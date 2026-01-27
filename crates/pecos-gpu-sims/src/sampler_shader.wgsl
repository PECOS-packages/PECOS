// GPU Measurement Sampler Shader
//
// Samples measurement outcomes based on dependencies computed by SymbolicSparseStab.
// Uses column-major processing: each thread handles one "word" (32 shots) and
// processes all measurements sequentially.
//
// RNG: Stateless hash-based (MurmurHash3 finalizer) - race-free and independent per measurement.

// Measurement metadata buffer
// Packed format: [type: 4 bits][flip: 1 bit][dep_count: 4 bits][source: 23 bits]
@group(0) @binding(0) var<storage, read> measurement_meta: array<u32>;

// Dependency indices (MAX_DEPS_PER_MEASUREMENT per measurement)
@group(0) @binding(1) var<storage, read> deps: array<u32>;

// Parameters
struct Params {
    num_measurements: u32,
    num_words: u32,
    num_shots: u32,
    error_threshold: u32,  // For noisy sampling (fixed-point: threshold = error_rate * 2^32)
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
    _padding4: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

// RNG seed data (4 u32s per word, read-only - used as input to stateless hash)
@group(0) @binding(3) var<storage, read_write> rng_state: array<u32>;

// Results: results[measurement * num_words + word_idx]
@group(0) @binding(4) var<storage, read_write> results: array<u32>;

// Statistics output: counts[measurement] = popcount of all shots for that measurement
@group(0) @binding(5) var<storage, read_write> counts: array<atomic<u32>>;

// Measurement type constants
const TYPE_FIXED_0: u32 = 0u;
const TYPE_FIXED_1: u32 = 1u;
const TYPE_RANDOM: u32 = 2u;
const TYPE_COPY: u32 = 3u;
const TYPE_COPY_FLIPPED: u32 = 4u;
const TYPE_COMPUTED: u32 = 5u;

// Maximum dependencies per measurement (must match Rust code)
const MAX_DEPS: u32 = 16u;

// ============================================================================
// Stateless Hash-Based Random Number Generator
// ============================================================================
// Uses per-word seed data (read-only) combined with a key parameter to produce
// independent random values. No mutable state means no data races when multiple
// threads share the same word_idx but use different keys.

// Generate one random u32 from the per-word seed and a unique key.
// Different keys produce independent random values for the same word.
fn random_u32(word_idx: u32, key: u32) -> u32 {
    let base = word_idx * 4u;

    // Combine all 4 seed values with the key using multiplicative hashing
    var h = rng_state[base] + (key * 2654435761u); // golden ratio constant
    h ^= rng_state[base + 1u];
    h *= 2246822519u;
    h ^= rng_state[base + 2u];
    h += rng_state[base + 3u];
    h ^= key;

    // MurmurHash3 32-bit finalizer for good avalanche properties
    h ^= h >> 16u;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13u;
    h *= 0xC2B2AE35u;
    h ^= h >> 16u;

    return h;
}

// ============================================================================
// Main Sampling Kernel
// ============================================================================

@compute @workgroup_size(256)
fn sample_measurements(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let word_idx = global_id.x;
    if (word_idx >= params.num_words) {
        return;
    }

    // Process each measurement in order
    for (var m: u32 = 0u; m < params.num_measurements; m = m + 1u) {
        let mdata = measurement_meta[m];
        let mtype = mdata & 0xFu;
        let flip = ((mdata >> 4u) & 1u) != 0u;
        let dep_count = (mdata >> 5u) & 0xFu;
        let source = (mdata >> 9u) & 0x7FFFFFu;

        var result: u32 = 0u;

        switch (mtype) {
            case TYPE_FIXED_0: {
                result = 0u;
            }
            case TYPE_FIXED_1: {
                result = 0xFFFFFFFFu;
            }
            case TYPE_RANDOM: {
                result = random_u32(word_idx, m);
            }
            case TYPE_COPY: {
                let src_idx = source * params.num_words + word_idx;
                result = results[src_idx];
            }
            case TYPE_COPY_FLIPPED: {
                let src_idx = source * params.num_words + word_idx;
                result = ~results[src_idx];
            }
            case TYPE_COMPUTED: {
                // Start with flip value
                if (flip) {
                    result = 0xFFFFFFFFu;
                } else {
                    result = 0u;
                }

                // XOR all dependencies
                let deps_base = m * MAX_DEPS;
                for (var d: u32 = 0u; d < dep_count; d = d + 1u) {
                    let dep_m = deps[deps_base + d];
                    let dep_idx = dep_m * params.num_words + word_idx;
                    result = result ^ results[dep_idx];
                }
            }
            default: {
                result = 0u;
            }
        }

        // Store result
        let out_idx = m * params.num_words + word_idx;
        results[out_idx] = result;
    }
}

// ============================================================================
// Parallel Sampling Kernel (for Fixed/Random heavy workloads)
// ============================================================================
// Each thread handles one (measurement, word) pair.
// Only works for Fixed and Random measurements (no dependencies).

@compute @workgroup_size(256)
fn sample_parallel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // 2D dispatch: flatten to linear index
    let idx = global_id.x + global_id.y * 65535u * 256u;
    let total_elements = params.num_measurements * params.num_words;

    if (idx >= total_elements) {
        return;
    }

    let m = idx / params.num_words;
    let word_idx = idx % params.num_words;

    let mdata = measurement_meta[m];
    let mtype = mdata & 0xFu;

    var result: u32 = 0u;

    switch (mtype) {
        case TYPE_FIXED_0: {
            result = 0u;
        }
        case TYPE_FIXED_1: {
            result = 0xFFFFFFFFu;
        }
        case TYPE_RANDOM: {
            result = random_u32(word_idx, m);
        }
        default: {
            // Skip Copy/CopyFlipped/Computed - handled by dependent kernel
            return;
        }
    }

    let out_idx = m * params.num_words + word_idx;
    results[out_idx] = result;
}

// ============================================================================
// Dependent Measurements Kernel
// ============================================================================
// Processes measurements with dependencies in a single pass.
// Uses the dependency buffer to control ordering.
// measurement_meta2 contains: [output_idx (16 bits)][type (4 bits)][flags (12 bits)]

@compute @workgroup_size(256)
fn sample_dependent(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let word_idx = global_id.x;
    if (word_idx >= params.num_words) {
        return;
    }

    // params.error_threshold is repurposed as num_dependent_measurements
    let num_dependent = params.error_threshold;

    // Process dependent measurements sequentially per thread
    // The deps buffer now contains [num_deps per dependent measurement, followed by their indices]
    for (var d: u32 = 0u; d < num_dependent; d = d + 1u) {
        let mdata = measurement_meta[d];
        let output_m = (mdata >> 16u) & 0xFFFFu;
        let mtype = mdata & 0xFu;
        let flip = ((mdata >> 4u) & 1u) != 0u;
        let dep_count = (mdata >> 5u) & 0xFu;
        let source = (mdata >> 9u) & 0x7Fu; // Only 7 bits for source in dependent mode

        var result: u32 = 0u;

        switch (mtype) {
            case TYPE_COPY: {
                let src_idx = source * params.num_words + word_idx;
                result = results[src_idx];
            }
            case TYPE_COPY_FLIPPED: {
                let src_idx = source * params.num_words + word_idx;
                result = ~results[src_idx];
            }
            case TYPE_COMPUTED: {
                if (flip) {
                    result = 0xFFFFFFFFu;
                } else {
                    result = 0u;
                }

                let deps_base = d * MAX_DEPS;
                for (var i: u32 = 0u; i < dep_count; i = i + 1u) {
                    let dep_m = deps[deps_base + i];
                    let dep_idx = dep_m * params.num_words + word_idx;
                    result = result ^ results[dep_idx];
                }
            }
            default: {
                result = 0u;
            }
        }

        let out_idx = output_m * params.num_words + word_idx;
        results[out_idx] = result;
    }
}

// ============================================================================
// Noise Application Kernel
// ============================================================================
// Applies bit flips to simulate measurement errors.
// Each bit has probability error_threshold/2^32 of being flipped.

// ============================================================================
// Count Reduction Kernel
// ============================================================================
// Computes popcount of each measurement column and atomically adds to counts buffer.
// Run after sampling to get statistics without full data transfer.

@compute @workgroup_size(256)
fn count_ones(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // 2D dispatch for large workloads
    let idx = global_id.x + global_id.y * 65535u * 256u;
    let total_elements = params.num_measurements * params.num_words;

    if (idx >= total_elements) {
        return;
    }

    let m = idx / params.num_words;
    let word_idx = idx % params.num_words;

    // Get the result word and count its bits
    let result_idx = m * params.num_words + word_idx;
    let word = results[result_idx];
    let bits = countOneBits(word);

    // Atomically add to the count for this measurement
    atomicAdd(&counts[m], bits);
}

// ============================================================================
// Sample and Count Combined Kernel (for stats-only use case)
// ============================================================================
// Samples measurements and accumulates counts without storing full results.
// Much faster when only statistics are needed, not full shot data.

@compute @workgroup_size(256)
fn sample_and_count(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // 2D dispatch for large workloads
    let idx = global_id.x + global_id.y * 65535u * 256u;
    let total_elements = params.num_measurements * params.num_words;

    if (idx >= total_elements) {
        return;
    }

    let m = idx / params.num_words;
    let word_idx = idx % params.num_words;

    let mdata = measurement_meta[m];
    let mtype = mdata & 0xFu;

    var result: u32 = 0u;

    switch (mtype) {
        case TYPE_FIXED_0: {
            result = 0u;
        }
        case TYPE_FIXED_1: {
            result = 0xFFFFFFFFu;
        }
        case TYPE_RANDOM: {
            result = random_u32(word_idx, m);
        }
        default: {
            // For dependent measurements, we need to fall back to sequential
            // This kernel only handles independent measurements
            return;
        }
    }

    // For the last word, mask out bits beyond num_shots
    let is_last_word = (word_idx == params.num_words - 1u);
    let remaining_bits = params.num_shots % 32u;
    if (is_last_word && remaining_bits != 0u) {
        // Create mask for valid bits (e.g., 0b111111 for 6 remaining bits)
        let mask = (1u << remaining_bits) - 1u;
        result = result & mask;
    }

    // Count bits and add atomically
    let bits = countOneBits(result);
    atomicAdd(&counts[m], bits);
}

@compute @workgroup_size(256)
fn apply_noise(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Support 2D dispatch: idx = x + y * 65535 * 256 (workgroup_size)
    let idx = global_id.x + global_id.y * 65535u * 256u;
    let total_elements = params.num_measurements * params.num_words;

    if (idx >= total_elements) {
        return;
    }

    let m = idx / params.num_words;
    let word_idx = idx % params.num_words;

    // Skip if no error rate
    if (params.error_threshold == 0u) {
        return;
    }

    // Generate 32 random bits and compare each to threshold.
    // Use keys in a separate namespace (high bit set) to avoid collisions with sampling keys.
    var flip_mask: u32 = 0u;
    let noise_key_base = 0x80000000u | (m * 32u);

    for (var bit: u32 = 0u; bit < 32u; bit = bit + 1u) {
        let rand_val = random_u32(word_idx, noise_key_base + bit);
        if (rand_val < params.error_threshold) {
            flip_mask = flip_mask | (1u << bit);
        }
    }

    // Apply flips
    let out_idx = m * params.num_words + word_idx;
    results[out_idx] = results[out_idx] ^ flip_mask;
}
