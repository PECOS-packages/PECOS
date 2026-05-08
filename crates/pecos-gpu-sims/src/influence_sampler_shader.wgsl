// GPU Influence Map Sampler Shader v2 - Optimized
//
// Key optimization: Each thread handles ONE shot and ALL locations.
// Since each shot has its own output region, there's NO atomic contention.
//
// Previous approach: threads = num_locations × shot_words, atomic XOR on shared output
// New approach: threads = num_shots, direct writes to per-shot output regions
//
// This trades parallelism (fewer threads) for elimination of atomic contention,
// which is a net win for QEC sampling where:
// - num_shots is large (100k+)
// - num_locations is moderate (100-10000)
// - Many locations affect the same detectors (high contention in old approach)

struct Params {
    num_locations: u32,
    num_shots: u32,
    num_detectors: u32,
    num_dem_outputs: u32,
    p_error_threshold: u32,  // Fixed-point threshold (p * 0xFFFFFFFF)
    detector_words: u32,     // ceil(num_detectors / 32)
    dem_output_words: u32,   // ceil(num_dem_outputs / 32)
    _padding: u32,
}

@group(0) @binding(0) var<uniform> params: Params;

// Detector influence CSR arrays
@group(0) @binding(1) var<storage, read> det_offsets_x: array<u32>;
@group(0) @binding(2) var<storage, read> det_data_x: array<u32>;
@group(0) @binding(3) var<storage, read> det_offsets_y: array<u32>;
@group(0) @binding(4) var<storage, read> det_data_y: array<u32>;
@group(0) @binding(5) var<storage, read> det_offsets_z: array<u32>;
@group(0) @binding(6) var<storage, read> det_data_z: array<u32>;

// DEM-output influence CSR arrays
@group(0) @binding(7) var<storage, read> dem_output_offsets_x: array<u32>;
@group(0) @binding(8) var<storage, read> dem_output_data_x: array<u32>;
@group(0) @binding(9) var<storage, read> dem_output_offsets_y: array<u32>;
@group(0) @binding(10) var<storage, read> dem_output_data_y: array<u32>;
@group(0) @binding(11) var<storage, read> dem_output_offsets_z: array<u32>;
@group(0) @binding(12) var<storage, read> dem_output_data_z: array<u32>;

// Random seeds (one per shot)
@group(0) @binding(13) var<storage, read> random_seeds: array<u32>;

// Output: detector and DEM-output flips
// Layout: [shot * words + word_idx] - each shot has its own contiguous region
// NO atomics needed since each shot is processed by exactly one thread
@group(0) @binding(14) var<storage, read_write> detector_flips: array<u32>;
@group(0) @binding(15) var<storage, read_write> dem_output_flips: array<u32>;

// PCG-style hash function for deterministic randomness
fn hash(seed: u32, loc: u32, extra: u32) -> u32 {
    var h = seed ^ (loc * 0x9E3779B9u) ^ (extra * 0x85EBCA6Bu);
    h = h ^ (h >> 16u);
    h = h * 0x85EBCA6Bu;
    h = h ^ (h >> 13u);
    h = h * 0xC2B2AE35u;
    h = h ^ (h >> 16u);
    return h;
}

// XOR a bit into a word in global memory (non-atomic, safe since each shot has its own region)
fn xor_detector(shot_base: u32, det_idx: u32, detector_words: u32) {
    let word = det_idx / 32u;
    let bit = det_idx % 32u;
    if (word < detector_words) {
        let idx = shot_base + word;
        detector_flips[idx] = detector_flips[idx] ^ (1u << bit);
    }
}

fn xor_dem_output(shot_base: u32, dem_output_idx: u32, dem_output_words: u32) {
    let word = dem_output_idx / 32u;
    let bit = dem_output_idx % 32u;
    if (word < dem_output_words) {
        let idx = shot_base + word;
        dem_output_flips[idx] = dem_output_flips[idx] ^ (1u << bit);
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let shot = global_id.x;

    if (shot >= params.num_shots) {
        return;
    }

    let seed = random_seeds[shot];
    let det_base = shot * params.detector_words;
    let dem_output_base = shot * params.dem_output_words;

    // Initialize this shot's output to zero
    for (var w = 0u; w < params.detector_words; w = w + 1u) {
        detector_flips[det_base + w] = 0u;
    }
    for (var w = 0u; w < params.dem_output_words; w = w + 1u) {
        dem_output_flips[dem_output_base + w] = 0u;
    }

    // Process ALL locations for this shot
    for (var loc = 0u; loc < params.num_locations; loc = loc + 1u) {
        let rand_error = hash(seed, loc, 0u);

        // Check if error occurs at this location
        if (rand_error >= params.p_error_threshold) {
            continue;  // No error
        }

        // Error occurred - select Pauli type: 0=X, 1=Y, 2=Z
        let rand_pauli = hash(seed, loc, 1u);
        let pauli = rand_pauli % 3u;

        // Process detector and DEM-output influences based on Pauli type
        if (pauli == 0u) {
            // X fault - process detector influences
            let det_start = det_offsets_x[loc];
            let det_end = det_offsets_x[loc + 1u];
            for (var i = det_start; i < det_end; i = i + 1u) {
                xor_detector(det_base, det_data_x[i], params.detector_words);
            }

            // X fault - process DEM-output influences
            let dem_output_start = dem_output_offsets_x[loc];
            let dem_output_end = dem_output_offsets_x[loc + 1u];
            for (var i = dem_output_start; i < dem_output_end; i = i + 1u) {
                xor_dem_output(dem_output_base, dem_output_data_x[i], params.dem_output_words);
            }
        } else if (pauli == 1u) {
            // Y fault - process detector influences
            let det_start = det_offsets_y[loc];
            let det_end = det_offsets_y[loc + 1u];
            for (var i = det_start; i < det_end; i = i + 1u) {
                xor_detector(det_base, det_data_y[i], params.detector_words);
            }

            // Y fault - process DEM-output influences
            let dem_output_start = dem_output_offsets_y[loc];
            let dem_output_end = dem_output_offsets_y[loc + 1u];
            for (var i = dem_output_start; i < dem_output_end; i = i + 1u) {
                xor_dem_output(dem_output_base, dem_output_data_y[i], params.dem_output_words);
            }
        } else {
            // Z fault - process detector influences
            let det_start = det_offsets_z[loc];
            let det_end = det_offsets_z[loc + 1u];
            for (var i = det_start; i < det_end; i = i + 1u) {
                xor_detector(det_base, det_data_z[i], params.detector_words);
            }

            // Z fault - process DEM-output influences
            let dem_output_start = dem_output_offsets_z[loc];
            let dem_output_end = dem_output_offsets_z[loc + 1u];
            for (var i = dem_output_start; i < dem_output_end; i = i + 1u) {
                xor_dem_output(dem_output_base, dem_output_data_z[i], params.dem_output_words);
            }
        }
    }
}
