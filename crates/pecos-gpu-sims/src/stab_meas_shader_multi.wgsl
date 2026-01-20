// GPU Stabilizer Measurement Shader - Multi-Shot Version
//
// Implements stabilizer measurement entirely on GPU for multiple shots in parallel.
// Each shot is independent, enabling massive parallelization.
//
// The measurement algorithm for non-deterministic measurements:
// 1. Find first anticommuting stabilizer (one with X on measured qubit)
// 2. XOR chosen stabilizer into all other anticommuting stabilizers
// 3. XOR chosen stabilizer into anticommuting destabilizers
// 4. Copy old stabilizer to destabilizer (for chosen generator)
// 5. Replace chosen stabilizer with Z_q
// 6. Set sign based on measurement outcome

// Tableau buffers (same layout as gate shader)
@group(0) @binding(0) var<storage, read_write> stab_x: array<u32>;
@group(0) @binding(1) var<storage, read_write> stab_z: array<u32>;
@group(0) @binding(2) var<storage, read_write> destab_x: array<u32>;
@group(0) @binding(3) var<storage, read_write> destab_z: array<u32>;
@group(0) @binding(4) var<storage, read_write> sign_minus: array<u32>;
@group(0) @binding(7) var<storage, read_write> sign_i: array<u32>;

// Parameters
struct MeasParams {
    num_qubits: u32,
    gen_words: u32,
    num_gens: u32,
    num_shots: u32,
    measured_qubit: u32,
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
}

@group(0) @binding(5) var<uniform> params: MeasParams;

// Per-shot measurement data
// Layout: [shot_id] contains:
//   chosen_gen: u32 (0xFFFFFFFF if deterministic)
//   outcome: u32 (0 or 1)
//   is_deterministic: u32 (1 if deterministic, 0 if not)
@group(1) @binding(0) var<storage, read_write> meas_data: array<u32>;

// Random bits for non-deterministic outcomes (one per shot)
@group(1) @binding(1) var<storage, read> random_bits: array<u32>;

// Noise seeds for per-shot randomness
@group(1) @binding(2) var<storage, read> noise_seeds: array<u32>;

// Output: measurement results (one per shot)
@group(1) @binding(3) var<storage, read_write> results: array<u32>;

// Constants for meas_data layout
const MEAS_CHOSEN_GEN: u32 = 0u;
const MEAS_OUTCOME: u32 = 1u;
const MEAS_IS_DETERMINISTIC: u32 = 2u;
const MEAS_DATA_STRIDE: u32 = 4u;  // 4 u32s per shot

// ============================================================================
// Stage 1: Find anticommuting generator for each shot
// ============================================================================
// One thread per shot. Scans generators to find first with X on measured qubit.

@compute @workgroup_size(256)
fn meas_find_anticommuting(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let shot_id = global_id.x;
    if (shot_id >= params.num_shots) {
        return;
    }

    let qubit = params.measured_qubit;
    let tableau_stride = params.num_qubits * params.gen_words;
    let shot_tableau_base = shot_id * tableau_stride;

    // Find first generator with X on measured qubit
    var chosen_gen: u32 = 0xFFFFFFFFu;  // Sentinel for "deterministic"

    // For each generator, check if it has X on measured qubit
    // The X component for generator g, qubit q is stored at:
    //   stab_x[shot_tableau_base + q * gen_words + (g / 32)]
    // with bit position (g % 32)
    for (var gen_idx: u32 = 0u; gen_idx < params.num_qubits; gen_idx = gen_idx + 1u) {
        let word_idx = gen_idx / 32u;
        let bit_idx = gen_idx % 32u;
        let row_offset = shot_tableau_base + qubit * params.gen_words + word_idx;
        let has_x = ((stab_x[row_offset] >> bit_idx) & 1u) != 0u;

        if (has_x) {
            chosen_gen = gen_idx;
            break;
        }
    }

    // Store result in meas_data
    let data_offset = shot_id * MEAS_DATA_STRIDE;

    if (chosen_gen == 0xFFFFFFFFu) {
        // Deterministic case
        meas_data[data_offset + MEAS_CHOSEN_GEN] = 0xFFFFFFFFu;
        meas_data[data_offset + MEAS_IS_DETERMINISTIC] = 1u;
        // Outcome will be computed in next stage
    } else {
        // Non-deterministic case
        meas_data[data_offset + MEAS_CHOSEN_GEN] = chosen_gen;
        meas_data[data_offset + MEAS_IS_DETERMINISTIC] = 0u;
        meas_data[data_offset + MEAS_OUTCOME] = random_bits[shot_id] & 1u;
    }
}

// ============================================================================
// Stage 2: Compute deterministic outcomes
// ============================================================================
// For deterministic measurements, compute outcome from destabilizer phase.
// One thread per shot.

@compute @workgroup_size(256)
fn meas_compute_deterministic(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let shot_id = global_id.x;
    if (shot_id >= params.num_shots) {
        return;
    }

    let data_offset = shot_id * MEAS_DATA_STRIDE;
    let is_deterministic = meas_data[data_offset + MEAS_IS_DETERMINISTIC];

    if (is_deterministic == 0u) {
        return;  // Non-deterministic, skip
    }

    let qubit = params.measured_qubit;
    let tableau_stride = params.num_qubits * params.gen_words;
    let shot_tableau_base = shot_id * tableau_stride;
    let shot_sign_base = shot_id * params.gen_words;

    // Compute outcome using rowsum algorithm
    // Product of destabilizers that have X on measured qubit
    var minus_count: u32 = 0u;
    var i_count: u32 = 0u;

    for (var gen_idx: u32 = 0u; gen_idx < params.num_qubits; gen_idx = gen_idx + 1u) {
        let word_idx = gen_idx / 32u;
        let bit_idx = gen_idx % 32u;

        // Check if destabilizer has X on measured qubit
        let destab_offset = shot_tableau_base + qubit * params.gen_words + word_idx;
        let has_x = ((destab_x[destab_offset] >> bit_idx) & 1u) != 0u;

        if (has_x) {
            // Include this destabilizer's phase
            let sign_word = sign_minus[shot_sign_base + word_idx];
            let i_word = sign_i[shot_sign_base + word_idx];

            if ((sign_word >> bit_idx) & 1u) != 0u {
                minus_count = minus_count + 1u;
            }
            if ((i_word >> bit_idx) & 1u) != 0u {
                i_count = i_count + 1u;
            }

            // Also need to count intersections between destabilizers
            // This is O(n^2) - simplified version, may need optimization
        }
    }

    // Outcome is 1 if total phase is negative
    let outcome = (minus_count + (i_count / 2u)) % 2u;
    meas_data[data_offset + MEAS_OUTCOME] = outcome;
}

// ============================================================================
// Stage 3: XOR chosen stabilizer into anticommuting stabilizers
// ============================================================================
// For non-deterministic measurements. One thread per (shot_id, word_idx).

@compute @workgroup_size(256)
fn meas_xor_stabilizers(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let thread_id = global_id.x;
    let total_threads = params.num_shots * params.gen_words;

    if (thread_id >= total_threads) {
        return;
    }

    let shot_id = thread_id / params.gen_words;
    let word_idx = thread_id % params.gen_words;

    let data_offset = shot_id * MEAS_DATA_STRIDE;
    let is_deterministic = meas_data[data_offset + MEAS_IS_DETERMINISTIC];

    if (is_deterministic != 0u) {
        return;  // Deterministic, no tableau update needed
    }

    let chosen_gen = meas_data[data_offset + MEAS_CHOSEN_GEN];
    let qubit = params.measured_qubit;
    let tableau_stride = params.num_qubits * params.gen_words;
    let shot_tableau_base = shot_id * tableau_stride;
    let shot_sign_base = shot_id * params.gen_words;

    // Get chosen generator's data for this word
    let chosen_word_for_gen = chosen_gen / 32u;
    let chosen_bit_for_gen = chosen_gen % 32u;

    // For each generator (by checking bits in this word)
    for (var bit: u32 = 0u; bit < 32u; bit = bit + 1u) {
        let gen_idx = word_idx * 32u + bit;
        if (gen_idx >= params.num_qubits || gen_idx == chosen_gen) {
            continue;
        }

        // Check if this generator anticommutes (has X on measured qubit)
        let gen_x_offset = shot_tableau_base + qubit * params.gen_words + (gen_idx / 32u);
        let gen_bit = gen_idx % 32u;
        let has_x = ((stab_x[gen_x_offset] >> gen_bit) & 1u) != 0u;

        if (!has_x) {
            continue;
        }

        // XOR chosen generator's row into this generator's row
        // For each qubit q, XOR chosen's X[q] and Z[q] into this gen's X[q] and Z[q]
        for (var q: u32 = 0u; q < params.num_qubits; q = q + 1u) {
            let q_offset = shot_tableau_base + q * params.gen_words;

            // Get chosen generator's bits for qubit q
            let chosen_x = (stab_x[q_offset + chosen_word_for_gen] >> chosen_bit_for_gen) & 1u;
            let chosen_z = (stab_z[q_offset + chosen_word_for_gen] >> chosen_bit_for_gen) & 1u;

            // XOR into current generator
            let curr_word = gen_idx / 32u;
            let curr_bit = gen_idx % 32u;

            if (chosen_x != 0u) {
                // Atomic XOR would be ideal, but use regular for now
                // This works because each thread handles different generators
                let mask = 1u << curr_bit;
                stab_x[q_offset + curr_word] ^= mask;
            }
            if (chosen_z != 0u) {
                let mask = 1u << curr_bit;
                stab_z[q_offset + curr_word] ^= mask;
            }
        }

        // Update sign (simplified - full phase tracking would need more work)
        // For now, just XOR the chosen sign into this sign
        let chosen_sign_word = sign_minus[shot_sign_base + chosen_word_for_gen];
        if ((chosen_sign_word >> chosen_bit_for_gen) & 1u) != 0u {
            let curr_word = gen_idx / 32u;
            let curr_bit = gen_idx % 32u;
            sign_minus[shot_sign_base + curr_word] ^= (1u << curr_bit);
        }
    }
}

// ============================================================================
// Stage 4: XOR chosen stabilizer into anticommuting destabilizers
// ============================================================================
// Similar to stage 3, but for destabilizers. One thread per (shot_id, word_idx).

@compute @workgroup_size(256)
fn meas_xor_destabilizers(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let thread_id = global_id.x;
    let total_threads = params.num_shots * params.gen_words;

    if (thread_id >= total_threads) {
        return;
    }

    let shot_id = thread_id / params.gen_words;
    let word_idx = thread_id % params.gen_words;

    let data_offset = shot_id * MEAS_DATA_STRIDE;
    let is_deterministic = meas_data[data_offset + MEAS_IS_DETERMINISTIC];

    if (is_deterministic != 0u) {
        return;
    }

    let chosen_gen = meas_data[data_offset + MEAS_CHOSEN_GEN];
    let qubit = params.measured_qubit;
    let tableau_stride = params.num_qubits * params.gen_words;
    let shot_tableau_base = shot_id * tableau_stride;

    let chosen_word_for_gen = chosen_gen / 32u;
    let chosen_bit_for_gen = chosen_gen % 32u;

    // For each destabilizer that anticommutes
    for (var bit: u32 = 0u; bit < 32u; bit = bit + 1u) {
        let gen_idx = word_idx * 32u + bit;
        if (gen_idx >= params.num_qubits || gen_idx == chosen_gen) {
            continue;
        }

        // Check if destabilizer has X on measured qubit
        let destab_x_offset = shot_tableau_base + qubit * params.gen_words + (gen_idx / 32u);
        let gen_bit = gen_idx % 32u;
        let has_x = ((destab_x[destab_x_offset] >> gen_bit) & 1u) != 0u;

        if (!has_x) {
            continue;
        }

        // XOR chosen stabilizer's row into this destabilizer's row
        for (var q: u32 = 0u; q < params.num_qubits; q = q + 1u) {
            let q_offset = shot_tableau_base + q * params.gen_words;

            let chosen_x = (stab_x[q_offset + chosen_word_for_gen] >> chosen_bit_for_gen) & 1u;
            let chosen_z = (stab_z[q_offset + chosen_word_for_gen] >> chosen_bit_for_gen) & 1u;

            let curr_word = gen_idx / 32u;
            let curr_bit = gen_idx % 32u;

            if (chosen_x != 0u) {
                destab_x[q_offset + curr_word] ^= (1u << curr_bit);
            }
            if (chosen_z != 0u) {
                destab_z[q_offset + curr_word] ^= (1u << curr_bit);
            }
        }
    }
}

// ============================================================================
// Stage 5: Finalize measurement - update chosen generator
// ============================================================================
// Copy old stabilizer to destabilizer, set new stabilizer to Z_q.
// One thread per (shot_id, qubit) pair.

@compute @workgroup_size(256)
fn meas_finalize(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let thread_id = global_id.x;
    let total_threads = params.num_shots * params.num_qubits;

    if (thread_id >= total_threads) {
        return;
    }

    let shot_id = thread_id / params.num_qubits;
    let q = thread_id % params.num_qubits;

    let data_offset = shot_id * MEAS_DATA_STRIDE;
    let is_deterministic = meas_data[data_offset + MEAS_IS_DETERMINISTIC];

    if (is_deterministic != 0u) {
        return;
    }

    let chosen_gen = meas_data[data_offset + MEAS_CHOSEN_GEN];
    let outcome = meas_data[data_offset + MEAS_OUTCOME];
    let measured_qubit = params.measured_qubit;
    let tableau_stride = params.num_qubits * params.gen_words;
    let shot_tableau_base = shot_id * tableau_stride;
    let shot_sign_base = shot_id * params.gen_words;

    let chosen_word = chosen_gen / 32u;
    let chosen_bit = chosen_gen % 32u;
    let q_offset = shot_tableau_base + q * params.gen_words;
    let mask = 1u << chosen_bit;

    // Get old stabilizer values for this qubit
    let old_stab_x = (stab_x[q_offset + chosen_word] >> chosen_bit) & 1u;
    let old_stab_z = (stab_z[q_offset + chosen_word] >> chosen_bit) & 1u;

    // Copy to destabilizer (set bit if old value was 1, clear otherwise)
    if (old_stab_x != 0u) {
        destab_x[q_offset + chosen_word] |= mask;
    } else {
        destab_x[q_offset + chosen_word] &= ~mask;
    }
    if (old_stab_z != 0u) {
        destab_z[q_offset + chosen_word] |= mask;
    } else {
        destab_z[q_offset + chosen_word] &= ~mask;
    }

    // Set new stabilizer to Z_q (Z only on measured qubit)
    if (q == measured_qubit) {
        stab_z[q_offset + chosen_word] |= mask;
    } else {
        stab_z[q_offset + chosen_word] &= ~mask;
    }
    stab_x[q_offset + chosen_word] &= ~mask;  // Clear X for all qubits

    // Set sign based on outcome (first thread for this shot handles signs)
    if (q == 0u) {
        // Clear i phase
        sign_i[shot_sign_base + chosen_word] &= ~mask;
        // Set minus based on outcome
        if (outcome != 0u) {
            sign_minus[shot_sign_base + chosen_word] |= mask;
        } else {
            sign_minus[shot_sign_base + chosen_word] &= ~mask;
        }
    }
}

// ============================================================================
// Stage 6: Write final results
// ============================================================================
// Copy outcomes to results buffer. One thread per shot.

@compute @workgroup_size(256)
fn meas_write_results(
    @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let shot_id = global_id.x;
    if (shot_id >= params.num_shots) {
        return;
    }

    let data_offset = shot_id * MEAS_DATA_STRIDE;
    results[shot_id] = meas_data[data_offset + MEAS_OUTCOME];
}
