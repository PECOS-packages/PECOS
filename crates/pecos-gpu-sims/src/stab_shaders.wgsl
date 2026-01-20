// GPU Stabilizer Tableau Simulation Shaders
//
// TRANSPOSED memory layout - qubits as rows, generators as columns.
// This layout enables efficient parallel gate operations on contiguous memory.
//
// For n qubits, we have n stabilizer generators and n destabilizer generators.
// Memory layout (row = qubit, col = generator):
//   stab_x[qubit * gen_words + word_idx]   - X bits for all stabilizers on this qubit
//   stab_z[qubit * gen_words + word_idx]   - Z bits for all stabilizers on this qubit
//   destab_x[qubit * gen_words + word_idx] - X bits for all destabilizers on this qubit
//   destab_z[qubit * gen_words + word_idx] - Z bits for all destabilizers on this qubit
//   sign_minus[word_idx] bit i             - minus sign for generator (word_idx * 32 + i)
//   sign_i[word_idx] bit i                 - i phase for generator (word_idx * 32 + i)
//
// Each word contains bits for generators [word_idx*32, word_idx*32+32).

// Stabilizer tableau buffers
@group(0) @binding(0) var<storage, read_write> stab_x: array<u32>;
@group(0) @binding(1) var<storage, read_write> stab_z: array<u32>;
@group(0) @binding(2) var<storage, read_write> destab_x: array<u32>;
@group(0) @binding(3) var<storage, read_write> destab_z: array<u32>;
@group(0) @binding(4) var<storage, read_write> sign_minus: array<u32>;
@group(0) @binding(7) var<storage, read_write> sign_i: array<u32>;

// Gate parameters
struct StabParams {
    num_qubits: u32,      // Total number of qubits
    gen_words: u32,       // Number of u32 words per qubit row (ceil(num_qubits / 32))
    num_gens: u32,        // Number of generators per type (= num_qubits)
    target_qubit: u32,    // Target qubit for single-qubit gates
    control_qubit: u32,   // Control qubit for two-qubit gates
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
}

@group(0) @binding(5) var<uniform> params: StabParams;

// Workgroup size
const WORKGROUP_SIZE: u32 = 256u;

// Sign bit constants
const SIGN_MINUS: u32 = 1u;  // bit 0: minus sign
const SIGN_I: u32 = 2u;      // bit 1: i phase

// Helper to get bit at position in a word
fn get_bit(word: u32, bit_pos: u32) -> bool {
    return (word & (1u << bit_pos)) != 0u;
}

// Helper to toggle bit at position in a word
fn toggle_bit(word: u32, bit_pos: u32) -> u32 {
    return word ^ (1u << bit_pos);
}

// Helper to set bit at position
fn set_bit(word: u32, bit_pos: u32) -> u32 {
    return word | (1u << bit_pos);
}

// Helper to clear bit at position
fn clear_bit(word: u32, bit_pos: u32) -> u32 {
    return word & ~(1u << bit_pos);
}

// Get word index and bit position for a generator index
fn gen_to_word_bit(gen: u32) -> vec2<u32> {
    return vec2<u32>(gen / 32u, gen % 32u);
}

// =============================================================================
// Hadamard Gate (H)
// =============================================================================
// H transforms: X -> Z, Z -> X
// Sign update: if both X and Z are set, multiply by -1
//
// With transposed layout, H on qubit q swaps row q of stab_x with row q of stab_z
// Each thread handles one word of the row
@compute @workgroup_size(256)
fn apply_h(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let word_idx = global_id.x;

    if (word_idx >= params.gen_words) {
        return;
    }

    let q = params.target_qubit;
    let row_offset = q * params.gen_words + word_idx;

    // Read ORIGINAL values before swapping
    let orig_stab_x_word = stab_x[row_offset];
    let orig_stab_z_word = stab_z[row_offset];

    // Swap stab_x and stab_z for this qubit
    stab_x[row_offset] = orig_stab_z_word;
    stab_z[row_offset] = orig_stab_x_word;

    // Swap destab_x and destab_z for this qubit
    let destab_x_word = destab_x[row_offset];
    let destab_z_word = destab_z[row_offset];
    destab_x[row_offset] = destab_z_word;
    destab_z[row_offset] = destab_x_word;

    // Update signs for generators in THIS word only (no race condition)
    // Each thread handles generators [word_idx*32, min((word_idx+1)*32, num_gens))
    // H(XZ) = H(iY) = -iY = -XZ, so we get a minus sign when both X and Z were set
    // With packed signs, we can compute a mask and XOR once
    let mask = orig_stab_x_word & orig_stab_z_word;  // Generators with both X and Z
    sign_minus[word_idx] = sign_minus[word_idx] ^ mask;
}

// =============================================================================
// S Gate (Phase gate, sqrt(Z))
// =============================================================================
// S transforms: X -> Y = iXZ, Z -> Z
// In tableau: Z ^= X (add X to Z support)
// Sign update: if X is set and had i phase, toggle minus (i*i = -1), then toggle i
//
// With transposed layout, S on qubit q: stab_z[q] ^= stab_x[q]
@compute @workgroup_size(256)
fn apply_s(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let word_idx = global_id.x;

    if (word_idx >= params.gen_words) {
        return;
    }

    let q = params.target_qubit;
    let row_offset = q * params.gen_words + word_idx;

    // Read X bits for this word (need these for sign updates)
    let stab_x_word = stab_x[row_offset];
    let destab_x_word = destab_x[row_offset];

    // For each generator with X on this qubit, toggle its Z bit
    stab_z[row_offset] = stab_z[row_offset] ^ stab_x_word;
    destab_z[row_offset] = destab_z[row_offset] ^ destab_x_word;

    // Update signs for generators in THIS word only (no race condition)
    // With packed signs: for generators with X, toggle i phase and handle i*i = -1
    // Generators with both X and existing i phase get their minus sign toggled
    let had_i_mask = sign_i[word_idx];
    let toggle_minus_mask = stab_x_word & had_i_mask;  // X and had i -> toggle minus
    sign_minus[word_idx] = sign_minus[word_idx] ^ toggle_minus_mask;
    // Toggle i phase for all generators with X
    sign_i[word_idx] = sign_i[word_idx] ^ stab_x_word;
}

// =============================================================================
// CX Gate (CNOT)
// =============================================================================
// CX(c,t) transforms:
//   XI -> XX  (X on control propagates to target)
//   IX -> IX  (X on target stays)
//   ZI -> ZI  (Z on control stays)
//   IZ -> ZZ  (Z on target propagates back to control)
//
// In tableau (transposed):
//   Row t of X ^= Row c of X  (for stab and destab)
//   Row c of Z ^= Row t of Z  (for stab and destab)
//
// Each thread handles one word of the rows
@compute @workgroup_size(256)
fn apply_cx_simple(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let word_idx = global_id.x;

    if (word_idx >= params.gen_words) {
        return;
    }

    let c = params.control_qubit;
    let t = params.target_qubit;

    let ctrl_offset = c * params.gen_words + word_idx;
    let tgt_offset = t * params.gen_words + word_idx;

    // X_target ^= X_control (for all generators in this word)
    stab_x[tgt_offset] = stab_x[tgt_offset] ^ stab_x[ctrl_offset];
    destab_x[tgt_offset] = destab_x[tgt_offset] ^ destab_x[ctrl_offset];

    // Z_control ^= Z_target (for all generators in this word)
    stab_z[ctrl_offset] = stab_z[ctrl_offset] ^ stab_z[tgt_offset];
    destab_z[ctrl_offset] = destab_z[ctrl_offset] ^ destab_z[tgt_offset];

    // Note: CX does NOT require sign updates.
}

// =============================================================================
// X Gate (Pauli X)
// =============================================================================
// X transforms: X -> X, Z -> -Z, Y -> -Y
// In tableau: if Z bit is set for this qubit, toggle minus sign
@compute @workgroup_size(256)
fn apply_x(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let word_idx = global_id.x;

    if (word_idx >= params.gen_words) {
        return;
    }

    let q = params.target_qubit;
    let row_offset = q * params.gen_words + word_idx;

    // Read Z bits for this word
    let stab_z_word = stab_z[row_offset];

    // X gate doesn't change tableau bits, only signs
    // With packed signs: toggle minus for generators with Z on this qubit
    sign_minus[word_idx] = sign_minus[word_idx] ^ stab_z_word;
}

// =============================================================================
// Z Gate (Pauli Z)
// =============================================================================
// Z transforms: X -> -X, Z -> Z, Y -> -Y
// In tableau: if X bit is set for this qubit, toggle minus sign
@compute @workgroup_size(256)
fn apply_z(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let word_idx = global_id.x;

    if (word_idx >= params.gen_words) {
        return;
    }

    let q = params.target_qubit;
    let row_offset = q * params.gen_words + word_idx;

    // Read X bits for this word
    let stab_x_word = stab_x[row_offset];

    // Z gate doesn't change tableau bits, only signs
    // With packed signs: toggle minus for generators with X on this qubit
    sign_minus[word_idx] = sign_minus[word_idx] ^ stab_x_word;
}

// =============================================================================
// Y Gate (Pauli Y)
// =============================================================================
// Y transforms: X -> -X, Z -> -Z, Y -> Y
// In tableau: if exactly one of X or Z is set, toggle minus sign
@compute @workgroup_size(256)
fn apply_y(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let word_idx = global_id.x;

    if (word_idx >= params.gen_words) {
        return;
    }

    let q = params.target_qubit;
    let row_offset = q * params.gen_words + word_idx;

    // Read X and Z bits for this word
    let stab_x_word = stab_x[row_offset];
    let stab_z_word = stab_z[row_offset];

    // Y gate doesn't change tableau bits, only signs
    // With packed signs: toggle minus if exactly one of X or Z is set (XOR)
    let xor_mask = stab_x_word ^ stab_z_word;  // Exactly one of X or Z
    sign_minus[word_idx] = sign_minus[word_idx] ^ xor_mask;
}

// =============================================================================
// CZ Gate (Controlled-Z)
// =============================================================================
// CZ(c,t) transforms:
//   XI -> XZ  (X on control picks up Z on target)
//   IX -> ZX  (X on target picks up Z on control)
//   ZI -> ZI  (Z unchanged)
//   IZ -> IZ  (Z unchanged)
//
// In tableau (transposed):
//   Row t of Z ^= Row c of X  (Z_t gets X_c)
//   Row c of Z ^= Row t of X  (Z_c gets X_t)
@compute @workgroup_size(256)
fn apply_cz(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let word_idx = global_id.x;

    if (word_idx >= params.gen_words) {
        return;
    }

    let c = params.control_qubit;
    let t = params.target_qubit;

    let ctrl_offset = c * params.gen_words + word_idx;
    let tgt_offset = t * params.gen_words + word_idx;

    // Read X values from both qubits for this word
    let ctrl_x_stab = stab_x[ctrl_offset];
    let tgt_x_stab = stab_x[tgt_offset];
    let ctrl_x_destab = destab_x[ctrl_offset];
    let tgt_x_destab = destab_x[tgt_offset];

    // Z_target ^= X_control
    stab_z[tgt_offset] = stab_z[tgt_offset] ^ ctrl_x_stab;
    destab_z[tgt_offset] = destab_z[tgt_offset] ^ ctrl_x_destab;

    // Z_control ^= X_target
    stab_z[ctrl_offset] = stab_z[ctrl_offset] ^ tgt_x_stab;
    destab_z[ctrl_offset] = destab_z[ctrl_offset] ^ tgt_x_destab;

    // CZ sign update: if both X_c and X_t are set for a generator, toggle minus
    // With packed signs: toggle minus for generators with both X_c and X_t
    let both_x_mask = ctrl_x_stab & tgt_x_stab;  // Generators with X on both qubits
    sign_minus[word_idx] = sign_minus[word_idx] ^ both_x_mask;
}

// =============================================================================
// SWAP Gate
// =============================================================================
// SWAP(a,b) exchanges qubits a and b
// In tableau (transposed): swap row a with row b for all buffers
@compute @workgroup_size(256)
fn apply_swap(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let word_idx = global_id.x;

    if (word_idx >= params.gen_words) {
        return;
    }

    let a = params.control_qubit;
    let b = params.target_qubit;

    let a_offset = a * params.gen_words + word_idx;
    let b_offset = b * params.gen_words + word_idx;

    // Swap stab_x rows
    let tmp_stab_x = stab_x[a_offset];
    stab_x[a_offset] = stab_x[b_offset];
    stab_x[b_offset] = tmp_stab_x;

    // Swap stab_z rows
    let tmp_stab_z = stab_z[a_offset];
    stab_z[a_offset] = stab_z[b_offset];
    stab_z[b_offset] = tmp_stab_z;

    // Swap destab_x rows
    let tmp_destab_x = destab_x[a_offset];
    destab_x[a_offset] = destab_x[b_offset];
    destab_x[b_offset] = tmp_destab_x;

    // Swap destab_z rows
    let tmp_destab_z = destab_z[a_offset];
    destab_z[a_offset] = destab_z[b_offset];
    destab_z[b_offset] = tmp_destab_z;

    // SWAP does NOT require sign updates
}

// =============================================================================
// Reset to |0...0> state
// =============================================================================
// Initialize stabilizers as Z_i and destabilizers as X_i
// In transposed layout:
//   stab_z[i, i] = 1, all other stab entries = 0
//   destab_x[i, i] = 1, all other destab entries = 0
@compute @workgroup_size(256)
fn reset_state(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    // Each thread handles one element of the arrays
    let total_elements = params.num_qubits * params.gen_words;

    if (idx >= total_elements) {
        return;
    }

    let qubit = idx / params.gen_words;
    let word_idx = idx % params.gen_words;

    // In transposed layout:
    // stab_z[qubit, gen] = 1 iff qubit == gen (Z_i stabilizer)
    // destab_x[qubit, gen] = 1 iff qubit == gen (X_i destabilizer)
    let gen_wb = gen_to_word_bit(qubit);

    var stab_z_val = 0u;
    var destab_x_val = 0u;

    if (gen_wb.x == word_idx) {
        stab_z_val = 1u << gen_wb.y;
        destab_x_val = 1u << gen_wb.y;
    }

    stab_x[idx] = 0u;
    stab_z[idx] = stab_z_val;
    destab_x[idx] = destab_x_val;
    destab_z[idx] = 0u;

    // Reset signs (only first row of threads, one per word)
    // With packed format, each word_idx thread clears its sign words
    if (qubit == 0u && word_idx < params.gen_words) {
        sign_minus[word_idx] = 0u;
        sign_i[word_idx] = 0u;
    }
}

// =============================================================================
// Measurement: Find generators with X support on measured qubit
// =============================================================================
// In transposed layout, row q of stab_x contains bits indicating which
// generators have X support on qubit q.
@group(1) @binding(0) var<storage, read_write> anticommuting: array<u32>;

@compute @workgroup_size(256)
fn find_anticommuting(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gen_idx = global_id.x;

    if (gen_idx >= params.num_gens) {
        return;
    }

    let q = params.target_qubit;
    let wb = gen_to_word_bit(gen_idx);
    let offset = q * params.gen_words + wb.x;

    let x_word = stab_x[offset];
    let has_x = get_bit(x_word, wb.y);

    // Store 1 if this generator anticommutes (has X on measured qubit), 0 otherwise
    anticommuting[gen_idx] = select(0u, 1u, has_x);
}

// =============================================================================
// Measurement GPU Shaders (Transposed Layout)
// =============================================================================

// Measurement buffer for storing intermediate results
// Layout: [min_weight, chosen_row, outcome, measurement_qubit, ...row_weights...]
@group(1) @binding(0) var<storage, read_write> measurement_data: array<atomic<u32>>;

// Saved generator data for XOR operations
// In transposed layout, we save the X and Z support of the chosen generator across all qubits
// saved_row_x[qubit] and saved_row_z[qubit] - one bit per qubit
@group(1) @binding(1) var<storage, read_write> saved_row_x: array<u32>;
@group(1) @binding(2) var<storage, read_write> saved_row_z: array<u32>;

// Constants for measurement_data layout
const MEAS_MIN_WEIGHT: u32 = 0u;
const MEAS_CHOSEN_ROW: u32 = 1u;
const MEAS_OUTCOME: u32 = 2u;
const MEAS_QUBIT: u32 = 3u;
const MEAS_WEIGHTS_OFFSET: u32 = 4u;

// =============================================================================
// Stage 1: Compute row weights and find minimum weight anticommuting generator
// =============================================================================
// Each thread handles one generator
@compute @workgroup_size(256)
fn measurement_compute_weights(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gen_idx = global_id.x;

    if (gen_idx >= params.num_gens) {
        return;
    }

    let qubit = atomicLoad(&measurement_data[MEAS_QUBIT]);
    let wb = gen_to_word_bit(gen_idx);

    // Check if this generator has X on the measured qubit (anticommutes)
    let gen_offset = qubit * params.gen_words + wb.x;
    let x_word = stab_x[gen_offset];
    let has_x = get_bit(x_word, wb.y);

    if (!has_x) {
        // Not anticommuting - store max weight to exclude from min finding
        atomicStore(&measurement_data[MEAS_WEIGHTS_OFFSET + gen_idx], 0xFFFFFFFFu);
        return;
    }

    // Compute weight (popcount of X and Z bits across all qubits)
    var weight: u32 = 0u;
    for (var q: u32 = 0u; q < params.num_qubits; q = q + 1u) {
        let q_offset = q * params.gen_words + wb.x;
        // Check if generator gen_idx has X or Z on qubit q
        if (get_bit(stab_x[q_offset], wb.y)) {
            weight = weight + 1u;
        }
        if (get_bit(stab_z[q_offset], wb.y)) {
            weight = weight + 1u;
        }
    }

    // Store weight for this generator
    atomicStore(&measurement_data[MEAS_WEIGHTS_OFFSET + gen_idx], weight);

    // Pack (weight, gen_index) for atomic min comparison
    let packed = (weight << 16u) | gen_idx;
    atomicMin(&measurement_data[MEAS_MIN_WEIGHT], packed);
}

// =============================================================================
// Stage 2: Extract chosen generator and save its data
// =============================================================================
@compute @workgroup_size(1)
fn measurement_extract_chosen() {
    let packed_min = atomicLoad(&measurement_data[MEAS_MIN_WEIGHT]);
    let chosen_gen = packed_min & 0xFFFFu;

    atomicStore(&measurement_data[MEAS_CHOSEN_ROW], chosen_gen);

    let wb = gen_to_word_bit(chosen_gen);

    // Save the chosen generator's X and Z support across all qubits
    // We pack multiple qubits into each u32 word
    let qubit_words = (params.num_qubits + 31u) / 32u;
    for (var w: u32 = 0u; w < qubit_words; w = w + 1u) {
        var x_bits: u32 = 0u;
        var z_bits: u32 = 0u;

        for (var b: u32 = 0u; b < 32u; b = b + 1u) {
            let q = w * 32u + b;
            if (q < params.num_qubits) {
                let q_offset = q * params.gen_words + wb.x;
                if (get_bit(stab_x[q_offset], wb.y)) {
                    x_bits = x_bits | (1u << b);
                }
                if (get_bit(stab_z[q_offset], wb.y)) {
                    z_bits = z_bits | (1u << b);
                }
            }
        }

        saved_row_x[w] = x_bits;
        saved_row_z[w] = z_bits;
    }
}

// =============================================================================
// Stage 3: XOR chosen generator into other anticommuting generators
// =============================================================================
// IMPORTANT: This kernel iterates over (qubit, word_idx) pairs to avoid race conditions.
// Each thread handles one word of data for all qubits, processing all anticommuting
// generators in that word sequentially.
@compute @workgroup_size(256)
fn measurement_xor_rows(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let word_idx = global_id.x;

    if (word_idx >= params.gen_words) {
        return;
    }

    let chosen_gen = atomicLoad(&measurement_data[MEAS_CHOSEN_ROW]);
    let meas_qubit = atomicLoad(&measurement_data[MEAS_QUBIT]);
    let chosen_word = chosen_gen / 32u;
    let chosen_bit = chosen_gen % 32u;

    // Read chosen generator's sign from packed format
    let chosen_has_minus = get_bit(sign_minus[chosen_word], chosen_bit);
    let chosen_has_i = get_bit(sign_i[chosen_word], chosen_bit);

    // For generators in this word, find which ones anticommute (have X on measured qubit)
    let meas_offset = meas_qubit * params.gen_words + word_idx;
    let anticom_mask = stab_x[meas_offset];

    // Exclude the chosen generator if it's in this word
    var active_mask = anticom_mask;
    if (word_idx == chosen_word) {
        active_mask = active_mask & ~(1u << chosen_bit);
    }

    if (active_mask == 0u) {
        return;  // No anticommuting generators in this word (excluding chosen)
    }

    // Handle sign propagation from chosen generator (vectorized for all active generators)
    if (chosen_has_minus) {
        sign_minus[word_idx] = sign_minus[word_idx] ^ active_mask;
    }
    if (chosen_has_i) {
        // If active gen has i and chosen has i: i*i = -1, toggle minus
        let both_i_mask = active_mask & sign_i[word_idx];
        sign_minus[word_idx] = sign_minus[word_idx] ^ both_i_mask;
        // Toggle i for all active generators
        sign_i[word_idx] = sign_i[word_idx] ^ active_mask;
    }

    // Process each anticommuting generator in this word for intersection counting
    let start_gen = word_idx * 32u;
    let end_gen = min(start_gen + 32u, params.num_gens);

    for (var gen_idx: u32 = start_gen; gen_idx < end_gen; gen_idx = gen_idx + 1u) {
        let bit_pos = gen_idx % 32u;

        if (!get_bit(active_mask, bit_pos)) {
            continue;  // This generator doesn't anticommute
        }

        // Count intersections for sign update (Z of chosen * X of current)
        var num_minuses: u32 = 0u;
        let qubit_words = (params.num_qubits + 31u) / 32u;
        for (var w: u32 = 0u; w < qubit_words; w = w + 1u) {
            var current_x_bits: u32 = 0u;
            for (var b: u32 = 0u; b < 32u; b = b + 1u) {
                let q = w * 32u + b;
                if (q < params.num_qubits) {
                    let q_offset = q * params.gen_words + word_idx;
                    if (get_bit(stab_x[q_offset], bit_pos)) {
                        current_x_bits = current_x_bits | (1u << b);
                    }
                }
            }
            num_minuses = num_minuses + countOneBits(saved_row_z[w] & current_x_bits);
        }
        if ((num_minuses % 2u) != 0u) {
            sign_minus[word_idx] = toggle_bit(sign_minus[word_idx], bit_pos);
        }
    }

    // Now XOR the chosen generator's data into all anticommuting generators
    // Process each qubit's word for this word_idx
    for (var q: u32 = 0u; q < params.num_qubits; q = q + 1u) {
        let q_offset = q * params.gen_words + word_idx;
        let qubit_word = q / 32u;
        let qubit_bit = q % 32u;

        // Check if chosen generator has X or Z on qubit q
        let chosen_has_x = get_bit(saved_row_x[qubit_word], qubit_bit);
        let chosen_has_z = get_bit(saved_row_z[qubit_word], qubit_bit);

        // XOR the chosen's data into all anticommuting generators in this word
        if (chosen_has_x) {
            stab_x[q_offset] = stab_x[q_offset] ^ active_mask;
        }
        if (chosen_has_z) {
            stab_z[q_offset] = stab_z[q_offset] ^ active_mask;
        }
    }
}

// =============================================================================
// Stage 4: Update destabilizers - XOR saved generator into anticommuting destabs
// =============================================================================
// Same pattern as Stage 3 - iterate over word_idx to avoid race conditions
@compute @workgroup_size(256)
fn measurement_xor_destabs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let word_idx = global_id.x;

    if (word_idx >= params.gen_words) {
        return;
    }

    let chosen_gen = atomicLoad(&measurement_data[MEAS_CHOSEN_ROW]);
    let meas_qubit = atomicLoad(&measurement_data[MEAS_QUBIT]);
    let chosen_word = chosen_gen / 32u;
    let chosen_bit = chosen_gen % 32u;

    // For destabilizers in this word, find which ones anticommute (have X on measured qubit)
    let meas_offset = meas_qubit * params.gen_words + word_idx;
    let anticom_mask = destab_x[meas_offset];

    // Exclude the chosen generator if it's in this word
    var active_mask = anticom_mask;
    if (word_idx == chosen_word) {
        active_mask = active_mask & ~(1u << chosen_bit);
    }

    if (active_mask == 0u) {
        return;  // No anticommuting destabilizers in this word (excluding chosen)
    }

    // XOR the saved generator's data into all anticommuting destabilizers
    for (var q: u32 = 0u; q < params.num_qubits; q = q + 1u) {
        let q_offset = q * params.gen_words + word_idx;
        let qubit_word = q / 32u;
        let qubit_bit = q % 32u;

        let chosen_has_x = get_bit(saved_row_x[qubit_word], qubit_bit);
        let chosen_has_z = get_bit(saved_row_z[qubit_word], qubit_bit);

        if (chosen_has_x) {
            destab_x[q_offset] = destab_x[q_offset] ^ active_mask;
        }
        if (chosen_has_z) {
            destab_z[q_offset] = destab_z[q_offset] ^ active_mask;
        }
    }
}

// =============================================================================
// Stage 5: Finalize - replace chosen stabilizer with Z_q, update destabilizer
// =============================================================================
@compute @workgroup_size(1)
fn measurement_finalize() {
    let chosen_gen = atomicLoad(&measurement_data[MEAS_CHOSEN_ROW]);
    let qubit = atomicLoad(&measurement_data[MEAS_QUBIT]);
    let outcome = atomicLoad(&measurement_data[MEAS_OUTCOME]);
    let wb = gen_to_word_bit(chosen_gen);

    // Set chosen destabilizer to the saved (removed) stabilizer
    for (var q: u32 = 0u; q < params.num_qubits; q = q + 1u) {
        let q_offset = q * params.gen_words + wb.x;
        let qubit_word = q / 32u;
        let qubit_bit = q % 32u;

        let saved_x = get_bit(saved_row_x[qubit_word], qubit_bit);
        let saved_z = get_bit(saved_row_z[qubit_word], qubit_bit);

        if (saved_x) {
            destab_x[q_offset] = set_bit(destab_x[q_offset], wb.y);
        } else {
            destab_x[q_offset] = clear_bit(destab_x[q_offset], wb.y);
        }

        if (saved_z) {
            destab_z[q_offset] = set_bit(destab_z[q_offset], wb.y);
        } else {
            destab_z[q_offset] = clear_bit(destab_z[q_offset], wb.y);
        }
    }

    // Replace chosen stabilizer with Z_q (only Z on measured qubit)
    for (var q: u32 = 0u; q < params.num_qubits; q = q + 1u) {
        let q_offset = q * params.gen_words + wb.x;

        // Clear X for this generator on all qubits
        stab_x[q_offset] = clear_bit(stab_x[q_offset], wb.y);

        // Set Z only on the measured qubit
        if (q == qubit) {
            stab_z[q_offset] = set_bit(stab_z[q_offset], wb.y);
        } else {
            stab_z[q_offset] = clear_bit(stab_z[q_offset], wb.y);
        }
    }

    // Set sign based on outcome (using packed format)
    // Clear i phase for the new stabilizer
    sign_i[wb.x] = clear_bit(sign_i[wb.x], wb.y);
    // Set or clear minus sign based on outcome
    if (outcome != 0u) {
        sign_minus[wb.x] = set_bit(sign_minus[wb.x], wb.y);  // |1> outcome
    } else {
        sign_minus[wb.x] = clear_bit(sign_minus[wb.x], wb.y);  // |0> outcome
    }
}

// =============================================================================
// Deterministic measurement helper
// =============================================================================
@compute @workgroup_size(256)
fn measurement_count_destab_x(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gen_idx = global_id.x;

    if (gen_idx >= params.num_gens) {
        return;
    }

    let qubit = atomicLoad(&measurement_data[MEAS_QUBIT]);
    let wb = gen_to_word_bit(gen_idx);

    // Check if this destabilizer has X on measured qubit
    let gen_offset = qubit * params.gen_words + wb.x;
    let x_word = destab_x[gen_offset];
    let has_x = get_bit(x_word, wb.y);

    atomicStore(&measurement_data[MEAS_WEIGHTS_OFFSET + gen_idx], select(0u, 1u, has_x));
}

// =============================================================================
// Batched Find Anticommuting - Check multiple qubits in one dispatch
// =============================================================================
// For each qubit in the batch, find the first generator that anticommutes.
// This allows measuring many qubits with a single GPU dispatch.
//
// Input:
//   batch_qubits[0] = number of qubits to measure
//   batch_qubits[1..N+1] = qubit indices to measure
//
// Output:
//   batch_results[i] = first anticommuting generator for qubit i, or 0xFFFFFFFF if none

@group(1) @binding(1) var<storage, read> batch_qubits: array<u32>;
@group(1) @binding(2) var<storage, read_write> batch_results: array<atomic<u32>>;
@group(1) @binding(3) var<storage, read> batch_random: array<u32>;

// Optimized anticommuting detection - one thread per measured qubit
// Instead of iterating over generators, scan words to find first anticommuting generator
@compute @workgroup_size(256)
fn find_anticommuting_batch(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    let num_batch_qubits = batch_qubits[0];

    if (batch_idx >= num_batch_qubits) {
        return;
    }

    let qubit = batch_qubits[batch_idx + 1u];
    let row_base = qubit * params.gen_words;

    // Scan through words to find first anticommuting generator
    // (first non-zero word, then first set bit in that word)
    for (var w: u32 = 0u; w < params.gen_words; w = w + 1u) {
        let x_word = stab_x[row_base + w];

        if (x_word != 0u) {
            // Found a word with anticommuting generators
            let first_bit = countTrailingZeros(x_word);
            let first_gen = w * 32u + first_bit;

            // Only update if this generator is smaller than current result
            atomicMin(&batch_results[batch_idx], first_gen);
            return;  // Found the first, no need to continue
        }
    }
    // If no anticommuting generator found, result stays at 0xFFFFFFFF (deterministic)
}

// =============================================================================
// Compute All Measurement Outcomes on GPU
// =============================================================================
// Computes outcomes for all measurements in the batch:
// - Deterministic measurements: compute from stabilizer tableau
// - Non-deterministic measurements: use pre-generated random bit from batch_random
//
// Input:
//   batch_qubits[0] = number of qubits
//   batch_qubits[1..N+1] = qubit indices
//   batch_results[i] = 0xFFFFFFFF if deterministic, else anticommuting generator index
//   batch_random[i] = pre-generated random bit (0 or 1) for non-deterministic cases
//
// Output:
//   batch_results[i] = outcome (0 or 1) for all qubits

@compute @workgroup_size(256)
fn compute_deterministic_outcomes(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    let num_batch_qubits = batch_qubits[0];

    if (batch_idx >= num_batch_qubits) {
        return;
    }

    // Check if this qubit is deterministic (no anticommuting generator found)
    let anticom_result = atomicLoad(&batch_results[batch_idx]);
    if (anticom_result != 0xFFFFFFFFu) {
        // Non-deterministic - use pre-generated random bit
        let random_outcome = batch_random[batch_idx] & 1u;
        atomicStore(&batch_results[batch_idx], random_outcome);
        return;
    }

    let qubit = batch_qubits[batch_idx + 1u];

    // Compute deterministic outcome using optimized algorithm
    // Phase 1: Find all contributing generators (those with destab X on measured qubit)
    // and count signs, stored as a bitmask per word
    var num_minuses: u32 = 0u;
    var num_is: u32 = 0u;

    // First pass: collect contributing generators and count signs
    // We process in word chunks for efficiency
    for (var w: u32 = 0u; w < params.gen_words; w = w + 1u) {
        let destab_offset = qubit * params.gen_words + w;
        let contrib_mask = destab_x[destab_offset];  // Generators in this word that contribute

        if (contrib_mask == 0u) {
            continue;  // No contributing generators in this word
        }

        // Count sign contributions from these generators
        num_minuses = num_minuses + countOneBits(contrib_mask & sign_minus[w]);
        num_is = num_is + countOneBits(contrib_mask & sign_i[w]);
    }

    // Phase 2: Count pairwise phase contributions
    // For each pair of contributing generators (prev, curr) where prev < curr,
    // count qubits where X(prev) AND Z(curr)
    //
    // Optimization: Process by word pairs to reduce iteration overhead
    for (var w_curr: u32 = 0u; w_curr < params.gen_words; w_curr = w_curr + 1u) {
        let destab_offset_curr = qubit * params.gen_words + w_curr;
        let contrib_curr = destab_x[destab_offset_curr];

        if (contrib_curr == 0u) {
            continue;
        }

        // Process each current generator in this word
        var curr_mask = contrib_curr;
        while (curr_mask != 0u) {
            let curr_bit = countTrailingZeros(curr_mask);
            let curr_gen = w_curr * 32u + curr_bit;
            curr_mask = curr_mask & (curr_mask - 1u);  // Clear lowest set bit

            // Count phase contributions from all previous contributing generators
            for (var w_prev: u32 = 0u; w_prev <= w_curr; w_prev = w_prev + 1u) {
                let destab_offset_prev = qubit * params.gen_words + w_prev;
                var contrib_prev = destab_x[destab_offset_prev];

                // For same word, only consider bits before curr_bit
                if (w_prev == w_curr) {
                    contrib_prev = contrib_prev & ((1u << curr_bit) - 1u);
                }

                if (contrib_prev == 0u) {
                    continue;
                }

                // For each previous contributing generator
                var prev_mask = contrib_prev;
                while (prev_mask != 0u) {
                    let prev_bit = countTrailingZeros(prev_mask);
                    prev_mask = prev_mask & (prev_mask - 1u);

                    // Count X(prev) AND Z(curr) across all qubits
                    // Optimized: iterate over qubits in word chunks
                    var intersection_count: u32 = 0u;
                    for (var q: u32 = 0u; q < params.num_qubits; q = q + 1u) {
                        let q_offset_prev = q * params.gen_words + w_prev;
                        let q_offset_curr = q * params.gen_words + w_curr;

                        let prev_has_x = get_bit(stab_x[q_offset_prev], prev_bit);
                        let curr_has_z = get_bit(stab_z[q_offset_curr], curr_bit);

                        if (prev_has_x && curr_has_z) {
                            intersection_count = intersection_count + 1u;
                        }
                    }
                    num_minuses = num_minuses + intersection_count;
                }
            }
        }
    }

    // Handle i phase: add 1 to minuses if num_is % 4 != 0
    // (matching CPU implementation)
    if ((num_is & 3u) != 0u) {
        num_minuses = num_minuses + 1u;
    }

    // Outcome is 1 if num_minuses is odd
    let outcome = num_minuses & 1u;
    atomicStore(&batch_results[batch_idx], outcome);
}
