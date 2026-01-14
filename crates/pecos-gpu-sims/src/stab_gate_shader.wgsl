// GPU Stabilizer Gate Shader
//
// Processes a queue of gates in a single dispatch for efficiency.
// Each thread handles one word_idx and processes all gates sequentially.
// No cross-workgroup barriers needed because different words don't share data.
//
// Signs are packed as bits: sign_minus[word_idx] and sign_i[word_idx] each
// contain one bit per generator, enabling bitwise sign updates.

// Stabilizer tableau buffers
@group(0) @binding(0) var<storage, read_write> stab_x: array<u32>;
@group(0) @binding(1) var<storage, read_write> stab_z: array<u32>;
@group(0) @binding(2) var<storage, read_write> destab_x: array<u32>;
@group(0) @binding(3) var<storage, read_write> destab_z: array<u32>;

// Packed sign bits: one bit per generator
// sign_minus[word_idx] bit i = minus sign for generator (word_idx * 32 + i)
// sign_i[word_idx] bit i = i phase for generator (word_idx * 32 + i)
@group(0) @binding(4) var<storage, read_write> sign_minus: array<u32>;
@group(0) @binding(7) var<storage, read_write> sign_i: array<u32>;

// Parameters (written once, not per-flush)
struct PersistentParams {
    num_qubits: u32,
    gen_words: u32,
    num_gens: u32,
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
    _padding4: u32,
    _padding5: u32,
}

@group(0) @binding(5) var<uniform> params: PersistentParams;

// Gate queue layout:
// [0]: num_gates (updated per-flush)
// [1..]: packed gates - each gate is one u32:
//   bits 0-3: gate type
//   bits 4-17: target qubit (14 bits, up to 16384 qubits)
//   bits 18-31: control qubit (14 bits, for 2-qubit gates)
@group(0) @binding(6) var<storage, read> gate_queue: array<u32>;

// Gate type constants
const GATE_H: u32 = 0u;
const GATE_S: u32 = 1u;
const GATE_SDG: u32 = 2u;
const GATE_X: u32 = 3u;
const GATE_Y: u32 = 4u;
const GATE_Z: u32 = 5u;
const GATE_CX: u32 = 6u;
const GATE_CZ: u32 = 7u;
const GATE_SWAP: u32 = 8u;

// Decode gate from packed format
fn decode_gate(packed: u32) -> vec3<u32> {
    let gate_type = packed & 0xFu;
    let tgt = (packed >> 4u) & 0x3FFFu;
    let ctrl = (packed >> 18u) & 0x3FFFu;
    return vec3<u32>(gate_type, tgt, ctrl);
}

// =============================================================================
// Inline gate implementations with packed sign updates
// =============================================================================

fn apply_h_inline(word_idx: u32, q: u32) {
    let row_offset = q * params.gen_words + word_idx;

    let orig_stab_x = stab_x[row_offset];
    let orig_stab_z = stab_z[row_offset];

    // Swap X and Z
    stab_x[row_offset] = orig_stab_z;
    stab_z[row_offset] = orig_stab_x;

    let destab_x_word = destab_x[row_offset];
    let destab_z_word = destab_z[row_offset];
    destab_x[row_offset] = destab_z_word;
    destab_z[row_offset] = destab_x_word;

    // H: flip minus sign when both X and Z were set (Y -> -Y)
    sign_minus[word_idx] ^= (orig_stab_x & orig_stab_z);
}

fn apply_s_inline(word_idx: u32, q: u32) {
    let row_offset = q * params.gen_words + word_idx;

    let orig_stab_x = stab_x[row_offset];
    let orig_stab_z = stab_z[row_offset];

    // S: Z -> Z, X -> XZ (Y with i phase)
    stab_z[row_offset] = orig_stab_z ^ orig_stab_x;

    let orig_destab_x = destab_x[row_offset];
    let orig_destab_z = destab_z[row_offset];
    destab_z[row_offset] = orig_destab_z ^ orig_destab_x;

    // S: when X is set, add i phase
    // When both X and Z were set (was Y), also flip minus (i * i = -1 component)
    let had_xz = orig_stab_x & orig_stab_z;
    sign_minus[word_idx] ^= had_xz;
    sign_i[word_idx] ^= orig_stab_x;
}

fn apply_sdg_inline(word_idx: u32, q: u32) {
    let row_offset = q * params.gen_words + word_idx;

    let orig_stab_x = stab_x[row_offset];
    let orig_stab_z = stab_z[row_offset];

    stab_z[row_offset] = orig_stab_z ^ orig_stab_x;

    let orig_destab_x = destab_x[row_offset];
    let orig_destab_z = destab_z[row_offset];
    destab_z[row_offset] = orig_destab_z ^ orig_destab_x;

    // S†: when X is set, add -i phase (i^3)
    // -i = toggle i, and if already had i, also toggle minus
    // For each generator with X: if sign_i was set, toggle minus; then toggle i
    let had_i = sign_i[word_idx];
    sign_minus[word_idx] ^= (orig_stab_x & had_i);
    sign_i[word_idx] ^= orig_stab_x;
}

fn apply_x_inline(word_idx: u32, q: u32) {
    let row_offset = q * params.gen_words + word_idx;
    let stab_z_word = stab_z[row_offset];

    // X: flip minus sign when Z is present
    sign_minus[word_idx] ^= stab_z_word;
}

fn apply_y_inline(word_idx: u32, q: u32) {
    let row_offset = q * params.gen_words + word_idx;

    let stab_x_word = stab_x[row_offset];
    let stab_z_word = stab_z[row_offset];

    // Y: flip minus sign when exactly one of X or Z (XOR)
    sign_minus[word_idx] ^= (stab_x_word ^ stab_z_word);
}

fn apply_z_inline(word_idx: u32, q: u32) {
    let row_offset = q * params.gen_words + word_idx;
    let stab_x_word = stab_x[row_offset];

    // Z: flip minus sign when X is present
    sign_minus[word_idx] ^= stab_x_word;
}

fn apply_cx_inline(word_idx: u32, ctrl: u32, tgt: u32) {
    let ctrl_offset = ctrl * params.gen_words + word_idx;
    let tgt_offset = tgt * params.gen_words + word_idx;

    // Read before update
    let ctrl_x = stab_x[ctrl_offset];
    let tgt_z = stab_z[tgt_offset];

    // CX: X_tgt ^= X_ctrl, Z_ctrl ^= Z_tgt
    stab_x[tgt_offset] = stab_x[tgt_offset] ^ ctrl_x;
    stab_z[ctrl_offset] = stab_z[ctrl_offset] ^ tgt_z;

    let ctrl_destab_x = destab_x[ctrl_offset];
    let tgt_destab_z = destab_z[tgt_offset];

    destab_x[tgt_offset] = destab_x[tgt_offset] ^ ctrl_destab_x;
    destab_z[ctrl_offset] = destab_z[ctrl_offset] ^ tgt_destab_z;

    // Sign update: flip when ctrl_x AND tgt_z AND (ctrl_z_new == tgt_x_new)
    // ctrl_z_new = ctrl_z ^ tgt_z, tgt_x_new = tgt_x ^ ctrl_x
    // (cz_new == tx_new) means they are both 0 or both 1, i.e., NOT(cz_new XOR tx_new)
    let ctrl_z_new = stab_z[ctrl_offset];
    let tgt_x_new = stab_x[tgt_offset];
    let same = ~(ctrl_z_new ^ tgt_x_new);
    sign_minus[word_idx] ^= (ctrl_x & tgt_z & same);
}

fn apply_cz_inline(word_idx: u32, a: u32, b: u32) {
    let a_offset = a * params.gen_words + word_idx;
    let b_offset = b * params.gen_words + word_idx;

    let a_x = stab_x[a_offset];
    let b_x = stab_x[b_offset];

    // CZ: Z_a ^= X_b, Z_b ^= X_a
    stab_z[a_offset] = stab_z[a_offset] ^ b_x;
    stab_z[b_offset] = stab_z[b_offset] ^ a_x;

    let a_destab_x = destab_x[a_offset];
    let b_destab_x = destab_x[b_offset];

    destab_z[a_offset] = destab_z[a_offset] ^ b_destab_x;
    destab_z[b_offset] = destab_z[b_offset] ^ a_destab_x;

    // Sign update: flip when both have X AND (az_new == bz_new)
    let az_new = stab_z[a_offset];
    let bz_new = stab_z[b_offset];
    let same = ~(az_new ^ bz_new);
    sign_minus[word_idx] ^= (a_x & b_x & same);
}

fn apply_swap_inline(word_idx: u32, a: u32, b: u32) {
    let a_offset = a * params.gen_words + word_idx;
    let b_offset = b * params.gen_words + word_idx;

    // Swap all arrays
    let tmp_stab_x = stab_x[a_offset];
    stab_x[a_offset] = stab_x[b_offset];
    stab_x[b_offset] = tmp_stab_x;

    let tmp_stab_z = stab_z[a_offset];
    stab_z[a_offset] = stab_z[b_offset];
    stab_z[b_offset] = tmp_stab_z;

    let tmp_destab_x = destab_x[a_offset];
    destab_x[a_offset] = destab_x[b_offset];
    destab_x[b_offset] = tmp_destab_x;

    let tmp_destab_z = destab_z[a_offset];
    destab_z[a_offset] = destab_z[b_offset];
    destab_z[b_offset] = tmp_destab_z;

    // SWAP has no sign updates
}

// =============================================================================
// Main kernel
// =============================================================================

@compute @workgroup_size(256)
fn process_gate_queue(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let word_idx = global_id.x;

    if (word_idx >= params.gen_words) {
        return;
    }

    // Read num_gates from first element of gate queue
    let num_gates = gate_queue[0];

    // Process all gates in sequence (gates start at index 1)
    for (var gate_idx: u32 = 0u; gate_idx < num_gates; gate_idx = gate_idx + 1u) {
        let gate = decode_gate(gate_queue[gate_idx + 1u]);
        let gate_type = gate.x;
        let tgt_qubit = gate.y;
        let ctrl_qubit = gate.z;

        switch (gate_type) {
            case GATE_H: {
                apply_h_inline(word_idx, tgt_qubit);
            }
            case GATE_S: {
                apply_s_inline(word_idx, tgt_qubit);
            }
            case GATE_SDG: {
                apply_sdg_inline(word_idx, tgt_qubit);
            }
            case GATE_X: {
                apply_x_inline(word_idx, tgt_qubit);
            }
            case GATE_Y: {
                apply_y_inline(word_idx, tgt_qubit);
            }
            case GATE_Z: {
                apply_z_inline(word_idx, tgt_qubit);
            }
            case GATE_CX: {
                apply_cx_inline(word_idx, ctrl_qubit, tgt_qubit);
            }
            case GATE_CZ: {
                apply_cz_inline(word_idx, ctrl_qubit, tgt_qubit);
            }
            case GATE_SWAP: {
                apply_swap_inline(word_idx, ctrl_qubit, tgt_qubit);
            }
            default: {}
        }
    }
}
