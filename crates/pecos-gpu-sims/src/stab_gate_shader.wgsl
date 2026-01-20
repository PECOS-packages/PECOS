// GPU Stabilizer Gate Shader
//
// Processes a queue of gates in a single dispatch for efficiency.
// Each thread handles one word_idx and processes all gates sequentially.
// No cross-workgroup barriers needed because different words don't share data.
//
// Signs are packed as bits: sign_minus[word_idx] and sign_i[word_idx] each
// contain one bit per generator, enabling bitwise sign updates.
//
// OPTIMIZATION: Signs are cached in local variables to reduce global memory traffic.
// Only one read at start and one write at end, instead of per-gate reads/writes.

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
// Main kernel with sign caching optimization
// =============================================================================

// Shared memory for broadcasting num_gates to all threads in workgroup
var<workgroup> shared_num_gates: u32;

@compute @workgroup_size(256)
fn process_gate_queue(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32
) {
    // First thread loads num_gates, then all threads sync
    if (local_idx == 0u) {
        shared_num_gates = gate_queue[0];
    }
    workgroupBarrier();

    let word_idx = global_id.x;
    if (word_idx >= params.gen_words) {
        return;
    }

    let num_gates = shared_num_gates;
    if (num_gates == 0u) {
        return;
    }

    // Cache signs in local variables - reduces global memory traffic
    var local_sign_minus = sign_minus[word_idx];
    var local_sign_i = sign_i[word_idx];

    // Process all gates in sequence (gates start at index 1)
    for (var gate_idx: u32 = 0u; gate_idx < num_gates; gate_idx = gate_idx + 1u) {
        let gate = decode_gate(gate_queue[gate_idx + 1u]);
        let gate_type = gate.x;
        let tgt_qubit = gate.y;
        let ctrl_qubit = gate.z;

        switch (gate_type) {
            case GATE_H: {
                let row_offset = tgt_qubit * params.gen_words + word_idx;
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
                local_sign_minus ^= (orig_stab_x & orig_stab_z);
            }
            case GATE_S: {
                let row_offset = tgt_qubit * params.gen_words + word_idx;
                let orig_stab_x = stab_x[row_offset];
                let orig_stab_z = stab_z[row_offset];
                // S: Z -> Z, X -> XZ (Y with i phase)
                stab_z[row_offset] = orig_stab_z ^ orig_stab_x;
                let orig_destab_x = destab_x[row_offset];
                let orig_destab_z = destab_z[row_offset];
                destab_z[row_offset] = orig_destab_z ^ orig_destab_x;
                // S: when X is set with existing i phase, toggle minus (i*i = -1)
                // Then toggle i phase for all with X
                let toggle_minus_mask = orig_stab_x & local_sign_i;
                local_sign_minus ^= toggle_minus_mask;
                local_sign_i ^= orig_stab_x;
            }
            case GATE_SDG: {
                let row_offset = tgt_qubit * params.gen_words + word_idx;
                let orig_stab_x = stab_x[row_offset];
                let orig_stab_z = stab_z[row_offset];
                stab_z[row_offset] = orig_stab_z ^ orig_stab_x;
                let orig_destab_x = destab_x[row_offset];
                let orig_destab_z = destab_z[row_offset];
                destab_z[row_offset] = orig_destab_z ^ orig_destab_x;
                // S†: Sdg multiplies phase by -i when X=1: flip sign_minus when sign_i was 0
                let had_i = local_sign_i;
                local_sign_minus ^= (orig_stab_x & ~had_i);
                local_sign_i ^= orig_stab_x;
            }
            case GATE_X: {
                let row_offset = tgt_qubit * params.gen_words + word_idx;
                let stab_z_word = stab_z[row_offset];
                // X: flip minus sign when Z is present
                local_sign_minus ^= stab_z_word;
            }
            case GATE_Y: {
                let row_offset = tgt_qubit * params.gen_words + word_idx;
                let stab_x_word = stab_x[row_offset];
                let stab_z_word = stab_z[row_offset];
                // Y: flip minus sign when exactly one of X or Z (XOR)
                local_sign_minus ^= (stab_x_word ^ stab_z_word);
            }
            case GATE_Z: {
                let row_offset = tgt_qubit * params.gen_words + word_idx;
                let stab_x_word = stab_x[row_offset];
                // Z: flip minus sign when X is present
                local_sign_minus ^= stab_x_word;
            }
            case GATE_CX: {
                let ctrl_offset = ctrl_qubit * params.gen_words + word_idx;
                let tgt_offset = tgt_qubit * params.gen_words + word_idx;
                // CX: X_tgt ^= X_ctrl, Z_ctrl ^= Z_tgt
                stab_x[tgt_offset] = stab_x[tgt_offset] ^ stab_x[ctrl_offset];
                stab_z[ctrl_offset] = stab_z[ctrl_offset] ^ stab_z[tgt_offset];
                destab_x[tgt_offset] = destab_x[tgt_offset] ^ destab_x[ctrl_offset];
                destab_z[ctrl_offset] = destab_z[ctrl_offset] ^ destab_z[tgt_offset];
                // CX does NOT require sign updates
            }
            case GATE_CZ: {
                let a_offset = ctrl_qubit * params.gen_words + word_idx;
                let b_offset = tgt_qubit * params.gen_words + word_idx;
                let a_x = stab_x[a_offset];
                let b_x = stab_x[b_offset];
                // CZ: Z_a ^= X_b, Z_b ^= X_a
                stab_z[a_offset] = stab_z[a_offset] ^ b_x;
                stab_z[b_offset] = stab_z[b_offset] ^ a_x;
                let a_destab_x = destab_x[a_offset];
                let b_destab_x = destab_x[b_offset];
                destab_z[a_offset] = destab_z[a_offset] ^ b_destab_x;
                destab_z[b_offset] = destab_z[b_offset] ^ a_destab_x;
                // Sign update: CZ flips sign when both qubits have X
                local_sign_minus ^= (a_x & b_x);
            }
            case GATE_SWAP: {
                let a_offset = ctrl_qubit * params.gen_words + word_idx;
                let b_offset = tgt_qubit * params.gen_words + word_idx;
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
            default: {}
        }
    }

    // Write cached signs back to global memory (once instead of per-gate)
    sign_minus[word_idx] = local_sign_minus;
    sign_i[word_idx] = local_sign_i;
}
