// GPU Stabilizer Gate Shader - Interleaved Memory Layout
//
// This version uses an interleaved memory layout where X and Z components
// for the same qubit/word are adjacent in memory, improving cache locality.
//
// Layout: stab[qubit * gen_words * 2 + word_idx * 2 + 0] = X
//         stab[qubit * gen_words * 2 + word_idx * 2 + 1] = Z
//
// Same for destab buffer.

// Interleaved tableau buffers (X and Z pairs)
@group(0) @binding(0) var<storage, read_write> stab: array<u32>;    // Interleaved X/Z
@group(0) @binding(1) var<storage, read_write> destab: array<u32>;  // Interleaved X/Z

// Packed sign bits
@group(0) @binding(2) var<storage, read_write> sign_minus: array<u32>;
@group(0) @binding(3) var<storage, read_write> sign_i: array<u32>;

// Parameters
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

@group(0) @binding(4) var<uniform> params: PersistentParams;

// Gate queue
@group(0) @binding(5) var<storage, read> gate_queue: array<u32>;

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

fn decode_gate(packed: u32) -> vec3<u32> {
    let gate_type = packed & 0xFu;
    let tgt = (packed >> 4u) & 0x3FFFu;
    let ctrl = (packed >> 18u) & 0x3FFFu;
    return vec3<u32>(gate_type, tgt, ctrl);
}

// Helper to compute interleaved offset for X component
fn x_offset(qubit: u32, word_idx: u32, gen_words: u32) -> u32 {
    return qubit * gen_words * 2u + word_idx * 2u;
}

// Helper to compute interleaved offset for Z component
fn z_offset(qubit: u32, word_idx: u32, gen_words: u32) -> u32 {
    return qubit * gen_words * 2u + word_idx * 2u + 1u;
}

var<workgroup> shared_num_gates: u32;

@compute @workgroup_size(256)
fn process_gate_queue_interleaved(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32
) {
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

    var local_sign_minus = sign_minus[word_idx];
    var local_sign_i = sign_i[word_idx];

    let gw = params.gen_words;

    for (var gate_idx: u32 = 0u; gate_idx < num_gates; gate_idx = gate_idx + 1u) {
        let gate = decode_gate(gate_queue[gate_idx + 1u]);
        let gate_type = gate.x;
        let tgt_qubit = gate.y;
        let ctrl_qubit = gate.z;

        switch (gate_type) {
            case GATE_H: {
                let x_off = x_offset(tgt_qubit, word_idx, gw);
                let z_off = z_offset(tgt_qubit, word_idx, gw);
                // Read X and Z (adjacent in memory!)
                let orig_x = stab[x_off];
                let orig_z = stab[z_off];
                // Swap
                stab[x_off] = orig_z;
                stab[z_off] = orig_x;
                // Destab
                let dx_off = x_offset(tgt_qubit, word_idx, gw);
                let dz_off = z_offset(tgt_qubit, word_idx, gw);
                let orig_dx = destab[dx_off];
                let orig_dz = destab[dz_off];
                destab[dx_off] = orig_dz;
                destab[dz_off] = orig_dx;
                // Sign
                local_sign_minus ^= (orig_x & orig_z);
            }
            case GATE_S: {
                let x_off = x_offset(tgt_qubit, word_idx, gw);
                let z_off = z_offset(tgt_qubit, word_idx, gw);
                let orig_x = stab[x_off];
                let orig_z = stab[z_off];
                stab[z_off] = orig_z ^ orig_x;
                let dx_off = x_offset(tgt_qubit, word_idx, gw);
                let dz_off = z_offset(tgt_qubit, word_idx, gw);
                let orig_dx = destab[dx_off];
                let orig_dz = destab[dz_off];
                destab[dz_off] = orig_dz ^ orig_dx;
                // S: when X is set with existing i phase, toggle minus (i*i = -1)
                let toggle_minus_mask = orig_x & local_sign_i;
                local_sign_minus ^= toggle_minus_mask;
                local_sign_i ^= orig_x;
            }
            case GATE_SDG: {
                let x_off = x_offset(tgt_qubit, word_idx, gw);
                let z_off = z_offset(tgt_qubit, word_idx, gw);
                let orig_x = stab[x_off];
                let orig_z = stab[z_off];
                stab[z_off] = orig_z ^ orig_x;
                let dx_off = x_offset(tgt_qubit, word_idx, gw);
                let dz_off = z_offset(tgt_qubit, word_idx, gw);
                let orig_dx = destab[dx_off];
                let orig_dz = destab[dz_off];
                destab[dz_off] = orig_dz ^ orig_dx;
                // Sdg multiplies phase by -i when X=1: flip sign_minus when sign_i was 0
                let had_i = local_sign_i;
                local_sign_minus ^= (orig_x & ~had_i);
                local_sign_i ^= orig_x;
            }
            case GATE_X: {
                let z_off = z_offset(tgt_qubit, word_idx, gw);
                let stab_z_word = stab[z_off];
                local_sign_minus ^= stab_z_word;
            }
            case GATE_Y: {
                let x_off = x_offset(tgt_qubit, word_idx, gw);
                let z_off = z_offset(tgt_qubit, word_idx, gw);
                let stab_x_word = stab[x_off];
                let stab_z_word = stab[z_off];
                local_sign_minus ^= (stab_x_word ^ stab_z_word);
            }
            case GATE_Z: {
                let x_off = x_offset(tgt_qubit, word_idx, gw);
                let stab_x_word = stab[x_off];
                local_sign_minus ^= stab_x_word;
            }
            case GATE_CX: {
                let ctrl_x_off = x_offset(ctrl_qubit, word_idx, gw);
                let ctrl_z_off = z_offset(ctrl_qubit, word_idx, gw);
                let tgt_x_off = x_offset(tgt_qubit, word_idx, gw);
                let tgt_z_off = z_offset(tgt_qubit, word_idx, gw);

                // CX: X_tgt ^= X_ctrl, Z_ctrl ^= Z_tgt
                stab[tgt_x_off] = stab[tgt_x_off] ^ stab[ctrl_x_off];
                stab[ctrl_z_off] = stab[ctrl_z_off] ^ stab[tgt_z_off];

                destab[tgt_x_off] = destab[tgt_x_off] ^ destab[ctrl_x_off];
                destab[ctrl_z_off] = destab[ctrl_z_off] ^ destab[tgt_z_off];
                // CX does NOT require sign updates
            }
            case GATE_CZ: {
                let a_x_off = x_offset(ctrl_qubit, word_idx, gw);
                let a_z_off = z_offset(ctrl_qubit, word_idx, gw);
                let b_x_off = x_offset(tgt_qubit, word_idx, gw);
                let b_z_off = z_offset(tgt_qubit, word_idx, gw);

                let a_x = stab[a_x_off];
                let b_x = stab[b_x_off];

                stab[a_z_off] = stab[a_z_off] ^ b_x;
                stab[b_z_off] = stab[b_z_off] ^ a_x;

                let a_dx = destab[a_x_off];
                let b_dx = destab[b_x_off];
                destab[a_z_off] = destab[a_z_off] ^ b_dx;
                destab[b_z_off] = destab[b_z_off] ^ a_dx;

                // CZ: flip sign when both qubits have X
                local_sign_minus ^= (a_x & b_x);
            }
            case GATE_SWAP: {
                let a_x_off = x_offset(ctrl_qubit, word_idx, gw);
                let a_z_off = z_offset(ctrl_qubit, word_idx, gw);
                let b_x_off = x_offset(tgt_qubit, word_idx, gw);
                let b_z_off = z_offset(tgt_qubit, word_idx, gw);

                let tmp_x = stab[a_x_off];
                let tmp_z = stab[a_z_off];
                stab[a_x_off] = stab[b_x_off];
                stab[a_z_off] = stab[b_z_off];
                stab[b_x_off] = tmp_x;
                stab[b_z_off] = tmp_z;

                let tmp_dx = destab[a_x_off];
                let tmp_dz = destab[a_z_off];
                destab[a_x_off] = destab[b_x_off];
                destab[a_z_off] = destab[b_z_off];
                destab[b_x_off] = tmp_dx;
                destab[b_z_off] = tmp_dz;
            }
            default: {}
        }
    }

    sign_minus[word_idx] = local_sign_minus;
    sign_i[word_idx] = local_sign_i;
}
