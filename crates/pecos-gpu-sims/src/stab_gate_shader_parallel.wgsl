// GPU Stabilizer Gate Shader - Gate-Parallel Version
//
// Processes a batch of INDEPENDENT gates in parallel.
// Each thread handles one (gate_idx, word_idx) pair.
// Gates in the batch must not share any qubits.
//
// This dramatically increases parallelism:
// - Original: gen_words threads (56 at d=21)
// - Parallel: num_gates * gen_words threads (e.g., 100 gates * 56 = 5,600 threads)
//
// The sign update is tricky because it's per-word, not per-gate.
// We use atomic XOR for sign updates since XOR is commutative.

@group(0) @binding(0) var<storage, read_write> stab_x: array<u32>;
@group(0) @binding(1) var<storage, read_write> stab_z: array<u32>;
@group(0) @binding(2) var<storage, read_write> destab_x: array<u32>;
@group(0) @binding(3) var<storage, read_write> destab_z: array<u32>;

@group(0) @binding(4) var<storage, read_write> sign_minus: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> sign_i: array<atomic<u32>>;

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

// Gate queue: [0] = num_gates, [1..] = packed gates
@group(0) @binding(6) var<storage, read> gate_queue: array<u32>;

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

var<workgroup> shared_num_gates: u32;

// Process independent gates in parallel
// Thread ID = gate_idx * gen_words + word_idx
@compute @workgroup_size(256)
fn process_gates_parallel(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32
) {
    if (local_idx == 0u) {
        shared_num_gates = gate_queue[0];
    }
    workgroupBarrier();

    let num_gates = shared_num_gates;
    if (num_gates == 0u) {
        return;
    }

    let thread_id = global_id.x;
    let total_threads = num_gates * params.gen_words;

    if (thread_id >= total_threads) {
        return;
    }

    // Decode which gate and word this thread handles
    let gate_idx = thread_id / params.gen_words;
    let word_idx = thread_id % params.gen_words;

    let gate = decode_gate(gate_queue[gate_idx + 1u]);
    let gate_type = gate.x;
    let tgt_qubit = gate.y;
    let ctrl_qubit = gate.z;

    // Process this single gate for this single word
    switch (gate_type) {
        case GATE_H: {
            let row_offset = tgt_qubit * params.gen_words + word_idx;
            let orig_stab_x = stab_x[row_offset];
            let orig_stab_z = stab_z[row_offset];
            stab_x[row_offset] = orig_stab_z;
            stab_z[row_offset] = orig_stab_x;
            let destab_x_word = destab_x[row_offset];
            let destab_z_word = destab_z[row_offset];
            destab_x[row_offset] = destab_z_word;
            destab_z[row_offset] = destab_x_word;
            // Atomic XOR for sign update
            let sign_update = orig_stab_x & orig_stab_z;
            if (sign_update != 0u) {
                atomicXor(&sign_minus[word_idx], sign_update);
            }
        }
        case GATE_S: {
            let row_offset = tgt_qubit * params.gen_words + word_idx;
            let orig_stab_x = stab_x[row_offset];
            let orig_stab_z = stab_z[row_offset];
            stab_z[row_offset] = orig_stab_z ^ orig_stab_x;
            let orig_destab_x = destab_x[row_offset];
            let orig_destab_z = destab_z[row_offset];
            destab_z[row_offset] = orig_destab_z ^ orig_destab_x;
            // S: when X is set with existing i phase, toggle minus (i*i = -1)
            let had_i = atomicLoad(&sign_i[word_idx]);
            let toggle_minus_mask = orig_stab_x & had_i;
            if (toggle_minus_mask != 0u) {
                atomicXor(&sign_minus[word_idx], toggle_minus_mask);
            }
            if (orig_stab_x != 0u) {
                atomicXor(&sign_i[word_idx], orig_stab_x);
            }
        }
        case GATE_SDG: {
            let row_offset = tgt_qubit * params.gen_words + word_idx;
            let orig_stab_x = stab_x[row_offset];
            let orig_stab_z = stab_z[row_offset];
            stab_z[row_offset] = orig_stab_z ^ orig_stab_x;
            let orig_destab_x = destab_x[row_offset];
            let orig_destab_z = destab_z[row_offset];
            destab_z[row_offset] = orig_destab_z ^ orig_destab_x;
            // SDG sign update: Sdg multiplies phase by -i when X=1
            // Flip sign_minus when sign_i was 0 (i.e., use ~had_i)
            // This is problematic for parallel execution - sign_i may be updated by other gates
            // For now, read the atomic value (may have race conditions with other SDG gates)
            let had_i = atomicLoad(&sign_i[word_idx]);
            let sign_minus_update = orig_stab_x & ~had_i;
            if (sign_minus_update != 0u) {
                atomicXor(&sign_minus[word_idx], sign_minus_update);
            }
            if (orig_stab_x != 0u) {
                atomicXor(&sign_i[word_idx], orig_stab_x);
            }
        }
        case GATE_X: {
            let row_offset = tgt_qubit * params.gen_words + word_idx;
            let stab_z_word = stab_z[row_offset];
            if (stab_z_word != 0u) {
                atomicXor(&sign_minus[word_idx], stab_z_word);
            }
        }
        case GATE_Y: {
            let row_offset = tgt_qubit * params.gen_words + word_idx;
            let stab_x_word = stab_x[row_offset];
            let stab_z_word = stab_z[row_offset];
            let sign_update = stab_x_word ^ stab_z_word;
            if (sign_update != 0u) {
                atomicXor(&sign_minus[word_idx], sign_update);
            }
        }
        case GATE_Z: {
            let row_offset = tgt_qubit * params.gen_words + word_idx;
            let stab_x_word = stab_x[row_offset];
            if (stab_x_word != 0u) {
                atomicXor(&sign_minus[word_idx], stab_x_word);
            }
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
            stab_z[a_offset] = stab_z[a_offset] ^ b_x;
            stab_z[b_offset] = stab_z[b_offset] ^ a_x;
            let a_destab_x = destab_x[a_offset];
            let b_destab_x = destab_x[b_offset];
            destab_z[a_offset] = destab_z[a_offset] ^ b_destab_x;
            destab_z[b_offset] = destab_z[b_offset] ^ a_destab_x;
            // CZ: flip sign when both qubits have X
            let sign_update = a_x & b_x;
            if (sign_update != 0u) {
                atomicXor(&sign_minus[word_idx], sign_update);
            }
        }
        case GATE_SWAP: {
            let a_offset = ctrl_qubit * params.gen_words + word_idx;
            let b_offset = tgt_qubit * params.gen_words + word_idx;
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
        }
        default: {}
    }
}
