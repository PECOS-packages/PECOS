// GPU Stabilizer Gate Shader - Multi-Shot Version
//
// Processes N independent stabilizer simulations in parallel.
// Each thread handles one (shot_id, word_idx) pair.
// All shots process the same gate queue simultaneously.
//
// Memory layout (shot-major):
//   stab_x[shot * tableau_stride + qubit * gen_words + word_idx]
//   where tableau_stride = num_qubits * gen_words

// Stabilizer tableau buffers (sized for num_shots * single_shot_size)
@group(0) @binding(0) var<storage, read_write> stab_x: array<u32>;
@group(0) @binding(1) var<storage, read_write> stab_z: array<u32>;
@group(0) @binding(2) var<storage, read_write> destab_x: array<u32>;
@group(0) @binding(3) var<storage, read_write> destab_z: array<u32>;

// Packed sign bits (sized for num_shots * gen_words)
@group(0) @binding(4) var<storage, read_write> sign_minus: array<u32>;
@group(0) @binding(7) var<storage, read_write> sign_i: array<u32>;

// Parameters
// Persistent parameters including noise thresholds.
// Noise params are packed here (instead of a separate binding) to avoid
// Metal/naga issues with small uniform buffers where the third field
// reads as 0 on Apple Paravirtual devices.
struct PersistentParams {
    num_qubits: u32,
    gen_words: u32,
    num_gens: u32,
    num_shots: u32,
    noise_enabled: u32,
    noise_p1_threshold: u32,
    noise_p2_threshold: u32,
    noise_p_meas_threshold: u32,
}

@group(0) @binding(5) var<uniform> params: PersistentParams;

// Gate queue: [0] = num_gates, [1..] = packed gates
@group(0) @binding(6) var<storage, read> gate_queue: array<u32>;

// Noise support
// Per-shot seeds for deterministic noise
@group(0) @binding(8) var<storage, read> noise_seeds: array<u32>;

// PCG-style hash for deterministic per-gate noise
// Combines shot seed, gate index, and qubit to produce independent random values
fn hash_noise(seed: u32, gate_idx: u32, qubit: u32) -> u32 {
    var h = seed ^ (gate_idx * 0x9E3779B9u) ^ (qubit * 0x85EBCA6Bu);
    h = h ^ (h >> 16u);
    h = h * 0x85EBCA6Bu;
    h = h ^ (h >> 13u);
    h = h * 0xC2B2AE35u;
    h = h ^ (h >> 16u);
    return h;
}

// Compute depolarizing noise XOR mask for a single qubit.
// Returns a bitmask to XOR into the sign word. Returns 0 if no error occurs.
// All needed values are passed as parameters to avoid Metal shader compiler
// issues with reading global/uniform buffers from within functions called
// in switch contexts.
fn noise_mask_1q(
    noise_enabled: u32,
    seed: u32,
    gate_idx: u32,
    qubit: u32,
    threshold: u32,
    stab_x_val: u32,
    stab_z_val: u32,
) -> u32 {
    if (noise_enabled == 0u) { return 0u; }

    let rand = hash_noise(seed, gate_idx, qubit);

    // Check if error occurs (compare lower 16 bits against threshold)
    if ((rand & 0xFFFFu) >= threshold) { return 0u; }

    // Select Pauli error: use var assignment (single return) to avoid
    // potential Metal compiler issues with multiple return paths
    let pauli = (rand >> 16u) % 3u;
    var result = stab_z_val;  // default: X error (pauli=0)
    if (pauli == 1u) {
        result = stab_x_val ^ stab_z_val;  // Y error
    } else if (pauli == 2u) {
        result = stab_x_val;  // Z error
    }
    return result;
}

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

// Shared memory for broadcasting num_gates
var<workgroup> shared_num_gates: u32;

@compute @workgroup_size(256)
fn process_gate_queue_multi(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_index) local_idx: u32
) {
    // First thread loads num_gates
    if (local_idx == 0u) {
        shared_num_gates = gate_queue[0];
    }
    workgroupBarrier();

    // Compute shot_id and word_idx from global thread id
    // Thread id = shot_id * gen_words + word_idx
    let thread_id = global_id.x;
    let total_words = params.num_shots * params.gen_words;

    if (thread_id >= total_words) {
        return;
    }

    let shot_id = thread_id / params.gen_words;
    let word_idx = thread_id % params.gen_words;

    let num_gates = shared_num_gates;
    if (num_gates == 0u) {
        return;
    }

    // Compute base offsets for this shot
    let tableau_stride = params.num_qubits * params.gen_words;
    let shot_tableau_base = shot_id * tableau_stride;
    let shot_sign_base = shot_id * params.gen_words;

    // Cache signs in local variables
    var local_sign_minus = sign_minus[shot_sign_base + word_idx];
    var local_sign_i = sign_i[shot_sign_base + word_idx];

    // Pre-read noise params into locals to pass as function arguments.
    // This avoids Metal shader compiler issues with reading uniform/storage
    // buffers from within called functions.
    let noise_enabled = params.noise_enabled;
    let noise_p1 = params.noise_p1_threshold;
    let noise_p2 = params.noise_p2_threshold;
    let noise_seed = noise_seeds[shot_id];
    let gen_w = params.gen_words;

    // Process all gates in sequence
    for (var gate_idx: u32 = 0u; gate_idx < num_gates; gate_idx = gate_idx + 1u) {
        let gate = decode_gate(gate_queue[gate_idx + 1u]);
        let gate_type = gate.x;
        let tgt_qubit = gate.y;
        let ctrl_qubit = gate.z;

        switch (gate_type) {
            case GATE_H: {
                let row_offset = shot_tableau_base + tgt_qubit * gen_w + word_idx;
                let orig_stab_x = stab_x[row_offset];
                let orig_stab_z = stab_z[row_offset];
                stab_x[row_offset] = orig_stab_z;
                stab_z[row_offset] = orig_stab_x;
                let destab_x_word = destab_x[row_offset];
                let destab_z_word = destab_z[row_offset];
                destab_x[row_offset] = destab_z_word;
                destab_z[row_offset] = destab_x_word;
                local_sign_minus ^= (orig_stab_x & orig_stab_z);
                // Apply 1Q noise after gate (read stab values post-gate for noise)
                let n_sx = stab_x[row_offset];
                let n_sz = stab_z[row_offset];
                local_sign_minus ^= noise_mask_1q(noise_enabled, noise_seed, gate_idx, tgt_qubit, noise_p1, n_sx, n_sz);
            }
            case GATE_S: {
                let row_offset = shot_tableau_base + tgt_qubit * gen_w + word_idx;
                let orig_stab_x = stab_x[row_offset];
                let orig_stab_z = stab_z[row_offset];
                stab_z[row_offset] = orig_stab_z ^ orig_stab_x;
                let orig_destab_x = destab_x[row_offset];
                let orig_destab_z = destab_z[row_offset];
                destab_z[row_offset] = orig_destab_z ^ orig_destab_x;
                let toggle_minus_mask = orig_stab_x & local_sign_i;
                local_sign_minus ^= toggle_minus_mask;
                local_sign_i ^= orig_stab_x;
                let n_sx = stab_x[row_offset];
                let n_sz = stab_z[row_offset];
                local_sign_minus ^= noise_mask_1q(noise_enabled, noise_seed, gate_idx, tgt_qubit, noise_p1, n_sx, n_sz);
            }
            case GATE_SDG: {
                let row_offset = shot_tableau_base + tgt_qubit * gen_w + word_idx;
                let orig_stab_x = stab_x[row_offset];
                let orig_stab_z = stab_z[row_offset];
                stab_z[row_offset] = orig_stab_z ^ orig_stab_x;
                let orig_destab_x = destab_x[row_offset];
                let orig_destab_z = destab_z[row_offset];
                destab_z[row_offset] = orig_destab_z ^ orig_destab_x;
                let had_i = local_sign_i;
                local_sign_minus ^= (orig_stab_x & ~had_i);
                local_sign_i ^= orig_stab_x;
                let n_sx = stab_x[row_offset];
                let n_sz = stab_z[row_offset];
                local_sign_minus ^= noise_mask_1q(noise_enabled, noise_seed, gate_idx, tgt_qubit, noise_p1, n_sx, n_sz);
            }
            case GATE_X: {
                let row_offset = shot_tableau_base + tgt_qubit * gen_w + word_idx;
                let stab_z_word = stab_z[row_offset];
                local_sign_minus ^= stab_z_word;
                let n_sx = stab_x[row_offset];
                let n_sz = stab_z[row_offset];
                local_sign_minus ^= noise_mask_1q(noise_enabled, noise_seed, gate_idx, tgt_qubit, noise_p1, n_sx, n_sz);
            }
            case GATE_Y: {
                let row_offset = shot_tableau_base + tgt_qubit * gen_w + word_idx;
                let stab_x_word = stab_x[row_offset];
                let stab_z_word = stab_z[row_offset];
                local_sign_minus ^= (stab_x_word ^ stab_z_word);
                local_sign_minus ^= noise_mask_1q(noise_enabled, noise_seed, gate_idx, tgt_qubit, noise_p1, stab_x_word, stab_z_word);
            }
            case GATE_Z: {
                let row_offset = shot_tableau_base + tgt_qubit * gen_w + word_idx;
                let stab_x_word = stab_x[row_offset];
                local_sign_minus ^= stab_x_word;
                let n_sz = stab_z[row_offset];
                local_sign_minus ^= noise_mask_1q(noise_enabled, noise_seed, gate_idx, tgt_qubit, noise_p1, stab_x_word, n_sz);
            }
            case GATE_CX: {
                let ctrl_offset = shot_tableau_base + ctrl_qubit * gen_w + word_idx;
                let tgt_offset = shot_tableau_base + tgt_qubit * gen_w + word_idx;
                stab_x[tgt_offset] = stab_x[tgt_offset] ^ stab_x[ctrl_offset];
                stab_z[ctrl_offset] = stab_z[ctrl_offset] ^ stab_z[tgt_offset];
                destab_x[tgt_offset] = destab_x[tgt_offset] ^ destab_x[ctrl_offset];
                destab_z[ctrl_offset] = destab_z[ctrl_offset] ^ destab_z[tgt_offset];
                // Noise on ctrl qubit
                let c_sx = stab_x[ctrl_offset];
                let c_sz = stab_z[ctrl_offset];
                local_sign_minus ^= noise_mask_1q(noise_enabled, noise_seed, gate_idx, ctrl_qubit, noise_p2, c_sx, c_sz);
                // Noise on tgt qubit
                let t_sx = stab_x[tgt_offset];
                let t_sz = stab_z[tgt_offset];
                local_sign_minus ^= noise_mask_1q(noise_enabled, noise_seed, gate_idx + 0x8000u, tgt_qubit, noise_p2, t_sx, t_sz);
            }
            case GATE_CZ: {
                let a_offset = shot_tableau_base + ctrl_qubit * gen_w + word_idx;
                let b_offset = shot_tableau_base + tgt_qubit * gen_w + word_idx;
                let a_x = stab_x[a_offset];
                let b_x = stab_x[b_offset];
                stab_z[a_offset] = stab_z[a_offset] ^ b_x;
                stab_z[b_offset] = stab_z[b_offset] ^ a_x;
                let a_destab_x = destab_x[a_offset];
                let b_destab_x = destab_x[b_offset];
                destab_z[a_offset] = destab_z[a_offset] ^ b_destab_x;
                destab_z[b_offset] = destab_z[b_offset] ^ a_destab_x;
                local_sign_minus ^= (a_x & b_x);
                let a_sx = stab_x[a_offset];
                let a_sz = stab_z[a_offset];
                local_sign_minus ^= noise_mask_1q(noise_enabled, noise_seed, gate_idx, ctrl_qubit, noise_p2, a_sx, a_sz);
                let b_sx = stab_x[b_offset];
                let b_sz = stab_z[b_offset];
                local_sign_minus ^= noise_mask_1q(noise_enabled, noise_seed, gate_idx + 0x8000u, tgt_qubit, noise_p2, b_sx, b_sz);
            }
            case GATE_SWAP: {
                let a_offset = shot_tableau_base + ctrl_qubit * gen_w + word_idx;
                let b_offset = shot_tableau_base + tgt_qubit * gen_w + word_idx;
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
                let a_sx = stab_x[a_offset];
                let a_sz = stab_z[a_offset];
                local_sign_minus ^= noise_mask_1q(noise_enabled, noise_seed, gate_idx, ctrl_qubit, noise_p2, a_sx, a_sz);
                let b_sx = stab_x[b_offset];
                let b_sz = stab_z[b_offset];
                local_sign_minus ^= noise_mask_1q(noise_enabled, noise_seed, gate_idx + 0x8000u, tgt_qubit, noise_p2, b_sx, b_sz);
            }
            default: {}
        }
    }

    // Write cached signs back
    sign_minus[shot_sign_base + word_idx] = local_sign_minus;
    sign_i[shot_sign_base + word_idx] = local_sign_i;
}
