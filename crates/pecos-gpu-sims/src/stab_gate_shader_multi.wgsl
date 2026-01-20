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
struct PersistentParams {
    num_qubits: u32,
    gen_words: u32,
    num_gens: u32,
    num_shots: u32,
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
    _padding4: u32,
}

@group(0) @binding(5) var<uniform> params: PersistentParams;

// Gate queue: [0] = num_gates, [1..] = packed gates
@group(0) @binding(6) var<storage, read> gate_queue: array<u32>;

// Noise support
// Per-shot seeds for deterministic noise
@group(0) @binding(8) var<storage, read> noise_seeds: array<u32>;

// Noise parameters
struct NoiseParams {
    enabled: u32,           // 0 = disabled, 1 = enabled
    p1_threshold: u32,      // Fixed-point threshold for 1Q gate error (p * 0xFFFF)
    p2_threshold: u32,      // Fixed-point threshold for 2Q gate error
    p_meas_threshold: u32,  // Fixed-point threshold for measurement error (used on CPU)
}
@group(0) @binding(9) var<uniform> noise_params: NoiseParams;

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

// Apply depolarizing noise to a single qubit
// With probability p, applies a random Pauli (X, Y, or Z)
fn apply_noise_1q(
    shot_id: u32,
    gate_idx: u32,
    qubit: u32,
    word_idx: u32,
    shot_tableau_base: u32,
    threshold: u32,
    local_sign_minus: ptr<function, u32>
) {
    if (noise_params.enabled == 0u) { return; }

    let seed = noise_seeds[shot_id];
    let rand = hash_noise(seed, gate_idx, qubit);

    // Check if error occurs (compare lower 16 bits against threshold)
    if ((rand & 0xFFFFu) >= threshold) { return; }

    // Select Pauli: 0=X, 1=Y, 2=Z (use upper bits)
    let pauli = (rand >> 16u) % 3u;
    let row_offset = shot_tableau_base + qubit * params.gen_words + word_idx;

    switch (pauli) {
        case 0u: { // X: flip sign where Z=1
            *local_sign_minus ^= stab_z[row_offset];
        }
        case 1u: { // Y: flip sign where X != Z
            *local_sign_minus ^= (stab_x[row_offset] ^ stab_z[row_offset]);
        }
        case 2u: { // Z: flip sign where X=1
            *local_sign_minus ^= stab_x[row_offset];
        }
        default: {}
    }
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

    // Process all gates in sequence
    for (var gate_idx: u32 = 0u; gate_idx < num_gates; gate_idx = gate_idx + 1u) {
        let gate = decode_gate(gate_queue[gate_idx + 1u]);
        let gate_type = gate.x;
        let tgt_qubit = gate.y;
        let ctrl_qubit = gate.z;

        switch (gate_type) {
            case GATE_H: {
                let row_offset = shot_tableau_base + tgt_qubit * params.gen_words + word_idx;
                let orig_stab_x = stab_x[row_offset];
                let orig_stab_z = stab_z[row_offset];
                stab_x[row_offset] = orig_stab_z;
                stab_z[row_offset] = orig_stab_x;
                let destab_x_word = destab_x[row_offset];
                let destab_z_word = destab_z[row_offset];
                destab_x[row_offset] = destab_z_word;
                destab_z[row_offset] = destab_x_word;
                local_sign_minus ^= (orig_stab_x & orig_stab_z);
                // Apply 1Q noise after gate
                apply_noise_1q(shot_id, gate_idx, tgt_qubit, word_idx, shot_tableau_base, noise_params.p1_threshold, &local_sign_minus);
            }
            case GATE_S: {
                let row_offset = shot_tableau_base + tgt_qubit * params.gen_words + word_idx;
                let orig_stab_x = stab_x[row_offset];
                let orig_stab_z = stab_z[row_offset];
                stab_z[row_offset] = orig_stab_z ^ orig_stab_x;
                let orig_destab_x = destab_x[row_offset];
                let orig_destab_z = destab_z[row_offset];
                destab_z[row_offset] = orig_destab_z ^ orig_destab_x;
                // S: when X is set with existing i phase, toggle minus (i*i = -1)
                let toggle_minus_mask = orig_stab_x & local_sign_i;
                local_sign_minus ^= toggle_minus_mask;
                local_sign_i ^= orig_stab_x;
                // Apply 1Q noise after gate
                apply_noise_1q(shot_id, gate_idx, tgt_qubit, word_idx, shot_tableau_base, noise_params.p1_threshold, &local_sign_minus);
            }
            case GATE_SDG: {
                let row_offset = shot_tableau_base + tgt_qubit * params.gen_words + word_idx;
                let orig_stab_x = stab_x[row_offset];
                let orig_stab_z = stab_z[row_offset];
                stab_z[row_offset] = orig_stab_z ^ orig_stab_x;
                let orig_destab_x = destab_x[row_offset];
                let orig_destab_z = destab_z[row_offset];
                destab_z[row_offset] = orig_destab_z ^ orig_destab_x;
                let had_i = local_sign_i;
                // Sdg multiplies phase by -i when X=1: flip sign_minus when sign_i was 0
                local_sign_minus ^= (orig_stab_x & ~had_i);
                local_sign_i ^= orig_stab_x;
                // Apply 1Q noise after gate
                apply_noise_1q(shot_id, gate_idx, tgt_qubit, word_idx, shot_tableau_base, noise_params.p1_threshold, &local_sign_minus);
            }
            case GATE_X: {
                let row_offset = shot_tableau_base + tgt_qubit * params.gen_words + word_idx;
                let stab_z_word = stab_z[row_offset];
                local_sign_minus ^= stab_z_word;
                // Apply 1Q noise after gate
                apply_noise_1q(shot_id, gate_idx, tgt_qubit, word_idx, shot_tableau_base, noise_params.p1_threshold, &local_sign_minus);
            }
            case GATE_Y: {
                let row_offset = shot_tableau_base + tgt_qubit * params.gen_words + word_idx;
                let stab_x_word = stab_x[row_offset];
                let stab_z_word = stab_z[row_offset];
                local_sign_minus ^= (stab_x_word ^ stab_z_word);
                // Apply 1Q noise after gate
                apply_noise_1q(shot_id, gate_idx, tgt_qubit, word_idx, shot_tableau_base, noise_params.p1_threshold, &local_sign_minus);
            }
            case GATE_Z: {
                let row_offset = shot_tableau_base + tgt_qubit * params.gen_words + word_idx;
                let stab_x_word = stab_x[row_offset];
                local_sign_minus ^= stab_x_word;
                // Apply 1Q noise after gate
                apply_noise_1q(shot_id, gate_idx, tgt_qubit, word_idx, shot_tableau_base, noise_params.p1_threshold, &local_sign_minus);
            }
            case GATE_CX: {
                let ctrl_offset = shot_tableau_base + ctrl_qubit * params.gen_words + word_idx;
                let tgt_offset = shot_tableau_base + tgt_qubit * params.gen_words + word_idx;
                // CX: X_tgt ^= X_ctrl, Z_ctrl ^= Z_tgt
                stab_x[tgt_offset] = stab_x[tgt_offset] ^ stab_x[ctrl_offset];
                stab_z[ctrl_offset] = stab_z[ctrl_offset] ^ stab_z[tgt_offset];
                destab_x[tgt_offset] = destab_x[tgt_offset] ^ destab_x[ctrl_offset];
                destab_z[ctrl_offset] = destab_z[ctrl_offset] ^ destab_z[tgt_offset];
                // CX does NOT require sign updates
                // Apply 2Q noise: independent noise on each qubit
                apply_noise_1q(shot_id, gate_idx, ctrl_qubit, word_idx, shot_tableau_base, noise_params.p2_threshold, &local_sign_minus);
                apply_noise_1q(shot_id, gate_idx + 0x8000u, tgt_qubit, word_idx, shot_tableau_base, noise_params.p2_threshold, &local_sign_minus);
            }
            case GATE_CZ: {
                let a_offset = shot_tableau_base + ctrl_qubit * params.gen_words + word_idx;
                let b_offset = shot_tableau_base + tgt_qubit * params.gen_words + word_idx;
                let a_x = stab_x[a_offset];
                let b_x = stab_x[b_offset];
                stab_z[a_offset] = stab_z[a_offset] ^ b_x;
                stab_z[b_offset] = stab_z[b_offset] ^ a_x;
                let a_destab_x = destab_x[a_offset];
                let b_destab_x = destab_x[b_offset];
                destab_z[a_offset] = destab_z[a_offset] ^ b_destab_x;
                destab_z[b_offset] = destab_z[b_offset] ^ a_destab_x;
                // CZ: flip sign when both qubits have X
                local_sign_minus ^= (a_x & b_x);
                // Apply 2Q noise: independent noise on each qubit
                apply_noise_1q(shot_id, gate_idx, ctrl_qubit, word_idx, shot_tableau_base, noise_params.p2_threshold, &local_sign_minus);
                apply_noise_1q(shot_id, gate_idx + 0x8000u, tgt_qubit, word_idx, shot_tableau_base, noise_params.p2_threshold, &local_sign_minus);
            }
            case GATE_SWAP: {
                let a_offset = shot_tableau_base + ctrl_qubit * params.gen_words + word_idx;
                let b_offset = shot_tableau_base + tgt_qubit * params.gen_words + word_idx;
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
                // Apply 2Q noise: independent noise on each qubit
                apply_noise_1q(shot_id, gate_idx, ctrl_qubit, word_idx, shot_tableau_base, noise_params.p2_threshold, &local_sign_minus);
                apply_noise_1q(shot_id, gate_idx + 0x8000u, tgt_qubit, word_idx, shot_tableau_base, noise_params.p2_threshold, &local_sign_minus);
            }
            default: {}
        }
    }

    // Write cached signs back
    sign_minus[shot_sign_base + word_idx] = local_sign_minus;
    sign_i[shot_sign_base + word_idx] = local_sign_i;
}
