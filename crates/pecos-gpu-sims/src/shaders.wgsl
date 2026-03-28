// State vector quantum simulation shaders
//
// State vector layout: array of vec2<f32> where .x = real, .y = imaginary
// For n qubits, we have 2^n amplitudes.

// Shared state vector buffer (read-write)
@group(0) @binding(0)
var<storage, read_write> state: array<vec2<f32>>;

// Gate parameters
struct GateParams {
    target_qubit: u32,      // Target qubit index
    control_qubit: u32,     // Control qubit index (for 2-qubit gates)
    num_qubits: u32,        // Total number of qubits
    _padding: u32,
    // 2x2 gate matrix (for arbitrary single-qubit gates)
    // [[a, b], [c, d]] stored as two vec4s:
    // matrix_row0 = (a_re, a_im, b_re, b_im)
    // matrix_row1 = (c_re, c_im, d_re, d_im)
    matrix_row0: vec4<f32>,
    matrix_row1: vec4<f32>,
}

@group(0) @binding(1)
var<uniform> params: GateParams;

// Workgroup size constant (must match @workgroup_size in all compute shaders)
const WORKGROUP_SIZE: u32 = 256u;

// Compute linear thread index from potentially 2D dispatch
// linear_idx = global_id.y * (num_workgroups.x * WORKGROUP_SIZE) + global_id.x
fn get_linear_idx(global_id: vec3<u32>, num_workgroups: vec3<u32>) -> u32 {
    let threads_per_y_row = num_workgroups.x * WORKGROUP_SIZE;
    return global_id.y * threads_per_y_row + global_id.x;
}

// Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

// Complex addition
fn cadd(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return a + b;
}

// Apply arbitrary single-qubit gate
// Each thread handles one pair of amplitudes that differ in the target qubit bit
@compute @workgroup_size(256)
fn apply_single_gate(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let pair_idx = get_linear_idx(global_id, num_workgroups);
    let num_pairs = 1u << (params.num_qubits - 1u);

    if (pair_idx >= num_pairs) {
        return;
    }

    // Compute indices of the two amplitudes in this pair
    // Insert a 0 bit at position target_qubit
    let low_mask = (1u << params.target_qubit) - 1u;
    let high_bits = pair_idx >> params.target_qubit;
    let low_bits = pair_idx & low_mask;

    let idx0 = (high_bits << (params.target_qubit + 1u)) | low_bits;
    let idx1 = idx0 | (1u << params.target_qubit);

    // Load amplitudes
    let amp0 = state[idx0];
    let amp1 = state[idx1];

    // Load matrix elements from vec4s
    let a = params.matrix_row0.xy;
    let b = params.matrix_row0.zw;
    let c = params.matrix_row1.xy;
    let d = params.matrix_row1.zw;

    // Apply gate: [a b; c d] * [amp0; amp1]
    let new_amp0 = cadd(cmul(a, amp0), cmul(b, amp1));
    let new_amp1 = cadd(cmul(c, amp0), cmul(d, amp1));

    // Store results
    state[idx0] = new_amp0;
    state[idx1] = new_amp1;
}

// Apply CNOT (CX) gate
// Only flips target when control is |1>
@compute @workgroup_size(256)
fn apply_cx(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let idx = get_linear_idx(global_id, num_workgroups);
    let num_amplitudes = 1u << params.num_qubits;

    if (idx >= num_amplitudes) {
        return;
    }

    // Only process if control qubit is 1 and target qubit is 0
    // (we swap with the state where target is 1)
    let control_mask = 1u << params.control_qubit;
    let target_mask = 1u << params.target_qubit;

    let control_set = (idx & control_mask) != 0u;
    let target_set = (idx & target_mask) != 0u;

    // Only process pairs once: when control=1 and target=0
    if (control_set && !target_set) {
        let partner_idx = idx | target_mask;

        // Swap amplitudes
        let amp0 = state[idx];
        let amp1 = state[partner_idx];
        state[idx] = amp1;
        state[partner_idx] = amp0;
    }
}

// Apply CZ gate
// Applies phase of -1 when both control and target are |1>
@compute @workgroup_size(256)
fn apply_cz(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let idx = get_linear_idx(global_id, num_workgroups);
    let num_amplitudes = 1u << params.num_qubits;

    if (idx >= num_amplitudes) {
        return;
    }

    let control_mask = 1u << params.control_qubit;
    let target_mask = 1u << params.target_qubit;

    // Apply -1 phase when both qubits are |1>
    if ((idx & control_mask) != 0u && (idx & target_mask) != 0u) {
        state[idx] = -state[idx];
    }
}

// Apply RZZ(theta) gate: exp(-i * theta/2 * Z⊗Z)
// Phase depends on parity of the two qubits:
// |00⟩ → e^{-iθ/2} |00⟩  (same parity: negative phase)
// |01⟩ → e^{+iθ/2} |01⟩  (different parity: positive phase)
// |10⟩ → e^{+iθ/2} |10⟩  (different parity: positive phase)
// |11⟩ → e^{-iθ/2} |11⟩  (same parity: negative phase)
// Angle theta is passed in matrix_row0.x
@compute @workgroup_size(256)
fn apply_rzz(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let idx = get_linear_idx(global_id, num_workgroups);
    let num_amplitudes = 1u << params.num_qubits;

    if (idx >= num_amplitudes) {
        return;
    }

    let q1_mask = 1u << params.control_qubit;  // First qubit
    let q2_mask = 1u << params.target_qubit;   // Second qubit

    let q1_set = (idx & q1_mask) != 0u;
    let q2_set = (idx & q2_mask) != 0u;

    // Same parity (00 or 11) → phase = -theta/2
    // Different parity (01 or 10) → phase = +theta/2
    let theta = params.matrix_row0.x;
    let half_theta = theta / 2.0;
    let phase = select(half_theta, -half_theta, q1_set == q2_set);

    // Apply phase rotation: amplitude *= e^{i*phase} = cos(phase) + i*sin(phase)
    let c = cos(phase);
    let s = sin(phase);
    let amp = state[idx];
    // (a + bi) * (c + si) = (ac - bs) + (as + bc)i
    state[idx] = vec2<f32>(amp.x * c - amp.y * s, amp.x * s + amp.y * c);
}

// Collapse state after measurement
// Zeros out amplitudes inconsistent with measurement result and renormalizes
struct MeasureParams {
    target_qubit: u32,
    outcome: u32,           // 0 or 1
    norm_factor: f32,       // 1/sqrt(probability of outcome)
    _padding: u32,
}

@group(0) @binding(3)
var<uniform> measure_params: MeasureParams;

@compute @workgroup_size(256)
fn collapse_state(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let idx = get_linear_idx(global_id, num_workgroups);
    let num_amplitudes = 1u << params.num_qubits;

    if (idx >= num_amplitudes) {
        return;
    }

    let target_mask = 1u << measure_params.target_qubit;
    let qubit_value = select(0u, 1u, (idx & target_mask) != 0u);

    if (qubit_value == measure_params.outcome) {
        // Keep and renormalize
        state[idx] = state[idx] * measure_params.norm_factor;
    } else {
        // Zero out
        state[idx] = vec2<f32>(0.0, 0.0);
    }
}

// GPU-side workgroup reduction for marginal probability P(target_qubit = 1).
// Each workgroup of 256 threads computes a partial sum via shared memory reduction.
// The CPU reads back the partial sums (one per workgroup) and does the final sum.
// This avoids reading back all 2^n probabilities — only ~2^n/256 floats.
@group(0) @binding(4)
var<storage, read_write> partial_sums: array<f32>;

var<workgroup> shared_prob: array<f32, 256>;

@compute @workgroup_size(256)
fn reduce_marginal_probability(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let idx = get_linear_idx(global_id, num_workgroups);
    let num_amplitudes = 1u << params.num_qubits;
    let lid = local_id.x;

    // Each thread loads |amplitude|^2 if target qubit bit is 1, else 0
    if (idx < num_amplitudes) {
        let target_mask = 1u << params.target_qubit;
        if ((idx & target_mask) != 0u) {
            let amp = state[idx];
            shared_prob[lid] = amp.x * amp.x + amp.y * amp.y;
        } else {
            shared_prob[lid] = 0.0;
        }
    } else {
        shared_prob[lid] = 0.0;
    }

    workgroupBarrier();

    // Tree reduction within workgroup
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            shared_prob[lid] += shared_prob[lid + stride];
        }
        workgroupBarrier();
    }

    // Thread 0 writes workgroup partial sum
    if (lid == 0u) {
        let wg_idx = workgroup_id.y * num_workgroups.x + workgroup_id.x;
        partial_sums[wg_idx] = shared_prob[0];
    }
}
