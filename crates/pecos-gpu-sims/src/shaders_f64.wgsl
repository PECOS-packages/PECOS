// State vector quantum simulation shaders (f64 precision)
//
// State vector layout: array of vec2<f64> where .x = real, .y = imaginary
// For n qubits, we have 2^n amplitudes.
//
// Requires SHADER_F64 feature (Vulkan shaderFloat64 capability).

// Shared state vector buffer (read-write)
@group(0) @binding(0)
var<storage, read_write> state: array<vec2<f64>>;

// Gate parameters
struct GateParams {
    target_qubit: u32,
    control_qubit: u32,
    num_qubits: u32,
    _padding: u32,
    // 2x2 gate matrix stored as 8 f64 values:
    // a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im
    a_re: f64,
    a_im: f64,
    b_re: f64,
    b_im: f64,
    c_re: f64,
    c_im: f64,
    d_re: f64,
    d_im: f64,
}

@group(0) @binding(1)
var<uniform> params: GateParams;

const WORKGROUP_SIZE: u32 = 256u;

fn get_linear_idx(global_id: vec3<u32>, num_workgroups: vec3<u32>) -> u32 {
    let threads_per_y_row = num_workgroups.x * WORKGROUP_SIZE;
    return global_id.y * threads_per_y_row + global_id.x;
}

// Complex multiplication: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
fn cmul(a: vec2<f64>, b: vec2<f64>) -> vec2<f64> {
    return vec2<f64>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

// Apply diagonal single-qubit gate: [[a, 0], [0, d]]
// Each thread handles ONE amplitude (not a pair), applying the appropriate
// diagonal element based on the qubit bit. Fully coalesced memory access.
@compute @workgroup_size(256)
fn apply_diagonal_gate(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let idx = get_linear_idx(global_id, num_workgroups);
    let num_amplitudes = 1u << params.num_qubits;

    if (idx >= num_amplitudes) {
        return;
    }

    // Select diagonal element based on target qubit bit
    let bit = (idx >> params.target_qubit) & 1u;
    let phase_re = select(params.a_re, params.d_re, bit == 1u);
    let phase_im = select(params.a_im, params.d_im, bit == 1u);
    let phase = vec2<f64>(phase_re, phase_im);

    state[idx] = cmul(phase, state[idx]);
}

// Apply arbitrary single-qubit gate
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

    let low_mask = (1u << params.target_qubit) - 1u;
    let high_bits = pair_idx >> params.target_qubit;
    let low_bits = pair_idx & low_mask;

    let idx0 = (high_bits << (params.target_qubit + 1u)) | low_bits;
    let idx1 = idx0 | (1u << params.target_qubit);

    let amp0 = state[idx0];
    let amp1 = state[idx1];

    let a = vec2<f64>(params.a_re, params.a_im);
    let b = vec2<f64>(params.b_re, params.b_im);
    let c = vec2<f64>(params.c_re, params.c_im);
    let d = vec2<f64>(params.d_re, params.d_im);

    let new_amp0 = cmul(a, amp0) + cmul(b, amp1);
    let new_amp1 = cmul(c, amp0) + cmul(d, amp1);

    state[idx0] = new_amp0;
    state[idx1] = new_amp1;
}

// Apply CNOT (CX) gate
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

    let control_mask = 1u << params.control_qubit;
    let target_mask = 1u << params.target_qubit;

    let control_set = (idx & control_mask) != 0u;
    let target_set = (idx & target_mask) != 0u;

    if (control_set && !target_set) {
        let partner_idx = idx | target_mask;
        let amp0 = state[idx];
        let amp1 = state[partner_idx];
        state[idx] = amp1;
        state[partner_idx] = amp0;
    }
}

// Apply CZ gate
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

    if ((idx & control_mask) != 0u && (idx & target_mask) != 0u) {
        state[idx] = -state[idx];
    }
}

// Apply CY gate: controlled-Y
// When control is |1> and target is |0>, swap and apply phase
// CY|c,t> = |c> (Y|t>) when c=1, else |c,t>
// Y = [[0, -i], [i, 0]]
@compute @workgroup_size(256)
fn apply_cy(
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

    let control_set = (idx & control_mask) != 0u;
    let target_set = (idx & target_mask) != 0u;

    // Process pairs once: when control=1 and target=0
    if (control_set && !target_set) {
        let partner_idx = idx | target_mask;
        let amp0 = state[idx];         // |...0...> (target=0)
        let amp1 = state[partner_idx]; // |...1...> (target=1)

        // Y|0> = i|1>, Y|1> = -i|0>
        // new amp0 = -i * amp1 = (amp1.y, -amp1.x)
        // new amp1 = i * amp0 = (-amp0.y, amp0.x)
        state[idx] = vec2<f64>(amp1.y, -amp1.x);
        state[partner_idx] = vec2<f64>(-amp0.y, amp0.x);
    }
}

// Apply SWAP gate: exchange amplitudes between two qubits
@compute @workgroup_size(256)
fn apply_swap(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let idx = get_linear_idx(global_id, num_workgroups);
    let num_amplitudes = 1u << params.num_qubits;

    if (idx >= num_amplitudes) {
        return;
    }

    let mask_a = 1u << params.control_qubit;
    let mask_b = 1u << params.target_qubit;

    let bit_a = (idx & mask_a) != 0u;
    let bit_b = (idx & mask_b) != 0u;

    // Only swap when bits differ: (a=0,b=1) swaps with (a=1,b=0)
    // Process once: when a=0 and b=1
    if (!bit_a && bit_b) {
        let partner = (idx & ~mask_b) | mask_a;
        let amp0 = state[idx];
        let amp1 = state[partner];
        state[idx] = amp1;
        state[partner] = amp0;
    }
}

// Apply RXX(theta) gate: exp(-i * theta/2 * X x X)
@compute @workgroup_size(256)
fn apply_rxx(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let idx = get_linear_idx(global_id, num_workgroups);
    let num_amplitudes = 1u << params.num_qubits;
    if (idx >= num_amplitudes) { return; }

    let mask_a = 1u << params.control_qubit;
    let mask_b = 1u << params.target_qubit;
    // cos(theta/2) and sin(theta/2) are precomputed on the CPU and passed
    // via (a_re, a_im). wgpu+Vulkan f64 cos/sin is unreliable.
    let partner = idx ^ mask_a ^ mask_b;
    if (idx < partner) {
        let c = params.a_re;
        let s = params.a_im;
        let amp0 = state[idx];
        let amp1 = state[partner];
        state[idx] = vec2<f64>(amp0.x * c + amp1.y * s, amp0.y * c - amp1.x * s);
        state[partner] = vec2<f64>(amp1.x * c + amp0.y * s, amp1.y * c - amp0.x * s);
    }
}

// Apply RYY(theta) gate: exp(-i * theta/2 * Y x Y)
@compute @workgroup_size(256)
fn apply_ryy(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let idx = get_linear_idx(global_id, num_workgroups);
    let num_amplitudes = 1u << params.num_qubits;
    if (idx >= num_amplitudes) { return; }

    let mask_a = 1u << params.control_qubit;
    let mask_b = 1u << params.target_qubit;
    // RYY acts on all 4 basis states but the coupling sign differs between
    // the (|00>,|11>) same-parity pair (+i*sin) and the (|01>,|10>)
    // diff-parity pair (-i*sin).
    let bit_a = (idx & mask_a) != 0u;
    let bit_b = (idx & mask_b) != 0u;
    let partner = idx ^ mask_a ^ mask_b;
    if (idx < partner) {
        let c = params.a_re;
        let s_abs = params.a_im;
        let s = select(s_abs, -s_abs, bit_a == bit_b);
        let amp0 = state[idx];
        let amp1 = state[partner];
        state[idx] = vec2<f64>(amp0.x * c + amp1.y * s, amp0.y * c - amp1.x * s);
        state[partner] = vec2<f64>(amp1.x * c + amp0.y * s, amp1.y * c - amp0.x * s);
    }
}

// Apply RZZ(theta) gate
// Angle theta is passed in a_re field
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

    let q1_mask = 1u << params.control_qubit;
    let q2_mask = 1u << params.target_qubit;

    let q1_set = (idx & q1_mask) != 0u;
    let q2_set = (idx & q2_mask) != 0u;

    // RZZ phase is -theta/2 when both bits match, +theta/2 otherwise.
    // cos/sin precomputed on CPU: a_re = cos(theta/2), a_im = sin(theta/2).
    let c = params.a_re;
    let s_abs = params.a_im;
    let s = select(s_abs, -s_abs, q1_set == q2_set);
    let amp = state[idx];
    state[idx] = vec2<f64>(amp.x * c - amp.y * s, amp.x * s + amp.y * c);
}

// Collapse state after measurement
struct MeasureParams {
    target_qubit: u32,
    outcome: u32,
    norm_factor: f64,
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
        state[idx] = state[idx] * measure_params.norm_factor;
    } else {
        state[idx] = vec2<f64>(0.0, 0.0);
    }
}

// GPU-side workgroup reduction for marginal probability
@group(0) @binding(4)
var<storage, read_write> partial_sums: array<f64>;

var<workgroup> shared_prob: array<f64, 256>;

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

    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid < stride) {
            shared_prob[lid] += shared_prob[lid + stride];
        }
        workgroupBarrier();
    }

    if (lid == 0u) {
        let wg_idx = workgroup_id.y * num_workgroups.x + workgroup_id.x;
        partial_sums[wg_idx] = shared_prob[0];
    }
}
