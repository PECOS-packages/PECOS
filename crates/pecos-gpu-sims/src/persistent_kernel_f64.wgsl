// Persistent kernel for f64 state vectors.
// SHARED_SIZE is templated at runtime based on the GPU's actual shared memory.

@group(0) @binding(0)
var<storage, read_write> state: array<vec2<f64>>;

// Gate queue stored as array<f64>. Metadata stored as f64-encoded u32 values.
// Layout per gate (12 f64): [type, tgt, ctrl, pad, a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im]
// Header: [num_gates, num_qubits] as f64.
@group(0) @binding(5)
var<storage, read> gate_queue_f64: array<f64>;

var<workgroup> shared_state: array<vec2<f64>, {SHARED_SIZE}>;

fn cmul(a: vec2<f64>, b: vec2<f64>) -> vec2<f64> {
    return vec2<f64>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

const GATE_SINGLE: u32 = 0u;
const GATE_DIAGONAL: u32 = 1u;
const GATE_CX: u32 = 2u;
const GATE_CY: u32 = 3u;
const GATE_CZ: u32 = 4u;
const GATE_SWAP: u32 = 5u;
const GATE_RXX: u32 = 6u;
const GATE_RYY: u32 = 7u;
const GATE_RZZ: u32 = 8u;
const GATE_STRIDE: u32 = 12u;

@compute @workgroup_size(256)
fn apply_gate_queue_persistent(
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let tid = local_id.x;
    let num_gates = u32(gate_queue_f64[0]);
    let num_qubits = u32(gate_queue_f64[1]);
    let num_amplitudes = 1u << num_qubits;

    for (var i = tid; i < num_amplitudes; i += 256u) {
        shared_state[i] = state[i];
    }
    workgroupBarrier();

    for (var g = 0u; g < num_gates; g++) {
        let base = 2u + g * GATE_STRIDE;
        let gate_type = u32(gate_queue_f64[base]);
        let tgt = u32(gate_queue_f64[base + 1u]);
        let ctrl = u32(gate_queue_f64[base + 2u]);
        let num_pairs = num_amplitudes >> 1u;

        switch (gate_type) {
            case GATE_SINGLE: {
                let a = vec2<f64>(gate_queue_f64[base + 4u], gate_queue_f64[base + 5u]);
                let b = vec2<f64>(gate_queue_f64[base + 6u], gate_queue_f64[base + 7u]);
                let c = vec2<f64>(gate_queue_f64[base + 8u], gate_queue_f64[base + 9u]);
                let d = vec2<f64>(gate_queue_f64[base + 10u], gate_queue_f64[base + 11u]);

                let low_mask = (1u << tgt) - 1u;
                for (var pair_idx = tid; pair_idx < num_pairs; pair_idx += 256u) {
                    let high_bits = pair_idx >> tgt;
                    let low_bits = pair_idx & low_mask;
                    let idx0 = (high_bits << (tgt + 1u)) | low_bits;
                    let idx1 = idx0 | (1u << tgt);
                    let amp0 = shared_state[idx0];
                    let amp1 = shared_state[idx1];
                    shared_state[idx0] = cmul(a, amp0) + cmul(b, amp1);
                    shared_state[idx1] = cmul(c, amp0) + cmul(d, amp1);
                }
            }
            case GATE_DIAGONAL: {
                let a = vec2<f64>(gate_queue_f64[base + 4u], gate_queue_f64[base + 5u]);
                let d = vec2<f64>(gate_queue_f64[base + 10u], gate_queue_f64[base + 11u]);
                for (var i = tid; i < num_amplitudes; i += 256u) {
                    let bit = (i >> tgt) & 1u;
                    let phase = select(a, d, bit == 1u);
                    shared_state[i] = cmul(phase, shared_state[i]);
                }
            }
            case GATE_CX: {
                for (var i = tid; i < num_amplitudes; i += 256u) {
                    if ((i & (1u << ctrl)) != 0u && (i & (1u << tgt)) == 0u) {
                        let partner = i | (1u << tgt);
                        let tmp = shared_state[i];
                        shared_state[i] = shared_state[partner];
                        shared_state[partner] = tmp;
                    }
                }
            }
            case GATE_CY: {
                for (var i = tid; i < num_amplitudes; i += 256u) {
                    if ((i & (1u << ctrl)) != 0u && (i & (1u << tgt)) == 0u) {
                        let partner = i | (1u << tgt);
                        let amp0 = shared_state[i];
                        let amp1 = shared_state[partner];
                        shared_state[i] = vec2<f64>(amp1.y, -amp1.x);
                        shared_state[partner] = vec2<f64>(-amp0.y, amp0.x);
                    }
                }
            }
            case GATE_CZ: {
                for (var i = tid; i < num_amplitudes; i += 256u) {
                    if ((i & (1u << ctrl)) != 0u && (i & (1u << tgt)) != 0u) {
                        shared_state[i] = -shared_state[i];
                    }
                }
            }
            case GATE_SWAP: {
                for (var i = tid; i < num_amplitudes; i += 256u) {
                    let bit_a = (i & (1u << ctrl)) != 0u;
                    let bit_b = (i & (1u << tgt)) != 0u;
                    if (!bit_a && bit_b) {
                        let partner = (i & ~(1u << tgt)) | (1u << ctrl);
                        let tmp = shared_state[i];
                        shared_state[i] = shared_state[partner];
                        shared_state[partner] = tmp;
                    }
                }
            }
            case GATE_RXX: {
                // cos/sin precomputed on CPU: base+4 = cos(t/2), base+5 = sin(t/2).
                let c_val = gate_queue_f64[base + 4u];
                let s_val = gate_queue_f64[base + 5u];
                for (var i = tid; i < num_amplitudes; i += 256u) {
                    let partner = i ^ (1u << ctrl) ^ (1u << tgt);
                    if (i < partner) {
                        let amp0 = shared_state[i];
                        let amp1 = shared_state[partner];
                        shared_state[i] = vec2<f64>(amp0.x * c_val + amp1.y * s_val, amp0.y * c_val - amp1.x * s_val);
                        shared_state[partner] = vec2<f64>(amp1.x * c_val + amp0.y * s_val, amp1.y * c_val - amp0.x * s_val);
                    }
                }
            }
            case GATE_RYY: {
                let c_val = gate_queue_f64[base + 4u];
                let s_abs = gate_queue_f64[base + 5u];
                for (var i = tid; i < num_amplitudes; i += 256u) {
                    let partner = i ^ (1u << ctrl) ^ (1u << tgt);
                    if (i < partner) {
                        let bit_a = (i & (1u << ctrl)) != 0u;
                        let bit_b = (i & (1u << tgt)) != 0u;
                        let s_val = select(s_abs, -s_abs, bit_a == bit_b);
                        let amp0 = shared_state[i];
                        let amp1 = shared_state[partner];
                        shared_state[i] = vec2<f64>(amp0.x * c_val + amp1.y * s_val, amp0.y * c_val - amp1.x * s_val);
                        shared_state[partner] = vec2<f64>(amp1.x * c_val + amp0.y * s_val, amp1.y * c_val - amp0.x * s_val);
                    }
                }
            }
            case GATE_RZZ: {
                let c_val = gate_queue_f64[base + 4u];
                let s_abs = gate_queue_f64[base + 5u];
                for (var i = tid; i < num_amplitudes; i += 256u) {
                    let q1_set = (i & (1u << ctrl)) != 0u;
                    let q2_set = (i & (1u << tgt)) != 0u;
                    let s_val = select(s_abs, -s_abs, q1_set == q2_set);
                    let amp = shared_state[i];
                    shared_state[i] = vec2<f64>(amp.x * c_val - amp.y * s_val, amp.x * s_val + amp.y * c_val);
                }
            }
            default: {}
        }
        workgroupBarrier();
    }

    for (var i = tid; i < num_amplitudes; i += 256u) {
        state[i] = shared_state[i];
    }
}
