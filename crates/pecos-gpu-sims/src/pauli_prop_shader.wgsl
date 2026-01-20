// GPU Pauli Propagator Shader
//
// Tracks X and Z fault bits through Clifford gates across many shots in parallel.
// Each workgroup item handles one (qubit, shot_word) pair.

struct Params {
    num_qubits: u32,
    num_shots: u32,
    shot_words: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> x_faults: array<u32>;
@group(0) @binding(2) var<storage, read_write> z_faults: array<u32>;
@group(0) @binding(3) var<storage, read> gate_queue: array<u32>;
@group(0) @binding(4) var<storage, read> random_bits: array<u32>;

// Gate type constants
const GATE_H: u32 = 1u;
const GATE_SZ: u32 = 2u;
const GATE_SZDG: u32 = 3u;
const GATE_X: u32 = 4u;
const GATE_Y: u32 = 5u;
const GATE_Z: u32 = 6u;
const GATE_CX: u32 = 7u;
const GATE_CZ: u32 = 8u;
const GATE_SWAP: u32 = 9u;

// Fault injection constants
const FAULT_X: u32 = 16u;
const FAULT_Z: u32 = 17u;
const FAULT_Y: u32 = 18u;
const FAULT_DEPOL1: u32 = 19u;
const FAULT_DEPOL2: u32 = 20u;

// Hash function for generating per-shot randomness
fn hash(seed: u32, shot: u32, gate_idx: u32) -> u32 {
    var h = seed ^ (shot * 0x9E3779B9u) ^ (gate_idx * 0x85EBCA6Bu);
    h = h ^ (h >> 16u);
    h = h * 0x85EBCA6Bu;
    h = h ^ (h >> 13u);
    h = h * 0xC2B2AE35u;
    h = h ^ (h >> 16u);
    return h;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_items = params.num_qubits * params.shot_words;

    if (idx >= total_items) {
        return;
    }

    // Decode idx into qubit and shot_word
    let qubit = idx / params.shot_words;
    let word_idx = idx % params.shot_words;

    // Load gate count
    let gate_count = gate_queue[0];

    // Load local copies of fault bits for this qubit/word
    let offset = qubit * params.shot_words + word_idx;
    var local_x = x_faults[offset];
    var local_z = z_faults[offset];

    // Process gates
    var i = 1u;
    var gate_idx = 0u;
    while (i <= gate_count) {
        let gate_type = gate_queue[i];
        let qubit1 = gate_queue[i + 1u];
        let qubit2 = gate_queue[i + 2u];

        // Single-qubit gates: only process if this is our qubit
        if (gate_type == GATE_H && qubit1 == qubit) {
            // H: X <-> Z
            let tmp = local_x;
            local_x = local_z;
            local_z = tmp;
        }
        else if (gate_type == GATE_SZ && qubit1 == qubit) {
            // SZ: X -> XZ (add Z where X is present)
            local_z = local_z ^ local_x;
        }
        else if (gate_type == GATE_SZDG && qubit1 == qubit) {
            // SZDG: X -> X(-Z) = XZ (same bit operation, sign tracked separately if needed)
            local_z = local_z ^ local_x;
        }
        else if (gate_type == GATE_X && qubit1 == qubit) {
            // X: toggle X fault
            local_x = ~local_x;
        }
        else if (gate_type == GATE_Y && qubit1 == qubit) {
            // Y: toggle both X and Z
            local_x = ~local_x;
            local_z = ~local_z;
        }
        else if (gate_type == GATE_Z && qubit1 == qubit) {
            // Z: toggle Z fault
            local_z = ~local_z;
        }
        // Two-qubit gates: need to read from the other qubit
        else if (gate_type == GATE_CX) {
            let ctrl = qubit1;
            let tgt = qubit2;

            if (qubit == ctrl) {
                // Control: Z propagates from target
                let tgt_offset = tgt * params.shot_words + word_idx;
                let tgt_z = z_faults[tgt_offset];
                local_z = local_z ^ tgt_z;
            }
            else if (qubit == tgt) {
                // Target: X propagates from control
                let ctrl_offset = ctrl * params.shot_words + word_idx;
                let ctrl_x = x_faults[ctrl_offset];
                local_x = local_x ^ ctrl_x;
            }
        }
        else if (gate_type == GATE_CZ) {
            let qa = qubit1;
            let qb = qubit2;

            if (qubit == qa) {
                // qa: Z += qb's X
                let qb_offset = qb * params.shot_words + word_idx;
                let qb_x = x_faults[qb_offset];
                local_z = local_z ^ qb_x;
            }
            else if (qubit == qb) {
                // qb: Z += qa's X
                let qa_offset = qa * params.shot_words + word_idx;
                let qa_x = x_faults[qa_offset];
                local_z = local_z ^ qa_x;
            }
        }
        else if (gate_type == GATE_SWAP) {
            let qa = qubit1;
            let qb = qubit2;

            if (qubit == qa) {
                // Swap with qb
                let qb_offset = qb * params.shot_words + word_idx;
                let tmp_x = local_x;
                let tmp_z = local_z;
                local_x = x_faults[qb_offset];
                local_z = z_faults[qb_offset];
                // Note: qb will pick up our values when it processes this gate
            }
            else if (qubit == qb) {
                // Swap with qa
                let qa_offset = qa * params.shot_words + word_idx;
                let tmp_x = local_x;
                let tmp_z = local_z;
                local_x = x_faults[qa_offset];
                local_z = z_faults[qa_offset];
            }
        }
        // Fault injection
        else if (gate_type == FAULT_X && qubit1 == qubit) {
            // Inject X fault on all shots
            local_x = 0xFFFFFFFFu;
        }
        else if (gate_type == FAULT_Z && qubit1 == qubit) {
            // Inject Z fault on all shots
            local_z = 0xFFFFFFFFu;
        }
        else if (gate_type == FAULT_Y && qubit1 == qubit) {
            // Inject Y fault on all shots (X and Z)
            local_x = 0xFFFFFFFFu;
            local_z = 0xFFFFFFFFu;
        }
        else if (gate_type == FAULT_DEPOL1 && qubit1 == qubit) {
            // Probabilistic single-qubit depolarizing
            // qubit2 contains the threshold
            let threshold = qubit2;

            // Process each shot in this word
            for (var bit = 0u; bit < 32u; bit = bit + 1u) {
                let shot = word_idx * 32u + bit;
                if (shot >= params.num_shots) {
                    break;
                }

                // Get random value for this shot
                let rand = hash(random_bits[shot], shot, gate_idx);

                // Check if fault occurs
                if (rand < threshold) {
                    // Select Pauli: 0=X, 1=Y, 2=Z
                    let pauli = (rand >> 16u) % 3u;
                    let bit_mask = 1u << bit;

                    if (pauli == 0u) {
                        // X fault
                        local_x = local_x ^ bit_mask;
                    } else if (pauli == 1u) {
                        // Y fault (X and Z)
                        local_x = local_x ^ bit_mask;
                        local_z = local_z ^ bit_mask;
                    } else {
                        // Z fault
                        local_z = local_z ^ bit_mask;
                    }
                }
            }
        }
        else if (gate_type == FAULT_DEPOL2) {
            // Two-qubit depolarizing: uses 4 words in queue
            let qa = qubit1;
            let qb = qubit2;
            let threshold = gate_queue[i + 3u];

            if (qubit == qa || qubit == qb) {
                for (var bit = 0u; bit < 32u; bit = bit + 1u) {
                    let shot = word_idx * 32u + bit;
                    if (shot >= params.num_shots) {
                        break;
                    }

                    let rand = hash(random_bits[shot], shot, gate_idx);

                    if (rand < threshold) {
                        // Select one of 15 non-II Paulis
                        let selection = (rand >> 16u) % 15u;
                        let pauli_a = selection / 4u;  // 0=I, 1=X, 2=Y, 3=Z
                        var pauli_b = selection % 4u;

                        // Avoid II (when pauli_a == 0 and pauli_b == 0)
                        if (pauli_a == 0u && pauli_b == 0u) {
                            pauli_b = 1u;  // Make it IX instead
                        }

                        let bit_mask = 1u << bit;

                        // Apply fault to this qubit based on which one we are
                        let my_pauli = select(pauli_b, pauli_a, qubit == qa);

                        if (my_pauli == 1u) {
                            // X
                            local_x = local_x ^ bit_mask;
                        } else if (my_pauli == 2u) {
                            // Y
                            local_x = local_x ^ bit_mask;
                            local_z = local_z ^ bit_mask;
                        } else if (my_pauli == 3u) {
                            // Z
                            local_z = local_z ^ bit_mask;
                        }
                        // my_pauli == 0 means I (no fault)
                    }
                }
            }

            // Skip extra word for 2Q depol
            i = i + 1u;
        }

        i = i + 3u;
        gate_idx = gate_idx + 1u;
    }

    // Write back
    x_faults[offset] = local_x;
    z_faults[offset] = local_z;
}
