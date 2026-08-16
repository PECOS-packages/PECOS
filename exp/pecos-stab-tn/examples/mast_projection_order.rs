// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
// express or implied. See the License for the specific language governing permissions and
// limitations under the License.

//! Compare deferred-projection ordering for seeded random Clifford+T circuits.
//!
//! Circuit generation follows the `next_rng` xorshift helper and random
//! Clifford-layer pattern in `disent_firing_rate.rs`.

use pecos_core::{Angle64, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable};
use pecos_stab_tn::stab_mps::mast::{Mast, ProjectionOrder};
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
enum CircuitGate {
    H(usize),
    Sz(usize),
    Cx(usize, usize),
    T(usize),
}

/// Same xorshift generator as `examples/disent_firing_rate.rs`.
fn next_rng(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn random_clifford_t_circuit(num_qubits: usize, t_count: usize, seed: u64) -> Vec<CircuitGate> {
    let mut gates = (0..num_qubits).map(CircuitGate::H).collect::<Vec<_>>();
    let mut rng_state = seed.wrapping_add(1);

    for _ in 0..t_count {
        for _ in 0..3 {
            let gate_type = next_rng(&mut rng_state) % 3;
            let q0 = (next_rng(&mut rng_state) % num_qubits as u64) as usize;
            match gate_type {
                0 => gates.push(CircuitGate::H(q0)),
                1 => gates.push(CircuitGate::Sz(q0)),
                _ => {
                    let q1 = loop {
                        let candidate = (next_rng(&mut rng_state) % num_qubits as u64) as usize;
                        if candidate != q0 {
                            break candidate;
                        }
                    };
                    gates.push(CircuitGate::Cx(q0, q1));
                }
            }
        }
        let target = (next_rng(&mut rng_state) % num_qubits as u64) as usize;
        gates.push(CircuitGate::T(target));

        // Make the benchmark sensitive to correction timing: after roughly
        // half of the injections, change the target's basis before projection.
        if next_rng(&mut rng_state) & 1 != 0 {
            gates.push(CircuitGate::H(target));
        }
    }
    gates
}

fn run_circuit(
    num_qubits: usize,
    t_count: usize,
    simulator_seed: u64,
    gates: &[CircuitGate],
    order: ProjectionOrder,
) -> (usize, usize, usize, usize, Duration) {
    let start = Instant::now();
    let mut mast = Mast::with_seed(num_qubits, t_count, simulator_seed).projection_order(order);
    let t = Angle64::QUARTER_TURN / 2u64;
    for &gate in gates {
        match gate {
            CircuitGate::H(q) => mast.h(&[QubitId(q)]),
            CircuitGate::Sz(q) => mast.sz(&[QubitId(q)]),
            CircuitGate::Cx(control, target) => mast.cx(&[(QubitId(control), QubitId(target))]),
            CircuitGate::T(q) => mast.rz(t, &[QubitId(q)]),
        };
    }
    mast.project_all();
    let post_projection_bond = mast.max_bond_dim();
    let mut exact_measure_peak = post_projection_bond;
    for q in 0..num_qubits {
        let _ = mast.mz(&[QubitId(q)]);
        exact_measure_peak = exact_measure_peak.max(mast.max_bond_dim());
    }
    let elapsed = start.elapsed();
    (
        mast.projection_peak_bond(),
        post_projection_bond,
        exact_measure_peak,
        mast.max_bond_dim(),
        elapsed,
    )
}

fn main() {
    let circuit_seed = 0x5eed_c1ff_07d0_2026;
    let simulator_seed = 0x5eed_5a1f_2026_0814;

    println!("MAST deferred-projection order comparison");
    println!(
        "{:<6} {:<8} {:<10} {:>15} {:>12} {:>15} {:>12} {:>14}",
        "data",
        "T-count",
        "order",
        "proj peak",
        "post-proj",
        "exact-mz peak",
        "final bond",
        "wall time (s)"
    );
    println!("{:-<108}", "");

    // Exact sample-then-force data measurement can saturate the default MPS
    // bond cap. Keep this routine benchmark small enough to finish while still
    // showing how that compensation cost scales with projection order.
    for num_qubits in [8usize, 10, 12] {
        for t_count in [num_qubits, 2 * num_qubits] {
            let gates = random_clifford_t_circuit(num_qubits, t_count, circuit_seed);
            for order in [ProjectionOrder::Input, ProjectionOrder::MinSpan] {
                let (
                    projection_peak,
                    post_projection_bond,
                    exact_measure_peak,
                    final_bond,
                    elapsed,
                ) = run_circuit(num_qubits, t_count, simulator_seed, &gates, order);
                println!(
                    "{num_qubits:<6} {t_count:<8} {order:<10?} {projection_peak:>15} \
                     {post_projection_bond:>12} {exact_measure_peak:>15} {final_bond:>12} \
                     {:>14.6}",
                    elapsed.as_secs_f64()
                );
            }
        }
    }
}
