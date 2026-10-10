// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use pecos_core::Angle64;
use pecos_quantum::pass::{CancelInverses, CircuitPass, MergeAdjacentRotations, StripIdentities};
use pecos_quantum::{DagCircuit, TickCircuit};
use pecos_random::{PecosRng, RngExt};
use std::fmt::Write;

pub fn snapshots() -> String {
    let mut output = String::new();
    for seed in [0, 1, 2, 31] {
        let mut rng = PecosRng::seed_from_u64(seed);
        let mut base = DagCircuit::new();
        for _ in 0..60 {
            let q = rng.random_range(0..3);
            match rng.random_range(0..10) {
                0 => {
                    base.h(&[q]);
                }
                2 => {
                    base.rz(Angle64::ZERO, &[q]);
                }
                3 => {
                    base.cx(&[(q, (q + 1) % 3)]);
                }
                4 => {
                    base.pz(&[q]);
                }
                5 => {
                    base.mz(&[q]);
                }
                6 => {
                    base.rzz(Angle64::ZERO, &[(q, (q + 1) % 3)]);
                }
                7 | 8 => {
                    base.rzz(Angle64::QUARTER_TURN, &[(q, (q + 1) % 3)]);
                    base.rzz(Angle64::QUARTER_TURN, &[(q, (q + 1) % 3)]);
                }
                _ => {
                    base.rz(Angle64::QUARTER_TURN, &[q]);
                }
            }
        }
        for pass in ["merge", "strip", "pipeline"] {
            let mut dag = base.clone();
            match pass {
                "merge" => MergeAdjacentRotations.apply_dag(&mut dag),
                "strip" => StripIdentities.apply_dag(&mut dag),
                _ => {
                    MergeAdjacentRotations.apply_dag(&mut dag);
                    StripIdentities.apply_dag(&mut dag);
                    CancelInverses.apply_dag(&mut dag);
                }
            }
            let ticks = TickCircuit::from(&dag);
            let gates: Vec<_> = ticks
                .ticks()
                .iter()
                .map(|tick| {
                    tick.gate_batches()
                        .iter()
                        .map(|gate| (gate.gate_type, gate.qubits.clone()))
                        .collect::<Vec<_>>()
                })
                .collect();
            writeln!(
                output,
                "seed={seed} pass={pass}\ntopo={:?}\nlayers={:?}\nticks={gates:?}",
                dag.topological_order(),
                dag.layers().collect::<Vec<_>>()
            )
            .unwrap();
        }
    }
    output
}
