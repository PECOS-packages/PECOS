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
use pecos_core::Gate;
use pecos_quantum::pass::{
    AbsorbBasisGates, CancelInverses, CircuitPass, MergeAdjacentRotations, PeepholeOptimize,
    SimplifyRotations, StripIdentities, StripIdles,
};
use pecos_quantum::{DagCircuit, QubitId, TickCircuit};
use pecos_random::{PecosRng, RngExt};
use std::collections::BTreeMap;
use std::fmt::Write;

pub fn snapshots() -> String {
    let mut output = String::new();
    for seed in [0, 1, 2, 31] {
        let mut rng = PecosRng::seed_from_u64(seed);
        let mut base = DagCircuit::new();
        for _ in 0..60 {
            let q = rng.random_range(0..3);
            match rng.random_range(0..13) {
                0 => {
                    base.h(&[q]);
                }
                1 => {
                    base.rzz(Angle64::HALF_TURN, &[(q, (q + 1) % 3)]);
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
                10 => {
                    base.add_gate_auto_wire(Gate::idle(1.0, vec![QubitId(q)]));
                }
                11 => {
                    base.z(&[q]);
                }
                12 => {
                    base.h(&[q]);
                    base.cx(&[((q + 1) % 3, q)]);
                    base.h(&[q]);
                }
                _ => {
                    base.rz(Angle64::QUARTER_TURN, &[q]);
                }
            }
        }
        for pass in [
            "merge", "strip", "pipeline", "simplify", "idles", "cancel", "peephole", "absorb",
        ] {
            let mut dag = base.clone();
            match pass {
                "merge" => MergeAdjacentRotations.apply_dag(&mut dag),
                "strip" => StripIdentities.apply_dag(&mut dag),
                "simplify" => SimplifyRotations.apply_dag(&mut dag),
                "idles" => StripIdles.apply_dag(&mut dag),
                "cancel" => CancelInverses.apply_dag(&mut dag),
                "peephole" => PeepholeOptimize.apply_dag(&mut dag),
                "absorb" => AbsorbBasisGates.apply_dag(&mut dag),
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
            let gate_payloads: Vec<_> = dag
                .nodes()
                .into_iter()
                .map(|node| {
                    let gate = dag.gate(node).unwrap();
                    (
                        node,
                        gate.gate_type,
                        gate.qubits.clone(),
                        gate.angles.clone(),
                    )
                })
                .collect();
            writeln!(
                output,
                "seed={seed} pass={pass}\ngates={gate_payloads:?}\nwires={:?}\ntopo={:?}\nlayers={:?}\nticks={gates:?}",
                labeled_wires(&dag),
                dag.topological_order(),
                dag.layers().collect::<Vec<_>>()
            )
            .unwrap();
        }
    }
    output
}

/// Sorted `(from, to, qubit)` wires, joined by edge ID from each gate's own
/// incoming and outgoing wires so parallel wires keep their labels.
fn labeled_wires(dag: &DagCircuit) -> Vec<(usize, usize, QubitId)> {
    let mut sources = BTreeMap::new();
    for node in dag.nodes() {
        for (edge, _) in dag.outgoing_wires(node) {
            sources.insert(edge, node);
        }
    }
    let mut wires: Vec<_> = dag
        .nodes()
        .into_iter()
        .flat_map(|node| {
            dag.incoming_wires(node)
                .into_iter()
                .map(move |(edge, qubit)| (edge, node, qubit))
        })
        .map(|(edge, to, qubit)| (sources[&edge], to, qubit))
        .collect();
    wires.sort_unstable();
    wires
}
