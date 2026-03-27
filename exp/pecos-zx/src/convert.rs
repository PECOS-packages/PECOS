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

//! Conversion between PECOS circuits and QuiZX ZX graphs.

use num_traits::One;
use pecos_core::Angle64;
use pecos_core::gate_type::GateType;
use pecos_quantum::DagCircuit;
use quizx::circuit::Circuit as ZxCircuit;
use quizx::extract::ToCircuit;
use quizx::gate::{GType, Gate as ZxGate};
use quizx::phase::Phase;

use crate::ZxGraph;

/// Errors that can occur during conversion.
#[derive(Debug, thiserror::Error)]
pub enum ConvertError {
    #[error("unsupported gate type for ZX conversion: {0}")]
    UnsupportedGate(String),
    #[error("circuit extraction failed: {0}")]
    ExtractionFailed(String),
}

/// Convert an `Angle64` (full turns) to a QuiZX `Phase` (half turns, i.e. multiples of pi).
///
/// PECOS stores angles as fractions of a full turn (2pi radians).
/// QuiZX stores phases as multiples of pi (half turns).
/// Conversion: `half_turns = 2.0 * full_turns`.
fn angle_to_phase(angle: Angle64) -> Phase {
    let full_turns = angle.to_radians() / std::f64::consts::TAU;
    let half_turns = 2.0 * full_turns;
    Phase::from_f64(half_turns)
}

/// Convert a QuiZX `Phase` (half turns) to an `Angle64` (full turns).
fn phase_to_angle(phase: Phase) -> Angle64 {
    let half_turns = phase.to_f64();
    let full_turns = half_turns / 2.0;
    Angle64::from_turns(full_turns)
}

/// Emit the best-matching PECOS gate for a `ZPhase` with the given phase.
///
/// Recognizes common rational multiples of pi and emits the specific gate
/// (S, Sdg, T, Tdg, Z) instead of a generic RZ.
fn zphase_to_dag_gate(dag: &mut DagCircuit, phase: Phase, q: usize) {
    if phase == Phase::new((1, 2)) {
        dag.sz(&[q]);
    } else if phase == Phase::new((-1, 2)) {
        dag.szdg(&[q]);
    } else if phase == Phase::new((1, 4)) {
        dag.t(&[q]);
    } else if phase == Phase::new((-1, 4)) {
        dag.tdg(&[q]);
    } else if phase == Phase::one() {
        dag.z(&[q]);
    } else {
        dag.rz(phase_to_angle(phase), &[q]);
    }
}

/// Emit the best-matching PECOS gate for an `XPhase` with the given phase.
///
/// Recognizes common rational multiples of pi and emits the specific gate
/// (SX, SXdg, X) instead of a generic RX.
fn xphase_to_dag_gate(dag: &mut DagCircuit, phase: Phase, q: usize) {
    if phase == Phase::new((1, 2)) {
        dag.sx(&[q]);
    } else if phase == Phase::new((-1, 2)) {
        dag.sxdg(&[q]);
    } else if phase == Phase::one() {
        dag.x(&[q]);
    } else {
        dag.rx(phase_to_angle(phase), &[q]);
    }
}

/// Convert a PECOS `DagCircuit` to a QuiZX ZX graph.
///
/// The circuit is first translated into a QuiZX `Circuit`, then converted
/// to a ZX graph using QuiZX's `to_graph()`.
///
/// # Errors
///
/// Returns `ConvertError::UnsupportedGate` if the circuit contains gate types
/// that have no ZX calculus equivalent (e.g., `Idle`, `CCX`).
pub fn dag_to_zx(dag: &DagCircuit) -> Result<ZxGraph, ConvertError> {
    let zx_circuit = dag_to_zx_circuit(dag)?;
    Ok(zx_circuit.to_graph())
}

/// Convert a PECOS `DagCircuit` to a QuiZX `Circuit`.
///
/// # Errors
///
/// Returns `ConvertError::UnsupportedGate` for unsupported gate types.
pub fn dag_to_zx_circuit(dag: &DagCircuit) -> Result<ZxCircuit, ConvertError> {
    let num_qubits = dag.width();
    let mut zx_circ = ZxCircuit::new(num_qubits);

    for (_node, gate) in dag.iter_gates_topo() {
        let arity = gate.quantum_arity();
        let qubits = &gate.qubits;

        // Process each individual gate operation within this Gate
        // (PECOS gates can contain multiple operations, e.g., Gate::x(&[0, 1, 2]))
        for chunk in qubits.chunks(arity) {
            let qs: Vec<usize> = chunk.iter().map(|q| usize::from(*q)).collect();

            match gate.gate_type {
                // Single-qubit Clifford gates
                GateType::H => {
                    zx_circ.push(ZxGate::new(GType::HAD, qs));
                }
                GateType::X => {
                    zx_circ.push(ZxGate::new(GType::NOT, qs));
                }
                GateType::Z => {
                    zx_circ.push(ZxGate::new(GType::Z, qs));
                }
                GateType::Y => {
                    // Y = i*X*Z, represent as ZPhase(pi) then XPhase(pi)
                    zx_circ.push(ZxGate::new_with_phase(
                        GType::ZPhase,
                        qs.clone(),
                        Phase::one(),
                    ));
                    zx_circ.push(ZxGate::new_with_phase(GType::XPhase, qs, Phase::one()));
                }
                GateType::SZ => {
                    zx_circ.push(ZxGate::new(GType::S, qs));
                }
                GateType::SZdg => {
                    zx_circ.push(ZxGate::new(GType::Sdg, qs));
                }
                GateType::T => {
                    zx_circ.push(ZxGate::new(GType::T, qs));
                }
                GateType::Tdg => {
                    zx_circ.push(ZxGate::new(GType::Tdg, qs));
                }
                GateType::SX => {
                    // SX = RX(pi/2) = XPhase(1/2) in half-turns
                    zx_circ.push(ZxGate::new_with_phase(
                        GType::XPhase,
                        qs,
                        Phase::new((1, 2)),
                    ));
                }
                GateType::SXdg => {
                    zx_circ.push(ZxGate::new_with_phase(
                        GType::XPhase,
                        qs,
                        Phase::new((-1, 2)),
                    ));
                }

                // Rotation gates (parameterized)
                GateType::RZ => {
                    let phase = angle_to_phase(gate.angles[0]);
                    zx_circ.push(ZxGate::new_with_phase(GType::ZPhase, qs, phase));
                }
                GateType::RX => {
                    let phase = angle_to_phase(gate.angles[0]);
                    zx_circ.push(ZxGate::new_with_phase(GType::XPhase, qs, phase));
                }

                // Two-qubit gates
                GateType::CX => {
                    zx_circ.push(ZxGate::new(GType::CNOT, qs));
                }
                GateType::CZ => {
                    zx_circ.push(ZxGate::new(GType::CZ, qs));
                }
                GateType::SWAP => {
                    zx_circ.push(ZxGate::new(GType::SWAP, qs));
                }

                // Measurement and prep
                GateType::MZ | GateType::MeasureFree => {
                    zx_circ.push(ZxGate::new(GType::Measure, qs));
                }
                GateType::PZ | GateType::QAlloc => {
                    zx_circ.push(ZxGate::new(GType::InitAncilla, qs));
                }

                // Skip identity
                GateType::I => {}

                other => {
                    return Err(ConvertError::UnsupportedGate(format!("{other:?}")));
                }
            }
        }
    }

    Ok(zx_circ)
}

/// Convert a QuiZX ZX graph back to a PECOS `DagCircuit`.
///
/// Uses QuiZX's circuit extraction to recover a circuit from the ZX graph,
/// then translates the QuiZX circuit to a PECOS `DagCircuit`.
///
/// # Errors
///
/// Returns `ConvertError::ExtractionFailed` if circuit extraction fails
/// (e.g., the graph has no gflow), or `ConvertError::UnsupportedGate`
/// if the extracted circuit uses gate types not supported in PECOS.
pub fn zx_to_dag(graph: &ZxGraph) -> Result<DagCircuit, ConvertError> {
    let zx_circ = graph
        .to_circuit()
        .map_err(|e| ConvertError::ExtractionFailed(e.0))?;
    zx_circuit_to_dag(&zx_circ)
}

/// Convert a QuiZX `Circuit` to a PECOS `DagCircuit`.
///
/// # Errors
///
/// Returns `ConvertError::UnsupportedGate` for unsupported QuiZX gate types.
pub fn zx_circuit_to_dag(zx_circ: &ZxCircuit) -> Result<DagCircuit, ConvertError> {
    let mut dag = DagCircuit::new();

    for zx_gate in &zx_circ.gates {
        match zx_gate.t {
            GType::HAD => {
                for &q in &zx_gate.qs {
                    dag.h(&[q]);
                }
            }
            GType::NOT => {
                for &q in &zx_gate.qs {
                    dag.x(&[q]);
                }
            }
            GType::Z => {
                for &q in &zx_gate.qs {
                    dag.z(&[q]);
                }
            }
            GType::S => {
                for &q in &zx_gate.qs {
                    dag.sz(&[q]);
                }
            }
            GType::Sdg => {
                for &q in &zx_gate.qs {
                    dag.szdg(&[q]);
                }
            }
            GType::T => {
                for &q in &zx_gate.qs {
                    dag.t(&[q]);
                }
            }
            GType::Tdg => {
                for &q in &zx_gate.qs {
                    dag.tdg(&[q]);
                }
            }
            GType::ZPhase => {
                for &q in &zx_gate.qs {
                    zphase_to_dag_gate(&mut dag, zx_gate.phase, q);
                }
            }
            GType::XPhase => {
                for &q in &zx_gate.qs {
                    xphase_to_dag_gate(&mut dag, zx_gate.phase, q);
                }
            }
            GType::CNOT => {
                assert!(
                    zx_gate.qs.len() == 2,
                    "CNOT gate must have exactly 2 qubits"
                );
                dag.cx(&[(zx_gate.qs[0], zx_gate.qs[1])]);
            }
            GType::CZ => {
                assert!(zx_gate.qs.len() == 2, "CZ gate must have exactly 2 qubits");
                dag.cz(&[(zx_gate.qs[0], zx_gate.qs[1])]);
            }
            GType::SWAP => {
                assert!(
                    zx_gate.qs.len() == 2,
                    "SWAP gate must have exactly 2 qubits"
                );
                dag.swap(&[(zx_gate.qs[0], zx_gate.qs[1])]);
            }
            GType::Measure | GType::MeasureReset => {
                for &q in &zx_gate.qs {
                    dag.mz(&[q]);
                }
            }
            GType::InitAncilla => {
                for &q in &zx_gate.qs {
                    dag.pz(&[q]);
                }
            }
            other => {
                return Err(ConvertError::UnsupportedGate(format!("{other:?}")));
            }
        }
    }

    Ok(dag)
}

#[cfg(test)]
mod tests {
    use super::*;
    use quizx::graph::GraphLike;

    #[test]
    fn test_angle_phase_roundtrip() {
        let angle = Angle64::HALF_TURN;
        let phase = angle_to_phase(angle);
        let roundtrip = phase_to_angle(phase);
        let diff = (angle.to_radians() - roundtrip.to_radians()).abs();
        assert!(diff < 1e-6, "roundtrip error: {diff}");
    }

    #[test]
    fn test_angle_phase_quarter_turn() {
        let angle = Angle64::QUARTER_TURN;
        let phase = angle_to_phase(angle);
        assert!((phase.to_f64() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_bell_state_roundtrip() {
        let mut dag = DagCircuit::new();
        dag.h(&[0]);
        dag.cx(&[(0, 1)]);

        let graph = dag_to_zx(&dag).expect("conversion should succeed");
        assert!(graph.num_vertices() > 0);
    }

    #[test]
    fn test_dag_to_zx_circuit_basic() {
        let mut dag = DagCircuit::new();
        dag.h(&[0]);
        dag.cx(&[(0, 1)]);
        dag.sz(&[0]);
        dag.t(&[1]);

        let zx_circ = dag_to_zx_circuit(&dag).expect("conversion should succeed");
        assert_eq!(zx_circ.num_qubits(), 2);
        assert_eq!(zx_circ.num_gates(), 4);
    }

    #[test]
    fn test_sx_roundtrip() {
        // SX -> ZX circuit -> DagCircuit should recover SX (not RX)
        let mut dag = DagCircuit::new();
        dag.sx(&[0]);
        dag.sxdg(&[1]);

        let zx_circ = dag_to_zx_circuit(&dag).expect("forward conversion");
        let recovered = zx_circuit_to_dag(&zx_circ).expect("reverse conversion");

        let gates: Vec<_> = recovered.iter_gates_topo().collect();
        assert_eq!(gates.len(), 2);
        assert_eq!(gates[0].1.gate_type, GateType::SX);
        assert_eq!(gates[1].1.gate_type, GateType::SXdg);
    }

    #[test]
    fn test_zphase_common_values() {
        // ZPhase with known rational phases should recover specific gates
        let mut zx_circ = ZxCircuit::new(5);
        zx_circ.push(ZxGate::new_with_phase(
            GType::ZPhase,
            vec![0],
            Phase::new((1, 2)),
        ));
        zx_circ.push(ZxGate::new_with_phase(
            GType::ZPhase,
            vec![1],
            Phase::new((-1, 2)),
        ));
        zx_circ.push(ZxGate::new_with_phase(
            GType::ZPhase,
            vec![2],
            Phase::new((1, 4)),
        ));
        zx_circ.push(ZxGate::new_with_phase(
            GType::ZPhase,
            vec![3],
            Phase::new((-1, 4)),
        ));
        zx_circ.push(ZxGate::new_with_phase(GType::ZPhase, vec![4], Phase::one()));

        let dag = zx_circuit_to_dag(&zx_circ).expect("conversion");

        // Collect gate types by qubit (order-independent since gates are on different qubits)
        let mut gate_by_qubit = std::collections::HashMap::new();
        for (_, gate) in dag.iter_gates_topo() {
            gate_by_qubit.insert(usize::from(gate.qubits[0]), gate.gate_type);
        }

        assert_eq!(gate_by_qubit[&0], GateType::SZ);
        assert_eq!(gate_by_qubit[&1], GateType::SZdg);
        assert_eq!(gate_by_qubit[&2], GateType::T);
        assert_eq!(gate_by_qubit[&3], GateType::Tdg);
        assert_eq!(gate_by_qubit[&4], GateType::Z);
    }

    #[test]
    fn test_xphase_fallback_to_rx() {
        // Non-special XPhase values should fall through to RX
        let mut zx_circ = ZxCircuit::new(1);
        zx_circ.push(ZxGate::new_with_phase(
            GType::XPhase,
            vec![0],
            Phase::new((1, 3)), // 1/3 half-turns -- not a named gate
        ));

        let dag = zx_circuit_to_dag(&zx_circ).expect("conversion");
        let gates: Vec<_> = dag.iter_gates_topo().collect();
        assert_eq!(gates.len(), 1);
        assert_eq!(gates[0].1.gate_type, GateType::RX);
    }

    #[test]
    fn test_zphase_fallback_to_rz() {
        // Non-special ZPhase values should fall through to RZ
        let mut zx_circ = ZxCircuit::new(1);
        zx_circ.push(ZxGate::new_with_phase(
            GType::ZPhase,
            vec![0],
            Phase::new((1, 3)), // 1/3 half-turns -- not a named gate
        ));

        let dag = zx_circuit_to_dag(&zx_circ).expect("conversion");
        let gates: Vec<_> = dag.iter_gates_topo().collect();
        assert_eq!(gates.len(), 1);
        assert_eq!(gates[0].1.gate_type, GateType::RZ);
    }

    #[test]
    fn test_full_circuit_roundtrip() {
        // Full dag -> ZX -> dag roundtrip preserving gate identities
        let mut dag = DagCircuit::new();
        dag.h(&[0]);
        dag.sz(&[0]);
        dag.cx(&[(0, 1)]);
        dag.t(&[1]);

        let zx_circ = dag_to_zx_circuit(&dag).expect("forward conversion");
        let recovered = zx_circuit_to_dag(&zx_circ).expect("reverse conversion");

        let gate_types: Vec<_> = recovered
            .iter_gates_topo()
            .map(|(_, g)| g.gate_type)
            .collect();

        assert_eq!(gate_types.len(), 4);
        assert!(gate_types.contains(&GateType::H));
        assert!(gate_types.contains(&GateType::SZ));
        assert!(gate_types.contains(&GateType::CX));
        assert!(gate_types.contains(&GateType::T));
    }

    #[test]
    fn test_unsupported_gate() {
        let mut dag = DagCircuit::new();
        dag.ccx(0, 1, 2);

        let result = dag_to_zx(&dag);
        assert!(result.is_err());
    }
}
