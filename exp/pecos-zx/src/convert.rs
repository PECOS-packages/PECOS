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
/// Export uses a grid with denominator 2^62 in half turns (reduction may lower
/// the denominator). Even `Angle64` fractions are exact. Every odd fraction is
/// exactly midway between grid points and moves by exactly one `Angle64` tick
/// (2^-64 turn), resolving ties away from zero on the signed principal
/// representative in (-pi, pi]. A half turn is positive; a value rounded to
/// -pi is normalized by QuiZX to +1 half turn, the same cyclic phase.
fn angle_to_phase(angle: Angle64) -> Phase {
    let fraction = angle.fraction();
    let negative = fraction > (1_u64 << 63);
    let magnitude = if negative {
        fraction.wrapping_neg()
    } else {
        fraction
    };
    // A grid point is two ticks. Round the magnitude up at each odd-tick tie.
    let numerator = i64::try_from(magnitude.div_ceil(2))
        .expect("the rounded principal half-turn numerator is at most 2^62 and fits i64");
    Phase::new((if negative { -numerator } else { numerator }, 1_i64 << 62))
}

/// Convert the exact rational phase to ticks, rounding once (ties away from zero).
fn phase_to_angle(phase: Phase) -> Angle64 {
    let rational = phase.to_rational();
    let numerator = u128::from(rational.numer().unsigned_abs());
    let denominator = u128::from(rational.denom().unsigned_abs());
    // Rational64's numerator occupies at most 64 bits, so shifting by 63 fits.
    let ticks = ((numerator << 63) + denominator / 2) / denominator;
    let fraction = u64::try_from(ticks & u128::from(u64::MAX))
        .expect("mask restricts the rational phase to 64 fraction bits");
    Angle64::new(if rational.numer().is_negative() {
        fraction.wrapping_neg()
    } else {
        fraction
    })
}

/// Emit the best-matching PECOS gate for a `ZPhase` with the given phase.
///
/// Recognizes common rational multiples of pi and emits the specific gate
/// (S, Sdg, T, Tdg, Z) instead of a generic phase-shaped U.
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
        dag.u(Angle64::ZERO, Angle64::ZERO, phase_to_angle(phase), &[q]);
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

            if let Some(lambda) = gate.phase_angle() {
                zx_circ.push(ZxGate::new_with_phase(
                    GType::ZPhase,
                    qs,
                    angle_to_phase(lambda),
                ));
                continue;
            }
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
    use pecos_core::Gate;
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
    fn phase_conversion_rounds_odd_ties_away_from_zero() {
        let half = 1_u64 << 63;
        let denominator = 1_i64 << 62;
        // Independent integer pins: the half-turn numerator is the signed tick
        // magnitude divided by two, rounded up on odd magnitudes. Rational
        // reduction is explicit in these pins; QuiZX normalizes -1 to +1.
        for (fraction, numerator, divisor, rounded) in [
            (1, 1, denominator, 2),
            (3, 1, denominator / 2, 4),
            (5, 3, denominator, 6),
            (u64::MAX, -1, denominator, u64::MAX - 1),
            (u64::MAX - 2, -1, denominator / 2, u64::MAX - 3),
            (u64::MAX - 4, -3, denominator, u64::MAX - 5),
            (half - 1, 1, 1, half),
            (half + 1, 1, 1, half),
            (half - 3, denominator - 1, denominator, half - 2),
            (half + 3, 1 - denominator, denominator, half + 2),
        ] {
            let phase = angle_to_phase(Angle64::new(fraction));
            let rational = phase.to_rational();
            assert_eq!(
                (*rational.numer(), *rational.denom()),
                (numerator, divisor),
                "fraction {fraction}"
            );
            assert_eq!(
                phase_to_angle(phase).fraction(),
                rounded,
                "fraction {fraction}"
            );
        }
    }

    #[test]
    fn phase_conversion_preserves_even_fractions() {
        let half = 1_u64 << 63;
        let denominator = 1_i64 << 62;
        for (fraction, numerator, divisor) in [
            (0, 0, 1),
            (2, 1, denominator),
            (6, 3, denominator),
            (half / 2, 1, 2),
            (half, 1, 1),
            (half + half / 2, -1, 2),
            (half - 2, denominator - 1, denominator),
            (half + 2, 1 - denominator, denominator),
            (u64::MAX - 1, -1, denominator),
        ] {
            let angle = Angle64::new(fraction);
            let phase = angle_to_phase(angle);
            let rational = phase.to_rational();
            assert_eq!(
                (*rational.numer(), *rational.denom()),
                (numerator, divisor),
                "fraction {fraction}"
            );
            assert_eq!(phase_to_angle(phase), angle);
        }
        // Exercise every bit position with both signs, including both sides of pi.
        for bit in 1..64 {
            for fraction in [1_u64 << bit, (1_u64 << bit) - 2] {
                for angle in [Angle64::new(fraction), -Angle64::new(fraction)] {
                    assert_eq!(phase_to_angle(angle_to_phase(angle)), angle);
                }
            }
        }
    }

    #[test]
    fn phase_conversion_rounds_all_export_paths() {
        for (fraction, numerator) in [(1, 1), (u64::MAX, -1)] {
            let angle = Angle64::new(fraction);
            let mut dag = DagCircuit::new();
            dag.u(Angle64::ZERO, Angle64::ZERO, angle, &[0]);
            dag.rx(angle, &[1]);
            dag.rz(angle, &[2]);
            let circuit = dag_to_zx_circuit(&dag).expect("every Angle64 can be exported");
            assert_eq!(circuit.gates.len(), 3);
            for gate in &circuit.gates {
                let rational = gate.phase.to_rational();
                assert_eq!(
                    (*rational.numer(), *rational.denom()),
                    (numerator, 1_i64 << 62)
                );
            }
        }
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
    fn rejected_dag_update_leaves_zx_conversion_safe() {
        let mut dag = DagCircuit::new();
        let rotation = dag.add_gate_auto_wire(Gate::rz(Angle64::QUARTER_TURN, &[0]));
        let before = dag.gate(rotation).cloned().expect("rotation exists");

        let error = dag
            .update_gate(rotation, |gate| gate.angles.clear())
            .expect_err("a stored RZ cannot lose its required angle");
        assert_eq!(
            error.to_string(),
            "Invalid gate at DAG node 0: Gate RZ expected 1 angle parameters, got 0"
        );
        assert_eq!(dag.gate(rotation), Some(&before));
        dag_to_zx_circuit(&dag).expect("the refused update must leave valid ZX input");
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
    fn test_zphase_generic_preserves_phase() {
        // Generic ZPhase values retain the exact diagonal phase convention.
        let mut zx_circ = ZxCircuit::new(1);
        zx_circ.push(ZxGate::new_with_phase(
            GType::ZPhase,
            vec![0],
            Phase::new((1, 3)), // 1/3 half-turns -- not a named gate
        ));

        let dag = zx_circuit_to_dag(&zx_circ).expect("conversion");
        let gates: Vec<_> = dag.iter_gates_topo().collect();
        assert_eq!(gates.len(), 1);
        assert_eq!(
            gates[0].1.phase_angle(),
            Some(phase_to_angle(Phase::new((1, 3))))
        );
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
