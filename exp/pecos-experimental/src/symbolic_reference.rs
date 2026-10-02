// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

//! Exact small-circuit oracle: dense registry matrices and state-vector projections.
//! This deliberately does not use symbolic execution or Clifford lowering.

use pecos_core::gate_type::GateType;
use pecos_core::{Angle64, Gate, QubitId, Unitary, UnitaryRep};
use pecos_quantum::{DagCircuit, unitary_matrix::to_matrix_with_size};
use pecos_simulators::{
    ArbitraryRotationGateable, CliffordGateable, DenseStateVec, SymbolicSparseStab,
};

use crate::symbolic_executor::{execute_circuit_symbolic, named_action};

pub(crate) fn gate_cases() -> Vec<Gate> {
    let mut gates: Vec<_> = (0..=u8::MAX)
        .filter_map(|value| GateType::try_from(value).ok())
        .filter(|&gate| named_action(gate, 0).is_ok())
        .map(|gate| {
            Gate::new(
                gate,
                vec![],
                vec![0.0; gate.classical_arity()],
                (0..gate.quantum_arity()).map(QubitId).collect::<Vec<_>>(),
            )
        })
        .collect();
    for angle in [
        Angle64::ZERO,
        Angle64::QUARTER_TURN,
        Angle64::HALF_TURN,
        Angle64::THREE_QUARTERS_TURN,
    ] {
        for gate in [
            GateType::RX,
            GateType::RY,
            GateType::RZ,
            GateType::RXX,
            GateType::RYY,
            GateType::RZZ,
        ] {
            gates.push(Gate::with_angles(
                gate,
                vec![angle],
                (0..gate.quantum_arity()).map(QubitId).collect::<Vec<_>>(),
            ));
        }
        for axis in [
            Angle64::ZERO,
            Angle64::QUARTER_TURN,
            Angle64::HALF_TURN,
            Angle64::THREE_QUARTERS_TURN,
        ] {
            gates.push(Gate::rxy1q(angle, axis, &[0]));
            gates.push(Gate::rxyxy2q(angle, axis, &[(0, 1)]));
        }
        gates.push(Gate::u(Angle64::ZERO, Angle64::ZERO, angle, &[0]));
    }
    gates.push(Gate::rxy1q(
        Angle64::from_turns(0.25 + 1e-12),
        Angle64::from_turns(0.5 - 1e-12),
        &[0],
    ));
    gates
}

/// Product Pauli eigenstates and all X/Y/Z readouts give process tomography.
pub(crate) fn tomography_circuit(gate: &Gate, input: usize, readout: usize) -> (Vec<Gate>, usize) {
    let mut gates = Vec::new();
    for (q, basis) in [(0, input % 4), (1, input / 4)] {
        if basis == 1 {
            gates.push(Gate::x(&[q]));
        }
        if basis >= 2 {
            gates.push(Gate::h(&[q]));
        }
        if basis == 3 {
            gates.push(Gate::sz(&[q]));
        }
    }
    let target = gates.len();
    gates.push(gate.clone());
    if matches!(gate.gate_type, GateType::QFree | GateType::MeasureFree) {
        gates.push(Gate::pz(&[0]));
    }
    for (q, basis) in [(0, readout % 3), (1, readout / 3)] {
        if basis == 2 {
            gates.push(Gate::szdg(&[q]));
        }
        if basis > 0 {
            gates.push(Gate::h(&[q]));
        }
        gates.push(Gate::mz(&[q]));
    }
    (gates, target)
}

pub(crate) fn circuit(gates: &[Gate]) -> DagCircuit {
    let mut circuit = DagCircuit::new();
    for gate in gates {
        circuit.add_gate_auto_wire(gate.clone());
    }
    circuit
}

fn reference_unitary(sim: &mut DenseStateVec, gate: &Gate) {
    let qs = &gate.qubits;
    let angles = &gate.angles;
    match gate.gate_type {
        GateType::RX => {
            sim.rx(angles[0], qs);
        }
        GateType::RY => {
            sim.ry(angles[0], qs);
        }
        GateType::RZ => {
            sim.rz(angles[0], qs);
        }
        GateType::RXX => {
            sim.rxx(angles[0], &[(qs[0], qs[1])]);
        }
        GateType::RYY => {
            sim.ryy(angles[0], &[(qs[0], qs[1])]);
        }
        GateType::RZZ => {
            sim.rzz(angles[0], &[(qs[0], qs[1])]);
        }
        GateType::RXY1Q => {
            sim.rxy1q(angles[0], angles[1], qs);
        }
        GateType::RXYXY2Q => {
            sim.rxyxy2q(angles[0], angles[1], &[(qs[0], qs[1])]);
        }
        GateType::U => {
            sim.u(angles[0], angles[1], angles[2], qs);
        }
        GateType::QFree
        | GateType::Idle
        | GateType::MeasCrosstalkGlobalPayload
        | GateType::MeasCrosstalkLocalPayload
        | GateType::TrackedPauliMeta => {}
        _ => {
            let unitary = Unitary::named(gate.gate_type);
            let rep = UnitaryRep::Gate(unitary, qs.iter().map(QubitId::index).collect());
            let matrix = to_matrix_with_size(&rep, 2);
            let amplitudes = sim.state();
            for row in 0..4 {
                sim.set_amplitude(
                    row,
                    (0..4).map(|col| matrix[(row, col)] * amplitudes[col]).sum(),
                );
            }
        }
    }
}

/// Branch on both projective outcomes, never sampling the reference simulator.
pub(crate) fn reference_distribution(gates: &[Gate]) -> Vec<f64> {
    let mut branches = vec![(DenseStateVec::new(2), 0usize, 1.0)];
    let mut records = 0;
    for gate in gates {
        let measured = matches!(
            gate.gate_type,
            GateType::MX
                | GateType::MZ
                | GateType::MPZ
                | GateType::MeasureFree
                | GateType::MeasureLeaked
        );
        let prepared = matches!(
            gate.gate_type,
            GateType::PZ | GateType::PX | GateType::QAlloc
        );
        if !measured && !prepared {
            for (sim, _, _) in &mut branches {
                reference_unitary(sim, gate);
            }
            continue;
        }
        let q = gate.qubits[0].index();
        let mut next = Vec::new();
        for (mut sim, record, weight) in branches {
            if gate.gate_type == GateType::MX {
                sim.h(&gate.qubits);
            }
            let amplitudes = sim.state();
            for outcome in 0..2 {
                let probability: f64 = amplitudes
                    .iter()
                    .enumerate()
                    .filter(|(basis, _)| (basis >> q) & 1 == outcome)
                    .map(|(_, amplitude)| amplitude.norm_sqr())
                    .sum();
                if probability < 1e-14 {
                    continue;
                }
                let mut projected = sim.clone();
                for (basis, &amplitude) in amplitudes.iter().enumerate() {
                    projected.set_amplitude(
                        basis,
                        if (basis >> q) & 1 == outcome {
                            amplitude / probability.sqrt()
                        } else {
                            0.0.into()
                        },
                    );
                }
                if gate.gate_type == GateType::MX {
                    projected.h(&gate.qubits);
                }
                if (prepared || gate.gate_type == GateType::MPZ) && outcome == 1 {
                    projected.x(&gate.qubits);
                }
                if gate.gate_type == GateType::PX {
                    projected.h(&gate.qubits);
                }
                next.push((
                    projected,
                    record | if measured { outcome << records } else { 0 },
                    weight * probability,
                ));
            }
        }
        branches = next;
        records += usize::from(measured);
    }
    let mut probabilities = vec![0.0; 1 << records];
    for (_, record, weight) in branches {
        probabilities[record] += weight;
    }
    probabilities
}

pub(crate) fn symbolic_distribution(gates: &[Gate]) -> Vec<f64> {
    let mut sim = SymbolicSparseStab::new(2);
    execute_circuit_symbolic(&mut sim, &circuit(gates)).unwrap();
    let history = sim.measurement_history();
    (0..1 << history.len())
        .map(|record| {
            let mut probability = 1.0;
            for (index, measurement) in history.iter().enumerate() {
                if measurement.is_deterministic {
                    let expected = measurement
                        .outcome
                        .iter()
                        .fold(measurement.flip, |bit, dep| {
                            bit ^ ((record >> dep) & 1 != 0)
                        });
                    if expected != ((record >> index) & 1 != 0) {
                        return 0.0;
                    }
                } else {
                    probability *= 0.5;
                }
            }
            probability
        })
        .collect()
}

pub(crate) fn assert_distribution(actual: &[f64], expected: &[f64], gates: &[Gate]) {
    assert_eq!(actual.len(), expected.len());
    for (outcome, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - b).abs() < 1e-9,
            "outcome {outcome}: {actual:?} != {expected:?}; gates={gates:?}"
        );
    }
}

#[test]
fn every_supported_gate_matches_exact_reference() {
    for gate in gate_cases() {
        for input in 0..16 {
            for readout in 0..9 {
                let (gates, _) = tomography_circuit(&gate, input, readout);
                assert_distribution(
                    &symbolic_distribution(&gates),
                    &reference_distribution(&gates),
                    &gates,
                );
            }
        }
    }
}

#[test]
fn cy_phase_matches_exact_reference() {
    let gates = [
        Gate::h(&[0]),
        Gate::cy(&[(0, 1)]),
        Gate::cx(&[(0, 1)]),
        Gate::szdg(&[0]),
        Gate::h(&[0]),
        Gate::mz(&[0]),
    ];
    assert_distribution(
        &symbolic_distribution(&gates),
        &reference_distribution(&gates),
        &gates,
    );
}

#[test]
fn mx_back_action_matches_exact_reference() {
    for second in [GateType::MX, GateType::MZ] {
        let gates = [
            Gate::simple(GateType::MX, vec![QubitId(0)]),
            Gate::simple(second, vec![QubitId(0)]),
        ];
        assert_distribution(
            &symbolic_distribution(&gates),
            &reference_distribution(&gates),
            &gates,
        );
    }
}
