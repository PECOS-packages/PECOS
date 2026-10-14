// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

//! Raw dense-matrix regressions for controlled-rotation lowering, including global phase.

use pecos_core::controlled_rotations::{lower_crx, lower_cry, lower_crz};
use pecos_core::gate_type::GateType;
use pecos_core::unitary_rep::RotationType;
use pecos_core::{Angle64, Gate, QubitId, UnitaryRep};
use pecos_quantum::unitary_matrix::{UnitaryMatrix, to_matrix_with_size};
use std::f64::consts::TAU;

fn dense(gates: &[Gate]) -> UnitaryMatrix {
    gates
        .iter()
        .fold(UnitaryMatrix::identity(4), |matrix, gate| {
            let qubits: Vec<_> = gate.qubits.iter().map(QubitId::index).collect();
            let op = match gate.gate_type {
                GateType::RZ => {
                    UnitaryRep::rotation(RotationType::RZ, gate.angles[0], qubits.as_slice())
                }
                GateType::RZZ => {
                    UnitaryRep::rotation(RotationType::RZZ, gate.angles[0], qubits.as_slice())
                }
                kind => UnitaryRep::gate(kind, qubits.as_slice()),
            };
            to_matrix_with_size(&op, 2) * matrix
        })
}

fn old_lowering(theta: f64) -> [Gate; 2] {
    [
        Gate::rzz(
            Angle64::from_radians(-theta / 2.0),
            &[(QubitId(1), QubitId(0))],
        ),
        Gate::rz(Angle64::from_radians(theta / 2.0), &[QubitId(0)]),
    ]
}

#[test]
fn dense_controlled_rotations_are_continuous_and_exact_at_odd_turns() {
    let control_z = dense(&[Gate::z(&[QubitId(1)])]);
    for theta in [TAU, -TAU, 3.0 * TAU, -3.0 * TAU] {
        let fixed = dense(&lower_crz(theta, QubitId(1), QubitId(0)));
        let old = dense(&old_lowering(theta));
        let error = (&fixed - &control_z).norm();
        let old_error = (&old - &control_z).norm();
        assert!(error < 1e-12);
        assert!((old_error - 4.0).abs() < 1e-12);
        for offset in [-1e-7, 1e-7] {
            let near = dense(&lower_crz(theta + offset, QubitId(1), QubitId(0)));
            let jump = (&fixed - &near).norm();
            let old_jump = (&old - dense(&old_lowering(theta + offset))).norm();
            println!(
                "theta={theta}, offset={offset}: fixed jump={jump:.4}, old jump={old_jump:.4}"
            );
            assert!(jump < 1e-7);
            assert!((old_jump - 4.0).abs() < 1e-7);
        }
        for lowering in [lower_crx, lower_cry] {
            let matrix = dense(&lowering(theta, QubitId(1), QubitId(0)));
            assert!((&matrix - &control_z).norm() < 1e-12);
        }
        println!("theta={theta}: fixed Z error={error:.6}, old Z error={old_error:.6}");
    }
}

#[test]
fn dense_reduced_lowering_matches_raw_half_at_800_angles() {
    let mut worst = 0.0_f64;
    for sample in 0..800 {
        let theta = -200.0 + 400.0 * (f64::from(sample) + 0.5) / 800.0;
        let fixed = dense(&lower_crz(theta, QubitId(1), QubitId(0)));
        worst = worst.max((&fixed - dense(&old_lowering(theta))).norm());
        let shifted = dense(&lower_crz(theta + 2.0 * TAU, QubitId(1), QubitId(0)));
        assert!((&fixed - shifted).norm() < 1e-12);
    }
    println!("800-angle worst raw-half disagreement={worst:.3e}");
    assert!(worst < 1e-12);
}
