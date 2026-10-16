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
            Angle64::try_from_radians(-theta / 2.0).unwrap(),
            &[(QubitId(1), QubitId(0))],
        ),
        Gate::rz(
            Angle64::try_from_radians(theta / 2.0).unwrap(),
            &[QubitId(0)],
        ),
    ]
}

#[test]
fn dense_controlled_rotations_are_continuous_and_exact_at_odd_turns() {
    let control_z = dense(&[Gate::z(&[QubitId(1)])]);
    for theta in [TAU, -TAU, 3.0 * TAU, -3.0 * TAU] {
        let fixed = dense(&lower_crz(theta, QubitId(1), QubitId(0)).unwrap());
        let old = dense(&old_lowering(theta));
        let error = (&fixed - &control_z).norm();
        let old_error = (&old - &control_z).norm();
        assert!(error < 1e-12);
        assert!((old_error - 4.0).abs() < 1e-12);
        for offset in [-1e-7, 1e-7] {
            let near = dense(&lower_crz(theta + offset, QubitId(1), QubitId(0)).unwrap());
            let jump = (&fixed - &near).norm();
            let old_jump = (&old - dense(&old_lowering(theta + offset))).norm();
            println!(
                "theta={theta}, offset={offset}: fixed jump={jump:.4}, old jump={old_jump:.4}"
            );
            assert!(jump < 1e-7);
            assert!((old_jump - 4.0).abs() < 1e-7);
        }
        for lowering in [lower_crx, lower_cry] {
            let matrix = dense(&lowering(theta, QubitId(1), QubitId(0)).unwrap());
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
        let fixed = dense(&lower_crz(theta, QubitId(1), QubitId(0)).unwrap());
        worst = worst.max((&fixed - dense(&old_lowering(theta))).norm());
        let shifted = dense(&lower_crz(theta + 2.0 * TAU, QubitId(1), QubitId(0)).unwrap());
        assert!((&fixed - shifted).norm() < 1e-12);
    }
    println!("800-angle worst raw-half disagreement={worst:.3e}");
    assert!(worst < 1e-12);
}

// Independent integer oracle for these small, positive, normal f64 inputs.
// Form the entire exact theta * 2^63 / TAU ratio in u128 and round once;
// production instead divides two carried base-2^64 digits after halving.
fn exact_small_half(theta: f64) -> Angle64 {
    let parts = |value: f64| {
        let bits = value.to_bits();
        (
            u128::from((bits & ((1_u64 << 52) - 1)) | (1_u64 << 52)),
            i32::try_from((bits >> 52) & 0x7ff).unwrap(),
        )
    };
    let (n, ne) = parts(theta);
    let (d, de) = parts(TAU);
    let shift = ne - de + 63;
    let numerator = n << u32::try_from(shift.max(0)).unwrap();
    let denominator = d << u32::try_from((-shift).max(0)).unwrap();
    Angle64::new(u64::try_from((numerator + denominator / 2) / denominator).unwrap())
}

#[test]
fn crz_fixed_point_legs_are_exact_negatives_and_match_dense_reference() {
    use num_complex::Complex64;
    use std::f64::consts::PI;
    let mut angles = vec![0.0, -0.0, f64::MIN_POSITIVE, -f64::MIN_POSITIVE];
    for tiny in [1e-18, 1e-17, 1e-16, 1e-15, 1e-12] {
        angles.extend([tiny, -tiny]);
    }
    for fraction in 1..=1024 {
        let theta = Angle64::new(fraction).to_radians();
        angles.extend([theta, -theta]);
        let positive = lower_crz(theta, QubitId(1), QubitId(0)).unwrap();
        let negative = lower_crz(-theta, QubitId(1), QubitId(0)).unwrap();
        let half = exact_small_half(theta);
        assert_eq!(positive[1].angles[0], half);
        assert_eq!(negative[1].angles[0], -half);
    }
    for boundary in [-2.0 * TAU, -TAU, -PI, PI, TAU, 2.0 * TAU] {
        angles.extend([
            boundary.next_down(),
            boundary,
            boundary.next_up(),
            boundary - 1e-7,
            boundary + 1e-7,
        ]);
    }
    for sample in -512..=512 {
        angles.push(f64::from(sample) * TAU / 128.0);
    }
    for theta in angles {
        let gates = lower_crz(theta, QubitId(1), QubitId(0)).unwrap();
        let rzz = &gates[gates.len() - 2];
        let rz = &gates[gates.len() - 1];
        assert_eq!(rzz.angles[0], -rz.angles[0], "theta={theta}");
        // The existing dense evaluator keeps the control-off block and global
        // phase, so this also detects a missing or spurious parity correction.
        let actual = dense(&gates);
        let mut expected = UnitaryMatrix::identity(4);
        expected[(2, 2)] = Complex64::from_polar(1.0, -theta / 2.0);
        expected[(3, 3)] = Complex64::from_polar(1.0, theta / 2.0);
        assert!((&actual - expected).norm() < 4e-15, "theta={theta}");
        // For tiny inputs an absolute 1e-12 test alone would hide a lost angle.
        // Quantizing the floating-point half contributes at most half a tick.
        // The existing dense evaluator also truncates half-angle fractions and
        // octant complements; allow three ticks in this numerical comparison.
        // The stored-angle assertions above remain exact, including odd ticks.
        if theta.abs() < 1e-12 {
            let tick = TAU * 2.0_f64.powi(-64);
            assert!(
                (actual[(2, 2)].im + (theta / 2.0).sin()).abs() <= 3.0 * tick,
                "tiny theta={theta}: {}",
                actual[(2, 2)]
            );
        }
    }
}
