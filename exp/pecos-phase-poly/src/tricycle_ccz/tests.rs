// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

use super::*;
use crate::PhasePoly;
use crate::runner::Program;
use pecos_random::PecosRng;
use std::time::Instant;

fn codes() -> [BivariateTricycleCode; 3] {
    [
        (3, [(2, 1), (2, 2)], [(0, 0), (1, 2)], [(1, 0), (2, 1)]),
        (5, [(1, 0), (0, 2)], [(0, 0), (1, 2)], [(1, 0), (1, 3)]),
        (9, [(1, 3), (2, 1)], [(0, 0), (1, 8)], [(2, 4), (2, 6)]),
    ]
    .map(|(m, a, b, c)| BivariateTricycleCode::new(3, m, &a, &b, &c).unwrap())
}

#[test]
fn noiseless_runs_are_deterministic() {
    for (code, detectors) in codes().iter().zip([72, 126, 234]) {
        for scoring in [false, true] {
            let circuit = magic_state_circuit(code, 0.0, scoring).unwrap();
            let program = Program::compile(&circuit).unwrap();
            assert_eq!(program.num_qubits(), 3 * code.num_qubits());
            assert_eq!(program.detectors().len(), detectors);
            assert_eq!(program.observables().len(), if scoring { 9 } else { 0 });
            assert_eq!(
                program.num_records(),
                detectors + if scoring { 9 } else { 0 }
            );
            for shot in program.run_shots(32, 8108).unwrap() {
                assert!(shot.detectors.iter().all(|bit| !bit));
                assert!(shot.observables.iter().all(|bit| !bit));
            }
        }
    }
}

fn support_faults(row: &[u8], offset: usize, kind: GateType) -> Vec<(usize, GateType)> {
    row.iter()
        .enumerate()
        .filter_map(|(q, &bit)| (bit == 1).then_some((offset + q, kind)))
        .collect()
}

#[test]
fn forced_faults() {
    let code = &codes()[0];
    let encoder = Encoder::new(code);
    let n = code.num_qubits();
    // Every single-qubit Pauli on every data qubit is detected.
    for q in 0..3 * n {
        for kind in [GateType::X, GateType::Y, GateType::Z] {
            let circuit = build(code, 0.0, true, &[(q, kind)]).unwrap();
            let program = Program::compile(&circuit).unwrap();
            for shot in program.run_shots(4, 9101).unwrap() {
                assert!(
                    shot.detectors.iter().any(|&bit| bit),
                    "undetected {kind:?} on {q}"
                );
            }
        }
    }
    // X stabilizers and X logicals leave the prepared logical plus state fixed.
    for block in 0..3 {
        for row in code.hx().rows().iter().chain(&encoder.logicals) {
            let faults = support_faults(row, block * n, GateType::X);
            let program = Program::compile(&build(code, 0.0, true, &faults).unwrap()).unwrap();
            for shot in program.run_shots(4, 9103).unwrap() {
                assert!(shot.detectors.iter().all(|bit| !bit));
                assert!(shot.observables.iter().all(|bit| !bit));
            }
        }
    }
    // A Z logical flips the observables paired with it in the encoder basis.
    for row in code.logical_z().rows() {
        let faults = support_faults(&row, n, GateType::Z);
        let pattern = encoder
            .logicals
            .iter()
            .map(|logical| {
                logical
                    .iter()
                    .zip(&row)
                    .fold(0, |parity, (&x, &z)| parity ^ (x & z))
                    != 0
            })
            .collect::<Vec<_>>();
        assert!(pattern.iter().any(|&bit| bit));
        let program = Program::compile(&build(code, 0.0, true, &faults).unwrap()).unwrap();
        for shot in program.run_shots(8, 9105).unwrap() {
            assert!(shot.detectors.iter().all(|bit| !bit));
            assert_eq!(&shot.observables[..3], &[false; 3]);
            assert_eq!(&shot.observables[3..6], pattern);
            assert_eq!(&shot.observables[6..], &[false; 3]);
        }
    }
}

#[test]
fn acceptance_matches_exact_css_oracle() {
    // Independent closed form, computed outside this crate. For a Pauli error
    // X^ex Z^ez before a code-space-preserving diagonal circuit on |+>_L, a shot is
    // accepted iff Hz ex = 0 and Hx ez = 0. Per block at p = 0.01 an exact Fourier
    // sum over rowspace(Hx), cross-checked by Pauli Monte Carlo, gives
    // 0.7623435107148606; the constant is its cube over three blocks. The
    // logical-error probability given acceptance is about 3e-6, far below any
    // affordable sampling bound, so target scoring is checked by `forced_faults`.
    let exact = 0.443_049_370_094_204;
    let shots = 4_000_u32;
    let code = &codes()[0];
    let started = Instant::now();
    let program = Program::compile(&magic_state_circuit(code, 0.01, false).unwrap()).unwrap();
    let mut sim = PhasePoly::with_seed(program.num_qubits(), 10_002);
    let mut rng = PecosRng::seed_from_u64(10_001);
    let mut accepted = 0_u32;
    for _ in 0..shots {
        let shot = program.run_shot(&mut sim, &mut rng).unwrap();
        accepted += u32::from(shot.detectors.iter().all(|bit| !bit));
    }
    let estimate = f64::from(accepted) / f64::from(shots);
    let bound = 5.0 * (exact * (1.0 - exact) / f64::from(shots)).sqrt();
    println!(
        "shots={shots}, accepted={accepted}, estimate={estimate:.16}, exact={exact:.16}, bound={bound:.16}, runtime={:?}",
        started.elapsed()
    );
    assert!(
        (estimate - exact).abs() <= bound,
        "estimate={estimate}, exact={exact}, bound={bound}"
    );
}

#[test]
fn invalid_probabilities() {
    let code = &codes()[0];
    for p in [f64::NAN, -0.1, 1.5, f64::INFINITY, f64::NEG_INFINITY] {
        for scoring in [false, true] {
            assert_eq!(
                magic_state_circuit(code, p, scoring).unwrap_err(),
                "p must be finite and in [0,1]"
            );
        }
    }
    assert!(magic_state_circuit(code, 1.0, true).is_ok());
}

#[test]
fn even_tori_and_disconnected_codes() {
    // Binomials also permit even torus sizes and more than three logical qubits.
    for (l, m, terms) in [(1, 2, [(0, 0), (0, 1)]), (2, 2, [(0, 0), (1, 0)])] {
        let code = BivariateTricycleCode::new(l, m, &terms, &terms, &terms).unwrap();
        let encoder = Encoder::new(&code);
        let k = code.num_logical_qubits();
        assert_eq!(encoder.logicals.len(), k);
        assert_eq!(encoder.stabilizers.len(), code.hx().rank());
        assert_eq!(encoder.non_pivots.len(), code.hz().rank());
        let program = Program::compile(&magic_state_circuit(&code, 0.0, true).unwrap()).unwrap();
        assert_eq!(program.observables().len(), 3 * k);
        assert_eq!(program.detectors().len(), 3 * (code.num_qubits() - k));
        for shot in program.run_shots(8, 1201).unwrap() {
            assert!(shot.detectors.iter().all(|bit| !bit));
            assert!(shot.observables.iter().all(|bit| !bit));
        }
    }
}
