// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.
use super::tests::{reconstruct, reference_unitary};
use super::*;
use pecos_simulators::state_vector_test_utils::normalized_z_projection;

#[test]
fn fixed_seed_trait_bits() {
    let mut state = StabActive::with_seed(3, 731);
    let mut records = Vec::new();
    let mut reference = pecos_simulators::StateVec::new(3).state();
    let project_x = |reference: &mut Vec<Complex64>, qubit, outcome| {
        reference_unitary(reference, 0, qubit, qubit, Angle64::ZERO);
        *reference =
            normalized_z_projection(reference, qubit, outcome, "seeded trait dense oracle");
        reference_unitary(reference, 0, qubit, qubit, Angle64::ZERO);
    };
    for _ in 0..4 {
        state.h(&[QubitId(0)]).sz(&[QubitId(1)]);
        state.rx(pecos_core::radians!(0.321), &[QubitId(0), QubitId(2)]);
        state.ry(pecos_core::radians!(-0.789), &[QubitId(1)]);
        state.rzz(pecos_core::radians!(0.456), &[(QubitId(0), QubitId(1))]);
        state.cx(&[(QubitId(1), QubitId(2))]);
        reference_unitary(&mut reference, 0, 0, 0, Angle64::ZERO);
        reference_unitary(&mut reference, 1, 1, 1, Angle64::ZERO);
        reference_unitary(&mut reference, 9, 0, 0, pecos_core::radians!(0.321));
        reference_unitary(&mut reference, 9, 2, 2, pecos_core::radians!(0.321));
        reference_unitary(&mut reference, 11, 1, 1, pecos_core::radians!(-0.789));
        reference_unitary(&mut reference, 10, 0, 1, pecos_core::radians!(0.456));
        reference_unitary(&mut reference, 6, 1, 2, Angle64::ZERO);
        let outcome = state.mx(&[QubitId(0)])[0].outcome;
        records.push(outcome);
        project_x(&mut reference, 0, outcome);
        // px delegates to mpx; observe its branch without changing gates or RNG draws.
        let preparation = state.mpx(&[QubitId(2)])[0].outcome;
        project_x(&mut reference, 2, preparation);
        if preparation {
            reference_unitary(&mut reference, 5, 2, 2, Angle64::ZERO);
        }
        // Reconstruct from physical tableau rows, independent of active rotation kernels.
        // Global phase is immaterial; compare every physical amplitude tightly.
        let reconstructed = reconstruct(&state);
        let overlap: Complex64 = reconstructed
            .iter()
            .zip(&reference)
            .map(|(a, b)| a.conj() * b)
            .sum();
        let phase = overlap / overlap.norm();
        for (actual, expected) in reconstructed.iter().zip(&reference) {
            assert!((*actual * phase - expected).norm() < 1e-12);
        }
    }
    let bits: Vec<_> = state
        .amplitudes
        .iter()
        .map(|a| (a.re.to_bits(), a.im.to_bits()))
        .collect();
    assert_eq!(records, [false, false, false, false]);
    assert_eq!(
        bits,
        [
            (13_815_872_805_157_858_747, 13_830_198_988_909_340_100),
            (4_590_476_583_996_472_602, 4_597_922_288_466_730_066)
        ]
    );
}
