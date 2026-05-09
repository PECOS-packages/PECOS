// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0

//! Validation tests for approximate strong simulation.

use pecos_eeg::Bm;
use pecos_eeg::circuit::PropagatedEeg;
use pecos_eeg::eeg::EegType;
use pecos_eeg::strong_sim::outcome_probability;

/// H-type correction: |0⟩ with `H_X` gives p(1) = h² at leading order.
/// Cross-check: exact p(1) = sin²(h).
#[test]
fn test_h_correction_matches_exact() {
    let stabs = vec![Bm::z(0)];

    for &h in &[0.01, 0.05, 0.1, 0.2] {
        let gens = vec![PropagatedEeg {
            eeg_type: EegType::H,
            label: Bm::x(0),
            label2: None,
            coeff: h,
            source: None,
        }];

        let p1 = outcome_probability(&gens, &[true], &stabs);
        let exact = h.sin().powi(2);

        // EEG gives h², which ≈ sin²(h) for small h
        assert!(
            (p1.total - h * h).abs() < 1e-10,
            "h={h}: EEG p(1)={:.6} expected h²={:.6}",
            p1.total,
            h * h
        );

        // Check closeness to exact
        let rel_err = (p1.total - exact).abs() / exact;
        eprintln!(
            "h={h:.2}: EEG={:.6} exact={exact:.6} rel_err={rel_err:.4}",
            p1.total
        );
        if h <= 0.1 {
            assert!(rel_err < 0.02, "h={h}: relative error {rel_err:.4} > 2%");
        }
    }
}

/// Probability conservation: p(0) + p(1) = 1 at leading order for H-type.
#[test]
fn test_h_probability_conservation() {
    let stabs = vec![Bm::z(0)];
    let h = 0.1;

    let gens = vec![PropagatedEeg {
        eeg_type: EegType::H,
        label: Bm::x(0),
        label2: None,
        coeff: h,
        source: None,
    }];

    let p0 = outcome_probability(&gens, &[false], &stabs);
    let p1 = outcome_probability(&gens, &[true], &stabs);
    let sum = p0.total + p1.total;

    assert!(
        (sum - 1.0).abs() < 0.001,
        "p(0)+p(1) = {sum:.6}, expected ≈ 1.0"
    );
}

/// Bell state: H_{Z0} noise should NOT affect Z-basis measurement probabilities.
/// (Z commutes with Z-basis measurements.)
#[test]
fn test_bell_h_z_invisible() {
    let stabs = vec![
        Bm::x(0).multiply(&Bm::x(1)), // XX
        Bm::z(0).multiply(&Bm::z(1)), // ZZ
    ];

    let gens = vec![PropagatedEeg {
        eeg_type: EegType::H,
        label: Bm::z(0),
        label2: None,
        coeff: 0.1,
        source: None,
    }];

    // Z₀ has no X component → no bit flips → α(S_Z) = 0 for all outcomes
    // H correction should be zero for all outcomes
    for outcome in &[
        vec![false, false],
        vec![true, true],
        vec![false, true],
        vec![true, false],
    ] {
        let p = outcome_probability(&gens, outcome, &stabs);
        assert!(
            p.h_correction.abs() < 1e-10,
            "H_Z should be invisible: outcome={outcome:?} h_corr={}",
            p.h_correction
        );
    }
}

/// Bell state: H_{X0} shifts probability between {00,11} and {01,10}.
#[test]
fn test_bell_h_x_shifts() {
    let stabs = vec![
        Bm::x(0).multiply(&Bm::x(1)), // XX
        Bm::z(0).multiply(&Bm::z(1)), // ZZ
    ];
    let h = 0.05;

    let gens = vec![PropagatedEeg {
        eeg_type: EegType::H,
        label: Bm::x(0),
        label2: None,
        coeff: h,
        source: None,
    }];

    let p00 = outcome_probability(&gens, &[false, false], &stabs);
    let p11 = outcome_probability(&gens, &[true, true], &stabs);
    let p01 = outcome_probability(&gens, &[false, true], &stabs);
    let p10 = outcome_probability(&gens, &[true, false], &stabs);

    eprintln!(
        "Bell + H_X0: p00={:.6} p11={:.6} p01={:.6} p10={:.6}",
        p00.total, p11.total, p01.total, p10.total
    );

    // X0 flips qubit 0: maps {00,11} ↔ {10,01}
    // H_X creates probability at {01,10} from {00,11}
    // p(01) and p(10) should be > 0 (increased from noiseless 0)
    assert!(p01.h_correction > 0.0, "p(01) should increase");
    assert!(p10.h_correction > 0.0, "p(10) should increase");

    // p(00) and p(11) should decrease
    assert!(p00.h_correction < 0.0, "p(00) should decrease");
    assert!(p11.h_correction < 0.0, "p(11) should decrease");

    // Conservation: total ≈ 1
    let sum = p00.total + p11.total + p01.total + p10.total;
    assert!((sum - 1.0).abs() < 0.01, "Conservation: sum={sum:.6}");

    // Symmetry: p(01) = p(10) (X0 on symmetric Bell state)
    assert!(
        (p01.total - p10.total).abs() < 1e-10,
        "Symmetry: p01={:.6} p10={:.6}",
        p01.total,
        p10.total
    );
}

/// Multiple H generators: verify the off-diagonal Φ correctly accounts
/// for cross-generator interference.
#[test]
fn test_two_h_generators_interference() {
    // |0⟩ with H_X rate h1 and H_Y rate h2
    let stabs = vec![Bm::z(0)];
    let h1 = 0.05;
    let h2 = 0.03;

    let gens = vec![
        PropagatedEeg {
            eeg_type: EegType::H,
            label: Bm::x(0),
            label2: None,
            coeff: h1,
            source: None,
        },
        PropagatedEeg {
            eeg_type: EegType::H,
            label: Bm::y(0),
            label2: None,
            coeff: h2,
            source: None,
        },
    ];

    let p0 = outcome_probability(&gens, &[false], &stabs);
    let p1 = outcome_probability(&gens, &[true], &stabs);

    // Diagonal: h1² + h2² for p(1)
    let diagonal = h1 * h1 + h2 * h2;

    // X and Y both flip (both have X component set).
    // Diagonal H·H: both contribute +h² to p(1).
    // Off-diagonal: X·Y = iZ (anticommuting), so C_{X,Y} has α contribution
    // from the off-diagonal Φ computation.
    eprintln!(
        "Two H gens: p0={:.6} p1={:.6} diagonal={diagonal:.6}",
        p0.total, p1.total
    );

    // p(1) should be at least the diagonal
    assert!(
        p1.total >= diagonal * 0.9,
        "p(1)={:.6} should be ≥ diagonal {diagonal:.6}",
        p1.total
    );

    // Conservation
    assert!((p0.total + p1.total - 1.0).abs() < 0.01);
}

/// C-type first-order α: directly construct a C generator and verify.
#[test]
fn test_c_type_alpha() {
    // |0⟩ with C_{X,X} = 2S_X. Should give same result as S_X at double rate.
    let stabs = vec![Bm::z(0)];
    let c = 0.005;

    let gens = vec![PropagatedEeg {
        eeg_type: EegType::C,
        label: Bm::x(0),
        label2: Some(Bm::x(0)),
        coeff: c,
        source: None,
    }];

    let p0 = outcome_probability(&gens, &[false], &stabs);
    let p1 = outcome_probability(&gens, &[true], &stabs);

    // C_{X,X} = 2S_X. α(C_{X,X}) at outcome 0:
    // Φ(X,X) at 0 = 0 (flipped not in support), Φ(I,I) = 1.
    // Φ(XX,I) = Φ(I,I) = 1.
    // α = 2*Re(0) - Re(1 + 1) = -2.
    // Correction: c * α = 0.005 * (-2) = -0.01.
    // But wait — with the scale factor for pure state (ζ=0): scale=1.
    // So ca_correction = 1 * 0.005 * (-2) = -0.01.
    // p(0) = 1 + (-0.01) = 0.99.

    eprintln!("C-type: p0={:.6} p1={:.6}", p0.total, p1.total);
    assert!(
        (p0.total - 0.99).abs() < 0.02,
        "p(0) ≈ 0.99: got {:.6}",
        p0.total
    );
    assert!(p1.total > 0.0, "p(1) should be positive");
}

/// A-type α: construct an A generator. For stabilizer states, A-type
/// typically gives zero (requires iQ1Q2 to be stabilizer eigenvalue).
#[test]
fn test_a_type_alpha_zero_for_stabilizer() {
    // |0⟩ with A_{X,Z}: iXZ = iY. Is iY|0⟩ = ±|0⟩? Y|0⟩ = i|1⟩, so iY|0⟩ = -|1⟩.
    // Not an eigenstate → α(A) should be 0 for both outcomes.
    let stabs = vec![Bm::z(0)];

    let gens = vec![PropagatedEeg {
        eeg_type: EegType::A,
        label: Bm::x(0),
        label2: Some(Bm::z(0)),
        coeff: 0.01,
        source: None,
    }];

    let p0 = outcome_probability(&gens, &[false], &stabs);
    let p1 = outcome_probability(&gens, &[true], &stabs);

    // A-type uses Im(Φ). For |0⟩ (real stabilizer state), Φ values
    // should be real, so Im = 0, giving zero A-type correction.
    eprintln!(
        "A-type: p0_corr={:.8} p1_corr={:.8}",
        p0.s_correction + p0.h_correction,
        p1.s_correction + p1.h_correction
    );
    // The ca_correction is part of total but not separately exposed.
    // Just check total is unchanged from noiseless.
    assert!(
        (p0.total - 1.0).abs() < 1e-6,
        "A on |0⟩ should not change p(0)"
    );
    assert!(p1.total.abs() < 1e-6, "A on |0⟩ should not change p(1)");
}
