// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

//! Packet #1209: independent dense reference and phase-sensitive regressions.

#[path = "support/dense_eeg.rs"]
mod dense;

use dense::{Generator, evolved, lindbladian, pattern, prepared};
use pecos_core::gate_type::GateType;
use pecos_core::{Gate, GateAngles, GateParams, QuarterPhase, QubitId};
use pecos_eeg::Bm;
use pecos_eeg::circuit::{NoiseSource, PropagatedEeg, analyze_with_noise};
use pecos_eeg::dem_mapping::{
    BchOrder, Detector, EegConfig, HFormula, build_dem_configured, build_dem_with_stabilizers,
    sensitivity_matrix,
};
use pecos_eeg::eeg::EegType::{self, A, C, H, S};
use pecos_eeg::noise::{NoiseInjection, NoiseSpec};
use pecos_eeg::stabilizer::StabilizerGroup;
use pecos_eeg::strong_sim::outcome_probability;

fn gate(kind: GateType, qubits: &[usize]) -> Gate {
    Gate {
        gate_type: kind,
        qubits: qubits.iter().map(|&q| QubitId(q)).collect(),
        angles: GateAngles::new(),
        params: GateParams::new(),
        meas_ids: pecos_core::GateMeasIds::new(),
        channel: None,
    }
}

fn bm(label: &str) -> Bm {
    label
        .bytes()
        .enumerate()
        .fold(Bm::default(), |acc, (q, p)| {
            // Each factor acts on a distinct qubit: no multiplication phase.
            acc.multiply(&match p {
                b'X' => Bm::x(q),
                b'Y' => Bm::y(q),
                b'Z' => Bm::z(q),
                b'I' => Bm::default(),
                _ => panic!("invalid Pauli"),
            })
        })
}

fn generator(kind: EegType, p: &'static str, q: Option<&'static str>, coeff: f64) -> Generator {
    Generator { kind, p, q, coeff }
}

fn propagated(generators: &[Generator]) -> Vec<PropagatedEeg> {
    generators
        .iter()
        .map(|g| PropagatedEeg {
            eeg_type: g.kind,
            label: bm(g.p),
            label2: g.q.map(bm),
            coeff: g.coeff,
            source: None,
        })
        .collect()
}

fn detectors(labels: &[&str]) -> Vec<Detector> {
    labels
        .iter()
        .enumerate()
        .map(|(id, label)| Detector {
            id,
            stabilizer: bm(label),
        })
        .collect()
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual:.16e}, expected={expected:.16e}, tolerance={tolerance:.2e}"
    );
}

fn reference(n: usize, gates: &[Gate], gens: &[Generator], dets: &[&str]) -> f64 {
    let rho = prepared(n, gates);
    pattern(
        &rho,
        &evolved(&rho, gens),
        &dets.iter().map(|&d| (d, true)).collect::<Vec<_>>(),
    )
}

fn dem(n: usize, gates: &[Gate], gens: &[Generator], dets: &[&str], formula: HFormula) -> f64 {
    let group = StabilizerGroup::from_circuit(gates, n);
    let entries = build_dem_configured(
        &propagated(gens),
        &detectors(dets),
        &[],
        Some(&group),
        &EegConfig {
            h_formula: formula,
            ..Default::default()
        },
    );
    assert!(
        entries.len() <= 1,
        "oracle comparisons must use one event class"
    );
    entries.first().map_or(0.0, |e| {
        assert_eq!(e.event.detectors.len(), dets.len());
        e.probability
    })
}

fn ca_dem(n: usize, gates: &[Gate], gens: &[Generator], dets: &[&str]) -> f64 {
    let group = StabilizerGroup::from_circuit(gates, n);
    let entries =
        build_dem_with_stabilizers(&propagated(gens), &detectors(dets), &[], Some(&group));
    assert!(entries.len() <= 1);
    entries.first().map_or(0.0, |entry| entry.probability)
}

#[test]
fn reference_r1_destructive_xx_yy() {
    let gens = [
        generator(H, "XX", None, 0.1),
        generator(H, "YY", None, 0.05),
    ];
    close(
        reference(2, &[], &gens, &["ZI"]),
        0.05_f64.sin().powi(2),
        1e-12,
    );
}

#[test]
fn reference_r2_c_derivatives() {
    let rho = prepared(2, &[]);
    for (q, expected) in [("YY", -2.0), ("XX", 2.0)] {
        let derivative = lindbladian(&rho, &[generator(C, "XX", Some(q), 1.0)]);
        close(pattern(&rho, &derivative, &[("ZI", true)]), expected, 1e-12);
    }
}

#[test]
fn reference_r3_a_derivative() {
    let rho = prepared(1, &[]);
    let derivative = lindbladian(&rho, &[generator(A, "X", Some("Y"), 1.0)]);
    close(pattern(&rho, &derivative, &[("Z", true)]), 2.0, 1e-12);
}

#[test]
fn reference_r4_s_channel() {
    for s in [-0.01_f64, -0.2, 0.03] {
        close(
            reference(1, &[], &[generator(S, "Y", None, s)], &["Z"]),
            (1.0 - (2.0 * s).exp()) / 2.0,
            1e-12,
        );
    }
}

#[test]
fn reference_trace_and_hermiticity() {
    let rho = prepared(
        2,
        &[
            gate(GateType::H, &[0]),
            gate(GateType::SZ, &[0]),
            gate(GateType::CX, &[0, 1]),
        ],
    );
    for g in [
        generator(H, "YX", None, -0.17),
        generator(S, "YY", None, -0.13),
        generator(C, "XY", Some("YX"), -0.09),
        generator(A, "XY", Some("YY"), 0.11),
    ] {
        let image = lindbladian(&rho, &[g]);
        let trace = dense::trace(&image);
        close(trace.re, 0.0, 1e-12);
        close(trace.im, 0.0, 1e-12);
        for row in 0..4 {
            for col in 0..4 {
                close(image[row * 4 + col].re, image[col * 4 + row].re, 1e-12);
                close(image[row * 4 + col].im, -image[col * 4 + row].im, 1e-12);
            }
        }
    }
}

#[test]
fn ordered_product_expectations() {
    let zero = StabilizerGroup::from_circuit(&[], 2);
    assert_eq!(
        zero.product_expectation([&bm("XX"), &bm("YY")]),
        Some(QuarterPhase::MinusOne)
    );
    assert_eq!(
        zero.product_expectation([&bm("XI"), &bm("YI")]),
        Some(QuarterPhase::PlusI)
    );
    assert_eq!(
        zero.product_expectation([&bm("YI"), &bm("XI")]),
        Some(QuarterPhase::MinusI)
    );
    assert_eq!(zero.product_expectation([&bm("XI")]), None);
    assert_eq!(
        zero.product_expectation(std::iter::empty()),
        Some(QuarterPhase::PlusOne)
    );
    let one = StabilizerGroup::from_circuit(&[gate(GateType::X, &[1])], 2);
    assert_eq!(
        one.product_expectation([&bm("XX"), &bm("YY")]),
        Some(QuarterPhase::PlusOne)
    );
    assert_eq!(
        one.product_expectation([&bm("IZ")]),
        Some(QuarterPhase::MinusOne)
    );
}

#[test]
fn h_formulas_phase_and_signed_coefficients() {
    for gates in [vec![], vec![gate(GateType::X, &[1])]] {
        for h2 in [0.05, -0.05] {
            let gens = [generator(H, "XX", None, 0.1), generator(H, "YY", None, h2)];
            let exact = reference(2, &gates, &gens, &["ZI"]);
            for formula in [HFormula::Taylor, HFormula::SinSquared] {
                close(
                    dem(2, &gates, &gens, &["ZI"], formula),
                    exact,
                    2.0 * (0.1 + h2.abs()).powi(4),
                );
            }
            close(
                dem(2, &gates, &gens, &["ZI"], HFormula::ExactSubset),
                exact,
                1e-12,
            );
            // Documented independent-rotation property; NOT the exact R1 answer.
            close(
                dem(2, &gates, &gens, &["ZI"], HFormula::ExactCommuting),
                (1.0 - 0.2_f64.cos() * (2.0 * h2).cos()) / 2.0,
                1e-12,
            );
        }
    }
}

#[test]
fn h_formulas_independent_rotations_domain() {
    // YY is not a stabilizer of |00>; no interference products exist.
    let gens = [
        generator(H, "YI", None, -0.17),
        generator(H, "IY", None, 0.08),
    ];
    let exact = reference(2, &[], &gens, &["ZZ"]);
    for formula in [HFormula::ExactCommuting, HFormula::ExactSubset] {
        close(dem(2, &[], &gens, &["ZZ"], formula), exact, 1e-12);
    }
    for formula in [HFormula::Taylor, HFormula::SinSquared] {
        close(
            dem(2, &[], &gens, &["ZZ"], formula),
            exact,
            2.0 * 0.25_f64.powi(4),
        );
    }
}

#[test]
fn exact_subset_four_body_interference() {
    let gens = [
        generator(H, "XXX", None, 0.09),
        generator(H, "YYX", None, -0.03),
        generator(H, "YXY", None, 0.07),
        generator(H, "XYY", None, -0.02),
    ];
    for gates in [vec![], vec![gate(GateType::X, &[1])]] {
        for dets in [vec!["ZII"], vec!["ZII", "IIZ"]] {
            close(
                dem(3, &gates, &gens, &dets, HFormula::ExactSubset),
                reference(3, &gates, &gens, &dets),
                1e-12,
            );
        }
    }
}

#[test]
fn c_finite_both_product_signs_and_two_detectors() {
    for (q, c, sigma) in [("YY", -0.01, -1.0), ("XX", 0.01, 1.0)] {
        let gens = [generator(C, "XX", Some(q), c)];
        for dets in [vec!["ZI"], vec!["ZI", "IZ"]] {
            let exact = reference(2, &[], &gens, &dets);
            close(exact, (1.0 - (-4.0_f64 * c * sigma).exp()) / 2.0, 1e-12);
            close(ca_dem(2, &[], &gens, &dets), exact, 1e-12);
        }
    }
    let gates = [gate(GateType::X, &[1])];
    let gens = [generator(C, "XX", Some("YY"), 0.02)];
    close(
        ca_dem(2, &gates, &gens, &["ZI"]),
        reference(2, &gates, &gens, &["ZI"]),
        1e-12,
    );
}

#[test]
fn signed_c_and_s_share_event() {
    // <XX YY> = -1: positive c REDUCES the S flip rate. Taking |delta p|
    // would incorrectly increase it. Net s_eff = -0.04 + 2*0.01 = -0.02.
    let gens = [
        generator(S, "XX", None, -0.04),
        generator(C, "XX", Some("YY"), 0.01),
    ];
    let exact = reference(2, &[], &gens, &["ZI"]);
    close(exact, (1.0 - (-0.04_f64).exp()) / 2.0, 1e-12);
    close(ca_dem(2, &[], &gens, &["ZI"]), exact, 1e-12);
}

#[test]
fn a_linear_reference_and_fold_error() {
    for (p, q, a) in [("X", "Y", 0.0001), ("Y", "X", -0.0001)] {
        let gens = [generator(A, p, Some(q), a)];
        let exact = reference(1, &[], &gens, &["Z"]);
        close(exact, 2.0 * a.abs(), 1e-12);
        close(ca_dem(1, &[], &gens, &["Z"]), exact, 8.0 * a * a);
    }
}

#[test]
fn ca_commutation_zero_and_same_label_identity() {
    for g in [
        generator(C, "X", Some("Y"), 0.02),
        generator(A, "Y", Some("Y"), -0.02),
    ] {
        close(reference(1, &[], &[g], &["Z"]), 0.0, 1e-12);
        close(ca_dem(1, &[], &[g], &["Z"]), 0.0, 0.0);
    }
    for c in [0.001, 0.1] {
        let c_gens = [generator(C, "YY", Some("YY"), c)];
        let s_gens = [generator(S, "YY", None, -2.0 * c)];
        let expected = reference(2, &[], &c_gens, &["ZI", "IZ"]);
        close(ca_dem(2, &[], &c_gens, &["ZI", "IZ"]), expected, 1e-12);
        close(
            ca_dem(2, &[], &c_gens, &["ZI", "IZ"]),
            ca_dem(2, &[], &s_gens, &["ZI", "IZ"]),
            0.0,
        );
    }
}

fn check_strong_first_order(gens: &[Generator]) {
    let rho = prepared(2, &[]);
    let exact = evolved(&rho, gens);
    let first = lindbladian(&rho, gens);
    for bits in [[false, false], [true, false], [true, true]] {
        let dets = [("ZI", bits[0]), ("IZ", bits[1])];
        let actual = outcome_probability(&propagated(gens), &bits, &[bm("ZI"), bm("IZ")]).total;
        close(
            actual,
            pattern(&rho, &rho, &dets) + pattern(&rho, &first, &dets),
            1e-12,
        );
        close(
            actual,
            pattern(&rho, &exact, &dets),
            8.0 * gens[0].coeff.powi(2),
        );
    }
}

#[test]
fn strong_c_with_y() {
    check_strong_first_order(&[generator(C, "XX", Some("YY"), -0.0001)]);
}

#[test]
fn strong_a_with_y() {
    check_strong_first_order(&[generator(A, "XI", Some("YI"), 0.0001)]);
}

#[test]
fn strong_phi_reduced_rows_retain_original_products() {
    let gates = [
        gate(GateType::H, &[0]),
        gate(GateType::H, &[1]),
        gate(GateType::CX, &[0, 2]),
        gate(GateType::CX, &[1, 2]),
    ];
    let rho = prepared(3, &gates);
    // Equivalent +1 generators with overlapping X pivots.
    let stabs = [bm("XXI"), bm("XIX"), bm("ZZZ")];
    let gens = [generator(C, "YII", Some("YXX"), 0.001)];
    let first = lindbladian(&rho, &gens);
    for index in 0..8 {
        let bits = [index & 1 != 0, index & 2 != 0, index & 4 != 0];
        close(
            outcome_probability(&propagated(&gens), &bits, &stabs).total,
            rho[index * 8 + index].re + first[index * 8 + index].re,
            1e-12,
        );
    }
}

#[test]
fn sensitivity_pair_beta_and_binding_replacement() {
    let mut gens = propagated(&[
        generator(H, "XX", None, 0.1),
        generator(H, "YY", None, 0.05),
    ]);
    for (i, g) in gens.iter_mut().enumerate() {
        g.source = Some(NoiseSource {
            gate_index: i,
            qubit: 0,
        });
    }
    let group = StabilizerGroup::from_circuit(&[], 2);
    let dets = detectors(&["ZI", "IZ"]);
    let matrix = sensitivity_matrix(&gens, &dets, &[], Some(&group));
    for (_, entries) in matrix {
        assert_eq!(entries.len(), 4);
        for (i, j, beta) in entries {
            close(
                beta,
                if i.gate_index == j.gate_index {
                    1.0
                } else {
                    -1.0
                },
                0.0,
            );
        }
    }
    // eeg_per_detector now calls exactly this crate API on singleton detectors.
    for det in &dets {
        let entries = build_dem_configured(
            &gens,
            std::slice::from_ref(det),
            &[],
            Some(&group),
            &EegConfig::new(),
        );
        close(entries[0].probability, (0.1_f64 - 0.05).powi(2), 1e-12);
    }
}

struct Injections(Vec<NoiseInjection>);
impl NoiseSpec for Injections {
    fn noise_after_gate(
        &self,
        _index: usize,
        kind: GateType,
        _qubits: &[usize],
    ) -> Vec<NoiseInjection> {
        if kind == GateType::I {
            self.0.clone()
        } else {
            vec![]
        }
    }
}

#[test]
fn reference_agrees_with_heisenberg_hs() {
    let h = 0.12;
    let s = -0.07_f64;
    let noise = Injections(vec![
        NoiseInjection {
            eeg_type: H,
            label: bm("Y"),
            label2: None,
            rate: h,
        },
        NoiseInjection {
            eeg_type: S,
            label: bm("Y"),
            label2: None,
            rate: -(1.0 - (2.0 * s).exp()) / 2.0,
        },
    ]);
    let gates = [
        gate(GateType::PZ, &[0]),
        gate(GateType::I, &[0]),
        gate(GateType::MZ, &[0]),
    ];
    let exact =
        pecos_eeg::heisenberg::heisenberg_exact_from_circuit(&gates, &[0], &noise, 1).unwrap();
    let gens = [generator(H, "Y", None, h), generator(S, "Y", None, s)];
    close(exact, reference(1, &[], &gens, &["Z"]), 1e-12);
}

#[test]
fn ca_requires_group_and_label_at_every_consumer() {
    for kind in [C, A] {
        for coeff in [0.0, 0.01] {
            let mut gens = propagated(&[generator(kind, "X", Some("Y"), coeff)]);
            assert!(
                std::panic::catch_unwind(|| build_dem_with_stabilizers(
                    &gens,
                    &detectors(&["Z"]),
                    &[],
                    None
                ))
                .is_err()
            );
            gens[0].label2 = None;
            let group = StabilizerGroup::from_circuit(&[], 1);
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    build_dem_with_stabilizers(&gens, &detectors(&["Z"]), &[], Some(&group))
                }))
                .is_err()
            );
            assert!(
                std::panic::catch_unwind(|| outcome_probability(&gens, &[false], &[bm("Z")]))
                    .is_err()
            );
            let noise = Injections(vec![NoiseInjection {
                eeg_type: kind,
                label: bm("X"),
                label2: None,
                rate: coeff,
            }]);
            assert!(
                std::panic::catch_unwind(|| analyze_with_noise(
                    &[gate(GateType::I, &[0])],
                    &noise,
                    &[false]
                ))
                .is_err()
            );
        }
    }
}

#[test]
#[should_panic(expected = "event DemEvent")]
fn negative_s_probability_names_event() {
    ca_dem(1, &[], &[generator(S, "X", None, 0.001)], &["Z"]);
}

#[test]
#[should_panic(expected = "event DemEvent")]
fn negative_c_probability_is_not_masked() {
    ca_dem(2, &[], &[generator(C, "XX", Some("YY"), 0.001)], &["ZI"]);
}

#[test]
fn roundoff_is_clamped_and_internal_bch_c_is_valid() {
    close(
        ca_dem(1, &[], &[generator(S, "X", None, 5e-13)], &["Z"]),
        0.0,
        0.0,
    );
    let gens = propagated(&[generator(H, "X", None, 0.1), generator(S, "Y", None, -0.01)]);
    let entries = build_dem_configured(
        &gens,
        &detectors(&["Z"]),
        &[],
        None,
        &EegConfig {
            bch_order: BchOrder::Second,
            h_formula: HFormula::Taylor,
        },
    );
    assert_eq!(entries.len(), 1);
}

#[test]
#[should_panic(expected = "ExactSubset limit is 25 H labels per event")]
fn exact_subset_limit_is_explicit() {
    // 26 distinct, mutually commuting labels all flip the same detector.
    let gens: Vec<_> = (1..=26)
        .map(|q| PropagatedEeg {
            eeg_type: H,
            label: Bm::x(0).multiply(&Bm::x(q)),
            label2: None,
            coeff: 0.001,
            source: None,
        })
        .collect();
    let group = StabilizerGroup::from_circuit(&[], 27);
    let _ = build_dem_configured(
        &gens,
        &detectors(&["Z"]),
        &[],
        Some(&group),
        &EegConfig::new().exact_subset(),
    );
}

#[test]
fn strong_h_pair_retains_product_phase() {
    let gens = [
        generator(H, "XX", None, 0.1),
        generator(H, "YY", None, 0.05),
    ];
    let rho = prepared(2, &[]);
    let second = lindbladian(&lindbladian(&rho, &gens), &gens);
    for (index, bits) in [(0, [false, false]), (3, [true, true])] {
        let expected = rho[index * 4 + index].re + 0.5 * second[index * 4 + index].re;
        close(
            outcome_probability(&propagated(&gens), &bits, &[bm("ZI"), bm("IZ")]).total,
            expected,
            1e-12,
        );
    }
}

#[test]
fn strong_ca_with_y_stabilizer_phase() {
    let rho = prepared(1, &[gate(GateType::H, &[0]), gate(GateType::SZ, &[0])]);
    for g in [
        generator(C, "Y", Some("Z"), 0.0001),
        generator(A, "X", Some("Y"), 0.0001),
    ] {
        let first = lindbladian(&rho, &[g]);
        let exact = evolved(&rho, &[g]);
        for (index, bit) in [(0, false), (1, true)] {
            let actual = outcome_probability(&propagated(&[g]), &[bit], &[bm("Y")]).total;
            close(
                actual,
                rho[index * 2 + index].re + first[index * 2 + index].re,
                1e-12,
            );
            close(actual, exact[index * 2 + index].re, 8.0 * g.coeff.powi(2));
        }
    }
}

#[test]
fn exact_subset_validation_matches_build_limit() {
    use pecos_eeg::dem_mapping::{Observable, exact_subset_violation};

    let mut gens: Vec<_> = (1..=26)
        .map(|q| PropagatedEeg {
            eeg_type: H,
            label: Bm::x(0).multiply(&Bm::x(q)),
            label2: None,
            coeff: 0.001,
            source: None,
        })
        .collect();
    let dets = detectors(&["Z"]);
    let config = EegConfig::new().exact_subset();
    assert_eq!(
        exact_subset_violation(&gens[..25], &dets, &[], &config),
        None
    );
    let message = exact_subset_violation(&gens, &dets, &[], &config).unwrap();
    assert!(message.contains("got 26 for event"));
    let panic = std::panic::catch_unwind(|| build_dem_configured(&gens, &dets, &[], None, &config))
        .unwrap_err();
    assert_eq!(panic.downcast_ref::<String>(), Some(&message));
    for formula in [
        HFormula::Taylor,
        HFormula::SinSquared,
        HFormula::ExactCommuting,
    ] {
        assert_eq!(
            exact_subset_violation(
                &gens,
                &dets,
                &[],
                &EegConfig {
                    h_formula: formula,
                    ..config
                }
            ),
            None
        );
    }

    // Full events split into sizes 1 and 25; singleton-detector pooling gives 26.
    assert_eq!(
        exact_subset_violation(&gens, &detectors(&["ZI", "IZ"]), &[], &config),
        None
    );
    assert_eq!(
        exact_subset_violation(
            &gens,
            &dets,
            &[Observable {
                id: 0,
                pauli: Bm::z(1)
            }],
            &config
        ),
        None
    );
    assert!(
        exact_subset_violation(
            &gens,
            &[],
            &[Observable {
                id: 0,
                pauli: Bm::z(0)
            }],
            &config
        )
        .is_some()
    );
    assert_eq!(exact_subset_violation(&gens, &[], &[], &config), None);

    // Count combined non-negligible labels, not input generators.
    let mut cancellation = gens[25].clone();
    cancellation.coeff = -0.001;
    gens.push(cancellation);
    assert_eq!(exact_subset_violation(&gens, &dets, &[], &config), None);
    gens[25].coeff = 1e-21;
    gens.pop();
    assert_eq!(exact_subset_violation(&gens, &dets, &[], &config), None);
}

#[test]
fn exact_subset_validation_counts_bch_generated_labels() {
    use pecos_eeg::dem_mapping::exact_subset_violation;

    let mut gens: Vec<_> = (1..=25)
        .map(|q| PropagatedEeg {
            eeg_type: H,
            label: Bm::x(0).multiply(&Bm::x(q)),
            label2: None,
            coeff: 0.001,
            source: None,
        })
        .collect();
    // Z0 is invisible to the detector but its BCH commutators produce 25
    // distinct Y0 Xq labels in the event, alongside the original 25 X0 Xq.
    gens.push(PropagatedEeg {
        eeg_type: H,
        label: Bm::z(0),
        label2: None,
        coeff: 0.001,
        source: None,
    });
    let dets = detectors(&["Z"]);
    let first = EegConfig::new().exact_subset();
    assert_eq!(exact_subset_violation(&gens, &dets, &[], &first), None);
    let second = first.bch_second();
    let message = exact_subset_violation(&gens, &dets, &[], &second).unwrap();
    assert!(message.contains("got 50 for event"));
    let panic = std::panic::catch_unwind(|| build_dem_configured(&gens, &dets, &[], None, &second))
        .unwrap_err();
    assert_eq!(panic.downcast_ref::<String>(), Some(&message));
}
