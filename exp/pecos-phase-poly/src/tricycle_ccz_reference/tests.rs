// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

use super::*;
use crate::tricycle_ccz_noisy::{CczModel, MetaChecks, NoiseModel, NoisyCcz, PAPER_C27_SCHEDULE};

fn c27() -> BivariateTricycleCode {
    BivariateTricycleCode::new(
        3,
        3,
        &[(2, 1), (2, 2)],
        &[(0, 0), (1, 2)],
        &[(1, 0), (2, 1)],
    )
    .unwrap()
}

#[test]
fn paper_single_faults_are_detected() {
    let code = c27();
    for schedule in [
        paper_schedule(),
        ExtractionSchedule::edge_coloring(&code).unwrap(),
    ] {
        let catalog = paper_catalog(&code, &schedule, 0.001).unwrap();
        for fault in catalog.fault_configurations(1) {
            assert!(
                !fault.affected_detectors.is_empty() || fault.affected_observables.is_empty(),
                "undetected logical single fault: {fault:?}"
            );
        }
    }
}

#[test]
fn paper_reference_matches_published_c27() {
    let result = estimate(
        &paper_catalog(&c27(), &paper_schedule(), 0.001).unwrap(),
        3,
        8,
        100_000,
        251_007_269,
    )
    .unwrap();
    let e = &result.estimate;
    let ratio = e.failure_given_survival.unwrap();
    let se = e.ratio_standard_error.unwrap();
    assert!(se <= 0.1 * 1.944_315_872_789_761e-6, "LER SE={se}");
    assert!(
        (e.survival_probability - 0.704_103_7).abs()
            <= 4.0 * 1.44e-5_f64.hypot(e.survival_standard_error),
        "success={}, SE={}, oracle=0.7041037, oracle SE=1.44e-5",
        e.survival_probability,
        e.survival_standard_error
    );
    assert!(
        (ratio - 1.944_315_872_789_761e-6).abs() <= 4.0 * 5.2549e-8_f64.hypot(se),
        "LER={ratio}, SE={se}, oracle=1.944315872789761e-6, oracle SE=5.2549e-8"
    );
}

#[test]
fn noisy_ccz_paper_noise_matches_reference_acceptance() {
    let code = c27();
    let protocol = NoisyCcz::new(
        &code,
        &ExtractionSchedule::edge_coloring(&code).unwrap(),
        0.001,
        CczModel::PaperSurrogate,
        NoiseModel::Paper,
        MetaChecks::Complete,
    )
    .unwrap();
    let shots = 4000_u32;
    let counts = protocol.run_counts(shots as usize, 251_007_269).unwrap();
    assert_eq!(counts.repair_failures, 0, "counts={counts:?}");
    let reference = estimate(
        &paper_catalog(
            &code,
            &ExtractionSchedule::edge_coloring(&code).unwrap(),
            0.001,
        )
        .unwrap(),
        1,
        8,
        20_000,
        251_007_269,
    )
    .unwrap()
    .estimate;
    let accepted = f64::from(u32::try_from(counts.accepted).unwrap()) / f64::from(shots);
    let expected = reference.survival_probability;
    let se = (expected * (1.0 - expected) / f64::from(shots)).sqrt();
    assert!(
        (accepted - expected).abs() <= 4.0 * se,
        "accepted={accepted}, reference={expected}, SE={se}, counts={counts:?}"
    );
}

#[test]
fn paper_catalog_channels_and_surrogate_records() {
    let code = c27();
    let p = 0.001;
    let catalog = paper_catalog(&code, &paper_schedule(), p).unwrap();
    let mut explicit_count = 0;
    for location in &catalog.locations {
        let expected = match location.channel {
            FaultChannel::P1 => panic!("paper circuit has no noisy single-qubit gates"),
            FaultChannel::P2 => {
                assert_eq!(location.faults.len(), 15);
                p
            }
            FaultChannel::PPrep if location.gate_type == GateType::PX => 0.0,
            FaultChannel::PPrep | FaultChannel::PMeas => 2.0 * p / 3.0,
            FaultChannel::Explicit => {
                explicit_count += 1;
                assert_eq!(location.faults.len(), 1);
                assert_eq!(location.qubits.len(), 5);
                let alternative = &location.faults[0];
                assert_eq!(alternative.affected_measurements.len(), 5);
                assert!((alternative.conditional_probability - 1.0).abs() < f64::EPSILON);
                let mut detectors = BTreeSet::new();
                let mut observables = BTreeSet::new();
                for &q in &location.qubits {
                    let measurement = catalog
                        .locations
                        .iter()
                        .find(|l| l.gate_type == GateType::MX && l.qubits == [q])
                        .unwrap();
                    let record = &measurement.faults[0];
                    assert!(
                        alternative
                            .affected_measurements
                            .contains(&record.affected_measurements[0])
                    );
                    xor_into(&mut detectors, &record.affected_detectors);
                    xor_into(&mut observables, &record.affected_observables);
                }
                assert_eq!(
                    alternative.affected_detectors,
                    detectors.into_iter().collect::<Vec<_>>()
                );
                assert_eq!(
                    alternative.affected_observables,
                    observables.into_iter().collect::<Vec<_>>()
                );
                p
            }
        };
        assert!((location.channel_probability - expected).abs() < f64::EPSILON);
        assert!((location.no_fault_probability - (1.0 - expected)).abs() < f64::EPSILON);
        for alternative in &location.faults {
            assert!(
                (alternative.absolute_probability - expected * alternative.conditional_probability)
                    .abs()
                    < f64::EPSILON
            );
        }
    }
    assert_eq!(explicit_count, 3 * code.num_qubits());
}

#[test]
fn exact_and_sampled_estimator_and_omitted_mass() {
    let mut circuit = TickCircuit::new();
    circuit.tick().px(&[0]);
    let record = circuit.tick().mx(&[0]);
    circuit.observable(&record).unwrap();
    let catalog =
        FaultCatalog::from_circuit(&circuit)
            .unwrap()
            .parameterized(&StochasticNoiseParams {
                p1: 0.0,
                p2: 0.0,
                p_meas: 0.2,
                p_prep: 0.1,
            });
    let exact = estimate(&catalog, 2, 2, 0, 72).unwrap();
    assert!((exact.estimate.survival_probability - 1.0).abs() < 1e-14);
    assert!((exact.estimate.failure_probability - 0.26).abs() < 1e-14);
    assert!(exact.estimate.failure_standard_error.abs() < f64::EPSILON);
    assert!(exact.estimate.survival_standard_error.abs() < f64::EPSILON);
    let sampled = estimate(&catalog, 0, 2, 10, 72).unwrap();
    // Each conditional stratum has a deterministic outcome in this circuit.
    assert!((sampled.estimate.failure_probability - 0.26).abs() < 1e-14);
    assert!((sampled.estimate.survival_probability - 1.0).abs() < 1e-14);
    let omitted = estimate(&catalog, 0, 2, 0, 72).unwrap();
    assert!((omitted.estimate.unsampled_mass - 0.28).abs() < 1e-14);
    assert!((omitted.estimate.survival_probability - 0.72).abs() < 1e-14);
    assert!(estimate(&catalog, 2, 1, 0, 72).is_err());
}

#[test]
fn configuration_counts_match_enumeration() {
    // Fifteen CX alternatives, two measurement flips, and inactive preparations.
    let mut circuit = TickCircuit::new();
    circuit.tick().px(&[0]);
    circuit.tick_at(0).pz(&[1]);
    circuit.tick().cx(&[(0, 1)]);
    circuit.tick().mz(&[1]);
    circuit.tick().mx(&[0]);
    let catalog =
        FaultCatalog::from_circuit(&circuit)
            .unwrap()
            .parameterized(&StochasticNoiseParams {
                p1: 0.0,
                p2: 0.01,
                p_meas: 0.02,
                p_prep: 0.0,
            });
    for k in 0..=3 {
        let enumerated = u64::try_from(catalog.fault_configurations(k).count()).unwrap();
        assert_eq!(
            configuration_count(&catalog, k).unwrap(),
            enumerated,
            "k={k}"
        );
    }
}

#[test]
fn paper_circuit_has_deterministic_checks_and_logicals() {
    let code = c27();
    let program =
        crate::runner::Program::compile(&paper_circuit(&code, &paper_schedule()).unwrap()).unwrap();
    assert_eq!(program.detectors().len(), 3 * code.hx().num_checks());
    assert_eq!(program.observables().len(), 3 * code.num_logical_qubits());
    for shot in program.run_shots(16, 72).unwrap() {
        assert!(shot.detectors.iter().all(|&b| !b));
        assert!(shot.observables.iter().all(|&b| !b));
    }
}

fn paper_schedule() -> ExtractionSchedule {
    ExtractionSchedule::from_layers(&c27(), PAPER_C27_SCHEDULE.map(Vec::from).into()).unwrap()
}

#[test]
fn weight_two_undetected_logicals_depend_on_schedule() {
    let code = c27();
    let failure = |schedule: &ExtractionSchedule| {
        paper_catalog(&code, schedule, 0.001)
            .unwrap()
            .fault_configurations(2)
            .filter(|c| c.affected_detectors.is_empty() && !c.affected_observables.is_empty())
            .map(|c| c.configuration_probability)
            .sum::<f64>()
    };
    let hook_safe = failure(&ExtractionSchedule::hook_safe(&code).unwrap());
    assert_eq!(hook_safe.abs().to_bits(), 0.0_f64.to_bits());
    let paper = failure(&paper_schedule());
    let coloring = failure(&ExtractionSchedule::edge_coloring(&code).unwrap());
    assert_eq!(
        paper.abs().to_bits(),
        0.0_f64.to_bits(),
        "k=2 failure: paper={paper}, coloring={coloring}"
    );
    assert!(
        coloring > 0.0,
        "k=2 failure: paper={paper}, coloring={coloring}"
    );
}

#[test]
fn extraction_schedule_validation() {
    let code = c27();
    let valid = paper_schedule();
    assert!(ExtractionSchedule::edge_coloring(&code).is_ok());
    let mut missing = valid.layers().to_vec();
    missing[0].pop();
    assert!(
        ExtractionSchedule::from_layers(&code, missing)
            .unwrap_err()
            .contains("missing")
    );
    let rows = code.hz().rows();
    let non_edge = (0..code.hz().num_checks())
        .flat_map(|c| (0..code.num_qubits()).map(move |q| (q, c)))
        .find(|&(q, c)| rows[c][q] == 0)
        .unwrap();
    let mut invalid = valid.layers().to_vec();
    invalid[0][0] = non_edge;
    assert!(
        ExtractionSchedule::from_layers(&code, invalid)
            .unwrap_err()
            .contains("not an H_Z edge")
    );
    for endpoint in [0, 1] {
        let mut repeated = valid.layers().to_vec();
        let first = repeated[0][0];
        let other = repeated[1..]
            .iter()
            .flatten()
            .copied()
            .find(|&pair| {
                if endpoint == 0 {
                    pair.0 == first.0
                } else {
                    pair.1 == first.1
                }
            })
            .unwrap();
        repeated[0].push(other);
        assert!(
            ExtractionSchedule::from_layers(&code, repeated)
                .unwrap_err()
                .contains("repeated")
        );
    }
    let mut duplicate = valid.layers().to_vec();
    duplicate.push(vec![duplicate[0][0]]);
    assert!(
        ExtractionSchedule::from_layers(&code, duplicate)
            .unwrap_err()
            .contains("more than once")
    );
}

fn hook_safe_full_catalog_distance(code: &BivariateTricycleCode, expected: usize) {
    let schedule = ExtractionSchedule::hook_safe(code).unwrap();
    let block_distance =
        pecos_qec::connected_cluster_code_distance(code.hx(), code.logical_x(), code.num_qubits())
            .unwrap()
            .distance;
    assert_eq!(block_distance, expected);
    let catalog = paper_catalog(code, &schedule, 0.001).unwrap();
    assert_eq!(
        crate::tricycle_ccz_noisy::catalog_fault_distance(
            &catalog,
            3 * code.hx().num_checks(),
            3 * code.logical_x().num_checks(),
            block_distance
        )
        .unwrap(),
        expected
    );
}

#[test]
fn hook_safe_full_catalog_distance_c27() {
    hook_safe_full_catalog_distance(&c27(), 3);
}

#[test]
fn hook_safe_full_catalog_distance_c81() {
    let code = BivariateTricycleCode::new(
        3,
        9,
        &[(1, 3), (2, 1)],
        &[(0, 0), (1, 8)],
        &[(2, 4), (2, 6)],
    )
    .unwrap();
    hook_safe_full_catalog_distance(&code, 5);
}

#[test]
fn low_weight_c27_matches_fixed_oracle() {
    let code = c27();
    let catalog =
        paper_catalog(&code, &ExtractionSchedule::hook_safe(&code).unwrap(), 0.001).unwrap();
    let result = low_weight_estimate(&catalog, 3).unwrap();
    assert_eq!(result.distance, 3);
    assert_eq!(result.terms.len(), 2);
    for (term, (weight, count, sum)) in result
        .terms
        .iter()
        .zip([(3, 315, 1.957_925e-6), (4, 3780, 3.190_297e-8)])
    {
        assert_eq!(term.weight, weight);
        assert_eq!(term.number_of_sets, count);
        assert!((term.coefficient_sum / sum - 1.0).abs() < 1e-6, "{term:?}");
    }
    assert!((result.total / (1.957_925e-6 + 3.190_297e-8) - 1.0).abs() < 1e-6);
    assert!(low_weight_estimate(&catalog, 2).is_err());
}

#[test]
fn low_weight_coefficient_matches_exact_fault_stratum() {
    let code = c27();
    let catalog =
        paper_catalog(&code, &ExtractionSchedule::hook_safe(&code).unwrap(), 0.001).unwrap();
    let columns = catalog_signature_columns(&catalog, 3 * code.hx().num_checks(), 9).unwrap();
    let sets = undetectable_logical_sets(&columns.h, &columns.l, 3).unwrap();
    let owners =
        signature_odds(&catalog, &columns, |location| location.no_fault_probability).unwrap();
    let coefficient = logical_set_term(3, &sets, &owners).coefficient_sum;
    let no_fault: f64 = catalog
        .locations
        .iter()
        .map(|location| location.no_fault_probability)
        .product();
    let exact = catalog
        .exact_fault_stratum(3)
        .unwrap()
        .failure_probability();
    assert!(
        (coefficient * no_fault / exact - 1.0).abs() < 1e-9,
        "coefficient={coefficient}, no_fault={no_fault}, exact={exact}"
    );
}

#[test]
fn low_weight_assignments_require_distinct_locations() {
    let owners = vec![vec![(0, 0.2), (1, 0.3)], vec![(0, 0.5), (2, 0.7)]];
    let actual = assignment_coefficient(&[0, 1], &owners, &mut Vec::new());
    assert!((actual - (0.2 * 0.7 + 0.3 * 0.5 + 0.3 * 0.7)).abs() < 1e-15);
}

fn signature_fault(
    detectors: &[usize],
    observables: &[usize],
    probability: f64,
) -> FaultAlternative {
    FaultAlternative {
        kind: FaultKind::Pauli,
        pauli: None,
        affected_measurements: Vec::new(),
        affected_detectors: detectors.to_vec(),
        affected_observables: observables.to_vec(),
        affected_tracked_paulis: Vec::new(),
        conditional_probability: 0.0,
        absolute_probability: probability,
    }
}

fn signature_location(mut faults: Vec<FaultAlternative>) -> FaultLocation {
    let probability: f64 = faults.iter().map(|fault| fault.absolute_probability).sum();
    for fault in &mut faults {
        fault.conditional_probability = fault.absolute_probability / probability;
    }
    FaultLocation {
        tick: 0,
        gate_index: 0,
        gate_type: GateType::Z,
        qubits: Vec::new(),
        channel: FaultChannel::Explicit,
        channel_probability: probability,
        no_fault_probability: 1.0 - probability,
        num_alternatives: faults.len(),
        faults,
    }
}

/// Independent exact DP over detector parities and one observable. Each location
/// contributes either no fault or exactly one alternative, including empty signatures.
fn exact_signature_ler(catalog: &FaultCatalog, num_detectors: usize) -> f64 {
    let observable_mask = 1 << num_detectors;
    let mut probabilities = vec![0.0; 2 * observable_mask];
    probabilities[0] = 1.0;
    for location in &catalog.locations {
        let mut next = vec![0.0; probabilities.len()];
        for (state, probability) in probabilities.iter().enumerate() {
            next[state] += probability * location.no_fault_probability;
            for fault in &location.faults {
                let mut effect = 0;
                for &detector in &fault.affected_detectors {
                    effect ^= 1 << detector;
                }
                for &observable in &fault.affected_observables {
                    assert_eq!(observable, 0);
                    effect ^= observable_mask;
                }
                next[state ^ effect] += probability * fault.absolute_probability;
            }
        }
        probabilities = next;
    }
    probabilities[observable_mask] / (probabilities[0] + probabilities[observable_mask])
}

#[test]
fn low_weight_rejects_unrealizable_signature_distance() {
    let p = 0.001;
    let catalog = FaultCatalog {
        locations: vec![
            signature_location(vec![
                signature_fault(&[0], &[0], p / 2.0), // A
                signature_fault(&[0], &[], p / 2.0),  // B
            ]),
            signature_location(vec![
                signature_fault(&[0, 1], &[], p / 2.0), // C
                signature_fault(&[1], &[], p / 2.0),    // D
            ]),
            signature_location(vec![signature_fault(&[1, 2], &[], p)]), // E
            signature_location(vec![signature_fault(&[2], &[], p)]),    // F
        ],
    };
    // A+B is a weight-2 signature logical but needs two alternatives at one
    // location. The first realizable logical is A+C+E+F, of weight four.
    let exact = exact_signature_ler(&catalog, 3);
    assert!((exact / 2.51e-13 - 1.0).abs() < 0.001, "exact={exact}");
    let error = low_weight_estimate(&catalog, 2).unwrap_err();
    assert_eq!(
        error,
        "weight-2 coefficient sum is not positive; cannot certify that signature distance 2 is the circuit fault distance"
    );
}

fn repetition_bit_fault(bits: &[usize], probability: f64) -> FaultAlternative {
    let mut flipped = [false; 5];
    for &bit in bits {
        flipped[bit] ^= true;
    }
    let detectors: Vec<_> = (0..4).filter(|&i| flipped[i] ^ flipped[i + 1]).collect();
    let observables: Vec<_> = flipped[0].then_some(0).into_iter().collect();
    signature_fault(&detectors, &observables, probability)
}

#[test]
fn low_weight_matches_independent_postselected_oracle() {
    let p = 0.01;
    let mut locations = Vec::new();
    for bit in 0..5 {
        locations.push(signature_location(vec![
            repetition_bit_fault(&[bit], p / 2.0),
            signature_fault(&[], &[], p / 2.0),
        ]));
    }
    for bit in 0..4 {
        locations.push(signature_location(vec![
            repetition_bit_fault(&[bit], p / 3.0),
            repetition_bit_fault(&[bit + 1], p / 3.0),
            repetition_bit_fault(&[bit, bit + 1], p / 3.0),
        ]));
    }
    for detector in 0..4 {
        locations.push(signature_location(vec![signature_fault(
            &[detector],
            &[],
            p,
        )]));
    }
    let catalog = FaultCatalog { locations };
    let exact = exact_signature_ler(&catalog, 4);
    let estimate = low_weight_estimate(&catalog, 2).unwrap();
    assert_eq!(estimate.distance, 2);
    // C=3 bounds the relative O(p^2) remainder for this toy at p=1e-2.
    let tolerance = 3.0 * p * p * exact;
    assert!(
        (estimate.total - exact).abs() < tolerance,
        "exact={exact}, estimate={}, tolerance={tolerance}",
        estimate.total
    );
    let next_term = estimate.terms.iter().find(|term| term.weight == 3).unwrap();
    assert!(
        next_term.coefficient_sum > tolerance,
        "weight-3 contribution={}, tolerance={tolerance}",
        next_term.coefficient_sum
    );
}
