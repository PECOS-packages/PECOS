use super::*;

fn parse(text: &str) -> StructuredDem {
    StructuredDem::from_dem_str(text).unwrap()
}

fn project(
    dem: &StructuredDem,
    policy: IrreduciblePolicy,
) -> Result<GraphlikeProjection, GraphlikeProjectionError> {
    project_graphlike(dem, policy)
}

fn parts(text: &str) -> Vec<StructuredDemComponent> {
    parse(&format!("error(0.1) {text}"))
        .errors
        .remove(0)
        .components
}

const TWO_CHOICES: &str = "error(0.01) D0 D1
error(0.01) D2 D3
error(0.4) D0 D2
error(0.4) D1 D3
error(0.123456789) D3 D1 D0 D2";

// contract: cross preference precedes weight; all emitted pairs are counted.
#[test]
fn cross_preference_beats_weight() {
    let mut dem = parse(TWO_CHOICES);
    dem.detector_fault_sectors = vec![
        Some("XX".into()),
        Some("XX".into()),
        Some("YY".into()),
        Some("YY".into()),
    ];
    let result = project_graphlike(&dem, IrreduciblePolicy::Refuse).unwrap();
    assert_eq!(result.model.errors[4].components, parts("D0 D1 ^ D2 D3"));
    assert_eq!(result.report.cross_sector_components, 2);
}

// contract: only unequal known labels are cross; unset and boundaries never count.
#[test]
fn unset_and_boundaries_are_not_cross() {
    for left in [None, Some("XX"), Some("YY"), Some("ZZ")] {
        for right in [None, Some("XX"), Some("YY"), Some("ZZ")] {
            let sectors = [left.map(str::to_owned), right.map(str::to_owned)];
            let expected = left.is_some() && right.is_some() && left != right;
            assert_eq!(is_cross(&[0, 1], &sectors), expected);
            assert_eq!(is_cross(&[1, 0], &sectors), expected);
            assert!(!is_cross(&[0], &sectors));
            assert!(!is_cross(&[1], &sectors));
            assert!(!is_cross(&[], &sectors));
        }
    }
    // The split through the unset detector has worse weight; cross preference wins.
    let mut dem = parse(
        "error(0.4) D0 D1\nerror(0.4) D2\nerror(0.01) D0 D2\nerror(0.01) D1\nerror(0.1) D0 D1 D2",
    );
    dem.detector_fault_sectors = vec![Some("XX".into()), Some("YY".into()), None];
    let result = project_graphlike(&dem, IrreduciblePolicy::Refuse).unwrap();
    assert_eq!(result.model.errors[4].components, parts("D0 D2 ^ D1"));
}

// contract: three opaque sectors prefer equal labels regardless of their spelling.
#[test]
fn opaque_sector_equality_controls_cross_preference() {
    for labels in [["XX", "YY", "ZZ"], ["ZZ", "XX", "YY"], ["x", "X", " "]] {
        let mut dem = parse(
            "error(0.01) D0 D1\nerror(0.01) D2 D3\nerror(0.01) D4 D5\n\
             error(0.4) D0 D2\nerror(0.4) D1 D4\nerror(0.4) D3 D5\n\
             error(0.2) D0 D1 D2 D3 D4 D5",
        );
        dem.detector_fault_sectors = labels
            .into_iter()
            .flat_map(|label| [Some(label.to_owned()), Some(label.to_owned())])
            .collect();
        let result = project(&dem, IrreduciblePolicy::Refuse).unwrap();
        assert_eq!(
            result.model.errors[6].components,
            parts("D0 D1 ^ D2 D3 ^ D4 D5")
        );
        assert_eq!(result.report.cross_sector_components, 3);
        assert_eq!(result.report.distinct_sectors, 3);
        assert_eq!(result.report.unlabelled_detectors, 0);
    }
}

// contract: even an unused detector with an empty label invalidates the public model.
#[test]
fn empty_sector_is_a_distinct_projection_error() {
    let mut dem = parse("error(0.1) D0\ndetector D3");
    dem.detector_fault_sectors[3] = Some(String::new());
    for policy in [IrreduciblePolicy::Drop, IrreduciblePolicy::Refuse] {
        let error = project(&dem, policy).unwrap_err();
        assert_eq!(error, GraphlikeProjectionError::EmptySector { detector: 3 });
        assert_eq!(error.to_string(), "detector D3 fault_sector is empty");
    }
}

// contract: weight wins, exact ties use tuple/Vec order; input order is irrelevant to ties.
#[test]
fn weight_and_canonical_ties() {
    let dem = parse(TWO_CHOICES);
    let result = project(&dem, IrreduciblePolicy::Refuse).unwrap();
    assert_eq!(result.model.errors[4].components, parts("D0 D2 ^ D1 D3"));
    let dem = parse(
        "error(0.1) D1 D3\nerror(0.1) D0 D2\nerror(0.1) D2 D3\nerror(0.1) D0 D1\nerror(0.2) D0 D1 D2 D3",
    );
    let result = project(&dem, IrreduciblePolicy::Refuse).unwrap();
    assert_eq!(result.model.errors[4].components, parts("D0 D1 ^ D2 D3"));
    // Singleton vectors precede pairs sharing the same first detector.
    let dem = parse(
        "error(0.5) D0 D1\nerror(0.5) D2\nerror(0.5) D0\nerror(0.5) D1 D2\nerror(0.2) D0 D1 D2",
    );
    assert_eq!(
        project(&dem, IrreduciblePolicy::Refuse)
            .unwrap()
            .model
            .errors[4]
            .components,
        parts("D0 ^ D1 D2")
    );
}

// contract: weights combine duplicates by Bernoulli XOR in input order from zero.
#[test]
fn duplicate_standalone_probabilities() {
    let dem = parse(
        "error(0.2) D0 D1\nerror(0.2) D1 D0\nerror(0.2) D2 D3\nerror(0.2) D3 D2\nerror(0.3) D0 D2\nerror(0.3) D1 D3\nerror(0.1) D0 D1 D2 D3",
    );
    assert_eq!(
        project(&dem, IrreduciblePolicy::Refuse)
            .unwrap()
            .model
            .errors[6]
            .components,
        parts("D0 D1 ^ D2 D3")
    );
    let dem = parse("error(0.91) D0\nerror(0.000000000000001) D0\nerror(0.333) D0\nerror(0.03) D0");
    let effects: Vec<_> = dem
        .errors
        .iter()
        .enumerate()
        .map(|(i, e)| validate(&dem, i, e).unwrap())
        .collect();
    let probability = dem.errors.iter().fold(0.0, |p, e| {
        p * (1.0 - e.probability) + e.probability * (1.0 - p)
    });
    assert_eq!(
        standalone_mechanisms(&dem, &effects)[&vec![0]][&vec![]].to_bits(),
        ((1.0 - probability) / probability).ln().to_bits()
    );
}

// contract: observable parity constrains the candidate, including labels above bit 64.
#[test]
fn observable_xor_and_label_order() {
    let dem = parse(
        "error(0.4) D0 D1 L0\nerror(0.4) D2 D3\nerror(0.01) D0 D2 L70\nerror(0.01) D1 D3 L70\nerror(0.1) D0 D1 D2 D3",
    );
    assert_eq!(
        project(&dem, IrreduciblePolicy::Refuse)
            .unwrap()
            .model
            .errors[4]
            .components,
        parts("D0 D2 L70 ^ D1 D3 L70")
    );
    let dem = parse("error(0.1) D0 D1 L0\nerror(0.1) D2\nerror(0.1) D0 D1 D2");
    assert!(matches!(
        project(&dem, IrreduciblePolicy::Refuse),
        Err(GraphlikeProjectionError::Irreducible { count: 1, .. })
    ));
    let dem = parse(
        "error(0.5) D0 L1\nerror(0.5) D0 L0\nerror(0.5) D1 D2 L1\nerror(0.5) D1 D2 L0\nerror(0.1) D0 D1 D2",
    );
    assert_eq!(
        project(&dem, IrreduciblePolicy::Refuse)
            .unwrap()
            .model
            .errors[4]
            .components,
        parts("D0 L0 ^ D1 D2 L0")
    );
}

// contract: equal effects share one cached choice, even before their standalones.
#[test]
fn repeated_effects_and_future_standalones() {
    let dem = parse(&format!("error(0.333) D0 D1 D2 D3\n{TWO_CHOICES}"));
    let original = dem.clone();
    let result = project(&dem, IrreduciblePolicy::Refuse).unwrap();
    assert_eq!(
        result.model.errors[0].components,
        result.model.errors[5].components
    );
    assert_eq!(
        result.model.errors[0].probability.to_bits(),
        dem.errors[0].probability.to_bits()
    );
    assert_eq!(
        result.model.errors[5].probability.to_bits(),
        dem.errors[5].probability.to_bits()
    );
    assert_eq!(dem, original);
}

// contract: passthrough preserves zero/single/pair effects, written order, metadata, and bits.
#[test]
fn passthrough_exactly_as_written() {
    let mut dem = parse(
        "error(0.1) L2 L0\nerror(0.2) D1 L2 L0\nerror(0.3) D2 D0 L2 L0\ndetector(1,2,3) D5\nlogical_observable L7",
    );
    dem.errors.push(StructuredDemError {
        probability: 0.7,
        components: vec![StructuredDemComponent {
            detectors: vec![],
            observables: vec![],
        }],
    });
    let result = project(&dem, IrreduciblePolicy::Refuse).unwrap();
    assert_eq!(result.model, dem);
    for (a, b) in dem.errors.iter().zip(&result.model.errors) {
        assert_eq!(a.probability.to_bits(), b.probability.to_bits());
    }
    assert_eq!(result.report.unlabelled_detectors, 6);
}

// contract: duplicates are rejected in both namespaces, with the offending index.
#[test]
fn repeated_targets_rejected() {
    let mut dem = parse("error(0.1) D0 L0\nerror(0.1) D1 L1");
    dem.errors[1].components[0].detectors.push(1);
    assert_eq!(
        project(&dem, IrreduciblePolicy::Drop).unwrap_err(),
        GraphlikeProjectionError::RepeatedDetector {
            error_index: 1,
            detector: 1
        }
    );
    dem.errors[1].components[0].detectors.pop();
    dem.errors[1].components[0].observables.push(1);
    assert_eq!(
        project(&dem, IrreduciblePolicy::Drop).unwrap_err(),
        GraphlikeProjectionError::RepeatedObservable {
            error_index: 1,
            observable: 1
        }
    );
}

// contract: direct public-field models receive bounds checks before indexing sectors.
#[test]
fn out_of_range_targets_rejected() {
    let mut dem = parse("error(0.1) D0 L0\ndetector D2");
    dem.errors[0].components[0].detectors = vec![99];
    assert_eq!(
        project(&dem, IrreduciblePolicy::Drop).unwrap_err(),
        GraphlikeProjectionError::DetectorOutOfRange {
            error_index: 0,
            detector: 99,
            num_detectors: 3
        }
    );
    dem.errors[0].components[0].detectors = vec![0];
    dem.errors[0].components[0].observables = vec![99];
    assert_eq!(
        project(&dem, IrreduciblePolicy::Drop).unwrap_err(),
        GraphlikeProjectionError::ObservableOutOfRange {
            error_index: 0,
            observable: 99,
            num_observables: 1
        }
    );
}

// contract: dimensions include gaps and declarations, not just referenced targets.
#[test]
fn dimensions_and_sector_length() {
    let dem = parse("error(0.1) D7\ndetector D10");
    assert_eq!(dem.num_detectors, 11);
    assert_eq!(project(&dem, IrreduciblePolicy::Drop).unwrap().model, dem);
    for length in [8, 12] {
        let mut malformed = dem.clone();
        malformed.detector_fault_sectors.resize(length, None);
        assert_eq!(
            project_graphlike(&malformed, IrreduciblePolicy::Drop).unwrap_err(),
            GraphlikeProjectionError::SectorsLength {
                expected: 11,
                actual: length,
            }
        );
    }
    assert_eq!(
        project(&parse(""), IrreduciblePolicy::Refuse)
            .unwrap()
            .report,
        ProjectionReport::default()
    );
}

// contract: exactly one component and finite probabilities in [0, 1).
#[test]
fn component_count_and_probabilities() {
    let mut dem = parse("error(0.1) D0 ^ D1");
    assert_eq!(
        project(&dem, IrreduciblePolicy::Drop).unwrap_err(),
        GraphlikeProjectionError::ComponentCount {
            error_index: 0,
            count: 2
        }
    );
    dem.errors[0].components.clear();
    assert_eq!(
        project(&dem, IrreduciblePolicy::Drop).unwrap_err(),
        GraphlikeProjectionError::ComponentCount {
            error_index: 0,
            count: 0
        }
    );
    dem.errors[0].components = parts("D0");
    for p in [1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.1, 1.1] {
        dem.errors[0].probability = p;
        let GraphlikeProjectionError::InvalidProbability {
            error_index,
            probability,
        } = project(&dem, IrreduciblePolicy::Drop).unwrap_err()
        else {
            panic!("wrong error");
        };
        assert_eq!(error_index, 0);
        assert_eq!(probability.to_bits(), p.to_bits());
    }
    for p in [
        f64::from_bits(1),
        0.5,
        0.9,
        f64::from_bits(1.0_f64.to_bits() - 1),
    ] {
        dem.errors[0].probability = p;
        assert!(project(&dem, IrreduciblePolicy::Refuse).is_ok());
    }
}

const OVERLAP: &str =
    "error(0.1) D0 D1\nerror(0.1) D0 D2\nerror(0.1) D0 D3\nerror(0.1) D0 D1 D2 D3";

// contract: a GF(2) reduction with overlapping support is a distinct immediate error.
#[test]
fn overlap_only_and_observable_span() {
    let dem = parse(OVERLAP);
    for policy in [IrreduciblePolicy::Refuse, IrreduciblePolicy::Drop] {
        assert_eq!(
            project(&dem, policy).unwrap_err(),
            GraphlikeProjectionError::OverlapOnly {
                error_index: 3,
                detectors: vec![0, 1, 2, 3],
                observables: vec![]
            }
        );
    }
    let dem = parse(&format!("{OVERLAP} L0"));
    assert!(matches!(
        project(&dem, IrreduciblePolicy::Refuse),
        Err(GraphlikeProjectionError::Irreducible { count: 1, .. })
    ));
    // Standalones involving outside detectors and silent logicals cannot enter the span.
    let dem = parse(
        "error(0.1) D0 D4\nerror(0.1) D1 D4\nerror(0.1) D2 D4\nerror(0.1) D3 D4\nerror(0.1) L0\nerror(0.1) D0 D1 D2 D3",
    );
    assert!(matches!(
        project(&dem, IrreduciblePolicy::Refuse),
        Err(GraphlikeProjectionError::Irreducible { .. })
    ));
}

fn budget_line() -> String {
    format!(
        "error(0.1) {}",
        (0..13)
            .map(|id| format!("D{id}"))
            .collect::<Vec<_>>()
            .join(" ")
    )
}

// contract: 13 detectors exceed the budget; exactly 12 are searched exhaustively.
#[test]
fn search_budget_boundary() {
    assert_eq!(
        project(&parse(&budget_line()), IrreduciblePolicy::Drop).unwrap_err(),
        GraphlikeProjectionError::SearchBudget {
            error_index: 0,
            detectors: (0..13).collect()
        }
    );
    let mut lines: Vec<_> = (0..12).map(|id| format!("error(0.1) D{id}")).collect();
    lines.push(format!(
        "error(0.2) {}",
        (0..12)
            .map(|id| format!("D{id}"))
            .collect::<Vec<_>>()
            .join(" ")
    ));
    let result = project(&parse(&lines.join("\n")), IrreduciblePolicy::Refuse).unwrap();
    assert_eq!(result.model.errors[12].components.len(), 12);
}

// contract: later immediate failures supersede collected irreducibility, under either policy.
#[test]
fn immediate_error_precedence() {
    for policy in [IrreduciblePolicy::Refuse, IrreduciblePolicy::Drop] {
        let dem = parse(&format!("error(0.1) D20 D21 D22\n{}", budget_line()));
        assert!(matches!(
            project(&dem, policy),
            Err(GraphlikeProjectionError::SearchBudget { error_index: 1, .. })
        ));
        let dem = parse(&format!("error(0.1) D20 D21 D22\n{OVERLAP}"));
        assert!(matches!(
            project(&dem, policy),
            Err(GraphlikeProjectionError::OverlapOnly { error_index: 4, .. })
        ));
        let dem = parse("error(0.1) D0 D1 D2\nerror(1) D0");
        assert!(matches!(
            project(&dem, policy),
            Err(GraphlikeProjectionError::InvalidProbability { error_index: 1, .. })
        ));
        // Every malformed line takes precedence over earlier reduction failures.
        let dem = parse(&format!("{OVERLAP}\nerror(1) D0"));
        assert!(matches!(
            project(&dem, policy),
            Err(GraphlikeProjectionError::InvalidProbability { error_index: 4, .. })
        ));
        let dem = parse(&format!("{}\nerror(1) D0", budget_line()));
        assert!(matches!(
            project(&dem, policy),
            Err(GraphlikeProjectionError::InvalidProbability { error_index: 1, .. })
        ));
        let dem = parse(&format!("error(1) D0\n{OVERLAP}"));
        assert!(matches!(
            project(&dem, policy),
            Err(GraphlikeProjectionError::InvalidProbability { error_index: 0, .. })
        ));
    }
}

// contract: refusal counts all lines and samples only the first five; dropping preserves indices.
#[test]
fn irreducible_refuse_and_drop() {
    let text = "error(0.1) D0 D1 D2\nerror(0.2) D3\n".repeat(7);
    let dem = parse(&text);
    assert_eq!(
        project(&dem, IrreduciblePolicy::Refuse).unwrap_err(),
        GraphlikeProjectionError::Irreducible {
            count: 7,
            first_indices: vec![0, 2, 4, 6, 8]
        }
    );
    let result = project(&dem, IrreduciblePolicy::Drop).unwrap();
    assert_eq!(result.report.dropped_indices, vec![0, 2, 4, 6, 8, 10, 12]);
    assert_eq!(result.model.errors, vec![dem.errors[1].clone(); 7]);
    assert_eq!(result.report.input_hyperedges, 7);
    assert_eq!(result.report.projected, 0);
}

// contract: label conflicts are counted per detector set, never rejected; all report counters.
#[test]
fn diagnostics_and_report_counts() {
    let mut dem = parse(
        "error(0.1) D0 D1\nerror(0.2) D0 D1 L0\nerror(0.3) D1 D0 L0\nerror(0.1) D2 L0\nerror(0.2) D2 L1\nerror(0.3) D2 L2\nerror(0.1) D0 D1 D2\nerror(0.2) D0 D1 D2\nerror(0.1) D3 D4 D5\ndetector(2,3,4) D6",
    );
    dem.detector_fault_sectors = vec![
        Some("XX".into()),
        Some("YY".into()),
        Some("ZZ".into()),
        None,
        None,
        None,
        Some("ZZ".into()),
    ];
    let result = project_graphlike(&dem, IrreduciblePolicy::Drop).unwrap();
    assert_eq!(
        result.report,
        ProjectionReport {
            input_hyperedges: 3,
            projected: 2,
            dropped_indices: vec![8],
            zero_probability_omitted: vec![],
            cross_sector_components: 5,
            label_conflicts: 2,
            unlabelled_detectors: 3,
            distinct_sectors: 3,
        }
    );
    // contract: projection retains every declaration, including unused detectors.
    assert_eq!(
        result.model.detector_fault_sectors,
        dem.detector_fault_sectors
    );
    assert_eq!(result.model.detector_coords, dem.detector_coords);
    assert_eq!(result.model.num_detectors, dem.num_detectors);
    assert_eq!(result.model.num_observables, dem.num_observables);
}

// contract: the structural assertion catches invented labels, wrong effects, size, and probability.
#[test]
fn structural_invariant_is_checked() {
    let dem = parse("error(0.1) D0\nerror(0.1) D1 D2 L0\nerror(0.2) D0 D1 D2 L0");
    let effects: Vec<_> = dem
        .errors
        .iter()
        .enumerate()
        .map(|(i, e)| validate(&dem, i, e).unwrap())
        .collect();
    let standalone = standalone_mechanisms(&dem, &effects);
    let input = &dem.errors[2];
    let output = project(&dem, IrreduciblePolicy::Refuse)
        .unwrap()
        .model
        .errors[2]
        .clone();
    assert!(assert_replacement(2, input, &output, &standalone).is_ok());
    let mut mutations = vec![output.clone(); 5];
    mutations[0].probability = f64::from_bits(output.probability.to_bits() + 1);
    mutations[1].components[0].observables.push(9);
    mutations[2].components.pop();
    mutations[3].components = input.components.clone();
    mutations[4].components.reverse();
    for mutated in mutations {
        assert!(matches!(
            assert_replacement(2, input, &mutated, &standalone),
            Err(GraphlikeProjectionError::InternalInvariant { error_index: 2, .. })
        ));
    }
    // Every pair is standalone and the XOR is correct, but D0 is shared.
    let dem = parse(OVERLAP);
    let effects: Vec<_> = dem
        .errors
        .iter()
        .enumerate()
        .map(|(i, e)| validate(&dem, i, e).unwrap())
        .collect();
    let standalone = standalone_mechanisms(&dem, &effects);
    let overlapping = StructuredDemError {
        probability: dem.errors[3].probability,
        components: parts("D0 D1 ^ D0 D2 ^ D0 D3"),
    };
    assert!(matches!(
        assert_replacement(3, &dem.errors[3], &overlapping, &standalone),
        Err(GraphlikeProjectionError::InternalInvariant { error_index: 3, .. })
    ));
}

// contract: exact f64 comparisons must not turn nearby weights into canonical ties.
#[test]
fn close_weights_and_infinite_weights() {
    let dem = parse(
        "error(0.1) D0 D1\nerror(0.1) D2 D3\nerror(0.100000000000001) D0 D2\nerror(0.1) D1 D3\nerror(0.1) D0 D1 D2 D3",
    );
    assert_eq!(
        project(&dem, IrreduciblePolicy::Refuse)
            .unwrap()
            .model
            .errors[4]
            .components,
        parts("D0 D2 ^ D1 D3")
    );
    // Interior subnormal probabilities may legitimately yield positive infinite weights.
    let mut dem = parse(
        "error(0.1) D0 D1\nerror(0.1) D2\nerror(0.1) D0\nerror(0.1) D1 D2\nerror(0.1) D0 D1 D2",
    );
    for error in &mut dem.errors[..4] {
        error.probability = f64::from_bits(1);
    }
    assert_eq!(
        project(&dem, IrreduciblePolicy::Refuse)
            .unwrap()
            .model
            .errors[4]
            .components,
        parts("D0 ^ D1 D2")
    );
}

// contract: weight accumulation is left to right in canonical component order.
#[test]
fn canonical_weight_accumulation() {
    let blocks: Vec<_> = [1e-15, 30.0, -30.0]
        .into_iter()
        .enumerate()
        .map(|(id, weight)| {
            vec![Block {
                mask: 1 << id,
                effect: (vec![u32::try_from(id).unwrap()], vec![]),
                weight,
                cross: 0,
            }]
        })
        .collect();
    let mut current = Candidate {
        cross: 0,
        weight: 0.0,
        components: vec![],
    };
    let mut best = None;
    Search::new(&[], &blocks).visit(7, &mut current, &mut ObsMask::new(), &mut best);
    assert_eq!(best.unwrap().weight.to_bits(), 0.0_f64.to_bits());
}

// contract: a detector-silent mechanism passes through but is not a standalone search block.
#[test]
fn silent_mechanism_cannot_supply_missing_labels() {
    let dem = parse("error(0.1) L0\nerror(0.1) D0 D1 L0\nerror(0.1) D2\nerror(0.1) D0 D1 D2");
    let result = project(&dem, IrreduciblePolicy::Drop).unwrap();
    assert_eq!(result.report.dropped_indices, vec![3]);
    assert_eq!(result.model.errors, dem.errors[..3]);
}

// contract: all validation variants outrank earlier reduction failures, and the first malformed line wins.
#[test]
fn later_malformed_lines_precede_reduction() {
    for prefix in [OVERLAP.to_owned(), budget_line()] {
        let mut base = parse(&format!("{prefix}\nerror(0.1) D0 L0\nerror(1) D0"));
        let index = base.errors.len() - 2;
        let valid = base.errors[index].clone();
        let mut cases = Vec::new();
        let mut error = valid.clone();
        error.components.clear();
        cases.push((
            error,
            GraphlikeProjectionError::ComponentCount {
                error_index: index,
                count: 0,
            },
        ));
        let mut error = valid.clone();
        error.probability = 1.0;
        cases.push((
            error,
            GraphlikeProjectionError::InvalidProbability {
                error_index: index,
                probability: 1.0,
            },
        ));
        let mut error = valid.clone();
        error.components[0].detectors.push(99);
        cases.push((
            error,
            GraphlikeProjectionError::DetectorOutOfRange {
                error_index: index,
                detector: 99,
                num_detectors: base.num_detectors,
            },
        ));
        let mut error = valid.clone();
        error.components[0].observables.push(99);
        cases.push((
            error,
            GraphlikeProjectionError::ObservableOutOfRange {
                error_index: index,
                observable: 99,
                num_observables: 1,
            },
        ));
        let mut error = valid.clone();
        error.components[0].detectors.push(0);
        cases.push((
            error,
            GraphlikeProjectionError::RepeatedDetector {
                error_index: index,
                detector: 0,
            },
        ));
        let mut error = valid;
        error.components[0].observables.push(0);
        cases.push((
            error,
            GraphlikeProjectionError::RepeatedObservable {
                error_index: index,
                observable: 0,
            },
        ));
        for (error, expected) in cases {
            base.errors[index] = error;
            for policy in [IrreduciblePolicy::Refuse, IrreduciblePolicy::Drop] {
                assert_eq!(project(&base, policy).unwrap_err(), expected);
            }
        }
    }
}

// contract: negative future weights disable weight pruning, and lower cross always precedes weight.
#[test]
fn pruning_preserves_negative_weights_and_lower_cross() {
    let dem = parse(
        "error(0.9) D0\nerror(0.1) D1 D2\nerror(0.01) D0 D1\nerror(0.999) D2\nerror(0.1) D0 D1 D2",
    );
    assert_eq!(
        project(&dem, IrreduciblePolicy::Refuse)
            .unwrap()
            .model
            .errors[4]
            .components,
        parts("D0 D1 ^ D2")
    );
    let mut dem = parse(
        "error(0.4) D0 D1\nerror(0.4) D2 D3\nerror(0.01) D0 D2\nerror(0.01) D1 D3\nerror(0.1) D0 D1 D2 D3",
    );
    dem.detector_fault_sectors = vec![
        Some("XX".into()),
        Some("YY".into()),
        Some("XX".into()),
        Some("YY".into()),
    ];
    let result = project_graphlike(&dem, IrreduciblePolicy::Refuse).unwrap();
    assert_eq!(result.model.errors[4].components, parts("D0 D2 ^ D1 D3"));
}

// contract: equality must remain searchable so an incumbent can improve by canonical order.
#[test]
fn pruning_never_discards_equality() {
    let dem = parse(
        "error(0.5) D0 D1\nerror(0.5) D2 D3\nerror(0.5) D0 D2\nerror(0.5) D1 D3\nerror(0.1) D0 D1 D2 D3",
    );
    let effects: Vec<_> = dem
        .errors
        .iter()
        .enumerate()
        .map(|(i, e)| validate(&dem, i, e).unwrap())
        .collect();
    let standalone = standalone_mechanisms(&dem, &effects);
    let blocks = local_blocks(&effects[4].0, &standalone, &[const { None }; 4]);
    let mut best = Some(Candidate {
        cross: 0,
        weight: 0.0,
        components: vec![&effects[2], &effects[3]],
    });
    let mut current = Candidate {
        cross: 0,
        weight: 0.0,
        components: Vec::new(),
    };
    Search::new(&[], &blocks).visit(15, &mut current, &mut ObsMask::new(), &mut best);
    assert_eq!(best.unwrap().components, vec![&effects[0], &effects[1]]);
}

// The small-case oracle is deliberately unpruned and independent of Search.
// Two-bit labels and an eight-entry stack avoid allocations at every leaf.
struct ReferenceBlock {
    detectors: u16,
    observables: u8,
    cross: usize,
    weight: f64,
    canonical_index: usize,
}

struct ReferencePath {
    cross: usize,
    weight: f64,
    components: [usize; 8],
    len: usize,
}

struct UnprunedReference {
    blocks: Vec<Vec<ReferenceBlock>>,
    best: Option<(usize, f64, Vec<usize>)>,
}

impl UnprunedReference {
    fn visit(&mut self, remaining: u16, parity: u8, path: &mut ReferencePath) {
        if remaining == 0 {
            if parity == 0
                && self
                    .best
                    .as_ref()
                    .is_none_or(|(cross, weight, components)| {
                        (path.cross, path.weight, &path.components[..path.len]).partial_cmp(&(
                            *cross,
                            *weight,
                            components.as_slice(),
                        )) == Some(Ordering::Less)
                    })
            {
                self.best = Some((
                    path.cross,
                    path.weight,
                    path.components[..path.len].to_vec(),
                ));
            }
            return;
        }
        let first = remaining.trailing_zeros() as usize;
        for index in 0..self.blocks[first].len() {
            let block = &self.blocks[first][index];
            if remaining & block.detectors == block.detectors {
                let old_weight = path.weight;
                let old_cross = path.cross;
                path.components[path.len] = block.canonical_index;
                path.len += 1;
                path.weight += block.weight;
                path.cross += block.cross;
                self.visit(
                    remaining ^ block.detectors,
                    parity ^ block.observables,
                    path,
                );
                path.weight = old_weight;
                path.cross = old_cross;
                path.len -= 1;
            }
        }
    }
}

fn reference_choice(dem: &StructuredDem, sectors: &[Option<String>]) -> Option<Vec<Effect>> {
    let (target, standalones) = dem.errors.split_last().unwrap();
    let mut components: Vec<_> = standalones
        .iter()
        .map(|error| {
            let part = &error.components[0];
            (
                (part.detectors.clone(), part.observables.clone()),
                ((1.0 - error.probability) / error.probability).ln(),
            )
        })
        .collect();
    components.sort_by(|a, b| a.0.cmp(&b.0));
    let mut reference = UnprunedReference {
        blocks: (0..dem.num_detectors).map(|_| Vec::new()).collect(),
        best: None,
    };
    for (canonical_index, ((detectors, observables), weight)) in components.iter().enumerate() {
        let cross = detectors.len() == 2
            && matches!(
                (
                    &sectors[detectors[0] as usize],
                    &sectors[detectors[1] as usize]
                ),
                (Some(left), Some(right)) if left != right
            );
        reference.blocks[detectors[0] as usize].push(ReferenceBlock {
            detectors: detectors.iter().fold(0, |mask, id| mask | (1 << id)),
            observables: observables.iter().fold(0, |mask, id| mask | (1 << id)),
            cross: usize::from(cross),
            weight: *weight,
            canonical_index,
        });
    }
    let parity = target.components[0]
        .observables
        .iter()
        .fold(0, |mask, id| mask | (1 << id));
    reference.visit(
        (1 << dem.num_detectors) - 1,
        parity,
        &mut ReferencePath {
            cross: 0,
            weight: 0.0,
            components: [0; 8],
            len: 0,
        },
    );
    reference.best.map(|(_, _, indices)| {
        indices
            .into_iter()
            .map(|index| components[index].0.clone())
            .collect()
    })
}

fn check_search_against_reference(
    dem: &StructuredDem,
    sectors: &[Option<String>],
    case: usize,
) -> bool {
    let effects: Vec<_> = dem
        .errors
        .iter()
        .enumerate()
        .map(|(index, error)| validate(dem, index, error).unwrap())
        .collect();
    let standalone = standalone_mechanisms(dem, &effects);
    let target = effects.last().unwrap();
    let blocks = local_blocks(&target.0, &standalone, sectors);
    let expected = reference_choice(dem, sectors);
    assert_eq!(
        choose(target, &blocks),
        expected,
        "case {case}, sectors {sectors:?}, model {dem:?}"
    );
    expected.is_some()
}

struct FixedRng(u32);

impl FixedRng {
    fn below(&mut self, bound: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0 as usize % bound
    }
}

fn dense_model(
    num_detectors: u32,
    detector_fault_sectors: Vec<Option<String>>,
    mut mechanisms: impl FnMut() -> Vec<(f64, Vec<u32>)>,
    observables: Vec<u32>,
) -> StructuredDem {
    let mut errors = Vec::new();
    for first in 0..num_detectors {
        for second in first..num_detectors {
            for (probability, observables) in mechanisms() {
                errors.push(StructuredDemError {
                    probability,
                    components: vec![StructuredDemComponent {
                        detectors: if first == second {
                            vec![first]
                        } else {
                            vec![first, second]
                        },
                        observables,
                    }],
                });
            }
        }
    }
    errors.push(StructuredDemError {
        probability: 0.2,
        components: vec![StructuredDemComponent {
            detectors: (0..num_detectors).collect(),
            observables,
        }],
    });
    StructuredDem {
        errors,
        detector_coords: vec![None; num_detectors as usize],
        detector_fault_sectors,
        num_detectors: num_detectors as usize,
        num_observables: 2,
    }
}

// contract: seeded differential coverage includes exact ties, negative weights,
// all sector declarations, random target parity, and every size from 4 through 8.
#[test]
fn randomized_dense_search_matches_unpruned_reference() {
    let mut rng = FixedRng(0x9e37_79b9);
    let mut nonnegative = 0;
    let mut negative = 0;
    let mut found = 0;
    let mut absent = 0;
    for case in 0..320 {
        let size = 4 + case % 5;
        let probabilities: &[f64] = match case % 4 {
            0 => &[0.5, 0.5],
            1 => &[0.1, 0.1, 0.2, 0.5],
            2 => &[0.1, 0.1, 0.5, 0.8, 0.9],
            _ => &[0.6, 0.6, 0.8, 0.9],
        };
        let sectors: Vec<_> = (0..size)
            .map(|_| [Some("XX"), Some("YY"), Some("ZZ"), None][rng.below(4)].map(str::to_owned))
            .collect();
        let target = rng.below(4);
        let observables = (0..2).filter(|id| target & (1 << id) != 0).collect();
        // Correlated single-label cases also exercise unreachable target parity.
        let shared_label = (case % 16 == 0).then(|| 1 << rng.below(3));
        let dem = dense_model(
            u32::try_from(size).unwrap(),
            sectors.clone(),
            || {
                let labels = shared_label.unwrap_or_else(|| 1 + rng.below(7));
                (0..3)
                    .filter(|label| labels & (1 << label) != 0)
                    .map(|label| {
                        let observables = match label {
                            0 => vec![],
                            1 => vec![0],
                            _ => vec![1],
                        };
                        (probabilities[rng.below(probabilities.len())], observables)
                    })
                    .collect()
            },
            observables,
        );
        if dem.errors.iter().all(|error| error.probability <= 0.5) {
            nonnegative += 1;
        } else {
            negative += 1;
        }
        if check_search_against_reference(&dem, &sectors, case) {
            found += 1;
        } else {
            absent += 1;
        }
    }
    assert!(nonnegative >= 80 && negative >= 80);
    assert!(found > 0 && absent > 0);
}

// contract: a readable fully labelled eight-detector example agrees with exhaustive search.
#[test]
fn dense_eight_detector_search_matches_unpruned_reference() {
    let dem = dense_model(
        8,
        vec![None; 8],
        || vec![(0.1, vec![]), (0.1, vec![0]), (0.1, vec![1])],
        vec![],
    );
    assert!(check_search_against_reference(
        &dem,
        &[const { None }; 8],
        0
    ));
    let result = project(&dem, IrreduciblePolicy::Refuse).unwrap();
    assert_eq!(
        result.model.errors.last().unwrap().components,
        parts("D0 D1 ^ D2 D3 ^ D4 D5 ^ D6 D7")
    );
}

// contract: incremental parity preserves sparse labels and more than 64 local observable bits.
#[test]
fn incremental_parity_handles_wide_and_sparse_labels() {
    let mut dem = parse("error(0.1) D0\nerror(0.1) D1 D2\nerror(0.1) D0 D1 D2");
    let shared: Vec<_> = (0..70).collect();
    dem.errors[0].components[0].observables = shared.clone();
    dem.errors[1].components[0].observables = shared;
    dem.errors[1].components[0].observables.push(u32::MAX - 1);
    dem.errors[2].components[0].observables = vec![u32::MAX - 1];
    dem.num_observables = u32::MAX as usize;
    let result = project(&dem, IrreduciblePolicy::Refuse).unwrap();
    assert_eq!(
        result.model.errors[2].components,
        vec![
            dem.errors[0].components[0].clone(),
            dem.errors[1].components[0].clone()
        ]
    );
}

// contract: both zero signs are omitted exactly at any support size, before reduction handling.
#[test]
fn zero_probability_errors_are_omitted_exactly() {
    for zero in [0.0, -0.0] {
        let mut dem = parse(&format!(
            "error(0.2) D0 L0\nerror(0) L0\nerror(0) D0\nerror(0) D0 D1\n\
             error(0) D0 D1 D2\n{}\nerror(0.3) D1 D2\nerror(0.4) D0 D1 D2 L0",
            budget_line(),
        ));
        // Include a detector-silent, target-free line, valid for projection's raw input.
        dem.errors[1].components[0].observables.clear();
        for error in &mut dem.errors[1..6] {
            error.probability = zero;
        }
        for policy in [IrreduciblePolicy::Drop, IrreduciblePolicy::Refuse] {
            let result = project(&dem, policy).unwrap();
            assert_eq!(result.report.zero_probability_omitted, [1, 2, 3, 4, 5]);
            assert_eq!(result.report.dropped_indices, [] as [usize; 0]);
            assert_eq!(result.report.input_hyperedges, 1);
            assert_eq!(result.report.projected, 1);
            let mut expected = dem.clone();
            expected.errors = vec![
                dem.errors[0].clone(),
                dem.errors[6].clone(),
                StructuredDemError {
                    probability: 0.4,
                    components: parts("D0 L0 ^ D1 D2"),
                },
            ];
            assert_eq!(result.model, expected);
        }
        let mut overlap = parse(OVERLAP);
        overlap.errors[3].probability = zero;
        let result = project(&overlap, IrreduciblePolicy::Refuse).unwrap();
        assert_eq!(result.report.zero_probability_omitted, [3]);
        assert_eq!(result.report.input_hyperedges, 0);
        assert_eq!(result.model.errors, overlap.errors[..3]);
    }
}

// contract: zero-probability standalones supply neither catalogue blocks nor label conflicts.
#[test]
fn zero_probability_standalones_do_not_enter_catalogue() {
    let dem = parse("error(0) D0 L0\nerror(0.2) D1 D2\nerror(0) D1 D2 L0\nerror(0.3) D0 D1 D2 L0");
    let result = project(&dem, IrreduciblePolicy::Drop).unwrap();
    assert_eq!(result.report.zero_probability_omitted, [0, 2]);
    assert_eq!(result.report.dropped_indices, [3]);
    assert_eq!(result.report.label_conflicts, 0);
    assert_eq!(result.report.input_hyperedges, 1);
    assert_eq!(result.model.errors, [dem.errors[1].clone()]);
    assert_eq!(
        project(&dem, IrreduciblePolicy::Refuse).unwrap_err(),
        GraphlikeProjectionError::Irreducible {
            count: 1,
            first_indices: vec![3]
        }
    );
}

// contract: zero omissions still undergo every structural validation before reduction starts.
#[test]
fn zero_probability_lines_are_fully_validated() {
    use GraphlikeProjectionError as E;
    let mut dem = parse(&format!("{}\nerror(0) D0 L0", budget_line()));
    let index = dem.errors.len() - 1;
    let original = dem.errors[index].clone();
    for (components, expected) in [
        (
            vec![],
            E::ComponentCount {
                error_index: index,
                count: 0,
            },
        ),
        (
            parts("D0 ^ D1"),
            E::ComponentCount {
                error_index: index,
                count: 2,
            },
        ),
        (
            parts("D99"),
            E::DetectorOutOfRange {
                error_index: index,
                detector: 99,
                num_detectors: dem.num_detectors,
            },
        ),
        (
            parts("D0 L99"),
            E::ObservableOutOfRange {
                error_index: index,
                observable: 99,
                num_observables: 1,
            },
        ),
        (
            vec![StructuredDemComponent {
                detectors: vec![0, 0],
                observables: vec![],
            }],
            E::RepeatedDetector {
                error_index: index,
                detector: 0,
            },
        ),
        (
            vec![StructuredDemComponent {
                detectors: vec![0],
                observables: vec![0, 0],
            }],
            E::RepeatedObservable {
                error_index: index,
                observable: 0,
            },
        ),
    ] {
        for zero in [0.0, -0.0] {
            dem.errors[index] = StructuredDemError {
                probability: zero,
                components: components.clone(),
            };
            for policy in [IrreduciblePolicy::Drop, IrreduciblePolicy::Refuse] {
                assert_eq!(project(&dem, policy).unwrap_err(), expected);
            }
        }
    }
    dem.errors[index] = original;
    dem.errors[0].probability = 0.0;
    dem.errors[index].probability = 1.0;
    assert_eq!(
        project(&dem, IrreduciblePolicy::Refuse).unwrap_err(),
        E::InvalidProbability {
            error_index: index,
            probability: 1.0
        }
    );
}
