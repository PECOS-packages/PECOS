// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

#[path = "support/grammar_contract.rs"]
mod contract;

use contract::{assert_outcome, unescape};
use pecos_decoder_core::dem::grammar::{Instruction, Kind, Target, parse_line};
use pecos_decoder_core::dem::{
    DemCheckMatrix, DemMatchingGraph, SparseDem, parse_detector_coords, utils,
};
use pecos_decoder_core::window::StructuredDem;

// Dense per-detector storage is sized by the maximum index.
const DENSE_INDEX_THRESHOLD: u64 = 1 << 24;

const FIXTURE: &str = include_str!("fixtures/stim_dem_grammar.tsv");

fn tokenize(text: &str) -> Result<Vec<Instruction>, String> {
    text.lines()
        .filter_map(|line| parse_line(line).transpose())
        .collect::<Result<_, _>>()
        .map_err(|error| error.to_string())
}

#[test]
fn tokenizer_matches_stim_oracle() {
    for row in FIXTURE.lines().skip(1) {
        let fields: Vec<_> = row.split('\t').collect();
        let input = unescape(fields[0]);
        let result = tokenize(&input);
        assert_eq!(
            result.is_ok(),
            fields[1] == "accept",
            "{input:?}: {result:?}"
        );
        if fields[1] == "accept" {
            assert_eq!(
                result.unwrap(),
                tokenize(&unescape(fields.get(2).copied().unwrap_or_default())).unwrap(),
                "{input:?}"
            );
        }
    }
}

#[test]
fn flat_consumers_match_grammar_verdicts_and_dimensions() {
    for row in FIXTURE.lines().skip(1) {
        let fields: Vec<_> = row.split('\t').collect();
        let input = unescape(fields[0]);
        let accepted = fields[1] == "accept";
        let expected = if accepted {
            utils::parse_dem_metadata(&unescape(fields.get(2).copied().unwrap_or_default())).ok()
        } else {
            None
        };
        let largest_index = tokenize(&input)
            .ok()
            .into_iter()
            .flatten()
            .flat_map(|instruction| instruction.targets)
            .filter_map(|target| match target {
                Target::Detector(index) | Target::Observable(index) => Some(index),
                _ => None,
            })
            .max();
        // Indices above u32::MAX must still exercise each reader's overflow check.
        let check_dense_readers = largest_index
            .is_none_or(|index| index < DENSE_INDEX_THRESHOLD || index > u64::from(u32::MAX));
        let mut outcomes = vec![
            (
                "SparseDem",
                SparseDem::from_dem_str(&input)
                    .map(|dem| Some((dem.num_detectors, dem.num_observables))),
            ),
            ("metadata", utils::parse_dem_metadata(&input).map(Some)),
            ("validate", utils::validate_dem(&input).map(|()| None)),
            ("coordinates", parse_detector_coords(&input).map(|_| None)),
            (
                "CorrelationTable",
                pecos_decoder_core::correlation_table::CorrelationTable::from_dem_str(
                    &input,
                    &std::collections::BTreeMap::new(),
                    0,
                )
                .map(|_| None),
            ),
            (
                "ghost",
                pecos_decoder_core::ghost_protocol::extract_ghost_edges_from_dem(&input, &vec![])
                    .map(|_| None),
            ),
            (
                "perturb",
                pecos_decoder_core::perturbed::perturb_dem(&input, 0.5, &mut || 0.5).map(
                    |rendered| {
                        let original = tokenize(&input).unwrap();
                        let perturbed = tokenize(&rendered).unwrap();
                        assert_eq!(original.len(), perturbed.len(), "{input:?}");
                        for (mut before, after) in original.into_iter().zip(perturbed) {
                            if before.kind == Kind::Error {
                                before.args.clone_from(&after.args);
                            }
                            assert_eq!(before, after, "{input:?}");
                        }
                        None
                    },
                ),
            ),
        ];
        if check_dense_readers {
            outcomes.extend([
                (
                    "DemCheckMatrix",
                    DemCheckMatrix::from_dem_str(&input)
                        .map(|dem| Some((dem.num_detectors, dem.num_observables))),
                ),
                (
                    "DemMatchingGraph",
                    DemMatchingGraph::from_dem_str(&input)
                        .map(|dem| Some((dem.num_detectors, dem.num_observables))),
                ),
                (
                    "StructuredDem",
                    StructuredDem::from_dem_str(&input)
                        .map(|dem| Some((dem.num_detectors, dem.num_observables))),
                ),
            ]);
        }
        for (consumer, result) in outcomes {
            if accepted
                && tokenize(&input).unwrap().iter().any(|instruction| {
                    matches!(
                        instruction.kind,
                        Kind::Repeat | Kind::ShiftDetectors | Kind::EndRepeat
                    )
                })
            {
                assert!(
                    result
                        .as_ref()
                        .unwrap_err()
                        .to_string()
                        .contains("requires a flattened DEM")
                );
            }
            assert_outcome(&input, accepted, consumer, result, expected);
        }
    }
}

#[test]
fn sparse_indices_report_their_limit() {
    let error = SparseDem::from_dem_str("error(0.1) D4294967296").unwrap_err();
    assert!(matches!(
        error,
        pecos_decoder_core::errors::DecoderError::InvalidConfiguration(_)
    ));
    assert!(
        error
            .to_string()
            .contains("detector index 4294967296 exceeds the supported maximum 4294967295")
    );
}

#[test]
fn instruction_tags_comments_and_components_retain_their_meaning() {
    let dem = "ERROR[tag](0.1) d0 L0 ^ D0 l1 # D99\ndetector(1, 2) d0\nlogical_observable l2";
    let sparse = SparseDem::from_dem_str(dem).unwrap();
    assert_eq!(sparse.mechanisms, vec![(0.1, vec![], vec![0, 1])]);
    let structured = StructuredDem::from_dem_str(dem).unwrap();
    assert_eq!(structured.errors[0].components.len(), 2);
    assert_eq!(structured.errors[0].components[0].detectors, [0]);
    let graph = DemMatchingGraph::from_dem_str(dem).unwrap();
    assert_eq!((graph.num_detectors, graph.num_observables), (1, 3));
    assert_eq!(
        DemMatchingGraph::from_dem_str("error() D0")
            .unwrap()
            .edges
            .len(),
        0
    );
    assert!(
        SparseDem::from_dem_str("error() D0").unwrap().mechanisms[0]
            .0
            .abs()
            < f64::EPSILON
    );
}

#[test]
fn grammar_validates_declarations_unknown_names_and_tag_escapes() {
    for text in [
        "unknown(0.1) D0",
        "\u{a0}}",
        "}\u{a0}",
        "error(\u{a0}0.1) D0",
        "error(0.1\u{a0}) D0",
        "error(\u{c}0.1) D0",
        "error(0.1\u{c}) D0",
        "detector L0",
        "detector D0 ^ D1",
        "logical_observable D0",
        "logical_observable(1) L0",
        "error(0.1) D0\u{000b}D1",
        "error[bad\\t](0.1) D0",
    ] {
        assert!(parse_line(text).is_err(), "{text}");
    }
    for text in [
        "error[a\\Cb\\Bc\\n](.1) D0",
        "error[]() D0",
        "detector() D0",
    ] {
        let instruction = parse_line(text).unwrap().unwrap();
        assert_eq!(
            parse_line(&instruction.to_string()).unwrap(),
            Some(instruction)
        );
    }
}

#[test]
fn extensions_require_explicit_opt_in() {
    use pecos_decoder_core::dem::grammar::{Options, parse_line_with_options};
    for text in [
        "error(0.1) D0 TP0",
        "pecos_observable {\"id\":0}",
        "pecos_tracked_pauli {\"id\":0}",
    ] {
        assert!(parse_line(text).is_err());
        assert!(
            parse_line_with_options(
                text,
                Options {
                    pecos_extensions: true
                }
            )
            .is_ok()
        );
    }
    assert!(
        parse_line_with_options(
            "pecos_unknown {}",
            Options {
                pecos_extensions: true
            }
        )
        .is_err()
    );
}

#[test]
fn correlations_and_perturbation_use_tagged_instructions() {
    use pecos_decoder_core::correlation_table::CorrelationTable;
    use std::collections::BTreeMap;
    let edges = BTreeMap::from([((0, u32::MAX), 0), ((1, u32::MAX), 1)]);
    let dem = "ERROR[tag](0.1) d0 ^ d1 # D99";
    let table = CorrelationTable::from_dem_str(dem, &edges, 2).unwrap();
    assert!(table.has_correlations());
    let filtered = CorrelationTable::from_dem_str("error[tag](0.8) D0 ^ D1", &edges, 2).unwrap();
    assert!(!filtered.has_correlations());
    let rendered = pecos_decoder_core::perturbed::perturb_dem(dem, 0.5, &mut || 0.5).unwrap();
    let original = tokenize(dem).unwrap().remove(0);
    let perturbed = tokenize(&rendered).unwrap().remove(0);
    assert_eq!(original.tag, perturbed.tag);
    assert_eq!(original.targets, perturbed.targets);
    assert!((original.args[0] - perturbed.args[0]).abs() > f64::EPSILON);
}

#[test]
fn perturbation_cannot_render_non_finite_probabilities() {
    let error = pecos_decoder_core::perturbed::perturb_dem("error(0) D0", 1000.0, &mut || 0.001)
        .unwrap_err();
    assert!(error.to_string().contains("non-finite probability"));
}

#[test]
fn check_matrix_cancels_duplicate_targets() {
    let matrix = DemCheckMatrix::from_dem_str("error(0.1) D0 D0").unwrap();
    assert_eq!(matrix.check_matrix.dim(), (1, 1));
    assert!(matrix.check_matrix.column(0).iter().all(|&bit| bit == 0));
}

#[test]
fn leading_whitespace_and_repeat_forms_match_stim() {
    for prefix in ["\u{000b}", "\u{000c}"] {
        assert_eq!(
            parse_line(&format!("{prefix}error(0.1) D0")).unwrap(),
            parse_line("error(0.1) D0").unwrap()
        );
    }
    for text in ["repeat 2{", "repeat 2 {}", "repeat 0 {"] {
        let instruction = parse_line(text).unwrap().unwrap();
        assert_eq!(instruction.kind, Kind::Repeat);
        assert_eq!(
            parse_line(&instruction.to_string()).unwrap(),
            Some(instruction)
        );
        let error = SparseDem::from_dem_str(text).unwrap_err();
        assert!(matches!(
            error,
            pecos_decoder_core::errors::DecoderError::InvalidConfiguration(_)
        ));
        assert!(error.to_string().contains("requires a flattened DEM:"));
    }
}

#[test]
fn all_tokenizer_errors_have_the_syntax_discriminant() {
    use pecos_decoder_core::dem::grammar::{Options, parse_line_with_options};
    use pecos_decoder_core::errors::DecoderError;
    for text in [
        "error[unclosed(0.1) D0",
        r"error[bad\t](0.1) D0",
        "repeat 2",
        "repeat {",
        "error(0.1) D18446744073709551616",
        "pecos_observable{}",
        "@bad",
    ] {
        let error = parse_line_with_options(
            text,
            Options {
                pecos_extensions: true,
            },
        )
        .unwrap_err();
        assert!(
            matches!(error, DecoderError::InvalidDemSyntax(_)),
            "{text}: {error}"
        );
    }
    assert!(parse_line("@bad").unwrap_err().to_string().contains("@bad"));
}
