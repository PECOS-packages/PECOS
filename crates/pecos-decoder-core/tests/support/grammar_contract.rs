// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

use pecos_decoder_core::errors::DecoderError;

pub fn unescape(text: &str) -> String {
    let mut result = String::new();
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        result.push(if c == '\\' {
            match chars.next().unwrap() {
                'n' => '\n',
                't' => '\t',
                '\\' => '\\',
                'u' => {
                    assert_eq!(chars.next(), Some('{'));
                    let digits: String = chars.by_ref().take_while(|&c| c != '}').collect();
                    char::from_u32(u32::from_str_radix(&digits, 16).unwrap()).unwrap()
                }
                other => panic!("unknown fixture escape: {other}"),
            }
        } else {
            c
        });
    }
    result
}

pub fn assert_outcome(
    input: &str,
    accepted: bool,
    consumer: &str,
    result: Result<Option<(usize, usize)>, DecoderError>,
    expected: Option<(usize, usize)>,
) {
    match result {
        Ok(counts) => {
            assert!(accepted, "{consumer} accepted invalid grammar: {input:?}");
            if let (Some(counts), Some(expected)) = (counts, expected) {
                assert_eq!(counts, expected, "{consumer}: {input:?}");
            }
        }
        Err(DecoderError::InvalidDemSyntax(message)) => {
            assert!(!accepted, "{consumer}: {input:?}: {message}");
        }
        Err(DecoderError::InvalidConfiguration(message)) if accepted => {
            let overflow = message.split_once(" index ").is_some_and(|(kind, rest)| {
                matches!(kind, "detector" | "observable")
                    && rest
                        .split_once(" exceeds the supported maximum ")
                        .is_some_and(|(index, maximum)| {
                            input.contains(index)
                                && matches!(
                                    (index.parse::<u64>(), maximum.parse::<u64>()),
                                    (Ok(index), Ok(maximum)) if index > maximum
                                )
                        })
            });
            let supports_index_limit = matches!(
                consumer,
                "SparseDem"
                    | "DemCheckMatrix"
                    | "DemMatchingGraph"
                    | "StructuredDem"
                    | "metadata"
                    | "coordinates"
                    | "CorrelationTable"
                    | "ghost"
            );
            assert!(
                (supports_index_limit && overflow)
                    || message.contains(" requires a flattened DEM: "),
                "{consumer}: unexpected semantic rejection of {input:?}: {message}"
            );
        }
        Err(error) => panic!("{consumer}: unexpected rejection of {input:?}: {error}"),
    }
}

pub fn assert_mechanism_effects(input: &str, dense_index_threshold: u64) {
    use pecos_decoder_core::dem::grammar::{Kind, Target, parse_line};
    use pecos_decoder_core::dem::{DemCheckMatrix, DemMatchingGraph, SparseDem};
    use std::collections::{BTreeMap, BTreeSet};

    let mut instructions: Vec<_> = input
        .lines()
        .filter_map(|line| parse_line(line).unwrap())
        .collect();
    let mut detector_ids = BTreeSet::new();
    let mut observable_ids = BTreeSet::new();
    for instruction in &instructions {
        for target in &instruction.targets {
            match target {
                Target::Detector(id) => {
                    detector_ids.insert(*id);
                }
                Target::Observable(id) => {
                    observable_ids.insert(*id);
                }
                _ => {}
            }
        }
    }
    // Dimensions and index limits are tested separately; compact huge indices to test parity too.
    let text = if detector_ids
        .iter()
        .chain(&observable_ids)
        .any(|&id| id >= dense_index_threshold)
    {
        let detectors: BTreeMap<_, _> = detector_ids.into_iter().zip(0u64..).collect();
        let observables: BTreeMap<_, _> = observable_ids.into_iter().zip(0u64..).collect();
        for instruction in &mut instructions {
            for target in &mut instruction.targets {
                match target {
                    Target::Detector(id) => *id = detectors[id],
                    Target::Observable(id) => *id = observables[id],
                    _ => {}
                }
            }
        }
        instructions
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        input.to_string()
    };
    let sparse = SparseDem::from_dem_str(&text).unwrap();
    let matrix = DemCheckMatrix::from_dem_str(&text).unwrap();
    assert_eq!(sparse.mechanisms.len(), matrix.num_mechanisms, "{input:?}");
    let errors: Vec<_> = instructions
        .into_iter()
        .filter(|i| i.kind == Kind::Error)
        .collect();
    assert_eq!(sparse.mechanisms.len(), errors.len(), "{input:?}");
    for (column, ((probability, detectors, observables), mut instruction)) in
        sparse.mechanisms.iter().zip(errors).enumerate()
    {
        assert!(
            (probability - matrix.error_priors[column]).abs() < f64::EPSILON,
            "{input:?}"
        );
        let column_detectors: Vec<_> = matrix
            .check_matrix
            .column(column)
            .iter()
            .enumerate()
            .filter(|&(_, &bit)| bit != 0)
            .map(|(row, _)| u32::try_from(row).unwrap())
            .collect();
        let column_observables: Vec<_> = matrix
            .observable_matrix
            .column(column)
            .iter()
            .enumerate()
            .filter(|&(_, &bit)| bit != 0)
            .map(|(row, _)| u32::try_from(row).unwrap())
            .collect();
        assert_eq!(
            *detectors, column_detectors,
            "{input:?}, mechanism {column}"
        );
        assert_eq!(
            *observables, column_observables,
            "{input:?}, mechanism {column}"
        );

        // Isolate each fault before independent parallel edges lose their fault provenance.
        let mut graph = DemMatchingGraph::from_dem_str(&instruction.to_string()).unwrap();
        if *probability == 0.0 {
            assert!(graph.edges.is_empty(), "{input:?}");
            // A zero-probability mechanism still has an effect when it fires.
            instruction.args[0] = 0.5;
            graph = DemMatchingGraph::from_dem_str(&instruction.to_string()).unwrap();
        }
        assert_eq!(graph.skipped_hyperedges, 0, "{input:?}");
        let mut endpoints = vec![0; sparse.num_detectors];
        for edge in graph.edges {
            assert_eq!(edge.fault_id, 0, "{input:?}");
            endpoints[edge.node1 as usize] ^= 1;
            if let Some(node2) = edge.node2 {
                endpoints[node2 as usize] ^= 1;
            }
        }
        assert_eq!(
            endpoints,
            matrix.check_matrix.column(column).to_vec(),
            "{input:?}, mechanism {column}"
        );
    }
}
