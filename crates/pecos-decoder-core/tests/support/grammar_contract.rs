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
                            matches!(
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
