// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

use pecos_decoder_core::dem::SparseDem;
use pecos_decoder_core::obs_mask::ObsMask;
use pecos_frontier::{FrontierConfig, FrontierDecoder, FrontierResult};
use serde::Deserialize;
use std::collections::BTreeMap;

const FIXTURES_JSON: &str = include_str!("fixtures/upstream_fixtures.json");

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixtureFile {
    generator: String,
    fixtures: Vec<Fixture>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    name: String,
    mechanisms: Vec<(f64, Vec<u32>, Vec<u32>)>,
    num_detectors: usize,
    num_observables: usize,
    syndromes: Vec<u128>,
    pruned: PruningConfig,
    expected_unpruned: Vec<ExpectedResult>,
    expected_pruned: Vec<ExpectedResult>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PruningConfig {
    k: usize,
    delta: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpectedResult {
    syndrome: u128,
    status: String,
    logical_hat: Option<u128>,
    log_evidence: Option<f64>,
    terminal_log_masses: BTreeMap<String, f64>,
    engine: String,
}

fn parse_fixtures() -> FixtureFile {
    serde_json::from_str(FIXTURES_JSON).expect("upstream fixture file must parse")
}

fn dense_syndrome(mask: u128, num_detectors: usize) -> Vec<u8> {
    (0..num_detectors)
        .map(|bit| u8::from(mask & (1_u128 << bit) != 0))
        .collect()
}

fn mask_as_u128(mask: &ObsMask) -> u128 {
    assert!(
        mask.words().iter().skip(2).all(|&word| word == 0),
        "fixture labels must fit in u128"
    );
    u128::from(mask.words().first().copied().unwrap_or(0))
        | (u128::from(mask.words().get(1).copied().unwrap_or(0)) << 64)
}

fn actual_masses(result: &FrontierResult) -> BTreeMap<u128, f64> {
    result
        .logical_masses
        .iter()
        .map(|entry| (mask_as_u128(&entry.logical), entry.log_mass))
        .collect()
}

fn assert_expected_results(
    fixture: &Fixture,
    config: FrontierConfig,
    expected_results: &[ExpectedResult],
    regime: &str,
) {
    let dem = SparseDem {
        mechanisms: fixture.mechanisms.clone(),
        detector_coords: BTreeMap::new(),
        num_detectors: fixture.num_detectors,
        num_observables: fixture.num_observables,
    };
    let mut decoder = FrontierDecoder::from_sparse_dem(&dem, config).unwrap_or_else(|error| {
        panic!(
            "{} {regime}: decoder construction failed: {error}",
            fixture.name
        )
    });

    let mut expected_by_syndrome = BTreeMap::new();
    for expected in expected_results {
        assert!(
            expected_by_syndrome
                .insert(expected.syndrome, expected)
                .is_none(),
            "{} {regime}: duplicate expected syndrome",
            fixture.name
        );
    }
    assert_eq!(
        expected_by_syndrome.len(),
        fixture.syndromes.len(),
        "{} {regime}: expected result count differs from syndrome count",
        fixture.name
    );

    for &syndrome_mask in &fixture.syndromes {
        let expected = expected_by_syndrome
            .remove(&syndrome_mask)
            .unwrap_or_else(|| panic!("{} {regime}: missing expected result", fixture.name));
        assert_eq!(
            expected.engine, "native_binary",
            "{} {regime} syndrome {syndrome_mask}: unexpected engine",
            fixture.name
        );
        let syndrome = dense_syndrome(syndrome_mask, fixture.num_detectors);
        let decoded = decoder.decode(&syndrome);

        match expected.status.as_str() {
            "ok" => {
                let result = decoded.unwrap_or_else(|error| {
                    panic!(
                        "{} {regime} syndrome {syndrome_mask}: expected success, got {error}",
                        fixture.name
                    )
                });
                let logical_hat = expected.logical_hat.expect("ok result needs logical_hat");
                assert_eq!(
                    mask_as_u128(&result.predicted),
                    logical_hat,
                    "{} {regime} syndrome {syndrome_mask}: predicted label",
                    fixture.name
                );

                let expected_masses: BTreeMap<u128, f64> = expected
                    .terminal_log_masses
                    .iter()
                    .map(|(label, &mass)| {
                        (label.parse().expect("logical label must fit in u128"), mass)
                    })
                    .collect();
                let actual_masses = actual_masses(&result);
                assert_eq!(
                    actual_masses.len(),
                    expected_masses.len(),
                    "{} {regime} syndrome {syndrome_mask}: terminal-label count",
                    fixture.name
                );
                for (label, expected_mass) in expected_masses {
                    let actual_mass = actual_masses.get(&label).unwrap_or_else(|| {
                        panic!(
                            "{} {regime} syndrome {syndrome_mask}: missing label {label}",
                            fixture.name
                        )
                    });
                    assert!(
                        (actual_mass - expected_mass).abs() <= 1e-9,
                        "{} {regime} syndrome {syndrome_mask}, label {label}: expected {expected_mass}, got {actual_mass}",
                        fixture.name
                    );
                }

                let expected_evidence = expected
                    .log_evidence
                    .expect("ok result needs finite log_evidence");
                assert!(
                    (result.log_evidence - expected_evidence).abs() <= 1e-9,
                    "{} {regime} syndrome {syndrome_mask}: expected evidence {expected_evidence}, got {}",
                    fixture.name,
                    result.log_evidence
                );
            }
            "no_path" => {
                assert!(
                    expected.logical_hat.is_none(),
                    "{} {regime} syndrome {syndrome_mask}: no-path logical_hat must be null",
                    fixture.name
                );
                assert!(
                    expected.log_evidence.is_none(),
                    "{} {regime} syndrome {syndrome_mask}: no-path log_evidence must be null",
                    fixture.name
                );
                assert!(
                    expected.terminal_log_masses.is_empty(),
                    "{} {regime} syndrome {syndrome_mask}: no-path masses must be empty",
                    fixture.name
                );
                assert!(
                    decoded.is_err(),
                    "{} {regime} syndrome {syndrome_mask}: expected no path",
                    fixture.name
                );
            }
            status => panic!(
                "{} {regime} syndrome {syndrome_mask}: unknown fixture status {status}",
                fixture.name
            ),
        }
    }

    assert!(
        expected_by_syndrome.is_empty(),
        "{} {regime}: expected results contain extra syndromes",
        fixture.name
    );
}

#[test]
fn unpruned_and_pruned_results_match_upstream_golden_fixtures() {
    let fixture_file = parse_fixtures();
    assert_eq!(fixture_file.generator, "generate_upstream_fixtures.py");

    for fixture in fixture_file.fixtures {
        assert_expected_results(
            &fixture,
            FrontierConfig {
                k: usize::MAX,
                delta: f64::INFINITY,
                score_alpha: 0.8,
                column_order: None,
                merge_indistinguishable: false,
                bp_score_iterations: 0,
            },
            &fixture.expected_unpruned,
            "unpruned",
        );
        assert_expected_results(
            &fixture,
            FrontierConfig {
                k: fixture.pruned.k,
                delta: fixture.pruned.delta,
                score_alpha: 0.8,
                column_order: None,
                merge_indistinguishable: false,
                bp_score_iterations: 0,
            },
            &fixture.expected_pruned,
            "pruned",
        );
    }
}
