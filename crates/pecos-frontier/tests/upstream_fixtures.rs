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
struct FixtureFile {
    fixtures: Vec<Fixture>,
}

#[derive(Debug, Deserialize)]
struct Fixture {
    name: String,
    mechanisms: Vec<(f64, Vec<u32>, Vec<u32>)>,
    num_detectors: usize,
    num_observables: usize,
    syndromes: Vec<u128>,
    expected_unpruned: Vec<ExpectedResult>,
}

#[derive(Debug, Deserialize)]
struct ExpectedResult {
    syndrome: u128,
    status: String,
    logical_hat: Option<u128>,
    log_evidence: Option<f64>,
    terminal_log_masses: BTreeMap<String, f64>,
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

#[test]
fn unpruned_results_match_upstream_golden_fixtures() {
    let fixture_file = parse_fixtures();

    for fixture in fixture_file.fixtures {
        let dem = SparseDem {
            mechanisms: fixture.mechanisms,
            detector_coords: BTreeMap::new(),
            num_detectors: fixture.num_detectors,
            num_observables: fixture.num_observables,
        };
        let config = FrontierConfig {
            k: usize::MAX,
            delta: f64::INFINITY,
            column_order: None,
        };
        let mut decoder = FrontierDecoder::from_sparse_dem(&dem, config).unwrap_or_else(|error| {
            panic!("{}: decoder construction failed: {error}", fixture.name)
        });

        let mut expected_by_syndrome = BTreeMap::new();
        for expected in fixture.expected_unpruned {
            assert!(
                expected_by_syndrome
                    .insert(expected.syndrome, expected)
                    .is_none(),
                "{}: duplicate expected syndrome",
                fixture.name
            );
        }
        assert_eq!(
            expected_by_syndrome.len(),
            fixture.syndromes.len(),
            "{}: expected result count differs from syndrome count",
            fixture.name
        );

        for syndrome_mask in fixture.syndromes {
            let expected = expected_by_syndrome
                .remove(&syndrome_mask)
                .unwrap_or_else(|| panic!("{}: missing expected result", fixture.name));
            let syndrome = dense_syndrome(syndrome_mask, fixture.num_detectors);
            let decoded = decoder.decode(&syndrome);

            match expected.status.as_str() {
                "ok" => {
                    let result = decoded.unwrap_or_else(|error| {
                        panic!(
                            "{} syndrome {syndrome_mask}: expected success, got {error}",
                            fixture.name
                        )
                    });
                    let logical_hat = expected.logical_hat.expect("ok result needs logical_hat");
                    assert_eq!(
                        mask_as_u128(&result.predicted),
                        logical_hat,
                        "{} syndrome {syndrome_mask}: predicted label",
                        fixture.name
                    );

                    let expected_masses: BTreeMap<u128, f64> = expected
                        .terminal_log_masses
                        .into_iter()
                        .map(|(label, mass)| {
                            (label.parse().expect("logical label must fit in u128"), mass)
                        })
                        .collect();
                    let actual_masses = actual_masses(&result);
                    assert_eq!(
                        actual_masses.len(),
                        expected_masses.len(),
                        "{} syndrome {syndrome_mask}: terminal-label count",
                        fixture.name
                    );
                    for (label, expected_mass) in expected_masses {
                        let actual_mass = actual_masses.get(&label).unwrap_or_else(|| {
                            panic!(
                                "{} syndrome {syndrome_mask}: missing label {label}",
                                fixture.name
                            )
                        });
                        assert!(
                            (actual_mass - expected_mass).abs() <= 1e-9,
                            "{} syndrome {syndrome_mask}, label {label}: expected {expected_mass}, got {actual_mass}",
                            fixture.name
                        );
                    }

                    let expected_evidence = expected
                        .log_evidence
                        .expect("ok result needs finite log_evidence");
                    assert!(
                        (result.log_evidence - expected_evidence).abs() <= 1e-9,
                        "{} syndrome {syndrome_mask}: expected evidence {expected_evidence}, got {}",
                        fixture.name,
                        result.log_evidence
                    );
                }
                "no_path" => assert!(
                    decoded.is_err(),
                    "{} syndrome {syndrome_mask}: expected no path",
                    fixture.name
                ),
                status => panic!(
                    "{} syndrome {syndrome_mask}: unknown fixture status {status}",
                    fixture.name
                ),
            }
        }

        assert!(
            expected_by_syndrome.is_empty(),
            "{}: expected results contain extra syndromes",
            fixture.name
        );
    }

    // `expected_pruned` is intentionally ignored: upstream pruning includes a
    // score_alpha mixing parameter that v1 deliberately did not port, so
    // pruned-path parity is out of scope.
}
