// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific language governing permissions and
// limitations under the License.

use pecos_frontier::{
    FrontierConfig, FrontierDecoder, FrontierResult, FrontierStatus, MetricMode, ObsMask, SparseDem,
};
use serde::Deserialize;
use std::collections::BTreeMap;

const FIXTURES_JSON: &str = include_str!("fixtures/upstream_maxlog_fixtures.json");

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
    expected: BTreeMap<String, ExpectedRegimes>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PruningConfig {
    k: usize,
    delta: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpectedRegimes {
    wide: Vec<ExpectedResult>,
    pruned: Vec<ExpectedResult>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpectedResult {
    syndrome: u128,
    status: String,
    logical_hat: Option<u128>,
    log_evidence: Option<f64>,
    terminal_log_masses: BTreeMap<String, f64>,
    terminal_top_log_mass_gap: Option<f64>,
    engine: String,
}

fn dense_syndrome(mask: u128, num_detectors: usize) -> Vec<u8> {
    (0..num_detectors)
        .map(|bit| u8::from(mask & (1_u128 << bit) != 0))
        .collect()
}

fn mask_as_u128(mask: &ObsMask) -> u128 {
    assert!(mask.words().iter().skip(2).all(|&word| word == 0));
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
    scale: i32,
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
    let mut decoder = FrontierDecoder::from_sparse_dem(&dem, config)
        .unwrap_or_else(|error| panic!("{} scale {scale} {regime}: {error}", fixture.name));
    let expected_by_syndrome: BTreeMap<_, _> = expected_results
        .iter()
        .map(|expected| (expected.syndrome, expected))
        .collect();
    assert_eq!(expected_by_syndrome.len(), fixture.syndromes.len());

    for &syndrome_mask in &fixture.syndromes {
        let expected = expected_by_syndrome
            .get(&syndrome_mask)
            .unwrap_or_else(|| panic!("{} {regime}: missing syndrome", fixture.name));
        assert_eq!(expected.engine, "native_binary");
        let decoded = decoder.decode(&dense_syndrome(syndrome_mask, fixture.num_detectors));
        match expected.status.as_str() {
            "ok" => {
                let result = decoded.unwrap_or_else(|error| {
                    panic!(
                        "{} scale {scale} {regime} syndrome {syndrome_mask}: expected success, got {error}",
                        fixture.name
                    )
                });
                if regime == "wide" {
                    assert_eq!(result.status, FrontierStatus::Exact);
                }
                assert_eq!(
                    mask_as_u128(&result.predicted),
                    expected
                        .logical_hat
                        .expect("successful fixture needs a label")
                );
                let expected_masses: BTreeMap<u128, f64> = expected
                    .terminal_log_masses
                    .iter()
                    .map(|(label, &mass)| (label.parse().unwrap(), mass))
                    .collect();
                let actual_masses = actual_masses(&result);
                assert_eq!(actual_masses.len(), expected_masses.len());
                for (label, expected_mass) in expected_masses {
                    let actual_mass = actual_masses.get(&label).unwrap_or_else(|| {
                        panic!(
                            "{} scale {scale} {regime} syndrome {syndrome_mask}: missing label {label}",
                            fixture.name
                        )
                    });
                    assert!(
                        (actual_mass - expected_mass).abs() <= 1e-9,
                        "{} scale {scale} {regime} syndrome {syndrome_mask}, label {label}: expected {expected_mass}, got {actual_mass}",
                        fixture.name
                    );
                }
                let expected_evidence = expected.log_evidence.unwrap();
                assert!(
                    (result.log_evidence - expected_evidence).abs() <= 1e-9,
                    "{} scale {scale} {regime} syndrome {syndrome_mask}: expected evidence {expected_evidence}, got {}",
                    fixture.name,
                    result.log_evidence
                );
                let expected_gap = expected.terminal_top_log_mass_gap;
                let gap_matches = match (result.runner_up_gap, expected_gap) {
                    (Some(actual), Some(expected)) => (actual - expected).abs() <= 1e-9,
                    (None, None) => true,
                    _ => false,
                };
                assert!(
                    gap_matches,
                    "{} scale {scale} {regime} syndrome {syndrome_mask}: expected gap {expected_gap:?}, got {:?}",
                    fixture.name, result.runner_up_gap
                );
            }
            "no_path" => assert!(
                decoded.is_err(),
                "{} scale {scale} {regime} syndrome {syndrome_mask}: expected no path",
                fixture.name
            ),
            status => panic!("{}: unknown fixture status {status}", fixture.name),
        }
    }
}

#[test]
fn maxlog_matches_upstream_binary_fixtures() {
    let fixture_file: FixtureFile = serde_json::from_str(FIXTURES_JSON).unwrap();
    assert_eq!(
        fixture_file.generator,
        "generate_upstream_maxlog_fixtures.py"
    );

    for fixture in fixture_file.fixtures {
        for (scale_text, expected) in &fixture.expected {
            let scale: i32 = scale_text.parse().unwrap();
            assert_expected_results(
                &fixture,
                scale,
                FrontierConfig {
                    k: 1_000_000_000,
                    delta: 1_000_000.0,
                    metric_mode: MetricMode::MaxLogInt,
                    int_metric_scale: scale,
                    ..FrontierConfig::default()
                },
                &expected.wide,
                "wide",
            );
            assert_expected_results(
                &fixture,
                scale,
                FrontierConfig {
                    k: fixture.pruned.k,
                    delta: fixture.pruned.delta,
                    metric_mode: MetricMode::MaxLogInt,
                    int_metric_scale: scale,
                    ..FrontierConfig::default()
                },
                &expected.pruned,
                "pruned",
            );
        }
    }
}

#[test]
fn maxlog_metric_flips_the_m1_winner() {
    let fixture_file: FixtureFile = serde_json::from_str(FIXTURES_JSON).unwrap();
    let fixture = fixture_file
        .fixtures
        .iter()
        .find(|fixture| fixture.name == "maxlog_metric_flips_winner")
        .unwrap();
    let dem = SparseDem {
        mechanisms: fixture.mechanisms.clone(),
        detector_coords: BTreeMap::new(),
        num_detectors: fixture.num_detectors,
        num_observables: fixture.num_observables,
    };
    let syndrome = dense_syndrome(1, fixture.num_detectors);
    let mut float = FrontierDecoder::from_sparse_dem(&dem, FrontierConfig::default()).unwrap();
    let mut maxlog = FrontierDecoder::from_sparse_dem(
        &dem,
        FrontierConfig {
            k: 1_000_000_000,
            delta: 1_000_000.0,
            metric_mode: MetricMode::MaxLogInt,
            ..FrontierConfig::default()
        },
    )
    .unwrap();
    assert_eq!(mask_as_u128(&float.decode(&syndrome).unwrap().predicted), 0);
    assert_eq!(
        mask_as_u128(&maxlog.decode(&syndrome).unwrap().predicted),
        1
    );
}
