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

use pecos_frontier::{
    FrontierCommittee, FrontierConfig, FrontierDecoder, FrontierResult, SparseDem,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

const UPSTREAM_FIXTURES_JSON: &str = include_str!("fixtures/upstream_fixtures.json");
const ORDER_FIXTURES_JSON: &str = include_str!("fixtures/upstream_order_fixtures.json");

#[derive(Debug, Deserialize)]
struct FixtureFile<T> {
    fixtures: Vec<T>,
}

#[derive(Debug, Deserialize)]
struct Fixture {
    name: String,
    mechanisms: Vec<(f64, Vec<u32>, Vec<u32>)>,
    num_detectors: usize,
    num_observables: usize,
    syndromes: Vec<u128>,
    pruned: PruningConfig,
}

#[derive(Debug, Deserialize)]
struct OrderFixture {
    name: String,
    mechanisms: Vec<(f64, Vec<u32>, Vec<u32>)>,
    num_detectors: usize,
    num_observables: usize,
    syndromes: Vec<u128>,
    pruned: PruningConfig,
    forward_ordering: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct PruningConfig {
    k: usize,
    delta: f64,
}

#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
struct SnapshotFile {
    scenarios: Vec<ScenarioSnapshot>,
}

#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
struct ScenarioSnapshot {
    name: String,
    #[serde(flatten)]
    outcome: SnapshotOutcome,
}

#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
enum SnapshotOutcome {
    Ok {
        predicted: Vec<String>,
        log_evidence: String,
        logical_masses: Vec<LogicalMassSnapshot>,
    },
    NoPath,
}

#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
struct LogicalMassSnapshot {
    label: Vec<String>,
    log_mass: String,
}

fn sparse_dem(
    mechanisms: Vec<(f64, Vec<u32>, Vec<u32>)>,
    num_detectors: usize,
    num_observables: usize,
) -> SparseDem {
    SparseDem {
        mechanisms,
        detector_coords: BTreeMap::new(),
        num_detectors,
        num_observables,
    }
}

fn dense_syndrome(mask: u128, num_detectors: usize) -> Vec<u8> {
    (0..num_detectors)
        .map(|bit| u8::from(mask & (1_u128 << bit) != 0))
        .collect()
}

fn hex_word(word: u64) -> String {
    format!("0x{word:016x}")
}

fn snapshot_result(result: FrontierResult) -> SnapshotOutcome {
    SnapshotOutcome::Ok {
        predicted: result
            .predicted
            .words()
            .iter()
            .copied()
            .map(hex_word)
            .collect(),
        log_evidence: hex_word(result.log_evidence.to_bits()),
        logical_masses: result
            .logical_masses
            .into_iter()
            .map(|mass| LogicalMassSnapshot {
                label: mass.logical.words().iter().copied().map(hex_word).collect(),
                log_mass: hex_word(mass.log_mass.to_bits()),
            })
            .collect(),
    }
}

fn collect_snapshot() -> SnapshotFile {
    let fixtures: FixtureFile<Fixture> =
        serde_json::from_str(UPSTREAM_FIXTURES_JSON).expect("upstream fixtures must parse");
    let order_fixtures: FixtureFile<OrderFixture> =
        serde_json::from_str(ORDER_FIXTURES_JSON).expect("order fixtures must parse");
    let mut scenarios = Vec::new();

    for fixture in fixtures.fixtures {
        let dem = sparse_dem(
            fixture.mechanisms,
            fixture.num_detectors,
            fixture.num_observables,
        );
        for (regime, config) in [
            (
                "unpruned",
                FrontierConfig {
                    k: usize::MAX,
                    delta: f64::INFINITY,
                    score_alpha: 0.8,
                    column_order: None,
                },
            ),
            (
                "pruned",
                FrontierConfig {
                    k: fixture.pruned.k,
                    delta: fixture.pruned.delta,
                    score_alpha: 0.8,
                    column_order: None,
                },
            ),
        ] {
            let mut decoder = FrontierDecoder::from_sparse_dem(&dem, config)
                .expect("fixture decoder construction must succeed");
            for &syndrome_mask in &fixture.syndromes {
                let syndrome = dense_syndrome(syndrome_mask, fixture.num_detectors);
                let outcome = decoder
                    .decode(&syndrome)
                    .map_or(SnapshotOutcome::NoPath, snapshot_result);
                scenarios.push(ScenarioSnapshot {
                    name: format!(
                        "upstream/{}/{regime}/syndrome=0x{syndrome_mask:x}",
                        fixture.name
                    ),
                    outcome,
                });
            }
        }
    }

    for fixture in order_fixtures.fixtures {
        let dem = sparse_dem(
            fixture.mechanisms,
            fixture.num_detectors,
            fixture.num_observables,
        );
        let mut committee = FrontierCommittee::from_sparse_dem(
            &dem,
            FrontierConfig {
                k: fixture.pruned.k,
                delta: fixture.pruned.delta,
                score_alpha: 0.8,
                column_order: Some(fixture.forward_ordering),
            },
        )
        .expect("fixture committee construction must succeed");
        for syndrome_mask in fixture.syndromes {
            let syndrome = dense_syndrome(syndrome_mask, fixture.num_detectors);
            let outcome = committee
                .decode(&syndrome)
                .map_or(SnapshotOutcome::NoPath, |result| {
                    snapshot_result(result.selected)
                });
            scenarios.push(ScenarioSnapshot {
                name: format!(
                    "order/{}/committee/syndrome=0x{syndrome_mask:x}",
                    fixture.name
                ),
                outcome,
            });
        }
    }

    SnapshotFile { scenarios }
}

fn snapshot_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/bitwise_snapshot.json")
}

#[test]
fn decode_outputs_match_bitwise_snapshot() {
    let snapshot_json = std::fs::read_to_string(snapshot_path())
        .expect("bitwise snapshot must exist; run the ignored regeneration test");
    let expected: SnapshotFile =
        serde_json::from_str(&snapshot_json).expect("bitwise snapshot must parse");
    let actual = collect_snapshot();

    assert_eq!(actual.scenarios.len(), 128, "scenario corpus changed");
    assert_eq!(actual, expected);
}

#[test]
#[ignore = "manually regenerates the committed bitwise snapshot"]
fn regenerate_bitwise_snapshot() {
    let path = snapshot_path();
    assert!(
        !path.exists() || std::env::var("PECOS_REGEN_SNAPSHOT").as_deref() == Ok("1"),
        "refusing to overwrite {}; set PECOS_REGEN_SNAPSHOT=1 to allow it",
        path.display()
    );

    let snapshot = collect_snapshot();
    assert_eq!(snapshot.scenarios.len(), 128, "scenario corpus changed");
    let mut json = serde_json::to_string_pretty(&snapshot).expect("snapshot must serialize");
    json.push('\n');
    std::fs::write(&path, json).expect("snapshot must be writable");
}
