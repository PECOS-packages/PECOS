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

//! Bitwise coverage; one compact JSON object per scenario line.
//! Every configuration runs every syndrome in its fixture's single list.
//! All four orderings occur on `ordering_duplicates`; other fixtures use deadline.
//! Merge-on cases use `degeneracy_ml_vs_mle`, `ordering_duplicates`, and
//! `defaults_identity` and `forced_duplicate` (repeated supports), plus every default case.
//! Non-empty ladders occur only on `overpruning_ladder`,
//! `parity_locked_and_untouched`, and `bp_sensitive_ladder`, with prunable bases.
//!
//! | Arm | Metric | Regime | BP iterations | Merge | Ladder |
//! |---|---|---|---|---|---|
//! | Binary | float | default | 5 | on | empty |
//! | Binary | float | unpruned | 0, 5 | off, on | empty |
//! | Binary | float | k, delta, k+delta | 0, 5 | off, on | empty, one, three, exhausted |
//!
//! | Fixture / scenario | Additional semantic coverage |
//! |---|---|
//! | `forced_duplicate`, default, 0x41 | Forced D0 shares a probabilistic mechanism, forced L1/L2 seed the label, D6 is forced-only, and p=0 on D7 is filtered. Merge-on preserves the forced layer. BP's forced residual affects dropped mass. |
//! | `bp_sensitive_ladder/k`, deadline, 0x2 | Alpha=.8; BP=0/5 both fail the base and succeed at k=16, with different logical masses, dropped mass, and transitions. The [1,1,16] ladder succeeds on rung 3 (transitions 258 versus 248). |
//!
//! BP is skipped on the unpruned path. The alpha=0 `overpruning_ladder`
//! also preserves the independent retry case. Base success, rung success,
//! all-rung failure, and untouched-detector failure are frozen.
//! N-ary and maxlog are unavailable through `BpTrellisConfig` (no runtime cell).
//! Zero-width ladder rungs are rejected: `escalation_ks[1]: TrellisConfig.k must be at least 1`,
//! pinned in `rejected_ladder_pins_invalid_configuration_message`.

use pecos_trellis::bp_trellis::{BpTrellisConfig, BpTrellisDecoder, TrellisOrdering};

use pecos_trellis::{DecoderError, SparseDem, TrellisResult, TrellisStatus};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

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
        runner_up_gap: Option<String>,
        peak_retained_states: usize,
        processed_columns: usize,
        transitions: u64,
        dropped_states: u64,
        dropped_log_mass: String,
        escalation_rungs_used: u32,
        status: String,
        bp_ran: bool,
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

fn snapshot_result(result: TrellisResult) -> SnapshotOutcome {
    SnapshotOutcome::Ok {
        predicted: result
            .predicted
            .words()
            .iter()
            .copied()
            .map(hex_word)
            .collect(),
        log_evidence: hex_word(result.log_evidence.to_bits()),
        runner_up_gap: result.runner_up_gap.map(|gap| hex_word(gap.to_bits())),
        peak_retained_states: result.peak_retained_states,
        processed_columns: result.processed_columns,
        transitions: result.transitions,
        dropped_states: result.dropped_states,
        dropped_log_mass: hex_word(result.dropped_log_mass.to_bits()),
        escalation_rungs_used: result.escalation_rungs_used,
        status: match result.status {
            TrellisStatus::Exact => "exact",
            TrellisStatus::Pruned {
                k_capped: true,
                delta_pruned: false,
            } => "pruned:k",
            TrellisStatus::Pruned {
                k_capped: false,
                delta_pruned: true,
            } => "pruned:delta",
            TrellisStatus::Pruned {
                k_capped: true,
                delta_pruned: true,
            } => "pruned:k+delta",
            TrellisStatus::Pruned {
                k_capped: false,
                delta_pruned: false,
            } => panic!("pruned without a cause"),
        }
        .into(),
        bp_ran: result.bp_seconds > 0.0,
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

fn snapshot_outcome(result: Result<TrellisResult, DecoderError>) -> SnapshotOutcome {
    result.map_or_else(
        |error| {
            assert!(
                matches!(error, DecoderError::DecodingFailed(_)),
                "snapshot hit an engine fault: {error}"
            );
            SnapshotOutcome::NoPath
        },
        snapshot_result,
    )
}

fn snapshot_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/bp_trellis/bitwise_snapshot.json")
}

#[test]
fn decode_outputs_match_bitwise_snapshot() {
    let actual = collect_snapshot();
    assert_eq!(
        actual.scenarios.len(),
        SCENARIO_COUNT,
        "scenario corpus changed"
    );
    if std::env::var("PECOS_REGEN_SNAPSHOT").as_deref() == Ok("1") {
        let lines: Vec<String> = actual
            .scenarios
            .iter()
            .map(|scenario| serde_json::to_string(scenario).expect("scenario must serialize"))
            .collect();
        let json = format!("{{\"scenarios\":[\n{}\n]}}\n", lines.join(",\n"));
        std::fs::write(snapshot_path(), json).expect("snapshot must be writable");
        return;
    }
    let json = std::fs::read_to_string(snapshot_path())
        .expect("bitwise snapshot must exist; run with PECOS_REGEN_SNAPSHOT=1");
    let expected: SnapshotFile = serde_json::from_str(&json).expect("snapshot must parse");
    assert_eq!(
        actual.scenarios.len(),
        expected.scenarios.len(),
        "scenario count"
    );
    for (actual, expected) in actual.scenarios.iter().zip(&expected.scenarios) {
        assert_eq!(actual.name, expected.name, "scenario order");
        let actual_fields = serde_json::to_value(&actual.outcome).unwrap();
        let expected_fields = serde_json::to_value(&expected.outcome).unwrap();
        for (field, value) in actual_fields.as_object().unwrap() {
            assert_eq!(
                Some(value),
                expected_fields.get(field),
                "scenario {} field {field}",
                actual.name
            );
        }
        assert_eq!(actual.outcome, expected.outcome, "scenario {}", actual.name);
    }
}
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
    score_alpha: f64,
}

fn configurations(fixture: &Fixture) -> Vec<(String, BpTrellisConfig)> {
    let columns = fixture.mechanisms.len();
    let alpha = fixture.score_alpha;
    let has_duplicates =
        fixture
            .mechanisms
            .iter()
            .enumerate()
            .any(|(index, (_, detectors, observables))| {
                fixture.mechanisms[..index]
                    .iter()
                    .any(|(_, other_detectors, other_observables)| {
                        detectors == other_detectors && observables == other_observables
                    })
            });
    let has_no_path = matches!(
        fixture.name.as_str(),
        "overpruning_ladder" | "parity_locked_and_untouched" | "bp_sensitive_ladder"
    );
    let mut configs = vec![(
        "default/order=deadline/bp=5/merge=true/ladder=empty".into(),
        BpTrellisConfig::default(),
    )];
    for (order, ordering) in [
        ("deadline", TrellisOrdering::Deadline),
        ("backward", TrellisOrdering::BackwardDeadline),
        ("input", TrellisOrdering::TimeOrder),
        (
            "reverse",
            TrellisOrdering::Explicit((0..columns).rev().collect()),
        ),
    ] {
        if order != "deadline" && fixture.name != "ordering_duplicates" {
            continue;
        }
        for bp in [0, 5] {
            for merge in [false, true] {
                if merge && !has_duplicates {
                    continue;
                }
                for (regime, k, delta) in [
                    ("unpruned", usize::MAX, f64::INFINITY),
                    ("k", 1, 100.0),
                    ("delta", usize::MAX, 0.01),
                    ("k+delta", 3, 0.5),
                ] {
                    for (ladder, escalation_ks) in [
                        ("empty", vec![]),
                        ("one", vec![16]),
                        ("three", vec![1, 1, 16]),
                        ("exhausted", vec![1, 1, 1]),
                    ] {
                        if ladder != "empty" && (!has_no_path || regime == "unpruned") {
                            continue;
                        }
                        configs.push((
                            format!("{regime}/order={order}/bp={bp}/merge={merge}/ladder={ladder}"),
                            BpTrellisConfig {
                                k,
                                delta,
                                score_alpha: alpha,
                                bp_score_iterations: bp,
                                merge_indistinguishable: merge,
                                ordering: ordering.clone(),
                                escalation_ks,
                            },
                        ));
                    }
                }
            }
        }
    }
    configs
}

fn collect_snapshot() -> SnapshotFile {
    let fixtures: FixtureFile =
        serde_json::from_str(include_str!("fixtures/bp_trellis/models.json"))
            .expect("fixtures must parse");
    let mut scenarios = Vec::new();
    for fixture in fixtures.fixtures {
        let configs = configurations(&fixture);
        let dem = sparse_dem(
            fixture.mechanisms,
            fixture.num_detectors,
            fixture.num_observables,
        );
        for (config_name, config) in configs {
            let mut decoder = BpTrellisDecoder::from_sparse_dem(&dem, config).unwrap();
            for &mask in &fixture.syndromes {
                scenarios.push(ScenarioSnapshot {
                    name: format!(
                        "binary/float/{}/{config_name}/syndrome=0x{mask:x}",
                        fixture.name
                    ),
                    outcome: snapshot_outcome(
                        decoder.decode(&dense_syndrome(mask, fixture.num_detectors)),
                    ),
                });
            }
        }
    }
    SnapshotFile { scenarios }
}

#[test]
fn rejected_ladder_pins_invalid_configuration_message() {
    let dem = sparse_dem(vec![], 0, 0);
    let error = BpTrellisDecoder::from_sparse_dem(
        &dem,
        BpTrellisConfig {
            escalation_ks: vec![16, 0],
            ..BpTrellisConfig::default()
        },
    )
    .unwrap_err();
    assert!(
        matches!(&error, DecoderError::InvalidConfiguration(message) if message == "escalation_ks[1]: TrellisConfig.k must be at least 1"),
        "{error}"
    );
}

const SCENARIO_COUNT: usize = 658;
