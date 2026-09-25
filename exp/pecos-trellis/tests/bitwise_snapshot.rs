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

//! Bitwise coverage: input order throughout; explicit reverse and rotation orders for BP=0
//! on every fixture and BP=5 on `degeneracy_ml_vs_mle` only.
//! Every configuration runs every syndrome in its fixture's single list.
//! Scenarios are serialized as one compact JSON object per line.
//!
//! | Arm | Metric | Regime | BP iterations | Merge |
//! |---|---|---|---|---|
//! | Binary | float | unpruned, k, delta, k+delta | 0, 5 (unpruned BP=5 pins `bp_ran=false`) | off, on |
//! | Binary | maxlog | k, delta, k+delta | 0, 5 | off |
//! | N-ary | float | unpruned | 0, 5 (BP=5 pins `bp_ran=false`) | off |
//! | N-ary | float/maxlog | k, delta, k+delta | 0 | off |
//! | Either | maxlog | unpruned | any | rejected: delta must be finite |
//! | Binary | maxlog | any | any | on rejected: merging sums coset mass |
//! | N-ary | either | prunable | 5 | rejected: BP requires binary model |
//! | N-ary | either | any | any | on rejected: merging requires binary mechanisms |
//!
//! | Fixture / scenario | Additional semantic coverage |
//! |---|---|
//! | `forced_duplicate`, 0x41 and 0x43 | Forced D0 shares a probabilistic toggle; forced L1/L2 seed predicted labels; merge-on keeps the forced duplicate separate. D6 is forced-only, and p=0 on D7 is filtered. Syndromes 0 and 0xc1 exercise the precheck. |
//! | `forced_factors`, 0x4 and 0x5 | N-ary normalization removes p=0 outcomes and seeds forced syndrome/logical masks; 0 and 0xc exercise prechecks. |
//! | `wide_detector_70` | D64/D68 suffix and BP indices cross a word boundary, D69 is forced-only, D67 is zero-only. |
//! | `three_route_collision/float/unpruned`, rotate, 0x0 | The [.17, .31, .52] factor has three identical empty toggles, so three log masses reduce into one state. In input order the same collision occurs for each preceding logical label. Reversing outcome iteration changes the rotated scenario's evidence bits. |
//! | `three_route_collision`, k+delta, input, 0x0 | In both metrics, the five [.6, .2, .1, .07, .03] labels exceed k=3; the second and third lie outside delta=.5. Thus both flags are set. |
//!
//! Merge-on is restricted to fixtures with repeated detector/observable supports:
//! `degeneracy_ml_vs_mle`, `random_seed11`, and `forced_duplicate`.
//! Reverse and rotation run at BP=0, plus BP=5 on `degeneracy_ml_vs_mle`.
//! The focused fixtures cover the full matrix above. Imported `random_seed11`
//! runs float/unpruned/BP=0; `rep_chain_hyperedge` and `random_seed23` run
//! k-pruned/BP=0 under both metrics. Every retained configuration uses its
//! fixture's complete syndrome list, including all no-path syndromes.
//!
//! Exact rejection messages are pinned in `rejected_cells_pin_invalid_configuration_messages`.
//! Fixtures include duplicate mechanisms, wide labels, parity-locked empty columns,
//! untouched detectors, and prefixes lost to pruning. `NoPath` records transitions
//! and `bp_ran`; engine faults fail the test. No clock values enter the oracle.

use pecos_trellis::factor::{Factor, FactorModel, Outcome};
use pecos_trellis::{MetricMode, TrellisConfig, TrellisDecodeAttempt, TrellisDecoder};

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
    NoPath {
        transitions: u64,
        bp_ran: bool,
    },
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

fn snapshot_outcome(attempt: TrellisDecodeAttempt) -> SnapshotOutcome {
    match attempt {
        TrellisDecodeAttempt::Success(result) => snapshot_result(result),
        TrellisDecodeAttempt::NoPath {
            dropped_states: _,
            error,
            transitions,
            bp_seconds,
        } => {
            assert!(
                matches!(error, DecoderError::DecodingFailed(_)),
                "unexpected no-path error: {error}"
            );
            SnapshotOutcome::NoPath {
                transitions,
                bp_ran: bp_seconds > 0.0,
            }
        }
        TrellisDecodeAttempt::Error(error) => panic!("snapshot hit an engine fault: {error}"),
    }
}

fn snapshot_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/bitwise_snapshot.json")
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
struct Fixtures {
    binary: Vec<BinaryFixture>,
    nary: Vec<NaryFixture>,
}

#[derive(Debug, Deserialize)]
struct BinaryFixture {
    name: String,
    mechanisms: Vec<(f64, Vec<u32>, Vec<u32>)>,
    num_detectors: usize,
    num_observables: usize,
    syndromes: Vec<u128>,
    score_alpha: f64,
}

type FactorOutcome = (f64, Vec<u32>, Vec<u32>);

#[derive(Debug, Deserialize)]
struct NaryFixture {
    name: String,
    factors: Vec<Vec<FactorOutcome>>,
    num_detectors: usize,
    num_observables: usize,
    syndromes: Vec<u128>,
    score_alpha: f64,
}

impl NaryFixture {
    fn model(&self) -> FactorModel {
        FactorModel::new(
            self.factors
                .iter()
                .map(|outcomes| Factor {
                    outcomes: outcomes
                        .iter()
                        .map(|(probability, detectors, observables)| Outcome {
                            probability: *probability,
                            detectors: detectors.clone(),
                            observables: observables.clone(),
                        })
                        .collect(),
                })
                .collect(),
            self.num_detectors,
            self.num_observables,
        )
        .unwrap()
    }
}

fn fixtures() -> Fixtures {
    serde_json::from_str(include_str!("fixtures/models.json")).expect("fixtures must parse")
}

fn has_indistinguishable_mechanisms(dem: &SparseDem) -> bool {
    dem.mechanisms
        .iter()
        .enumerate()
        .any(|(index, (probability, detectors, observables))| {
            *probability > 0.0
                && dem.mechanisms[..index].iter().any(
                    |(other_probability, other_detectors, other_observables)| {
                        *other_probability > 0.0
                            && detectors == other_detectors
                            && observables == other_observables
                    },
                )
        })
}

fn configurations(
    nary: bool,
    columns: usize,
    alpha: f64,
    name: &str,
    has_duplicates: bool,
) -> Vec<(String, TrellisConfig)> {
    let mut configs = Vec::new();
    for (metric, metric_mode) in [
        ("float", MetricMode::LogSumExpFloat),
        ("maxlog", MetricMode::MaxLogInt),
    ] {
        for (regime, k, delta) in [
            ("unpruned", usize::MAX, f64::INFINITY),
            ("k", 1, 100.0),
            ("delta", usize::MAX, 0.01),
            ("k+delta", 3, 0.5),
        ] {
            if metric_mode == MetricMode::MaxLogInt && delta.is_infinite() {
                continue;
            }
            for bp in [0, 5] {
                // Imported broad models supplement the focused full-matrix fixtures.
                // Keep the float reduction oracle and the pruning/no-path corpora
                // without repeating every configuration on each large syndrome list.
                if name == "random_seed11"
                    && (metric_mode != MetricMode::LogSumExpFloat
                        || regime != "unpruned"
                        || bp != 0)
                {
                    continue;
                }
                if matches!(name, "rep_chain_hyperedge" | "random_seed23")
                    && (regime != "k" || bp != 0)
                {
                    continue;
                }
                if nary && bp > 0 && regime != "unpruned" {
                    continue;
                }
                for merge in [false, true] {
                    if merge && (!has_duplicates || nary || metric_mode == MetricMode::MaxLogInt) {
                        continue;
                    }
                    for (order, column_order) in [
                        ("input", None),
                        ("reverse", Some((0..columns).rev().collect())),
                        (
                            "rotate",
                            Some((1..columns).chain(0..columns.min(1)).collect()),
                        ),
                    ] {
                        if order != "input" && bp == 5 && name != "degeneracy_ml_vs_mle" {
                            continue;
                        }
                        configs.push((
                            format!("{metric}/{regime}/bp={bp}/merge={merge}/order={order}"),
                            TrellisConfig {
                                k,
                                delta,
                                score_alpha: alpha,
                                bp_score_iterations: bp,
                                merge_indistinguishable: merge,
                                column_order,
                                metric_mode,
                                int_metric_scale: 1024,
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
    let fixtures = fixtures();
    let mut scenarios = Vec::new();
    for fixture in fixtures.binary {
        let dem = sparse_dem(
            fixture.mechanisms,
            fixture.num_detectors,
            fixture.num_observables,
        );
        for (config_name, config) in configurations(
            false,
            dem.mechanisms.len(),
            fixture.score_alpha,
            &fixture.name,
            has_indistinguishable_mechanisms(&dem),
        ) {
            let mut decoder = TrellisDecoder::from_sparse_dem(&dem, config).unwrap();
            for &mask in &fixture.syndromes {
                scenarios.push(ScenarioSnapshot {
                    name: format!("binary/{}/{config_name}/syndrome=0x{mask:x}", fixture.name),
                    outcome: snapshot_outcome(
                        decoder.decode_attempt(&dense_syndrome(mask, fixture.num_detectors)),
                    ),
                });
            }
        }
    }
    for fixture in fixtures.nary {
        let model = fixture.model();
        for (config_name, config) in configurations(
            true,
            fixture.factors.len(),
            fixture.score_alpha,
            &fixture.name,
            false,
        ) {
            let mut decoder = TrellisDecoder::from_factor_model(&model, config).unwrap();
            for &mask in &fixture.syndromes {
                scenarios.push(ScenarioSnapshot {
                    name: format!("nary/{}/{config_name}/syndrome=0x{mask:x}", fixture.name),
                    outcome: snapshot_outcome(
                        decoder.decode_attempt(&dense_syndrome(mask, fixture.num_detectors)),
                    ),
                });
            }
        }
    }
    SnapshotFile { scenarios }
}

#[test]
fn rejected_cells_pin_invalid_configuration_messages() {
    let fixtures = fixtures();
    let binary = &fixtures.binary[0];
    let dem = sparse_dem(
        binary.mechanisms.clone(),
        binary.num_detectors,
        binary.num_observables,
    );
    let model = fixtures.nary[0].model();
    for nary in [false, true] {
        for (config, message) in [
            (
                TrellisConfig {
                    metric_mode: MetricMode::MaxLogInt,
                    delta: f64::INFINITY,
                    ..TrellisConfig::default()
                },
                "delta must be finite under maxlog_int; infinite delta would quantize to zero and prune to score-ties",
            ),
            (
                TrellisConfig {
                    metric_mode: MetricMode::MaxLogInt,
                    merge_indistinguishable: true,
                    ..TrellisConfig::default()
                },
                "indistinguishable-mechanism merging sums coset mass and is incompatible with the max-log route metric",
            ),
        ] {
            let message = if nary && config.merge_indistinguishable {
                "indistinguishable-mechanism merging is defined for binary mechanisms only"
            } else {
                message
            };
            let error = if nary {
                TrellisDecoder::from_factor_model(&model, config)
            } else {
                TrellisDecoder::from_sparse_dem(&dem, config)
            }
            .unwrap_err();
            assert!(
                matches!(&error, DecoderError::InvalidConfiguration(actual) if actual == message),
                "{error}"
            );
        }
    }
    for metric_mode in [MetricMode::LogSumExpFloat, MetricMode::MaxLogInt] {
        let error = TrellisDecoder::from_factor_model(
            &model,
            TrellisConfig {
                metric_mode,
                bp_score_iterations: 5,
                ..TrellisConfig::default()
            },
        )
        .unwrap_err();
        assert!(
            matches!(&error, DecoderError::InvalidConfiguration(message) if message == "BP-guided pruning requires a binary model"),
            "{error}"
        );
    }
    let error = TrellisDecoder::from_factor_model(
        &model,
        TrellisConfig {
            merge_indistinguishable: true,
            ..TrellisConfig::default()
        },
    )
    .unwrap_err();
    assert!(
        matches!(&error, DecoderError::InvalidConfiguration(message) if message == "indistinguishable-mechanism merging is defined for binary mechanisms only"),
        "{error}"
    );
}

const SCENARIO_COUNT: usize = 1242;
