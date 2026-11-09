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

//! Parity against frozen outputs of `relay-bp` 0.2.2 on the gross-code DEM.
//!
//! Fixture headers are the sole source of each decoder configuration. For
//! c1, c2, c5, c6, c7 and c8, flags and total iterations agree on at least
//! 98/100 shots; jointly converged counts are within two of the smaller
//! convergence count, and at least 19/20 jointly converged corrections match.
//! Both successes and failures must occur in each such configuration.
//! For c3, independent random streams give convergence counts within five;
//! native shot seed equals shot index and its 76/100 count pins draw order.
//! For c4, native must converge wherever the oracle did and have no more than
//! five additional logical errors. Every shot checks the expected leg count.
//!
//! Known non-bitwise differences: floating-point summation order; hard
//! decisions at exactly zero (external `<= 0`, native `< 0`); saturation
//! (external `f64::MAX`, native 30); all-legs-failed fallback (external leg
//! zero, native last leg); and check-message sign at -0.0 (external uses
//! `is_sign_negative`, native counts -0.0 as positive).
//!
//! c7's external failed-shot correction is stale from a previous decode when
//! `pre_iter = 0`. Compare flags, iterations and jointly converged corrections
//! only, with no logical-error assertion for that configuration.

use pecos_bp::{Alpha, BpGraph, RelayBp, RelayConfig, RelayOutcome, Schedule, StopRule};
use pecos_decoder_core::dem::DemCheckMatrix;
use std::collections::BTreeMap;

/// One fixture shot: `(detector bits, true observable flips)`.
type Shot = (Vec<u8>, Vec<u8>);

const SHOTS: usize = 100;
/// Per-shot agreement floor for the deterministic comparisons.
const MIN_AGREEMENT: usize = 98;

fn load_fixture() -> (DemCheckMatrix, Vec<Shot>) {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/");
    let dem = std::fs::read_to_string(format!("{dir}gross_144_12_12.dem")).unwrap();
    let dcm = DemCheckMatrix::from_dem_str(&dem).unwrap();
    let shots = std::fs::read_to_string(format!("{dir}gross_144_12_12_shots.txt")).unwrap();
    let parse = |s: &str| s.bytes().map(|b| b - b'0').collect::<Vec<u8>>();
    let shots = shots
        .lines()
        .map(|line| {
            let (d, o) = line.split_once(' ').unwrap();
            (parse(d), parse(o))
        })
        .collect::<Vec<_>>();
    assert_eq!(dcm.num_detectors, 1008);
    assert_eq!(dcm.num_mechanisms, 8785);
    assert_eq!(dcm.num_observables, 12);
    assert_eq!(shots.len(), SHOTS);
    (dcm, shots)
}

fn observables(dcm: &DemCheckMatrix, correction: &[u8]) -> Vec<u8> {
    (0..dcm.num_observables)
        .map(|o| {
            (0..dcm.num_mechanisms).fold(0u8, |acc, v| {
                acc ^ (dcm.observable_matrix[[o, v]] & correction[v])
            })
        })
        .collect()
}

/// Deterministic gamma vector in `[-0.24, 0.66)` (no RNG dependency).
fn shared_gamma_vector(n: usize, mut state: u64) -> Vec<f64> {
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = f64::from(u32::try_from(state >> 32).unwrap()) / 4_294_967_296.0;
            -0.24 + 0.90 * u
        })
        .collect()
}

struct Header<'a>(BTreeMap<&'a str, &'a str>);

impl<'a> Header<'a> {
    fn get(&self, key: &str) -> &'a str {
        self.0
            .get(key)
            .unwrap_or_else(|| panic!("missing fixture key {key}"))
    }

    fn number<T: std::str::FromStr>(&self, key: &str) -> T {
        self.get(key)
            .parse()
            .unwrap_or_else(|_| panic!("malformed {key}={}", self.get(key)))
    }

    fn some(&self, key: &str) -> f64 {
        let value = self.get(key);
        value
            .strip_prefix("Some(")
            .and_then(|s| s.strip_suffix(')'))
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| panic!("malformed {key}={value}"))
    }

    fn config(&self) -> RelayConfig {
        let scaling_factor = self.number::<f64>("alpha_iteration_scaling_factor");
        // Required even where informational or unused by the external builder.
        self.get("seed");
        self.get("max_iter");
        match self.get("decoder") {
            "MinSumBpBuilder" => {
                assert_eq!(self.get("alpha"), "None");
                assert_eq!(self.get("gamma0"), "None");
                assert_eq!(self.get("seed"), "not applicable (plain min-sum)");
                assert_eq!(
                    self.get("stopping_criterion"),
                    "not applicable (plain min-sum)"
                );
                RelayConfig {
                    schedule: Schedule::Flooding,
                    pre_iterations: self.number("max_iter"),
                    num_legs: 0,
                    gamma0: 0.0,
                    alpha: Alpha::Constant(1.0),
                    stop_rule: StopRule::Converged(1),
                    ..RelayConfig::default()
                }
            }
            "RelayBpBuilder" => {
                let alpha = self.some("alpha");
                let stopping = self.get("stopping_criterion");
                let stop_rule = match stopping {
                    "PreIter" => StopRule::PreIter,
                    "All" => StopRule::Converged(usize::MAX),
                    other => StopRule::Converged(
                        other
                            .strip_prefix("NConv { stop_after: ")
                            .and_then(|s| s.strip_suffix(" }"))
                            .and_then(|s| s.parse().ok())
                            .unwrap_or_else(|| panic!("malformed stopping_criterion={other}")),
                    ),
                };
                let interval = self.get("gamma_dist_interval");
                let (lo, hi) = interval
                    .strip_prefix('(')
                    .and_then(|s| s.strip_suffix(')'))
                    .and_then(|s| s.split_once(", "))
                    .unwrap_or_else(|| panic!("malformed gamma_dist_interval={interval}"));
                let parse_endpoint = |s: &str| {
                    s.parse::<f64>()
                        .unwrap_or_else(|_| panic!("malformed gamma_dist_interval={interval}"))
                };
                let explicit = self.get("explicit_gammas").split(" (").next().unwrap();
                let explicit_gammas = match explicit {
                    "None" => None,
                    "[shared_gamma_vector(8785, 0x00C0_FFEE)]" => {
                        Some(vec![shared_gamma_vector(8785, 0x00C0_FFEE)])
                    }
                    "[shared_gamma_vector(8785, 0x00C0_FFEE), shared_gamma_vector(8785, 0x0BAD_F00D)]" => {
                        Some(vec![
                            shared_gamma_vector(8785, 0x00C0_FFEE),
                            shared_gamma_vector(8785, 0x0BAD_F00D),
                        ])
                    }
                    other => panic!("malformed explicit_gammas={other}"),
                };
                if let Some(rows) = &explicit_gammas {
                    for row in rows {
                        assert!(row.iter().any(|&g| g < 0.0) && row.iter().any(|&g| g > 0.0));
                    }
                }
                RelayConfig {
                    schedule: Schedule::Flooding,
                    alpha: if alpha == 0.0 {
                        Alpha::Dynamic { scaling_factor }
                    } else {
                        Alpha::Constant(alpha)
                    },
                    gamma0: self.some("gamma0"),
                    pre_iterations: self.number("pre_iter"),
                    num_legs: self.number("num_sets"),
                    leg_iterations: self.number("set_max_iter"),
                    gamma_range: (parse_endpoint(lo), parse_endpoint(hi)),
                    stop_rule,
                    explicit_gammas,
                }
            }
            other => panic!("unknown decoder={other}"),
        }
    }
}

struct OracleShot {
    converged: bool,
    iterations: usize,
    correction: Vec<u8>,
}

fn parse_oracle(data: &str, mechanisms: usize) -> (RelayConfig, Vec<OracleShot>) {
    let mut header = Header(BTreeMap::new());
    let mut shots = Vec::new();
    for line in data.lines() {
        if let Some(comment) = line.strip_prefix("# ") {
            if let Some((key, value)) = comment.split_once('=') {
                assert!(header.0.insert(key, value).is_none(), "duplicate key {key}");
            } else {
                assert_eq!(
                    comment,
                    "relay-bp 0.2.2 external oracle, generated by crates/pecos-bp/tests/relay_oracle_dump.rs"
                );
            }
            continue;
        }
        let fields: Vec<_> = line.split_whitespace().collect();
        assert_eq!(fields.len(), 3, "malformed shot: {line}");
        let converged = match fields[0] {
            "0" => false,
            "1" => true,
            _ => panic!("malformed convergence flag: {line}"),
        };
        let iterations = fields[1]
            .parse()
            .unwrap_or_else(|_| panic!("malformed iterations: {line}"));
        let mut correction = vec![0; mechanisms];
        if fields[2] != "-" {
            let mut previous = None;
            for index in fields[2].split(',') {
                let index: usize = index
                    .parse()
                    .unwrap_or_else(|_| panic!("malformed correction: {line}"));
                assert!(
                    index < mechanisms && previous.is_none_or(|p| p < index),
                    "invalid correction index {index}"
                );
                correction[index] = 1;
                previous = Some(index);
            }
        }
        shots.push(OracleShot {
            converged,
            iterations,
            correction,
        });
    }
    assert_eq!(shots.len(), SHOTS);
    (header.config(), shots)
}

fn assert_legs(config: &RelayConfig, outcome: &RelayOutcome) {
    let expected = match config.stop_rule {
        StopRule::PreIter if outcome.legs[0].converged => 1,
        StopRule::PreIter | StopRule::Converged(usize::MAX) => config.num_legs + 1,
        StopRule::Converged(target) => outcome
            .legs
            .iter()
            .enumerate()
            .filter(|(_, leg)| leg.converged)
            .nth(target - 1)
            .map_or(config.num_legs + 1, |(index, _)| index + 1),
    };
    assert_eq!(outcome.legs.len(), expected);
    if config.pre_iterations == 0 {
        assert_eq!(outcome.legs[0].iterations, 0);
        assert!(!outcome.legs[0].converged);
        assert_eq!(outcome.legs[0].weight, None);
        assert_eq!(outcome.legs[0].correction, None);
    }
}

#[derive(Default)]
struct Tally {
    native_converged: usize,
    external_converged: usize,
    native_errors: usize,
    external_errors: usize,
    flag_agreement: usize,
    iteration_agreement: usize,
    both_converged: usize,
    same_correction: usize,
    external_max_iterations: usize,
}

fn compare(name: &str, dcm: &DemCheckMatrix, shots: &[Shot]) -> Tally {
    let path = format!(
        "{}/tests/data/relay_oracle/{name}.txt",
        env!("CARGO_MANIFEST_DIR")
    );
    let data = std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("{path}: {error}"));
    let (config, oracle) = parse_oracle(&data, dcm.num_mechanisms);
    let mut native = RelayBp::new(BpGraph::from_dcm(dcm), config.clone()).unwrap();
    let mut tally = Tally::default();
    let compare_logicals = name != "c7_pre0_explicit";
    for (shot, ((syndrome, truth), external)) in shots.iter().zip(&oracle).enumerate() {
        let outcome = native
            .decode(syndrome, u64::try_from(shot).unwrap())
            .unwrap();
        assert_legs(&config, &outcome);
        tally.native_converged += usize::from(outcome.converged);
        tally.external_converged += usize::from(external.converged);
        if compare_logicals {
            tally.native_errors += usize::from(observables(dcm, &outcome.correction) != *truth);
            tally.external_errors += usize::from(observables(dcm, &external.correction) != *truth);
        }
        tally.flag_agreement += usize::from(outcome.converged == external.converged);
        tally.iteration_agreement += usize::from(outcome.total_iterations == external.iterations);
        if outcome.converged && external.converged {
            tally.both_converged += 1;
            tally.same_correction += usize::from(outcome.correction == external.correction);
        }
        tally.external_max_iterations = tally.external_max_iterations.max(external.iterations);
        if name == "c4_operating_point" {
            assert!(
                !external.converged || outcome.converged,
                "c4 lost convergence on shot {shot}"
            );
        }
    }
    if config.num_legs > 0 {
        assert!(tally.external_max_iterations > config.pre_iterations);
    } else {
        assert!(tally.external_max_iterations <= config.pre_iterations);
    }
    let logicals = if compare_logicals {
        format!(
            "logical errors native {}/{SHOTS}, external {}/{SHOTS}",
            tally.native_errors, tally.external_errors
        )
    } else {
        "logical errors not compared (stale external failed-shot corrections)".into()
    };
    println!(
        "{name}: converged native {}/{SHOTS}, external {}/{SHOTS}; flags {}/{SHOTS}; iterations {}/{SHOTS}; corrections {}/{} jointly converged; {logicals}",
        tally.native_converged,
        tally.external_converged,
        tally.flag_agreement,
        tally.iteration_agreement,
        tally.same_correction,
        tally.both_converged
    );
    tally
}

#[test]
fn frozen_oracle_per_shot_contracts() {
    let (dcm, shots) = load_fixture();
    for name in [
        "c1_leg0_pre8",
        "c1_leg0_pre16",
        "c2_explicit_one_row",
        "c2_explicit_two_rows",
        "c5_dynamic_alpha_pre8",
        "c5_dynamic_alpha_pre16",
        "c6_preiter_run_all",
        "c7_pre0_explicit",
        "c8_min_sum_iter8",
        "c8_min_sum_iter16",
        "c9_preiter_pre8_relay",
        "c10_dynamic_alpha_relay",
    ] {
        let tally = compare(name, &dcm, &shots);
        assert!(
            (1..SHOTS).contains(&tally.native_converged),
            "{name}: native degenerate"
        );
        assert!(
            (1..SHOTS).contains(&tally.external_converged),
            "{name}: external degenerate"
        );
        assert!(
            tally.flag_agreement >= MIN_AGREEMENT,
            "{name}: flag agreement"
        );
        assert!(
            tally.iteration_agreement >= MIN_AGREEMENT,
            "{name}: iteration agreement"
        );
        assert!(
            tally.both_converged + 2 >= tally.native_converged.min(tally.external_converged),
            "{name}: jointly converged"
        );
        assert!(
            tally.same_correction * 20 >= tally.both_converged * 19,
            "{name}: correction agreement"
        );
    }
}

#[test]
fn frozen_random_gammas_aggregate_and_draw_order() {
    let (dcm, shots) = load_fixture();
    let tally = compare("c3_random_gammas", &dcm, &shots);
    assert_eq!(
        tally.native_converged, 76,
        "per-shot RNG draw order changed"
    );
    assert!(tally.native_converged.abs_diff(tally.external_converged) <= 5);
}

#[test]
fn frozen_operating_point_contract() {
    let (dcm, shots) = load_fixture();
    let tally = compare("c4_operating_point", &dcm, &shots);
    assert_eq!(tally.native_converged, SHOTS);
    assert_eq!(tally.external_converged, SHOTS);
    assert!(tally.native_errors <= tally.external_errors + 5);
}

#[test]
fn frozen_headers_reject_missing_and_malformed_configuration() {
    let relay = include_str!("data/relay_oracle/c2_explicit_two_rows.txt");
    let min_sum = include_str!("data/relay_oracle/c8_min_sum_iter8.txt");
    for (data, keys) in [
        (
            relay,
            &[
                "decoder",
                "alpha",
                "gamma0",
                "pre_iter",
                "num_sets",
                "set_max_iter",
                "stopping_criterion",
                "gamma_dist_interval",
                "alpha_iteration_scaling_factor",
                "explicit_gammas",
                "seed",
                "max_iter",
            ][..],
        ),
        (
            min_sum,
            &[
                "decoder",
                "max_iter",
                "alpha",
                "gamma0",
                "stopping_criterion",
                "seed",
                "alpha_iteration_scaling_factor",
            ][..],
        ),
    ] {
        for key in keys {
            let prefix = format!("# {key}=");
            let missing = data
                .lines()
                .filter(|line| !line.starts_with(&prefix))
                .collect::<Vec<_>>()
                .join("\n");
            assert!(
                std::panic::catch_unwind(|| parse_oracle(&missing, 8785)).is_err(),
                "accepted missing {key}"
            );
        }
    }
    for key in [
        "decoder",
        "alpha",
        "gamma0",
        "pre_iter",
        "num_sets",
        "set_max_iter",
        "stopping_criterion",
        "gamma_dist_interval",
        "alpha_iteration_scaling_factor",
        "explicit_gammas",
    ] {
        let prefix = format!("# {key}=");
        let malformed = relay
            .lines()
            .map(|line| {
                if line.starts_with(&prefix) {
                    format!("{prefix}malformed")
                } else {
                    line.to_owned()
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            std::panic::catch_unwind(|| parse_oracle(&malformed, 8785)).is_err(),
            "accepted malformed {key}"
        );
    }
}
