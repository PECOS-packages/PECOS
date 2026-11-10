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

use super::*;

fn map(entries: &[(&str, f64)]) -> BTreeMap<String, f64> {
    entries
        .iter()
        .map(|&(key, value)| (key.to_string(), value))
        .collect()
}

fn error(options: NoiseOptions) -> String {
    NoiseConfig::try_from(options).unwrap_err()
}

#[test]
fn base_idle_conflicts_and_missing_partners() {
    for (options, message) in [
        (
            NoiseOptions {
                p_idle: Some(0.0),
                t1: Some(1.0),
                ..NoiseOptions::default()
            },
            "p_idle cannot be combined with t1/t2; the T1/T2 channel replaces the depolarizing base idle channel, so p_idle would be ignored",
        ),
        (
            NoiseOptions {
                p_idle: Some(0.0),
                t2: Some(1.0),
                ..NoiseOptions::default()
            },
            "p_idle cannot be combined with t1/t2; the T1/T2 channel replaces the depolarizing base idle channel, so p_idle would be ignored",
        ),
        (
            NoiseOptions {
                p_idle: Some(0.0),
                idle_rz: Some(0.0),
                ..NoiseOptions::default()
            },
            "idle_rz cannot be combined with p_idle; the coherent RZ conversion replaces the base idle channel, so p_idle would be ignored",
        ),
        (
            NoiseOptions {
                idle_rz: Some(0.0),
                t1: Some(1.0),
                ..NoiseOptions::default()
            },
            "idle_rz cannot be combined with t1/t2; the coherent RZ conversion overwrites the T1/T2 channel with an equivalent T2",
        ),
        (
            NoiseOptions {
                idle_rz: Some(0.0),
                t2: Some(1.0),
                ..NoiseOptions::default()
            },
            "idle_rz cannot be combined with t1/t2; the coherent RZ conversion overwrites the T1/T2 channel with an equivalent T2",
        ),
        (
            NoiseOptions {
                t1: Some(1.0),
                ..NoiseOptions::default()
            },
            "t1 requires t2; both relaxation times must be supplied",
        ),
        (
            NoiseOptions {
                t2: Some(1.0),
                ..NoiseOptions::default()
            },
            "t2 requires t1; both relaxation times must be supplied",
        ),
    ] {
        assert_eq!(error(options), message);
    }
}

#[test]
fn weight_errors() {
    for (entries, message) in [
        (
            vec![("A", 1.0)],
            "p1_weights keys must be one of ['X', 'Y', 'Z'], got \"A\"",
        ),
        (
            vec![("X", -1.0)],
            "p1_weights[\"X\"] must be finite and non-negative, got -1",
        ),
        (
            vec![("X", 0.5)],
            "p1_weights relative probabilities must sum to 1.0, got 0.5",
        ),
    ] {
        assert_eq!(
            error(NoiseOptions {
                p1_weights: Some(map(&entries)),
                ..NoiseOptions::default()
            }),
            message
        );
    }
    for (entries, message) in [
        (vec![("*XX", 1.0)], "p2_weights replacement label \"*XX\" uses the removed '*' syntax; use \"~XX\" (or \":replace:XX\") instead".to_string()),
        (vec![("~XX", 0.5), (":replace:XX", 0.5)], "p2_weights contains duplicate label \"~XX\" after normalization; use only one spelling for each branch".to_string()),
        (vec![("II", 1.0)], "plain p2_weights cannot contain identity pair 'II'; use '~II' (or ':replace:II') for a replacement branch that only omits the gate".to_string()),
        (vec![("AA", 1.0)], format!("p2_weights keys must be one of {PAULI_2Q_ORDER:?}, or use '~' / ':replace:' before a two-qubit Pauli label for a replacement branch; got \"AA\"")),
        (vec![("XX", -1.0)], "p2_weights[\"XX\"] must be finite and non-negative, got -1".to_string()),
        (vec![("XX", 0.5)], "p2_weights relative probabilities must sum to 1.0, got 0.5".to_string()),
    ] {
        assert_eq!(error(NoiseOptions { p2_weights: Some(map(&entries)), ..NoiseOptions::default() }), message);
    }
    for rate in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            parse_p1_weights(map(&[("X", rate)])).unwrap_err(),
            format!("p1_weights[\"X\"] must be finite and non-negative, got {rate}")
        );
        assert_eq!(
            parse_p2_weights(map(&[("XX", rate)])).unwrap_err(),
            format!("p2_weights[\"XX\"] must be finite and non-negative, got {rate}")
        );
    }
    assert!(parse_p1_weights(map(&[(" x ", 1.0 + 0.5e-6)])).is_ok());
    assert!(parse_p2_weights(map(&[(" xx ", 1.0 + 0.5e-6)])).is_ok());
    assert!(parse_p1_weights(map(&[("X", 1.0 + 2.0e-6)])).is_err());
    assert!(parse_p2_weights(map(&[("XX", 1.0 + 2.0e-6)])).is_err());
}

#[test]
fn gate_rate_errors_and_names() {
    for (family, label, rate, expected) in [
        (
            "p1",
            "CX",
            1.0,
            "p1_gate_rates keys must name single-qubit gates, got \"CX\"".to_string(),
        ),
        (
            "p2",
            "X",
            1.0,
            "p2_gate_rates keys must name two-qubit gates, got \"X\"".to_string(),
        ),
        (
            "p1",
            "X",
            -1.0,
            "p1_gate_rates[\"X\"] must be finite and non-negative, got -1".to_string(),
        ),
        (
            "p2",
            "CX",
            -1.0,
            "p2_gate_rates[\"CX\"] must be finite and non-negative, got -1".to_string(),
        ),
        (
            "p1",
            "invalid",
            1.0,
            format!(
                "unsupported p1_gate_rates gate label \"invalid\": {}",
                GateType::from_str("invalid").unwrap_err()
            ),
        ),
        (
            "p2",
            "invalid",
            1.0,
            format!(
                "unsupported p2_gate_rates gate label \"invalid\": {}",
                GateType::from_str("invalid").unwrap_err()
            ),
        ),
    ] {
        let mut options = NoiseOptions::default();
        if family == "p1" {
            options.p1_gate_rates = Some(map(&[(label, rate)]));
        } else {
            options.p2_gate_rates = Some(map(&[(label, rate)]));
        }
        assert_eq!(error(options), expected);
    }
    for rate in [f64::NAN, f64::INFINITY] {
        assert_eq!(
            parse_p1_gate_rates(map(&[("X", rate)])).unwrap_err(),
            format!("p1_gate_rates[\"X\"] must be finite and non-negative, got {rate}")
        );
        assert_eq!(
            parse_p2_gate_rates(map(&[("CX", rate)])).unwrap_err(),
            format!("p2_gate_rates[\"CX\"] must be finite and non-negative, got {rate}")
        );
    }
    assert_eq!(
        parse_p1_gate_rates(map(&[(" x ", 0.2)])).unwrap(),
        BTreeMap::from([(GateType::X, 0.2)])
    );
    assert_eq!(
        parse_p2_gate_rates(map(&[(" cx ", 0.3)])).unwrap(),
        BTreeMap::from([(GateType::CX, 0.3)])
    );
}

#[test]
fn replacement_and_crosstalk_aliases() {
    for (aliases, expected) in [
        (
            vec!["pauli_twirl_omitted_gate", "pauli_twirl", "twirl"],
            ReplacementBranchApproximation::PauliTwirlOmittedGate,
        ),
        (
            vec!["branch_impact", "replacement_branch_impact", "impact"],
            ReplacementBranchApproximation::BranchImpact,
        ),
        (
            vec![
                "exact_branch_replay",
                "exact_replay",
                "exact_branch",
                "exact",
            ],
            ReplacementBranchApproximation::ExactBranchReplay,
        ),
        (
            vec![
                "ignore_gate_removal",
                "ignore_removal",
                "post_gate",
                "postgate",
            ],
            ReplacementBranchApproximation::IgnoreGateRemoval,
        ),
    ] {
        for alias in aliases {
            let input = format!(" {} ", alias.to_uppercase().replace('_', "-"));
            assert_eq!(
                parse_replacement_approximation(Some(input)).unwrap(),
                expected
            );
        }
    }
    assert_eq!(
        error(NoiseOptions {
            p2_replacement_approximation: Some("bad".into()),
            ..NoiseOptions::default()
        }),
        "p2_replacement_approximation must be 'pauli_twirl_omitted_gate', 'branch_impact', 'exact_branch_replay', or 'ignore_gate_removal'"
    );
    for (aliases, expected) in [
        (
            vec!["omitted", "omit", "none", "off"],
            MeasurementCrosstalkDemMode::Omitted,
        ),
        (
            vec!["exact_deterministic", "exact", "deterministic"],
            MeasurementCrosstalkDemMode::ExactDeterministic,
        ),
        (
            vec![
                "exact_deterministic_leakage_as_depolarizing",
                "exact_leakage_as_depolarizing",
                "deterministic_leakage_as_depolarizing",
                "leakage_as_depolarizing",
            ],
            MeasurementCrosstalkDemMode::ExactDeterministicLeakageAsDepolarizing,
        ),
        (
            vec![
                "averaged_hidden_leakage_as_depolarizing",
                "average_hidden_leakage_as_depolarizing",
                "state_averaged_leakage_as_depolarizing",
                "averaged_leakage_as_depolarizing",
            ],
            MeasurementCrosstalkDemMode::AveragedHiddenLeakageAsDepolarizing,
        ),
    ] {
        for alias in aliases {
            let input = format!(" {} ", alias.to_uppercase().replace('_', " "));
            assert_eq!(
                parse_measurement_crosstalk_dem_mode(Some(input)).unwrap(),
                expected
            );
        }
    }
    assert_eq!(
        error(NoiseOptions {
            measurement_crosstalk_dem_mode: Some("bad".into()),
            ..NoiseOptions::default()
        }),
        "measurement_crosstalk_dem_mode must be 'omitted', 'exact_deterministic', 'exact_deterministic_leakage_as_depolarizing', or 'averaged_hidden_leakage_as_depolarizing'"
    );
}

#[test]
fn crosstalk_transition_rules() {
    for rate in [-1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            error(NoiseOptions {
                p_meas_crosstalk_model: Some(map(&[("0->1", rate)])),
                ..NoiseOptions::default()
            }),
            "measurement crosstalk transition probability for \"0->1\" must be finite and non-negative"
        );
    }
    assert_eq!(
        error(NoiseOptions {
            p_meas_crosstalk_model: Some(map(&[("bad", 0.0)])),
            ..NoiseOptions::default()
        }),
        "unsupported measurement crosstalk transition key \"bad\"; expected 0->0, 0->1, 0->L, 1->0, 1->1, or 1->L"
    );
    for row in [
        vec![("0->1", 0.6), ("0->L", 0.5)],
        vec![("1->0", 0.6), ("1->L", 0.5)],
    ] {
        assert_eq!(
            error(NoiseOptions {
                p_meas_crosstalk_model: Some(map(&row)),
                ..NoiseOptions::default()
            }),
            "measurement crosstalk transition rows must sum to <= 1"
        );
    }
    let parsed = parse_measurement_crosstalk_transition_model(Some(map(&[
        ("0 -> 0", 2.0),
        ("1 -> 1", 2.0),
        ("0 -> 1", 0.2),
        ("0 -> l", 0.3),
        ("1 -> 0", 0.4),
        ("1 -> l", 0.5),
    ])))
    .unwrap();
    assert_eq!(
        parsed,
        MeasurementCrosstalkTransitionModel {
            p_0_to_1: 0.2,
            p_0_to_leak: 0.3,
            p_1_to_0: 0.4,
            p_1_to_leak: 0.5
        }
    );
}

fn structured(
    family: &str,
    rate: Option<f64>,
    model: Option<BTreeMap<String, f64>>,
) -> NoiseOptions {
    let mut options = NoiseOptions::default();
    match family {
        "linear" => {
            options.p_idle_linear = rate;
            options.p_idle_linear_model = model;
        }
        "sin_squared" => {
            options.p_idle_sin_squared = rate;
            options.p_idle_sin_squared_model = model;
        }
        "coherent" => {
            options.p_idle_coherent = rate;
            options.p_idle_coherent_model = model;
        }
        _ => unreachable!(),
    }
    options
}

#[test]
fn structured_family_messages() {
    for family in ["linear", "sin_squared", "coherent"] {
        assert_eq!(
            error(structured(family, None, Some(BTreeMap::new()))),
            format!("p_idle_{family}_model requires p_idle_{family}; otherwise the model is inert")
        );
        for rate in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                error(structured(family, Some(rate), None)),
                format!("p_idle_{family} must be a finite, non-negative float")
            );
        }
        let keys = if family == "coherent" {
            "\"RX\", \"RY\", \"RZ\""
        } else {
            "\"L\", \"X\", \"Y\", \"Z\""
        };
        assert_eq!(
            error(structured(family, Some(0.0), Some(map(&[("bad", 1.0)])))),
            format!("invalid p_idle_{family}_model key \"bad\"; expected {keys}")
        );
        let key = if family == "coherent" { "RZ" } else { "Z" };
        for multiplier in [-1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                error(structured(
                    family,
                    Some(0.0),
                    Some(map(&[(key, multiplier)]))
                )),
                format!(
                    "p_idle_{family}_model multiplier for {key:?} must be a finite, non-negative float"
                )
            );
        }
    }
    for family in ["linear", "sin_squared"] {
        assert_eq!(
            error(structured(family, Some(0.0), Some(map(&[("L", 1.0)])))),
            format!(
                "p_idle_{family}_model key \"L\" has a nonzero multiplier; DEM fault propagation is Pauli-only; the engines simulators support and consume leakage models"
            )
        );
    }
    for rate in [0.01, 1.0] {
        assert_eq!(
            error(structured("coherent", Some(rate), None)),
            concat!(
                "the standard DEM builder cannot represent coherent idle noise; its previous behavior silently stored ",
                "the Pauli twirl, discarding exactly the coherence that was requested. The EEG coherent route in ",
                "exp/pecos-eeg is the consumer that can represent it, and only with an RZ generator even there. The ",
                "honest stochastic equivalent, which is the exact Pauli twirl of RZ(rate * t), is ",
                "p_idle_sin_squared=rate/2 with p_idle_sin_squared_model={'Z': 1.0}"
            )
        );
    }
}

#[test]
fn structured_conflicts() {
    let options = NoiseOptions {
        p_idle_linear_model: Some(BTreeMap::new()),
        p_idle_linear_rate: Some(0.0),
        p_idle_x_linear_rate: Some(0.0),
        p_idle_y_linear_rate: Some(0.0),
        p_idle_z_linear_rate: Some(0.0),
        ..NoiseOptions::default()
    };
    assert_eq!(
        error(options),
        "p_idle_linear/p_idle_linear_model cannot be combined with low-level idle rate(s): p_idle_linear_rate, p_idle_x_linear_rate, p_idle_y_linear_rate, p_idle_z_linear_rate"
    );
    let options = NoiseOptions {
        p_idle_sin_squared: Some(0.0),
        p_idle_quadratic_sine_rate: Some(0.0),
        p_idle_x_quadratic_sine_rate: Some(0.0),
        p_idle_y_quadratic_sine_rate: Some(0.0),
        p_idle_z_quadratic_sine_rate: Some(0.0),
        ..NoiseOptions::default()
    };
    assert_eq!(
        error(options),
        "p_idle_sin_squared/p_idle_sin_squared_model cannot be combined with sine-law idle rate(s): p_idle_quadratic_sine_rate, p_idle_x_quadratic_sine_rate, p_idle_y_quadratic_sine_rate, p_idle_z_quadratic_sine_rate"
    );
}

#[test]
fn structured_defaults_normalization_and_absent_axes() {
    assert_eq!(
        NoiseOptions::default().translate_idle_noise().unwrap(),
        [None; 6]
    );
    assert_eq!(
        structured("linear", Some(0.03), None)
            .translate_idle_noise()
            .unwrap(),
        [
            Some(0.009_999_999_999_999_998),
            Some(0.009_999_999_999_999_998),
            Some(0.009_999_999_999_999_998),
            None,
            None,
            None
        ]
    );
    assert_eq!(
        structured("sin_squared", Some(0.03), None)
            .translate_idle_noise()
            .unwrap(),
        [None, None, None, Some(0.03), Some(0.03), Some(0.03)]
    );
    assert_eq!(
        structured("coherent", Some(0.0), None)
            .translate_idle_noise()
            .unwrap(),
        [None; 6]
    );
    assert_eq!(
        structured("sin_squared", Some(0.0), Some(map(&[("Z", 2.0)])))
            .translate_idle_noise()
            .unwrap(),
        [None, None, None, None, None, Some(0.0)]
    );
    assert_eq!(
        structured(
            "sin_squared",
            Some(0.2),
            Some(map(&[("X", 2.0), ("Z", 0.0)]))
        )
        .translate_idle_noise()
        .unwrap(),
        [None, None, None, Some(0.4), None, None]
    );
    for total in [0.0, 0.8, 1.00002] {
        assert_eq!(
            error(structured("linear", Some(0.1), Some(map(&[("Z", total)])))),
            format!(
                "p_idle_linear_model multipliers must sum to 1.0 within tolerance 1e-05; got {total}"
            )
        );
    }
    for total in [1.0 - 0.5e-5, 1.0 + 0.5e-5] {
        assert_eq!(
            structured("linear", Some(0.1), Some(map(&[("Z", total)])))
                .translate_idle_noise()
                .unwrap(),
            [Some(0.0), Some(0.0), Some(0.1), None, None, None]
        );
    }
    let total = 1.0 + 0.5e-10;
    assert_eq!(
        structured("linear", Some(0.1), Some(map(&[("Z", total), ("L", 0.0)])))
            .translate_idle_noise()
            .unwrap(),
        [Some(0.0), Some(0.0), Some(0.1 * total), None, None, None]
    );
}

#[test]
fn representative_configs_match_previous_binding_assembly() {
    use pecos_core::pauli::{X, Z};
    let options = NoiseOptions {
        p1: 0.01,
        p2: 0.02,
        p_meas: 0.03,
        p_prep: 0.04,
        p_idle: Some(0.05),
        p_idle_linear_rate: Some(0.7),
        p_idle_z_linear_rate: Some(0.3),
        p_idle_x_linear_rate: Some(0.1),
        p_idle_y_linear_rate: Some(0.2),
        p_idle_quadratic_rate: Some(0.8),
        p_idle_z_quadratic_rate: Some(0.0),
        p_idle_x_quadratic_rate: Some(0.4),
        p_idle_y_quadratic_rate: Some(0.5),
        p_idle_quadratic_sine_rate: Some(0.9),
        p_idle_z_quadratic_sine_rate: Some(0.0),
        p_idle_x_quadratic_sine_rate: Some(0.6),
        p1_weights: Some(map(&[("X", 0.25), ("Z", 0.75)])),
        p2_weights: Some(map(&[("XZ", 0.25), ("~II", 0.75)])),
        p1_gate_rates: Some(map(&[("X", 0.1)])),
        p2_gate_rates: Some(map(&[("CX", 0.2)])),
        p2_replacement_approximation: Some("exact".into()),
        p_meas_crosstalk_local: Some(0.12),
        p_meas_crosstalk_global: Some(0.13),
        p_meas_crosstalk_model: Some(map(&[
            ("0->1", 0.2),
            ("0->L", 0.3),
            ("1->0", 0.4),
            ("1->L", 0.5),
        ])),
        measurement_crosstalk_dem_mode: Some("exact".into()),
        ..NoiseOptions::default()
    };
    let expected = NoiseConfig {
        idle_noise_parameters: options.idle_noise_parameters(),
        p1: 0.01,
        p2: 0.02,
        p_meas: 0.03,
        p_prep: 0.04,
        p_idle: 0.05,
        t1: None,
        t2: None,
        idle_rz: 0.0,
        p1_gate_rates: BTreeMap::from([(GateType::X, 0.1)]),
        p2_gate_rates: BTreeMap::from([(GateType::CX, 0.2)]),
        p1_weights: Some(PauliWeights::new(vec![(X(0), 0.25), (Z(0), 0.75)])),
        p2_weights: Some(PauliWeights::with_replacement(
            vec![(X(0) & Z(1), 0.25)],
            vec![(pecos_core::PauliString::identity(), 0.75)],
        )),
        p2_replacement_approximation: ReplacementBranchApproximation::ExactBranchReplay,
        p_idle_linear: IdleNoiseFamily::new(1.0, map(&[("X", 0.1), ("Y", 0.2), ("Z", 0.3)])),
        p_idle_quadratic: IdleNoiseFamily::new(1.0, map(&[("X", 0.4), ("Y", 0.5), ("Z", 0.0)])),
        p_idle_quadratic_sine: IdleNoiseFamily::new(
            1.0,
            map(&[("X", 0.6), ("Y", 0.0), ("Z", 0.0)]),
        ),
        p_meas_crosstalk_local: 0.12,
        p_meas_crosstalk_global: 0.13,
        p_meas_crosstalk_model: MeasurementCrosstalkTransitionModel {
            p_0_to_1: 0.2,
            p_0_to_leak: 0.3,
            p_1_to_0: 0.4,
            p_1_to_leak: 0.5,
        },
        measurement_crosstalk_dem_mode: MeasurementCrosstalkDemMode::ExactDeterministic,
    };
    // NoiseConfig has no PartialEq; compare its complete derived representation.
    assert_eq!(
        format!("{:?}", NoiseConfig::try_from(options).unwrap()),
        format!("{expected:?}")
    );
    let options = NoiseOptions {
        t1: Some(10.0),
        t2: Some(15.0),
        ..NoiseOptions::default()
    };
    let expected = NoiseConfig {
        idle_noise_parameters: options.idle_noise_parameters(),
        t1: Some(10.0),
        t2: Some(15.0),
        ..NoiseConfig::new(0.0, 0.0, 0.0, 0.0)
    };
    assert_eq!(
        format!("{:?}", NoiseConfig::try_from(options).unwrap()),
        format!("{expected:?}")
    );
    let options = NoiseOptions {
        idle_rz: Some(0.2),
        ..NoiseOptions::default()
    };
    let expected = NoiseConfig {
        idle_noise_parameters: options.idle_noise_parameters(),
        idle_rz: 0.2,
        t1: Some(1e15),
        t2: Some(-1.0 / (1.0 - 2.0 * 0.1_f64.sin().powi(2)).ln()),
        ..NoiseConfig::new(0.0, 0.0, 0.0, 0.0)
    };
    assert_eq!(
        format!("{:?}", NoiseConfig::try_from(options).unwrap()),
        format!("{expected:?}")
    );
}

#[test]
fn bare_aliases_and_zero_families() {
    let options = NoiseOptions {
        p_idle_linear_rate: Some(0.1),
        p_idle_quadratic_rate: Some(0.2),
        p_idle_quadratic_sine_rate: Some(0.3),
        ..NoiseOptions::default()
    };
    let noise = NoiseConfig::try_from(options).unwrap();
    assert_eq!(
        noise.p_idle_linear,
        IdleNoiseFamily::new(1.0, map(&[("X", 0.0), ("Y", 0.0), ("Z", 0.1)]))
    );
    assert_eq!(
        noise.p_idle_quadratic,
        IdleNoiseFamily::new(1.0, map(&[("X", 0.0), ("Y", 0.0), ("Z", 0.2)]))
    );
    assert_eq!(
        noise.p_idle_quadratic_sine,
        IdleNoiseFamily::new(1.0, map(&[("X", 0.0), ("Y", 0.0), ("Z", 0.3)]))
    );
    let noise = NoiseConfig::try_from(NoiseOptions::default()).unwrap();
    let expected = NoiseConfig::new(0.0, 0.0, 0.0, 0.0);
    assert_eq!(format!("{noise:?}"), format!("{expected:?}"));
    assert_eq!(noise.p_idle_linear, IdleNoiseFamily::default());
    assert_eq!(noise.p_idle_quadratic, IdleNoiseFamily::default());
    assert_eq!(noise.p_idle_quadratic_sine, IdleNoiseFamily::default());
}

mod p2_weight_parser_tests {
    use super::*;

    fn parse_error(weights: BTreeMap<String, f64>) -> String {
        parse_p2_weights(weights).unwrap_err()
    }

    #[test]
    fn accepts_both_replacement_label_spellings_case_insensitively() {
        use pecos_core::pauli::X;

        for label in [" ~xx ", " :RePlAcE:xX "] {
            let parsed = parse_p2_weights(BTreeMap::from([(label.to_string(), 1.0)])).unwrap();

            assert_eq!(parsed.entries(), []);
            assert_eq!(parsed.replacement_entries(), &[(X(0) & X(1), 1.0)]);
        }
    }

    #[test]
    fn accepts_omission_only_replacement_label() {
        let parsed = parse_p2_weights(BTreeMap::from([("~II".to_string(), 1.0)])).unwrap();

        assert_eq!(parsed.entries(), []);
        assert_eq!(
            parsed.replacement_entries(),
            &[(pecos_core::PauliString::identity(), 1.0)]
        );
    }

    #[test]
    fn rejects_removed_star_replacement_syntax_with_migration_hint() {
        let message = parse_error(BTreeMap::from([("*XX".to_string(), 1.0)]));

        assert!(message.contains("~XX"));
        assert!(message.contains(":replace:XX"));
    }

    #[test]
    fn rejects_duplicate_replacement_aliases_after_normalization() {
        let message = parse_error(BTreeMap::from([
            ("~XX".to_string(), 0.5),
            (":replace:XX".to_string(), 0.5),
        ]));

        assert!(message.contains("duplicate label"));
    }

    #[test]
    fn rejects_plain_identity_with_replacement_spelling_hint() {
        let message = parse_error(BTreeMap::from([("II".to_string(), 1.0)]));

        assert!(message.contains("~II"));
        assert!(message.contains(":replace:II"));
    }
}

#[test]
fn structured_config_and_coefficient_composition() {
    let options = NoiseOptions {
        p_idle: Some(0.01),
        p_idle_linear: Some(0.2),
        p_idle_linear_model: Some(map(&[("Z", 0.75), ("X", 0.25), ("L", 0.0)])),
        p_idle_sin_squared: Some(0.3),
        p_idle_sin_squared_model: Some(map(&[("Y", 2.0)])),
        p_idle_coherent: Some(0.0),
        p_idle_x_quadratic_rate: Some(0.4),
        ..NoiseOptions::default()
    };
    let expected = NoiseConfig {
        idle_noise_parameters: options.idle_noise_parameters(),
        p_idle: 0.01,
        p_idle_linear: IdleNoiseFamily::new(
            1.0,
            map(&[("X", 0.05), ("Y", 0.0), ("Z", 0.150_000_000_000_000_02)]),
        ),
        p_idle_quadratic: IdleNoiseFamily::new(1.0, map(&[("X", 0.4), ("Y", 0.0), ("Z", 0.0)])),
        p_idle_quadratic_sine: IdleNoiseFamily::new(
            1.0,
            map(&[("X", 0.0), ("Y", 0.6), ("Z", 0.0)]),
        ),
        ..NoiseConfig::new(0.0, 0.0, 0.0, 0.0)
    };
    assert_eq!(
        format!("{:?}", NoiseConfig::try_from(options).unwrap()),
        format!("{expected:?}")
    );
}

#[test]
fn normalization_uses_engines_sum_and_rust_display() {
    assert_eq!(
        error(structured("linear", Some(0.1), Some(BTreeMap::new()))),
        "p_idle_linear_model multipliers must sum to 1.0 within tolerance 1e-05; got 0"
    );
    for (total, text) in [(1.0e-6, "0.000001"), (1.0e20, "100000000000000000000")] {
        assert_eq!(
            error(structured("linear", Some(0.1), Some(map(&[("Z", total)])))),
            format!(
                "p_idle_linear_model multipliers must sum to 1.0 within tolerance 1e-05; got {text}"
            )
        );
    }
    let entries = map(&[("Z", 1e-16), ("Y", 1e-16), ("X", 1.000_001)]);
    let axes = structured("linear", Some(1.0), Some(entries))
        .translate_idle_noise()
        .unwrap();
    // Key-order summation adds X first, rounding away both small terms.
    let total = 1.000_001;
    assert_eq!(
        axes,
        [
            Some(1.0),
            Some(1e-16 / total),
            Some(1e-16 / total),
            None,
            None,
            None
        ]
    );
}

#[test]
fn idle_diagnostics_preserve_public_parameters() {
    macro_rules! check_rate {
        ($($name:ident),+ $(,)?) => {
            $(
                let noise = NoiseConfig::try_from(NoiseOptions {
                    $name: Some(0.01),
                    ..NoiseOptions::default()
                }).unwrap();
                assert_eq!(noise.idle_noise_request().unwrap(), concat!(stringify!($name), "=0.01"));
                let zero = NoiseConfig::try_from(NoiseOptions {
                    $name: Some(0.0),
                    ..NoiseOptions::default()
                }).unwrap();
                assert_eq!(zero.idle_noise_request(), None);
            )+
        };
    }
    check_rate!(
        p_idle,
        idle_rz,
        p_idle_linear,
        p_idle_sin_squared,
        p_idle_linear_rate,
        p_idle_quadratic_rate,
        p_idle_quadratic_sine_rate,
        p_idle_x_linear_rate,
        p_idle_y_linear_rate,
        p_idle_z_linear_rate,
        p_idle_x_quadratic_rate,
        p_idle_y_quadratic_rate,
        p_idle_z_quadratic_rate,
        p_idle_x_quadratic_sine_rate,
        p_idle_y_quadratic_sine_rate,
        p_idle_z_quadratic_sine_rate,
    );
    let noise = NoiseConfig::try_from(NoiseOptions {
        t1: Some(100.0),
        t2: Some(0.0),
        p_idle_x_linear_rate: Some(0.01),
        p_idle_y_quadratic_rate: Some(0.02),
        ..NoiseOptions::default()
    })
    .unwrap();
    assert_eq!(
        noise.idle_noise_request().unwrap(),
        "p_idle_x_linear_rate=0.01, p_idle_y_quadratic_rate=0.02, t1=100, t2=0"
    );
    let raw =
        NoiseConfig::default().set_idle_linear(IdleNoiseFamily::new(0.01, map(&[("X", 1.0)])));
    assert_eq!(raw.idle_noise_request().unwrap(), "p_idle_linear = 0.01");
    let reconfigured = NoiseConfig::try_from(NoiseOptions {
        p_idle_x_linear_rate: Some(0.01),
        ..NoiseOptions::default()
    })
    .unwrap()
    .set_idle(0.02);
    assert_eq!(
        reconfigured.idle_noise_request().unwrap(),
        "p_idle = 0.02, p_idle_linear = 1"
    );
}
