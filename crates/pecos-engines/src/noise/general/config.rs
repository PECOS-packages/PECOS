//! The general-noise parameter schema shared by engine and adapter consumers.

use super::{GeneralNoiseModel, GeneralNoiseModelBuilder};
use crate::noise::{
    P2PauliLeakageStep, P2TransitionStep, PauliLeakageChannel, StateTransitionChannel,
    TwoQubitPauliLeakageChannel, TwoQubitStateTransitionChannel,
};
use pecos_core::{errors::PecosError, gate_type::GateType};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::BTreeMap, str::FromStr};

fn input_error(error: impl std::fmt::Display) -> PecosError {
    PecosError::Input(error.to_string())
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum TransitionConfig {
    Flat(BTreeMap<String, f64>),
    Rows(BTreeMap<String, BTreeMap<String, f64>>),
}

impl TransitionConfig {
    fn arity(&self) -> usize {
        let label = match self {
            Self::Flat(rows) => rows
                .keys()
                .next()
                .map(|key| key.split("->").next().unwrap_or(key)),
            Self::Rows(rows) => rows.keys().next().map(String::as_str),
        };
        label.map_or(1, |label| label.trim().len())
    }

    fn single(&self, probability: f64) -> Result<StateTransitionChannel, PecosError> {
        Ok(match self {
            Self::Flat(rows) => {
                StateTransitionChannel::from_transitions(probability, rows).map_err(input_error)?
            }
            Self::Rows(rows) => {
                StateTransitionChannel::new(probability, rows).map_err(input_error)?
            }
        })
    }

    fn joint(&self, probability: f64) -> Result<TwoQubitStateTransitionChannel, PecosError> {
        Ok(match self {
            Self::Flat(rows) => TwoQubitStateTransitionChannel::from_transitions(probability, rows)
                .map_err(input_error)?,
            Self::Rows(rows) => {
                TwoQubitStateTransitionChannel::new(probability, rows).map_err(input_error)?
            }
        })
    }
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum HookConfig {
    Transition {
        probability: f64,
        transitions: TransitionConfig,
    },
    PauliLeakage {
        probability: f64,
        events: BTreeMap<String, f64>,
    },
    Independent {
        first: Box<HookConfig>,
        second: Box<HookConfig>,
    },
}

// Read JSON values directly: serde's buffered enum deserializers cannot read
// serde_json numbers when workspace feature unification enables arbitrary_precision.
impl<'de> Deserialize<'de> for HookConfig {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::from_value(&Value::deserialize(deserializer)?)
    }
}

fn hook_number<E: serde::de::Error>(value: &Value) -> Result<f64, E> {
    value
        .as_f64()
        .ok_or_else(|| E::custom("invalid type: expected f64"))
}

fn hook_distribution<E: serde::de::Error>(value: &Value) -> Result<BTreeMap<String, f64>, E> {
    value
        .as_object()
        .ok_or_else(|| E::custom("invalid type: expected a map of probabilities"))?
        .iter()
        .map(|(label, value)| Ok((label.clone(), hook_number(value)?)))
        .collect()
}

impl HookConfig {
    fn from_value<E: serde::de::Error>(value: &Value) -> Result<Self, E> {
        let object = value
            .as_object()
            .ok_or_else(|| E::custom("invalid type: expected a hook object"))?;
        let field = |name| object.get(name).ok_or_else(|| E::missing_field(name));
        let kind = field("kind")?
            .as_str()
            .ok_or_else(|| E::custom("invalid type: expected a string for `kind`"))?;
        let fields: &[&str] = match kind {
            "transition" => &["probability", "transitions"],
            "pauli_leakage" => &["probability", "events"],
            "independent" => &["first", "second"],
            _ => {
                return Err(E::unknown_variant(
                    kind,
                    &["transition", "pauli_leakage", "independent"],
                ));
            }
        };
        for name in object.keys() {
            if name != "kind" && !fields.contains(&name.as_str()) {
                return Err(E::unknown_field(name, fields));
            }
        }
        match kind {
            "transition" => {
                let probability = hook_number(field("probability")?)?;
                let transitions = field("transitions")?;
                let rows = transitions
                    .as_object()
                    .ok_or_else(|| E::custom("invalid type: expected a transitions map"))?;
                let transitions = if rows.values().all(Value::is_number) {
                    TransitionConfig::Flat(hook_distribution(transitions)?)
                } else {
                    TransitionConfig::Rows(
                        rows.iter()
                            .map(|(label, row)| Ok((label.clone(), hook_distribution(row)?)))
                            .collect::<Result<_, E>>()?,
                    )
                };
                Ok(Self::Transition {
                    probability,
                    transitions,
                })
            }
            "pauli_leakage" => Ok(Self::PauliLeakage {
                probability: hook_number(field("probability")?)?,
                events: hook_distribution(field("events")?)?,
            }),
            "independent" => Ok(Self::Independent {
                first: Box::new(Self::from_value(field("first")?)?),
                second: Box::new(Self::from_value(field("second")?)?),
            }),
            _ => unreachable!("hook kind was validated above"),
        }
    }

    fn apply(
        &self,
        builder: GeneralNoiseModelBuilder,
        two: bool,
        before: bool,
    ) -> Result<GeneralNoiseModelBuilder, PecosError> {
        match self {
            Self::Transition {
                probability,
                transitions,
            } if !two => {
                let channel = transitions.single(*probability)?;
                Ok(if before {
                    builder.add_p1_transition_channel_before_gate(&channel)
                } else {
                    builder.add_p1_transition_channel_after_gate(&channel)
                })
            }
            Self::PauliLeakage {
                probability,
                events,
            } if !two => {
                let channel =
                    PauliLeakageChannel::new(*probability, events).map_err(input_error)?;
                Ok(if before {
                    builder.add_p1_pauli_leakage_channel_before_gate(&channel)
                } else {
                    builder.add_p1_pauli_leakage_channel_after_gate(&channel)
                })
            }
            Self::Transition {
                probability,
                transitions,
            } => {
                if transitions.arity() == 1 {
                    let channel = transitions.single(*probability)?;
                    Ok(if before {
                        builder.add_p2_transition_channel_before_gate(&channel)
                    } else {
                        builder.add_p2_transition_channel_after_gate(&channel)
                    })
                } else {
                    let channel = transitions.joint(*probability)?;
                    Ok(if before {
                        builder.add_p2_transition_channel_before_gate(&channel)
                    } else {
                        builder.add_p2_transition_channel_after_gate(&channel)
                    })
                }
            }
            Self::PauliLeakage {
                probability,
                events,
            } => {
                if events
                    .keys()
                    .next()
                    .is_none_or(|label| label.trim().len() == 1)
                {
                    let channel =
                        PauliLeakageChannel::new(*probability, events).map_err(input_error)?;
                    Ok(if before {
                        builder.add_p2_pauli_leakage_channel_before_gate(&channel)
                    } else {
                        builder.add_p2_pauli_leakage_channel_after_gate(&channel)
                    })
                } else {
                    let step = P2PauliLeakageStep::joint(
                        TwoQubitPauliLeakageChannel::new(*probability, events)
                            .map_err(input_error)?,
                    );
                    Ok(if before {
                        builder.add_p2_pauli_leakage_step_before_gate(&step)
                    } else {
                        builder.add_p2_pauli_leakage_step_after_gate(&step)
                    })
                }
            }
            Self::Independent { first, second } => {
                if !two {
                    return Err(input_error("independent hook requires a two-qubit site"));
                }
                match (first.as_ref(), second.as_ref()) {
                    (
                        Self::Transition {
                            probability: p,
                            transitions: a,
                        },
                        Self::Transition {
                            probability: q,
                            transitions: b,
                        },
                    ) => {
                        let step = P2TransitionStep::independent(a.single(*p)?, b.single(*q)?);
                        Ok(if before {
                            builder.add_p2_transition_step_before_gate(&step)
                        } else {
                            builder.add_p2_transition_step_after_gate(&step)
                        })
                    }
                    (
                        Self::PauliLeakage {
                            probability: p,
                            events: a,
                        },
                        Self::PauliLeakage {
                            probability: q,
                            events: b,
                        },
                    ) => {
                        let step = P2PauliLeakageStep::independent(
                            PauliLeakageChannel::new(*p, a).map_err(input_error)?,
                            PauliLeakageChannel::new(*q, b).map_err(input_error)?,
                        );
                        Ok(if before {
                            builder.add_p2_pauli_leakage_step_before_gate(&step)
                        } else {
                            builder.add_p2_pauli_leakage_step_after_gate(&step)
                        })
                    }
                    _ => Err(input_error(
                        "independent hook requires two one-qubit channels of the same family",
                    )),
                }
            }
        }
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct PreparationConfig {
    probability: Option<f64>,
    leakage_ratio: Option<f64>,
    crosstalk_probability: Option<f64>,
    average_crosstalk_probability: Option<f64>,
    scale: Option<f64>,
    crosstalk_scale: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct GateConfig {
    probability: Option<f64>,
    average_infidelity: Option<f64>,
    pauli_model: Option<BTreeMap<String, f64>>,
    emission_ratio: Option<f64>,
    emission_model: Option<BTreeMap<String, f64>>,
    seepage_probability: Option<f64>,
    scale: Option<f64>,
    hooks_before: Vec<HookConfig>,
    hooks_after: Vec<HookConfig>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct TwoQubitConfig {
    probability: Option<f64>,
    average_infidelity: Option<f64>,
    pauli_model: Option<BTreeMap<String, f64>>,
    emission_ratio: Option<f64>,
    emission_model: Option<BTreeMap<String, f64>>,
    seepage_probability: Option<f64>,
    scale: Option<f64>,
    angle_coefficients: Option<[f64; 4]>,
    angle_power: Option<f64>,
    idle_after_gate: Option<f64>,
    hooks_before: Vec<HookConfig>,
    hooks_after: Vec<HookConfig>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct IdleConfig {
    linear_rate: Option<f64>,
    linear_model: Option<BTreeMap<String, f64>>,
    sin_squared_rate: Option<f64>,
    sin_squared_model: Option<BTreeMap<String, f64>>,
    coherent_rate: Option<f64>,
    coherent_model: Option<BTreeMap<String, f64>>,
    scale: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct MeasurementConfig {
    p0_to_1: Option<f64>,
    p1_to_0: Option<f64>,
    global_crosstalk_probability: Option<f64>,
    local_crosstalk_probability: Option<f64>,
    crosstalk_model: Option<BTreeMap<String, f64>>,
    scale: Option<f64>,
    crosstalk_scale: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ScalingConfig {
    overall: Option<f64>,
    leakage: Option<f64>,
    emission: Option<f64>,
}

/// Device-neutral general-noise parameters shared by engine and plugin consumers.
///
/// Unset fields preserve an existing builder's configuration when applied.
#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GeneralNoiseConfig {
    preparation: PreparationConfig,
    measurement: MeasurementConfig,
    single_qubit: GateConfig,
    two_qubit: TwoQubitConfig,
    idle: IdleConfig,
    scaling: ScalingConfig,
    noiseless_gates: Vec<String>,
}

impl GeneralNoiseConfig {
    /// Parse the engine-owned model schema from JSON.
    ///
    /// # Errors
    /// Returns an input error for malformed JSON or unknown fields.
    pub fn from_json(json: &str) -> Result<Self, PecosError> {
        serde_json::from_str(json).map_err(input_error)
    }

    /// Apply set fields to an existing builder, retaining its other parameters.
    ///
    /// Cross-field checks are deferred so callers can continue chaining setters.
    ///
    /// # Errors
    /// Returns schema or setter errors, including an error already recorded on the builder.
    pub fn apply_to(
        &self,
        mut builder: GeneralNoiseModelBuilder,
    ) -> Result<GeneralNoiseModelBuilder, PecosError> {
        if let Some(message) = builder.setter_error() {
            return Err(input_error(message));
        }

        if let Some(value) = self.scaling.overall {
            builder = builder.with_scale(value);
        }
        if let Some(value) = self.scaling.leakage {
            builder = builder.with_leakage_scale(value);
        }
        if let Some(value) = self.scaling.emission {
            builder = builder.with_emission_scale(value);
        }
        for name in &self.noiseless_gates {
            let gate = GateType::from_str(name).map_err(|message| {
                input_error(format!("invalid noiseless gate {name:?}: {message}"))
            })?;
            builder = builder.with_noiseless_gate(gate);
        }

        let prep = &self.preparation;
        if !(prep.crosstalk_probability.is_none() || prep.average_crosstalk_probability.is_none()) {
            return Err(input_error(
                "preparation accepts crosstalk_probability or average_crosstalk_probability, not both",
            ));
        }
        if let Some(value) = prep.probability {
            builder = builder.with_p_prep(value);
        }
        if let Some(value) = prep.leakage_ratio {
            builder = builder.with_prep_leak_ratio(value);
        }
        if let Some(value) = prep.crosstalk_probability {
            builder = builder.with_p_prep_crosstalk(value);
        }
        if let Some(value) = prep.average_crosstalk_probability {
            builder = builder.with_average_p_prep_crosstalk(value);
        }
        if let Some(value) = prep.scale {
            builder = builder.with_prep_scale(value);
        }
        if let Some(value) = prep.crosstalk_scale {
            builder = builder.with_p_prep_crosstalk_scale(value);
        }

        let one = &self.single_qubit;
        if !(one.probability.is_none() || one.average_infidelity.is_none()) {
            return Err(input_error(
                "single_qubit accepts probability or average_infidelity, not both",
            ));
        }
        if let Some(value) = one.probability {
            builder = builder.with_p1(value);
        }
        if let Some(value) = one.average_infidelity {
            builder = builder.with_average_p1(value);
        }
        if let Some(value) = &one.pauli_model {
            builder = builder.with_p1_pauli_model(value);
        }
        if let Some(value) = one.emission_ratio {
            builder = builder.with_p1_emission_ratio(value);
        }
        if let Some(value) = &one.emission_model {
            builder = builder.with_p1_emission_model(value);
        }
        if let Some(value) = one.seepage_probability {
            builder = builder.with_p1_seepage_prob(value);
        }
        if let Some(value) = one.scale {
            builder = builder.with_p1_scale(value);
        }

        let two = &self.two_qubit;
        if !(two.probability.is_none() || two.average_infidelity.is_none()) {
            return Err(input_error(
                "two_qubit accepts probability or average_infidelity, not both",
            ));
        }
        if let Some(value) = two.probability {
            builder = builder.with_p2(value);
        }
        if let Some(value) = two.average_infidelity {
            builder = builder.with_average_p2(value);
        }
        if let Some(value) = &two.pauli_model {
            builder = builder.with_p2_pauli_model(value);
        }
        if let Some(value) = two.emission_ratio {
            builder = builder.with_p2_emission_ratio(value);
        }
        if let Some(value) = &two.emission_model {
            builder = builder.with_p2_emission_model(value);
        }
        if let Some(value) = two.seepage_probability {
            builder = builder.with_p2_seepage_prob(value);
        }
        if let Some(value) = two.scale {
            builder = builder.with_p2_scale(value);
        }
        if let Some([a, b, c, d]) = two.angle_coefficients {
            builder = builder.with_p2_angle_params(a, b, c, d);
        }
        if let Some(value) = two.angle_power {
            builder = builder.with_p2_angle_power(value);
        }
        if let Some(value) = two.idle_after_gate {
            builder = builder.with_idle_after_2q(value);
        }

        for (hooks, two, before) in [
            (&one.hooks_before, false, true),
            (&one.hooks_after, false, false),
            (&two.hooks_before, true, true),
            (&two.hooks_after, true, false),
        ] {
            for hook in hooks {
                builder = hook.apply(builder, two, before)?;
            }
        }

        let idle = &self.idle;
        if let Some(rate) = idle.linear_rate {
            let model = idle.linear_model.clone().unwrap_or_else(|| {
                BTreeMap::from([
                    ("X".to_string(), 1.0 / 3.0),
                    ("Y".to_string(), 1.0 / 3.0),
                    ("Z".to_string(), 1.0 / 3.0),
                ])
            });
            builder = builder.with_p_idle_linear(rate, &model);
        } else if idle.linear_model.is_some() {
            return Err(input_error("idle.linear_model requires idle.linear_rate"));
        }
        if let Some(rate) = idle.sin_squared_rate {
            let model = idle
                .sin_squared_model
                .clone()
                .unwrap_or_else(|| BTreeMap::from([("Z".to_string(), 1.0)]));
            builder = builder.with_p_idle_sin_squared(rate, &model);
        } else if idle.sin_squared_model.is_some() {
            return Err(input_error(
                "idle.sin_squared_model requires idle.sin_squared_rate",
            ));
        }
        if let Some(rate) = idle.coherent_rate {
            let model = idle
                .coherent_model
                .clone()
                .unwrap_or_else(|| BTreeMap::from([("RZ".to_string(), 1.0)]));
            builder = builder.with_p_idle_coherent(rate, &model);
        } else if idle.coherent_model.is_some() {
            return Err(input_error(
                "idle.coherent_model requires idle.coherent_rate",
            ));
        }
        if let Some(value) = idle.scale {
            builder = builder.with_idle_scale(value);
        }

        let measurement = &self.measurement;
        if let Some(value) = measurement.p0_to_1 {
            builder = builder.with_p_meas_0(value);
        }
        if let Some(value) = measurement.p1_to_0 {
            builder = builder.with_p_meas_1(value);
        }
        if let Some(value) = measurement.global_crosstalk_probability {
            builder = builder.with_p_meas_crosstalk_global(value);
        }
        if let Some(value) = measurement.local_crosstalk_probability {
            builder = builder.with_p_meas_crosstalk_local(value);
        }
        if let Some(value) = &measurement.crosstalk_model {
            builder = builder.with_p_meas_crosstalk_model(value);
        }
        if let Some(value) = measurement.scale {
            builder = builder.with_meas_scale(value);
        }
        if let Some(value) = measurement.crosstalk_scale {
            builder = builder.with_p_meas_crosstalk_scale(value);
        }

        if let Some(message) = builder.setter_error() {
            return Err(input_error(message));
        }
        Ok(builder)
    }

    /// Build a model from this configuration and the engine defaults.
    ///
    /// # Errors
    /// Returns schema, setter, or cross-field configuration errors.
    pub fn build(&self) -> Result<GeneralNoiseModel, PecosError> {
        self.apply_to(GeneralNoiseModel::builder())?.try_build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hook_configuration_and_validation() {
        let valid = r#"{"single_qubit":{"hooks_before":[{"kind":"transition","probability":1.0,"transitions":{"L->0":1.0}}]}}"#;
        let config: GeneralNoiseConfig = serde_json::from_str(valid).unwrap();
        assert!(config.build().is_ok());
        assert!(
            serde_json::from_str::<GeneralNoiseConfig>(
                &valid.replace("\"probability\":1.0", "\"probability\":1.0,\"unknown\":0")
            )
            .is_err()
        );
        let invalid = valid.replace("L->0", "0->L");
        let config: GeneralNoiseConfig = serde_json::from_str(&invalid).unwrap();
        let expected = StateTransitionChannel::from_transitions(
            1.0,
            &BTreeMap::from([("0->L".to_string(), 1.0)]),
        )
        .unwrap_err()
        .to_string();
        assert_eq!(
            config.build().err().unwrap().to_string(),
            format!("Input error: {expected}")
        );
    }

    #[test]
    fn hook_json_round_trip_and_unknown_fields() {
        let values = [
            serde_json::json!({"kind":"transition","probability":0.5,"transitions":{"L->0":1.0}}),
            serde_json::json!({"kind":"transition","probability":0.5,"transitions":{"L":{"0":1.0}}}),
            serde_json::json!({"kind":"pauli_leakage","probability":0.5,"events":{"L":1.0}}),
        ];
        for value in values {
            let hook: HookConfig = serde_json::from_value(value.clone()).unwrap();
            assert_eq!(serde_json::to_value(hook).unwrap(), value);
            let independent =
                serde_json::json!({"kind":"independent","first":value,"second":value});
            let hook: HookConfig = serde_json::from_value(independent.clone()).unwrap();
            assert_eq!(serde_json::to_value(hook).unwrap(), independent);
            for mut invalid in [value, independent] {
                invalid["unknown"] = serde_json::json!(true);
                assert!(serde_json::from_value::<HookConfig>(invalid).is_err());
            }
        }
    }

    #[test]
    fn hook_json_numbers_parse_from_text() {
        for json in [
            r#"{"kind":"transition","probability":0.25,"transitions":{"L->0":1}}"#,
            r#"{"kind":"transition","probability":2.5e-1,"transitions":{"L":{"0":1.0}}}"#,
            r#"{"kind":"pauli_leakage","probability":0.25,"events":{"L":1}}"#,
        ] {
            let hook: HookConfig = serde_json::from_str(json).unwrap();
            let expected: serde_json::Value = serde_json::from_str(json).unwrap();
            // Keep the Value-based deserializer: opt-in mwpf workspace builds enable
            // serde_json/arbitrary_precision, which preserves number spelling.
            let encoded = serde_json::to_value(hook).unwrap();
            assert_eq!(encoded["probability"].as_f64(), Some(0.25));
            assert_eq!(encoded["kind"], expected["kind"]);
            let independent = serde_json::json!({
                "kind": "independent", "first": expected, "second": expected
            });
            let hook: HookConfig = serde_json::from_str(&independent.to_string()).unwrap();
            assert!(
                hook.apply(super::GeneralNoiseModel::builder(), true, false)
                    .is_ok()
            );
        }
    }

    #[test]
    fn hook_json_rejects_non_numeric_probabilities_and_missing_kind() {
        for json in [
            r#"{"kind":"transition","probability":"0.5","transitions":{"L->0":1}}"#,
            r#"{"kind":"pauli_leakage","probability":false,"events":{"L":1}}"#,
            r#"{"kind":"pauli_leakage","probability":null,"events":{"L":1}}"#,
            r#"{"kind":"transition","probability":1,"transitions":{"L->0":"1"}}"#,
            r#"{"kind":"transition","probability":1,"transitions":{"L":{"0":false}}}"#,
            r#"{"kind":"pauli_leakage","probability":1,"events":{"L":[]}}"#,
            r#"{"kind":"transition","probability":1,"transitions":{"L":{"0":1},"0->0":1}}"#,
            r#"{"probability":1,"events":{"L":1}}"#,
        ] {
            assert!(serde_json::from_str::<HookConfig>(json).is_err(), "{json}");
            let value: serde_json::Value = serde_json::from_str(json).unwrap();
            for leg in ["first", "second"] {
                let valid = serde_json::json!({
                    "kind": "pauli_leakage", "probability": 1, "events": {"L": 1}
                });
                let mut independent = serde_json::json!({
                    "kind": "independent", "first": valid, "second": valid
                });
                independent[leg] = value.clone();
                assert!(
                    serde_json::from_str::<HookConfig>(&independent.to_string()).is_err(),
                    "{independent}"
                );
            }
        }
        let error = serde_json::from_str::<HookConfig>(r#"{"probability":1}"#).unwrap_err();
        assert!(error.to_string().contains("missing field `kind`"));
    }

    #[test]
    fn scaled_hook_probabilities_are_rejected() {
        for site in ["single_qubit", "two_qubit"] {
            for timing in ["hooks_before", "hooks_after"] {
                let hook = serde_json::json!({"kind":"pauli_leakage","probability":0.6,"events":{"L":1.0}});
                let mut hooks = vec![hook.clone()];
                if site == "two_qubit" {
                    hooks
                        .push(serde_json::json!({"kind":"independent","first":hook,"second":hook}));
                }
                for hook in hooks {
                    let value = serde_json::json!({site:{timing:[hook],"scale":2.0}});
                    let config: GeneralNoiseConfig = serde_json::from_value(value).unwrap();
                    assert!(config.build().is_err());
                }
            }
        }
    }

    #[test]
    fn rich_configuration_builds_the_current_pecos_model() {
        let json = r#"{
            "preparation":{
                "probability":0.001,
                "leakage_ratio":0.2,
                "average_crosstalk_probability":0.0001
            },
            "measurement":{
                "p0_to_1":0.002,
                "p1_to_0":0.003,
                "global_crosstalk_probability":0.0001,
                "local_crosstalk_probability":0.0002,
                "crosstalk_model":{
                    "0->0":0.9,"0->1":0.1,
                    "1->0":0.2,"1->1":0.8
                }
            },
            "single_qubit":{"average_infidelity":0.0001},
            "two_qubit":{
                "average_infidelity":0.001,
                "angle_coefficients":[1.0,0.0,1.0,0.0],
                "angle_power":1.0,
                "idle_after_gate":0.000005
            },
            "idle":{
                "linear_rate":0.01,
                "sin_squared_rate":0.02,
                "coherent_rate":0.03
            },
            "scaling":{"overall":0.5,"leakage":0.75,"emission":1.25},
            "noiseless_gates":["RZ"]
        }"#;
        let config: GeneralNoiseConfig = serde_json::from_str(json).unwrap();
        config.build().unwrap();
    }

    const ALL_MODEL_FIELDS: &str = r#"{
        "scaling":{"overall":0.51,"leakage":0.52,"emission":0.53},
        "preparation":{
            "probability":0.011,"leakage_ratio":0.012,
            "crosstalk_probability":0.013,"scale":0.54,"crosstalk_scale":0.55
        },
        "single_qubit":{
            "probability":0.014,"pauli_model":{"X":0.21,"Y":0.79},
            "emission_ratio":0.015,"emission_model":{"Z":0.22,"L":0.78},
            "seepage_probability":0.016,"scale":0.56
        },
        "two_qubit":{
            "probability":0.017,"pauli_model":{"XI":0.23,"YZ":0.77},
            "emission_ratio":0.018,"emission_model":{"IZ":0.24,"LL":0.76},
            "seepage_probability":0.019,"scale":0.57,
            "angle_coefficients":[0.31,0.32,0.33,0.34],
            "angle_power":0.35,"idle_after_gate":0.036
        },
        "idle":{
            "linear_rate":0.041,"linear_model":{"Y":0.25,"L":0.75},
            "sin_squared_rate":0.042,"sin_squared_model":{"X":0.26,"Z":0.74},
            "coherent_rate":0.043,"coherent_model":{"RX":0.27,"RY":0.73},
            "scale":0.58
        },
        "measurement":{
            "p0_to_1":0.044,"p1_to_0":0.045,
            "global_crosstalk_probability":0.046,"local_crosstalk_probability":0.047,
            "crosstalk_model":{"0->0":0.28,"0->1":0.72,"1->0":0.29,"1->1":0.71},
            "scale":0.59,"crosstalk_scale":0.61
        },
        "noiseless_gates":["RZ","H"]
    }"#;

    fn assert_scalar(name: &str, actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "{name}: expected {expected}, got {actual}"
        );
    }

    fn assert_distribution(actual: &BTreeMap<String, f64>, expected: [(&str, f64); 2]) {
        assert_eq!(actual.len(), expected.len());
        for (key, weight) in expected {
            assert_scalar(key, actual[key], weight);
        }
    }

    #[test]
    fn every_json_field_maps_to_the_expected_model_value() {
        // Model fields are visible to this child module; the sibling builder's
        // fields are private. Check explicit scaled results rather than another
        // builder path, which could duplicate the same mapping mistake.
        let model = GeneralNoiseConfig::from_json(ALL_MODEL_FIELDS)
            .unwrap()
            .build()
            .unwrap();
        for (name, actual, expected) in [
            ("leakage", model.leakage_scale, 0.52),
            ("preparation", model.p_prep, 0.011 * 0.54 * 0.51),
            ("preparation leakage", model.p_prep_leak_ratio, 0.012 * 0.51),
            (
                "preparation crosstalk",
                model.p_prep_crosstalk,
                0.013 * 0.55 * 0.54 * 0.51,
            ),
            ("single-qubit probability", model.p1, 0.014 * 0.56 * 0.51),
            (
                "single-qubit emission",
                model.p1_emission_ratio,
                0.015 * 0.53 * 0.51,
            ),
            ("single-qubit seepage", model.p1_seepage_prob, 0.016),
            ("two-qubit probability", model.p2, 0.017 * 0.57 * 0.51),
            (
                "two-qubit emission",
                model.p2_emission_ratio,
                0.018 * 0.53 * 0.51,
            ),
            ("two-qubit seepage", model.p2_seepage_prob, 0.019),
            ("angle a", model.p2_angle_a, 0.31),
            ("angle b", model.p2_angle_b, 0.32),
            ("angle c", model.p2_angle_c, 0.33),
            ("angle d", model.p2_angle_d, 0.34),
            ("angle power", model.p2_angle_power, 0.35),
            ("idle after gate", model.idle_after_2q, 0.036),
            ("linear idle", model.p_idle_linear_rate, 0.041 * 0.51 * 0.58),
            (
                "sine-squared idle",
                model.p_idle_sin_squared_rate,
                0.042 * 0.51 * 0.58,
            ),
            (
                "coherent idle",
                model.p_idle_coherent_rate,
                0.043 * 0.51 * 0.58,
            ),
            ("measurement 0 to 1", model.p_meas_0, 0.044 * 0.59 * 0.51),
            ("measurement 1 to 0", model.p_meas_1, 0.045 * 0.59 * 0.51),
            (
                "global crosstalk",
                model.p_meas_crosstalk_global,
                0.046 * 0.61 * 0.59 * 0.51,
            ),
            (
                "local crosstalk",
                model.p_meas_crosstalk_local,
                0.047 * 0.61 * 0.59 * 0.51,
            ),
        ] {
            assert_scalar(name, actual, expected);
        }
        assert_distribution(
            model.p1_pauli_model.get_weighted_map(),
            [("X", 0.21), ("Y", 0.79)],
        );
        assert_distribution(
            model.p1_emission_model.get_weighted_map(),
            [("Z", 0.22), ("L", 0.78)],
        );
        assert_distribution(
            model.p2_pauli_model.get_weighted_map(),
            [("XI", 0.23), ("YZ", 0.77)],
        );
        assert_distribution(
            model.p2_emission_model.get_weighted_map(),
            [("IZ", 0.24), ("LL", 0.76)],
        );
        assert_distribution(
            model.p_idle_linear_model.get_weighted_map(),
            [("Y", 0.25), ("L", 0.75)],
        );
        assert_distribution(&model.p_idle_sin_squared_model, [("X", 0.26), ("Z", 0.74)]);
        assert_distribution(&model.p_idle_coherent_model, [("RX", 0.27), ("RY", 0.73)]);
        assert_distribution(
            model.p_meas_crosstalk_model.get_weighted_map(0),
            [("0->0", 0.28), ("0->1", 0.72)],
        );
        assert_distribution(
            model.p_meas_crosstalk_model.get_weighted_map(1),
            [("1->0", 0.29), ("1->1", 0.71)],
        );
        assert_eq!(
            model.noiseless_gates,
            std::collections::BTreeSet::from([GateType::RZ, GateType::H])
        );
    }

    #[test]
    fn average_json_fields_map_to_converted_model_probabilities() {
        // Each mutually exclusive alternative is exercised independently.
        let prep = GeneralNoiseConfig::from_json(
            r#"{"preparation":{"average_crosstalk_probability":0.048}}"#,
        )
        .unwrap()
        .build()
        .unwrap();
        assert_scalar(
            "average preparation crosstalk",
            prep.p_prep_crosstalk,
            0.048 * 18.0 / 5.0,
        );
        let one = GeneralNoiseConfig::from_json(r#"{"single_qubit":{"average_infidelity":0.049}}"#)
            .unwrap()
            .build()
            .unwrap();
        assert_scalar("single-qubit average infidelity", one.p1, 0.049 * 1.5);
        let two = GeneralNoiseConfig::from_json(r#"{"two_qubit":{"average_infidelity":0.050}}"#)
            .unwrap()
            .build()
            .unwrap();
        assert_scalar("two-qubit average infidelity", two.p2, 0.050 * 1.25);
    }

    #[test]
    fn unknown_configuration_fields_are_rejected() {
        let error = serde_json::from_str::<GeneralNoiseConfig>(r#"{"device_profile":"example"}"#)
            .expect_err("device-specific fields must not enter the generic API");
        assert!(error.to_string().contains("unknown field"));
    }
    #[test]
    fn config_reports_each_former_setter_panic_as_input() {
        for (json, expected) in [
            (
                r#"{"single_qubit":{"probability":1.5}}"#,
                "Probability must be between 0 and 1",
            ),
            (
                r#"{"preparation":{"crosstalk_scale":-1}}"#,
                "Preparation crosstalk rescale factor must be non-negative",
            ),
            (
                r#"{"measurement":{"crosstalk_scale":-1}}"#,
                "Measurement crosstalk rescale factor must be non-negative",
            ),
            (
                r#"{"single_qubit":{"pauli_model":{"bad":1}}}"#,
                "invalid key 'bad'",
            ),
            (
                r#"{"single_qubit":{"emission_model":{"X":2}}}"#,
                "total weight 2 deviates",
            ),
            (
                r#"{"two_qubit":{"pauli_model":{"X":1}}}"#,
                "must be exactly 2 characters",
            ),
            (
                r#"{"two_qubit":{"emission_model":{"II":1}}}"#,
                "key 'II' is not allowed",
            ),
            (
                r#"{"two_qubit":{"pauli_model":{"XA":1}}}"#,
                "invalid character 'A'",
            ),
            (
                r#"{"two_qubit":{"emission_model":{"XI":-1}}}"#,
                "weights must be finite and non-negative",
            ),
            (
                r#"{"measurement":{"crosstalk_model":{"bad":1}}}"#,
                "invalid key 'bad'",
            ),
            (
                r#"{"measurement":{"crosstalk_model":{"0->1":1}}}"#,
                "weighted_map cannot be empty",
            ),
            (
                r#"{"measurement":{"crosstalk_model":{"0->1":1,"1->0":2}}}"#,
                "total weight 2 deviates",
            ),
            (
                r#"{"idle":{"linear_rate":-1}}"#,
                "linear idling rate must be non-negative",
            ),
            (
                r#"{"idle":{"linear_rate":1,"linear_model":{"X":2}}}"#,
                "total weight 2 deviates",
            ),
            (
                r#"{"idle":{"sin_squared_rate":-1}}"#,
                "sine-squared idling rate must be finite and non-negative",
            ),
            (
                r#"{"idle":{"sin_squared_rate":1,"sin_squared_model":{"A":1}}}"#,
                "invalid key 'A'",
            ),
            (
                r#"{"idle":{"sin_squared_rate":1,"sin_squared_model":{"X":-1}}}"#,
                "multiplier for 'X' must be finite and non-negative",
            ),
            (
                r#"{"idle":{"coherent_rate":-1}}"#,
                "coherent idling rate must be finite and non-negative",
            ),
            (
                r#"{"idle":{"coherent_rate":1,"coherent_model":{"L":1}}}"#,
                "invalid key 'L'",
            ),
            (
                r#"{"idle":{"coherent_rate":1,"coherent_model":{"RX":-1}}}"#,
                "multiplier for 'RX' must be finite and non-negative",
            ),
            (
                r#"{"two_qubit":{"angle_power":0}}"#,
                "RZZ power parameter must be positive",
            ),
            (
                r#"{"two_qubit":{"idle_after_gate":-1}}"#,
                "Duration must be finite and non-negative",
            ),
            (
                r#"{"preparation":{"average_crosstalk_probability":0.3,"scale":0.01}}"#,
                "average preparation crosstalk probability must be between 0 and 5/18",
            ),
        ] {
            let config = GeneralNoiseConfig::from_json(json).unwrap();
            for error in [
                config.build().err().unwrap(),
                config.apply_to(GeneralNoiseModel::builder()).unwrap_err(),
            ] {
                assert!(
                    matches!(error, PecosError::Input(ref message) if message.contains(expected)),
                    "{json}: {error}"
                );
            }
        }
    }

    #[test]
    fn apply_preserves_unset_fields_and_defers_cross_field_checks() {
        let original = GeneralNoiseModel::builder().with_p1(0.8).with_p_meas_0(0.2);
        let config = GeneralNoiseConfig::from_json(r#"{"scaling":{"overall":2}}"#).unwrap();
        let intermediate = config.apply_to(original).unwrap();
        assert!(intermediate.setter_error().is_none());
        assert!(intermediate.validate_configuration().is_err());
        let model = intermediate.with_p1_scale(0.5).try_build().unwrap();
        assert!((model.p1 - 0.8).abs() < f64::EPSILON);
        assert!((model.p_meas_0 - 0.4).abs() < f64::EPSILON);
        let config =
            GeneralNoiseConfig::from_json(r#"{"single_qubit":{"probability":0.1}}"#).unwrap();
        let error = config
            .apply_to(GeneralNoiseModel::builder().with_p1(1.5))
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "Input error: Probability must be between 0 and 1, got 1.5"
        );
        assert!(
            GeneralNoiseConfig::from_json(r#"{"scaling":{"overall":-1}}"#)
                .unwrap()
                .build()
                .is_err()
        );
    }

    #[test]
    fn idle_rate_defaults_and_schema_rules_are_preserved() {
        let config = GeneralNoiseConfig::from_json(
            r#"{"idle":{"linear_rate":0.1,"sin_squared_rate":0.2,"coherent_rate":0.3}}"#,
        )
        .unwrap();
        let model = config.build().unwrap();
        assert_eq!(
            model.p_idle_linear_model.get_weighted_map(),
            &BTreeMap::from([
                ("X".to_string(), 1.0 / 3.0),
                ("Y".to_string(), 1.0 / 3.0),
                ("Z".to_string(), 1.0 / 3.0),
            ])
        );
        assert_eq!(
            model.p_idle_sin_squared_model,
            BTreeMap::from([("Z".to_string(), 1.0)])
        );
        assert_eq!(
            model.p_idle_coherent_model,
            BTreeMap::from([("RZ".to_string(), 1.0)])
        );
        assert!((model.p_idle_linear_rate - 0.1).abs() < f64::EPSILON);
        assert!((model.p_idle_sin_squared_rate - 0.2).abs() < f64::EPSILON);
        assert!((model.p_idle_coherent_rate - 0.3).abs() < f64::EPSILON);
        for json in [
            r#"{"preparation":{"crosstalk_probability":0.1,"average_crosstalk_probability":0.1}}"#,
            r#"{"single_qubit":{"probability":0.1,"average_infidelity":0.1}}"#,
            r#"{"two_qubit":{"probability":0.1,"average_infidelity":0.1}}"#,
            r#"{"idle":{"linear_model":{"X":1}}}"#,
            r#"{"idle":{"sin_squared_model":{"Z":1}}}"#,
            r#"{"idle":{"coherent_model":{"RZ":1}}}"#,
            r#"{"noiseless_gates":["invalid"]}"#,
        ] {
            let error = GeneralNoiseConfig::from_json(json)
                .unwrap()
                .build()
                .err()
                .unwrap();
            assert!(matches!(error, PecosError::Input(_)), "{json}: {error}");
        }
        for json in [
            "{",
            r#"{"measurement":{"local_groups":[]}}"#,
            r#"{"idle":{"unknown":1}}"#,
        ] {
            assert!(matches!(
                GeneralNoiseConfig::from_json(json),
                Err(PecosError::Input(_))
            ));
        }
    }
}
