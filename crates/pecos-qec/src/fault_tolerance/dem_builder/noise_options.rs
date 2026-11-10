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

//! High-level DEM noise options shared by language bindings and Rust callers.

use super::{
    IdleNoiseFamily, MeasurementCrosstalkDemMode, MeasurementCrosstalkTransitionModel, NoiseConfig,
    PAULI_2Q_ORDER, PauliWeights, ReplacementBranchApproximation,
};
use pecos_core::gate_type::GateType;
use std::collections::{BTreeMap, BTreeSet};
use std::str::FromStr;

// Match NORMALIZATION_TOLERANCE and FLOAT_EPSILON in pecos-engines/src/noise/weighted_sampler.rs.
const NORMALIZATION_TOLERANCE: f64 = 1e-5;
const FLOAT_EPSILON: f64 = 1e-10;

/// High-level noise arguments before parsing labels and translating idle families.
///
/// Defaults disable noise. Structured models normalize in key order, like the engines.
#[derive(Debug, Clone, Default)]
pub struct NoiseOptions {
    /// Single-qubit gate error rate.
    pub p1: f64,
    /// Two-qubit gate error rate.
    pub p2: f64,
    /// Measurement error rate.
    pub p_meas: f64,
    /// Preparation error rate.
    pub p_prep: f64,
    /// Depolarizing base idle rate per time unit.
    pub p_idle: Option<f64>,
    /// Relaxation time; requires `t2`.
    pub t1: Option<f64>,
    /// Dephasing time; requires `t1`.
    pub t2: Option<f64>,
    /// Coherent RZ angle per time unit, converted by `NoiseConfig::set_idle_rz`.
    pub idle_rz: Option<f64>,
    /// Legacy Z-axis linear alias; an explicit Z-axis rate takes precedence.
    pub p_idle_linear_rate: Option<f64>,
    /// Legacy Z-axis quadratic alias; an explicit Z-axis rate takes precedence.
    pub p_idle_quadratic_rate: Option<f64>,
    /// The `p_idle_x_linear_rate` noise option.
    pub p_idle_x_linear_rate: Option<f64>,
    /// The `p_idle_y_linear_rate` noise option.
    pub p_idle_y_linear_rate: Option<f64>,
    /// The `p_idle_z_linear_rate` noise option.
    pub p_idle_z_linear_rate: Option<f64>,
    /// The `p_idle_x_quadratic_rate` noise option.
    pub p_idle_x_quadratic_rate: Option<f64>,
    /// The `p_idle_y_quadratic_rate` noise option.
    pub p_idle_y_quadratic_rate: Option<f64>,
    /// The `p_idle_z_quadratic_rate` noise option.
    pub p_idle_z_quadratic_rate: Option<f64>,
    /// Legacy Z-axis sine-squared alias; an explicit Z-axis rate takes precedence.
    pub p_idle_quadratic_sine_rate: Option<f64>,
    /// The `p_idle_x_quadratic_sine_rate` noise option.
    pub p_idle_x_quadratic_sine_rate: Option<f64>,
    /// The `p_idle_y_quadratic_sine_rate` noise option.
    pub p_idle_y_quadratic_sine_rate: Option<f64>,
    /// The `p_idle_z_quadratic_sine_rate` noise option.
    pub p_idle_z_quadratic_sine_rate: Option<f64>,
    /// Relative single-qubit Pauli weights, keyed by label.
    pub p1_weights: Option<BTreeMap<String, f64>>,
    /// Relative two-qubit Pauli weights, including replacement labels.
    pub p2_weights: Option<BTreeMap<String, f64>>,
    /// Named approximation for replacement branches.
    pub p2_replacement_approximation: Option<String>,
    /// Local measurement-crosstalk event rate.
    pub p_meas_crosstalk_local: Option<f64>,
    /// Global measurement-crosstalk event rate.
    pub p_meas_crosstalk_global: Option<f64>,
    /// Hidden-measurement transitions keyed by state transition label.
    pub p_meas_crosstalk_model: Option<BTreeMap<String, f64>>,
    /// Named policy for rendering measurement crosstalk in the DEM.
    pub measurement_crosstalk_dem_mode: Option<String>,
    /// Two-qubit rate overrides, keyed by scheduled gate name.
    pub p2_gate_rates: Option<BTreeMap<String, f64>>,
    /// Single-qubit rate overrides, keyed by scheduled gate name.
    pub p1_gate_rates: Option<BTreeMap<String, f64>>,
    /// Structured categorical Pauli rate per time unit.
    pub p_idle_linear: Option<f64>,
    /// Relative Pauli multipliers for the linear family.
    pub p_idle_linear_model: Option<BTreeMap<String, f64>>,
    /// Structured sine-squared family rate.
    pub p_idle_sin_squared: Option<f64>,
    /// Relative-rate multipliers for this family.
    pub p_idle_sin_squared_model: Option<BTreeMap<String, f64>>,
    /// Structured coherent rate; only zero is representable by the standard DEM.
    pub p_idle_coherent: Option<f64>,
    /// Relative-rate multipliers for this family.
    pub p_idle_coherent_model: Option<BTreeMap<String, f64>>,
}

impl TryFrom<NoiseOptions> for NoiseConfig {
    type Error = String;

    fn try_from(options: NoiseOptions) -> Result<Self, Self::Error> {
        let idle_noise_parameters = options.idle_noise_parameters();
        let [
            p_idle_x_linear_rate,
            p_idle_y_linear_rate,
            p_idle_z_linear_rate,
            p_idle_x_quadratic_sine_rate,
            p_idle_y_quadratic_sine_rate,
            p_idle_z_quadratic_sine_rate,
        ] = options.translate_idle_noise()?;
        let NoiseOptions {
            p1,
            p2,
            p_meas,
            p_prep,
            p_idle,
            t1,
            t2,
            idle_rz,
            p_idle_linear_rate,
            p_idle_quadratic_rate,
            p_idle_x_quadratic_rate,
            p_idle_y_quadratic_rate,
            p_idle_z_quadratic_rate,
            p_idle_quadratic_sine_rate,
            p1_weights,
            p2_weights,
            p2_replacement_approximation,
            p_meas_crosstalk_local,
            p_meas_crosstalk_global,
            p_meas_crosstalk_model,
            measurement_crosstalk_dem_mode,
            p2_gate_rates,
            p1_gate_rates,
            ..
        } = options;
        let mut noise = NoiseConfig::new(p1, p2, p_meas, p_prep);
        // Reject the base-idle-channel combinations this function would otherwise
        // resolve silently: `set_t1_t2` makes T1/T2 the base channel that shadows
        // `p_idle`, and `set_idle_rz` zeroes `p_idle` and overwrites T1/T2 with a
        // synthetic T2. Each combination discards a caller-supplied rate without
        // any signal (issue #426).
        if p_idle.is_some() && (t1.is_some() || t2.is_some()) {
            return Err(String::from(
                "p_idle cannot be combined with t1/t2; the T1/T2 channel replaces the \
             depolarizing base idle channel, so p_idle would be ignored",
            ));
        }
        if idle_rz.is_some() {
            if p_idle.is_some() {
                return Err(String::from(
                    "idle_rz cannot be combined with p_idle; the coherent RZ conversion \
                 replaces the base idle channel, so p_idle would be ignored",
                ));
            }
            if t1.is_some() || t2.is_some() {
                return Err(String::from(
                    "idle_rz cannot be combined with t1/t2; the coherent RZ conversion \
                 overwrites the T1/T2 channel with an equivalent T2",
                ));
            }
        }

        if t1.is_some() && t2.is_none() {
            return Err("t1 requires t2; both relaxation times must be supplied".to_string());
        }
        if t2.is_some() && t1.is_none() {
            return Err("t2 requires t1; both relaxation times must be supplied".to_string());
        }
        noise.p_idle = p_idle.unwrap_or(0.0);
        // Channel validation belongs to the DEM builder, including idle-target
        // checks. The infallible setter asserts the T2/T1 bound prematurely.
        noise.t1 = t1;
        noise.t2 = t2;
        if let Some(rz) = idle_rz {
            noise = noise.set_idle_rz(rz);
        }
        noise.p_idle_linear = idle_family_from_axis_rates(
            p_idle_x_linear_rate.unwrap_or(0.0),
            p_idle_y_linear_rate.unwrap_or(0.0),
            p_idle_z_linear_rate.or(p_idle_linear_rate).unwrap_or(0.0),
        );
        noise.p_idle_quadratic = idle_family_from_axis_rates(
            p_idle_x_quadratic_rate.unwrap_or(0.0),
            p_idle_y_quadratic_rate.unwrap_or(0.0),
            p_idle_z_quadratic_rate
                .or(p_idle_quadratic_rate)
                .unwrap_or(0.0),
        );
        noise.p_idle_quadratic_sine = idle_family_from_axis_rates(
            p_idle_x_quadratic_sine_rate.unwrap_or(0.0),
            p_idle_y_quadratic_sine_rate.unwrap_or(0.0),
            p_idle_z_quadratic_sine_rate
                .or(p_idle_quadratic_sine_rate)
                .unwrap_or(0.0),
        );
        if let Some(weights) = p1_weights {
            noise = noise.set_p1_weights(parse_p1_weights(weights)?);
        }
        if let Some(weights) = p2_weights {
            noise = noise.set_p2_weights(parse_p2_weights(weights)?);
        }
        if let Some(rates) = p2_gate_rates {
            for (gate_type, rate) in parse_p2_gate_rates(rates)? {
                noise = noise.set_p2_gate_rate(gate_type, rate);
            }
        }
        if let Some(rates) = p1_gate_rates {
            for (gate_type, rate) in parse_p1_gate_rates(rates)? {
                noise = noise.set_p1_gate_rate(gate_type, rate);
            }
        }
        noise = noise.set_p2_replacement_approximation(parse_replacement_approximation(
            p2_replacement_approximation,
        )?);
        if let Some(rate) = p_meas_crosstalk_local {
            noise = noise.set_measurement_crosstalk_local_rate(rate);
        }
        if let Some(rate) = p_meas_crosstalk_global {
            noise = noise.set_measurement_crosstalk_global_rate(rate);
        }
        noise = noise.set_measurement_crosstalk_transition_model(
            parse_measurement_crosstalk_transition_model(p_meas_crosstalk_model)?,
        );
        noise = noise.set_measurement_crosstalk_dem_mode(parse_measurement_crosstalk_dem_mode(
            measurement_crosstalk_dem_mode,
        )?);
        noise.idle_noise_parameters = idle_noise_parameters;
        Ok(noise)
    }
}

fn idle_family_from_axis_rates(px: f64, py: f64, pz: f64) -> IdleNoiseFamily {
    if px == 0.0 && py == 0.0 && pz == 0.0 {
        return IdleNoiseFamily::default();
    }
    IdleNoiseFamily::new(
        1.0,
        BTreeMap::from([
            ("X".to_string(), px),
            ("Y".to_string(), py),
            ("Z".to_string(), pz),
        ]),
    )
}

fn parse_p1_weights(weights: BTreeMap<String, f64>) -> Result<PauliWeights, String> {
    use pecos_core::pauli::{X, Y, Z};

    let mut entries = Vec::with_capacity(weights.len());
    let mut sum = 0.0;
    for (label, weight) in weights {
        let label = label.trim().to_ascii_uppercase();
        let pauli = match label.as_str() {
            "X" => X(0),
            "Y" => Y(0),
            "Z" => Z(0),
            _ => {
                let msg = format!("p1_weights keys must be one of ['X', 'Y', 'Z'], got {label:?}");
                return Err(msg);
            }
        };
        if !weight.is_finite() || weight < 0.0 {
            let msg =
                format!("p1_weights[{label:?}] must be finite and non-negative, got {weight}");
            return Err(msg);
        }
        sum += weight;
        entries.push((pauli, weight));
    }
    if (sum - 1.0).abs() >= 1.0e-6 {
        let msg = format!("p1_weights relative probabilities must sum to 1.0, got {sum}");
        return Err(msg);
    }
    Ok(PauliWeights::new(entries))
}

fn parse_p2_weights(weights: BTreeMap<String, f64>) -> Result<PauliWeights, String> {
    use pecos_core::pauli::{X, Y, Z};

    let mut entries = Vec::with_capacity(weights.len());
    let mut replacement_entries = Vec::new();
    let mut normalized_labels = BTreeSet::new();
    let mut sum = 0.0;
    for (label, weight) in weights {
        let input_label = label.trim().to_ascii_uppercase();
        let (replacement, label) = if let Some(stripped) = input_label.strip_prefix(":REPLACE:") {
            (true, stripped.to_string())
        } else if let Some(stripped) = input_label.strip_prefix('~') {
            (true, stripped.to_string())
        } else if let Some(stripped) = input_label.strip_prefix('*') {
            let replacement = format!("~{stripped}");
            let msg = format!(
                "p2_weights replacement label {input_label:?} uses the removed '*' syntax; use {replacement:?} (or \":replace:{stripped}\") instead"
            );
            return Err(msg);
        } else {
            (false, input_label.clone())
        };
        if !normalized_labels.insert((replacement, label.clone())) {
            let canonical = if replacement {
                format!("~{label}")
            } else {
                label.clone()
            };
            let msg = format!(
                "p2_weights contains duplicate label {canonical:?} after normalization; use only one spelling for each branch"
            );
            return Err(msg);
        }
        let identity_pair = label == "II";
        if !identity_pair && !PAULI_2Q_ORDER.contains(&label.as_str()) {
            let msg = format!(
                "p2_weights keys must be one of {PAULI_2Q_ORDER:?}, or use '~' / ':replace:' before a two-qubit Pauli label for a replacement branch; got {input_label:?}"
            );
            return Err(msg);
        }
        if !weight.is_finite() || weight < 0.0 {
            let msg = format!(
                "p2_weights[{input_label:?}] must be finite and non-negative, got {weight}"
            );
            return Err(msg);
        }
        let mut pauli = None;
        for (qubit, ch) in label.chars().enumerate() {
            let term = match ch {
                'I' => None,
                'X' => Some(X(qubit)),
                'Y' => Some(Y(qubit)),
                'Z' => Some(Z(qubit)),
                _ => unreachable!("validated p2_weights label contains only I/X/Y/Z"),
            };
            pauli = match (pauli, term) {
                (None, None) => None,
                (Some(existing), None) => Some(existing),
                (None, Some(term)) => Some(term),
                (Some(existing), Some(term)) => Some(existing & term),
            };
        }
        let pauli = if let Some(pauli) = pauli {
            pauli
        } else if replacement {
            pecos_core::PauliString::identity()
        } else {
            return Err(String::from(
                "plain p2_weights cannot contain identity pair 'II'; use '~II' (or ':replace:II') for a replacement branch that only omits the gate",
            ));
        };
        sum += weight;
        if replacement {
            replacement_entries.push((pauli, weight));
        } else {
            entries.push((pauli, weight));
        }
    }
    if (sum - 1.0).abs() >= 1.0e-6 {
        let msg = format!("p2_weights relative probabilities must sum to 1.0, got {sum}");
        return Err(msg);
    }
    Ok(PauliWeights::with_replacement(entries, replacement_entries))
}

fn parse_replacement_approximation(
    value: Option<String>,
) -> Result<ReplacementBranchApproximation, String> {
    let Some(value) = value else {
        return Ok(ReplacementBranchApproximation::default());
    };
    match value
        .trim()
        .to_ascii_lowercase()
        .replace(['-', ' '], "_")
        .as_str()
    {
        "pauli_twirl_omitted_gate" | "pauli_twirl" | "twirl" => {
            Ok(ReplacementBranchApproximation::PauliTwirlOmittedGate)
        }
        "branch_impact" | "replacement_branch_impact" | "impact" => {
            Ok(ReplacementBranchApproximation::BranchImpact)
        }
        "exact_branch_replay" | "exact_replay" | "exact_branch" | "exact" => {
            Ok(ReplacementBranchApproximation::ExactBranchReplay)
        }
        "ignore_gate_removal" | "ignore_removal" | "post_gate" | "postgate" => {
            Ok(ReplacementBranchApproximation::IgnoreGateRemoval)
        }
        _ => Err(String::from(
            "p2_replacement_approximation must be 'pauli_twirl_omitted_gate', 'branch_impact', 'exact_branch_replay', or 'ignore_gate_removal'",
        )),
    }
}

fn parse_p2_gate_rates(rates: BTreeMap<String, f64>) -> Result<BTreeMap<GateType, f64>, String> {
    let mut parsed = BTreeMap::new();
    for (label, rate) in rates {
        if !rate.is_finite() || rate < 0.0 {
            let msg =
                format!("p2_gate_rates[{label:?}] must be finite and non-negative, got {rate}");
            return Err(msg);
        }
        let gate_type = GateType::from_str(label.trim())
            .map_err(|err| format!("unsupported p2_gate_rates gate label {label:?}: {err}"))?;
        if !gate_type.is_two_qubit() {
            let msg = format!("p2_gate_rates keys must name two-qubit gates, got {label:?}");
            return Err(msg);
        }
        parsed.insert(gate_type, rate);
    }
    Ok(parsed)
}

fn parse_p1_gate_rates(rates: BTreeMap<String, f64>) -> Result<BTreeMap<GateType, f64>, String> {
    let mut parsed = BTreeMap::new();
    for (label, rate) in rates {
        if !rate.is_finite() || rate < 0.0 {
            let msg =
                format!("p1_gate_rates[{label:?}] must be finite and non-negative, got {rate}");
            return Err(msg);
        }
        let gate_type = GateType::from_str(label.trim())
            .map_err(|err| format!("unsupported p1_gate_rates gate label {label:?}: {err}"))?;
        if !gate_type.is_single_qubit() {
            let msg = format!("p1_gate_rates keys must name single-qubit gates, got {label:?}");
            return Err(msg);
        }
        parsed.insert(gate_type, rate);
    }
    Ok(parsed)
}

fn parse_measurement_crosstalk_dem_mode(
    value: Option<String>,
) -> Result<MeasurementCrosstalkDemMode, String> {
    let Some(value) = value else {
        return Ok(MeasurementCrosstalkDemMode::default());
    };
    match value
        .trim()
        .to_ascii_lowercase()
        .replace(['-', ' '], "_")
        .as_str()
    {
        "omitted" | "omit" | "none" | "off" => Ok(MeasurementCrosstalkDemMode::Omitted),
        "exact_deterministic" | "exact" | "deterministic" => {
            Ok(MeasurementCrosstalkDemMode::ExactDeterministic)
        }
        "exact_deterministic_leakage_as_depolarizing"
        | "exact_leakage_as_depolarizing"
        | "deterministic_leakage_as_depolarizing"
        | "leakage_as_depolarizing" => {
            Ok(MeasurementCrosstalkDemMode::ExactDeterministicLeakageAsDepolarizing)
        }
        "averaged_hidden_leakage_as_depolarizing"
        | "average_hidden_leakage_as_depolarizing"
        | "state_averaged_leakage_as_depolarizing"
        | "averaged_leakage_as_depolarizing" => {
            Ok(MeasurementCrosstalkDemMode::AveragedHiddenLeakageAsDepolarizing)
        }
        _ => Err(String::from(
            "measurement_crosstalk_dem_mode must be 'omitted', 'exact_deterministic', 'exact_deterministic_leakage_as_depolarizing', or 'averaged_hidden_leakage_as_depolarizing'",
        )),
    }
}

fn parse_measurement_crosstalk_transition_model(
    value: Option<BTreeMap<String, f64>>,
) -> Result<MeasurementCrosstalkTransitionModel, String> {
    let Some(value) = value else {
        return Ok(MeasurementCrosstalkTransitionModel::default());
    };
    let mut model = MeasurementCrosstalkTransitionModel::default();
    for (key, probability) in value {
        if !probability.is_finite() || probability < 0.0 {
            let msg = format!(
                "measurement crosstalk transition probability for {key:?} must be finite and non-negative"
            );
            return Err(msg);
        }
        match key.trim().to_ascii_uppercase().replace(' ', "").as_str() {
            "0->0" | "1->1" => {}
            "0->1" => model.p_0_to_1 = probability,
            "0->L" => model.p_0_to_leak = probability,
            "1->0" => model.p_1_to_0 = probability,
            "1->L" => model.p_1_to_leak = probability,
            _ => {
                let msg = format!(
                    "unsupported measurement crosstalk transition key {key:?}; expected 0->0, 0->1, 0->L, 1->0, 1->1, or 1->L"
                );
                return Err(msg);
            }
        }
    }
    if !model.is_valid() {
        return Err(String::from(
            "measurement crosstalk transition rows must sum to <= 1",
        ));
    }
    Ok(model)
}

impl NoiseOptions {
    fn idle_noise_parameters(&self) -> BTreeMap<String, f64> {
        [
            ("p_idle", self.p_idle),
            ("t1", self.t1),
            ("t2", self.t2),
            ("idle_rz", self.idle_rz),
            ("p_idle_linear_rate", self.p_idle_linear_rate),
            ("p_idle_quadratic_rate", self.p_idle_quadratic_rate),
            ("p_idle_x_linear_rate", self.p_idle_x_linear_rate),
            ("p_idle_y_linear_rate", self.p_idle_y_linear_rate),
            ("p_idle_z_linear_rate", self.p_idle_z_linear_rate),
            ("p_idle_x_quadratic_rate", self.p_idle_x_quadratic_rate),
            ("p_idle_y_quadratic_rate", self.p_idle_y_quadratic_rate),
            ("p_idle_z_quadratic_rate", self.p_idle_z_quadratic_rate),
            (
                "p_idle_x_quadratic_sine_rate",
                self.p_idle_x_quadratic_sine_rate,
            ),
            (
                "p_idle_y_quadratic_sine_rate",
                self.p_idle_y_quadratic_sine_rate,
            ),
            (
                "p_idle_z_quadratic_sine_rate",
                self.p_idle_z_quadratic_sine_rate,
            ),
            (
                "p_idle_quadratic_sine_rate",
                self.p_idle_quadratic_sine_rate,
            ),
            ("p_idle_linear", self.p_idle_linear),
            ("p_idle_sin_squared", self.p_idle_sin_squared),
            ("p_idle_coherent", self.p_idle_coherent),
        ]
        .into_iter()
        .filter_map(|(name, value)| {
            value
                .filter(|value| matches!(name, "t1" | "t2") || *value != 0.0)
                .map(|value| (name.to_string(), value))
        })
        .collect()
    }

    /// Translate structured idle families to linear and sine-squared X/Y/Z rates.
    ///
    /// Absent sine axes remain absent, as required by idle-target validation.
    ///
    /// # Errors
    ///
    /// Rejects conflicting families, inert models, invalid rates or multipliers,
    /// leakage, and coherent noise that the standard DEM cannot represent.
    pub fn translate_idle_noise(&self) -> Result<[Option<f64>; 6], String> {
        validate_idle_family(
            self.p_idle_coherent,
            self.p_idle_coherent_model.as_ref(),
            "coherent",
        )?;
        for (family, rate, model, primitives, law) in [
            (
                "linear",
                self.p_idle_linear,
                &self.p_idle_linear_model,
                [
                    ("p_idle_linear_rate", self.p_idle_linear_rate),
                    ("p_idle_x_linear_rate", self.p_idle_x_linear_rate),
                    ("p_idle_y_linear_rate", self.p_idle_y_linear_rate),
                    ("p_idle_z_linear_rate", self.p_idle_z_linear_rate),
                ],
                "low-level",
            ),
            (
                "sin_squared",
                self.p_idle_sin_squared,
                &self.p_idle_sin_squared_model,
                [
                    (
                        "p_idle_quadratic_sine_rate",
                        self.p_idle_quadratic_sine_rate,
                    ),
                    (
                        "p_idle_x_quadratic_sine_rate",
                        self.p_idle_x_quadratic_sine_rate,
                    ),
                    (
                        "p_idle_y_quadratic_sine_rate",
                        self.p_idle_y_quadratic_sine_rate,
                    ),
                    (
                        "p_idle_z_quadratic_sine_rate",
                        self.p_idle_z_quadratic_sine_rate,
                    ),
                ],
                "sine-law",
            ),
        ] {
            let conflicts: Vec<_> = primitives
                .iter()
                .filter_map(|(name, value)| value.map(|_| *name))
                .collect();
            if (rate.is_some() || model.is_some()) && !conflicts.is_empty() {
                return Err(format!(
                    "p_idle_{family}/p_idle_{family}_model cannot be combined with {law} idle rate(s): {}",
                    conflicts.join(", ")
                ));
            }
        }
        let mut axes = [
            self.p_idle_x_linear_rate,
            self.p_idle_y_linear_rate,
            self.p_idle_z_linear_rate,
            self.p_idle_x_quadratic_sine_rate,
            self.p_idle_y_quadratic_sine_rate,
            self.p_idle_z_quadratic_sine_rate,
        ];
        if let Some((rate, model)) = validate_idle_family(
            self.p_idle_linear,
            self.p_idle_linear_model.as_ref(),
            "linear",
        )? {
            for (axis, key) in axes[..3].iter_mut().zip(["X", "Y", "Z"]) {
                *axis = Some(rate * model.get(key).copied().unwrap_or(0.0));
            }
        }
        if let Some((rate, model)) = validate_idle_family(
            self.p_idle_sin_squared,
            self.p_idle_sin_squared_model.as_ref(),
            "sin_squared",
        )? {
            for (axis, key) in axes[3..].iter_mut().zip(["X", "Y", "Z"]) {
                let multiplier = model.get(key).copied().unwrap_or(0.0);
                *axis = (multiplier != 0.0).then_some(rate * multiplier);
            }
        }
        Ok(axes)
    }
}

type ValidatedIdleFamily = (f64, BTreeMap<String, f64>);

fn validate_idle_family(
    rate: Option<f64>,
    model: Option<&BTreeMap<String, f64>>,
    family: &str,
) -> Result<Option<ValidatedIdleFamily>, String> {
    let rate_name = format!("p_idle_{family}");
    let model_name = format!("{rate_name}_model");
    if model.is_some() && rate.is_none() {
        return Err(format!(
            "{model_name} requires {rate_name}; otherwise the model is inert"
        ));
    }
    let Some(rate) = rate else {
        return Ok(None);
    };
    if !rate.is_finite() || rate < 0.0 {
        return Err(format!("{rate_name} must be a finite, non-negative float"));
    }
    if family == "coherent" && rate != 0.0 {
        return Err(concat!(
            "the standard DEM builder cannot represent coherent idle noise; its previous behavior silently stored ",
            "the Pauli twirl, discarding exactly the coherence that was requested. The EEG coherent route in ",
            "exp/pecos-eeg is the consumer that can represent it, and only with an RZ generator even there. The ",
            "honest stochastic equivalent, which is the exact Pauli twirl of RZ(rate * t), is ",
            "p_idle_sin_squared=rate/2 with p_idle_sin_squared_model={'Z': 1.0}"
        ).to_string());
    }
    let keys: &[&str] = if family == "coherent" {
        &["RX", "RY", "RZ"]
    } else {
        &["L", "X", "Y", "Z"]
    };
    let mut model = model.map_or_else(
        || {
            let default = if family == "linear" { 1.0 / 3.0 } else { 1.0 };
            keys.iter()
                .filter(|&&key| key != "L")
                .map(|&key| (key.to_string(), default))
                .collect()
        },
        Clone::clone,
    );
    for (key, multiplier) in &model {
        if !keys.contains(&key.as_str()) {
            let expected = keys
                .iter()
                .map(|key| format!("{key:?}"))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "invalid {model_name} key {key:?}; expected {expected}"
            ));
        }
        if !multiplier.is_finite() || *multiplier < 0.0 {
            return Err(format!(
                "{model_name} multiplier for {key:?} must be a finite, non-negative float"
            ));
        }
    }
    if family == "linear" {
        // Fold from +0.0: an empty `f64` sum is -0.0, which would print as "-0".
        let total = model.values().fold(0.0, |total, value| total + value);
        if total <= 0.0 || (total - 1.0).abs() > NORMALIZATION_TOLERANCE {
            return Err(format!(
                "{model_name} multipliers must sum to 1.0 within tolerance 1e-05; got {total}"
            ));
        }
        if (total - 1.0).abs() > FLOAT_EPSILON {
            for multiplier in model.values_mut() {
                *multiplier /= total;
            }
        }
    }
    if model.get("L").copied().unwrap_or(0.0) != 0.0 {
        return Err(format!(
            "{model_name} key {:?} has a nonzero multiplier; DEM fault propagation is Pauli-only; the engines simulators support and consume leakage models",
            "L"
        ));
    }
    Ok(Some((rate, model)))
}

#[cfg(test)]
mod tests;
