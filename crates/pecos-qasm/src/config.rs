// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Configuration structures for QASM simulation
//!
//! This module provides JSON-serializable configuration structures for
//! noise models and quantum engines used in QASM simulations.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use pecos_engines::sim_builder::QuantumEngineType;
use pecos_engines::GateType;
use pecos_engines::noise::{
    BiasedDepolarizingNoiseModel, DepolarizingNoiseModel, GeneralNoiseModel,
    GeneralNoiseModelBuilder, PassThroughNoiseModel,
};

/// Quantum engine configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum QuantumEngineConfig {
    /// State vector engine for general circuits
    StateVector,
    /// Sparse stabilizer engine for Clifford circuits
    SparseStabilizer,
}

impl From<QuantumEngineConfig> for QuantumEngineType {
    fn from(config: QuantumEngineConfig) -> Self {
        match config {
            QuantumEngineConfig::StateVector => QuantumEngineType::StateVector,
            QuantumEngineConfig::SparseStabilizer => QuantumEngineType::SparseStabilizer,
        }
    }
}

/// General noise configuration fields
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneralNoiseFields {
    // Global parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub noiseless_gates: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scale: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub leakage_scale: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emission_scale: Option<f64>,

    // Idle noise parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_idle_coherent: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_idle_linear_rate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_idle_linear_model: Option<BTreeMap<String, f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_idle_quadratic_rate: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_idle_coherent_to_incoherent_factor: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub idle_scale: Option<f64>,

    // Preparation noise parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_prep: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_prep_leak_ratio: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_prep_crosstalk: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prep_scale: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_prep_crosstalk_scale: Option<f64>,

    // Single-qubit gate noise parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p1: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p1_emission_ratio: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p1_emission_model: Option<BTreeMap<String, f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p1_seepage_prob: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p1_pauli_model: Option<BTreeMap<String, f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p1_scale: Option<f64>,

    // Two-qubit gate noise parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p2: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p2_angle_params: Option<(f64, f64, f64, f64)>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p2_angle_power: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p2_emission_ratio: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p2_emission_model: Option<BTreeMap<String, f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p2_seepage_prob: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p2_pauli_model: Option<BTreeMap<String, f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p2_idle: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p2_scale: Option<f64>,

    // Measurement noise parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_meas_0: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_meas_1: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_meas_crosstalk: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meas_scale: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_meas_crosstalk_scale: Option<f64>,
}

/// Noise model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum NoiseConfig {
    /// No noise - ideal simulation
    PassThroughNoise,

    /// Standard depolarizing noise
    DepolarizingNoise {
        #[serde(default = "default_probability")]
        p: f64,
    },

    /// Custom depolarizing noise with per-operation probabilities
    DepolarizingCustomNoise {
        #[serde(default = "default_probability")]
        p_prep: f64,
        #[serde(default = "default_probability")]
        p_meas: f64,
        #[serde(default = "default_probability")]
        p1: f64,
        #[serde(default = "default_p2")]
        p2: f64,
    },

    /// Biased depolarizing noise
    BiasedDepolarizingNoise {
        #[serde(default = "default_probability")]
        p: f64,
    },

    /// General noise model with full configuration
    GeneralNoise(Box<GeneralNoiseFields>),
}

fn default_probability() -> f64 {
    0.001
}

fn default_p2() -> f64 {
    0.002
}

impl GeneralNoiseFields {
    /// Apply global parameters to the builder
    fn apply_global_params(
        &self,
        mut builder: GeneralNoiseModelBuilder,
    ) -> GeneralNoiseModelBuilder {
        if let Some(gates) = &self.noiseless_gates {
            for gate_str in gates {
                if let Some(gate_type) = parse_gate_type_from_string(gate_str) {
                    builder = builder.with_noiseless_gate(gate_type);
                }
            }
        }
        if let Some(s) = self.seed {
            builder = builder.with_seed(s);
        }
        if let Some(v) = self.scale {
            builder = builder.with_scale(v);
        }
        if let Some(v) = self.leakage_scale {
            builder = builder.with_leakage_scale(v);
        }
        if let Some(v) = self.emission_scale {
            builder = builder.with_emission_scale(v);
        }
        builder
    }

    /// Apply idle noise parameters to the builder
    fn apply_idle_params(&self, mut builder: GeneralNoiseModelBuilder) -> GeneralNoiseModelBuilder {
        if let Some(v) = self.p_idle_coherent {
            builder = builder.with_p_idle_coherent(v);
        }
        if let Some(v) = self.p_idle_linear_rate {
            builder = builder.with_p_idle_linear_rate(v);
        }
        if let Some(model) = self.p_idle_linear_model.as_ref() {
            builder = builder.with_p_idle_linear_model(model);
        }
        if let Some(v) = self.p_idle_quadratic_rate {
            builder = builder.with_p_idle_quadratic_rate(v);
        }
        if let Some(v) = self.p_idle_coherent_to_incoherent_factor {
            builder = builder.with_p_idle_coherent_to_incoherent_factor(v);
        }
        if let Some(v) = self.idle_scale {
            builder = builder.with_idle_scale(v);
        }
        builder
    }

    /// Apply prep noise parameters to the builder
    fn apply_prep_params(&self, mut builder: GeneralNoiseModelBuilder) -> GeneralNoiseModelBuilder {
        if let Some(v) = self.p_prep {
            builder = builder.with_prep_probability(v);
        }
        if let Some(v) = self.p_prep_leak_ratio {
            builder = builder.with_prep_leak_ratio(v);
        }
        if let Some(v) = self.p_prep_crosstalk {
            builder = builder.with_p_prep_crosstalk(v);
        }
        if let Some(v) = self.prep_scale {
            builder = builder.with_prep_scale(v);
        }
        if let Some(v) = self.p_prep_crosstalk_scale {
            builder = builder.with_p_prep_crosstalk_scale(v);
        }
        builder
    }

    /// Apply single-qubit gate noise parameters to the builder
    fn apply_single_qubit_params(
        &self,
        mut builder: GeneralNoiseModelBuilder,
    ) -> GeneralNoiseModelBuilder {
        if let Some(v) = self.p1 {
            builder = builder.with_p1_probability(v);
        }
        if let Some(v) = self.p1_emission_ratio {
            builder = builder.with_p1_emission_ratio(v);
        }
        if let Some(model) = self.p1_emission_model.as_ref() {
            builder = builder.with_p1_emission_model(model);
        }
        if let Some(v) = self.p1_seepage_prob {
            builder = builder.with_p1_seepage_prob(v);
        }
        if let Some(model) = self.p1_pauli_model.as_ref() {
            builder = builder.with_p1_pauli_model(model);
        }
        if let Some(v) = self.p1_scale {
            builder = builder.with_p1_scale(v);
        }
        builder
    }

    /// Apply two-qubit gate noise parameters to the builder
    fn apply_two_qubit_params(
        &self,
        mut builder: GeneralNoiseModelBuilder,
    ) -> GeneralNoiseModelBuilder {
        if let Some(v) = self.p2 {
            builder = builder.with_p2_probability(v);
        }
        if let Some((a, b, c, d)) = self.p2_angle_params {
            builder = builder.with_p2_angle_params(a, b, c, d);
        }
        if let Some(v) = self.p2_angle_power {
            builder = builder.with_p2_angle_power(v);
        }
        if let Some(v) = self.p2_emission_ratio {
            builder = builder.with_p2_emission_ratio(v);
        }
        if let Some(model) = self.p2_emission_model.as_ref() {
            builder = builder.with_p2_emission_model(model);
        }
        if let Some(v) = self.p2_seepage_prob {
            builder = builder.with_p2_seepage_prob(v);
        }
        if let Some(model) = self.p2_pauli_model.as_ref() {
            builder = builder.with_p2_pauli_model(model);
        }
        if let Some(v) = self.p2_idle {
            builder = builder.with_p2_idle(v);
        }
        if let Some(v) = self.p2_scale {
            builder = builder.with_p2_scale(v);
        }
        builder
    }

    /// Apply measurement noise parameters to the builder
    fn apply_meas_params(&self, mut builder: GeneralNoiseModelBuilder) -> GeneralNoiseModelBuilder {
        if let Some(v) = self.p_meas_0 {
            builder = builder.with_meas_0_probability(v);
        }
        if let Some(v) = self.p_meas_1 {
            builder = builder.with_meas_1_probability(v);
        }
        if let Some(v) = self.p_meas_crosstalk {
            builder = builder.with_p_meas_crosstalk(v);
        }
        if let Some(v) = self.meas_scale {
            builder = builder.with_meas_scale(v);
        }
        if let Some(v) = self.p_meas_crosstalk_scale {
            builder = builder.with_p_meas_crosstalk_scale(v);
        }
        builder
    }
}

/// Parse a gate type from a string
#[must_use]
pub fn parse_gate_type_from_string(gate_str: &str) -> Option<GateType> {
    match gate_str.to_uppercase().as_str() {
        "I" => Some(GateType::I),
        "X" => Some(GateType::X),
        "Y" => Some(GateType::Y),
        "Z" => Some(GateType::Z),
        "H" => Some(GateType::H),
        "CX" | "CNOT" => Some(GateType::CX),
        "RZ" => Some(GateType::RZ),
        "RZZ" => Some(GateType::RZZ),
        "SZZ" => Some(GateType::SZZ),
        "SZZDAG" | "SZZDG" => Some(GateType::SZZdg),
        "U" => Some(GateType::U),
        "R1XY" => Some(GateType::R1XY),
        "MEASURE" | "M" => Some(GateType::MZ),
        "PREP" => Some(GateType::PZ),
        "IDLE" => Some(GateType::Idle),
        _ => None, // Ignore unknown gate types
    }
}

// Note: The old impl From<NoiseConfig> for NoiseModelType has been removed
// as NoiseModelType is no longer used in the unified API.
// Use the noise model builders directly with the simulation API instead.
