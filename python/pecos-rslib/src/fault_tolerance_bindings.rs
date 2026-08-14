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

//! Python bindings for PECOS fault tolerance analysis.
//!
//! This module provides Python bindings for the fault tolerance infrastructure,
//! enabling PECOS-native DEM (Detector Error Model) generation from quantum circuits.
//!
//! # Main Types
//!
//! - `DagFaultAnalyzer` - Builds fault influence maps from DAG circuits
//! - `DagFaultInfluenceMap` - CSR-optimized influence map (equivalent to DEM)
//! - `FaultLocation` - Represents a fault location in spacetime
//!
//! # Example
//!
//! ```python
//! from pecos_rslib import DagCircuit
//! from pecos_rslib.qec import DagFaultAnalyzer
//!
//! # Build a syndrome extraction circuit
//! dag = DagCircuit()
//! dag.pz(2)      # Prep ancilla
//! dag.cx(0, 2)   # CNOT data -> ancilla
//! dag.cx(1, 2)   # CNOT data -> ancilla
//! dag.mz(2)      # Measure ancilla
//!
//! # Build fault influence map
//! analyzer = DagFaultAnalyzer(dag)
//! influence_map = analyzer.build_influence_map()
//!
//! # Query fault influence (O(1) lookup)
//! has_syndrome, causes_logical = influence_map.classify_fault(0, 1)  # loc 0, X fault
//! ```

use crate::code_matrix_bindings::PyParityCheckMatrix;
use crate::dag_circuit_bindings::PyTickCircuit;
use crate::decoder_spec_bindings::PyDecoderSpec;
use crate::pecos_array::{Array, ArrayData};
use crate::stabilizer_code_spec_bindings::PyStabilizerCodeSpec;
use pecos_core::gate_type::GateType;
use pecos_qec::fault_tolerance::dem_builder::{
    ComparisonMethod as RustComparisonMethod,
    ContributionEffectSummary as RustContributionEffectSummary,
    ContributionRenderRecord as RustContributionRenderRecord,
    ContributionRenderStrategy as RustContributionRenderStrategy,
    ContributionRenderSummary as RustContributionRenderSummary, DemBuilder as RustDemBuilder,
    DemSampler as RustNewDemSampler, DemSamplerBuilder as RustNewDemSamplerBuilder,
    DetectorErrorModel as RustDetectorErrorModel, DirectSourceFamily as RustDirectSourceFamily,
    EquivalenceResult as RustEquivalenceResult, FaultContribution as RustFaultContribution,
    FaultSourceType as RustFaultSourceType, IdleNoiseFamily, MeasurementCrosstalkDemMode,
    MeasurementCrosstalkTransitionModel, NoiseConfig, OutputMode, PAULI_2Q_ORDER,
    ParsedDem as RustParsedDem, PauliWeights, ReplacementBranchApproximation,
    TwoDetectorDirectRenderPolicy as RustTwoDetectorDirectRenderPolicy,
    compare_dems_exact as rust_compare_dems_exact,
    compare_dems_statistical as rust_compare_dems_statistical,
    verify_dem_equivalence as rust_verify_dem_equivalence,
};
use pecos_qec::fault_tolerance::fault_distance::{
    FaultDistanceResult as RustFaultDistanceResult,
    connected_cluster_fault_distance as rust_connected_cluster_fault_distance,
    exhaustive_fault_distance as rust_exhaustive_fault_distance,
    graphlike_fault_distance as rust_graphlike_fault_distance,
    per_observable_fault_distances as rust_per_observable_fault_distances,
};
use pecos_qec::fault_tolerance::fault_distance_upper_bound::{
    FaultDistanceBpMethod as RustFaultDistanceBpMethod,
    FaultDistanceBpSchedule as RustFaultDistanceBpSchedule,
    FaultDistanceObservableSubsetStrategy as RustFaultDistanceObservableSubsetStrategy,
    FaultDistanceOsdMethod as RustFaultDistanceOsdMethod,
    FaultDistanceUpperBoundConfig as RustFaultDistanceUpperBoundConfig,
    FaultDistanceUpperBoundResult as RustFaultDistanceUpperBoundResult,
    randomized_code_distance_upper_bound as rust_randomized_code_distance_upper_bound,
    randomized_fault_distance_upper_bound as rust_randomized_fault_distance_upper_bound,
};
use pecos_qec::fault_tolerance::influence_builder::InfluenceBuilder as RustInfluenceBuilder;
use pecos_qec::fault_tolerance::propagator::{
    DagFaultAnalyzer as RustDagFaultAnalyzer, DagFaultInfluenceMap as RustDagFaultInfluenceMap,
    DagSpacetimeLocation, Pauli,
};
use pecos_qec::fault_tolerance::{
    CircuitDistanceResult as RustCircuitDistanceResult, FaultCheckConfig, FaultChecker,
    FaultConfiguration, FlagFaultToleranceReport as RustFlagFaultToleranceReport,
    FlagViolation as RustFlagViolation, HookError as RustHookError,
    HookErrorReport as RustHookErrorReport, PauliFrameLookup as RustPauliFrameLookup,
    PauliPropChecker, SpacetimeLocation,
};
use pecos_qec::{
    BbMemoryBasis as RustBbMemoryBasis, BivariateBicycleCode as RustBivariateBicycleCode,
    bb_memory_circuit as rust_bb_memory_circuit,
    coloration_memory_circuit as rust_coloration_memory_circuit,
};
use pecos_qec::{
    BoundedEnumerationDistance as RustBoundedEnumerationDistance,
    CertifiedDistance as RustCertifiedDistance,
    ClassicalDistanceSearchOutcome as RustClassicalDistanceSearchOutcome,
    DistanceProblem as RustDistanceProblem, DistanceResult as RustDistanceResult,
    StabilizerDistanceSearchOutcome as RustStabilizerDistanceSearchOutcome,
    bounded_enumeration_code_distance as rust_bounded_enumeration_code_distance,
    bounded_enumeration_stabilizer_distance as rust_bounded_enumeration_stabilizer_distance,
    bounded_enumeration_x_distance as rust_bounded_enumeration_x_distance,
    bounded_enumeration_z_distance as rust_bounded_enumeration_z_distance,
    certified_distance as rust_certified_distance,
    connected_cluster_code_distance as rust_connected_cluster_code_distance,
    stabilizer_code_distance as rust_stabilizer_code_distance,
    subsystem_dressed_distance as rust_subsystem_dressed_distance, x_distance as rust_x_distance,
    z_distance as rust_z_distance,
};
use pecos_quantum::DagCircuit;
use pecos_quantum::QubitId;
use pyo3::Py;
use pyo3::prelude::*;
use pyo3::types::PyString;

use crate::observable_flips_bindings::{PyObservableFlips, obsmask_to_py, py_to_obsmask};
use std::collections::{BTreeMap, BTreeSet};
use std::str::FromStr;

mod batch_decode;
mod decoder_comparison;
mod decoder_scoring;
mod sample_corpus;
mod sampler_decode;

use decoder_comparison::{
    PyDecoderComparisonResult, compare_decoder_outcomes, validate_comparison_arguments,
};
use decoder_scoring::{
    MaskedObservableDecoder, ShotDecodeError, TimedObservableDecoder, count_decoder_mismatches,
};
use sample_corpus::{CorpusError, CorpusToSave, LoadedCorpus};

/// Resolve a caller-supplied worker count.
///
/// `None` means "use rayon's default". An explicit zero is caller error: the
/// parallel decode paths divide the shot count by this value to size chunks,
/// so a zero would panic across the FFI boundary instead of reporting a
/// problem the caller can fix.
fn resolve_worker_count(num_workers: Option<usize>) -> PyResult<usize> {
    match num_workers {
        Some(0) => Err(pyo3::exceptions::PyValueError::new_err(
            "num_workers must be at least 1 (omit it to use one worker per CPU)",
        )),
        Some(count) => Ok(count),
        None => Ok(rayon::current_num_threads()),
    }
}

type PyDemMechanismTuple = (f64, Vec<u32>, Vec<u32>);
type PyDemFitResult = (Vec<PyDemMechanismTuple>, Vec<f64>);
/// Per-shot detector rows paired with per-shot observable/DEM-output rows.
type PyDetectorObservableRows = (Vec<Vec<bool>>, Vec<Vec<bool>>);

fn map_shot_decode_error(error: ShotDecodeError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(error.to_string())
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

fn parse_p1_weights(weights: BTreeMap<String, f64>) -> PyResult<PauliWeights> {
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
                return Err(pyo3::exceptions::PyValueError::new_err(msg));
            }
        };
        if !weight.is_finite() || weight < 0.0 {
            let msg =
                format!("p1_weights[{label:?}] must be finite and non-negative, got {weight}");
            return Err(pyo3::exceptions::PyValueError::new_err(msg));
        }
        sum += weight;
        entries.push((pauli, weight));
    }
    if (sum - 1.0).abs() >= 1.0e-6 {
        let msg = format!("p1_weights relative probabilities must sum to 1.0, got {sum}");
        return Err(pyo3::exceptions::PyValueError::new_err(msg));
    }
    Ok(PauliWeights::new(entries))
}

fn parse_p2_weights(weights: BTreeMap<String, f64>) -> PyResult<PauliWeights> {
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
            return Err(pyo3::exceptions::PyValueError::new_err(msg));
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
            return Err(pyo3::exceptions::PyValueError::new_err(msg));
        }
        let replacement_identity = replacement && label == "II";
        if !replacement_identity && !PAULI_2Q_ORDER.contains(&label.as_str()) {
            let msg = format!(
                "p2_weights keys must be one of {PAULI_2Q_ORDER:?}, or use '~' / ':replace:' before a two-qubit Pauli label for a replacement branch; got {input_label:?}"
            );
            return Err(pyo3::exceptions::PyValueError::new_err(msg));
        }
        if !weight.is_finite() || weight < 0.0 {
            let msg = format!(
                "p2_weights[{input_label:?}] must be finite and non-negative, got {weight}"
            );
            return Err(pyo3::exceptions::PyValueError::new_err(msg));
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
            pecos_core::PauliString::with_phase_and_paulis(
                pecos_core::QuarterPhase::PlusOne,
                Vec::new(),
            )
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(
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
        return Err(pyo3::exceptions::PyValueError::new_err(msg));
    }
    Ok(PauliWeights::with_replacement(entries, replacement_entries))
}

#[cfg(test)]
mod p2_weight_parser_tests {
    use super::*;

    #[test]
    fn accepts_compact_and_explicit_replacement_labels() {
        let weights = BTreeMap::from([("~II".to_string(), 0.4), (":replace:XX".to_string(), 0.6)]);

        let parsed = parse_p2_weights(weights).unwrap();

        assert!(parsed.entries().is_empty());
        assert_eq!(parsed.replacement_entries().len(), 2);
    }

    #[test]
    fn rejects_removed_star_replacement_syntax_with_migration_hint() {
        let error = parse_p2_weights(BTreeMap::from([("*XX".to_string(), 1.0)])).unwrap_err();
        let message = error.to_string();

        assert!(message.contains("removed '*' syntax"));
        assert!(message.contains("~XX"));
        assert!(message.contains(":replace:XX"));
    }

    #[test]
    fn rejects_duplicate_replacement_aliases_after_normalization() {
        let error = parse_p2_weights(BTreeMap::from([
            ("~XX".to_string(), 0.5),
            (":replace:XX".to_string(), 0.5),
        ]))
        .unwrap_err();

        assert!(error.to_string().contains("duplicate label"));
    }
}

fn parse_replacement_approximation(
    value: Option<String>,
) -> PyResult<ReplacementBranchApproximation> {
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
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "p2_replacement_approximation must be 'pauli_twirl_omitted_gate', 'branch_impact', 'exact_branch_replay', or 'ignore_gate_removal'",
        )),
    }
}

fn parse_p2_gate_rates(rates: BTreeMap<String, f64>) -> PyResult<BTreeMap<GateType, f64>> {
    let mut parsed = BTreeMap::new();
    for (label, rate) in rates {
        if !rate.is_finite() || rate < 0.0 {
            let msg =
                format!("p2_gate_rates[{label:?}] must be finite and non-negative, got {rate}");
            return Err(pyo3::exceptions::PyValueError::new_err(msg));
        }
        let gate_type = GateType::from_str(label.trim()).map_err(|err| {
            let msg = format!("unsupported p2_gate_rates gate label {label:?}: {err}");
            pyo3::exceptions::PyValueError::new_err(msg)
        })?;
        if !gate_type.is_two_qubit() {
            let msg = format!("p2_gate_rates keys must name two-qubit gates, got {label:?}");
            return Err(pyo3::exceptions::PyValueError::new_err(msg));
        }
        parsed.insert(gate_type, rate);
    }
    Ok(parsed)
}

fn parse_p1_gate_rates(rates: BTreeMap<String, f64>) -> PyResult<BTreeMap<GateType, f64>> {
    let mut parsed = BTreeMap::new();
    for (label, rate) in rates {
        if !rate.is_finite() || rate < 0.0 {
            let msg =
                format!("p1_gate_rates[{label:?}] must be finite and non-negative, got {rate}");
            return Err(pyo3::exceptions::PyValueError::new_err(msg));
        }
        let gate_type = GateType::from_str(label.trim()).map_err(|err| {
            let msg = format!("unsupported p1_gate_rates gate label {label:?}: {err}");
            pyo3::exceptions::PyValueError::new_err(msg)
        })?;
        if !gate_type.is_single_qubit() {
            let msg = format!("p1_gate_rates keys must name single-qubit gates, got {label:?}");
            return Err(pyo3::exceptions::PyValueError::new_err(msg));
        }
        parsed.insert(gate_type, rate);
    }
    Ok(parsed)
}

fn parse_measurement_crosstalk_dem_mode(
    value: Option<String>,
) -> PyResult<MeasurementCrosstalkDemMode> {
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
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "measurement_crosstalk_dem_mode must be 'omitted', 'exact_deterministic', 'exact_deterministic_leakage_as_depolarizing', or 'averaged_hidden_leakage_as_depolarizing'",
        )),
    }
}

fn parse_measurement_crosstalk_transition_model(
    value: Option<BTreeMap<String, f64>>,
) -> PyResult<MeasurementCrosstalkTransitionModel> {
    let Some(value) = value else {
        return Ok(MeasurementCrosstalkTransitionModel::default());
    };
    let mut model = MeasurementCrosstalkTransitionModel::default();
    for (key, probability) in value {
        if !probability.is_finite() || probability < 0.0 {
            let msg = format!(
                "measurement crosstalk transition probability for {key:?} must be finite and non-negative"
            );
            return Err(pyo3::exceptions::PyValueError::new_err(msg));
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
                return Err(pyo3::exceptions::PyValueError::new_err(msg));
            }
        }
    }
    if !model.is_valid() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "measurement crosstalk transition rows must sum to <= 1",
        ));
    }
    Ok(model)
}

fn apply_noise_options(
    mut noise: NoiseConfig,
    p_idle: Option<f64>,
    t1: Option<f64>,
    t2: Option<f64>,
    idle_rz: Option<f64>,
    p_idle_linear_rate: Option<f64>,
    p_idle_quadratic_rate: Option<f64>,
    p_idle_x_linear_rate: Option<f64>,
    p_idle_y_linear_rate: Option<f64>,
    p_idle_z_linear_rate: Option<f64>,
    p_idle_x_quadratic_rate: Option<f64>,
    p_idle_y_quadratic_rate: Option<f64>,
    p_idle_z_quadratic_rate: Option<f64>,
    p_idle_quadratic_sine_rate: Option<f64>,
    p_idle_x_quadratic_sine_rate: Option<f64>,
    p_idle_y_quadratic_sine_rate: Option<f64>,
    p_idle_z_quadratic_sine_rate: Option<f64>,
    p1_weights: Option<BTreeMap<String, f64>>,
    p2_weights: Option<BTreeMap<String, f64>>,
    p2_replacement_approximation: Option<String>,
    p_meas_crosstalk_local: Option<f64>,
    p_meas_crosstalk_global: Option<f64>,
    p_meas_crosstalk_model: Option<BTreeMap<String, f64>>,
    measurement_crosstalk_dem_mode: Option<String>,
    p2_gate_rates: Option<BTreeMap<String, f64>>,
    p1_gate_rates: Option<BTreeMap<String, f64>>,
) -> PyResult<NoiseConfig> {
    // Reject the base-idle-channel combinations this function would otherwise
    // resolve silently: `set_t1_t2` makes T1/T2 the base channel that shadows
    // `p_idle`, and `set_idle_rz` zeroes `p_idle` and overwrites T1/T2 with a
    // synthetic T2. Each combination discards a caller-supplied rate without
    // any signal (issue #426).
    if p_idle.is_some() && (t1.is_some() || t2.is_some()) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "p_idle cannot be combined with t1/t2; the T1/T2 channel replaces the \
             depolarizing base idle channel, so p_idle would be ignored",
        ));
    }
    if idle_rz.is_some() {
        if p_idle.is_some() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "idle_rz cannot be combined with p_idle; the coherent RZ conversion \
                 replaces the base idle channel, so p_idle would be ignored",
            ));
        }
        if t1.is_some() || t2.is_some() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "idle_rz cannot be combined with t1/t2; the coherent RZ conversion \
                 overwrites the T1/T2 channel with an equivalent T2",
            ));
        }
    }

    noise.p_idle = p_idle.unwrap_or(0.0);
    if let (Some(t1_val), Some(t2_val)) = (t1, t2) {
        noise = noise.set_t1_t2(t1_val, t2_val);
    }
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
    Ok(noise)
}

// Adapter for decoder factories that require `Send + Sync` trait objects.
// Decoder implementations own their state; Python access remains GIL-mediated.
struct SendWrapper(Box<dyn pecos_decoders::ObservableDecoder>);
unsafe impl Send for SendWrapper {}
unsafe impl Sync for SendWrapper {}
impl pecos_decoders::ObservableDecoder for SendWrapper {
    fn decode_obs(
        &mut self,
        syndrome: &[u8],
    ) -> Result<pecos_decoder_core::obs_mask::ObsMask, pecos_decoders::DecoderError> {
        self.0.decode_obs(syndrome)
    }
}

// =============================================================================
// Fault Location Types
// =============================================================================

/// A spacetime location for a fault in a DAG circuit.
///
/// Identifies where a fault can occur: the DAG node, qubits involved,
/// whether it's before or after the gate, and the gate type.
///
/// # Attributes
///
/// * `node` - DAG node index
/// * `qubits` - List of qubit indices involved
/// * `before` - Whether fault occurs before (True) or after (False) the gate
/// * `gate_type` - Name of the gate type
#[pyclass(
    name = "FaultLocation",
    module = "pecos_rslib.qec",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyFaultLocation {
    node: usize,
    qubits: Vec<usize>,
    before: bool,
    gate_type: String,
}

#[pymethods]
impl PyFaultLocation {
    /// DAG node index.
    #[getter]
    fn node(&self) -> usize {
        self.node
    }

    /// Qubit indices involved in this fault location.
    #[getter]
    fn qubits(&self) -> Vec<usize> {
        self.qubits.clone()
    }

    /// Whether the fault occurs before the gate (True) or after (False).
    #[getter]
    fn before(&self) -> bool {
        self.before
    }

    /// Gate type name.
    #[getter]
    fn gate_type(&self) -> String {
        self.gate_type.clone()
    }

    fn __repr__(&self) -> String {
        let timing = if self.before { "before" } else { "after" };
        format!(
            "FaultLocation(node={}, qubits={:?}, {}, gate={})",
            self.node, self.qubits, timing, self.gate_type
        )
    }
}

impl From<&DagSpacetimeLocation> for PyFaultLocation {
    fn from(loc: &DagSpacetimeLocation) -> Self {
        Self {
            node: loc.node,
            qubits: loc.qubits.iter().map(QubitId::index).collect(),
            before: loc.before,
            gate_type: format!("{:?}", loc.gate_type),
        }
    }
}

// =============================================================================
// Fault Influence Map
// =============================================================================

/// A fault influence map built from a DAG circuit.
///
/// Maps fault locations to their effects on detectors and DEM outputs.
/// Uses CSR (Compressed Sparse Row) layout for cache-efficient storage.
///
/// This is functionally equivalent to a Detector Error Model (DEM) but stored
/// in a format optimized for fast querying during sampling.
///
/// # Example
///
/// ```python
/// # Build influence map from analyzer
/// influence_map = analyzer.build_influence_map()
///
/// # Query fault influence
/// has_syndrome, flips_dem_output = influence_map.classify_fault(loc_idx=0, pauli=1)
///
/// # Get detector indices flipped by this fault
/// detector_indices = influence_map.get_detector_indices(loc_idx=0, pauli=1)
/// ```
#[pyclass(name = "DagFaultInfluenceMap", module = "pecos_rslib.qec")]
pub struct PyDagFaultInfluenceMap {
    inner: RustDagFaultInfluenceMap,
}

#[pymethods]
impl PyDagFaultInfluenceMap {
    /// Number of fault locations in the map.
    #[getter]
    fn num_locations(&self) -> usize {
        self.inner.locations.len()
    }

    /// Number of detectors (measurement-based).
    #[getter]
    fn num_detectors(&self) -> usize {
        self.inner.detectors.len()
    }

    /// Total number of outputs in the DEM `L<n>` namespace.
    #[getter]
    fn num_dem_outputs(&self) -> usize {
        self.inner.num_dem_outputs()
    }

    /// Number of observable DEM outputs.
    #[getter]
    fn num_observables(&self) -> usize {
        self.inner.num_observables()
    }

    /// Number of tracked Paulis.
    #[getter]
    fn num_tracked_paulis(&self) -> usize {
        self.inner.num_tracked_paulis()
    }

    /// Get all fault locations.
    ///
    /// Returns:
    ///     List of `FaultLocation` objects.
    fn get_locations(&self) -> Vec<PyFaultLocation> {
        self.inner
            .locations
            .iter()
            .map(PyFaultLocation::from)
            .collect()
    }

    /// Get a specific fault location by index.
    ///
    /// Args:
    ///     `loc_idx`: Location index.
    ///
    /// Returns:
    ///     `FaultLocation` object or None if index is out of range.
    fn get_location(&self, loc_idx: usize) -> Option<PyFaultLocation> {
        self.inner.get_location(loc_idx).map(PyFaultLocation::from)
    }

    /// Classify a fault at the given location.
    ///
    /// Args:
    ///     `loc_idx`: Location index.
    ///     pauli: Pauli type (1=X, 2=Y, 3=Z).
    ///
    /// Returns:
    ///     Tuple (`has_syndrome`, `flips_dem_output`).
    ///     - `has_syndrome`: True if the fault flips at least one detector.
    ///     - `flips_dem_output`: True if the fault flips at least one standard observable DEM output.
    fn classify_fault(&self, loc_idx: usize, pauli: u8) -> (bool, bool) {
        (
            self.inner
                .influences
                .has_detector_flips(loc_idx, Pauli::from_u8(pauli)),
            self.inner.has_observable_flips(loc_idx, pauli),
        )
    }

    /// Get detector indices flipped by a fault.
    ///
    /// Args:
    ///     `loc_idx`: Location index.
    ///     pauli: Pauli type (1=X, 2=Y, 3=Z).
    ///
    /// Returns:
    ///     List of detector indices that are flipped by this fault.
    fn get_detector_indices(&self, loc_idx: usize, pauli: u8) -> Vec<u32> {
        self.inner.get_detector_indices(loc_idx, pauli).to_vec()
    }

    /// Get standard DEM `L<n>` observable indices flipped by a fault.
    fn get_dem_output_indices(&self, loc_idx: usize, pauli: u8) -> Vec<u32> {
        self.inner.get_observable_indices(loc_idx, pauli)
    }

    /// Get raw internal non-detector influence indices flipped by a fault.
    ///
    /// These are implementation indices used to propagate both observables and
    /// tracked Paulis. Prefer `get_dem_output_indices`,
    /// `get_observable_indices`, or `get_tracked_pauli_indices` for public DEM
    /// semantics.
    fn get_internal_dem_output_indices(&self, loc_idx: usize, pauli: u8) -> Vec<u32> {
        self.inner.get_dem_output_indices(loc_idx, pauli).to_vec()
    }

    /// Get tracked-Pauli indices flipped by a fault.
    ///
    /// Args:
    ///     `loc_idx`: Location index.
    ///     pauli: Pauli type (1=X, 2=Y, 3=Z).
    ///
    /// Returns:
    ///     List of tracked-Pauli indices that are flipped by this fault.
    fn get_tracked_pauli_indices(&self, loc_idx: usize, pauli: u8) -> Vec<u32> {
        self.inner.get_tracked_pauli_indices(loc_idx, pauli)
    }

    /// Get observable indices flipped by a fault.
    fn get_observable_indices(&self, loc_idx: usize, pauli: u8) -> Vec<u32> {
        self.inner.get_observable_indices(loc_idx, pauli)
    }

    /// Check if a fault at the given location flips any detector.
    ///
    /// Args:
    ///     `loc_idx`: Location index.
    ///     pauli: Pauli type (1=X, 2=Y, 3=Z).
    ///
    /// Returns:
    ///     True if the fault flips at least one detector.
    fn has_detector_flips(&self, loc_idx: usize, pauli: u8) -> bool {
        self.inner
            .influences
            .has_detector_flips(loc_idx, Pauli::from_u8(pauli))
    }

    /// Check if a fault at the given location flips any standard DEM output.
    fn has_dem_output_flips(&self, loc_idx: usize, pauli: u8) -> bool {
        self.inner.has_observable_flips(loc_idx, pauli)
    }

    /// Check if a fault at the given location flips any observable.
    fn has_observable_flips(&self, loc_idx: usize, pauli: u8) -> bool {
        self.inner.has_observable_flips(loc_idx, pauli)
    }

    /// Replace this map's non-detector DEM outputs with another map's outputs.
    ///
    /// This is the Python equivalent of the canonical Rust DEM builder's
    /// annotation merge: detector influence from `DagFaultAnalyzer` is kept,
    /// while observable/tracked-Pauli outputs from `InfluenceBuilder` are used
    /// for DEM output propagation.
    fn merge_dem_outputs_from(&mut self, other: &PyDagFaultInfluenceMap) {
        self.inner.merge_dem_outputs_from(&other.inner);
    }

    /// Check if a fault at the given location flips any tracked Pauli.
    ///
    /// Args:
    ///     `loc_idx`: Location index.
    ///     pauli: Pauli type (1=X, 2=Y, 3=Z).
    ///
    /// Returns:
    ///     True if the fault flips at least one tracked Pauli.
    fn has_tracked_pauli_flips(&self, loc_idx: usize, pauli: u8) -> bool {
        self.inner.has_tracked_pauli_flips(loc_idx, pauli)
    }

    /// Get memory statistics for this influence map.
    ///
    /// Returns:
    ///     Dictionary with memory usage statistics.
    fn memory_stats(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        let stats = self.inner.memory_stats();
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("num_locations", stats.num_locations)?;
        dict.set_item("total_detector_entries", stats.total_detector_entries)?;
        dict.set_item("total_dem_output_entries", stats.total_dem_output_entries)?;
        dict.set_item("offset_bytes", stats.offset_bytes)?;
        dict.set_item("data_bytes", stats.data_bytes)?;
        dict.set_item("total_bytes", stats.total_bytes)?;
        Ok(dict.unbind())
    }

    /// Export CSR data for external use (e.g., GPU sampling).
    ///
    /// Returns:
    ///     Dictionary containing all CSR arrays:
    ///     - `num_locations`, `num_detectors`, `num_dem_outputs`
    ///     - `num_internal_dem_outputs` for the raw CSR bit-plane width
    ///     - `detector_offsets_x`, `detector_data_x`
    ///     - `detector_offsets_y`, `detector_data_y`
    ///     - `detector_offsets_z`, `detector_data_z`
    ///     - `dem_output_offsets_x`, `dem_output_data_x`
    ///     - `dem_output_offsets_y`, `dem_output_data_y`
    ///     - `dem_output_offsets_z`, `dem_output_data_z`
    fn export_csr(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        let num_internal_dem_outputs = self
            .inner
            .influences
            .max_dem_output_index()
            .map_or(0, |idx| idx + 1);
        let (
            num_locations,
            num_detectors,
            num_dem_outputs,
            det_off_x,
            det_data_x,
            det_off_y,
            det_data_y,
            det_off_z,
            det_data_z,
            dem_output_offsets_x,
            dem_output_data_x,
            dem_output_offsets_y,
            dem_output_data_y,
            dem_output_offsets_z,
            dem_output_data_z,
        ) = self.inner.export_csr();

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("num_locations", num_locations)?;
        dict.set_item("num_detectors", num_detectors)?;
        dict.set_item("num_dem_outputs", num_dem_outputs)?;
        dict.set_item("num_internal_dem_outputs", num_internal_dem_outputs)?;
        dict.set_item("num_observables", self.num_observables())?;
        dict.set_item("num_tracked_paulis", self.num_tracked_paulis())?;
        dict.set_item("detector_offsets_x", det_off_x)?;
        dict.set_item("detector_data_x", det_data_x)?;
        dict.set_item("detector_offsets_y", det_off_y)?;
        dict.set_item("detector_data_y", det_data_y)?;
        dict.set_item("detector_offsets_z", det_off_z)?;
        dict.set_item("detector_data_z", det_data_z)?;
        dict.set_item("dem_output_offsets_x", &dem_output_offsets_x)?;
        dict.set_item("dem_output_data_x", &dem_output_data_x)?;
        dict.set_item("dem_output_offsets_y", &dem_output_offsets_y)?;
        dict.set_item("dem_output_data_y", &dem_output_data_y)?;
        dict.set_item("dem_output_offsets_z", &dem_output_offsets_z)?;
        dict.set_item("dem_output_data_z", &dem_output_data_z)?;
        Ok(dict.unbind())
    }

    /// Get the measurements in order (node, qubit, basis).
    ///
    /// Returns:
    ///     List of (`node_id`, qubit, basis) tuples representing measurements
    ///     in the order used by the influence map.
    fn measurements(&self) -> Vec<(usize, usize, u8)> {
        self.inner
            .measurements
            .iter()
            .map(|&(node, qubit, basis)| (node, qubit, basis))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "DagFaultInfluenceMap(locations={}, detectors={}, tracked_paulis={})",
            self.num_locations(),
            self.num_detectors(),
            self.num_tracked_paulis()
        )
    }

    fn __len__(&self) -> usize {
        self.inner.locations.len()
    }
}

// =============================================================================
// DAG Fault Analyzer
// =============================================================================

/// Analyzes fault tolerance properties of a DAG circuit.
///
/// Builds fault influence maps by backward propagation from measurements.
/// Uses sparse traversal that only visits gates touching qubits with
/// non-trivial Paulis, providing 5-50x speedup over tick-based analysis.
///
/// # Performance
///
/// | Circuit Size | Tick-based | DAG-based | Speedup |
/// |--------------|------------|-----------|---------|
/// | d=3 (17 qubits) | 64 us | 16 us | 4x |
/// | d=5 (49 qubits) | 205 us | 38 us | 5x |
/// | d=7 (97 qubits) | 569 us | 49 us | 11x |
/// | d=11 (241 qubits) | 6529 us | 125 us | 52x |
///
/// # Example
///
/// ```python
/// from pecos_rslib import DagCircuit
/// from pecos_rslib.qec import DagFaultAnalyzer
///
/// dag = DagCircuit()
/// dag.pz(2)
/// dag.cx(0, 2)
/// dag.cx(1, 2)
/// dag.mz(2)
///
/// analyzer = DagFaultAnalyzer(dag)
/// influence_map = analyzer.build_influence_map()
/// ```
#[pyclass(name = "DagFaultAnalyzer", module = "pecos_rslib.qec")]
pub struct PyDagFaultAnalyzer {
    // We need to own the DagCircuit since RustDagFaultAnalyzer borrows it
    dag: DagCircuit,
}

#[pymethods]
impl PyDagFaultAnalyzer {
    /// Create a new DAG fault analyzer.
    ///
    /// Args:
    ///     dag: A `DagCircuit` to analyze.
    #[new]
    fn new(dag: &crate::dag_circuit_bindings::PyDagCircuit) -> Self {
        Self {
            dag: dag.inner.clone(),
        }
    }

    /// Build the complete fault influence map.
    ///
    /// Performs backward propagation from all measurements and creates a
    /// lookup table for fault classification.
    ///
    /// Returns:
    ///     `DagFaultInfluenceMap` with O(1) fault classification.
    fn build_influence_map(&self) -> PyDagFaultInfluenceMap {
        let analyzer = RustDagFaultAnalyzer::new(&self.dag);
        let inner = analyzer.build_influence_map();
        PyDagFaultInfluenceMap { inner }
    }

    /// Maximum node index in the DAG.
    #[getter]
    fn max_node(&self) -> usize {
        let analyzer = RustDagFaultAnalyzer::new(&self.dag);
        analyzer.max_node()
    }

    /// Maximum qubit index in the DAG.
    #[getter]
    fn max_qubit(&self) -> usize {
        let analyzer = RustDagFaultAnalyzer::new(&self.dag);
        analyzer.max_qubit()
    }

    fn __repr__(&self) -> String {
        let analyzer = RustDagFaultAnalyzer::new(&self.dag);
        format!(
            "DagFaultAnalyzer(max_node={}, max_qubit={})",
            analyzer.max_node(),
            analyzer.max_qubit()
        )
    }
}

// =============================================================================
// Influence Builder
// =============================================================================

/// Builder for fault influence maps with proper detector definitions.
///
/// This integrates forward symbolic simulation with backward propagation
/// to create complete influence maps (DEM equivalents) suitable for noisy sampling.
///
/// Unlike `DagFaultAnalyzer` which treats each measurement as a detector,
/// `InfluenceBuilder` uses symbolic simulation to identify which measurements
/// are deterministic (and thus define proper detectors).
///
/// # Example
///
/// ```python
/// from pecos_rslib import DagCircuit
/// from pecos_rslib.qec import InfluenceBuilder
///
/// dag = DagCircuit()
/// # ... build circuit ...
///
/// # Build influence map with tracked Paulis
/// builder = InfluenceBuilder(dag)
/// builder.with_tracked_z([0, 1, 2])  # Track a Z string on these qubits
/// influence_map = builder.build()
/// ```
#[pyclass(name = "InfluenceBuilder", module = "pecos_rslib.qec")]
pub struct PyInfluenceBuilder {
    dag: DagCircuit,
    tracked_x_qubits: Vec<usize>,
    tracked_z_qubits: Vec<usize>,
    tracked_paulis: Vec<pecos_core::PauliString>,
    use_circuit_tracked_paulis: bool,
}

#[pymethods]
impl PyInfluenceBuilder {
    /// Create a new influence builder for the given circuit.
    ///
    /// Args:
    ///     dag: A `DagCircuit` to analyze.
    #[new]
    fn new(dag: &crate::dag_circuit_bindings::PyDagCircuit) -> Self {
        Self {
            dag: dag.inner.clone(),
            tracked_x_qubits: Vec::new(),
            tracked_z_qubits: Vec::new(),
            tracked_paulis: Vec::new(),
            use_circuit_tracked_paulis: false,
        }
    }

    /// Add an X-string tracked Pauli.
    ///
    /// The tracked Pauli is X on all specified qubits and is sensitive to Z errors.
    ///
    /// Args:
    ///     qubits: List of qubit indices for the tracked X Pauli.
    ///
    /// Returns:
    ///     Self for method chaining.
    fn with_tracked_x(mut slf: PyRefMut<'_, Self>, qubits: Vec<usize>) -> PyRefMut<'_, Self> {
        slf.tracked_x_qubits = qubits;
        slf
    }

    /// Add a Z-string tracked Pauli.
    ///
    /// The tracked Pauli is Z on all specified qubits and is sensitive to X errors.
    ///
    /// Args:
    ///     qubits: List of qubit indices for the tracked Z Pauli.
    ///
    /// Returns:
    ///     Self for method chaining.
    fn with_tracked_z(mut slf: PyRefMut<'_, Self>, qubits: Vec<usize>) -> PyRefMut<'_, Self> {
        slf.tracked_z_qubits = qubits;
        slf
    }

    /// Add a tracked Pauli.
    ///
    /// Each entry is a `(qubit, pauli)` tuple where pauli is "X", "Y", or "Z".
    ///
    /// Args:
    ///     entries: List of (`qubit_index`, `pauli_str`) tuples.
    ///
    /// Returns:
    ///     Self for method chaining.
    fn with_tracked_pauli(
        mut slf: PyRefMut<'_, Self>,
        entries: Vec<(usize, String)>,
    ) -> PyResult<PyRefMut<'_, Self>> {
        let paulis: Vec<(pecos_core::Pauli, pecos_core::QubitId)> = entries
            .iter()
            .map(|(qubit, p)| {
                let pauli = match p.to_uppercase().as_str() {
                    "X" => Ok(pecos_core::Pauli::X),
                    "Y" => Ok(pecos_core::Pauli::Y),
                    "Z" => Ok(pecos_core::Pauli::Z),
                    _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Invalid Pauli type: {p}. Expected 'X', 'Y', or 'Z'."
                    ))),
                }?;
                Ok((pauli, pecos_core::QubitId::from(*qubit)))
            })
            .collect::<PyResult<_>>()?;
        slf.tracked_paulis
            .push(pecos_core::PauliString::with_phase_and_paulis(
                pecos_core::QuarterPhase::PlusOne,
                paulis,
            ));
        Ok(slf)
    }

    /// Use annotations from the circuit (observables and tracked Paulis).
    ///
    /// Extracts observable and `tracked_pauli()` annotations from the
    /// circuit. Tracked Paulis are propagated with positional awareness
    /// (only faults before each annotation's position affect it).
    ///
    /// Returns:
    ///     Self for method chaining.
    fn with_circuit_annotations(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.use_circuit_tracked_paulis = true;
        slf
    }

    /// Build the fault influence map.
    ///
    /// This performs:
    /// 1. Forward symbolic simulation to identify deterministic measurements
    /// 2. Detector extraction from deterministic measurement correlations
    /// 3. Backward propagation to build the influence map
    ///
    /// Returns:
    ///     `DagFaultInfluenceMap` with proper detector definitions and tracked Paulis.
    ///
    /// Raises:
    ///     ValueError: A circuit annotation cannot be resolved -- an observable
    ///         referencing a missing node or a non-measurement gate, or a
    ///         tracked Pauli with no meta gate.
    fn build(&self) -> PyResult<PyDagFaultInfluenceMap> {
        let mut builder = RustInfluenceBuilder::new(&self.dag);

        if !self.tracked_x_qubits.is_empty() {
            builder = builder.with_x(&self.tracked_x_qubits);
        }
        if !self.tracked_z_qubits.is_empty() {
            builder = builder.with_z(&self.tracked_z_qubits);
        }

        if self.use_circuit_tracked_paulis {
            builder = builder
                .with_circuit_annotations()
                .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
        }
        for pauli in &self.tracked_paulis {
            builder = builder.with_tracked_pauli(pauli.clone());
        }

        let inner = builder
            .build()
            .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
        Ok(PyDagFaultInfluenceMap { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "InfluenceBuilder(tracked_x={:?}, tracked_z={:?}, tracked_paulis={}, circuit_annotations={})",
            self.tracked_x_qubits,
            self.tracked_z_qubits,
            self.tracked_paulis.len(),
            self.use_circuit_tracked_paulis,
        )
    }
}

// =============================================================================
// Pauli Frame Lookup
// =============================================================================

#[pyclass(name = "PauliFrameLookup", module = "pecos_rslib.qec")]
pub struct PyPauliFrameLookup {
    inner: RustPauliFrameLookup,
}

#[pymethods]
impl PyPauliFrameLookup {
    /// Build a Pauli-frame lookup from positional tracked-Pauli annotations.
    ///
    /// Args:
    ///     dag: A `DagCircuit` carrying tracked-Pauli meta-gates.
    ///     detectors: Detector definitions as measurement-record offsets.
    ///     observables: Observable definitions as measurement-record offsets.
    #[staticmethod]
    #[pyo3(signature = (dag, detectors, observables))]
    fn from_circuit(
        dag: &crate::dag_circuit_bindings::PyDagCircuit,
        detectors: Vec<Vec<i32>>,
        observables: Vec<Vec<i32>>,
    ) -> PyResult<Self> {
        let inner = RustPauliFrameLookup::from_circuit(&dag.inner, &detectors, &observables)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Number of Pauli-twirl mask sites.
    #[getter]
    fn num_pauli_sites(&self) -> usize {
        self.inner.num_pauli_sites()
    }

    /// Number of tracked-Pauli rows.
    #[getter]
    fn num_tracked_paulis(&self) -> usize {
        self.inner.num_tracked_paulis()
    }

    /// Number of detector columns.
    #[getter]
    fn num_detectors(&self) -> usize {
        self.inner.num_detectors()
    }

    /// Number of observable columns.
    #[getter]
    fn num_observables(&self) -> usize {
        self.inner.num_observables()
    }

    /// Return one tracked-Pauli row as `(detectors, observables)`.
    fn row(&self, tracked_idx: usize) -> PyResult<(Vec<u32>, Vec<u32>)> {
        let Some((detectors, observables)) = self.inner.row_effects(tracked_idx) else {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "tracked_idx {tracked_idx} is out of range"
            )));
        };
        Ok((detectors.to_vec(), observables.to_vec()))
    }

    /// Decode a Pauli mask array into tracked-row firings.
    fn mask_firings(&self, pauli_masks: &Bound<'_, pyo3::PyAny>) -> PyResult<Vec<Vec<bool>>> {
        let (values, rows, cols) = extract_pauli_mask_values(pauli_masks)?;
        self.inner
            .mask_firings(&values, rows, cols)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Compute per-shot detector/observable XOR patterns for the given masks.
    fn compute_mask_xor(
        &self,
        pauli_masks: &Bound<'_, pyo3::PyAny>,
    ) -> PyResult<PyDetectorObservableRows> {
        let (values, rows, cols) = extract_pauli_mask_values(pauli_masks)?;
        self.inner
            .compute_mask_xor(&values, rows, cols)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "PauliFrameLookup(num_pauli_sites={}, num_tracked_paulis={}, num_detectors={}, num_observables={})",
            self.num_pauli_sites(),
            self.num_tracked_paulis(),
            self.num_detectors(),
            self.num_observables(),
        )
    }
}

fn extract_pauli_mask_values(
    pauli_masks: &Bound<'_, pyo3::PyAny>,
) -> PyResult<(Vec<u8>, usize, usize)> {
    let array = Array::from_python_value(pauli_masks, None)?;
    match &array.data {
        ArrayData::I8(arr) => collect_signed_pauli_mask_values(arr),
        ArrayData::I16(arr) => collect_signed_pauli_mask_values(arr),
        ArrayData::I32(arr) => collect_signed_pauli_mask_values(arr),
        ArrayData::I64(arr) => collect_signed_pauli_mask_values(arr),
        ArrayData::U8(arr) => collect_unsigned_pauli_mask_values(arr),
        ArrayData::U16(arr) => collect_unsigned_pauli_mask_values(arr),
        ArrayData::U32(arr) => collect_unsigned_pauli_mask_values(arr),
        ArrayData::U64(arr) => collect_unsigned_pauli_mask_values(arr),
        ArrayData::Bool(_)
        | ArrayData::F32(_)
        | ArrayData::F64(_)
        | ArrayData::Complex64(_)
        | ArrayData::Complex128(_)
        | ArrayData::Pauli(_)
        | ArrayData::PauliString(_) => Err(pyo3::exceptions::PyTypeError::new_err(
            "pauli_masks must be an integer Array with values 0=I, 1=X, 2=Y, 3=Z",
        )),
    }
}

fn pauli_mask_shape<T>(arr: &ndarray::ArrayD<T>) -> PyResult<(usize, usize)> {
    let shape = arr.shape();
    if shape.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "pauli_masks must be 2-D with shape (num_shots, num_pauli_sites), got shape {shape:?}"
        )));
    }
    Ok((shape[0], shape[1]))
}

fn collect_signed_pauli_mask_values<T>(
    arr: &ndarray::ArrayD<T>,
) -> PyResult<(Vec<u8>, usize, usize)>
where
    T: Copy + Into<i64>,
{
    let (rows, cols) = pauli_mask_shape(arr)?;
    let mut values = Vec::with_capacity(arr.len());
    for (idx, value) in arr.iter().copied().enumerate() {
        let value = value.into();
        if !(0..=3).contains(&value) {
            let row = idx / cols;
            let col = idx % cols;
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "pauli_masks[{row}, {col}]={value} is outside 0..=3"
            )));
        }
        values.push(u8::try_from(value).expect("validated pauli mask value fits in u8"));
    }
    Ok((values, rows, cols))
}

fn collect_unsigned_pauli_mask_values<T>(
    arr: &ndarray::ArrayD<T>,
) -> PyResult<(Vec<u8>, usize, usize)>
where
    T: Copy + Into<u64>,
{
    let (rows, cols) = pauli_mask_shape(arr)?;
    let mut values = Vec::with_capacity(arr.len());
    for (idx, value) in arr.iter().copied().enumerate() {
        let value = value.into();
        if value > 3 {
            let row = idx / cols;
            let col = idx % cols;
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "pauli_masks[{row}, {col}]={value} is outside 0..=3"
            )));
        }
        values.push(u8::try_from(value).expect("validated pauli mask value fits in u8"));
    }
    Ok((values, rows, cols))
}

// =============================================================================
// Detector Error Model
// =============================================================================

/// Result of a unit-weight mechanism-distance search.
#[pyclass(
    name = "FaultDistanceResult",
    module = "pecos_rslib.qec",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyFaultDistanceResult {
    #[pyo3(get)]
    distance: usize,
    #[pyo3(get)]
    mechanism_indices: Vec<usize>,
}

impl From<RustFaultDistanceResult> for PyFaultDistanceResult {
    fn from(result: RustFaultDistanceResult) -> Self {
        Self {
            distance: result.distance,
            mechanism_indices: result.mechanism_indices,
        }
    }
}

#[pymethods]
impl PyFaultDistanceResult {
    fn __repr__(&self) -> String {
        format!(
            "FaultDistanceResult(distance={}, mechanism_indices={:?})",
            self.distance, self.mechanism_indices
        )
    }
}

/// Fully explicit randomized fault-distance upper-bound configuration.
#[pyclass(
    frozen,
    name = "FaultDistanceUpperBoundConfig",
    module = "pecos_rslib.qec",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyFaultDistanceUpperBoundConfig {
    inner: RustFaultDistanceUpperBoundConfig,
}

fn parse_fault_distance_subset_strategy(
    value: &str,
) -> PyResult<RustFaultDistanceObservableSubsetStrategy> {
    match value {
        "each_single_then_random" => {
            Ok(RustFaultDistanceObservableSubsetStrategy::EachSingleThenRandom)
        }
        "random_nonempty" => Ok(RustFaultDistanceObservableSubsetStrategy::RandomNonempty),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "observable_subset_strategy must be 'each_single_then_random' or 'random_nonempty', got {value:?}"
        ))),
    }
}

fn parse_fault_distance_bp_method(value: &str) -> PyResult<RustFaultDistanceBpMethod> {
    match value {
        "product_sum" => Ok(RustFaultDistanceBpMethod::ProductSum),
        "minimum_sum" => Ok(RustFaultDistanceBpMethod::MinimumSum),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "bp_method must be 'product_sum' or 'minimum_sum', got {value:?}"
        ))),
    }
}

fn parse_fault_distance_bp_schedule(value: &str) -> PyResult<RustFaultDistanceBpSchedule> {
    match value {
        "serial" => Ok(RustFaultDistanceBpSchedule::Serial),
        "parallel" => Ok(RustFaultDistanceBpSchedule::Parallel),
        "serial_relative" => Ok(RustFaultDistanceBpSchedule::SerialRelative),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "bp_schedule must be 'serial', 'parallel', or 'serial_relative', got {value:?}"
        ))),
    }
}

fn parse_fault_distance_osd_method(value: &str) -> PyResult<RustFaultDistanceOsdMethod> {
    match value {
        "off" => Ok(RustFaultDistanceOsdMethod::Off),
        "osd_0" => Ok(RustFaultDistanceOsdMethod::Osd0),
        "osd_e" => Ok(RustFaultDistanceOsdMethod::OsdE),
        "osd_cs" => Ok(RustFaultDistanceOsdMethod::OsdCs),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "osd_method must be 'off', 'osd_0', 'osd_e', or 'osd_cs', got {value:?}"
        ))),
    }
}

#[pymethods]
impl PyFaultDistanceUpperBoundConfig {
    #[new]
    #[pyo3(signature = (samples, seed, observable_subset_strategy, error_rate, max_iterations, bp_method, bp_schedule, min_sum_scaling_factor, osd_method, osd_order, omp_threads))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        samples: usize,
        seed: u64,
        observable_subset_strategy: &str,
        error_rate: f64,
        max_iterations: usize,
        bp_method: &str,
        bp_schedule: &str,
        min_sum_scaling_factor: f64,
        osd_method: &str,
        osd_order: usize,
        omp_threads: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: RustFaultDistanceUpperBoundConfig {
                samples,
                seed,
                observable_subset_strategy: parse_fault_distance_subset_strategy(
                    observable_subset_strategy,
                )?,
                error_rate,
                max_iterations,
                bp_method: parse_fault_distance_bp_method(bp_method)?,
                bp_schedule: parse_fault_distance_bp_schedule(bp_schedule)?,
                min_sum_scaling_factor,
                osd_method: parse_fault_distance_osd_method(osd_method)?,
                osd_order,
                omp_threads,
            },
        })
    }

    #[getter]
    fn samples(&self) -> usize {
        self.inner.samples
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    fn __repr__(&self) -> String {
        format!(
            "FaultDistanceUpperBoundConfig(samples={}, seed={}, observable_subset_strategy={:?}, error_rate={}, max_iterations={}, bp_method={:?}, bp_schedule={:?}, min_sum_scaling_factor={}, osd_method={:?}, osd_order={}, omp_threads={})",
            self.inner.samples,
            self.inner.seed,
            self.inner.observable_subset_strategy,
            self.inner.error_rate,
            self.inner.max_iterations,
            self.inner.bp_method,
            self.inner.bp_schedule,
            self.inner.min_sum_scaling_factor,
            self.inner.osd_method,
            self.inner.osd_order,
            self.inner.omp_threads,
        )
    }
}

/// Natively verified randomized fault-distance upper bound.
#[pyclass(
    frozen,
    name = "FaultDistanceUpperBoundResult",
    module = "pecos_rslib.qec",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyFaultDistanceUpperBoundResult {
    #[pyo3(get)]
    weight: usize,
    #[pyo3(get)]
    mechanism_indices: Vec<usize>,
    #[pyo3(get)]
    samples_run: usize,
}

impl From<RustFaultDistanceUpperBoundResult> for PyFaultDistanceUpperBoundResult {
    fn from(result: RustFaultDistanceUpperBoundResult) -> Self {
        Self {
            weight: result.weight,
            mechanism_indices: result.mechanism_indices,
            samples_run: result.samples_run,
        }
    }
}

#[pymethods]
impl PyFaultDistanceUpperBoundResult {
    #[getter]
    fn bound_kind(&self) -> &'static str {
        "upper_bound"
    }

    fn __repr__(&self) -> String {
        format!(
            "FaultDistanceUpperBoundResult(weight={}, mechanism_indices={:?}, samples_run={}, bound_kind='upper_bound')",
            self.weight, self.mechanism_indices, self.samples_run
        )
    }
}

/// A Detector Error Model (DEM) in standard DEM text format.
///
/// This represents the error model of a quantum circuit, mapping error
/// mechanisms to their probabilities. It can be exported as DEM text for use
/// with compatible decoders.
///
/// # Example
///
/// ```python
/// from pecos_rslib.qec import DemBuilder
///
/// # Build DEM from influence map
/// builder = DemBuilder(influence_map)
/// builder.with_noise(0.01, 0.01, 0.01, 0.01)
/// builder.with_detectors_json(detectors_json)
/// dem = builder.build()
///
/// # Output in DEM format
/// print(dem.to_string())
/// ```
#[pyclass(subclass, name = "DetectorErrorModel", module = "pecos_rslib.qec")]
pub struct PyDetectorErrorModel {
    inner: RustDetectorErrorModel,
}

fn split_dem_outputs_for_dem(
    dem_outputs: &[u32],
    dem: &RustDetectorErrorModel,
) -> (Vec<u32>, Vec<u32>) {
    if dem
        .dem_outputs()
        .iter()
        .all(|output| output.kind.is_none() && output.records.is_empty() && output.pauli.is_none())
    {
        return (dem_outputs.to_vec(), Vec::new());
    }

    let mut observables = Vec::new();
    let mut tracked_paulis = Vec::new();
    for &output_id in dem_outputs {
        if let Some(output) = dem.dem_outputs().get(output_id as usize) {
            if output.is_observable() {
                observables.push(output_id);
            }
            if output.is_tracked_pauli() {
                tracked_paulis.push(output_id);
            }
        }
    }
    (observables, tracked_paulis)
}

fn contribution_summary_to_pydict(
    py: Python<'_>,
    summary: RustContributionEffectSummary,
    dem: &RustDetectorErrorModel,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("detectors", summary.effect.detectors.to_vec())?;
    let dem_outputs = summary.effect.dem_outputs.to_vec();
    let (observables, tracked_paulis) = split_dem_outputs_for_dem(&dem_outputs, dem);
    dict.set_item("dem_outputs", &dem_outputs)?;
    dict.set_item("observables", observables)?;
    dict.set_item("tracked_paulis", tracked_paulis)?;
    dict.set_item("num_contributions", summary.num_contributions)?;
    dict.set_item("total_probability", summary.total_probability)?;
    dict.set_item("direct_count", summary.direct_count)?;
    dict.set_item("direct_probability", summary.direct_probability)?;
    dict.set_item("y_decomposed_count", summary.y_decomposed_count)?;
    dict.set_item("y_decomposed_probability", summary.y_decomposed_probability)?;
    dict.set_item(
        "graphlike_decomposable_count",
        summary.graphlike_decomposable_count,
    )?;
    Ok(dict.unbind())
}

fn contribution_render_summary_to_pydict(
    py: Python<'_>,
    summary: RustContributionRenderSummary,
    dem: &RustDetectorErrorModel,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("detectors", summary.effect.detectors.to_vec())?;
    let dem_outputs = summary.effect.dem_outputs.to_vec();
    let (observables, tracked_paulis) = split_dem_outputs_for_dem(&dem_outputs, dem);
    dict.set_item("dem_outputs", &dem_outputs)?;
    dict.set_item("observables", observables)?;
    dict.set_item("tracked_paulis", tracked_paulis)?;
    dict.set_item("rendered_targets", summary.rendered_targets)?;
    dict.set_item("num_contributions", summary.num_contributions)?;
    dict.set_item("total_probability", summary.total_probability)?;
    dict.set_item("combined_probability", summary.combined_probability)?;
    dict.set_item("source_type_counts", summary.source_type_counts)?;
    dict.set_item(
        "source_type_probabilities",
        summary.source_type_probabilities,
    )?;
    dict.set_item(
        "direct_source_family_counts",
        summary.direct_source_family_counts,
    )?;
    dict.set_item(
        "direct_source_family_probabilities",
        summary.direct_source_family_probabilities,
    )?;
    Ok(dict.unbind())
}

fn contribution_render_record_to_pydict(
    py: Python<'_>,
    record: RustContributionRenderRecord,
    dem: &RustDetectorErrorModel,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let dict = contribution_record_to_pydict(py, record.contribution, dem)?;
    let render_strategy = match record.render_strategy {
        RustContributionRenderStrategy::SourceComponents => "SourceComponents",
        RustContributionRenderStrategy::RecordedComponents => "RecordedComponents",
        RustContributionRenderStrategy::TwoDetectorDirect => "TwoDetectorDirect",
        RustContributionRenderStrategy::HyperedgeGraphlike => "HyperedgeGraphlike",
        RustContributionRenderStrategy::EffectDirect => "EffectDirect",
    };
    dict.bind(py)
        .set_item("rendered_targets", record.rendered_targets)?;
    dict.bind(py).set_item("render_strategy", render_strategy)?;
    if let Some(targets) = record.recorded_component_targets {
        dict.bind(py)
            .set_item("recorded_component_targets", targets)?;
    }
    Ok(dict)
}

fn parse_two_detector_direct_render_policy(
    policy: &str,
) -> PyResult<RustTwoDetectorDirectRenderPolicy> {
    match policy {
        "KeepDirect" => Ok(RustTwoDetectorDirectRenderPolicy::KeepDirect),
        "PreferRecordedComponents" => {
            Ok(RustTwoDetectorDirectRenderPolicy::PreferRecordedComponents)
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown two-detector direct render policy: {policy}"
        ))),
    }
}

fn contribution_record_to_pydict(
    py: Python<'_>,
    contribution: RustFaultContribution,
    dem: &RustDetectorErrorModel,
) -> PyResult<Py<pyo3::types::PyDict>> {
    fn pauli_label(pauli: Pauli) -> &'static str {
        match pauli {
            Pauli::I => "I",
            Pauli::X => "X",
            Pauli::Y => "Y",
            Pauli::Z => "Z",
        }
    }

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("detectors", contribution.effect.detectors.to_vec())?;
    let dem_outputs = contribution.effect.dem_outputs.to_vec();
    let (observables, tracked_paulis) = split_dem_outputs_for_dem(&dem_outputs, dem);
    dict.set_item("dem_outputs", &dem_outputs)?;
    dict.set_item("observables", observables)?;
    dict.set_item("tracked_paulis", tracked_paulis)?;
    dict.set_item("probability", contribution.probability)?;
    dict.set_item("location_indices", contribution.location_indices.to_vec())?;
    dict.set_item(
        "pauli_labels",
        contribution
            .paulis
            .iter()
            .map(|pauli| pauli_label(*pauli))
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "gate_type_labels",
        contribution
            .source_gate_types
            .iter()
            .map(|gate_type| format!("{gate_type:?}"))
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("before_flags", contribution.source_before_flags.to_vec())?;
    if let Some(family) = contribution.direct_source_family {
        let family_label = match family {
            RustDirectSourceFamily::ExclusiveSignature => "ExclusiveSignature",
            RustDirectSourceFamily::SingleLocation => "SingleLocation",
            RustDirectSourceFamily::SingleLocationY => "SingleLocationY",
            RustDirectSourceFamily::TwoLocationPlainY => "TwoLocationPlainY",
            RustDirectSourceFamily::TwoLocationComponent => "TwoLocationComponent",
            RustDirectSourceFamily::TwoLocationOneSidedComponent => "TwoLocationOneSidedComponent",
            RustDirectSourceFamily::TwoLocationReplacementBranchImpact => {
                "TwoLocationReplacementBranchImpact"
            }
            RustDirectSourceFamily::TwoLocationExactReplacementBranch => {
                "TwoLocationExactReplacementBranch"
            }
            RustDirectSourceFamily::MeasurementCrosstalk => "MeasurementCrosstalk",
            RustDirectSourceFamily::Other => "Other",
        };
        dict.set_item("direct_source_family", family_label)?;
    }
    dict.set_item("replacement_branch", contribution.replacement_branch)?;
    if let Some(parts) = &contribution.source_component_effects {
        dict.set_item(
            "source_component_detectors",
            parts
                .iter()
                .map(|part| part.detectors.to_vec())
                .collect::<Vec<_>>(),
        )?;
        dict.set_item(
            "source_component_dem_outputs",
            parts
                .iter()
                .map(|part| part.dem_outputs.to_vec())
                .collect::<Vec<_>>(),
        )?;
    }

    match contribution.source_type {
        RustFaultSourceType::Direct => {
            dict.set_item("source_type", "Direct")?;
            if let Some((first, second)) = contribution.direct_component_effects {
                dict.set_item("component_1_detectors", first.detectors.to_vec())?;
                dict.set_item("component_1_dem_outputs", first.dem_outputs.to_vec())?;
                dict.set_item("component_2_detectors", second.detectors.to_vec())?;
                dict.set_item("component_2_dem_outputs", second.dem_outputs.to_vec())?;
            }
        }
        RustFaultSourceType::DirectOneSidedComponent => {
            dict.set_item("source_type", "DirectOneSidedComponent")?;
            if let Some((first, second)) = contribution.direct_component_effects {
                dict.set_item("component_1_detectors", first.detectors.to_vec())?;
                dict.set_item("component_1_dem_outputs", first.dem_outputs.to_vec())?;
                dict.set_item("component_2_detectors", second.detectors.to_vec())?;
                dict.set_item("component_2_dem_outputs", second.dem_outputs.to_vec())?;
            }
        }
        RustFaultSourceType::YDecomposed {
            x_detectors,
            x_dem_outputs,
            z_detectors,
            z_dem_outputs,
        } => {
            dict.set_item("source_type", "YDecomposed")?;
            dict.set_item("x_detectors", x_detectors.to_vec())?;
            dict.set_item("x_dem_outputs", x_dem_outputs.to_vec())?;
            dict.set_item("z_detectors", z_detectors.to_vec())?;
            dict.set_item("z_dem_outputs", z_dem_outputs.to_vec())?;
        }
    }

    Ok(dict.unbind())
}

#[pymethods]
impl PyDetectorErrorModel {
    /// Build a DetectorErrorModel directly from a circuit and noise.
    ///
    /// Accepts both `TickCircuit` and `DagCircuit`. Reads detector/tracked-Pauli
    /// definitions from circuit metadata.
    ///
    /// Example:
    ///     >>> dem = DetectorErrorModel.from_circuit(tc, p2=0.01)
    ///     >>> print(dem.to_string())
    ///     >>> sampler = dem.to_sampler()
    #[staticmethod]
    #[pyo3(signature = (circuit, p1=0.001, p2=0.01, p_meas=0.001, p_prep=0.001, p_idle=None, t1=None, t2=None, idle_rz=None, p_idle_linear_rate=None, p_idle_quadratic_rate=None, p_idle_x_linear_rate=None, p_idle_y_linear_rate=None, p_idle_z_linear_rate=None, p_idle_x_quadratic_rate=None, p_idle_y_quadratic_rate=None, p_idle_z_quadratic_rate=None, p_idle_quadratic_sine_rate=None, p_idle_x_quadratic_sine_rate=None, p_idle_y_quadratic_sine_rate=None, p_idle_z_quadratic_sine_rate=None, p1_weights=None, p2_weights=None, p2_replacement_approximation=None, p_meas_crosstalk_local=None, p_meas_crosstalk_global=None, p_meas_crosstalk_model=None, measurement_crosstalk_dem_mode=None, p2_gate_rates=None, p1_gate_rates=None))]
    #[allow(clippy::too_many_arguments)]
    fn from_circuit(
        circuit: &pyo3::Bound<'_, pyo3::PyAny>,
        p1: f64,
        p2: f64,
        p_meas: f64,
        p_prep: f64,
        p_idle: Option<f64>,
        t1: Option<f64>,
        t2: Option<f64>,
        idle_rz: Option<f64>,
        p_idle_linear_rate: Option<f64>,
        p_idle_quadratic_rate: Option<f64>,
        p_idle_x_linear_rate: Option<f64>,
        p_idle_y_linear_rate: Option<f64>,
        p_idle_z_linear_rate: Option<f64>,
        p_idle_x_quadratic_rate: Option<f64>,
        p_idle_y_quadratic_rate: Option<f64>,
        p_idle_z_quadratic_rate: Option<f64>,
        p_idle_quadratic_sine_rate: Option<f64>,
        p_idle_x_quadratic_sine_rate: Option<f64>,
        p_idle_y_quadratic_sine_rate: Option<f64>,
        p_idle_z_quadratic_sine_rate: Option<f64>,
        p1_weights: Option<BTreeMap<String, f64>>,
        p2_weights: Option<BTreeMap<String, f64>>,
        p2_replacement_approximation: Option<String>,
        p_meas_crosstalk_local: Option<f64>,
        p_meas_crosstalk_global: Option<f64>,
        p_meas_crosstalk_model: Option<BTreeMap<String, f64>>,
        measurement_crosstalk_dem_mode: Option<String>,
        p2_gate_rates: Option<BTreeMap<String, f64>>,
        p1_gate_rates: Option<BTreeMap<String, f64>>,
    ) -> PyResult<Self> {
        use pecos_qec::fault_tolerance::dem_builder::DemBuilder;

        let noise = apply_noise_options(
            NoiseConfig::new(p1, p2, p_meas, p_prep),
            p_idle,
            t1,
            t2,
            idle_rz,
            p_idle_linear_rate,
            p_idle_quadratic_rate,
            p_idle_x_linear_rate,
            p_idle_y_linear_rate,
            p_idle_z_linear_rate,
            p_idle_x_quadratic_rate,
            p_idle_y_quadratic_rate,
            p_idle_z_quadratic_rate,
            p_idle_quadratic_sine_rate,
            p_idle_x_quadratic_sine_rate,
            p_idle_y_quadratic_sine_rate,
            p_idle_z_quadratic_sine_rate,
            p1_weights,
            p2_weights,
            p2_replacement_approximation,
            p_meas_crosstalk_local,
            p_meas_crosstalk_global,
            p_meas_crosstalk_model,
            measurement_crosstalk_dem_mode,
            p2_gate_rates,
            p1_gate_rates,
        )?;
        if let Ok(dag) =
            circuit.extract::<pyo3::PyRef<'_, crate::dag_circuit_bindings::PyDagCircuit>>()
        {
            let inner = DemBuilder::try_from_circuit_with_noise_config(&dag.inner, noise)
                .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
            Ok(Self { inner })
        } else if let Ok(tc) =
            circuit.extract::<pyo3::PyRef<'_, crate::dag_circuit_bindings::PyTickCircuit>>()
        {
            let inner = DemBuilder::try_from_tick_circuit_with_noise_config(&tc.inner, noise)
                .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
            Ok(Self { inner })
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "from_circuit() expects a DagCircuit or TickCircuit",
            ))
        }
    }

    /// Build a DetectorErrorModel from PECOS DEM metadata JSON.
    ///
    /// This imports observable and tracked-Pauli metadata only; mechanism
    /// errors must be provided through DEM text or built from a circuit.
    ///
    /// Raises:
    ///     `ValueError`: If the metadata JSON is malformed or uses unsupported fields.
    #[staticmethod]
    fn from_pecos_metadata_json(json: &str) -> PyResult<Self> {
        let inner = RustDetectorErrorModel::new()
            .with_pecos_metadata_json(json)
            .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
        Ok(Self { inner })
    }

    /// Number of detectors in the model.
    #[getter]
    fn num_detectors(&self) -> usize {
        self.inner.num_detectors()
    }

    /// Number of observables in the model.
    #[getter]
    fn num_observables(&self) -> usize {
        self.inner.num_observables()
    }

    /// Total number of outputs in the DEM `L<n>` namespace.
    #[getter]
    fn num_dem_outputs(&self) -> usize {
        self.inner.num_dem_outputs()
    }

    /// Number of tracked Paulis in the model.
    #[getter]
    fn num_tracked_paulis(&self) -> usize {
        self.inner.num_tracked_paulis()
    }

    /// Compute exact fault distance when every mechanism is graphlike.
    ///
    /// Raises:
    ///     `ValueError`: If any mechanism flips more than two detectors.
    fn graphlike_fault_distance(&self) -> PyResult<Option<PyFaultDistanceResult>> {
        rust_graphlike_fault_distance(&self.inner)
            .map(|result| result.map(PyFaultDistanceResult::from))
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
    }

    /// Compute exact fault distance up to an explicit mechanism-count budget using
    /// connected-cluster pruning.
    fn connected_cluster_fault_distance(&self, max_weight: usize) -> Option<PyFaultDistanceResult> {
        rust_connected_cluster_fault_distance(&self.inner, max_weight)
            .map(PyFaultDistanceResult::from)
    }

    /// Compute one connected-cluster fault distance per observable.
    fn per_observable_fault_distances(
        &self,
        max_weight: usize,
    ) -> Vec<Option<PyFaultDistanceResult>> {
        rust_per_observable_fault_distances(&self.inner, max_weight)
            .into_iter()
            .map(|result| result.map(PyFaultDistanceResult::from))
            .collect()
    }

    /// Exhaustively compute fault distance up to an explicit mechanism-count budget.
    ///
    /// This supports hyperedges but has combinatorial cost in the number of mechanisms.
    fn exhaustive_fault_distance(&self, max_weight: usize) -> Option<PyFaultDistanceResult> {
        rust_exhaustive_fault_distance(&self.inner, max_weight).map(PyFaultDistanceResult::from)
    }

    /// Sample natively verified decoder witnesses for a fault-distance upper bound.
    ///
    /// A return value is only an upper bound and never certifies exactness. Invalid decoder
    /// vectors are discarded by native detector and observable parity checks.
    fn randomized_fault_distance_upper_bound(
        &self,
        config: &PyFaultDistanceUpperBoundConfig,
    ) -> PyResult<Option<PyFaultDistanceUpperBoundResult>> {
        rust_randomized_fault_distance_upper_bound(&self.inner, &config.inner)
            .map(|result| result.map(PyFaultDistanceUpperBoundResult::from))
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
    }

    /// Convert the DEM to a string in standard DEM format.
    ///
    /// Each error mechanism is output with its total probability, with no
    /// splitting into decomposed forms.
    ///
    /// Returns:
    ///     A string in DEM format with one entry per mechanism.
    #[allow(clippy::inherent_to_string)] // PyO3 binding - two string formats
    fn to_string(&self) -> String {
        self.inner.to_string()
    }

    /// Convert the DEM to a string with source-decomposed representations.
    ///
    /// Faults are decomposed only using component structure attached to the
    /// original source contribution. Residual hyperedges remain hyperedges
    /// instead of being rewritten by an ambient graphlike search.
    ///
    /// Returns:
    ///     A string in DEM format with decomposed representations.
    fn to_string_decomposed(&self) -> String {
        self.inner.to_string_decomposed()
    }

    /// Convert the DEM to source-decomposed text.
    ///
    /// Only decomposition components attached to the original fault source are
    /// used. Residual hyperedges remain hyperedges instead of being rewritten
    /// by an ambient graphlike search.
    fn to_string_source_decomposed(&self) -> String {
        self.inner.to_string_source_decomposed()
    }

    /// Convert the DEM to a source-informed graphlike decomposition.
    ///
    /// Source-carried components are recursively decomposed only using
    /// graphlike pieces that are themselves source-carried components in this
    /// DEM. Residual hyperedges remain hyperedges.
    fn to_string_source_graphlike_decomposed(&self) -> String {
        self.inner.to_string_source_graphlike_decomposed()
    }

    /// Convert the DEM to a terminal-only graphlike projection.
    ///
    /// Raw mechanisms are first grouped exactly as in `to_string()`. Each raw
    /// effect is then projected to graphlike terminal components using detector
    /// coordinates. This is a decoder-facing representation for graph matchers,
    /// not source-proof decomposition.
    fn to_string_terminal_graphlike_decomposed(&self) -> String {
        self.inner.to_string_terminal_graphlike_decomposed()
    }

    /// Convert the DEM using the explicit historical graphlike-search renderer.
    ///
    /// This may decompose residual hyperedges by searching for graphlike
    /// mechanisms elsewhere in the DEM, so it should be treated as a
    /// compatibility/diagnostic representation rather than source proof.
    fn to_string_graphlike_search_decomposed(&self) -> String {
        self.inner.to_string_graphlike_search_decomposed()
    }

    /// Convert the DEM to a string with an explicit direct-2det render policy.
    fn to_string_decomposed_with_two_detector_direct_policy(
        &self,
        policy: &str,
    ) -> PyResult<String> {
        let policy = parse_two_detector_direct_render_policy(policy)?;
        Ok(self
            .inner
            .to_string_decomposed_with_two_detector_direct_policy(policy))
    }

    /// Convert the DEM to a maximally decomposed graphlike representation.
    ///
    /// When possible, graphlike 2-detector mechanisms are further rewritten
    /// into XORs of standalone singleton detector effects.
    fn to_string_decomposed_maximally(&self) -> String {
        self.inner.to_string_decomposed_maximally()
    }

    /// Number of tracked error contributions.
    #[getter]
    fn num_contributions(&self) -> usize {
        self.inner.num_contributions()
    }

    /// Quantified residuals from infeasible categorical-to-independent conversions.
    ///
    /// Each dictionary reports the channel kind, fault location, representative
    /// flip signature, total-variation magnitude, requested channel weight, and
    /// their relative magnitude. An empty list means every categorical conversion
    /// was exact.
    #[getter]
    fn idle_noise_residuals(&self, py: Python<'_>) -> PyResult<Vec<Py<pyo3::types::PyDict>>> {
        self.inner
            .idle_noise_residuals()
            .iter()
            .map(|residual| {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("location_index", residual.location_index)?;
                dict.set_item("channel_kind", residual.channel_kind.as_str())?;
                dict.set_item("detectors", residual.effect.detectors.to_vec())?;
                dict.set_item("dem_outputs", residual.effect.dem_outputs.to_vec())?;
                dict.set_item("tracked_paulis", residual.effect.tracked_paulis.to_vec())?;
                dict.set_item("magnitude", residual.magnitude)?;
                dict.set_item("channel_weight", residual.channel_weight)?;
                dict.set_item("relative_magnitude", residual.relative_magnitude())?;
                Ok(dict.unbind())
            })
            .collect()
    }

    /// Returns debug info about contributions for a specific mechanism.
    ///
    /// Args:
    ///     detectors: List of detector IDs that define the mechanism.
    ///
    /// Returns:
    ///     Debug string showing source types and probabilities for matching contributions.
    fn contributions_for_mechanism(&self, detectors: Vec<u32>) -> String {
        self.inner.contributions_for_mechanism(&detectors)
    }

    /// Returns debug info about all unique contribution effects.
    ///
    /// Shows each unique detector/DEM-output pattern and how many contributions
    /// target it with their total probability.
    fn all_contribution_effects(&self) -> String {
        self.inner.all_contribution_effects()
    }

    /// Build a `DemSampler` directly from this DEM — no string round-trip.
    fn to_sampler(&self) -> PyResult<PyDemSampler> {
        use pecos_qec::fault_tolerance::dem_builder::DemSampler;

        let inner = DemSampler::from_detector_error_model(&self.inner);
        Ok(PyDemSampler { inner })
    }

    /// Returns structured summaries for all unique contribution effects.
    fn contribution_effect_summaries(
        &self,
        py: Python<'_>,
    ) -> PyResult<Vec<Py<pyo3::types::PyDict>>> {
        self.inner
            .contribution_effect_summaries()
            .into_iter()
            .map(|summary| contribution_summary_to_pydict(py, summary, &self.inner))
            .collect()
    }

    /// Returns structured summaries for render buckets before final regrouping.
    fn contribution_render_summaries(
        &self,
        py: Python<'_>,
    ) -> PyResult<Vec<Py<pyo3::types::PyDict>>> {
        self.inner
            .contribution_render_summaries()
            .into_iter()
            .map(|summary| contribution_render_summary_to_pydict(py, summary, &self.inner))
            .collect()
    }

    /// Returns structured summaries for render buckets under an explicit
    /// direct-2det render policy.
    fn contribution_render_summaries_with_two_detector_direct_policy(
        &self,
        py: Python<'_>,
        policy: &str,
    ) -> PyResult<Vec<Py<pyo3::types::PyDict>>> {
        let policy = parse_two_detector_direct_render_policy(policy)?;
        self.inner
            .contribution_render_summaries_with_two_detector_direct_policy(policy)
            .into_iter()
            .map(|summary| contribution_render_summary_to_pydict(py, summary, &self.inner))
            .collect()
    }

    /// Returns per-contribution render records before final regrouping.
    fn contribution_render_records(
        &self,
        py: Python<'_>,
    ) -> PyResult<Vec<Py<pyo3::types::PyDict>>> {
        self.inner
            .contribution_render_records()
            .into_iter()
            .map(|record| contribution_render_record_to_pydict(py, record, &self.inner))
            .collect()
    }

    /// Returns per-contribution render records for the source-informed
    /// graphlike renderer.
    fn contribution_source_graphlike_render_records(
        &self,
        py: Python<'_>,
    ) -> PyResult<Vec<Py<pyo3::types::PyDict>>> {
        self.inner
            .contribution_source_graphlike_render_records()
            .into_iter()
            .map(|record| contribution_render_record_to_pydict(py, record, &self.inner))
            .collect()
    }

    /// Returns per-contribution render records under an explicit direct-2det
    /// render policy.
    fn contribution_render_records_with_two_detector_direct_policy(
        &self,
        py: Python<'_>,
        policy: &str,
    ) -> PyResult<Vec<Py<pyo3::types::PyDict>>> {
        let policy = parse_two_detector_direct_render_policy(policy)?;
        self.inner
            .contribution_render_records_with_two_detector_direct_policy(policy)
            .into_iter()
            .map(|record| contribution_render_record_to_pydict(py, record, &self.inner))
            .collect()
    }

    /// Returns source-tracked contributions for a full detector/DEM-output effect.
    fn contributions_for_effect(
        &self,
        py: Python<'_>,
        detectors: Vec<u32>,
        dem_outputs: Vec<u32>,
    ) -> PyResult<Vec<Py<pyo3::types::PyDict>>> {
        self.inner
            .contributions_for_effect(&detectors, &dem_outputs)
            .into_iter()
            .map(|contribution| contribution_record_to_pydict(py, contribution, &self.inner))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "DetectorErrorModel(detectors={}, dem_outputs={}, observables={}, tracked_paulis={}, contributions={})",
            self.num_detectors(),
            self.num_dem_outputs(),
            self.num_observables(),
            self.num_tracked_paulis(),
            self.num_contributions()
        )
    }

    fn __str__(&self) -> String {
        self.to_string()
    }
}

// =============================================================================
// DEM Builder
// =============================================================================

/// Advanced builder for Detector Error Models (DEMs).
///
/// For most use cases, prefer `DetectorErrorModel.from_circuit()` or
/// `DemSampler.from_circuit()` which handle everything automatically.
///
/// Use `DemBuilder` directly when you need:
/// - A custom fault influence map
/// - Non-standard noise configuration
/// - Manual detector and observable definitions
///
/// # Example (advanced)
///
/// ```python
/// from pecos_rslib.qec import DagFaultAnalyzer, DemBuilder
///
/// # Build influence map
/// analyzer = DagFaultAnalyzer(dag)
/// influence_map = analyzer.build_influence_map()
///
/// # Build DEM
/// builder = DemBuilder(influence_map)
/// builder.with_noise(0.01, 0.01, 0.01, 0.01)
/// builder.with_detectors_json(
///     '[{"id": 0, "coords": [0, 0, 0], "records": [-1]}, '
///     '{"detector_id": 1, "coords": [1, 0, 0], "records": [-2]}]'
/// )
/// builder.with_observables_json(
///     '[{"id": 0, "records": [-1]}, {"observable_id": 1, "records": [-2]}]'
/// )
/// dem = builder.build()
///
/// print(dem.to_string())
/// ```
#[pyclass(name = "DemBuilder", module = "pecos_rslib.qec")]
pub struct PyDemBuilder {
    influence_map: RustDagFaultInfluenceMap,
    noise: NoiseConfig,
    detectors_json: Option<String>,
    observables_json: Option<String>,
    num_measurements: Option<usize>,
    measurement_order: Option<Vec<usize>>,
    exact_branch_circuit: Option<DagCircuit>,
}

#[pymethods]
impl PyDemBuilder {
    /// Create a new DEM builder from a fault influence map.
    ///
    /// Args:
    ///     `influence_map`: A `DagFaultInfluenceMap` from `DagFaultAnalyzer`.
    #[new]
    fn new(influence_map: &PyDagFaultInfluenceMap) -> Self {
        Self {
            influence_map: influence_map.inner.clone(),
            noise: NoiseConfig::default(),
            detectors_json: None,
            observables_json: None,
            num_measurements: None,
            measurement_order: None,
            exact_branch_circuit: None,
        }
    }

    /// Set the noise parameters.
    ///
    /// Args:
    ///     p1: Single-qubit depolarizing error rate.
    ///     p2: Two-qubit depolarizing error rate.
    ///     `p_meas`: Measurement error rate.
    ///     `p_prep`: Initialization (prep) error rate.
    ///     `p_idle`: Optional idle noise rate per time unit.
    ///     t1: Optional T1 relaxation time.
    ///     t2: Optional T2 dephasing time.
    ///
    /// Returns:
    ///     Self for method chaining.
    #[pyo3(signature = (p1, p2, p_meas, p_prep, p_idle=None, t1=None, t2=None, idle_rz=None, p_idle_linear_rate=None, p_idle_quadratic_rate=None, p_idle_x_linear_rate=None, p_idle_y_linear_rate=None, p_idle_z_linear_rate=None, p_idle_x_quadratic_rate=None, p_idle_y_quadratic_rate=None, p_idle_z_quadratic_rate=None, p_idle_quadratic_sine_rate=None, p_idle_x_quadratic_sine_rate=None, p_idle_y_quadratic_sine_rate=None, p_idle_z_quadratic_sine_rate=None, p1_weights=None, p2_weights=None, p2_replacement_approximation=None, p_meas_crosstalk_local=None, p_meas_crosstalk_global=None, p_meas_crosstalk_model=None, measurement_crosstalk_dem_mode=None, p2_gate_rates=None, p1_gate_rates=None))]
    #[allow(clippy::too_many_arguments)]
    fn with_noise(
        mut slf: PyRefMut<'_, Self>,
        p1: f64,
        p2: f64,
        p_meas: f64,
        p_prep: f64,
        p_idle: Option<f64>,
        t1: Option<f64>,
        t2: Option<f64>,
        idle_rz: Option<f64>,
        p_idle_linear_rate: Option<f64>,
        p_idle_quadratic_rate: Option<f64>,
        p_idle_x_linear_rate: Option<f64>,
        p_idle_y_linear_rate: Option<f64>,
        p_idle_z_linear_rate: Option<f64>,
        p_idle_x_quadratic_rate: Option<f64>,
        p_idle_y_quadratic_rate: Option<f64>,
        p_idle_z_quadratic_rate: Option<f64>,
        p_idle_quadratic_sine_rate: Option<f64>,
        p_idle_x_quadratic_sine_rate: Option<f64>,
        p_idle_y_quadratic_sine_rate: Option<f64>,
        p_idle_z_quadratic_sine_rate: Option<f64>,
        p1_weights: Option<BTreeMap<String, f64>>,
        p2_weights: Option<BTreeMap<String, f64>>,
        p2_replacement_approximation: Option<String>,
        p_meas_crosstalk_local: Option<f64>,
        p_meas_crosstalk_global: Option<f64>,
        p_meas_crosstalk_model: Option<BTreeMap<String, f64>>,
        measurement_crosstalk_dem_mode: Option<String>,
        p2_gate_rates: Option<BTreeMap<String, f64>>,
        p1_gate_rates: Option<BTreeMap<String, f64>>,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.noise = apply_noise_options(
            NoiseConfig::new(p1, p2, p_meas, p_prep),
            p_idle,
            t1,
            t2,
            idle_rz,
            p_idle_linear_rate,
            p_idle_quadratic_rate,
            p_idle_x_linear_rate,
            p_idle_y_linear_rate,
            p_idle_z_linear_rate,
            p_idle_x_quadratic_rate,
            p_idle_y_quadratic_rate,
            p_idle_z_quadratic_rate,
            p_idle_quadratic_sine_rate,
            p_idle_x_quadratic_sine_rate,
            p_idle_y_quadratic_sine_rate,
            p_idle_z_quadratic_sine_rate,
            p1_weights,
            p2_weights,
            p2_replacement_approximation,
            p_meas_crosstalk_local,
            p_meas_crosstalk_global,
            p_meas_crosstalk_model,
            measurement_crosstalk_dem_mode,
            p2_gate_rates,
            p1_gate_rates,
        )?;
        Ok(slf)
    }

    /// Set the detector definitions from JSON.
    ///
    /// Args:
    ///     json: JSON string with detector definitions.
    ///           Format: [{"id": 0, "coords": [x, y, t], "records": [-1, -5]}, ...]
    ///           Public surface descriptors using "`detector_id`" are also accepted.
    ///
    /// Returns:
    ///     Self for method chaining.
    fn with_detectors_json(mut slf: PyRefMut<'_, Self>, json: String) -> PyRefMut<'_, Self> {
        slf.detectors_json = Some(json);
        slf
    }

    /// Set the observable definitions from JSON.
    ///
    /// Tracked Paulis are carried by the influence map; this helper is for
    /// observable metadata.
    fn with_observables_json(mut slf: PyRefMut<'_, Self>, json: String) -> PyRefMut<'_, Self> {
        slf.observables_json = Some(json);
        slf
    }

    /// Set the number of measurements (for record offset calculation).
    ///
    /// Args:
    ///     num: Total number of measurements in the circuit.
    ///
    /// Returns:
    ///     Self for method chaining.
    fn with_num_measurements(mut slf: PyRefMut<'_, Self>, num: usize) -> PyRefMut<'_, Self> {
        slf.num_measurements = Some(num);
        slf
    }

    /// Set the measurement order from the original circuit.
    ///
    /// The measurement order is a list of qubits in the order they were measured
    /// in the original circuit (e.g., `TickCircuit`). This allows proper mapping
    /// between record offsets (which use `TickCircuit` order) and influence map
    /// indices (which may use a different order based on DAG topology).
    ///
    /// Args:
    ///     order: List of qubit indices in measurement execution order.
    ///            order[i] is the qubit measured at `TickCircuit` measurement index i.
    ///
    /// Returns:
    ///     Self for method chaining.
    fn with_measurement_order(
        mut slf: PyRefMut<'_, Self>,
        order: Vec<usize>,
    ) -> PyRefMut<'_, Self> {
        slf.measurement_order = Some(order);
        slf
    }

    /// Attach the original circuit for exact replacement-branch replay.
    ///
    /// This is only needed when using `p2_replacement_approximation="exact_branch_replay"`
    /// with p2 replacement branches. The influence map still determines
    /// ordinary Pauli propagation; the circuit context lets PECOS replay the
    /// omitted-gate branch and fail loudly if it is not DEM-representable.
    fn with_exact_branch_replay_circuit<'py>(
        mut slf: PyRefMut<'py, Self>,
        circuit: &crate::dag_circuit_bindings::PyDagCircuit,
    ) -> PyRefMut<'py, Self> {
        slf.exact_branch_circuit = Some(circuit.inner.clone());
        slf
    }

    /// Build the Detector Error Model.
    ///
    /// Returns:
    ///     A `DetectorErrorModel` that can be converted to string format.
    ///
    /// Raises:
    ///     `ValueError`: If the detector or observable JSON is malformed, or
    ///         a used record offset / `meas_id` is out of range for the
    ///         configured measurement count.
    fn build(&self) -> PyResult<PyDetectorErrorModel> {
        let mut builder =
            RustDemBuilder::new(&self.influence_map).with_noise_config(self.noise.clone());

        if let Some(ref circuit) = self.exact_branch_circuit {
            builder = builder.with_exact_branch_replay_context(circuit);
        }

        if let Some(num) = self.num_measurements {
            builder = builder.with_num_measurements(num);
        }

        if let Some(ref order) = self.measurement_order {
            builder = builder.with_measurement_order(order.clone());
        }

        if let Some(ref json) = self.detectors_json {
            builder = builder
                .with_detectors_json(json)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        }

        if let Some(ref json) = self.observables_json {
            builder = builder
                .with_observables_json(json)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        }

        let inner = builder
            .try_build()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyDetectorErrorModel { inner })
    }

    /// Alias for `build()` - provided for backward compatibility.
    fn build_with_source_tracking(&self) -> PyResult<PyDetectorErrorModel> {
        self.build()
    }

    fn __repr__(&self) -> String {
        format!(
            "DemBuilder(p1={}, p2={}, p_meas={}, p_prep={}, p_idle={:?})",
            self.noise.p1, self.noise.p2, self.noise.p_meas, self.noise.p_prep, self.noise.p_idle
        )
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Convert a `DemMatchingGraph` to a DEM string for inner decoder construction.
fn subgraph_to_dem_string(graph: &pecos_decoder_core::DemMatchingGraph) -> String {
    let mut lines = Vec::new();
    for edge in &graph.edges {
        let p = edge.probability;
        let mut targets = Vec::new();
        targets.push(format!("D{}", edge.node1));
        if let Some(n2) = edge.node2 {
            targets.push(format!("D{n2}"));
        }
        for &obs in &edge.observables {
            targets.push(format!("L{obs}"));
        }
        lines.push(format!("error({p}) {}", targets.join(" ")));
    }
    lines.join("\n")
}

/// Convert a decoder-spec parse error into Python's invalid-value exception.
/// Convert a decoder type-string parse error into the legacy `ValueError`.
pub(crate) fn decoder_parse_error_to_py(error: pecos_decoders::DecoderError) -> PyErr {
    let message = match error {
        pecos_decoders::DecoderError::InvalidConfiguration(message) => message,
        error => error.to_string(),
    };
    PyErr::new::<pyo3::exceptions::PyValueError, _>(message)
}

/// Convert a decoder build error into the legacy Python exception categories.
fn decoder_build_error_to_py(error: pecos_decoders::DecoderError) -> PyErr {
    match &error {
        pecos_decoders::DecoderError::BackendUnavailable { family: "mwpf", .. } => {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "MWPF decoder is not available in this build. \
                 Install cmake (run `pecos setup`) and rebuild. \
                 See: https://github.com/PECOS-packages/PECOS/blob/dev/docs/user-guide/cmake-setup.md",
            )
        }
        pecos_decoders::DecoderError::BackendUnavailable { .. } => {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(error.to_string())
        }
        pecos_decoders::DecoderError::InternalError(message) => {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(message.clone())
        }
        _ => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(error.to_string()),
    }
}

/// Create an `ObservableDecoder` from a DEM string and decoder type name.
///
/// This is the shared factory used by `SampleBatch.decode_count`,
/// `DemSampler.sample_decode_count`, and the parallel variants.
fn create_observable_decoder(
    dem: &str,
    decoder_type: &str,
) -> PyResult<Box<dyn pecos_decoders::ObservableDecoder>> {
    let spec =
        pecos_decoders::DecoderSpec::parse(decoder_type).map_err(decoder_parse_error_to_py)?;
    let model = spec.embedded_hybrid_full_dem().map_or_else(
        || pecos_decoders::DecodeModel::SingleDem(dem.to_string()),
        |full| pecos_decoders::DecodeModel::HybridDem {
            full: full.to_string(),
            decomposed: dem.to_string(),
        },
    );
    spec.build(&model).map_err(decoder_build_error_to_py)
}

/// Pre-generated sample batch held in Rust memory.
///
/// Created by `DemSampler.sample_batch()`. Can be decoded by multiple
/// decoders without re-sampling, and without crossing the Rust/Python boundary
/// per shot.
///
/// A batch produced by a raw-measurement `DemSampler` uses the same container,
/// but its detector columns contain raw measurements rather than detector
/// events. Data accessors remain available for those batches; decode methods
/// reject them because raw measurements are not decoder syndromes.
///
/// # Example
///
/// ```python
/// from pecos.decoders import pymatching, tesseract
///
/// samples = sampler.sample_batch(10000, seed=42)
/// pm_errors = samples.decode(dem, pymatching(correlated=True)).num_errors
/// ts_errors = samples.decode(dem, tesseract()).num_errors
/// # Both decoders ran on the exact same samples.
/// ```
#[pyclass(name = "SampleBatch", module = "pecos_rslib.qec")]
pub struct PySampleBatch {
    /// Columnar bit-packed detector columns: det_columns[det_idx][word_idx]
    det_columns: Vec<Vec<u64>>,
    /// Columnar bit-packed observable columns: obs_columns[obs_idx][word_idx]
    obs_columns: Vec<Vec<u64>>,
    num_detectors: usize,
    num_shots: usize,
    raw_measurements: bool,
    seed: Option<u64>,
    dem: Option<String>,
    metadata_json: Option<String>,
    generator: Option<String>,
    format_version: Option<u32>,
}

impl PySampleBatch {
    /// Extract syndrome for one shot into a pre-allocated buffer.
    fn extract_syndrome(&self, shot: usize, buf: &mut [u8]) {
        buf.fill(0);
        let word_idx = shot / 64;
        let bit_mask = 1u64 << (shot % 64);
        for (det_idx, col) in self.det_columns.iter().enumerate() {
            if col[word_idx] & bit_mask != 0 {
                buf[det_idx] = 1;
            }
        }
    }

    /// Reject raw-measurement batches before treating their rows as syndromes.
    fn ensure_detector_events(&self) -> PyResult<()> {
        if self.raw_measurements {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "raw-measurement SampleBatch rows carry measurements, not detector events, and cannot be decoded",
            ));
        }
        Ok(())
    }

    /// Extract the observable mask for one shot as a wide [`ObsMask`], with no
    /// 64-observable cap (the columnar storage already supports >64 columns).
    fn extract_obs_mask_wide(&self, shot: usize) -> pecos_decoder_core::obs_mask::ObsMask {
        let word_idx = shot / 64;
        let bit_mask = 1u64 << (shot % 64);
        let mut mask = pecos_decoder_core::obs_mask::ObsMask::new();
        for (obs_idx, col) in self.obs_columns.iter().enumerate() {
            if col[word_idx] & bit_mask != 0 {
                mask.set(obs_idx);
            }
        }
        mask
    }

    /// Build from columnar sampling data.
    fn from_columnar(
        det_columns: Vec<Vec<u64>>,
        obs_columns: Vec<Vec<u64>>,
        num_shots: usize,
        seed: Option<u64>,
    ) -> Self {
        let num_detectors = det_columns.len();
        Self {
            det_columns,
            obs_columns,
            num_detectors,
            num_shots,
            raw_measurements: false,
            seed,
            dem: None,
            metadata_json: None,
            generator: None,
            format_version: None,
        }
    }

    /// Build from rectangular row-major boolean detector and observable data.
    ///
    /// Both outer lists must have equal length, and each list's rows must have
    /// the same width as its row 0.
    fn from_bool_rows(
        detection_events: Vec<Vec<bool>>,
        observable_flips: Vec<Vec<bool>>,
        raw_measurements: bool,
        seed: Option<u64>,
    ) -> Self {
        debug_assert_eq!(observable_flips.len(), detection_events.len());
        let num_shots = detection_events.len();
        let num_detectors = detection_events.first().map_or(0, Vec::len);
        let num_observables = observable_flips.first().map_or(0, Vec::len);
        debug_assert!(
            detection_events
                .iter()
                .all(|row| row.len() == num_detectors)
        );
        debug_assert!(
            observable_flips
                .iter()
                .all(|row| row.len() == num_observables)
        );
        let num_words = num_shots.div_ceil(64);
        let mut det_columns = vec![vec![0u64; num_words]; num_detectors];
        let mut obs_columns = vec![vec![0u64; num_words]; num_observables];

        for (shot, row) in detection_events.iter().enumerate() {
            let word_idx = shot / 64;
            let bit_mask = 1u64 << (shot % 64);
            for (det_idx, &value) in row.iter().enumerate() {
                if value {
                    det_columns[det_idx][word_idx] |= bit_mask;
                }
            }
        }
        for (shot, row) in observable_flips.iter().enumerate() {
            let word_idx = shot / 64;
            let bit_mask = 1u64 << (shot % 64);
            for (obs_idx, &value) in row.iter().enumerate() {
                if value {
                    obs_columns[obs_idx][word_idx] |= bit_mask;
                }
            }
        }

        let mut batch = Self::from_columnar(det_columns, obs_columns, num_shots, seed);
        batch.raw_measurements = raw_measurements;
        batch
    }

    /// Materialize bit-packed columns as shots-major boolean rows.
    fn columns_as_rows(columns: &[Vec<u64>], num_shots: usize) -> Vec<Vec<bool>> {
        (0..num_shots)
            .map(|shot| {
                let word_idx = shot / 64;
                let bit_mask = 1u64 << (shot % 64);
                columns
                    .iter()
                    .map(|column| column[word_idx] & bit_mask != 0)
                    .collect()
            })
            .collect()
    }

    /// Build from row-major data (from Python constructor). Observable masks are
    /// wide [`ObsMask`]es, so more than 64 observables are stored without loss.
    fn from_row_major(
        detection_events: Vec<Vec<u8>>,
        observable_masks: &[pecos_decoder_core::obs_mask::ObsMask],
        num_observables: usize,
    ) -> Self {
        let num_shots = detection_events.len();
        let num_detectors = detection_events.first().map_or(0, Vec::len);
        let num_words = num_shots.div_ceil(64);

        // Convert row-major → columnar
        let mut det_columns = vec![vec![0u64; num_words]; num_detectors];
        for (shot, row) in detection_events.iter().enumerate() {
            let word_idx = shot / 64;
            let bit_mask = 1u64 << (shot % 64);
            for (det_idx, &val) in row.iter().enumerate() {
                if val != 0 {
                    det_columns[det_idx][word_idx] |= bit_mask;
                }
            }
        }

        let mut obs_columns = vec![vec![0u64; num_words]; num_observables];
        for (shot, mask) in observable_masks.iter().enumerate() {
            let word_idx = shot / 64;
            let bit_mask = 1u64 << (shot % 64);
            for obs_idx in mask.iter_set_bits() {
                obs_columns[obs_idx][word_idx] |= bit_mask;
            }
        }

        Self {
            det_columns,
            obs_columns,
            num_detectors,
            num_shots,
            raw_measurements: false,
            seed: None,
            dem: None,
            metadata_json: None,
            generator: None,
            format_version: None,
        }
    }

    fn from_corpus(corpus: LoadedCorpus) -> Self {
        Self {
            num_detectors: corpus.det_columns.len(),
            det_columns: corpus.det_columns,
            obs_columns: corpus.obs_columns,
            num_shots: corpus.num_shots,
            raw_measurements: false,
            seed: corpus.seed,
            dem: Some(corpus.dem),
            metadata_json: corpus.metadata_json,
            generator: Some(corpus.generator),
            format_version: Some(corpus.format_version),
        }
    }

    fn ensure_dem_matches(&self, dem: &str, allow_dem_mismatch: bool) -> PyResult<()> {
        self.ensure_detector_events()?;
        if allow_dem_mismatch {
            return Ok(());
        }
        if let Some(embedded_dem) = &self.dem
            && embedded_dem != dem
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "supplied DEM differs from the DEM embedded in this loaded SampleBatch; pass \
                 allow_dem_mismatch=True to use a different model deliberately",
            ));
        }
        Ok(())
    }

    fn map_corpus_error(error: CorpusError, path: &std::path::Path) -> PyErr {
        match error {
            CorpusError::Io(error) => match error.raw_os_error() {
                Some(errno) => pyo3::exceptions::PyOSError::new_err((
                    errno,
                    error.to_string(),
                    path.as_os_str().to_os_string(),
                )),
                None => error.into(),
            },
            CorpusError::Invalid(message) => pyo3::exceptions::PyValueError::new_err(message),
        }
    }
}

#[pymethods]
impl PySampleBatch {
    /// Build a SampleBatch from detection event arrays and observable masks.
    ///
    /// Args:
    ///     detection_events: List of syndromes, each a list of u8 (0/1).
    ///     observable_masks: List of true observable flip masks as Python ints
    ///         (arbitrary precision; bit ``i`` = observable ``i``, so more than 64
    ///         observables are supported).
    ///     num_observables: Optional exact observable-column width. Every set
    ///         mask bit must be below this width. When omitted, the width is
    ///         inferred as one greater than the highest set bit across all masks;
    ///         consequently, all-zero masks infer zero observable columns.
    #[new]
    #[pyo3(signature = (detection_events, observable_masks, *, num_observables=None))]
    fn new(
        detection_events: Vec<Vec<u8>>,
        observable_masks: Vec<pyo3::Bound<'_, pyo3::PyAny>>,
        num_observables: Option<usize>,
    ) -> PyResult<Self> {
        if detection_events.len() != observable_masks.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "detection_events ({}) and observable_masks ({}) must have same length",
                detection_events.len(),
                observable_masks.len(),
            )));
        }
        let expected_len = detection_events.first().map_or(0, Vec::len);
        for (i, row) in detection_events.iter().enumerate() {
            if row.len() != expected_len {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "detection_events row {i} has length {} but expected {expected_len} \
                     (matching row 0)",
                    row.len()
                )));
            }
        }
        let masks: Vec<pecos_decoder_core::obs_mask::ObsMask> = observable_masks
            .iter()
            .map(py_to_obsmask)
            .collect::<PyResult<_>>()?;
        let observable_width = if let Some(width) = num_observables {
            if let Some(bit) = masks
                .iter()
                .flat_map(pecos_decoder_core::obs_mask::ObsMask::iter_set_bits)
                .find(|&bit| bit >= width)
            {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "observable mask bit {bit} is outside num_observables={width}",
                )));
            }
            width
        } else {
            masks
                .iter()
                .filter_map(|mask| mask.iter_set_bits().max())
                .max()
                .map_or(0, |bit| bit + 1)
        };
        Ok(Self::from_row_major(
            detection_events,
            &masks,
            observable_width,
        ))
    }

    /// Number of shots in this batch.
    #[getter]
    fn num_shots(&self) -> usize {
        self.num_shots
    }

    /// Stored observable-column width, i.e. the length of one row of
    /// [`observable_flips`] and of every [`get_observable_flips`] value.
    ///
    /// This is the constructor's `num_observables` when supplied. Sampler-produced
    /// columns hold all DEM outputs, which can be a superset of the logical
    /// observables, so this is a width rather than a promise about the code.
    #[getter]
    fn num_observables(&self) -> usize {
        self.obs_columns.len()
    }

    /// Resolved random seed used to generate this batch, if known.
    #[getter]
    const fn seed(&self) -> Option<u64> {
        self.seed
    }

    /// Exact detector error model stored with a loaded corpus, if any.
    #[getter]
    fn dem(&self) -> Option<&str> {
        self.dem.as_deref()
    }

    /// Opaque caller metadata JSON stored with a loaded corpus, if any.
    #[getter]
    fn metadata_json(&self) -> Option<&str> {
        self.metadata_json.as_deref()
    }

    /// PECOS writer identity stored with a loaded corpus, if any.
    #[getter]
    fn generator(&self) -> Option<&str> {
        self.generator.as_deref()
    }

    /// Corpus format version for a loaded batch, if any.
    #[getter]
    const fn format_version(&self) -> Option<u32> {
        self.format_version
    }

    /// Save this serially captured shot batch as a self-describing corpus.
    ///
    /// Args:
    ///     path: Destination file path.
    ///     dem: DEM text associated with the samples. For a generated or
    ///         Python-constructed batch, only detector and observable dimensions
    ///         can be checked. This catches gross mismatches, but cannot prove DEM
    ///         identity or detect a different model with the same dimensions. For
    ///         a loaded corpus, the text must exactly match its embedded DEM unless
    ///         `allow_dem_mismatch` is true.
    ///     `metadata_json`: Optional syntactically valid JSON string. ``None``
    ///         preserves metadata already carried by a loaded batch. A supplied
    ///         value replaces it.
    ///     `clear_metadata`: Explicitly omit metadata when true. Cannot be combined
    ///         with a supplied `metadata_json` value.
    ///     `allow_dem_mismatch`: Permit deliberately saving a loaded batch with a
    ///         DEM different from its embedded model.
    #[pyo3(signature = (path, *, dem, metadata_json=None, clear_metadata=false, allow_dem_mismatch=false))]
    fn save(
        &self,
        path: std::path::PathBuf,
        dem: &str,
        metadata_json: Option<&str>,
        clear_metadata: bool,
        allow_dem_mismatch: bool,
    ) -> PyResult<()> {
        self.ensure_dem_matches(dem, allow_dem_mismatch)?;
        if clear_metadata && metadata_json.is_some() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "metadata_json and clear_metadata=True are mutually exclusive",
            ));
        }
        let metadata_json = if clear_metadata {
            None
        } else {
            metadata_json.or(self.metadata_json.as_deref())
        };
        sample_corpus::save(
            &path,
            CorpusToSave {
                det_columns: &self.det_columns,
                obs_columns: &self.obs_columns,
                num_shots: self.num_shots,
                seed: self.seed,
                dem,
                metadata_json,
            },
        )
        .map_err(|error| Self::map_corpus_error(error, &path))
    }

    /// Load and validate a self-describing shot corpus.
    #[staticmethod]
    fn load(path: std::path::PathBuf) -> PyResult<Self> {
        sample_corpus::load(&path)
            .map(Self::from_corpus)
            .map_err(|error| Self::map_corpus_error(error, &path))
    }

    /// Get the syndrome for shot `i` as a list of u8 values.
    fn get_syndrome(&self, i: usize) -> PyResult<Vec<u8>> {
        if i >= self.num_shots {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Shot index {i} out of range (num_shots={})",
                self.num_shots
            )));
        }
        let mut buf = vec![0u8; self.num_detectors];
        self.extract_syndrome(i, &mut buf);
        Ok(buf)
    }

    /// Return all detector events as shots-major boolean lists.
    ///
    /// The result has shape (`num_shots`, `num_detectors`).
    fn detector_events(&self) -> Vec<Vec<bool>> {
        Self::columns_as_rows(&self.det_columns, self.num_shots)
    }

    /// Return all observable flips as shots-major boolean lists.
    ///
    /// The result has shape (`num_shots`, stored observable-column width) and
    /// does not truncate batches containing more than 64 observables. For the
    /// Python constructor, the width is `num_observables` when supplied and is
    /// otherwise inferred from the highest set mask bit (all-zero masks infer
    /// width zero). Sampler-produced columns contain all DEM outputs, which can
    /// be a superset of the logical observables.
    fn observable_flips(&self) -> Vec<Vec<bool>> {
        Self::columns_as_rows(&self.obs_columns, self.num_shots)
    }

    /// Observable flips for shot `i` as an [`ObservableFlips`] value.
    ///
    /// This is the single-shot form of [`observable_flips`], and compares
    /// directly against a decoder result's `observable_flips`. Its length is the
    /// stored observable-column width, so it matches one row of
    /// [`observable_flips`] and carries the same caveat about sampler-produced
    /// columns being a superset of the logical observables.
    fn get_observable_flips(&self, i: usize) -> PyResult<PyObservableFlips> {
        if i >= self.num_shots {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Shot index {i} out of range (num_shots={})",
                self.num_shots
            )));
        }
        Ok(PyObservableFlips::from_mask_value(
            self.extract_obs_mask_wide(i),
            self.obs_columns.len(),
        ))
    }

    /// Decode and score every shot using a typed decoder specification or a
    /// legacy decoder string.
    ///
    /// `dem=None` uses the exact DEM embedded by `SampleBatch.load`; generated
    /// batches require an explicit DEM. Automatic execution honors decoder
    /// statefulness, uses native batching where available, and otherwise chooses
    /// sequential or bounded parallel per-shot execution. Set `workers` to opt
    /// into an exact worker count, `predictions` to retain wide per-shot masks,
    /// and `timing` to retain per-shot elapsed-time statistics.
    #[pyo3(signature = (dem=None, decoder=None, *, workers=None, predictions=false, timing=false, allow_dem_mismatch=false))]
    fn decode(
        &self,
        py: Python<'_>,
        dem: Option<&str>,
        decoder: Option<&Bound<'_, PyAny>>,
        workers: Option<i64>,
        predictions: bool,
        timing: bool,
        allow_dem_mismatch: bool,
    ) -> PyResult<batch_decode::PyDecodeResult> {
        let resolved_dem = dem.or(self.dem.as_deref()).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "dem is required because this SampleBatch has no embedded DEM",
            )
        })?;

        // Preserve the legacy validation precedence: reject raw rows and an
        // embedded-model mismatch before inspecting or constructing a decoder.
        self.ensure_dem_matches(resolved_dem, allow_dem_mismatch)?;

        let decoder = decoder.ok_or_else(|| {
            pyo3::exceptions::PyTypeError::new_err("decoder is a required argument")
        })?;
        let spec = if decoder.is_instance_of::<PyString>() {
            let decoder_type = decoder.extract::<&str>()?;
            pecos_decoders::DecoderSpec::parse(decoder_type).map_err(decoder_parse_error_to_py)?
        } else if let Ok(spec) = decoder.extract::<PyRef<'_, PyDecoderSpec>>() {
            spec.inner.clone()
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "decoder must be a pecos.decoders.DecoderSpec or legacy decoder string",
            ));
        };

        let explicit_workers = workers
            .map(|workers| {
                if workers <= 0 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "workers must be at least 1",
                    ));
                }
                usize::try_from(workers).map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(
                        "workers is too large for this platform",
                    )
                })
            })
            .transpose()?;
        let traits = spec.execution_traits();
        let plan =
            pecos_decoders::batch::plan_execution(pecos_decoders::batch::ExecutionPlanInputs {
                traits,
                num_shots: self.num_shots,
                native_batch_capable: spec.native_batch_capable(),
                timing,
                explicit_workers,
                available_threads: rayon::current_num_threads(),
            })
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;

        let output = py
            .detach(|| batch_decode::execute(self, resolved_dem, &spec, &plan, predictions, timing))
            .map_err(batch_decode::BatchExecutionError::into_pyerr)?;
        batch_decode::PyDecodeResult::from_execution(py, self.num_shots, plan, output)
    }

    /// Decode all samples with the given decoder type and return the error count.
    ///
    /// This runs entirely in Rust -- no per-shot Python crossing.
    ///
    /// Args:
    ///     dem: DEM string in standard DEM text format for the decoder.
    ///     `decoder_type`: "pymatching", "`pymatching_correlated`",
    ///                   "`pymatching_uncorrelated`", "tesseract", "`bp_osd`",
    ///                   "`bp_lsd`", "`union_find`", "`relay_bp`", or "`min_sum_bp`".
    ///
    /// Returns:
    ///     Number of logical errors.
    #[pyo3(signature = (dem, decoder_type="pymatching", *, allow_dem_mismatch=false))]
    fn decode_count(
        &self,
        dem: &str,
        decoder_type: &str,
        allow_dem_mismatch: bool,
    ) -> PyResult<usize> {
        self.ensure_dem_matches(dem, allow_dem_mismatch)?;
        let mut decoder = create_observable_decoder(dem, decoder_type)?;
        let mut syndrome = vec![0u8; self.num_detectors];
        count_decoder_mismatches(
            0..self.num_shots,
            &mut syndrome,
            |shot, buffer| {
                self.extract_syndrome(shot, buffer);
                self.extract_obs_mask_wide(shot)
            },
            decoder.as_mut(),
        )
        .map_err(map_shot_decode_error)
    }

    /// Decode every shot and return the predicted observable mask per shot.
    ///
    /// Mirrors `decode_count` but returns the raw per-shot predictions instead
    /// of an aggregate error count, so callers can localize disagreements
    /// against a reference decoder.
    ///
    /// Args:
    ///     dem: DEM string for the decoder.
    ///     `decoder_type`: Decoder type string.
    ///
    /// Returns:
    ///     List of predicted observable masks (Python ints; arbitrary precision,
    ///     so more than 64 observables are not truncated), one per shot.
    #[pyo3(signature = (dem, decoder_type="pymatching", *, allow_dem_mismatch=false))]
    fn decode_each(
        &self,
        py: Python<'_>,
        dem: &str,
        decoder_type: &str,
        allow_dem_mismatch: bool,
    ) -> PyResult<Vec<Py<pyo3::PyAny>>> {
        self.ensure_dem_matches(dem, allow_dem_mismatch)?;
        let mut decoder = create_observable_decoder(dem, decoder_type)?;
        let mut predictions = Vec::with_capacity(self.num_shots);
        let mut syndrome = vec![0u8; self.num_detectors];
        for shot in 0..self.num_shots {
            self.extract_syndrome(shot, &mut syndrome);
            // Propagate a decode failure rather than masking it as a sentinel
            // observable value (which would read as a spurious disagreement).
            let predicted = decoder.decode_obs(&syndrome).map_err(|error| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "decoder failed on shot {shot}: {error}"
                ))
            })?;
            predictions.push(obsmask_to_py(py, &predicted)?);
        }
        Ok(predictions)
    }

    /// Decode every shot with a decoder under test (DUT) and a reference decoder.
    ///
    /// Both decoders receive the same shots in the same order. Each result is
    /// independently classified as correct, mismatch, or decode error, and a
    /// decode error is counted for that shot without aborting the comparison.
    /// Predictions and truth are compared as wide observable masks, with no
    /// 64-observable limit.
    ///
    /// Args:
    ///     dem: DEM string shared by both decoders.
    ///     `dut_decoder_type`: Decoder type string for the decoder under test.
    ///     `reference_decoder_type`: Decoder type string for the reference.
    ///     alpha: Tail probability for equal-tailed Jeffreys intervals.
    ///
    /// Returns:
    ///     A `DecoderComparisonResult` containing the raw 3x3 counts and
    ///     headline DUT-only-failure and both-failed proportions.
    #[pyo3(signature = (dem, dut_decoder_type, reference_decoder_type, alpha=0.05, *, allow_dem_mismatch=false))]
    fn compare_decoders(
        &self,
        dem: &str,
        dut_decoder_type: &str,
        reference_decoder_type: &str,
        alpha: f64,
        allow_dem_mismatch: bool,
    ) -> PyResult<PyDecoderComparisonResult> {
        validate_comparison_arguments(self.num_shots, alpha)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        self.ensure_dem_matches(dem, allow_dem_mismatch)?;
        let mut dut = create_observable_decoder(dem, dut_decoder_type)?;
        let mut reference = create_observable_decoder(dem, reference_decoder_type)?;
        let mut syndrome = vec![0u8; self.num_detectors];
        let counts = compare_decoder_outcomes(
            self.num_shots,
            &mut syndrome,
            |shot, buffer| {
                self.extract_syndrome(shot, buffer);
                self.extract_obs_mask_wide(shot)
            },
            dut.as_mut(),
            reference.as_mut(),
        );
        PyDecoderComparisonResult::new(counts, alpha)
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
    }

    /// Parallel decode: distributes samples across rayon workers.
    ///
    /// Each worker creates its own decoder instance. Faster for slow decoders.
    ///
    /// Args:
    ///     dem: DEM string for the decoder.
    ///     `decoder_type`: Decoder type string.
    ///     `num_workers`: Number of parallel workers (default: number of CPUs).
    ///
    /// Returns:
    ///     Number of logical errors.
    #[pyo3(signature = (dem, decoder_type="pymatching", num_workers=None, *, allow_dem_mismatch=false))]
    fn decode_count_parallel(
        &self,
        dem: &str,
        decoder_type: &str,
        num_workers: Option<usize>,
        allow_dem_mismatch: bool,
    ) -> PyResult<usize> {
        use rayon::prelude::*;

        self.ensure_dem_matches(dem, allow_dem_mismatch)?;
        drop(create_observable_decoder(dem, decoder_type)?);
        let n_workers = resolve_worker_count(num_workers)?;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_workers)
            .build()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let dem_str = dem.to_string();
        let dt = decoder_type.to_string();
        let n = self.num_shots;
        let num_dets = self.num_detectors;

        // Materialize row-major data for parallel decode.
        let detection_events: Vec<Vec<u8>> = (0..n)
            .map(|i| {
                let mut s = vec![0u8; num_dets];
                self.extract_syndrome(i, &mut s);
                s
            })
            .collect();
        let observable_masks: Vec<pecos_decoder_core::obs_mask::ObsMask> =
            (0..n).map(|i| self.extract_obs_mask_wide(i)).collect();

        let worker_results: Vec<Result<usize, ShotDecodeError>> = pool.install(|| {
            let chunk_size = n.div_ceil(n_workers);
            (0..n_workers)
                .into_par_iter()
                .map(|worker_id| {
                    let start = worker_id * chunk_size;
                    let end = (start + chunk_size).min(n);
                    if start >= end {
                        return Ok(0);
                    }

                    // Safe after the identical factory call was validated above.
                    let mut decoder = create_observable_decoder(&dem_str, &dt).unwrap();
                    let mut syndrome = vec![0u8; num_dets];
                    count_decoder_mismatches(
                        start..end,
                        &mut syndrome,
                        |shot, buffer| {
                            buffer.copy_from_slice(&detection_events[shot]);
                            observable_masks[shot].clone()
                        },
                        decoder.as_mut(),
                    )
                })
                .collect()
        });

        worker_results
            .into_iter()
            .try_fold(0usize, |total, result| {
                result
                    .map(|count| total + count)
                    .map_err(map_shot_decode_error)
            })
    }

    /// Batch decode all samples at once using `PyMatching`'s batch API.
    ///
    /// Sends all detection events in a single flat array to the decoder,
    /// which can vectorize across shots. Faster than per-shot decode for
    /// `PyMatching`. Only supports pymatching decoder.
    ///
    /// Returns:
    ///     Number of logical errors.
    #[pyo3(signature = (dem, *, allow_dem_mismatch=false))]
    fn decode_count_batch(&self, dem: &str, allow_dem_mismatch: bool) -> PyResult<usize> {
        use pecos_decoders::{BatchConfig, PyMatchingDecoder};

        self.ensure_dem_matches(dem, allow_dem_mismatch)?;
        let mut decoder = PyMatchingDecoder::from_dem(dem)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let num_detectors = decoder.num_detectors();

        // Flatten all detection events into a single contiguous array
        let mut flat = Vec::with_capacity(self.num_shots * num_detectors);
        let mut syndrome = vec![0u8; self.num_detectors];
        for i in 0..self.num_shots {
            self.extract_syndrome(i, &mut syndrome);
            // Pad or truncate to decoder's num_detectors
            let take = syndrome.len().min(num_detectors);
            flat.extend_from_slice(&syndrome[..take]);
            flat.extend(std::iter::repeat_n(0, num_detectors - take));
        }

        let config = BatchConfig {
            bit_packed_input: false,
            bit_packed_output: false,
            return_weights: false,
        };

        let result = decoder
            .decode_batch_with_config(&flat, self.num_shots, num_detectors, config)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Count errors by comparing predictions to true observable masks. The
        // predicted mask is a wide ObsMask (inline for <=64 observables, correct
        // beyond), so a DEM with more than 64 observables is not truncated.
        let num_observables = decoder.num_observables();
        let mut num_errors = 0usize;
        for (i, prediction) in result.predictions.iter().enumerate() {
            let mut predicted = pecos_decoder_core::obs_mask::ObsMask::new();
            for (j, &v) in prediction.iter().enumerate() {
                if v != 0 && j < num_observables {
                    predicted.set(j);
                }
            }
            if predicted != self.extract_obs_mask_wide(i) {
                num_errors += 1;
            }
        }

        Ok(num_errors)
    }

    /// Decode all samples and collect per-shot timing statistics.
    ///
    /// Returns a `DecodeStats` with error count, total time, median, and
    /// percentile per-shot decode times. Useful for understanding decoder
    /// performance characteristics (heavy tails, etc.).
    ///
    /// Args:
    ///     dem: DEM string for the decoder.
    ///     `decoder_type`: Decoder type string.
    ///
    /// Returns:
    ///     `DecodeStats` with timing breakdown.
    #[pyo3(signature = (dem, decoder_type="pymatching", *, allow_dem_mismatch=false))]
    fn decode_stats(
        &self,
        dem: &str,
        decoder_type: &str,
        allow_dem_mismatch: bool,
    ) -> PyResult<PyDecodeStats> {
        self.ensure_dem_matches(dem, allow_dem_mismatch)?;
        let decoder = create_observable_decoder(dem, decoder_type)?;
        let mut decoder = TimedObservableDecoder::new(decoder, self.num_shots);
        let mut syndrome = vec![0u8; self.num_detectors];
        let num_errors = count_decoder_mismatches(
            0..self.num_shots,
            &mut syndrome,
            |shot, buffer| {
                self.extract_syndrome(shot, buffer);
                self.extract_obs_mask_wide(shot)
            },
            &mut decoder,
        )
        .map_err(map_shot_decode_error)?;
        let per_shot_seconds = decoder.into_times();

        Ok(PyDecodeStats::from_times(
            self.num_shots,
            num_errors,
            per_shot_seconds,
        ))
    }

    /// Decode all shots with per-shot timing, using parallel workers.
    ///
    /// Like `decode_stats` but distributes shots across rayon threads.
    /// Useful for slow decoders (MWPF, Tesseract, BP+OSD) where a single
    /// shot can take seconds.
    ///
    /// Per-shot timing is still collected (each worker times its own shots).
    /// The total wall-clock time is approximately `serial_total / num_workers`.
    ///
    /// Args:
    ///     dem: DEM string for the decoder.
    ///     `decoder_type`: Decoder type string.
    ///     `num_workers`: Number of parallel workers (default: number of CPUs).
    #[pyo3(signature = (dem, decoder_type="mwpf", num_workers=None, *, allow_dem_mismatch=false))]
    fn decode_stats_parallel(
        &self,
        dem: &str,
        decoder_type: &str,
        num_workers: Option<usize>,
        allow_dem_mismatch: bool,
    ) -> PyResult<PyDecodeStats> {
        use rayon::prelude::*;

        self.ensure_dem_matches(dem, allow_dem_mismatch)?;
        let n_workers = resolve_worker_count(num_workers)?;

        // Validate decoder type early.
        create_observable_decoder(dem, decoder_type)?;

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_workers)
            .build()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let dem_str = dem.to_string();
        let dt = decoder_type.to_string();
        let num_dets = self.num_detectors;

        // Materialize row-major data for parallel decode.
        let detection_events: Vec<Vec<u8>> = (0..self.num_shots)
            .map(|i| {
                let mut s = vec![0u8; num_dets];
                self.extract_syndrome(i, &mut s);
                s
            })
            .collect();
        let observable_masks: Vec<pecos_decoder_core::obs_mask::ObsMask> = (0..self.num_shots)
            .map(|i| self.extract_obs_mask_wide(i))
            .collect();

        // Each worker decodes a contiguous slice and returns its count and timings.
        let results: Vec<Result<(usize, Vec<f64>), ShotDecodeError>> = pool.install(|| {
            let chunk_size = self.num_shots.div_ceil(n_workers);
            (0..n_workers)
                .into_par_iter()
                .map(|worker_id| {
                    let start = worker_id * chunk_size;
                    let end = (start + chunk_size).min(self.num_shots);
                    if start >= end {
                        return Ok((0, Vec::new()));
                    }

                    // Safe after the identical factory call was validated above.
                    let decoder = create_observable_decoder(&dem_str, &dt).unwrap();
                    let mut decoder = TimedObservableDecoder::new(decoder, end - start);
                    let mut syndrome = vec![0u8; num_dets];
                    let errors = count_decoder_mismatches(
                        start..end,
                        &mut syndrome,
                        |shot, buffer| {
                            buffer.copy_from_slice(&detection_events[shot]);
                            observable_masks[shot].clone()
                        },
                        &mut decoder,
                    )?;
                    Ok((errors, decoder.into_times()))
                })
                .collect()
        });

        let mut total_errors = 0usize;
        let mut all_times = Vec::with_capacity(self.num_shots);
        for result in results {
            let (errs, times) = result.map_err(map_shot_decode_error)?;
            total_errors += errs;
            all_times.extend(times);
        }

        Ok(PyDecodeStats::from_times(
            self.num_shots,
            total_errors,
            all_times,
        ))
    }

    fn __repr__(&self) -> String {
        format!("SampleBatch(num_shots={})", self.num_shots)
    }
}

/// Per-shot decode timing statistics.
#[pyclass(name = "DecodeStats", module = "pecos_rslib.qec", skip_from_py_object)]
#[derive(Clone)]
pub struct PyDecodeStats {
    #[pyo3(get)]
    pub num_shots: usize,
    #[pyo3(get)]
    pub num_errors: usize,
    #[pyo3(get)]
    pub logical_error_rate: f64,
    #[pyo3(get)]
    pub total_seconds: f64,
    /// End-to-end elapsed time from decoder construction through Rust scoring.
    #[pyo3(get)]
    pub wall_elapsed: f64,
    /// Sum of the elapsed durations of the individual `decode_obs` calls.
    #[pyo3(get)]
    pub summed_decode_elapsed: f64,
    /// Number of individual per-shot timing samples summarized below.
    #[pyo3(get)]
    pub num_timing_samples: usize,
    #[pyo3(get)]
    pub per_shot_mean: f64,
    #[pyo3(get)]
    pub per_shot_median: f64,
    #[pyo3(get)]
    pub per_shot_p99: f64,
    #[pyo3(get)]
    pub per_shot_min: f64,
    #[pyo3(get)]
    pub per_shot_max: f64,
    /// Quantile summary for distribution visualization (violin plots).
    /// 21 values at percentiles [0, 5, 10, 15, ..., 90, 95, 100].
    #[pyo3(get)]
    pub quantiles: Vec<f64>,
}

impl PyDecodeStats {
    // Shot counts and error counts are well within f64 mantissa range (2^52).
    // Percentile index computation is bounded by array length.
    fn from_times(num_shots: usize, num_errors: usize, times: Vec<f64>) -> Self {
        // Legacy timing methods do not define an end-to-end measurement
        // boundary. Keep their two newer elapsed fields at the documented zero
        // default while preserving every existing statistic.
        Self::from_times_with_elapsed(num_shots, num_errors, times, 0.0, 0.0)
    }

    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn from_times_with_elapsed(
        num_shots: usize,
        num_errors: usize,
        mut times: Vec<f64>,
        wall_elapsed: f64,
        summed_decode_elapsed: f64,
    ) -> Self {
        let num_timing_samples = times.len();
        let total_seconds: f64 = times.iter().sum();
        let per_shot_mean = if num_shots > 0 {
            total_seconds / num_shots as f64
        } else {
            0.0
        };

        times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let percentile = |p: f64| -> f64 {
            if times.is_empty() {
                return 0.0;
            }
            let idx = (p / 100.0 * (times.len() - 1) as f64).round() as usize;
            times[idx.min(times.len() - 1)]
        };

        // 21 quantiles at [0, 5, 10, ..., 95, 100] for violin plots
        let quantiles: Vec<f64> = (0..=20).map(|i| percentile(f64::from(i) * 5.0)).collect();

        Self {
            num_shots,
            num_errors,
            logical_error_rate: if num_shots > 0 {
                num_errors as f64 / num_shots as f64
            } else {
                0.0
            },
            total_seconds,
            wall_elapsed,
            summed_decode_elapsed,
            num_timing_samples,
            per_shot_mean,
            per_shot_median: percentile(50.0),
            per_shot_p99: percentile(99.0),
            per_shot_min: times.first().copied().unwrap_or(0.0),
            per_shot_max: times.last().copied().unwrap_or(0.0),
            quantiles,
        }
    }
}

#[pymethods]
impl PyDecodeStats {
    fn __repr__(&self) -> String {
        format!(
            "DecodeStats(shots={}, errors={}, LER={:.4}, median={:.2e}s, p99={:.2e}s, max={:.2e}s)",
            self.num_shots,
            self.num_errors,
            self.logical_error_rate,
            self.per_shot_median,
            self.per_shot_p99,
            self.per_shot_max,
        )
    }
}

#[pyclass(name = "DemSampler", module = "pecos_rslib.qec")]
pub struct PyDemSampler {
    inner: RustNewDemSampler,
}

#[pymethods]
impl PyDemSampler {
    /// Build a sampler directly from a circuit and noise parameters.
    ///
    /// This is the simplest path: builds the influence map, extracts
    /// annotations, and configures the sampler in one step.
    ///
    /// Args:
    ///     circuit: A `DagCircuit` with gates and annotations.
    ///     p1: Single-qubit depolarizing error rate.
    ///     p2: Two-qubit depolarizing error rate.
    ///     `p_meas`: Measurement error rate.
    ///     `p_prep`: Initialization error rate.
    ///     `p_idle`: Optional idle noise rate per time unit.
    ///
    /// Example:
    ///     >>> sampler = DemSampler.from_circuit(dag, p1=0.001, p2=0.01)
    ///     >>> sampler = DemSampler.from_circuit(tc, p2=0.01)  # TickCircuit also works
    #[staticmethod]
    #[pyo3(signature = (circuit, p1=0.001, p2=0.01, p_meas=0.001, p_prep=0.001, p_idle=None, t1=None, t2=None, idle_rz=None, p_idle_linear_rate=None, p_idle_quadratic_rate=None, p_idle_x_linear_rate=None, p_idle_y_linear_rate=None, p_idle_z_linear_rate=None, p_idle_x_quadratic_rate=None, p_idle_y_quadratic_rate=None, p_idle_z_quadratic_rate=None, p_idle_quadratic_sine_rate=None, p_idle_x_quadratic_sine_rate=None, p_idle_y_quadratic_sine_rate=None, p_idle_z_quadratic_sine_rate=None, p1_weights=None, p2_weights=None, p2_replacement_approximation=None, p_meas_crosstalk_local=None, p_meas_crosstalk_global=None, p_meas_crosstalk_model=None, measurement_crosstalk_dem_mode=None, p2_gate_rates=None, p1_gate_rates=None))]
    #[allow(clippy::too_many_arguments)]
    fn from_circuit(
        circuit: &Bound<'_, pyo3::PyAny>,
        p1: f64,
        p2: f64,
        p_meas: f64,
        p_prep: f64,
        p_idle: Option<f64>,
        t1: Option<f64>,
        t2: Option<f64>,
        idle_rz: Option<f64>,
        p_idle_linear_rate: Option<f64>,
        p_idle_quadratic_rate: Option<f64>,
        p_idle_x_linear_rate: Option<f64>,
        p_idle_y_linear_rate: Option<f64>,
        p_idle_z_linear_rate: Option<f64>,
        p_idle_x_quadratic_rate: Option<f64>,
        p_idle_y_quadratic_rate: Option<f64>,
        p_idle_z_quadratic_rate: Option<f64>,
        p_idle_quadratic_sine_rate: Option<f64>,
        p_idle_x_quadratic_sine_rate: Option<f64>,
        p_idle_y_quadratic_sine_rate: Option<f64>,
        p_idle_z_quadratic_sine_rate: Option<f64>,
        p1_weights: Option<BTreeMap<String, f64>>,
        p2_weights: Option<BTreeMap<String, f64>>,
        p2_replacement_approximation: Option<String>,
        p_meas_crosstalk_local: Option<f64>,
        p_meas_crosstalk_global: Option<f64>,
        p_meas_crosstalk_model: Option<BTreeMap<String, f64>>,
        measurement_crosstalk_dem_mode: Option<String>,
        p2_gate_rates: Option<BTreeMap<String, f64>>,
        p1_gate_rates: Option<BTreeMap<String, f64>>,
    ) -> PyResult<Self> {
        let noise = apply_noise_options(
            NoiseConfig::new(p1, p2, p_meas, p_prep),
            p_idle,
            t1,
            t2,
            idle_rz,
            p_idle_linear_rate,
            p_idle_quadratic_rate,
            p_idle_x_linear_rate,
            p_idle_y_linear_rate,
            p_idle_z_linear_rate,
            p_idle_x_quadratic_rate,
            p_idle_y_quadratic_rate,
            p_idle_z_quadratic_rate,
            p_idle_quadratic_sine_rate,
            p_idle_x_quadratic_sine_rate,
            p_idle_y_quadratic_sine_rate,
            p_idle_z_quadratic_sine_rate,
            p1_weights,
            p2_weights,
            p2_replacement_approximation,
            p_meas_crosstalk_local,
            p_meas_crosstalk_global,
            p_meas_crosstalk_model,
            measurement_crosstalk_dem_mode,
            p2_gate_rates,
            p1_gate_rates,
        )?;

        // Accept both DagCircuit and TickCircuit
        if let Ok(dag) =
            circuit.extract::<pyo3::PyRef<'_, crate::dag_circuit_bindings::PyDagCircuit>>()
        {
            let inner = RustNewDemSampler::from_circuit(&dag.inner, &noise)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            Ok(Self { inner })
        } else if let Ok(tc) =
            circuit.extract::<pyo3::PyRef<'_, crate::dag_circuit_bindings::PyTickCircuit>>()
        {
            let inner = RustNewDemSampler::from_tick_circuit(&tc.inner, &noise)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            Ok(Self { inner })
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "from_circuit() expects a DagCircuit or TickCircuit",
            ))
        }
    }

    /// Create a sampler from a standard DEM-format string.
    ///
    /// Parses `error(p) D0 D3 L0` lines and builds a sampling engine.
    /// Useful for sampling from DEMs produced by EEG analysis.
    ///
    /// Example:
    ///     >>> from pecos_rslib_exp import eeg_heisenberg_dem
    ///     >>> dem_str = eeg_heisenberg_dem(tc, idle_rz=0.05)
    ///     >>> sampler = DemSampler.from_dem_string(dem_str)
    ///     >>> results = sampler.sample_batch(shots=1000000)
    #[staticmethod]
    #[pyo3(signature = (dem_string))]
    fn from_dem_string(dem_string: &str) -> PyResult<Self> {
        use pecos_qec::fault_tolerance::dem_builder::SamplingEngine;

        // Detector and observable counts come from the canonical parser, which also
        // honours bare `detector D<n>` and `logical_observable L<n>` declarations.
        // Deriving them from `error(...)` lines alone undercounts: Stim emits
        // `logical_observable Lk` precisely for logicals that no mechanism flips, and
        // dropping those made the sampler's width disagree with the decoders' -- which
        // silently turned every shot into a logical error when the widths were compared.
        let (num_detectors, num_observables) =
            pecos_decoder_core::dem::utils::parse_dem_metadata(dem_string).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("invalid DEM: {e}"))
            })?;

        let mut mechanisms = Vec::new();

        for line in dem_string.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse: error(prob) D0 D3 L0
            let Some(rest) = line.strip_prefix("error(") else {
                continue;
            };
            let Some(paren_end) = rest.find(')') else {
                continue;
            };
            let prob: f64 = rest[..paren_end].parse().map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("bad probability: {e}"))
            })?;
            let tokens = rest[paren_end + 1..].split_whitespace();
            let mut dets = Vec::new();
            let mut obs = Vec::new();
            for tok in tokens {
                if let Some(d) = tok.strip_prefix('D') {
                    let id: u32 = d.parse().map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!("bad detector: {e}"))
                    })?;
                    dets.push(id);
                } else if let Some(l) = tok.strip_prefix('L') {
                    let id: u32 = l.parse().map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!("bad observable: {e}"))
                    })?;
                    obs.push(id);
                }
            }
            if prob > 0.0 {
                mechanisms.push((prob, dets, obs));
            }
        }

        let engine = SamplingEngine::from_mechanisms(mechanisms, num_detectors, num_observables);
        let inner = RustNewDemSampler::from_engine(engine);
        Ok(Self { inner })
    }

    /// Create a sampler in raw measurement mode with uniform noise.
    #[staticmethod]
    #[pyo3(signature = (influence_map, p_error))]
    fn raw_uniform(influence_map: &PyDagFaultInfluenceMap, p_error: f64) -> PyResult<Self> {
        Self::from_influence_map(influence_map, p_error)
    }

    /// Create a sampler in raw measurement mode with circuit-level noise.
    #[staticmethod]
    #[pyo3(signature = (influence_map, p1, p2, p_meas, p_prep))]
    fn raw(
        influence_map: &PyDagFaultInfluenceMap,
        p1: f64,
        p2: f64,
        p_meas: f64,
        p_prep: f64,
    ) -> PyResult<Self> {
        Self::from_influence_map_circuit_noise(influence_map, p1, p2, p_meas, p_prep)
    }

    /// Create a sampler in detector-event mode.
    ///
    /// The `observables` argument defines observables.
    #[staticmethod]
    #[pyo3(signature = (influence_map, detectors, observables, p1, p2, p_meas, p_prep, p_idle=None, t1=None, t2=None, idle_rz=None, p_idle_linear_rate=None, p_idle_quadratic_rate=None, p_idle_x_linear_rate=None, p_idle_y_linear_rate=None, p_idle_z_linear_rate=None, p_idle_x_quadratic_rate=None, p_idle_y_quadratic_rate=None, p_idle_z_quadratic_rate=None, p_idle_quadratic_sine_rate=None, p_idle_x_quadratic_sine_rate=None, p_idle_y_quadratic_sine_rate=None, p_idle_z_quadratic_sine_rate=None, p1_weights=None, p2_weights=None, p2_replacement_approximation=None, p_meas_crosstalk_local=None, p_meas_crosstalk_global=None, p_meas_crosstalk_model=None, measurement_crosstalk_dem_mode=None, p2_gate_rates=None, p1_gate_rates=None))]
    #[allow(clippy::too_many_arguments)]
    fn with_detectors(
        influence_map: &PyDagFaultInfluenceMap,
        detectors: Vec<Vec<i32>>,
        observables: Vec<Vec<i32>>,
        p1: f64,
        p2: f64,
        p_meas: f64,
        p_prep: f64,
        p_idle: Option<f64>,
        t1: Option<f64>,
        t2: Option<f64>,
        idle_rz: Option<f64>,
        p_idle_linear_rate: Option<f64>,
        p_idle_quadratic_rate: Option<f64>,
        p_idle_x_linear_rate: Option<f64>,
        p_idle_y_linear_rate: Option<f64>,
        p_idle_z_linear_rate: Option<f64>,
        p_idle_x_quadratic_rate: Option<f64>,
        p_idle_y_quadratic_rate: Option<f64>,
        p_idle_z_quadratic_rate: Option<f64>,
        p_idle_quadratic_sine_rate: Option<f64>,
        p_idle_x_quadratic_sine_rate: Option<f64>,
        p_idle_y_quadratic_sine_rate: Option<f64>,
        p_idle_z_quadratic_sine_rate: Option<f64>,
        p1_weights: Option<BTreeMap<String, f64>>,
        p2_weights: Option<BTreeMap<String, f64>>,
        p2_replacement_approximation: Option<String>,
        p_meas_crosstalk_local: Option<f64>,
        p_meas_crosstalk_global: Option<f64>,
        p_meas_crosstalk_model: Option<BTreeMap<String, f64>>,
        measurement_crosstalk_dem_mode: Option<String>,
        p2_gate_rates: Option<BTreeMap<String, f64>>,
        p1_gate_rates: Option<BTreeMap<String, f64>>,
    ) -> PyResult<Self> {
        let noise = apply_noise_options(
            NoiseConfig::new(p1, p2, p_meas, p_prep),
            p_idle,
            t1,
            t2,
            idle_rz,
            p_idle_linear_rate,
            p_idle_quadratic_rate,
            p_idle_x_linear_rate,
            p_idle_y_linear_rate,
            p_idle_z_linear_rate,
            p_idle_x_quadratic_rate,
            p_idle_y_quadratic_rate,
            p_idle_z_quadratic_rate,
            p_idle_quadratic_sine_rate,
            p_idle_x_quadratic_sine_rate,
            p_idle_y_quadratic_sine_rate,
            p_idle_z_quadratic_sine_rate,
            p1_weights,
            p2_weights,
            p2_replacement_approximation,
            p_meas_crosstalk_local,
            p_meas_crosstalk_global,
            p_meas_crosstalk_model,
            measurement_crosstalk_dem_mode,
            p2_gate_rates,
            p1_gate_rates,
        )?;
        let inner = RustNewDemSamplerBuilder::new(&influence_map.inner)
            .with_noise_config(noise)
            .with_detectors(detectors, observables)
            .build()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Create a sampler directly from an influence map with uniform noise.
    ///
    /// Args:
    ///     `influence_map`: A `DagFaultInfluenceMap` from `DagFaultAnalyzer` or `InfluenceBuilder`.
    ///     `p_error`: Uniform depolarizing error probability per fault location.
    #[staticmethod]
    fn from_influence_map(influence_map: &PyDagFaultInfluenceMap, p_error: f64) -> PyResult<Self> {
        let inner = RustNewDemSamplerBuilder::new(&influence_map.inner)
            .with_uniform_noise(p_error)
            .raw_measurements()
            .build()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Create a sampler from an influence map with circuit-level noise.
    #[staticmethod]
    fn from_influence_map_circuit_noise(
        influence_map: &PyDagFaultInfluenceMap,
        p1: f64,
        p2: f64,
        p_meas: f64,
        p_prep: f64,
    ) -> PyResult<Self> {
        let inner = RustNewDemSamplerBuilder::new(&influence_map.inner)
            .with_noise(p1, p2, p_meas, p_prep)
            .raw_measurements()
            .build()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Number of mechanisms in the sampler.
    #[getter]
    fn num_mechanisms(&self) -> usize {
        self.inner.num_mechanisms()
    }

    /// Number of output channels (detectors or measurements).
    #[getter]
    fn num_outputs(&self) -> usize {
        self.inner.num_outputs()
    }

    /// Number of detectors (alias for `num_outputs`).
    #[getter]
    fn num_detectors(&self) -> usize {
        self.inner.num_outputs()
    }

    /// Number of observables when sampler metadata is known.
    #[getter]
    fn num_observables(&self) -> usize {
        self.inner.num_observables()
    }

    /// Total number of outputs in the DEM `L<n>` namespace.
    #[getter]
    fn num_dem_outputs(&self) -> usize {
        self.inner.num_dem_outputs()
    }

    /// Number of tracked Paulis.
    #[getter]
    fn num_tracked_paulis(&self) -> usize {
        self.inner.num_tracked_paulis()
    }

    /// Sample a single shot.
    ///
    /// Args:
    ///     seed: Optional random seed for reproducibility.
    ///
    /// Returns:
    ///     Tuple of (`detection_events`, `dem_output_flips`) as boolean lists.
    #[pyo3(signature = (seed=None))]
    fn sample(&self, seed: Option<u64>) -> (Vec<bool>, Vec<bool>) {
        use pecos_random::PecosRng;
        use rand::RngExt;

        let mut rng = match seed {
            Some(s) => PecosRng::seed_from_u64(s),
            None => PecosRng::seed_from_u64(rand::rng().random()),
        };

        self.inner.sample(&mut rng)
    }

    /// Sample multiple shots into a `SampleBatch` held in Rust memory.
    ///
    /// The batch can be decoded by multiple decoders without re-sampling or
    /// materialized as shots-major Python lists with
    /// `SampleBatch.detector_events()` and `SampleBatch.observable_flips()`.
    /// For a raw-measurement sampler, the first set of columns contains raw
    /// measurements rather than detector events, so the batch data accessors
    /// work but its decode methods raise `ValueError`.
    ///
    /// Args:
    ///     num_shots: Number of shots to sample.
    ///     seed: Optional random seed for reproducibility.
    ///
    /// Returns:
    ///     `SampleBatch` object with samples held in Rust memory.
    #[pyo3(signature = (num_shots, seed=None))]
    fn sample_batch(&self, num_shots: usize, seed: Option<u64>) -> PySampleBatch {
        use pecos_random::PecosRng;
        use rand::RngExt;

        let actual_seed = seed.unwrap_or_else(|| rand::rng().random());
        let mut rng = PecosRng::seed_from_u64(actual_seed);

        if self.inner.mode() == OutputMode::RawMeasurements {
            let (detection_events, observable_flips) = self.inner.sample_batch(num_shots, &mut rng);
            return PySampleBatch::from_bool_rows(
                detection_events,
                observable_flips,
                true,
                Some(actual_seed),
            );
        }
        let (det_columns, obs_columns) = self.inner.sample_batch_geometric(num_shots, &mut rng);
        PySampleBatch::from_columnar(det_columns, obs_columns, num_shots, Some(actual_seed))
    }

    /// Sample multiple shots and XOR a known Pauli-frame mask into the outputs.
    ///
    /// Args:
    ///     num_shots: Number of shots to sample.
    ///     lookup: Pauli-frame lookup built from the same circuit metadata.
    ///     pauli_masks: Integer array with shape `(num_shots, num_pauli_sites)`.
    ///         Values are 0=I, 1=X, 2=Y, 3=Z.
    ///     seed: Optional random seed for reproducibility.
    ///
    /// Returns:
    ///     `SampleBatch` containing the sampled and XOR-adjusted outputs.
    #[pyo3(signature = (num_shots, lookup, pauli_masks, seed=None))]
    fn sample_batch_with_pauli_masks(
        &self,
        num_shots: usize,
        lookup: &PyPauliFrameLookup,
        pauli_masks: &Bound<'_, pyo3::PyAny>,
        seed: Option<u64>,
    ) -> PyResult<PySampleBatch> {
        use pecos_random::PecosRng;
        use rand::RngExt;

        if lookup.inner.num_detectors() != self.inner.num_outputs() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "pauli frame lookup has {} detector(s), sampler has {}",
                lookup.inner.num_detectors(),
                self.inner.num_outputs()
            )));
        }
        if lookup.inner.num_observables() != self.inner.num_dem_outputs() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "pauli frame lookup has {} observable(s), sampler has {}",
                lookup.inner.num_observables(),
                self.inner.num_dem_outputs()
            )));
        }

        let (mask_values, mask_rows, mask_cols) = extract_pauli_mask_values(pauli_masks)?;
        let actual_seed = seed.unwrap_or_else(|| rand::rng().random());
        let mut rng = PecosRng::seed_from_u64(actual_seed);

        let (mut det_events, mut obs_flips) = self.inner.sample_batch(num_shots, &mut rng);
        lookup
            .inner
            .apply_mask_values(
                &mask_values,
                mask_rows,
                mask_cols,
                &mut det_events,
                &mut obs_flips,
            )
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PySampleBatch::from_bool_rows(
            det_events,
            obs_flips,
            self.inner.mode() == OutputMode::RawMeasurements,
            Some(actual_seed),
        ))
    }

    /// Sample direct tracked-Pauli flips.
    ///
    /// Raises:
    ///     RuntimeError: If this sampler carries tracked Paulis but the
    ///         backend cannot evaluate tracked-Pauli flips directly.
    #[pyo3(signature = (seed=None))]
    fn sample_tracked_paulis(&self, seed: Option<u64>) -> PyResult<Vec<bool>> {
        use pecos_random::PecosRng;
        use rand::RngExt;

        let mut rng = match seed {
            Some(s) => PecosRng::seed_from_u64(s),
            None => PecosRng::seed_from_u64(rand::rng().random()),
        };

        self.inner
            .sample_tracked_pauli_flips(&mut rng)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Sample direct tracked-Pauli flips for multiple shots.
    ///
    /// Raises:
    ///     RuntimeError: If this sampler carries tracked Paulis but the
    ///         backend cannot evaluate tracked-Pauli flips directly.
    #[pyo3(signature = (num_shots, seed=None))]
    fn sample_tracked_pauli_batch(
        &self,
        num_shots: usize,
        seed: Option<u64>,
    ) -> PyResult<Vec<Vec<bool>>> {
        use pecos_random::PecosRng;
        use rand::RngExt;

        let mut rng = match seed {
            Some(s) => PecosRng::seed_from_u64(s),
            None => PecosRng::seed_from_u64(rand::rng().random()),
        };

        self.inner
            .sample_tracked_pauli_batch(num_shots, &mut rng)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Compute statistics without storing individual shots.
    ///
    /// This is the most efficient method for threshold estimation when you
    /// only need aggregate statistics (logical error rate, syndrome rate).
    ///
    /// Args:
    ///     `num_shots`: Number of shots to sample.
    ///     seed: Optional random seed for reproducibility.
    ///
    /// Returns:
    ///     Dictionary with statistics:
    ///     - `total_shots`: Number of shots
    ///     - `logical_error_count`: Shots with selected observable flips
    ///     - `syndrome_count`: Shots with non-trivial syndrome
    ///     - `undetectable_count`: Shots with observable flips and no syndrome
    ///     - `logical_error_rate`: Fraction with selected observable flips
    ///     - `syndrome_rate`: Fraction with syndromes
    ///     - `undetectable_rate`: Fraction with undetectable errors
    #[pyo3(signature = (num_shots, seed=None))]
    fn sample_statistics(
        &self,
        num_shots: usize,
        seed: Option<u64>,
        py: Python<'_>,
    ) -> PyResult<Py<pyo3::types::PyDict>> {
        use rand::RngExt;

        let actual_seed = seed.unwrap_or_else(|| rand::rng().random());
        let stats = self.inner.sample_statistics(num_shots, actual_seed);
        let observable_indices = self.inner.observable_ids();
        let tracked_pauli_result = self.inner.tracked_pauli_ids();
        let tracked_pauli_statistics_error =
            tracked_pauli_result.as_ref().err().map(ToString::to_string);
        let tracked_pauli_indices = tracked_pauli_result.unwrap_or_default();
        let per_observable = stats.observable_counts(&observable_indices);
        let per_tracked_pauli: Vec<usize> = tracked_pauli_indices
            .iter()
            .filter_map(|&idx| stats.dem_output_counts().get(idx).copied())
            .collect();
        let logical_rates = stats.logical_rates(&observable_indices);
        #[allow(clippy::cast_precision_loss)] // Counts are converted to rates for Python reporting.
        let n = stats.total_shots as f64;
        #[allow(clippy::cast_precision_loss)] // Counts are converted to rates for Python reporting.
        let tracked_pauli_rates: Vec<f64> = per_tracked_pauli
            .iter()
            .map(|&count| count as f64 / n)
            .collect();

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("total_shots", stats.total_shots)?;
        dict.set_item("logical_error_count", stats.logical_error_count)?;
        dict.set_item("syndrome_count", stats.syndrome_count)?;
        dict.set_item("undetectable_count", stats.undetectable_count)?;
        dict.set_item("logical_error_rate", stats.logical_error_rate())?;
        dict.set_item("syndrome_rate", stats.syndrome_rate())?;
        dict.set_item("undetectable_rate", stats.undetectable_rate())?;
        dict.set_item("per_detector", &stats.per_detector)?;
        dict.set_item("per_observable", per_observable)?;
        dict.set_item("per_tracked_pauli", per_tracked_pauli)?;
        dict.set_item("per_dem_output", stats.dem_output_counts())?;
        dict.set_item("detector_rates", stats.detector_rates())?;
        dict.set_item("logical_rates", logical_rates)?;
        dict.set_item("tracked_pauli_rates", tracked_pauli_rates)?;
        dict.set_item("dem_output_rates", stats.dem_output_rates())?;
        dict.set_item(
            "tracked_pauli_statistics_supported",
            tracked_pauli_statistics_error.is_none(),
        )?;
        if let Some(error) = tracked_pauli_statistics_error {
            dict.set_item("tracked_pauli_statistics_error", error)?;
        }
        Ok(dict.unbind())
    }

    /// Get labels for the sampler's output channels.
    ///
    /// Returns a dict with:
    ///     - `outputs`: labels for output channels (raw measurements or detectors)
    ///     - `dem_outputs`: labels for all DEM `L<n>` targets
    ///     - `observables`: labels for observables
    ///     - `tracked_paulis`: labels for tracked Paulis
    ///     - `dual_detectors`: labels for dual-output detector channels
    fn labels(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        let labels = self.inner.labels();
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("outputs", &labels.outputs)?;
        dict.set_item("dem_outputs", &labels.dem_output_labels)?;
        dict.set_item("observables", &labels.dem_output_labels)?;
        dict.set_item("tracked_paulis", &labels.tracked_pauli_labels)?;
        dict.set_item("dual_detectors", &labels.dual_detectors)?;
        Ok(dict.unbind())
    }

    /// Sample, decode, and score shots through the planned batch executor.
    ///
    /// Sampling ABI v1 guarantees that a fixed `(seed, num_shots)` produces the
    /// same SHOT stream for every worker count and execution path, including
    /// sequential and native. Each canonical 1024-shot chunk owns a
    /// deterministic RNG stream and consumes exactly one single-shot sampler
    /// call per shot.
    ///
    /// Predictions and error counts match as well, EXCEPT when
    /// `reproducibility_warnings` on the result is non-empty: a wall-clock-limited
    /// decoder (for example `mwpf(timeout=...)`) run in parallel can decode
    /// differently because CPU contention changes which shots reach the solver's
    /// deadline. Timing measurements are always outside the guarantee.
    ///
    /// Parallelism is granted per 1024-shot chunk, so the effective concurrency
    /// is capped at `ceil(num_shots / 1024)` regardless of the requested workers.
    ///
    /// Predictions and truth are restricted to the sampler's observable DEM
    /// outputs, matching `sample_decode_count`; `SampleBatch.decode` scores
    /// against the full DEM-output row instead, so the two can differ on
    /// samplers whose DEM outputs exceed their observables.
    ///
    /// Args:
    ///     dem: DEM text used to construct the decoder. It may deliberately be
    ///         a different projection from the sampler's own model.
    ///     `num_shots`: Number of shots to sample and decode.
    ///     decoder: A typed `DecoderSpec` or legacy decoder string.
    ///     seed: Optional sampling seed. The resolved seed is returned as
    ///         `sampling_seed_used` and can replay the run.
    ///     workers: Optional exact worker count.
    ///     predictions: Retain predictions in absolute shot order.
    ///     timing: Retain decode-call timings. Sampling time is excluded from
    ///         individual samples but included in `wall_elapsed`.
    ///
    /// Returns:
    ///     `DecodeResult` scored against the sampler's observable flips.
    #[pyo3(signature = (dem, num_shots, decoder=None, *, seed=None, workers=None, predictions=false, timing=false))]
    fn decode(
        &self,
        py: Python<'_>,
        dem: &str,
        num_shots: usize,
        decoder: Option<&Bound<'_, PyAny>>,
        seed: Option<u64>,
        workers: Option<i64>,
        predictions: bool,
        timing: bool,
    ) -> PyResult<batch_decode::PyDecodeResult> {
        use rand::RngExt;

        // Raw samples are measurements, not syndromes. Preserve the established
        // precedence by rejecting that mode before inspecting the decoder.
        if self.inner.mode() == OutputMode::RawMeasurements {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "raw-measurement DemSampler outputs are measurements, not detector events, and cannot be decoded",
            ));
        }

        let decoder = decoder.ok_or_else(|| {
            pyo3::exceptions::PyTypeError::new_err("decoder is a required argument")
        })?;
        let spec = if decoder.is_instance_of::<PyString>() {
            let decoder_type = decoder.extract::<&str>()?;
            pecos_decoders::DecoderSpec::parse(decoder_type).map_err(decoder_parse_error_to_py)?
        } else if let Ok(spec) = decoder.extract::<PyRef<'_, PyDecoderSpec>>() {
            spec.inner.clone()
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "decoder must be a pecos.decoders.DecoderSpec or legacy decoder string",
            ));
        };

        let explicit_workers = workers
            .map(|workers| {
                if workers <= 0 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "workers must be at least 1",
                    ));
                }
                usize::try_from(workers).map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(
                        "workers is too large for this platform",
                    )
                })
            })
            .transpose()?;
        let mut plan =
            pecos_decoders::batch::plan_execution(pecos_decoders::batch::ExecutionPlanInputs {
                traits: spec.execution_traits(),
                num_shots,
                native_batch_capable: spec.native_batch_capable(),
                timing,
                explicit_workers,
                available_threads: rayon::current_num_threads(),
            })
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;

        // Report the workers that can actually run: the fused unit of work is a
        // sampling chunk, so anything beyond one worker per chunk would idle.
        if plan.path == pecos_decoders::batch::ExecutionPath::Parallel {
            plan.workers_used = plan
                .workers_used
                .min(pecos_decoders::batch::fused_worker_cap(num_shots));
        }

        // Resolve entropy once before dispatch, including for an empty run.
        let actual_seed = seed.unwrap_or_else(|| rand::rng().random());
        let output = py
            .detach(|| {
                sampler_decode::execute(
                    &self.inner,
                    dem,
                    &spec,
                    &plan,
                    num_shots,
                    actual_seed,
                    sampler_decode::DecodeOptions::new(predictions, timing),
                )
            })
            .map_err(batch_decode::BatchExecutionError::into_pyerr)?;
        batch_decode::PyDecodeResult::from_sampler_execution(
            py,
            num_shots,
            plan,
            output,
            actual_seed,
        )
    }

    /// Sample and decode in a tight Rust loop, returning only the error count.
    ///
    /// This is the fastest path for threshold estimation -- no per-shot data
    /// crosses the Rust/Python boundary. The sampler produces detection events,
    /// the decoder decodes them via the `ObservableDecoder` trait, and errors
    /// are counted, all in Rust.
    ///
    /// Args:
    ///     dem: DEM string in standard DEM text format for the decoder.
    ///     `num_shots`: Number of shots to sample and decode.
    ///     `decoder_type`: "pymatching", "`pymatching_correlated`",
    ///                   "`pymatching_uncorrelated`", "tesseract", or another
    ///                   decoder accepted by `create_observable_decoder`.
    ///     seed: Optional random seed for reproducibility.
    ///
    /// Returns:
    ///     Number of logical errors (mismatches between decoder prediction and true flip).
    #[pyo3(signature = (dem, num_shots, decoder_type="pymatching", seed=None))]
    fn sample_decode_count(
        &self,
        dem: &str,
        num_shots: usize,
        decoder_type: &str,
        seed: Option<u64>,
    ) -> PyResult<usize> {
        use pecos_random::PecosRng;
        use rand::RngExt;

        let actual_seed = seed.unwrap_or_else(|| rand::rng().random());
        let mut rng = PecosRng::seed_from_u64(actual_seed);

        let decoder = create_observable_decoder(dem, decoder_type)?;
        let observable_mask = self.inner.observable_dem_output_mask();
        let mut decoder = MaskedObservableDecoder::new(decoder, observable_mask.clone());

        // Tight sample+decode loop -- no Python involvement.
        // Single-threaded: sample and decode sequentially.
        let mut syndrome = vec![0u8; self.inner.num_detectors()];
        count_decoder_mismatches(
            0..num_shots,
            &mut syndrome,
            |_, buffer| {
                let (det_events, obs_flips) = self.inner.sample(&mut rng);
                debug_assert_eq!(det_events.len(), buffer.len());
                for (value, event) in buffer.iter_mut().zip(det_events) {
                    *value = u8::from(event);
                }
                self.inner
                    .observable_mask_from_dem_output_flips(&obs_flips, &observable_mask)
            },
            &mut decoder,
        )
        .map_err(map_shot_decode_error)
    }

    /// Parallel sample+decode: distributes shots across threads.
    ///
    /// Each thread gets its own sampler clone and decoder instance.
    /// Much faster for slow decoders (Tesseract) where decode time dominates.
    ///
    /// Args:
    ///     dem: DEM string in standard DEM text format for the decoder.
    ///     `num_shots`: Number of shots to sample and decode.
    ///     `decoder_type`: "pymatching", "`pymatching_correlated`",
    ///                   "`pymatching_uncorrelated`", "tesseract", "`bp_osd`",
    ///                   "`bp_lsd`", or "`union_find`".
    ///     seed: Optional base random seed. Each thread gets seed + `thread_id`.
    ///     `num_workers`: Number of parallel workers (default: number of CPUs).
    ///
    /// Returns:
    ///     Number of logical errors.
    #[pyo3(signature = (dem, num_shots, decoder_type="pymatching", seed=None, num_workers=None))]
    fn sample_decode_count_parallel(
        &self,
        dem: &str,
        num_shots: usize,
        decoder_type: &str,
        seed: Option<u64>,
        num_workers: Option<usize>,
    ) -> PyResult<usize> {
        use rayon::prelude::*;

        let actual_seed = seed.unwrap_or(0);
        let n_workers = resolve_worker_count(num_workers)?;

        // Validate decoder type early
        create_observable_decoder(dem, decoder_type)?;

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_workers)
            .build()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let shots_per_worker = num_shots / n_workers;
        let remainder = num_shots % n_workers;

        let sampler = &self.inner;
        let observable_mask = sampler.observable_dem_output_mask();
        let dem_str = dem.to_string();
        let dt = decoder_type.to_string();

        let worker_results: Vec<Result<usize, ShotDecodeError>> = pool.install(|| {
            (0..n_workers)
                .into_par_iter()
                .map(|worker_id| {
                    use pecos_random::PecosRng;

                    let my_shots = shots_per_worker + usize::from(worker_id < remainder);
                    if my_shots == 0 {
                        return Ok(0);
                    }
                    let start = worker_id * shots_per_worker + worker_id.min(remainder);
                    let end = start + my_shots;

                    let my_sampler = sampler.clone();
                    let mut my_rng =
                        PecosRng::seed_from_u64(actual_seed.wrapping_add(worker_id as u64));
                    // unwrap is safe: we validated above
                    let decoder = create_observable_decoder(&dem_str, &dt).unwrap();
                    let mut decoder =
                        MaskedObservableDecoder::new(decoder, observable_mask.clone());
                    let mut syndrome = vec![0u8; my_sampler.num_detectors()];
                    count_decoder_mismatches(
                        start..end,
                        &mut syndrome,
                        |_, buffer| {
                            let (det_events, obs_flips) = my_sampler.sample(&mut my_rng);
                            debug_assert_eq!(det_events.len(), buffer.len());
                            for (value, event) in buffer.iter_mut().zip(det_events) {
                                *value = u8::from(event);
                            }
                            my_sampler
                                .observable_mask_from_dem_output_flips(&obs_flips, &observable_mask)
                        },
                        &mut decoder,
                    )
                })
                .collect()
        });

        worker_results
            .into_iter()
            .try_fold(0usize, |total, result| {
                result
                    .map(|count| total + count)
                    .map_err(map_shot_decode_error)
            })
    }

    fn __repr__(&self) -> String {
        format!(
            "DemSampler(mechanisms={}, outputs={}, dem_outputs={}, observables={}, tracked_paulis={})",
            self.num_mechanisms(),
            self.num_outputs(),
            self.num_dem_outputs(),
            self.num_observables(),
            self.num_tracked_paulis(),
        )
    }
}

/// Builder for `DemSampler`.
///
/// Constructs a `DemSampler` from a fault influence map, noise parameters,
/// and explicit detector / observable definitions.
#[pyclass(name = "DemSamplerBuilder", module = "pecos_rslib.qec")]
pub struct PyDemSamplerBuilder {
    influence_map: RustDagFaultInfluenceMap,
    noise: NoiseConfig,
    detectors_json: Option<String>,
    observables_json: Option<String>,
    measurement_order: Option<Vec<usize>>,
}

#[pymethods]
impl PyDemSamplerBuilder {
    /// Create a new builder from a fault influence map.
    #[new]
    fn new(influence_map: &PyDagFaultInfluenceMap) -> Self {
        Self {
            influence_map: influence_map.inner.clone(),
            noise: NoiseConfig::default(),
            detectors_json: None,
            observables_json: None,
            measurement_order: None,
        }
    }

    /// Set noise parameters.
    #[pyo3(signature = (p1, p2, p_meas, p_prep, p_idle=None, t1=None, t2=None, idle_rz=None, p_idle_linear_rate=None, p_idle_quadratic_rate=None, p_idle_x_linear_rate=None, p_idle_y_linear_rate=None, p_idle_z_linear_rate=None, p_idle_x_quadratic_rate=None, p_idle_y_quadratic_rate=None, p_idle_z_quadratic_rate=None, p_idle_quadratic_sine_rate=None, p_idle_x_quadratic_sine_rate=None, p_idle_y_quadratic_sine_rate=None, p_idle_z_quadratic_sine_rate=None, p1_weights=None, p2_weights=None, p2_replacement_approximation=None, p_meas_crosstalk_local=None, p_meas_crosstalk_global=None, p_meas_crosstalk_model=None, measurement_crosstalk_dem_mode=None, p2_gate_rates=None, p1_gate_rates=None))]
    #[allow(clippy::too_many_arguments)]
    fn with_noise(
        mut slf: PyRefMut<'_, Self>,
        p1: f64,
        p2: f64,
        p_meas: f64,
        p_prep: f64,
        p_idle: Option<f64>,
        t1: Option<f64>,
        t2: Option<f64>,
        idle_rz: Option<f64>,
        p_idle_linear_rate: Option<f64>,
        p_idle_quadratic_rate: Option<f64>,
        p_idle_x_linear_rate: Option<f64>,
        p_idle_y_linear_rate: Option<f64>,
        p_idle_z_linear_rate: Option<f64>,
        p_idle_x_quadratic_rate: Option<f64>,
        p_idle_y_quadratic_rate: Option<f64>,
        p_idle_z_quadratic_rate: Option<f64>,
        p_idle_quadratic_sine_rate: Option<f64>,
        p_idle_x_quadratic_sine_rate: Option<f64>,
        p_idle_y_quadratic_sine_rate: Option<f64>,
        p_idle_z_quadratic_sine_rate: Option<f64>,
        p1_weights: Option<BTreeMap<String, f64>>,
        p2_weights: Option<BTreeMap<String, f64>>,
        p2_replacement_approximation: Option<String>,
        p_meas_crosstalk_local: Option<f64>,
        p_meas_crosstalk_global: Option<f64>,
        p_meas_crosstalk_model: Option<BTreeMap<String, f64>>,
        measurement_crosstalk_dem_mode: Option<String>,
        p2_gate_rates: Option<BTreeMap<String, f64>>,
        p1_gate_rates: Option<BTreeMap<String, f64>>,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.noise = apply_noise_options(
            NoiseConfig::new(p1, p2, p_meas, p_prep),
            p_idle,
            t1,
            t2,
            idle_rz,
            p_idle_linear_rate,
            p_idle_quadratic_rate,
            p_idle_x_linear_rate,
            p_idle_y_linear_rate,
            p_idle_z_linear_rate,
            p_idle_x_quadratic_rate,
            p_idle_y_quadratic_rate,
            p_idle_z_quadratic_rate,
            p_idle_quadratic_sine_rate,
            p_idle_x_quadratic_sine_rate,
            p_idle_y_quadratic_sine_rate,
            p_idle_z_quadratic_sine_rate,
            p1_weights,
            p2_weights,
            p2_replacement_approximation,
            p_meas_crosstalk_local,
            p_meas_crosstalk_global,
            p_meas_crosstalk_model,
            measurement_crosstalk_dem_mode,
            p2_gate_rates,
            p1_gate_rates,
        )?;
        Ok(slf)
    }

    /// Set detector definitions from JSON.
    ///
    /// Accepts either legacy detector rows with an `"id"` key or public surface
    /// descriptor rows with a `"detector_id"` key.
    fn with_detectors_json(mut slf: PyRefMut<'_, Self>, json: String) -> PyRefMut<'_, Self> {
        slf.detectors_json = Some(json);
        slf
    }

    /// Set observable definitions from JSON.
    ///
    /// Tracked Paulis are carried by the influence map; this helper is for
    /// observable metadata.
    fn with_observables_json(mut slf: PyRefMut<'_, Self>, json: String) -> PyRefMut<'_, Self> {
        slf.observables_json = Some(json);
        slf
    }

    /// Set the measurement order mapping from `TickCircuit`.
    fn with_measurement_order(
        mut slf: PyRefMut<'_, Self>,
        order: Vec<usize>,
    ) -> PyRefMut<'_, Self> {
        slf.measurement_order = Some(order);
        slf
    }

    /// Build the `DemSampler`.
    fn build(&self) -> PyResult<PyDemSampler> {
        let mut builder = RustNewDemSamplerBuilder::new(&self.influence_map)
            .with_noise_config(self.noise.clone());

        if let Some(ref json) = self.detectors_json {
            builder = builder
                .with_detectors_json(json)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
        }

        if let Some(ref json) = self.observables_json {
            builder = builder
                .with_observables_json(json)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
        }

        if let Some(ref order) = self.measurement_order {
            builder = builder.with_measurement_order(order.clone());
        }

        let inner = builder
            .build()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyDemSampler { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "DemSamplerBuilder(p1={}, p2={}, p_meas={}, p_prep={}, p_idle={:?})",
            self.noise.p1, self.noise.p2, self.noise.p_meas, self.noise.p_prep, self.noise.p_idle
        )
    }
}

// =============================================================================
// DEM Equivalence Validation
// =============================================================================

/// Result of DEM equivalence comparison.
///
/// Contains detailed information about whether two DEMs are equivalent
/// and what differences were found.
///
/// # Example
///
/// ```python
/// from pecos_rslib.qec import compare_dems_exact
///
/// result = compare_dems_exact(dem1_str, dem2_str, prob_tolerance=0.001)
/// if result.equivalent:
///     print("DEMs are equivalent")
/// else:
///     print(f"Max rate difference: {result.max_rate_difference}")
///     for mech in result.only_in_dem1:
///         print(f"Only in DEM1: {mech}")
/// ```
#[pyclass(name = "EquivalenceResult", module = "pecos_rslib.qec")]
pub struct PyEquivalenceResult {
    inner: RustEquivalenceResult,
}

#[pymethods]
impl PyEquivalenceResult {
    /// Whether the DEMs are equivalent within tolerance.
    #[getter]
    fn equivalent(&self) -> bool {
        self.inner.equivalent
    }

    /// Maximum absolute difference in rates/probabilities.
    #[getter]
    fn max_rate_difference(&self) -> f64 {
        self.inner.max_rate_difference
    }

    /// Maximum relative difference in rates/probabilities.
    #[getter]
    fn max_relative_difference(&self) -> f64 {
        self.inner.max_relative_difference
    }

    /// Correlation of detector rates (statistical comparison).
    #[getter]
    fn correlation(&self) -> f64 {
        self.inner.correlation
    }

    /// Alias for correlation (matches Python API).
    #[getter]
    fn syndrome_rate_correlation(&self) -> f64 {
        self.inner.correlation
    }

    /// Per-detector rate differences (statistical comparison).
    #[getter]
    fn detector_rate_differences(&self) -> Vec<f64> {
        self.inner.detector_rate_differences.clone()
    }

    /// Per-observable rate differences (statistical comparison).
    #[getter]
    fn observable_rate_differences(&self) -> Vec<f64> {
        self.inner.observable_rate_differences.clone()
    }

    /// Number of mechanisms in first DEM.
    #[getter]
    fn dem1_mechanism_count(&self) -> usize {
        self.inner.details.dem1_mechanism_count
    }

    /// Number of mechanisms in second DEM.
    #[getter]
    fn dem2_mechanism_count(&self) -> usize {
        self.inner.details.dem2_mechanism_count
    }

    /// Mechanisms only in first DEM.
    #[getter]
    fn only_in_dem1(&self) -> Vec<String> {
        self.inner.details.only_in_dem1.clone()
    }

    /// Mechanisms only in second DEM.
    #[getter]
    fn only_in_dem2(&self) -> Vec<String> {
        self.inner.details.only_in_dem2.clone()
    }

    /// Get comparison details as a dictionary.
    fn details(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item(
            "dem1_mechanism_count",
            self.inner.details.dem1_mechanism_count,
        )?;
        dict.set_item(
            "dem2_mechanism_count",
            self.inner.details.dem2_mechanism_count,
        )?;
        dict.set_item("only_in_dem1", self.inner.details.only_in_dem1.clone())?;
        dict.set_item("only_in_dem2", self.inner.details.only_in_dem2.clone())?;

        let mismatches: Vec<_> = self
            .inner
            .details
            .prob_mismatches
            .iter()
            .map(|m| (m.target.clone(), m.dem1_prob, m.dem2_prob, m.difference))
            .collect();
        dict.set_item("prob_mismatches", mismatches)?;

        Ok(dict.unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "EquivalenceResult(equivalent={}, max_rate_diff={:.6})",
            self.inner.equivalent, self.inner.max_rate_difference
        )
    }
}

/// A parsed Detector Error Model.
///
/// Parses standard and PECOS DEM strings and provides methods for
/// aggregation and sampling.
///
/// # Example
///
/// ```python
/// from pecos_rslib.qec import ParsedDem
///
/// dem = ParsedDem.from_string("error(0.01) D0 D1\\nerror(0.02) D1 D2")
/// print(f"Mechanisms: {dem.num_mechanisms}")
/// print(f"Detectors: {dem.num_detectors}")
/// ```
#[pyclass(name = "ParsedDem", module = "pecos_rslib.qec")]
pub struct PyParsedDem {
    inner: RustParsedDem,
}

#[pymethods]
impl PyParsedDem {
    /// Parse a DEM from a string.
    ///
    /// Args:
    ///     `dem_str`: DEM string in standard or PECOS DEM text format.
    ///
    /// Returns:
    ///     `ParsedDem` object.
    ///
    /// Raises:
    ///     `ValueError`: If the DEM string is malformed.
    #[staticmethod]
    fn from_string(dem_str: &str) -> PyResult<Self> {
        let inner = dem_str
            .parse::<RustParsedDem>()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Number of mechanisms in the DEM.
    #[getter]
    fn num_mechanisms(&self) -> usize {
        self.inner.mechanisms.len()
    }

    /// Number of detectors (max ID + 1).
    #[getter]
    fn num_detectors(&self) -> u32 {
        self.inner.num_detectors
    }

    /// Number of observables.
    #[getter]
    fn num_observables(&self) -> u32 {
        self.inner.num_observables()
    }

    /// Total number of outputs in the DEM `L<n>` namespace.
    #[getter]
    fn num_dem_outputs(&self) -> u32 {
        self.inner.num_dem_outputs()
    }

    /// Number of tracked Paulis.
    #[getter]
    fn num_tracked_paulis(&self) -> u32 {
        self.inner.num_tracked_paulis()
    }

    /// Convert to a decomposed (graphlike) DEM string.
    ///
    /// Mechanisms with <= 2 detectors pass through unchanged.
    /// Hyperedges (3+ detectors) cannot be decomposed without Pauli
    /// provenance and will raise an error.
    ///
    /// For proper decomposition, use ``coherent_dem_decomposed()``
    /// or ``noise_characterization()`` which track X/Z components.
    fn to_string_decomposed(&self) -> PyResult<String> {
        self.inner
            .to_string_decomposed()
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Aggregate mechanisms by their effect.
    ///
    /// Returns a dictionary mapping (`detector_tuple`, `observable_tuple`) to
    /// combined probability. Probabilities are combined using the independent
    /// error formula: p1*(1-p2) + p2*(1-p1).
    ///
    /// Returns:
    ///     Dictionary of {(detectors, observables): probability}.
    fn aggregate(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        let agg = self.inner.aggregate();
        let dict = pyo3::types::PyDict::new(py);

        for (key, prob) in agg {
            let det_tuple = pyo3::types::PyTuple::new(py, key.detectors.iter())?;
            let obs_tuple = pyo3::types::PyTuple::new(py, key.observables.iter())?;
            let key_tuple =
                pyo3::types::PyTuple::new(py, [det_tuple.as_any(), obs_tuple.as_any()])?;
            dict.set_item(key_tuple, prob)?;
        }

        Ok(dict.unbind())
    }

    /// Sample from this DEM.
    ///
    /// Args:
    ///     seed: Optional random seed for reproducibility.
    ///
    /// Returns:
    ///     Tuple of (`detector_events`, `dem_output_flips`) as boolean lists.
    #[pyo3(signature = (seed=None))]
    fn sample(&self, seed: Option<u64>) -> (Vec<bool>, Vec<bool>) {
        use pecos_random::PecosRng;
        use rand::RngExt;

        let mut rng = match seed {
            Some(s) => PecosRng::seed_from_u64(s),
            None => PecosRng::seed_from_u64(rand::rng().random()),
        };

        self.inner.sample(&mut rng)
    }

    /// Sample multiple shots from this DEM into a `SampleBatch`.
    ///
    /// Args:
    ///     num_shots: Number of shots to sample.
    ///     seed: Optional random seed for reproducibility.
    ///
    /// Returns:
    ///     `SampleBatch` object with samples held in Rust memory.
    #[pyo3(signature = (num_shots, seed=None))]
    fn sample_batch(&self, num_shots: usize, seed: Option<u64>) -> PySampleBatch {
        use pecos_random::PecosRng;
        use rand::RngExt;

        let actual_seed = seed.unwrap_or_else(|| rand::rng().random());
        let mut rng = PecosRng::seed_from_u64(actual_seed);

        let (detector_events, observable_flips) = self.inner.sample_batch(num_shots, &mut rng);
        PySampleBatch::from_bool_rows(detector_events, observable_flips, false, Some(actual_seed))
    }

    /// Convert to an optimized `DemSampler` for fast batch sampling.
    ///
    /// The `DemSampler` uses geometric skip sampling and parallel chunked
    /// processing, which is significantly faster than `sample_batch` for
    /// large shot counts and low error rates.
    ///
    /// Returns:
    ///     `DemSampler`: Optimized sampler for this DEM.
    ///
    /// Example:
    ///     >>> dem = `ParsedDem.from_string("error(0.01)` D0 D1")
    ///     >>> sampler = `dem.to_dem_sampler()`
    ///     >>> stats = `sampler.sample_statistics(100000`, seed=42)
    fn to_dem_sampler(&self) -> PyDemSampler {
        PyDemSampler {
            inner: self.inner.to_dem_sampler(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ParsedDem(mechanisms={}, detectors={}, dem_outputs={}, observables={}, tracked_paulis={})",
            self.inner.mechanisms.len(),
            self.inner.num_detectors,
            self.inner.num_dem_outputs(),
            self.inner.num_observables(),
            self.inner.num_tracked_paulis()
        )
    }
}

/// Compare two DEMs for exact mechanism match.
///
/// This comparison aggregates mechanisms by effect and compares probabilities.
/// Appropriate for non-decomposed DEMs or when exact match is required.
///
/// Args:
///     dem1: First DEM string or `ParsedDem`.
///     dem2: Second DEM string or `ParsedDem`.
///     `prob_tolerance`: Relative tolerance for probability comparison (default 1e-6).
///
/// Returns:
///     `EquivalenceResult` with comparison statistics.
///
/// Example:
///     >>> result = `compare_dems_exact(dem1_str`, `dem2_str`, `prob_tolerance=0.001`)
///     >>> if result.equivalent:
///     ...     print("DEMs are equivalent")
#[pyfunction]
#[pyo3(signature = (dem1, dem2, prob_tolerance=1e-6))]
fn compare_dems_exact(
    dem1: &str,
    dem2: &str,
    prob_tolerance: f64,
) -> PyResult<PyEquivalenceResult> {
    let parsed1 = dem1
        .parse::<RustParsedDem>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("DEM1 parse error: {e}")))?;
    let parsed2 = dem2
        .parse::<RustParsedDem>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("DEM2 parse error: {e}")))?;

    let inner = rust_compare_dems_exact(&parsed1, &parsed2, prob_tolerance);
    Ok(PyEquivalenceResult { inner })
}

/// Compare two DEMs statistically by sampling.
///
/// This is the most robust comparison method as it accounts for all
/// decomposition strategies and probability combinations. It compares
/// the joint distribution of syndrome patterns, not just marginal rates.
///
/// Args:
///     dem1: First DEM string or `ParsedDem`.
///     dem2: Second DEM string or `ParsedDem`.
///     `num_shots`: Number of shots for sampling (default 100,000).
///     seed: Random seed (default 42).
///     tolerance: Maximum relative difference to consider equivalent (default 0.05).
///
/// Returns:
///     `EquivalenceResult` with comparison statistics.
///
/// Example:
///     >>> result = `compare_dems_statistical(dem1_str`, `dem2_str`, `num_shots=50000`)
///     >>> print(f"Correlation: {result.correlation}")
#[pyfunction]
#[pyo3(signature = (dem1, dem2, num_shots=100_000, seed=42, tolerance=0.05))]
fn compare_dems_statistical(
    dem1: &str,
    dem2: &str,
    num_shots: usize,
    seed: u64,
    tolerance: f64,
) -> PyResult<PyEquivalenceResult> {
    let parsed1 = dem1
        .parse::<RustParsedDem>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("DEM1 parse error: {e}")))?;
    let parsed2 = dem2
        .parse::<RustParsedDem>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("DEM2 parse error: {e}")))?;

    let inner = rust_compare_dems_statistical(&parsed1, &parsed2, num_shots, seed, tolerance);
    Ok(PyEquivalenceResult { inner })
}

/// Convenience function to verify DEM equivalence.
///
/// Args:
///     dem1: First DEM string.
///     dem2: Second DEM string.
///     method: Comparison method - "exact" or "statistical" (default "exact").
///     `prob_tolerance`: For exact: probability tolerance (default 1e-6).
///     `num_shots`: For statistical: number of shots (default 100,000).
///     tolerance: For statistical: rate tolerance (default 0.05).
///     seed: For statistical: random seed (default 42).
///
/// Returns:
///     True if DEMs are equivalent within tolerance.
///
/// Example:
///     >>> if `verify_dem_equivalence(dem1`, dem2, method="exact"):
///     ...     print("DEMs match exactly")
#[pyfunction]
#[pyo3(signature = (dem1, dem2, method="exact", prob_tolerance=1e-6, num_shots=100_000, tolerance=0.05, seed=42))]
fn verify_dem_equivalence(
    dem1: &str,
    dem2: &str,
    method: &str,
    prob_tolerance: f64,
    num_shots: usize,
    tolerance: f64,
    seed: u64,
) -> PyResult<bool> {
    let comparison_method = match method {
        "exact" => RustComparisonMethod::Exact { prob_tolerance },
        "statistical" => RustComparisonMethod::Statistical {
            num_shots,
            seed,
            tolerance,
        },
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "method must be 'exact' or 'statistical'",
            ));
        }
    };

    rust_verify_dem_equivalence(dem1, dem2, &comparison_method)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Assert that two DEMs are equivalent, raising an error if not.
///
/// This is a convenience function for testing that raises `AssertionError`
/// if the DEMs are not equivalent.
///
/// Args:
///     dem1: First DEM string.
///     dem2: Second DEM string.
///     method: Comparison method - "exact" or "statistical" (default "exact").
///     `prob_tolerance`: For exact: probability tolerance (default 1e-6).
///     `num_shots`: For statistical: number of shots (default 100,000).
///     tolerance: For statistical: rate tolerance (default 0.05).
///     seed: For statistical: random seed (default 42).
///
/// Raises:
///     `AssertionError`: If DEMs are not equivalent.
///
/// Example:
///     >>> `assert_dems_equivalent(dem1`, dem2, method="exact")  # Raises if not equivalent
#[pyfunction]
#[pyo3(signature = (dem1, dem2, method="exact", prob_tolerance=1e-6, num_shots=100_000, tolerance=0.05, seed=42))]
fn assert_dems_equivalent(
    dem1: &str,
    dem2: &str,
    method: &str,
    prob_tolerance: f64,
    num_shots: usize,
    tolerance: f64,
    seed: u64,
) -> PyResult<()> {
    let parsed1 = dem1
        .parse::<RustParsedDem>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("DEM1 parse error: {e}")))?;
    let parsed2 = dem2
        .parse::<RustParsedDem>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("DEM2 parse error: {e}")))?;

    let result = match method {
        "exact" => rust_compare_dems_exact(&parsed1, &parsed2, prob_tolerance),
        "statistical" => {
            rust_compare_dems_statistical(&parsed1, &parsed2, num_shots, seed, tolerance)
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "method must be 'exact' or 'statistical'",
            ));
        }
    };

    if result.equivalent {
        Ok(())
    } else {
        let msg = format!(
            "DEMs are not equivalent: max_rate_diff={:.6}, only_in_dem1={:?}, only_in_dem2={:?}",
            result.max_rate_difference, result.details.only_in_dem1, result.details.only_in_dem2
        );
        Err(pyo3::exceptions::PyAssertionError::new_err(msg))
    }
}

// =============================================================================
// CSS UF Decoder (UIUF)
// =============================================================================

/// CSS-aware Union-Find decoder using the UIUF algorithm.
///
/// Takes separate X and Z DEM strings and decodes them jointly, exploiting
/// Y-error identification through cluster intersection.
///
/// `Example::`
///
///     decoder = CssUfDecoder(x_dem_str, z_dem_str)
///     x_obs, z_obs = decoder.decode_css(x_syndrome, z_syndrome)
///
#[pyclass(name = "CssUfDecoder", module = "pecos_rslib.qec")]
pub struct PyCssUfDecoder {
    inner: pecos_decoders::CssUfDecoder,
}

#[pymethods]
impl PyCssUfDecoder {
    /// Create a CSS UF decoder from X and Z DEM strings.
    ///
    /// The qubit-edge mapping is auto-detected from detector coordinates.
    /// If coordinates are missing, falls back to independent X/Z decoding.
    #[new]
    fn new(x_dem: &str, z_dem: &str) -> PyResult<Self> {
        let inner = pecos_decoders::CssUfDecoder::from_dems(
            x_dem,
            z_dem,
            pecos_decoders::UfDecoderConfig::accurate(),
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Decode X and Z syndromes jointly using UIUF.
    ///
    /// Args:
    ///     `x_syndrome`: X-basis detection events (bytes).
    ///     `z_syndrome`: Z-basis detection events (bytes).
    ///
    /// Returns:
    ///     Tuple of (`x_observable_mask`, `z_observable_mask`).
    fn decode_css(&mut self, x_syndrome: &[u8], z_syndrome: &[u8]) -> PyResult<(u64, u64)> {
        self.inner
            .decode_css(x_syndrome, z_syndrome)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Number of matched qubit pairs between X and Z graphs.
    /// 0 means no mapping was found (falls back to independent decode).
    #[getter]
    fn num_qubit_pairs(&self) -> usize {
        self.inner.num_qubit_pairs()
    }

    /// Count erasures the intersection would produce for given syndromes.
    fn count_erasures(&mut self, x_syndrome: &[u8], z_syndrome: &[u8]) -> usize {
        self.inner
            .count_intersection_erasures(x_syndrome, z_syndrome)
    }

    /// Decode a batch of syndromes and return the error count.
    ///
    /// Each shot has concatenated `[x_syndrome | z_syndrome]`.
    /// The `x_syndrome` length is specified by `x_num_detectors`.
    ///
    /// Args:
    ///     syndromes: List of concatenated syndrome byte arrays.
    ///     `true_obs_masks`: True observable masks for each shot.
    ///     `x_num_detectors`: Length of the X syndrome prefix.
    ///
    /// Returns:
    ///     Number of logical errors.
    fn decode_count_batch(
        &mut self,
        syndromes: Vec<Vec<u8>>,
        true_obs_masks: Vec<u64>,
        x_num_detectors: usize,
    ) -> PyResult<usize> {
        let mut errors = 0;
        for (syn, &true_obs) in syndromes.iter().zip(true_obs_masks.iter()) {
            let x_syn = &syn[..x_num_detectors.min(syn.len())];
            let z_syn = &syn[x_num_detectors.min(syn.len())..];
            let (x_obs, z_obs) = self
                .inner
                .decode_css(x_syn, z_syn)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            let predicted = x_obs ^ z_obs;
            if predicted != true_obs {
                errors += 1;
            }
        }
        Ok(errors)
    }
}

// =============================================================================
// Observable Subgraph Decoder (Python class)
// =============================================================================

/// Per-logical-operator subgraph decoder for transversal gates.
///
/// Partitions a DEM into per-logical-operator graphlike subgraphs using
/// stabilizer coordinate information, then decodes each independently.
///
/// Args:
///     dem: DEM string with detector coordinate declarations.
///     `stab_coords`: List of dicts, one per logical qubit. Each dict has
///         keys "X" and "Z" mapping to lists of (x, y) ancilla coordinates.
///     `inner_decoder`: Inner decoder type string (default
///         "`fusion_blossom_serial`", exact MWPM -- accurate and fast across
///         distances, bundled). The best choice is circuit-dependent:
///         `pecos_uf:bp` (PECOS-native belief-propagation + union-find,
///         dependency-free) is competitive on memory and at small distance and is
///         the right pick when you want the pure-native path, but its grow+peel
///         matching is both LESS accurate and SLOWER at higher distance /
///         multi-observable circuits. `belief_matching` matches fusion's accuracy
///         but is slower.
///
/// Example:
///     >>> decoder = `LogicalSubgraphDecoder`(
///     ...     `dem_str`,
///     ...     [{"X": [(1,0), (3,1)], "Z": [(0,3), (1,1)]}],
///     ...     "`fusion_blossom_serial`",
///     ... )
///     >>> obs = decoder.decode(syndrome)
#[pyclass(name = "LogicalSubgraphDecoder", module = "pecos_rslib.qec")]
pub struct PyLogicalSubgraphDecoder {
    inner: pecos_decoder_core::logical_subgraph::LogicalSubgraphDecoder,
    /// The inner per-observable decoder backend selected at construction
    /// (e.g. `"fusion_blossom_serial"`). `decode_count_parallel` reuses this so
    /// the parallel workers match the serial path unless the caller overrides.
    inner_decoder: String,
}

#[pymethods]
impl PyLogicalSubgraphDecoder {
    // Default inner is `fusion_blossom_serial`: exact MWPM on each per-observable
    // subgraph, bundled (no optional dependency). This is now backed by a powered
    // threshold/CI study (`examples/surface/inner_decoder_study.py`; memory +
    // transversal-CX, d=3/5/7, 3 seeds pooled = 150-300k shots/cell, Jeffreys
    // intervals), NOT just policy:
    //   * Accuracy: fusion is statistically tied with pymatching/belief_matching/
    //     tesseract (these per-observable DEMs are graphlike, so exact MWPM is
    //     optimal) and STRICTLY beats `pecos_uf:bp` at every d>=5 cell -- 1.4-2.7x
    //     lower LER with DISJOINT Jeffreys intervals, both families. Tied at d=3.
    //   * Threshold: fusion ~0.9% vs `pecos_uf:bp` ~0.7% (bp also breaks down sooner).
    //   * Speed: at d=7 bp's grow+peel blows up (CX 7.1ms/shot vs fusion 1.2ms);
    //     "bp is the fast native one" is false at depth.
    // Only `pymatching` is faster (~6x) but it is an EXTERNAL dep with zero accuracy
    // or threshold gain, so it is the documented speed option, not the default.
    // SCOPE: the study families are graphlike, so it does not distinguish fusion
    // from hyperedge decoders (tesseract/mwpf) -- re-run with those if non-graphlike
    // per-observable DEMs (biased/correlated noise) ever arise. See
    // pecos-docs/design/inner-decoder-threshold-study.md.
    //
    // `pecos_uf:bp` remains the pure-native, dependency-free path (it does suppress
    // with distance -- the predecoder bug that broke it at d>=5 is fixed -- just at
    // a worse prefactor and lower threshold). See
    // pecos-docs/design/lomatching-paper-additional-learnings.md and
    // logical-subgraph-backprop-region-builder.md.
    #[new]
    #[pyo3(signature = (dem, stab_coords, inner_decoder="fusion_blossom_serial", max_time_radius=None))]
    fn new(
        dem: &str,
        stab_coords: Vec<pyo3::Bound<'_, pyo3::types::PyDict>>,
        inner_decoder: &str,
        max_time_radius: Option<i64>,
    ) -> PyResult<Self> {
        use pecos_decoder_core::logical_subgraph::{LogicalSubgraphDecoder, QubitStabCoords};

        // Parse stab_coords from Python dicts
        let mut rust_stab_coords = Vec::with_capacity(stab_coords.len());
        for dict in &stab_coords {
            let x_list: Vec<(f64, f64)> = dict
                .get_item("X")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'X' key"))?
                .extract()?;
            let z_list: Vec<(f64, f64)> = dict
                .get_item("Z")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'Z' key"))?
                .extract()?;
            rust_stab_coords.push(QubitStabCoords {
                x_positions: x_list,
                z_positions: z_list,
            });
        }

        let inner = LogicalSubgraphDecoder::from_dem_windowed(
            dem,
            &rust_stab_coords,
            max_time_radius,
            |subgraph| {
                let sub_dem = subgraph_to_dem_string(subgraph);
                let decoder = create_observable_decoder(&sub_dem, inner_decoder)
                    .map_err(|e| pecos_decoders::DecoderError::InternalError(e.to_string()))?;
                Ok(Box::new(SendWrapper(decoder))
                    as Box<dyn pecos_decoders::ObservableDecoder + Send + Sync>)
            },
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(Self {
            inner,
            inner_decoder: inner_decoder.to_string(),
        })
    }

    /// Build from a precomputed per-observable detector membership instead of
    /// from `stab_coords`.
    ///
    /// `membership` is a list (one entry per observable) of full-DEM detector
    /// ids. This lets callers supply an alternative observing-region
    /// construction (e.g. the paper's back-propagation / detecting-region set)
    /// and decode with the same machinery for direct comparison.
    #[staticmethod]
    #[pyo3(signature = (dem, membership, inner_decoder="fusion_blossom_serial"))]
    fn from_membership(
        dem: &str,
        membership: Vec<Vec<usize>>,
        inner_decoder: &str,
    ) -> PyResult<Self> {
        use pecos_decoder_core::logical_subgraph::LogicalSubgraphDecoder;

        let inner = LogicalSubgraphDecoder::from_membership(dem, &membership, |subgraph| {
            let sub_dem = subgraph_to_dem_string(subgraph);
            let decoder = create_observable_decoder(&sub_dem, inner_decoder)
                .map_err(|e| pecos_decoders::DecoderError::InternalError(e.to_string()))?;
            Ok(Box::new(SendWrapper(decoder))
                as Box<dyn pecos_decoders::ObservableDecoder + Send + Sync>)
        })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(Self {
            inner,
            inner_decoder: inner_decoder.to_string(),
        })
    }

    /// Decode a syndrome and return observable flip predictions.
    ///
    /// Returns a Python ``int`` (bit ``i`` = observable ``i``). The integer is
    /// arbitrary precision, so decoders with more than 64 observables are
    /// returned without truncation.
    fn decode(&mut self, py: Python<'_>, syndrome: Vec<u8>) -> PyResult<Py<pyo3::PyAny>> {
        use pecos_decoder_core::ObservableDecoder;
        let mask = self
            .inner
            .decode_obs(&syndrome)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        obsmask_to_py(py, &mask)
    }

    /// Number of observables this decoder handles.
    fn num_observables(&self) -> usize {
        self.inner.num_observables()
    }

    /// The inner per-observable decoder backend selected at construction.
    ///
    /// `decode_count_parallel` reuses this unless the caller overrides it, so
    /// the serial and parallel paths agree by default.
    #[getter]
    fn inner_decoder(&self) -> &str {
        &self.inner_decoder
    }

    /// Decode a batch of syndromes and return observable predictions.
    ///
    /// Args:
    ///     syndromes: 2D numpy array of shape (`num_shots`, `num_detectors`).
    ///
    /// Returns:
    ///     List of observable flip masks (one Python ``int`` per shot; arbitrary
    ///     precision, so more than 64 observables are not truncated).
    fn decode_batch(
        &mut self,
        py: Python<'_>,
        syndromes: Vec<Vec<u8>>,
    ) -> PyResult<Vec<Py<pyo3::PyAny>>> {
        use pecos_decoder_core::ObservableDecoder;
        let mut results = Vec::with_capacity(syndromes.len());
        for syn in &syndromes {
            let mask = self
                .inner
                .decode_obs(syn)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            results.push(obsmask_to_py(py, &mask)?);
        }
        Ok(results)
    }

    /// Decode a `SampleBatch` and return the number of logical errors.
    ///
    /// This runs entirely in Rust — no Python per-shot overhead.
    ///
    /// Args:
    ///     batch: A `SampleBatch` from `DemSampler.sample_batch()`.
    ///
    /// Returns:
    ///     Number of logical errors.
    fn decode_count(&mut self, batch: &PySampleBatch) -> PyResult<usize> {
        batch.ensure_detector_events()?;
        let detection_events: Vec<Vec<u8>> = (0..batch.num_shots)
            .map(|i| {
                let mut s = vec![0u8; batch.num_detectors];
                batch.extract_syndrome(i, &mut s);
                s
            })
            .collect();
        let observable_masks: Vec<pecos_decoder_core::obs_mask::ObsMask> = (0..batch.num_shots)
            .map(|i| batch.extract_obs_mask_wide(i))
            .collect();
        self.inner
            .decode_count_batched(&detection_events, &observable_masks)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Decode a `SampleBatch` in parallel using rayon.
    ///
    /// Creates per-worker decoder instances to avoid lock contention.
    /// Requires the DEM string for reconstruction. `inner_decoder` defaults to
    /// the backend selected at construction (so the parallel path matches the
    /// serial `decode_count` path); pass an explicit value only to override it.
    #[pyo3(signature = (batch, dem, stab_coords, inner_decoder=None, num_workers=None, max_time_radius=None))]
    fn decode_count_parallel(
        &self,
        batch: &PySampleBatch,
        dem: &str,
        stab_coords: Vec<pyo3::Bound<'_, pyo3::types::PyDict>>,
        inner_decoder: Option<&str>,
        num_workers: Option<usize>,
        max_time_radius: Option<i64>,
    ) -> PyResult<usize> {
        use pecos_decoder_core::logical_subgraph::{LogicalSubgraphDecoder, QubitStabCoords};
        use rayon::prelude::*;

        batch.ensure_detector_events()?;
        // Parse stab_coords
        let mut sc = Vec::with_capacity(stab_coords.len());
        for dict in &stab_coords {
            let x: Vec<(f64, f64)> = dict
                .get_item("X")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("X"))?
                .extract()?;
            let z: Vec<(f64, f64)> = dict
                .get_item("Z")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Z"))?
                .extract()?;
            sc.push(QubitStabCoords {
                x_positions: x,
                z_positions: z,
            });
        }

        let dem_str = dem.to_string();
        // Reuse the backend chosen at construction unless the caller overrides,
        // so parallel workers decode identically to the serial path.
        let inner_str = inner_decoder
            .unwrap_or(self.inner_decoder.as_str())
            .to_string();
        let n = batch.num_shots;

        // Materialize row-major data for parallel decode.
        let events: Vec<Vec<u8>> = (0..n)
            .map(|i| {
                let mut s = vec![0u8; batch.num_detectors];
                batch.extract_syndrome(i, &mut s);
                s
            })
            .collect();
        let masks: Vec<pecos_decoder_core::obs_mask::ObsMask> =
            (0..n).map(|i| batch.extract_obs_mask_wide(i)).collect();

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_workers.unwrap_or(0))
            .build()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Propagate worker construction and decode errors instead of panicking
        // across the FFI boundary or silently scoring a failed chunk as
        // all-failures (which would inflate the reported logical error rate).
        let errors: Result<usize, pecos_decoders::DecoderError> = pool.install(|| {
            // Split into chunks, each chunk gets its own decoder + batch decode
            let chunk_size = n.div_ceil(rayon::current_num_threads());
            (0..n)
                .collect::<Vec<_>>()
                .par_chunks(chunk_size.max(1))
                .map(|chunk| {
                    // Build a fresh decoder for this worker
                    let mut dec = LogicalSubgraphDecoder::from_dem_windowed(
                        &dem_str,
                        &sc,
                        max_time_radius,
                        |subgraph| {
                            let sub_dem = subgraph_to_dem_string(subgraph);
                            let d =
                                create_observable_decoder(&sub_dem, &inner_str).map_err(|e| {
                                    pecos_decoders::DecoderError::InternalError(e.to_string())
                                })?;
                            Ok(Box::new(SendWrapper(d))
                                as Box<dyn pecos_decoders::ObservableDecoder + Send + Sync>)
                        },
                    )?;

                    // Collect chunk syndromes and masks for batch decode
                    let chunk_syns: Vec<Vec<u8>> =
                        chunk.iter().map(|&i| events[i].clone()).collect();
                    let chunk_masks: Vec<pecos_decoder_core::obs_mask::ObsMask> =
                        chunk.iter().map(|&i| masks[i].clone()).collect();
                    dec.decode_count_batched(&chunk_syns, &chunk_masks)
                })
                .try_reduce(|| 0, |a, b| Ok(a + b))
        });

        errors.map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Number of detectors in each subgraph.
    fn subgraph_sizes(&self) -> Vec<usize> {
        (0..self.inner.num_observables())
            .map(|i| self.inner.subgraph(i).map_or(0, |sg| sg.detector_map.len()))
            .collect()
    }

    /// Per-observable observing regions: a list (one entry per observable) of
    /// sorted full-DEM detector ids in that observable's subgraph.
    ///
    /// Exposed for differential testing against reference implementations such
    /// as `lomatching.get_detector_indices_for_subgraphs`.
    fn observing_regions(&self) -> Vec<Vec<usize>> {
        self.inner.observing_regions()
    }

    /// Diagnostics: (`num_edges`, `skipped_hyperedges`) for each subgraph.
    fn subgraph_diagnostics(&self) -> Vec<(usize, usize)> {
        (0..self.inner.num_observables())
            .map(|i| {
                self.inner.subgraph(i).map_or((0, 0), |sg| {
                    (sg.graph.edges.len(), sg.graph.skipped_hyperedges)
                })
            })
            .collect()
    }

    /// Count ghost edges (3-detector cross-qubit hyperedges) in the DEM.
    ///
    /// These are the hyperedges that the ghost protocol decomposes for
    /// modular per-qubit decoding. Returns (`total_ghost_edges`, `num_qubits`).
    #[staticmethod]
    fn count_ghost_edges(
        dem: &str,
        stab_coords: Vec<pyo3::Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<(usize, usize)> {
        use pecos_decoder_core::ghost_protocol::extract_ghost_edges_from_dem;
        use pecos_decoder_core::logical_subgraph::QubitStabCoords;

        let mut sc = Vec::with_capacity(stab_coords.len());
        for dict in &stab_coords {
            let x: Vec<(f64, f64)> = dict
                .get_item("X")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("X"))?
                .extract()?;
            let z: Vec<(f64, f64)> = dict
                .get_item("Z")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Z"))?
                .extract()?;
            sc.push(QubitStabCoords {
                x_positions: x,
                z_positions: z,
            });
        }

        let edges = extract_ghost_edges_from_dem(dem, &sc);
        let num_qubits = sc.len();
        Ok((edges.len(), num_qubits))
    }

    /// Get the per-subgraph DEM strings (graphlike, local detector IDs 0..N).
    ///
    /// NOTE: these strings carry NO `detector(...)` coordinate lines (subgraph
    /// graphs drop coordinates), so they are NOT suitable for *time-windowed*
    /// decoding -- a windowed decoder would see no detector times and collapse to
    /// a single window. For windowing, use the coord-preserving
    /// `LogicalSubgraphWindowPlan` path (the `WindowedLogicalSubgraphDecoder` /
    /// logical-circuit windowed budget already do). These strings are fine for
    /// full (non-windowed) per-subgraph decoding.
    fn subgraph_dems(&self) -> Vec<String> {
        (0..self.inner.num_observables())
            .map(|i| {
                self.inner
                    .subgraph(i)
                    .map_or(String::new(), |sg| subgraph_to_dem_string(&sg.graph))
            })
            .collect()
    }

    /// Get the detector map for each subgraph (local → global index mapping).
    fn subgraph_detector_maps(&self) -> Vec<Vec<usize>> {
        (0..self.inner.num_observables())
            .map(|i| {
                self.inner
                    .subgraph(i)
                    .map_or(Vec::new(), |sg| sg.detector_map.clone())
            })
            .collect()
    }
}

// =============================================================================
// Windowed logical-subgraph decoding Decoder (Python class)
// =============================================================================

/// Windowed observable subgraph decoder for deep circuits.
///
/// Splits the DEM into time windows, runs logical-subgraph decoder within each window.
/// Prevents the observing region from spanning the full circuit.
///
/// Partitions the DEM per observable, then windows each subgraph with proper
/// sliding-window core-commit (only correction edges whose both endpoints lie
/// in a window's core are committed). The inner decoder is the native
/// edge-tracking union-find decoder, which core-commit requires.
///
/// Args:
///     dem: DEM string.
///     `stab_coords`: Stabilizer coordinates per logical qubit.
///     step: Core window size in time steps.
///     buffer: Buffer size on each side for matching context (0 =
///         non-overlapping; recommend ~code distance).
#[pyclass(name = "WindowedLogicalSubgraphDecoder", module = "pecos_rslib.qec")]
pub struct PyWindowedLogicalSubgraphDecoder {
    inner: pecos_decoders::WindowedLogicalSubgraphDecoder,
}

#[pymethods]
impl PyWindowedLogicalSubgraphDecoder {
    #[new]
    #[pyo3(signature = (dem, stab_coords, step=8, buffer=4))]
    fn new(
        dem: &str,
        stab_coords: Vec<pyo3::Bound<'_, pyo3::types::PyDict>>,
        step: usize,
        buffer: usize,
    ) -> PyResult<Self> {
        use pecos_decoder_core::logical_subgraph::QubitStabCoords;

        let mut sc = Vec::with_capacity(stab_coords.len());
        for dict in &stab_coords {
            let x: Vec<(f64, f64)> = dict
                .get_item("X")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("X"))?
                .extract()?;
            let z: Vec<(f64, f64)> = dict
                .get_item("Z")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Z"))?
                .extract()?;
            sc.push(QubitStabCoords {
                x_positions: x,
                z_positions: z,
            });
        }

        let config = pecos_decoders::WindowedConfig {
            step_size: step,
            buffer_size: buffer,
            ..Default::default()
        };

        let inner =
            pecos_decoders::WindowedLogicalSubgraphDecoder::from_dem(dem, &sc, None, config)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(Self { inner })
    }

    fn decode(&mut self, py: Python<'_>, syndrome: Vec<u8>) -> PyResult<Py<pyo3::PyAny>> {
        use pecos_decoder_core::ObservableDecoder;
        let mask = self
            .inner
            .decode_obs(&syndrome)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        obsmask_to_py(py, &mask)
    }

    fn decode_count(&mut self, batch: &PySampleBatch) -> PyResult<usize> {
        use pecos_decoder_core::ObservableDecoder;
        batch.ensure_detector_events()?;
        let mut errors = 0usize;
        let mut syndrome = vec![0u8; batch.num_detectors];
        for i in 0..batch.num_shots {
            batch.extract_syndrome(i, &mut syndrome);
            let predicted = self
                .inner
                .decode_obs(&syndrome)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            if predicted != batch.extract_obs_mask_wide(i) {
                errors += 1;
            }
        }
        Ok(errors)
    }

    fn num_windows(&self) -> usize {
        self.inner.num_windows()
    }
}

// =============================================================================
// Logical Algorithm Decoder (Python class)
// =============================================================================

/// Read a required `u32` bit field off a boundary-gate descriptor dict, returning
/// a clear `PyErr` (not a panic) when a malformed descriptor omits the field.
/// Shared by the two algorithm-decoder bindings below.
fn req_bit(
    dict: &pyo3::Bound<'_, pyo3::types::PyDict>,
    key: &str,
    gate_type: &str,
) -> PyResult<u32> {
    let bit: u32 = dict
        .get_item(key)?
        .ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "boundary gate '{gate_type}' missing required field '{key}'"
            ))
        })?
        .extract()?;
    // Every boundary-gate bit indexes a u64 observable frame (`1u64 << bit`), so
    // it must be < 64 -- reject out-of-range here rather than shift-overflow later.
    if bit >= 64 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "boundary gate '{gate_type}' field '{key}' = {bit} exceeds the 64-observable frame limit"
        )));
    }
    Ok(bit)
}

/// Decoder for logical quantum algorithms with per-segment logical-subgraph decoder and
/// Pauli frame propagation at transversal gate boundaries.
///
/// Built from a descriptor dict produced by
/// ``LogicalCircuitBuilder.build_algorithm_descriptor()``.
///
/// Supports both batch mode (``decode``, ``decode_count``) and
/// streaming mode (``feed_sparse``, ``flush``, ``reset``).
#[pyclass(name = "LogicalAlgorithmDecoder", module = "pecos_rslib.qec")]
pub struct PyLogicalAlgorithmDecoder {
    inner: pecos_decoder_core::logical_algorithm::StreamingLogicalDecoder,
}

#[pymethods]
impl PyLogicalAlgorithmDecoder {
    /// Build from a descriptor dict and inner decoder type.
    ///
    /// Args:
    ///     descriptor: Dict from ``LogicalCircuitBuilder.build_algorithm_descriptor()``.
    ///     `inner_decoder`: Decoder type string for each segment's logical-subgraph decoder inner decoder.
    #[new]
    #[pyo3(signature = (descriptor, inner_decoder="pymatching"))]
    fn new(
        descriptor: &pyo3::Bound<'_, pyo3::types::PyDict>,
        inner_decoder: &str,
    ) -> PyResult<Self> {
        use pecos_decoder_core::logical_algorithm::{
            AlgorithmDescriptor, BoundaryGate, LogicalAlgorithmDecoder, SegmentDescriptor,
        };
        use pecos_decoder_core::logical_subgraph::{LogicalSubgraphDecoder, QubitStabCoords};

        // Parse full DEM and stab_coords for full-circuit logical-subgraph decoder
        let full_dem: String = descriptor
            .get_item("full_dem")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("full_dem"))?
            .extract()?;

        // Use first segment's stab_coords as the base (they have the
        // original X/Z assignment; the full-circuit DEM uses original coords).
        let seg_list: Vec<pyo3::Bound<'_, pyo3::types::PyDict>> = descriptor
            .get_item("segments")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("segments"))?
            .extract()?;

        let num_obs: usize = descriptor
            .get_item("num_observables")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("num_observables"))?
            .extract()?;

        // Parse stab_coords from the first segment (original orientation)
        let first_seg = seg_list.first().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("algorithm descriptor has no segments")
        })?;
        let sc_list: Vec<pyo3::Bound<'_, pyo3::types::PyDict>> = first_seg
            .get_item("stab_coords")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("stab_coords"))?
            .extract()?;
        let mut rust_sc = Vec::with_capacity(sc_list.len());
        for sc_dict in &sc_list {
            let x: Vec<(f64, f64)> = sc_dict
                .get_item("X")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("X"))?
                .extract()?;
            let z: Vec<(f64, f64)> = sc_dict
                .get_item("Z")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Z"))?
                .extract()?;
            rust_sc.push(QubitStabCoords {
                x_positions: x,
                z_positions: z,
            });
        }

        let inner_str = inner_decoder.to_string();

        // Build full-circuit logical-subgraph decoder from the full DEM
        let full_osd = LogicalSubgraphDecoder::from_dem(&full_dem, &rust_sc, |subgraph| {
            let sub_dem = subgraph_to_dem_string(subgraph);
            let d = create_observable_decoder(&sub_dem, &inner_str)
                .map_err(|e| pecos_decoders::DecoderError::InternalError(e.to_string()))?;
            Ok(Box::new(SendWrapper(d))
                as Box<dyn pecos_decoders::ObservableDecoder + Send + Sync>)
        })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Parse segment descriptors (for metadata)
        let mut seg_descs = Vec::with_capacity(seg_list.len());
        for seg_dict in &seg_list {
            let n_det: usize = seg_dict
                .get_item("num_detectors")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("num_detectors"))?
                .extract()?;
            seg_descs.push(SegmentDescriptor {
                num_detectors: n_det,
                num_observables: num_obs,
            });
        }

        // Parse boundary gates
        let bg_list: Vec<Vec<pyo3::Bound<'_, pyo3::types::PyDict>>> = descriptor
            .get_item("boundary_gates")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("boundary_gates"))?
            .extract()?;

        let mut boundary_gates = Vec::with_capacity(bg_list.len());
        for gates in &bg_list {
            let mut bg_vec = Vec::new();
            for gate_dict in gates {
                let gate_type: String = gate_dict
                    .get_item("type")?
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("type"))?
                    .extract()?;
                match gate_type.as_str() {
                    "Hadamard" => {
                        bg_vec.push(BoundaryGate::Hadamard {
                            x_obs_bit: req_bit(gate_dict, "x_obs_bit", &gate_type)?,
                            z_obs_bit: req_bit(gate_dict, "z_obs_bit", &gate_type)?,
                        });
                    }
                    "Cnot" => {
                        bg_vec.push(BoundaryGate::Cnot {
                            ctrl_x_bit: req_bit(gate_dict, "ctrl_x_bit", &gate_type)?,
                            ctrl_z_bit: req_bit(gate_dict, "ctrl_z_bit", &gate_type)?,
                            tgt_x_bit: req_bit(gate_dict, "tgt_x_bit", &gate_type)?,
                            tgt_z_bit: req_bit(gate_dict, "tgt_z_bit", &gate_type)?,
                        });
                    }
                    "SGate" => {
                        bg_vec.push(BoundaryGate::SGate {
                            x_obs_bit: req_bit(gate_dict, "x_obs_bit", &gate_type)?,
                            z_obs_bit: req_bit(gate_dict, "z_obs_bit", &gate_type)?,
                        });
                    }
                    "TGateInjection" => {
                        let z = req_bit(gate_dict, "z_obs_bit", &gate_type)?;
                        let a = req_bit(gate_dict, "ancilla_z_bit", &gate_type)?;
                        bg_vec.push(BoundaryGate::TGateInjection {
                            z_obs_bit: z,
                            ancilla_z_bit: a,
                        });
                    }
                    _ => {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Unknown gate type: {gate_type}"
                        )));
                    }
                }
            }
            boundary_gates.push(bg_vec);
        }

        let algo_desc = AlgorithmDescriptor {
            segments: seg_descs,
            boundary_gates,
            num_observables: num_obs,
        };

        let algo_dec = LogicalAlgorithmDecoder::new(Box::new(full_osd), algo_desc);
        let inner = pecos_decoder_core::logical_algorithm::StreamingLogicalDecoder::new(algo_dec);
        Ok(Self { inner })
    }

    // -- Batch mode --

    /// Decode a single syndrome and return the observable flip mask as a Python
    /// ``int`` (arbitrary precision; more than 64 observables are not truncated).
    fn decode(&mut self, py: Python<'_>, syndrome: Vec<u8>) -> PyResult<Py<pyo3::PyAny>> {
        self.inner.reset();
        let mask = self
            .inner
            .decode_shot_obs(&syndrome)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        obsmask_to_py(py, &mask)
    }

    /// Decode a batch of samples and count logical errors (wide observable masks).
    fn decode_count(&mut self, batch: &PySampleBatch) -> PyResult<usize> {
        batch.ensure_detector_events()?;
        let mut errors = 0usize;
        let mut syndrome = vec![0u8; batch.num_detectors];
        for i in 0..batch.num_shots {
            batch.extract_syndrome(i, &mut syndrome);
            self.inner.reset();
            let predicted = self
                .inner
                .decode_shot_obs(&syndrome)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            if predicted != batch.extract_obs_mask_wide(i) {
                errors += 1;
            }
        }
        Ok(errors)
    }

    // -- Streaming mode --

    /// Feed sparse detection events: list of (`detector_index`, value) pairs.
    fn feed_sparse(&mut self, detectors: Vec<(u32, u8)>) {
        self.inner.feed_sparse(&detectors);
    }

    /// Feed a dense syndrome (all detectors in order).
    fn feed_dense(&mut self, syndrome: Vec<u8>) {
        self.inner.feed_dense(&syndrome);
    }

    /// Decode the accumulated syndrome. Call at segment boundaries or end.
    fn flush(&mut self) -> PyResult<u64> {
        self.inner
            .flush()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Reset syndrome buffer for the next shot.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Current accumulated observable correction, narrowed to `u64`.
    ///
    /// Raises if the accumulated mask exceeds 64 observables; use
    /// `accumulated_obs_mask` for the wide value.
    fn accumulated_obs(&self) -> PyResult<u64> {
        self.inner
            .accumulated_obs()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Current accumulated observable correction as an arbitrary-precision int.
    fn accumulated_obs_mask(&self, py: Python<'_>) -> PyResult<Py<pyo3::PyAny>> {
        obsmask_to_py(py, self.inner.accumulated_obs_mask())
    }

    // -- Metadata --

    /// Number of segments.
    fn num_segments(&self) -> usize {
        self.inner.num_segments()
    }

    /// Rounds fed so far.
    fn rounds_fed(&self) -> usize {
        self.inner.rounds_fed()
    }
}

// =============================================================================
// Logical Circuit Decoder with Budget (Python class)
// =============================================================================

/// Budget-aware decoder for logical quantum circuits.
///
/// Selects decode strategy based on available reaction time:
/// - ``"unlimited"``: full-circuit logical-subgraph decoder (Clifford circuits, offline)
/// - ``"windowed"``: default windowed logical-subgraph decoder (~1ms reaction time)
/// - ``"10ms"``, ``"1000us"``, etc.: explicit reaction time budget
///
/// The reaction time is the time available at feed-forward decision
/// points (T gates, magic state injection). For Clifford-only circuits,
/// use ``"unlimited"`` since there are no mid-circuit decisions.
///
/// `Example::`
///
///     desc = builder.build_algorithm_descriptor(p1=0.001, p2=0.001)
///     decoder = LogicalCircuitDecoder(desc, budget="unlimited")
///     errors = decoder.decode_count(batch)
#[pyclass(name = "LogicalCircuitDecoder", module = "pecos_rslib.qec")]
pub struct PyLogicalCircuitDecoder {
    inner: pecos_decoder_core::logical_algorithm::LogicalCircuitDecoder,
    /// How the decode actually windows: "unlimited" (full circuit),
    /// "full_fallback" (per-observable full decode behind a windowed budget),
    /// or "real_windowed" (genuine sliding-window; not yet enabled).
    effective_windowing: String,
    /// Window count actually used, one entry per non-empty subgraph (1 == full
    /// decode); not indexed by global observable id.
    actual_num_windows: Vec<usize>,
    /// Whether genuine time-windowing is *possible* for this circuit (deep
    /// enough), independent of whether it is enabled. False for "unlimited".
    can_window: bool,
}

#[pymethods]
impl PyLogicalCircuitDecoder {
    #[new]
    #[pyo3(signature = (descriptor, budget="unlimited", inner_decoder="pymatching", strict=false))]
    fn new(
        descriptor: &pyo3::Bound<'_, pyo3::types::PyDict>,
        budget: &str,
        inner_decoder: &str,
        strict: bool,
    ) -> PyResult<Self> {
        use pecos_decoder_core::decode_budget::DecodeBudget;
        use pecos_decoder_core::logical_algorithm::{
            AlgorithmDescriptor, BoundaryGate, FullCircuitStrategy, LogicalCircuitDecoder,
            SegmentDescriptor,
        };
        use pecos_decoder_core::logical_subgraph::{LogicalSubgraphDecoder, QubitStabCoords};

        // Parse full DEM
        let full_dem: String = descriptor
            .get_item("full_dem")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("full_dem"))?
            .extract()?;

        let seg_list: Vec<pyo3::Bound<'_, pyo3::types::PyDict>> = descriptor
            .get_item("segments")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("segments"))?
            .extract()?;

        let num_obs: usize = descriptor
            .get_item("num_observables")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("num_observables"))?
            .extract()?;

        // Parse stab_coords from first segment
        let first_seg = seg_list.first().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("algorithm descriptor has no segments")
        })?;
        let sc_list: Vec<pyo3::Bound<'_, pyo3::types::PyDict>> = first_seg
            .get_item("stab_coords")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("stab_coords"))?
            .extract()?;
        let mut rust_sc = Vec::with_capacity(sc_list.len());
        for sc_dict in &sc_list {
            let x: Vec<(f64, f64)> = sc_dict
                .get_item("X")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("X"))?
                .extract()?;
            let z: Vec<(f64, f64)> = sc_dict
                .get_item("Z")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Z"))?
                .extract()?;
            rust_sc.push(QubitStabCoords {
                x_positions: x,
                z_positions: z,
            });
        }
        let num_qubits = rust_sc.len();

        let inner_str = inner_decoder.to_string();
        let full_osd = LogicalSubgraphDecoder::from_dem(&full_dem, &rust_sc, |subgraph| {
            let sub_dem = subgraph_to_dem_string(subgraph);
            let d = create_observable_decoder(&sub_dem, &inner_str)
                .map_err(|e| pecos_decoders::DecoderError::InternalError(e.to_string()))?;
            Ok(Box::new(SendWrapper(d))
                as Box<dyn pecos_decoders::ObservableDecoder + Send + Sync>)
        })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Parse segments
        let mut seg_descs = Vec::with_capacity(seg_list.len());
        for seg_dict in &seg_list {
            let n_det: usize = seg_dict
                .get_item("num_detectors")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("num_detectors"))?
                .extract()?;
            seg_descs.push(SegmentDescriptor {
                num_detectors: n_det,
                num_observables: num_obs,
            });
        }

        // Parse boundary gates
        let bg_list: Vec<Vec<pyo3::Bound<'_, pyo3::types::PyDict>>> = descriptor
            .get_item("boundary_gates")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("boundary_gates"))?
            .extract()?;

        let mut boundary_gates = Vec::with_capacity(bg_list.len());
        for gates in &bg_list {
            let mut bg_vec = Vec::new();
            for gate_dict in gates {
                let gate_type: String = gate_dict
                    .get_item("type")?
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("type"))?
                    .extract()?;
                match gate_type.as_str() {
                    "Hadamard" => {
                        bg_vec.push(BoundaryGate::Hadamard {
                            x_obs_bit: req_bit(gate_dict, "x_obs_bit", &gate_type)?,
                            z_obs_bit: req_bit(gate_dict, "z_obs_bit", &gate_type)?,
                        });
                    }
                    "Cnot" => {
                        bg_vec.push(BoundaryGate::Cnot {
                            ctrl_x_bit: req_bit(gate_dict, "ctrl_x_bit", &gate_type)?,
                            ctrl_z_bit: req_bit(gate_dict, "ctrl_z_bit", &gate_type)?,
                            tgt_x_bit: req_bit(gate_dict, "tgt_x_bit", &gate_type)?,
                            tgt_z_bit: req_bit(gate_dict, "tgt_z_bit", &gate_type)?,
                        });
                    }
                    "SGate" => {
                        bg_vec.push(BoundaryGate::SGate {
                            x_obs_bit: req_bit(gate_dict, "x_obs_bit", &gate_type)?,
                            z_obs_bit: req_bit(gate_dict, "z_obs_bit", &gate_type)?,
                        });
                    }
                    "TGateInjection" => {
                        let z = req_bit(gate_dict, "z_obs_bit", &gate_type)?;
                        let a = req_bit(gate_dict, "ancilla_z_bit", &gate_type)?;
                        bg_vec.push(BoundaryGate::TGateInjection {
                            z_obs_bit: z,
                            ancilla_z_bit: a,
                        });
                    }
                    _ => {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Unknown gate type: {gate_type}"
                        )));
                    }
                }
            }
            boundary_gates.push(bg_vec);
        }

        let algo_desc = AlgorithmDescriptor {
            segments: seg_descs,
            boundary_gates,
            num_observables: num_obs,
        };

        // Select budget: "unlimited" for full-circuit, "windowed" for
        // bounded-latency, or a cycle time in microseconds like "1000us".
        //
        // Use the REAL physical code distance from the descriptor (used for the
        // windowing step / latency bound). `num_qubits = rust_sc.len()` is the
        // number of logical patches, NOT a distance -- deriving distance from it
        // (e.g. sqrt) is wrong (a single d=7 patch would yield distance 1 and
        // make `can_window`/`strict` dishonest). Fall back to the old patch-count
        // heuristic only for legacy descriptors that predate the `distance` field.
        let distance: usize = descriptor
            .get_item("distance")?
            .and_then(|v| v.extract::<usize>().ok())
            .filter(|&d| d > 0)
            .unwrap_or_else(|| {
                let mut d = 0usize;
                while d.saturating_mul(d) < num_qubits {
                    d += 1;
                }
                d.max(1)
            });
        let decode_budget = match budget {
            "unlimited" | "offline" => DecodeBudget::unlimited(),
            "windowed" => {
                DecodeBudget::from_reaction_time(std::time::Duration::from_millis(1), distance)
            }
            s if s.ends_with("us") => {
                let us: u64 = s[..s.len() - 2].parse().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Invalid cycle time: {s}"
                    ))
                })?;
                DecodeBudget::from_reaction_time(std::time::Duration::from_micros(us), distance)
            }
            s if s.ends_with("ms") => {
                let ms: u64 = s[..s.len() - 2].parse().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Invalid cycle time: {s}"
                    ))
                })?;
                DecodeBudget::from_reaction_time(std::time::Duration::from_millis(ms), distance)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown budget: {budget}. Use: unlimited, windowed, or a cycle time like 1000us, 10ms"
                )));
            }
        };

        // Select strategy based on budget.
        let mut effective_windowing = String::from("unlimited");
        let mut actual_num_windows: Vec<usize> = Vec::new();
        let mut can_window = false;
        let strategy: Box<dyn pecos_decoder_core::decode_budget::DecodeStrategy + Send + Sync> =
            if decode_budget.is_unlimited() {
                // Unlimited: full-circuit logical-subgraph decoder (maximum accuracy)
                Box::new(FullCircuitStrategy::new(Box::new(full_osd)))
            } else {
                // A bounded-latency ("windowed") budget was requested. Genuine
                // per-observable sliding-window LOM decoding does not yet
                // suppress (the windowed-LOM time-like-snake limitation; needs
                // the anti-snake machinery), so we do an EXPLICIT full-decode
                // fallback per observable -- accurate, but NOT bounded latency --
                // and surface that honestly via `effective_windowing()` /
                // `actual_num_windows()`. No silent fallback. `strict=True` turns
                // the unmet latency budget into a hard error.
                use pecos_decoder_core::logical_algorithm::WindowedLogicalSubgraphStrategy;
                use pecos_decoder_core::logical_subgraph::window_plan::EffectiveWindowing;

                // Coord-preserving window plan (reports whether real windowing is
                // even possible for this circuit depth).
                let full_coords = pecos_decoder_core::DemMatchingGraph::from_dem_str(&full_dem)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
                    .detector_coords;
                let plan = full_osd.window_plan(&full_coords);
                let step = decode_budget.code_distance.max(1);
                can_window = plan.effective_windowing(step) == EffectiveWindowing::RealWindowed;

                // `strict` rejects only when genuine windowing was POSSIBLE (the
                // circuit is deep enough) but is being skipped. When `!can_window`
                // the circuit is a single window anyway, so a full decode IS the
                // bounded-latency answer -- no degradation to reject.
                if strict && can_window {
                    return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        "bounded-latency ('windowed') budget requested with strict=True, \
                         but accurate windowed logical-subgraph decoding is not yet \
                         available (windowed-LOM anti-snake machinery pending). This \
                         circuit is deep enough to time-window (can_window=True), so a \
                         full per-observable decode would forgo the requested latency \
                         bound. Use budget='unlimited', or pass strict=False to accept \
                         the full-decode fallback."
                            .to_string(),
                    ));
                }

                let sub_dems = plan.sub_dems();
                let det_maps = plan.detector_maps();
                let obs_indices: Vec<usize> =
                    plan.entries().iter().map(|e| e.observable_idx).collect();
                // The fallback runs a full (non-windowed) inner per observable, so
                // the actual window count is 1 each by construction. (The Layer C
                // real-windowed path must instead derive these from the windowed
                // inners.) The label is single-sourced from the plan's enum.
                effective_windowing = EffectiveWindowing::FullFallback.as_str().to_string();
                actual_num_windows = vec![1usize; sub_dems.len()];

                let fallback_inner = inner_decoder.to_string();
                let wosd = WindowedLogicalSubgraphStrategy::new(
                    sub_dems,
                    det_maps,
                    obs_indices,
                    |dem_str| {
                        let dec =
                            create_observable_decoder(dem_str, &fallback_inner).map_err(|e| {
                                pecos_decoders::DecoderError::InternalError(e.to_string())
                            })?;
                        Ok(Box::new(SendWrapper(dec))
                            as Box<dyn pecos_decoders::ObservableDecoder + Send + Sync>)
                    },
                )
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

                Box::new(wosd)
            };

        let inner = LogicalCircuitDecoder::new(algo_desc, strategy, decode_budget, num_qubits);
        Ok(Self {
            inner,
            effective_windowing,
            actual_num_windows,
            can_window,
        })
    }

    /// How the decode actually windows: ``"unlimited"`` (full-circuit decode),
    /// ``"full_fallback"`` (per-observable full decode behind a windowed
    /// budget -- accurate but NOT bounded latency), or ``"real_windowed"``
    /// (genuine sliding-window; not yet enabled pending the windowed-LOM
    /// anti-snake machinery). Lets callers/tests assert the effective mode
    /// instead of trusting a silent fallback.
    #[getter]
    fn effective_windowing(&self) -> &str {
        &self.effective_windowing
    }

    /// Window count actually used, one entry per *non-empty* subgraph in
    /// surviving-subgraph order (empty-region observables are dropped, so this
    /// is not indexed by global observable id). ``1`` == full decode. All ``1``
    /// in the current full-fallback path; empty for the unlimited budget.
    #[getter]
    fn actual_num_windows(&self) -> Vec<usize> {
        self.actual_num_windows.clone()
    }

    /// Whether genuine time-windowing is *possible* for this circuit (deep
    /// enough), independent of whether it is enabled. ``False`` for unlimited.
    #[getter]
    fn can_window(&self) -> bool {
        self.can_window
    }

    /// Decode a single syndrome. Returns a Python ``int`` (arbitrary precision;
    /// more than 64 observables are not truncated).
    fn decode(&mut self, py: Python<'_>, syndrome: Vec<u8>) -> PyResult<Py<pyo3::PyAny>> {
        use pecos_decoder_core::ObservableDecoder;
        let mask = self
            .inner
            .decode_obs(&syndrome)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        obsmask_to_py(py, &mask)
    }

    /// Decode a batch and count errors (wide observable masks).
    fn decode_count(&mut self, batch: &PySampleBatch) -> PyResult<usize> {
        use pecos_decoder_core::ObservableDecoder;
        batch.ensure_detector_events()?;
        let mut errors = 0usize;
        let mut syndrome = vec![0u8; batch.num_detectors];
        for i in 0..batch.num_shots {
            batch.extract_syndrome(i, &mut syndrome);
            let predicted = self
                .inner
                .decode_obs(&syndrome)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            if predicted != batch.extract_obs_mask_wide(i) {
                errors += 1;
            }
        }
        Ok(errors)
    }

    /// Number of segments.
    fn num_segments(&self) -> usize {
        self.inner.num_segments()
    }

    /// Total detectors.
    fn total_detectors(&self) -> usize {
        self.inner.total_detectors()
    }

    /// Whether the circuit has feed-forward decision points (T gates).
    /// If False, the reaction time budget doesn't matter — Clifford only.
    fn has_decision_points(&self) -> bool {
        self.inner.has_decision_points()
    }

    /// Number of decision points.
    fn num_decision_points(&self) -> usize {
        self.inner.num_decision_points()
    }

    /// Reset for next shot.
    fn reset(&mut self) {
        self.inner.reset();
    }
}

// =============================================================================
// Correlation Analysis Functions
// =============================================================================

/// Compute a detector flip frequency matrix from fired-detector lists.
///
/// Args:
///     fired_per_shot: List of lists, each inner list contains the detector
///         indices that fired in that shot (sorted ascending).
///     num_detectors: Total number of detectors.
///
/// Returns:
///     Flat list of length ``num_detectors^2`` (row-major). Diagonal entries
///     are marginal rates; off-diagonal ``M[i*n+j]`` = 0.5 * P(i AND j fire).
#[pyfunction]
#[pyo3(signature = (fired_per_shot, num_detectors))]
fn detector_flip_matrix(fired_per_shot: Vec<Vec<u32>>, num_detectors: usize) -> Vec<f64> {
    pecos_qec::fault_tolerance::correlation::flip_matrix_from_fired(&fired_per_shot, num_detectors)
}

/// Compute per-round detector flip frequency matrices.
///
/// Returns a list of flat matrices, one per round.
#[pyfunction]
#[pyo3(signature = (fired_per_shot, num_detectors, dets_per_round))]
fn detector_flip_matrices_by_round(
    fired_per_shot: Vec<Vec<u32>>,
    num_detectors: usize,
    dets_per_round: usize,
) -> Vec<Vec<f64>> {
    pecos_qec::fault_tolerance::correlation::flip_matrices_by_round(
        &fired_per_shot,
        num_detectors,
        dets_per_round,
    )
}

/// Compute k-body detector firing rates up to a given order.
///
/// Returns a list of ``(detector_indices, rate)`` pairs where
/// ``detector_indices`` is a tuple of sorted detector indices.
#[pyfunction]
#[pyo3(signature = (fired_per_shot, num_detectors, max_order=3))]
fn detector_k_body_rates(
    fired_per_shot: Vec<Vec<u32>>,
    num_detectors: usize,
    max_order: usize,
) -> Vec<(Vec<u32>, f64)> {
    pecos_qec::fault_tolerance::correlation::k_body_rates(&fired_per_shot, num_detectors, max_order)
        .into_iter()
        .collect()
}

/// Compute per-round k-body detector firing rates.
///
/// Returns a list (one per round) of lists of ``(local_indices, rate)`` pairs.
#[pyfunction]
#[pyo3(signature = (fired_per_shot, num_detectors, dets_per_round, max_order=3))]
fn detector_k_body_rates_by_round(
    fired_per_shot: Vec<Vec<u32>>,
    num_detectors: usize,
    dets_per_round: usize,
    max_order: usize,
) -> Vec<Vec<(Vec<u32>, f64)>> {
    pecos_qec::fault_tolerance::correlation::k_body_rates_by_round(
        &fired_per_shot,
        num_detectors,
        dets_per_round,
        max_order,
    )
    .into_iter()
    .map(|m| m.into_iter().collect())
    .collect()
}

/// Compare two flat flip matrices. Returns (max_rel_err, frob_rel_err, worst_i, worst_j).
#[pyfunction]
#[pyo3(signature = (sim, dem, num_detectors, min_rate=0.0005))]
fn compare_flip_matrices_rs(
    sim: Vec<f64>,
    dem: Vec<f64>,
    num_detectors: usize,
    min_rate: f64,
) -> (f64, f64, usize, usize) {
    pecos_qec::fault_tolerance::correlation::compare_flip_matrices(
        &sim,
        &dem,
        num_detectors,
        min_rate,
    )
}

/// Compare k-body rates grouped by order.
///
/// Args:
///     sim: List of ``(detector_indices, rate)`` from simulation.
///     dem: List of ``(detector_indices, rate)`` from DEM.
///     min_rate: Minimum rate to consider.
///
/// Returns:
///     List of ``(order, max_rel_err, rms_rel_err, worst_event)`` tuples.
#[pyfunction]
#[pyo3(signature = (sim, dem, min_rate=0.0005))]
fn compare_k_body_rates_rs(
    sim: Vec<(Vec<u32>, f64)>,
    dem: Vec<(Vec<u32>, f64)>,
    min_rate: f64,
) -> Vec<(usize, f64, f64, Vec<u32>)> {
    let sim_map: std::collections::BTreeMap<Vec<u32>, f64> = sim.into_iter().collect();
    let dem_map: std::collections::BTreeMap<Vec<u32>, f64> = dem.into_iter().collect();
    pecos_qec::fault_tolerance::correlation::compare_k_body(&sim_map, &dem_map, min_rate)
        .into_iter()
        .map(|(order, (me, rms, worst))| (order, me, rms, worst))
        .collect()
}

/// Fit DEM mechanism probabilities to match target detector marginals.
///
/// Takes the mechanism structure (from a stochastic DEM) and exact
/// per-detector marginals (from Heisenberg EEG), and adjusts mechanism
/// probabilities so the DEM reproduces those marginals.
///
/// Args:
///     mechanisms: List of ``(probability, detector_indices, observable_indices)``
///         from the stochastic DEM.
///     target_marginals: Exact per-detector rates from Heisenberg EEG.
///     max_iterations: Maximum fitting iterations (default 200).
///     tolerance: Convergence threshold (default 1e-12).
///
/// Returns:
///     Tuple of ``(fitted_mechanisms, residuals)`` where
///     ``fitted_mechanisms`` has the same format as input but with
///     adjusted probabilities, and ``residuals`` is the per-detector
///     absolute error after fitting.
#[pyfunction]
#[pyo3(signature = (mechanisms, target_marginals, max_iterations=200, tolerance=1e-12))]
fn fit_dem_to_marginals(
    mechanisms: Vec<PyDemMechanismTuple>,
    target_marginals: Vec<f64>,
    max_iterations: usize,
    tolerance: f64,
) -> PyDemFitResult {
    use pecos_qec::fault_tolerance::correlation::{
        DemMechanism, fit_dem_to_marginals as fit_inner,
    };

    let mechs: Vec<DemMechanism> = mechanisms
        .iter()
        .map(|(p, d, o)| DemMechanism {
            probability: *p,
            detectors: d.clone(),
            observables: o.clone(),
        })
        .collect();

    let (fitted, residuals) = fit_inner(&mechs, &target_marginals, max_iterations, tolerance);

    let result: Vec<(f64, Vec<u32>, Vec<u32>)> = fitted
        .iter()
        .map(|m| (m.probability, m.detectors.clone(), m.observables.clone()))
        .collect();

    (result, residuals)
}

/// Format DEM mechanisms as a standard DEM string.
#[pyfunction]
fn mechanisms_to_dem_string(mechanisms: Vec<(f64, Vec<u32>, Vec<u32>)>) -> String {
    use pecos_qec::fault_tolerance::correlation::{
        DemMechanism, mechanisms_to_dem_string as fmt_inner,
    };

    let mechs: Vec<DemMechanism> = mechanisms
        .iter()
        .map(|(p, d, o)| DemMechanism {
            probability: *p,
            detectors: d.clone(),
            observables: o.clone(),
        })
        .collect();

    fmt_inner(&mechs)
}

/// Query whether a decoder type requires decomposed (graphlike) DEMs.
///
/// Returns ``"graphlike"`` for MWPM decoders that need decomposed DEMs
/// (hyperedges cause errors), ``"any"`` for decoders that handle both
/// raw and decomposed DEMs.
///
/// Raises ``ValueError`` for unknown decoder types.
#[pyfunction]
fn decoder_dem_requirement(decoder_type: &str) -> PyResult<String> {
    let base = decoder_type.split(':').next().unwrap_or(decoder_type);
    // "perturbed" wraps an arbitrary inner decoder ("perturbed:K=15,inner=TYPE"),
    // so its requirement is the inner decoder's. `inner=` takes the rest of the
    // string, matching how create_observable_decoder parses nested specs.
    if base == "perturbed" {
        let inner = decoder_type
            .split_once("inner=")
            .map_or("pymatching", |(_, rest)| rest);
        return decoder_dem_requirement(inner);
    }
    match base {
        "pymatching"
        | "pymatching_correlated"
        | "pymatching_uncorrelated"
        | "fusion_blossom"
        | "fusion_blossom_serial"
        | "fusion_blossom_parallel"
        | "fusion_blossom_correlated"
        | "pecos_uf"
        | "pecos_uf_correlated"
        | "windowed"
        | "k_mwpm"
        | "perturbed_fb_corr"
        | "perturbed_fb"
        | "beamsearch"
        | "belief_matching"
        | "belief_matching_correlated"
        | "belief_matching_mgbp"
        | "belief_matching_hybrid"
        | "ensemble" => Ok("graphlike".to_string()),
        "tesseract" | "astar" | "astar_full" | "bp_osd" | "bp_lsd" | "belief_find"
        | "union_find" | "min_sum_bp" | "relay_bp" | "mwpf" | "chromobius" => Ok("any".to_string()),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown decoder type: {decoder_type:?}",
        ))),
    }
}

// =============================================================================
// Circuit fault-tolerance diagnosis and distance certification
// =============================================================================

/// A gate location in a tick circuit where a Pauli fault is injected.
#[pyclass(
    name = "CircuitFaultLocation",
    module = "pecos_rslib.qec",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyCircuitFaultLocation {
    tick: usize,
    gate_type: String,
    qubits: Vec<usize>,
    gate_index: usize,
    before: bool,
}

impl From<&SpacetimeLocation> for PyCircuitFaultLocation {
    fn from(location: &SpacetimeLocation) -> Self {
        Self {
            tick: location.tick,
            gate_type: format!("{:?}", location.gate_type),
            qubits: location.qubits.iter().map(QubitId::index).collect(),
            gate_index: location.gate_index,
            before: location.before,
        }
    }
}

#[pymethods]
impl PyCircuitFaultLocation {
    #[getter]
    fn tick(&self) -> usize {
        self.tick
    }

    #[getter]
    fn gate_type(&self) -> String {
        self.gate_type.clone()
    }

    #[getter]
    fn qubits(&self) -> Vec<usize> {
        self.qubits.clone()
    }

    #[getter]
    fn gate_index(&self) -> usize {
        self.gate_index
    }

    #[getter]
    fn before(&self) -> bool {
        self.before
    }

    fn __repr__(&self) -> String {
        format!(
            "CircuitFaultLocation(tick={}, gate_type={:?}, qubits={:?}, gate_index={}, before={})",
            self.tick, self.gate_type, self.qubits, self.gate_index, self.before
        )
    }
}

type PyCircuitFault = (PyCircuitFaultLocation, Vec<usize>);

fn python_faults(configuration: &FaultConfiguration) -> Vec<PyCircuitFault> {
    configuration
        .faults
        .iter()
        .map(|fault| {
            (
                PyCircuitFaultLocation::from(&fault.location),
                fault.paulis.iter().copied().map(usize::from).collect(),
            )
        })
        .collect()
}

/// A single-location fault that amplifies into a multi-qubit data error.
#[pyclass(name = "HookError", module = "pecos_rslib.qec", skip_from_py_object)]
#[derive(Clone)]
pub struct PyHookError {
    location: PyCircuitFaultLocation,
    fault_paulis: Vec<usize>,
    data_support: Vec<usize>,
    data_weight: usize,
    detected: bool,
    causes_logical_error: bool,
}

impl From<RustHookError> for PyHookError {
    fn from(error: RustHookError) -> Self {
        Self {
            location: PyCircuitFaultLocation::from(&error.location),
            fault_paulis: error.fault_paulis.into_iter().map(usize::from).collect(),
            data_support: error.data_support,
            data_weight: error.data_weight,
            detected: error.detected,
            causes_logical_error: error.causes_logical_error,
        }
    }
}

#[pymethods]
impl PyHookError {
    #[getter]
    fn location(&self) -> PyCircuitFaultLocation {
        self.location.clone()
    }

    #[getter]
    fn fault_paulis(&self) -> Vec<usize> {
        self.fault_paulis.clone()
    }

    #[getter]
    fn data_support(&self) -> Vec<usize> {
        self.data_support.clone()
    }

    #[getter]
    fn data_weight(&self) -> usize {
        self.data_weight
    }

    #[getter]
    fn detected(&self) -> bool {
        self.detected
    }

    #[getter]
    fn causes_logical_error(&self) -> bool {
        self.causes_logical_error
    }

    fn __repr__(&self) -> String {
        format!(
            "HookError(location={}, fault_paulis={:?}, data_support={:?}, data_weight={}, detected={}, causes_logical_error={})",
            self.location.__repr__(),
            self.fault_paulis,
            self.data_support,
            self.data_weight,
            self.detected,
            self.causes_logical_error
        )
    }
}

/// Summary of hook-error diagnosis over the selected Pauli fault set.
#[pyclass(
    name = "HookErrorReport",
    module = "pecos_rslib.qec",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyHookErrorReport {
    hook_errors: Vec<PyHookError>,
    total_faults_examined: usize,
    max_data_weight: usize,
}

impl From<RustHookErrorReport> for PyHookErrorReport {
    fn from(report: RustHookErrorReport) -> Self {
        Self {
            hook_errors: report
                .hook_errors
                .into_iter()
                .map(PyHookError::from)
                .collect(),
            total_faults_examined: report.total_faults_examined,
            max_data_weight: report.max_data_weight,
        }
    }
}

#[pymethods]
impl PyHookErrorReport {
    #[getter]
    fn hook_errors(&self) -> Vec<PyHookError> {
        self.hook_errors.clone()
    }

    #[getter]
    fn total_faults_examined(&self) -> usize {
        self.total_faults_examined
    }

    #[getter]
    fn max_data_weight(&self) -> usize {
        self.max_data_weight
    }

    fn __repr__(&self) -> String {
        format!(
            "HookErrorReport(hook_errors={}, total_faults_examined={}, max_data_weight={})",
            self.hook_errors.len(),
            self.total_faults_examined,
            self.max_data_weight
        )
    }
}

/// A counterexample to the propagated-fault condition for a flag circuit.
#[pyclass(
    name = "FlagViolation",
    module = "pecos_rslib.qec",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyFlagViolation {
    faults: Vec<PyCircuitFault>,
    num_faults: usize,
    error_weight: usize,
}

impl From<RustFlagViolation> for PyFlagViolation {
    fn from(violation: RustFlagViolation) -> Self {
        Self {
            faults: python_faults(&violation.faults),
            num_faults: violation.num_faults,
            error_weight: violation.error_weight,
        }
    }
}

#[pymethods]
impl PyFlagViolation {
    #[getter]
    fn faults(&self) -> Vec<PyCircuitFault> {
        self.faults.clone()
    }

    #[getter]
    fn num_faults(&self) -> usize {
        self.num_faults
    }

    #[getter]
    fn error_weight(&self) -> usize {
        self.error_weight
    }

    fn __repr__(&self) -> String {
        format!(
            "FlagViolation(num_faults={}, error_weight={}, faults={})",
            self.num_faults,
            self.error_weight,
            self.faults.len()
        )
    }
}

/// Result of checking the propagated-fault condition through fault weight ``t``.
#[pyclass(
    name = "FlagFaultToleranceReport",
    module = "pecos_rslib.qec",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyFlagFaultToleranceReport {
    fault_condition_satisfied: bool,
    t: usize,
    violations: Vec<PyFlagViolation>,
    total_configurations_tested: usize,
}

impl From<RustFlagFaultToleranceReport> for PyFlagFaultToleranceReport {
    fn from(report: RustFlagFaultToleranceReport) -> Self {
        Self {
            fault_condition_satisfied: report.fault_condition_satisfied,
            t: report.t,
            violations: report
                .violations
                .into_iter()
                .map(PyFlagViolation::from)
                .collect(),
            total_configurations_tested: report.total_configurations_tested,
        }
    }
}

#[pymethods]
impl PyFlagFaultToleranceReport {
    #[getter]
    fn fault_condition_satisfied(&self) -> bool {
        self.fault_condition_satisfied
    }

    #[getter]
    fn t(&self) -> usize {
        self.t
    }

    #[getter]
    fn violations(&self) -> Vec<PyFlagViolation> {
        self.violations.clone()
    }

    #[getter]
    fn total_configurations_tested(&self) -> usize {
        self.total_configurations_tested
    }

    fn __repr__(&self) -> String {
        format!(
            "FlagFaultToleranceReport(fault_condition_satisfied={}, t={}, violations={}, total_configurations_tested={})",
            self.fault_condition_satisfied,
            self.t,
            self.violations.len(),
            self.total_configurations_tested
        )
    }
}

/// A minimum circuit fault distance and its first iterator-ordered witness.
#[pyclass(
    name = "CircuitDistanceResult",
    module = "pecos_rslib.qec",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyCircuitDistanceResult {
    distance: usize,
    witness: Vec<PyCircuitFault>,
    logical_index: usize,
}

impl From<RustCircuitDistanceResult> for PyCircuitDistanceResult {
    fn from(result: RustCircuitDistanceResult) -> Self {
        Self {
            distance: result.distance,
            witness: python_faults(&result.witness),
            logical_index: result.logical_index,
        }
    }
}

#[pymethods]
impl PyCircuitDistanceResult {
    #[getter]
    fn distance(&self) -> usize {
        self.distance
    }

    #[getter]
    fn witness(&self) -> Vec<PyCircuitFault> {
        self.witness.clone()
    }

    #[getter]
    fn logical_index(&self) -> usize {
        self.logical_index
    }

    fn __repr__(&self) -> String {
        format!(
            "CircuitDistanceResult(distance={}, logical_index={}, witness_faults={})",
            self.distance,
            self.logical_index,
            self.witness.len()
        )
    }
}

fn selected_fault_config(x_only: bool, y_only: bool, z_only: bool) -> FaultCheckConfig {
    let restricted = x_only || y_only || z_only;
    FaultCheckConfig {
        include_x: !restricted || x_only,
        include_y: !restricted || y_only,
        include_z: !restricted || z_only,
        ..FaultCheckConfig::default()
    }
}

fn logical_slices(logicals: &[(Vec<usize>, Vec<usize>)]) -> Vec<(&[usize], &[usize])> {
    logicals
        .iter()
        .map(|(xs, zs)| (xs.as_slice(), zs.as_slice()))
        .collect()
}

/// Fault-tolerance diagnostics and circuit distance searches for a tick circuit.
///
/// The analyzer owns a clone of the supplied circuit. Rust's ``PauliPropChecker`` and
/// ``FaultChecker`` intentionally borrow a circuit, so each method constructs a fresh checker;
/// checker construction only extracts circuit locations and is cheap at these analysis scales.
#[pyclass(name = "CircuitFaultAnalyzer", module = "pecos_rslib.qec")]
pub struct PyCircuitFaultAnalyzer {
    circuit: pecos_quantum::TickCircuit,
}

#[pymethods]
impl PyCircuitFaultAnalyzer {
    #[new]
    fn new(circuit: &PyTickCircuit) -> Self {
        Self {
            circuit: circuit.inner.clone(),
        }
    }

    /// Diagnose single-location faults that amplify across the data block.
    ///
    /// With no Pauli-selection keyword, X, Y, and Z faults are all included. Setting any of
    /// ``x_only``, ``y_only``, or ``z_only`` restricts enumeration to the selected union.
    #[pyo3(signature = (data_qubits, z_ancillas, x_ancillas, logicals, min_data_weight, *, x_only=false, y_only=false, z_only=false))]
    fn hook_errors(
        &self,
        data_qubits: Vec<usize>,
        z_ancillas: Vec<usize>,
        x_ancillas: Vec<usize>,
        logicals: Vec<(Vec<usize>, Vec<usize>)>,
        min_data_weight: usize,
        x_only: bool,
        y_only: bool,
        z_only: bool,
    ) -> PyHookErrorReport {
        // Checkers borrow TickCircuit by design. The Python owner retains a clone and checker
        // construction (location extraction) is cheap enough to repeat for each method call.
        let checker = PauliPropChecker::new(&self.circuit)
            .with_config(selected_fault_config(x_only, y_only, z_only));
        let logicals = logical_slices(&logicals);
        checker
            .diagnose_hook_errors(
                &data_qubits,
                &z_ancillas,
                &x_ancillas,
                &logicals,
                min_data_weight,
            )
            .into()
    }

    /// Verify the propagated-fault part of the Chao-Reichardt t-flag condition.
    #[pyo3(signature = (data_qubits, flag_qubits, measured_stabilizer, t, *, x_only=false, y_only=false, z_only=false))]
    fn flag_fault_condition(
        &self,
        data_qubits: Vec<usize>,
        flag_qubits: Vec<usize>,
        measured_stabilizer: (Vec<usize>, Vec<usize>),
        t: usize,
        x_only: bool,
        y_only: bool,
        z_only: bool,
    ) -> PyFlagFaultToleranceReport {
        let checker = PauliPropChecker::new(&self.circuit)
            .with_config(selected_fault_config(x_only, y_only, z_only));
        checker
            .verify_flag_fault_tolerance(
                &data_qubits,
                &flag_qubits,
                (&measured_stabilizer.0, &measured_stabilizer.1),
                t,
            )
            .into()
    }

    /// Find the minimum undetectable logical fault weight through ``max_weight``.
    #[pyo3(signature = (z_ancillas, x_ancillas, logicals, max_weight, *, x_only=false, y_only=false, z_only=false))]
    fn fault_distance(
        &self,
        z_ancillas: Vec<usize>,
        x_ancillas: Vec<usize>,
        logicals: Vec<(Vec<usize>, Vec<usize>)>,
        max_weight: usize,
        x_only: bool,
        y_only: bool,
        z_only: bool,
    ) -> Option<PyCircuitDistanceResult> {
        let checker = FaultChecker::new(&self.circuit)
            .with_config(selected_fault_config(x_only, y_only, z_only));
        let logicals = logical_slices(&logicals);
        checker
            .circuit_fault_distance(&z_ancillas, &x_ancillas, &logicals, max_weight)
            .map(PyCircuitDistanceResult::from)
    }

    /// Find one fault distance result for each supplied logical operator.
    #[pyo3(signature = (z_ancillas, x_ancillas, logicals, max_weight, *, x_only=false, y_only=false, z_only=false))]
    fn per_logical_fault_distances(
        &self,
        z_ancillas: Vec<usize>,
        x_ancillas: Vec<usize>,
        logicals: Vec<(Vec<usize>, Vec<usize>)>,
        max_weight: usize,
        x_only: bool,
        y_only: bool,
        z_only: bool,
    ) -> Vec<Option<PyCircuitDistanceResult>> {
        let checker = FaultChecker::new(&self.circuit)
            .with_config(selected_fault_config(x_only, y_only, z_only));
        let logicals = logical_slices(&logicals);
        checker
            .per_logical_circuit_fault_distances(&z_ancillas, &x_ancillas, &logicals, max_weight)
            .into_iter()
            .map(|result| result.map(PyCircuitDistanceResult::from))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "CircuitFaultAnalyzer(ticks={}, gates={})",
            self.circuit.num_ticks(),
            self.circuit.gate_count()
        )
    }
}

/// A natively checked SAT witness and the solver-trusted UNSAT prefix below it.
#[pyclass(
    name = "CertifiedDistance",
    module = "pecos_rslib.qec",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyCertifiedDistance {
    distance: usize,
    witness: Vec<bool>,
    sat_certified: bool,
    unsat_trusted_below: usize,
}

impl From<RustCertifiedDistance> for PyCertifiedDistance {
    fn from(result: RustCertifiedDistance) -> Self {
        Self {
            distance: result.distance,
            witness: result.witness,
            sat_certified: result.sat_certified,
            unsat_trusted_below: result.unsat_trusted_below,
        }
    }
}

#[pymethods]
impl PyCertifiedDistance {
    #[getter]
    fn distance(&self) -> usize {
        self.distance
    }

    #[getter]
    fn witness(&self) -> Vec<bool> {
        self.witness.clone()
    }

    #[getter]
    fn sat_certified(&self) -> bool {
        self.sat_certified
    }

    #[getter]
    fn unsat_trusted_below(&self) -> usize {
        self.unsat_trusted_below
    }

    fn __repr__(&self) -> String {
        format!(
            "CertifiedDistance(distance={}, witness_weight={}, sat_certified={}, unsat_trusted_below={})",
            self.distance, self.distance, self.sat_certified, self.unsat_trusted_below
        )
    }
}

/// Result of a budgeted stabilizer-code distance search.
#[pyclass(
    name = "StabilizerDistanceSearchResult",
    module = "pecos_rslib.qec",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyStabilizerDistanceSearchResult {
    result: Option<RustDistanceResult>,
    max_weight: Option<usize>,
}

impl From<RustStabilizerDistanceSearchOutcome> for PyStabilizerDistanceSearchResult {
    fn from(outcome: RustStabilizerDistanceSearchOutcome) -> Self {
        match outcome {
            RustStabilizerDistanceSearchOutcome::Certified(result) => Self {
                result: Some(result),
                max_weight: None,
            },
            RustStabilizerDistanceSearchOutcome::BudgetExhausted { max_weight } => Self {
                result: None,
                max_weight: Some(max_weight),
            },
        }
    }
}

#[pymethods]
impl PyStabilizerDistanceSearchResult {
    #[getter]
    fn certified(&self) -> bool {
        self.result.is_some()
    }

    #[getter]
    fn distance(&self) -> Option<usize> {
        self.result.as_ref().map(|result| result.distance)
    }

    #[getter]
    fn min_weight_operator(&self) -> Option<crate::pauli_bindings::PauliString> {
        self.result.as_ref().map(|result| {
            crate::pauli_bindings::PauliString::from_rust(result.min_weight_operator.clone())
        })
    }

    #[getter]
    fn lower_bound(&self) -> usize {
        self.result.as_ref().map_or_else(
            || self.max_weight.unwrap_or(0) + 1,
            |result| result.distance,
        )
    }

    #[getter]
    fn max_weight(&self) -> Option<usize> {
        self.max_weight
    }

    fn __repr__(&self) -> String {
        match &self.result {
            Some(result) => format!(
                "StabilizerDistanceSearchResult(certified=True, distance={})",
                result.distance
            ),
            None => format!(
                "StabilizerDistanceSearchResult(certified=False, lower_bound={}, max_weight={})",
                self.lower_bound(),
                self.max_weight.unwrap_or(0)
            ),
        }
    }
}

/// Result of a budgeted classical-code distance certification.
#[pyclass(
    name = "ClassicalDistanceSearchResult",
    module = "pecos_rslib.qec",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyClassicalDistanceSearchResult {
    outcome: RustClassicalDistanceSearchOutcome,
}

impl From<RustClassicalDistanceSearchOutcome> for PyClassicalDistanceSearchResult {
    fn from(outcome: RustClassicalDistanceSearchOutcome) -> Self {
        Self { outcome }
    }
}

#[pymethods]
impl PyClassicalDistanceSearchResult {
    #[getter]
    fn certified(&self) -> bool {
        !matches!(
            self.outcome,
            RustClassicalDistanceSearchOutcome::BudgetExhausted { .. }
        )
    }

    #[getter]
    fn distance(&self) -> Option<usize> {
        match &self.outcome {
            RustClassicalDistanceSearchOutcome::Certified(result) => Some(result.distance),
            RustClassicalDistanceSearchOutcome::NoNonzeroCodeword
            | RustClassicalDistanceSearchOutcome::BudgetExhausted { .. } => None,
        }
    }

    #[getter]
    fn witness(&self) -> Option<Vec<bool>> {
        match &self.outcome {
            RustClassicalDistanceSearchOutcome::Certified(result) => Some(result.witness.clone()),
            RustClassicalDistanceSearchOutcome::NoNonzeroCodeword
            | RustClassicalDistanceSearchOutcome::BudgetExhausted { .. } => None,
        }
    }

    #[getter]
    fn lower_bound(&self) -> Option<usize> {
        match &self.outcome {
            RustClassicalDistanceSearchOutcome::Certified(result) => Some(result.distance),
            RustClassicalDistanceSearchOutcome::BudgetExhausted { max_weight } => {
                Some(max_weight + 1)
            }
            RustClassicalDistanceSearchOutcome::NoNonzeroCodeword => None,
        }
    }

    #[getter]
    fn max_weight(&self) -> Option<usize> {
        match self.outcome {
            RustClassicalDistanceSearchOutcome::BudgetExhausted { max_weight } => Some(max_weight),
            RustClassicalDistanceSearchOutcome::Certified(_)
            | RustClassicalDistanceSearchOutcome::NoNonzeroCodeword => None,
        }
    }

    #[getter]
    fn no_nonzero_codeword(&self) -> bool {
        matches!(
            self.outcome,
            RustClassicalDistanceSearchOutcome::NoNonzeroCodeword
        )
    }

    fn __repr__(&self) -> String {
        match &self.outcome {
            RustClassicalDistanceSearchOutcome::Certified(result) => format!(
                "ClassicalDistanceSearchResult(certified=True, distance={})",
                result.distance
            ),
            RustClassicalDistanceSearchOutcome::NoNonzeroCodeword => {
                "ClassicalDistanceSearchResult(certified=True, no_nonzero_codeword=True)"
                    .to_string()
            }
            RustClassicalDistanceSearchOutcome::BudgetExhausted { max_weight } => format!(
                "ClassicalDistanceSearchResult(certified=False, lower_bound={}, max_weight={max_weight})",
                max_weight + 1
            ),
        }
    }
}

/// Native lower and upper bounds from bounded generator-row enumeration.
#[pyclass(
    name = "BoundedEnumerationDistance",
    module = "pecos_rslib.qec",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyBoundedEnumerationDistance {
    lower_bound: usize,
    upper_bound: usize,
    witness: Vec<bool>,
    certified: bool,
    level: Option<usize>,
    max_level: Option<usize>,
    lb_certified: bool,
}

impl From<RustBoundedEnumerationDistance> for PyBoundedEnumerationDistance {
    fn from(result: RustBoundedEnumerationDistance) -> Self {
        match result {
            RustBoundedEnumerationDistance::CertifiedByBounds {
                distance,
                witness,
                lower_bound,
                level,
                lb_certified,
            } => Self {
                lower_bound,
                upper_bound: distance,
                witness,
                certified: true,
                level: Some(level),
                max_level: None,
                lb_certified,
            },
            RustBoundedEnumerationDistance::LevelLimitReached {
                lower_bound,
                upper_bound,
                witness,
                max_level,
                lb_certified,
            } => Self {
                lower_bound,
                upper_bound,
                witness,
                certified: false,
                level: None,
                max_level: Some(max_level),
                lb_certified,
            },
        }
    }
}

#[pymethods]
impl PyBoundedEnumerationDistance {
    #[getter]
    fn lower_bound(&self) -> usize {
        self.lower_bound
    }

    #[getter]
    fn upper_bound(&self) -> usize {
        self.upper_bound
    }

    #[getter]
    fn distance(&self) -> Option<usize> {
        self.certified.then_some(self.upper_bound)
    }

    #[getter]
    fn witness(&self) -> Vec<bool> {
        self.witness.clone()
    }

    #[getter]
    fn certified(&self) -> bool {
        self.certified
    }

    #[getter]
    fn level(&self) -> Option<usize> {
        self.level
    }

    #[getter]
    fn max_level(&self) -> Option<usize> {
        self.max_level
    }

    #[getter]
    fn lb_certified(&self) -> bool {
        self.lb_certified
    }

    fn __repr__(&self) -> String {
        format!(
            "BoundedEnumerationDistance(lower_bound={}, upper_bound={}, certified={}, witness_weight={})",
            self.lower_bound,
            self.upper_bound,
            self.certified,
            self.witness.iter().filter(|&&selected| selected).count()
        )
    }
}

fn certify_python_problem(
    problem: &RustDistanceProblem,
    max_weight: usize,
) -> PyResult<Option<PyCertifiedDistance>> {
    rust_certified_distance(problem, max_weight)
        .map(|result| result.map(PyCertifiedDistance::from))
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
}

/// A binary problem whose solutions are undetectable with nonzero logical effect.
#[pyclass(
    name = "DistanceProblem",
    module = "pecos_rslib.qec",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyDistanceProblem {
    inner: RustDistanceProblem,
}

#[pymethods]
impl PyDistanceProblem {
    #[classmethod]
    fn from_css_checks(
        _cls: &Bound<'_, pyo3::types::PyType>,
        hx: &PyParityCheckMatrix,
        lx: &PyParityCheckMatrix,
    ) -> PyResult<Self> {
        RustDistanceProblem::from_css_checks(&hx.inner, &lx.inner)
            .map(|inner| Self { inner })
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
    }

    #[classmethod]
    fn from_css_code_x_distance(
        _cls: &Bound<'_, pyo3::types::PyType>,
        spec: &PyStabilizerCodeSpec,
    ) -> PyResult<Self> {
        RustDistanceProblem::from_css_code_x_distance(&spec.inner)
            .map(|inner| Self { inner })
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
    }

    #[classmethod]
    fn from_css_code_z_distance(
        _cls: &Bound<'_, pyo3::types::PyType>,
        spec: &PyStabilizerCodeSpec,
    ) -> PyResult<Self> {
        RustDistanceProblem::from_css_code_z_distance(&spec.inner)
            .map(|inner| Self { inner })
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
    }

    #[classmethod]
    fn from_dem(_cls: &Bound<'_, pyo3::types::PyType>, dem: &PyDetectorErrorModel) -> Self {
        Self {
            inner: RustDistanceProblem::from_dem(&dem.inner),
        }
    }

    #[getter]
    fn num_vars(&self) -> usize {
        self.inner.num_vars()
    }

    fn to_dimacs(&self, max_weight: usize) -> String {
        self.inner.to_dimacs(max_weight)
    }

    fn to_wcnf(&self) -> String {
        self.inner.to_wcnf()
    }

    fn verify_witness(&self, witness: Vec<bool>) -> PyResult<usize> {
        self.inner
            .verify_witness(&witness)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
    }

    /// Certifies distance through ``max_weight`` using the in-process batsat SAT solver.
    ///
    /// A fresh deterministic solver instance is built for each weight from the internal clause
    /// encoding. SAT answers are certified natively with ``DistanceProblem.verify_witness`` before
    /// they are accepted. UNSAT answers, and therefore the exactness of a returned distance, rest
    /// on trusting the solver. ``None`` means batsat reported every weight through ``max_weight``
    /// UNSAT.
    fn certified_distance(&self, max_weight: usize) -> PyResult<Option<PyCertifiedDistance>> {
        certify_python_problem(&self.inner, max_weight)
    }

    fn __repr__(&self) -> String {
        format!("DistanceProblem(num_vars={})", self.inner.num_vars())
    }
}

/// Certifies distance through ``max_weight`` using the in-process batsat SAT solver.
///
/// A fresh deterministic solver instance is built for each weight from the internal clause
/// encoding. SAT answers are certified natively with ``DistanceProblem.verify_witness`` before
/// they are accepted. UNSAT answers, and therefore the exactness of a returned distance, rest
/// on trusting the solver. ``None`` means batsat reported every weight through ``max_weight``
/// UNSAT.
#[pyfunction]
fn certified_distance(
    problem: &PyDistanceProblem,
    max_weight: usize,
) -> PyResult<Option<PyCertifiedDistance>> {
    certify_python_problem(&problem.inner, max_weight)
}

/// Certified minimum weight of ``representative + rowspan(group)`` over GF(2).
///
/// Weight 0 means the representative is in the group; certified without a solver call.
/// SAT answers are natively verified; UNSAT answers are solver-trusted.
#[pyfunction]
fn certified_coset_weight(
    group: &PyParityCheckMatrix,
    representative: Vec<u8>,
    max_weight: usize,
) -> PyResult<Option<PyCertifiedDistance>> {
    pecos_qec::certified_coset_weight(&group.inner, &representative, max_weight)
        .map(|result| result.map(PyCertifiedDistance::from))
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

/// Certified minimum qubit-support weight of ``operator * stabilizer group`` for any code.
#[pyfunction]
fn certified_stabilizer_coset_weight(
    code: &PyStabilizerCodeSpec,
    operator: &crate::pauli_bindings::PauliString,
    max_weight: usize,
) -> PyResult<Option<PyCertifiedDistance>> {
    pecos_qec::certified_stabilizer_coset_weight(&code.inner, &operator.to_rust(), max_weight)
        .map(|result| result.map(PyCertifiedDistance::from))
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

/// Certified coset weight of every supplied logical generator (Z generators, then X generators).
///
/// The minimum of this list is not the code distance: two weight-two supplied generators can,
/// for example, have a weight-one product in another logical coset.
/// The supplied generators are measured as given; logical-basis completeness is not required.
#[pyfunction]
fn logical_generator_coset_weights(
    code: &PyStabilizerCodeSpec,
    max_weight: usize,
) -> PyResult<Vec<Option<PyCertifiedDistance>>> {
    pecos_qec::logical_generator_coset_weights(&code.inner, max_weight)
        .map(|profile| {
            profile
                .into_iter()
                .map(|entry| entry.map(PyCertifiedDistance::from))
                .collect()
        })
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

/// Certified minimum weight of a nonzero kernel element of a classical parity-check matrix.
#[pyfunction]
fn certified_classical_distance(
    h: &PyParityCheckMatrix,
    max_weight: usize,
) -> PyResult<PyClassicalDistanceSearchResult> {
    pecos_qec::certified_classical_distance(&h.inner, max_weight)
        .map(PyClassicalDistanceSearchResult::from)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

/// Computes binary ``(H, L)`` distance using native bounded row enumeration.
#[pyfunction]
fn bounded_enumeration_code_distance(
    h: &PyParityCheckMatrix,
    l: &PyParityCheckMatrix,
    max_level: usize,
) -> Option<PyBoundedEnumerationDistance> {
    rust_bounded_enumeration_code_distance(&h.inner, &l.inner, max_level)
        .map(PyBoundedEnumerationDistance::from)
}

/// Computes pure-X bounded-enumeration distance for a CSS stabilizer code.
#[pyfunction]
fn bounded_enumeration_x_distance(
    code: &PyStabilizerCodeSpec,
    max_level: usize,
) -> PyResult<Option<PyBoundedEnumerationDistance>> {
    rust_bounded_enumeration_x_distance(&code.inner, max_level)
        .map(|result| result.map(PyBoundedEnumerationDistance::from))
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

/// Computes pure-Z bounded-enumeration distance for a CSS stabilizer code.
#[pyfunction]
fn bounded_enumeration_z_distance(
    code: &PyStabilizerCodeSpec,
    max_level: usize,
) -> PyResult<Option<PyBoundedEnumerationDistance>> {
    rust_bounded_enumeration_z_distance(&code.inner, max_level)
        .map(|result| result.map(PyBoundedEnumerationDistance::from))
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

/// Computes bounded-enumeration distance for any stabilizer code.
#[pyfunction]
fn bounded_enumeration_stabilizer_distance(
    code: &PyStabilizerCodeSpec,
    max_level: usize,
) -> PyResult<Option<PyBoundedEnumerationDistance>> {
    rust_bounded_enumeration_stabilizer_distance(&code.inner, max_level)
        .map(|result| result.map(PyBoundedEnumerationDistance::from))
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

/// Computes connected-cluster distance for a binary check/logical matrix pair.
#[pyfunction]
fn connected_cluster_code_distance(
    h: &PyParityCheckMatrix,
    l: &PyParityCheckMatrix,
    max_weight: usize,
) -> Option<PyFaultDistanceResult> {
    rust_connected_cluster_code_distance(&h.inner, &l.inner, max_weight)
        .map(PyFaultDistanceResult::from)
}

/// Samples natively verified qubit witnesses for a binary code-distance upper bound.
///
/// Returned ``mechanism_indices`` are qubit indices. A return value is only an upper bound and
/// never certifies exactness.
#[pyfunction]
fn randomized_code_distance_upper_bound(
    h: &PyParityCheckMatrix,
    l: &PyParityCheckMatrix,
    config: &PyFaultDistanceUpperBoundConfig,
) -> PyResult<Option<PyFaultDistanceUpperBoundResult>> {
    rust_randomized_code_distance_upper_bound(&h.inner, &l.inner, &config.inner)
        .map(|result| result.map(PyFaultDistanceUpperBoundResult::from))
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

/// Computes pure-X connected-cluster distance for a CSS stabilizer code.
#[pyfunction]
fn x_distance(
    code: &PyStabilizerCodeSpec,
    max_weight: usize,
) -> PyResult<Option<PyFaultDistanceResult>> {
    rust_x_distance(&code.inner, max_weight)
        .map(|result| result.map(PyFaultDistanceResult::from))
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

/// Computes pure-Z connected-cluster distance for a CSS stabilizer code.
#[pyfunction]
fn z_distance(
    code: &PyStabilizerCodeSpec,
    max_weight: usize,
) -> PyResult<Option<PyFaultDistanceResult>> {
    rust_z_distance(&code.inner, max_weight)
        .map(|result| result.map(PyFaultDistanceResult::from))
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

/// Computes connected-cluster distance for any stabilizer code.
#[pyfunction]
fn stabilizer_code_distance(
    code: &PyStabilizerCodeSpec,
    max_weight: usize,
) -> PyResult<PyStabilizerDistanceSearchResult> {
    rust_stabilizer_code_distance(&code.inner, max_weight)
        .map(PyStabilizerDistanceSearchResult::from)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

/// A minimum-weight logical operator and its weight.
#[pyclass(name = "DistanceResult", module = "pecos_rslib.qec")]
pub struct PyDistanceResult {
    inner: RustDistanceResult,
}

impl From<RustDistanceResult> for PyDistanceResult {
    fn from(inner: RustDistanceResult) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyDistanceResult {
    /// The distance: the weight of the minimum-weight logical operator.
    #[getter]
    fn distance(&self) -> usize {
        self.inner.distance
    }

    /// A logical operator achieving the minimum weight.
    #[getter]
    fn min_weight_operator(&self) -> crate::pauli_bindings::PauliString {
        crate::pauli_bindings::PauliString::from_rust(self.inner.min_weight_operator.clone())
    }

    fn __repr__(&self) -> String {
        format!("DistanceResult(distance={})", self.inner.distance)
    }
}

/// Computes the dressed distance of a subsystem (gauge) code by qubit-support weight.
///
/// `stabilizers` are the stabilizer generators, `gauge_generators` the remaining gauge
/// generators, and the logicals bare representatives commuting with the full gauge group.
/// Returns `None` if no logical operator exists at weight at most `max_weight`.
///
/// Raises `ValueError` if the specification is ill-formed (for example a logical that
/// anticommutes with a stabilizer, or a gauge/center split that cannot describe paired
/// gauge qubits).
#[pyfunction]
fn subsystem_dressed_distance(
    num_qubits: usize,
    stabilizers: Vec<crate::pauli_bindings::PauliString>,
    gauge_generators: Vec<crate::pauli_bindings::PauliString>,
    logical_zs: Vec<crate::pauli_bindings::PauliString>,
    logical_xs: Vec<crate::pauli_bindings::PauliString>,
    max_weight: usize,
) -> PyResult<Option<PyDistanceResult>> {
    let stabilizers = stabilizers.into_iter().map(|p| p.to_rust()).collect();
    let gauge_generators: Vec<_> = gauge_generators.into_iter().map(|p| p.to_rust()).collect();
    let logical_zs = logical_zs.into_iter().map(|p| p.to_rust()).collect();
    let logical_xs = logical_xs.into_iter().map(|p| p.to_rust()).collect();

    rust_subsystem_dressed_distance(
        num_qubits,
        stabilizers,
        &gauge_generators,
        logical_zs,
        logical_xs,
        max_weight,
    )
    .map(|result| result.map(PyDistanceResult::from))
    .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
}

// =============================================================================
// Module Registration
// =============================================================================

/// A hypergraph-product CSS code built from two classical parity-check matrices.
#[pyclass(name = "HypergraphProductCode", module = "pecos_rslib.qec")]
pub struct PyHypergraphProductCode {
    inner: pecos_qec::HypergraphProductCode,
}

#[pymethods]
impl PyHypergraphProductCode {
    /// Build the hypergraph product of two classical parity-check matrices.
    #[new]
    fn new(h1: &PyParityCheckMatrix, h2: &PyParityCheckMatrix) -> PyResult<Self> {
        pecos_qec::HypergraphProductCode::new(&h1.inner, &h2.inner)
            .map(|inner| Self { inner })
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))
    }

    #[getter]
    fn hx(&self) -> PyParityCheckMatrix {
        PyParityCheckMatrix {
            inner: self.inner.hx().clone(),
        }
    }

    #[getter]
    fn hz(&self) -> PyParityCheckMatrix {
        PyParityCheckMatrix {
            inner: self.inner.hz().clone(),
        }
    }

    #[getter]
    fn logical_x(&self) -> PyParityCheckMatrix {
        PyParityCheckMatrix {
            inner: self.inner.logical_x().clone(),
        }
    }

    #[getter]
    fn logical_z(&self) -> PyParityCheckMatrix {
        PyParityCheckMatrix {
            inner: self.inner.logical_z().clone(),
        }
    }

    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    fn num_logical_qubits(&self) -> usize {
        self.inner.num_logical_qubits()
    }

    fn __repr__(&self) -> String {
        format!(
            "HypergraphProductCode(n={}, k={})",
            self.inner.num_qubits(),
            self.inner.num_logical_qubits()
        )
    }
}

/// A validated bivariate-bicycle CSS code.
#[pyclass(
    name = "BivariateBicycleCode",
    module = "pecos_rslib.qec",
    skip_from_py_object
)]
#[derive(Clone, Debug)]
pub struct PyBivariateBicycleCode {
    inner: RustBivariateBicycleCode,
}

#[pymethods]
impl PyBivariateBicycleCode {
    /// Construct `QC(A, B)` from canonical `(x_power, y_power)` exponent lists.
    #[new]
    fn new(
        l: usize,
        m: usize,
        a_terms: Vec<(usize, usize)>,
        b_terms: Vec<(usize, usize)>,
    ) -> PyResult<Self> {
        let inner = RustBivariateBicycleCode::new(l, m, &a_terms, &b_terms)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        Ok(Self { inner })
    }

    #[getter]
    fn l_order(&self) -> usize {
        self.inner.dimensions().0
    }

    #[getter]
    fn m_order(&self) -> usize {
        self.inner.dimensions().1
    }

    #[getter]
    fn hx(&self) -> PyParityCheckMatrix {
        PyParityCheckMatrix {
            inner: self.inner.hx().clone(),
        }
    }

    #[getter]
    fn hz(&self) -> PyParityCheckMatrix {
        PyParityCheckMatrix {
            inner: self.inner.hz().clone(),
        }
    }

    #[getter]
    fn logical_x(&self) -> PyParityCheckMatrix {
        PyParityCheckMatrix {
            inner: self.inner.logical_x().clone(),
        }
    }

    #[getter]
    fn logical_z(&self) -> PyParityCheckMatrix {
        PyParityCheckMatrix {
            inner: self.inner.logical_z().clone(),
        }
    }

    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    fn num_logical_qubits(&self) -> usize {
        self.inner.num_logical_qubits()
    }

    fn __repr__(&self) -> String {
        format!(
            "BivariateBicycleCode(l={}, m={}, n={}, k={})",
            self.l_order(),
            self.m_order(),
            self.num_qubits(),
            self.num_logical_qubits()
        )
    }
}

/// Build the Table 5 bivariate-bicycle memory circuit.
#[pyfunction]
fn bb_memory_circuit(
    l: usize,
    m: usize,
    a_terms: Vec<(usize, usize)>,
    b_terms: Vec<(usize, usize)>,
    rounds: usize,
    basis: &str,
) -> PyResult<PyTickCircuit> {
    let basis = match basis.to_ascii_uppercase().as_str() {
        "X" => RustBbMemoryBasis::X,
        "Z" => RustBbMemoryBasis::Z,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "basis must be 'X' or 'Z', got {basis:?}"
            )));
        }
    };
    let inner = rust_bb_memory_circuit(l, m, &a_terms, &b_terms, rounds, basis)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    Ok(PyTickCircuit { inner })
}

/// Build a generic CSS memory circuit from exact Tanner-graph edge colorings.
#[pyfunction]
fn coloration_memory_circuit(
    hx: &PyParityCheckMatrix,
    hz: &PyParityCheckMatrix,
    rounds: usize,
    basis: &str,
) -> PyResult<PyTickCircuit> {
    let basis = match basis.to_ascii_uppercase().as_str() {
        "X" => RustBbMemoryBasis::X,
        "Z" => RustBbMemoryBasis::Z,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "basis must be 'X' or 'Z', got {basis:?}"
            )));
        }
    };
    let inner = rust_coloration_memory_circuit(&hx.inner, &hz.inner, rounds, basis)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    Ok(PyTickCircuit { inner })
}

/// Register the QEC fault tolerance module.
pub fn register_qec_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let qec = PyModule::new(m.py(), "qec")?;

    qec.add_class::<PyObservableFlips>()?;
    qec.add_class::<PyFaultLocation>()?;
    qec.add_class::<PyDagFaultInfluenceMap>()?;
    qec.add_class::<PyDagFaultAnalyzer>()?;
    qec.add_class::<PyInfluenceBuilder>()?;
    qec.add_class::<PyPauliFrameLookup>()?;
    qec.add_class::<PyFaultDistanceResult>()?;
    qec.add_class::<PyFaultDistanceUpperBoundConfig>()?;
    qec.add_class::<PyFaultDistanceUpperBoundResult>()?;
    qec.add_class::<PyDetectorErrorModel>()?;
    qec.add_class::<PyDemBuilder>()?;
    qec.add_class::<PySampleBatch>()?;
    qec.add_class::<batch_decode::PyDecodeResult>()?;
    qec.add_class::<PyDecoderComparisonResult>()?;
    qec.add_class::<PyCssUfDecoder>()?;
    qec.add_class::<PyLogicalSubgraphDecoder>()?;
    qec.add_class::<PyWindowedLogicalSubgraphDecoder>()?;
    qec.add_class::<PyLogicalAlgorithmDecoder>()?;
    qec.add_class::<PyLogicalCircuitDecoder>()?;
    qec.add_class::<PyDecodeStats>()?;
    qec.add_class::<PyDemSampler>()?;
    qec.add_class::<PyDemSamplerBuilder>()?;
    qec.add_class::<PyEquivalenceResult>()?;
    qec.add_class::<PyParsedDem>()?;
    qec.add_class::<PyCircuitFaultLocation>()?;
    qec.add_class::<PyHookError>()?;
    qec.add_class::<PyHookErrorReport>()?;
    qec.add_class::<PyFlagViolation>()?;
    qec.add_class::<PyFlagFaultToleranceReport>()?;
    qec.add_class::<PyCircuitDistanceResult>()?;
    qec.add_class::<PyCircuitFaultAnalyzer>()?;
    qec.add_class::<PyCertifiedDistance>()?;
    qec.add_class::<PyStabilizerDistanceSearchResult>()?;
    qec.add_class::<PyClassicalDistanceSearchResult>()?;
    qec.add_class::<PyBoundedEnumerationDistance>()?;
    qec.add_class::<PyDistanceProblem>()?;
    qec.add_class::<PyBivariateBicycleCode>()?;
    qec.add_class::<PyHypergraphProductCode>()?;

    // Add DEM equivalence functions
    qec.add_function(wrap_pyfunction!(compare_dems_exact, &qec)?)?;
    qec.add_function(wrap_pyfunction!(compare_dems_statistical, &qec)?)?;
    qec.add_function(wrap_pyfunction!(verify_dem_equivalence, &qec)?)?;
    qec.add_function(wrap_pyfunction!(assert_dems_equivalent, &qec)?)?;
    qec.add_function(wrap_pyfunction!(connected_cluster_code_distance, &qec)?)?;
    qec.add_function(wrap_pyfunction!(
        randomized_code_distance_upper_bound,
        &qec
    )?)?;
    qec.add_function(wrap_pyfunction!(bounded_enumeration_code_distance, &qec)?)?;
    qec.add_function(wrap_pyfunction!(bounded_enumeration_x_distance, &qec)?)?;
    qec.add_function(wrap_pyfunction!(bounded_enumeration_z_distance, &qec)?)?;
    qec.add_function(wrap_pyfunction!(
        bounded_enumeration_stabilizer_distance,
        &qec
    )?)?;
    qec.add_function(wrap_pyfunction!(x_distance, &qec)?)?;
    qec.add_function(wrap_pyfunction!(z_distance, &qec)?)?;
    qec.add_function(wrap_pyfunction!(stabilizer_code_distance, &qec)?)?;
    qec.add_class::<PyDistanceResult>()?;
    qec.add_function(wrap_pyfunction!(subsystem_dressed_distance, &qec)?)?;

    // Correlation analysis
    qec.add_function(wrap_pyfunction!(detector_flip_matrix, &qec)?)?;
    qec.add_function(wrap_pyfunction!(detector_flip_matrices_by_round, &qec)?)?;
    qec.add_function(wrap_pyfunction!(detector_k_body_rates, &qec)?)?;
    qec.add_function(wrap_pyfunction!(detector_k_body_rates_by_round, &qec)?)?;
    qec.add_function(wrap_pyfunction!(compare_flip_matrices_rs, &qec)?)?;
    qec.add_function(wrap_pyfunction!(compare_k_body_rates_rs, &qec)?)?;
    qec.add_function(wrap_pyfunction!(fit_dem_to_marginals, &qec)?)?;
    qec.add_function(wrap_pyfunction!(mechanisms_to_dem_string, &qec)?)?;
    qec.add_function(wrap_pyfunction!(decoder_dem_requirement, &qec)?)?;
    qec.add_function(wrap_pyfunction!(certified_distance, &qec)?)?;
    qec.add_function(wrap_pyfunction!(certified_coset_weight, &qec)?)?;
    qec.add_function(wrap_pyfunction!(certified_stabilizer_coset_weight, &qec)?)?;
    qec.add_function(wrap_pyfunction!(logical_generator_coset_weights, &qec)?)?;
    qec.add_function(wrap_pyfunction!(certified_classical_distance, &qec)?)?;
    qec.add_function(wrap_pyfunction!(bb_memory_circuit, &qec)?)?;
    qec.add_function(wrap_pyfunction!(coloration_memory_circuit, &qec)?)?;

    // Add Pauli constants
    qec.add("PAULI_I", 0u8)?;
    qec.add("PAULI_X", 1u8)?;
    qec.add("PAULI_Y", 2u8)?;
    qec.add("PAULI_Z", 3u8)?;

    m.add_submodule(&qec)?;

    // Keep the common DEM sampler import available at the package root for
    // scripts that use `from pecos_rslib import DemSampler`.
    m.add("DemSampler", qec.getattr("DemSampler")?)?;

    // Register in sys.modules so 'from pecos_rslib.qec import ...' works
    let sys = m.py().import("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("pecos_rslib.qec", &qec)?;

    Ok(())
}
