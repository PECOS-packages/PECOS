// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

//! High-level DEM construction from circuits or runtime adapters.
//!
//! Runtime adapters own compilation, execution and schedule/provenance checks.
//! The QEC layer owns annotation inference, circuit preparation, fault
//! propagation and measurement evaluation, without depending on a runtime.

use crate::dem_inference::{
    DemInferenceData, DemInferenceError, DemInferenceReport, DemInferenceSettings,
};
use crate::fault_tolerance::circuit_definitions::{
    CircuitDefinitions, DefinitionError, definitions_from_tick_circuit, tick_circuit_emission,
};
use crate::{DemBuilder, DemBuilderError, DetectorErrorModel, NoiseConfig};
use pecos_decoder_core::obs_mask::ObsMask;
use pecos_quantum::{Attribute, TickCircuit, pass};
use std::collections::{BTreeMap, BTreeSet};

/// Input prepared by a caller or runtime adapter. Probe IDs must be bound to
/// this circuit's measurements; the builder checks complete identity coverage.
pub struct DemBuildInput {
    pub circuit: TickCircuit,
    pub probes: Option<DemInferenceData>,
}

/// Execution options passed to a runtime adapter before building a DEM.
pub struct DemProgramRequest<'a> {
    pub num_qubits: usize,
    pub seed: u64,
    pub inference: Option<&'a DemInferenceSettings>,
}

/// A program adapter owns its runtime and prepares a circuit plus optional
/// coin-toss probes. Implementations must validate the program's schedule and
/// honor `require_raw_provenance`; a successful fit does not prove either.
pub trait DemProgram {
    /// Execute and prepare the requested program.
    ///
    /// # Errors
    /// Return an error if execution, schedule validation or provenance fails.
    fn prepare(&mut self, request: &DemProgramRequest<'_>) -> Result<DemBuildInput, DemBuildError>;
}

/// Errors from configuration, runtime preparation, inference or propagation.
#[derive(Debug, thiserror::Error)]
pub enum DemBuildError {
    #[error("{0}")]
    Configuration(String),
    #[error("{0}")]
    Preparation(String),
    #[error(transparent)]
    Inference(#[from] DemInferenceError),
    #[error(transparent)]
    Definitions(#[from] DefinitionError),
    #[error(transparent)]
    Propagation(#[from] DemBuilderError),
}

/// A DEM and the resolved measurement schema used to construct it.
pub struct DemBuild {
    pub dem: DetectorErrorModel,
    pub circuit: TickCircuit,
    pub definitions: CircuitDefinitions,
    /// Stable measurement IDs in runtime emission order.
    pub measurement_ids: Vec<usize>,
    pub inference: Option<DemInferenceReport>,
    raw_tag: Option<String>,
}

impl DemBuild {
    /// Evaluate a physical measurement record in circuit emission order.
    /// These are measurement parities; no unknown noiseless references are invented.
    ///
    /// # Errors
    /// Rejects invalid bits, wrong widths and tracked-Pauli outputs.
    pub fn evaluate_runtime_record(
        &self,
        values: &[u8],
    ) -> Result<(Vec<u8>, ObsMask), DemBuildError> {
        if values.len() != self.measurement_ids.len() || values.iter().any(|&bit| bit > 1) {
            return Err(DemBuildError::Configuration(
                "measurement record must contain one bit per circuit measurement".into(),
            ));
        }
        let parity = |positions: &[usize]| {
            positions
                .iter()
                .fold(0, |value, &index| value ^ values[index])
        };
        let mut detectors = vec![0; self.dem.num_detectors()];
        for detector in &self.definitions.detectors {
            let event = detectors.get_mut(detector.id as usize).ok_or_else(|| {
                DemBuildError::Configuration("detector IDs must fit the DEM detector vector".into())
            })?;
            *event = parity(&detector.measurements);
        }
        let mut observables = ObsMask::new();
        for observable in &self.definitions.observables {
            if observable.pauli.is_some() {
                return Err(DemBuildError::Configuration(
                    "tracked-Pauli outputs cannot be evaluated from measurement records".into(),
                ));
            }
            if parity(&observable.measurements) != 0 {
                observables.set(observable.id as usize);
            }
        }
        Ok((detectors, observables))
    }

    /// Evaluate a record keyed by stable measurement ID.
    ///
    /// # Errors
    /// Rejects missing measurements or invalid bits.
    pub fn evaluate_measurements(
        &self,
        values: &BTreeMap<usize, u8>,
    ) -> Result<(Vec<u8>, ObsMask), DemBuildError> {
        let row = self
            .measurement_ids
            .iter()
            .map(|id| {
                values.get(id).copied().ok_or_else(|| {
                    DemBuildError::Configuration(format!("missing measurement ID {id}"))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.evaluate_runtime_record(&row)
    }

    /// Evaluate inferred programs' raw result columns, including reordered arrays.
    ///
    /// # Errors
    /// Requires inferred raw-output bindings and correctly sized bit rows.
    pub fn evaluate_result_columns(
        &self,
        columns: &BTreeMap<String, Vec<Vec<u8>>>,
    ) -> Result<Vec<(Vec<u8>, ObsMask)>, DemBuildError> {
        let report = self.inference.as_ref().ok_or_else(|| DemBuildError::Configuration("result-column evaluation requires an inferred raw-output binding; use evaluate_measurements for explicit definitions".into()))?;
        let tag = self
            .raw_tag
            .as_ref()
            .ok_or_else(|| DemBuildError::Configuration("missing raw output tag".into()))?;
        let rows = columns.get(tag).ok_or_else(|| {
            DemBuildError::Configuration(format!("missing raw result tag {tag:?}"))
        })?;
        rows.iter()
            .map(|row| {
                if row.len() != report.raw_measurement_ids.len() {
                    return Err(DemBuildError::Configuration(
                        "raw result width does not match inferred binding".into(),
                    ));
                }
                self.evaluate_measurements(
                    &report
                        .raw_measurement_ids
                        .iter()
                        .copied()
                        .zip(row.iter().copied())
                        .collect(),
                )
            })
            .collect()
    }
}

/// Fluent high-level construction. Explicit metadata and inference are exclusive.
/// Validation occurs before a runtime adapter executes.
#[derive(Default)]
pub struct DemBuildBuilder {
    circuit: Option<TickCircuit>,
    program: Option<Box<dyn DemProgram>>,
    qubits: Option<usize>,
    seed: u64,
    probes: Option<DemInferenceData>,
    inference: Option<DemInferenceSettings>,
    detectors: Option<String>,
    observables: Option<String>,
    noise: Option<NoiseConfig>,
    idle_after_2q: Option<f64>,
    strip_idles: Option<bool>,
    configuration_error: Option<String>,
}

impl DetectorErrorModel {
    /// Configure a DEM from a circuit, bound probes, or a runtime adapter.
    #[must_use]
    pub fn builder() -> DemBuildBuilder {
        DemBuildBuilder::default()
    }
}

impl DemBuildBuilder {
    #[must_use]
    pub fn with_circuit(mut self, circuit: TickCircuit) -> Self {
        if self.circuit.is_some() {
            self.configuration_error = Some("with_circuit may only be called once".into());
        }
        self.circuit = Some(circuit);
        self
    }
    /// Supply a runtime adapter. The adapter owns its selected runtime.
    #[must_use]
    pub fn with_program(mut self, program: impl DemProgram + 'static) -> Self {
        if self.program.is_some() {
            self.configuration_error = Some("with_program may only be called once".into());
        }
        self.program = Some(Box::new(program));
        self
    }
    #[must_use]
    pub fn with_qubits(mut self, qubits: usize) -> Self {
        self.qubits = Some(qubits);
        self
    }
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
    #[must_use]
    pub fn with_inference(mut self, settings: DemInferenceSettings) -> Self {
        if self.inference.is_some() {
            self.configuration_error = Some("with_inference may only be called once".into());
        }
        self.inference = Some(settings);
        self
    }
    /// Supply already-bound probes alongside `with_circuit`. This is a trusted
    /// boundary: the caller establishes provenance and a fixed quantum schedule.
    #[must_use]
    pub fn with_inference_data(mut self, data: DemInferenceData) -> Self {
        self.probes = Some(data);
        self
    }
    #[must_use]
    pub fn with_detectors_json(mut self, json: impl Into<String>) -> Self {
        self.detectors = Some(json.into());
        self
    }
    #[must_use]
    pub fn with_observables_json(mut self, json: impl Into<String>) -> Self {
        self.observables = Some(json.into());
        self
    }
    /// Supply detector supports as stable measurement IDs; IDs are assigned in list order.
    #[must_use]
    pub fn with_detectors(self, supports: Vec<Vec<usize>>) -> Self {
        self.with_detectors_json(supports_json(supports))
    }
    /// Supply observable supports as stable measurement IDs; IDs are assigned in list order.
    #[must_use]
    pub fn with_observables(self, supports: Vec<Vec<usize>>) -> Self {
        self.with_observables_json(supports_json(supports))
    }
    #[must_use]
    pub fn with_noise(mut self, noise: NoiseConfig) -> Self {
        self.noise = Some(noise);
        self
    }
    #[must_use]
    pub fn with_idle_after_2q(mut self, duration: f64) -> Self {
        self.idle_after_2q = Some(duration);
        self
    }
    #[must_use]
    pub fn with_strip_traced_idles(mut self, strip: bool) -> Self {
        self.strip_idles = Some(strip);
        self
    }

    /// Resolve annotations, apply circuit passes and build through the existing
    /// native DEM pipeline. Already supplied circuits retain their gate schedule;
    /// runtime adapters are responsible for normalizing their lowered traces.
    ///
    /// # Errors
    /// Rejects conflicting inputs, invalid settings/metadata, failed inference,
    /// and unsupported circuits or noise models.
    pub fn build(mut self) -> Result<DemBuild, DemBuildError> {
        if let Some(error) = self.configuration_error {
            return Err(DemBuildError::Configuration(error));
        }
        if self.inference.is_some() && (self.detectors.is_some() || self.observables.is_some()) {
            return Err(DemBuildError::Configuration(
                "inference cannot be combined with explicit detector/observable definitions".into(),
            ));
        }
        if self.probes.is_some() && (self.inference.is_none() || self.program.is_some()) {
            return Err(DemBuildError::Configuration(
                "supplied probes require with_circuit and with_inference".into(),
            ));
        }
        if let Some(settings) = &self.inference {
            settings.validate()?;
        }
        if self
            .idle_after_2q
            .is_some_and(|duration| !duration.is_finite() || duration <= 0.0)
        {
            return Err(DemBuildError::Configuration(
                "idle_after_2q must be a finite positive duration".into(),
            ));
        }
        let input = match (self.circuit, self.program.as_mut()) {
            (Some(circuit), None) => DemBuildInput {
                circuit,
                probes: self.probes,
            },
            (None, Some(program)) => program.prepare(&DemProgramRequest {
                num_qubits: self.qubits.filter(|&n| n > 0).ok_or_else(|| {
                    DemBuildError::Configuration(
                        "with_program requires a positive with_qubits".into(),
                    )
                })?,
                seed: self.seed,
                inference: self.inference.as_ref(),
            })?,
            _ => {
                return Err(DemBuildError::Configuration(
                    "supply exactly one of with_program or with_circuit".into(),
                ));
            }
        };
        let mut circuit = input.circuit;
        // Inference IDs must already exist before any IDs can be assigned.
        let original_ids = tick_circuit_emission(&circuit);
        if self.inference.is_some() && original_ids.iter().any(Option::is_none) {
            return Err(DemBuildError::Configuration(
                "inference requires stable IDs on every circuit measurement".into(),
            ));
        }
        pass::assign_missing_meas_ids(&mut circuit);
        let measurement_ids: Vec<_> = tick_circuit_emission(&circuit)
            .into_iter()
            .flatten()
            .map(pecos_core::MeasId::index)
            .collect();
        if measurement_ids.iter().collect::<BTreeSet<_>>().len() != measurement_ids.len() {
            return Err(DemBuildError::Configuration(
                "circuit measurement IDs must be unique".into(),
            ));
        }
        let report = if let Some(settings) = &self.inference {
            if circuit.get_meta("detectors").is_some()
                || circuit.get_meta("observables").is_some()
                || !circuit.annotations().is_empty()
            {
                return Err(DemBuildError::Configuration(
                    "inference requires a circuit without existing detector/observable annotations"
                        .into(),
                ));
            }
            let data = input.probes.ok_or_else(|| {
                DemBuildError::Configuration("inference requires bound probe data".into())
            })?;
            if data.raw_measurement_ids.iter().collect::<BTreeSet<_>>()
                != measurement_ids.iter().collect::<BTreeSet<_>>()
            {
                return Err(DemBuildError::Configuration(
                    "probe IDs must cover every circuit measurement exactly once".into(),
                ));
            }
            let report = settings.infer(&data)?;
            circuit.set_meta(
                "detectors",
                Attribute::String(supports_json(&report.detector_supports)),
            );
            circuit.set_meta(
                "observables",
                Attribute::String(supports_json(&report.observable_supports)),
            );
            Some(report)
        } else {
            if input.probes.is_some() {
                return Err(DemBuildError::Configuration(
                    "unexpected probe data without inference settings".into(),
                ));
            }
            if let Some(json) = self.detectors {
                circuit.set_meta("detectors", Attribute::String(json));
            }
            if let Some(json) = self.observables {
                circuit.set_meta("observables", Attribute::String(json));
            }
            None
        };
        if self.strip_idles.unwrap_or(self.idle_after_2q.is_some()) {
            pass::strip_idles(&mut circuit);
        }
        if let Some(duration) = self.idle_after_2q {
            circuit.insert_idle_after_two_qubit_gates(duration);
        }
        let noise = self
            .noise
            .unwrap_or_else(|| NoiseConfig::new(0.001, 0.01, 0.001, 0.001));
        if (noise.uses_dedicated_idle_noise() || noise.idle_rz != 0.0)
            && !circuit
                .iter_gate_batches()
                .any(|batch| batch.gate_type == pecos_core::gate_type::GateType::Idle)
        {
            return Err(DemBuildError::Configuration(
                "idle-noise parameters have no idle gates to attach to".into(),
            ));
        }
        let definitions = definitions_from_tick_circuit(&circuit)?;
        let dem = DemBuilder::try_from_tick_circuit_with_noise_config(&circuit, noise)?;
        Ok(DemBuild {
            dem,
            circuit,
            definitions,
            measurement_ids,
            inference: report,
            raw_tag: self.inference.map(|s| s.raw_tag),
        })
    }
}

fn supports_json(supports: impl IntoIterator<Item = impl AsRef<[usize]>>) -> String {
    serde_json::Value::Array(
        supports
            .into_iter()
            .enumerate()
            .map(|(id, support)| serde_json::json!({"id": id, "meas_ids": support.as_ref()}))
            .collect(),
    )
    .to_string()
}
