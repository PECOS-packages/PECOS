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

/// A selected element in a tag's flattened per-shot output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DemOutputSelector {
    pub tag: String,
    pub element: usize,
}

/// How a single detector or observable obtains its measurement parity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DemDefinitionSource {
    Measurements(Vec<usize>),
    Inferred(DemOutputSelector),
}

/// A definition with an explicit ID, independent of its preparation method.
#[derive(Clone, Debug, PartialEq)]
pub struct DemDefinition {
    pub id: usize,
    pub source: DemDefinitionSource,
    /// Optional detector coordinates (x, y, t). Rejected for observables.
    pub coords: Option<[f64; 3]>,
    /// Additional JSON fields preserved in the circuit schema.
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

impl DemDefinition {
    #[must_use]
    pub fn explicit(id: usize, measurement_ids: Vec<usize>) -> Self {
        Self {
            id,
            source: DemDefinitionSource::Measurements(measurement_ids),
            coords: None,
            metadata: serde_json::Map::new(),
        }
    }
    #[must_use]
    pub fn inferred(id: usize, tag: impl Into<String>, element: usize) -> Self {
        Self {
            id,
            source: DemDefinitionSource::Inferred(DemOutputSelector {
                tag: tag.into(),
                element,
            }),
            coords: None,
            metadata: serde_json::Map::new(),
        }
    }

    /// Set detector coordinates (x, y, t); all values must be finite.
    #[must_use]
    pub fn with_coords(mut self, coords: [f64; 3]) -> Self {
        self.coords = Some(coords);
        self
    }

    /// Add or replace a custom JSON field, such as `label` or `reference`.
    ///
    /// Identity, support, coordinates and inference provenance fields are
    /// reserved and rejected at build time, before runtime preparation.
    #[must_use]
    pub fn with_metadata(
        mut self,
        key: impl Into<String>,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Evidence used to obtain a definition's measurement parity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DemAnnotationOrigin {
    Explicit,
    Inferred(DemOutputSelector),
    /// Column in the concatenation of these default tags. Probe data without
    /// resolved selectors has no per-tag widths, so its exact selector is unknown.
    InferredDefaults {
        tags: Vec<String>,
        column: usize,
    },
}

/// Per-definition origin. Inferred definitions share the build's inference evidence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DemDefinitionOrigin {
    pub role: &'static str,
    pub id: usize,
    pub source: DemAnnotationOrigin,
}

/// Input prepared by a caller or runtime adapter. Probe IDs must be bound to
/// this circuit's measurements; the builder checks complete identity coverage.
pub struct DemBuildInput {
    pub circuit: TickCircuit,
    pub probes: Option<DemInferenceData>,
    /// Resolved detector columns, when the adapter knows the per-tag widths.
    pub resolved_detector_outputs: Option<Vec<DemOutputSelector>>,
    /// Resolved observable columns, when the adapter knows the per-tag widths.
    pub resolved_observable_outputs: Option<Vec<DemOutputSelector>>,
}

/// Execution options passed to a runtime adapter before building a DEM.
pub struct DemProgramRequest<'a> {
    pub num_qubits: usize,
    pub seed: u64,
    pub inference: Option<&'a DemInferenceSettings>,
    /// None requests all outputs under the settings' default detector tag.
    /// Some requests exactly these columns, in order; an empty slice skips this role.
    pub detector_outputs: Option<&'a [DemOutputSelector]>,
    /// None requests all outputs under the settings' default observable tags.
    pub observable_outputs: Option<&'a [DemOutputSelector]>,
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
    pub definition_origins: Vec<DemDefinitionOrigin>,
    raw_tag: Option<String>,
}

impl DemBuild {
    /// Evaluate a physical measurement record in circuit emission order.
    /// These are measurement parities; no unknown noiseless references are invented.
    ///
    /// # Errors
    /// Rejects invalid bits and wrong widths. Tracked Paulis are not readout definitions.
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

/// Fluent construction from explicit, inferred, or mixed definitions.
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
    detector_definitions: Option<Vec<DemDefinition>>,
    observable_definitions: Option<Vec<DemDefinition>>,
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
        if self.detectors.is_some() || self.detector_definitions.is_some() {
            self.configuration_error =
                Some("detector definitions may only be supplied once".into());
        }
        self.detectors = Some(json.into());
        self
    }
    #[must_use]
    pub fn with_observables_json(mut self, json: impl Into<String>) -> Self {
        if self.observables.is_some() || self.observable_definitions.is_some() {
            self.configuration_error =
                Some("observable definitions may only be supplied once".into());
        }
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
    /// Supply explicit and inferred detector definitions in one list.
    #[must_use]
    pub fn with_detector_definitions(mut self, definitions: Vec<DemDefinition>) -> Self {
        if self.detectors.is_some() || self.detector_definitions.is_some() {
            self.configuration_error =
                Some("detector definitions may only be supplied once".into());
        }
        self.detector_definitions = Some(definitions);
        self
    }
    /// Supply explicit and inferred observable definitions in one list.
    #[must_use]
    pub fn with_observable_definitions(mut self, definitions: Vec<DemDefinition>) -> Self {
        if self.observables.is_some() || self.observable_definitions.is_some() {
            self.configuration_error =
                Some("observable definitions may only be supplied once".into());
        }
        self.observable_definitions = Some(definitions);
        self
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
        let detector_outputs = requested_outputs(
            self.detector_definitions.as_deref(),
            self.detectors.is_some(),
            self.inference.is_some(),
            true,
        )?;
        let observable_outputs = requested_outputs(
            self.observable_definitions.as_deref(),
            self.observables.is_some(),
            self.inference.is_some(),
            false,
        )?;
        let needs_inference = [&detector_outputs, &observable_outputs]
            .iter()
            .any(|outputs| outputs.as_ref().is_none_or(|items| !items.is_empty()));
        let settings = needs_inference.then(|| self.inference.clone().unwrap_or_default());
        if self.probes.is_some() && (!needs_inference || self.program.is_some()) {
            return Err(DemBuildError::Configuration(
                "supplied probes require with_circuit and at least one inferred definition".into(),
            ));
        }
        if let Some(settings) = &settings {
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
                resolved_detector_outputs: None,
                resolved_observable_outputs: None,
            },
            (None, Some(program)) => program.prepare(&DemProgramRequest {
                num_qubits: self.qubits.filter(|&n| n > 0).ok_or_else(|| {
                    DemBuildError::Configuration(
                        "with_program requires a positive with_qubits".into(),
                    )
                })?,
                seed: self.seed,
                inference: settings.as_ref(),
                detector_outputs: detector_outputs.as_deref(),
                observable_outputs: observable_outputs.as_deref(),
            })?,
            _ => {
                return Err(DemBuildError::Configuration(
                    "supply exactly one of with_program or with_circuit".into(),
                ));
            }
        };
        let mut circuit = input.circuit;
        let sets_definitions = needs_inference
            || self.detectors.is_some()
            || self.observables.is_some()
            || self.detector_definitions.is_some()
            || self.observable_definitions.is_some();
        if sets_definitions {
            let mut existing = Vec::new();
            for key in ["detectors", "observables"] {
                if circuit.get_meta(key).is_some() {
                    existing.push(format!("{key} metadata"));
                }
            }
            for annotation in circuit.annotations() {
                match annotation.kind {
                    pecos_quantum::AnnotationKind::Detector { .. } => {
                        existing.push("Detector annotation".into());
                    }
                    pecos_quantum::AnnotationKind::Observable { .. } => {
                        existing.push("Observable annotation".into());
                    }
                    pecos_quantum::AnnotationKind::TrackedPauli { .. } => {}
                }
            }
            if !existing.is_empty() {
                return Err(DemBuildError::Configuration(format!(
                    "circuit already carries {}; supply definitions on either the circuit or builder",
                    existing.join(", ")
                )));
            }
        }
        // Inference IDs must already exist before any IDs can be assigned.
        let original_ids = tick_circuit_emission(&circuit);
        if needs_inference && original_ids.iter().any(Option::is_none) {
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
        let report = if let Some(settings) = &settings {
            let mut data = input.probes.ok_or_else(|| {
                DemBuildError::Configuration("inference requires bound probe data".into())
            })?;
            if data.raw_measurement_ids.iter().collect::<BTreeSet<_>>()
                != measurement_ids.iter().collect::<BTreeSet<_>>()
            {
                return Err(DemBuildError::Configuration(
                    "probe IDs must cover every circuit measurement exactly once".into(),
                ));
            }
            select_probe_columns(
                &mut data.detector_rows,
                detector_outputs.as_deref(),
                data.raw_rows.len(),
            )?;
            select_probe_columns(
                &mut data.observable_rows,
                observable_outputs.as_deref(),
                data.raw_rows.len(),
            )?;
            Some(settings.infer(&data)?)
        } else {
            if input.probes.is_some() {
                return Err(DemBuildError::Configuration(
                    "unexpected probe data without inferred definitions".into(),
                ));
            }
            None
        };
        let mut definition_origins = Vec::new();
        for (role, specs, json, outputs, supports, default_tags, resolved_outputs) in [
            (
                "detector",
                &self.detector_definitions,
                &self.detectors,
                &detector_outputs,
                report
                    .as_ref()
                    .map(|r| r.detector_supports.as_slice())
                    .unwrap_or_default(),
                settings
                    .as_ref()
                    .map(|s| vec![s.detector_tag.clone()])
                    .unwrap_or_default(),
                input.resolved_detector_outputs.as_deref(),
            ),
            (
                "observable",
                &self.observable_definitions,
                &self.observables,
                &observable_outputs,
                report
                    .as_ref()
                    .map(|r| r.observable_supports.as_slice())
                    .unwrap_or_default(),
                settings
                    .as_ref()
                    .map(|s| s.observable_tags.clone())
                    .unwrap_or_default(),
                input.resolved_observable_outputs.as_deref(),
            ),
        ] {
            let resolved = resolve_definitions(
                role,
                specs.as_deref(),
                json.as_deref(),
                outputs.as_deref(),
                supports,
                DefaultOutputs {
                    tags: &default_tags,
                    resolved: resolved_outputs,
                },
                &mut definition_origins,
            )?;
            if let Some(json) = resolved {
                circuit.set_meta(
                    if role == "detector" {
                        "detectors"
                    } else {
                        "observables"
                    },
                    Attribute::String(json),
                );
            }
        }
        if self.strip_idles.unwrap_or(self.idle_after_2q.is_some()) {
            pass::strip_idles(&mut circuit);
        }
        if let Some(duration) = self.idle_after_2q {
            circuit.insert_idle_after_two_qubit_gates(duration);
        }
        let noise = self
            .noise
            .unwrap_or_else(|| NoiseConfig::new(0.001, 0.01, 0.001, 0.001));
        let definitions = definitions_from_tick_circuit(&circuit)?;
        let detector_ids: Vec<_> = definitions
            .detectors
            .iter()
            .map(|d| d.id as usize)
            .collect();
        let unique: BTreeSet<_> = detector_ids.iter().copied().collect();
        if unique.len() != detector_ids.len() || unique.into_iter().ne(0..detector_ids.len()) {
            return Err(DemBuildError::Configuration(format!(
                "detector IDs {detector_ids:?} must be unique and dense from zero; renumber the definitions"
            )));
        }
        for (role, ids) in [
            (
                "detector",
                definitions
                    .detectors
                    .iter()
                    .map(|d| d.id as usize)
                    .collect::<Vec<_>>(),
            ),
            (
                "observable",
                definitions
                    .observables
                    .iter()
                    .map(|d| d.id as usize)
                    .collect::<Vec<_>>(),
            ),
        ] {
            for id in ids {
                if !definition_origins
                    .iter()
                    .any(|origin| origin.role == role && origin.id == id)
                {
                    definition_origins.push(DemDefinitionOrigin {
                        role,
                        id,
                        source: DemAnnotationOrigin::Explicit,
                    });
                }
            }
        }
        let dem = DemBuilder::try_from_tick_circuit_with_noise_config(&circuit, noise)?;
        Ok(DemBuild {
            dem,
            circuit,
            definitions,
            measurement_ids,
            inference: report,
            definition_origins,
            raw_tag: settings.map(|s| s.raw_tag),
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

/// None means the shorthand all-output request; Some(empty) means no inference.
fn requested_outputs(
    definitions: Option<&[DemDefinition]>,
    json_supplied: bool,
    shorthand: bool,
    detectors: bool,
) -> Result<Option<Vec<DemOutputSelector>>, DemBuildError> {
    let Some(definitions) = definitions else {
        return Ok(if shorthand && !json_supplied {
            None
        } else {
            Some(Vec::new())
        });
    };
    let ids: BTreeSet<_> = definitions.iter().map(|d| d.id).collect();
    // Fail before runtime preparation; the build rechecks every source.
    if ids.len() != definitions.len() || (detectors && ids.iter().copied().ne(0..ids.len())) {
        return Err(DemBuildError::Configuration(format!(
            "definition IDs must be unique, and detector IDs dense from zero; got {:?}",
            definitions.iter().map(|d| d.id).collect::<Vec<_>>()
        )));
    }
    let mut outputs = Vec::new();
    for definition in definitions {
        if let Some(coords) = definition.coords
            && (!detectors || coords.iter().any(|value| !value.is_finite()))
        {
            return Err(DemBuildError::Configuration(
                "coordinates require a detector and three finite values".into(),
            ));
        }
        for key in [
            "id",
            "records",
            "meas_ids",
            "result_tags",
            "coords",
            "inferred_from_result_tag",
            "result_element",
        ] {
            if definition.metadata.contains_key(key) {
                return Err(DemBuildError::Configuration(format!(
                    "definition metadata uses reserved key: {key}"
                )));
            }
        }
        if let DemDefinitionSource::Inferred(output) = &definition.source {
            if output.tag.is_empty() {
                return Err(DemBuildError::Configuration(
                    "inferred output tags must be nonempty".into(),
                ));
            }
            outputs.push(output.clone());
        }
    }
    Ok(Some(outputs))
}

fn select_probe_columns(
    rows: &mut Vec<Vec<u8>>,
    outputs: Option<&[DemOutputSelector]>,
    shots: usize,
) -> Result<(), DemBuildError> {
    if let Some(outputs) = outputs {
        if outputs.is_empty() {
            *rows = vec![Vec::new(); shots];
        } else if rows.iter().any(|row| row.len() != outputs.len()) {
            return Err(DemBuildError::Configuration(
                "probe columns must match requested inferred outputs in order".into(),
            ));
        }
    }
    Ok(())
}

/// A role's default-tag expansion, with the adapter's resolved columns when it knows them.
#[derive(Clone, Copy)]
struct DefaultOutputs<'a> {
    tags: &'a [String],
    resolved: Option<&'a [DemOutputSelector]>,
}

fn resolve_definitions(
    role: &'static str,
    definitions: Option<&[DemDefinition]>,
    json: Option<&str>,
    outputs: Option<&[DemOutputSelector]>,
    supports: &[Vec<usize>],
    defaults: DefaultOutputs<'_>,
    origins: &mut Vec<DemDefinitionOrigin>,
) -> Result<Option<String>, DemBuildError> {
    if let Some(definitions) = definitions {
        let mut inferred = supports.iter();
        let mut entries = Vec::new();
        for definition in definitions {
            let (ids, output) = match &definition.source {
                DemDefinitionSource::Measurements(ids) => (ids, None),
                DemDefinitionSource::Inferred(output) => (
                    inferred.next().ok_or_else(|| {
                        DemBuildError::Configuration("missing inferred definition".into())
                    })?,
                    Some(output.clone()),
                ),
            };
            let mut entry = definition.metadata.clone();
            entry.insert("id".into(), serde_json::json!(definition.id));
            entry.insert("meas_ids".into(), serde_json::json!(ids));
            if let Some(coords) = definition.coords {
                entry.insert("coords".into(), serde_json::json!(coords));
            }
            if let Some(output) = &output {
                entry.insert(
                    "inferred_from_result_tag".into(),
                    serde_json::json!(output.tag),
                );
                entry.insert("result_element".into(), serde_json::json!(output.element));
            }
            origins.push(DemDefinitionOrigin {
                role,
                id: definition.id,
                source: output.map_or(DemAnnotationOrigin::Explicit, DemAnnotationOrigin::Inferred),
            });
            entries.push(serde_json::Value::Object(entry));
        }
        return Ok(Some(serde_json::Value::Array(entries).to_string()));
    }
    if outputs.is_none() {
        if let Some(selectors) = defaults.resolved {
            if selectors.len() != supports.len() {
                return Err(DemBuildError::Configuration(format!(
                    "resolved {role} selector count {} differs from inferred support count {}",
                    selectors.len(),
                    supports.len(),
                )));
            }
            let mut entries = Vec::new();
            for (id, (support, selector)) in supports.iter().zip(selectors).enumerate() {
                origins.push(DemDefinitionOrigin {
                    role,
                    id,
                    source: DemAnnotationOrigin::Inferred(selector.clone()),
                });
                let mut entry = serde_json::json!({
                    "id": id,
                    "meas_ids": support,
                    "inferred_from_result_tag": selector.tag,
                });
                if role == "observable" {
                    entry["result_element"] = serde_json::json!(selector.element);
                }
                entries.push(entry);
            }
            return Ok(Some(serde_json::Value::Array(entries).to_string()));
        }
        // Probes without resolved selectors do not carry per-tag widths.
        for id in 0..supports.len() {
            origins.push(DemDefinitionOrigin {
                role,
                id,
                source: DemAnnotationOrigin::InferredDefaults {
                    tags: defaults.tags.to_vec(),
                    column: id,
                },
            });
        }
        return Ok(Some(supports_json(supports)));
    }
    if let Some(json) = json {
        let entries: Vec<serde_json::Value> =
            serde_json::from_str(json).map_err(|e| DemBuildError::Configuration(e.to_string()))?;
        for entry in entries {
            if let Some(id) = entry.get("id").and_then(serde_json::Value::as_u64) {
                origins.push(DemDefinitionOrigin {
                    role,
                    id: usize::try_from(id)
                        .map_err(|e| DemBuildError::Configuration(e.to_string()))?,
                    source: DemAnnotationOrigin::Explicit,
                });
            }
        }
        return Ok(Some(json.to_string()));
    }
    Ok(None)
}
