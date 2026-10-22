// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

//! High-level DEM construction from circuits or runtime adapters.
//!
//! Runtime adapters own compilation, execution and schedule/provenance checks.
//! The QEC layer owns annotation inference, circuit preparation, fault
//! propagation and measurement evaluation, without depending on a runtime.

mod audit;
pub mod bindings;
mod certificate;
mod evaluation;
mod evidence;
mod fingerprint;
mod references;
pub mod specifications;

pub use audit::{
    DemBuildAudit, DemInferenceAudit, NoiseResidualSummary, summarize_noise_residuals,
};
#[cfg(feature = "selene")]
pub(crate) use evidence::require_static_schedule;
pub use evidence::{DemExecutionEvidence, DemScheduleEvidence, GeneratorLayout};
pub use fingerprint::schema_fingerprint;
pub use references::{NoiselessReferences, noiseless_references};
pub use specifications::{MeasurementRef, ResultRef};

pub use certificate::{check_generator_certificate_digest, generator_certificate_digest};

use crate::dem_inference::{
    DemInferenceData, DemInferenceError, DemInferenceReport, DemInferenceSettings,
};
use crate::fault_tolerance::circuit_definitions::{
    CircuitDefinitions, DefinitionError, definitions_from_tick_circuit, tick_circuit_emission,
};
use crate::fault_tolerance::dem_builder::NoiseChannelResidual;
use crate::{DemBuilder, DemBuilderError, DetectorErrorModel, NoiseConfig};
use pecos_quantum::{Attribute, TickCircuit, pass};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// A selected element in a tag's flattened per-shot output.
/// Python checks string tags and non-boolean, non-negative integer elements
/// before converting its selectors to this value type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DemOutputSelector {
    pub tag: String,
    pub element: usize,
}

/// How a single detector or observable obtains its measurement parity.
/// Python checks reference object shapes and non-boolean, non-negative integer
/// measurement IDs before conversion; inferred selectors have their own checks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DemDefinitionSource {
    Measurements(Vec<usize>),
    /// XOR-normalized typed references; cancelled support is rejected.
    References(Vec<MeasurementRef>),
    Inferred(DemOutputSelector),
}

/// A definition independent of its preparation method; an omitted ID uses list order.
/// Python checks non-boolean, non-negative integer IDs, coordinate numeric
/// conversion and metadata JSON-serializability before constructing this value.
#[derive(Clone, Debug, PartialEq)]
pub struct DemDefinition {
    pub id: Option<usize>,
    pub source: DemDefinitionSource,
    /// Optional detector coordinates. Rejected for observables.
    pub coords: Option<Vec<f64>>,
    /// Additional JSON fields preserved in the circuit schema.
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

impl DemDefinition {
    #[must_use]
    pub fn explicit(id: usize, measurement_ids: Vec<usize>) -> Self {
        Self {
            id: Some(id),
            source: DemDefinitionSource::Measurements(measurement_ids),
            coords: None,
            metadata: serde_json::Map::new(),
        }
    }
    #[must_use]
    pub fn inferred(id: usize, tag: impl Into<String>, element: usize) -> Self {
        Self {
            id: Some(id),
            source: DemDefinitionSource::Inferred(DemOutputSelector {
                tag: tag.into(),
                element,
            }),
            coords: None,
            metadata: serde_json::Map::new(),
        }
    }

    /// Construct a parity from typed measurement references.
    /// # Errors
    /// Rejects an empty reference list.
    pub fn references(
        id: usize,
        references: Vec<MeasurementRef>,
    ) -> Result<Self, bindings::DemBindingError> {
        Ok(Self {
            id: Some(id),
            source: DemDefinitionSource::references(references)?,
            coords: None,
            metadata: serde_json::Map::new(),
        })
    }

    /// Set detector coordinates; all values must be finite.
    #[must_use]
    pub fn with_coords(mut self, coords: impl Into<Vec<f64>>) -> Self {
        self.coords = Some(coords.into());
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
#[derive(Default)]
pub struct DemBuildInput {
    pub circuit: TickCircuit,
    /// Adapter evidence; a successful inference solve adds no schedule evidence.
    pub evidence: DemExecutionEvidence,
    /// Runtime result calls, preserving their widths and encounter order.
    /// Without compiler result-tag analysis or a generator layout, a custom adapter
    /// owns certification of these bindings. Selene supplies compiler analysis
    /// when static evidence is required and never trusts read timing.
    pub result_traces: Vec<bindings::ResultTrace>,
    /// Source measurement order before replay compaction, required for certification.
    pub source_measurement_ids: Vec<usize>,
    /// Compiler result-tag analysis of the exact executed QIS, including certification failures.
    pub result_tag_analysis:
        Option<Result<bindings::ResultTagMeasurements, bindings::DemBindingError>>,
    /// Program-bound generator certificate, verified before use.
    pub generator_layout: Option<GeneratorLayout>,
    pub probes: Option<DemInferenceData>,
    /// Resolved detector columns, when the adapter knows the per-tag widths.
    pub resolved_detector_outputs: Option<Vec<DemOutputSelector>>,
    /// Resolved observable columns, when the adapter knows the per-tag widths.
    pub resolved_observable_outputs: Option<Vec<DemOutputSelector>>,
}

/// Execution options passed to a runtime adapter before building a DEM.
pub struct DemProgramRequest<'a> {
    /// Resolved policy, independent of whether inference settings were supplied.
    pub requires_static_schedule: bool,
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
    /// Report static schedule evidence without executing the program.
    /// Custom adapters must report the same source in `DemBuildInput::evidence`.
    /// # Errors
    /// Rejects invalid certificates or unsupported program structure.
    fn static_schedule_evidence(&self) -> Result<Option<String>, DemBuildError> {
        Ok(None)
    }

    /// Explain missing static evidence for this adapter's input route.
    fn missing_static_schedule_error(&self) -> DemBuildError {
        DemBuildError::Preparation(
            "DEM program requires static schedule evidence; the adapter must report a nonempty static schedule source before execution".into(),
        )
    }

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

/// A DEM built from a prepared measurement schema.
pub struct DemBuild {
    prepared: PreparedDem,
    dem: DetectorErrorModel,
}

impl DemBuild {
    #[must_use]
    pub fn prepared(&self) -> &PreparedDem {
        &self.prepared
    }

    #[must_use]
    pub fn dem(&self) -> &DetectorErrorModel {
        &self.dem
    }

    #[must_use]
    pub fn idle_noise_residuals(&self) -> &[NoiseChannelResidual] {
        self.dem.idle_noise_residuals()
    }

    /// Summarize this DEM's residuals above the relative threshold.
    /// # Errors
    /// Rejects non-finite thresholds and thresholds outside [0, 1].
    pub fn residual_summary(
        &self,
        relative_threshold: f64,
    ) -> Result<Vec<NoiseResidualSummary>, DemBuildError> {
        summarize_noise_residuals(self.idle_noise_residuals(), relative_threshold)
    }
}

/// Noise-free circuit and schema, finalized together before DEM construction.
#[derive(Clone)]
pub struct PreparedDem {
    inference: Option<DemInferenceReport>,
    evidence: DemExecutionEvidence,
    generator_layout: bool,
    requires_static_schedule: bool,
    /// Detectors first, then observables, with ascending IDs within each role.
    definition_origins: Vec<DemDefinitionOrigin>,
    schema: Arc<DemBuildSchema>,
}

/// Finalized together so references, evaluators, audit data and digest agree.
struct DemBuildSchema {
    circuit: TickCircuit,
    definitions: CircuitDefinitions,
    measurement_ids: Vec<usize>,
    named_result_measurements: BTreeMap<String, Vec<Option<usize>>>,
    schema_fingerprint: String,
    measurement_ledger: Vec<bindings::MeasurementLedgerEntry>,
}

impl DemBuildSchema {
    fn new(
        mut circuit: TickCircuit,
        mut definitions: CircuitDefinitions,
        measurement_ids: Vec<usize>,
        traces: &[bindings::ResultTrace],
        raw_array_tag: Option<&str>,
    ) -> Result<Self, DemBuildError> {
        let references = noiseless_references(&circuit, &definitions)?;
        let (detectors, observables) = references::write_references(
            &mut circuit,
            &mut definitions,
            &references,
            &measurement_ids,
        )?;
        let index = bindings::ResultTraceIndex::new(&measurement_ids, traces)
            .map_err(|error| DemBuildError::Configuration(error.to_string()))?;
        let named_result_measurements = index.named_bindings(raw_array_tag);
        let schema_fingerprint = schema_fingerprint(
            &detectors,
            &observables,
            &measurement_ids,
            &named_result_measurements,
        )?;
        circuit.set_meta(
            "dem_schema_fingerprint",
            Attribute::String(schema_fingerprint.clone()),
        );
        Ok(Self {
            circuit,
            definitions,
            measurement_ids,
            named_result_measurements,
            schema_fingerprint,
            measurement_ledger: index.ledger(),
        })
    }
}

impl PreparedDem {
    /// Build with noise using a separate copy of the prepared circuit.
    /// # Errors
    /// Rejects idle noise without idle gates and unsupported noise models.
    pub fn build_dem(&self, noise: NoiseConfig) -> Result<DemBuild, DemBuildError> {
        // The DEM builder only reads the circuit, so the prepared schema stays shared.
        let dem = DemBuilder::try_from_tick_circuit_with_noise_config(&self.schema.circuit, noise)?;
        Ok(DemBuild {
            prepared: self.clone(),
            dem,
        })
    }

    #[must_use]
    pub fn inference(&self) -> Option<&DemInferenceReport> {
        self.inference.as_ref()
    }

    #[must_use]
    pub fn definition_origins(&self) -> &[DemDefinitionOrigin] {
        &self.definition_origins
    }

    /// The final circuit, including computed definition references and schema digest.
    #[must_use]
    pub fn circuit(&self) -> &TickCircuit {
        &self.schema.circuit
    }

    /// Resolved definitions, the source of truth for noiseless references.
    #[must_use]
    pub fn definitions(&self) -> &CircuitDefinitions {
        &self.schema.definitions
    }

    /// Stable measurement IDs in runtime emission order.
    #[must_use]
    pub fn measurement_ids(&self) -> &[usize] {
        &self.schema.measurement_ids
    }

    /// A snapshot of the exact references stored in the resolved definitions.
    #[must_use]
    pub fn references(&self) -> NoiselessReferences {
        NoiselessReferences {
            detectors: self
                .schema
                .definitions
                .detectors
                .iter()
                .filter_map(|d| d.reference.map(|reference| (d.id as usize, reference)))
                .collect(),
            observables: self
                .schema
                .definitions
                .observables
                .iter()
                .filter_map(|d| d.reference.map(|reference| (d.id as usize, reference)))
                .collect(),
        }
    }

    /// Flattened named result slots, with None for unsupported provenance.
    #[must_use]
    pub fn named_result_measurements(&self) -> &BTreeMap<String, Vec<Option<usize>>> {
        &self.schema.named_result_measurements
    }

    /// Version-2 canonical schema digest, also stored on the circuit.
    #[must_use]
    pub fn schema_fingerprint(&self) -> &str {
        &self.schema.schema_fingerprint
    }

    /// Measurement identity and result provenance in runtime emission order.
    #[must_use]
    pub fn measurement_ledger(&self) -> &[bindings::MeasurementLedgerEntry] {
        &self.schema.measurement_ledger
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
    /// Supply direct, reference-based and inferred detector definitions in one list.
    #[must_use]
    pub fn with_detector_definitions(mut self, definitions: Vec<DemDefinition>) -> Self {
        if self.detectors.is_some() || self.detector_definitions.is_some() {
            self.configuration_error =
                Some("detector definitions may only be supplied once".into());
        }
        self.detector_definitions = Some(definitions);
        self
    }
    /// Supply direct, reference-based and inferred observable definitions in one list.
    #[must_use]
    pub fn with_observable_definitions(mut self, definitions: Vec<DemDefinition>) -> Self {
        if self.observables.is_some() || self.observable_definitions.is_some() {
            self.configuration_error =
                Some("observable definitions may only be supplied once".into());
        }
        self.observable_definitions = Some(definitions);
        self
    }
    /// Supply the noise configuration required by `build`.
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

    /// Prepare the measurement schema and build through the native DEM pipeline.
    /// # Errors
    /// Requires noise and rejects invalid preparation or unsupported noise models.
    pub fn build(mut self) -> Result<DemBuild, DemBuildError> {
        if let Some(error) = self.configuration_error.take() {
            return Err(DemBuildError::Configuration(error));
        }
        let noise = self.noise.take().ok_or_else(|| {
            DemBuildError::Configuration("supply a noise configuration with with_noise".into())
        })?;
        self.prepare()?.build_dem(noise)
    }

    /// Resolve annotations, apply circuit passes and finalize the noiseless schema.
    /// Supplied noise is ignored. Already supplied circuits retain their gate schedule;
    /// runtime adapters are responsible for normalizing their lowered traces.
    /// # Errors
    /// Rejects conflicting inputs, invalid settings/metadata, failed inference,
    /// and unsupported circuits or nondeterministic noiseless parities.
    pub fn prepare(mut self) -> Result<PreparedDem, DemBuildError> {
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
        let required_tags = self
            .detector_definitions
            .iter()
            .chain(&self.observable_definitions)
            .flatten()
            .filter_map(|definition| match &definition.source {
                DemDefinitionSource::References(refs) => Some(refs),
                _ => None,
            })
            .flatten()
            .filter_map(|reference| match reference {
                MeasurementRef::Result(result) => Some(result.tag().to_string()),
                MeasurementRef::Record(_) => None,
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let detector_json_needs_tags = json_uses_result_tags(self.detectors.as_deref())?;
        let observable_json_needs_tags = json_uses_result_tags(self.observables.as_deref())?;
        let json_needs_tags = detector_json_needs_tags || observable_json_needs_tags;
        let requires_static_schedule =
            !needs_inference || !required_tags.is_empty() || json_needs_tags;
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
            (Some(circuit), None) => {
                let result_traces = self
                    .probes
                    .as_ref()
                    .zip(settings.as_ref())
                    .map(|(data, settings)| bindings::ResultTrace {
                        name: settings.raw_tag.clone(),
                        occurrence: Some(0),
                        width: data.raw_measurement_ids.len(),
                        measurement_ids: Some(data.raw_measurement_ids.clone()),
                    })
                    .into_iter()
                    .collect();
                DemBuildInput {
                    circuit,
                    evidence: DemExecutionEvidence {
                        schedule: DemScheduleEvidence {
                            static_source: Some("fixed_circuit".into()),
                            ..Default::default()
                        },
                        raw_binding: self
                            .probes
                            .as_ref()
                            .map(|_| "caller_bound_measurement_ids".into()),
                        ..Default::default()
                    },
                    probes: self.probes,
                    result_traces,
                    ..Default::default()
                }
            }
            (None, Some(program)) => {
                let request = DemProgramRequest {
                    requires_static_schedule,
                    num_qubits: self.qubits.filter(|&n| n > 0).ok_or_else(|| {
                        DemBuildError::Configuration(
                            "with_program requires a positive with_qubits".into(),
                        )
                    })?,
                    seed: self.seed,
                    inference: settings.as_ref(),
                    detector_outputs: detector_outputs.as_deref(),
                    observable_outputs: observable_outputs.as_deref(),
                };
                let static_source = program.static_schedule_evidence()?;
                evidence::require_static_schedule(
                    program.as_ref(),
                    requires_static_schedule,
                    static_source.as_deref(),
                )?;
                let input = program.prepare(&request)?;
                if static_source.is_some() && input.evidence.schedule.static_source != static_source
                {
                    return Err(DemBuildError::Preparation(
                        "adapter static schedule evidence changed during preparation".into(),
                    ));
                }
                input
            }
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
        let generator_layout = requires_static_schedule && input.generator_layout.is_some();
        let traces = evidence::certify_traces(
            &input.result_traces,
            &input.source_measurement_ids,
            &measurement_ids,
            requires_static_schedule.then_some(evidence::TraceCertification {
                analysis: input.result_tag_analysis.as_ref(),
                generator: input.generator_layout.as_ref(),
            }),
            &required_tags,
            report.as_ref().zip(settings.as_ref()),
        )?;
        if json_needs_tags {
            let (tags, count) = input
                .result_tag_analysis
                .as_ref()
                .ok_or_else(|| {
                    DemBuildError::Configuration(evidence::JSON_CERTIFICATION_ERROR.into())
                })?
                .as_ref()
                .map_err(|error| match error {
                    bindings::DemBindingError::Certification(_) => {
                        DemBuildError::Configuration(evidence::JSON_CERTIFICATION_ERROR.into())
                    }
                    _ => DemBuildError::Configuration(error.to_string()),
                })?;
            let (detectors, observables) =
                crate::fault_tolerance::dem_builder::resolve_result_tags(
                    self.detectors.as_deref().unwrap_or("[]"),
                    self.observables.as_deref().unwrap_or("[]"),
                    tags,
                    *count,
                    &input.source_measurement_ids,
                    &measurement_ids,
                )?;
            if self.detectors.is_some() {
                self.detectors = Some(detectors);
            }
            if self.observables.is_some() {
                self.observables = Some(observables);
            }
        }
        let index = bindings::ResultTraceIndex::new(&measurement_ids, &traces)
            .map_err(|error| DemBuildError::Configuration(error.to_string()))?;
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
                    index: &index,
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
        let definitions = definitions_from_tick_circuit(&circuit).map_err(|error| match error {
            DefinitionError::Parse { .. } => DemBuildError::Configuration(error.to_string()),
            _ => DemBuildError::Definitions(error),
        })?;
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
        let mut recorded: BTreeSet<_> = definition_origins
            .iter()
            .map(|origin| (origin.role, origin.id))
            .collect();
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
                if recorded.insert((role, id)) {
                    definition_origins.push(DemDefinitionOrigin {
                        role,
                        id,
                        source: DemAnnotationOrigin::Explicit,
                    });
                }
            }
        }
        definition_origins.sort_by_key(|origin| (origin.role != "detector", origin.id));
        let schema = DemBuildSchema::new(
            circuit,
            definitions,
            measurement_ids,
            &traces,
            settings.as_ref().map(|s| s.raw_tag.as_str()),
        )?;
        Ok(PreparedDem {
            inference: report,
            evidence: input.evidence,
            generator_layout,
            requires_static_schedule,
            definition_origins,
            schema: Arc::new(schema),
        })
    }
}

fn json_uses_result_tags(json: Option<&str>) -> Result<bool, DemBuildError> {
    let Some(json) = json.filter(|json| !json.trim().is_empty()) else {
        return Ok(false);
    };
    let value: serde_json::Value = serde_json::from_str(json).map_err(|error| {
        DemBuildError::Configuration(format!("invalid detector/observable JSON: {error}"))
    })?;
    Ok(value.as_array().is_some_and(|entries| {
        entries.iter().any(|entry| {
            entry
                .as_object()
                .is_some_and(|entry| entry.contains_key("result_tags"))
        })
    }))
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
    let ids: BTreeSet<_> = definitions
        .iter()
        .enumerate()
        .map(|(position, d)| d.id.unwrap_or(position))
        .collect();
    // Fail before runtime preparation; the build rechecks every source.
    if ids.len() != definitions.len() || (detectors && ids.iter().copied().ne(0..ids.len())) {
        return Err(DemBuildError::Configuration(format!(
            "definition IDs must be unique, and detector IDs dense from zero; got {:?}",
            definitions
                .iter()
                .enumerate()
                .map(|(position, d)| d.id.unwrap_or(position))
                .collect::<Vec<_>>()
        )));
    }
    let mut outputs = Vec::new();
    for definition in definitions {
        if let Some(coords) = &definition.coords
            && (!detectors || coords.iter().any(|value| !value.is_finite()))
        {
            return Err(DemBuildError::Configuration(
                "coordinates require a detector and finite values".into(),
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
        if let DemDefinitionSource::References(refs) = &definition.source {
            DemDefinitionSource::references(refs.clone())
                .map_err(|error| DemBuildError::Configuration(error.to_string()))?;
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
    index: &'a bindings::ResultTraceIndex,
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
        let resolved = specifications::resolve_role(
            role,
            definitions,
            defaults.index,
            |_| inferred.next().map(Vec::as_slice),
            origins,
        )
        .map_err(|error| DemBuildError::Configuration(error.to_string()))?;
        let entries = resolved
            .into_iter()
            .map(|definition| {
                let mut entry = definition.metadata;
                entry.insert("id".into(), serde_json::json!(definition.id));
                entry.insert(
                    "meas_ids".into(),
                    serde_json::json!(definition.measurement_ids),
                );
                if let Some(coords) = definition.coords {
                    entry.insert("coords".into(), serde_json::json!(coords));
                }
                serde_json::Value::Object(entry)
            })
            .collect();
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
        return Ok(Some(json.to_string()));
    }
    Ok(None)
}

#[cfg(test)]
mod tests;
