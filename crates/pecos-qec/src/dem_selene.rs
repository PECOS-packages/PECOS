// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

//! Optional native program preparation for the DEM builder.
//!
//! QIS programs run through Selene; the `hugr` feature also enables native
//! HUGR lowering. Circuit-only construction does not enable either dependency.
//! Explicit builds and mixed builds with named references require static schedule
//! evidence before execution. Explicit builds capture one baseline; inference
//! validates the schedule across its probes and records that empirical evidence
//! independently of static schedule and named-result binding evidence.

use crate::dem_inference::correlate_measurement_ids;
use crate::{
    DemBuildError, DemBuildInput, DemExecutionEvidence, DemInferenceData, DemInferenceSettings,
    DemOutputSelector, DemProgram, DemProgramRequest, DemScheduleEvidence, GeneratorLayout,
};
use pecos_engines::{coin_toss, sim_builder};
use pecos_programs::{Qis, QisContent};
use pecos_qis::trace_replay::HostedReplayOptions;
use pecos_qis::{
    OperationTraceChunk, OperationTraceStore, QisEngineBuilder, SeleneRuntime,
    helios_interface_builder, qis_engine, selene_simple_runtime,
};
use pecos_qis_ffi_types::{NamedResult, NamedResultTrace};
use pecos_quantum::{Attribute, TickCircuit, pass};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

trait ProgramSource: Send + Sync {
    fn lower(&self) -> Result<Qis, DemBuildError>;

    fn hugr_bytes(&self) -> Option<&[u8]> {
        None
    }

    fn static_schedule_verdict(&self) -> Result<Option<bool>, DemBuildError> {
        Ok(None)
    }
}

impl ProgramSource for Qis {
    fn lower(&self) -> Result<Qis, DemBuildError> {
        Ok(self.clone())
    }
}

struct FrontendCompiledSource {
    program: Qis,
    hugr_bytes: Vec<u8>,
}

impl ProgramSource for FrontendCompiledSource {
    fn lower(&self) -> Result<Qis, DemBuildError> {
        Ok(self.program.clone())
    }

    fn hugr_bytes(&self) -> Option<&[u8]> {
        Some(&self.hugr_bytes)
    }
}

#[cfg(feature = "hugr")]
struct HugrSource {
    bytes: Vec<u8>,
    platform: crate::QSystemPlatform,
    opt_level: crate::OptimizationLevel,
}

#[cfg(feature = "hugr")]
impl ProgramSource for HugrSource {
    fn lower(&self) -> Result<Qis, DemBuildError> {
        super::dem_hugr::lower(&self.bytes, self.platform, self.opt_level)
    }

    fn hugr_bytes(&self) -> Option<&[u8]> {
        Some(&self.bytes)
    }

    fn static_schedule_verdict(&self) -> Result<Option<bool>, DemBuildError> {
        let (_, nontrivial) = super::dem_hugr::parse(&self.bytes)?;
        Ok(Some(!nontrivial))
    }
}

/// Built-in runtime adapter for QIS, and (with `hugr`) HUGR envelopes.
///
/// Supply this directly to `DetectorErrorModel::builder().with_program(...)`.
/// The default runtime is Selene simple. Runtime selection is lazy: creating
/// a program or configuring invalid definitions does not load a runtime.
#[derive(Clone)]
pub struct SeleneDemProgram {
    source: Arc<dyn ProgramSource>,
    runtime: Option<SeleneRuntime>,
    schedule_shots: usize,
    hosted: HostedReplayOptions,
    hosted_context: String,
    frontend_static_source: Option<String>,
    generator_certificate: Option<(Vec<(String, usize)>, String)>,
    simplify_cliffords: bool,
}

impl SeleneDemProgram {
    /// Use QIS LLVM IR. Bitcode is rejected because the runtime's dynamic
    /// execution path currently requires IR text.
    #[must_use]
    pub fn new(program: Qis) -> Self {
        Self {
            source: Arc::new(program),
            runtime: None,
            schedule_shots: 32,
            hosted: HostedReplayOptions::default(),
            hosted_context: "SeleneDemProgram::prepare".into(),
            frontend_static_source: None,
            generator_certificate: None,
            simplify_cliffords: false,
        }
    }

    /// Assert that the front end compiled these exact HUGR bytes to this QIS.
    /// The front end owns compilation and certificate-carrier trust; retaining
    /// the bytes permits the same generator digest validation as native lowering.
    #[must_use]
    pub fn from_frontend_compiled_hugr(program: Qis, hugr_bytes: Vec<u8>) -> Self {
        let mut adapter = Self::new(program.clone());
        adapter.source = Arc::new(FrontendCompiledSource {
            program,
            hugr_bytes,
        });
        adapter
    }

    /// Compile a HUGR envelope natively for the chosen platform and optimization level.
    #[cfg(feature = "hugr")]
    #[must_use]
    pub fn from_hugr(
        bytes: impl Into<Vec<u8>>,
        platform: crate::QSystemPlatform,
        opt_level: crate::OptimizationLevel,
    ) -> Self {
        Self {
            source: Arc::new(HugrSource {
                bytes: bytes.into(),
                platform,
                opt_level,
            }),
            runtime: None,
            schedule_shots: 32,
            hosted: HostedReplayOptions::default(),
            hosted_context: "SeleneDemProgram::prepare".into(),
            frontend_static_source: None,
            generator_certificate: None,
            simplify_cliffords: false,
        }
    }

    #[must_use]
    pub fn with_runtime(mut self, runtime: SeleneRuntime) -> Self {
        self.runtime = Some(runtime);
        self
    }

    /// Require local operations to precede their unique host.
    #[must_use]
    pub fn with_require_hosted_operation_order(mut self, required: bool) -> Self {
        self.hosted.require_hosted_operation_order = required;
        self
    }

    /// Bound absolute hosted tick distance, requiring unique host IDs.
    #[must_use]
    pub fn with_max_hosted_tick_separation(mut self, maximum: Option<u64>) -> Self {
        self.hosted.max_hosted_tick_separation = maximum;
        self
    }

    /// Minimum number of schedule comparisons on empirical inference routes.
    /// The number of additional shots is the maximum of this minimum and the sum
    /// of inference and provenance probe counts. Explicit static builds ignore
    /// this setting and capture exactly one baseline.
    #[must_use]
    pub fn with_schedule_validation_shots(mut self, shots: usize) -> Self {
        self.schedule_shots = shots;
        self
    }

    /// Assert that a front end inspected the HUGR that produced this exact QIS
    /// and established a static measurement schedule (the Python HUGR preflight).
    /// The audit records `source`; this assertion supplies no named-result binding.
    #[must_use]
    pub fn with_frontend_static_schedule(mut self, source: impl Into<String>) -> Self {
        self.frontend_static_source = Some(source.into());
        self
    }

    /// Assert that a trusted built-in generator stamped this layout and certificate
    /// on the definition compiled to the exact bytes supplied to `from_hugr`
    /// or `from_frontend_compiled_hugr`.
    /// This assertion permits control flow; the digest only binds the layout to
    /// those bytes and does not establish trust in their producer. The audit records
    /// `generator_layout_v2_program_bound` as the asserted source. QIS-only sources
    /// without a front-end byte binding cannot use this certificate.
    #[must_use]
    pub fn with_trusted_generator_layout(
        mut self,
        layout: Vec<(String, usize)>,
        expected_digest: impl Into<String>,
    ) -> Self {
        self.generator_certificate = Some((layout, expected_digest.into()));
        self
    }

    /// Context attached to hosted validation errors after replay and probe checks.
    #[must_use]
    pub fn with_hosted_validation_context(mut self, context: impl Into<String>) -> Self {
        self.hosted_context = context.into();
        self
    }

    /// Enable or disable Clifford chain simplification during trace normalization.
    /// Disabled by default to preserve each physical gate as an independent noise site.
    #[must_use]
    pub fn with_simplify_single_qubit_clifford_chains(mut self, simplify: bool) -> Self {
        self.simplify_cliffords = simplify;
        self
    }

    fn verified_generator_layout(&self) -> Result<Option<GeneratorLayout>, DemBuildError> {
        let Some((layout, expected_digest)) = &self.generator_certificate else {
            return Ok(None);
        };
        let bytes = self.source.hugr_bytes().ok_or_else(|| {
            preparation("generator layout requires the exact HUGR bytes supplied to from_hugr")
        })?;
        let generator = GeneratorLayout {
            hugr_bytes: bytes.to_vec(),
            layout: layout.clone(),
            expected_digest: expected_digest.clone(),
        };
        generator.verify().map_err(preparation)?;
        Ok(Some(generator))
    }
}

fn preparation(error: impl std::fmt::Display) -> DemBuildError {
    DemBuildError::Preparation(error.to_string())
}

type TraceShot = Vec<OperationTraceChunk>;

fn capture(
    engine: &QisEngineBuilder,
    qubits: usize,
    shots: usize,
    seed: u64,
) -> Result<Vec<TraceShot>, DemBuildError> {
    let store = OperationTraceStore::default();
    sim_builder()
        .classical(
            engine
                .clone()
                .trace_operations_in_memory_to(Arc::clone(&store)),
        )
        .quantum(coin_toss())
        .qubits(qubits)
        .workers(1)
        .seed(seed)
        .run(shots)
        .map_err(preparation)?;
    let chunks = std::mem::take(&mut *store.lock().map_err(preparation)?);
    let mut grouped = BTreeMap::<(u64, usize), TraceShot>::new();
    for chunk in chunks {
        grouped
            .entry((chunk.engine_trace_id, chunk.shot_index))
            .or_default()
            .push(chunk);
    }
    let mut result = Vec::with_capacity(shots);
    for mut shot in grouped.into_values() {
        shot.sort_by_key(|chunk| chunk.chunk_index);
        result.push(shot);
    }
    Ok(result)
}

fn validate_shot(shot: &[OperationTraceChunk]) -> Result<(), DemBuildError> {
    pecos_qis::trace_replay::validate_shot(
        &shot
            .iter()
            .map(pecos_qis::trace_replay::StoredTraceChunk::from)
            .collect::<Vec<_>>(),
    )
    .map_err(preparation)
}

fn source_ids(shot: &[OperationTraceChunk]) -> Result<Vec<usize>, DemBuildError> {
    pecos_qis::trace_replay::source_ids(
        &shot
            .iter()
            .map(pecos_qis::trace_replay::StoredTraceChunk::from)
            .collect::<Vec<_>>(),
    )
    .map_err(preparation)
}

fn replay(shot: &[OperationTraceChunk], qubits: usize) -> Result<TickCircuit, DemBuildError> {
    pecos_qis::trace_replay::replay_in_memory(shot, qubits).map_err(preparation)
}

fn check_schedule(
    baseline: &[OperationTraceChunk],
    baseline_source: &[usize],
    shot: &[OperationTraceChunk],
) -> Result<(), DemBuildError> {
    let a: Vec<_> = baseline
        .iter()
        .flat_map(|c| &c.lowered_quantum_ops)
        .collect();
    let b: Vec<_> = shot.iter().flat_map(|c| &c.lowered_quantum_ops).collect();
    if a.len() != b.len()
        || a.iter().zip(b).any(|(a, b)| {
            a.gate_type != b.gate_type
                || a.qubits != b.qubits
                || a.angles != b.angles
                || a.params != b.params
                || a.measurement_result_ids != b.measurement_result_ids
                || a.metadata != b.metadata
        })
        || baseline_source != source_ids(shot)?
    {
        return Err(preparation(
            "quantum operation schedule changed; a single static DEM cannot represent this program",
        ));
    }
    Ok(())
}

type NamedRows<'a> = BTreeMap<&'a str, &'a NamedResult>;

fn named_rows(shot: &[OperationTraceChunk]) -> NamedRows<'_> {
    shot.iter()
        .flat_map(|chunk| &chunk.named_results)
        .map(|(tag, values)| (tag.as_str(), values))
        .collect()
}

fn named_row<'a>(rows: &NamedRows<'a>, tag: &str) -> Result<&'a NamedResult, DemBuildError> {
    rows.get(tag)
        .copied()
        .filter(|row| !row.is_empty())
        .ok_or_else(|| preparation(format!("missing or empty required output tag {tag:?}")))
}

fn selected_bit(row: &NamedResult, tag: &str, element: usize) -> Result<u8, DemBuildError> {
    if element >= row.len() {
        return Err(preparation(format!(
            "output {tag:?} element {element} is out of range"
        )));
    }
    let value = match row {
        NamedResult::Bool(values) => return Ok(u8::from(values[element])),
        NamedResult::I64(values) => {
            if let Ok(bit @ (0 | 1)) = u8::try_from(values[element]) {
                return Ok(bit);
            }
            values[element].to_string()
        }
        NamedResult::U64(values) => {
            if let Ok(bit @ (0 | 1)) = u8::try_from(values[element]) {
                return Ok(bit);
            }
            values[element].to_string()
        }
        NamedResult::F64(values) => format!("{:?}", values[element]),
    };
    Err(preparation(format!(
        "output {tag:?} element {element} has value {value}; emit bool or integer 0/1"
    )))
}

fn raw_row(
    shot: &[OperationTraceChunk],
    rows: &NamedRows<'_>,
    tag: &str,
) -> Result<Vec<u8>, DemBuildError> {
    let error = || {
        preparation(format!(
            "raw tag {tag:?}: raw measurements must be emitted as bool results"
        ))
    };
    let values = named_row(rows, tag)?;
    let complete = (0..values.len())
        .map(|i| selected_bit(values, tag, i))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| error())?;
    let traced: Vec<_> = shot
        .iter()
        .flat_map(|chunk| &chunk.named_result_traces)
        .filter(|item| item.name == tag)
        .flat_map(|item| item.values.iter().copied().map(u8::from))
        .collect();
    if complete != traced {
        return Err(error());
    }
    Ok(complete)
}

fn physical_row(shot: &[OperationTraceChunk], ids: &[usize]) -> Result<Vec<u8>, DemBuildError> {
    let outcomes = &shot
        .last()
        .ok_or_else(|| preparation("missing terminal trace"))?
        .measurement_results;
    if outcomes.keys().copied().collect::<BTreeSet<_>>() != ids.iter().copied().collect() {
        return Err(preparation(format!(
            "QIS probe measurement ids differ from the source trace: outcome ids={:?}, source ids={ids:?}",
            outcomes.keys().collect::<Vec<_>>()
        )));
    }
    ids.iter()
        .map(|id| match outcomes.get(id) {
            Some(&bit @ (0 | 1)) => u8::try_from(bit).map_err(preparation),
            value => Err(preparation(format!(
                "missing or non-Boolean physical measurement outcome for ID {id}: {value:?}"
            ))),
        })
        .collect()
}

fn selectors(
    rows: &NamedRows<'_>,
    requested: Option<&[DemOutputSelector]>,
    defaults: &[String],
) -> Result<Vec<DemOutputSelector>, DemBuildError> {
    if let Some(requested) = requested {
        return Ok(requested.to_vec());
    }
    let mut result = Vec::new();
    for tag in defaults {
        result.extend(
            (0..named_row(rows, tag)?.len()).map(|element| DemOutputSelector {
                tag: tag.clone(),
                element,
            }),
        );
    }
    Ok(result)
}

fn selected_row(
    rows: &NamedRows<'_>,
    outputs: &[DemOutputSelector],
) -> Result<Vec<u8>, DemBuildError> {
    outputs
        .iter()
        .map(|output| selected_bit(named_row(rows, &output.tag)?, &output.tag, output.element))
        .collect()
}

/// Capture evidence before Clifford normalization. Explicit preparations have no probes.
pub struct SeleneDemEvidence {
    pub circuit: TickCircuit,
    pub probes: Option<DemInferenceData>,
    pub binding: Option<&'static str>,
    pub detector_outputs: Vec<DemOutputSelector>,
    pub observable_outputs: Vec<DemOutputSelector>,
    pub probe_shots: usize,
    pub correlation_shots: usize,
    pub schedule_shots: usize,
    pub named_result_traces: Vec<NamedResultTrace>,
    pub schedule_evidence: DemScheduleEvidence,
    pub source_measurement_ids: Vec<usize>,
    pub result_tag_analysis: Option<
        Result<
            crate::dem_build::bindings::ResultTagMeasurements,
            crate::dem_build::bindings::DemBindingError,
        >,
    >,
}

struct BaselineAnalysis<'a> {
    baseline: TraceShot,
    source: Vec<usize>,
    evidence: SeleneDemEvidence,
    inference: Option<BaselineInference<'a>>,
    probe_seed: u64,
}

struct BaselineInference<'a> {
    settings: &'a DemInferenceSettings,
    raw_width: usize,
    direct: bool,
    ids: Vec<usize>,
    widths: BTreeMap<String, usize>,
}

impl SeleneDemProgram {
    /// Capture coin-toss shots and analyze them, without normalizing the replayed circuit.
    ///
    /// # Errors
    /// Returns an error, before any capture, for missing required static evidence
    /// or invalid inference settings; otherwise for invalid
    /// traces, outputs, schedules, or measurement bindings.
    pub fn capture_and_analyze(
        &self,
        request: &DemProgramRequest<'_>,
    ) -> Result<SeleneDemEvidence, DemBuildError> {
        self.capture_with(request, capture)
    }

    fn capture_with(
        &self,
        request: &DemProgramRequest<'_>,
        mut capture_shots: impl FnMut(
            &QisEngineBuilder,
            usize,
            usize,
            u64,
        ) -> Result<Vec<TraceShot>, DemBuildError>,
    ) -> Result<SeleneDemEvidence, DemBuildError> {
        let static_source = self.static_schedule_evidence()?;
        crate::dem_build::require_static_schedule(
            self,
            request.requires_static_schedule,
            static_source.as_deref(),
        )?;
        if let Some(settings) = request.inference {
            settings.validate()?;
        }
        let program = self.source.lower()?;
        let QisContent::Ir(source_ir) = program.content else {
            return Err(preparation(
                "Selene DEM preparation requires QIS IR text, not bitcode",
            ));
        };
        let runtime = match &self.runtime {
            Some(runtime) => runtime.clone(),
            None => selene_simple_runtime().map_err(preparation)?,
        };
        let engine = qis_engine()
            .runtime(runtime)
            .interface(helios_interface_builder())
            .program_ir(source_ir.clone());
        let mut baselines = capture_shots(&engine, request.num_qubits, 1, request.seed)?;
        if baselines.len() != 1 {
            return Err(preparation(format!(
                "runtime returned {} baseline shots; expected 1",
                baselines.len()
            )));
        }
        let analysis = analyze_baseline(
            baselines.remove(0),
            request,
            if request.inference.is_some() {
                self.schedule_shots
            } else {
                0
            },
        )?;
        let traces = if analysis.evidence.schedule_shots == 0 {
            Vec::new()
        } else {
            capture_shots(
                &engine,
                request.num_qubits,
                analysis.evidence.schedule_shots,
                analysis.probe_seed,
            )?
        };
        let mut evidence = self.analyze_hosted_probes(analysis, &traces)?;
        evidence.schedule_evidence.static_source = static_source;
        if request.requires_static_schedule {
            evidence.result_tag_analysis = Some(
                pecos_qis::text::extract_result_tag_measurements(&source_ir).map_err(Into::into),
            );
        }
        Ok(evidence)
    }

    fn analyze_hosted_probes(
        &self,
        analysis: BaselineAnalysis<'_>,
        traces: &[TraceShot],
    ) -> Result<SeleneDemEvidence, DemBuildError> {
        let evidence = analyze_probes(analysis, traces)?;
        self.hosted
            .validate(&evidence.circuit, &self.hosted_context)
            .map_err(preparation)?;
        Ok(evidence)
    }
}

fn analyze_baseline<'a>(
    baseline: TraceShot,
    request: &DemProgramRequest<'a>,
    schedule_shots: usize,
) -> Result<BaselineAnalysis<'a>, DemBuildError> {
    let source = source_ids(&baseline)?;
    let circuit = replay(&baseline, request.num_qubits)?;
    let mut evidence = SeleneDemEvidence {
        circuit,
        schedule_evidence: DemScheduleEvidence::default(),
        source_measurement_ids: source.clone(),
        result_tag_analysis: None,
        probes: None,
        binding: None,
        detector_outputs: Vec::new(),
        observable_outputs: Vec::new(),
        probe_shots: 0,
        correlation_shots: 0,
        schedule_shots,
        named_result_traces: baseline
            .iter()
            .flat_map(|c| c.named_result_traces.clone())
            .collect(),
    };
    let probe_seed = request.seed.wrapping_add(1);
    let Some(settings) = request.inference else {
        return Ok(BaselineAnalysis {
            baseline,
            source,
            evidence,
            inference: None,
            probe_seed,
        });
    };
    let baseline_rows = named_rows(&baseline);
    let raw = raw_row(&baseline, &baseline_rows, &settings.raw_tag)?;
    if raw.len() != source.len() {
        return Err(preparation(format!(
            "result tag {:?} must expose every physical measurement exactly once",
            settings.raw_tag
        )));
    }
    let raw_traces: Vec<_> = baseline
        .iter()
        .flat_map(|c| &c.named_result_traces)
        .filter(|r| r.name == settings.raw_tag)
        .collect();
    let direct = raw_traces
        .iter()
        .all(|r| r.result_ids.len() == r.values.len());
    let ids: Vec<_> = raw_traces
        .iter()
        .flat_map(|r| r.result_ids.iter().copied())
        .collect();
    if direct {
        let context = format!("tag ids={ids:?}, source ids={source:?}");
        if ids.iter().copied().collect::<BTreeSet<_>>() != source.iter().copied().collect() {
            return Err(preparation(format!(
                "result tag {:?} must expose every physical measurement exactly once; {context}",
                settings.raw_tag
            )));
        }
        if ids != source {
            return Err(preparation(format!(
                "result tag {:?} must expose physical measurements in certified source order; {context}",
                settings.raw_tag
            )));
        }
    }
    let detector_outputs = selectors(
        &baseline_rows,
        request.detector_outputs,
        std::slice::from_ref(&settings.detector_tag),
    )?;
    let observable_outputs = selectors(
        &baseline_rows,
        request.observable_outputs,
        &settings.observable_tags,
    )?;
    // Validate selectors before spending shots on provenance or fitting.
    selected_row(&baseline_rows, &detector_outputs)?;
    selected_row(&baseline_rows, &observable_outputs)?;
    let mut widths = BTreeMap::new();
    for output in detector_outputs.iter().chain(&observable_outputs) {
        widths.insert(
            output.tag.clone(),
            named_row(&baseline_rows, &output.tag)?.len(),
        );
    }
    let correlation_shots = if !direct && settings.require_raw_provenance {
        settings.provenance_shots
    } else {
        0
    };
    let probe_shots = settings.resolved_probe_shots(raw.len())?;
    let shots = probe_shots
        .checked_add(correlation_shots)
        .ok_or_else(|| preparation("probe count overflow"))?;
    evidence.detector_outputs = detector_outputs;
    evidence.observable_outputs = observable_outputs;
    evidence.probe_shots = probe_shots;
    evidence.correlation_shots = correlation_shots;
    evidence.schedule_shots = shots.max(schedule_shots);
    Ok(BaselineAnalysis {
        baseline,
        source,
        evidence,
        inference: Some(BaselineInference {
            settings,
            raw_width: raw.len(),
            direct,
            ids,
            widths,
        }),
        probe_seed,
    })
}

fn analyze_probes(
    analysis: BaselineAnalysis<'_>,
    traces: &[TraceShot],
) -> Result<SeleneDemEvidence, DemBuildError> {
    let BaselineAnalysis {
        baseline,
        source,
        mut evidence,
        inference,
        ..
    } = analysis;
    if traces.len() != evidence.schedule_shots {
        return Err(preparation(format!(
            "QIS probe trace contains {} shots; expected {}",
            traces.len(),
            evidence.schedule_shots
        )));
    }
    for shot in traces {
        validate_shot(shot)?;
        check_schedule(&baseline, &source, shot)?;
    }
    if !traces.is_empty() {
        evidence.schedule_evidence.validation_shots = traces.len();
        evidence.circuit.set_meta(
            "dem_schedule_validation",
            Attribute::String("empirical_trace_comparison".into()),
        );
        evidence.circuit.set_meta(
            "dem_schedule_validation_shots",
            Attribute::String(evidence.schedule_shots.to_string()),
        );
    }
    let Some(BaselineInference {
        settings,
        raw_width,
        direct,
        mut ids,
        widths,
    }) = inference
    else {
        return Ok(evidence);
    };
    let correlation_shots = evidence.correlation_shots;
    let detector_outputs = &evidence.detector_outputs;
    let observable_outputs = &evidence.observable_outputs;
    let rows: Vec<_> = traces.iter().map(|shot| named_rows(shot)).collect();
    let raw_rows = traces
        .iter()
        .zip(&rows)
        .map(|(shot, rows)| raw_row(shot, rows, &settings.raw_tag))
        .collect::<Result<Vec<_>, _>>()?;
    let binding = if direct {
        "runtime_result_ids"
    } else if settings.require_raw_provenance {
        let physical = traces[..correlation_shots]
            .iter()
            .map(|s| physical_row(s, &source))
            .collect::<Result<Vec<_>, _>>()?;
        ids = correlate_measurement_ids(&raw_rows[..correlation_shots], &physical, &source)?;
        "probe_correlated_result_ids"
    } else {
        ids.clone_from(&source);
        "assumed_canonical_result_order"
    };
    let mut data = DemInferenceData {
        raw_measurement_ids: ids,
        raw_rows: Vec::new(),
        detector_rows: Vec::new(),
        observable_rows: Vec::new(),
    };
    for ((shot, rows), row) in traces
        .iter()
        .zip(&rows)
        .zip(raw_rows)
        .skip(correlation_shots)
    {
        for (tag, width) in &widths {
            if named_row(rows, tag)?.len() != *width {
                return Err(preparation(format!(
                    "output {tag:?} width changed across probes"
                )));
            }
        }
        if row.len() != raw_width {
            return Err(preparation(format!(
                "output {:?} width changed across probes",
                settings.raw_tag
            )));
        }
        if settings.require_raw_provenance && row != physical_row(shot, &data.raw_measurement_ids)?
        {
            return Err(preparation(
                "raw measurement binding changed across inference probes",
            ));
        }
        let detectors = selected_row(rows, detector_outputs)?;
        let observables = selected_row(rows, observable_outputs)?;
        if data.raw_rows.len() < evidence.probe_shots {
            data.raw_rows.push(row);
            data.detector_rows.push(detectors);
            data.observable_rows.push(observables);
        }
    }
    evidence.circuit.set_meta(
        "dem_raw_measurement_binding",
        Attribute::String(binding.into()),
    );
    evidence.binding = Some(binding);
    evidence.probes = Some(data);
    Ok(evidence)
}

impl DemProgram for SeleneDemProgram {
    fn static_schedule_evidence(&self) -> Result<Option<String>, DemBuildError> {
        if self.verified_generator_layout()?.is_some() {
            return Ok(Some("generator_layout_v2_program_bound".into()));
        }
        if let Some(verdict) = self.source.static_schedule_verdict()? {
            return Ok(verdict.then(|| "native_hugr_control_flow".into()));
        }
        if self
            .frontend_static_source
            .as_ref()
            .is_some_and(String::is_empty)
        {
            return Err(preparation(
                "front-end static schedule source must be nonempty",
            ));
        }
        Ok(self.frontend_static_source.clone())
    }

    fn missing_static_schedule_error(&self) -> DemBuildError {
        if self.source.hugr_bytes().is_some() {
            preparation(
                "SeleneDemProgram requires a statically straight-line Guppy program unless it carries a trusted generator-owned measurement layout; branching or looping control flow cannot be certified from one runtime trace",
            )
        } else {
            preparation(
                "DEM program requires static schedule evidence; QIS-only input cannot be certified as statically scheduled. Use SeleneDemProgram::from_hugr, a front-end HUGR preflight, or with_circuit for circuit inputs",
            )
        }
    }

    fn prepare(&mut self, request: &DemProgramRequest<'_>) -> Result<DemBuildInput, DemBuildError> {
        let evidence = self.capture_and_analyze(request)?;
        let mut circuit = evidence.circuit;
        pass::normalize_traced_tick_circuit(&mut circuit, self.simplify_cliffords);
        Ok(DemBuildInput {
            circuit,
            evidence: DemExecutionEvidence {
                schedule: evidence.schedule_evidence,
                raw_binding: evidence.binding.map(str::to_string),
                provenance_shots: evidence.binding.map(|_| evidence.correlation_shots),
            },
            result_trace_values: evidence
                .named_result_traces
                .iter()
                .map(|trace| trace.values.clone())
                .collect(),
            result_traces: evidence
                .named_result_traces
                .into_iter()
                .map(|trace| crate::dem_build::bindings::ResultTrace {
                    name: trace.name,
                    occurrence: None,
                    width: trace.values.len(),
                    measurement_ids: (!trace.result_ids.is_empty()
                        && trace.result_ids.len() == trace.values.len())
                    .then_some(trace.result_ids),
                })
                .collect(),
            source_measurement_ids: evidence.source_measurement_ids,
            result_tag_analysis: evidence.result_tag_analysis,
            generator_layout: if request.requires_static_schedule {
                self.verified_generator_layout()?
            } else {
                None
            },
            probes: evidence.probes,
            resolved_detector_outputs: Some(evidence.detector_outputs),
            resolved_observable_outputs: Some(evidence.observable_outputs),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::{MeasId, gate_type::GateType};
    use pecos_qis::LoweredQuantumGateTrace;
    use pecos_qis_ffi_types::{Operation, QuantumOp, RuntimeRequest, SeleneRequest};

    fn chunk(index: usize, stage: &str) -> OperationTraceChunk {
        OperationTraceChunk {
            format: "pecos_qis_operation_trace_v1",
            engine_trace_id: 1,
            shot_index: 0,
            chunk_index: index,
            stage: stage.into(),
            waiting_for_result_id: None,
            current_shot_seed: Some(42),
            simulated_op_count: 0,
            num_operations: 0,
            operations: vec![],
            lowered_quantum_ops: vec![],
            lowered_quantum_ops_complete: true,
            runtime_lowering_deferred: false,
            runtime_events_complete: true,
            named_result_traces: vec![],
            named_results: BTreeMap::new(),
            measurement_results: BTreeMap::new(),
        }
    }

    fn trace() -> TraceShot {
        let mut first = chunk(0, "batch");
        first.operations = vec![QuantumOp::Measure(0, 7).into()];
        first.num_operations = 1;
        first.lowered_quantum_ops.push(LoweredQuantumGateTrace {
            gate_type: "MZ".into(),
            angles: vec![],
            params: vec![],
            qubits: vec![0],
            measurement_result_ids: vec![7],
            metadata: BTreeMap::new(),
        });
        let mut last = chunk(1, "trace_complete");
        last.measurement_results.insert(7, 1);
        vec![first, last]
    }

    #[test]
    fn native_measurements_preserve_dense_source_ids_and_require_lowered_output() {
        let mut shot = trace();
        shot[0].operations = vec![Operation::RuntimeRequest {
            request: RuntimeRequest::Native(SeleneRequest::Measure {
                qubit: 0,
                result_id: 7,
                leaked: false,
            }),
            result_id: Some(pecos_qis_ffi_types::RUNTIME_REQUEST_RESULT_ID),
        }];
        assert_eq!(source_ids(&shot).unwrap(), [7]);
        validate_shot(&shot).unwrap();
        shot[0].lowered_quantum_ops.clear();
        assert!(validate_shot(&shot).is_err());
        shot[0].runtime_lowering_deferred = true;
        validate_shot(&shot).unwrap();
    }

    fn deferred_trace() -> TraceShot {
        let mut shot = trace();
        let mut lowered = chunk(1, "result_read");
        lowered.lowered_quantum_ops = std::mem::take(&mut shot[0].lowered_quantum_ops);
        shot[0].runtime_lowering_deferred = true;
        shot[1].chunk_index = 2;
        shot.insert(1, lowered);
        shot
    }

    #[test]
    fn explicit_static_capture_runs_exactly_one_shot() {
        let program = SeleneDemProgram::new(Qis::from_string("not executed by this test"))
            .with_frontend_static_schedule("test_frontend_hugr_preflight");
        let request = DemProgramRequest {
            requires_static_schedule: true,
            num_qubits: 1,
            seed: 42,
            inference: None,
            detector_outputs: None,
            observable_outputs: None,
        };
        let mut counts = Vec::new();
        let evidence = program
            .capture_with(&request, |_, _, shots, _| {
                counts.push(shots);
                Ok(vec![trace()])
            })
            .unwrap();
        assert_eq!(counts, [1]);
        assert_eq!(
            evidence.schedule_evidence.static_source.as_deref(),
            Some("test_frontend_hugr_preflight")
        );
        assert_eq!(evidence.schedule_evidence.validation_shots, 0);
        assert!(
            evidence
                .circuit
                .get_meta("dem_schedule_validation")
                .is_none()
        );
    }

    #[test]
    fn missing_static_evidence_rejects_before_capture() {
        let program = SeleneDemProgram::new(Qis::from_string("must not execute"))
            .with_schedule_validation_shots(0);
        let request = DemProgramRequest {
            requires_static_schedule: true,
            num_qubits: 1,
            seed: 42,
            inference: None,
            detector_outputs: None,
            observable_outputs: None,
        };
        let error = program
            .capture_with(&request, |_, _, _, _| panic!("capture ran"))
            .err()
            .unwrap();
        assert!(
            error
                .to_string()
                .contains("requires static schedule evidence")
        );
    }

    #[test]
    #[cfg(feature = "hugr")]
    fn invalid_generator_digest_rejects_before_capture() {
        let bytes = include_bytes!("../tests/fixtures/dem_programs/explicit_only.hugr");
        let program = SeleneDemProgram::from_hugr(
            bytes.to_vec(),
            crate::QSystemPlatform::Helios,
            crate::OptimizationLevel::Default,
        )
        .with_trusted_generator_layout(vec![("measurement".into(), 0)], "bad digest");
        let request = DemProgramRequest {
            requires_static_schedule: true,
            num_qubits: 1,
            seed: 42,
            inference: None,
            detector_outputs: None,
            observable_outputs: None,
        };
        let mut captures = 0;
        let error = program
            .capture_with(&request, |_, _, _, _| {
                captures += 1;
                Ok(vec![trace()])
            })
            .err()
            .unwrap();
        assert_eq!(captures, 0);
        assert!(error.to_string().contains("certificate does not match"));
    }

    #[test]
    #[cfg(feature = "hugr")]
    fn trusted_generator_assertion_can_override_native_control_flow_verdict() {
        let bytes = include_bytes!("../tests/fixtures/dem_programs/changing_schedule.hugr");
        let program = SeleneDemProgram::from_hugr(
            bytes.to_vec(),
            crate::QSystemPlatform::Helios,
            crate::OptimizationLevel::Default,
        );
        assert_eq!(program.static_schedule_evidence().unwrap(), None);
        let layout = vec![("raw measurements".into(), 0)];
        let digest = crate::dem_build::generator_certificate_digest(bytes, &layout);
        let trusted = program.with_trusted_generator_layout(layout, digest);
        assert_eq!(
            trusted.static_schedule_evidence().unwrap().as_deref(),
            Some("generator_layout_v2_program_bound")
        );
    }

    #[test]
    fn empirical_capture_does_not_analyze_result_tags() {
        let program = SeleneDemProgram::new(Qis::from_string("not valid QIS analysis"))
            .with_schedule_validation_shots(0);
        let settings = analysis_settings();
        let request = analysis_request(&settings);
        let mut counts = Vec::new();
        let evidence = program
            .capture_with(&request, |_, _, shots, _| {
                counts.push(shots);
                if shots == 1 {
                    Ok(vec![inference_trace(true, [false, false])])
                } else {
                    Ok(inference_probes(true))
                }
            })
            .unwrap();
        assert_eq!(counts, [1, 4]);
        assert!(evidence.result_tag_analysis.is_none());
        assert_eq!(evidence.schedule_evidence.validation_shots, 4);
    }

    #[test]
    fn sampled_evidence_requires_actual_comparisons() {
        let settings = analysis_settings();
        let request = analysis_request(&settings);
        let analysis =
            analyze_baseline(inference_trace(true, [false, false]), &request, 0).unwrap();
        assert_eq!(analysis.evidence.schedule_evidence.validation_shots, 0);
        let evidence = analyze_probes(analysis, &inference_probes(true)).unwrap();
        assert_eq!(evidence.schedule_evidence.validation_shots, 4);
    }

    #[test]
    fn extra_schedule_comparisons_do_not_add_fitting_rows() {
        let settings = analysis_settings();
        let request = analysis_request(&settings);
        let analysis =
            analyze_baseline(inference_trace(true, [false, false]), &request, 8).unwrap();
        let probes = inference_probes(true)
            .into_iter()
            .cycle()
            .take(8)
            .collect::<Vec<_>>();
        let evidence = analyze_probes(analysis, &probes).unwrap();
        assert_eq!(evidence.probes.unwrap().raw_rows.len(), 4);
        assert_eq!(evidence.schedule_evidence.validation_shots, 8);
    }

    #[test]
    fn hosted_options_reject_order_drift_only_when_requested() {
        let mut baseline = trace();
        let host = LoweredQuantumGateTrace {
            gate_type: "H".into(),
            angles: vec![],
            params: vec![],
            qubits: vec![0],
            measurement_result_ids: vec![],
            metadata: BTreeMap::from([("host_id".into(), "host:a".into())]),
        };
        let mut local = host.clone();
        local
            .metadata
            .insert("local_role".into(), "basis_prefix".into());
        baseline[0].lowered_quantum_ops.splice(0..0, [host, local]);
        let request = DemProgramRequest {
            requires_static_schedule: false,
            num_qubits: 1,
            seed: 42,
            inference: None,
            detector_outputs: None,
            observable_outputs: None,
        };
        for required in [false, true] {
            let program = SeleneDemProgram::new(Qis::from_string(""))
                .with_require_hosted_operation_order(required);
            let analysis = analyze_baseline(baseline.clone(), &request, 1).unwrap();
            let result = program.analyze_hosted_probes(analysis, std::slice::from_ref(&baseline));
            if required {
                let analysis = analyze_baseline(baseline.clone(), &request, 1).unwrap();
                let probe_error = program.analyze_hosted_probes(analysis, &[]).err().unwrap();
                assert!(
                    probe_error
                        .to_string()
                        .contains("QIS probe trace contains 0 shots")
                );
                assert!(result.err().unwrap().to_string().contains("ordering drift"));
            } else {
                assert!(result.is_ok());
            }
        }
        let mut repeated = baseline.clone();
        let host = repeated[0].lowered_quantum_ops[0].clone();
        repeated[0].lowered_quantum_ops.insert(0, host);
        let program =
            SeleneDemProgram::new(Qis::from_string("")).with_max_hosted_tick_separation(Some(10));
        let analysis = analyze_baseline(repeated.clone(), &request, 1).unwrap();
        assert!(
            program
                .analyze_hosted_probes(analysis, &[repeated])
                .err()
                .unwrap()
                .to_string()
                .contains("appears on 2 host gates")
        );
        let program =
            SeleneDemProgram::new(Qis::from_string("")).with_max_hosted_tick_separation(Some(0));
        let analysis = analyze_baseline(baseline.clone(), &request, 1).unwrap();
        assert!(
            program
                .analyze_hosted_probes(analysis, &[baseline])
                .err()
                .unwrap()
                .to_string()
                .contains("signed tick separation -1")
        );
    }

    fn inference_trace(direct: bool, bits: [bool; 2]) -> TraceShot {
        let mut shot = trace();
        shot[0].operations.push(QuantumOp::Measure(0, 9).into());
        shot[0].num_operations = 2;
        let mut second = shot[0].lowered_quantum_ops[0].clone();
        second.measurement_result_ids = vec![9];
        shot[0].lowered_quantum_ops.push(second);
        shot[0].named_results = BTreeMap::from([
            ("raw measurements".into(), NamedResult::Bool(bits.to_vec())),
            (
                "DETECTOR".into(),
                NamedResult::I64(vec![i64::from(bits[0] ^ bits[1])]),
            ),
            ("obs".into(), NamedResult::Bool(vec![bits[0]])),
        ]);
        shot[0].named_result_traces.push(NamedResultTrace {
            name: "raw measurements".into(),
            values: bits.to_vec(),
            result_ids: if direct { vec![7, 9] } else { vec![] },
        });
        shot[1].measurement_results =
            BTreeMap::from([(7, u32::from(bits[0])), (9, u32::from(bits[1]))]);
        shot
    }

    fn analysis_settings() -> DemInferenceSettings {
        DemInferenceSettings::default()
            .with_probe_shots(4)
            .with_provenance_shots(2)
            .with_validation_rows(1)
    }

    fn analysis_request(settings: &DemInferenceSettings) -> DemProgramRequest<'_> {
        DemProgramRequest {
            requires_static_schedule: false,
            num_qubits: 1,
            seed: u64::MAX,
            inference: Some(settings),
            detector_outputs: None,
            observable_outputs: None,
        }
    }

    fn analyze_test_shots(
        baseline: TraceShot,
        shots: &[TraceShot],
    ) -> Result<SeleneDemEvidence, DemBuildError> {
        let settings = analysis_settings();
        let request = analysis_request(&settings);
        let analysis = analyze_baseline(baseline, &request, 0)?;
        assert_eq!(analysis.probe_seed, 0);
        analyze_probes(analysis, shots)
    }

    fn inference_probes(direct: bool) -> Vec<TraceShot> {
        (0..if direct { 4 } else { 6 })
            .map(|index| inference_trace(direct, [index % 2 == 0, index % 2 == 1]))
            .collect()
    }

    fn analysis_error(result: Result<SeleneDemEvidence, DemBuildError>, expected: &str) {
        match result {
            Err(error) => assert!(error.to_string().contains(expected), "{error}"),
            Ok(_) => panic!("analysis accepted corrupt evidence; expected {expected}"),
        }
    }

    #[test]
    fn baseline_rejects_short_probe_count_before_capture() {
        let settings = analysis_settings().with_probe_shots(3);
        let request = analysis_request(&settings);
        for direct in [false, true] {
            let result = analyze_baseline(inference_trace(direct, [false, false]), &request, 32);
            assert!(matches!(result,
                Err(DemBuildError::Inference(crate::DemInferenceError::InvalidSettings(message)))
                if message == "explicit probe_shots 3 for raw width 2 requires at least 4 rows"));
        }
    }

    #[test]
    fn analysis_rejects_permuted_direct_ids_in_baseline() {
        let mut baseline = inference_trace(true, [true, false]);
        baseline[0].named_result_traces[0].result_ids = vec![9, 7];
        analysis_error(
            analyze_test_shots(baseline, &inference_probes(true)),
            "certified source order; tag ids=[9, 7], source ids=[7, 9]",
        );
    }

    #[test]
    fn analysis_rejects_incomplete_direct_ids_in_baseline() {
        let mut baseline = inference_trace(true, [true, false]);
        baseline[0].named_result_traces[0].result_ids = vec![7, 7];
        analysis_error(
            analyze_test_shots(baseline, &inference_probes(true)),
            "exactly once; tag ids=[7, 7], source ids=[7, 9]",
        );
    }

    #[test]
    fn analysis_checks_schedule_with_direct_ids() {
        let baseline = inference_trace(true, [true, false]);
        let mut probes = inference_probes(true);
        probes[0][0].lowered_quantum_ops[0].qubits = vec![99];
        analysis_error(analyze_test_shots(baseline, &probes), "schedule changed");
    }

    #[test]
    fn analysis_checks_last_fitted_schedule() {
        let baseline = inference_trace(false, [true, false]);
        let mut probes = inference_probes(false);
        probes.last_mut().unwrap()[0].lowered_quantum_ops[0].qubits = vec![99];
        analysis_error(analyze_test_shots(baseline, &probes), "schedule changed");
    }

    #[test]
    fn analysis_checks_last_fitted_binding() {
        let baseline = inference_trace(false, [true, false]);
        let mut probes = inference_probes(false);
        let shot = probes.last_mut().unwrap();
        shot[0].named_result_traces[0].values[0] = true;
        shot[0].named_results.insert(
            "raw measurements".into(),
            NamedResult::Bool(vec![true, true]),
        );
        analysis_error(analyze_test_shots(baseline, &probes), "binding changed");
    }

    #[test]
    fn analysis_checks_last_fitted_width() {
        let baseline = inference_trace(false, [true, false]);
        let mut probes = inference_probes(false);
        probes.last_mut().unwrap()[0]
            .named_results
            .insert("DETECTOR".into(), NamedResult::I64(vec![1, 0]));
        analysis_error(
            analyze_test_shots(baseline, &probes),
            "output \"DETECTOR\" width changed",
        );
    }

    #[test]
    fn analysis_checks_last_fitted_raw_width() {
        let baseline = inference_trace(false, [true, false]);
        let mut probes = inference_probes(false);
        let shot = probes.last_mut().unwrap();
        shot[0].named_results.insert(
            "raw measurements".into(),
            NamedResult::Bool(vec![false, true, false]),
        );
        shot[0].named_result_traces[0].values.push(false);
        analysis_error(
            analyze_test_shots(baseline, &probes),
            "output \"raw measurements\" width changed",
        );
    }

    #[test]
    fn analysis_checks_last_fitted_terminal() {
        let baseline = inference_trace(false, [true, false]);
        let mut probes = inference_probes(false);
        probes.last_mut().unwrap().pop();
        analysis_error(
            analyze_test_shots(baseline, &probes),
            "exactly one trace_complete",
        );
    }

    #[test]
    fn analysis_checks_last_fitted_lowering_attestation() {
        let baseline = inference_trace(false, [true, false]);
        let mut probes = inference_probes(false);
        probes.last_mut().unwrap()[0].lowered_quantum_ops_complete = false;
        analysis_error(analyze_test_shots(baseline, &probes), "incomplete");
    }

    fn add_unlowered_gate(shot: &mut TraceShot) {
        let mut missing = chunk(1, "batch");
        missing.operations.push(QuantumOp::H(0).into());
        missing.num_operations = 1;
        shot.insert(1, missing);
        shot[2].chunk_index = 2;
    }

    #[test]
    fn analysis_checks_baseline_empty_lowering() {
        let mut baseline = inference_trace(true, [true, false]);
        add_unlowered_gate(&mut baseline);
        analysis_error(
            analyze_test_shots(baseline, &inference_probes(true)),
            "lowerable quantum operations",
        );
    }

    #[test]
    fn analysis_checks_probe_empty_lowering() {
        let baseline = inference_trace(false, [true, false]);
        let mut probes = inference_probes(false);
        add_unlowered_gate(probes.last_mut().unwrap());
        analysis_error(
            analyze_test_shots(baseline, &probes),
            "lowerable quantum operations",
        );
    }

    #[test]
    fn analysis_rejects_selected_float_in_baseline() {
        for value in [0.0, 1.0, 2.5] {
            let mut baseline = inference_trace(true, [true, false]);
            baseline[0]
                .named_results
                .insert("float output".into(), NamedResult::F64(vec![value]));
            assert!(analyze_test_shots(baseline.clone(), &inference_probes(true)).is_ok());
            let settings = analysis_settings();
            let selected = [DemOutputSelector {
                tag: "float output".into(),
                element: 0,
            }];
            let request = DemProgramRequest {
                detector_outputs: Some(&selected),
                ..analysis_request(&settings)
            };
            match analyze_baseline(baseline, &request, 32) {
                Err(error) => assert!(
                    error.to_string().contains(&format!(
                        "output \"float output\" element 0 has value {value:?}"
                    )),
                    "{error}"
                ),
                Ok(_) => panic!("selected float was accepted"),
            }
        }
    }

    #[test]
    fn analysis_requires_raw_trace_in_baseline() {
        let mut baseline = inference_trace(true, [true, false]);
        baseline[0].named_results.insert(
            "raw measurements".into(),
            NamedResult::Bool(vec![true, false, false]),
        );
        analysis_error(
            analyze_test_shots(baseline, &inference_probes(true)),
            "raw measurements\": raw measurements must be emitted as bool results",
        );
    }

    #[test]
    fn analysis_requires_raw_trace_in_first_probe() {
        let baseline = inference_trace(false, [true, false]);
        let mut probes = inference_probes(false);
        probes[0][0].named_results.insert(
            "raw measurements".into(),
            NamedResult::Bool(vec![true, false, false]),
        );
        analysis_error(
            analyze_test_shots(baseline, &probes),
            "raw measurements\": raw measurements must be emitted as bool results",
        );
    }

    #[test]
    fn analysis_requires_complete_named_results_per_shot() {
        let baseline = inference_trace(false, [true, false]);
        let mut probes = inference_probes(false);
        probes.last_mut().unwrap()[0]
            .named_results
            .remove("DETECTOR");
        analysis_error(
            analyze_test_shots(baseline, &probes),
            "missing or empty required output tag \"DETECTOR\"",
        );
    }

    fn duplicate_named_results(shot: &mut TraceShot) {
        let mut duplicate = chunk(1, "named_results");
        duplicate.named_results.clone_from(&shot[0].named_results);
        shot.insert(1, duplicate);
        shot[2].chunk_index = 2;
    }

    #[test]
    fn analysis_rejects_duplicate_complete_results_in_baseline() {
        let mut baseline = inference_trace(true, [true, false]);
        duplicate_named_results(&mut baseline);
        analysis_error(
            analyze_test_shots(baseline, &inference_probes(true)),
            "more than one chunk",
        );
    }

    #[test]
    fn analysis_rejects_duplicate_complete_results_in_probe() {
        let baseline = inference_trace(false, [true, false]);
        let mut probes = inference_probes(false);
        duplicate_named_results(probes.last_mut().unwrap());
        analysis_error(analyze_test_shots(baseline, &probes), "more than one chunk");
    }

    #[test]
    fn complete_named_results_appear_in_at_most_one_chunk() {
        let mut shot = trace();
        let mut named = chunk(1, "named_results");
        named
            .named_results
            .insert("raw".into(), NamedResult::Bool(vec![true]));
        shot.insert(1, named.clone());
        shot[2].chunk_index = 2;
        assert!(validate_shot(&shot).is_ok());
        named.chunk_index = 2;
        shot.insert(2, named);
        shot[3].chunk_index = 3;
        let error = validate_shot(&shot).unwrap_err().to_string();
        assert!(error.contains("more than one chunk"), "{error}");
    }

    #[test]
    fn selected_values_use_complete_typed_rows() {
        let mut shot = trace();
        shot[0].named_results = BTreeMap::from([
            ("checks".into(), NamedResult::I64(vec![0, 1])),
            ("counts".into(), NamedResult::U64(vec![2, 1])),
            ("float".into(), NamedResult::F64(vec![0.0, 1.0, 2.5])),
        ]);
        let rows = named_rows(&shot);
        let outputs = selectors(&rows, None, &["checks".into()]).unwrap();
        assert_eq!(selected_row(&rows, &outputs).unwrap(), [0, 1]);
        assert_eq!(selected_bit(rows["counts"], "counts", 1).unwrap(), 1);
        let error = selected_bit(rows["counts"], "counts", 0)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("counts") && error.contains("element 0") && error.contains("value 2"),
            "{error}"
        );
        for (index, value) in ["0.0", "1.0", "2.5"].iter().enumerate() {
            let error = selected_bit(rows["float"], "float", index)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("float")
                    && error.contains(&format!("element {index}"))
                    && error.contains(value),
                "{error}"
            );
        }
        assert!(selected_bit(rows["checks"], "checks", 2).is_err());
    }

    #[test]
    fn raw_outputs_require_complete_trace_values() {
        let mut shot = trace();
        shot[0]
            .named_results
            .insert("raw".into(), NamedResult::Bool(vec![false, true]));
        shot[0]
            .named_result_traces
            .push(pecos_qis_ffi_types::NamedResultTrace {
                name: "raw".into(),
                values: vec![true],
                result_ids: vec![7],
            });
        let error = raw_row(&shot, &named_rows(&shot), "raw")
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("raw") && error.contains("must be emitted as bool results"),
            "{error}"
        );
        shot[0].named_result_traces[0].values = vec![false, true];
        assert_eq!(raw_row(&shot, &named_rows(&shot), "raw").unwrap(), [0, 1]);
    }

    #[test]
    fn replay_rejects_gates_outside_python_gate_set() {
        for name in ["MX", "QAlloc", "Channel"] {
            let mut shot = trace();
            shot[0].lowered_quantum_ops[0].gate_type = name.into();
            let error = replay(&shot, 1).unwrap_err().to_string();
            assert!(
                error.contains(name) && error.contains("unsupported lowered traced gate"),
                "{error}"
            );
        }
    }

    #[test]
    fn replay_preserves_stable_measurement_identity() {
        let circuit = replay(&trace(), 1).unwrap();
        let ids: Vec<_> = circuit
            .iter_gate_batches()
            .flat_map(|gate| gate.as_gate().meas_ids.clone())
            .collect();
        assert_eq!(ids, vec![MeasId::from_raw(7)]);
        assert_eq!(circuit.num_measurements(), 8);
        assert_eq!(physical_row(&trace(), &[7]).unwrap(), [1]);
    }

    #[test]
    fn replay_accepts_deferred_lowering_with_stable_measurement_identity() {
        let shot = deferred_trace();
        let circuit = replay(&shot, 1).unwrap();
        let gates: Vec<_> = circuit.iter_gate_batches().collect();
        assert_eq!(gates.len(), 1);
        assert_eq!(gates[0].as_gate().gate_type, GateType::MZ);
        assert_eq!(
            gates[0].as_gate().meas_ids.as_slice(),
            [MeasId::from_raw(7)]
        );
        assert_eq!(physical_row(&shot, &[7]).unwrap(), [1]);
    }

    #[test]
    fn deferred_lowering_requires_complete_framed_and_handled_trace() {
        let mut shot = deferred_trace();
        shot[0].runtime_lowering_deferred = false;
        assert!(
            replay(&shot, 1)
                .unwrap_err()
                .to_string()
                .contains("no lowered_quantum_ops")
        );
        let mut shot = deferred_trace();
        shot.pop();
        assert!(
            replay(&shot, 1)
                .unwrap_err()
                .to_string()
                .contains("exactly one trace_complete")
        );
        for index in 0..3 {
            let mut shot = deferred_trace();
            shot[index].lowered_quantum_ops_complete = false;
            assert!(replay(&shot, 1).is_err());
            let mut shot = deferred_trace();
            shot[index].runtime_events_complete = false;
            assert!(
                replay(&shot, 1)
                    .unwrap_err()
                    .to_string()
                    .contains("unhandled runtime custom events")
            );
            let mut shot = deferred_trace();
            shot[index].chunk_index += 1;
            assert!(replay(&shot, 1).is_err());
        }
        let mut shot = deferred_trace();
        shot[1].lowered_quantum_ops.clear();
        assert!(replay(&shot, 1).is_err());
    }

    #[test]
    fn replay_rejects_partial_traces_and_missing_identity() {
        let mut shot = trace();
        shot.pop();
        assert!(replay(&shot, 1).is_err());
        let mut shot = trace();
        shot[0].lowered_quantum_ops_complete = false;
        assert!(replay(&shot, 1).is_err());
        let mut shot = trace();
        shot[1].runtime_events_complete = false;
        assert!(
            replay(&shot, 1)
                .unwrap_err()
                .to_string()
                .contains("unhandled runtime custom events")
        );
        let mut shot = trace();
        shot[0].lowered_quantum_ops[0]
            .measurement_result_ids
            .clear();
        assert!(replay(&shot, 1).is_err());
        let mut shot = trace();
        shot[0].lowered_quantum_ops[0].measurement_result_ids = vec![8];
        assert!(replay(&shot, 1).is_err());
        let mut shot = trace();
        shot[1].chunk_index = 2;
        assert!(replay(&shot, 1).is_err());
    }

    #[test]
    fn schedule_comparison_includes_identity_and_metadata() {
        let baseline = trace();
        let mut changed = trace();
        changed[0].lowered_quantum_ops[0].measurement_result_ids = vec![8];
        assert!(check_schedule(&baseline, &source_ids(&baseline).unwrap(), &changed).is_err());
        let mut changed = trace();
        changed[0].lowered_quantum_ops[0]
            .metadata
            .insert("role".into(), "changed".into());
        assert!(check_schedule(&baseline, &source_ids(&baseline).unwrap(), &changed).is_err());
    }

    #[test]
    fn replay_converts_idle_seconds_and_keeps_metadata() {
        let mut shot = trace();
        shot[0].lowered_quantum_ops.insert(
            0,
            LoweredQuantumGateTrace {
                gate_type: "Idle".into(),
                angles: vec![],
                params: vec![1.5e-9],
                qubits: vec![0],
                measurement_result_ids: vec![],
                metadata: BTreeMap::from([("role".into(), "waiting".into())]),
            },
        );
        let circuit = replay(&shot, 1).unwrap();
        let idle = circuit
            .iter_gate_batches()
            .find(|g| g.as_gate().gate_type == GateType::Idle)
            .unwrap();
        assert_eq!(idle.as_gate().params.as_slice(), [2.0]);
        assert_eq!(
            idle.get_attr("role"),
            Some(&Attribute::String("waiting".into()))
        );
    }

    #[test]
    fn replay_rejects_invalid_gate_parameters_and_bounds() {
        // Python replay constructs TimeUnits(u64), so finite f64 durations
        // must also fit that range after conversion to nanoseconds.
        let mut baseline = inference_trace(true, [true, false]);
        baseline[0].lowered_quantum_ops.insert(
            0,
            LoweredQuantumGateTrace {
                gate_type: "Idle".into(),
                angles: vec![],
                params: vec![2.0_f64.powi(64) / 1e9],
                qubits: vec![0],
                measurement_result_ids: vec![],
                metadata: BTreeMap::new(),
            },
        );
        analysis_error(
            analyze_test_shots(baseline, &inference_probes(true)),
            "fit in u64 nanoseconds",
        );
        for (qubit, angle) in [(1, 0.0), (0, f64::NAN)] {
            let mut shot = trace();
            shot[0].lowered_quantum_ops[0].qubits = vec![qubit];
            shot[0].lowered_quantum_ops[0].angles = vec![angle];
            assert!(replay(&shot, 1).is_err());
        }
    }
}
