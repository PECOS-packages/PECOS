// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

//! Optional native program preparation for the DEM builder.
//!
//! QIS programs run through Selene; the `hugr` feature also enables native
//! HUGR lowering. Circuit-only construction does not enable either dependency.
//! Schedule checks are empirical, including for explicit QIS definitions.

use crate::dem_inference::correlate_measurement_ids;
use crate::{
    DemBuildError, DemBuildInput, DemInferenceData, DemOutputSelector, DemProgram,
    DemProgramRequest,
};
use pecos_core::{Angle64, Gate, MeasId, QubitId, gate_type::GateType};
use pecos_engines::{coin_toss, sim_builder};
use pecos_programs::{Qis, QisContent};
use pecos_qis::{
    OperationTraceChunk, OperationTraceStore, QisEngineBuilder, SeleneRuntime,
    helios_interface_builder, qis_engine, selene_simple_runtime,
};
use pecos_qis_ffi_types::{NamedResult, Operation, QuantumOp};
use pecos_quantum::{Attribute, TickCircuit, pass};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

#[derive(Clone)]
enum ProgramSource {
    Qis(Qis),
    #[cfg(feature = "hugr")]
    Hugr(Vec<u8>),
}

/// Built-in runtime adapter for QIS, and (with `hugr`) HUGR envelopes.
///
/// Supply this directly to `DetectorErrorModel::builder().with_program(...)`.
/// The default runtime is Selene simple. Runtime selection is lazy: creating
/// a program or configuring invalid definitions does not load a runtime.
#[derive(Clone)]
pub struct SeleneDemProgram {
    source: ProgramSource,
    runtime: Option<SeleneRuntime>,
    schedule_shots: usize,
}

impl SeleneDemProgram {
    /// Use QIS LLVM IR. Bitcode is rejected because the runtime's dynamic
    /// execution path currently requires IR text.
    #[must_use]
    pub fn new(program: Qis) -> Self {
        Self {
            source: ProgramSource::Qis(program),
            runtime: None,
            schedule_shots: 32,
        }
    }

    /// Compile a HUGR envelope natively, with no Python interpreter.
    #[cfg(feature = "hugr")]
    #[must_use]
    pub fn from_hugr(bytes: impl Into<Vec<u8>>) -> Self {
        Self {
            source: ProgramSource::Hugr(bytes.into()),
            runtime: None,
            schedule_shots: 32,
        }
    }

    #[must_use]
    pub fn with_runtime(mut self, runtime: SeleneRuntime) -> Self {
        self.runtime = Some(runtime);
        self
    }

    /// Number of additional schedule-check shots for explicit programs.
    /// Inferred programs instead validate every inference/provenance shot.
    /// Must be positive for explicit builds; unused for inferred builds.
    /// A sample comparison is not a compiler proof.
    #[must_use]
    pub fn with_schedule_validation_shots(mut self, shots: usize) -> Self {
        self.schedule_shots = shots;
        self
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
    if grouped.len() != shots {
        return Err(preparation(
            "runtime did not return exactly one complete trace per requested shot",
        ));
    }
    let mut result = Vec::with_capacity(shots);
    for mut shot in grouped.into_values() {
        shot.sort_by_key(|chunk| chunk.chunk_index);
        validate_shot(&shot)?;
        result.push(shot);
    }
    Ok(result)
}

fn validate_shot(shot: &[OperationTraceChunk]) -> Result<(), DemBuildError> {
    let Some(first) = shot.first() else {
        return Err(preparation("empty runtime trace"));
    };
    for (index, chunk) in shot.iter().enumerate() {
        if chunk.format != "pecos_qis_operation_trace_v1"
            || chunk.engine_trace_id != first.engine_trace_id
            || chunk.shot_index != first.shot_index
            || chunk.chunk_index != index
            || chunk.num_operations != chunk.operations.len()
            || !chunk.lowered_quantum_ops_complete
        {
            return Err(preparation(
                "runtime trace is incomplete or has inconsistent framing",
            ));
        }
        if (chunk.stage == "trace_complete") != (index + 1 == shot.len()) {
            return Err(preparation(
                "runtime trace must end with exactly one trace_complete chunk",
            ));
        }
    }
    let terminal = &shot[shot.len() - 1];
    if !terminal.operations.is_empty()
        || !terminal.lowered_quantum_ops.is_empty()
        || !terminal.named_result_traces.is_empty()
        || !terminal.named_results.is_empty()
    {
        return Err(preparation(
            "terminal trace chunk must contain no operations or named outputs",
        ));
    }
    if shot
        .iter()
        .filter(|chunk| !chunk.named_results.is_empty())
        .count()
        > 1
    {
        return Err(preparation(
            "runtime trace carries complete named results in more than one chunk",
        ));
    }
    Ok(())
}

fn source_ids(shot: &[OperationTraceChunk]) -> Result<Vec<usize>, DemBuildError> {
    let ids: Vec<_> = shot
        .iter()
        .flat_map(|chunk| &chunk.operations)
        .filter_map(|operation| match operation {
            Operation::Quantum(QuantumOp::Measure(_, id) | QuantumOp::MeasureLeaked(_, id)) => {
                Some(*id)
            }
            _ => None,
        })
        .collect();
    if ids.iter().collect::<BTreeSet<_>>().len() != ids.len() {
        return Err(preparation("source trace repeats measurement IDs"));
    }
    Ok(ids)
}

/// Replay only runtime-lowered operations; never reconstruct measurement IDs
/// from gate positions or silently fall back to the source operation stream.
fn replay(
    shot: &[OperationTraceChunk],
    qubits: usize,
    source: &[usize],
) -> Result<TickCircuit, DemBuildError> {
    validate_shot(shot)?;
    let mut circuit = TickCircuit::new();
    let mut runtime_ids = BTreeSet::new();
    for trace in shot.iter().flat_map(|chunk| &chunk.lowered_quantum_ops) {
        // Same gate set as the Python replay chain in pecos/_qis_trace_replay.py.
        let kind: GateType = match trace.gate_type.as_str() {
            "MeasureLeaked" => GateType::MZ,
            name @ ("H"
            | "X"
            | "Y"
            | "Z"
            | "SZ"
            | "SZdg"
            | "T"
            | "Tdg"
            | "PZ"
            | "Idle"
            | "MZ"
            | "MeasCrosstalkGlobalPayload"
            | "MeasCrosstalkLocalPayload"
            | "RX"
            | "RY"
            | "RZ"
            | "RXY1Q"
            | "R1XY"
            | "CX"
            | "CY"
            | "CZ"
            | "CH"
            | "CRZ"
            | "SZZ"
            | "SZZdg"
            | "RZZ"
            | "RXYXY2Q"
            | "CCX") => name.parse().map_err(preparation)?,
            name => {
                return Err(preparation(format!(
                    "unsupported lowered traced gate {name:?}"
                )));
            }
        };
        if trace
            .angles
            .iter()
            .chain(&trace.params)
            .any(|x| !x.is_finite())
            || trace.qubits.is_empty()
            || trace.qubits.iter().any(|&q| q >= qubits)
            || kind.quantum_arity() == 0
            || !trace.qubits.len().is_multiple_of(kind.quantum_arity())
        {
            return Err(preparation("lowered gate has invalid parameters or qubits"));
        }
        let mut params = trace.params.clone();
        if kind == GateType::Idle {
            if params.len() != 1 || params[0] < 0.0 || !(params[0] * 1e9).is_finite() {
                return Err(preparation(
                    "lowered idle duration must be finite and non-negative",
                ));
            }
            params[0] = if params[0] == 0.0 {
                0.0
            } else {
                (params[0] * 1e9).round_ties_even().max(1.0)
            };
        } else if !params.is_empty() {
            return Err(preparation(
                "unexpected non-angle parameters on a lowered gate",
            ));
        }
        let mut gate = Gate::try_new(
            kind,
            trace
                .angles
                .iter()
                .map(|&a| Angle64::try_from_radians(a))
                .collect::<Result<Vec<_>, _>>()
                .map_err(preparation)?,
            params,
            trace
                .qubits
                .iter()
                .copied()
                .map(QubitId)
                .collect::<Vec<_>>(),
        )
        .map_err(preparation)?;
        if kind == GateType::MZ {
            if trace.measurement_result_ids.len() != trace.qubits.len() {
                return Err(preparation(
                    "lowered measurement is missing stable measurement IDs",
                ));
            }
            for &id in &trace.measurement_result_ids {
                if !runtime_ids.insert(id) {
                    return Err(preparation("lowered trace repeats measurement IDs"));
                }
            }
            gate.meas_ids = trace
                .measurement_result_ids
                .iter()
                .copied()
                .map(MeasId::from_raw)
                .collect();
        } else if !trace.measurement_result_ids.is_empty() {
            return Err(preparation("non-measurement gate carries measurement IDs"));
        }
        circuit
            .tick()
            .try_add_gate(gate)
            .map_err(preparation)?
            .metas(
                trace
                    .metadata
                    .iter()
                    .map(|(key, value)| (key.clone(), Attribute::String(value.clone())))
                    .collect(),
            );
    }
    if runtime_ids != source.iter().copied().collect() {
        return Err(preparation(
            "source and runtime-lowered measurement IDs differ",
        ));
    }
    circuit.compact_ticks();
    pass::lower_clifford_rotations(&mut circuit);
    pass::simplify_single_qubit_clifford_chains(&mut circuit);
    circuit.set_meta(
        "qis_source_measurement_ids",
        Attribute::String(serde_json::to_string(&source).map_err(preparation)?),
    );
    Ok(circuit)
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
    if outcomes.len() != ids.len() {
        return Err(preparation(
            "physical outcomes do not cover all source measurements",
        ));
    }
    ids.iter()
        .map(|id| match outcomes.get(id) {
            Some(&bit @ (0 | 1)) => u8::try_from(bit).map_err(preparation),
            _ => Err(preparation(
                "missing or non-Boolean physical measurement outcome",
            )),
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

impl DemProgram for SeleneDemProgram {
    fn prepare(&mut self, request: &DemProgramRequest<'_>) -> Result<DemBuildInput, DemBuildError> {
        if request.inference.is_none() && self.schedule_shots == 0 {
            return Err(preparation("schedule validation shots must be positive"));
        }
        let program = match &self.source {
            ProgramSource::Qis(program) => program.clone(),
            #[cfg(feature = "hugr")]
            ProgramSource::Hugr(bytes) => super::dem_hugr::lower(bytes)?,
        };
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
            .program_ir(source_ir);
        let baseline = capture(&engine, request.num_qubits, 1, request.seed)?.remove(0);
        let source = source_ids(&baseline)?;
        let mut circuit = replay(&baseline, request.num_qubits, &source)?;
        let Some(settings) = request.inference else {
            let checks = capture(
                &engine,
                request.num_qubits,
                self.schedule_shots,
                request.seed.wrapping_add(1),
            )?;
            for shot in &checks {
                check_schedule(&baseline, &source, shot)?;
            }
            circuit.set_meta(
                "dem_schedule_validation",
                Attribute::String("empirical_trace_comparison".into()),
            );
            circuit.set_meta(
                "dem_schedule_validation_shots",
                Attribute::String(self.schedule_shots.to_string()),
            );
            return Ok(DemBuildInput {
                circuit,
                probes: None,
            });
        };
        let baseline_rows = named_rows(&baseline);
        let raw = raw_row(&baseline, &baseline_rows, &settings.raw_tag)?;
        if raw.len() != source.len() {
            return Err(preparation(
                "raw output must expose every physical measurement exactly once",
            ));
        }
        let raw_traces: Vec<_> = baseline
            .iter()
            .flat_map(|c| &c.named_result_traces)
            .filter(|r| r.name == settings.raw_tag)
            .collect();
        let direct = raw_traces
            .iter()
            .all(|r| r.result_ids.len() == r.values.len());
        let mut ids: Vec<_> = raw_traces
            .iter()
            .flat_map(|r| r.result_ids.iter().copied())
            .collect();
        if direct && ids != source {
            return Err(preparation(
                "direct raw measurement IDs must preserve certified source order",
            ));
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
        let traces = capture(
            &engine,
            request.num_qubits,
            shots,
            request.seed.wrapping_add(1),
        )?;
        for shot in &traces {
            check_schedule(&baseline, &source, shot)?;
        }
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
            if row.len() != raw.len() {
                return Err(preparation("raw output width changed across probes"));
            }
            if settings.require_raw_provenance
                && row != physical_row(shot, &data.raw_measurement_ids)?
            {
                return Err(preparation(
                    "raw measurement binding changed across inference probes",
                ));
            }
            data.raw_rows.push(row);
            data.detector_rows
                .push(selected_row(rows, &detector_outputs)?);
            data.observable_rows
                .push(selected_row(rows, &observable_outputs)?);
        }
        circuit.set_meta(
            "dem_schedule_validation",
            Attribute::String("empirical_trace_comparison".into()),
        );
        circuit.set_meta(
            "dem_schedule_validation_shots",
            Attribute::String(shots.to_string()),
        );
        circuit.set_meta(
            "dem_raw_measurement_binding",
            Attribute::String(binding.into()),
        );
        Ok(DemBuildInput {
            circuit,
            probes: Some(data),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_qis::LoweredQuantumGateTrace;

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
            let error = replay(&shot, 1, &[7]).unwrap_err().to_string();
            assert!(
                error.contains(name) && error.contains("unsupported lowered traced gate"),
                "{error}"
            );
        }
    }

    #[test]
    fn replay_preserves_stable_measurement_identity() {
        let circuit = replay(&trace(), 1, &[7]).unwrap();
        let ids: Vec<_> = circuit
            .iter_gate_batches()
            .flat_map(|gate| gate.as_gate().meas_ids.clone())
            .collect();
        assert_eq!(ids, vec![MeasId::from_raw(7)]);
        assert_eq!(physical_row(&trace(), &[7]).unwrap(), [1]);
    }

    #[test]
    fn replay_rejects_partial_traces_and_missing_identity() {
        let mut shot = trace();
        shot.pop();
        assert!(replay(&shot, 1, &[7]).is_err());
        let mut shot = trace();
        shot[0].lowered_quantum_ops_complete = false;
        assert!(replay(&shot, 1, &[7]).is_err());
        let mut shot = trace();
        shot[0].lowered_quantum_ops[0]
            .measurement_result_ids
            .clear();
        assert!(replay(&shot, 1, &[7]).is_err());
        let mut shot = trace();
        shot[0].lowered_quantum_ops[0].measurement_result_ids = vec![8];
        assert!(replay(&shot, 1, &[7]).is_err());
        let mut shot = trace();
        shot[1].chunk_index = 2;
        assert!(replay(&shot, 1, &[7]).is_err());
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
        let circuit = replay(&shot, 1, &[7]).unwrap();
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
        for (qubit, angle) in [(1, 0.0), (0, f64::NAN)] {
            let mut shot = trace();
            shot[0].lowered_quantum_ops[0].qubits = vec![qubit];
            shot[0].lowered_quantum_ops[0].angles = vec![angle];
            assert!(replay(&shot, 1, &[7]).is_err());
        }
    }
}
