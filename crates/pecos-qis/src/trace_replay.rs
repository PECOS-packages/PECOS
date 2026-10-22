// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

//! Audited runtime QIS trace replay, independent of LLVM.
//!
//! Stored chunks have owned strings; the runtime's in-memory structs stay in
//! `ccengine`. Both producers serialize the same fields. Older Python fixtures
//! omit capture diagnostics, named outputs and empty gate fields.
//!
//! | Wire fields | Required / default |
//! | --- | --- |
//! | `format`, `engine_trace_id`, `shot_index`, `chunk_index`, `stage` | Required |
//! | `num_operations`, `operations`, `lowered_quantum_ops` | Required |
//! | `lowered_quantum_ops_complete` | Required on every chunk |
//! | `runtime_lowering_deferred` | false |
//! | `runtime_events_complete` | true |
//! | `named_result_traces` | `[]` |
//! | `named_results`, `measurement_results` | `{}` |
//! | `waiting_for_result_id`, `current_shot_seed` | `null` |
//! | `simulated_op_count` | 0 |
//! | gate: `gate_type`, `qubits` | Required |
//! | gate: `angles`, `params`, `measurement_result_ids` | `[]` |
//! | gate: `metadata` | `{}` |
//!
//! Missing measurement IDs still fail replay for measurements. Unknown fields
//! are ignored, as with serde's normal struct decoding. JSON metadata is kept
//! typed, including nested values via `Attribute::Json`.
//! JSON integer tokens must fit i64 or u64, including inside nested metadata.
//! Raw JSON values preserve tokens until that check, preventing float coercion.
//! Bookkeeping-only completed traces are valid. Quantum operations and runtime requests
//! must contain lowered gates or attest deferred lowering in a fully drained stream.

use crate::OperationTraceChunk;
use pecos_core::{Angle64, Gate, MeasId, QubitId, gate_type::GateType};
use pecos_qis_ffi_types::{
    NamedResult, NamedResultTrace, Operation, QuantumOp, RuntimeRequest, SeleneRequest,
};
use pecos_quantum::{Attribute, TickCircuit};
use serde::{Deserialize, Deserializer, de::DeserializeOwned};
use serde_json::{Map, Value, value::RawValue};
use std::collections::{BTreeMap, BTreeSet};

/// Invalid stored data or an uncertifiable runtime gate stream.
#[derive(Debug, Clone)]
pub struct TraceReplayError(String);

impl std::fmt::Display for TraceReplayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TraceReplayError {}

fn replay_error(error: impl std::fmt::Display) -> TraceReplayError {
    TraceReplayError(error.to_string())
}

fn field<T: DeserializeOwned>(
    fields: &mut BTreeMap<String, Box<RawValue>>,
    name: &str,
    default: Option<Value>,
) -> Result<T, String> {
    match fields.remove(name) {
        Some(value) => serde_json::from_str(value.get()),
        None => match default {
            Some(value) => serde_json::from_value(value),
            None => return Err(format!("field {name}: missing required field")),
        },
    }
    .map_err(|error| format!("field {name}: {error}"))
}

fn metadata_fields(
    fields: &mut BTreeMap<String, Box<RawValue>>,
) -> Result<BTreeMap<String, Value>, String> {
    let values: BTreeMap<String, Box<RawValue>> =
        field(fields, "metadata", Some(Value::Object(Map::new())))?;
    values
        .into_iter()
        .map(|(key, raw)| {
            metadata_value(&raw)
                .map(|value| (key.clone(), value))
                .map_err(|error| format!("field metadata key {key:?}: {error}"))
        })
        .collect()
}

fn metadata_value(raw: &RawValue) -> Result<Value, String> {
    let token = raw.get().trim();
    match token.as_bytes().first() {
        Some(b'{') => {
            let fields: BTreeMap<String, Box<RawValue>> =
                serde_json::from_str(token).map_err(|error| error.to_string())?;
            fields
                .into_iter()
                .map(|(key, value)| metadata_value(&value).map(|value| (key, value)))
                .collect::<Result<Map<_, _>, _>>()
                .map(Value::Object)
        }
        Some(b'[') => {
            let values: Vec<Box<RawValue>> =
                serde_json::from_str(token).map_err(|error| error.to_string())?;
            values
                .iter()
                .map(|value| metadata_value(value))
                .collect::<Result<Vec<_>, _>>()
                .map(Value::Array)
        }
        Some(b'-' | b'0'..=b'9') if !token.contains(['.', 'e', 'E']) => {
            if let Ok(value) = token.parse::<i64>() {
                Ok(Value::from(value))
            } else if let Ok(value) = token.parse::<u64>() {
                Ok(Value::from(value))
            } else {
                Err("integer cannot be represented losslessly as i64 or u64".into())
            }
        }
        _ => serde_json::from_str(token).map_err(|error| error.to_string()),
    }
}

/// Owned gate payload in a stored trace.
#[derive(Debug, Clone)]
pub struct StoredGate {
    pub gate_type: String,
    pub angles: Vec<f64>,
    pub params: Vec<f64>,
    pub qubits: Vec<usize>,
    pub measurement_result_ids: Vec<usize>,
    pub metadata: BTreeMap<String, Value>,
}

impl<'de> Deserialize<'de> for StoredGate {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut fields = BTreeMap::<String, Box<RawValue>>::deserialize(deserializer)?;
        let mut parse = || -> Result<Self, String> {
            Ok(Self {
                gate_type: field(&mut fields, "gate_type", None)?,
                qubits: field(&mut fields, "qubits", None)?,
                angles: field(&mut fields, "angles", Some(Value::Array(vec![])))?,
                params: field(&mut fields, "params", Some(Value::Array(vec![])))?,
                measurement_result_ids: field(
                    &mut fields,
                    "measurement_result_ids",
                    Some(Value::Array(vec![])),
                )?,
                metadata: metadata_fields(&mut fields)?,
            })
        };
        parse().map_err(serde::de::Error::custom)
    }
}

/// Owned wire representation of a captured chunk.
#[derive(Debug, Clone)]
pub struct StoredTraceChunk {
    pub format: String,
    pub engine_trace_id: u64,
    pub shot_index: usize,
    pub chunk_index: usize,
    pub stage: String,
    pub waiting_for_result_id: Option<u64>,
    pub current_shot_seed: Option<u64>,
    pub simulated_op_count: usize,
    pub num_operations: usize,
    pub operations: Vec<Operation>,
    pub lowered_quantum_ops: Vec<StoredGate>,
    pub lowered_quantum_ops_complete: bool,
    pub runtime_lowering_deferred: bool,
    pub runtime_events_complete: bool,
    pub named_result_traces: Vec<NamedResultTrace>,
    pub named_results: BTreeMap<String, NamedResult>,
    pub measurement_results: BTreeMap<usize, u32>,
}

impl<'de> Deserialize<'de> for StoredTraceChunk {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut fields = BTreeMap::<String, Box<RawValue>>::deserialize(deserializer)?;
        let mut parse = || -> Result<Self, String> {
            Ok(Self {
                format: field(&mut fields, "format", None)?,
                engine_trace_id: field(&mut fields, "engine_trace_id", None)?,
                shot_index: field(&mut fields, "shot_index", None)?,
                chunk_index: field(&mut fields, "chunk_index", None)?,
                stage: field(&mut fields, "stage", None)?,
                num_operations: field(&mut fields, "num_operations", None)?,
                operations: field(&mut fields, "operations", None)?,
                lowered_quantum_ops: {
                    let gates: Vec<Box<RawValue>> =
                        field(&mut fields, "lowered_quantum_ops", None)?;
                    gates
                        .iter()
                        .enumerate()
                        .map(|(index, gate)| {
                            serde_json::from_str(gate.get()).map_err(|error| {
                                format!("field lowered_quantum_ops gate {index}: {error}")
                            })
                        })
                        .collect::<Result<_, _>>()?
                },
                lowered_quantum_ops_complete: field(
                    &mut fields,
                    "lowered_quantum_ops_complete",
                    None,
                )?,
                runtime_lowering_deferred: field(
                    &mut fields,
                    "runtime_lowering_deferred",
                    Some(Value::Bool(false)),
                )?,
                runtime_events_complete: field(
                    &mut fields,
                    "runtime_events_complete",
                    Some(Value::Bool(true)),
                )?,
                waiting_for_result_id: field(
                    &mut fields,
                    "waiting_for_result_id",
                    Some(Value::Null),
                )?,
                current_shot_seed: field(&mut fields, "current_shot_seed", Some(Value::Null))?,
                simulated_op_count: field(&mut fields, "simulated_op_count", Some(Value::from(0)))?,
                named_result_traces: field(
                    &mut fields,
                    "named_result_traces",
                    Some(Value::Array(vec![])),
                )?,
                named_results: field(
                    &mut fields,
                    "named_results",
                    Some(Value::Object(Map::new())),
                )?,
                measurement_results: field(
                    &mut fields,
                    "measurement_results",
                    Some(Value::Object(Map::new())),
                )?,
            })
        };
        parse().map_err(serde::de::Error::custom)
    }
}

impl From<&OperationTraceChunk> for StoredTraceChunk {
    fn from(chunk: &OperationTraceChunk) -> Self {
        Self {
            format: chunk.format.to_owned(),
            engine_trace_id: chunk.engine_trace_id,
            shot_index: chunk.shot_index,
            chunk_index: chunk.chunk_index,
            stage: chunk.stage.clone(),
            num_operations: chunk.num_operations,
            operations: chunk.operations.clone(),
            lowered_quantum_ops: chunk
                .lowered_quantum_ops
                .iter()
                .map(|gate| StoredGate {
                    gate_type: gate.gate_type.clone(),
                    angles: gate.angles.clone(),
                    params: gate.params.clone(),
                    qubits: gate.qubits.clone(),
                    measurement_result_ids: gate.measurement_result_ids.clone(),
                    metadata: gate
                        .metadata
                        .iter()
                        .map(|(key, value)| (key.clone(), Value::String(value.clone())))
                        .collect(),
                })
                .collect(),
            lowered_quantum_ops_complete: chunk.lowered_quantum_ops_complete,
            runtime_lowering_deferred: chunk.runtime_lowering_deferred,
            runtime_events_complete: chunk.runtime_events_complete,
            waiting_for_result_id: chunk.waiting_for_result_id,
            current_shot_seed: chunk.current_shot_seed,
            simulated_op_count: chunk.simulated_op_count,
            named_result_traces: chunk.named_result_traces.clone(),
            named_results: chunk.named_results.clone(),
            measurement_results: chunk.measurement_results.clone(),
        }
    }
}

/// Decode stored chunks with chunk and field context on conversion errors.
///
/// # Errors
/// Returns an error for malformed JSON or incorrectly typed fields.
pub fn parse_trace(json: &str) -> Result<Vec<StoredTraceChunk>, TraceReplayError> {
    let values: Vec<Box<RawValue>> = serde_json::from_str(json).map_err(replay_error)?;
    values
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            serde_json::from_str(value.get())
                .map_err(|error| replay_error(format!("chunk {index}: {error}")))
        })
        .collect()
}

/// Replay in-memory chunks without a JSON round trip.
///
/// # Errors
/// Returns an error if the stream cannot be certified.
pub fn replay_in_memory(
    shot: &[OperationTraceChunk],
    qubits: usize,
) -> Result<TickCircuit, TraceReplayError> {
    replay(
        &shot.iter().map(StoredTraceChunk::from).collect::<Vec<_>>(),
        Some(qubits),
        None,
    )
}

/// Replay runtime chunks and validate hosted metadata after compaction.
///
/// # Errors
/// Returns replay or hosted validation errors.
pub fn replay_in_memory_with_hosted_options(
    shot: &[OperationTraceChunk],
    qubits: usize,
    hosted: &HostedReplayOptions,
    context: &str,
) -> Result<TickCircuit, TraceReplayError> {
    replay_with_hosted_options(
        &shot.iter().map(StoredTraceChunk::from).collect::<Vec<_>>(),
        Some(qubits),
        None,
        hosted,
        context,
    )
}

/// Extract named result provenance in emission order.
#[must_use]
pub fn named_result_traces(shot: &[StoredTraceChunk]) -> Vec<NamedResultTrace> {
    shot.iter()
        .flat_map(|chunk| chunk.named_result_traces.iter().cloned())
        .collect()
}

fn metadata_attribute(value: &Value) -> Attribute {
    match value {
        Value::String(value) => Attribute::String(value.clone()),
        Value::Bool(value) => Attribute::Bool(*value),
        Value::Number(number) => match (number.as_i64(), number.as_u64(), number.as_f64()) {
            (Some(integer), _, _) => Attribute::Int(integer),
            (None, None, Some(float)) => Attribute::Float(float),
            (None, Some(_), _) | (None, None, None) => Attribute::Json(value.clone()),
        },
        value => Attribute::Json(value.clone()),
    }
}

/// Validate audited framing and lowering attestations.
///
/// # Errors
/// Rejects incomplete or inconsistent streams.
pub fn validate_shot(shot: &[StoredTraceChunk]) -> Result<(), TraceReplayError> {
    let Some(first) = shot.first() else {
        return Err(replay_error("empty runtime trace"));
    };
    for (index, chunk) in shot.iter().enumerate() {
        if chunk.format != "pecos_qis_operation_trace_v1"
            || chunk.engine_trace_id != first.engine_trace_id
            || chunk.shot_index != first.shot_index
            || chunk.chunk_index != index
            || chunk.num_operations != chunk.operations.len()
            || !chunk.lowered_quantum_ops_complete
        {
            return Err(replay_error(format!(
                "runtime trace chunk {index} is incomplete or has inconsistent framing: format={:?}, engine={}, shot={}, chunk={}, num_operations={}, operations={}, lowered_quantum_ops_complete={}",
                chunk.format,
                chunk.engine_trace_id,
                chunk.shot_index,
                chunk.chunk_index,
                chunk.num_operations,
                chunk.operations.len(),
                chunk.lowered_quantum_ops_complete
            )));
        }
        if !chunk.runtime_events_complete {
            return Err(replay_error(format!(
                "runtime trace chunk {index} captured unhandled runtime custom events; configure a runtime batch adapter"
            )));
        }
        if chunk
            .operations
            .iter()
            .any(|op| matches!(op, Operation::Quantum(_) | Operation::RuntimeRequest { .. }))
            && chunk.lowered_quantum_ops.is_empty()
            && !chunk.runtime_lowering_deferred
        {
            return Err(replay_error(format!(
                "runtime trace chunk {index} carries lowerable quantum operations but no lowered_quantum_ops"
            )));
        }
        if (chunk.stage == "trace_complete") != (index + 1 == shot.len()) {
            return Err(replay_error(
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
        return Err(replay_error(
            "terminal trace chunk must contain no operations or named outputs",
        ));
    }
    if shot
        .iter()
        .filter(|chunk| !chunk.named_results.is_empty())
        .count()
        > 1
    {
        return Err(replay_error(
            "runtime trace carries complete named results in more than one chunk",
        ));
    }
    Ok(())
}

/// Extract unique source measurement IDs in execution order.
///
/// # Errors
/// Rejects repeated IDs.
pub fn source_ids(shot: &[StoredTraceChunk]) -> Result<Vec<usize>, TraceReplayError> {
    let ids: Vec<_> = shot
        .iter()
        .flat_map(|chunk| &chunk.operations)
        .filter_map(|operation| match operation {
            Operation::Quantum(QuantumOp::Measure(_, id) | QuantumOp::MeasureLeaked(_, id))
            | Operation::RuntimeRequest {
                request: RuntimeRequest::Native(SeleneRequest::Measure { result_id: id, .. }),
                ..
            } => Some(*id),
            _ => None,
        })
        .collect();
    if ids.iter().collect::<BTreeSet<_>>().len() != ids.len() {
        return Err(replay_error("source trace repeats measurement IDs"));
    }
    Ok(ids)
}

/// Optional hosted checks on the compacted replay circuit.
#[derive(Clone, Debug, Default)]
pub struct HostedReplayOptions {
    pub require_hosted_operation_order: bool,
    pub max_hosted_tick_separation: Option<u64>,
}

impl HostedReplayOptions {
    /// Validate after replay, or after evidence capture for inference callers.
    ///
    /// # Errors
    /// Returns a hosted metadata violation when checks are requested.
    pub fn validate(&self, circuit: &TickCircuit, context: &str) -> Result<(), TraceReplayError> {
        if self.require_hosted_operation_order || self.max_hosted_tick_separation.is_some() {
            pecos_quantum::hosted::validate_hosted_circuit(
                circuit,
                "local_role",
                &pecos_quantum::hosted::HostedOperationOptions {
                    require_host_after_local: self.require_hosted_operation_order,
                    max_tick_separation: self.max_hosted_tick_separation,
                    require_unique_host_id: true,
                    context,
                    ..Default::default()
                },
            )
            .map_err(replay_error)?;
        }
        Ok(())
    }
}

/// Replay and validate hosted metadata after compaction.
///
/// # Errors
/// Returns replay or hosted validation errors.
pub fn replay_with_hosted_options(
    shot: &[StoredTraceChunk],
    qubits: Option<usize>,
    measurement_crosstalk_topology: Option<&str>,
    hosted: &HostedReplayOptions,
    context: &str,
) -> Result<TickCircuit, TraceReplayError> {
    let circuit = replay(shot, qubits, measurement_crosstalk_topology)?;
    hosted.validate(&circuit, context)?;
    Ok(circuit)
}

/// Replay only runtime-lowered operations; never reconstruct measurement IDs
/// from gate positions or silently fall back to the source operation stream.
///
/// # Errors
/// Rejects invalid framing, gates, provenance, or crosstalk options.
pub fn replay(
    shot: &[StoredTraceChunk],
    qubits: Option<usize>,
    measurement_crosstalk_topology: Option<&str>,
) -> Result<TickCircuit, TraceReplayError> {
    let global = match measurement_crosstalk_topology {
        None | Some("none" | "runtime_payloads") => false,
        Some("global_from_measurements") => true,
        Some(_) => {
            return Err(replay_error(
                "measurement_crosstalk_topology must be None, 'runtime_payloads', or 'global_from_measurements'",
            ));
        }
    };
    validate_shot(shot)?;
    let source = source_ids(shot)?;
    let qubits = match qubits {
        Some(bound) => bound,
        None => shot
            .iter()
            .flat_map(|chunk| &chunk.lowered_quantum_ops)
            .flat_map(|gate| &gate.qubits)
            .max()
            .map_or(Ok(0), |max| {
                max.checked_add(1)
                    .ok_or_else(|| replay_error("qubits: qubit bound overflows usize"))
            })?,
    };
    let mut circuit = TickCircuit::new();
    let mut runtime_ids = BTreeSet::new();
    for trace in shot.iter().flat_map(|chunk| &chunk.lowered_quantum_ops) {
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
            | "CCX") => name.parse().map_err(replay_error)?,
            name => {
                return Err(replay_error(format!(
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
            return Err(replay_error(format!(
                "lowered gate {:?} has invalid angles {:?}, parameters {:?}, or qubits {:?} (num_qubits={qubits})",
                trace.gate_type, trace.angles, trace.params, trace.qubits
            )));
        }
        let mut params = trace.params.clone();
        if kind == GateType::Idle {
            if params.len() != 1
                || params[0] < 0.0
                || !(params[0] * 1e9).is_finite()
                || (params[0] * 1e9).round_ties_even() >= 2.0_f64.powi(64)
            {
                return Err(replay_error(format!(
                    "lowered Idle duration must be finite and non-negative and fit in u64 nanoseconds; params={params:?}"
                )));
            }
            params[0] = if params[0] == 0.0 {
                0.0
            } else {
                (params[0] * 1e9).round_ties_even().max(1.0)
            };
        } else if !params.is_empty() {
            return Err(replay_error(format!(
                "unexpected non-angle parameters {params:?} on lowered gate {:?}",
                trace.gate_type
            )));
        }
        let mut gate = Gate::try_new(
            kind,
            trace
                .angles
                .iter()
                .map(|&a| Angle64::try_from_radians(a))
                .collect::<Result<Vec<_>, _>>()
                .map_err(replay_error)?,
            params,
            trace
                .qubits
                .iter()
                .copied()
                .map(QubitId)
                .collect::<Vec<_>>(),
        )
        .map_err(|error| {
            replay_error(format!(
                "lowered gate {:?} with angles {:?}, params {:?}, qubits {:?}: {error}",
                trace.gate_type, trace.angles, trace.params, trace.qubits
            ))
        })?;
        if kind == GateType::MZ {
            if trace.measurement_result_ids.len() != trace.qubits.len() {
                return Err(replay_error(format!(
                    "lowered {:?} measurement carries IDs {:?} for qubits {:?}",
                    trace.gate_type, trace.measurement_result_ids, trace.qubits
                )));
            }
            for &id in &trace.measurement_result_ids {
                if !runtime_ids.insert(id) {
                    return Err(replay_error(format!(
                        "lowered trace repeats measurement ID {id}"
                    )));
                }
            }
            gate.meas_ids = trace
                .measurement_result_ids
                .iter()
                .copied()
                .map(MeasId::from_raw)
                .collect();
        } else if !trace.measurement_result_ids.is_empty() {
            return Err(replay_error("non-measurement gate carries measurement IDs"));
        }
        if global && kind == GateType::MZ {
            circuit
                .tick()
                .try_add_gate(
                    Gate::try_new(
                        GateType::MeasCrosstalkGlobalPayload,
                        vec![],
                        vec![],
                        gate.qubits.to_vec(),
                    )
                    .map_err(replay_error)?,
                )
                .map_err(replay_error)?;
        }
        circuit
            .tick()
            .try_add_gate(gate)
            .map_err(replay_error)?
            .metas(
                trace
                    .metadata
                    .iter()
                    .map(|(key, value)| (key.clone(), metadata_attribute(value)))
                    .collect(),
            );
    }
    if runtime_ids != source.iter().copied().collect() {
        return Err(replay_error(format!(
            "source and runtime-lowered measurement IDs differ: source ids={source:?}, runtime ids={runtime_ids:?}"
        )));
    }
    // External measurement IDs must also reserve their successors, just as the
    // TickCircuit measurement builders do. Generic gate insertion does not
    // allocate measurement records.
    if let Some(highest) = runtime_ids.last() {
        let next = highest.checked_add(1).ok_or_else(|| {
            replay_error(format!(
                "measurement ID {highest} leaves no room for a later ID"
            ))
        })?;
        circuit
            .try_advance_meas_counter(next)
            .map_err(replay_error)?;
    }
    circuit.compact_ticks();
    circuit.set_meta(
        "qis_source_measurement_ids",
        Attribute::String(serde_json::to_string(&source).map_err(replay_error)?),
    );
    Ok(circuit)
}

#[cfg(test)]
mod tests;
