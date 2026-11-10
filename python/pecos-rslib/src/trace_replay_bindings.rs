// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

//! Stored QIS trace replay without LLVM.

use crate::dag_circuit_bindings::PyTickCircuit;
use pecos_qis::trace_replay::{self, StoredTraceChunk};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

fn stored_trace(chunks: &Bound<'_, PyList>) -> PyResult<Vec<StoredTraceChunk>> {
    let py = chunks.py();
    let kwargs = PyDict::new(py);
    kwargs.set_item("allow_nan", false)?;
    let json: String = py
        .import("json")?
        .call_method("dumps", (chunks,), Some(&kwargs))?
        .extract()?;
    trace_replay::parse_trace(&json).map_err(|error| PyValueError::new_err(error.to_string()))
}

#[pyfunction]
#[pyo3(signature = (chunks, measurement_crosstalk_topology = None))]
fn _replay_qis_trace(
    chunks: &Bound<'_, PyList>,
    measurement_crosstalk_topology: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyTickCircuit> {
    let topology = measurement_crosstalk_topology
        .map(pyo3::types::PyAnyMethods::extract::<String>)
        .transpose()
        .map_err(|_| PyValueError::new_err(
            "measurement_crosstalk_topology must be None, 'runtime_payloads', or 'global_from_measurements'",
        ))?;
    let shot = stored_trace(chunks)?;
    let inner = trace_replay::replay(&shot, None, topology.as_deref())
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyTickCircuit { inner })
}

#[pyfunction]
fn _qis_named_result_traces(chunks: &Bound<'_, PyList>) -> PyResult<Py<PyAny>> {
    let shot = stored_trace(chunks)?;
    let json = serde_json::to_string(&trace_replay::named_result_traces(&shot))
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(chunks
        .py()
        .import("json")?
        .call_method1("loads", (json,))?
        .unbind())
}

#[pyfunction]
fn _qis_source_measurement_ids(chunks: &Bound<'_, PyList>) -> PyResult<Vec<usize>> {
    let shot = stored_trace(chunks)?;
    trace_replay::source_ids(&shot).map_err(|error| PyValueError::new_err(error.to_string()))
}

type HostedRecord = (usize, usize, String, Vec<usize>, String, String);

#[pyfunction]
#[pyo3(signature = (records, max_tick_separation, require_shared_qubit, require_host_after_local, require_unique_host_id, host_id_key, context))]
fn _validate_hosted_operations(
    records: Vec<HostedRecord>,
    max_tick_separation: Option<&Bound<'_, PyAny>>,
    require_shared_qubit: bool,
    require_host_after_local: bool,
    require_unique_host_id: bool,
    host_id_key: &str,
    context: &str,
) -> PyResult<Vec<(usize, usize, i128)>> {
    use pecos_quantum::hosted::{
        HostedGateRecord, HostedOperationOptions, validate_hosted_operations,
    };
    let records: Vec<_> = records
        .into_iter()
        .map(
            |(tick_index, gate_index, gate_name, qubits, host_id, local_role)| HostedGateRecord {
                tick_index,
                gate_index,
                gate_name,
                qubits,
                host_id,
                local_role,
            },
        )
        .collect();
    let max_tick_separation = max_tick_separation
        .map(|value| {
            crate::pauli_bindings::extract_unsigned_index(
                value,
                format!("{context}: max_tick_separation"),
            )
        })
        .transpose()?;
    let options = HostedOperationOptions {
        max_tick_separation,
        require_shared_qubit,
        require_host_after_local,
        require_unique_host_id,
        host_id_key,
        context,
    };
    validate_hosted_operations(&records, &options)
        .map(|bindings| {
            bindings
                .into_iter()
                .map(|b| (b.local, b.host, b.tick_separation))
                .collect()
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_validate_hosted_operations, m)?)?;
    m.add_function(wrap_pyfunction!(_replay_qis_trace, m)?)?;
    m.add_function(wrap_pyfunction!(_qis_named_result_traces, m)?)?;
    m.add_function(wrap_pyfunction!(_qis_source_measurement_ids, m)?)?;
    Ok(())
}
