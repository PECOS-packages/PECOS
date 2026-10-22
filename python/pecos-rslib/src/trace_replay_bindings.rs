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

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_replay_qis_trace, m)?)?;
    m.add_function(wrap_pyfunction!(_qis_named_result_traces, m)?)?;
    m.add_function(wrap_pyfunction!(_qis_source_measurement_ids, m)?)?;
    Ok(())
}
