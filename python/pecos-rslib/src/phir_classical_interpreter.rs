// Copyright 2024 The PECOS Developers
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

//! PyO3 wrapper for the Rust `PhirClassicalInterpreter`.
//!
//! Exposes the Rust classical interpreter to Python as a drop-in replacement
//! for `pecos.classical_interpreters.PhirClassicalInterpreter`.

use pecos_phir_json::v0_1::ast::{Operation, PHIRProgram};
use pecos_phir_json::v0_1::classical_interpreter::{
    MeasKey, PhirClassicalInterpreter as RustInterpreter, QOpArgs, ResultValue, YieldedOp,
};
use pecos_phir_json::v0_1::environment::DataType;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

/// Python-exposed classical interpreter backed by Rust.
///
/// Drop-in replacement for `pecos.classical_interpreters.PhirClassicalInterpreter`.
#[pyclass(name = "RustPhirClassicalInterpreter", module = "pecos_rslib")]
pub struct PyPhirClassicalInterpreter {
    inner: Arc<Mutex<RustInterpreter>>,
    /// Cached program JSON for re-parsing during iteration
    program_json: Option<String>,
    /// Whether to validate PHIR
    #[pyo3(get, set)]
    phir_validate: bool,
}

#[pymethods]
impl PyPhirClassicalInterpreter {
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(RustInterpreter::new())),
            program_json: None,
            phir_validate: true,
        }
    }

    /// Initialize with a PHIR program. Returns num_qubits.
    #[pyo3(signature = (program, foreign_obj=None))]
    fn init(
        &mut self,
        py: Python<'_>,
        program: &Bound<'_, PyAny>,
        foreign_obj: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<usize> {
        // Convert program to JSON string
        let json_str = if let Ok(s) = program.extract::<String>() {
            s
        } else if program.is_instance_of::<pyo3::types::PyDict>() {
            let json_mod = py.import("json")?;
            json_mod
                .call_method1("dumps", (program,))?
                .extract::<String>()?
        } else {
            // Try to_phir_dict() for PhirConvertible objects
            let phir_dict = program.call_method0("to_phir_dict")?;
            let json_mod = py.import("json")?;
            json_mod
                .call_method1("dumps", (&phir_dict,))?
                .extract::<String>()?
        };

        if foreign_obj.is_some() {
            log::warn!("foreign_object not yet supported in Rust interpreter, ignoring");
        }

        self.program_json = Some(json_str.clone());

        let mut inner = self
            .inner
            .lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Lock error: {e}")))?;
        inner
            .init(&json_str, None)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))
    }

    /// Reset to initial state.
    fn reset(&mut self) -> PyResult<()> {
        self.inner = Arc::new(Mutex::new(RustInterpreter::new()));
        self.program_json = None;
        Ok(())
    }

    /// Reset variable values for a new shot.
    fn shot_reinit(&self) -> PyResult<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Lock error: {e}")))?;
        inner.shot_reinit();
        Ok(())
    }

    /// Add a classical variable dynamically.
    fn add_cvar(&self, _py: Python<'_>, cvar: &str, dtype: &Bound<'_, PyAny>, size: usize) -> PyResult<()> {
        let dtype_str = dtype.str()?.to_string();
        let data_type = map_python_dtype(&dtype_str)?;

        let mut inner = self
            .inner
            .lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Lock error: {e}")))?;
        inner
            .add_cvar(cvar, data_type, size)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))
    }

    /// Execute the program, returning an iterator that yields batches of ops.
    ///
    /// The `sequence` argument is accepted for protocol compatibility but ignored --
    /// the iterator always uses the program ops stored at init time.
    fn execute(&self, _sequence: Option<&Bound<'_, PyAny>>) -> PyResult<PyPhirExecuteIter> {
        let json = self.program_json.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("No program initialized")
        })?;
        let program: PHIRProgram = serde_json::from_str(json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Failed to parse: {e}"))
        })?;

        Ok(PyPhirExecuteIter {
            interp: Arc::clone(&self.inner),
            ops: program.ops,
            stack: vec![(0, OpsRef::Root)],
            buffer: Vec::new(),
            done: false,
        })
    }

    /// Receive measurement results from the quantum simulator.
    fn receive_results(&self, _py: Python<'_>, qsim_results: &Bound<'_, PyList>) -> PyResult<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Lock error: {e}")))?;

        let mut results = Vec::new();
        for item in qsim_results.iter() {
            let dict = item.cast::<PyDict>()?;
            let mut meas = BTreeMap::new();
            for (key, val) in dict.iter() {
                let v: i64 = val.extract()?;
                if let Ok(tuple) = key.downcast::<PyTuple>() {
                    let name: String = tuple.get_item(0)?.extract()?;
                    let idx: usize = tuple.get_item(1)?.extract()?;
                    meas.insert(MeasKey::Bit(name, idx), v);
                } else if let Ok(name) = key.extract::<String>() {
                    meas.insert(MeasKey::Var(name), v);
                }
            }
            results.push(meas);
        }

        inner
            .receive_results(&results)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
    }

    /// Extract measurement bits, filtering private variables.
    #[pyo3(signature = (bits, *, filter_private=true))]
    #[allow(unused_variables)]
    fn result_bits(
        &self,
        py: Python<'_>,
        bits: &Bound<'_, PyAny>,
        filter_private: bool,
    ) -> PyResult<Py<PyAny>> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Lock error: {e}")))?;

        // Convert Python bits (list of dicts) to Rust format
        let mut measurements = Vec::new();
        let iter = bits.try_iter()?;
        for item in iter {
            let item = item?;
            let dict = item.cast::<PyDict>()?;
            let mut meas = BTreeMap::new();
            for (key, val) in dict.iter() {
                let tuple = key.cast::<PyTuple>()?;
                let name: String = tuple.get_item(0)?.extract()?;
                let idx: usize = tuple.get_item(1)?.extract()?;
                let v: i64 = val.extract()?;
                meas.insert((name, idx), v);
            }
            measurements.push(meas);
        }

        let result = inner.result_bits(&measurements);

        let dict = PyDict::new(py);
        for ((name, idx), val) in &result {
            let key = PyTuple::new(py, &[
                name.into_pyobject(py)?.into_any(),
                idx.into_pyobject(py)?.into_any(),
            ])?;
            dict.set_item(key, val)?;
        }
        Ok(dict.into_any().unbind())
    }

    /// Return final results dict.
    #[pyo3(signature = (*, return_int=true))]
    fn results(&self, py: Python<'_>, return_int: bool) -> PyResult<Py<PyAny>> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Lock error: {e}")))?;
        let results = inner.results(return_int);

        let dict = PyDict::new(py);
        for (name, val) in &results {
            match val {
                ResultValue::Int(v) => dict.set_item(name, v)?,
                ResultValue::BitString(s) => dict.set_item(name, s)?,
            }
        }
        Ok(dict.into_any().unbind())
    }

    /// Expose `program` attribute for HybridEngine compatibility.
    ///
    /// HybridEngine accesses `cinterp.program.ops` and passes it to execute().
    /// Our execute() ignores the argument, so this returns a simple wrapper.
    #[getter]
    fn program(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let num_qubits = {
            let inner = self.inner.lock().map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Lock error: {e}"))
            })?;
            inner.num_qubits()
        };

        let wrapper = PyProgramWrapper { num_qubits };
        Ok(wrapper.into_pyobject(py)?.into_any().unbind())
    }
}

/// Wrapper for the `program` attribute.
///
/// HybridEngine accesses `cinterp.program.ops` and `cinterp.program.num_qubits`.
/// The `ops` returns None since our execute() uses its own internal ops.
#[pyclass(name = "_PhirProgramWrapper", module = "pecos_rslib")]
struct PyProgramWrapper {
    #[pyo3(get)]
    num_qubits: usize,
}

#[pymethods]
impl PyProgramWrapper {
    #[getter]
    fn ops(&self, py: Python<'_>) -> Py<PyAny> {
        py.None()
    }
}

// ── Execute Iterator ────────────────────────────────────────────────

/// Where the current frame's ops come from.
enum OpsRef {
    /// Root level -- use the owned ops Vec
    Root,
    /// Owned ops from a block
    Owned(Vec<Operation>),
}

/// Python iterator that yields batches of QOp/MOp objects.
#[pyclass(name = "_PhirExecuteIter", module = "pecos_rslib")]
pub struct PyPhirExecuteIter {
    interp: Arc<Mutex<RustInterpreter>>,
    /// Owned copy of the program ops
    ops: Vec<Operation>,
    /// Stack of (current_index, ops_source)
    stack: Vec<(usize, OpsRef)>,
    /// Buffer of yielded ops accumulated until a measurement
    buffer: Vec<YieldedOp>,
    done: bool,
}

#[pymethods]
impl PyPhirExecuteIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        if self.done {
            return Ok(None);
        }

        let batch = self.advance()?;
        match batch {
            Some(ops) => {
                let py_list = convert_batch_to_python(py, &ops)?;
                Ok(Some(py_list))
            }
            None => Ok(None),
        }
    }
}

impl PyPhirExecuteIter {
    fn get_ops_slice<'a>(ops: &'a [Operation], stack_entry: &'a OpsRef) -> &'a [Operation] {
        match stack_entry {
            OpsRef::Root => ops,
            OpsRef::Owned(owned) => owned,
        }
    }

    /// Advance through operations until the next measurement boundary or end.
    fn advance(&mut self) -> PyResult<Option<Vec<YieldedOp>>> {
        let mut interp = self
            .interp
            .lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Lock error: {e}")))?;

        loop {
            let stack_len = self.stack.len();
            if stack_len == 0 {
                self.done = true;
                if self.buffer.is_empty() {
                    return Ok(None);
                }
                return Ok(Some(std::mem::take(&mut self.buffer)));
            }

            let (idx, ref ops_ref) = self.stack[stack_len - 1];
            let ops_slice = Self::get_ops_slice(&self.ops, ops_ref);

            if idx >= ops_slice.len() {
                self.stack.pop();
                continue;
            }

            // Clone the operation to avoid borrow issues
            let op = ops_slice[idx].clone();
            self.stack[stack_len - 1].0 += 1;

            match &op {
                Operation::VariableDefinition { .. } | Operation::Comment { .. } => {}

                Operation::MetaInstruction { meta, .. } => {
                    if meta == "barrier" {
                        // skip
                    }
                }

                Operation::QuantumOp {
                    qop,
                    angles,
                    args,
                    returns,
                    metadata,
                } => {
                    let yielded = interp
                        .make_qop(qop, angles, args, returns, metadata)
                        .map_err(|e| {
                            pyo3::exceptions::PyRuntimeError::new_err(format!("{e}"))
                        })?;
                    let is_measure =
                        matches!(qop.as_str(), "measure Z" | "Measure" | "Measure +Z");
                    self.buffer.push(YieldedOp::QOp(yielded));

                    if is_measure {
                        return Ok(Some(std::mem::take(&mut self.buffer)));
                    }
                }

                Operation::MachineOp {
                    mop,
                    args,
                    duration,
                    metadata,
                } => {
                    let yielded = interp
                        .make_mop(mop, args, duration, metadata)
                        .map_err(|e| {
                            pyo3::exceptions::PyRuntimeError::new_err(format!("{e}"))
                        })?;
                    self.buffer.push(YieldedOp::MOp(yielded));
                }

                Operation::ClassicalOp {
                    cop,
                    args,
                    returns,
                    function,
                    ..
                } => {
                    interp.handle_cop(cop, args, returns, function).map_err(|e| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!("{e}"))
                    })?;
                }

                Operation::Block {
                    block,
                    ops: block_ops,
                    condition,
                    true_branch,
                    false_branch,
                    ..
                } => match block.as_str() {
                    "sequence" | "qparallel" => {
                        self.stack.push((0, OpsRef::Owned(block_ops.clone())));
                    }
                    "if" => {
                        let cond = condition.as_ref().ok_or_else(|| {
                            pyo3::exceptions::PyValueError::new_err("If block missing condition")
                        })?;
                        let cond_val = interp.eval_expr(cond).map_err(|e| {
                            pyo3::exceptions::PyRuntimeError::new_err(format!("{e}"))
                        })?;
                        if cond_val != 0 {
                            if let Some(tb) = true_branch {
                                self.stack.push((0, OpsRef::Owned(tb.clone())));
                            }
                        } else if let Some(fb) = false_branch {
                            self.stack.push((0, OpsRef::Owned(fb.clone())));
                        }
                    }
                    other => {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Unknown block type: {other}"
                        )));
                    }
                },
            }
        }
    }
}

// ── Python object conversion ────────────────────────────────────────

/// Convert a batch of YieldedOps to a Python list of actual QOp/MOp objects.
///
/// Creates real Python `pecos.reps.pyphir.op_types.QOp` and `MOp` instances
/// so that `isinstance()` checks in QuantumSimulator and GenericOpProc work.
fn convert_batch_to_python(py: Python<'_>, ops: &[YieldedOp]) -> PyResult<Py<PyAny>> {
    let op_types = py.import("pecos.reps.pyphir.op_types")?;
    let qop_cls = op_types.getattr("QOp")?;
    let mop_cls = op_types.getattr("MOp")?;

    let list = PyList::empty(py);

    for op in ops {
        match op {
            YieldedOp::QOp(qop) => {
                let py_args = qop_args_to_python(py, &qop.args)?;
                let py_returns: Py<PyAny> = match &qop.returns {
                    Some(rets) => {
                        let r = PyList::empty(py);
                        for (name, idx) in rets {
                            let pair = PyList::new(py, &[
                                name.into_pyobject(py)?.into_any(),
                                idx.into_pyobject(py)?.into_any(),
                            ])?;
                            r.append(pair)?;
                        }
                        r.into_any().unbind()
                    }
                    None => py.None(),
                };
                let py_metadata = metadata_to_python(py, &qop.metadata)?;
                let py_angles: Py<PyAny> = match &qop.angles {
                    Some(angs) => PyTuple::new(py, angs)?.into_any().unbind(),
                    None => py.None(),
                };

                let obj = qop_cls.call1((
                    &qop.name,
                    py_args,
                    py_returns,
                    py_metadata,
                    py_angles,
                    &qop.sim_name,
                ))?;
                list.append(obj)?;
            }
            YieldedOp::MOp(mop) => {
                let py_args: Py<PyAny> = match &mop.args {
                    Some(args) => qop_args_to_python(py, args)?,
                    None => py.None(),
                };
                let py_metadata: Py<PyAny> = match &mop.metadata {
                    Some(meta) => metadata_to_python(py, meta)?,
                    None => py.None(),
                };
                let obj = mop_cls.call1((&mop.name, py_args, py.None(), py_metadata))?;
                list.append(obj)?;
            }
        }
    }

    Ok(list.into_any().unbind())
}

/// Convert QOpArgs to Python representation.
fn qop_args_to_python(py: Python<'_>, args: &QOpArgs) -> PyResult<Py<PyAny>> {
    match args {
        QOpArgs::Single(ids) => {
            let list = PyList::new(py, ids)?;
            Ok(list.into_any().unbind())
        }
        QOpArgs::Multi(groups) => {
            let list = PyList::empty(py);
            for group in groups {
                let tuple = PyTuple::new(py, group)?;
                list.append(tuple)?;
            }
            Ok(list.into_any().unbind())
        }
    }
}

/// Convert BTreeMap metadata to a Python dict.
fn metadata_to_python(
    py: Python<'_>,
    metadata: &BTreeMap<String, serde_json::Value>,
) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    for (key, val) in metadata {
        dict.set_item(key, json_value_to_python(py, val)?)?;
    }
    Ok(dict.into_any().unbind())
}

/// Convert a serde_json::Value to a Python object.
fn json_value_to_python(py: Python<'_>, val: &serde_json::Value) -> PyResult<Py<PyAny>> {
    match val {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => {
            let pyb = b.into_pyobject(py)?;
            Ok(pyb.to_owned().into_any().unbind())
        }
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)?.into_any().unbind())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py)?.into_any().unbind())
            } else {
                Ok(py.None())
            }
        }
        serde_json::Value::String(s) => Ok(s.into_pyobject(py)?.into_any().unbind()),
        serde_json::Value::Array(arr) => {
            let list = PyList::empty(py);
            for item in arr {
                list.append(json_value_to_python(py, item)?)?;
            }
            Ok(list.into_any().unbind())
        }
        serde_json::Value::Object(obj) => {
            let dict = PyDict::new(py);
            for (key, val) in obj {
                dict.set_item(key, json_value_to_python(py, val)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}

/// Map a Python dtype string to Rust DataType.
fn map_python_dtype(dtype_str: &str) -> PyResult<DataType> {
    let clean = dtype_str
        .trim_start_matches("<class '")
        .trim_end_matches("'>")
        .trim_start_matches("pecos.dtypes.");

    DataType::from_str(clean)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Unknown dtype '{dtype_str}': {e}")))
}
