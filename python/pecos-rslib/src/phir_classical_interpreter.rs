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

//! `PyO3` wrapper for the Rust `PhirClassicalInterpreter`.
//!
//! Exposes the Rust classical interpreter to Python as a drop-in replacement
//! for `pecos.classical_interpreters.PhirClassicalInterpreter`.

use pecos_core::errors::PecosError;
use pecos_phir_json::v0_1::ast::{Operation, PHIRProgram};
use pecos_phir_json::v0_1::classical_interpreter::{
    MeasKey, PhirClassicalInterpreter as RustInterpreter, QOpArgs, ResultValue, YieldedOp,
};
use pecos_phir_json::v0_1::environment::DataType;
use pecos_wasm::ForeignObject;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::any::Any;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

// ── Python ForeignObject bridge ──────────────────────────────────────

/// Wraps a Python `ForeignObjectProtocol` as a Rust `ForeignObject`.
///
/// Calls into Python via the GIL for `exec()`. Implements `Send + Sync`
/// because `Py<PyAny>` is `Send` and we acquire the GIL on each call.
struct PyForeignObject {
    obj: Py<PyAny>,
}

// SAFETY: Py<PyAny> is Send. We always acquire the GIL before using it.
unsafe impl Sync for PyForeignObject {}

impl std::fmt::Debug for PyForeignObject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PyForeignObject(<python object>)")
    }
}

impl ForeignObject for PyForeignObject {
    fn clone_box(&self) -> Box<dyn ForeignObject> {
        Python::attach(|py| {
            Box::new(PyForeignObject {
                obj: self.obj.clone_ref(py),
            }) as Box<dyn ForeignObject>
        })
    }

    fn init(&mut self) -> Result<(), PecosError> {
        Python::attach(|py| {
            self.obj
                .call_method0(py, "init")
                .map_err(|e| PecosError::Input(format!("ForeignObject.init() failed: {e}")))?;
            Ok(())
        })
    }

    fn new_instance(&mut self) -> Result<(), PecosError> {
        // Python ForeignObjectProtocol doesn't have new_instance, just init
        Ok(())
    }

    fn get_funcs(&self) -> Vec<String> {
        Python::attach(|py| {
            let result = self.obj.call_method0(py, "get_funcs");
            match result {
                Ok(list) => list.extract::<Vec<String>>(py).unwrap_or_default(),
                Err(_) => vec![],
            }
        })
    }

    fn exec(&mut self, func_name: &str, args: &[i64]) -> Result<Vec<i64>, PecosError> {
        Python::attach(|py| {
            let py_args = PyList::new(py, args)
                .map_err(|e| PecosError::Input(format!("Failed to create args list: {e}")))?;
            let result = self
                .obj
                .call_method1(py, "exec", (func_name, py_args))
                .map_err(|e| {
                    PecosError::Input(format!("ForeignObject.exec({func_name}) failed: {e}"))
                })?;

            // Result can be int or tuple/list of ints
            if let Ok(val) = result.extract::<i64>(py) {
                Ok(vec![val])
            } else if let Ok(vals) = result.extract::<Vec<i64>>(py) {
                Ok(vals)
            } else {
                // Try extracting as tuple
                let tuple = result.bind(py).cast::<PyTuple>().map_err(|e| {
                    PecosError::Input(format!("ForeignObject.exec() returned non-int/tuple: {e}"))
                })?;
                let mut vals = Vec::new();
                for item in tuple.iter() {
                    vals.push(item.extract::<i64>().map_err(|e| {
                        PecosError::Input(format!("ForeignObject.exec() return item not int: {e}"))
                    })?);
                }
                Ok(vals)
            }
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

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

    /// Support pickling for multiprocessing.
    ///
    /// The interpreter state is re-initialized by `HybridEngine.run()` via `init()`,
    /// so we only need to preserve the configuration (`phir_validate`).
    fn __reduce__(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let _ = slf; // PyO3 pickle protocol requires the bound reference
        let cls = py
            .import("pecos_rslib")?
            .getattr("RustPhirClassicalInterpreter")?
            .unbind();
        let args = PyTuple::empty(py).into_any().unbind();
        Ok((cls, args))
    }

    fn __getstate__(&self) -> bool {
        self.phir_validate
    }

    fn __setstate__(&mut self, state: bool) {
        self.phir_validate = state;
    }

    /// Initialize with a PHIR program. Returns `num_qubits`.
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

        let rust_foreign = foreign_obj.map(|fo| {
            let py_fo = PyForeignObject {
                obj: fo.clone().unbind(),
            };
            Box::new(py_fo) as Box<dyn ForeignObject>
        });

        self.program_json = Some(json_str.clone());

        let mut inner = self
            .inner
            .lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Lock error: {e}")))?;
        inner
            .init(&json_str, rust_foreign)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))
    }

    /// Reset to initial state.
    fn reset(&mut self) {
        self.inner = Arc::new(Mutex::new(RustInterpreter::new()));
        self.program_json = None;
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
    fn add_cvar(
        &self,
        _py: Python<'_>,
        cvar: &str,
        dtype: &Bound<'_, PyAny>,
        size: usize,
    ) -> PyResult<()> {
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
    /// When `sequence` is None or the program's own ops, uses the Rust AST walker.
    /// When `sequence` is a list of Python QOp/MOp objects (inner interpreter case),
    /// passes them through directly, buffering at measurement boundaries.
    fn execute(&self, py: Python<'_>, sequence: Option<&Bound<'_, PyAny>>) -> PyResult<Py<PyAny>> {
        // Check if sequence is a list of Python QOp objects (inner interpreter mode)
        if let Some(seq) = sequence
            && !seq.is_none()
        {
            if let Ok(py_list) = seq.cast::<PyList>() {
                // Inner interpreter mode: pass through Python QOp objects
                return Ok(PyPhirPassthroughIter::new(py_list)
                    .into_pyobject(py)?
                    .into_any()
                    .unbind());
            }
            // If it's some other iterable (like a list), try to convert
            if let Ok(iter) = seq.try_iter() {
                let items: Vec<Py<PyAny>> = iter
                    .map(|item| item.map(pyo3::Bound::unbind))
                    .collect::<PyResult<_>>()?;
                let py_list = PyList::new(py, &items)?;
                return Ok(PyPhirPassthroughIter::new(&py_list)
                    .into_pyobject(py)?
                    .into_any()
                    .unbind());
            }
        }

        // Outer interpreter mode: use Rust AST walker
        let json = self
            .program_json
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No program initialized"))?;
        let program: PHIRProgram = serde_json::from_str(json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Failed to parse: {e}"))
        })?;

        Ok(PyPhirExecuteIter {
            interp: Arc::clone(&self.inner),
            ops: program.ops,
            stack: vec![(0, OpsRef::Root)],
            buffer: Vec::new(),
            done: false,
            qop_cls: None,
            mop_cls: None,
        }
        .into_pyobject(py)?
        .into_any()
        .unbind())
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
                if let Ok(tuple) = key.cast::<PyTuple>() {
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

    /// Extract measurement bits, optionally filtering private variables.
    #[pyo3(signature = (bits, *, filter_private=true))]
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
                if filter_private && name.starts_with("__") {
                    continue;
                }
                let v: i64 = val.extract()?;
                meas.insert((name, idx), v);
            }
            measurements.push(meas);
        }

        let result = inner.result_bits(&measurements);

        let dict = PyDict::new(py);
        for ((name, idx), val) in &result {
            let key = PyTuple::new(
                py,
                &[
                    name.into_pyobject(py)?.into_any(),
                    idx.into_pyobject(py)?.into_any(),
                ],
            )?;
            dict.set_item(key, val)?;
        }
        Ok(dict.into_any().unbind())
    }

    /// Return final results dict.
    ///
    /// When `return_int=True`, returns PECOS dtype objects (e.g. `i32(42)`, `u32(7)`)
    /// matching the Python `PhirClassicalInterpreter` behavior.
    #[pyo3(signature = (*, return_int=true))]
    fn results(&self, py: Python<'_>, return_int: bool) -> PyResult<Py<PyAny>> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Lock error: {e}")))?;
        let results = inner.results(return_int);

        let dict = PyDict::new(py);
        if return_int {
            // Access dtypes through pecos_rslib module
            let pecos_rslib = py.import("pecos_rslib")?;
            let dtypes = pecos_rslib.getattr("dtypes")?;
            for (name, val) in &results {
                match val {
                    ResultValue::Int(v, dtype_name) => {
                        let dtype_cls = dtypes.getattr(dtype_name.as_str())?;
                        let typed_val = dtype_cls.call1((*v,))?;
                        dict.set_item(name, typed_val)?;
                    }
                    ResultValue::UInt(v, dtype_name) => {
                        let dtype_cls = dtypes.getattr(dtype_name.as_str())?;
                        let typed_val = dtype_cls.call1((*v,))?;
                        dict.set_item(name, typed_val)?;
                    }
                    ResultValue::BitString(s) => dict.set_item(name, s)?,
                }
            }
        } else {
            for (name, val) in &results {
                match val {
                    ResultValue::Int(v, _) => dict.set_item(name, v)?,
                    ResultValue::UInt(v, _) => dict.set_item(name, v)?,
                    ResultValue::BitString(s) => dict.set_item(name, s)?,
                }
            }
        }
        Ok(dict.into_any().unbind())
    }

    /// Expose `program` attribute for `HybridEngine` compatibility.
    ///
    /// Returns a wrapper with `ops` and `num_qubits` attributes.
    /// `ops` returns a sentinel that our `execute()` recognizes.
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

    /// Expose `foreign_obj` attribute for protocol compatibility.
    #[getter]
    #[allow(clippy::unused_self)] // PyO3 requires &self for getters
    fn foreign_obj(&self) -> Option<()> {
        // Foreign objects are stored internally in the Rust interpreter.
        // We return None here for protocol compatibility -- the actual
        // foreign object was passed via init() and is held in Rust.
        None
    }
}

/// Wrapper for the `program` attribute.
///
/// `HybridEngine` accesses `cinterp.program.ops` and `cinterp.program.num_qubits`.
/// The `ops` returns None since our `execute()` uses its own internal ops.
#[pyclass(name = "_PhirProgramWrapper", module = "pecos_rslib")]
struct PyProgramWrapper {
    #[pyo3(get)]
    num_qubits: usize,
}

#[pymethods]
impl PyProgramWrapper {
    #[getter]
    #[allow(clippy::unused_self)] // PyO3 requires &self for getters
    fn ops(&self, py: Python<'_>) -> Py<PyAny> {
        py.None()
    }
}

// ── Passthrough Iterator (for inner interpreter) ────────────────────

/// Iterator that passes Python QOp/MOp objects through, buffering at measurement boundaries.
///
/// Used by the inner interpreter which receives noisy `QOp` objects from `op_processor.process()`.
#[pyclass(name = "_PhirPassthroughIter", module = "pecos_rslib")]
struct PyPhirPassthroughIter {
    /// All ops as Python objects
    ops: Vec<Py<PyAny>>,
    /// Current position
    idx: usize,
    /// Buffer for current batch
    buffer: Vec<Py<PyAny>>,
    done: bool,
}

impl PyPhirPassthroughIter {
    fn new(ops: &Bound<'_, PyList>) -> Self {
        let items: Vec<Py<PyAny>> = ops.iter().map(pyo3::Bound::unbind).collect();
        Self {
            ops: items,
            idx: 0,
            buffer: Vec::new(),
            done: false,
        }
    }
}

#[pymethods]
impl PyPhirPassthroughIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        if self.done {
            return Ok(None);
        }

        loop {
            if self.idx >= self.ops.len() {
                self.done = true;
                if self.buffer.is_empty() {
                    return Ok(None);
                }
                let batch: Vec<Py<PyAny>> = std::mem::take(&mut self.buffer);
                let list = PyList::new(py, &batch)?;
                return Ok(Some(list.into_any().unbind()));
            }

            let op = &self.ops[self.idx];
            self.idx += 1;

            // Check if this is a measurement op by reading .name
            let name: String = op.getattr(py, "name")?.extract(py)?;
            let is_measure = matches!(name.as_str(), "measure Z" | "Measure" | "Measure +Z");

            self.buffer.push(op.clone_ref(py));

            if is_measure {
                let batch: Vec<Py<PyAny>> = std::mem::take(&mut self.buffer);
                let list = PyList::new(py, &batch)?;
                return Ok(Some(list.into_any().unbind()));
            }
        }
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
    /// Stack of (`current_index`, `ops_source`)
    stack: Vec<(usize, OpsRef)>,
    /// Buffer of yielded ops accumulated until a measurement
    buffer: Vec<YieldedOp>,
    done: bool,
    /// Cached Python `QOp` class (avoid repeated import)
    qop_cls: Option<Py<PyAny>>,
    /// Cached Python `MOp` class
    mop_cls: Option<Py<PyAny>>,
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

        // Cache class lookups on first call
        if self.qop_cls.is_none() {
            let op_types = py.import("pecos.reps.pyphir.op_types")?;
            self.qop_cls = Some(op_types.getattr("QOp")?.unbind());
            self.mop_cls = Some(op_types.getattr("MOp")?.unbind());
        }

        let batch = self.advance()?;
        match batch {
            Some(ops) => {
                let qop_cls = self.qop_cls.as_ref().unwrap().bind(py);
                let mop_cls = self.mop_cls.as_ref().unwrap().bind(py);
                let py_list = convert_batch_to_python_cached(py, &ops, qop_cls, mop_cls)?;
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
                Operation::VariableDefinition { .. }
                | Operation::DataExport { .. }
                | Operation::Comment { .. } => {}

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
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
                    let is_measure = matches!(qop.as_str(), "measure Z" | "Measure" | "Measure +Z");
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
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
                    self.buffer.push(YieldedOp::MOp(yielded));
                }

                Operation::ClassicalOp {
                    cop,
                    args,
                    returns,
                    function,
                    ..
                } => {
                    interp
                        .handle_cop(cop, args, returns, function)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
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

/// Convert a batch of `YieldedOps` to a Python list of actual QOp/MOp objects.
///
/// Uses pre-cached class references for performance.
fn convert_batch_to_python_cached<'py>(
    py: Python<'py>,
    ops: &[YieldedOp],
    qop_cls: &Bound<'py, PyAny>,
    mop_cls: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    let list = PyList::empty(py);

    for op in ops {
        match op {
            YieldedOp::QOp(qop) => {
                let py_args = qop_args_to_python(py, &qop.args)?;
                let py_returns: Py<PyAny> = match &qop.returns {
                    Some(rets) => {
                        let r = PyList::empty(py);
                        for (name, idx) in rets {
                            let pair = PyList::new(
                                py,
                                &[
                                    name.into_pyobject(py)?.into_any(),
                                    idx.into_pyobject(py)?.into_any(),
                                ],
                            )?;
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

/// Convert `QOpArgs` to Python representation.
fn qop_args_to_python(py: Python<'_>, args: &QOpArgs) -> PyResult<Py<PyAny>> {
    match args {
        QOpArgs::Single(ids) => {
            let list = PyList::new(py, ids)?;
            Ok(list.into_any().unbind())
        }
        QOpArgs::Multi(groups) => {
            let list = PyList::empty(py);
            for group in groups {
                // Python PyPHIR uses list (not tuple) for multi-qubit arg groups
                let inner = PyList::new(py, group)?;
                list.append(inner)?;
            }
            Ok(list.into_any().unbind())
        }
    }
}

/// Convert `BTreeMap` metadata to a Python dict.
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

/// Convert a `serde_json::Value` to a Python object.
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

/// Map a Python dtype string to Rust `DataType`.
fn map_python_dtype(dtype_str: &str) -> PyResult<DataType> {
    let clean = dtype_str
        .trim_start_matches("<class '")
        .trim_end_matches("'>")
        .trim_start_matches("pecos.dtypes.")
        .trim_start_matches("pecos_rslib.dtypes.");

    // Map Python dtype names to Rust DataType names
    // str(pc.dtypes.i64) gives "int64", repr gives "dtypes.i64"
    let mapped = match clean {
        "int8" => "i8",
        "int16" => "i16",
        "int32" => "i32",
        "int64" => "i64",
        "uint8" => "u8",
        "uint16" => "u16",
        "uint32" => "u32",
        "uint64" => "u64",
        other => other,
    };

    mapped.parse::<DataType>().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Unknown dtype '{dtype_str}': {e}"))
    })
}

/// Build a Rust noise model from a Python object.
///
/// Accepts:
/// - Rust noise model builders (`DepolarizingNoiseModelBuilder`, `GeneralNoiseModelBuilder`, etc.)
/// - Python `GenericErrorModel` (mapped to Rust `DepolarizingNoiseModel`)
/// - Python `NoErrorModel` (mapped to `PassThroughNoiseModel`)
/// - A float (shorthand for uniform depolarizing probability)
fn build_noise_model(
    _py: Python<'_>,
    obj: &Bound<'_, PyAny>,
) -> PyResult<Box<dyn pecos_engines::noise::NoiseModel>> {
    use crate::engine_builders::{
        PyBiasedDepolarizingNoiseModelBuilder, PyDepolarizingNoiseModelBuilder,
        PyGeneralNoiseModelBuilder,
    };
    use pecos_engines::noise::{DepolarizingNoiseModel, PassThroughNoiseModel};

    // Try Rust noise model builders first
    if let Ok(builder) = obj.extract::<PyDepolarizingNoiseModelBuilder>() {
        return Ok(Box::new(builder.inner.build()));
    }
    if let Ok(builder) = obj.extract::<PyGeneralNoiseModelBuilder>() {
        return Ok(Box::new(builder.inner.build()));
    }
    if let Ok(builder) = obj.extract::<PyBiasedDepolarizingNoiseModelBuilder>() {
        return Ok(Box::new(builder.inner.build()));
    }

    // Try float (shorthand for uniform depolarizing)
    if let Ok(p) = obj.extract::<f64>() {
        return Ok(Box::new(DepolarizingNoiseModel::new_uniform(p)));
    }

    // Try Python GenericErrorModel
    let type_name = obj.get_type().qualname()?;
    if type_name == "GenericErrorModel" {
        let error_params = obj.getattr("error_params")?;
        let p1: f64 = error_params
            .get_item("p1")
            .ok()
            .and_then(|v| v.extract().ok())
            .unwrap_or(0.0);
        return Ok(Box::new(DepolarizingNoiseModel::new_uniform(p1)));
    }

    // Python NoErrorModel -> PassThrough
    if type_name == "NoErrorModel" {
        return Ok(Box::new(PassThroughNoiseModel::new()));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "Unsupported noise model type: {type_name}. Use a Rust noise model builder \
         (e.g., depolarizing_noise().with_uniform_probability(0.01)), a Python \
         GenericErrorModel, or a float for uniform depolarizing."
    )))
}

// ── Full Rust simulation ────────────────────────────────────────────

/// Run a PHIR program entirely in Rust.
///
/// Uses the existing Rust `PhirJsonEngine` + `HybridEngine` + `MonteCarloEngine`
/// pipeline. Zero Python boundary crossings per shot.
///
/// Returns a dict of `{register_name: [bitstring, ...]}` matching
/// `HybridEngine.run()` output format.
#[pyfunction]
#[pyo3(signature = (phir_json, *, shots=1, seed=None, quantum="stabilizer", foreign_object=None, noise_model=None))]
#[allow(clippy::too_many_arguments)]
pub fn run_phir_sim(
    py: Python<'_>,
    phir_json: &str,
    shots: usize,
    seed: Option<u64>,
    quantum: &str,
    foreign_object: Option<&Bound<'_, PyAny>>,
    noise_model: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    use pecos_engines::classical::ClassicalEngine;
    use pecos_engines::hybrid::HybridEngineBuilder;
    use pecos_engines::monte_carlo::MonteCarloEngineBuilder;
    use pecos_engines::noise::PassThroughNoiseModel;
    use pecos_engines::quantum::SparseStabEngine;
    use pecos_engines::{QuantumSystem, StateVecEngine};
    use pecos_phir_json::v0_1::engine::PhirJsonEngine;

    // Parse and create the classical engine
    let mut engine = PhirJsonEngine::from_json(phir_json).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Failed to parse PHIR: {e}"))
    })?;

    // Set foreign object if provided (WASM or Python ForeignObjectProtocol)
    if let Some(fo) = foreign_object {
        #[cfg(feature = "wasm")]
        {
            use crate::wasm_foreign_object_bindings::PyWasmForeignObject;
            if let Ok(wasm_ref) = fo.cast::<PyWasmForeignObject>() {
                engine.set_foreign_object(wasm_ref.borrow().clone_boxed());
            } else {
                let py_fo = PyForeignObject {
                    obj: fo.clone().unbind(),
                };
                engine.set_foreign_object(Box::new(py_fo));
            }
        }
        #[cfg(not(feature = "wasm"))]
        {
            let py_fo = PyForeignObject {
                obj: fo.clone().unbind(),
            };
            engine.set_foreign_object(Box::new(py_fo));
        }
    }

    let num_qubits = engine.num_qubits();

    // Build quantum engine
    let quantum_engine: Box<dyn pecos_engines::QuantumEngine> = match quantum {
        "stabilizer" => Box::new(SparseStabEngine::new(num_qubits)),
        "state-vector" => Box::new(StateVecEngine::new(num_qubits)),
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown quantum backend: '{other}'. Use 'stabilizer' or 'state-vector'."
            )));
        }
    };

    // Build quantum system with optional noise model
    let rust_noise: Box<dyn pecos_engines::noise::NoiseModel> = if let Some(nm) = noise_model {
        build_noise_model(py, nm)?
    } else {
        Box::new(PassThroughNoiseModel::new())
    };
    let quantum_system = QuantumSystem::new(rust_noise, quantum_engine);

    // Build hybrid engine
    let hybrid = HybridEngineBuilder::new()
        .with_classical_engine(Box::new(engine))
        .with_quantum_system(quantum_system)
        .build();

    // Build and run Monte Carlo engine
    let mut mc_builder = MonteCarloEngineBuilder::new().with_hybrid_engine(hybrid);

    if let Some(s) = seed {
        mc_builder = mc_builder.with_seed(s);
    }

    let mut mc = mc_builder.build();

    let shot_vec = mc
        .run(shots)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Simulation error: {e}")))?;

    // Convert ShotVec to Python dict matching HybridEngine.run() format
    // {register_name: [bitstring, bitstring, ...]}
    let binary_results = shot_vec.format_as_binary_strings();

    let result_dict = PyDict::new(py);
    for (name, values) in &binary_results {
        let py_list = PyList::new(py, values)?;
        result_dict.set_item(name, py_list)?;
    }

    Ok(result_dict.into_any().unbind())
}
