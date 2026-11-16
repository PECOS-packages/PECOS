//! Python bridge for runtime-owned event interpretation on the ordinary QIS path.
use crate::scheduled_adapter::CallbackScope;
use pecos_qis::runtime::{Result, RuntimeError};
use pecos_qis::runtime_batch_adapter::MAX_ADAPTER_OPERATIONS;
use pecos_qis::scheduled::RuntimeScheduledOp;
use pecos_qis::{RuntimeBatch, RuntimeBatchAdapter, SeleneRuntime};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;

struct PythonBatchAdapter {
    object: Py<PyAny>,
}

fn callback_error(error: impl std::fmt::Display) -> RuntimeError {
    RuntimeError::ExecutionError(format!("Python runtime batch adapter: {error}"))
}

impl RuntimeBatchAdapter for PythonBatchAdapter {
    fn normalize(&mut self, batch: &RuntimeBatch) -> Result<Vec<RuntimeScheduledOp>> {
        let _scope = CallbackScope::enter();
        let input = serde_json::to_string(batch).map_err(callback_error)?;
        Python::attach(|py| -> PyResult<Vec<RuntimeScheduledOp>> {
            let json = py.import("json")?;
            let snapshot = json.call_method1("loads", (input,))?;
            let result = self.object.call_method1(py, "normalize", (snapshot,))?;
            let output = result.bind(py).cast::<PyList>()?;
            if output.len() > MAX_ADAPTER_OPERATIONS {
                return Err(PyValueError::new_err(
                    "runtime batch adapter expansion limit",
                ));
            }
            let encoded: String = json.call_method1("dumps", (output,))?.extract()?;
            serde_json::from_str(&encoded).map_err(|e| {
                PyValueError::new_err(format!("invalid runtime batch adapter output: {e}"))
            })
        })
        .map_err(callback_error)
    }
}

pub(crate) fn configure(
    runtime: &mut SeleneRuntime,
    factory: Option<Py<PyAny>>,
    py: Python<'_>,
) -> PyResult<()> {
    let Some(factory) = factory else {
        return Ok(());
    };
    if !factory.bind(py).is_callable() {
        return Err(PyTypeError::new_err(
            "batch_adapter_factory must be callable",
        ));
    }
    runtime.set_batch_adapter_factory(move || {
        let _scope = CallbackScope::enter();
        Python::attach(|py| -> PyResult<Box<dyn RuntimeBatchAdapter>> {
            let object = factory.call0(py)?;
            if !object.getattr(py, "normalize")?.bind(py).is_callable() {
                return Err(PyTypeError::new_err("adapter.normalize must be callable"));
            }
            Ok(Box::new(PythonBatchAdapter { object }))
        })
        .map_err(callback_error)
    });
    Ok(())
}
