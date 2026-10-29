//! Python provider bridge. Only observed bytes and public request IDs cross it.
use crate::scheduled_adapter::CallbackScope;
use pecos_core::errors::PecosError;
use pecos_qis::runtime_decoder::{RuntimeDecoderFactory, RuntimeDecoderSession};
use pyo3::prelude::*;
use std::sync::Arc;

struct PythonDecoderSession(Py<PyAny>);

fn callback_error(error: PyErr) -> PecosError {
    PecosError::Processing(format!("Python runtime decoder: {error}"))
}

impl RuntimeDecoderSession for PythonDecoderSession {
    fn decode(&mut self, window: u64, instance: u64, syndrome: &[u8]) -> Result<u64, PecosError> {
        let _scope = CallbackScope::enter();
        Python::attach(|py| {
            self.0
                .call_method1(py, "decode", (window, instance, syndrome.to_vec()))?
                .extract(py)
        })
        .map_err(callback_error)
    }
}

pub(crate) fn factory(object: Py<PyAny>, py: Python<'_>) -> PyResult<RuntimeDecoderFactory> {
    if !object.bind(py).is_callable() {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "runtime decoder factory must be callable",
        ));
    }
    Ok(Arc::new(move || {
        let _scope = CallbackScope::enter();
        Python::attach(|py| -> PyResult<Box<dyn RuntimeDecoderSession>> {
            let session = object.call0(py)?;
            if !session.getattr(py, "decode")?.bind(py).is_callable() {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "runtime decoder session.decode must be callable",
                ));
            }
            Ok(Box::new(PythonDecoderSession(session)))
        })
        .map_err(callback_error)
    }))
}
