//! Python provider bridge for observations and optional outcome-free batch data.
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

    fn decode_v1(
        &mut self,
        window: u64,
        instance: u64,
        syndrome: &[u8],
        context: &[u8],
    ) -> Result<u64, PecosError> {
        let _scope = CallbackScope::enter();
        Python::attach(|py| {
            self.0
                .call_method1(
                    py,
                    "decode_v1",
                    (window, instance, syndrome.to_vec(), context.to_vec()),
                )?
                .extract(py)
        })
        .map_err(callback_error)
    }

    fn decode_observed_with_batches_v1(
        &mut self,
        window: u64,
        instance: u64,
        syndrome: &[u8],
        context: &[u8],
        reads: &[pecos_qis::runtime_decoder::RuntimeObservedRead],
        prefix: &pecos_qis::decoder_batches::DecoderBatchPrefix,
    ) -> Result<u64, PecosError> {
        let _scope = CallbackScope::enter();
        let observations: Vec<_> = reads
            .iter()
            .map(|r| (r.name.clone(), r.occurrence, r.source_id, r.value))
            .collect();
        let encoded =
            serde_json::to_vec(prefix).map_err(|e| PecosError::Processing(e.to_string()))?;
        Python::attach(|py| {
            self.0
                .call_method1(
                    py,
                    "decode_observed_with_batches_v1",
                    (
                        window,
                        instance,
                        syndrome.to_vec(),
                        context.to_vec(),
                        observations,
                        pyo3::types::PyBytes::new(py, &encoded),
                    ),
                )?
                .extract(py)
        })
        .map_err(callback_error)
    }

    fn decode_observed_v1(
        &mut self,
        window: u64,
        instance: u64,
        syndrome: &[u8],
        context: &[u8],
        reads: &[pecos_qis::runtime_decoder::RuntimeObservedRead],
    ) -> Result<u64, PecosError> {
        let _scope = CallbackScope::enter();
        let observations: Vec<_> = reads
            .iter()
            .map(|r| (r.name.clone(), r.occurrence, r.source_id, r.value))
            .collect();
        Python::attach(|py| {
            self.0
                .call_method1(
                    py,
                    "decode_observed_v1",
                    (
                        window,
                        instance,
                        syndrome.to_vec(),
                        context.to_vec(),
                        observations,
                    ),
                )?
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
