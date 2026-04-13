//! Python-implemented decoder plugin.
//!
//! Wraps a Python object that implements `decode()`, `check_count()`, and `bit_count()`
//! into a Rust type that implements [`pecos_decoder_core::Decoder`].

use ndarray::ArrayView1;
use pecos_decoder_core::{Decoder, DecodingResultTrait};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::fmt;

/// Error from a Python-implemented decoder.
#[derive(Debug)]
pub struct PyForeignDecoderError(pub String);

impl fmt::Display for PyForeignDecoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Python decoder error: {}", self.0)
    }
}

impl std::error::Error for PyForeignDecoderError {}

/// Decoded result from a Python decoder.
#[derive(Debug, Clone)]
pub struct PyForeignDecodingResult {
    pub observable: Vec<u8>,
    pub weight: f64,
    pub converged: Option<bool>,
}

impl DecodingResultTrait for PyForeignDecodingResult {
    fn is_successful(&self) -> bool {
        self.converged.unwrap_or(true)
    }

    fn cost(&self) -> Option<f64> {
        Some(self.weight)
    }
}

/// A decoder implemented in Python, usable by PECOS's Rust engine.
///
/// The Python object must implement:
/// - `decode(syndrome: bytes) -> dict` with keys `"observable"` (bytes/list) and `"weight"` (float),
///   optionally `"converged"` (bool)
/// - `check_count() -> int`
/// - `bit_count() -> int`
///
/// # Example (Python side)
///
/// ```python
/// class MyDecoder:
///     def decode(self, syndrome: bytes) -> dict:
///         return {"observable": bytes([0, 1]), "weight": 1.0}
///     def check_count(self) -> int:
///         return 10
///     def bit_count(self) -> int:
///         return 5
///
/// decoder = pecos_rslib.PyForeignDecoder(MyDecoder())
/// ```
#[pyclass(name = "PyForeignDecoder", module = "pecos_rslib")]
pub struct PyForeignDecoder {
    inner: Py<PyAny>,
    cached_check_count: usize,
    cached_bit_count: usize,
}

// SAFETY: Py<PyAny> is Send (atomic refcount). GIL is acquired via Python::attach()
// before any access to the Python object.
unsafe impl Send for PyForeignDecoder {}

#[pymethods]
impl PyForeignDecoder {
    /// Wrap a Python decoder object for use in PECOS.
    ///
    /// Validates that the object has the required methods and caches
    /// structural properties (`check_count`, `bit_count`).
    #[new]
    fn new(py_obj: Py<PyAny>) -> PyResult<Self> {
        Python::attach(|py| {
            let obj = py_obj.bind(py);

            for method in &["decode", "check_count", "bit_count"] {
                if !obj.hasattr(*method)? {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                        "Python decoder must have a '{method}' method"
                    )));
                }
            }

            let cached_check_count: usize = obj.call_method0("check_count")?.extract()?;
            let cached_bit_count: usize = obj.call_method0("bit_count")?.extract()?;

            Ok(Self {
                inner: py_obj,
                cached_check_count,
                cached_bit_count,
            })
        })
    }

    /// Decode a syndrome, returning a dict with `observable`, `weight`, and optionally `converged`.
    fn decode(&mut self, py: Python<'_>, syndrome: Vec<u8>) -> PyResult<Py<PyAny>> {
        let arr = ndarray::Array1::from_vec(syndrome);
        let result = Decoder::decode(self, &arr.view())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("observable", &result.observable)?;
        dict.set_item("weight", result.weight)?;
        if let Some(conv) = result.converged {
            dict.set_item("converged", conv)?;
        }
        Ok(dict.into_any().unbind())
    }

    #[getter]
    fn check_count(&self) -> usize {
        self.cached_check_count
    }

    #[getter]
    fn bit_count(&self) -> usize {
        self.cached_bit_count
    }

    fn __repr__(&self) -> String {
        format!(
            "PyForeignDecoder(checks={}, bits={})",
            self.cached_check_count, self.cached_bit_count
        )
    }
}

impl Decoder for PyForeignDecoder {
    type Result = PyForeignDecodingResult;
    type Error = PyForeignDecoderError;

    fn decode(&mut self, input: &ArrayView1<u8>) -> Result<Self::Result, Self::Error> {
        let input_vec: Vec<u8> = input.to_vec();

        Python::attach(|py| {
            let py_bytes = PyBytes::new(py, &input_vec);
            let result = self
                .inner
                .call_method1(py, "decode", (py_bytes,))
                .map_err(|e| PyForeignDecoderError(format!("decode() failed: {e}")))?;

            let result_bound = result.bind(py);

            let observable: Vec<u8> = result_bound
                .get_item("observable")
                .map_err(|e| PyForeignDecoderError(format!("missing 'observable': {e}")))?
                .extract::<Vec<u8>>()
                .map_err(|e| PyForeignDecoderError(format!("'observable' not bytes: {e}")))?;

            let weight: f64 = result_bound
                .get_item("weight")
                .map_err(|e| PyForeignDecoderError(format!("missing 'weight': {e}")))?
                .extract::<f64>()
                .map_err(|e| PyForeignDecoderError(format!("'weight' not float: {e}")))?;

            let converged: Option<bool> = result_bound
                .get_item("converged")
                .ok()
                .and_then(|v| v.extract::<bool>().ok());

            Ok(PyForeignDecodingResult {
                observable,
                weight,
                converged,
            })
        })
    }

    fn check_count(&self) -> usize {
        self.cached_check_count
    }

    fn bit_count(&self) -> usize {
        self.cached_bit_count
    }
}
