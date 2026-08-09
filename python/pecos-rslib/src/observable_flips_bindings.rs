// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Shared Python value type for logical-observable flip predictions and ground truth.

use pecos_decoder_core::obs_mask::ObsMask;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyInt, PyList};

/// Logical-observable flips with an explicit observable count.
#[pyclass(
    name = "ObservableFlips",
    module = "pecos_rslib.decoders",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyObservableFlips {
    mask: ObsMask,
    num_observables: usize,
}

impl PyObservableFlips {
    pub(crate) fn from_mask_value(mask: ObsMask, num_observables: usize) -> Self {
        debug_assert!(mask.iter_set_bits().all(|index| index < num_observables));
        Self {
            mask,
            num_observables,
        }
    }

    pub(crate) fn from_u8_bits(bits: &[u8]) -> Self {
        let mut mask = ObsMask::new();
        for (index, &bit) in bits.iter().enumerate() {
            if bit != 0 {
                mask.set(index);
            }
        }
        Self::from_mask_value(mask, bits.len())
    }

    fn normalize_index(&self, index: isize) -> PyResult<usize> {
        let normalized = if index < 0 {
            self.num_observables.checked_add_signed(index)
        } else {
            usize::try_from(index).ok()
        };
        normalized
            .filter(|&i| i < self.num_observables)
            .ok_or_else(|| {
                pyo3::exceptions::PyIndexError::new_err(format!(
                    "Observable index {index} out of range (num_observables={})",
                    self.num_observables
                ))
            })
    }

    fn validate_mask(
        mask: &ObsMask,
        mask_display: &str,
        num_observables: usize,
    ) -> Result<(), String> {
        if let Some(index) = mask
            .iter_set_bits()
            .filter(|&index| index >= num_observables)
            .max()
        {
            return Err(format!(
                "mask={mask_display} has bit {index} set at or above \
                 num_observables={num_observables}"
            ));
        }
        Ok(())
    }

    fn value_eq(&self, other: &Self) -> bool {
        self.num_observables == other.num_observables && self.mask == other.mask
    }
}

#[pymethods]
impl PyObservableFlips {
    fn __len__(&self) -> usize {
        self.num_observables
    }

    fn __getitem__(&self, index: isize) -> PyResult<bool> {
        self.normalize_index(index).map(|i| self.mask.get(i))
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let bits = (0..self.num_observables).map(|index| self.mask.get(index));
        Ok(PyList::new(py, bits)?.call_method0("__iter__")?.unbind())
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let Ok(other) = other.extract::<PyRef<'_, Self>>() else {
            return Ok(py.NotImplemented());
        };
        Ok(self
            .value_eq(&other)
            .into_pyobject(py)?
            .to_owned()
            .into_any()
            .unbind())
    }

    fn indices(&self) -> Vec<usize> {
        self.mask.iter_set_bits().collect()
    }

    #[getter]
    fn mask(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        obsmask_to_py(py, &self.mask)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let mask = obsmask_to_py(py, &self.mask)?;
        Ok(format!(
            "ObservableFlips(num_observables={}, mask={})",
            self.num_observables,
            mask.bind(py).str()?.to_str()?
        ))
    }

    #[staticmethod]
    fn from_mask(mask: &Bound<'_, PyAny>, num_observables: usize) -> PyResult<Self> {
        let mask = as_index(mask)?;
        let mask_display = mask.str()?.to_str()?.to_owned();
        if mask.lt(0)? {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "mask={mask_display} is negative; an observable mask is unsigned"
            )));
        }
        let mask_value = py_to_obsmask(&mask)?;
        Self::validate_mask(&mask_value, &mask_display, num_observables)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self::from_mask_value(mask_value, num_observables))
    }

    #[staticmethod]
    fn from_bits(bits: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut mask = ObsMask::new();
        let mut len = 0usize;
        for (index, item) in bits.try_iter()?.enumerate() {
            if bit_value(&item?, index)? {
                mask.set(index);
            }
            len = index + 1;
        }
        Ok(Self::from_mask_value(mask, len))
    }
}

/// Normalize an integer-like Python object to a true `int` via `__index__`.
///
/// This is the protocol Python itself uses wherever an integer is required, so
/// `bool` and NumPy integer scalars are accepted on the same footing as `int`.
/// This also accepts values returned by integer-oriented libraries, while
/// observable masks routinely arrive as NumPy scalars.
fn as_index<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    match value.call_method0("__index__") {
        Ok(index) => Ok(index),
        // A missing `__index__` means "not an integer", which Python reports as a
        // TypeError. A `__index__` that exists and raises is a real error: pass it on.
        Err(err) if err.is_instance_of::<pyo3::exceptions::PyAttributeError>(value.py()) => {
            let type_name = value
                .get_type()
                .name()
                .map_or_else(|_| "object".to_owned(), |name| name.to_string());
            Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "'{type_name}' object cannot be interpreted as an integer"
            )))
        }
        Err(err) => Err(err),
    }
}

/// Read one entry of a `from_bits` iterable.
///
/// Booleans (including NumPy booleans) are taken directly; anything integer-like
/// must be exactly 0 or 1. Truthiness is deliberately not used -- a non-bit value
/// is an error, not something to coerce.
fn bit_value(item: &Bound<'_, PyAny>, index: usize) -> PyResult<bool> {
    if let Ok(bit) = item.extract::<bool>() {
        return Ok(bit);
    }
    match as_index(item)?.extract::<i64>()? {
        0 => Ok(false),
        1 => Ok(true),
        value => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "bit at index {index} must be 0 or 1, got {value}"
        ))),
    }
}

/// Convert a wide observable mask to a Python integer (arbitrary precision).
pub(crate) fn obsmask_to_py(py: Python<'_>, mask: &ObsMask) -> PyResult<Py<PyAny>> {
    if let Some(value) = mask.to_u64() {
        return Ok(value.into_pyobject(py)?.into_any().unbind());
    }
    let mut bytes = Vec::with_capacity(mask.words().len() * 8);
    for &word in mask.words() {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    let py_bytes = PyBytes::new(py, &bytes);
    Ok(py
        .get_type::<PyInt>()
        .call_method1("from_bytes", (py_bytes, "little"))?
        .unbind())
}

/// Convert a Python integer (arbitrary precision) to a wide observable mask.
pub(crate) fn py_to_obsmask(value: &Bound<'_, PyAny>) -> PyResult<ObsMask> {
    let bit_length: usize = value.call_method0("bit_length")?.extract()?;
    let nbytes = bit_length.div_ceil(8).max(1);
    let bytes: Vec<u8> = value
        .call_method1("to_bytes", (nbytes, "little"))?
        .extract()?;
    let words = bytes
        .chunks(8)
        .map(|chunk| {
            let mut buf = [0u8; 8];
            buf[..chunk.len()].copy_from_slice(chunk);
            u64::from_le_bytes(buf)
        })
        .collect::<Vec<_>>();
    Ok(ObsMask::from_words(&words))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_normalization_checks_both_ends() {
        let flips = PyObservableFlips::from_u8_bits(&[1, 0]);

        assert_eq!(flips.normalize_index(0).unwrap(), 0);
        assert_eq!(flips.normalize_index(-1).unwrap(), 1);
        assert!(flips.normalize_index(2).is_err());
        assert!(flips.normalize_index(-3).is_err());
    }

    #[test]
    fn equality_includes_length() {
        let short = PyObservableFlips::from_mask_value(ObsMask::from_u64(1), 1);
        let long = PyObservableFlips::from_mask_value(ObsMask::from_u64(1), 2);

        assert!(!short.value_eq(&long));
    }

    #[test]
    fn mask_validation_rejects_bits_outside_length() {
        let mask = ObsMask::from_u64(4);
        let error = PyObservableFlips::validate_mask(&mask, "4", 2).unwrap_err();

        assert!(error.contains("mask=4"));
        assert!(error.contains("num_observables=2"));
    }
}
