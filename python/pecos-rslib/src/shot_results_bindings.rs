//! `PyO3` bindings for `ShotVec` and `ShotMap` types
//!
//! This module provides Python-friendly wrappers around the Rust shot result types,
//! allowing direct access to the data and providing convenient conversion methods.

use pecos::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

/// Python wrapper for `ShotVec`
#[pyclass(name = "ShotVec", module = "_pecos_rslib")]
pub struct PyShotVec {
    pub(crate) inner: ShotVec,
}

impl PyShotVec {
    /// Create a new `PyShotVec` from a Rust `ShotVec`
    pub fn new(inner: ShotVec) -> Self {
        PyShotVec { inner }
    }
}

#[pymethods]
impl PyShotVec {
    /// Get the number of shots
    #[getter]
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Convert to `ShotMap` for columnar access
    ///
    /// Returns:
    ///     `ShotMap`: A columnar representation of the shot data
    ///
    /// Raises:
    ///     `RuntimeError`: If conversion fails
    fn to_shot_map(&self) -> PyResult<PyShotMap> {
        let shot_map = self
            .inner
            .try_as_shot_map()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyShotMap { inner: shot_map })
    }

    /// Convert to a Python dictionary with integer values
    ///
    /// This is the default format, where bit vectors are converted to integers.
    ///
    /// Returns:
    ///     dict[str, list[int]]: Register names mapped to lists of integer values
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        shot_vec_to_dict_integers(py, &self.inner)
    }

    /// Convert to a Python dictionary with binary string values
    ///
    /// Bit vectors are formatted as binary strings (e.g., "0101").
    ///
    /// Returns:
    ///     dict[str, list[str]]: Register names mapped to lists of binary strings
    fn to_binary_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        shot_vec_to_dict_binary(py, &self.inner)
    }

    fn __repr__(&self) -> String {
        format!("ShotVec(shots={})", self.inner.len())
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

/// Python wrapper for `ShotMap`
#[pyclass(name = "ShotMap", module = "_pecos_rslib")]
pub struct PyShotMap {
    inner: ShotMap,
}

#[pymethods]
impl PyShotMap {
    /// Get all register names
    #[getter]
    fn register_names(&self) -> Vec<String> {
        self.inner
            .register_names()
            .into_iter()
            .map(std::string::ToString::to_string)
            .collect()
    }

    /// Get the number of shots
    #[getter]
    fn shots(&self) -> usize {
        self.inner.num_shots()
    }

    /// Get values from a register as integers
    ///
    /// Args:
    ///     register: Name of the register
    ///
    /// Returns:
    ///     list[int]: List of integer values
    ///
    /// Raises:
    ///     `RuntimeError`: If register doesn't exist or contains non-integer data
    fn get_integers(&self, register: &str) -> PyResult<Vec<i64>> {
        // Try different integer types in order
        if let Ok(u64_values) = self.inner.try_bits_as_u64(register) {
            // Convert u64 to i64, saturating at i64::MAX if the value is too large
            Ok(u64_values
                .into_iter()
                .map(|v| i64::try_from(v).unwrap_or(i64::MAX))
                .collect())
        } else if let Ok(i64_values) = self.inner.try_i64s(register) {
            Ok(i64_values)
        } else if let Ok(u32_values) = self.inner.try_u32s(register) {
            Ok(u32_values.into_iter().map(i64::from).collect())
        } else {
            Err(PyRuntimeError::new_err(format!(
                "Register '{register}' doesn't exist or contains non-integer data"
            )))
        }
    }

    /// Get values from a register as binary strings
    ///
    /// Args:
    ///     register: Name of the register
    ///
    /// Returns:
    ///     list[str]: List of binary string values (e.g., `["0101", "1010"]`)
    ///
    /// Raises:
    ///     `RuntimeError`: If register doesn't exist or contains non-bit data
    fn get_binary_strings(&self, register: &str) -> PyResult<Vec<String>> {
        self.inner
            .try_bits_as_binary(register)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Get values from a register as decimal strings
    ///
    /// Args:
    ///     register: Name of the register
    ///
    /// Returns:
    ///     list[str]: List of decimal string values
    ///
    /// Raises:
    ///     `RuntimeError`: If register doesn't exist or contains non-bit data
    fn get_decimal_strings(&self, register: &str) -> PyResult<Vec<String>> {
        self.inner
            .try_bits_as_decimal(register)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Get values from a register as hexadecimal strings
    ///
    /// Args:
    ///     register: Name of the register
    ///
    /// Returns:
    ///     list[str]: List of hex string values
    ///
    /// Raises:
    ///     `RuntimeError`: If register doesn't exist or contains non-bit data
    fn get_hex_strings(&self, register: &str) -> PyResult<Vec<String>> {
        self.inner
            .try_bits_as_hex(register)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    /// Convert to a Python dictionary with integer values
    ///
    /// Returns:
    ///     dict[str, list[int]]: Register names mapped to lists of integer values
    fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        shot_map_to_dict_integers(py, &self.inner)
    }

    /// Convert to a Python dictionary with binary string values
    ///
    /// Returns:
    ///     dict[str, list[str]]: Register names mapped to lists of binary strings
    fn to_binary_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        shot_map_to_dict_binary(py, &self.inner)
    }

    fn __repr__(&self) -> String {
        let registers = self.inner.register_names().join(", ");
        format!(
            "ShotMap(shots={}, registers=[{}])",
            self.inner.num_shots(),
            registers
        )
    }
}

// Helper functions for conversion

/// Convert `ShotVec` to Python dict with integer values
pub(crate) fn shot_vec_to_dict_integers(py: Python<'_>, shot_vec: &ShotVec) -> PyResult<Py<PyAny>> {
    let shot_map = shot_vec
        .try_as_shot_map()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    shot_map_to_dict_integers(py, &shot_map)
}

/// Convert `ShotVec` to Python dict with binary string values
pub(crate) fn shot_vec_to_dict_binary(py: Python<'_>, shot_vec: &ShotVec) -> PyResult<Py<PyAny>> {
    let shot_map = shot_vec
        .try_as_shot_map()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    shot_map_to_dict_binary(py, &shot_map)
}

/// Convert `ShotMap` to Python dict with integer values
pub(crate) fn shot_map_to_dict_integers(py: Python<'_>, shot_map: &ShotMap) -> PyResult<Py<PyAny>> {
    let py_dict = PyDict::new(py);

    for reg_name in shot_map.register_names() {
        let py_list = PyList::empty(py);

        // Try different data types in order
        if let Ok(biguint_values) = shot_map.try_bits_as_biguint(reg_name) {
            // Convert BigUint to Python integers
            for val in biguint_values {
                let bytes = val.to_bytes_le();
                let py_int: Py<PyAny> = if bytes.is_empty() {
                    0u32.into_pyobject(py)?.into()
                } else {
                    let py_bytes = PyBytes::new(py, &bytes);
                    let int_type = py.import("builtins")?.getattr("int")?;
                    int_type
                        .call_method1("from_bytes", (py_bytes, "little"))?
                        .into()
                };
                py_list.append(py_int)?;
            }
            py_dict.set_item(reg_name, py_list)?;
        } else if let Ok(u32_values) = shot_map.try_u32s(reg_name) {
            for val in u32_values {
                py_list.append(val)?;
            }
            py_dict.set_item(reg_name, py_list)?;
        } else if let Ok(i64_values) = shot_map.try_i64s(reg_name) {
            for val in i64_values {
                py_list.append(val)?;
            }
            py_dict.set_item(reg_name, py_list)?;
        } else if let Ok(f64_values) = shot_map.try_f64s(reg_name) {
            for val in f64_values {
                py_list.append(val)?;
            }
            py_dict.set_item(reg_name, py_list)?;
        } else if let Ok(bool_values) = shot_map.try_bools(reg_name) {
            for val in bool_values {
                py_list.append(val)?;
            }
            py_dict.set_item(reg_name, py_list)?;
        }
        // Skip registers we can't handle
    }

    Ok(py_dict.into())
}

/// Convert `ShotMap` to Python dict with binary string values
pub(crate) fn shot_map_to_dict_binary(py: Python<'_>, shot_map: &ShotMap) -> PyResult<Py<PyAny>> {
    let py_dict = PyDict::new(py);

    for reg_name in shot_map.register_names() {
        let py_list = PyList::empty(py);

        // Try to get as binary strings
        if let Ok(binary_values) = shot_map.try_bits_as_binary(reg_name) {
            for val in binary_values {
                py_list.append(val.into_pyobject(py)?)?;
            }
            py_dict.set_item(reg_name, py_list)?;
        } else if let Ok(u32_values) = shot_map.try_u32s(reg_name) {
            // Fallback for non-bit data
            for val in u32_values {
                py_list.append(val)?;
            }
            py_dict.set_item(reg_name, py_list)?;
        } else if let Ok(i64_values) = shot_map.try_i64s(reg_name) {
            for val in i64_values {
                py_list.append(val)?;
            }
            py_dict.set_item(reg_name, py_list)?;
        } else if let Ok(f64_values) = shot_map.try_f64s(reg_name) {
            for val in f64_values {
                py_list.append(val)?;
            }
            py_dict.set_item(reg_name, py_list)?;
        } else if let Ok(bool_values) = shot_map.try_bools(reg_name) {
            for val in bool_values {
                py_list.append(val)?;
            }
            py_dict.set_item(reg_name, py_list)?;
        }
        // Skip registers we can't handle
    }

    Ok(py_dict.into())
}

impl From<ShotVec> for PyShotVec {
    fn from(shot_vec: ShotVec) -> Self {
        PyShotVec { inner: shot_vec }
    }
}

impl From<ShotMap> for PyShotMap {
    fn from(shot_map: ShotMap) -> Self {
        PyShotMap { inner: shot_map }
    }
}
