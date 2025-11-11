// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Rust-backed dtype system and scalar types for PECOS numerical computing.
//!
//! This module provides:
//! - A clean, type-safe dtype system with Rust naming conventions
//! - Rust-backed scalar types (F64, I64, Complex128, etc.)

// Allow Clippy pedantic lints that are not applicable to PyO3 bindings
#![allow(clippy::trivially_copy_pass_by_ref)] // PyO3 requires &self for methods
#![allow(clippy::match_same_arms)] // Intentional duplication for clarity
#![allow(clippy::unused_self)] // PyO3 property getters require &self
#![allow(clippy::wrong_self_convention)] // to_* methods are correct in this context

use num_complex::Complex64;
use pyo3::prelude::*;

/// Dtype enum representing supported data types
#[pyclass(name = "DType", module = "pecos_rslib.dtypes")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// 64-bit floating point (f64, double precision)
    F64,
    /// 32-bit floating point (f32, single precision)
    F32,
    /// 64-bit integer (i64, signed long)
    I64,
    /// 32-bit integer (i32, signed int)
    I32,
    /// 16-bit integer (i16, signed short)
    I16,
    /// 8-bit integer (i8, signed byte)
    I8,
    /// 128-bit complex (Complex<f64>, double precision complex)
    Complex128,
    /// 64-bit complex (Complex<f32>, single precision complex)
    Complex64,
}

#[pymethods]
impl DType {
    /// String representation of the dtype
    fn __repr__(&self) -> String {
        match self {
            DType::F64 => "dtypes.f64".to_string(),
            DType::F32 => "dtypes.f32".to_string(),
            DType::I64 => "dtypes.i64".to_string(),
            DType::I32 => "dtypes.i32".to_string(),
            DType::I16 => "dtypes.i16".to_string(),
            DType::I8 => "dtypes.i8".to_string(),
            DType::Complex128 => "dtypes.complex128".to_string(),
            DType::Complex64 => "dtypes.complex64".to_string(),
        }
    }

    /// String name of the dtype
    #[allow(clippy::trivially_copy_pass_by_ref)] // PyO3 requires &self for __str__
    fn __str__(&self) -> String {
        self.to_numpy_str().to_string()
    }

    /// Convert to NumPy-compatible dtype string (Python method)
    #[pyo3(name = "numpy_str")]
    #[allow(clippy::trivially_copy_pass_by_ref)] // PyO3 requires &self for methods
    fn py_numpy_str(&self) -> &'static str {
        self.to_numpy_str()
    }

    /// Check if this is a floating point dtype
    #[getter]
    #[allow(clippy::trivially_copy_pass_by_ref)] // PyO3 requires &self for getters
    fn is_float(&self) -> bool {
        matches!(self, DType::F64 | DType::F32)
    }

    /// Check if this is an integer dtype
    #[getter]
    #[allow(clippy::trivially_copy_pass_by_ref)] // PyO3 requires &self for getters
    fn is_int(&self) -> bool {
        matches!(self, DType::I64 | DType::I32 | DType::I16 | DType::I8)
    }

    /// Check if this is a complex dtype
    #[getter]
    #[allow(clippy::trivially_copy_pass_by_ref)] // PyO3 requires &self for getters
    fn is_complex(&self) -> bool {
        matches!(self, DType::Complex128 | DType::Complex64)
    }

    /// Item size in bytes
    #[getter]
    fn itemsize(&self) -> usize {
        match self {
            DType::F64 => 8,
            DType::F32 => 4,
            DType::I64 => 8,
            DType::I32 => 4,
            DType::I16 => 2,
            DType::I8 => 1,
            DType::Complex128 => 16,
            DType::Complex64 => 8,
        }
    }

    /// Make `DType` callable as a type constructor (returns Rust-backed scalars)
    fn __call__<'py>(&self, py: Python<'py>, value: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        match self {
            DType::F64 => {
                // Convert to f64 and create Rust-backed scalar
                let float_val = value.extract::<f64>()?;
                Ok(Py::new(py, ScalarF64::new(float_val))?.into_any())
            }
            DType::F32 => {
                // For now, convert f32 to f64 scalar (we can add ScalarF32 later if needed)
                let float_val = f64::from(value.extract::<f32>()?);
                Ok(Py::new(py, ScalarF64::new(float_val))?.into_any())
            }
            DType::I64 => {
                // Convert to i64 and create Rust-backed scalar
                let int_val = value.extract::<i64>()?;
                Ok(Py::new(py, ScalarI64::new(int_val))?.into_any())
            }
            DType::I32 => {
                // For now, convert i32 to i64 scalar (we can add ScalarI32 later if needed)
                let int_val = i64::from(value.extract::<i32>()?);
                Ok(Py::new(py, ScalarI64::new(int_val))?.into_any())
            }
            DType::I16 => {
                // Convert i16 to i64 scalar
                let int_val = i64::from(value.extract::<i16>()?);
                Ok(Py::new(py, ScalarI64::new(int_val))?.into_any())
            }
            DType::I8 => {
                // Convert i8 to i64 scalar
                let int_val = i64::from(value.extract::<i8>()?);
                Ok(Py::new(py, ScalarI64::new(int_val))?.into_any())
            }
            DType::Complex128 => {
                // Convert to Complex64 and create Rust-backed scalar
                let complex_val = value.extract::<Complex64>()?;
                Ok(Py::new(py, ScalarComplex128::new(complex_val))?.into_any())
            }
            DType::Complex64 => {
                // For now, convert to Complex128 scalar (we can add ScalarComplex64 later)
                let complex_val = value.extract::<Complex64>()?;
                Ok(Py::new(py, ScalarComplex128::new(complex_val))?.into_any())
            }
        }
    }
}

impl DType {
    /// Convert to NumPy-compatible dtype string (public Rust method)
    pub fn to_numpy_str(&self) -> &'static str {
        match self {
            DType::F64 => "float64",
            DType::F32 => "float32",
            DType::I64 => "int64",
            DType::I32 => "int32",
            DType::I16 => "int16",
            DType::I8 => "int8",
            DType::Complex128 => "complex128",
            DType::Complex64 => "complex64",
        }
    }

    /// Parse from a string (supports both Rust-style and NumPy-style names)
    pub fn from_str(s: &str) -> PyResult<Self> {
        match s.to_lowercase().as_str() {
            // Rust-style names
            "f64" | "float64" => Ok(DType::F64),
            "f32" | "float32" => Ok(DType::F32),
            "i64" | "int64" => Ok(DType::I64),
            "i32" | "int32" => Ok(DType::I32),
            "i16" | "int16" => Ok(DType::I16),
            "i8" | "int8" => Ok(DType::I8),
            "complex128" | "complex" => Ok(DType::Complex128),
            "complex64" => Ok(DType::Complex64),
            // Common aliases
            "double" => Ok(DType::F64),
            "float" => Ok(DType::F32),
            "long" | "int" => Ok(DType::I64),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown dtype: {s}"
            ))),
        }
    }
}

// ============================================================================
// Rust-backed Scalar Types
// ============================================================================

/// Rust-backed f64 scalar
#[pyclass(name = "f64", module = "pecos_rslib.dtypes")]
#[derive(Debug, Clone, Copy)]
pub struct ScalarF64 {
    value: f64,
}

#[pymethods]
impl ScalarF64 {
    #[new]
    fn new(value: f64) -> Self {
        Self { value }
    }

    fn __repr__(&self) -> String {
        format!("f64({})", self.value)
    }

    fn __str__(&self) -> String {
        self.value.to_string()
    }

    fn __float__(&self) -> f64 {
        self.value
    }

    /// Convert to `NumPy` scalar
    fn as_np<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let np = py.import("numpy")?;
        np.getattr("float64")?.call1((self.value,))
    }

    /// Get the dtype
    #[getter]
    fn dtype(&self) -> DType {
        DType::F64
    }
}

/// Rust-backed i64 scalar
#[pyclass(name = "i64", module = "pecos_rslib.dtypes")]
#[derive(Debug, Clone, Copy)]
pub struct ScalarI64 {
    value: i64,
}

#[pymethods]
impl ScalarI64 {
    #[new]
    fn new(value: i64) -> Self {
        Self { value }
    }

    fn __repr__(&self) -> String {
        format!("i64({})", self.value)
    }

    fn __str__(&self) -> String {
        self.value.to_string()
    }

    fn __int__(&self) -> i64 {
        self.value
    }

    /// Convert to `NumPy` scalar
    fn as_np<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let np = py.import("numpy")?;
        np.getattr("int64")?.call1((self.value,))
    }

    /// Get the dtype
    #[getter]
    fn dtype(&self) -> DType {
        DType::I64
    }
}

/// Rust-backed complex128 scalar
#[pyclass(name = "complex128", module = "pecos_rslib.dtypes")]
#[derive(Debug, Clone, Copy)]
pub struct ScalarComplex128 {
    value: Complex64,
}

#[pymethods]
impl ScalarComplex128 {
    #[new]
    fn new(value: Complex64) -> Self {
        Self { value }
    }

    fn __repr__(&self) -> String {
        format!("complex128({}+{}j)", self.value.re, self.value.im)
    }

    fn __str__(&self) -> String {
        format!("{}+{}j", self.value.re, self.value.im)
    }

    fn __complex__(&self) -> Complex64 {
        self.value
    }

    /// Get real part
    #[getter]
    fn real(&self) -> f64 {
        self.value.re
    }

    /// Get imaginary part
    #[getter]
    fn imag(&self) -> f64 {
        self.value.im
    }

    /// Convert to `NumPy` scalar
    fn as_np<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let np = py.import("numpy")?;
        let py_complex = pyo3::types::PyComplex::from_doubles(py, self.value.re, self.value.im);
        np.getattr("complex128")?.call1((py_complex,))
    }

    /// Get the dtype
    #[getter]
    fn dtype(&self) -> DType {
        DType::Complex128
    }
}

/// Module constants for dtype singletons
pub fn register_dtypes_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let dtypes = PyModule::new(parent_module.py(), "dtypes")?;

    // Register the DType class
    dtypes.add_class::<DType>()?;

    // Register scalar types
    dtypes.add_class::<ScalarF64>()?;
    dtypes.add_class::<ScalarI64>()?;
    dtypes.add_class::<ScalarComplex128>()?;

    // Create singleton instances for each dtype
    dtypes.add("f64", DType::F64)?;
    dtypes.add("f32", DType::F32)?;
    dtypes.add("i64", DType::I64)?;
    dtypes.add("i32", DType::I32)?;
    dtypes.add("complex128", DType::Complex128)?;
    dtypes.add("complex64", DType::Complex64)?;

    // Aliases for convenience
    dtypes.add("float64", DType::F64)?;
    dtypes.add("float32", DType::F32)?;
    dtypes.add("int64", DType::I64)?;
    dtypes.add("int32", DType::I32)?;
    dtypes.add("complex", DType::Complex128)?; // Default complex is 128-bit

    // More intuitive aliases
    dtypes.add("float", DType::F64)?; // Default float is 64-bit
    dtypes.add("int", DType::I64)?; // Default int is 64-bit

    parent_module.add_submodule(&dtypes)?;
    Ok(())
}
