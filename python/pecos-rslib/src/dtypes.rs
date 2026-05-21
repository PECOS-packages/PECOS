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
use pecos_core::Angle64;
use pyo3::basic::CompareOp;
use pyo3::prelude::*;
use pyo3::types::PyBool;

/// Dtype enum representing supported data types
#[pyclass(name = "DType", module = "pecos_rslib.dtypes", from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// Boolean (bool)
    Bool,
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
    /// 64-bit unsigned integer (u64, unsigned long)
    U64,
    /// 32-bit unsigned integer (u32, unsigned int)
    U32,
    /// 16-bit unsigned integer (u16, unsigned short)
    U16,
    /// 8-bit unsigned integer (u8, unsigned byte)
    U8,
    /// 128-bit complex (Complex<f64>, double precision complex)
    Complex128,
    /// 64-bit complex (Complex<f32>, single precision complex)
    Complex64,
    /// Pauli operator (I, X, Y, Z)
    Pauli,
    /// Pauli string (sequence of Pauli operators)
    PauliString,
}

#[pymethods]
impl DType {
    /// String representation of the dtype
    fn __repr__(&self) -> String {
        match self {
            DType::Bool => "dtypes.bool".to_string(),
            DType::F64 => "dtypes.f64".to_string(),
            DType::F32 => "dtypes.f32".to_string(),
            DType::I64 => "dtypes.i64".to_string(),
            DType::I32 => "dtypes.i32".to_string(),
            DType::I16 => "dtypes.i16".to_string(),
            DType::I8 => "dtypes.i8".to_string(),
            DType::U64 => "dtypes.u64".to_string(),
            DType::U32 => "dtypes.u32".to_string(),
            DType::U16 => "dtypes.u16".to_string(),
            DType::U8 => "dtypes.u8".to_string(),
            DType::Complex128 => "dtypes.complex128".to_string(),
            DType::Complex64 => "dtypes.complex64".to_string(),
            DType::Pauli => "dtypes.pauli".to_string(),
            DType::PauliString => "dtypes.paulistring".to_string(),
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

    /// Check if this is an integer dtype (signed or unsigned)
    #[getter]
    #[allow(clippy::trivially_copy_pass_by_ref)] // PyO3 requires &self for getters
    fn is_int(&self) -> bool {
        matches!(
            self,
            DType::I64
                | DType::I32
                | DType::I16
                | DType::I8
                | DType::U64
                | DType::U32
                | DType::U16
                | DType::U8
        )
    }

    /// Check if this is a complex dtype
    #[getter]
    #[allow(clippy::trivially_copy_pass_by_ref)] // PyO3 requires &self for getters
    fn is_complex(&self) -> bool {
        matches!(self, DType::Complex128 | DType::Complex64)
    }

    /// Check if this is a boolean dtype
    #[getter]
    #[allow(clippy::trivially_copy_pass_by_ref)] // PyO3 requires &self for getters
    fn is_bool(&self) -> bool {
        matches!(self, DType::Bool)
    }

    /// Item size in bytes
    #[getter]
    fn itemsize(&self) -> usize {
        match self {
            DType::Bool => 1,
            DType::F64 => 8,
            DType::F32 => 4,
            DType::I64 => 8,
            DType::I32 => 4,
            DType::I16 => 2,
            DType::I8 => 1,
            DType::U64 => 8,
            DType::U32 => 4,
            DType::U16 => 2,
            DType::U8 => 1,
            DType::Complex128 => 16,
            DType::Complex64 => 8,
            DType::Pauli => 1,       // Pauli is stored as 2 bits but we use 1 byte
            DType::PauliString => 8, // PauliString size varies, return pointer size
        }
    }

    /// Minimum value for numeric types (None for non-numeric types)
    /// For integers: returns the minimum value
    /// For unsigned integers: returns 0
    /// For floats: returns the smallest finite value (most negative)
    /// For other types: returns None
    #[getter]
    fn min<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        match self {
            // Signed integers
            DType::I64 => Ok(Some(i64::MIN.into_pyobject(py)?.into_any())),
            DType::I32 => Ok(Some(i32::MIN.into_pyobject(py)?.into_any())),
            DType::I16 => Ok(Some(i16::MIN.into_pyobject(py)?.into_any())),
            DType::I8 => Ok(Some(i8::MIN.into_pyobject(py)?.into_any())),
            // Unsigned integers - min is always 0
            DType::U64 => Ok(Some(0_u64.into_pyobject(py)?.into_any())),
            DType::U32 => Ok(Some(0_u32.into_pyobject(py)?.into_any())),
            DType::U16 => Ok(Some(0_u16.into_pyobject(py)?.into_any())),
            DType::U8 => Ok(Some(0_u8.into_pyobject(py)?.into_any())),
            // Floats - smallest finite value
            DType::F64 => Ok(Some(f64::MIN.into_pyobject(py)?.into_any())),
            DType::F32 => Ok(Some(f32::MIN.into_pyobject(py)?.into_any())),
            // Other types don't have a meaningful min
            _ => Ok(None),
        }
    }

    /// Maximum value for numeric types (None for non-numeric types)
    /// For integers: returns the maximum value
    /// For floats: returns the largest finite value
    /// For other types: returns None
    #[getter]
    fn max<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        match self {
            // Signed integers
            DType::I64 => Ok(Some(i64::MAX.into_pyobject(py)?.into_any())),
            DType::I32 => Ok(Some(i32::MAX.into_pyobject(py)?.into_any())),
            DType::I16 => Ok(Some(i16::MAX.into_pyobject(py)?.into_any())),
            DType::I8 => Ok(Some(i8::MAX.into_pyobject(py)?.into_any())),
            // Unsigned integers
            DType::U64 => Ok(Some(u64::MAX.into_pyobject(py)?.into_any())),
            DType::U32 => Ok(Some(u32::MAX.into_pyobject(py)?.into_any())),
            DType::U16 => Ok(Some(u16::MAX.into_pyobject(py)?.into_any())),
            DType::U8 => Ok(Some(u8::MAX.into_pyobject(py)?.into_any())),
            // Floats - largest finite value
            DType::F64 => Ok(Some(f64::MAX.into_pyobject(py)?.into_any())),
            DType::F32 => Ok(Some(f32::MAX.into_pyobject(py)?.into_any())),
            // Other types don't have a meaningful max
            _ => Ok(None),
        }
    }

    /// Python rich comparison (allows comparison with `NumPy` dtypes)
    fn __richcmp__(
        &self,
        other: &Bound<'_, PyAny>,
        op: pyo3::pyclass::CompareOp,
    ) -> PyResult<Py<PyAny>> {
        use pyo3::pyclass::CompareOp;

        let py = other.py();

        // Try to convert other to DType
        let other_dtype: Option<DType> = if other.is_instance_of::<DType>() {
            // Direct DType comparison
            Some(other.extract::<DType>()?)
        } else if other.hasattr("name")? {
            // NumPy dtype instance comparison - get the name and convert to DType
            let name: String = other.getattr("name")?.extract()?;
            DType::from_str(&name).ok()
        } else if other.hasattr("__name__")? {
            // NumPy scalar type class comparison (e.g., np.float64)
            let name: String = other.getattr("__name__")?.extract()?;
            DType::from_str(&name).ok()
        } else {
            None
        };

        let result = match (op, other_dtype) {
            (CompareOp::Eq, Some(other)) => self == &other,
            (CompareOp::Ne, Some(other)) => self != &other,
            (CompareOp::Eq, None) => false, // Can't compare, so not equal
            (CompareOp::Ne, None) => true,  // Can't compare, so not equal
            _ => {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "DType only supports == and != comparisons",
                ));
            }
        };

        Ok(pyo3::types::PyBool::new(py, result)
            .to_owned()
            .into_any()
            .unbind())
    }

    /// Python hash implementation
    fn __hash__(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        (*self as u8).hash(&mut hasher);
        hasher.finish()
    }

    /// Make `DType` callable as a type constructor (returns Rust-backed scalars)
    fn __call__<'py>(&self, py: Python<'py>, value: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        match self {
            DType::Bool => {
                // Convert to bool and return as Python bool
                let bool_val = value.extract::<bool>()?;
                Ok(PyBool::new(py, bool_val).to_owned().into_any().unbind())
            }
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
                // Try i64 first, then u64 (for values > i64::MAX that truncate)
                #[allow(clippy::cast_possible_wrap)]
                let int_val = if let Ok(v) = value.extract::<i64>() {
                    v
                } else if let Ok(v) = value.extract::<u64>() {
                    v as i64
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "i64() argument must be an integer",
                    ));
                };
                Ok(Py::new(py, ScalarI64::new(int_val))?.into_any())
            }
            DType::I32 => {
                let int_val = value.extract::<i64>()?;
                Ok(Py::new(py, ScalarI32::new(int_val))?.into_any())
            }
            DType::I16 => {
                let int_val = value.extract::<i64>()?;
                Ok(Py::new(py, ScalarI16::new(int_val))?.into_any())
            }
            DType::I8 => {
                let int_val = value.extract::<i64>()?;
                Ok(Py::new(py, ScalarI8::new(int_val))?.into_any())
            }
            #[allow(clippy::cast_sign_loss)]
            DType::U64 => {
                let int_val = if let Ok(v) = value.extract::<u64>() {
                    v
                } else {
                    value.extract::<i64>()? as u64
                };
                Ok(Py::new(py, ScalarU64::new(int_val))?.into_any())
            }
            DType::U32 => {
                let int_val = value.extract::<i64>()?;
                Ok(Py::new(py, ScalarU32::new(int_val))?.into_any())
            }
            DType::U16 => {
                let int_val = value.extract::<i64>()?;
                Ok(Py::new(py, ScalarU16::new(int_val))?.into_any())
            }
            DType::U8 => {
                let int_val = value.extract::<i64>()?;
                Ok(Py::new(py, ScalarU8::new(int_val))?.into_any())
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
            DType::Pauli => {
                // Import Pauli type
                use crate::pauli_bindings::Pauli;

                // Try to extract as Pauli directly
                if let Ok(pauli) = value.extract::<Pauli>() {
                    return Ok(Py::new(py, pauli)?.into_any());
                }

                // Try to convert from string
                if let Ok(s) = value.extract::<&str>() {
                    let pauli = Pauli::from_str(s)?;
                    return Ok(Py::new(py, pauli)?.into_any());
                }

                Err(pyo3::exceptions::PyTypeError::new_err(
                    "Value must be a Pauli or string ('I', 'X', 'Y', 'Z')",
                ))
            }
            DType::PauliString => {
                // Import PauliString type
                use crate::pauli_bindings::PauliString;

                // Try to extract as PauliString directly
                if let Ok(ps) = value.extract::<PauliString>() {
                    return Ok(Py::new(py, ps)?.into_any());
                }

                Err(pyo3::exceptions::PyTypeError::new_err(
                    "Value must be a PauliString",
                ))
            }
        }
    }

    /// NumPy-compatible `.type` property that returns the scalar class
    ///
    /// In `NumPy`, `arr.dtype.type` returns the scalar class (e.g., np.int64 class).
    /// This allows code like: `dtype_cls = arr.dtype.type; val = dtype_cls(42)`
    #[getter]
    fn r#type(&self, py: Python<'_>) -> Py<PyAny> {
        match self {
            DType::Bool => {
                // For Bool, return Python's bool type
                py.get_type::<PyBool>().into_any().unbind()
            }
            DType::F64 => {
                // Return the ScalarF64 class
                py.get_type::<ScalarF64>().into_any().unbind()
            }
            DType::F32 => {
                // Return the ScalarF32 class
                py.get_type::<ScalarF32>().into_any().unbind()
            }
            DType::I64 => {
                // Return the ScalarI64 class
                py.get_type::<ScalarI64>().into_any().unbind()
            }
            DType::I32 => {
                // Return the ScalarI32 class
                py.get_type::<ScalarI32>().into_any().unbind()
            }
            DType::I16 => {
                // Return the ScalarI16 class
                py.get_type::<ScalarI16>().into_any().unbind()
            }
            DType::I8 => {
                // Return the ScalarI8 class
                py.get_type::<ScalarI8>().into_any().unbind()
            }
            DType::U64 => {
                // Return the ScalarU64 class
                py.get_type::<ScalarU64>().into_any().unbind()
            }
            DType::U32 => {
                // Return the ScalarU32 class
                py.get_type::<ScalarU32>().into_any().unbind()
            }
            DType::U16 => {
                // Return the ScalarU16 class
                py.get_type::<ScalarU16>().into_any().unbind()
            }
            DType::U8 => {
                // Return the ScalarU8 class
                py.get_type::<ScalarU8>().into_any().unbind()
            }
            DType::Complex128 | DType::Complex64 => {
                // Return the ScalarComplex128 class
                py.get_type::<ScalarComplex128>().into_any().unbind()
            }
            DType::Pauli => {
                // Return the Pauli class
                use crate::pauli_bindings::Pauli;
                py.get_type::<Pauli>().into_any().unbind()
            }
            DType::PauliString => {
                // Return the PauliString class
                use crate::pauli_bindings::PauliString;
                py.get_type::<PauliString>().into_any().unbind()
            }
        }
    }
}

impl DType {
    /// Convert to NumPy-compatible dtype string (public Rust method)
    pub fn to_numpy_str(&self) -> &'static str {
        match self {
            DType::Bool => "bool",
            DType::F64 => "float64",
            DType::F32 => "float32",
            DType::I64 => "int64",
            DType::I32 => "int32",
            DType::I16 => "int16",
            DType::I8 => "int8",
            DType::U64 => "uint64",
            DType::U32 => "uint32",
            DType::U16 => "uint16",
            DType::U8 => "uint8",
            DType::Complex128 => "complex128",
            DType::Complex64 => "complex64",
            DType::Pauli => "object", // Pauli arrays are stored as object arrays in NumPy
            DType::PauliString => "object", // PauliString arrays are stored as object arrays in NumPy
        }
    }

    /// Parse from a string (supports both Rust-style and NumPy-style names)
    pub fn from_str(s: &str) -> PyResult<Self> {
        match s.to_lowercase().as_str() {
            // Boolean type
            "bool" => Ok(DType::Bool),
            // Rust-style names (signed integers)
            "f64" | "float64" => Ok(DType::F64),
            "f32" | "float32" => Ok(DType::F32),
            "i64" | "int64" => Ok(DType::I64),
            "i32" | "int32" => Ok(DType::I32),
            "i16" | "int16" => Ok(DType::I16),
            "i8" | "int8" => Ok(DType::I8),
            // Unsigned integers
            "u64" | "uint64" => Ok(DType::U64),
            "u32" | "uint32" => Ok(DType::U32),
            "u16" | "uint16" => Ok(DType::U16),
            "u8" | "uint8" => Ok(DType::U8),
            // Complex numbers
            "complex128" | "complex" => Ok(DType::Complex128),
            "complex64" => Ok(DType::Complex64),
            // Pauli types
            "pauli" => Ok(DType::Pauli),
            "paulistring" => Ok(DType::PauliString),
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
#[pyclass(name = "f64", module = "pecos_rslib.dtypes", from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct ScalarF64 {
    value: f64,
}

#[pymethods]
impl ScalarF64 {
    /// Item size in bytes (class attribute)
    #[classattr]
    #[allow(non_upper_case_globals)] // Python API expects lowercase 'itemsize'
    const itemsize: usize = 8;

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

    /// Rich comparison support for ==, !=, <, <=, >, >=
    fn __richcmp__(&self, other: &Bound<PyAny>, op: CompareOp) -> PyResult<bool> {
        let other_value = if other.is_instance_of::<ScalarF64>() {
            other.extract::<ScalarF64>()?.value
        } else if let Ok(val) = other.extract::<f64>() {
            val
        } else if let Ok(val) = other.extract::<i64>() {
            // Allow precision loss: This is intentional for NumPy compatibility.
            // When comparing f64 with i64, NumPy converts i64 to f64, accepting
            // potential precision loss for large integers beyond f64's mantissa range.
            #[allow(clippy::cast_precision_loss)]
            let result = val as f64;
            result
        } else {
            return Ok(false);
        };

        // Allow exact float comparison: This is intentional for NumPy compatibility.
        // NumPy uses exact bitwise equality for == and != on floats. While this can
        // be surprising with floating-point arithmetic, it matches NumPy's behavior.
        #[allow(clippy::float_cmp)]
        match op {
            CompareOp::Lt => Ok(self.value < other_value),
            CompareOp::Le => Ok(self.value <= other_value),
            CompareOp::Eq => Ok(self.value == other_value),
            CompareOp::Ne => Ok(self.value != other_value),
            CompareOp::Gt => Ok(self.value > other_value),
            CompareOp::Ge => Ok(self.value >= other_value),
        }
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

    // Mathematical constants (f64 precision)
    #[classattr]
    #[allow(non_upper_case_globals)]
    const pi: f64 = std::f64::consts::PI;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const tau: f64 = std::f64::consts::TAU;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const e: f64 = std::f64::consts::E;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const frac_pi_2: f64 = std::f64::consts::FRAC_PI_2;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const frac_pi_3: f64 = std::f64::consts::FRAC_PI_3;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const frac_pi_4: f64 = std::f64::consts::FRAC_PI_4;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const frac_pi_6: f64 = std::f64::consts::FRAC_PI_6;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const frac_pi_8: f64 = std::f64::consts::FRAC_PI_8;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const frac_1_pi: f64 = std::f64::consts::FRAC_1_PI;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const frac_2_pi: f64 = std::f64::consts::FRAC_2_PI;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const frac_2_sqrt_pi: f64 = std::f64::consts::FRAC_2_SQRT_PI;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const sqrt_2: f64 = std::f64::consts::SQRT_2;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const frac_1_sqrt_2: f64 = std::f64::consts::FRAC_1_SQRT_2;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const ln_2: f64 = std::f64::consts::LN_2;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const ln_10: f64 = std::f64::consts::LN_10;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const log2_e: f64 = std::f64::consts::LOG2_E;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const log10_e: f64 = std::f64::consts::LOG10_E;
}

/// Rust-backed f32 scalar
#[pyclass(name = "f32", module = "pecos_rslib.dtypes", from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct ScalarF32 {
    value: f32,
}

#[pymethods]
impl ScalarF32 {
    /// Item size in bytes (class attribute)
    #[classattr]
    #[allow(non_upper_case_globals)] // Python API expects lowercase 'itemsize'
    const itemsize: usize = 4;

    #[new]
    fn new(value: f32) -> Self {
        Self { value }
    }

    fn __repr__(&self) -> String {
        format!("f32({})", self.value)
    }

    fn __str__(&self) -> String {
        self.value.to_string()
    }

    fn __float__(&self) -> f32 {
        self.value
    }

    /// Rich comparison support for ==, !=, <, <=, >, >=
    fn __richcmp__(&self, other: &Bound<PyAny>, op: CompareOp) -> PyResult<bool> {
        let other_value = if other.is_instance_of::<ScalarF32>() {
            other.extract::<ScalarF32>()?.value
        } else if let Ok(val) = other.extract::<f32>() {
            val
        } else if let Ok(val) = other.extract::<i32>() {
            // Allow precision loss: Intentional for Python compatibility
            // f32 mantissa is 23 bits, so i32 values may lose precision
            #[allow(clippy::cast_precision_loss)]
            let result = val as f32;
            result
        } else {
            return Ok(false);
        };

        // Allow exact float comparison: This is intentional for NumPy compatibility.
        // NumPy uses exact bitwise equality for == and != on floats. While this can
        // be surprising with floating-point arithmetic, it matches NumPy's behavior.
        #[allow(clippy::float_cmp)]
        match op {
            CompareOp::Lt => Ok(self.value < other_value),
            CompareOp::Le => Ok(self.value <= other_value),
            CompareOp::Eq => Ok(self.value == other_value),
            CompareOp::Ne => Ok(self.value != other_value),
            CompareOp::Gt => Ok(self.value > other_value),
            CompareOp::Ge => Ok(self.value >= other_value),
        }
    }

    /// Convert to `NumPy` scalar
    fn as_np<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let np = py.import("numpy")?;
        np.getattr("float32")?.call1((self.value,))
    }

    /// Get the dtype
    #[getter]
    fn dtype(&self) -> DType {
        DType::F32
    }

    // Mathematical constants (f32 precision)
    #[classattr]
    #[allow(non_upper_case_globals)]
    const pi: f32 = std::f32::consts::PI;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const tau: f32 = std::f32::consts::TAU;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const e: f32 = std::f32::consts::E;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const frac_pi_2: f32 = std::f32::consts::FRAC_PI_2;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const frac_pi_3: f32 = std::f32::consts::FRAC_PI_3;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const frac_pi_4: f32 = std::f32::consts::FRAC_PI_4;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const frac_pi_6: f32 = std::f32::consts::FRAC_PI_6;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const frac_pi_8: f32 = std::f32::consts::FRAC_PI_8;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const frac_1_pi: f32 = std::f32::consts::FRAC_1_PI;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const frac_2_pi: f32 = std::f32::consts::FRAC_2_PI;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const frac_2_sqrt_pi: f32 = std::f32::consts::FRAC_2_SQRT_PI;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const sqrt_2: f32 = std::f32::consts::SQRT_2;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const frac_1_sqrt_2: f32 = std::f32::consts::FRAC_1_SQRT_2;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const ln_2: f32 = std::f32::consts::LN_2;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const ln_10: f32 = std::f32::consts::LN_10;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const log2_e: f32 = std::f32::consts::LOG2_E;
    #[classattr]
    #[allow(non_upper_case_globals)]
    const log10_e: f32 = std::f32::consts::LOG10_E;
}

/// Rust-backed u8 scalar
#[pyclass(name = "u8", module = "pecos_rslib.dtypes", from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct ScalarU8 {
    value: u8,
}

#[pymethods]
impl ScalarU8 {
    /// Item size in bytes (class attribute)
    #[classattr]
    #[allow(non_upper_case_globals)] // Python API expects lowercase 'itemsize'
    const itemsize: usize = 1;

    #[new]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn new(value: i64) -> Self {
        Self { value: value as u8 }
    }

    fn __repr__(&self) -> String {
        format!("u8({})", self.value)
    }

    fn __str__(&self) -> String {
        self.value.to_string()
    }

    fn __int__(&self) -> u8 {
        self.value
    }

    fn __index__(&self) -> u8 {
        self.value
    }

    fn __bool__(&self) -> bool {
        self.value != 0
    }

    fn __format__(&self, format_spec: &str) -> String {
        // Handle various format specifications for u8
        if format_spec.is_empty() || format_spec == "d" {
            self.value.to_string()
        } else if format_spec == "b" {
            format!("{:b}", self.value)
        } else if format_spec == "x" {
            format!("{:x}", self.value)
        } else if format_spec == "X" {
            format!("{:X}", self.value)
        } else if format_spec == "o" {
            format!("{:o}", self.value)
        } else if format_spec.starts_with('0') && format_spec.len() > 1 {
            // Handle padding format like "08b" or "02x"
            let rest = &format_spec[1..];
            if let Some(format_type_pos) = rest.rfind(|c: char| !c.is_ascii_digit()) {
                let width_str = &rest[..format_type_pos];
                let format_type = &rest[format_type_pos..];
                if let Ok(width) = width_str.parse::<usize>() {
                    match format_type {
                        "b" => format!("{:0width$b}", self.value, width = width),
                        "x" => format!("{:0width$x}", self.value, width = width),
                        "X" => format!("{:0width$X}", self.value, width = width),
                        "o" => format!("{:0width$o}", self.value, width = width),
                        "d" => format!("{:0width$}", self.value, width = width),
                        _ => self.value.to_string(),
                    }
                } else {
                    self.value.to_string()
                }
            } else {
                self.value.to_string()
            }
        } else {
            // Fallback for unsupported format specs
            self.value.to_string()
        }
    }

    // Arithmetic operations with Python int
    fn __add__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarU8>() {
            other.extract::<ScalarU8>()?.value
        } else if let Ok(val) = other.extract::<u8>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for +: 'u8' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value.wrapping_add(other_value),
        })
    }

    fn __radd__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__add__(other)
    }

    fn __sub__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarU8>() {
            other.extract::<ScalarU8>()?.value
        } else if let Ok(val) = other.extract::<u8>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for -: 'u8' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value.wrapping_sub(other_value),
        })
    }

    fn __rsub__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(other_value) = other.extract::<u8>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for -: '{}' and 'u8'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: other_value.wrapping_sub(self.value),
        })
    }

    fn __mul__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarU8>() {
            other.extract::<ScalarU8>()?.value
        } else if let Ok(val) = other.extract::<u8>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for *: 'u8' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value.wrapping_mul(other_value),
        })
    }

    fn __rmul__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__mul__(other)
    }

    fn __floordiv__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarU8>() {
            other.extract::<ScalarU8>()?.value
        } else if let Ok(val) = other.extract::<u8>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for //: 'u8' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value / other_value,
        })
    }

    fn __rfloordiv__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(other_value) = other.extract::<u8>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for //: '{}' and 'u8'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: other_value / self.value,
        })
    }

    fn __mod__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarU8>() {
            other.extract::<ScalarU8>()?.value
        } else if let Ok(val) = other.extract::<u8>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for %: 'u8' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value % other_value,
        })
    }

    fn __rmod__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(other_value) = other.extract::<u8>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for %: '{}' and 'u8'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: other_value % self.value,
        })
    }

    // Bitwise operations
    fn __and__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarU8>() {
            other.extract::<ScalarU8>()?.value
        } else if let Ok(val) = other.extract::<u8>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for &: 'u8' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value & other_value,
        })
    }

    fn __rand__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__and__(other)
    }

    fn __or__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarU8>() {
            other.extract::<ScalarU8>()?.value
        } else if let Ok(val) = other.extract::<u8>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for |: 'u8' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value | other_value,
        })
    }

    fn __ror__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__or__(other)
    }

    fn __xor__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarU8>() {
            other.extract::<ScalarU8>()?.value
        } else if let Ok(val) = other.extract::<u8>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for ^: 'u8' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value ^ other_value,
        })
    }

    fn __rxor__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__xor__(other)
    }

    fn __invert__(&self) -> Self {
        Self { value: !self.value }
    }

    // Logical right shift for unsigned types
    fn __lshift__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let shift_amount = if other.is_instance_of::<ScalarU8>() {
            u32::from(other.extract::<ScalarU8>()?.value)
        } else if let Ok(val) = other.extract::<u32>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for <<: 'u8' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value.wrapping_shl(shift_amount),
        })
    }

    fn __rlshift__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(base_value) = other.extract::<u8>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for <<: '{}' and 'u8'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: base_value.wrapping_shl(u32::from(self.value)),
        })
    }

    fn __rshift__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let shift_amount = if other.is_instance_of::<ScalarU8>() {
            u32::from(other.extract::<ScalarU8>()?.value)
        } else if let Ok(val) = other.extract::<u32>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for >>: 'u8' and '{}'",
                other.get_type().name()?
            )));
        };
        // Logical right shift (no sign extension for unsigned)
        Ok(Self {
            value: self.value.wrapping_shr(shift_amount),
        })
    }

    fn __rrshift__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(base_value) = other.extract::<u8>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for >>: '{}' and 'u8'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: base_value.wrapping_shr(u32::from(self.value)),
        })
    }

    /// Rich comparison support for ==, !=, <, <=, >, >=
    fn __richcmp__(&self, other: &Bound<PyAny>, op: CompareOp) -> PyResult<bool> {
        let other_value = if other.is_instance_of::<ScalarU8>() {
            other.extract::<ScalarU8>()?.value
        } else if let Ok(val) = other.extract::<u8>() {
            val
        } else {
            return Ok(false);
        };

        match op {
            CompareOp::Lt => Ok(self.value < other_value),
            CompareOp::Le => Ok(self.value <= other_value),
            CompareOp::Eq => Ok(self.value == other_value),
            CompareOp::Ne => Ok(self.value != other_value),
            CompareOp::Gt => Ok(self.value > other_value),
            CompareOp::Ge => Ok(self.value >= other_value),
        }
    }

    /// Convert to `NumPy` scalar
    fn as_np<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let np = py.import("numpy")?;
        np.getattr("uint8")?.call1((self.value,))
    }

    /// Get the dtype
    #[getter]
    fn dtype(&self) -> DType {
        DType::U8
    }
}

/// Rust-backed u16 scalar
#[pyclass(name = "u16", module = "pecos_rslib.dtypes", from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct ScalarU16 {
    value: u16,
}

#[pymethods]
impl ScalarU16 {
    /// Item size in bytes (class attribute)
    #[classattr]
    #[allow(non_upper_case_globals)] // Python API expects lowercase 'itemsize'
    const itemsize: usize = 2;

    #[new]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn new(value: i64) -> Self {
        Self {
            value: value as u16,
        }
    }

    fn __repr__(&self) -> String {
        format!("u16({})", self.value)
    }

    fn __str__(&self) -> String {
        self.value.to_string()
    }

    fn __int__(&self) -> u16 {
        self.value
    }

    fn __index__(&self) -> u16 {
        self.value
    }

    fn __bool__(&self) -> bool {
        self.value != 0
    }

    fn __format__(&self, format_spec: &str) -> String {
        // Handle various format specifications for u16
        if format_spec.is_empty() || format_spec == "d" {
            self.value.to_string()
        } else if format_spec == "b" {
            format!("{:b}", self.value)
        } else if format_spec == "x" {
            format!("{:x}", self.value)
        } else if format_spec == "X" {
            format!("{:X}", self.value)
        } else if format_spec == "o" {
            format!("{:o}", self.value)
        } else if format_spec.starts_with('0') && format_spec.len() > 1 {
            // Handle padding format like "08b" or "04x"
            let rest = &format_spec[1..];
            if let Some(format_type_pos) = rest.rfind(|c: char| !c.is_ascii_digit()) {
                let width_str = &rest[..format_type_pos];
                let format_type = &rest[format_type_pos..];
                if let Ok(width) = width_str.parse::<usize>() {
                    match format_type {
                        "b" => format!("{:0width$b}", self.value, width = width),
                        "x" => format!("{:0width$x}", self.value, width = width),
                        "X" => format!("{:0width$X}", self.value, width = width),
                        "o" => format!("{:0width$o}", self.value, width = width),
                        "d" => format!("{:0width$}", self.value, width = width),
                        _ => self.value.to_string(),
                    }
                } else {
                    self.value.to_string()
                }
            } else {
                self.value.to_string()
            }
        } else {
            // Fallback for unsupported format specs
            self.value.to_string()
        }
    }

    // Arithmetic operations with Python int
    fn __add__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarU16>() {
            other.extract::<ScalarU16>()?.value
        } else if let Ok(val) = other.extract::<u16>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for +: 'u16' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value.wrapping_add(other_value),
        })
    }

    fn __radd__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__add__(other)
    }

    fn __sub__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarU16>() {
            other.extract::<ScalarU16>()?.value
        } else if let Ok(val) = other.extract::<u16>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for -: 'u16' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value.wrapping_sub(other_value),
        })
    }

    fn __rsub__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(other_value) = other.extract::<u16>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for -: '{}' and 'u16'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: other_value.wrapping_sub(self.value),
        })
    }

    fn __mul__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarU16>() {
            other.extract::<ScalarU16>()?.value
        } else if let Ok(val) = other.extract::<u16>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for *: 'u16' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value.wrapping_mul(other_value),
        })
    }

    fn __rmul__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__mul__(other)
    }

    fn __floordiv__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarU16>() {
            other.extract::<ScalarU16>()?.value
        } else if let Ok(val) = other.extract::<u16>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for //: 'u16' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value / other_value,
        })
    }

    fn __rfloordiv__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(other_value) = other.extract::<u16>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for //: '{}' and 'u16'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: other_value / self.value,
        })
    }

    fn __mod__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarU16>() {
            other.extract::<ScalarU16>()?.value
        } else if let Ok(val) = other.extract::<u16>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for %: 'u16' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value % other_value,
        })
    }

    fn __rmod__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(other_value) = other.extract::<u16>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for %: '{}' and 'u16'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: other_value % self.value,
        })
    }

    // Bitwise operations
    fn __and__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarU16>() {
            other.extract::<ScalarU16>()?.value
        } else if let Ok(val) = other.extract::<u16>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for &: 'u16' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value & other_value,
        })
    }

    fn __rand__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__and__(other)
    }

    fn __or__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarU16>() {
            other.extract::<ScalarU16>()?.value
        } else if let Ok(val) = other.extract::<u16>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for |: 'u16' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value | other_value,
        })
    }

    fn __ror__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__or__(other)
    }

    fn __xor__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarU16>() {
            other.extract::<ScalarU16>()?.value
        } else if let Ok(val) = other.extract::<u16>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for ^: 'u16' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value ^ other_value,
        })
    }

    fn __rxor__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__xor__(other)
    }

    fn __invert__(&self) -> Self {
        Self { value: !self.value }
    }

    // Logical right shift for unsigned types
    fn __lshift__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let shift_amount = if other.is_instance_of::<ScalarU16>() {
            u32::from(other.extract::<ScalarU16>()?.value)
        } else if let Ok(val) = other.extract::<u32>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for <<: 'u16' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value.wrapping_shl(shift_amount),
        })
    }

    fn __rlshift__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(base_value) = other.extract::<u16>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for <<: '{}' and 'u16'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: base_value.wrapping_shl(u32::from(self.value)),
        })
    }

    fn __rshift__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let shift_amount = if other.is_instance_of::<ScalarU16>() {
            u32::from(other.extract::<ScalarU16>()?.value)
        } else if let Ok(val) = other.extract::<u32>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for >>: 'u16' and '{}'",
                other.get_type().name()?
            )));
        };
        // Logical right shift (no sign extension for unsigned)
        Ok(Self {
            value: self.value.wrapping_shr(shift_amount),
        })
    }

    fn __rrshift__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(base_value) = other.extract::<u16>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for >>: '{}' and 'u16'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: base_value.wrapping_shr(u32::from(self.value)),
        })
    }

    /// Rich comparison support for ==, !=, <, <=, >, >=
    fn __richcmp__(&self, other: &Bound<PyAny>, op: CompareOp) -> PyResult<bool> {
        let other_value = if other.is_instance_of::<ScalarU16>() {
            other.extract::<ScalarU16>()?.value
        } else if let Ok(val) = other.extract::<u16>() {
            val
        } else {
            return Ok(false);
        };

        match op {
            CompareOp::Lt => Ok(self.value < other_value),
            CompareOp::Le => Ok(self.value <= other_value),
            CompareOp::Eq => Ok(self.value == other_value),
            CompareOp::Ne => Ok(self.value != other_value),
            CompareOp::Gt => Ok(self.value > other_value),
            CompareOp::Ge => Ok(self.value >= other_value),
        }
    }

    /// Convert to `NumPy` scalar
    fn as_np<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let np = py.import("numpy")?;
        np.getattr("uint16")?.call1((self.value,))
    }

    /// Get the dtype
    #[getter]
    fn dtype(&self) -> DType {
        DType::U16
    }
}

/// Rust-backed u32 scalar
#[pyclass(name = "u32", module = "pecos_rslib.dtypes", from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct ScalarU32 {
    value: u32,
}

#[pymethods]
impl ScalarU32 {
    /// Item size in bytes (class attribute)
    #[classattr]
    #[allow(non_upper_case_globals)] // Python API expects lowercase 'itemsize'
    const itemsize: usize = 4;

    #[new]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn new(value: i64) -> Self {
        Self {
            value: value as u32,
        }
    }

    fn __repr__(&self) -> String {
        format!("u32({})", self.value)
    }

    fn __str__(&self) -> String {
        self.value.to_string()
    }

    fn __int__(&self) -> u32 {
        self.value
    }

    fn __index__(&self) -> u32 {
        self.value
    }

    fn __bool__(&self) -> bool {
        self.value != 0
    }

    fn __format__(&self, format_spec: &str) -> String {
        // Handle various format specifications for u32
        if format_spec.is_empty() || format_spec == "d" {
            self.value.to_string()
        } else if format_spec == "b" {
            format!("{:b}", self.value)
        } else if format_spec == "x" {
            format!("{:x}", self.value)
        } else if format_spec == "X" {
            format!("{:X}", self.value)
        } else if format_spec == "o" {
            format!("{:o}", self.value)
        } else if format_spec.starts_with('0') && format_spec.len() > 1 {
            // Handle padding format like "08b" or "08x"
            let rest = &format_spec[1..];
            if let Some(format_type_pos) = rest.rfind(|c: char| !c.is_ascii_digit()) {
                let width_str = &rest[..format_type_pos];
                let format_type = &rest[format_type_pos..];
                if let Ok(width) = width_str.parse::<usize>() {
                    match format_type {
                        "b" => format!("{:0width$b}", self.value, width = width),
                        "x" => format!("{:0width$x}", self.value, width = width),
                        "X" => format!("{:0width$X}", self.value, width = width),
                        "o" => format!("{:0width$o}", self.value, width = width),
                        "d" => format!("{:0width$}", self.value, width = width),
                        _ => self.value.to_string(),
                    }
                } else {
                    self.value.to_string()
                }
            } else {
                self.value.to_string()
            }
        } else {
            // Fallback for unsupported format specs
            self.value.to_string()
        }
    }

    // Arithmetic operations with Python int
    fn __add__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarU32>() {
            other.extract::<ScalarU32>()?.value
        } else if let Ok(val) = other.extract::<u32>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for +: 'u32' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value.wrapping_add(other_value),
        })
    }

    fn __radd__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__add__(other)
    }

    fn __sub__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarU32>() {
            other.extract::<ScalarU32>()?.value
        } else if let Ok(val) = other.extract::<u32>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for -: 'u32' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value.wrapping_sub(other_value),
        })
    }

    fn __rsub__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(other_value) = other.extract::<u32>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for -: '{}' and 'u32'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: other_value.wrapping_sub(self.value),
        })
    }

    fn __mul__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarU32>() {
            other.extract::<ScalarU32>()?.value
        } else if let Ok(val) = other.extract::<u32>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for *: 'u32' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value.wrapping_mul(other_value),
        })
    }

    fn __rmul__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__mul__(other)
    }

    fn __floordiv__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarU32>() {
            other.extract::<ScalarU32>()?.value
        } else if let Ok(val) = other.extract::<u32>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for //: 'u32' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value / other_value,
        })
    }

    fn __rfloordiv__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(other_value) = other.extract::<u32>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for //: '{}' and 'u32'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: other_value / self.value,
        })
    }

    fn __mod__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarU32>() {
            other.extract::<ScalarU32>()?.value
        } else if let Ok(val) = other.extract::<u32>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for %: 'u32' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value % other_value,
        })
    }

    fn __rmod__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(other_value) = other.extract::<u32>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for %: '{}' and 'u32'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: other_value % self.value,
        })
    }

    // Bitwise operations
    fn __and__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarU32>() {
            other.extract::<ScalarU32>()?.value
        } else if let Ok(val) = other.extract::<u32>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for &: 'u32' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value & other_value,
        })
    }

    fn __rand__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__and__(other)
    }

    fn __or__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarU32>() {
            other.extract::<ScalarU32>()?.value
        } else if let Ok(val) = other.extract::<u32>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for |: 'u32' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value | other_value,
        })
    }

    fn __ror__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__or__(other)
    }

    fn __xor__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarU32>() {
            other.extract::<ScalarU32>()?.value
        } else if let Ok(val) = other.extract::<u32>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for ^: 'u32' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value ^ other_value,
        })
    }

    fn __rxor__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__xor__(other)
    }

    fn __invert__(&self) -> Self {
        Self { value: !self.value }
    }

    // Logical right shift for unsigned types
    fn __lshift__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let shift_amount = if other.is_instance_of::<ScalarU32>() {
            other.extract::<ScalarU32>()?.value
        } else if let Ok(val) = other.extract::<u32>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for <<: 'u32' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value.wrapping_shl(shift_amount),
        })
    }

    fn __rlshift__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(base_value) = other.extract::<u32>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for <<: '{}' and 'u32'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: base_value.wrapping_shl(self.value),
        })
    }

    fn __rshift__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let shift_amount = if other.is_instance_of::<ScalarU32>() {
            other.extract::<ScalarU32>()?.value
        } else if let Ok(val) = other.extract::<u32>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for >>: 'u32' and '{}'",
                other.get_type().name()?
            )));
        };
        // Logical right shift (no sign extension for unsigned)
        Ok(Self {
            value: self.value.wrapping_shr(shift_amount),
        })
    }

    fn __rrshift__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(base_value) = other.extract::<u32>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for >>: '{}' and 'u32'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: base_value.wrapping_shr(self.value),
        })
    }

    /// Rich comparison support for ==, !=, <, <=, >, >=
    fn __richcmp__(&self, other: &Bound<PyAny>, op: CompareOp) -> PyResult<bool> {
        let other_value = if other.is_instance_of::<ScalarU32>() {
            other.extract::<ScalarU32>()?.value
        } else if let Ok(val) = other.extract::<u32>() {
            val
        } else {
            return Ok(false);
        };

        match op {
            CompareOp::Lt => Ok(self.value < other_value),
            CompareOp::Le => Ok(self.value <= other_value),
            CompareOp::Eq => Ok(self.value == other_value),
            CompareOp::Ne => Ok(self.value != other_value),
            CompareOp::Gt => Ok(self.value > other_value),
            CompareOp::Ge => Ok(self.value >= other_value),
        }
    }

    /// Convert to `NumPy` scalar
    fn as_np<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let np = py.import("numpy")?;
        np.getattr("uint32")?.call1((self.value,))
    }

    /// Get the dtype
    #[getter]
    fn dtype(&self) -> DType {
        DType::U32
    }
}

/// Rust-backed u64 scalar
#[pyclass(name = "u64", module = "pecos_rslib.dtypes", from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct ScalarU64 {
    value: u64,
}

impl ScalarU64 {
    /// Rust-only constructor (for internal use).
    pub fn new(value: u64) -> Self {
        Self { value }
    }
}

#[pymethods]
impl ScalarU64 {
    /// Item size in bytes (class attribute)
    #[classattr]
    #[allow(non_upper_case_globals)] // Python API expects lowercase 'itemsize'
    const itemsize: usize = 8;

    #[new]
    #[allow(clippy::cast_sign_loss)]
    fn py_new(value: &Bound<'_, pyo3::PyAny>) -> PyResult<Self> {
        // Try u64 first (for values > i64::MAX), then i64 (for negative values that wrap)
        if let Ok(v) = value.extract::<u64>() {
            Ok(Self { value: v })
        } else if let Ok(v) = value.extract::<i64>() {
            Ok(Self { value: v as u64 })
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "u64() argument must be an integer, not '{}'",
                value.get_type().name()?
            )))
        }
    }

    fn __repr__(&self) -> String {
        format!("u64({})", self.value)
    }

    fn __str__(&self) -> String {
        self.value.to_string()
    }

    fn __int__<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        // Properly convert u64 to unsigned Python int
        self.value
            .into_pyobject(py)
            .expect("u64 to Python conversion failed")
            .into_any()
    }

    fn __index__<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        // Properly convert u64 to unsigned Python int for indexing
        self.value
            .into_pyobject(py)
            .expect("u64 to Python conversion failed")
            .into_any()
    }

    fn __bool__(&self) -> bool {
        self.value != 0
    }

    fn __format__(&self, format_spec: &str) -> String {
        // Handle various format specifications for u8
        if format_spec.is_empty() || format_spec == "d" {
            self.value.to_string()
        } else if format_spec == "b" {
            format!("{:b}", self.value)
        } else if format_spec == "x" {
            format!("{:x}", self.value)
        } else if format_spec == "X" {
            format!("{:X}", self.value)
        } else if format_spec == "o" {
            format!("{:o}", self.value)
        } else if format_spec.starts_with('0') && format_spec.len() > 1 {
            // Handle padding format like "08b" or "02x"
            let rest = &format_spec[1..];
            if let Some(format_type_pos) = rest.rfind(|c: char| !c.is_ascii_digit()) {
                let width_str = &rest[..format_type_pos];
                let format_type = &rest[format_type_pos..];
                if let Ok(width) = width_str.parse::<usize>() {
                    match format_type {
                        "b" => format!("{:0width$b}", self.value, width = width),
                        "x" => format!("{:0width$x}", self.value, width = width),
                        "X" => format!("{:0width$X}", self.value, width = width),
                        "o" => format!("{:0width$o}", self.value, width = width),
                        "d" => format!("{:0width$}", self.value, width = width),
                        _ => self.value.to_string(),
                    }
                } else {
                    self.value.to_string()
                }
            } else {
                self.value.to_string()
            }
        } else {
            // Fallback for unsupported format specs
            self.value.to_string()
        }
    }

    // Arithmetic operations with Python int
    fn __add__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarU64>() {
            other.extract::<ScalarU64>()?.value
        } else if let Ok(val) = other.extract::<u64>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for +: 'u8' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value.wrapping_add(other_value),
        })
    }

    fn __radd__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__add__(other)
    }

    fn __sub__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarU64>() {
            other.extract::<ScalarU64>()?.value
        } else if let Ok(val) = other.extract::<u64>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for -: 'u8' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value.wrapping_sub(other_value),
        })
    }

    fn __rsub__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(other_value) = other.extract::<u64>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for -: '{}' and 'u8'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: other_value.wrapping_sub(self.value),
        })
    }

    fn __mul__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarU64>() {
            other.extract::<ScalarU64>()?.value
        } else if let Ok(val) = other.extract::<u64>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for *: 'u8' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value.wrapping_mul(other_value),
        })
    }

    fn __rmul__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__mul__(other)
    }

    fn __floordiv__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarU64>() {
            other.extract::<ScalarU64>()?.value
        } else if let Ok(val) = other.extract::<u64>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for //: 'u8' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value / other_value,
        })
    }

    fn __rfloordiv__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(other_value) = other.extract::<u64>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for //: '{}' and 'u8'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: other_value / self.value,
        })
    }

    fn __mod__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarU64>() {
            other.extract::<ScalarU64>()?.value
        } else if let Ok(val) = other.extract::<u64>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for %: 'u8' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value % other_value,
        })
    }

    fn __rmod__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(other_value) = other.extract::<u64>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for %: '{}' and 'u8'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: other_value % self.value,
        })
    }

    // Bitwise operations
    fn __and__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarU64>() {
            other.extract::<ScalarU64>()?.value
        } else if let Ok(val) = other.extract::<u64>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for &: 'u8' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value & other_value,
        })
    }

    fn __rand__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__and__(other)
    }

    fn __or__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarU64>() {
            other.extract::<ScalarU64>()?.value
        } else if let Ok(val) = other.extract::<u64>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for |: 'u8' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value | other_value,
        })
    }

    fn __ror__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__or__(other)
    }

    fn __xor__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarU64>() {
            other.extract::<ScalarU64>()?.value
        } else if let Ok(val) = other.extract::<u64>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for ^: 'u8' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value ^ other_value,
        })
    }

    fn __rxor__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__xor__(other)
    }

    fn __invert__(&self) -> Self {
        Self { value: !self.value }
    }

    // Logical right shift for unsigned types
    #[allow(clippy::cast_possible_truncation)] // Shift amounts >u32::MAX would be invalid anyway
    fn __lshift__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let shift_amount = if other.is_instance_of::<ScalarU64>() {
            other.extract::<ScalarU64>()?.value as u32
        } else if let Ok(val) = other.extract::<u32>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for <<: 'u8' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value.wrapping_shl(shift_amount),
        })
    }

    #[allow(clippy::cast_possible_truncation)] // Shift amounts >u32::MAX would be invalid anyway
    fn __rlshift__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(base_value) = other.extract::<u64>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for <<: '{}' and 'u8'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: base_value.wrapping_shl(self.value as u32),
        })
    }

    #[allow(clippy::cast_possible_truncation)] // Shift amounts >u32::MAX would be invalid anyway
    fn __rshift__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let shift_amount = if other.is_instance_of::<ScalarU64>() {
            other.extract::<ScalarU64>()?.value as u32
        } else if let Ok(val) = other.extract::<u32>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for >>: 'u8' and '{}'",
                other.get_type().name()?
            )));
        };
        // Logical right shift (no sign extension for unsigned)
        Ok(Self {
            value: self.value.wrapping_shr(shift_amount),
        })
    }

    #[allow(clippy::cast_possible_truncation)] // Shift amounts >u32::MAX would be invalid anyway
    fn __rrshift__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(base_value) = other.extract::<u64>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for >>: '{}' and 'u8'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: base_value.wrapping_shr(self.value as u32),
        })
    }

    /// Rich comparison support for ==, !=, <, <=, >, >=
    fn __richcmp__(&self, other: &Bound<PyAny>, op: CompareOp) -> PyResult<bool> {
        let other_value = if other.is_instance_of::<ScalarU64>() {
            other.extract::<ScalarU64>()?.value
        } else if let Ok(val) = other.extract::<u64>() {
            val
        } else {
            return Ok(false);
        };

        match op {
            CompareOp::Lt => Ok(self.value < other_value),
            CompareOp::Le => Ok(self.value <= other_value),
            CompareOp::Eq => Ok(self.value == other_value),
            CompareOp::Ne => Ok(self.value != other_value),
            CompareOp::Gt => Ok(self.value > other_value),
            CompareOp::Ge => Ok(self.value >= other_value),
        }
    }

    /// Convert to `NumPy` scalar
    fn as_np<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let np = py.import("numpy")?;
        np.getattr("uint64")?.call1((self.value,))
    }

    /// Get the dtype
    #[getter]
    fn dtype(&self) -> DType {
        DType::U64
    }
}

/// Rust-backed i8 scalar
#[pyclass(name = "i8", module = "pecos_rslib.dtypes", from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct ScalarI8 {
    value: i8,
}

#[pymethods]
impl ScalarI8 {
    /// Item size in bytes (class attribute)
    #[classattr]
    #[allow(non_upper_case_globals)] // Python API expects lowercase 'itemsize'
    const itemsize: usize = 1;

    #[new]
    #[allow(clippy::cast_possible_truncation)]
    fn new(value: i64) -> Self {
        Self { value: value as i8 }
    }

    fn __repr__(&self) -> String {
        format!("i8({})", self.value)
    }

    fn __str__(&self) -> String {
        self.value.to_string()
    }

    fn __int__(&self) -> i8 {
        self.value
    }

    fn __index__(&self) -> i8 {
        self.value
    }

    fn __bool__(&self) -> bool {
        self.value != 0
    }

    fn __format__(&self, format_spec: &str) -> String {
        // Handle various format specifications for i8
        // For negative values, Python includes a minus sign prefix for b/x/X/o formats
        if format_spec.is_empty() || format_spec == "d" {
            self.value.to_string()
        } else if format_spec == "b" {
            if self.value < 0 {
                format!("-{:b}", self.value.wrapping_neg())
            } else {
                format!("{:b}", self.value)
            }
        } else if format_spec == "x" {
            if self.value < 0 {
                format!("-{:x}", self.value.wrapping_neg())
            } else {
                format!("{:x}", self.value)
            }
        } else if format_spec == "X" {
            if self.value < 0 {
                format!("-{:X}", self.value.wrapping_neg())
            } else {
                format!("{:X}", self.value)
            }
        } else if format_spec == "o" {
            if self.value < 0 {
                format!("-{:o}", self.value.wrapping_neg())
            } else {
                format!("{:o}", self.value)
            }
        } else if format_spec.starts_with('0') && format_spec.len() > 1 {
            // Handle padding format like "08b" or "02x"
            let rest = &format_spec[1..];
            if let Some(format_type_pos) = rest.rfind(|c: char| !c.is_ascii_digit()) {
                let width_str = &rest[..format_type_pos];
                let format_type = &rest[format_type_pos..];
                if let Ok(width) = width_str.parse::<usize>() {
                    match format_type {
                        "b" => {
                            if self.value < 0 {
                                format!("-{:0width$b}", self.value.wrapping_neg(), width = width)
                            } else {
                                format!("{:0width$b}", self.value, width = width)
                            }
                        }
                        "x" => {
                            if self.value < 0 {
                                format!("-{:0width$x}", self.value.wrapping_neg(), width = width)
                            } else {
                                format!("{:0width$x}", self.value, width = width)
                            }
                        }
                        "X" => {
                            if self.value < 0 {
                                format!("-{:0width$X}", self.value.wrapping_neg(), width = width)
                            } else {
                                format!("{:0width$X}", self.value, width = width)
                            }
                        }
                        "o" => {
                            if self.value < 0 {
                                format!("-{:0width$o}", self.value.wrapping_neg(), width = width)
                            } else {
                                format!("{:0width$o}", self.value, width = width)
                            }
                        }
                        "d" => format!("{:0width$}", self.value, width = width),
                        _ => self.value.to_string(),
                    }
                } else {
                    self.value.to_string()
                }
            } else {
                self.value.to_string()
            }
        } else {
            // Fallback for unsupported format specs
            self.value.to_string()
        }
    }

    fn __add__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarI8>() {
            other.extract::<ScalarI8>()?.value
        } else if let Ok(val) = other.extract::<i8>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for +: 'i8' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value.wrapping_add(other_value),
        })
    }

    fn __radd__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__add__(other)
    }

    fn __sub__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarI8>() {
            other.extract::<ScalarI8>()?.value
        } else if let Ok(val) = other.extract::<i8>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for -: 'i8' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value.wrapping_sub(other_value),
        })
    }

    fn __rsub__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(other_value) = other.extract::<i8>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for -: '{}' and 'i8'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: other_value.wrapping_sub(self.value),
        })
    }

    fn __mul__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarI8>() {
            other.extract::<ScalarI8>()?.value
        } else if let Ok(val) = other.extract::<i8>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for *: 'i8' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value.wrapping_mul(other_value),
        })
    }

    fn __rmul__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__mul__(other)
    }

    fn __floordiv__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarI8>() {
            other.extract::<ScalarI8>()?.value
        } else if let Ok(val) = other.extract::<i8>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for //: 'i8' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value / other_value,
        })
    }

    fn __rfloordiv__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(other_value) = other.extract::<i8>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for //: '{}' and 'i8'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: other_value / self.value,
        })
    }

    fn __mod__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarI8>() {
            other.extract::<ScalarI8>()?.value
        } else if let Ok(val) = other.extract::<i8>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for %: 'i8' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value % other_value,
        })
    }

    fn __rmod__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(other_value) = other.extract::<i8>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for %: '{}' and 'i8'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: other_value % self.value,
        })
    }

    fn __and__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarI8>() {
            other.extract::<ScalarI8>()?.value
        } else if let Ok(val) = other.extract::<i8>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for &: 'i8' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value & other_value,
        })
    }

    fn __rand__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__and__(other)
    }

    fn __or__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarI8>() {
            other.extract::<ScalarI8>()?.value
        } else if let Ok(val) = other.extract::<i8>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for |: 'i8' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value | other_value,
        })
    }

    fn __ror__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__or__(other)
    }

    fn __xor__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarI8>() {
            other.extract::<ScalarI8>()?.value
        } else if let Ok(val) = other.extract::<i8>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for ^: 'i8' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value ^ other_value,
        })
    }

    fn __rxor__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__xor__(other)
    }

    fn __invert__(&self) -> Self {
        Self { value: !self.value }
    }

    #[allow(clippy::cast_sign_loss)] // Rust shift ops require u32; negative shifts would be invalid
    fn __lshift__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let shift_amount = if other.is_instance_of::<ScalarI8>() {
            other.extract::<ScalarI8>()?.value as u32
        } else if let Ok(val) = other.extract::<u32>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for <<: 'i8' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value.wrapping_shl(shift_amount),
        })
    }

    #[allow(clippy::cast_sign_loss)] // Rust shift ops require u32; negative shifts would be invalid
    fn __rlshift__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(base_value) = other.extract::<i8>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for <<: '{}' and 'i8'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: base_value.wrapping_shl(self.value as u32),
        })
    }

    #[allow(clippy::cast_sign_loss)] // Rust shift ops require u32; negative shifts would be invalid
    fn __rshift__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let shift_amount = if other.is_instance_of::<ScalarI8>() {
            other.extract::<ScalarI8>()?.value as u32
        } else if let Ok(val) = other.extract::<u32>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for >>: 'i8' and '{}'",
                other.get_type().name()?
            )));
        };
        // Arithmetic right shift (sign extension for signed)
        Ok(Self {
            value: self.value.wrapping_shr(shift_amount),
        })
    }

    #[allow(clippy::cast_sign_loss)] // Rust shift ops require u32; negative shifts would be invalid
    fn __rrshift__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(base_value) = other.extract::<i8>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for >>: '{}' and 'i8'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: base_value.wrapping_shr(self.value as u32),
        })
    }

    fn __richcmp__(&self, other: &Bound<PyAny>, op: CompareOp) -> PyResult<bool> {
        let other_value = if other.is_instance_of::<ScalarI8>() {
            other.extract::<ScalarI8>()?.value
        } else if let Ok(val) = other.extract::<i8>() {
            val
        } else {
            return Ok(false);
        };

        match op {
            CompareOp::Lt => Ok(self.value < other_value),
            CompareOp::Le => Ok(self.value <= other_value),
            CompareOp::Eq => Ok(self.value == other_value),
            CompareOp::Ne => Ok(self.value != other_value),
            CompareOp::Gt => Ok(self.value > other_value),
            CompareOp::Ge => Ok(self.value >= other_value),
        }
    }

    fn as_np<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let np = py.import("numpy")?;
        np.getattr("int8")?.call1((self.value,))
    }

    #[getter]
    fn dtype(&self) -> DType {
        DType::I8
    }
}

/// Rust-backed i16 scalar
#[pyclass(name = "i16", module = "pecos_rslib.dtypes", from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct ScalarI16 {
    value: i16,
}

#[pymethods]
impl ScalarI16 {
    #[classattr]
    #[allow(non_upper_case_globals)] // Python API expects lowercase 'itemsize'
    const itemsize: usize = 2;

    #[new]
    #[allow(clippy::cast_possible_truncation)]
    fn new(value: i64) -> Self {
        Self {
            value: value as i16,
        }
    }

    fn __repr__(&self) -> String {
        format!("i16({})", self.value)
    }

    fn __str__(&self) -> String {
        self.value.to_string()
    }

    fn __int__(&self) -> i16 {
        self.value
    }

    fn __index__(&self) -> i16 {
        self.value
    }

    fn __bool__(&self) -> bool {
        self.value != 0
    }

    fn __format__(&self, format_spec: &str) -> String {
        // Handle various format specifications for i16
        // For negative values, Python includes a minus sign prefix for b/x/X/o formats
        if format_spec.is_empty() || format_spec == "d" {
            self.value.to_string()
        } else if format_spec == "b" {
            if self.value < 0 {
                format!("-{:b}", self.value.wrapping_neg())
            } else {
                format!("{:b}", self.value)
            }
        } else if format_spec == "x" {
            if self.value < 0 {
                format!("-{:x}", self.value.wrapping_neg())
            } else {
                format!("{:x}", self.value)
            }
        } else if format_spec == "X" {
            if self.value < 0 {
                format!("-{:X}", self.value.wrapping_neg())
            } else {
                format!("{:X}", self.value)
            }
        } else if format_spec == "o" {
            if self.value < 0 {
                format!("-{:o}", self.value.wrapping_neg())
            } else {
                format!("{:o}", self.value)
            }
        } else if format_spec.starts_with('0') && format_spec.len() > 1 {
            // Handle padding format like "016b" or "04x"
            let rest = &format_spec[1..];
            if let Some(format_type_pos) = rest.rfind(|c: char| !c.is_ascii_digit()) {
                let width_str = &rest[..format_type_pos];
                let format_type = &rest[format_type_pos..];
                if let Ok(width) = width_str.parse::<usize>() {
                    match format_type {
                        "b" => {
                            if self.value < 0 {
                                format!("-{:0width$b}", self.value.wrapping_neg(), width = width)
                            } else {
                                format!("{:0width$b}", self.value, width = width)
                            }
                        }
                        "x" => {
                            if self.value < 0 {
                                format!("-{:0width$x}", self.value.wrapping_neg(), width = width)
                            } else {
                                format!("{:0width$x}", self.value, width = width)
                            }
                        }
                        "X" => {
                            if self.value < 0 {
                                format!("-{:0width$X}", self.value.wrapping_neg(), width = width)
                            } else {
                                format!("{:0width$X}", self.value, width = width)
                            }
                        }
                        "o" => {
                            if self.value < 0 {
                                format!("-{:0width$o}", self.value.wrapping_neg(), width = width)
                            } else {
                                format!("{:0width$o}", self.value, width = width)
                            }
                        }
                        "d" => format!("{:0width$}", self.value, width = width),
                        _ => self.value.to_string(),
                    }
                } else {
                    self.value.to_string()
                }
            } else {
                self.value.to_string()
            }
        } else {
            // Fallback for unsupported format specs
            self.value.to_string()
        }
    }

    fn __add__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarI16>() {
            other.extract::<ScalarI16>()?.value
        } else if let Ok(val) = other.extract::<i16>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for +: 'i16' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value.wrapping_add(other_value),
        })
    }

    fn __radd__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__add__(other)
    }

    fn __sub__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarI16>() {
            other.extract::<ScalarI16>()?.value
        } else if let Ok(val) = other.extract::<i16>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for -: 'i16' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value.wrapping_sub(other_value),
        })
    }

    fn __rsub__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(other_value) = other.extract::<i16>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for -: '{}' and 'i16'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: other_value.wrapping_sub(self.value),
        })
    }

    fn __mul__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarI16>() {
            other.extract::<ScalarI16>()?.value
        } else if let Ok(val) = other.extract::<i16>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for *: 'i16' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value.wrapping_mul(other_value),
        })
    }

    fn __rmul__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__mul__(other)
    }

    fn __floordiv__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarI16>() {
            other.extract::<ScalarI16>()?.value
        } else if let Ok(val) = other.extract::<i16>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for //: 'i16' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value / other_value,
        })
    }

    fn __rfloordiv__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(other_value) = other.extract::<i16>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for //: '{}' and 'i16'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: other_value / self.value,
        })
    }

    fn __mod__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarI16>() {
            other.extract::<ScalarI16>()?.value
        } else if let Ok(val) = other.extract::<i16>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for %: 'i16' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value % other_value,
        })
    }

    fn __rmod__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(other_value) = other.extract::<i16>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for %: '{}' and 'i16'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: other_value % self.value,
        })
    }

    fn __and__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarI16>() {
            other.extract::<ScalarI16>()?.value
        } else if let Ok(val) = other.extract::<i16>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for &: 'i16' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value & other_value,
        })
    }

    fn __rand__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__and__(other)
    }

    fn __or__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarI16>() {
            other.extract::<ScalarI16>()?.value
        } else if let Ok(val) = other.extract::<i16>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for |: 'i16' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value | other_value,
        })
    }

    fn __ror__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__or__(other)
    }

    fn __xor__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarI16>() {
            other.extract::<ScalarI16>()?.value
        } else if let Ok(val) = other.extract::<i16>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for ^: 'i16' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value ^ other_value,
        })
    }

    fn __rxor__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__xor__(other)
    }

    fn __invert__(&self) -> Self {
        Self { value: !self.value }
    }

    #[allow(clippy::cast_sign_loss)] // Rust shift ops require u32; negative shifts would be invalid
    fn __lshift__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let shift_amount = if other.is_instance_of::<ScalarI16>() {
            other.extract::<ScalarI16>()?.value as u32
        } else if let Ok(val) = other.extract::<u32>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for <<: 'i16' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value.wrapping_shl(shift_amount),
        })
    }

    #[allow(clippy::cast_sign_loss)] // Rust shift ops require u32; negative shifts would be invalid
    fn __rlshift__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(base_value) = other.extract::<i16>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for <<: '{}' and 'i16'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: base_value.wrapping_shl(self.value as u32),
        })
    }

    #[allow(clippy::cast_sign_loss)] // Rust shift ops require u32; negative shifts would be invalid
    fn __rshift__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let shift_amount = if other.is_instance_of::<ScalarI16>() {
            other.extract::<ScalarI16>()?.value as u32
        } else if let Ok(val) = other.extract::<u32>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for >>: 'i16' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value.wrapping_shr(shift_amount),
        })
    }

    #[allow(clippy::cast_sign_loss)] // Rust shift ops require u32; negative shifts would be invalid
    fn __rrshift__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(base_value) = other.extract::<i16>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for >>: '{}' and 'i16'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: base_value.wrapping_shr(self.value as u32),
        })
    }

    fn __richcmp__(&self, other: &Bound<PyAny>, op: CompareOp) -> PyResult<bool> {
        let other_value = if other.is_instance_of::<ScalarI16>() {
            other.extract::<ScalarI16>()?.value
        } else if let Ok(val) = other.extract::<i16>() {
            val
        } else {
            return Ok(false);
        };

        match op {
            CompareOp::Lt => Ok(self.value < other_value),
            CompareOp::Le => Ok(self.value <= other_value),
            CompareOp::Eq => Ok(self.value == other_value),
            CompareOp::Ne => Ok(self.value != other_value),
            CompareOp::Gt => Ok(self.value > other_value),
            CompareOp::Ge => Ok(self.value >= other_value),
        }
    }

    fn as_np<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let np = py.import("numpy")?;
        np.getattr("int16")?.call1((self.value,))
    }

    #[getter]
    fn dtype(&self) -> DType {
        DType::I16
    }
}

/// Rust-backed i32 scalar
#[pyclass(name = "i32", module = "pecos_rslib.dtypes", from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct ScalarI32 {
    value: i32,
}

#[pymethods]
impl ScalarI32 {
    #[classattr]
    #[allow(non_upper_case_globals)] // Python API expects lowercase 'itemsize'
    const itemsize: usize = 4;

    #[new]
    #[allow(clippy::cast_possible_truncation)]
    fn new(value: i64) -> Self {
        Self {
            value: value as i32,
        }
    }

    fn __repr__(&self) -> String {
        format!("i32({})", self.value)
    }

    fn __str__(&self) -> String {
        self.value.to_string()
    }

    fn __int__(&self) -> i32 {
        self.value
    }

    fn __index__(&self) -> i32 {
        self.value
    }

    fn __bool__(&self) -> bool {
        self.value != 0
    }

    fn __format__(&self, format_spec: &str) -> String {
        // Handle various format specifications for i32
        // For negative values, Python includes a minus sign prefix for b/x/X/o formats
        if format_spec.is_empty() || format_spec == "d" {
            self.value.to_string()
        } else if format_spec == "b" {
            if self.value < 0 {
                format!("-{:b}", self.value.wrapping_neg())
            } else {
                format!("{:b}", self.value)
            }
        } else if format_spec == "x" {
            if self.value < 0 {
                format!("-{:x}", self.value.wrapping_neg())
            } else {
                format!("{:x}", self.value)
            }
        } else if format_spec == "X" {
            if self.value < 0 {
                format!("-{:X}", self.value.wrapping_neg())
            } else {
                format!("{:X}", self.value)
            }
        } else if format_spec == "o" {
            if self.value < 0 {
                format!("-{:o}", self.value.wrapping_neg())
            } else {
                format!("{:o}", self.value)
            }
        } else if format_spec.starts_with('0') && format_spec.len() > 1 {
            // Handle padding format like "032b" or "08x"
            let rest = &format_spec[1..];
            if let Some(format_type_pos) = rest.rfind(|c: char| !c.is_ascii_digit()) {
                let width_str = &rest[..format_type_pos];
                let format_type = &rest[format_type_pos..];
                if let Ok(width) = width_str.parse::<usize>() {
                    match format_type {
                        "b" => {
                            if self.value < 0 {
                                format!("-{:0width$b}", self.value.wrapping_neg(), width = width)
                            } else {
                                format!("{:0width$b}", self.value, width = width)
                            }
                        }
                        "x" => {
                            if self.value < 0 {
                                format!("-{:0width$x}", self.value.wrapping_neg(), width = width)
                            } else {
                                format!("{:0width$x}", self.value, width = width)
                            }
                        }
                        "X" => {
                            if self.value < 0 {
                                format!("-{:0width$X}", self.value.wrapping_neg(), width = width)
                            } else {
                                format!("{:0width$X}", self.value, width = width)
                            }
                        }
                        "o" => {
                            if self.value < 0 {
                                format!("-{:0width$o}", self.value.wrapping_neg(), width = width)
                            } else {
                                format!("{:0width$o}", self.value, width = width)
                            }
                        }
                        "d" => format!("{:0width$}", self.value, width = width),
                        _ => self.value.to_string(),
                    }
                } else {
                    self.value.to_string()
                }
            } else {
                self.value.to_string()
            }
        } else {
            // Fallback for unsupported format specs
            self.value.to_string()
        }
    }

    fn __add__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarI32>() {
            other.extract::<ScalarI32>()?.value
        } else if let Ok(val) = other.extract::<i32>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for +: 'i32' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value.wrapping_add(other_value),
        })
    }

    fn __radd__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__add__(other)
    }

    fn __sub__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarI32>() {
            other.extract::<ScalarI32>()?.value
        } else if let Ok(val) = other.extract::<i32>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for -: 'i32' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value.wrapping_sub(other_value),
        })
    }

    fn __rsub__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(other_value) = other.extract::<i32>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for -: '{}' and 'i32'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: other_value.wrapping_sub(self.value),
        })
    }

    fn __mul__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarI32>() {
            other.extract::<ScalarI32>()?.value
        } else if let Ok(val) = other.extract::<i32>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for *: 'i32' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value.wrapping_mul(other_value),
        })
    }

    fn __rmul__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__mul__(other)
    }

    fn __floordiv__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarI32>() {
            other.extract::<ScalarI32>()?.value
        } else if let Ok(val) = other.extract::<i32>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for //: 'i32' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value / other_value,
        })
    }

    fn __rfloordiv__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(other_value) = other.extract::<i32>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for //: '{}' and 'i32'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: other_value / self.value,
        })
    }

    fn __mod__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarI32>() {
            other.extract::<ScalarI32>()?.value
        } else if let Ok(val) = other.extract::<i32>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for %: 'i32' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value % other_value,
        })
    }

    fn __rmod__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(other_value) = other.extract::<i32>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for %: '{}' and 'i32'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: other_value % self.value,
        })
    }

    fn __and__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarI32>() {
            other.extract::<ScalarI32>()?.value
        } else if let Ok(val) = other.extract::<i32>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for &: 'i32' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value & other_value,
        })
    }

    fn __rand__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__and__(other)
    }

    fn __or__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarI32>() {
            other.extract::<ScalarI32>()?.value
        } else if let Ok(val) = other.extract::<i32>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for |: 'i32' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value | other_value,
        })
    }

    fn __ror__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__or__(other)
    }

    fn __xor__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarI32>() {
            other.extract::<ScalarI32>()?.value
        } else if let Ok(val) = other.extract::<i32>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for ^: 'i32' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value ^ other_value,
        })
    }

    fn __rxor__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__xor__(other)
    }

    fn __invert__(&self) -> Self {
        Self { value: !self.value }
    }

    #[allow(clippy::cast_sign_loss)] // Rust shift ops require u32; negative shifts would be invalid
    fn __lshift__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let shift_amount = if other.is_instance_of::<ScalarI32>() {
            other.extract::<ScalarI32>()?.value as u32
        } else if let Ok(val) = other.extract::<u32>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for <<: 'i32' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value.wrapping_shl(shift_amount),
        })
    }

    #[allow(clippy::cast_sign_loss)] // Rust shift ops require u32; negative shifts would be invalid
    fn __rlshift__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(base_value) = other.extract::<i32>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for <<: '{}' and 'i32'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: base_value.wrapping_shl(self.value as u32),
        })
    }

    #[allow(clippy::cast_sign_loss)] // Rust shift ops require u32; negative shifts would be invalid
    fn __rshift__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let shift_amount = if other.is_instance_of::<ScalarI32>() {
            other.extract::<ScalarI32>()?.value as u32
        } else if let Ok(val) = other.extract::<u32>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for >>: 'i32' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value.wrapping_shr(shift_amount),
        })
    }

    #[allow(clippy::cast_sign_loss)] // Rust shift ops require u32; negative shifts would be invalid
    fn __rrshift__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(base_value) = other.extract::<i32>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for >>: '{}' and 'i32'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: base_value.wrapping_shr(self.value as u32),
        })
    }

    fn __richcmp__(&self, other: &Bound<PyAny>, op: CompareOp) -> PyResult<bool> {
        let other_value = if other.is_instance_of::<ScalarI32>() {
            other.extract::<ScalarI32>()?.value
        } else if let Ok(val) = other.extract::<i32>() {
            val
        } else {
            return Ok(false);
        };

        match op {
            CompareOp::Lt => Ok(self.value < other_value),
            CompareOp::Le => Ok(self.value <= other_value),
            CompareOp::Eq => Ok(self.value == other_value),
            CompareOp::Ne => Ok(self.value != other_value),
            CompareOp::Gt => Ok(self.value > other_value),
            CompareOp::Ge => Ok(self.value >= other_value),
        }
    }

    fn as_np<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let np = py.import("numpy")?;
        np.getattr("int32")?.call1((self.value,))
    }

    #[getter]
    fn dtype(&self) -> DType {
        DType::I32
    }
}

/// Rust-backed i64 scalar
#[pyclass(name = "i64", module = "pecos_rslib.dtypes", from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct ScalarI64 {
    value: i64,
}

impl ScalarI64 {
    /// Rust-only constructor (for internal use).
    pub fn new(value: i64) -> Self {
        Self { value }
    }
}

#[pymethods]
impl ScalarI64 {
    /// Item size in bytes (class attribute)
    #[classattr]
    #[allow(non_upper_case_globals)] // Python API expects lowercase 'itemsize'
    const itemsize: usize = 8;

    #[new]
    #[allow(clippy::cast_possible_wrap)]
    fn py_new(value: &Bound<'_, pyo3::PyAny>) -> PyResult<Self> {
        // Try i64 first (normal case), then u64 (for values > i64::MAX that truncate)
        if let Ok(v) = value.extract::<i64>() {
            Ok(Self { value: v })
        } else if let Ok(v) = value.extract::<u64>() {
            Ok(Self { value: v as i64 })
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "i64() argument must be an integer",
            ))
        }
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

    fn __index__(&self) -> i64 {
        self.value
    }

    fn __bool__(&self) -> bool {
        self.value != 0
    }

    fn __format__(&self, format_spec: &str) -> String {
        // Handle various format specifications for i64
        // For negative values, Python includes a minus sign prefix for b/x/X/o formats
        if format_spec.is_empty() || format_spec == "d" {
            self.value.to_string()
        } else if format_spec == "b" {
            if self.value < 0 {
                format!("-{:b}", self.value.wrapping_neg())
            } else {
                format!("{:b}", self.value)
            }
        } else if format_spec == "x" {
            if self.value < 0 {
                format!("-{:x}", self.value.wrapping_neg())
            } else {
                format!("{:x}", self.value)
            }
        } else if format_spec == "X" {
            if self.value < 0 {
                format!("-{:X}", self.value.wrapping_neg())
            } else {
                format!("{:X}", self.value)
            }
        } else if format_spec == "o" {
            if self.value < 0 {
                format!("-{:o}", self.value.wrapping_neg())
            } else {
                format!("{:o}", self.value)
            }
        } else if format_spec.starts_with('0') && format_spec.len() > 1 {
            // Handle padding format like "08b" or "016x"
            let rest = &format_spec[1..];
            if let Some(format_type_pos) = rest.rfind(|c: char| !c.is_ascii_digit()) {
                let width_str = &rest[..format_type_pos];
                let format_type = &rest[format_type_pos..];
                if let Ok(width) = width_str.parse::<usize>() {
                    match format_type {
                        "b" => {
                            if self.value < 0 {
                                format!("-{:0width$b}", self.value.wrapping_neg(), width = width)
                            } else {
                                format!("{:0width$b}", self.value, width = width)
                            }
                        }
                        "x" => {
                            if self.value < 0 {
                                format!("-{:0width$x}", self.value.wrapping_neg(), width = width)
                            } else {
                                format!("{:0width$x}", self.value, width = width)
                            }
                        }
                        "X" => {
                            if self.value < 0 {
                                format!("-{:0width$X}", self.value.wrapping_neg(), width = width)
                            } else {
                                format!("{:0width$X}", self.value, width = width)
                            }
                        }
                        "o" => {
                            if self.value < 0 {
                                format!("-{:0width$o}", self.value.wrapping_neg(), width = width)
                            } else {
                                format!("{:0width$o}", self.value, width = width)
                            }
                        }
                        "d" => format!("{:0width$}", self.value, width = width),
                        _ => self.value.to_string(),
                    }
                } else {
                    self.value.to_string()
                }
            } else {
                self.value.to_string()
            }
        } else {
            // Fallback for unsupported format specs
            self.value.to_string()
        }
    }

    // Bitwise operations
    fn __and__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarI64>() {
            other.extract::<ScalarI64>()?.value
        } else if let Ok(val) = other.extract::<i64>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for &: 'i64' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value & other_value,
        })
    }

    fn __rand__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__and__(other)
    }

    fn __or__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarI64>() {
            other.extract::<ScalarI64>()?.value
        } else if let Ok(val) = other.extract::<i64>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for |: 'i64' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value | other_value,
        })
    }

    fn __ror__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__or__(other)
    }

    fn __xor__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarI64>() {
            other.extract::<ScalarI64>()?.value
        } else if let Ok(val) = other.extract::<i64>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for ^: 'i64' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value ^ other_value,
        })
    }

    fn __rxor__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__xor__(other)
    }

    fn __invert__(&self) -> Self {
        Self { value: !self.value }
    }

    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)] // Rust shift ops require u32; shifts > 63 are masked by wrapping_shl anyway
    fn __lshift__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let shift_amount = if other.is_instance_of::<ScalarI64>() {
            other.extract::<ScalarI64>()?.value as u32
        } else if let Ok(val) = other.extract::<u32>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for <<: 'i64' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value.wrapping_shl(shift_amount),
        })
    }

    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)] // Rust shift ops require u32; shifts > 63 are masked by wrapping_shl anyway
    fn __rlshift__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(base_value) = other.extract::<i64>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for <<: '{}' and 'i64'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: base_value.wrapping_shl(self.value as u32),
        })
    }

    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)] // Rust shift ops require u32; shifts > 63 are masked by wrapping_shr anyway
    fn __rshift__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let shift_amount = if other.is_instance_of::<ScalarI64>() {
            other.extract::<ScalarI64>()?.value as u32
        } else if let Ok(val) = other.extract::<u32>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for >>: 'i64' and '{}'",
                other.get_type().name()?
            )));
        };
        // Arithmetic right shift (sign extension for signed)
        Ok(Self {
            value: self.value.wrapping_shr(shift_amount),
        })
    }

    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)] // Rust shift ops require u32; shifts > 63 are masked by wrapping_shr anyway
    fn __rrshift__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(base_value) = other.extract::<i64>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for >>: '{}' and 'i64'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: base_value.wrapping_shr(self.value as u32),
        })
    }

    // Arithmetic operations with Python int/float
    fn __add__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarI64>() {
            other.extract::<ScalarI64>()?.value
        } else if let Ok(val) = other.extract::<i64>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for +: 'i64' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value.wrapping_add(other_value),
        })
    }

    fn __radd__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__add__(other)
    }

    fn __sub__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarI64>() {
            other.extract::<ScalarI64>()?.value
        } else if let Ok(val) = other.extract::<i64>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for -: 'i64' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value.wrapping_sub(other_value),
        })
    }

    fn __rsub__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(other_value) = other.extract::<i64>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for -: '{}' and 'i64'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: other_value.wrapping_sub(self.value),
        })
    }

    fn __mul__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarI64>() {
            other.extract::<ScalarI64>()?.value
        } else if let Ok(val) = other.extract::<i64>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for *: 'i64' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value.wrapping_mul(other_value),
        })
    }

    fn __rmul__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        self.__mul__(other)
    }

    fn __floordiv__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarI64>() {
            other.extract::<ScalarI64>()?.value
        } else if let Ok(val) = other.extract::<i64>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for //: 'i64' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value / other_value,
        })
    }

    fn __rfloordiv__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(other_value) = other.extract::<i64>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for //: '{}' and 'i64'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: other_value / self.value,
        })
    }

    fn __mod__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let other_value = if other.is_instance_of::<ScalarI64>() {
            other.extract::<ScalarI64>()?.value
        } else if let Ok(val) = other.extract::<i64>() {
            val
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for %: 'i64' and '{}'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: self.value % other_value,
        })
    }

    fn __rmod__(&self, other: &Bound<PyAny>) -> PyResult<Self> {
        let Ok(other_value) = other.extract::<i64>() else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "unsupported operand type(s) for %: '{}' and 'i64'",
                other.get_type().name()?
            )));
        };
        Ok(Self {
            value: other_value % self.value,
        })
    }

    /// Rich comparison support for ==, !=, <, <=, >, >=
    fn __richcmp__(&self, other: &Bound<PyAny>, op: CompareOp) -> PyResult<bool> {
        let other_value = if other.is_instance_of::<ScalarI64>() {
            other.extract::<ScalarI64>()?.value
        } else if let Ok(val) = other.extract::<i64>() {
            val
        } else if let Ok(val) = other.extract::<f64>() {
            // For float comparison, convert to float
            // Allow exact comparison - matches Python's comparison semantics
            // Allow precision loss - this is expected when comparing i64 to f64
            #[allow(clippy::float_cmp, clippy::cast_precision_loss)]
            return match op {
                CompareOp::Lt => Ok((self.value as f64) < val),
                CompareOp::Le => Ok((self.value as f64) <= val),
                CompareOp::Eq => Ok((self.value as f64) == val),
                CompareOp::Ne => Ok((self.value as f64) != val),
                CompareOp::Gt => Ok((self.value as f64) > val),
                CompareOp::Ge => Ok((self.value as f64) >= val),
            };
        } else {
            return Ok(false);
        };

        match op {
            CompareOp::Lt => Ok(self.value < other_value),
            CompareOp::Le => Ok(self.value <= other_value),
            CompareOp::Eq => Ok(self.value == other_value),
            CompareOp::Ne => Ok(self.value != other_value),
            CompareOp::Gt => Ok(self.value > other_value),
            CompareOp::Ge => Ok(self.value >= other_value),
        }
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
#[pyclass(name = "complex128", module = "pecos_rslib.dtypes", from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct ScalarComplex128 {
    value: Complex64,
}

#[pymethods]
impl ScalarComplex128 {
    /// Item size in bytes (class attribute)
    #[classattr]
    #[allow(non_upper_case_globals)] // Python API expects lowercase 'itemsize'
    const itemsize: usize = 16;

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

    /// Rich comparison support for ==, != (ordering not supported for complex numbers)
    fn __richcmp__(&self, other: &Bound<PyAny>, op: CompareOp) -> PyResult<bool> {
        let other_value = if other.is_instance_of::<ScalarComplex128>() {
            other.extract::<ScalarComplex128>()?.value
        } else if let Ok(val) = other.extract::<Complex64>() {
            val
        } else if let Ok(val) = other.extract::<f64>() {
            Complex64::new(val, 0.0)
        } else if let Ok(val) = other.extract::<i64>() {
            #[allow(clippy::cast_precision_loss)]
            // i64 to f64 conversion expected for Python compatibility
            Complex64::new(val as f64, 0.0)
        } else {
            return Ok(false);
        };

        match op {
            CompareOp::Eq => Ok(self.value == other_value),
            CompareOp::Ne => Ok(self.value != other_value),
            _ => Ok(false), // Ordering not supported for complex numbers
        }
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

/// Rust-backed angle64 scalar (fixed-point angle with u64 precision)
#[pyclass(name = "angle64", module = "pecos_rslib.dtypes", from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct ScalarAngle64 {
    value: Angle64,
}

#[pymethods]
impl ScalarAngle64 {
    /// Item size in bytes
    #[classattr]
    #[allow(non_upper_case_globals)]
    const itemsize: usize = 8;

    /// Create from a raw u64 fraction (fraction of a full turn)
    #[new]
    fn new(fraction: u64) -> Self {
        Self {
            value: Angle64::new(fraction),
        }
    }

    /// Create from radians
    #[staticmethod]
    fn from_radians(radians: f64) -> Self {
        Self {
            value: Angle64::from_radians(radians),
        }
    }

    /// Create from turns (0.0 = 0, 0.5 = half turn, 1.0 = full turn)
    #[staticmethod]
    fn from_turns(turns: f64) -> Self {
        Self {
            value: Angle64::from_turns(turns),
        }
    }

    /// Convert to radians (unsigned, in [0, 2pi))
    fn to_radians(&self) -> f64 {
        self.value.to_radians()
    }

    /// Convert to signed radians (in (-pi, pi])
    fn to_radians_signed(&self) -> f64 {
        self.value.to_radians_signed()
    }

    /// Convert to turns (in [0, 1)) -- the inverse of `from_turns`
    fn to_turns(&self) -> f64 {
        self.value.to_turns()
    }

    /// Convert to signed turns (in (-0.5, 0.5])
    fn to_turns_signed(&self) -> f64 {
        self.value.to_turns_signed()
    }

    /// Convert to half-turns (in [0, 2)); pi radians = 1.0 half-turn
    fn to_half_turns(&self) -> f64 {
        self.value.to_half_turns()
    }

    /// Convert to signed half-turns (in (-1, 1])
    fn to_half_turns_signed(&self) -> f64 {
        self.value.to_half_turns_signed()
    }

    /// Get the raw u64 fraction
    #[getter]
    fn fraction(&self) -> u64 {
        self.value.fraction()
    }

    fn __repr__(&self) -> String {
        format!("angle64({:#x})", self.value.fraction())
    }

    fn __str__(&self) -> String {
        format!("{:.6} rad", self.value.to_radians())
    }

    fn __float__(&self) -> f64 {
        self.value.to_radians()
    }

    // -- Arithmetic --

    fn __neg__(&self) -> Self {
        Self { value: -self.value }
    }

    fn __add__(&self, other: &Self) -> Self {
        Self {
            value: self.value + other.value,
        }
    }

    fn __sub__(&self, other: &Self) -> Self {
        Self {
            value: self.value - other.value,
        }
    }

    fn __mul__(&self, rhs: u64) -> Self {
        Self {
            value: self.value * rhs,
        }
    }

    fn __rmul__(&self, lhs: u64) -> Self {
        Self {
            value: self.value * lhs,
        }
    }

    fn __truediv__(&self, rhs: u64) -> Self {
        Self {
            value: self.value / rhs,
        }
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> bool {
        match op {
            CompareOp::Eq => self.value == other.value,
            CompareOp::Ne => self.value != other.value,
            CompareOp::Lt => self.value < other.value,
            CompareOp::Le => self.value <= other.value,
            CompareOp::Gt => self.value > other.value,
            CompareOp::Ge => self.value >= other.value,
        }
    }

    fn __hash__(&self) -> u64 {
        self.value.fraction()
    }

    // -- Radian-named constants (mirrors f64 constants) --

    #[classattr]
    fn pi() -> Self {
        Self {
            value: Angle64::HALF_TURN,
        }
    }

    #[classattr]
    fn tau() -> Self {
        Self {
            value: Angle64::ZERO,
        }
    } // 2pi wraps to 0

    #[classattr]
    fn frac_pi_2() -> Self {
        Self {
            value: Angle64::QUARTER_TURN,
        }
    }

    #[classattr]
    fn frac_pi_3() -> Self {
        Self {
            value: Angle64::new(Angle64::HALF_TURN.fraction() / 3),
        }
    }

    #[classattr]
    fn frac_pi_4() -> Self {
        Self {
            value: Angle64::new(Angle64::QUARTER_TURN.fraction() / 2),
        }
    }

    #[classattr]
    fn frac_pi_6() -> Self {
        Self {
            value: Angle64::new(Angle64::QUARTER_TURN.fraction() / 3),
        }
    }

    #[classattr]
    fn frac_pi_8() -> Self {
        Self {
            value: Angle64::new(Angle64::QUARTER_TURN.fraction() / 4),
        }
    }

    // -- Turn-named constants --

    #[classattr]
    fn zero() -> Self {
        Self {
            value: Angle64::ZERO,
        }
    }

    #[classattr]
    fn quarter_turn() -> Self {
        Self {
            value: Angle64::QUARTER_TURN,
        }
    }

    #[classattr]
    fn half_turn() -> Self {
        Self {
            value: Angle64::HALF_TURN,
        }
    }

    #[classattr]
    fn three_quarter_turn() -> Self {
        Self {
            value: Angle64::THREE_QUARTERS_TURN,
        }
    }

    // -- Trig --

    fn sin(&self) -> f64 {
        self.value.sin()
    }

    fn cos(&self) -> f64 {
        self.value.cos()
    }
}

impl ScalarAngle64 {
    pub fn from_angle64(value: Angle64) -> Self {
        Self { value }
    }

    pub fn inner(&self) -> Angle64 {
        self.value
    }
}

/// Wrapper for extracting angle parameters from Python.
///
/// Accepts either a `ScalarAngle64` (exact fixed-point) or a plain `f64` (radians).
/// This allows Python code to pass either `pc.angle64.pi` or `3.14159` wherever an
/// angle is expected.
pub struct AngleParam(pub Angle64);

impl<'a, 'py> pyo3::FromPyObject<'a, 'py> for AngleParam {
    type Error = PyErr;

    fn extract(ob: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> PyResult<Self> {
        if let Ok(sa) = ob.extract::<ScalarAngle64>() {
            Ok(Self(sa.inner()))
        } else if let Ok(f) = ob.extract::<f64>() {
            Ok(Self(Angle64::from_radians(f)))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "expected angle64 or float for angle parameter",
            ))
        }
    }
}

/// Module constants for dtype singletons
pub fn register_dtypes_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let dtypes = PyModule::new(parent_module.py(), "dtypes")?;

    // Register the DType class
    dtypes.add_class::<DType>()?;

    // Register all scalar types so they can be imported directly
    // Signed integers
    dtypes.add_class::<ScalarI8>()?;
    dtypes.add_class::<ScalarI16>()?;
    dtypes.add_class::<ScalarI32>()?;
    dtypes.add_class::<ScalarI64>()?;
    // Unsigned integers
    dtypes.add_class::<ScalarU8>()?;
    dtypes.add_class::<ScalarU16>()?;
    dtypes.add_class::<ScalarU32>()?;
    dtypes.add_class::<ScalarU64>()?;
    // Floats
    dtypes.add_class::<ScalarF64>()?;
    dtypes.add_class::<ScalarF32>()?;
    // Complex
    dtypes.add_class::<ScalarComplex128>()?;
    // Angles
    dtypes.add_class::<ScalarAngle64>()?;

    // Create singleton instances for each dtype (Rust-based names)
    dtypes.add("bool", DType::Bool)?;
    // Signed integers
    dtypes.add("i8", DType::I8)?;
    dtypes.add("i16", DType::I16)?;
    dtypes.add("i32", DType::I32)?;
    dtypes.add("i64", DType::I64)?;
    // Unsigned integers
    dtypes.add("u8", DType::U8)?;
    dtypes.add("u16", DType::U16)?;
    dtypes.add("u32", DType::U32)?;
    dtypes.add("u64", DType::U64)?;
    // Floats
    dtypes.add("f32", DType::F32)?;
    dtypes.add("f64", DType::F64)?;
    // Complex
    dtypes.add("complex64", DType::Complex64)?;
    dtypes.add("complex128", DType::Complex128)?;
    // Quantum types
    dtypes.add("pauli", DType::Pauli)?;
    dtypes.add("paulistring", DType::PauliString)?;

    // NumPy-compatible aliases for convenience
    dtypes.add("int8", DType::I8)?;
    dtypes.add("int16", DType::I16)?;
    dtypes.add("int32", DType::I32)?;
    dtypes.add("int64", DType::I64)?;
    dtypes.add("uint8", DType::U8)?;
    dtypes.add("uint16", DType::U16)?;
    dtypes.add("uint32", DType::U32)?;
    dtypes.add("uint64", DType::U64)?;
    dtypes.add("float32", DType::F32)?;
    dtypes.add("float64", DType::F64)?;

    // Generic aliases (default to 64-bit)
    dtypes.add("int", DType::I64)?; // Default int is 64-bit
    dtypes.add("float", DType::F64)?; // Default float is 64-bit
    dtypes.add("complex", DType::Complex128)?; // Default complex is 128-bit

    parent_module.add_submodule(&dtypes)?;
    Ok(())
}
