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

//! `NumPy` array interoperability using `PyO3`'s buffer protocol
//!
//! This module provides zero-copy interop between Rust ndarray and Python `NumPy`
//! without depending on the rust-numpy crate.
//!
//! Design goals:
//! 1. Zero-copy data sharing with Python via buffer protocol
//! 2. Support all numeric dtypes (int8-64, float32-64, complex64-128, bool)
//! 3. NumPy-compatible API via __`array_interface`__
//! 4. No Python-side numpy dependency required
//!
//! Note: `PyO3` doesn't support generic #[pyclass], so we create concrete types for each dtype.

#![allow(clippy::unnecessary_wraps)] // PyResult is required for Python error handling
#![allow(clippy::needless_pass_by_value)] // PyO3 requires passing Bound by value

use ndarray::ArrayD;
use num_complex::{Complex32, Complex64};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

// Helper macros to reduce code duplication
// Macro for numeric types (supports all operators)
macro_rules! impl_numeric_array_view {
    ($name:ident, $dtype:ty, $typestr:expr) => {
        /// Wrapper that exposes ndarray to Python via `__array_interface__`
        #[pyclass(from_py_object)]
        #[derive(Clone)]
        pub struct $name {
            data: ArrayD<$dtype>,
        }

        impl $name {
            /// Create a new `ArrayView` wrapping an ndarray
            #[allow(dead_code)]
            pub fn new(data: ArrayD<$dtype>) -> Self {
                Self { data }
            }
        }

        #[pymethods]
        #[allow(clippy::cast_possible_wrap)] // Intentional: NumPy stride calculations
        #[allow(clippy::cast_sign_loss)] // Intentional: negative index handling
        #[allow(clippy::float_cmp)] // Intentional: element-wise equality comparison
        impl $name {
            /// Expose array to `NumPy` via array interface protocol
            #[getter]
            fn __array_interface__(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
                let dict = PyDict::new(py);

                // Shape (must be tuple)
                let shape: Vec<usize> = self.data.shape().to_vec();
                dict.set_item("shape", PyTuple::new(py, &shape)?)?;

                // Data type string (NumPy format)
                dict.set_item("typestr", $typestr)?;

                // Data pointer (address, read-only flag)
                let ptr = self.data.as_ptr() as usize;
                dict.set_item("data", (ptr, false))?;

                // Strides in bytes (must be tuple)
                let strides: Vec<isize> = self
                    .data
                    .strides()
                    .iter()
                    .map(|&s| s * std::mem::size_of::<$dtype>() as isize)
                    .collect();
                dict.set_item("strides", PyTuple::new(py, &strides)?)?;

                // Protocol version
                dict.set_item("version", 3)?;

                Ok(dict.into())
            }

            /// Get array length (number of elements in first dimension)
            fn __len__(&self) -> usize {
                self.data.shape().first().copied().unwrap_or(0)
            }

            /// String representation
            fn __repr__(&self) -> String {
                format!("{}({:?})", stringify!($name), self.data)
            }

            /// Get array shape property
            #[getter]
            fn shape(&self) -> Vec<usize> {
                self.data.shape().to_vec()
            }

            /// Get array ndim property
            #[getter]
            fn ndim(&self) -> usize {
                self.data.ndim()
            }

            /// Get array size (total number of elements)
            #[getter]
            fn size(&self) -> usize {
                self.data.len()
            }

            /// Less than or equal comparison (<=)
            fn __le__(&self, py: Python<'_>, other: $dtype) -> Py<BoolArrayView> {
                let result = self.data.mapv(|x| x <= other);
                Py::new(py, BoolArrayView::new(result)).unwrap()
            }

            /// Greater than or equal comparison (>=)
            fn __ge__(&self, py: Python<'_>, other: $dtype) -> Py<BoolArrayView> {
                let result = self.data.mapv(|x| x >= other);
                Py::new(py, BoolArrayView::new(result)).unwrap()
            }

            /// Less than comparison (<)
            fn __lt__(&self, py: Python<'_>, other: $dtype) -> Py<BoolArrayView> {
                let result = self.data.mapv(|x| x < other);
                Py::new(py, BoolArrayView::new(result)).unwrap()
            }

            /// Greater than comparison (>)
            fn __gt__(&self, py: Python<'_>, other: $dtype) -> Py<BoolArrayView> {
                let result = self.data.mapv(|x| x > other);
                Py::new(py, BoolArrayView::new(result)).unwrap()
            }

            /// Equality comparison (==)
            fn __eq__(&self, py: Python<'_>, other: $dtype) -> Py<BoolArrayView> {
                let result = self.data.mapv(|x| x == other);
                Py::new(py, BoolArrayView::new(result)).unwrap()
            }

            /// Not equal comparison (!=)
            fn __ne__(&self, py: Python<'_>, other: $dtype) -> Py<BoolArrayView> {
                let result = self.data.mapv(|x| x != other);
                Py::new(py, BoolArrayView::new(result)).unwrap()
            }

            /// Addition (+)
            fn __add__(&self, py: Python<'_>, other: $dtype) -> Py<$name> {
                let result = self.data.mapv(|x| x + other);
                Py::new(py, $name::new(result)).unwrap()
            }

            /// Subtraction (-)
            fn __sub__(&self, py: Python<'_>, other: $dtype) -> Py<$name> {
                let result = self.data.mapv(|x| x - other);
                Py::new(py, $name::new(result)).unwrap()
            }

            /// Multiplication (*)
            fn __mul__(&self, py: Python<'_>, other: $dtype) -> Py<$name> {
                let result = self.data.mapv(|x| x * other);
                Py::new(py, $name::new(result)).unwrap()
            }

            /// Division (/)
            fn __truediv__(&self, py: Python<'_>, other: $dtype) -> Py<$name> {
                let result = self.data.mapv(|x| x / other);
                Py::new(py, $name::new(result)).unwrap()
            }

            /// Power (**) - Converts to f64 for power operation, then back to original type
            fn __pow__(&self, py: Python<'_>, other: $dtype, _mod: Option<$dtype>) -> Py<$name> {
                #[allow(clippy::cast_lossless)] // f32 -> f64 is lossless, but triggers warning in generic code
                #[allow(clippy::cast_possible_truncation)] // f64 -> smaller type intentional for NumPy compat
                #[allow(clippy::cast_precision_loss)]
                // i64/u64 -> f64 loses precision, matches NumPy behavior
                let result = self
                    .data
                    .mapv(|x| ((x as f64).powf(other as f64)) as $dtype);
                Py::new(py, $name::new(result)).unwrap()
            }

            /// Indexing ([]) - Basic integer indexing for 1D arrays
            fn __getitem__(&self, index: isize) -> PyResult<$dtype> {
                let len = self.data.len();
                let idx = if index < 0 {
                    (len as isize + index) as usize
                } else {
                    index as usize
                };

                if idx >= len {
                    return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                        "index {} out of range for array of length {}",
                        index, len
                    )));
                }

                Ok(self.data[idx])
            }
        }
    };
}

// Macro for complex types (no ordering operators, only equality)
macro_rules! impl_complex_array_view {
    ($name:ident, $dtype:ty, $typestr:expr) => {
        /// Wrapper that exposes ndarray to Python via `__array_interface__`
        #[pyclass(from_py_object)]
        #[derive(Clone)]
        pub struct $name {
            data: ArrayD<$dtype>,
        }

        impl $name {
            /// Create a new `ArrayView` wrapping an ndarray
            #[allow(dead_code)]
            pub fn new(data: ArrayD<$dtype>) -> Self {
                Self { data }
            }
        }

        #[pymethods]
        #[allow(clippy::cast_possible_wrap)] // Intentional: NumPy stride calculations
        #[allow(clippy::cast_sign_loss)] // Intentional: negative index handling
        #[allow(clippy::float_cmp)] // Intentional: element-wise equality comparison
        impl $name {
            /// Expose array to `NumPy` via array interface protocol
            #[getter]
            fn __array_interface__(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
                let dict = PyDict::new(py);

                // Shape (must be tuple)
                let shape: Vec<usize> = self.data.shape().to_vec();
                dict.set_item("shape", PyTuple::new(py, &shape)?)?;

                // Data type string (NumPy format)
                dict.set_item("typestr", $typestr)?;

                // Data pointer (address, read-only flag)
                let ptr = self.data.as_ptr() as usize;
                dict.set_item("data", (ptr, false))?;

                // Strides in bytes (must be tuple)
                let strides: Vec<isize> = self
                    .data
                    .strides()
                    .iter()
                    .map(|&s| s * std::mem::size_of::<$dtype>() as isize)
                    .collect();
                dict.set_item("strides", PyTuple::new(py, &strides)?)?;

                // Protocol version
                dict.set_item("version", 3)?;

                Ok(dict.into())
            }

            /// Get array length (number of elements in first dimension)
            fn __len__(&self) -> usize {
                self.data.shape().first().copied().unwrap_or(0)
            }

            /// String representation
            fn __repr__(&self) -> String {
                format!("{}({:?})", stringify!($name), self.data)
            }

            /// Get array shape property
            #[getter]
            fn shape(&self) -> Vec<usize> {
                self.data.shape().to_vec()
            }

            /// Get array ndim property
            #[getter]
            fn ndim(&self) -> usize {
                self.data.ndim()
            }

            /// Get array size (total number of elements)
            #[getter]
            fn size(&self) -> usize {
                self.data.len()
            }

            /// Equality comparison (==)
            fn __eq__(&self, py: Python<'_>, other: $dtype) -> Py<BoolArrayView> {
                let result = self.data.mapv(|x| x == other);
                Py::new(py, BoolArrayView::new(result)).unwrap()
            }

            /// Not equal comparison (!=)
            fn __ne__(&self, py: Python<'_>, other: $dtype) -> Py<BoolArrayView> {
                let result = self.data.mapv(|x| x != other);
                Py::new(py, BoolArrayView::new(result)).unwrap()
            }

            /// Addition (+)
            fn __add__(&self, py: Python<'_>, other: $dtype) -> Py<$name> {
                let result = self.data.mapv(|x| x + other);
                Py::new(py, $name::new(result)).unwrap()
            }

            /// Subtraction (-)
            fn __sub__(&self, py: Python<'_>, other: $dtype) -> Py<$name> {
                let result = self.data.mapv(|x| x - other);
                Py::new(py, $name::new(result)).unwrap()
            }

            /// Multiplication (*)
            fn __mul__(&self, py: Python<'_>, other: $dtype) -> Py<$name> {
                let result = self.data.mapv(|x| x * other);
                Py::new(py, $name::new(result)).unwrap()
            }

            /// Division (/)
            fn __truediv__(&self, py: Python<'_>, other: $dtype) -> Py<$name> {
                let result = self.data.mapv(|x| x / other);
                Py::new(py, $name::new(result)).unwrap()
            }

            /// Power (**) - Complex power using powc
            fn __pow__(&self, py: Python<'_>, other: $dtype, _mod: Option<$dtype>) -> Py<$name> {
                let result = self.data.mapv(|x| x.powc(other));
                Py::new(py, $name::new(result)).unwrap()
            }

            /// Indexing ([]) - Basic integer indexing for 1D arrays
            fn __getitem__(&self, index: isize) -> PyResult<$dtype> {
                let len = self.data.len();
                let idx = if index < 0 {
                    (len as isize + index) as usize
                } else {
                    index as usize
                };

                if idx >= len {
                    return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                        "index {} out of range for array of length {}",
                        index, len
                    )));
                }

                Ok(self.data[idx])
            }
        }
    };
}

// Macro for bool type (special handling)
macro_rules! impl_bool_array_view {
    ($name:ident, $dtype:ty, $typestr:expr) => {
        /// Wrapper that exposes ndarray to Python via `__array_interface__`
        #[pyclass(from_py_object)]
        #[derive(Clone)]
        pub struct $name {
            data: ArrayD<$dtype>,
        }

        impl $name {
            /// Create a new `ArrayView` wrapping an ndarray
            #[allow(dead_code)]
            pub fn new(data: ArrayD<$dtype>) -> Self {
                Self { data }
            }
        }

        #[pymethods]
        #[allow(clippy::cast_possible_wrap)] // Intentional: NumPy stride calculations
        #[allow(clippy::cast_sign_loss)] // Intentional: negative index handling
        #[allow(clippy::float_cmp)] // Intentional: element-wise equality comparison
        impl $name {
            /// Expose array to `NumPy` via array interface protocol
            #[getter]
            fn __array_interface__(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
                let dict = PyDict::new(py);

                // Shape (must be tuple)
                let shape: Vec<usize> = self.data.shape().to_vec();
                dict.set_item("shape", PyTuple::new(py, &shape)?)?;

                // Data type string (NumPy format)
                dict.set_item("typestr", $typestr)?;

                // Data pointer (address, read-only flag)
                let ptr = self.data.as_ptr() as usize;
                dict.set_item("data", (ptr, false))?;

                // Strides in bytes (must be tuple)
                let strides: Vec<isize> = self
                    .data
                    .strides()
                    .iter()
                    .map(|&s| s * std::mem::size_of::<$dtype>() as isize)
                    .collect();
                dict.set_item("strides", PyTuple::new(py, &strides)?)?;

                // Protocol version
                dict.set_item("version", 3)?;

                Ok(dict.into())
            }

            /// Get array length (number of elements in first dimension)
            fn __len__(&self) -> usize {
                self.data.shape().first().copied().unwrap_or(0)
            }

            /// String representation
            fn __repr__(&self) -> String {
                format!("{}({:?})", stringify!($name), self.data)
            }

            /// Get array shape property
            #[getter]
            fn shape(&self) -> Vec<usize> {
                self.data.shape().to_vec()
            }

            /// Get array ndim property
            #[getter]
            fn ndim(&self) -> usize {
                self.data.ndim()
            }

            /// Get array size (total number of elements)
            #[getter]
            fn size(&self) -> usize {
                self.data.len()
            }

            /// Indexing ([]) - Basic integer indexing for 1D arrays
            fn __getitem__(&self, index: isize) -> PyResult<$dtype> {
                let len = self.data.len();
                let idx = if index < 0 {
                    (len as isize + index) as usize
                } else {
                    index as usize
                };

                if idx >= len {
                    return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                        "index {} out of range for array of length {}",
                        index, len
                    )));
                }

                Ok(self.data[idx])
            }
        }
    };
}

// Define concrete types for each dtype
// Bool must be defined first since comparison operators return BoolArrayView
impl_bool_array_view!(BoolArrayView, bool, "|b1");

impl_numeric_array_view!(F64ArrayView, f64, "<f8");
impl_numeric_array_view!(F32ArrayView, f32, "<f4");
impl_numeric_array_view!(I64ArrayView, i64, "<i8");
impl_numeric_array_view!(I32ArrayView, i32, "<i4");
impl_numeric_array_view!(I16ArrayView, i16, "<i2");
impl_numeric_array_view!(I8ArrayView, i8, "i1");
impl_numeric_array_view!(U64ArrayView, u64, "<u8");
impl_numeric_array_view!(U32ArrayView, u32, "<u4");
impl_numeric_array_view!(U16ArrayView, u16, "<u2");
impl_numeric_array_view!(U8ArrayView, u8, "u1");

impl_complex_array_view!(Complex64ArrayView, Complex64, "<c16");
impl_complex_array_view!(Complex32ArrayView, Complex32, "<c8");

// Helper macro to implement array extraction using __array_interface__
// Uses Python's builtin getattr to work around PyO3's .getattr() not handling
// data descriptors correctly in abi3 mode
macro_rules! impl_extract_array {
    ($fn_name:ident, $dtype:ty) => {
        /// Extract array from Python array-like object using `__array_interface__`
        #[allow(clippy::items_after_statements)] // use statement in unsafe block for clarity
        #[allow(clippy::cast_possible_wrap)] // Intentional casts for NumPy stride calculations
        #[allow(clippy::cast_sign_loss)] // Intentional casts for NumPy stride calculations
        #[allow(clippy::cast_precision_loss)] // Intentional u64/i64 to f64 for NumPy compatibility
        pub fn $fn_name(obj: &Bound<'_, PyAny>) -> PyResult<ArrayD<$dtype>> {
            use ndarray::{ArrayView, IxDyn};
            use pyo3::types::{PyDict, PyList};

            let py = obj.py();

            // Check if input is a Python list and handle it directly
            if obj.is_exact_instance_of::<PyList>() {
                let list = obj.clone().cast_into::<PyList>().unwrap();
                // Extract list elements directly in Rust
                let elements: Vec<$dtype> = list.extract()?;
                let arr = ndarray::Array1::from_vec(elements);
                return Ok(arr.into_dyn());
            }

            // Get __array_interface__ using Python's builtin getattr
            // IMPORTANT: Always use Python's builtin getattr() instead of PyO3's .getattr()
            // because PyO3's getattr doesn't correctly handle data descriptors in abi3 mode.
            // NumPy's __array_interface__ is implemented as a data descriptor.
            //
            // We cannot use py.import("builtins").getattr("getattr") because .getattr() has the
            // bug we're trying to work around. Instead, we use eval to directly access the function.
            let getattr_fn = py.eval(c"getattr", None, None)?;
            let array_iface = getattr_fn.call1((obj, "__array_interface__"))?;
            let interface: &Bound<'_, PyDict> = &array_iface.cast_into::<PyDict>()?;

            // Check dtype matches
            let typestr = interface.get_item("typestr")?.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("Missing 'typestr' in __array_interface__")
            })?;
            let typestr_value: String = typestr.extract()?;

            // Get expected typestr suffix for this dtype (ignoring byte order marker)
            let expected_typestr = std::any::type_name::<$dtype>()
                .split("::")
                .last()
                .unwrap_or("");
            let expected_suffix = match expected_typestr {
                "f64" => "f8",
                "f32" => "f4",
                "i64" => "i8",
                "i32" => "i4",
                "i16" => "i2",
                "i8" => "i1",
                "u64" => "u8",
                "u32" => "u4",
                "u16" => "u2",
                "u8" => "u1",
                "bool" => "b1",
                "Complex64" | "Complex<f64>" => "c16", // Complex64 is a type alias for Complex<f64>
                "Complex32" | "Complex<f32>" => "c8",  // Complex32 is a type alias for Complex<f32>
                _ => {
                    return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                        "Unknown type: {}",
                        expected_typestr
                    )))
                }
            };

            // Check if typestr matches, allowing any byte order marker (<, >, |, =)
            if !typestr_value.ends_with(expected_suffix) {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "Type mismatch: expected *{}, got {}",
                    expected_suffix, typestr_value
                )));
            }

            // Extract shape
            let shape_tuple = interface.get_item("shape")?.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("Missing 'shape' in __array_interface__")
            })?;
            let shape: Vec<usize> = shape_tuple.extract()?;

            // Extract strides (in bytes)
            let strides_opt = interface.get_item("strides")?;
            let byte_strides: Vec<isize> = if let Some(strides_tuple) = strides_opt {
                // Check if the value is None (Python None, not Rust None)
                if strides_tuple.is_none() {
                    // If strides is None, assume C-contiguous
                    let mut strides = Vec::with_capacity(shape.len());
                    let mut stride = std::mem::size_of::<$dtype>() as isize;
                    for &dim in shape.iter().rev() {
                        strides.push(stride);
                        stride *= dim as isize;
                    }
                    strides.reverse();
                    strides
                } else {
                    strides_tuple.extract()?
                }
            } else {
                // If no strides, assume C-contiguous
                let mut strides = Vec::with_capacity(shape.len());
                let mut stride = std::mem::size_of::<$dtype>() as isize;
                for &dim in shape.iter().rev() {
                    strides.push(stride);
                    stride *= dim as isize;
                }
                strides.reverse();
                strides
            };

            // Convert byte strides to element strides (as usize for ndarray)
            let elem_strides: Vec<usize> = byte_strides
                .iter()
                .map(|&s| (s / std::mem::size_of::<$dtype>() as isize) as usize)
                .collect();

            // Extract data pointer
            let data_tuple = interface.get_item("data")?.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("Missing 'data' in __array_interface__")
            })?;
            let data_info: (usize, bool) = data_tuple.extract()?;
            let ptr = data_info.0 as *const $dtype;

            // Create ArrayView from raw parts
            // SAFETY: __array_interface__ protocol guarantees data validity
            // We immediately convert to owned array to avoid lifetime issues
            use ndarray::ShapeBuilder;
            unsafe {
                let view =
                    ArrayView::from_shape_ptr(IxDyn(&shape).strides(IxDyn(&elem_strides)), ptr);
                Ok(view.to_owned())
            }
        }
    };
}

impl_extract_array!(extract_f64_array, f64);
impl_extract_array!(extract_f32_array, f32);
impl_extract_array!(extract_i64_array, i64);
impl_extract_array!(extract_i32_array, i32);
impl_extract_array!(extract_i16_array, i16);
impl_extract_array!(extract_i8_array, i8);
impl_extract_array!(extract_u64_array, u64);
impl_extract_array!(extract_u32_array, u32);
impl_extract_array!(extract_u16_array, u16);
impl_extract_array!(extract_u8_array, u8);
impl_extract_array!(extract_bool_array, bool);
impl_extract_array!(extract_complex64_array, Complex64);
impl_extract_array!(extract_complex32_array, Complex32);

// ============================================================================
// Smart conversion helpers for numeric functions
// ============================================================================

/// Extract a real-valued array as f64, accepting PECOS Arrays, Python sequences, and real numeric types.
///
/// This is a numpy-compatible helper that accepts:
/// - PECOS Arrays with `__array_interface__` (f64, f32, i64, i32, i16, i8, u64, u32, u16, u8)
/// - Python sequences (lists, tuples) of real numbers (int or float)
/// - Automatic dtype conversion to f64 for numerical operations
///
/// Performance: Zero-copy when input is already f64 PECOS Array, otherwise allocates.
///
/// # Arguments
/// * `obj` - Python object that is either:
///   - A PECOS Array with `__array_interface__`
///   - A Python sequence (list/tuple) of real numbers
/// * `param_name` - Name of the parameter for error messages (e.g., "xdata", "ydata")
///
/// # Returns
/// f64 ndarray suitable for real-valued numerical operations
///
/// # Errors
/// Returns detailed error message if:
/// - Object has no `__array_interface__` and is not a sequence
/// - Array has unsupported dtype (complex numbers or boolean)
/// - Sequence contains non-numeric values
pub fn ensure_f64_array(obj: &Bound<'_, PyAny>, param_name: &str) -> PyResult<ArrayD<f64>> {
    use pyo3::types::PySequence;
    let py = obj.py();

    // Strategy 1: Try __array_interface__ (PECOS Array path - fast, zero-copy for f64)
    if has_array_interface(obj)? {
        return extract_from_array_interface(obj, param_name);
    }

    // Strategy 2: Try Python sequence (list, tuple, etc.)
    if let Ok(seq) = obj.clone().cast_into::<PySequence>() {
        return extract_from_sequence(py, obj, &seq, param_name);
    }

    // Strategy 3: Neither array nor sequence - provide helpful error
    make_type_error(obj, param_name)
}

/// Check if object has `__array_interface__` using Python's builtin hasattr.
fn has_array_interface(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    // IMPORTANT: Always use Python's builtin hasattr() instead of PyO3's .hasattr()
    // because PyO3's hasattr doesn't correctly handle data descriptors in abi3 mode.
    let py = obj.py();
    let hasattr_fn = py.eval(c"hasattr", None, None)?;
    hasattr_fn.call1((obj, "__array_interface__"))?.extract()
}

/// Try to extract f64 array from object with `__array_interface__`.
fn extract_from_array_interface(obj: &Bound<'_, PyAny>, param_name: &str) -> PyResult<ArrayD<f64>> {
    // Try f64 first (zero-copy path)
    if let Ok(arr) = extract_f64_array(obj) {
        return Ok(arr);
    }

    // Try other numeric types and convert to f64
    if let Some(arr) = try_extract_and_convert(obj) {
        return Ok(arr);
    }

    // Extraction failed - provide helpful error about unsupported dtype
    make_unsupported_dtype_error(obj, param_name)
}

/// Try extracting various numeric array types and convert to f64.
#[allow(clippy::cast_precision_loss)] // Intentional: i64/u64 to f64 for NumPy compatibility
fn try_extract_and_convert(obj: &Bound<'_, PyAny>) -> Option<ArrayD<f64>> {
    // Signed integers
    if let Ok(arr) = extract_i64_array(obj) {
        return Some(arr.mapv(|x| x as f64));
    }
    if let Ok(arr) = extract_i32_array(obj) {
        return Some(arr.mapv(f64::from));
    }
    if let Ok(arr) = extract_i16_array(obj) {
        return Some(arr.mapv(f64::from));
    }
    if let Ok(arr) = extract_i8_array(obj) {
        return Some(arr.mapv(f64::from));
    }
    // Unsigned integers
    if let Ok(arr) = extract_u64_array(obj) {
        return Some(arr.mapv(|x| x as f64));
    }
    if let Ok(arr) = extract_u32_array(obj) {
        return Some(arr.mapv(f64::from));
    }
    if let Ok(arr) = extract_u16_array(obj) {
        return Some(arr.mapv(f64::from));
    }
    if let Ok(arr) = extract_u8_array(obj) {
        return Some(arr.mapv(f64::from));
    }
    // Float32
    if let Ok(arr) = extract_f32_array(obj) {
        return Some(arr.mapv(f64::from));
    }
    None
}

/// Create error for unsupported array dtype.
fn make_unsupported_dtype_error(obj: &Bound<'_, PyAny>, param_name: &str) -> PyResult<ArrayD<f64>> {
    let array_iface = obj.getattr("__array_interface__")?;
    let interface = array_iface.cast::<pyo3::types::PyDict>()?;

    if let Some(typestr) = interface.get_item("typestr")? {
        let typestr_value: String = typestr.extract()?;
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "Parameter '{param_name}': Unsupported array dtype '{typestr_value}'. \
             Supported: float64, float32, int64, int32, int16, int8, uint64, uint32, uint16, uint8."
        )));
    }

    Err(pyo3::exceptions::PyValueError::new_err(format!(
        "Parameter '{param_name}': Array has __array_interface__ but missing 'typestr' field"
    )))
}

/// Extract f64 array from Python sequence.
fn extract_from_sequence(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    seq: &Bound<'_, pyo3::types::PySequence>,
    param_name: &str,
) -> PyResult<ArrayD<f64>> {
    let len = seq.len()?;
    if len == 0 {
        return Ok(ArrayD::from_shape_vec(vec![0], vec![]).unwrap());
    }

    // Check for nested sequence (e.g., [[1, 2], [3, 4]])
    let first_item = seq.get_item(0)?;
    let is_nested = first_item.cast::<pyo3::types::PySequence>().is_ok()
        && !first_item.is_instance_of::<pyo3::types::PyString>();

    if is_nested {
        return extract_nested_sequence(py, obj);
    }

    extract_flat_sequence(seq, len, param_name)
}

/// Extract f64 array from nested sequence using PECOS `array()`.
fn extract_nested_sequence(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<ArrayD<f64>> {
    let pecos_rslib = py.import("pecos_rslib")?;
    let array_fn = pecos_rslib.getattr("array")?;
    let f64_dtype = pecos_rslib.getattr("dtypes")?.getattr("f64")?;
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("dtype", f64_dtype)?;
    let pecos_array = array_fn.call((obj,), Some(&kwargs))?;
    extract_f64_array(&pecos_array)
}

/// Extract f64 array from flat sequence.
fn extract_flat_sequence(
    seq: &Bound<'_, pyo3::types::PySequence>,
    len: usize,
    param_name: &str,
) -> PyResult<ArrayD<f64>> {
    let mut vec = Vec::with_capacity(len);
    for i in 0..len {
        let item = seq.get_item(i)?;
        match item.extract::<f64>() {
            Ok(val) => vec.push(val),
            Err(_) => return make_sequence_element_error(&item, i, param_name),
        }
    }
    ArrayD::from_shape_vec(vec![len], vec)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}")))
}

/// Create error for invalid sequence element.
fn make_sequence_element_error(
    item: &Bound<'_, PyAny>,
    index: usize,
    param_name: &str,
) -> PyResult<ArrayD<f64>> {
    let item_type = get_type_name(item);
    let item_repr = get_repr(item, &item_type);
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "Parameter '{param_name}': Cannot convert element at index {index} to float64. \
         Got {item_repr} of type '{item_type}'."
    )))
}

/// Create error for unsupported object type.
fn make_type_error(obj: &Bound<'_, PyAny>, param_name: &str) -> PyResult<ArrayD<f64>> {
    let obj_type = get_type_name(obj);
    let obj_repr = get_repr(obj, &obj_type);
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "Parameter '{param_name}': Expected PECOS Array or numeric sequence, got {obj_repr} of type '{obj_type}'."
    )))
}

/// Get Python type name as string.
fn get_type_name(obj: &Bound<'_, PyAny>) -> String {
    obj.get_type()
        .name()
        .and_then(|s| s.extract::<String>())
        .unwrap_or_else(|_| String::from("<unknown type>"))
}

/// Get Python object repr as string.
fn get_repr(obj: &Bound<'_, PyAny>, fallback_type: &str) -> String {
    obj.repr()
        .and_then(|r| r.extract::<String>())
        .or_else(|_| obj.str().and_then(|s| s.extract::<String>()))
        .unwrap_or_else(|_| format!("<{fallback_type}>"))
}

// ============================================================================
// Helper functions for creating Python arrays from ndarray (PyArray replacements)
// ============================================================================

/// Helper function to create a Python-accessible array from ndarray (f64, any dimensionality)
/// This is a drop-in replacement for `PyArray::from_array()`
pub fn f64_array_to_py(
    py: Python<'_>,
    arr: &ndarray::ArrayBase<impl ndarray::Data<Elem = f64>, impl ndarray::Dimension>,
) -> Py<F64ArrayView> {
    Py::new(py, F64ArrayView::new(arr.to_owned().into_dyn())).unwrap()
}

/// Helper function to create a Python-accessible array from ndarray (i64, any dimensionality)
pub fn i64_array_to_py(
    py: Python<'_>,
    arr: &ndarray::ArrayBase<impl ndarray::Data<Elem = i64>, impl ndarray::Dimension>,
) -> Py<I64ArrayView> {
    Py::new(py, I64ArrayView::new(arr.to_owned().into_dyn())).unwrap()
}

/// Helper function to create a Python-accessible array from ndarray (Complex64, any dimensionality)
pub fn complex64_array_to_py(
    py: Python<'_>,
    arr: &ndarray::ArrayBase<impl ndarray::Data<Elem = Complex64>, impl ndarray::Dimension>,
) -> Py<Complex64ArrayView> {
    Py::new(py, Complex64ArrayView::new(arr.to_owned().into_dyn())).unwrap()
}
