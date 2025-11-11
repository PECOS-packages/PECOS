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

//! `Array` - A numpy-independent array type for Python
//!
//! This module provides a custom array type that wraps Rust's ndarray
//! and exposes it to Python without requiring numpy on the Python side.
//!
//! Design goals:
//! 1. Zero-copy data sharing with Python via buffer protocol
//! 2. Support all numeric dtypes (int8-64, float32-64, complex64-128)
//! 3. Numpy-compatible API (shape, dtype, ndim, indexing, etc.)
//! 4. No Python-side numpy dependency

// Allow Clippy pedantic lints that are not applicable to this module
#![allow(clippy::similar_names)] // start/stop/step are standard slice terminology
#![allow(clippy::too_many_lines)] // Large module with many array operations
#![allow(clippy::cast_possible_truncation)] // Intentional truncation for dtype conversions
#![allow(clippy::cast_possible_wrap)] // Intentional wrap for Python-style negative indexing
#![allow(clippy::cast_sign_loss)] // Intentional sign loss for index conversions
#![allow(clippy::cast_precision_loss)] // Expected precision loss in numeric conversions
#![allow(clippy::unnecessary_wraps)] // PyResult is required for Python error handling
#![allow(clippy::needless_pass_by_value)] // PyO3 requires passing Bound by value

use num_complex::{Complex32, Complex64};
use numpy::ndarray::{ArrayD, Axis, Slice};
use pyo3::prelude::*;
use pyo3::types::{PySequence, PySlice, PySliceIndices, PyTuple};

use crate::dtypes::DType;

/// Internal storage for array data
/// We use separate variants for each dtype to maintain type safety
#[derive(Clone)]
pub enum ArrayData {
    Int8(ArrayD<i8>),
    Int16(ArrayD<i16>),
    Int32(ArrayD<i32>),
    Int64(ArrayD<i64>),
    Float32(ArrayD<f32>),
    Float64(ArrayD<f64>),
    Complex64(ArrayD<num_complex::Complex<f32>>),
    Complex128(ArrayD<num_complex::Complex<f64>>),
}

/// Represents an indexing operation: either an integer index or a slice
#[derive(Debug, Clone, Copy)]
enum IndexOp {
    Integer(isize),
    Slice(isize, isize, isize),
}

impl ArrayData {
    /// Get the dtype of this array
    fn dtype(&self) -> DType {
        match self {
            ArrayData::Int8(_) => DType::I8,
            ArrayData::Int16(_) => DType::I16,
            ArrayData::Int32(_) => DType::I32,
            ArrayData::Int64(_) => DType::I64,
            ArrayData::Float32(_) => DType::F32,
            ArrayData::Float64(_) => DType::F64,
            ArrayData::Complex64(_) => DType::Complex64,
            ArrayData::Complex128(_) => DType::Complex128,
        }
    }

    /// Get the shape of this array
    fn shape(&self) -> &[usize] {
        match self {
            ArrayData::Int8(arr) => arr.shape(),
            ArrayData::Int16(arr) => arr.shape(),
            ArrayData::Int32(arr) => arr.shape(),
            ArrayData::Int64(arr) => arr.shape(),
            ArrayData::Float32(arr) => arr.shape(),
            ArrayData::Float64(arr) => arr.shape(),
            ArrayData::Complex64(arr) => arr.shape(),
            ArrayData::Complex128(arr) => arr.shape(),
        }
    }

    /// Get the number of dimensions
    fn ndim(&self) -> usize {
        self.shape().len()
    }

    /// Get the total number of elements
    fn size(&self) -> usize {
        self.shape().iter().product()
    }
}

/// `Array` - A numpy-independent array type for Python
///
/// This struct wraps a Rust ndarray and provides numpy-like functionality
/// without requiring numpy on the Python side.
#[pyclass(name = "Array", module = "pecos_rslib")]
pub struct Array {
    pub(crate) data: ArrayData,
}

#[pymethods]
impl Array {
    /// Create a new `Array` from a numpy array
    ///
    /// Args:
    ///     array: A numpy array to wrap
    ///
    /// Returns:
    ///     A new `Array` wrapping the numpy array data
    #[new]
    fn py_new(array: &Bound<'_, PyAny>) -> PyResult<Self> {
        use numpy::{PyArrayDyn, PyArrayMethods};

        // Try to extract as each dtype in order of likelihood
        // Start with float64 as it's most common
        if let Ok(arr) = array.cast::<PyArrayDyn<f64>>() {
            let ndarray = arr.to_owned_array();
            return Ok(Self {
                data: ArrayData::Float64(ndarray),
            });
        }

        if let Ok(arr) = array.cast::<PyArrayDyn<i64>>() {
            let ndarray = arr.to_owned_array();
            return Ok(Self {
                data: ArrayData::Int64(ndarray),
            });
        }

        if let Ok(arr) = array.cast::<PyArrayDyn<num_complex::Complex<f64>>>() {
            let ndarray = arr.to_owned_array();
            return Ok(Self {
                data: ArrayData::Complex128(ndarray),
            });
        }

        if let Ok(arr) = array.cast::<PyArrayDyn<f32>>() {
            let ndarray = arr.to_owned_array();
            return Ok(Self {
                data: ArrayData::Float32(ndarray),
            });
        }

        if let Ok(arr) = array.cast::<PyArrayDyn<i32>>() {
            let ndarray = arr.to_owned_array();
            return Ok(Self {
                data: ArrayData::Int32(ndarray),
            });
        }

        if let Ok(arr) = array.cast::<PyArrayDyn<i16>>() {
            let ndarray = arr.to_owned_array();
            return Ok(Self {
                data: ArrayData::Int16(ndarray),
            });
        }

        if let Ok(arr) = array.cast::<PyArrayDyn<i8>>() {
            let ndarray = arr.to_owned_array();
            return Ok(Self {
                data: ArrayData::Int8(ndarray),
            });
        }

        if let Ok(arr) = array.cast::<PyArrayDyn<num_complex::Complex<f32>>>() {
            let ndarray = arr.to_owned_array();
            return Ok(Self {
                data: ArrayData::Complex64(ndarray),
            });
        }

        Err(pyo3::exceptions::PyTypeError::new_err(
            "Input must be a numpy array with a supported dtype (int8-64, float32-64, complex64-128)",
        ))
    }

    /// Get the shape of the array as a tuple
    #[getter]
    fn shape(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let shape_vec: Vec<usize> = self.data.shape().to_vec();
        Ok(PyTuple::new(py, &shape_vec)?.into())
    }

    /// Get the data type of the array
    #[getter]
    fn dtype(&self) -> String {
        self.data.dtype().to_numpy_str().to_string()
    }

    /// Get the number of dimensions
    #[getter]
    fn ndim(&self) -> usize {
        self.data.ndim()
    }

    /// Get the total number of elements
    #[getter]
    fn size(&self) -> usize {
        self.data.size()
    }

    /// Implement __len__ to return the size of the first dimension
    /// This matches `NumPy`'s behavior where len(arr) returns arr.shape[0]
    fn __len__(&self) -> PyResult<usize> {
        let shape = self.data.shape();
        if shape.is_empty() {
            // Scalar arrays (0-dimensional) don't have a length
            Err(pyo3::exceptions::PyTypeError::new_err(
                "len() of unsized object (0-dimensional array)",
            ))
        } else {
            // Return the size of the first dimension
            Ok(shape[0])
        }
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "Array(shape={:?}, dtype={})",
            self.data.shape(),
            self.data.dtype().to_numpy_str()
        )
    }

    fn __str__(&self) -> String {
        self.format_array()
    }

    /// Implement __array__ method for numpy compatibility
    /// This allows numpy to convert `Array` to numpy.ndarray via `np.asarray()`
    fn __array__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        use numpy::ToPyArray;

        match &self.data {
            ArrayData::Int8(arr) => Ok(arr.to_pyarray(py).unbind().into()),
            ArrayData::Int16(arr) => Ok(arr.to_pyarray(py).unbind().into()),
            ArrayData::Int32(arr) => Ok(arr.to_pyarray(py).unbind().into()),
            ArrayData::Int64(arr) => Ok(arr.to_pyarray(py).unbind().into()),
            ArrayData::Float32(arr) => Ok(arr.to_pyarray(py).unbind().into()),
            ArrayData::Float64(arr) => Ok(arr.to_pyarray(py).unbind().into()),
            ArrayData::Complex64(arr) => Ok(arr.to_pyarray(py).unbind().into()),
            ArrayData::Complex128(arr) => Ok(arr.to_pyarray(py).unbind().into()),
        }
    }

    /// Implement __setitem__ for slice assignment support
    /// Supports:
    /// - 1D slicing: arr[start:stop] = value (unit-step only)
    /// - Multi-dimensional slicing: arr[0:2, 1:3] = value (unit-step only)
    fn __setitem__(&mut self, index: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        // Check if index is a tuple (multi-dimensional slicing)
        if let Ok(tuple) = index.cast::<PyTuple>() {
            // Parse the tuple to extract slices
            // Copy shape to avoid borrow checker issues with mutable methods
            let shape: Vec<usize> = self.data.shape().to_vec();
            let ndim = shape.len();

            if tuple.len() > ndim {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "Too many indices for array: array is {}-dimensional, but {} were indexed",
                    ndim,
                    tuple.len()
                )));
            }

            // Parse indexing operations: collect integers and slices
            let mut index_ops = Vec::new();

            for (axis, item) in tuple.iter().enumerate() {
                // Check if this dimension is a slice
                if let Ok(slice) = item.cast::<PySlice>() {
                    let (start, stop, step) = Self::parse_slice(slice, shape[axis])?;
                    index_ops.push(IndexOp::Slice(start, stop, step));
                } else if let Ok(idx) = item.extract::<isize>() {
                    // Integer index
                    index_ops.push(IndexOp::Integer(idx));
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "indices must be integers or slices",
                    ));
                }
            }

            // Apply mixed indexing assignment
            self.apply_mixed_indexing_assignment(&index_ops, &shape, value)?;
            Ok(())
        } else if let Ok(slice) = index.cast::<PySlice>() {
            // Single slice: arr[start:stop:step] = value
            let shape = self.data.shape();
            if shape.len() != 1 {
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                    "Slice assignment only works on 1D arrays for now",
                ));
            }

            let (start, stop, step) = Self::parse_slice(slice, shape[0])?;

            // Apply 1D slice assignment (now supports arbitrary steps)
            self.apply_1d_slice_assignment_with_step(start, stop, step, value)?;
            Ok(())
        } else {
            // Integer indexing (not implemented)
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "Integer indexing assignment not yet implemented (use slicing for now)",
            ))
        }
    }

    /// Implement __getitem__ for slicing support
    /// Supports:
    /// - Single integer indexing: arr[i] (not yet implemented)
    /// - Multi-dimensional indexing: arr[i, j, k] (not yet implemented)
    /// - Slicing: arr[start:stop:step] (in progress)
    /// - Multi-dimensional slicing: arr[0:2, 1:5, :] (current focus)
    fn __getitem__(&self, index: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = index.py();

        // Check if index is a tuple (multi-dimensional indexing/slicing)
        if let Ok(tuple) = index.cast::<PyTuple>() {
            // Parse the tuple to extract slices/indices
            let shape = self.data.shape();
            let ndim = shape.len();

            if tuple.len() > ndim {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "Too many indices for array: array is {}-dimensional, but {} were indexed",
                    ndim,
                    tuple.len()
                )));
            }

            // Parse indexing operations: collect integers and slices
            let mut index_ops = Vec::new();

            for (axis, item) in tuple.iter().enumerate() {
                // Check if this dimension is a slice
                if let Ok(slice) = item.cast::<PySlice>() {
                    let (start, stop, step) = Self::parse_slice(slice, shape[axis])?;
                    index_ops.push(IndexOp::Slice(start, stop, step));
                } else if let Ok(idx) = item.extract::<isize>() {
                    // Integer index
                    index_ops.push(IndexOp::Integer(idx));
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "indices must be integers or slices",
                    ));
                }
            }

            // Apply mixed indexing
            let result = self.apply_mixed_indexing(&index_ops)?;
            Ok(Py::new(py, result)?.into_any())
        } else if let Ok(slice) = index.cast::<PySlice>() {
            // Single slice: arr[start:stop:step]
            // Handle 1D slicing
            let shape = self.data.shape();
            if shape.len() != 1 {
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                    "Single-dimension slicing only works on 1D arrays for now",
                ));
            }

            let (start, stop, step) = Self::parse_slice(slice, shape[0])?;
            let slices = vec![(0, start, stop, step)];
            let result = self.apply_multidim_slicing(slices)?;
            Ok(Py::new(py, result)?.into_any())
        } else if let Ok(idx) = index.extract::<isize>() {
            // Integer indexing: arr[i]
            let shape = self.data.shape();

            // Only 1D arrays support integer indexing with a single integer
            if shape.len() != 1 {
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                    "Single integer indexing only works on 1D arrays (use tuple indexing for multi-dimensional arrays, e.g., arr[i, j])",
                ));
            }

            // Normalize negative indices
            let size = shape[0] as isize;
            let normalized_idx = if idx < 0 { size + idx } else { idx };

            // Bounds checking
            if normalized_idx < 0 || normalized_idx >= size {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "Index {idx} is out of bounds for array of size {size}"
                )));
            }

            // Extract the element and return as Python scalar
            match &self.data {
                ArrayData::Int8(arr) => {
                    let val = arr[normalized_idx as usize];
                    Ok(val.into_pyobject(py)?.into_any().unbind())
                }
                ArrayData::Int16(arr) => {
                    let val = arr[normalized_idx as usize];
                    Ok(val.into_pyobject(py)?.into_any().unbind())
                }
                ArrayData::Int32(arr) => {
                    let val = arr[normalized_idx as usize];
                    Ok(val.into_pyobject(py)?.into_any().unbind())
                }
                ArrayData::Int64(arr) => {
                    let val = arr[normalized_idx as usize];
                    Ok(val.into_pyobject(py)?.into_any().unbind())
                }
                ArrayData::Float32(arr) => {
                    let val = arr[normalized_idx as usize];
                    Ok(val.into_pyobject(py)?.into_any().unbind())
                }
                ArrayData::Float64(arr) => {
                    let val = arr[normalized_idx as usize];
                    Ok(val.into_pyobject(py)?.into_any().unbind())
                }
                ArrayData::Complex64(arr) => {
                    let val = arr[normalized_idx as usize];
                    Ok(val.into_pyobject(py)?.into_any().unbind())
                }
                ArrayData::Complex128(arr) => {
                    let val = arr[normalized_idx as usize];
                    Ok(val.into_pyobject(py)?.into_any().unbind())
                }
            }
        } else if let Ok(seq) = index.cast::<PySequence>() {
            // Fancy indexing: arr[[4, 2, 0, 3, 1]]
            // Check if array is 1D
            let shape = self.data.shape();
            if shape.len() != 1 {
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                    "Fancy indexing currently only works on 1D arrays",
                ));
            }

            // Extract indices from the sequence
            let length = seq.len()?;
            let mut indices = Vec::with_capacity(length);
            for i in 0..length {
                let item = seq.get_item(i)?;
                let idx: isize = item.extract()?;
                indices.push(idx);
            }

            // Perform fancy indexing
            let result = self.apply_fancy_indexing(&indices)?;
            Ok(Py::new(py, result)?.into_any())
        } else {
            // Unsupported indexing type
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Invalid index type - expected int, slice, tuple, or sequence",
            ))
        }
    }

    // ============================================================
    // Arithmetic operations (element-wise)
    // ============================================================

    /// Add two arrays element-wise: self + other
    fn __add__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.binary_op(other, py, |a, b| a + b, "add")
    }

    /// Subtract arrays element-wise: self - other
    fn __sub__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.binary_op(other, py, |a, b| a - b, "subtract")
    }

    /// Multiply arrays element-wise: self * other
    fn __mul__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.binary_op(other, py, |a, b| a * b, "multiply")
    }

    /// Divide arrays element-wise: self / other
    fn __truediv__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.binary_op(other, py, |a, b| a / b, "divide")
    }

    // Reverse operations (for when the left operand is a scalar)

    /// Reverse add: other + self
    fn __radd__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // Addition is commutative, so radd is the same as add
        self.__add__(other, py)
    }

    /// Reverse subtract: other - self
    fn __rsub__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.binary_op_reverse(other, py, |a, b| a - b, "subtract")
    }

    /// Reverse multiply: other * self
    fn __rmul__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // Multiplication is commutative, so rmul is the same as mul
        self.__mul__(other, py)
    }

    /// Reverse divide: other / self
    fn __rtruediv__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.binary_op_reverse(other, py, |a, b| a / b, "divide")
    }
}

impl Array {
    /// Create a new `Array` from `ArrayData`
    pub fn new(data: ArrayData) -> Self {
        Self { data }
    }

    /// Create a new `Array` from a typed ndarray
    pub fn from_array_i64(arr: ArrayD<i64>) -> Self {
        Self {
            data: ArrayData::Int64(arr),
        }
    }

    pub fn from_array_f64(arr: ArrayD<f64>) -> Self {
        Self {
            data: ArrayData::Float64(arr),
        }
    }

    pub fn from_array_c128(arr: ArrayD<num_complex::Complex<f64>>) -> Self {
        Self {
            data: ArrayData::Complex128(arr),
        }
    }

    // TODO: Add more from_array_* methods for other types

    /// Helper method for binary arithmetic operations: self op other
    /// Handles both scalar and array operands
    /// F is a closure that performs the actual operation (e.g., |a, b| a + b)
    fn binary_op<F>(
        &self,
        other: &Bound<'_, PyAny>,
        py: Python<'_>,
        op: F,
        op_name: &str,
    ) -> PyResult<Py<PyAny>>
    where
        F: Fn(f64, f64) -> f64 + Copy,
    {
        use numpy::{PyArrayDyn, PyArrayMethods};
        use pyo3::types::PyComplex;

        // Try to extract as f64 scalar first
        if let Ok(scalar) = other.extract::<f64>() {
            // Scalar operation: apply to all elements
            match &self.data {
                ArrayData::Int8(arr) => {
                    let result = arr.mapv(|x| op(f64::from(x), scalar) as i8);
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::Int8(result),
                        },
                    )?
                    .into_any())
                }
                ArrayData::Int16(arr) => {
                    let result = arr.mapv(|x| op(f64::from(x), scalar) as i16);
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::Int16(result),
                        },
                    )?
                    .into_any())
                }
                ArrayData::Int32(arr) => {
                    let result = arr.mapv(|x| op(f64::from(x), scalar) as i32);
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::Int32(result),
                        },
                    )?
                    .into_any())
                }
                ArrayData::Int64(arr) => {
                    let result = arr.mapv(|x| op(x as f64, scalar) as i64);
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::Int64(result),
                        },
                    )?
                    .into_any())
                }
                ArrayData::Float32(arr) => {
                    let result = arr.mapv(|x| op(f64::from(x), scalar) as f32);
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::Float32(result),
                        },
                    )?
                    .into_any())
                }
                ArrayData::Float64(arr) => {
                    let result = arr.mapv(|x| op(x, scalar));
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::Float64(result),
                        },
                    )?
                    .into_any())
                }
                ArrayData::Complex64(arr) => {
                    // For f64 scalar with complex array:
                    // - For add/subtract: only modify real part (a+bi) + c = (a+c) + bi
                    // - For multiply/divide: modify both parts (a+bi) * c = (a*c) + (b*c)i
                    let result = match op_name {
                        "add" | "subtract" => arr.mapv(|x| {
                            let re = op(f64::from(x.re), scalar);
                            Complex32::new(re as f32, x.im)
                        }),
                        "multiply" | "divide" => arr.mapv(|x| {
                            let re = op(f64::from(x.re), scalar);
                            let im = op(f64::from(x.im), scalar);
                            Complex32::new(re as f32, im as f32)
                        }),
                        _ => {
                            return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                                "Operation {op_name} is not implemented for Complex64 with f64 scalar"
                            )));
                        }
                    };
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::Complex64(result),
                        },
                    )?
                    .into_any())
                }
                ArrayData::Complex128(arr) => {
                    // For f64 scalar with complex array:
                    // - For add/subtract: only modify real part (a+bi) + c = (a+c) + bi
                    // - For multiply/divide: modify both parts (a+bi) * c = (a*c) + (b*c)i
                    let result = match op_name {
                        "add" | "subtract" => arr.mapv(|x| {
                            let re = op(x.re, scalar);
                            Complex64::new(re, x.im)
                        }),
                        "multiply" | "divide" => arr.mapv(|x| {
                            let re = op(x.re, scalar);
                            let im = op(x.im, scalar);
                            Complex64::new(re, im)
                        }),
                        _ => {
                            return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                                "Operation {op_name} is not implemented for Complex128 with f64 scalar"
                            )));
                        }
                    };
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::Complex128(result),
                        },
                    )?
                    .into_any())
                }
            }
        } else if let Ok(complex_scalar) = other.cast::<PyComplex>() {
            // Complex scalar operation
            let c_real = complex_scalar.real();
            let c_imag = complex_scalar.imag();
            let c = Complex64::new(c_real, c_imag);

            // Complex scalar operations are only defined for complex arrays
            // and need special handling based on the operation
            match &self.data {
                ArrayData::Complex64(arr) => {
                    let result: PyResult<Vec<Complex32>> = arr
                        .iter()
                        .map(|&x| {
                            let x64 = Complex64::new(f64::from(x.re), f64::from(x.im));
                            let res = match op_name {
                                "add" => x64 + c,
                                "subtract" => x64 - c,
                                "multiply" => x64 * c,
                                "divide" => x64 / c,
                                _ => {
                                    return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                                        format!("Complex scalar {op_name} is not implemented"),
                                    ));
                                }
                            };
                            Ok(Complex32::new(res.re as f32, res.im as f32))
                        })
                        .collect();
                    let result_vec = result?;
                    let result_arr =
                        ArrayD::from_shape_vec(arr.raw_dim(), result_vec).map_err(|e| {
                            pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}"))
                        })?;
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::Complex64(result_arr),
                        },
                    )?
                    .into_any())
                }
                ArrayData::Complex128(arr) => {
                    let result: PyResult<Vec<Complex64>> = arr
                        .iter()
                        .map(|&x| {
                            let res = match op_name {
                                "add" => x + c,
                                "subtract" => x - c,
                                "multiply" => x * c,
                                "divide" => x / c,
                                _ => {
                                    return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                                        format!("Complex scalar {op_name} is not implemented"),
                                    ));
                                }
                            };
                            Ok(res)
                        })
                        .collect();
                    let result_vec = result?;
                    let result_arr =
                        ArrayD::from_shape_vec(arr.raw_dim(), result_vec).map_err(|e| {
                            pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}"))
                        })?;
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::Complex128(result_arr),
                        },
                    )?
                    .into_any())
                }
                _ => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "Complex scalar {op_name} is only supported for complex arrays"
                ))),
            }
        } else if let Ok(other_array) = other.cast::<Array>() {
            // Array-array operation
            let other_data = &other_array.borrow().data;

            match (&self.data, other_data) {
                (ArrayData::Float64(a), ArrayData::Float64(b)) => {
                    if a.shape() != b.shape() {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Shape mismatch for {}: {:?} vs {:?}",
                            op_name,
                            a.shape(),
                            b.shape()
                        )));
                    }
                    let result = a
                        .iter()
                        .zip(b.iter())
                        .map(|(x, y)| op(*x, *y))
                        .collect::<Vec<_>>();
                    let result_arr = ArrayD::from_shape_vec(a.raw_dim(), result).map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}"))
                    })?;
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::Float64(result_arr),
                        },
                    )?
                    .into_any())
                }
                (ArrayData::Int64(a), ArrayData::Int64(b)) => {
                    if a.shape() != b.shape() {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Shape mismatch for {}: {:?} vs {:?}",
                            op_name,
                            a.shape(),
                            b.shape()
                        )));
                    }
                    let result = a
                        .iter()
                        .zip(b.iter())
                        .map(|(x, y)| op(*x as f64, *y as f64) as i64)
                        .collect::<Vec<_>>();
                    let result_arr = ArrayD::from_shape_vec(a.raw_dim(), result).map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}"))
                    })?;
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::Int64(result_arr),
                        },
                    )?
                    .into_any())
                }
                (ArrayData::Complex128(a), ArrayData::Complex128(b)) => {
                    if a.shape() != b.shape() {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Shape mismatch for {}: {:?} vs {:?}",
                            op_name,
                            a.shape(),
                            b.shape()
                        )));
                    }
                    let result = a
                        .iter()
                        .zip(b.iter())
                        .map(|(x, y)| {
                            let re = op(x.re, y.re);
                            let im = op(x.im, y.im);
                            Complex64::new(re, im)
                        })
                        .collect::<Vec<_>>();
                    let result_arr = ArrayD::from_shape_vec(a.raw_dim(), result).map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}"))
                    })?;
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::Complex128(result_arr),
                        },
                    )?
                    .into_any())
                }
                _ => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "Unsupported dtype combination for {op_name}"
                ))),
            }
        } else if let Ok(np_arr) = other.cast::<PyArrayDyn<f64>>() {
            // Numpy array operation
            let other_arr = np_arr.to_owned_array();

            match &self.data {
                ArrayData::Float64(a) => {
                    if a.shape() != other_arr.shape() {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Shape mismatch for {}: {:?} vs {:?}",
                            op_name,
                            a.shape(),
                            other_arr.shape()
                        )));
                    }
                    let result = a
                        .iter()
                        .zip(other_arr.iter())
                        .map(|(x, y)| op(*x, *y))
                        .collect::<Vec<_>>();
                    let result_arr = ArrayD::from_shape_vec(a.raw_dim(), result).map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}"))
                    })?;
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::Float64(result_arr),
                        },
                    )?
                    .into_any())
                }
                _ => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "Dtype mismatch for {op_name}"
                ))),
            }
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Unsupported operand type for {op_name}"
            )))
        }
    }

    /// Helper method for reverse binary arithmetic operations: other op self
    /// Handles scalar op array (e.g., 2.0 - array)
    fn binary_op_reverse<F>(
        &self,
        other: &Bound<'_, PyAny>,
        py: Python<'_>,
        op: F,
        op_name: &str,
    ) -> PyResult<Py<PyAny>>
    where
        F: Fn(f64, f64) -> f64 + Copy,
    {
        // Try to extract as scalar
        if let Ok(scalar) = other.extract::<f64>() {
            // Scalar operation: apply to all elements with reversed operands
            match &self.data {
                ArrayData::Int8(arr) => {
                    let result = arr.mapv(|x| op(scalar, f64::from(x)) as i8);
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::Int8(result),
                        },
                    )?
                    .into_any())
                }
                ArrayData::Int16(arr) => {
                    let result = arr.mapv(|x| op(scalar, f64::from(x)) as i16);
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::Int16(result),
                        },
                    )?
                    .into_any())
                }
                ArrayData::Int32(arr) => {
                    let result = arr.mapv(|x| op(scalar, f64::from(x)) as i32);
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::Int32(result),
                        },
                    )?
                    .into_any())
                }
                ArrayData::Int64(arr) => {
                    let result = arr.mapv(|x| op(scalar, x as f64) as i64);
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::Int64(result),
                        },
                    )?
                    .into_any())
                }
                ArrayData::Float32(arr) => {
                    let result = arr.mapv(|x| op(scalar, f64::from(x)) as f32);
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::Float32(result),
                        },
                    )?
                    .into_any())
                }
                ArrayData::Float64(arr) => {
                    let result = arr.mapv(|x| op(scalar, x));
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::Float64(result),
                        },
                    )?
                    .into_any())
                }
                ArrayData::Complex64(arr) => {
                    let result = arr.mapv(|x| {
                        let re = op(scalar, f64::from(x.re));
                        let im = op(scalar, f64::from(x.im));
                        Complex32::new(re as f32, im as f32)
                    });
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::Complex64(result),
                        },
                    )?
                    .into_any())
                }
                ArrayData::Complex128(arr) => {
                    let result = arr.mapv(|x| {
                        let re = op(scalar, x.re);
                        let im = op(scalar, x.im);
                        Complex64::new(re, im)
                    });
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::Complex128(result),
                        },
                    )?
                    .into_any())
                }
            }
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Unsupported operand type for reverse {op_name}"
            )))
        }
    }

    /// Parse a Python slice object into (start, end, step) for a given axis size
    /// This properly handles:
    /// - Negative indices (converted to positive)
    /// - None values (replaced with defaults)
    /// - Out of bounds clamping
    /// - Step direction validation
    ///
    /// IMPORTANT: For negative-step slices with default bounds, `slice.indices()`
    /// returns stop=-1 (meaning "one past the beginning"). When used with ndarray
    /// slicing, we need to handle this specially to avoid misinterpretation as
    /// negative indexing.
    ///
    /// Returns: (start, stop, step, `needs_special_handling`)
    /// - `needs_special_handling=true` means stop should be treated as None (go to beginning)
    fn parse_slice(
        slice: &Bound<'_, PySlice>,
        axis_size: usize,
    ) -> PyResult<(isize, isize, isize)> {
        let indices: PySliceIndices = slice.indices(axis_size as isize)?;

        // For negative steps, if stop=-1, this indicates we should slice all the
        // way to the beginning. Python's slice.indices() returns stop=-1 which works
        // with range() but causes problems with ndarray's slice indexing where -1
        // means "second-to-last element", not "one past the beginning".
        //
        // We handle this by converting stop=-1 to a sentinel value that calling
        // code can recognize and handle appropriately.

        Ok((indices.start, indices.stop, indices.step))
    }

    /// Apply 1D slice assignment
    /// This leverages ndarray's built-in mutable slicing capabilities
    /// Only supports 1D arrays for now
    ///
    /// The value can be:
    /// - A scalar (broadcast to all elements in the slice)
    /// - A numpy array matching the slice shape
    fn apply_1d_slice_assignment(
        &mut self,
        start: usize,
        stop: usize,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        use numpy::{PyArrayDyn, PyArrayMethods};

        // Apply 1D slice assignment based on data type
        // Use ndarray's slice_mut() with Slice::from() for unit-step slicing
        match &mut self.data {
            ArrayData::Int8(arr) => {
                let slice = Slice::from(start..stop);
                let mut view = arr.slice_mut(numpy::ndarray::s![slice]);
                if let Ok(scalar_val) = value.extract::<i8>() {
                    view.fill(scalar_val);
                } else if let Ok(arr_val) = value.cast::<PyArrayDyn<i8>>() {
                    let np_arr = arr_val.to_owned_array();
                    view.assign(&np_arr);
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "Value must be a scalar or array matching the slice shape and dtype",
                    ));
                }
            }
            ArrayData::Int16(arr) => {
                let slice = Slice::from(start..stop);
                let mut view = arr.slice_mut(numpy::ndarray::s![slice]);
                if let Ok(scalar_val) = value.extract::<i16>() {
                    view.fill(scalar_val);
                } else if let Ok(arr_val) = value.cast::<PyArrayDyn<i16>>() {
                    let np_arr = arr_val.to_owned_array();
                    view.assign(&np_arr);
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "Value must be a scalar or array matching the slice shape and dtype",
                    ));
                }
            }
            ArrayData::Int32(arr) => {
                let slice = Slice::from(start..stop);
                let mut view = arr.slice_mut(numpy::ndarray::s![slice]);
                if let Ok(scalar_val) = value.extract::<i32>() {
                    view.fill(scalar_val);
                } else if let Ok(arr_val) = value.cast::<PyArrayDyn<i32>>() {
                    let np_arr = arr_val.to_owned_array();
                    view.assign(&np_arr);
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "Value must be a scalar or array matching the slice shape and dtype",
                    ));
                }
            }
            ArrayData::Int64(arr) => {
                let slice = Slice::from(start..stop);
                let mut view = arr.slice_mut(numpy::ndarray::s![slice]);
                if let Ok(scalar_val) = value.extract::<i64>() {
                    view.fill(scalar_val);
                } else if let Ok(arr_val) = value.cast::<PyArrayDyn<i64>>() {
                    let np_arr = arr_val.to_owned_array();
                    view.assign(&np_arr);
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "Value must be a scalar or array matching the slice shape and dtype",
                    ));
                }
            }
            ArrayData::Float32(arr) => {
                let slice = Slice::from(start..stop);
                let mut view = arr.slice_mut(numpy::ndarray::s![slice]);
                if let Ok(scalar_val) = value.extract::<f32>() {
                    view.fill(scalar_val);
                } else if let Ok(arr_val) = value.cast::<PyArrayDyn<f32>>() {
                    let np_arr = arr_val.to_owned_array();
                    view.assign(&np_arr);
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "Value must be a scalar or array matching the slice shape and dtype",
                    ));
                }
            }
            ArrayData::Float64(arr) => {
                let slice = Slice::from(start..stop);
                let mut view = arr.slice_mut(numpy::ndarray::s![slice]);
                if let Ok(scalar_val) = value.extract::<f64>() {
                    view.fill(scalar_val);
                } else if let Ok(arr_val) = value.cast::<PyArrayDyn<f64>>() {
                    let np_arr = arr_val.to_owned_array();
                    view.assign(&np_arr);
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "Value must be a scalar or array matching the slice shape and dtype",
                    ));
                }
            }
            ArrayData::Complex64(arr) => {
                let slice = Slice::from(start..stop);
                let mut view = arr.slice_mut(numpy::ndarray::s![slice]);
                if let Ok(scalar_val) = value.extract::<num_complex::Complex<f32>>() {
                    view.fill(scalar_val);
                } else if let Ok(arr_val) = value.cast::<PyArrayDyn<num_complex::Complex<f32>>>() {
                    let np_arr = arr_val.to_owned_array();
                    view.assign(&np_arr);
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "Value must be a scalar or array matching the slice shape and dtype",
                    ));
                }
            }
            ArrayData::Complex128(arr) => {
                let slice = Slice::from(start..stop);
                let mut view = arr.slice_mut(numpy::ndarray::s![slice]);
                if let Ok(scalar_val) = value.extract::<num_complex::Complex<f64>>() {
                    view.fill(scalar_val);
                } else if let Ok(arr_val) = value.cast::<PyArrayDyn<num_complex::Complex<f64>>>() {
                    let np_arr = arr_val.to_owned_array();
                    view.assign(&np_arr);
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "Value must be a scalar or array matching the slice shape and dtype",
                    ));
                }
            }
        }

        Ok(())
    }

    /// Apply 1D slice assignment with arbitrary step support
    /// Handles both unit-step (step=1) and non-unit step slicing
    ///
    /// For unit steps, uses ndarray's built-in `slice_mut()` for efficiency.
    /// For non-unit steps, manually iterates through indices.
    ///
    /// The value can be:
    /// - A scalar (broadcast to all elements in the slice)
    /// - A numpy array matching the slice shape
    fn apply_1d_slice_assignment_with_step(
        &mut self,
        start: isize,
        stop: isize,
        step: isize,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        use numpy::{PyArrayDyn, PyArrayMethods};

        // Handle unit-step case efficiently using existing method
        if step == 1 {
            let start_usize = start.max(0) as usize;
            let stop_usize = stop.max(0) as usize;
            return self.apply_1d_slice_assignment(start_usize, stop_usize, value);
        }

        // Handle non-unit step case by manually iterating through indices
        // Generate the list of indices: start, start+step, start+2*step, ..., < stop
        #[allow(clippy::maybe_infinite_iter)] // False positive: iteration is bounded by take_while
        let indices: Vec<usize> = if step > 0 {
            (0..)
                .map(|i| start + i * step)
                .take_while(|&idx| idx < stop)
                .map(|idx| idx as usize)
                .collect()
        } else {
            // Negative step
            (0..)
                .map(|i| start + i * step)
                .take_while(|&idx| idx > stop)
                .map(|idx| idx as usize)
                .collect()
        };

        if indices.is_empty() {
            return Ok(()); // Nothing to assign
        }

        // Apply assignment based on data type
        match &mut self.data {
            ArrayData::Int8(arr) => {
                if let Ok(scalar_val) = value.extract::<i8>() {
                    for &idx in &indices {
                        arr[idx] = scalar_val;
                    }
                } else if let Ok(arr_val) = value.cast::<PyArrayDyn<i8>>() {
                    let np_arr = arr_val.readonly();
                    let np_slice = np_arr.as_array();
                    if np_slice.len() != indices.len() {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Shape mismatch: cannot assign array of length {} to slice of length {}",
                            np_slice.len(),
                            indices.len()
                        )));
                    }
                    for (i, &idx) in indices.iter().enumerate() {
                        arr[idx] = np_slice[i];
                    }
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "Value must be a scalar or array matching the slice shape and dtype",
                    ));
                }
            }
            ArrayData::Int16(arr) => {
                if let Ok(scalar_val) = value.extract::<i16>() {
                    for &idx in &indices {
                        arr[idx] = scalar_val;
                    }
                } else if let Ok(arr_val) = value.cast::<PyArrayDyn<i16>>() {
                    let np_arr = arr_val.readonly();
                    let np_slice = np_arr.as_array();
                    if np_slice.len() != indices.len() {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Shape mismatch: cannot assign array of length {} to slice of length {}",
                            np_slice.len(),
                            indices.len()
                        )));
                    }
                    for (i, &idx) in indices.iter().enumerate() {
                        arr[idx] = np_slice[i];
                    }
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "Value must be a scalar or array matching the slice shape and dtype",
                    ));
                }
            }
            ArrayData::Int32(arr) => {
                if let Ok(scalar_val) = value.extract::<i32>() {
                    for &idx in &indices {
                        arr[idx] = scalar_val;
                    }
                } else if let Ok(arr_val) = value.cast::<PyArrayDyn<i32>>() {
                    let np_arr = arr_val.readonly();
                    let np_slice = np_arr.as_array();
                    if np_slice.len() != indices.len() {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Shape mismatch: cannot assign array of length {} to slice of length {}",
                            np_slice.len(),
                            indices.len()
                        )));
                    }
                    for (i, &idx) in indices.iter().enumerate() {
                        arr[idx] = np_slice[i];
                    }
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "Value must be a scalar or array matching the slice shape and dtype",
                    ));
                }
            }
            ArrayData::Int64(arr) => {
                if let Ok(scalar_val) = value.extract::<i64>() {
                    for &idx in &indices {
                        arr[idx] = scalar_val;
                    }
                } else if let Ok(arr_val) = value.cast::<PyArrayDyn<i64>>() {
                    let np_arr = arr_val.readonly();
                    let np_slice = np_arr.as_array();
                    if np_slice.len() != indices.len() {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Shape mismatch: cannot assign array of length {} to slice of length {}",
                            np_slice.len(),
                            indices.len()
                        )));
                    }
                    for (i, &idx) in indices.iter().enumerate() {
                        arr[idx] = np_slice[i];
                    }
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "Value must be a scalar or array matching the slice shape and dtype",
                    ));
                }
            }
            ArrayData::Float32(arr) => {
                if let Ok(scalar_val) = value.extract::<f32>() {
                    for &idx in &indices {
                        arr[idx] = scalar_val;
                    }
                } else if let Ok(arr_val) = value.cast::<PyArrayDyn<f32>>() {
                    let np_arr = arr_val.readonly();
                    let np_slice = np_arr.as_array();
                    if np_slice.len() != indices.len() {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Shape mismatch: cannot assign array of length {} to slice of length {}",
                            np_slice.len(),
                            indices.len()
                        )));
                    }
                    for (i, &idx) in indices.iter().enumerate() {
                        arr[idx] = np_slice[i];
                    }
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "Value must be a scalar or array matching the slice shape and dtype",
                    ));
                }
            }
            ArrayData::Float64(arr) => {
                if let Ok(scalar_val) = value.extract::<f64>() {
                    for &idx in &indices {
                        arr[idx] = scalar_val;
                    }
                } else if let Ok(arr_val) = value.cast::<PyArrayDyn<f64>>() {
                    let np_arr = arr_val.readonly();
                    let np_slice = np_arr.as_array();
                    if np_slice.len() != indices.len() {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Shape mismatch: cannot assign array of length {} to slice of length {}",
                            np_slice.len(),
                            indices.len()
                        )));
                    }
                    for (i, &idx) in indices.iter().enumerate() {
                        arr[idx] = np_slice[i];
                    }
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "Value must be a scalar or array matching the slice shape and dtype",
                    ));
                }
            }
            ArrayData::Complex64(arr) => {
                if let Ok(scalar_val) = value.extract::<Complex32>() {
                    for &idx in &indices {
                        arr[idx] = scalar_val;
                    }
                } else if let Ok(arr_val) = value.cast::<PyArrayDyn<Complex32>>() {
                    let np_arr = arr_val.readonly();
                    let np_slice = np_arr.as_array();
                    if np_slice.len() != indices.len() {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Shape mismatch: cannot assign array of length {} to slice of length {}",
                            np_slice.len(),
                            indices.len()
                        )));
                    }
                    for (i, &idx) in indices.iter().enumerate() {
                        arr[idx] = np_slice[i];
                    }
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "Value must be a scalar or array matching the slice shape and dtype",
                    ));
                }
            }
            ArrayData::Complex128(arr) => {
                if let Ok(scalar_val) = value.extract::<Complex64>() {
                    for &idx in &indices {
                        arr[idx] = scalar_val;
                    }
                } else if let Ok(arr_val) = value.cast::<PyArrayDyn<Complex64>>() {
                    let np_arr = arr_val.readonly();
                    let np_slice = np_arr.as_array();
                    if np_slice.len() != indices.len() {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Shape mismatch: cannot assign array of length {} to slice of length {}",
                            np_slice.len(),
                            indices.len()
                        )));
                    }
                    for (i, &idx) in indices.iter().enumerate() {
                        arr[idx] = np_slice[i];
                    }
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "Value must be a scalar or array matching the slice shape and dtype",
                    ));
                }
            }
        }

        Ok(())
    }

    /// Apply N-dimensional slice assignment with arbitrary step support
    /// This is a generalized solution that works for any number of dimensions
    ///
    /// Note: ndarray's `slice_mut()` doesn't support non-unit steps for mutation,
    /// so we must manually iterate through all index combinations.
    /// This approach generates all valid index combinations across all dimensions,
    /// then assigns values to those indices.
    ///
    /// Fancy indexing: Select elements from a 1D array using a list of integer indices
    /// Example: arr[[4, 2, 0, 3, 1]] returns elements at indices 4, 2, 0, 3, 1 in that order
    fn apply_fancy_indexing(&self, indices: &[isize]) -> PyResult<Self> {
        let shape = self.data.shape();
        let len = shape[0];

        // Macro to implement fancy indexing for each dtype
        macro_rules! impl_fancy_indexing {
            ($arr:expr) => {{
                // Create result array of the same length as indices
                let mut result_vec = Vec::with_capacity(indices.len());

                for &idx in indices {
                    // Resolve negative indices
                    let resolved_idx = if idx < 0 {
                        let size = len as isize;
                        let resolved = size + idx;
                        if resolved < 0 {
                            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                                "index {} is out of bounds for array of length {}",
                                idx, len
                            )));
                        }
                        resolved as usize
                    } else {
                        let idx_usize = idx as usize;
                        if idx_usize >= len {
                            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                                "index {} is out of bounds for array of length {}",
                                idx, len
                            )));
                        }
                        idx_usize
                    };

                    result_vec.push($arr[resolved_idx]);
                }

                // Convert to ndarray
                let result_arr =
                    ArrayD::from_shape_vec(vec![indices.len()], result_vec).map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Failed to create result array: {}",
                            e
                        ))
                    })?;

                result_arr
            }};
        }

        // Apply fancy indexing based on dtype
        let result_data = match &self.data {
            ArrayData::Int8(arr) => ArrayData::Int8(impl_fancy_indexing!(arr)),
            ArrayData::Int16(arr) => ArrayData::Int16(impl_fancy_indexing!(arr)),
            ArrayData::Int32(arr) => ArrayData::Int32(impl_fancy_indexing!(arr)),
            ArrayData::Int64(arr) => ArrayData::Int64(impl_fancy_indexing!(arr)),
            ArrayData::Float32(arr) => ArrayData::Float32(impl_fancy_indexing!(arr)),
            ArrayData::Float64(arr) => ArrayData::Float64(impl_fancy_indexing!(arr)),
            ArrayData::Complex64(arr) => ArrayData::Complex64(impl_fancy_indexing!(arr)),
            ArrayData::Complex128(arr) => ArrayData::Complex128(impl_fancy_indexing!(arr)),
        };

        Ok(Self { data: result_data })
    }

    /// Apply multi-dimensional slicing using iterative `slice_axis()`
    /// This leverages ndarray's built-in slicing capabilities
    /// Supports arbitrary step sizes including negative steps
    fn apply_multidim_slicing(
        &self,
        slices: Vec<(usize, isize, isize, isize)>, // (axis, start, stop, step)
    ) -> PyResult<Self> {
        // Apply slices iteratively using ndarray's slice_axis()
        // For negative steps, we convert to forward slice + invert_axis
        match &self.data {
            ArrayData::Int8(arr) => {
                let mut result = arr.clone();
                for (axis, start, stop, step) in slices {
                    if step < 0 {
                        // ndarray's Slice doesn't match NumPy for negative steps (see issue #312)
                        // We need to manually implement NumPy's behavior:
                        // 1. Slice forward [stop+1, start+1] with step=1
                        // 2. Reverse the axis
                        // 3. Apply step magnitude if > 1
                        let actual_start = if stop == -1 { 0 } else { stop + 1 };
                        let actual_end = start + 1;
                        let slice_info = Slice::new(actual_start, Some(actual_end), 1);
                        result = result.slice_axis(Axis(axis), slice_info).to_owned();
                        result.invert_axis(Axis(axis));

                        // Now apply step magnitude if it's not -1
                        let step_magnitude = step.abs();
                        if step_magnitude > 1 {
                            let slice_stepped = Slice::new(0, None, step_magnitude);
                            result = result.slice_axis(Axis(axis), slice_stepped).to_owned();
                        }
                    } else {
                        let slice_info = Slice::new(start, Some(stop), step);
                        result = result.slice_axis(Axis(axis), slice_info).to_owned();
                    }
                }
                Ok(Array {
                    data: ArrayData::Int8(result),
                })
            }
            ArrayData::Int16(arr) => {
                let mut result = arr.clone();
                for (axis, start, stop, step) in slices {
                    if step < 0 {
                        // ndarray's Slice doesn't match NumPy for negative steps (see issue #312)
                        // We need to manually implement NumPy's behavior:
                        // 1. Slice forward [stop+1, start+1] with step=1
                        // 2. Reverse the axis
                        // 3. Apply step magnitude if > 1
                        let actual_start = if stop == -1 { 0 } else { stop + 1 };
                        let actual_end = start + 1;
                        let slice_info = Slice::new(actual_start, Some(actual_end), 1);
                        result = result.slice_axis(Axis(axis), slice_info).to_owned();
                        result.invert_axis(Axis(axis));

                        // Now apply step magnitude if it's not -1
                        let step_magnitude = step.abs();
                        if step_magnitude > 1 {
                            let slice_stepped = Slice::new(0, None, step_magnitude);
                            result = result.slice_axis(Axis(axis), slice_stepped).to_owned();
                        }
                    } else {
                        let slice_info = Slice::new(start, Some(stop), step);
                        result = result.slice_axis(Axis(axis), slice_info).to_owned();
                    }
                }
                Ok(Array {
                    data: ArrayData::Int16(result),
                })
            }
            ArrayData::Int32(arr) => {
                let mut result = arr.clone();
                for (axis, start, stop, step) in slices {
                    if step < 0 {
                        // ndarray's Slice doesn't match NumPy for negative steps (see issue #312)
                        // We need to manually implement NumPy's behavior:
                        // 1. Slice forward [stop+1, start+1] with step=1
                        // 2. Reverse the axis
                        // 3. Apply step magnitude if > 1
                        let actual_start = if stop == -1 { 0 } else { stop + 1 };
                        let actual_end = start + 1;
                        let slice_info = Slice::new(actual_start, Some(actual_end), 1);
                        result = result.slice_axis(Axis(axis), slice_info).to_owned();
                        result.invert_axis(Axis(axis));

                        // Now apply step magnitude if it's not -1
                        let step_magnitude = step.abs();
                        if step_magnitude > 1 {
                            let slice_stepped = Slice::new(0, None, step_magnitude);
                            result = result.slice_axis(Axis(axis), slice_stepped).to_owned();
                        }
                    } else {
                        let slice_info = Slice::new(start, Some(stop), step);
                        result = result.slice_axis(Axis(axis), slice_info).to_owned();
                    }
                }
                Ok(Array {
                    data: ArrayData::Int32(result),
                })
            }
            ArrayData::Int64(arr) => {
                let mut result = arr.clone();
                for (axis, start, stop, step) in slices {
                    if step < 0 {
                        // ndarray's Slice doesn't match NumPy for negative steps (see issue #312)
                        // We need to manually implement NumPy's behavior:
                        // 1. Slice forward [stop+1, start+1] with step=1
                        // 2. Reverse the axis
                        // 3. Apply step magnitude if > 1
                        let actual_start = if stop == -1 { 0 } else { stop + 1 };
                        let actual_end = start + 1;
                        let slice_info = Slice::new(actual_start, Some(actual_end), 1);
                        result = result.slice_axis(Axis(axis), slice_info).to_owned();
                        result.invert_axis(Axis(axis));

                        // Now apply step magnitude if it's not -1
                        let step_magnitude = step.abs();
                        if step_magnitude > 1 {
                            let slice_stepped = Slice::new(0, None, step_magnitude);
                            result = result.slice_axis(Axis(axis), slice_stepped).to_owned();
                        }
                    } else {
                        let slice_info = Slice::new(start, Some(stop), step);
                        result = result.slice_axis(Axis(axis), slice_info).to_owned();
                    }
                }
                Ok(Array {
                    data: ArrayData::Int64(result),
                })
            }
            ArrayData::Float32(arr) => {
                let mut result = arr.clone();
                for (axis, start, stop, step) in slices {
                    if step < 0 {
                        // ndarray's Slice doesn't match NumPy for negative steps (see issue #312)
                        // We need to manually implement NumPy's behavior:
                        // 1. Slice forward [stop+1, start+1] with step=1
                        // 2. Reverse the axis
                        // 3. Apply step magnitude if > 1
                        let actual_start = if stop == -1 { 0 } else { stop + 1 };
                        let actual_end = start + 1;
                        let slice_info = Slice::new(actual_start, Some(actual_end), 1);
                        result = result.slice_axis(Axis(axis), slice_info).to_owned();
                        result.invert_axis(Axis(axis));

                        // Now apply step magnitude if it's not -1
                        let step_magnitude = step.abs();
                        if step_magnitude > 1 {
                            let slice_stepped = Slice::new(0, None, step_magnitude);
                            result = result.slice_axis(Axis(axis), slice_stepped).to_owned();
                        }
                    } else {
                        let slice_info = Slice::new(start, Some(stop), step);
                        result = result.slice_axis(Axis(axis), slice_info).to_owned();
                    }
                }
                Ok(Array {
                    data: ArrayData::Float32(result),
                })
            }
            ArrayData::Float64(arr) => {
                let mut result = arr.clone();
                for (axis, start, stop, step) in slices {
                    if step < 0 {
                        // ndarray's Slice doesn't match NumPy for negative steps (see issue #312)
                        // We need to manually implement NumPy's behavior:
                        // 1. Slice forward [stop+1, start+1] with step=1
                        // 2. Reverse the axis
                        // 3. Apply step magnitude if > 1
                        let actual_start = if stop == -1 { 0 } else { stop + 1 };
                        let actual_end = start + 1;
                        let slice_info = Slice::new(actual_start, Some(actual_end), 1);
                        result = result.slice_axis(Axis(axis), slice_info).to_owned();
                        result.invert_axis(Axis(axis));

                        // Now apply step magnitude if it's not -1
                        let step_magnitude = step.abs();
                        if step_magnitude > 1 {
                            let slice_stepped = Slice::new(0, None, step_magnitude);
                            result = result.slice_axis(Axis(axis), slice_stepped).to_owned();
                        }
                    } else {
                        let slice_info = Slice::new(start, Some(stop), step);
                        result = result.slice_axis(Axis(axis), slice_info).to_owned();
                    }
                }
                Ok(Array {
                    data: ArrayData::Float64(result),
                })
            }
            ArrayData::Complex64(arr) => {
                let mut result = arr.clone();
                for (axis, start, stop, step) in slices {
                    if step < 0 {
                        // ndarray's Slice doesn't match NumPy for negative steps (see issue #312)
                        // We need to manually implement NumPy's behavior:
                        // 1. Slice forward [stop+1, start+1] with step=1
                        // 2. Reverse the axis
                        // 3. Apply step magnitude if > 1
                        let actual_start = if stop == -1 { 0 } else { stop + 1 };
                        let actual_end = start + 1;
                        let slice_info = Slice::new(actual_start, Some(actual_end), 1);
                        result = result.slice_axis(Axis(axis), slice_info).to_owned();
                        result.invert_axis(Axis(axis));

                        // Now apply step magnitude if it's not -1
                        let step_magnitude = step.abs();
                        if step_magnitude > 1 {
                            let slice_stepped = Slice::new(0, None, step_magnitude);
                            result = result.slice_axis(Axis(axis), slice_stepped).to_owned();
                        }
                    } else {
                        let slice_info = Slice::new(start, Some(stop), step);
                        result = result.slice_axis(Axis(axis), slice_info).to_owned();
                    }
                }
                Ok(Array {
                    data: ArrayData::Complex64(result),
                })
            }
            ArrayData::Complex128(arr) => {
                let mut result = arr.clone();
                for (axis, start, stop, step) in slices {
                    if step < 0 {
                        // ndarray's Slice doesn't match NumPy for negative steps (see issue #312)
                        // We need to manually implement NumPy's behavior:
                        // 1. Slice forward [stop+1, start+1] with step=1
                        // 2. Reverse the axis
                        // 3. Apply step magnitude if > 1
                        let actual_start = if stop == -1 { 0 } else { stop + 1 };
                        let actual_end = start + 1;
                        let slice_info = Slice::new(actual_start, Some(actual_end), 1);
                        result = result.slice_axis(Axis(axis), slice_info).to_owned();
                        result.invert_axis(Axis(axis));

                        // Now apply step magnitude if it's not -1
                        let step_magnitude = step.abs();
                        if step_magnitude > 1 {
                            let slice_stepped = Slice::new(0, None, step_magnitude);
                            result = result.slice_axis(Axis(axis), slice_stepped).to_owned();
                        }
                    } else {
                        let slice_info = Slice::new(start, Some(stop), step);
                        result = result.slice_axis(Axis(axis), slice_info).to_owned();
                    }
                }
                Ok(Array {
                    data: ArrayData::Complex128(result),
                })
            }
        }
    }

    /// Format the array nicely like numpy
    /// For 1D: [1.0, 2.0, 3.0]
    /// For 2D: [[1.0, 2.0]
    ///          [3.0, 4.0]]
    /// For 3D: [[[1, 2], [3, 4]]
    ///          [[5, 6], [7, 8]]]
    fn format_array(&self) -> String {
        match &self.data {
            ArrayData::Float64(arr) => Self::format_array_typed(arr, "float64"),
            ArrayData::Float32(arr) => Self::format_array_typed(arr, "float32"),
            ArrayData::Int64(arr) => Self::format_array_typed(arr, "int64"),
            ArrayData::Int32(arr) => Self::format_array_typed(arr, "int32"),
            ArrayData::Int16(arr) => Self::format_array_typed(arr, "int16"),
            ArrayData::Int8(arr) => Self::format_array_typed(arr, "int8"),
            ArrayData::Complex64(arr) => Self::format_array_complex_f32(arr),
            ArrayData::Complex128(arr) => Self::format_array_complex_f64(arr),
        }
    }

    /// Format a typed array (non-complex)
    fn format_array_typed<T: std::fmt::Display>(arr: &ArrayD<T>, dtype_str: &str) -> String {
        let shape = arr.shape();
        let ndim = shape.len();

        match ndim {
            1 => {
                // 1D: [1.0, 2.0, 3.0]
                let elements: Vec<String> = arr.iter().map(|x| format!("{x}")).collect();
                format!("[{}]", elements.join(", "))
            }
            2 => {
                // 2D: [[1.0, 2.0]
                //      [3.0, 4.0]]
                let rows: Vec<String> = (0..shape[0])
                    .map(|i| {
                        let row_elements: Vec<String> =
                            (0..shape[1]).map(|j| format!("{}", arr[[i, j]])).collect();
                        format!("[{}]", row_elements.join(", "))
                    })
                    .collect();

                if rows.len() == 1 {
                    format!("[{}]", rows[0])
                } else {
                    let first_row = &rows[0];
                    let other_rows: Vec<String> =
                        rows[1..].iter().map(|row| format!(" {row}")).collect();
                    format!("[{}\n{}]", first_row, other_rows.join("\n"))
                }
            }
            3 => {
                // 3D: [[[1, 2], [3, 4]]
                //      [[5, 6], [7, 8]]]
                let planes: Vec<String> = (0..shape[0])
                    .map(|i| {
                        let rows: Vec<String> = (0..shape[1])
                            .map(|j| {
                                let row_elements: Vec<String> = (0..shape[2])
                                    .map(|k| format!("{}", arr[[i, j, k]]))
                                    .collect();
                                format!("[{}]", row_elements.join(", "))
                            })
                            .collect();
                        if rows.len() == 1 {
                            format!("[{}]", rows[0])
                        } else {
                            format!("[{}, {}]", rows[0], rows[1..].join(", "))
                        }
                    })
                    .collect();

                if planes.len() == 1 {
                    format!("[{}]", planes[0])
                } else {
                    let first_plane = &planes[0];
                    let other_planes: Vec<String> = planes[1..]
                        .iter()
                        .map(|plane| format!(" {plane}"))
                        .collect();
                    format!("[{}\n{}]", first_plane, other_planes.join("\n"))
                }
            }
            _ => {
                // For higher dimensions, just show shape and dtype
                format!("Array(shape={shape:?}, dtype={dtype_str})")
            }
        }
    }

    /// Format a complex array for f32
    fn format_array_complex_f32(arr: &ArrayD<num_complex::Complex<f32>>) -> String {
        Self::format_array_complex_generic(arr, 0.0_f32)
    }

    /// Format a complex array for f64
    fn format_array_complex_f64(arr: &ArrayD<num_complex::Complex<f64>>) -> String {
        Self::format_array_complex_generic(arr, 0.0_f64)
    }

    /// Generic complex array formatting
    fn format_array_complex_generic<T>(arr: &ArrayD<num_complex::Complex<T>>, zero: T) -> String
    where
        T: std::fmt::Display + PartialOrd,
    {
        let shape = arr.shape();
        let ndim = shape.len();

        match ndim {
            1 => {
                // 1D: [(1+2j), (3+4j)]
                let elements: Vec<String> = arr
                    .iter()
                    .map(|x| {
                        if x.im >= zero {
                            format!("({}+{}j)", x.re, x.im)
                        } else {
                            format!("({}{}j)", x.re, x.im)
                        }
                    })
                    .collect();
                format!("[{}]", elements.join(", "))
            }
            2 => {
                // 2D formatting for complex
                let rows: Vec<String> = (0..shape[0])
                    .map(|i| {
                        let row_elements: Vec<String> = (0..shape[1])
                            .map(|j| {
                                let x = &arr[[i, j]];
                                if x.im >= zero {
                                    format!("({}+{}j)", x.re, x.im)
                                } else {
                                    format!("({}{}j)", x.re, x.im)
                                }
                            })
                            .collect();
                        format!("[{}]", row_elements.join(", "))
                    })
                    .collect();

                if rows.len() == 1 {
                    format!("[{}]", rows[0])
                } else {
                    let first_row = &rows[0];
                    let other_rows: Vec<String> =
                        rows[1..].iter().map(|row| format!(" {row}")).collect();
                    format!("[{}\n{}]", first_row, other_rows.join("\n"))
                }
            }
            _ => {
                // For 3D+ complex, just show shape and dtype
                format!("Array(shape={shape:?}, dtype=complex)")
            }
        }
    }

    /// Apply mixed integer/slice indexing leveraging ndarray's `index_axis` and `slice_axis`
    /// This method handles cases like arr[0, 1:3] or arr[:, 0]
    /// where some dimensions are indexed by integers (reducing dimensionality)
    /// and others are sliced (preserving dimensionality)
    fn apply_mixed_indexing(&self, index_ops: &[IndexOp]) -> PyResult<Self> {
        // Check if all are integers (pure integer indexing)
        let all_integers = index_ops.iter().all(|op| matches!(op, IndexOp::Integer(_)));
        if all_integers {
            // Pure integer indexing - use existing implementation
            // This case is already handled by multi_index methods
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "Pure integer indexing should be handled by existing code",
            ));
        }

        // Check if all are slices (pure slice indexing)
        let all_slices = index_ops
            .iter()
            .all(|op| matches!(op, IndexOp::Slice(_, _, _)));
        if all_slices {
            // Pure slice indexing - use existing implementation
            let slices: Vec<(usize, isize, isize, isize)> = index_ops
                .iter()
                .enumerate()
                .map(|(axis, op)| {
                    if let IndexOp::Slice(start, stop, step) = op {
                        (axis, *start, *stop, *step)
                    } else {
                        unreachable!()
                    }
                })
                .collect();
            return self.apply_multidim_slicing(slices);
        }

        // Mixed indexing: combination of integers and slices
        // Strategy: Apply operations sequentially, but index parameters are ALREADY computed
        // based on the ORIGINAL array shape. We need to re-normalize them for the CURRENT array.

        // Macro to generate the mixed indexing logic for each dtype
        macro_rules! apply_mixed_indexing_impl {
            ($arr:expr, $variant:ident) => {{
                // Start with owned array
                let mut result = $arr.clone();
                let mut current_axis = 0;

                for op in index_ops.iter() {
                    match op {
                        IndexOp::Integer(idx) => {
                            // Get the current shape of the result array (which may have been reduced)
                            let current_shape = result.shape();

                            // current_axis should be within bounds of the current result shape
                            if current_axis >= current_shape.len() {
                                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                                    "Too many indices for array with {} dimensions",
                                    current_shape.len()
                                )));
                            }

                            let axis_size = current_shape[current_axis];

                            // Resolve negative index based on CURRENT axis size
                            // NOTE: The index was already validated against the ORIGINAL shape,
                            // but after dimension reduction, we need to re-validate
                            let resolved_idx = if *idx < 0 {
                                ((axis_size as isize) + idx) as usize
                            } else {
                                *idx as usize
                            };

                            // Bounds check against CURRENT axis size
                            if resolved_idx >= axis_size {
                                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                                    "Index {} is out of bounds for axis {} with size {}",
                                    idx, current_axis, axis_size
                                )));
                            }

                            // Use index_axis to select along this axis and convert to owned
                            // This reduces dimensionality
                            result = result.index_axis(Axis(current_axis), resolved_idx).to_owned();
                            // Don't increment current_axis because we removed a dimension
                        }
                        IndexOp::Slice(start, stop, step) => {
                            // The slice parameters (start, stop, step) were calculated by Python's
                            // slice.indices() based on the original array shape. These are correct for
                            // the SIZE of the axis. After dimension reduction from integer indexing,
                            // the axis SIZE doesn't change (only the axis NUMBER changes).
                            // So we can use the slice params as-is, just on the current_axis.

                            if *step < 0 {
                                // ndarray's Slice doesn't match NumPy for negative steps (see issue #312)
                                // We need to manually implement NumPy's behavior:
                                // 1. Slice forward [stop+1, start+1] with step=1
                                // 2. Reverse the axis
                                // 3. Apply step magnitude if > 1
                                let actual_start = if *stop == -1 { 0 } else { stop + 1 };
                                let actual_end = start + 1;
                                let slice_info = Slice::new(actual_start, Some(actual_end), 1);
                                result = result.slice_axis(Axis(current_axis), slice_info).to_owned();
                                result.invert_axis(Axis(current_axis));

                                // Now apply step magnitude if it's not -1
                                let step_magnitude = step.abs();
                                if step_magnitude > 1 {
                                    let slice_stepped = Slice::new(0, None, step_magnitude);
                                    result = result.slice_axis(Axis(current_axis), slice_stepped).to_owned();
                                }
                            } else {
                                // Positive step: use the slice as-is
                                let slice_info = Slice::new(*start, Some(*stop), *step);
                                result = result.slice_axis(Axis(current_axis), slice_info).to_owned();
                            }
                            current_axis += 1; // Move to next axis in the result
                        }
                    }
                }

                Ok(Self {
                    data: ArrayData::$variant(result),
                })
            }};
        }

        // Apply the operation to each dtype variant
        match &self.data {
            ArrayData::Float64(arr) => apply_mixed_indexing_impl!(arr, Float64),
            ArrayData::Float32(arr) => apply_mixed_indexing_impl!(arr, Float32),
            ArrayData::Int64(arr) => apply_mixed_indexing_impl!(arr, Int64),
            ArrayData::Int32(arr) => apply_mixed_indexing_impl!(arr, Int32),
            ArrayData::Int16(arr) => apply_mixed_indexing_impl!(arr, Int16),
            ArrayData::Int8(arr) => apply_mixed_indexing_impl!(arr, Int8),
            ArrayData::Complex128(arr) => apply_mixed_indexing_impl!(arr, Complex128),
            ArrayData::Complex64(arr) => apply_mixed_indexing_impl!(arr, Complex64),
        }
    }

    /// Apply mixed integer/slice indexing assignment to an array
    /// This method uses ndarray's `index_axis_mut()` and `slice_axis_mut()` for mutable views
    /// Similar to `apply_mixed_indexing` but for assignment operations
    fn apply_mixed_indexing_assignment(
        &mut self,
        index_ops: &[IndexOp],
        shape: &[usize],
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        use numpy::{PyArrayDyn, PyArrayMethods};

        // Macro to generate the mixed indexing assignment logic for each dtype
        macro_rules! apply_mixed_indexing_assignment_impl {
            ($arr:expr, $dtype:ty, $variant:ident) => {{
                // Strategy: Convert integers to single-element slices, then use slice_each_axis_mut
                // This avoids the borrow checker issues with chaining mutable slices

                use numpy::ndarray::SliceInfoElem;

                // Build slice info elements for each axis
                let mut slice_infos: Vec<SliceInfoElem> = Vec::new();
                let integer_axes: Vec<usize> = index_ops
                    .iter()
                    .enumerate()
                    .filter_map(|(i, op)| match op {
                        IndexOp::Integer(_) => Some(i),
                        _ => None,
                    })
                    .collect();

                for (original_axis, op) in index_ops.iter().enumerate() {
                    match op {
                        IndexOp::Integer(idx) => {
                            // Resolve negative index
                            let resolved_idx = if *idx < 0 {
                                let axis_size = shape[original_axis] as isize;
                                (axis_size + idx) as usize
                            } else {
                                *idx as usize
                            };

                            // Bounds check
                            if resolved_idx >= shape[original_axis] {
                                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                                    "Index {} is out of bounds for axis {} with size {}",
                                    idx, original_axis, shape[original_axis]
                                )));
                            }

                            // Use Index to reduce dimensionality directly
                            slice_infos.push(SliceInfoElem::Index(resolved_idx as isize));
                        }
                        IndexOp::Slice(start, stop, step) => {
                            // Add as a slice (this preserves dimensionality)
                            slice_infos.push(SliceInfoElem::Slice {
                                start: *start,
                                end: Some(*stop),
                                step: *step,
                            });
                        }
                    }
                }

                // Try to use ndarray's slice_mut with dynamic SliceInfo
                // Actually, let's use a different approach: ndarray's slice_each_axis_mut
                // which works better with dynamic dimensions

                // Use slice_each_axis_mut which returns an iterator
                // For now, let's use a workaround: manually index into the array

                // Actually, the simplest approach is to use ndarray's select API
                // But for mutable access, we need to be more careful

                // Let me use a different strategy: process each index operation one at a time
                // using slice_collapse for integers and slice_axis_mut for slices

                // First, let's check if we have only slices (no integers) - that's simpler
                if integer_axes.is_empty() {
                    // All slices - convert to ranges and use the recursive approach
                    // This avoids the borrow checker issue completely
                    let mut ranges: Vec<Vec<usize>> = Vec::new();

                    for op in index_ops.iter() {
                        if let IndexOp::Slice(start, stop, step) = op {
                            // Generate range of indices
                            let mut indices = Vec::new();
                            let mut i = *start;
                            while (*step > 0 && i < *stop) || (*step < 0 && i > *stop) {
                                indices.push(i as usize);
                                i += step;
                            }
                            ranges.push(indices);
                        }
                    }

                    // Calculate the shape of the result
                    let result_shape: Vec<usize> = ranges.iter().map(|r| r.len()).collect();

                    // Assign value
                    if let Ok(scalar_val) = value.extract::<$dtype>() {
                        // Scalar assignment - iterate over all target indices
                        Self::assign_to_mixed_indices($arr, &ranges, scalar_val);
                    } else if let Ok(arr_val) = value.cast::<PyArrayDyn<$dtype>>() {
                        let np_arr = arr_val.to_owned_array();

                        // Check shape compatibility
                        if np_arr.shape() != result_shape.as_slice() {
                            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                "Shape mismatch: target has shape {:?}, but source has shape {:?}",
                                result_shape,
                                np_arr.shape()
                            )));
                        }

                        // Since there are no integer axes, we can use a simpler assignment
                        let integer_axes_empty: Vec<usize> = Vec::new();
                        Self::assign_array_to_mixed_indices(
                            $arr,
                            &ranges,
                            &integer_axes_empty,
                            &np_arr,
                        )?;
                    } else {
                        return Err(pyo3::exceptions::PyTypeError::new_err(
                            "Value must be a scalar or array matching the slice shape and dtype",
                        ));
                    }
                } else {
                    // Mixed indexing with integers - need special handling
                    // Use nested iteration approach

                    // First, convert all operations to slice ranges for iteration
                    let mut ranges: Vec<Vec<usize>> = Vec::new();

                    for (axis, op) in index_ops.iter().enumerate() {
                        match op {
                            IndexOp::Integer(idx) => {
                                // Resolve negative index
                                let resolved_idx = if *idx < 0 {
                                    let axis_size = shape[axis] as isize;
                                    (axis_size + idx) as usize
                                } else {
                                    *idx as usize
                                };

                                // Single index
                                ranges.push(vec![resolved_idx]);
                            }
                            IndexOp::Slice(start, stop, step) => {
                                // Generate range of indices
                                let mut indices = Vec::new();
                                let mut i = *start;
                                while (*step > 0 && i < *stop) || (*step < 0 && i > *stop) {
                                    indices.push(i as usize);
                                    i += step;
                                }
                                ranges.push(indices);
                            }
                        }
                    }

                    // Calculate the shape of the result (only slice dimensions)
                    let result_shape: Vec<usize> = ranges
                        .iter()
                        .enumerate()
                        .filter_map(|(i, r)| {
                            if integer_axes.contains(&i) {
                                None
                            } else {
                                Some(r.len())
                            }
                        })
                        .collect();

                    // Now handle the value assignment
                    if let Ok(scalar_val) = value.extract::<$dtype>() {
                        // Scalar assignment - iterate over all target indices
                        // Generate all combinations of indices
                        Self::assign_to_mixed_indices($arr, &ranges, scalar_val);
                    } else if let Ok(arr_val) = value.cast::<PyArrayDyn<$dtype>>() {
                        let np_arr = arr_val.to_owned_array();

                        // Check shape compatibility
                        if np_arr.shape() != result_shape.as_slice() {
                            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                "Shape mismatch: target has shape {:?}, but source has shape {:?}",
                                result_shape,
                                np_arr.shape()
                            )));
                        }

                        // Assign array values - need to map result indices to target indices
                        Self::assign_array_to_mixed_indices($arr, &ranges, &integer_axes, &np_arr)?;
                    } else {
                        return Err(pyo3::exceptions::PyTypeError::new_err(
                            "Value must be a scalar or array matching the slice shape and dtype",
                        ));
                    }
                }

                Ok(())
            }};
        }

        // Apply the operation to each dtype variant
        match &mut self.data {
            ArrayData::Float64(arr) => apply_mixed_indexing_assignment_impl!(arr, f64, Float64),
            ArrayData::Float32(arr) => apply_mixed_indexing_assignment_impl!(arr, f32, Float32),
            ArrayData::Int64(arr) => apply_mixed_indexing_assignment_impl!(arr, i64, Int64),
            ArrayData::Int32(arr) => apply_mixed_indexing_assignment_impl!(arr, i32, Int32),
            ArrayData::Int16(arr) => apply_mixed_indexing_assignment_impl!(arr, i16, Int16),
            ArrayData::Int8(arr) => apply_mixed_indexing_assignment_impl!(arr, i8, Int8),
            ArrayData::Complex128(arr) => {
                apply_mixed_indexing_assignment_impl!(arr, num_complex::Complex<f64>, Complex128)
            }
            ArrayData::Complex64(arr) => {
                apply_mixed_indexing_assignment_impl!(arr, num_complex::Complex<f32>, Complex64)
            }
        }
    }

    // Helper method: Assign a scalar value to all indices specified by ranges
    fn assign_to_mixed_indices<T: Clone>(
        arr: &mut numpy::ndarray::ArrayD<T>,
        ranges: &[Vec<usize>],
        value: T,
    ) {
        // Recursively iterate through all combinations of indices
        fn assign_recursive<T: Clone>(
            arr: &mut numpy::ndarray::ArrayD<T>,
            ranges: &[Vec<usize>],
            current_indices: &mut Vec<usize>,
            value: &T,
        ) {
            if current_indices.len() == ranges.len() {
                // We have a complete set of indices - assign the value
                arr[current_indices.as_slice()] = value.clone();
            } else {
                // Recurse through the next dimension
                let dim = current_indices.len();
                for &idx in &ranges[dim] {
                    current_indices.push(idx);
                    assign_recursive(arr, ranges, current_indices, value);
                    current_indices.pop();
                }
            }
        }

        let mut current_indices = Vec::new();
        assign_recursive(arr, ranges, &mut current_indices, &value);
    }

    // Helper method: Assign array values to indices specified by ranges
    fn assign_array_to_mixed_indices<T: Clone>(
        arr: &mut numpy::ndarray::ArrayD<T>,
        ranges: &[Vec<usize>],
        integer_axes: &[usize],
        source: &numpy::ndarray::ArrayD<T>,
    ) -> PyResult<()> {
        use numpy::ndarray::IxDyn;

        // Recursively iterate through all combinations of indices
        fn assign_array_recursive<T: Clone>(
            arr: &mut numpy::ndarray::ArrayD<T>,
            ranges: &[Vec<usize>],
            integer_axes: &[usize],
            source: &numpy::ndarray::ArrayD<T>,
            current_target_indices: &mut Vec<usize>,
            current_source_indices: &mut Vec<usize>,
        ) {
            if current_target_indices.len() == ranges.len() {
                // We have a complete set of indices - assign the value
                let target_idx = IxDyn(current_target_indices);
                let source_idx = IxDyn(current_source_indices);
                arr[target_idx] = source[source_idx].clone();
            } else {
                // Recurse through the next dimension
                let dim = current_target_indices.len();
                let is_integer_axis = integer_axes.contains(&dim);

                for (i, &idx) in ranges[dim].iter().enumerate() {
                    current_target_indices.push(idx);

                    // Only add to source indices if this is NOT an integer axis
                    // (integer axes reduce dimensionality)
                    if !is_integer_axis {
                        current_source_indices.push(i);
                    }

                    assign_array_recursive(
                        arr,
                        ranges,
                        integer_axes,
                        source,
                        current_target_indices,
                        current_source_indices,
                    );

                    if !is_integer_axis {
                        current_source_indices.pop();
                    }
                    current_target_indices.pop();
                }
            }
        }

        let mut current_target_indices = Vec::new();
        let mut current_source_indices = Vec::new();
        assign_array_recursive(
            arr,
            ranges,
            integer_axes,
            source,
            &mut current_target_indices,
            &mut current_source_indices,
        );
        Ok(())
    }
}
