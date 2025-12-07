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

use ndarray::{ArrayD, Axis, IxDyn, Slice};
use num_complex::{Complex32, Complex64};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt, PySequence, PySlice, PySliceIndices, PyTuple, PyType};

use crate::dtypes::DType;
use crate::pauli_bindings::{Pauli, PauliString};

/// Internal storage for array data
/// We use separate variants for each dtype to maintain type safety
#[derive(Clone)]
pub enum ArrayData {
    Bool(ArrayD<bool>),
    I8(ArrayD<i8>),
    I16(ArrayD<i16>),
    I32(ArrayD<i32>),
    I64(ArrayD<i64>),
    U8(ArrayD<u8>),
    U16(ArrayD<u16>),
    U32(ArrayD<u32>),
    U64(ArrayD<u64>),
    F32(ArrayD<f32>),
    F64(ArrayD<f64>),
    Complex64(ArrayD<num_complex::Complex<f32>>),
    Complex128(ArrayD<num_complex::Complex<f64>>),
    Pauli(ArrayD<Pauli>),
    PauliString(ArrayD<PauliString>),
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
            ArrayData::Bool(_) => DType::Bool,
            ArrayData::I8(_) => DType::I8,
            ArrayData::I16(_) => DType::I16,
            ArrayData::I32(_) => DType::I32,
            ArrayData::I64(_) => DType::I64,
            ArrayData::U8(_) => DType::U8,
            ArrayData::U16(_) => DType::U16,
            ArrayData::U32(_) => DType::U32,
            ArrayData::U64(_) => DType::U64,
            ArrayData::F32(_) => DType::F32,
            ArrayData::F64(_) => DType::F64,
            ArrayData::Complex64(_) => DType::Complex64,
            ArrayData::Complex128(_) => DType::Complex128,
            ArrayData::Pauli(_) => DType::Pauli,
            ArrayData::PauliString(_) => DType::PauliString,
        }
    }

    /// Get the shape of this array
    fn shape(&self) -> &[usize] {
        match self {
            ArrayData::Bool(arr) => arr.shape(),
            ArrayData::I8(arr) => arr.shape(),
            ArrayData::I16(arr) => arr.shape(),
            ArrayData::I32(arr) => arr.shape(),
            ArrayData::I64(arr) => arr.shape(),
            ArrayData::U8(arr) => arr.shape(),
            ArrayData::U16(arr) => arr.shape(),
            ArrayData::U32(arr) => arr.shape(),
            ArrayData::U64(arr) => arr.shape(),
            ArrayData::F32(arr) => arr.shape(),
            ArrayData::F64(arr) => arr.shape(),
            ArrayData::Complex64(arr) => arr.shape(),
            ArrayData::Complex128(arr) => arr.shape(),
            ArrayData::Pauli(arr) => arr.shape(),
            ArrayData::PauliString(arr) => arr.shape(),
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

/// Element type tracking for nested sequence parsing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ElemType {
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    Complex64,
    Complex128,
    Pauli,
    PauliString,
}

#[pymethods]
impl Array {
    /// Create a new `Array` from a numpy array or Python sequence
    ///
    /// Args:
    ///     data: A numpy array or Python sequence (list/tuple)
    ///     dtype: Optional dtype specification (`DType` enum or None for auto-detection)
    ///
    /// Returns:
    ///     A new `Array` wrapping the data
    #[new]
    #[pyo3(signature = (data, dtype=None))]
    fn py_new(data: &Bound<'_, PyAny>, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        Self::from_python_value(data, dtype)
    }

    /// Support Array[dtype] syntax for type hints.
    ///
    /// This is a classmethod that allows type hint syntax like:
    ///     Array[f64]  # Array with float64 dtype
    ///     Array[i32]  # Array with int32 dtype
    ///
    /// The dtype parameter is only for type checkers and has no runtime effect.
    /// This method returns the Array type itself.
    #[classmethod]
    fn __class_getitem__(cls: &Bound<'_, PyType>, _dtype_hint: &Bound<'_, PyAny>) -> Py<PyType> {
        cls.clone().unbind()
    }

    /// Get the shape of the array as a tuple
    #[getter]
    fn shape(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let shape_vec: Vec<usize> = self.data.shape().to_vec();
        Ok(PyTuple::new(py, &shape_vec)?.into())
    }

    /// Get the data type of the array
    #[getter]
    pub fn dtype(&self) -> DType {
        self.data.dtype()
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

    /// Create a deep copy of the array
    ///
    /// Returns:
    ///     A new `Array` with the same data as this array
    ///
    /// # Examples
    ///
    /// ```python
    /// from pecos_rslib import Array
    /// import numpy as np
    ///
    /// arr = Array(np.array([1.0, 2.0, 3.0]))
    /// arr_copy = arr.copy()
    /// arr_copy[0] = 99.0  # Modifying the copy doesn't affect the original
    /// ```
    pub fn copy(&self) -> Self {
        match &self.data {
            ArrayData::Bool(arr) => Self {
                data: ArrayData::Bool(arr.clone()),
            },
            ArrayData::I8(arr) => Self {
                data: ArrayData::I8(arr.clone()),
            },
            ArrayData::I16(arr) => Self {
                data: ArrayData::I16(arr.clone()),
            },
            ArrayData::I32(arr) => Self {
                data: ArrayData::I32(arr.clone()),
            },
            ArrayData::I64(arr) => Self {
                data: ArrayData::I64(arr.clone()),
            },
            ArrayData::U8(arr) => Self {
                data: ArrayData::U8(arr.clone()),
            },
            ArrayData::U16(arr) => Self {
                data: ArrayData::U16(arr.clone()),
            },
            ArrayData::U32(arr) => Self {
                data: ArrayData::U32(arr.clone()),
            },
            ArrayData::U64(arr) => Self {
                data: ArrayData::U64(arr.clone()),
            },
            ArrayData::F32(arr) => Self {
                data: ArrayData::F32(arr.clone()),
            },
            ArrayData::F64(arr) => Self {
                data: ArrayData::F64(arr.clone()),
            },
            ArrayData::Complex64(arr) => Self {
                data: ArrayData::Complex64(arr.clone()),
            },
            ArrayData::Complex128(arr) => Self {
                data: ArrayData::Complex128(arr.clone()),
            },
            ArrayData::Pauli(arr) => Self {
                data: ArrayData::Pauli(arr.clone()),
            },
            ArrayData::PauliString(arr) => Self {
                data: ArrayData::PauliString(arr.clone()),
            },
        }
    }

    /// Check if all elements in the array are True (for boolean arrays)
    /// or non-zero (for numeric arrays).
    ///
    /// Args:
    ///     axis: Ignored (for `NumPy` compatibility)
    ///     out: Ignored (for `NumPy` compatibility)
    ///     keepdims: Ignored (for `NumPy` compatibility)
    ///
    /// Returns:
    ///     bool: True if all elements are True/non-zero, False otherwise
    ///
    /// # Examples
    ///
    /// ```python
    /// from pecos.num import array
    ///
    /// arr = array([True, True, True])
    /// assert arr.all() == True
    ///
    /// arr2 = array([True, False, True])
    /// assert arr2.all() == False
    /// ```
    #[pyo3(signature = (axis=None, out=None, keepdims=None, **_kwargs))]
    #[allow(unused_variables)]
    pub fn all(
        &self,
        axis: Option<Py<PyAny>>,
        out: Option<Py<PyAny>>,
        keepdims: Option<bool>,
        _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> bool {
        match &self.data {
            ArrayData::Bool(arr) => arr.iter().all(|&x| x),
            ArrayData::I8(arr) => arr.iter().all(|&x| x != 0),
            ArrayData::I16(arr) => arr.iter().all(|&x| x != 0),
            ArrayData::I32(arr) => arr.iter().all(|&x| x != 0),
            ArrayData::I64(arr) => arr.iter().all(|&x| x != 0),
            ArrayData::U8(arr) => arr.iter().all(|&x| x != 0),
            ArrayData::U16(arr) => arr.iter().all(|&x| x != 0),
            ArrayData::U32(arr) => arr.iter().all(|&x| x != 0),
            ArrayData::U64(arr) => arr.iter().all(|&x| x != 0),
            ArrayData::F32(arr) => arr.iter().all(|&x| x != 0.0),
            ArrayData::F64(arr) => arr.iter().all(|&x| x != 0.0),
            ArrayData::Complex64(arr) => arr.iter().all(|&x| x.re != 0.0 || x.im != 0.0),
            ArrayData::Complex128(arr) => arr.iter().all(|&x| x.re != 0.0 || x.im != 0.0),
            ArrayData::Pauli(_) | ArrayData::PauliString(_) => {
                // Pauli arrays don't have a meaningful all() operation
                // We'll return true if there are any elements
                self.data.size() > 0
            }
        }
    }

    /// Convert array to a different dtype
    /// This is a pure Rust implementation that does NOT use `NumPy` internally
    pub fn astype(&self, target_dtype: DType) -> Self {
        use num_complex::Complex;

        // If already the target dtype, just clone
        if self.data.dtype() == target_dtype {
            return Self {
                data: self.data.clone(),
            };
        }

        match &self.data {
            ArrayData::Bool(arr) => match target_dtype {
                DType::Bool => Self {
                    data: ArrayData::Bool(arr.clone()),
                },
                DType::I8 => Self {
                    data: ArrayData::I8(arr.mapv(i8::from)),
                },
                DType::I16 => Self {
                    data: ArrayData::I16(arr.mapv(i16::from)),
                },
                DType::I32 => Self {
                    data: ArrayData::I32(arr.mapv(i32::from)),
                },
                DType::I64 => Self {
                    data: ArrayData::I64(arr.mapv(i64::from)),
                },
                DType::U8 => Self {
                    data: ArrayData::U8(arr.mapv(u8::from)),
                },
                DType::U16 => Self {
                    data: ArrayData::U16(arr.mapv(u16::from)),
                },
                DType::U32 => Self {
                    data: ArrayData::U32(arr.mapv(u32::from)),
                },
                DType::U64 => Self {
                    data: ArrayData::U64(arr.mapv(u64::from)),
                },
                DType::F32 => Self {
                    data: ArrayData::F32(arr.mapv(|x| if x { 1.0f32 } else { 0.0f32 })),
                },
                DType::F64 => Self {
                    data: ArrayData::F64(arr.mapv(|x| if x { 1.0f64 } else { 0.0f64 })),
                },
                DType::Complex64 => Self {
                    data: ArrayData::Complex64(
                        arr.mapv(|x| Complex::new(if x { 1.0f32 } else { 0.0f32 }, 0.0f32)),
                    ),
                },
                DType::Complex128 => Self {
                    data: ArrayData::Complex128(
                        arr.mapv(|x| Complex::new(if x { 1.0f64 } else { 0.0f64 }, 0.0f64)),
                    ),
                },
                DType::Pauli => panic!("Cannot convert to Pauli type"),
                DType::PauliString => panic!("Cannot convert to PauliString type"),
            },
            ArrayData::I8(arr) => match target_dtype {
                DType::Bool => Self {
                    data: ArrayData::Bool(arr.mapv(|x| x != 0)),
                },
                DType::I8 => Self {
                    data: ArrayData::I8(arr.clone()),
                },
                DType::I16 => Self {
                    data: ArrayData::I16(arr.mapv(i16::from)),
                },
                DType::I32 => Self {
                    data: ArrayData::I32(arr.mapv(i32::from)),
                },
                DType::I64 => Self {
                    data: ArrayData::I64(arr.mapv(i64::from)),
                },
                DType::U8 => Self {
                    data: ArrayData::U8(arr.mapv(|x| x as u8)),
                },
                DType::U16 => Self {
                    data: ArrayData::U16(arr.mapv(|x| x as u16)),
                },
                DType::U32 => Self {
                    data: ArrayData::U32(arr.mapv(|x| x as u32)),
                },
                DType::U64 => Self {
                    data: ArrayData::U64(arr.mapv(|x| x as u64)),
                },
                DType::F32 => Self {
                    data: ArrayData::F32(arr.mapv(f32::from)),
                },
                DType::F64 => Self {
                    data: ArrayData::F64(arr.mapv(f64::from)),
                },
                DType::Complex64 => Self {
                    data: ArrayData::Complex64(arr.mapv(|x| Complex::new(f32::from(x), 0.0f32))),
                },
                DType::Complex128 => Self {
                    data: ArrayData::Complex128(arr.mapv(|x| Complex::new(f64::from(x), 0.0f64))),
                },
                DType::Pauli => panic!("Cannot convert to Pauli type"),
                DType::PauliString => panic!("Cannot convert to PauliString type"),
            },
            ArrayData::I16(arr) => match target_dtype {
                DType::Bool => Self {
                    data: ArrayData::Bool(arr.mapv(|x| x != 0)),
                },
                DType::I8 => Self {
                    data: ArrayData::I8(arr.mapv(|x| x as i8)),
                },
                DType::I16 => Self {
                    data: ArrayData::I16(arr.clone()),
                },
                DType::I32 => Self {
                    data: ArrayData::I32(arr.mapv(i32::from)),
                },
                DType::I64 => Self {
                    data: ArrayData::I64(arr.mapv(i64::from)),
                },
                DType::U8 => Self {
                    data: ArrayData::U8(arr.mapv(|x| x as u8)),
                },
                DType::U16 => Self {
                    data: ArrayData::U16(arr.mapv(|x| x as u16)),
                },
                DType::U32 => Self {
                    data: ArrayData::U32(arr.mapv(|x| x as u32)),
                },
                DType::U64 => Self {
                    data: ArrayData::U64(arr.mapv(|x| x as u64)),
                },
                DType::F32 => Self {
                    data: ArrayData::F32(arr.mapv(f32::from)),
                },
                DType::F64 => Self {
                    data: ArrayData::F64(arr.mapv(f64::from)),
                },
                DType::Complex64 => Self {
                    data: ArrayData::Complex64(arr.mapv(|x| Complex::new(f32::from(x), 0.0f32))),
                },
                DType::Complex128 => Self {
                    data: ArrayData::Complex128(arr.mapv(|x| Complex::new(f64::from(x), 0.0f64))),
                },
                DType::Pauli => panic!("Cannot convert to Pauli type"),
                DType::PauliString => panic!("Cannot convert to PauliString type"),
            },
            ArrayData::I32(arr) => match target_dtype {
                DType::Bool => Self {
                    data: ArrayData::Bool(arr.mapv(|x| x != 0)),
                },
                DType::I8 => Self {
                    data: ArrayData::I8(arr.mapv(|x| x as i8)),
                },
                DType::I16 => Self {
                    data: ArrayData::I16(arr.mapv(|x| x as i16)),
                },
                DType::I32 => Self {
                    data: ArrayData::I32(arr.clone()),
                },
                DType::I64 => Self {
                    data: ArrayData::I64(arr.mapv(i64::from)),
                },
                DType::U8 => Self {
                    data: ArrayData::U8(arr.mapv(|x| x as u8)),
                },
                DType::U16 => Self {
                    data: ArrayData::U16(arr.mapv(|x| x as u16)),
                },
                DType::U32 => Self {
                    data: ArrayData::U32(arr.mapv(|x| x as u32)),
                },
                DType::U64 => Self {
                    data: ArrayData::U64(arr.mapv(|x| x as u64)),
                },
                DType::F32 => Self {
                    data: ArrayData::F32(arr.mapv(|x| x as f32)),
                },
                DType::F64 => Self {
                    data: ArrayData::F64(arr.mapv(f64::from)),
                },
                DType::Complex64 => Self {
                    data: ArrayData::Complex64(arr.mapv(|x| Complex::new(x as f32, 0.0f32))),
                },
                DType::Complex128 => Self {
                    data: ArrayData::Complex128(arr.mapv(|x| Complex::new(f64::from(x), 0.0f64))),
                },
                DType::Pauli => panic!("Cannot convert to Pauli type"),
                DType::PauliString => panic!("Cannot convert to PauliString type"),
            },
            ArrayData::I64(arr) => match target_dtype {
                DType::Bool => Self {
                    data: ArrayData::Bool(arr.mapv(|x| x != 0)),
                },
                DType::I8 => Self {
                    data: ArrayData::I8(arr.mapv(|x| x as i8)),
                },
                DType::I16 => Self {
                    data: ArrayData::I16(arr.mapv(|x| x as i16)),
                },
                DType::I32 => Self {
                    data: ArrayData::I32(arr.mapv(|x| x as i32)),
                },
                DType::I64 => Self {
                    data: ArrayData::I64(arr.clone()),
                },
                DType::U8 => Self {
                    data: ArrayData::U8(arr.mapv(|x| x as u8)),
                },
                DType::U16 => Self {
                    data: ArrayData::U16(arr.mapv(|x| x as u16)),
                },
                DType::U32 => Self {
                    data: ArrayData::U32(arr.mapv(|x| x as u32)),
                },
                DType::U64 => Self {
                    data: ArrayData::U64(arr.mapv(|x| x as u64)),
                },
                DType::F32 => Self {
                    data: ArrayData::F32(arr.mapv(|x| x as f32)),
                },
                DType::F64 => Self {
                    data: ArrayData::F64(arr.mapv(|x| x as f64)),
                },
                DType::Complex64 => Self {
                    data: ArrayData::Complex64(arr.mapv(|x| Complex::new(x as f32, 0.0f32))),
                },
                DType::Complex128 => Self {
                    data: ArrayData::Complex128(arr.mapv(|x| Complex::new(x as f64, 0.0f64))),
                },
                DType::Pauli => panic!("Cannot convert to Pauli type"),
                DType::PauliString => panic!("Cannot convert to PauliString type"),
            },
            ArrayData::U8(arr) => match target_dtype {
                DType::Bool => Self {
                    data: ArrayData::Bool(arr.mapv(|x| x != 0)),
                },
                DType::I8 => Self {
                    data: ArrayData::I8(arr.mapv(|x| x as i8)),
                },
                DType::I16 => Self {
                    data: ArrayData::I16(arr.mapv(i16::from)),
                },
                DType::I32 => Self {
                    data: ArrayData::I32(arr.mapv(i32::from)),
                },
                DType::I64 => Self {
                    data: ArrayData::I64(arr.mapv(i64::from)),
                },
                DType::U8 => Self {
                    data: ArrayData::U8(arr.clone()),
                },
                DType::U16 => Self {
                    data: ArrayData::U16(arr.mapv(u16::from)),
                },
                DType::U32 => Self {
                    data: ArrayData::U32(arr.mapv(u32::from)),
                },
                DType::U64 => Self {
                    data: ArrayData::U64(arr.mapv(u64::from)),
                },
                DType::F32 => Self {
                    data: ArrayData::F32(arr.mapv(f32::from)),
                },
                DType::F64 => Self {
                    data: ArrayData::F64(arr.mapv(f64::from)),
                },
                DType::Complex64 => Self {
                    data: ArrayData::Complex64(arr.mapv(|x| Complex::new(f32::from(x), 0.0f32))),
                },
                DType::Complex128 => Self {
                    data: ArrayData::Complex128(arr.mapv(|x| Complex::new(f64::from(x), 0.0f64))),
                },
                DType::Pauli => panic!("Cannot convert to Pauli type"),
                DType::PauliString => panic!("Cannot convert to PauliString type"),
            },
            ArrayData::U16(arr) => match target_dtype {
                DType::Bool => Self {
                    data: ArrayData::Bool(arr.mapv(|x| x != 0)),
                },
                DType::I8 => Self {
                    data: ArrayData::I8(arr.mapv(|x| x as i8)),
                },
                DType::I16 => Self {
                    data: ArrayData::I16(arr.mapv(|x| x as i16)),
                },
                DType::I32 => Self {
                    data: ArrayData::I32(arr.mapv(i32::from)),
                },
                DType::I64 => Self {
                    data: ArrayData::I64(arr.mapv(i64::from)),
                },
                DType::U8 => Self {
                    data: ArrayData::U8(arr.mapv(|x| x as u8)),
                },
                DType::U16 => Self {
                    data: ArrayData::U16(arr.clone()),
                },
                DType::U32 => Self {
                    data: ArrayData::U32(arr.mapv(u32::from)),
                },
                DType::U64 => Self {
                    data: ArrayData::U64(arr.mapv(u64::from)),
                },
                DType::F32 => Self {
                    data: ArrayData::F32(arr.mapv(f32::from)),
                },
                DType::F64 => Self {
                    data: ArrayData::F64(arr.mapv(f64::from)),
                },
                DType::Complex64 => Self {
                    data: ArrayData::Complex64(arr.mapv(|x| Complex::new(f32::from(x), 0.0f32))),
                },
                DType::Complex128 => Self {
                    data: ArrayData::Complex128(arr.mapv(|x| Complex::new(f64::from(x), 0.0f64))),
                },
                DType::Pauli => panic!("Cannot convert to Pauli type"),
                DType::PauliString => panic!("Cannot convert to PauliString type"),
            },
            ArrayData::U32(arr) => match target_dtype {
                DType::Bool => Self {
                    data: ArrayData::Bool(arr.mapv(|x| x != 0)),
                },
                DType::I8 => Self {
                    data: ArrayData::I8(arr.mapv(|x| x as i8)),
                },
                DType::I16 => Self {
                    data: ArrayData::I16(arr.mapv(|x| x as i16)),
                },
                DType::I32 => Self {
                    data: ArrayData::I32(arr.mapv(|x| x as i32)),
                },
                DType::I64 => Self {
                    data: ArrayData::I64(arr.mapv(i64::from)),
                },
                DType::U8 => Self {
                    data: ArrayData::U8(arr.mapv(|x| x as u8)),
                },
                DType::U16 => Self {
                    data: ArrayData::U16(arr.mapv(|x| x as u16)),
                },
                DType::U32 => Self {
                    data: ArrayData::U32(arr.clone()),
                },
                DType::U64 => Self {
                    data: ArrayData::U64(arr.mapv(u64::from)),
                },
                DType::F32 => Self {
                    data: ArrayData::F32(arr.mapv(|x| x as f32)),
                },
                DType::F64 => Self {
                    data: ArrayData::F64(arr.mapv(f64::from)),
                },
                DType::Complex64 => Self {
                    data: ArrayData::Complex64(arr.mapv(|x| Complex::new(x as f32, 0.0f32))),
                },
                DType::Complex128 => Self {
                    data: ArrayData::Complex128(arr.mapv(|x| Complex::new(f64::from(x), 0.0f64))),
                },
                DType::Pauli => panic!("Cannot convert to Pauli type"),
                DType::PauliString => panic!("Cannot convert to PauliString type"),
            },
            ArrayData::U64(arr) => match target_dtype {
                DType::Bool => Self {
                    data: ArrayData::Bool(arr.mapv(|x| x != 0)),
                },
                DType::I8 => Self {
                    data: ArrayData::I8(arr.mapv(|x| x as i8)),
                },
                DType::I16 => Self {
                    data: ArrayData::I16(arr.mapv(|x| x as i16)),
                },
                DType::I32 => Self {
                    data: ArrayData::I32(arr.mapv(|x| x as i32)),
                },
                DType::I64 => Self {
                    data: ArrayData::I64(arr.mapv(|x| x as i64)),
                },
                DType::U8 => Self {
                    data: ArrayData::U8(arr.mapv(|x| x as u8)),
                },
                DType::U16 => Self {
                    data: ArrayData::U16(arr.mapv(|x| x as u16)),
                },
                DType::U32 => Self {
                    data: ArrayData::U32(arr.mapv(|x| x as u32)),
                },
                DType::U64 => Self {
                    data: ArrayData::U64(arr.clone()),
                },
                DType::F32 => Self {
                    data: ArrayData::F32(arr.mapv(|x| x as f32)),
                },
                DType::F64 => Self {
                    data: ArrayData::F64(arr.mapv(|x| x as f64)),
                },
                DType::Complex64 => Self {
                    data: ArrayData::Complex64(arr.mapv(|x| Complex::new(x as f32, 0.0f32))),
                },
                DType::Complex128 => Self {
                    data: ArrayData::Complex128(arr.mapv(|x| Complex::new(x as f64, 0.0f64))),
                },
                DType::Pauli => panic!("Cannot convert to Pauli type"),
                DType::PauliString => panic!("Cannot convert to PauliString type"),
            },
            ArrayData::F32(arr) => match target_dtype {
                DType::Bool => Self {
                    data: ArrayData::Bool(arr.mapv(|x| x != 0.0)),
                },
                DType::I8 => Self {
                    data: ArrayData::I8(arr.mapv(|x| x as i8)),
                },
                DType::I16 => Self {
                    data: ArrayData::I16(arr.mapv(|x| x as i16)),
                },
                DType::I32 => Self {
                    data: ArrayData::I32(arr.mapv(|x| x as i32)),
                },
                DType::I64 => Self {
                    data: ArrayData::I64(arr.mapv(|x| x as i64)),
                },
                DType::U8 => Self {
                    data: ArrayData::U8(arr.mapv(|x| x as u8)),
                },
                DType::U16 => Self {
                    data: ArrayData::U16(arr.mapv(|x| x as u16)),
                },
                DType::U32 => Self {
                    data: ArrayData::U32(arr.mapv(|x| x as u32)),
                },
                DType::U64 => Self {
                    data: ArrayData::U64(arr.mapv(|x| x as u64)),
                },
                DType::F32 => Self {
                    data: ArrayData::F32(arr.clone()),
                },
                DType::F64 => Self {
                    data: ArrayData::F64(arr.mapv(f64::from)),
                },
                DType::Complex64 => Self {
                    data: ArrayData::Complex64(arr.mapv(|x| Complex::new(x, 0.0f32))),
                },
                DType::Complex128 => Self {
                    data: ArrayData::Complex128(arr.mapv(|x| Complex::new(f64::from(x), 0.0f64))),
                },
                DType::Pauli => panic!("Cannot convert to Pauli type"),
                DType::PauliString => panic!("Cannot convert to PauliString type"),
            },
            ArrayData::F64(arr) => match target_dtype {
                DType::Bool => Self {
                    data: ArrayData::Bool(arr.mapv(|x| x != 0.0)),
                },
                DType::I8 => Self {
                    data: ArrayData::I8(arr.mapv(|x| x as i8)),
                },
                DType::I16 => Self {
                    data: ArrayData::I16(arr.mapv(|x| x as i16)),
                },
                DType::I32 => Self {
                    data: ArrayData::I32(arr.mapv(|x| x as i32)),
                },
                DType::I64 => Self {
                    data: ArrayData::I64(arr.mapv(|x| x as i64)),
                },
                DType::U8 => Self {
                    data: ArrayData::U8(arr.mapv(|x| x as u8)),
                },
                DType::U16 => Self {
                    data: ArrayData::U16(arr.mapv(|x| x as u16)),
                },
                DType::U32 => Self {
                    data: ArrayData::U32(arr.mapv(|x| x as u32)),
                },
                DType::U64 => Self {
                    data: ArrayData::U64(arr.mapv(|x| x as u64)),
                },
                DType::F32 => Self {
                    data: ArrayData::F32(arr.mapv(|x| x as f32)),
                },
                DType::F64 => Self {
                    data: ArrayData::F64(arr.clone()),
                },
                DType::Complex64 => Self {
                    data: ArrayData::Complex64(arr.mapv(|x| Complex::new(x as f32, 0.0f32))),
                },
                DType::Complex128 => Self {
                    data: ArrayData::Complex128(arr.mapv(|x| Complex::new(x, 0.0f64))),
                },
                DType::Pauli => panic!("Cannot convert to Pauli type"),
                DType::PauliString => panic!("Cannot convert to PauliString type"),
            },
            ArrayData::Complex64(arr) => match target_dtype {
                DType::Bool => Self {
                    data: ArrayData::Bool(arr.mapv(|x| x.re != 0.0)),
                },
                DType::I8 => Self {
                    data: ArrayData::I8(arr.mapv(|x| x.re as i8)),
                },
                DType::I16 => Self {
                    data: ArrayData::I16(arr.mapv(|x| x.re as i16)),
                },
                DType::I32 => Self {
                    data: ArrayData::I32(arr.mapv(|x| x.re as i32)),
                },
                DType::I64 => Self {
                    data: ArrayData::I64(arr.mapv(|x| x.re as i64)),
                },
                DType::U8 => Self {
                    data: ArrayData::U8(arr.mapv(|x| x.re as u8)),
                },
                DType::U16 => Self {
                    data: ArrayData::U16(arr.mapv(|x| x.re as u16)),
                },
                DType::U32 => Self {
                    data: ArrayData::U32(arr.mapv(|x| x.re as u32)),
                },
                DType::U64 => Self {
                    data: ArrayData::U64(arr.mapv(|x| x.re as u64)),
                },
                DType::F32 => Self {
                    data: ArrayData::F32(arr.mapv(|x| x.re)),
                },
                DType::F64 => Self {
                    data: ArrayData::F64(arr.mapv(|x| f64::from(x.re))),
                },
                DType::Complex64 => Self {
                    data: ArrayData::Complex64(arr.clone()),
                },
                DType::Complex128 => Self {
                    data: ArrayData::Complex128(
                        arr.mapv(|x| Complex::new(f64::from(x.re), f64::from(x.im))),
                    ),
                },
                DType::Pauli => panic!("Cannot convert to Pauli type"),
                DType::PauliString => panic!("Cannot convert to PauliString type"),
            },
            ArrayData::Complex128(arr) => match target_dtype {
                DType::Bool => Self {
                    data: ArrayData::Bool(arr.mapv(|x| x.re != 0.0)),
                },
                DType::I8 => Self {
                    data: ArrayData::I8(arr.mapv(|x| x.re as i8)),
                },
                DType::I16 => Self {
                    data: ArrayData::I16(arr.mapv(|x| x.re as i16)),
                },
                DType::I32 => Self {
                    data: ArrayData::I32(arr.mapv(|x| x.re as i32)),
                },
                DType::I64 => Self {
                    data: ArrayData::I64(arr.mapv(|x| x.re as i64)),
                },
                DType::U8 => Self {
                    data: ArrayData::U8(arr.mapv(|x| x.re as u8)),
                },
                DType::U16 => Self {
                    data: ArrayData::U16(arr.mapv(|x| x.re as u16)),
                },
                DType::U32 => Self {
                    data: ArrayData::U32(arr.mapv(|x| x.re as u32)),
                },
                DType::U64 => Self {
                    data: ArrayData::U64(arr.mapv(|x| x.re as u64)),
                },
                DType::F32 => Self {
                    data: ArrayData::F32(arr.mapv(|x| x.re as f32)),
                },
                DType::F64 => Self {
                    data: ArrayData::F64(arr.mapv(|x| x.re)),
                },
                DType::Complex64 => Self {
                    data: ArrayData::Complex64(
                        arr.mapv(|x| Complex::new(x.re as f32, x.im as f32)),
                    ),
                },
                DType::Complex128 => Self {
                    data: ArrayData::Complex128(arr.clone()),
                },
                DType::Pauli => panic!("Cannot convert to Pauli type"),
                DType::PauliString => panic!("Cannot convert to PauliString type"),
            },
            ArrayData::Pauli(arr) => match target_dtype {
                DType::Pauli => Self {
                    data: ArrayData::Pauli(arr.clone()),
                },
                _ => panic!("Cannot convert Pauli array to numeric type"),
            },
            ArrayData::PauliString(arr) => match target_dtype {
                DType::PauliString => Self {
                    data: ArrayData::PauliString(arr.clone()),
                },
                _ => panic!("Cannot convert PauliString array to numeric type"),
            },
        }
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

    /// Implement __`array_interface`__ property for `NumPy` compatibility
    /// This allows `NumPy` to consume our arrays via zero-copy protocol
    #[getter]
    fn __array_interface__(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        use pyo3::types::PyDict;

        let dict = PyDict::new(py);

        // Set shape (must be a tuple for NumPy)
        let shape: Vec<usize> = self.data.shape().to_vec();
        let shape_tuple = pyo3::types::PyTuple::new(py, &shape)?;
        dict.set_item("shape", shape_tuple)?;

        // Set typestr and data pointer based on the dtype
        match &self.data {
            ArrayData::Bool(arr) => {
                dict.set_item("typestr", "|b1")?;
                dict.set_item("data", (arr.as_ptr() as usize, false))?;
                let strides: Vec<isize> = arr
                    .strides()
                    .iter()
                    .map(|&s| s * std::mem::size_of::<bool>() as isize)
                    .collect();
                let strides_tuple = pyo3::types::PyTuple::new(py, &strides)?;
                dict.set_item("strides", strides_tuple)?;
            }
            ArrayData::I8(arr) => {
                dict.set_item("typestr", "i1")?;
                dict.set_item("data", (arr.as_ptr() as usize, false))?;
                let strides: Vec<isize> = arr
                    .strides()
                    .iter()
                    .map(|&s| s * std::mem::size_of::<i8>() as isize)
                    .collect();
                let strides_tuple = pyo3::types::PyTuple::new(py, &strides)?;
                dict.set_item("strides", strides_tuple)?;
            }
            ArrayData::I16(arr) => {
                dict.set_item("typestr", "<i2")?;
                dict.set_item("data", (arr.as_ptr() as usize, false))?;
                let strides: Vec<isize> = arr
                    .strides()
                    .iter()
                    .map(|&s| s * std::mem::size_of::<i16>() as isize)
                    .collect();
                let strides_tuple = pyo3::types::PyTuple::new(py, &strides)?;
                dict.set_item("strides", strides_tuple)?;
            }
            ArrayData::I32(arr) => {
                dict.set_item("typestr", "<i4")?;
                dict.set_item("data", (arr.as_ptr() as usize, false))?;
                let strides: Vec<isize> = arr
                    .strides()
                    .iter()
                    .map(|&s| s * std::mem::size_of::<i32>() as isize)
                    .collect();
                let strides_tuple = pyo3::types::PyTuple::new(py, &strides)?;
                dict.set_item("strides", strides_tuple)?;
            }
            ArrayData::I64(arr) => {
                dict.set_item("typestr", "<i8")?;
                dict.set_item("data", (arr.as_ptr() as usize, false))?;
                let strides: Vec<isize> = arr
                    .strides()
                    .iter()
                    .map(|&s| s * std::mem::size_of::<i64>() as isize)
                    .collect();
                let strides_tuple = pyo3::types::PyTuple::new(py, &strides)?;
                dict.set_item("strides", strides_tuple)?;
            }
            ArrayData::U8(arr) => {
                dict.set_item("typestr", "u1")?;
                dict.set_item("data", (arr.as_ptr() as usize, false))?;
                let strides: Vec<isize> = arr
                    .strides()
                    .iter()
                    .map(|&s| s * std::mem::size_of::<u8>() as isize)
                    .collect();
                let strides_tuple = pyo3::types::PyTuple::new(py, &strides)?;
                dict.set_item("strides", strides_tuple)?;
            }
            ArrayData::U16(arr) => {
                dict.set_item("typestr", "<u2")?;
                dict.set_item("data", (arr.as_ptr() as usize, false))?;
                let strides: Vec<isize> = arr
                    .strides()
                    .iter()
                    .map(|&s| s * std::mem::size_of::<u16>() as isize)
                    .collect();
                let strides_tuple = pyo3::types::PyTuple::new(py, &strides)?;
                dict.set_item("strides", strides_tuple)?;
            }
            ArrayData::U32(arr) => {
                dict.set_item("typestr", "<u4")?;
                dict.set_item("data", (arr.as_ptr() as usize, false))?;
                let strides: Vec<isize> = arr
                    .strides()
                    .iter()
                    .map(|&s| s * std::mem::size_of::<u32>() as isize)
                    .collect();
                let strides_tuple = pyo3::types::PyTuple::new(py, &strides)?;
                dict.set_item("strides", strides_tuple)?;
            }
            ArrayData::U64(arr) => {
                dict.set_item("typestr", "<u8")?;
                dict.set_item("data", (arr.as_ptr() as usize, false))?;
                let strides: Vec<isize> = arr
                    .strides()
                    .iter()
                    .map(|&s| s * std::mem::size_of::<u64>() as isize)
                    .collect();
                let strides_tuple = pyo3::types::PyTuple::new(py, &strides)?;
                dict.set_item("strides", strides_tuple)?;
            }
            ArrayData::F32(arr) => {
                dict.set_item("typestr", "<f4")?;
                dict.set_item("data", (arr.as_ptr() as usize, false))?;
                let strides: Vec<isize> = arr
                    .strides()
                    .iter()
                    .map(|&s| s * std::mem::size_of::<f32>() as isize)
                    .collect();
                let strides_tuple = pyo3::types::PyTuple::new(py, &strides)?;
                dict.set_item("strides", strides_tuple)?;
            }
            ArrayData::F64(arr) => {
                dict.set_item("typestr", "<f8")?;
                dict.set_item("data", (arr.as_ptr() as usize, false))?;
                let strides: Vec<isize> = arr
                    .strides()
                    .iter()
                    .map(|&s| s * std::mem::size_of::<f64>() as isize)
                    .collect();
                let strides_tuple = pyo3::types::PyTuple::new(py, &strides)?;
                dict.set_item("strides", strides_tuple)?;
            }
            ArrayData::Complex64(arr) => {
                dict.set_item("typestr", "<c8")?;
                dict.set_item("data", (arr.as_ptr() as usize, false))?;
                let strides: Vec<isize> = arr
                    .strides()
                    .iter()
                    .map(|&s| s * std::mem::size_of::<num_complex::Complex32>() as isize)
                    .collect();
                let strides_tuple = pyo3::types::PyTuple::new(py, &strides)?;
                dict.set_item("strides", strides_tuple)?;
            }
            ArrayData::Complex128(arr) => {
                dict.set_item("typestr", "<c16")?;
                dict.set_item("data", (arr.as_ptr() as usize, false))?;
                let strides: Vec<isize> = arr
                    .strides()
                    .iter()
                    .map(|&s| s * std::mem::size_of::<num_complex::Complex64>() as isize)
                    .collect();
                let strides_tuple = pyo3::types::PyTuple::new(py, &strides)?;
                dict.set_item("strides", strides_tuple)?;
            }
            ArrayData::Pauli(_) | ArrayData::PauliString(_) => {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Pauli and PauliString arrays cannot be converted to NumPy via __array_interface__ (use __array__() method instead)",
                ));
            }
        }

        // Set protocol version
        dict.set_item("version", 3)?;

        Ok(dict.into())
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
        } else if let Ok(idx) = index.extract::<isize>() {
            // Integer indexing: arr[i] = value
            let shape = self.data.shape();

            // Only 1D arrays support integer indexing with a single integer
            if shape.len() != 1 {
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                    "Single integer indexing assignment only works on 1D arrays (use tuple indexing for multi-dimensional arrays, e.g., arr[i, j] = value)",
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

            let idx_usize = normalized_idx as usize;

            // Assign the value based on array dtype
            match &mut self.data {
                ArrayData::Bool(arr) => {
                    let val: bool = value.extract()?;
                    arr[idx_usize] = val;
                }
                ArrayData::I8(arr) => {
                    let val: i8 = value.extract()?;
                    arr[idx_usize] = val;
                }
                ArrayData::I16(arr) => {
                    let val: i16 = value.extract()?;
                    arr[idx_usize] = val;
                }
                ArrayData::I32(arr) => {
                    let val: i32 = value.extract()?;
                    arr[idx_usize] = val;
                }
                ArrayData::I64(arr) => {
                    let val: i64 = value.extract()?;
                    arr[idx_usize] = val;
                }
                ArrayData::U8(arr) => {
                    let val: u8 = value.extract()?;
                    arr[idx_usize] = val;
                }
                ArrayData::U16(arr) => {
                    let val: u16 = value.extract()?;
                    arr[idx_usize] = val;
                }
                ArrayData::U32(arr) => {
                    let val: u32 = value.extract()?;
                    arr[idx_usize] = val;
                }
                ArrayData::U64(arr) => {
                    let val: u64 = value.extract()?;
                    arr[idx_usize] = val;
                }
                ArrayData::F32(arr) => {
                    let val: f32 = value.extract()?;
                    arr[idx_usize] = val;
                }
                ArrayData::F64(arr) => {
                    let val: f64 = value.extract()?;
                    arr[idx_usize] = val;
                }
                ArrayData::Complex64(arr) => {
                    let val: Complex32 = value.extract()?;
                    arr[idx_usize] = val;
                }
                ArrayData::Complex128(arr) => {
                    let val: Complex64 = value.extract()?;
                    arr[idx_usize] = val;
                }
                ArrayData::Pauli(arr) => {
                    let val: crate::pauli_bindings::Pauli = value.extract()?;
                    arr[idx_usize] = val;
                }
                ArrayData::PauliString(arr) => {
                    let val: crate::pauli_bindings::PauliString = value.extract()?;
                    arr[idx_usize] = val;
                }
            }
            Ok(())
        } else {
            // Unsupported index type
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Index must be an integer, slice, or tuple",
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

            // If result is 0-dimensional (scalar), extract the value instead of returning Array
            if result.data.shape().is_empty() {
                return result.extract_scalar(py);
            }

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
            // For multi-dimensional arrays, this selects along the first axis (like NumPy)
            let shape = self.data.shape();

            // Normalize negative indices
            let size = shape[0] as isize;
            let normalized_idx = if idx < 0 { size + idx } else { idx };

            // Bounds checking
            if normalized_idx < 0 || normalized_idx >= size {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "Index {idx} is out of bounds for array of size {size}"
                )));
            }

            // Use apply_mixed_indexing with a single integer index
            // This handles both 1D (returns scalar) and multi-D (returns sub-array) cases
            let index_ops = vec![IndexOp::Integer(normalized_idx)];
            let result = self.apply_mixed_indexing(&index_ops)?;

            // If result is 0-dimensional (scalar), extract the value instead of returning Array
            if result.data.shape().is_empty() {
                return result.extract_scalar(py);
            }

            Ok(Py::new(py, result)?.into_any())
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

    /// Power: self ** other
    fn __pow__(
        &self,
        other: &Bound<'_, PyAny>,
        _modulo: Option<&Bound<'_, PyAny>>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        self.binary_op(other, py, f64::powf, "power")
    }

    /// Reverse power: other ** self
    fn __rpow__(
        &self,
        other: &Bound<'_, PyAny>,
        _modulo: Option<&Bound<'_, PyAny>>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        self.binary_op_reverse(other, py, f64::powf, "power")
    }

    /// Absolute value: abs(self)
    fn __abs__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        use num_complex::ComplexFloat;

        match &self.data {
            ArrayData::Bool(_) => Err(pyo3::exceptions::PyTypeError::new_err(
                "abs() operation not supported on boolean arrays",
            )),
            ArrayData::F64(arr) => {
                let result = arr.abs();
                Ok(Py::new(
                    py,
                    Array {
                        data: ArrayData::F64(result),
                    },
                )?
                .into_any())
            }
            ArrayData::F32(arr) => {
                // Convert to f64 for consistency
                let result = arr.mapv(|v| f64::from(v.abs()));
                Ok(Py::new(
                    py,
                    Array {
                        data: ArrayData::F64(result),
                    },
                )?
                .into_any())
            }
            ArrayData::I8(arr) => {
                let result = arr.mapv(|v| f64::from(v.abs()));
                Ok(Py::new(
                    py,
                    Array {
                        data: ArrayData::F64(result),
                    },
                )?
                .into_any())
            }
            ArrayData::I16(arr) => {
                let result = arr.mapv(|v| f64::from(v.abs()));
                Ok(Py::new(
                    py,
                    Array {
                        data: ArrayData::F64(result),
                    },
                )?
                .into_any())
            }
            ArrayData::I32(arr) => {
                let result = arr.mapv(|v| f64::from(v.abs()));
                Ok(Py::new(
                    py,
                    Array {
                        data: ArrayData::F64(result),
                    },
                )?
                .into_any())
            }
            ArrayData::I64(arr) => {
                #[allow(clippy::cast_precision_loss)]
                let result = arr.mapv(|v| v.abs() as f64);
                Ok(Py::new(
                    py,
                    Array {
                        data: ArrayData::F64(result),
                    },
                )?
                .into_any())
            }
            ArrayData::U8(arr) => {
                let result = arr.mapv(f64::from);
                Ok(Py::new(
                    py,
                    Array {
                        data: ArrayData::F64(result),
                    },
                )?
                .into_any())
            }
            ArrayData::U16(arr) => {
                let result = arr.mapv(f64::from);
                Ok(Py::new(
                    py,
                    Array {
                        data: ArrayData::F64(result),
                    },
                )?
                .into_any())
            }
            ArrayData::U32(arr) => {
                let result = arr.mapv(f64::from);
                Ok(Py::new(
                    py,
                    Array {
                        data: ArrayData::F64(result),
                    },
                )?
                .into_any())
            }
            ArrayData::U64(arr) => {
                #[allow(clippy::cast_precision_loss)]
                let result = arr.mapv(|v| v as f64);
                Ok(Py::new(
                    py,
                    Array {
                        data: ArrayData::F64(result),
                    },
                )?
                .into_any())
            }
            ArrayData::Complex64(arr) => {
                let result = arr.mapv(|v| f64::from(v.abs()));
                Ok(Py::new(
                    py,
                    Array {
                        data: ArrayData::F64(result),
                    },
                )?
                .into_any())
            }
            ArrayData::Complex128(arr) => {
                let result = arr.mapv(num_complex::ComplexFloat::abs);
                Ok(Py::new(
                    py,
                    Array {
                        data: ArrayData::F64(result),
                    },
                )?
                .into_any())
            }
            ArrayData::Pauli(_) => Err(pyo3::exceptions::PyTypeError::new_err(
                "abs() operation not supported on Pauli arrays",
            )),
            ArrayData::PauliString(_) => Err(pyo3::exceptions::PyTypeError::new_err(
                "abs() operation not supported on PauliString arrays",
            )),
        }
    }

    /// Greater than: self > other
    fn __gt__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.comparison_op(
            other,
            py,
            |a, b| if a > b { 1.0 } else { 0.0 },
            "greater than",
        )
    }

    /// Greater than or equal: self >= other
    fn __ge__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.comparison_op(
            other,
            py,
            |a, b| if a >= b { 1.0 } else { 0.0 },
            "greater than or equal",
        )
    }

    /// Less than: self < other
    fn __lt__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.comparison_op(other, py, |a, b| if a < b { 1.0 } else { 0.0 }, "less than")
    }

    /// Less than or equal: self <= other
    fn __le__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.comparison_op(
            other,
            py,
            |a, b| if a <= b { 1.0 } else { 0.0 },
            "less than or equal",
        )
    }

    /// Equal: self == other
    /// Note: Uses exact float equality to match numpy behavior
    #[allow(clippy::float_cmp)]
    fn __eq__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.comparison_op(other, py, |a, b| if a == b { 1.0 } else { 0.0 }, "equal")
    }

    /// Not equal: self != other
    /// Note: Uses exact float equality to match numpy behavior
    #[allow(clippy::float_cmp)]
    fn __ne__(&self, other: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.comparison_op(
            other,
            py,
            |a, b| if a == b { 0.0 } else { 1.0 },
            "not equal",
        )
    }
}

impl Array {
    /// Create a new `Array` from `ArrayData`
    pub fn new(data: ArrayData) -> Self {
        Self { data }
    }

    /// Create an Array from Python value (`NumPy` array or sequence)
    pub fn from_python_value(
        data: &Bound<'_, PyAny>,
        dtype: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        use pyo3::types::PySequence;

        // First check if it's already an Array object
        if let Ok(arr) = data.extract::<PyRef<Array>>() {
            // If dtype is specified and different, convert; otherwise just copy
            if let Some(dt) = dtype {
                let target_dtype = Self::parse_dtype(dt)?;
                let target_dtype_obj = Self::elemtype_to_dtype(target_dtype)?;
                return Ok(arr.astype(target_dtype_obj));
            }
            return Ok(arr.copy());
        }

        // Then try NumPy array directly (for compatibility with existing NumPy arrays)
        if let Ok(arr) = Self::try_from_numpy(data) {
            return Ok(arr);
        }

        // Finally try Python sequence (list/tuple) - parse using pure Rust
        if let Ok(_seq) = data.cast::<PySequence>() {
            return Self::from_nested_sequence(data, dtype);
        }

        Err(pyo3::exceptions::PyTypeError::new_err(
            "Input must be a numpy array, Array, or Python sequence (list/tuple)",
        ))
    }

    /// Parse dtype from Python (string, `DType` object, or scalar class) to `ElemType`
    fn parse_dtype(dtype: &Bound<'_, PyAny>) -> PyResult<ElemType> {
        use crate::dtypes::DType;

        // Try to extract as string first
        if let Ok(s) = dtype.extract::<String>() {
            let dtype_obj = DType::from_str(&s)?;
            return Self::dtype_to_elemtype(dtype_obj);
        }

        // Try to extract as DType object
        if let Ok(dtype_obj) = dtype.extract::<DType>() {
            return Self::dtype_to_elemtype(dtype_obj);
        }

        // Try to match scalar class types (NumPy compatibility)
        // Check if it's a Python type/class by checking for __name__ attribute
        if let Ok(type_obj) = dtype.cast::<pyo3::types::PyType>()
            && let Ok(name) = type_obj.name()
        {
            let name_str = name.to_string();
            // Match on the scalar class names
            let dtype_obj = match name_str.as_str() {
                "i8" | "int8" => DType::I8,
                "i16" | "int16" => DType::I16,
                "i32" | "int32" => DType::I32,
                "i64" | "int64" | "int" => DType::I64, // Python's int -> i64
                "u8" | "uint8" => DType::U8,
                "u16" | "uint16" => DType::U16,
                "u32" | "uint32" => DType::U32,
                "u64" | "uint64" => DType::U64,
                "f32" | "float32" => DType::F32,
                "f64" | "float64" | "float" => DType::F64, // Python's float -> f64
                "complex64" => DType::Complex64,
                "complex128" | "complex" => DType::Complex128, // Python's complex -> complex128
                "bool" => DType::Bool,
                _ => {
                    return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                        "Unknown scalar type: {name_str}"
                    )));
                }
            };
            return Self::dtype_to_elemtype(dtype_obj);
        }

        Err(pyo3::exceptions::PyTypeError::new_err(
            "dtype must be a string, DType object, or scalar class (e.g., i64, f64)",
        ))
    }

    /// Convert `DType` to `ElemType`
    fn dtype_to_elemtype(dtype: DType) -> PyResult<ElemType> {
        use crate::dtypes::DType;

        match dtype {
            DType::Bool => Ok(ElemType::Bool),
            DType::I8 => Ok(ElemType::I8),
            DType::I16 => Ok(ElemType::I16),
            DType::I32 => Ok(ElemType::I32),
            DType::I64 => Ok(ElemType::I64),
            DType::U8 => Ok(ElemType::U8),
            DType::U16 => Ok(ElemType::U16),
            DType::U32 => Ok(ElemType::U32),
            DType::U64 => Ok(ElemType::U64),
            DType::F32 => Ok(ElemType::F32),
            DType::F64 => Ok(ElemType::F64),
            DType::Complex64 => Ok(ElemType::Complex64),
            DType::Complex128 => Ok(ElemType::Complex128),
            DType::Pauli => Ok(ElemType::Pauli),
            DType::PauliString => Ok(ElemType::PauliString),
        }
    }

    /// Convert `ElemType` to `DType`
    fn elemtype_to_dtype(elemtype: ElemType) -> PyResult<DType> {
        use crate::dtypes::DType;

        match elemtype {
            ElemType::Bool => Ok(DType::Bool),
            ElemType::I8 => Ok(DType::I8),
            ElemType::I16 => Ok(DType::I16),
            ElemType::I32 => Ok(DType::I32),
            ElemType::I64 => Ok(DType::I64),
            ElemType::U8 => Ok(DType::U8),
            ElemType::U16 => Ok(DType::U16),
            ElemType::U32 => Ok(DType::U32),
            ElemType::U64 => Ok(DType::U64),
            ElemType::F32 => Ok(DType::F32),
            ElemType::F64 => Ok(DType::F64),
            ElemType::Complex64 => Ok(DType::Complex64),
            ElemType::Complex128 => Ok(DType::Complex128),
            ElemType::Pauli => Ok(DType::Pauli),
            ElemType::PauliString => Ok(DType::PauliString),
        }
    }

    /// Parse nested Python sequences (lists/tuples) into Array - pure Rust implementation
    fn from_nested_sequence(
        data: &Bound<'_, PyAny>,
        dtype: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        // Determine shape and element type
        let shape = Self::infer_shape(data)?;
        let ndim = shape.len();

        if ndim == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Cannot create array from empty sequence",
            ));
        }

        // Parse dtype if provided, otherwise auto-detect
        let mut elem_type = if let Some(dt) = dtype {
            Self::parse_dtype(dt)?
        } else {
            // Use Int64 as default for auto-detection, will promote to float/complex if needed
            ElemType::I64
        };

        // Flatten and collect all elements
        let mut flat_f64: Vec<f64> = Vec::new();
        let mut flat_complex: Vec<num_complex::Complex<f64>> = Vec::new();
        let mut flat_pauli: Vec<Pauli> = Vec::new();
        let mut flat_paulistring: Vec<PauliString> = Vec::new();
        let mut flat_bool: Vec<bool> = Vec::new();
        let mut flat_i64: Vec<i64> = Vec::new();

        Self::flatten_sequence(
            data,
            &mut flat_f64,
            &mut flat_complex,
            &mut flat_pauli,
            &mut flat_paulistring,
            &mut flat_bool,
            &mut flat_i64,
            &mut elem_type,
            dtype.is_some(), // explicit_dtype flag
        )?;

        // Create ndarray with the inferred shape
        match elem_type {
            ElemType::Bool => {
                let arr = ArrayD::from_shape_vec(shape, flat_bool).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}"))
                })?;
                Ok(Self {
                    data: ArrayData::Bool(arr),
                })
            }
            ElemType::I8 => {
                // Convert i64 to i8
                let flat_i8: Vec<i8> = flat_i64.iter().map(|&x| x as i8).collect();
                let arr = ArrayD::from_shape_vec(shape, flat_i8).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}"))
                })?;
                Ok(Self {
                    data: ArrayData::I8(arr),
                })
            }
            ElemType::I16 => {
                // Convert i64 to i16
                let flat_i16: Vec<i16> = flat_i64.iter().map(|&x| x as i16).collect();
                let arr = ArrayD::from_shape_vec(shape, flat_i16).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}"))
                })?;
                Ok(Self {
                    data: ArrayData::I16(arr),
                })
            }
            ElemType::I32 => {
                // Convert i64 to i32
                let flat_i32: Vec<i32> = flat_i64.iter().map(|&x| x as i32).collect();
                let arr = ArrayD::from_shape_vec(shape, flat_i32).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}"))
                })?;
                Ok(Self {
                    data: ArrayData::I32(arr),
                })
            }
            ElemType::I64 => {
                let arr = ArrayD::from_shape_vec(shape, flat_i64).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}"))
                })?;
                Ok(Self {
                    data: ArrayData::I64(arr),
                })
            }
            ElemType::U8 => {
                // Convert i64 to u8
                let flat_u8: Vec<u8> = flat_i64.iter().map(|&x| x as u8).collect();
                let arr = ArrayD::from_shape_vec(shape, flat_u8).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}"))
                })?;
                Ok(Self {
                    data: ArrayData::U8(arr),
                })
            }
            ElemType::U16 => {
                // Convert i64 to u16
                let flat_u16: Vec<u16> = flat_i64.iter().map(|&x| x as u16).collect();
                let arr = ArrayD::from_shape_vec(shape, flat_u16).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}"))
                })?;
                Ok(Self {
                    data: ArrayData::U16(arr),
                })
            }
            ElemType::U32 => {
                // Convert i64 to u32
                let flat_u32: Vec<u32> = flat_i64.iter().map(|&x| x as u32).collect();
                let arr = ArrayD::from_shape_vec(shape, flat_u32).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}"))
                })?;
                Ok(Self {
                    data: ArrayData::U32(arr),
                })
            }
            ElemType::U64 => {
                // Convert i64 to u64
                let flat_u64: Vec<u64> = flat_i64.iter().map(|&x| x as u64).collect();
                let arr = ArrayD::from_shape_vec(shape, flat_u64).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}"))
                })?;
                Ok(Self {
                    data: ArrayData::U64(arr),
                })
            }
            ElemType::F32 => {
                // Convert f64 to f32
                let flat_f32: Vec<f32> = flat_f64.iter().map(|&x| x as f32).collect();
                let arr = ArrayD::from_shape_vec(shape, flat_f32).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}"))
                })?;
                Ok(Self {
                    data: ArrayData::F32(arr),
                })
            }
            ElemType::F64 => {
                let arr = ArrayD::from_shape_vec(shape, flat_f64).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}"))
                })?;
                Ok(Self {
                    data: ArrayData::F64(arr),
                })
            }
            ElemType::Complex64 => {
                // Convert Complex<f64> to Complex<f32>
                let flat_c64: Vec<num_complex::Complex<f32>> = flat_complex
                    .iter()
                    .map(|&c| num_complex::Complex::new(c.re as f32, c.im as f32))
                    .collect();
                let arr = ArrayD::from_shape_vec(shape, flat_c64).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}"))
                })?;
                Ok(Self {
                    data: ArrayData::Complex64(arr),
                })
            }
            ElemType::Complex128 => {
                let arr = ArrayD::from_shape_vec(shape, flat_complex).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}"))
                })?;
                Ok(Self {
                    data: ArrayData::Complex128(arr),
                })
            }
            ElemType::Pauli => {
                let arr = ArrayD::from_shape_vec(shape, flat_pauli).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}"))
                })?;
                Ok(Self {
                    data: ArrayData::Pauli(arr),
                })
            }
            ElemType::PauliString => {
                let arr = ArrayD::from_shape_vec(shape, flat_paulistring).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}"))
                })?;
                Ok(Self {
                    data: ArrayData::PauliString(arr),
                })
            }
        }
    }

    /// Infer the shape of a nested sequence
    fn infer_shape(data: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
        use pyo3::types::{PySequence, PyString};

        let mut shape = Vec::new();
        let mut current = data.clone();

        loop {
            // Check if this is a string first - strings are sequences but should be treated as scalars
            if current.is_instance_of::<PyString>() {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Arrays cannot contain string objects. Use Pauli objects instead of strings for Pauli symbols.",
                ));
            }

            // Check if this is an Array object - if so, add its shape and stop
            if let Ok(arr) = current.extract::<pyo3::PyRef<Array>>() {
                shape.extend(arr.data.shape());
                break;
            }

            if let Ok(seq) = current.cast::<PySequence>() {
                let len = seq.len()?;
                shape.push(len);

                if len > 0 {
                    current = seq.get_item(0)?;
                } else {
                    break;
                }
            } else {
                // Reached a scalar
                break;
            }
        }

        Ok(shape)
    }

    /// Flatten a nested sequence into a 1D vector
    fn flatten_sequence(
        data: &Bound<'_, PyAny>,
        flat_f64: &mut Vec<f64>,
        flat_complex: &mut Vec<num_complex::Complex<f64>>,
        flat_pauli: &mut Vec<Pauli>,
        flat_paulistring: &mut Vec<PauliString>,
        flat_bool: &mut Vec<bool>,
        flat_i64: &mut Vec<i64>,
        elem_type: &mut ElemType,
        explicit_dtype: bool,
    ) -> PyResult<()> {
        use pyo3::types::{PySequence, PyString};

        // Check if this is a string first - strings are sequences in Python but should be treated as scalars/objects
        // Arrays cannot contain arbitrary Python objects like strings
        if data.is_instance_of::<PyString>() {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Arrays cannot contain string objects. Use Pauli objects instead of strings for Pauli symbols.",
            ));
        }

        // Check if this is an Array object (before checking sequence)
        // If it is, we need to flatten its contents directly
        if let Ok(arr) = data.extract::<pyo3::PyRef<Array>>() {
            // It's an Array - flatten its raw data directly
            match &arr.data {
                ArrayData::Bool(ndarray) => {
                    for val in ndarray {
                        flat_bool.push(*val);
                    }
                    if !explicit_dtype && *elem_type != ElemType::Bool {
                        *elem_type = ElemType::Bool;
                    }
                }
                ArrayData::I8(ndarray) => {
                    for val in ndarray {
                        flat_i64.push(i64::from(*val));
                    }
                }
                ArrayData::I16(ndarray) => {
                    for val in ndarray {
                        flat_i64.push(i64::from(*val));
                    }
                }
                ArrayData::I32(ndarray) => {
                    for val in ndarray {
                        flat_i64.push(i64::from(*val));
                    }
                }
                ArrayData::I64(ndarray) => {
                    for val in ndarray {
                        flat_i64.push(*val);
                    }
                }
                ArrayData::U8(ndarray) => {
                    for val in ndarray {
                        flat_i64.push(i64::from(*val));
                    }
                }
                ArrayData::U16(ndarray) => {
                    for val in ndarray {
                        flat_i64.push(i64::from(*val));
                    }
                }
                ArrayData::U32(ndarray) => {
                    for val in ndarray {
                        flat_i64.push(i64::from(*val));
                    }
                }
                ArrayData::U64(ndarray) => {
                    for val in ndarray {
                        flat_i64.push(*val as i64);
                    }
                }
                ArrayData::F32(ndarray) => {
                    for val in ndarray {
                        flat_f64.push(f64::from(*val));
                    }
                    if !explicit_dtype {
                        *elem_type = ElemType::F64;
                    }
                }
                ArrayData::F64(ndarray) => {
                    for val in ndarray {
                        flat_f64.push(*val);
                    }
                    if !explicit_dtype {
                        *elem_type = ElemType::F64;
                    }
                }
                ArrayData::Complex64(ndarray) => {
                    for val in ndarray {
                        flat_complex.push(num_complex::Complex::new(
                            f64::from(val.re),
                            f64::from(val.im),
                        ));
                    }
                    if !explicit_dtype {
                        *elem_type = ElemType::Complex128;
                    }
                }
                ArrayData::Complex128(ndarray) => {
                    for val in ndarray {
                        flat_complex.push(*val);
                    }
                    if !explicit_dtype {
                        *elem_type = ElemType::Complex128;
                    }
                }
                ArrayData::Pauli(ndarray) => {
                    for val in ndarray {
                        flat_pauli.push(*val);
                    }
                    if !explicit_dtype {
                        *elem_type = ElemType::Pauli;
                    }
                }
                ArrayData::PauliString(ndarray) => {
                    for val in ndarray {
                        flat_paulistring.push(val.clone());
                    }
                    if !explicit_dtype {
                        *elem_type = ElemType::PauliString;
                    }
                }
            }
        } else if let Ok(seq) = data.cast::<PySequence>() {
            // It's a sequence - recurse
            for i in 0..seq.len()? {
                let item = seq.get_item(i)?;
                Self::flatten_sequence(
                    &item,
                    flat_f64,
                    flat_complex,
                    flat_pauli,
                    flat_paulistring,
                    flat_bool,
                    flat_i64,
                    elem_type,
                    explicit_dtype,
                )?;
            }
        } else {
            // It's a scalar - extract it based on explicit or inferred type
            if explicit_dtype {
                // Explicit dtype: convert value to target type
                Self::extract_and_convert_value(
                    data,
                    *elem_type,
                    flat_f64,
                    flat_complex,
                    flat_pauli,
                    flat_paulistring,
                    flat_bool,
                    flat_i64,
                )?;
            } else {
                // Auto-detect type (Priority 2, 3, and 4 will be added here)
                Self::extract_and_infer_type(
                    data,
                    elem_type,
                    flat_f64,
                    flat_complex,
                    flat_pauli,
                    flat_paulistring,
                    flat_bool,
                    flat_i64,
                )?;
            }
        }

        Ok(())
    }

    /// Extract value and convert to explicit dtype
    fn extract_and_convert_value(
        data: &Bound<'_, PyAny>,
        target_type: ElemType,
        flat_f64: &mut Vec<f64>,
        flat_complex: &mut Vec<num_complex::Complex<f64>>,
        flat_pauli: &mut Vec<Pauli>,
        flat_paulistring: &mut Vec<PauliString>,
        flat_bool: &mut Vec<bool>,
        flat_i64: &mut Vec<i64>,
    ) -> PyResult<()> {
        match target_type {
            ElemType::Bool => {
                // Try bool first, then convert from int
                if let Ok(val) = data.extract::<bool>() {
                    flat_bool.push(val);
                } else if let Ok(val) = data.extract::<i64>() {
                    flat_bool.push(val != 0);
                } else {
                    let val = data.extract::<f64>()?;
                    flat_bool.push(val != 0.0);
                }
            }
            ElemType::I8
            | ElemType::I16
            | ElemType::I32
            | ElemType::I64
            | ElemType::U8
            | ElemType::U16
            | ElemType::U32
            | ElemType::U64 => {
                let val = data.extract::<i64>()?;
                flat_i64.push(val);
            }
            ElemType::F32 | ElemType::F64 => {
                let val = data.extract::<f64>()?;
                flat_f64.push(val);
            }
            ElemType::Complex64 | ElemType::Complex128 => {
                // Try complex first, then convert float
                if let Ok(val) = data.extract::<num_complex::Complex<f64>>() {
                    flat_complex.push(val);
                } else {
                    let val = data.extract::<f64>()?;
                    flat_complex.push(num_complex::Complex::new(val, 0.0));
                }
            }
            ElemType::Pauli => {
                let val = data.extract::<Pauli>()?;
                flat_pauli.push(val);
            }
            ElemType::PauliString => {
                let val = data.extract::<PauliString>()?;
                flat_paulistring.push(val);
            }
        }
        Ok(())
    }

    /// Extract value and infer type automatically
    fn extract_and_infer_type(
        data: &Bound<'_, PyAny>,
        elem_type: &mut ElemType,
        flat_f64: &mut Vec<f64>,
        flat_complex: &mut Vec<num_complex::Complex<f64>>,
        flat_pauli: &mut Vec<Pauli>,
        flat_paulistring: &mut Vec<PauliString>,
        flat_bool: &mut Vec<bool>,
        flat_i64: &mut Vec<i64>,
    ) -> PyResult<()> {
        use pyo3::types::PyBool;

        // Priority order: PauliString > Pauli > Bool > Int > Complex > Float
        if data.is_instance_of::<PauliString>() {
            *elem_type = ElemType::PauliString;
            let paulistring = data.extract::<PauliString>()?;
            flat_paulistring.push(paulistring);
        } else if data.is_instance_of::<Pauli>() {
            *elem_type = ElemType::Pauli;
            let pauli = data.extract::<Pauli>()?;
            flat_pauli.push(pauli);
        } else if data.is_instance_of::<PyBool>() {
            // Priority 2: Auto-detect booleans
            if *elem_type != ElemType::Bool {
                // Type promotion needed - convert existing values
                Self::promote_type_to_bool(elem_type, flat_bool, flat_i64, flat_f64)?;
            }
            let val = data.extract::<bool>()?;
            flat_bool.push(val);
        } else if data.is_instance_of::<pyo3::types::PyComplex>() {
            // Found complex - promote if needed
            if matches!(*elem_type, ElemType::F64 | ElemType::I64 | ElemType::Bool) {
                Self::promote_type_to_complex(
                    elem_type,
                    flat_complex,
                    flat_f64,
                    flat_i64,
                    flat_bool,
                )?;
            }
            *elem_type = ElemType::Complex128;
            let val = data.extract::<num_complex::Complex<f64>>()?;
            flat_complex.push(val);
        } else {
            // Priority 3: Check if it's an integer by type name
            let type_name = data.get_type().name()?;

            if type_name == "int" {
                // It's a Python int
                let ival = data.extract::<i64>()?;
                match elem_type {
                    ElemType::Complex128 | ElemType::Complex64 => {
                        flat_complex.push(num_complex::Complex::new(ival as f64, 0.0));
                    }
                    ElemType::F64 | ElemType::F32 => {
                        flat_f64.push(ival as f64);
                    }
                    ElemType::Bool => {
                        flat_bool.push(ival != 0);
                    }
                    _ => {
                        // First value or already in int mode
                        *elem_type = ElemType::I64;
                        flat_i64.push(ival);
                    }
                }
                return Ok(());
            }

            // Try as float
            if let Ok(val) = data.extract::<f64>() {
                if matches!(*elem_type, ElemType::I64) {
                    Self::promote_type_to_float(elem_type, flat_f64, flat_i64)?;
                }
                if *elem_type == ElemType::Complex128 {
                    flat_complex.push(num_complex::Complex::new(val, 0.0));
                } else {
                    *elem_type = ElemType::F64;
                    flat_f64.push(val);
                }
                return Ok(());
            }

            // If we got here, extraction failed
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Cannot extract numeric value from {type_name}"
            )));
        }

        Ok(())
    }

    /// Promote existing values to bool
    fn promote_type_to_bool(
        elem_type: &mut ElemType,
        flat_bool: &mut Vec<bool>,
        flat_i64: &mut Vec<i64>,
        flat_f64: &mut Vec<f64>,
    ) -> PyResult<()> {
        match elem_type {
            ElemType::I64 => {
                for &i in flat_i64.iter() {
                    flat_bool.push(i != 0);
                }
                flat_i64.clear();
            }
            ElemType::F64 => {
                for &f in flat_f64.iter() {
                    flat_bool.push(f != 0.0);
                }
                flat_f64.clear();
            }
            _ => {}
        }
        *elem_type = ElemType::Bool;
        Ok(())
    }

    /// Promote existing values to float
    fn promote_type_to_float(
        elem_type: &mut ElemType,
        flat_f64: &mut Vec<f64>,
        flat_i64: &mut Vec<i64>,
    ) -> PyResult<()> {
        for &i in flat_i64.iter() {
            flat_f64.push(i as f64);
        }
        flat_i64.clear();
        *elem_type = ElemType::F64;
        Ok(())
    }

    /// Promote existing values to complex
    fn promote_type_to_complex(
        elem_type: &mut ElemType,
        flat_complex: &mut Vec<num_complex::Complex<f64>>,
        flat_f64: &mut Vec<f64>,
        flat_i64: &mut Vec<i64>,
        flat_bool: &mut Vec<bool>,
    ) -> PyResult<()> {
        match elem_type {
            ElemType::F64 => {
                for &f in flat_f64.iter() {
                    flat_complex.push(num_complex::Complex::new(f, 0.0));
                }
                flat_f64.clear();
            }
            ElemType::I64 => {
                for &i in flat_i64.iter() {
                    flat_complex.push(num_complex::Complex::new(i as f64, 0.0));
                }
                flat_i64.clear();
            }
            ElemType::Bool => {
                for &b in flat_bool.iter() {
                    flat_complex.push(num_complex::Complex::new(if b { 1.0 } else { 0.0 }, 0.0));
                }
                flat_bool.clear();
            }
            _ => {}
        }
        *elem_type = ElemType::Complex128;
        Ok(())
    }

    /// Try to create Array from `NumPy` array
    fn try_from_numpy(array: &Bound<'_, PyAny>) -> PyResult<Self> {
        use crate::array_buffer;
        use pyo3::types::PyDict;

        // Get __array_interface__ dict from the Python object
        // IMPORTANT: Always use Python's builtin getattr() instead of PyO3's .getattr()
        // because PyO3's getattr doesn't correctly handle data descriptors in abi3 mode.
        // NumPy's __array_interface__ is implemented as a data descriptor.
        //
        // We cannot use py.import("builtins").getattr("getattr") because .getattr() has the
        // bug we're trying to work around. Instead, we use eval to directly access the function.
        let py = array.py();
        let getattr_fn = py.eval(c"getattr", None, None)?;
        let array_iface = getattr_fn.call1((array, "__array_interface__"))?;
        let interface: &Bound<'_, PyDict> = &array_iface.cast_into::<PyDict>()?;

        // Extract typestr to determine dtype
        let typestr = interface.get_item("typestr")?.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Missing 'typestr' in __array_interface__")
        })?;
        let typestr_str: &str = typestr.extract()?;

        // Try to extract based on dtype
        // Support little-endian (<), big-endian (>), and native (=) byte orders
        match typestr_str {
            "<f8" | ">f8" | "=f8" => {
                let ndarray = array_buffer::extract_f64_array(array)?;
                Ok(Self {
                    data: ArrayData::F64(ndarray),
                })
            }
            "<i8" | ">i8" | "=i8" => {
                let ndarray = array_buffer::extract_i64_array(array)?;
                Ok(Self {
                    data: ArrayData::I64(ndarray),
                })
            }
            "<c16" | ">c16" | "=c16" => {
                let ndarray = array_buffer::extract_complex64_array(array)?;
                Ok(Self {
                    data: ArrayData::Complex128(ndarray),
                })
            }
            "<f4" | ">f4" | "=f4" => {
                let ndarray = array_buffer::extract_f32_array(array)?;
                Ok(Self {
                    data: ArrayData::F32(ndarray),
                })
            }
            "<i4" | ">i4" | "=i4" => {
                let ndarray = array_buffer::extract_i32_array(array)?;
                Ok(Self {
                    data: ArrayData::I32(ndarray),
                })
            }
            "<i2" | ">i2" | "=i2" => {
                let ndarray = array_buffer::extract_i16_array(array)?;
                Ok(Self {
                    data: ArrayData::I16(ndarray),
                })
            }
            "|i1" | "i1" | "=i1" | "<i1" | ">i1" => {
                let ndarray = array_buffer::extract_i8_array(array)?;
                Ok(Self {
                    data: ArrayData::I8(ndarray),
                })
            }
            "<c8" | ">c8" | "=c8" => {
                let ndarray = array_buffer::extract_complex32_array(array)?;
                Ok(Self {
                    data: ArrayData::Complex64(ndarray),
                })
            }
            "|b1" | "=b1" | "?1" => {
                let ndarray = array_buffer::extract_bool_array(array)?;
                Ok(Self {
                    data: ArrayData::Bool(ndarray),
                })
            }
            _ => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Unsupported dtype: {typestr_str}. Expected one of: f64, i64, complex128, f32, i32, i16, i8, complex64, bool"
            ))),
        }
    }

    /// Create a new `Array` from a typed ndarray
    pub fn from_array_i64(arr: ArrayD<i64>) -> Self {
        Self {
            data: ArrayData::I64(arr),
        }
    }

    pub fn from_array_f64(arr: ArrayD<f64>) -> Self {
        Self {
            data: ArrayData::F64(arr),
        }
    }

    pub fn from_array_c128(arr: ArrayD<num_complex::Complex<f64>>) -> Self {
        Self {
            data: ArrayData::Complex128(arr),
        }
    }

    pub fn from_array_u64(arr: ArrayD<u64>) -> Self {
        Self {
            data: ArrayData::U64(arr),
        }
    }

    pub fn from_array_u32(arr: ArrayD<u32>) -> Self {
        Self {
            data: ArrayData::U32(arr),
        }
    }

    pub fn from_array_u16(arr: ArrayD<u16>) -> Self {
        Self {
            data: ArrayData::U16(arr),
        }
    }

    pub fn from_array_u8(arr: ArrayD<u8>) -> Self {
        Self {
            data: ArrayData::U8(arr),
        }
    }

    pub fn from_array_f32(arr: ArrayD<f32>) -> Self {
        Self {
            data: ArrayData::F32(arr),
        }
    }

    pub fn from_array_i32(arr: ArrayD<i32>) -> Self {
        Self {
            data: ArrayData::I32(arr),
        }
    }

    pub fn from_array_i16(arr: ArrayD<i16>) -> Self {
        Self {
            data: ArrayData::I16(arr),
        }
    }

    pub fn from_array_i8(arr: ArrayD<i8>) -> Self {
        Self {
            data: ArrayData::I8(arr),
        }
    }

    pub fn from_array_bool(arr: ArrayD<bool>) -> Self {
        Self {
            data: ArrayData::Bool(arr),
        }
    }

    /// Compute the broadcast shape for two arrays following `NumPy` broadcasting rules.
    ///
    /// `NumPy` broadcasting rules:
    /// 1. If arrays have different number of dimensions, prepend 1s to the smaller one
    /// 2. For each dimension, the sizes must either:
    ///    - Be equal, or
    ///    - One of them is 1
    /// 3. The output shape is the maximum of the two shapes in each dimension
    ///
    /// Returns `Ok(broadcast_shape)` if broadcasting is possible, Err otherwise.
    fn broadcast_shape(shape1: &[usize], shape2: &[usize]) -> Result<Vec<usize>, String> {
        let ndim1 = shape1.len();
        let ndim2 = shape2.len();
        let max_ndim = ndim1.max(ndim2);

        let mut result = Vec::with_capacity(max_ndim);

        // Iterate from the trailing dimensions
        for i in 0..max_ndim {
            let dim1 = if i < ndim1 { shape1[ndim1 - 1 - i] } else { 1 };
            let dim2 = if i < ndim2 { shape2[ndim2 - 1 - i] } else { 1 };

            if dim1 == dim2 {
                result.push(dim1);
            } else if dim1 == 1 {
                result.push(dim2);
            } else if dim2 == 1 {
                result.push(dim1);
            } else {
                return Err(format!(
                    "Shape mismatch: cannot broadcast shapes {shape1:?} and {shape2:?}"
                ));
            }
        }

        // Reverse to get the correct order (we built it backwards)
        result.reverse();
        Ok(result)
    }

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
        use pyo3::types::PyComplex;

        // Try to extract as f64 scalar first
        if let Ok(scalar) = other.extract::<f64>() {
            // Scalar operation: apply to all elements
            match &self.data {
                ArrayData::Bool(_) => Err(pyo3::exceptions::PyTypeError::new_err(
                    "Arithmetic operations not supported on boolean arrays",
                )),
                ArrayData::I8(arr) => {
                    let result = arr.mapv(|x| op(f64::from(x), scalar) as i8);
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::I8(result),
                        },
                    )?
                    .into_any())
                }
                ArrayData::I16(arr) => {
                    let result = arr.mapv(|x| op(f64::from(x), scalar) as i16);
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::I16(result),
                        },
                    )?
                    .into_any())
                }
                ArrayData::I32(arr) => {
                    let result = arr.mapv(|x| op(f64::from(x), scalar) as i32);
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::I32(result),
                        },
                    )?
                    .into_any())
                }
                ArrayData::I64(arr) => {
                    let result = arr.mapv(|x| op(x as f64, scalar) as i64);
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::I64(result),
                        },
                    )?
                    .into_any())
                }
                ArrayData::U8(arr) => {
                    let result = arr.mapv(|x| op(f64::from(x), scalar) as u8);
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::U8(result),
                        },
                    )?
                    .into_any())
                }
                ArrayData::U16(arr) => {
                    let result = arr.mapv(|x| op(f64::from(x), scalar) as u16);
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::U16(result),
                        },
                    )?
                    .into_any())
                }
                ArrayData::U32(arr) => {
                    let result = arr.mapv(|x| op(f64::from(x), scalar) as u32);
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::U32(result),
                        },
                    )?
                    .into_any())
                }
                ArrayData::U64(arr) => {
                    let result = arr.mapv(|x| op(x as f64, scalar) as u64);
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::U64(result),
                        },
                    )?
                    .into_any())
                }
                ArrayData::F32(arr) => {
                    let result = arr.mapv(|x| op(f64::from(x), scalar) as f32);
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::F32(result),
                        },
                    )?
                    .into_any())
                }
                ArrayData::F64(arr) => {
                    let result = arr.mapv(|x| op(x, scalar));
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::F64(result),
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
                ArrayData::Pauli(_) => Err(pyo3::exceptions::PyTypeError::new_err(
                    "Arithmetic operations not supported on Pauli arrays",
                )),
                ArrayData::PauliString(_) => Err(pyo3::exceptions::PyTypeError::new_err(
                    "Arithmetic operations not supported on PauliString arrays",
                )),
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
                (ArrayData::F64(a), ArrayData::F64(b)) => {
                    // Compute broadcast shape
                    let broadcast_shape = Self::broadcast_shape(a.shape(), b.shape())
                        .map_err(pyo3::exceptions::PyValueError::new_err)?;

                    // Convert to IxDyn for broadcasting
                    let target_shape = IxDyn(&broadcast_shape);

                    // Broadcast both arrays to the target shape
                    let a_broadcast = a.broadcast(target_shape.clone()).ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Failed to broadcast array with shape {:?} to {:?}",
                            a.shape(),
                            broadcast_shape
                        ))
                    })?;
                    let b_broadcast = b.broadcast(target_shape.clone()).ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Failed to broadcast array with shape {:?} to {:?}",
                            b.shape(),
                            broadcast_shape
                        ))
                    })?;

                    // Apply operation element-wise on broadcasted arrays
                    let result = a_broadcast
                        .iter()
                        .zip(b_broadcast.iter())
                        .map(|(x, y)| op(*x, *y))
                        .collect::<Vec<_>>();

                    let result_arr = ArrayD::from_shape_vec(target_shape, result).map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}"))
                    })?;

                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::F64(result_arr),
                        },
                    )?
                    .into_any())
                }
                (ArrayData::I64(a), ArrayData::I64(b)) => {
                    // Compute broadcast shape
                    let broadcast_shape = Self::broadcast_shape(a.shape(), b.shape())
                        .map_err(pyo3::exceptions::PyValueError::new_err)?;

                    // Convert to IxDyn for broadcasting
                    let target_shape = IxDyn(&broadcast_shape);

                    // Broadcast both arrays to the target shape
                    let a_broadcast = a.broadcast(target_shape.clone()).ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Failed to broadcast array with shape {:?} to {:?}",
                            a.shape(),
                            broadcast_shape
                        ))
                    })?;
                    let b_broadcast = b.broadcast(target_shape.clone()).ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Failed to broadcast array with shape {:?} to {:?}",
                            b.shape(),
                            broadcast_shape
                        ))
                    })?;

                    // Apply operation element-wise on broadcasted arrays
                    let result = a_broadcast
                        .iter()
                        .zip(b_broadcast.iter())
                        .map(|(x, y)| op(*x as f64, *y as f64) as i64)
                        .collect::<Vec<_>>();

                    let result_arr = ArrayD::from_shape_vec(target_shape, result).map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}"))
                    })?;

                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::I64(result_arr),
                        },
                    )?
                    .into_any())
                }
                (ArrayData::Complex128(a), ArrayData::Complex128(b)) => {
                    // Compute broadcast shape
                    let broadcast_shape = Self::broadcast_shape(a.shape(), b.shape())
                        .map_err(pyo3::exceptions::PyValueError::new_err)?;

                    // Convert to IxDyn for broadcasting
                    let target_shape = IxDyn(&broadcast_shape);

                    // Broadcast both arrays to the target shape
                    let a_broadcast = a.broadcast(target_shape.clone()).ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Failed to broadcast array with shape {:?} to {:?}",
                            a.shape(),
                            broadcast_shape
                        ))
                    })?;
                    let b_broadcast = b.broadcast(target_shape.clone()).ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Failed to broadcast array with shape {:?} to {:?}",
                            b.shape(),
                            broadcast_shape
                        ))
                    })?;

                    // Apply operation element-wise on broadcasted arrays
                    let result = a_broadcast
                        .iter()
                        .zip(b_broadcast.iter())
                        .map(|(x, y)| {
                            let re = op(x.re, y.re);
                            let im = op(x.im, y.im);
                            Complex64::new(re, im)
                        })
                        .collect::<Vec<_>>();

                    let result_arr = ArrayD::from_shape_vec(target_shape, result).map_err(|e| {
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
        } else if let Ok(other_arr) = crate::array_buffer::extract_f64_array(other) {
            // Numpy array operation

            match &self.data {
                ArrayData::F64(a) => {
                    // Compute broadcast shape
                    let broadcast_shape = Self::broadcast_shape(a.shape(), other_arr.shape())
                        .map_err(pyo3::exceptions::PyValueError::new_err)?;

                    // Convert to IxDyn for broadcasting
                    let target_shape = IxDyn(&broadcast_shape);

                    // Broadcast both arrays to the target shape
                    let a_broadcast = a.broadcast(target_shape.clone()).ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(format!(
                            "Failed to broadcast array with shape {:?} to {:?}",
                            a.shape(),
                            broadcast_shape
                        ))
                    })?;
                    let b_broadcast =
                        other_arr.broadcast(target_shape.clone()).ok_or_else(|| {
                            pyo3::exceptions::PyValueError::new_err(format!(
                                "Failed to broadcast array with shape {:?} to {:?}",
                                other_arr.shape(),
                                broadcast_shape
                            ))
                        })?;

                    // Apply operation element-wise on broadcasted arrays
                    let result = a_broadcast
                        .iter()
                        .zip(b_broadcast.iter())
                        .map(|(x, y)| op(*x, *y))
                        .collect::<Vec<_>>();

                    let result_arr = ArrayD::from_shape_vec(target_shape, result).map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!("Shape error: {e}"))
                    })?;

                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::F64(result_arr),
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
                ArrayData::Bool(_) => Err(pyo3::exceptions::PyTypeError::new_err(
                    "Arithmetic operations not supported on boolean arrays",
                )),
                ArrayData::I8(arr) => {
                    let result = arr.mapv(|x| op(scalar, f64::from(x)) as i8);
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::I8(result),
                        },
                    )?
                    .into_any())
                }
                ArrayData::I16(arr) => {
                    let result = arr.mapv(|x| op(scalar, f64::from(x)) as i16);
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::I16(result),
                        },
                    )?
                    .into_any())
                }
                ArrayData::I32(arr) => {
                    let result = arr.mapv(|x| op(scalar, f64::from(x)) as i32);
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::I32(result),
                        },
                    )?
                    .into_any())
                }
                ArrayData::I64(arr) => {
                    let result = arr.mapv(|x| op(scalar, x as f64) as i64);
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::I64(result),
                        },
                    )?
                    .into_any())
                }
                ArrayData::U8(arr) => {
                    let result = arr.mapv(|x| op(scalar, f64::from(x)) as u8);
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::U8(result),
                        },
                    )?
                    .into_any())
                }
                ArrayData::U16(arr) => {
                    let result = arr.mapv(|x| op(scalar, f64::from(x)) as u16);
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::U16(result),
                        },
                    )?
                    .into_any())
                }
                ArrayData::U32(arr) => {
                    let result = arr.mapv(|x| op(scalar, f64::from(x)) as u32);
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::U32(result),
                        },
                    )?
                    .into_any())
                }
                ArrayData::U64(arr) => {
                    let result = arr.mapv(|x| op(scalar, x as f64) as u64);
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::U64(result),
                        },
                    )?
                    .into_any())
                }
                ArrayData::F32(arr) => {
                    let result = arr.mapv(|x| op(scalar, f64::from(x)) as f32);
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::F32(result),
                        },
                    )?
                    .into_any())
                }
                ArrayData::F64(arr) => {
                    let result = arr.mapv(|x| op(scalar, x));
                    Ok(Py::new(
                        py,
                        Array {
                            data: ArrayData::F64(result),
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
                ArrayData::Pauli(_) => Err(pyo3::exceptions::PyTypeError::new_err(
                    "Arithmetic operations not supported on Pauli arrays",
                )),
                ArrayData::PauliString(_) => Err(pyo3::exceptions::PyTypeError::new_err(
                    "Arithmetic operations not supported on PauliString arrays",
                )),
            }
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Unsupported operand type for reverse {op_name}"
            )))
        }
    }

    /// Helper method for comparison operations: self op other
    /// Always returns a float64 array with 1.0 for True and 0.0 for False
    /// F is a closure that performs the comparison (e.g., |a, b| if a > b { 1.0 } else { 0.0 })
    fn comparison_op<F>(
        &self,
        other: &Bound<'_, PyAny>,
        py: Python<'_>,
        op: F,
        op_name: &str,
    ) -> PyResult<Py<PyAny>>
    where
        F: Fn(f64, f64) -> f64 + Copy,
    {
        // Try to extract as f64 scalar first
        if let Ok(scalar) = other.extract::<f64>() {
            // Scalar comparison: apply to all elements, always return float64 array
            match &self.data {
                ArrayData::Bool(_) => Err(pyo3::exceptions::PyTypeError::new_err(
                    "Comparison operations with numeric scalars not supported on boolean arrays",
                )),
                ArrayData::I8(arr) => {
                    let result = arr.mapv(|x| op(f64::from(x), scalar));
                    Ok(Py::new(py, Array::from_array_f64(result))?.into_any())
                }
                ArrayData::I16(arr) => {
                    let result = arr.mapv(|x| op(f64::from(x), scalar));
                    Ok(Py::new(py, Array::from_array_f64(result))?.into_any())
                }
                ArrayData::I32(arr) => {
                    let result = arr.mapv(|x| op(f64::from(x), scalar));
                    Ok(Py::new(py, Array::from_array_f64(result))?.into_any())
                }
                ArrayData::I64(arr) => {
                    let result = arr.mapv(|x| op(x as f64, scalar));
                    Ok(Py::new(py, Array::from_array_f64(result))?.into_any())
                }
                ArrayData::U8(arr) => {
                    let result = arr.mapv(|x| op(f64::from(x), scalar));
                    Ok(Py::new(py, Array::from_array_f64(result))?.into_any())
                }
                ArrayData::U16(arr) => {
                    let result = arr.mapv(|x| op(f64::from(x), scalar));
                    Ok(Py::new(py, Array::from_array_f64(result))?.into_any())
                }
                ArrayData::U32(arr) => {
                    let result = arr.mapv(|x| op(f64::from(x), scalar));
                    Ok(Py::new(py, Array::from_array_f64(result))?.into_any())
                }
                ArrayData::U64(arr) => {
                    let result = arr.mapv(|x| op(x as f64, scalar));
                    Ok(Py::new(py, Array::from_array_f64(result))?.into_any())
                }
                ArrayData::F32(arr) => {
                    let result = arr.mapv(|x| op(f64::from(x), scalar));
                    Ok(Py::new(py, Array::from_array_f64(result))?.into_any())
                }
                ArrayData::F64(arr) => {
                    let result = arr.mapv(|x| op(x, scalar));
                    Ok(Py::new(py, Array::from_array_f64(result))?.into_any())
                }
                ArrayData::Complex64(_) | ArrayData::Complex128(_) => {
                    Err(pyo3::exceptions::PyTypeError::new_err(format!(
                        "Comparison {op_name} not supported for complex arrays"
                    )))
                }
                ArrayData::Pauli(_) => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "Comparison {op_name} not supported for Pauli arrays"
                ))),
                ArrayData::PauliString(_) => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "Comparison {op_name} not supported for PauliString arrays"
                ))),
            }
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Unsupported operand type for comparison {op_name}"
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
        // Apply 1D slice assignment based on data type
        // Use ndarray's slice_mut() with Slice::from() for unit-step slicing
        match &mut self.data {
            ArrayData::Bool(arr) => {
                let slice = Slice::from(start..stop);
                let mut view = arr.slice_mut(ndarray::s![slice]);
                if let Ok(scalar_val) = value.extract::<bool>() {
                    view.fill(scalar_val);
                } else if let Ok(np_arr) = crate::array_buffer::extract_bool_array(value) {
                    view.assign(&np_arr);
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "Value must be a scalar or array matching the slice shape and dtype",
                    ));
                }
            }
            ArrayData::I8(arr) => {
                let slice = Slice::from(start..stop);
                let mut view = arr.slice_mut(ndarray::s![slice]);
                if let Ok(scalar_val) = value.extract::<i8>() {
                    view.fill(scalar_val);
                } else if let Ok(np_arr) = crate::array_buffer::extract_i8_array(value) {
                    view.assign(&np_arr);
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "Value must be a scalar or array matching the slice shape and dtype",
                    ));
                }
            }
            ArrayData::I16(arr) => {
                let slice = Slice::from(start..stop);
                let mut view = arr.slice_mut(ndarray::s![slice]);
                if let Ok(scalar_val) = value.extract::<i16>() {
                    view.fill(scalar_val);
                } else if let Ok(np_arr) = crate::array_buffer::extract_i16_array(value) {
                    view.assign(&np_arr);
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "Value must be a scalar or array matching the slice shape and dtype",
                    ));
                }
            }
            ArrayData::I32(arr) => {
                let slice = Slice::from(start..stop);
                let mut view = arr.slice_mut(ndarray::s![slice]);
                if let Ok(scalar_val) = value.extract::<i32>() {
                    view.fill(scalar_val);
                } else if let Ok(np_arr) = crate::array_buffer::extract_i32_array(value) {
                    view.assign(&np_arr);
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "Value must be a scalar or array matching the slice shape and dtype",
                    ));
                }
            }
            ArrayData::I64(arr) => {
                let slice = Slice::from(start..stop);
                let mut view = arr.slice_mut(ndarray::s![slice]);
                if let Ok(scalar_val) = value.extract::<i64>() {
                    view.fill(scalar_val);
                } else if let Ok(np_arr) = crate::array_buffer::extract_i64_array(value) {
                    view.assign(&np_arr);
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "Value must be a scalar or array matching the slice shape and dtype",
                    ));
                }
            }
            ArrayData::U8(arr) => {
                let slice = Slice::from(start..stop);
                let mut view = arr.slice_mut(ndarray::s![slice]);
                if let Ok(scalar_val) = value.extract::<u8>() {
                    view.fill(scalar_val);
                } else if let Ok(np_arr) = crate::array_buffer::extract_u8_array(value) {
                    view.assign(&np_arr);
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "Value must be a scalar or array matching the slice shape and dtype",
                    ));
                }
            }
            ArrayData::U16(arr) => {
                let slice = Slice::from(start..stop);
                let mut view = arr.slice_mut(ndarray::s![slice]);
                if let Ok(scalar_val) = value.extract::<u16>() {
                    view.fill(scalar_val);
                } else if let Ok(np_arr) = crate::array_buffer::extract_u16_array(value) {
                    view.assign(&np_arr);
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "Value must be a scalar or array matching the slice shape and dtype",
                    ));
                }
            }
            ArrayData::U32(arr) => {
                let slice = Slice::from(start..stop);
                let mut view = arr.slice_mut(ndarray::s![slice]);
                if let Ok(scalar_val) = value.extract::<u32>() {
                    view.fill(scalar_val);
                } else if let Ok(np_arr) = crate::array_buffer::extract_u32_array(value) {
                    view.assign(&np_arr);
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "Value must be a scalar or array matching the slice shape and dtype",
                    ));
                }
            }
            ArrayData::U64(arr) => {
                let slice = Slice::from(start..stop);
                let mut view = arr.slice_mut(ndarray::s![slice]);
                if let Ok(scalar_val) = value.extract::<u64>() {
                    view.fill(scalar_val);
                } else if let Ok(np_arr) = crate::array_buffer::extract_u64_array(value) {
                    view.assign(&np_arr);
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "Value must be a scalar or array matching the slice shape and dtype",
                    ));
                }
            }
            ArrayData::F32(arr) => {
                let slice = Slice::from(start..stop);
                let mut view = arr.slice_mut(ndarray::s![slice]);
                if let Ok(scalar_val) = value.extract::<f32>() {
                    view.fill(scalar_val);
                } else if let Ok(np_arr) = crate::array_buffer::extract_f32_array(value) {
                    view.assign(&np_arr);
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "Value must be a scalar or array matching the slice shape and dtype",
                    ));
                }
            }
            ArrayData::F64(arr) => {
                let slice = Slice::from(start..stop);
                let mut view = arr.slice_mut(ndarray::s![slice]);
                if let Ok(scalar_val) = value.extract::<f64>() {
                    view.fill(scalar_val);
                } else if let Ok(np_arr) = crate::array_buffer::extract_f64_array(value) {
                    view.assign(&np_arr);
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "Value must be a scalar or array matching the slice shape and dtype",
                    ));
                }
            }
            ArrayData::Complex64(arr) => {
                let slice = Slice::from(start..stop);
                let mut view = arr.slice_mut(ndarray::s![slice]);
                if let Ok(scalar_val) = value.extract::<num_complex::Complex<f32>>() {
                    view.fill(scalar_val);
                } else if let Ok(np_arr) = crate::array_buffer::extract_complex32_array(value) {
                    view.assign(&np_arr);
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "Value must be a scalar or array matching the slice shape and dtype",
                    ));
                }
            }
            ArrayData::Complex128(arr) => {
                let slice = Slice::from(start..stop);
                let mut view = arr.slice_mut(ndarray::s![slice]);
                if let Ok(scalar_val) = value.extract::<num_complex::Complex<f64>>() {
                    view.fill(scalar_val);
                } else if let Ok(np_arr) = crate::array_buffer::extract_complex64_array(value) {
                    view.assign(&np_arr);
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "Value must be a scalar or array matching the slice shape and dtype",
                    ));
                }
            }
            ArrayData::Pauli(_) => {
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                    "Slice assignment not yet implemented for Pauli arrays",
                ));
            }
            ArrayData::PauliString(_) => {
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                    "Slice assignment not yet implemented for PauliString arrays",
                ));
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
            ArrayData::Bool(arr) => {
                if let Ok(scalar_val) = value.extract::<bool>() {
                    for &idx in &indices {
                        arr[idx] = scalar_val;
                    }
                } else if let Ok(np_arr) = crate::array_buffer::extract_bool_array(value) {
                    if np_arr.len() != indices.len() {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Array length {} does not match slice length {}",
                            np_arr.len(),
                            indices.len()
                        )));
                    }
                    for (i, &idx) in indices.iter().enumerate() {
                        arr[idx] = np_arr[i];
                    }
                } else {
                    return Err(pyo3::exceptions::PyTypeError::new_err(
                        "Value must be a scalar or array matching the slice shape and dtype",
                    ));
                }
            }
            ArrayData::I8(arr) => {
                if let Ok(scalar_val) = value.extract::<i8>() {
                    for &idx in &indices {
                        arr[idx] = scalar_val;
                    }
                } else if let Ok(np_arr) = crate::array_buffer::extract_i8_array(value) {
                    let np_slice = np_arr.view();
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
            ArrayData::I16(arr) => {
                if let Ok(scalar_val) = value.extract::<i16>() {
                    for &idx in &indices {
                        arr[idx] = scalar_val;
                    }
                } else if let Ok(np_arr) = crate::array_buffer::extract_i16_array(value) {
                    let np_slice = np_arr.view();
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
            ArrayData::I32(arr) => {
                if let Ok(scalar_val) = value.extract::<i32>() {
                    for &idx in &indices {
                        arr[idx] = scalar_val;
                    }
                } else if let Ok(np_arr) = crate::array_buffer::extract_i32_array(value) {
                    let np_slice = np_arr.view();
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
            ArrayData::I64(arr) => {
                if let Ok(scalar_val) = value.extract::<i64>() {
                    for &idx in &indices {
                        arr[idx] = scalar_val;
                    }
                } else if let Ok(np_arr) = crate::array_buffer::extract_i64_array(value) {
                    let np_slice = np_arr.view();
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
            ArrayData::U8(arr) => {
                if let Ok(scalar_val) = value.extract::<u8>() {
                    for &idx in &indices {
                        arr[idx] = scalar_val;
                    }
                } else if let Ok(np_arr) = crate::array_buffer::extract_u8_array(value) {
                    let np_slice = np_arr.view();
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
            ArrayData::U16(arr) => {
                if let Ok(scalar_val) = value.extract::<u16>() {
                    for &idx in &indices {
                        arr[idx] = scalar_val;
                    }
                } else if let Ok(np_arr) = crate::array_buffer::extract_u16_array(value) {
                    let np_slice = np_arr.view();
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
            ArrayData::U32(arr) => {
                if let Ok(scalar_val) = value.extract::<u32>() {
                    for &idx in &indices {
                        arr[idx] = scalar_val;
                    }
                } else if let Ok(np_arr) = crate::array_buffer::extract_u32_array(value) {
                    let np_slice = np_arr.view();
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
            ArrayData::U64(arr) => {
                if let Ok(scalar_val) = value.extract::<u64>() {
                    for &idx in &indices {
                        arr[idx] = scalar_val;
                    }
                } else if let Ok(np_arr) = crate::array_buffer::extract_u64_array(value) {
                    let np_slice = np_arr.view();
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
            ArrayData::F32(arr) => {
                if let Ok(scalar_val) = value.extract::<f32>() {
                    for &idx in &indices {
                        arr[idx] = scalar_val;
                    }
                } else if let Ok(np_arr) = crate::array_buffer::extract_f32_array(value) {
                    let np_slice = np_arr.view();
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
            ArrayData::F64(arr) => {
                if let Ok(scalar_val) = value.extract::<f64>() {
                    for &idx in &indices {
                        arr[idx] = scalar_val;
                    }
                } else if let Ok(np_arr) = crate::array_buffer::extract_f64_array(value) {
                    let np_slice = np_arr.view();
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
                } else if let Ok(np_arr) = crate::array_buffer::extract_complex32_array(value) {
                    let np_slice = np_arr.view();
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
                } else if let Ok(np_arr) = crate::array_buffer::extract_complex64_array(value) {
                    let np_slice = np_arr.view();
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
            ArrayData::Pauli(_) => {
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                    "Fancy indexing assignment not yet implemented for Pauli arrays",
                ));
            }
            ArrayData::PauliString(_) => {
                return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                    "Fancy indexing assignment not yet implemented for PauliString arrays",
                ));
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

                    result_vec.push($arr[resolved_idx].clone());
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
            ArrayData::Bool(arr) => ArrayData::Bool(impl_fancy_indexing!(arr)),
            ArrayData::I8(arr) => ArrayData::I8(impl_fancy_indexing!(arr)),
            ArrayData::I16(arr) => ArrayData::I16(impl_fancy_indexing!(arr)),
            ArrayData::I32(arr) => ArrayData::I32(impl_fancy_indexing!(arr)),
            ArrayData::I64(arr) => ArrayData::I64(impl_fancy_indexing!(arr)),
            ArrayData::U8(arr) => ArrayData::U8(impl_fancy_indexing!(arr)),
            ArrayData::U16(arr) => ArrayData::U16(impl_fancy_indexing!(arr)),
            ArrayData::U32(arr) => ArrayData::U32(impl_fancy_indexing!(arr)),
            ArrayData::U64(arr) => ArrayData::U64(impl_fancy_indexing!(arr)),
            ArrayData::F32(arr) => ArrayData::F32(impl_fancy_indexing!(arr)),
            ArrayData::F64(arr) => ArrayData::F64(impl_fancy_indexing!(arr)),
            ArrayData::Complex64(arr) => ArrayData::Complex64(impl_fancy_indexing!(arr)),
            ArrayData::Complex128(arr) => ArrayData::Complex128(impl_fancy_indexing!(arr)),
            ArrayData::Pauli(arr) => ArrayData::Pauli(impl_fancy_indexing!(arr)),
            ArrayData::PauliString(arr) => ArrayData::PauliString(impl_fancy_indexing!(arr)),
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
            ArrayData::Bool(arr) => {
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
                    data: ArrayData::Bool(result),
                })
            }
            ArrayData::I8(arr) => {
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
                    data: ArrayData::I8(result),
                })
            }
            ArrayData::I16(arr) => {
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
                    data: ArrayData::I16(result),
                })
            }
            ArrayData::I32(arr) => {
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
                    data: ArrayData::I32(result),
                })
            }
            ArrayData::I64(arr) => {
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
                    data: ArrayData::I64(result),
                })
            }
            ArrayData::U8(arr) => {
                let mut result = arr.clone();
                for (axis, start, stop, step) in slices {
                    if step < 0 {
                        let actual_start = if stop == -1 { 0 } else { stop + 1 };
                        let actual_end = start + 1;
                        let slice_info = Slice::new(actual_start, Some(actual_end), 1);
                        result = result.slice_axis(Axis(axis), slice_info).to_owned();
                        result.invert_axis(Axis(axis));

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
                    data: ArrayData::U8(result),
                })
            }
            ArrayData::U16(arr) => {
                let mut result = arr.clone();
                for (axis, start, stop, step) in slices {
                    if step < 0 {
                        let actual_start = if stop == -1 { 0 } else { stop + 1 };
                        let actual_end = start + 1;
                        let slice_info = Slice::new(actual_start, Some(actual_end), 1);
                        result = result.slice_axis(Axis(axis), slice_info).to_owned();
                        result.invert_axis(Axis(axis));

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
                    data: ArrayData::U16(result),
                })
            }
            ArrayData::U32(arr) => {
                let mut result = arr.clone();
                for (axis, start, stop, step) in slices {
                    if step < 0 {
                        let actual_start = if stop == -1 { 0 } else { stop + 1 };
                        let actual_end = start + 1;
                        let slice_info = Slice::new(actual_start, Some(actual_end), 1);
                        result = result.slice_axis(Axis(axis), slice_info).to_owned();
                        result.invert_axis(Axis(axis));

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
                    data: ArrayData::U32(result),
                })
            }
            ArrayData::U64(arr) => {
                let mut result = arr.clone();
                for (axis, start, stop, step) in slices {
                    if step < 0 {
                        let actual_start = if stop == -1 { 0 } else { stop + 1 };
                        let actual_end = start + 1;
                        let slice_info = Slice::new(actual_start, Some(actual_end), 1);
                        result = result.slice_axis(Axis(axis), slice_info).to_owned();
                        result.invert_axis(Axis(axis));

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
                    data: ArrayData::U64(result),
                })
            }
            ArrayData::F32(arr) => {
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
                    data: ArrayData::F32(result),
                })
            }
            ArrayData::F64(arr) => {
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
                    data: ArrayData::F64(result),
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
            ArrayData::Pauli(arr) => {
                let mut result = arr.clone();
                for (axis, start, stop, step) in slices {
                    if step < 0 {
                        let actual_start = if stop == -1 { 0 } else { stop + 1 };
                        let actual_end = start + 1;
                        let slice_info = Slice::new(actual_start, Some(actual_end), 1);
                        result = result.slice_axis(Axis(axis), slice_info).to_owned();
                        result.invert_axis(Axis(axis));

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
                    data: ArrayData::Pauli(result),
                })
            }
            ArrayData::PauliString(arr) => {
                let mut result = arr.clone();
                for (axis, start, stop, step) in slices {
                    if step < 0 {
                        let actual_start = if stop == -1 { 0 } else { stop + 1 };
                        let actual_end = start + 1;
                        let slice_info = Slice::new(actual_start, Some(actual_end), 1);
                        result = result.slice_axis(Axis(axis), slice_info).to_owned();
                        result.invert_axis(Axis(axis));

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
                    data: ArrayData::PauliString(result),
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
            ArrayData::Bool(arr) => Self::format_array_typed(arr, "bool"),
            ArrayData::F64(arr) => Self::format_array_typed(arr, "float64"),
            ArrayData::F32(arr) => Self::format_array_typed(arr, "float32"),
            ArrayData::I64(arr) => Self::format_array_typed(arr, "int64"),
            ArrayData::I32(arr) => Self::format_array_typed(arr, "int32"),
            ArrayData::I16(arr) => Self::format_array_typed(arr, "int16"),
            ArrayData::I8(arr) => Self::format_array_typed(arr, "int8"),
            ArrayData::U64(arr) => Self::format_array_typed(arr, "uint64"),
            ArrayData::U32(arr) => Self::format_array_typed(arr, "uint32"),
            ArrayData::U16(arr) => Self::format_array_typed(arr, "uint16"),
            ArrayData::U8(arr) => Self::format_array_typed(arr, "uint8"),
            ArrayData::Complex64(arr) => Self::format_array_complex_f32(arr),
            ArrayData::Complex128(arr) => Self::format_array_complex_f64(arr),
            ArrayData::Pauli(arr) => Self::format_array_pauli(arr),
            ArrayData::PauliString(arr) => Self::format_array_paulistring(arr),
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

    /// Format a Pauli array
    fn format_array_pauli(arr: &ArrayD<crate::pauli_bindings::Pauli>) -> String {
        use pecos::prelude::Pauli as RustPauli;
        let shape = arr.shape();
        let ndim = shape.len();

        match ndim {
            1 => {
                // 1D: [Pauli.X, Pauli.Z, Pauli.Y]
                let elements: Vec<String> = arr
                    .iter()
                    .map(|p| {
                        let rust_pauli: RustPauli = unsafe { std::mem::transmute_copy(p) };
                        match rust_pauli {
                            RustPauli::I => "Pauli.I",
                            RustPauli::X => "Pauli.X",
                            RustPauli::Y => "Pauli.Y",
                            RustPauli::Z => "Pauli.Z",
                        }
                        .to_string()
                    })
                    .collect();
                format!("[{}]", elements.join(", "))
            }
            _ => {
                // For 2D+ Pauli, just show shape and dtype
                format!("Array(shape={shape:?}, dtype=pauli)")
            }
        }
    }

    /// Format a `PauliString` array
    fn format_array_paulistring(arr: &ArrayD<crate::pauli_bindings::PauliString>) -> String {
        let shape = arr.shape();
        let ndim = shape.len();

        match ndim {
            1 => {
                // 1D: [PauliString(...), PauliString(...)]
                let elements: Vec<String> = arr.iter().map(|p| format!("{p:?}")).collect();
                format!("[{}]", elements.join(", "))
            }
            _ => {
                // For 2D+ PauliString, just show shape and dtype
                format!("Array(shape={shape:?}, dtype=paulistring)")
            }
        }
    }

    /// Extract scalar value from a 0-dimensional array
    /// Returns the actual Python scalar instead of an Array wrapper
    fn extract_scalar(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if !self.data.shape().is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Cannot extract scalar from non-zero-dimensional array",
            ));
        }

        match &self.data {
            ArrayData::Bool(arr) => {
                let val = *arr.first().unwrap();
                Ok(PyBool::new(py, val).to_owned().into_any().unbind())
            }
            ArrayData::I8(arr) => {
                let val = i64::from(*arr.first().unwrap());
                Ok(PyInt::new(py, val).clone().into_any().unbind())
            }
            ArrayData::I16(arr) => {
                let val = i64::from(*arr.first().unwrap());
                Ok(PyInt::new(py, val).clone().into_any().unbind())
            }
            ArrayData::I32(arr) => {
                let val = i64::from(*arr.first().unwrap());
                Ok(PyInt::new(py, val).clone().into_any().unbind())
            }
            ArrayData::I64(arr) => {
                let val = *arr.first().unwrap();
                Ok(PyInt::new(py, val).clone().into_any().unbind())
            }
            ArrayData::U8(arr) => {
                let val = u64::from(*arr.first().unwrap());
                Ok(PyInt::new(py, val).clone().into_any().unbind())
            }
            ArrayData::U16(arr) => {
                let val = u64::from(*arr.first().unwrap());
                Ok(PyInt::new(py, val).clone().into_any().unbind())
            }
            ArrayData::U32(arr) => {
                let val = u64::from(*arr.first().unwrap());
                Ok(PyInt::new(py, val).clone().into_any().unbind())
            }
            ArrayData::U64(arr) => {
                let val = *arr.first().unwrap();
                Ok(PyInt::new(py, val).clone().into_any().unbind())
            }
            ArrayData::F32(arr) => {
                let val = f64::from(*arr.first().unwrap());
                Ok(PyFloat::new(py, val).clone().into_any().unbind())
            }
            ArrayData::F64(arr) => {
                let val = *arr.first().unwrap();
                Ok(PyFloat::new(py, val).clone().into_any().unbind())
            }
            ArrayData::Complex64(arr) => {
                let val = arr.first().unwrap();
                Ok(
                    pyo3::types::PyComplex::from_doubles(py, f64::from(val.re), f64::from(val.im))
                        .into(),
                )
            }
            ArrayData::Complex128(arr) => {
                let val = arr.first().unwrap();
                Ok(pyo3::types::PyComplex::from_doubles(py, val.re, val.im).into())
            }
            ArrayData::Pauli(arr) => {
                let val = arr.first().unwrap();
                Ok(Py::new(py, *val)?.into_any())
            }
            ArrayData::PauliString(arr) => {
                let val = arr.first().unwrap();
                Ok(Py::new(py, val.clone())?.into_any())
            }
        }
    }

    /// Apply mixed integer/slice indexing leveraging ndarray's `index_axis` and `slice_axis`
    /// This method handles cases like arr[0, 1:3] or arr[:, 0]
    /// where some dimensions are indexed by integers (reducing dimensionality)
    /// and others are sliced (preserving dimensionality)
    fn apply_mixed_indexing(&self, index_ops: &[IndexOp]) -> PyResult<Self> {
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
            ArrayData::Bool(arr) => apply_mixed_indexing_impl!(arr, Bool),
            ArrayData::F64(arr) => apply_mixed_indexing_impl!(arr, F64),
            ArrayData::F32(arr) => apply_mixed_indexing_impl!(arr, F32),
            ArrayData::I64(arr) => apply_mixed_indexing_impl!(arr, I64),
            ArrayData::I32(arr) => apply_mixed_indexing_impl!(arr, I32),
            ArrayData::I16(arr) => apply_mixed_indexing_impl!(arr, I16),
            ArrayData::I8(arr) => apply_mixed_indexing_impl!(arr, I8),
            ArrayData::U64(arr) => apply_mixed_indexing_impl!(arr, U64),
            ArrayData::U32(arr) => apply_mixed_indexing_impl!(arr, U32),
            ArrayData::U16(arr) => apply_mixed_indexing_impl!(arr, U16),
            ArrayData::U8(arr) => apply_mixed_indexing_impl!(arr, U8),
            ArrayData::Complex128(arr) => apply_mixed_indexing_impl!(arr, Complex128),
            ArrayData::Complex64(arr) => apply_mixed_indexing_impl!(arr, Complex64),
            ArrayData::Pauli(arr) => apply_mixed_indexing_impl!(arr, Pauli),
            ArrayData::PauliString(arr) => apply_mixed_indexing_impl!(arr, PauliString),
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
        // Macro to generate the mixed indexing assignment logic for each dtype
        macro_rules! apply_mixed_indexing_assignment_impl {
            ($arr:expr, $dtype:ty, $variant:ident) => {{
                // Strategy: Convert integers to single-element slices, then use slice_each_axis_mut
                // This avoids the borrow checker issues with chaining mutable slices

                use ndarray::SliceInfoElem;

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
                    } else if let Ok(np_arr) =
                        Self::extract_array_for_dtype::<$dtype>(value, stringify!($variant))
                    {
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
                    } else if let Ok(np_arr) =
                        Self::extract_array_for_dtype::<$dtype>(value, stringify!($variant))
                    {
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
            ArrayData::Bool(arr) => apply_mixed_indexing_assignment_impl!(arr, bool, Bool),
            ArrayData::F64(arr) => apply_mixed_indexing_assignment_impl!(arr, f64, Float64),
            ArrayData::F32(arr) => apply_mixed_indexing_assignment_impl!(arr, f32, Float32),
            ArrayData::I64(arr) => apply_mixed_indexing_assignment_impl!(arr, i64, Int64),
            ArrayData::I32(arr) => apply_mixed_indexing_assignment_impl!(arr, i32, Int32),
            ArrayData::I16(arr) => apply_mixed_indexing_assignment_impl!(arr, i16, Int16),
            ArrayData::I8(arr) => apply_mixed_indexing_assignment_impl!(arr, i8, Int8),
            ArrayData::U64(arr) => apply_mixed_indexing_assignment_impl!(arr, u64, Uint64),
            ArrayData::U32(arr) => apply_mixed_indexing_assignment_impl!(arr, u32, Uint32),
            ArrayData::U16(arr) => apply_mixed_indexing_assignment_impl!(arr, u16, Uint16),
            ArrayData::U8(arr) => apply_mixed_indexing_assignment_impl!(arr, u8, Uint8),
            ArrayData::Complex128(arr) => {
                apply_mixed_indexing_assignment_impl!(arr, num_complex::Complex<f64>, Complex128)
            }
            ArrayData::Complex64(arr) => {
                apply_mixed_indexing_assignment_impl!(arr, num_complex::Complex<f32>, Complex64)
            }
            ArrayData::Pauli(_) => Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "Mixed integer/slice indexing assignment not yet implemented for Pauli arrays",
            )),
            ArrayData::PauliString(_) => Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "Mixed integer/slice indexing assignment not yet implemented for PauliString arrays",
            )),
        }
    }

    // Helper method: Extract array from Python based on dtype variant name
    fn extract_array_for_dtype<T: Clone>(
        value: &Bound<'_, PyAny>,
        variant: &str,
    ) -> PyResult<ndarray::ArrayD<T>> {
        use crate::array_buffer;

        // Map variant name to appropriate extraction function
        match variant {
            "Bool" => {
                let arr = array_buffer::extract_bool_array(value)?;
                // SAFETY: We know T is bool based on the macro invocation
                let transmuted = unsafe { std::mem::transmute_copy(&arr) };
                std::mem::forget(arr);
                Ok(transmuted)
            }
            "Float64" => {
                let arr = array_buffer::extract_f64_array(value)?;
                let transmuted = unsafe { std::mem::transmute_copy(&arr) };
                std::mem::forget(arr);
                Ok(transmuted)
            }
            "Float32" => {
                let arr = array_buffer::extract_f32_array(value)?;
                let transmuted = unsafe { std::mem::transmute_copy(&arr) };
                std::mem::forget(arr);
                Ok(transmuted)
            }
            "Int64" => {
                let arr = array_buffer::extract_i64_array(value)?;
                let transmuted = unsafe { std::mem::transmute_copy(&arr) };
                std::mem::forget(arr);
                Ok(transmuted)
            }
            "Int32" => {
                let arr = array_buffer::extract_i32_array(value)?;
                let transmuted = unsafe { std::mem::transmute_copy(&arr) };
                std::mem::forget(arr);
                Ok(transmuted)
            }
            "Int16" => {
                let arr = array_buffer::extract_i16_array(value)?;
                let transmuted = unsafe { std::mem::transmute_copy(&arr) };
                std::mem::forget(arr);
                Ok(transmuted)
            }
            "Int8" => {
                let arr = array_buffer::extract_i8_array(value)?;
                let transmuted = unsafe { std::mem::transmute_copy(&arr) };
                std::mem::forget(arr);
                Ok(transmuted)
            }
            "Uint64" => {
                let arr = array_buffer::extract_u64_array(value)?;
                let transmuted = unsafe { std::mem::transmute_copy(&arr) };
                std::mem::forget(arr);
                Ok(transmuted)
            }
            "Uint32" => {
                let arr = array_buffer::extract_u32_array(value)?;
                let transmuted = unsafe { std::mem::transmute_copy(&arr) };
                std::mem::forget(arr);
                Ok(transmuted)
            }
            "Uint16" => {
                let arr = array_buffer::extract_u16_array(value)?;
                let transmuted = unsafe { std::mem::transmute_copy(&arr) };
                std::mem::forget(arr);
                Ok(transmuted)
            }
            "Uint8" => {
                let arr = array_buffer::extract_u8_array(value)?;
                let transmuted = unsafe { std::mem::transmute_copy(&arr) };
                std::mem::forget(arr);
                Ok(transmuted)
            }
            "Complex128" => {
                let arr = array_buffer::extract_complex64_array(value)?;
                let transmuted = unsafe { std::mem::transmute_copy(&arr) };
                std::mem::forget(arr);
                Ok(transmuted)
            }
            "Complex64" => {
                let arr = array_buffer::extract_complex32_array(value)?;
                let transmuted = unsafe { std::mem::transmute_copy(&arr) };
                std::mem::forget(arr);
                Ok(transmuted)
            }
            _ => Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Unsupported dtype variant for array extraction: {variant}"
            ))),
        }
    }

    // Helper method: Assign a scalar value to all indices specified by ranges
    fn assign_to_mixed_indices<T: Clone>(
        arr: &mut ndarray::ArrayD<T>,
        ranges: &[Vec<usize>],
        value: T,
    ) {
        // Recursively iterate through all combinations of indices
        fn assign_recursive<T: Clone>(
            arr: &mut ndarray::ArrayD<T>,
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
        arr: &mut ndarray::ArrayD<T>,
        ranges: &[Vec<usize>],
        integer_axes: &[usize],
        source: &ndarray::ArrayD<T>,
    ) -> PyResult<()> {
        use ndarray::IxDyn;

        // Recursively iterate through all combinations of indices
        fn assign_array_recursive<T: Clone>(
            arr: &mut ndarray::ArrayD<T>,
            ranges: &[Vec<usize>],
            integer_axes: &[usize],
            source: &ndarray::ArrayD<T>,
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

/// Create an array from a Python sequence or `NumPy` array
///
/// This is a convenience function that wraps the Array constructor,
/// providing a NumPy-like interface without using `NumPy` in the implementation.
///
/// Args:
///     data: A `NumPy` array or Python sequence (list/tuple)
///     dtype: Optional dtype specification (`DType` enum or None for auto-detection)
///
/// Returns:
///     A new Array wrapping the data
///
/// Examples:
///     >>> from `pecos_rslib` import array, Pauli
///     >>> arr = array([1.0, 2.0, 3.0])
///     >>> `pauli_arr` = array([Pauli.X, Pauli.Y, Pauli.Z])
#[pyfunction]
#[pyo3(signature = (data, dtype=None))]
pub fn array(data: &Bound<'_, PyAny>, dtype: Option<&Bound<'_, PyAny>>) -> PyResult<Array> {
    Array::from_python_value(data, dtype)
}
