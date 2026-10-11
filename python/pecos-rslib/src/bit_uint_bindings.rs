// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Python bindings for the `BitUInt` unsigned fixed-width integer type.

use crate::bit_conversion;
use crate::bit_int_bindings::PyBitInt;
use crate::prelude::BitUInt;
use pyo3::basic::CompareOp;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyString};

/// Convert an inbound value, preserving same-class widths for the core operators.
fn operand_to_bituint(size: u16, other: &Bound<'_, PyAny>) -> PyResult<BitUInt> {
    if let Ok(value) = other.extract::<PyRef<PyBitUInt>>() {
        return Ok(value.inner.clone());
    }
    if let Ok(value) = other.extract::<PyRef<PyBitInt>>() {
        return Ok(value.inner.inner().resize_sign_extend(size));
    }
    if let Ok(value) = other.extract::<u64>() {
        return Ok(BitUInt::new(size, value));
    }
    if let Ok(value) = other.extract::<i64>() {
        return Ok(BitUInt::new(64, value.cast_unsigned()).resize_sign_extend(size));
    }
    let value = bit_conversion::operand_to_pyint(other)?;
    let words = bit_conversion::pyint_to_u64_words(&value, size)?;
    Ok(BitUInt::from_raw_words(size, words.into_boxed_slice()))
}

/// Convert a constructor or setter value, wrapping it to the declared width.
fn value_to_bituint(size: u16, value: &Bound<'_, PyAny>) -> PyResult<BitUInt> {
    let operand = operand_to_bituint(size, value)?;
    if operand.size() == size {
        return Ok(operand);
    }
    Ok(BitUInt::from_raw_words(
        size,
        operand.to_words().into_boxed_slice(),
    ))
}

/// An unsigned fixed-width integer with explicit bit width tracking.
///
/// `BitUInt(N)` stores values in N bits (1-65535) and always returns
/// non-negative values from `int()`.
///
/// Examples:
/// ```python
/// from pecos import BitUInt
///
/// u = BitUInt(1, 1)
/// assert int(u) == 1     # Always non-negative
///
/// u = BitUInt(8, 0b10101010)
/// u = BitUInt("01010101")
/// ```
#[pyclass(name = "BitUInt", from_py_object)]
#[derive(Clone)]
pub struct PyBitUInt {
    pub(crate) inner: BitUInt,
}

#[pymethods]
impl PyBitUInt {
    /// Create a new `BitUInt`.
    ///
    /// Can be called as:
    /// - `BitUInt(size, value=0)` - create with explicit size (1-65535)
    /// - `BitUInt("1010")` - create from binary string (size = string length)
    #[new]
    #[pyo3(signature = (size, value=None))]
    pub fn new(size: &Bound<'_, PyAny>, value: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        if let Ok(s) = size.extract::<String>() {
            return Self::from_binary(&s);
        }

        let size: u16 = size.extract().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "size must be an integer or binary string",
            )
        })?;

        if size == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "BitUInt size must be at least 1",
            ));
        }

        let inner = if let Some(value) = value {
            value_to_bituint(size, value)?
        } else {
            BitUInt::zero(size)
        };

        Ok(PyBitUInt { inner })
    }

    /// Create a `BitUInt` from a binary string.
    #[staticmethod]
    pub fn from_binary(s: &str) -> PyResult<Self> {
        if s.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Binary string must not be empty",
            ));
        }
        if !s.chars().all(|c| c == '0' || c == '1') {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Binary string must contain only '0' and '1' characters",
            ));
        }
        if s.len() > 65535 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Binary string too long",
            ));
        }
        Ok(PyBitUInt {
            inner: BitUInt::from_binary_str(s),
        })
    }

    /// Create a zero value with the given size.
    #[staticmethod]
    pub fn zeros(size: u16) -> PyResult<Self> {
        if size == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "BitUInt size must be at least 1",
            ));
        }
        Ok(PyBitUInt {
            inner: BitUInt::zero(size),
        })
    }

    /// Create an all-ones value with the given size.
    #[staticmethod]
    pub fn ones(size: u16) -> PyResult<Self> {
        if size == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "BitUInt size must be at least 1",
            ));
        }
        Ok(PyBitUInt {
            inner: BitUInt::ones(size),
        })
    }

    /// Returns the bit width.
    #[getter]
    pub fn size(&self) -> u16 {
        self.inner.size()
    }

    /// Always returns False (unsigned).
    #[getter]
    #[allow(clippy::unused_self)] // Python instance method
    pub fn signed(&self) -> bool {
        false
    }

    /// Returns the value as a Python int (always non-negative).
    pub fn to_int(&self) -> Option<i64> {
        self.inner.to_i64()
    }

    /// Set the value.
    pub fn set(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let size = self.inner.size();
        self.inner = value_to_bituint(size, value)?;
        Ok(())
    }

    /// Get a specific bit value.
    pub fn get_bit(&self, index: u16) -> PyResult<bool> {
        if index >= self.inner.size() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Bit index {} out of bounds for size {}",
                index,
                self.inner.size()
            )));
        }
        Ok(self.inner.get_bit(index))
    }

    /// Set a specific bit value.
    pub fn set_bit(&mut self, index: u16, value: bool) -> PyResult<()> {
        if index >= self.inner.size() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Bit index {} out of bounds for size {}",
                index,
                self.inner.size()
            )));
        }
        self.inner.set_bit(index, value);
        Ok(())
    }

    /// Returns the number of 1 bits.
    pub fn count_ones(&self) -> u32 {
        self.inner.count_ones()
    }

    /// Returns the number of 0 bits.
    pub fn count_zeros(&self) -> u32 {
        self.inner.count_zeros()
    }

    /// Returns True if the value is zero.
    pub fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

    /// Returns the number of bits required to represent the current value.
    pub fn num_bits(&self) -> u32 {
        if let Some(val) = self.inner.to_u64() {
            if val == 0 {
                1
            } else {
                64 - val.leading_zeros()
            }
        } else {
            u32::from(self.inner.size())
        }
    }

    /// Clamp the value to fit within the specified bit size.
    pub fn clamp(&mut self, size: u16) {
        if size < self.inner.size() {
            if size == 0 {
                self.inner = BitUInt::zero(self.inner.size());
            } else {
                let pattern =
                    BitUInt::from_raw_words(size, self.inner.to_words().into_boxed_slice());
                self.inner = BitUInt::from_raw_words(
                    self.inner.size(),
                    pattern.to_words().into_boxed_slice(),
                );
            }
        }
    }

    /// Set value with clipping to fit within the allocated size.
    pub fn set_clip(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.set(value)
    }

    // ========================================================================
    // Bitwise operations
    // ========================================================================

    pub fn __xor__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitUInt> {
        let other_uint = operand_to_bituint(self.inner.size(), other)?;
        Ok(PyBitUInt {
            inner: &self.inner ^ &other_uint,
        })
    }

    pub fn __rxor__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitUInt> {
        self.__xor__(other)
    }

    pub fn __and__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitUInt> {
        let other_uint = operand_to_bituint(self.inner.size(), other)?;
        Ok(PyBitUInt {
            inner: &self.inner & &other_uint,
        })
    }

    pub fn __rand__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitUInt> {
        self.__and__(other)
    }

    pub fn __or__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitUInt> {
        let other_uint = operand_to_bituint(self.inner.size(), other)?;
        Ok(PyBitUInt {
            inner: &self.inner | &other_uint,
        })
    }

    pub fn __ror__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitUInt> {
        self.__or__(other)
    }

    pub fn __invert__(&self) -> PyBitUInt {
        PyBitUInt {
            inner: !&self.inner,
        }
    }

    pub fn __lshift__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitUInt> {
        let n = bit_conversion::shift_count(other)?;
        Ok(PyBitUInt {
            inner: &self.inner << n,
        })
    }

    pub fn __rshift__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitUInt> {
        let n = bit_conversion::shift_count(other)?;
        Ok(PyBitUInt {
            inner: &self.inner >> n,
        })
    }

    // ========================================================================
    // Arithmetic operations
    // ========================================================================

    pub fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitUInt> {
        let other_uint = operand_to_bituint(self.inner.size(), other)?;
        Ok(PyBitUInt {
            inner: &self.inner + &other_uint,
        })
    }

    pub fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitUInt> {
        self.__add__(other)
    }

    pub fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitUInt> {
        let other_uint = operand_to_bituint(self.inner.size(), other)?;
        Ok(PyBitUInt {
            inner: &self.inner - &other_uint,
        })
    }

    pub fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitUInt> {
        let other_uint = operand_to_bituint(self.inner.size(), other)?;
        Ok(PyBitUInt {
            inner: &other_uint - &self.inner,
        })
    }

    pub fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitUInt> {
        let other_uint = operand_to_bituint(self.inner.size(), other)?;
        Ok(PyBitUInt {
            inner: &self.inner * &other_uint,
        })
    }

    pub fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitUInt> {
        self.__mul__(other)
    }

    pub fn __floordiv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitUInt> {
        let other_uint = operand_to_bituint(self.inner.size(), other)?;
        if other_uint.is_zero() {
            return Err(PyErr::new::<pyo3::exceptions::PyZeroDivisionError, _>(
                "division by zero",
            ));
        }
        Ok(PyBitUInt {
            inner: &self.inner / &other_uint,
        })
    }

    pub fn __mod__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitUInt> {
        let other_uint = operand_to_bituint(self.inner.size(), other)?;
        if other_uint.is_zero() {
            return Err(PyErr::new::<pyo3::exceptions::PyZeroDivisionError, _>(
                "modulo by zero",
            ));
        }
        Ok(PyBitUInt {
            inner: &self.inner % &other_uint,
        })
    }

    // ========================================================================
    // Comparison
    // ========================================================================

    pub fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let result = if let Ok(value) = other.extract::<PyRef<PyBitUInt>>() {
            op.matches(self.inner.cmp(&value.inner))
        } else if other.is_instance_of::<PyString>() || other.get_type().hasattr("__index__")? {
            self.__int__(py)?
                .rich_compare(bit_conversion::operand_to_pyint(other)?, op)?
                .is_truthy()?
        } else {
            // An operand with no integer value is not comparable: equality then
            // falls back to identity, so containers holding floats keep working.
            return Ok(py.NotImplemented());
        };
        Ok(PyBool::new(py, result).to_owned().into_any().unbind())
    }

    pub fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        self.__int__(py)?.hash()
    }

    pub fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    pub fn __repr__(&self) -> String {
        format!("BitUInt({}, 0b{})", self.inner.size(), self.inner)
    }

    #[pyo3(signature = (reverse_bits=false, separator=None))]
    pub fn to_binary_str(&self, reverse_bits: bool, separator: Option<&str>) -> String {
        let size = self.inner.size();
        let mut bits = Vec::with_capacity(size as usize);

        if reverse_bits {
            for i in 0..size {
                bits.push(if self.inner.get_bit(i) { '1' } else { '0' });
            }
        } else {
            for i in (0..size).rev() {
                bits.push(if self.inner.get_bit(i) { '1' } else { '0' });
            }
        }

        match separator {
            Some(sep) => bits
                .iter()
                .map(std::string::ToString::to_string)
                .collect::<Vec<_>>()
                .join(sep),
            None => bits.into_iter().collect(),
        }
    }

    pub fn __len__(&self) -> usize {
        self.inner.size() as usize
    }

    #[allow(clippy::cast_possible_wrap)]
    pub fn __getitem__(&self, index: isize) -> PyResult<i32> {
        let size = self.inner.size() as isize;
        let idx = if index < 0 { size + index } else { index };

        if idx < 0 || idx >= size {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Bit index {index} out of bounds for size {size}"
            )));
        }

        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        Ok(i32::from(self.inner.get_bit(idx as u16)))
    }

    #[allow(clippy::cast_possible_wrap)]
    pub fn __setitem__(&mut self, index: isize, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let size = self.inner.size() as isize;
        let idx = if index < 0 { size + index } else { index };

        if idx < 0 || idx >= size {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Bit index {index} out of bounds for size {size}"
            )));
        }

        let bit_value = if let Ok(v) = value.extract::<i64>() {
            v != 0
        } else if let Ok(v) = value.extract::<bool>() {
            v
        } else if let Ok(s) = value.extract::<String>() {
            match s.as_str() {
                "0" => false,
                "1" => true,
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "String value must be '0' or '1'",
                    ));
                }
            }
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Value must be int, bool, or '0'/'1' string",
            ));
        };

        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        self.inner.set_bit(idx as u16, bit_value);
        Ok(())
    }

    pub fn __bool__(&self) -> bool {
        !self.inner.is_zero()
    }

    /// Returns the unsigned integer value as a Python int (arbitrary precision).
    pub(crate) fn __int__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        // Fast path: value fits in u64
        if let Some(val) = self.inner.to_u64() {
            return Ok(val
                .into_pyobject(py)
                .expect("u64 to Python conversion failed")
                .into_any());
        }
        // Slow path: arbitrary precision
        let words = self.inner.to_words();
        bit_conversion::u64_words_to_pyint(py, &words)
    }

    fn __index__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.__int__(py)
    }
}
