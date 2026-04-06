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
use pyo3::types::PyInt;

/// Helper to extract a u64 value from Python objects (`BitUInt`, `BitInt`, int, or str).
fn extract_uint_operand_value(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    if let Ok(bit_uint) = obj.extract::<PyRef<PyBitUInt>>() {
        return Ok(bit_uint.inner.to_u64().unwrap_or(0));
    }

    if let Ok(bit_int) = obj.extract::<PyRef<PyBitInt>>() {
        let val = bit_int.to_int().unwrap_or(0);
        #[allow(clippy::cast_sign_loss)]
        return Ok(val as u64);
    }

    if let Ok(value) = obj.extract::<u64>() {
        return Ok(value);
    }

    if let Ok(value) = obj.extract::<i64>() {
        #[allow(clippy::cast_sign_loss)]
        return Ok(value as u64);
    }

    if let Ok(s) = obj.extract::<String>() {
        let stripped = s
            .strip_prefix("0b")
            .or_else(|| s.strip_prefix("0B"))
            .unwrap_or(&s);

        if stripped.chars().all(|c| c == '0' || c == '1') {
            return u64::from_str_radix(stripped, 2).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid binary string: {e}"
                ))
            });
        }
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "String must contain only '0' and '1' characters",
        ));
    }

    // Large Python int (doesn't fit in u64/i64): extract lower 64 bits
    if obj.is_instance_of::<PyInt>() {
        let words = bit_conversion::pyint_to_u64_words(obj, 1)?;
        return Ok(words[0]);
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "Operand must be BitUInt, BitInt, int, or binary string",
    ))
}

/// Helper to convert an operand to a `BitUInt` with the given size.
fn operand_to_bituint(size: u16, other: &Bound<'_, PyAny>) -> PyResult<BitUInt> {
    if let Ok(bit_uint) = other.extract::<PyRef<PyBitUInt>>() {
        return Ok(bit_uint.inner.clone());
    }

    // Fast path for values fitting in u64/i64
    if let Ok(val) = extract_uint_operand_value(other) {
        return Ok(BitUInt::new(size, val));
    }

    // Large Python int
    if other.is_instance_of::<PyInt>() {
        let n_words = (size as usize).div_ceil(64);
        let words = bit_conversion::pyint_to_u64_words(other, n_words)?;
        return Ok(BitUInt::from_raw_words(size, words.into_boxed_slice()));
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "Operand must be BitUInt, BitInt, int, or binary string",
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
            let s = s.as_str();
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

            let size_u16 = u16::try_from(s.len()).map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Binary string exceeds maximum BitUInt size",
                )
            })?;

            let val = u64::from_str_radix(s, 2).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid binary string: {e}"
                ))
            })?;

            return Ok(PyBitUInt {
                inner: BitUInt::new(size_u16, val),
            });
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

        let inner = if let Some(val_obj) = value {
            // Fast path: value fits in u64
            if let Ok(v) = val_obj.extract::<u64>() {
                BitUInt::new(size, v)
            } else if let Ok(v) = val_obj.extract::<i64>() {
                #[allow(clippy::cast_sign_loss)]
                BitUInt::new(size, v as u64)
            } else if val_obj.is_instance_of::<PyInt>() {
                // Arbitrary-precision Python int
                let n_words = (size as usize).div_ceil(64);
                let words = bit_conversion::pyint_to_u64_words(val_obj, n_words)?;
                BitUInt::from_raw_words(size, words.into_boxed_slice())
            } else {
                let v = extract_uint_operand_value(val_obj)?;
                BitUInt::new(size, v)
            }
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
        let size = u16::try_from(s.len()).map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>("Binary string too long")
        })?;
        let val = u64::from_str_radix(s, 2).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Invalid binary string: {e}"))
        })?;
        Ok(PyBitUInt {
            inner: BitUInt::new(size, val),
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
        if let Ok(v) = value.extract::<u64>() {
            self.inner = BitUInt::new(size, v);
        } else if let Ok(v) = value.extract::<i64>() {
            #[allow(clippy::cast_sign_loss)]
            {
                self.inner = BitUInt::new(size, v as u64);
            }
        } else {
            let n_words = (size as usize).div_ceil(64);
            let words = bit_conversion::pyint_to_u64_words(value, n_words)?;
            self.inner = BitUInt::from_raw_words(size, words.into_boxed_slice());
        }
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
        if size < self.inner.size()
            && let Some(v) = self.inner.to_u64()
        {
            let mask = if size >= 64 {
                u64::MAX
            } else {
                (1u64 << size) - 1
            };
            self.inner = BitUInt::new(self.inner.size(), v & mask);
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
        let n = extract_uint_operand_value(other)?;
        #[allow(clippy::cast_possible_truncation)]
        let n = n as u16;
        Ok(PyBitUInt {
            inner: &self.inner << n,
        })
    }

    pub fn __rshift__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitUInt> {
        let n = extract_uint_operand_value(other)?;
        #[allow(clippy::cast_possible_truncation)]
        let n = n as u16;
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

    pub fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<bool> {
        let other_uint = operand_to_bituint(self.inner.size(), other)?;
        Ok(match op {
            CompareOp::Eq => self.inner == other_uint,
            CompareOp::Ne => self.inner != other_uint,
            CompareOp::Lt => self.inner < other_uint,
            CompareOp::Le => self.inner <= other_uint,
            CompareOp::Gt => self.inner > other_uint,
            CompareOp::Ge => self.inner >= other_uint,
        })
    }

    pub fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.size().hash(&mut hasher);
        false.hash(&mut hasher);
        if let Some(val) = self.inner.to_u64() {
            val.hash(&mut hasher);
        } else {
            for word in self.inner.to_words() {
                word.hash(&mut hasher);
            }
        }
        hasher.finish()
    }

    pub fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    pub fn __repr__(&self) -> String {
        format!("BitUInt({}, 0b{})", self.inner.size(), self.inner,)
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
    fn __int__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
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
