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

//! Python bindings for the `BitInt` fixed-width signed integer type.

use crate::bit_conversion;
use crate::prelude::BitInt;
use pyo3::basic::CompareOp;
use pyo3::prelude::*;
use pyo3::types::PyInt;

/// Helper to extract a u64 value from Python objects (`BitInt`, `BitUInt`, int, or str).
fn extract_operand_value(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    if let Ok(bit_int) = obj.extract::<PyRef<PyBitInt>>() {
        return Ok(bit_int.inner.to_u64().unwrap_or(0));
    }

    // Try BitUInt
    if let Ok(bit_uint) = obj.extract::<PyRef<crate::bit_uint_bindings::PyBitUInt>>() {
        return Ok(bit_uint.inner.to_u64().unwrap_or(0));
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

    // Large Python int (doesn't fit in i64/u64): extract lower 64 bits
    if obj.is_instance_of::<PyInt>() {
        let words = bit_conversion::pyint_to_u64_words(obj, 1)?;
        return Ok(words[0]);
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "Operand must be BitInt, BitUInt, int, or binary string",
    ))
}

/// A fixed-width signed integer with explicit bit width tracking.
///
/// `BitInt(N)` is always signed. Internally wraps `BitUInt(N+1)` where
/// the extra bit is the sign bit.
///
/// Examples:
/// ```python
/// from pecos import BitInt
///
/// a = BitInt(8, 42)
/// assert int(a) == 42
///
/// b = BitInt(1, 1)
/// assert int(b) == 1    # Not -1 (extra sign bit)
///
/// c = BitInt(1, -1)
/// assert int(c) == -1
/// ```
#[pyclass(name = "BitInt", from_py_object)]
#[derive(Clone)]
pub struct PyBitInt {
    pub(crate) inner: BitInt,
}

/// Helper methods for `PyBitInt` not exposed to Python.
impl PyBitInt {
    /// Helper to create `BitInt` from operand.
    fn operand_to_bitint(&self, other: &Bound<'_, PyAny>) -> PyResult<BitInt> {
        if let Ok(bit_int) = other.extract::<PyRef<PyBitInt>>() {
            return Ok(bit_int.inner.clone());
        }

        if let Ok(bit_uint) = other.extract::<PyRef<crate::bit_uint_bindings::PyBitUInt>>() {
            let val = bit_uint.inner.to_u64().unwrap_or(0);
            #[allow(clippy::cast_possible_wrap)]
            return Ok(BitInt::new(self.inner.size(), val as i64));
        }

        if let Ok(value) = other.extract::<i64>() {
            return Ok(BitInt::new(self.inner.size(), value));
        }

        if let Ok(value) = other.extract::<u64>() {
            #[allow(clippy::cast_possible_wrap)]
            return Ok(BitInt::new(self.inner.size(), value as i64));
        }

        if let Ok(s) = other.extract::<String>() {
            let stripped = s
                .strip_prefix("0b")
                .or_else(|| s.strip_prefix("0B"))
                .unwrap_or(&s);

            if stripped.chars().all(|c| c == '0' || c == '1') {
                let val = u64::from_str_radix(stripped, 2).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Invalid binary string: {e}"
                    ))
                })?;
                return Ok(BitInt::new_from_u64(self.inner.size(), val));
            }
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "String must contain only '0' and '1' characters",
            ));
        }

        // Large Python int (doesn't fit in i64/u64)
        if other.is_instance_of::<PyInt>() {
            let size = self.inner.size();
            let internal_size = size + 1;
            let n_words = (internal_size as usize).div_ceil(64);
            let words = bit_conversion::pyint_to_u64_words(other, n_words)?;
            return Ok(BitInt::new_from_raw_inner(size, words.into_boxed_slice()));
        }

        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Operand must be BitInt, BitUInt, int, or binary string",
        ))
    }

    /// Get the signed integer value (for use by `PyBitUInt` bindings).
    pub fn to_int(&self) -> Option<i64> {
        self.inner.to_i64()
    }
}

#[pymethods]
impl PyBitInt {
    /// Create a new `BitInt`.
    ///
    /// Can be called as:
    /// - `BitInt(size, value=0)` - create with explicit size (1-65534)
    /// - `BitInt("1010")` - create from binary string (size = string length)
    #[new]
    #[pyo3(signature = (size, value=None))]
    pub fn new(size: &Bound<'_, PyAny>, value: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        // Check if size is a string (binary string constructor)
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

            let val = u64::from_str_radix(s, 2).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid binary string: {e}"
                ))
            })?;

            let size_u16 = u16::try_from(s.len()).map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Binary string exceeds maximum BitInt size (65534 bits)",
                )
            })?;

            if size_u16 > 65534 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "BitInt size must be at most 65534",
                ));
            }

            return Ok(PyBitInt {
                inner: BitInt::new_from_u64(size_u16, val),
            });
        }

        // Otherwise, size should be an integer
        let size: u16 = size.extract().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "size must be an integer or binary string",
            )
        })?;

        if size == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "`BitInt` size must be at least 1",
            ));
        }

        if size > 65534 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "BitInt size must be at most 65534",
            ));
        }

        let inner = if let Some(val_obj) = value {
            // Fast path: value fits in i64
            if let Ok(v) = val_obj.extract::<i64>() {
                BitInt::new(size, v)
            } else {
                // Arbitrary-precision Python int
                let internal_size = size + 1;
                let n_words = (internal_size as usize).div_ceil(64);
                let words = bit_conversion::pyint_to_u64_words(val_obj, n_words)?;
                BitInt::new_from_raw_inner(size, words.into_boxed_slice())
            }
        } else {
            BitInt::zero(size)
        };

        Ok(PyBitInt { inner })
    }

    /// Create a `BitInt` from a binary string.
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

        Ok(PyBitInt {
            inner: BitInt::from_binary_str(s),
        })
    }

    /// Create a zero value with the given size.
    #[staticmethod]
    pub fn zeros(size: u16) -> PyResult<Self> {
        if size == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "`BitInt` size must be at least 1",
            ));
        }
        Ok(PyBitInt {
            inner: BitInt::zero(size),
        })
    }

    /// Create an all-ones value with the given size.
    #[staticmethod]
    pub fn ones(size: u16) -> PyResult<Self> {
        if size == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "`BitInt` size must be at least 1",
            ));
        }
        Ok(PyBitInt {
            inner: BitInt::ones(size),
        })
    }

    #[getter]
    pub fn size(&self) -> u16 {
        self.inner.size()
    }

    /// Always returns True (signed).
    #[getter]
    #[allow(clippy::unused_self)] // Python instance method
    pub fn signed(&self) -> bool {
        true
    }

    pub fn set(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let size = self.inner.size();
        if let Ok(v) = value.extract::<i64>() {
            self.inner = BitInt::new(size, v);
        } else {
            let internal_size = size + 1;
            let n_words = (internal_size as usize).div_ceil(64);
            let words = bit_conversion::pyint_to_u64_words(value, n_words)?;
            self.inner = BitInt::new_from_raw_inner(size, words.into_boxed_slice());
        }
        Ok(())
    }

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

    pub fn count_ones(&self) -> u32 {
        self.inner.count_ones()
    }

    pub fn count_zeros(&self) -> u32 {
        self.inner.count_zeros()
    }

    pub fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }

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

    pub fn clamp(&mut self, size: u16) {
        if size < self.inner.size()
            && let Some(v) = self.inner.to_i64()
        {
            let mask: i64 = if size >= 63 {
                i64::MAX
            } else {
                (1i64 << size) - 1
            };
            self.inner = BitInt::new(self.inner.size(), v & mask);
        }
    }

    pub fn set_clip(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.set(value)
    }

    // Bitwise operations
    pub fn __xor__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitInt> {
        let other_int = self.operand_to_bitint(other)?;
        Ok(PyBitInt {
            inner: &self.inner ^ &other_int,
        })
    }

    pub fn __rxor__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitInt> {
        self.__xor__(other)
    }

    pub fn __and__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitInt> {
        let other_int = self.operand_to_bitint(other)?;
        Ok(PyBitInt {
            inner: &self.inner & &other_int,
        })
    }

    pub fn __rand__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitInt> {
        self.__and__(other)
    }

    pub fn __or__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitInt> {
        let other_int = self.operand_to_bitint(other)?;
        Ok(PyBitInt {
            inner: &self.inner | &other_int,
        })
    }

    pub fn __ror__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitInt> {
        self.__or__(other)
    }

    pub fn __invert__(&self) -> PyBitInt {
        PyBitInt {
            inner: !&self.inner,
        }
    }

    pub fn __lshift__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitInt> {
        let n = extract_operand_value(other)?;
        #[allow(clippy::cast_possible_truncation)]
        let n = n as u16;
        Ok(PyBitInt {
            inner: &self.inner << n,
        })
    }

    pub fn __rshift__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitInt> {
        let n = extract_operand_value(other)?;
        #[allow(clippy::cast_possible_truncation)]
        let n = n as u16;
        Ok(PyBitInt {
            inner: &self.inner >> n,
        })
    }

    // Arithmetic operations
    pub fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitInt> {
        let other_int = self.operand_to_bitint(other)?;
        Ok(PyBitInt {
            inner: &self.inner + &other_int,
        })
    }

    pub fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitInt> {
        self.__add__(other)
    }

    pub fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitInt> {
        let other_int = self.operand_to_bitint(other)?;
        Ok(PyBitInt {
            inner: &self.inner - &other_int,
        })
    }

    pub fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitInt> {
        let other_int = self.operand_to_bitint(other)?;
        Ok(PyBitInt {
            inner: &other_int - &self.inner,
        })
    }

    pub fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitInt> {
        let other_int = self.operand_to_bitint(other)?;
        Ok(PyBitInt {
            inner: &self.inner * &other_int,
        })
    }

    pub fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitInt> {
        self.__mul__(other)
    }

    pub fn __floordiv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitInt> {
        let other_int = self.operand_to_bitint(other)?;
        if other_int.is_zero() {
            return Err(PyErr::new::<pyo3::exceptions::PyZeroDivisionError, _>(
                "division by zero",
            ));
        }
        Ok(PyBitInt {
            inner: &self.inner / &other_int,
        })
    }

    pub fn __mod__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitInt> {
        let other_int = self.operand_to_bitint(other)?;
        if other_int.is_zero() {
            return Err(PyErr::new::<pyo3::exceptions::PyZeroDivisionError, _>(
                "modulo by zero",
            ));
        }
        Ok(PyBitInt {
            inner: &self.inner % &other_int,
        })
    }

    // Comparison operations (always signed)
    pub fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<bool> {
        let other_int = self.operand_to_bitint(other)?;
        Ok(match op {
            CompareOp::Eq => self.inner == other_int,
            CompareOp::Ne => self.inner != other_int,
            CompareOp::Lt => self.inner < other_int,
            CompareOp::Le => self.inner <= other_int,
            CompareOp::Gt => self.inner > other_int,
            CompareOp::Ge => self.inner >= other_int,
        })
    }

    pub fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.size().hash(&mut hasher);
        true.hash(&mut hasher); // signed = true
        if let Some(val) = self.inner.to_i64() {
            val.hash(&mut hasher);
        } else {
            for word in self.inner.inner_words() {
                word.hash(&mut hasher);
            }
        }
        hasher.finish()
    }

    pub fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    pub fn __repr__(&self) -> String {
        format!("BitInt({}, 0b{})", self.inner.size(), self.inner,)
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

    /// Returns the signed integer value as a Python int (arbitrary precision).
    fn __int__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        // Fast path: value fits in i64
        if let Some(val) = self.inner.to_i64() {
            return Ok(val
                .into_pyobject(py)
                .expect("i64 to Python conversion failed")
                .into_any());
        }
        // Slow path: arbitrary precision
        let words = self.inner.inner_words();
        let internal_size = self.inner.size() + 1;
        bit_conversion::u64_words_to_pyint_signed(py, &words, internal_size)
    }

    fn __index__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.__int__(py)
    }
}
