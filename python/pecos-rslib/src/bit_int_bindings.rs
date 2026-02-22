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

//! Python bindings for the `BitInt` fixed-width integer type.
//!
//! This module provides a drop-in replacement for `BinArray` with Rust performance.

use pecos::prelude::BitInt;
use pyo3::basic::CompareOp;
use pyo3::prelude::*;

/// A fixed-width integer with explicit bit width tracking.
///
/// This class provides a Rust-backed implementation of fixed-width integers
/// compatible with `BinArray`. It supports arbitrary bit widths and both
/// signed and unsigned semantics.
///
/// Examples:
/// ```python
/// from pecos import BitInt
///
/// # Create from size and value
/// a = BitInt(8, 0b10101010)
///
/// # Create from binary string (like BinArray)
/// b = BitInt("01010101")
///
/// # Operations work with BitInt, int, or str
/// c = a ^ b           # BitInt ^ BitInt
/// d = a ^ 0b11110000  # BitInt ^ int
/// e = a ^ "11110000"  # BitInt ^ str
///
/// # Bit access
/// a[0]      # Get bit (returns bool)
/// a[1] = 1  # Set bit
/// ```
#[pyclass(name = "BitInt", from_py_object)]
#[derive(Clone)]
pub struct PyBitInt {
    inner: BitInt,
}

/// Helper to extract a u64 value from Python objects (`BitInt`, int, or str).
fn extract_operand_value(obj: &Bound<'_, PyAny>) -> PyResult<u64> {
    // Try BitInt first
    if let Ok(bit_int) = obj.extract::<PyRef<PyBitInt>>() {
        return Ok(bit_int.inner.to_u64().unwrap_or(0));
    }

    // Try int
    if let Ok(value) = obj.extract::<i64>() {
        #[allow(clippy::cast_sign_loss)]
        return Ok(value as u64);
    }

    // Try str (binary string, with optional "0b" prefix)
    if let Ok(s) = obj.extract::<String>() {
        // Strip optional "0b" or "0B" prefix
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

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "Operand must be BitInt, int, or binary string",
    ))
}

/// Helper methods for `PyBitInt` that are not exposed to Python.
impl PyBitInt {
    /// Helper to create `BitInt` from operand with matching signedness to self.
    fn operand_to_bitint(&self, other: &Bound<'_, PyAny>) -> PyResult<BitInt> {
        // If other is already a PyBitInt, use it directly
        if let Ok(bit_int) = other.extract::<PyRef<PyBitInt>>() {
            return Ok(bit_int.inner.clone());
        }

        // For integers, respect signedness of self
        if let Ok(value) = other.extract::<i64>() {
            return Ok(if self.inner.is_signed() {
                BitInt::new_signed(self.inner.size(), value)
            } else {
                #[allow(clippy::cast_sign_loss)]
                BitInt::new_unsigned(self.inner.size(), value as u64)
            });
        }

        // For binary strings
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
                return Ok(if self.inner.is_signed() {
                    #[allow(clippy::cast_possible_wrap)]
                    BitInt::new_signed(self.inner.size(), val as i64)
                } else {
                    BitInt::new_unsigned(self.inner.size(), val)
                });
            }
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "String must contain only '0' and '1' characters",
            ));
        }

        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Operand must be BitInt, int, or binary string",
        ))
    }
}

#[pymethods]
impl PyBitInt {
    /// Create a new `BitInt`.
    ///
    /// Can be called as:
    /// - `BitInt(size, value=0, signed=False)` - create with explicit size
    /// - `BitInt("1010")` - create from binary string (size = string length)
    /// - `BitInt("1010", dtype=pc.u64)` - create from binary string with explicit dtype
    ///
    /// Args:
    ///     size: The bit width (1 to 65535) or a binary string
    ///     value: The initial value (default: 0), ignored if size is a string
    ///     signed: Whether to use signed semantics (default: True for `BinArray` compat)
    ///     dtype: Optional dtype (pc.i64, pc.u64, etc.) to specify signedness
    ///
    /// Returns:
    ///     A new `BitInt` instance
    ///
    /// Raises:
    ///     `ValueError`: If size is 0 or string contains non-binary characters
    #[new]
    #[pyo3(signature = (size, value=0, *, signed=None, dtype=None))]
    pub fn new(
        size: &Bound<'_, PyAny>,
        value: i64,
        signed: Option<bool>,
        dtype: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        // Helper to determine signedness from dtype
        let dtype_is_signed = if let Some(dt) = dtype {
            // Get the type name to determine if it's signed or unsigned
            // dtype can be a class (pc.u64) or an instance
            let type_name = if let Ok(name) = dt.getattr("__name__") {
                // It's a class/type, get __name__
                name.extract::<String>().ok()
            } else {
                // It's an instance, get the class name via __class__.__name__
                dt.get_type().name().ok().map(|s| s.to_string())
            };

            match type_name.as_deref() {
                Some("u8" | "u16" | "u32" | "u64") => Some(false), // unsigned
                Some("i8" | "i16" | "i32" | "i64") => Some(true),  // signed
                _ => None,                                         // unknown, use default
            }
        } else {
            None
        };

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

            // Determine signedness: signed param > dtype > default
            // For binary string construction, default to unsigned (matching BinArray behavior
            // where "1010" gives 10, not -6)
            let is_signed = signed.or(dtype_is_signed).unwrap_or(false);

            // Parse the binary string as unsigned first
            let val = u64::from_str_radix(s, 2).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid binary string: {e}"
                ))
            })?;

            let size_u16 = u16::try_from(s.len()).map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Binary string exceeds maximum BitInt size (65535 bits)",
                )
            })?;
            let inner = if is_signed {
                #[allow(clippy::cast_possible_wrap)]
                BitInt::new_signed(size_u16, val as i64)
            } else {
                BitInt::new_unsigned(size_u16, val)
            };

            return Ok(PyBitInt { inner });
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

        // Determine signedness: signed param > dtype > default (true for BinArray compat)
        let is_signed = signed.or(dtype_is_signed).unwrap_or(true);

        let inner = if is_signed {
            BitInt::new_signed(size, value)
        } else {
            #[allow(clippy::cast_sign_loss)]
            BitInt::new_unsigned(size, value as u64)
        };

        Ok(PyBitInt { inner })
    }

    /// Create a `BitInt` from a binary string.
    ///
    /// Args:
    ///     s: A binary string (e.g., "1010")
    ///
    /// Returns:
    ///     A new unsigned `BitInt` with size equal to the string length
    ///
    /// Raises:
    ///     `ValueError`: If the string is empty or contains non-binary characters
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
    ///
    /// Args:
    ///     size: The bit width
    ///     signed: Whether to use signed semantics (default: False)
    ///
    /// Returns:
    ///     A new `BitInt` with all bits set to 0
    #[staticmethod]
    #[pyo3(signature = (size, signed=false))]
    pub fn zeros(size: u16, signed: bool) -> PyResult<Self> {
        if size == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "`BitInt` size must be at least 1",
            ));
        }
        Ok(PyBitInt {
            inner: BitInt::zero(size, signed),
        })
    }

    /// Create an all-ones value with the given size.
    ///
    /// Args:
    ///     size: The bit width
    ///     signed: Whether to use signed semantics (default: False)
    ///
    /// Returns:
    ///     A new `BitInt` with all bits set to 1
    #[staticmethod]
    #[pyo3(signature = (size, signed=false))]
    pub fn ones(size: u16, signed: bool) -> PyResult<Self> {
        if size == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "`BitInt` size must be at least 1",
            ));
        }
        Ok(PyBitInt {
            inner: BitInt::ones(size, signed),
        })
    }

    /// Returns the bit width of this integer.
    #[getter]
    pub fn size(&self) -> u16 {
        self.inner.size()
    }

    /// Returns whether this integer uses signed semantics.
    #[getter]
    pub fn signed(&self) -> bool {
        self.inner.is_signed()
    }

    /// Returns the value as a Python int if it fits in 64 bits.
    ///
    /// Returns:
    ///     The integer value, or None if the value is too large
    pub fn to_int(&self) -> Option<i64> {
        if self.inner.is_signed() {
            self.inner.to_i64()
        } else {
            self.inner.to_u64().map(|v| {
                #[allow(clippy::cast_possible_wrap)]
                let result = v as i64;
                result
            })
        }
    }

    /// Set the value (like `BinArray.set()`).
    ///
    /// Args:
    ///     value: New value as int, binary string, or `BitInt`
    pub fn set(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let v = extract_operand_value(value)?;
        // Create a new BitInt with the same size and set the value
        self.inner = if self.inner.is_signed() {
            #[allow(clippy::cast_possible_wrap)]
            BitInt::new_signed(self.inner.size(), v as i64)
        } else {
            BitInt::new_unsigned(self.inner.size(), v)
        };
        Ok(())
    }

    /// Get the value of a specific bit (0-indexed from LSB).
    ///
    /// Args:
    ///     index: The bit index (0 is the least significant bit)
    ///
    /// Returns:
    ///     True if the bit is 1, False if it is 0
    ///
    /// Raises:
    ///     `IndexError`: If index >= size
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

    /// Set the value of a specific bit (0-indexed from LSB).
    ///
    /// Args:
    ///     index: The bit index (0 is the least significant bit)
    ///     value: True to set the bit to 1, False to set it to 0
    ///
    /// Raises:
    ///     `IndexError`: If index >= size
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

    /// Returns the number of 1 bits (population count).
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
    ///
    /// Like `BinArray.num_bits()`.
    pub fn num_bits(&self) -> u32 {
        if let Some(v) = self.inner.to_u64() {
            if v == 0 { 1 } else { 64 - v.leading_zeros() }
        } else {
            // For large values, return the size
            u32::from(self.inner.size())
        }
    }

    /// Clamp the value to fit within the specified bit size.
    ///
    /// Like `BinArray.clamp()`.
    ///
    /// Args:
    ///     size: Maximum number of bits allowed
    pub fn clamp(&mut self, size: u16) {
        if size < self.inner.size() {
            // Mask the value to the new size
            if let Some(v) = self.inner.to_u64() {
                let mask = if size >= 64 {
                    u64::MAX
                } else {
                    (1u64 << size) - 1
                };
                self.inner = BitInt::new_unsigned(self.inner.size(), v & mask);
            }
        }
    }

    /// Set value with clipping to fit within the allocated size.
    ///
    /// Like `BinArray.set_clip()`.
    ///
    /// Args:
    ///     value: Value to set, clipped if necessary
    pub fn set_clip(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let v = extract_operand_value(value)?;
        let size = self.inner.size();
        let mask = if size >= 64 {
            u64::MAX
        } else {
            (1u64 << size) - 1
        };
        self.inner = if self.inner.is_signed() {
            #[allow(clippy::cast_possible_wrap)]
            BitInt::new_signed(size, (v & mask) as i64)
        } else {
            BitInt::new_unsigned(size, v & mask)
        };
        Ok(())
    }

    // ========================================================================
    // Bitwise operations (support BitInt, int, or str operands)
    // ========================================================================

    /// Bitwise XOR. Accepts `BitInt`, int, or binary string.
    pub fn __xor__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitInt> {
        let other_val = extract_operand_value(other)?;
        let other_int = BitInt::new_unsigned(self.inner.size(), other_val);
        Ok(PyBitInt {
            inner: &self.inner ^ &other_int,
        })
    }

    /// Reverse XOR (for int ^ `BitInt`).
    pub fn __rxor__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitInt> {
        self.__xor__(other)
    }

    /// Bitwise AND. Accepts `BitInt`, int, or binary string.
    pub fn __and__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitInt> {
        let other_val = extract_operand_value(other)?;
        let other_int = BitInt::new_unsigned(self.inner.size(), other_val);
        Ok(PyBitInt {
            inner: &self.inner & &other_int,
        })
    }

    /// Reverse AND (for int & `BitInt`).
    pub fn __rand__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitInt> {
        self.__and__(other)
    }

    /// Bitwise OR. Accepts `BitInt`, int, or binary string.
    pub fn __or__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitInt> {
        let other_val = extract_operand_value(other)?;
        let other_int = BitInt::new_unsigned(self.inner.size(), other_val);
        Ok(PyBitInt {
            inner: &self.inner | &other_int,
        })
    }

    /// Reverse OR (for int | `BitInt`).
    pub fn __ror__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitInt> {
        self.__or__(other)
    }

    /// Bitwise NOT (inversion).
    pub fn __invert__(&self) -> PyBitInt {
        PyBitInt {
            inner: !&self.inner,
        }
    }

    /// Left shift. Accepts int or `BitInt`.
    pub fn __lshift__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitInt> {
        let n = extract_operand_value(other)?;
        #[allow(clippy::cast_possible_truncation)]
        let n = n as u16;
        Ok(PyBitInt {
            inner: &self.inner << n,
        })
    }

    /// Right shift. Accepts int or `BitInt`.
    pub fn __rshift__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitInt> {
        let n = extract_operand_value(other)?;
        #[allow(clippy::cast_possible_truncation)]
        let n = n as u16;
        Ok(PyBitInt {
            inner: &self.inner >> n,
        })
    }

    // ========================================================================
    // Arithmetic operations (support BitInt, int, or str operands)
    // ========================================================================

    /// Addition. Accepts `BitInt`, int, or binary string.
    pub fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitInt> {
        let other_int = self.operand_to_bitint(other)?;
        Ok(PyBitInt {
            inner: &self.inner + &other_int,
        })
    }

    /// Reverse addition (for int + `BitInt`).
    pub fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitInt> {
        self.__add__(other)
    }

    /// Subtraction. Accepts `BitInt`, int, or binary string.
    pub fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitInt> {
        let other_int = self.operand_to_bitint(other)?;
        Ok(PyBitInt {
            inner: &self.inner - &other_int,
        })
    }

    /// Reverse subtraction (for int - `BitInt`).
    pub fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitInt> {
        let other_int = self.operand_to_bitint(other)?;
        Ok(PyBitInt {
            inner: &other_int - &self.inner,
        })
    }

    /// Multiplication. Accepts `BitInt`, int, or binary string.
    pub fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitInt> {
        let other_int = self.operand_to_bitint(other)?;
        Ok(PyBitInt {
            inner: &self.inner * &other_int,
        })
    }

    /// Reverse multiplication (for int * `BitInt`).
    pub fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyBitInt> {
        self.__mul__(other)
    }

    /// Integer division. Accepts `BitInt`, int, or binary string.
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

    /// Remainder (modulo). Accepts `BitInt`, int, or binary string.
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

    // ========================================================================
    // Comparison operations
    // ========================================================================

    /// Rich comparison (==, !=, <, <=, >, >=).
    /// Compares raw values directly (like `BinArray`).
    pub fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<bool> {
        let other_val = extract_operand_value(other)?;
        let self_val = self.inner.to_u64().unwrap_or(0);

        Ok(match op {
            CompareOp::Eq => self_val == other_val,
            CompareOp::Ne => self_val != other_val,
            CompareOp::Lt => self_val < other_val,
            CompareOp::Le => self_val <= other_val,
            CompareOp::Gt => self_val > other_val,
            CompareOp::Ge => self_val >= other_val,
        })
    }

    /// Hash function for use in sets and dicts.
    pub fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.size().hash(&mut hasher);
        self.inner.is_signed().hash(&mut hasher);
        if let Some(v) = self.inner.to_u64() {
            v.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// String representation (as a binary string, like `BinArray`).
    pub fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    /// Detailed repr for debugging.
    pub fn __repr__(&self) -> String {
        let signed_str = if self.inner.is_signed() {
            ", signed=True"
        } else {
            ""
        };
        format!(
            "BitInt({}, 0b{}{})",
            self.inner.size(),
            self.inner,
            signed_str
        )
    }

    /// Get the binary string representation with configurable bit ordering.
    ///
    /// Args:
    ///     `reverse_bits`: If True, reverse the bit order (LSB on left instead of right).
    ///                   If False (default), use standard notation (MSB on left).
    ///     separator: Optional separator between bits (e.g., " " or "_").
    ///
    /// Returns:
    ///     Binary string representation.
    ///
    /// Examples:
    ///     >>> b = BitInt("1010")  # value 10
    ///     >>> `b.to_binary_str()`  # Standard: MSB first
    ///     "1010"
    ///     >>> `b.to_binary_str(reverse_bits=True)`  # Reversed: LSB first
    ///     "0101"
    ///     >>> `b.to_binary_str(separator`=" ")
    ///     "1 0 1 0"
    #[pyo3(signature = (reverse_bits=false, separator=None))]
    pub fn to_binary_str(&self, reverse_bits: bool, separator: Option<&str>) -> String {
        let size = self.inner.size();
        let mut bits = Vec::with_capacity(size as usize);

        if reverse_bits {
            // Reversed: bit 0 on the left
            for i in 0..size {
                bits.push(if self.inner.get_bit(i) { '1' } else { '0' });
            }
        } else {
            // Standard: bit 0 on the right (MSB first)
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

    /// Length returns the bit size.
    pub fn __len__(&self) -> usize {
        self.inner.size() as usize
    }

    /// Get bit at index (supports Python indexing with []).
    /// Returns int (0 or 1) like `BinArray` for compatibility.
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

    /// Set bit at index (supports Python indexing with [] = ).
    #[allow(clippy::cast_possible_wrap)]
    pub fn __setitem__(&mut self, index: isize, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let size = self.inner.size() as isize;
        let idx = if index < 0 { size + index } else { index };

        if idx < 0 || idx >= size {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Bit index {index} out of bounds for size {size}"
            )));
        }

        // Accept int (0/1) or bool or str ("0"/"1")
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

    /// Boolean conversion (True if non-zero).
    pub fn __bool__(&self) -> bool {
        !self.inner.is_zero()
    }

    /// Integer conversion.
    pub fn __int__(&self) -> PyResult<i64> {
        self.to_int().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyOverflowError, _>(
                "`BitInt` value too large to convert to Python int",
            )
        })
    }
}
