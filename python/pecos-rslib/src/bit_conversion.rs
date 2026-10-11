// Copyright 2026 The PECOS Developers
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

//! Conversion helpers for Python arbitrary-precision integers ↔ `u64` word arrays.
//!
//! These functions bridge Python's unlimited-precision `int` type with the
//! fixed-width `BitUInt`/`BitInt` types that store values as `Vec<u64>` words.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyInt};

/// Decode an integer's two's-complement value into the little-endian words that
/// cover `bits`. Bits above `bits` in the top word may be set; callers mask them
/// when building the fixed-width value.
pub fn pyint_to_u64_words(obj: &Bound<'_, PyAny>, bits: u16) -> PyResult<Vec<u64>> {
    let n_words = usize::from(bits).div_ceil(64);
    if let Ok(v) = obj.extract::<u64>() {
        let mut words = vec![0; n_words];
        words[0] = v;
        return Ok(words);
    }
    if let Ok(v) = obj.extract::<i64>() {
        let mut words = vec![if v < 0 { u64::MAX } else { 0 }; n_words];
        words[0] = v.cast_unsigned();
        return Ok(words);
    }

    let mask = 1u32.into_pyobject(obj.py())?.lshift(bits)?.sub(1)?;
    let wrapped = obj.bitand(mask)?;
    let bytes = wrapped.call_method1("to_bytes", (n_words * 8, "little"))?;
    Ok(bytes
        .cast::<PyBytes>()?
        .as_bytes()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| u64::from_le_bytes(*chunk))
        .collect())
}

/// Convert little-endian words with Python's linear-time byte decoder.
pub fn u64_words_to_pyint<'py>(py: Python<'py>, words: &[u64]) -> PyResult<Bound<'py, PyAny>> {
    if words.len() == 1 || words.iter().skip(1).all(|&w| w == 0) {
        return Ok(words
            .first()
            .copied()
            .unwrap_or(0)
            .into_pyobject(py)?
            .into_any());
    }
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    py.get_type::<PyInt>()
        .call_method1("from_bytes", (PyBytes::new(py, &bytes), "little"))
}

/// Read an accepted operand's mathematical value without receiver-width coercion.
pub fn operand_to_pyint<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if obj.is_instance_of::<PyInt>() {
        return Ok(obj.clone());
    }
    if let Ok(value) = obj.extract::<PyRef<crate::bit_int_bindings::PyBitInt>>() {
        return value.__int__(obj.py());
    }
    if let Ok(value) = obj.extract::<PyRef<crate::bit_uint_bindings::PyBitUInt>>() {
        return value.__int__(obj.py());
    }
    if let Ok(s) = obj.extract::<String>() {
        let bits = s
            .strip_prefix("0b")
            .or_else(|| s.strip_prefix("0B"))
            .unwrap_or(&s);
        if bits.is_empty() || !bits.bytes().all(|c| c == b'0' || c == b'1') {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Binary string must contain only '0' and '1' characters",
            ));
        }
        return obj.py().get_type::<PyInt>().call1((bits, 2));
    }
    obj.py().import("operator")?.getattr("index")?.call1((obj,))
}

/// Read a count once, rejecting negatives and saturating beyond all storage widths.
pub fn shift_count(obj: &Bound<'_, PyAny>) -> PyResult<u16> {
    let value = operand_to_pyint(obj)?;
    if value.lt(0)? {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "negative shift count",
        ));
    }
    if value.ge(u16::MAX)? {
        return Ok(u16::MAX);
    }
    value.extract()
}

/// Convert `u64` words (little-endian) representing a two's complement value
/// to a signed Python integer.
///
/// `internal_size` is the total number of bits in the internal representation
/// (e.g., `user_size + 1` for `BitInt`). The sign bit is at position `internal_size - 1`.
pub fn u64_words_to_pyint_signed<'py>(
    py: Python<'py>,
    words: &[u64],
    internal_size: u16,
) -> PyResult<Bound<'py, PyAny>> {
    // Fast path: fits in i64
    if internal_size <= 64 {
        let raw = words[0];
        let val = if internal_size == 64 {
            #[allow(clippy::cast_possible_wrap)]
            let r = raw as i64;
            r
        } else {
            let sign_bit = 1u64 << (internal_size - 1);
            if raw & sign_bit != 0 {
                let mask = !((1u64 << internal_size) - 1);
                #[allow(clippy::cast_possible_wrap)]
                let r = (raw | mask) as i64;
                r
            } else {
                #[allow(clippy::cast_possible_wrap)]
                let r = raw as i64;
                r
            }
        };
        return Ok(val
            .into_pyobject(py)
            .expect("i64 to Python conversion failed")
            .into_any());
    }

    // Slow path: build unsigned value, then check sign bit
    let unsigned = u64_words_to_pyint(py, words)?;

    let sign_bit_pos = internal_size - 1;
    let one = 1u32
        .into_pyobject(py)
        .expect("u32 to Python conversion failed")
        .into_any();
    let sign_mask = one.call_method1("__lshift__", (sign_bit_pos,))?;
    let sign_test = unsigned.call_method1("__and__", (&sign_mask,))?;

    if sign_test.is_truthy()? {
        // Subtract 2^internal_size to get negative value
        let modulus = one.call_method1("__lshift__", (internal_size,))?;
        unsigned.call_method1("__sub__", (&modulus,))
    } else {
        Ok(unsigned)
    }
}
