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

/// Extract a Python integer as a `Vec<u64>` words in little-endian order.
///
/// Works for arbitrarily large values, including negative ones.
/// For negative values, the result contains the two's complement representation.
/// The caller is responsible for masking to the desired bit width (typically
/// done by `BitUInt::from_raw_words` / `BitInt::new_from_raw_inner`).
pub fn pyint_to_u64_words(obj: &Bound<'_, PyAny>, n_words: usize) -> PyResult<Vec<u64>> {
    // Fast path: value fits in u64
    if let Ok(v) = obj.extract::<u64>() {
        let mut words = vec![0u64; n_words];
        words[0] = v;
        return Ok(words);
    }

    // Fast path: value fits in i64 (handles small negative values)
    if let Ok(v) = obj.extract::<i64>() {
        #[allow(clippy::cast_sign_loss)]
        let raw = v as u64;
        // Sign-extend: fill upper words with all-1s for negative, all-0s for positive
        let fill = if v < 0 { u64::MAX } else { 0u64 };
        let mut words = vec![fill; n_words];
        words[0] = raw;
        return Ok(words);
    }

    // Slow path: arbitrary precision Python int.
    // Extract word by word: word_i = (value >> (64*i)) & 0xFFFFFFFFFFFFFFFF
    let py = obj.py();
    let mask = u64::MAX.into_pyobject(py).unwrap().into_any();

    let mut words = Vec::with_capacity(n_words);
    let mut current = obj.clone();
    for _ in 0..n_words {
        let word_obj = current.call_method1("__and__", (&mask,))?;
        let word: u64 = word_obj.extract()?;
        words.push(word);
        current = current.call_method1("__rshift__", (64u32,))?;
    }

    Ok(words)
}

/// Convert `u64` words (little-endian, LSB first) to an unsigned Python integer.
pub fn u64_words_to_pyint<'py>(py: Python<'py>, words: &[u64]) -> PyResult<Bound<'py, PyAny>> {
    // Fast path: single word
    if words.len() == 1 || words.iter().skip(1).all(|&w| w == 0) {
        return Ok(words[0].into_pyobject(py).unwrap().into_any());
    }

    // Build value: iterate from MSB word to LSB word
    let mut result = 0u64.into_pyobject(py).unwrap().into_any();
    for &word in words.iter().rev() {
        result = result.call_method1("__lshift__", (64u32,))?;
        let word_py = word.into_pyobject(py).unwrap().into_any();
        result = result.call_method1("__or__", (&word_py,))?;
    }

    Ok(result)
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
        return Ok(val.into_pyobject(py).unwrap().into_any());
    }

    // Slow path: build unsigned value, then check sign bit
    let unsigned = u64_words_to_pyint(py, words)?;

    let sign_bit_pos = internal_size - 1;
    let one = 1u32.into_pyobject(py).unwrap().into_any();
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
