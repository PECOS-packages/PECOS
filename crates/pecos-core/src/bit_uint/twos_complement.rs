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

//! Two's-complement readings of fixed-width unsigned bit patterns.
//!
//! A [`BitUInt`] pattern read as signed has its sign in the top bit. These
//! helpers serve both the signed [`BitInt`](crate::BitInt) type and the shared
//! classical expression evaluator.

use crate::BitUInt;
use std::cmp::Ordering;

pub(crate) fn negative(value: &BitUInt) -> bool {
    value.get_bit(value.size() - 1)
}

pub(crate) fn negate(value: &BitUInt) -> BitUInt {
    &BitUInt::zero(value.size()) - value
}

pub(crate) fn signed_cmp(a: &BitUInt, b: &BitUInt) -> Ordering {
    // Both helpers read each operand's own top bit as its sign, so unequal
    // widths would compare different bit positions and return quiet nonsense.
    // Callers normalize operands to the same width before calling these helpers.
    debug_assert_eq!(a.size(), b.size(), "signed comparison needs equal widths");
    match (negative(a), negative(b)) {
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        _ => a.cmp(b),
    }
}

// Divide magnitudes and restore the sign at the original width. Negating the
// signed minimum wraps to itself, whose unsigned pattern is its magnitude;
// minimum / -1 consequently wraps to minimum, with remainder zero.
pub(crate) fn signed_div_rem(a: &BitUInt, b: &BitUInt, remainder: bool) -> BitUInt {
    debug_assert_eq!(a.size(), b.size(), "signed division needs equal widths");
    let a_negative = negative(a);
    let b_negative = negative(b);
    let magnitude_a = if a_negative { negate(a) } else { a.clone() };
    let magnitude_b = if b_negative { negate(b) } else { b.clone() };
    let magnitude = if remainder {
        &magnitude_a % &magnitude_b
    } else {
        &magnitude_a / &magnitude_b
    };
    // A quotient is negative when the signs differ; a remainder takes the
    // dividend's sign.
    let negative_result = if remainder {
        a_negative
    } else {
        a_negative != b_negative
    };
    if negative_result {
        negate(&magnitude)
    } else {
        magnitude
    }
}
