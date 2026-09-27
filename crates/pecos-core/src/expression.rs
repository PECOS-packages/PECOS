//! Shared classical expression values and eager operator evaluation.

use crate::{BitUInt, errors::PecosError};
use std::{cmp::Ordering, fmt};

/// Minimum width for integer expression evaluation.
pub const MIN_EVAL_WIDTH: u16 = 64;

/// Expression value using arbitrary-width integers.
///
/// All values use `BitUInt` internally (matching the hardware model where
/// everything is unsigned bits). The `Signed` variant interprets the pattern
/// as two's complement at its own width for signed operations.
///
/// All values are widened to at least [`MIN_EVAL_WIDTH`] bits during
/// evaluation, matching the hardware model.
#[derive(Debug, Clone, PartialEq)]
pub enum ExprValue {
    /// Signed value (stored as unsigned bits, sign-interpreted on read)
    Signed(BitUInt),
    /// Unsigned value
    Unsigned(BitUInt),
    /// Boolean value
    Boolean(bool),
}

impl ExprValue {
    /// Reads the pattern as `u64` and casts to `i64`, regardless of its tag.
    ///
    /// Returns zero if any bits above bit 63 are set. Narrow signed patterns
    /// are not sign-extended. This lossy accessor also defines shift-count errors.
    #[must_use]
    pub fn as_i64(&self) -> i64 {
        match self {
            ExprValue::Signed(v) | ExprValue::Unsigned(v) => v.to_u64().unwrap_or(0).cast_signed(),
            ExprValue::Boolean(v) => i64::from(*v),
        }
    }

    /// Reads the pattern as `u64`, returning zero if any bits above bit 63 are set.
    #[must_use]
    pub fn as_u64(&self) -> u64 {
        match self {
            ExprValue::Signed(v) | ExprValue::Unsigned(v) => v.to_u64().unwrap_or(0),
            ExprValue::Boolean(v) => u64::from(*v),
        }
    }

    /// Converts the expression value to boolean.
    #[must_use]
    pub fn as_bool(&self) -> bool {
        match self {
            ExprValue::Signed(v) | ExprValue::Unsigned(v) => !v.is_zero(),
            ExprValue::Boolean(v) => *v,
        }
    }

    /// Create a signed value at evaluation width from i64.
    #[must_use]
    pub fn signed(val: i64) -> Self {
        ExprValue::Signed(BitUInt::new(MIN_EVAL_WIDTH, val.cast_unsigned()))
    }

    /// Create an unsigned value at evaluation width from u64.
    #[must_use]
    pub fn unsigned(val: u64) -> Self {
        ExprValue::Unsigned(BitUInt::new(MIN_EVAL_WIDTH, val))
    }
}

impl fmt::Display for ExprValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExprValue::Signed(v) => write!(f, "{}", v.to_i64().unwrap_or(0)),
            ExprValue::Unsigned(v) => write!(f, "{}", v.to_u64().unwrap_or(0)),
            ExprValue::Boolean(v) => write!(f, "{v}"),
        }
    }
}

/// Widen a value according to its tag, promoting booleans as unsigned integers.
/// Reuse integer storage unchanged when no growth is needed.
fn widen_to(value: ExprValue, target: u16) -> BitUInt {
    match value {
        ExprValue::Signed(v) | ExprValue::Unsigned(v) if v.size() >= target => v,
        ExprValue::Signed(v) => v.resize_sign_extend(target),
        ExprValue::Unsigned(v) => BitUInt::from_raw_words(target, v.to_words().into_boxed_slice()),
        ExprValue::Boolean(v) => BitUInt::new(target, u64::from(v)),
    }
}

fn width(value: &ExprValue) -> u16 {
    match value {
        ExprValue::Signed(v) | ExprValue::Unsigned(v) => v.size(),
        ExprValue::Boolean(_) => 1,
    }
}

fn widen_pair(a: ExprValue, b: ExprValue) -> (BitUInt, BitUInt) {
    let target = width(&a).max(width(&b)).max(MIN_EVAL_WIDTH);
    (widen_to(a, target), widen_to(b, target))
}

fn negative(value: &BitUInt) -> bool {
    value.get_bit(value.size() - 1)
}

fn negate(value: &BitUInt) -> BitUInt {
    &BitUInt::zero(value.size()) - value
}

fn signed_cmp(a: &BitUInt, b: &BitUInt) -> Ordering {
    // Both helpers read each operand's own top bit as its sign, so unequal
    // widths would compare different bit positions and return quiet nonsense.
    // `widen_pair` is the only caller and guarantees equal widths.
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
fn signed_div_rem(a: &BitUInt, b: &BitUInt, remainder: bool) -> BitUInt {
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

fn arithmetic_shr(value: &BitUInt, count: u16) -> BitUInt {
    let count = count % value.size();
    let mut result = value >> count;
    if negative(value) {
        for bit in (value.size() - count)..value.size() {
            result.set_bit(bit, true);
        }
    }
    result
}

/// Evaluate a unary operator on an already-evaluated value.
///
/// Integer operators use at least [`MIN_EVAL_WIDTH`] bits. Unary minus always
/// produces a signed integer; boolean complement and logical not stay boolean.
///
/// # Errors
/// Returns an input error for an unsupported operator.
pub fn eval_unary_op(op: &str, value: ExprValue) -> Result<ExprValue, PecosError> {
    let target = width(&value).max(MIN_EVAL_WIDTH);
    match op {
        "~" => match value {
            ExprValue::Boolean(v) => Ok(ExprValue::Boolean(!v)),
            ExprValue::Signed(_) => Ok(ExprValue::Signed(!widen_to(value, target))),
            ExprValue::Unsigned(_) => Ok(ExprValue::Unsigned(!widen_to(value, target))),
        },
        "!" => Ok(ExprValue::Boolean(!value.as_bool())),
        "-" => Ok(ExprValue::Signed(negate(&widen_to(value, target)))),
        _ => Err(PecosError::Input(format!(
            "Unsupported unary operation: {op}"
        ))),
    }
}

/// Evaluate a binary operator on already-evaluated values.
///
/// Operands widen to the largest operand width or [`MIN_EVAL_WIDTH`], with
/// signed patterns sign-extended and booleans promoted as unsigned zero or one.
/// Arithmetic, bitwise operations and left shifts are signed only when both
/// inputs are signed. Right shifts follow the left input's tag. Comparisons
/// return unsigned 64-bit zero or one; logical operations return booleans.
///
/// Shift counts deliberately preserve PHIR's legacy rules: reject a negative
/// `as_i64()` reading, truncate the `as_u64()` reading to 16 bits, and reduce
/// modulo the evaluation width only for arithmetic right shifts.
///
/// The rejection depends on the count's width, not on its tag: a signed count
/// narrower than 64 bits is read at face value, so a signed eight-bit `0xFF`
/// shifts by 255 rather than erroring as negative one. The count is also read
/// before the operands widen, which is what keeps a 64-bit signed negative
/// count erroring instead of sign-extending into a large positive one. #879
/// covers whether to replace this family of rules.
///
/// # Errors
/// Returns an error for division by zero, a negative shift count, or an
/// unsupported operator.
pub fn eval_binary_op(op: &str, lhs: ExprValue, rhs: ExprValue) -> Result<ExprValue, PecosError> {
    // Read truth values and the original shift-count pattern before consuming
    // operands. Widening can change the lossy reading of a signed count.
    match op {
        "&&" => return Ok(ExprValue::Boolean(lhs.as_bool() && rhs.as_bool())),
        "||" => return Ok(ExprValue::Boolean(lhs.as_bool() || rhs.as_bool())),
        _ => {}
    }
    let shift_count = match op {
        "<<" | ">>" => rhs.as_u64(),
        _ => 0,
    };
    let lhs_signed = matches!(lhs, ExprValue::Signed(_));
    let rhs_signed = matches!(rhs, ExprValue::Signed(_));
    let result_signed = lhs_signed && rhs_signed;
    let (lhs_bits, rhs_bits) = widen_pair(lhs, rhs);
    let wrap = |value| {
        if result_signed {
            ExprValue::Signed(value)
        } else {
            ExprValue::Unsigned(value)
        }
    };
    match op {
        "+" => Ok(wrap(&lhs_bits + &rhs_bits)),
        "-" => Ok(wrap(&lhs_bits - &rhs_bits)),
        "*" => Ok(wrap(&lhs_bits * &rhs_bits)),
        "/" | "%" => {
            if rhs_bits.is_zero() {
                return Err(PecosError::RuntimeDivisionByZero);
            }
            if result_signed {
                Ok(wrap(signed_div_rem(&lhs_bits, &rhs_bits, op == "%")))
            } else if op == "/" {
                Ok(wrap(&lhs_bits / &rhs_bits))
            } else {
                Ok(wrap(&lhs_bits % &rhs_bits))
            }
        }
        "&" => Ok(wrap(&lhs_bits & &rhs_bits)),
        "|" => Ok(wrap(&lhs_bits | &rhs_bits)),
        "^" => Ok(wrap(&lhs_bits ^ &rhs_bits)),
        "<<" | ">>" => {
            let ri = shift_count.cast_signed();
            if ri < 0 {
                return Err(PecosError::Input(format!("Negative shift amount: {ri}")));
            }
            // Truncating to 16 bits is the rule, not an accident: PHIR and the
            // Python interpreter both do it, so a count of 65536 shifts by
            // nothing. #879 asks whether to replace it.
            #[allow(clippy::cast_possible_truncation)]
            let shift = shift_count as u16;
            if op == "<<" {
                Ok(wrap(&lhs_bits << shift))
            } else if lhs_signed {
                Ok(ExprValue::Signed(arithmetic_shr(&lhs_bits, shift)))
            } else {
                Ok(ExprValue::Unsigned(&lhs_bits >> shift))
            }
        }
        "==" | "!=" | "<" | ">" | "<=" | ">=" => {
            let order = if result_signed {
                signed_cmp(&lhs_bits, &rhs_bits)
            } else {
                lhs_bits.cmp(&rhs_bits)
            };
            let result = match op {
                "==" => order.is_eq(),
                "!=" => !order.is_eq(),
                "<" => order.is_lt(),
                ">" => order.is_gt(),
                "<=" => !order.is_gt(),
                ">=" => !order.is_lt(),
                _ => unreachable!(),
            };
            Ok(ExprValue::unsigned(u64::from(result)))
        }
        _ => Err(PecosError::Input(format!(
            "Unsupported binary operation: {op}"
        ))),
    }
}

#[cfg(test)]
mod tests;
