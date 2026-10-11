//! Shared classical expression values and eager operator evaluation.

use crate::bit_uint::twos_complement::{negate, negative, signed_cmp, signed_div_rem};
use crate::{BitUInt, errors::PecosError};
use std::fmt;

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
    /// are not sign-extended.
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

/// Format an integer pattern at its full width for diagnostics.
///
/// Negative patterns use their magnitude; widths above 128 use hexadecimal.
///
/// # Errors
/// Returns an error if the pattern cannot be converted to a bit vector.
pub fn format_integer(bits: &BitUInt, signed: bool) -> Result<String, PecosError> {
    let is_negative = signed && negative(bits);
    let magnitude = if is_negative {
        negate(bits)
    } else {
        bits.clone()
    };
    let magnitude =
        crate::bitvec::from_expr_value(&ExprValue::Unsigned(magnitude), usize::from(bits.size()))?;
    let magnitude = if magnitude.len() <= 128 {
        crate::bitvec::to_decimal_string(&magnitude)
    } else {
        crate::bitvec::to_hex_string(&magnitude)
    };
    Ok(if is_negative {
        format!("-{magnitude}")
    } else {
        magnitude
    })
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

fn arithmetic_shr(value: &BitUInt, count: u16) -> BitUInt {
    let count = count.min(value.size());
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
/// Shift counts use their full mathematical value and own signedness tag.
/// Negative signed counts are errors. Counts at or beyond the evaluation width
/// yield zero, or all sign bits for an arithmetic right shift.
///
/// # Errors
/// Returns an error for division by zero, a negative shift count, or an
/// unsupported operator.
pub fn eval_binary_op(op: &str, lhs: ExprValue, rhs: ExprValue) -> Result<ExprValue, PecosError> {
    // Inspect the count at its own width and tag before widening the operands.
    match op {
        "&&" => return Ok(ExprValue::Boolean(lhs.as_bool() && rhs.as_bool())),
        "||" => return Ok(ExprValue::Boolean(lhs.as_bool() || rhs.as_bool())),
        _ => {}
    }
    let shift_count = match op {
        "<<" | ">>" => match &rhs {
            ExprValue::Signed(bits) if negative(bits) => {
                return Err(PecosError::Input(format!(
                    "Negative shift amount: {}",
                    format_integer(bits, true)?
                )));
            }
            ExprValue::Signed(bits) | ExprValue::Unsigned(bits) => bits
                .to_u64()
                .and_then(|count| u16::try_from(count).ok())
                .unwrap_or(u16::MAX),
            ExprValue::Boolean(value) => u16::from(*value),
        },
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
            let shift = shift_count;
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
