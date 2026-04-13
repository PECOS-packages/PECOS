use crate::v0_1::ast::{ArgItem, Expression};
use crate::v0_1::environment::{BitValue, DataType, Environment};
use pecos_core::BitUInt;
use pecos_core::errors::PecosError;
use std::fmt;

/// Minimum evaluation width -- matches the hardware model where
/// everything is i64 under the hood.
const MIN_EVAL_WIDTH: u16 = 64;

/// Widen a `BitUInt` to a target width by zero-extending.
/// If already at or wider than target, returns as-is.
fn widen_to(v: BitUInt, target: u16) -> BitUInt {
    if v.size() >= target {
        return v;
    }
    // Create wider value from raw words (handles >64 bit)
    let words = v.to_words();
    BitUInt::from_raw_words(target, words.into_boxed_slice())
}

/// Expression value using arbitrary-width integers.
///
/// All values use `BitUInt` internally (matching the hardware model where
/// everything is unsigned bits). Sign interpretation happens at the API
/// boundary via `as_i64()`. The `Signed` variant tracks that the value
/// should be treated as signed for operations like comparison and shift.
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
    /// Converts the expression value to i64 (sign-extending for Signed).
    #[must_use]
    #[allow(clippy::cast_possible_wrap)]
    pub fn as_i64(&self) -> i64 {
        match self {
            ExprValue::Signed(v) | ExprValue::Unsigned(v) => v.to_u64().unwrap_or(0) as i64,
            ExprValue::Boolean(v) => i64::from(*v),
        }
    }

    /// Converts the expression value to u64.
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

    /// Converts a `BitValue` to an `ExprValue`, widening to evaluation width.
    ///
    /// For signed values, sign-extends from the type width (not zero-extends).
    /// For example, i8(-1) stored as 0xFF is widened to 0xFFFFFFFFFFFFFFFF.
    #[must_use]
    #[allow(clippy::cast_sign_loss)]
    pub fn from_bit_value(value: &BitValue) -> Self {
        if value.is_signed() {
            // Sign-extend via as_i64(), then store at eval width
            let signed_val = value.as_i64();
            ExprValue::Signed(BitUInt::new(MIN_EVAL_WIDTH, signed_val as u64))
        } else {
            let eval_width = MIN_EVAL_WIDTH.max(value.size());
            let raw = value.to_bituint();
            let widened = widen_to(raw, eval_width);
            ExprValue::Unsigned(widened)
        }
    }

    /// Create a signed value at evaluation width from i64.
    #[must_use]
    #[allow(clippy::cast_sign_loss)]
    fn signed(val: i64) -> Self {
        ExprValue::Signed(BitUInt::new(MIN_EVAL_WIDTH, val as u64))
    }

    /// Create an unsigned value at evaluation width from u64.
    #[must_use]
    fn unsigned(val: u64) -> Self {
        ExprValue::Unsigned(BitUInt::new(MIN_EVAL_WIDTH, val))
    }
}

impl PartialEq<i64> for ExprValue {
    fn eq(&self, other: &i64) -> bool {
        self.as_i64() == *other
    }
}

impl PartialEq<u64> for ExprValue {
    fn eq(&self, other: &u64) -> bool {
        self.as_u64() == *other
    }
}

/// Evaluator for expressions using arbitrary-width integers.
pub struct ExpressionEvaluator<'a> {
    /// Environment for variable lookups
    environment: &'a Environment,
}

impl<'a> ExpressionEvaluator<'a> {
    /// Creates a new expression evaluator with the given environment.
    #[must_use]
    pub fn new(environment: &'a Environment) -> Self {
        Self { environment }
    }

    /// Evaluates an expression.
    ///
    /// # Errors
    /// Returns an error if evaluation fails.
    pub fn eval_expr(&mut self, expr: &Expression) -> Result<ExprValue, PecosError> {
        match expr {
            Expression::Integer(val) => return Ok(ExprValue::signed(*val)),
            Expression::Variable(name) => {
                if let Some(value) = self.environment.get(name) {
                    let is_bool = self
                        .environment
                        .get_variable_info_opt(name)
                        .is_some_and(|info| info.data_type == DataType::Bool);
                    return Ok(if is_bool {
                        ExprValue::Boolean(value.as_bool())
                    } else {
                        ExprValue::from_bit_value(value)
                    });
                }
                return Err(PecosError::Input(format!("Variable '{name}' not found")));
            }
            Expression::Operation { .. } => {}
        }

        let result = match expr {
            Expression::Operation { cop, args } => match cop.as_str() {
                // Unary operations
                "~" | "!" => {
                    if args.len() != 1 {
                        return Err(PecosError::Input(format!(
                            "Unary operation '{cop}' requires exactly 1 argument"
                        )));
                    }
                    self.eval_unary_op(cop, &args[0])
                }
                // Short-circuit logical operations
                "&&" => {
                    if args.len() != 2 {
                        return Err(PecosError::Input(
                            "Logical AND requires exactly 2 arguments".to_string(),
                        ));
                    }
                    let lhs = self.eval_arg(&args[0])?;
                    if !lhs.as_bool() {
                        return Ok(ExprValue::Boolean(false));
                    }
                    let rhs = self.eval_arg(&args[1])?;
                    Ok(ExprValue::Boolean(rhs.as_bool()))
                }
                "||" => {
                    if args.len() != 2 {
                        return Err(PecosError::Input(
                            "Logical OR requires exactly 2 arguments".to_string(),
                        ));
                    }
                    let lhs = self.eval_arg(&args[0])?;
                    if lhs.as_bool() {
                        return Ok(ExprValue::Boolean(true));
                    }
                    let rhs = self.eval_arg(&args[1])?;
                    Ok(ExprValue::Boolean(rhs.as_bool()))
                }
                // Binary operations
                _ => {
                    if args.len() != 2 {
                        return Err(PecosError::Input(format!(
                            "Binary operation '{cop}' requires exactly 2 arguments"
                        )));
                    }
                    self.eval_binary_op(cop, &args[0], &args[1])
                }
            },
            _ => unreachable!("handled above"),
        }?;

        Ok(result)
    }

    /// Evaluates an argument to an `ExprValue`.
    ///
    /// # Errors
    /// Returns an error if evaluation fails.
    pub fn eval_arg(&mut self, arg: &ArgItem) -> Result<ExprValue, PecosError> {
        match arg {
            ArgItem::Simple(name) => {
                if let Some(value) = self.environment.get(name) {
                    let is_bool = self
                        .environment
                        .get_variable_info_opt(name)
                        .is_some_and(|info| info.data_type == DataType::Bool);
                    Ok(if is_bool {
                        ExprValue::Boolean(value.as_bool())
                    } else {
                        ExprValue::from_bit_value(value)
                    })
                } else {
                    Err(PecosError::Input(format!("Variable '{name}' not found")))
                }
            }
            ArgItem::Indexed((name, idx)) => {
                if let Ok(bit) = self.environment.get_bit(name, *idx) {
                    Ok(ExprValue::Boolean(bit.0))
                } else {
                    Err(PecosError::Input(format!(
                        "Failed to access bit {name}[{idx}]"
                    )))
                }
            }
            ArgItem::Integer(val) => Ok(ExprValue::signed(*val)),
            ArgItem::UInteger(val) => Ok(ExprValue::unsigned(*val)),
            ArgItem::Expression(expr) => self.eval_expr(expr),
        }
    }

    /// Evaluates a unary operation.
    fn eval_unary_op(&mut self, op: &str, arg: &ArgItem) -> Result<ExprValue, PecosError> {
        let val = self.eval_arg(arg)?;

        match op {
            "~" => {
                // Bitwise NOT -- flips all bits at evaluation width
                match val {
                    ExprValue::Signed(v) => Ok(ExprValue::Signed(!&v)),
                    ExprValue::Unsigned(v) => Ok(ExprValue::Unsigned(!&v)),
                    ExprValue::Boolean(v) => Ok(ExprValue::Boolean(!v)),
                }
            }
            "!" => Ok(ExprValue::Boolean(!val.as_bool())),
            _ => Err(PecosError::Input(format!(
                "Unsupported unary operation: {op}"
            ))),
        }
    }

    /// Evaluates a binary operation using `BitUInt` arithmetic directly.
    ///
    /// Both operands are widened to the same evaluation width before
    /// the operation. This works for any bit width -- not just <= 64.
    #[allow(clippy::too_many_lines)]
    fn eval_binary_op(
        &mut self,
        op: &str,
        lhs: &ArgItem,
        rhs: &ArgItem,
    ) -> Result<ExprValue, PecosError> {
        let lhs_val = self.eval_arg(lhs)?;
        let rhs_val = self.eval_arg(rhs)?;

        // Determine signedness of result
        let lhs_signed = matches!(lhs_val, ExprValue::Signed(_));
        let rhs_signed = matches!(rhs_val, ExprValue::Signed(_));
        let result_signed = lhs_signed && rhs_signed;

        // Extract inner BitUInt from both operands and widen to same width
        let (l, r) = Self::widen_pair(&lhs_val, &rhs_val);

        // Helper to wrap result in the right variant
        let wrap = |v: BitUInt| -> ExprValue {
            if result_signed {
                ExprValue::Signed(v)
            } else {
                ExprValue::Unsigned(v)
            }
        };

        // For signed operations we use i64 arithmetic (matching C/Rust behavior).
        // This is correct for values <= 64 bits, which covers all practical PHIR types.
        let li = lhs_val.as_i64();
        let ri = rhs_val.as_i64();

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        match op {
            // Arithmetic (BitUInt ops automatically wrap at the width)
            "+" => Ok(wrap(&l + &r)),
            "-" => Ok(wrap(&l - &r)),
            "*" => Ok(wrap(&l * &r)),
            "/" => {
                if r.is_zero() {
                    return Err(PecosError::RuntimeDivisionByZero);
                }
                if result_signed {
                    // Signed division truncates toward zero (C/Rust behavior)
                    let result = li.wrapping_div(ri);
                    Ok(ExprValue::signed(result))
                } else {
                    Ok(wrap(&l / &r))
                }
            }
            "%" => {
                if r.is_zero() {
                    return Err(PecosError::RuntimeDivisionByZero);
                }
                if result_signed {
                    // Signed remainder (C/Rust behavior: sign follows dividend)
                    let result = li.wrapping_rem(ri);
                    Ok(ExprValue::signed(result))
                } else {
                    Ok(wrap(&l % &r))
                }
            }

            // Bitwise
            "&" => Ok(wrap(&l & &r)),
            "|" => Ok(wrap(&l | &r)),
            "^" => Ok(wrap(&l ^ &r)),

            // Shifts -- RHS is the shift amount (must be non-negative)
            ">>" => {
                if ri < 0 {
                    return Err(PecosError::Input(format!("Negative shift amount: {ri}")));
                }
                if lhs_signed {
                    // Arithmetic right shift: sign-extends (C/Rust behavior for signed)
                    let result = li.wrapping_shr(ri as u32);
                    Ok(ExprValue::signed(result))
                } else {
                    let shift = r.to_u64().unwrap_or(0) as u16;
                    Ok(wrap(&l >> shift))
                }
            }
            "<<" => {
                if ri < 0 {
                    return Err(PecosError::Input(format!("Negative shift amount: {ri}")));
                }
                let shift = r.to_u64().unwrap_or(0) as u16;
                Ok(wrap(&l << shift))
            }

            // Comparisons -- signed when both operands are signed
            "==" => Ok(ExprValue::unsigned(u64::from(l == r))),
            "!=" => Ok(ExprValue::unsigned(u64::from(l != r))),
            "<" => {
                let result = if result_signed { li < ri } else { l < r };
                Ok(ExprValue::unsigned(u64::from(result)))
            }
            ">" => {
                let result = if result_signed { li > ri } else { l > r };
                Ok(ExprValue::unsigned(u64::from(result)))
            }
            "<=" => {
                let result = if result_signed { li <= ri } else { l <= r };
                Ok(ExprValue::unsigned(u64::from(result)))
            }
            ">=" => {
                let result = if result_signed { li >= ri } else { l >= r };
                Ok(ExprValue::unsigned(u64::from(result)))
            }

            // Logical
            "&&" => Ok(ExprValue::Boolean(lhs_val.as_bool() && rhs_val.as_bool())),
            "||" => Ok(ExprValue::Boolean(lhs_val.as_bool() || rhs_val.as_bool())),

            _ => Err(PecosError::Input(format!(
                "Unsupported binary operation: {op}"
            ))),
        }
    }

    /// Extract inner `BitUInt` from an `ExprValue`, widening both to the same width.
    fn widen_pair(a: &ExprValue, b: &ExprValue) -> (BitUInt, BitUInt) {
        let (a_bits, b_bits) = match (a, b) {
            (
                ExprValue::Signed(va) | ExprValue::Unsigned(va),
                ExprValue::Signed(vb) | ExprValue::Unsigned(vb),
            ) => (va.clone(), vb.clone()),
            (ExprValue::Boolean(v), ExprValue::Signed(vb) | ExprValue::Unsigned(vb)) => {
                (BitUInt::new(vb.size(), u64::from(*v)), vb.clone())
            }
            (ExprValue::Signed(va) | ExprValue::Unsigned(va), ExprValue::Boolean(v)) => {
                (va.clone(), BitUInt::new(va.size(), u64::from(*v)))
            }
            (ExprValue::Boolean(va), ExprValue::Boolean(vb)) => (
                BitUInt::new(MIN_EVAL_WIDTH, u64::from(*va)),
                BitUInt::new(MIN_EVAL_WIDTH, u64::from(*vb)),
            ),
        };

        // Widen to same width (max of the two)
        let target = a_bits.size().max(b_bits.size());
        let a_wide = widen_to(a_bits, target);
        let b_wide = widen_to(b_bits, target);
        (a_wide, b_wide)
    }

    /// Gets multiple bit values from a variable.
    ///
    /// # Errors
    /// Returns an error if any bit access fails.
    pub fn get_bits(&self, name: &str, indices: &[usize]) -> Result<Vec<bool>, PecosError> {
        let value = self
            .environment
            .get(name)
            .ok_or_else(|| PecosError::Input(format!("Variable '{name}' not found")))?;
        let value_u64 = value.as_u64();
        indices
            .iter()
            .map(|&idx| Ok(((value_u64 >> idx) & 1) != 0))
            .collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_environment() -> Environment {
        let mut env = Environment::new();
        env.add_variable("x", DataType::I32, 32).unwrap();
        env.add_variable("y", DataType::U8, 8).unwrap();
        env.add_variable("z", DataType::Bool, 1).unwrap();
        env.set_raw("x", 42).unwrap();
        env.set_raw("y", 255).unwrap();
        env.set_raw("z", 1).unwrap();
        env
    }

    #[test]
    fn test_basic_arithmetic() {
        let env = setup_environment();
        let mut evaluator = ExpressionEvaluator::new(&env);

        let expr = Expression::Operation {
            cop: "+".to_string(),
            args: vec![ArgItem::Simple("x".to_string()), ArgItem::Integer(8)],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert_eq!(result.as_i64(), 50);

        let expr = Expression::Operation {
            cop: "-".to_string(),
            args: vec![ArgItem::Simple("x".to_string()), ArgItem::Integer(2)],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert_eq!(result.as_i64(), 40);

        let expr = Expression::Operation {
            cop: "*".to_string(),
            args: vec![ArgItem::Simple("x".to_string()), ArgItem::Integer(2)],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert_eq!(result.as_i64(), 84);

        let expr = Expression::Operation {
            cop: "/".to_string(),
            args: vec![ArgItem::Simple("x".to_string()), ArgItem::Integer(2)],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert_eq!(result.as_i64(), 21);
    }

    #[test]
    fn test_bitwise_operations() {
        let env = setup_environment();
        let mut evaluator = ExpressionEvaluator::new(&env);

        // Test bitwise AND
        let expr = Expression::Operation {
            cop: "&".to_string(),
            args: vec![ArgItem::Simple("x".to_string()), ArgItem::Integer(15)],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert_eq!(result.as_i64(), 10); // 42 & 15 = 0b101010 & 0b1111 = 0b1010 = 10

        // Test bitwise XOR
        let expr = Expression::Operation {
            cop: "^".to_string(),
            args: vec![ArgItem::Simple("x".to_string()), ArgItem::Integer(15)],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert_eq!(result.as_i64(), 37); // 42 ^ 15 = 37

        // Test bitwise NOT on Bool
        let expr = Expression::Operation {
            cop: "~".to_string(),
            args: vec![ArgItem::Simple("z".to_string())],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert!(!result.as_bool()); // ~true = false
    }

    #[test]
    fn test_comparison_operations() {
        let env = setup_environment();
        let mut evaluator = ExpressionEvaluator::new(&env);

        let expr = Expression::Operation {
            cop: "==".to_string(),
            args: vec![ArgItem::Simple("x".to_string()), ArgItem::Integer(42)],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert!(result.as_bool());

        let expr = Expression::Operation {
            cop: "<".to_string(),
            args: vec![ArgItem::Simple("x".to_string()), ArgItem::Integer(100)],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert!(result.as_bool());
    }

    #[test]
    fn test_evaluation_at_64_bit_width() {
        // i32 variable, but arithmetic should happen at 64 bits
        let mut env = Environment::new();
        env.add_variable("a", DataType::I32, 32).unwrap();
        env.set_raw("a", 1).unwrap();

        let mut evaluator = ExpressionEvaluator::new(&env);

        // 1 << 33 should give 8589934592, not 2 (modulo-32) or 0 (truncate-32)
        let expr = Expression::Operation {
            cop: "<<".to_string(),
            args: vec![ArgItem::Simple("a".to_string()), ArgItem::Integer(33)],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert_eq!(result.as_i64(), 1i64 << 33);
    }

    #[test]
    fn test_not_at_full_width() {
        // ~(u32 size=1, val=1) should flip all 64 bits, giving a large number
        let mut env = Environment::new();
        env.add_variable("m", DataType::U32, 1).unwrap();
        env.set_raw("m", 1).unwrap();

        let mut evaluator = ExpressionEvaluator::new(&env);

        let expr = Expression::Operation {
            cop: "~".to_string(),
            args: vec![ArgItem::Simple("m".to_string())],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        // ~1u64 = 0xFFFFFFFFFFFFFFFE
        assert_eq!(result.as_u64(), !1u64);
    }

    #[test]
    fn test_signed_comparison() {
        // Integer literals are signed: -1 < 1 should be true
        let env = Environment::new();
        let mut evaluator = ExpressionEvaluator::new(&env);
        let expr = Expression::Operation {
            cop: "<".to_string(),
            args: vec![ArgItem::Integer(-1), ArgItem::Integer(1)],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert!(result.as_bool(), "-1 < 1 should be true");
    }

    #[test]
    fn test_signed_comparison_greater() {
        let env = Environment::new();
        let mut evaluator = ExpressionEvaluator::new(&env);
        let expr = Expression::Operation {
            cop: ">".to_string(),
            args: vec![ArgItem::Integer(-1), ArgItem::Integer(1)],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert!(!result.as_bool(), "-1 > 1 should be false");
    }

    #[test]
    fn test_signed_division() {
        // -7 / 2 should be -3 (truncation toward zero)
        let env = Environment::new();
        let mut evaluator = ExpressionEvaluator::new(&env);
        let expr = Expression::Operation {
            cop: "/".to_string(),
            args: vec![ArgItem::Integer(-7), ArgItem::Integer(2)],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert_eq!(result.as_i64(), -3, "-7 / 2 should be -3");
    }

    #[test]
    fn test_signed_modulo() {
        // -7 % 3 should be -1 (remainder, sign follows dividend)
        let env = Environment::new();
        let mut evaluator = ExpressionEvaluator::new(&env);
        let expr = Expression::Operation {
            cop: "%".to_string(),
            args: vec![ArgItem::Integer(-7), ArgItem::Integer(3)],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert_eq!(result.as_i64(), -1, "-7 % 3 should be -1");
    }

    #[test]
    fn test_signed_right_shift() {
        // -1 >> 1 should be -1 (arithmetic shift, sign-extends)
        let env = Environment::new();
        let mut evaluator = ExpressionEvaluator::new(&env);
        let expr = Expression::Operation {
            cop: ">>".to_string(),
            args: vec![ArgItem::Integer(-1), ArgItem::Integer(1)],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert_eq!(
            result.as_i64(),
            -1,
            "-1 >> 1 should be -1 (arithmetic shift)"
        );
    }

    #[test]
    fn test_sign_extension_from_narrow() {
        // i8 with value 0xFF stored in 7-bit size.
        // Type width is 8, so bit 7 is the sign bit.
        // 0xFF masked to 7 bits = 0x7F. Sign bit (bit 7) = 0, so positive.
        // But 0x7F in i8 is 127. as_i64() uses type_width=8, sign bit at 7.
        // 0x7F has bit 7 = 0, so it's +127.
        let mut env = Environment::new();
        env.add_variable("a", DataType::I8, 7).unwrap();
        env.set_raw("a", 0x7F).unwrap(); // 127 in i8 (max for 7-bit size)

        let mut evaluator = ExpressionEvaluator::new(&env);
        let result = evaluator
            .eval_arg(&ArgItem::Simple("a".to_string()))
            .unwrap();
        assert_eq!(result.as_i64(), 127, "i8 size=7 val=0x7F should be 127");

        // Now test sign extension with expression: 0 - 1 = -1 as signed
        let expr = Expression::Operation {
            cop: "-".to_string(),
            args: vec![ArgItem::Simple("a".to_string()), ArgItem::Integer(128)],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert_eq!(result.as_i64(), -1, "127 - 128 = -1 as signed");
    }
}
