use crate::v0_1::ast::{ArgItem, Expression};
use crate::v0_1::environment::{BitValue, DataType, Environment, TypedValue};
use pecos_core::BitUInt;
use pecos_core::errors::PecosError;
use std::collections::BTreeMap;
use std::fmt::{self, Write};

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

    /// Converts a `TypedValue` to an `ExprValue` (backward compatibility).
    #[must_use]
    #[allow(clippy::cast_sign_loss)]
    pub fn from_typed_value(value: TypedValue) -> Self {
        match value {
            TypedValue::I8(val) => ExprValue::Signed(BitUInt::new(MIN_EVAL_WIDTH, val as u64)),
            TypedValue::I16(val) => ExprValue::Signed(BitUInt::new(MIN_EVAL_WIDTH, val as u64)),
            TypedValue::I32(val) => ExprValue::Signed(BitUInt::new(MIN_EVAL_WIDTH, val as u64)),
            TypedValue::I64(val) => ExprValue::Signed(BitUInt::new(MIN_EVAL_WIDTH, val as u64)),
            TypedValue::U8(val) => {
                ExprValue::Unsigned(BitUInt::new(MIN_EVAL_WIDTH, u64::from(val)))
            }
            TypedValue::U16(val) => {
                ExprValue::Unsigned(BitUInt::new(MIN_EVAL_WIDTH, u64::from(val)))
            }
            TypedValue::U32(val) => {
                ExprValue::Unsigned(BitUInt::new(MIN_EVAL_WIDTH, u64::from(val)))
            }
            TypedValue::U64(val) => ExprValue::Unsigned(BitUInt::new(MIN_EVAL_WIDTH, val)),
            TypedValue::Bool(val) => ExprValue::Boolean(val),
        }
    }

    /// Converts a `BitValue` to an `ExprValue`, widening to evaluation width.
    #[must_use]
    pub fn from_bit_value(value: &BitValue) -> Self {
        let eval_width = MIN_EVAL_WIDTH.max(value.size());
        // Get the raw BitUInt from the value and widen
        let raw = value.to_bituint();
        let widened = widen_to(raw, eval_width);
        if value.is_signed() {
            ExprValue::Signed(widened)
        } else {
            ExprValue::Unsigned(widened)
        }
    }

    /// Converts an `ExprValue` to a `TypedValue` with a specific data type.
    ///
    /// # Errors
    /// Returns an error if the value cannot be converted.
    pub fn to_typed_value(&self, data_type: &DataType) -> Result<TypedValue, PecosError> {
        Ok(TypedValue::new(data_type, self.as_u64()))
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
    /// Cache for variable lookups
    var_cache: BTreeMap<String, ExprValue>,
    /// Cache for expression evaluation results
    expr_cache: BTreeMap<String, ExprValue>,
}

impl<'a> ExpressionEvaluator<'a> {
    /// Creates a new expression evaluator with the given environment.
    #[must_use]
    pub fn new(environment: &'a Environment) -> Self {
        Self {
            environment,
            var_cache: BTreeMap::new(),
            expr_cache: BTreeMap::new(),
        }
    }

    /// Creates a new expression evaluator with pre-allocated cache sizes.
    #[must_use]
    pub fn with_capacity(
        environment: &'a Environment,
        _var_capacity: usize,
        _expr_capacity: usize,
    ) -> Self {
        Self::new(environment)
    }

    /// Clears the expression cache but keeps variable cache.
    pub fn clear_expr_cache(&mut self) {
        self.expr_cache.clear();
    }

    /// Clears all caches.
    pub fn clear_caches(&mut self) {
        self.var_cache.clear();
        self.expr_cache.clear();
    }

    /// Converts an expression to a string for caching.
    fn expr_to_cache_key(expr: &Expression) -> String {
        match expr {
            Expression::Integer(val) => format!("int:{val}"),
            Expression::Variable(name) => format!("var:{name}"),
            Expression::Operation { cop, args } => {
                let mut key = format!("op:{cop}");
                for arg in args {
                    match arg {
                        ArgItem::Simple(name) => write!(&mut key, ",simple:{name}").unwrap(),
                        ArgItem::Indexed((name, idx)) => {
                            write!(&mut key, ",indexed:{name}[{idx}]").unwrap();
                        }
                        ArgItem::Integer(val) => write!(&mut key, ",int:{val}").unwrap(),
                        ArgItem::UInteger(val) => write!(&mut key, ",uint:{val}").unwrap(),
                        ArgItem::Expression(expr) => {
                            write!(&mut key, ",expr:{}", Self::expr_to_cache_key(expr)).unwrap();
                        }
                    }
                }
                key
            }
        }
    }

    /// Evaluates an expression.
    ///
    /// # Errors
    /// Returns an error if evaluation fails.
    pub fn eval_expr(&mut self, expr: &Expression) -> Result<ExprValue, PecosError> {
        // Handle simple cases without caching
        match expr {
            Expression::Integer(val) => return Ok(ExprValue::signed(*val)),
            Expression::Variable(name) => {
                if let Some(val) = self.var_cache.get(name) {
                    return Ok(val.clone());
                }
                if let Some(value) = self.environment.get(name) {
                    let is_bool = self
                        .environment
                        .get_variable_info_opt(name)
                        .is_some_and(|info| info.data_type == DataType::Bool);
                    let expr_val = if is_bool {
                        ExprValue::Boolean(value.as_bool())
                    } else {
                        ExprValue::from_bit_value(value)
                    };
                    self.var_cache.insert(name.clone(), expr_val.clone());
                    return Ok(expr_val);
                }
                return Err(PecosError::Input(format!("Variable '{name}' not found")));
            }
            Expression::Operation { .. } => {}
        }

        // For complex expressions, use caching
        let cache_key = Self::expr_to_cache_key(expr);
        if let Some(cached_value) = self.expr_cache.get(&cache_key) {
            return Ok(cached_value.clone());
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

        self.expr_cache.insert(cache_key, result.clone());
        Ok(result)
    }

    /// Evaluates an argument to an `ExprValue`.
    ///
    /// # Errors
    /// Returns an error if evaluation fails.
    pub fn eval_arg(&mut self, arg: &ArgItem) -> Result<ExprValue, PecosError> {
        match arg {
            ArgItem::Simple(name) => {
                if let Some(val) = self.var_cache.get(name) {
                    return Ok(val.clone());
                }
                if let Some(value) = self.environment.get(name) {
                    let is_bool = self
                        .environment
                        .get_variable_info_opt(name)
                        .is_some_and(|info| info.data_type == DataType::Bool);
                    let expr_val = if is_bool {
                        ExprValue::Boolean(value.as_bool())
                    } else {
                        ExprValue::from_bit_value(value)
                    };
                    self.var_cache.insert(name.clone(), expr_val.clone());
                    Ok(expr_val)
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

        #[allow(clippy::cast_possible_truncation)]
        match op {
            // Arithmetic (BitUInt ops automatically wrap at the width)
            "+" => Ok(wrap(&l + &r)),
            "-" => Ok(wrap(&l - &r)),
            "*" => Ok(wrap(&l * &r)),
            "/" => {
                if r.is_zero() {
                    return Err(PecosError::RuntimeDivisionByZero);
                }
                Ok(wrap(&l / &r))
            }
            "%" => {
                if r.is_zero() {
                    return Err(PecosError::RuntimeDivisionByZero);
                }
                Ok(wrap(&l % &r))
            }

            // Bitwise
            "&" => Ok(wrap(&l & &r)),
            "|" => Ok(wrap(&l | &r)),
            "^" => Ok(wrap(&l ^ &r)),

            // Shifts -- RHS is the shift amount
            ">>" => {
                let shift = r.to_u64().unwrap_or(0) as u16;
                Ok(wrap(&l >> shift))
            }
            "<<" => {
                let shift = r.to_u64().unwrap_or(0) as u16;
                Ok(wrap(&l << shift))
            }

            // Comparisons -- use BitUInt ordering (unsigned)
            // For signed comparisons, we'd need to check sign bits.
            // For now, numeric comparison via as_u64/as_i64 for <=64-bit,
            // and BitUInt ordering for wider values.
            "==" => Ok(ExprValue::unsigned(u64::from(l == r))),
            "!=" => Ok(ExprValue::unsigned(u64::from(l != r))),
            "<" => Ok(ExprValue::unsigned(u64::from(l < r))),
            ">" => Ok(ExprValue::unsigned(u64::from(l > r))),
            "<=" => Ok(ExprValue::unsigned(u64::from(l <= r))),
            ">=" => Ok(ExprValue::unsigned(u64::from(l >= r))),

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
}
