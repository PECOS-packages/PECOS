use crate::v0_1::ast::{ArgItem, Expression};
use crate::v0_1::environment::{DataType, Environment, TypedValue};
use pecos_core::errors::PecosError;
use std::collections::BTreeMap;
use std::fmt::{self, Write};

/// Expression value with type information
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExprValue {
    /// Integer value with sign information
    Integer(i64),
    /// Unsigned integer value
    UInteger(u64),
    /// Boolean value
    Boolean(bool),
}

impl ExprValue {
    /// Converts the expression value to i64 for calculations
    #[must_use]
    #[allow(clippy::cast_possible_wrap)]
    pub fn as_i64(&self) -> i64 {
        match self {
            ExprValue::Integer(val) => *val,
            ExprValue::UInteger(val) => *val as i64,
            ExprValue::Boolean(val) => i64::from(*val),
        }
    }

    /// Converts the expression value to u64 for calculations
    #[must_use]
    #[allow(clippy::cast_sign_loss)]
    pub fn as_u64(&self) -> u64 {
        match self {
            ExprValue::Integer(val) => *val as u64,
            ExprValue::UInteger(val) => *val,
            ExprValue::Boolean(val) => u64::from(*val),
        }
    }

    /// Converts the expression value to boolean
    #[must_use]
    pub fn as_bool(&self) -> bool {
        match self {
            ExprValue::Integer(val) => *val != 0,
            ExprValue::UInteger(val) => *val != 0,
            ExprValue::Boolean(val) => *val,
        }
    }

    /// Converts a `TypedValue` to an `ExprValue`
    #[must_use]
    pub fn from_typed_value(value: TypedValue) -> Self {
        match value {
            TypedValue::I8(val) => ExprValue::Integer(i64::from(val)),
            TypedValue::I16(val) => ExprValue::Integer(i64::from(val)),
            TypedValue::I32(val) => ExprValue::Integer(i64::from(val)),
            TypedValue::I64(val) => ExprValue::Integer(val),
            TypedValue::U8(val) => ExprValue::UInteger(u64::from(val)),
            TypedValue::U16(val) => ExprValue::UInteger(u64::from(val)),
            TypedValue::U32(val) => ExprValue::UInteger(u64::from(val)),
            TypedValue::U64(val) => ExprValue::UInteger(val),
            TypedValue::Bool(val) => ExprValue::Boolean(val),
        }
    }

    /// Helper function to convert signed integer with bounds checking
    fn convert_signed_int<T>(val: i64, type_name: &str) -> Result<T, PecosError>
    where
        T: TryFrom<i64> + Copy,
        T::Error: std::fmt::Debug,
    {
        T::try_from(val)
            .map_err(|_| PecosError::Input(format!("Value {val} out of range for {type_name}")))
    }

    /// Helper function to convert to unsigned type from any `ExprValue`
    fn convert_to_unsigned<T>(
        &self,
        type_name: &str,
        bool_converter: fn(bool) -> T,
    ) -> Result<T, PecosError>
    where
        T: TryFrom<i64> + TryFrom<u64> + Copy,
        <T as TryFrom<i64>>::Error: std::fmt::Debug,
        <T as TryFrom<u64>>::Error: std::fmt::Debug,
    {
        match self {
            ExprValue::Integer(val) => T::try_from(*val).map_err(|_| {
                PecosError::Input(format!("Value {val} out of range for {type_name}"))
            }),
            ExprValue::UInteger(val) => T::try_from(*val).map_err(|_| {
                PecosError::Input(format!("Value {val} out of range for {type_name}"))
            }),
            ExprValue::Boolean(val) => Ok(bool_converter(*val)),
        }
    }

    /// Converts an `ExprValue` to a `TypedValue` with a specific data type
    ///
    /// Returns an error if the value cannot be safely converted to the target type
    ///
    /// # Errors
    /// Returns an error if the value is out of range for the target data type.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn to_typed_value(&self, data_type: &DataType) -> Result<TypedValue, PecosError> {
        match data_type {
            DataType::I8 => {
                let val = self.as_i64();
                Ok(TypedValue::I8(Self::convert_signed_int(val, "i8")?))
            }
            DataType::I16 => {
                let val = self.as_i64();
                Ok(TypedValue::I16(Self::convert_signed_int(val, "i16")?))
            }
            DataType::I32 => {
                let val = self.as_i64();
                Ok(TypedValue::I32(Self::convert_signed_int(val, "i32")?))
            }
            DataType::I64 => Ok(TypedValue::I64(self.as_i64())),
            DataType::U8 => Ok(TypedValue::U8(self.convert_to_unsigned("u8", u8::from)?)),
            DataType::U16 => Ok(TypedValue::U16(self.convert_to_unsigned("u16", u16::from)?)),
            DataType::U32 => Ok(TypedValue::U32(self.convert_to_unsigned("u32", u32::from)?)),
            DataType::U64 | DataType::Qubits => {
                let typename = match data_type {
                    DataType::U64 => "u64",
                    DataType::Qubits => "qubits",
                    _ => unreachable!(),
                };
                Ok(TypedValue::U64(
                    self.convert_to_unsigned(typename, u64::from)?,
                ))
            }
            DataType::Bool => Ok(TypedValue::Bool(self.as_bool())),
        }
    }
}

/// Evaluator for expressions with type information
pub struct ExpressionEvaluator<'a> {
    /// Environment for variable lookups
    environment: &'a Environment,
    /// Cache for variable lookups to improve performance
    var_cache: BTreeMap<String, ExprValue>,
    /// Cache for expression evaluation results
    expr_cache: BTreeMap<String, ExprValue>,
}

impl<'a> ExpressionEvaluator<'a> {
    /// Creates a new expression evaluator with the given environment
    #[must_use]
    pub fn new(environment: &'a Environment) -> Self {
        Self {
            environment,
            var_cache: BTreeMap::new(),
            expr_cache: BTreeMap::new(),
        }
    }

    /// Creates a new expression evaluator with pre-allocated cache sizes
    #[must_use]
    pub fn with_capacity(
        environment: &'a Environment,
        _var_capacity: usize,
        _expr_capacity: usize,
    ) -> Self {
        Self {
            environment,
            var_cache: BTreeMap::new(),
            expr_cache: BTreeMap::new(),
        }
    }

    /// Clears the expression cache but keeps variable cache
    pub fn clear_expr_cache(&mut self) {
        self.expr_cache.clear();
    }

    /// Clears all caches
    pub fn clear_caches(&mut self) {
        self.var_cache.clear();
        self.expr_cache.clear();
    }

    /// Converts an expression to a string for caching
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
                        ArgItem::Expression(expr) => {
                            write!(&mut key, ",expr:{}", Self::expr_to_cache_key(expr)).unwrap();
                        }
                    }
                }
                key
            }
        }
    }

    /// Evaluates an expression to an `ExprValue` with caching
    ///
    /// # Errors
    /// Returns an error if:
    /// - Variables referenced in the expression don't exist
    /// - Binary/unary operations are unsupported
    /// - Arguments to operations are invalid
    pub fn eval_expr(&mut self, expr: &Expression) -> Result<ExprValue, PecosError> {
        // For simple expressions, don't bother with caching
        match expr {
            Expression::Integer(val) => {
                // Check if the value fits in i64
                if *val >= 0 {
                    return Ok(ExprValue::Integer(*val));
                }
                // This shouldn't happen as integers are parsed as positive
                return Ok(ExprValue::Integer(*val));
            }
            Expression::Variable(name) => {
                // Check if the variable exists in the cache
                if let Some(val) = self.var_cache.get(name) {
                    return Ok(*val);
                }

                // Lookup the variable in the environment
                if let Some(value) = self.environment.get(name) {
                    let expr_val = ExprValue::from_typed_value(value);
                    // Update cache for future lookups
                    self.var_cache.insert(name.clone(), expr_val);
                    return Ok(expr_val);
                }
                return Err(PecosError::Input(format!("Variable '{name}' not found")));
            }
            Expression::Operation { .. } => {}
        }

        // For complex expressions, use caching
        let cache_key = Self::expr_to_cache_key(expr);
        if let Some(cached_value) = self.expr_cache.get(&cache_key) {
            return Ok(*cached_value);
        }

        // If not in cache, evaluate and store result
        let result = match expr {
            Expression::Operation { cop, args } => {
                // Handle operations based on type
                match cop.as_str() {
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
                                "Logical AND operation requires exactly 2 arguments".to_string(),
                            ));
                        }
                        // Short-circuit evaluation
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
                                "Logical OR operation requires exactly 2 arguments".to_string(),
                            ));
                        }
                        // Short-circuit evaluation
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
                }
            }
            // These cases are handled above
            Expression::Integer(_) | Expression::Variable(_) => unreachable!(),
        }?;

        // Cache the result
        self.expr_cache.insert(cache_key, result);
        Ok(result)
    }

    /// Converts an `ExprValue` to a bit string of the specified width
    #[must_use]
    pub fn to_bit_string(&self, value: &ExprValue, width: usize) -> String {
        let bits = match value {
            ExprValue::Integer(val) => {
                // Use from_ne_bytes for a reinterpret cast to preserve bit pattern
                let unsigned = u64::from_ne_bytes((*val).to_ne_bytes());
                format!("{unsigned:b}")
            }
            ExprValue::UInteger(val) => format!("{val:b}"),
            ExprValue::Boolean(val) => {
                if *val {
                    "1".to_string()
                } else {
                    "0".to_string()
                }
            }
        };

        // Pad with zeros to the requested width
        format!("{bits:0>width$}")
    }

    /// Extract bits from a value as a vector of booleans
    #[must_use]
    pub fn extract_bits(&self, value: &ExprValue, indices: &[usize]) -> Vec<bool> {
        let value_u64 = value.as_u64();
        indices
            .iter()
            .map(|&idx| ((value_u64 >> idx) & 1) != 0)
            .collect()
    }

    /// Evaluates an argument to an `ExprValue`
    ///
    /// # Errors
    /// Returns an error if:
    /// - Variable referenced doesn't exist
    /// - Bit access is invalid
    /// - Nested expression evaluation fails
    pub fn eval_arg(&mut self, arg: &ArgItem) -> Result<ExprValue, PecosError> {
        match arg {
            ArgItem::Simple(name) => {
                // Simple variable reference
                // Check if the variable exists in the cache
                if let Some(val) = self.var_cache.get(name) {
                    return Ok(*val);
                }

                // Lookup the variable in the environment
                if let Some(value) = self.environment.get(name) {
                    let expr_val = ExprValue::from_typed_value(value);
                    // Update cache for future lookups
                    self.var_cache.insert(name.clone(), expr_val);
                    Ok(expr_val)
                } else {
                    Err(PecosError::Input(format!("Variable '{name}' not found")))
                }
            }
            ArgItem::Indexed((name, idx)) => {
                // Bit access
                if let Ok(bit) = self.environment.get_bit(name, *idx) {
                    Ok(ExprValue::Boolean(bit.0))
                } else {
                    Err(PecosError::Input(format!(
                        "Failed to access bit {name}[{idx}]"
                    )))
                }
            }
            ArgItem::Integer(val) => {
                // Integer literal
                if *val >= 0 {
                    Ok(ExprValue::Integer(*val))
                } else {
                    // This shouldn't happen as integers are parsed as positive
                    Ok(ExprValue::Integer(*val))
                }
            }
            ArgItem::Expression(expr) => {
                // Nested expression
                self.eval_expr(expr)
            }
        }
    }

    /// Evaluates a unary operation
    fn eval_unary_op(&mut self, op: &str, arg: &ArgItem) -> Result<ExprValue, PecosError> {
        let val = self.eval_arg(arg)?;

        match op {
            "~" => {
                // Bitwise NOT
                match val {
                    ExprValue::Integer(v) => Ok(ExprValue::Integer(!v)),
                    ExprValue::UInteger(v) => Ok(ExprValue::UInteger(!v)),
                    ExprValue::Boolean(v) => Ok(ExprValue::Boolean(!v)),
                }
            }
            "!" => {
                // Logical NOT
                Ok(ExprValue::Boolean(!val.as_bool()))
            }
            _ => Err(PecosError::Input(format!(
                "Unsupported unary operation: {op}"
            ))),
        }
    }

    /// Evaluates a binary operation with proper type handling
    #[allow(clippy::too_many_lines)]
    fn eval_binary_op(
        &mut self,
        op: &str,
        lhs: &ArgItem,
        rhs: &ArgItem,
    ) -> Result<ExprValue, PecosError> {
        let lhs_val = self.eval_arg(lhs)?;
        let rhs_val = self.eval_arg(rhs)?;

        // Promote types based on Python's promotion rules
        // If both operands are signed, result is signed
        // If any operand is unsigned, result is unsigned if it fits, otherwise signed
        let lhs_signed = matches!(lhs_val, ExprValue::Integer(_));
        let rhs_signed = matches!(rhs_val, ExprValue::Integer(_));

        let result_signed = lhs_signed && rhs_signed;

        match op {
            // Arithmetic operations
            "+" => {
                if result_signed {
                    Ok(ExprValue::Integer(
                        lhs_val.as_i64().wrapping_add(rhs_val.as_i64()),
                    ))
                } else {
                    Ok(ExprValue::UInteger(
                        lhs_val.as_u64().wrapping_add(rhs_val.as_u64()),
                    ))
                }
            }
            "-" => {
                if result_signed {
                    Ok(ExprValue::Integer(
                        lhs_val.as_i64().wrapping_sub(rhs_val.as_i64()),
                    ))
                } else {
                    Ok(ExprValue::UInteger(
                        lhs_val.as_u64().wrapping_sub(rhs_val.as_u64()),
                    ))
                }
            }
            "*" => {
                if result_signed {
                    Ok(ExprValue::Integer(
                        lhs_val.as_i64().wrapping_mul(rhs_val.as_i64()),
                    ))
                } else {
                    Ok(ExprValue::UInteger(
                        lhs_val.as_u64().wrapping_mul(rhs_val.as_u64()),
                    ))
                }
            }
            "/" => {
                if rhs_val == 0 {
                    return Err(PecosError::RuntimeDivisionByZero);
                }
                if result_signed {
                    Ok(ExprValue::Integer(lhs_val.as_i64() / rhs_val.as_i64()))
                } else {
                    Ok(ExprValue::UInteger(lhs_val.as_u64() / rhs_val.as_u64()))
                }
            }
            "%" => {
                if rhs_val == 0 {
                    return Err(PecosError::RuntimeDivisionByZero);
                }
                if result_signed {
                    Ok(ExprValue::Integer(lhs_val.as_i64() % rhs_val.as_i64()))
                } else {
                    Ok(ExprValue::UInteger(lhs_val.as_u64() % rhs_val.as_u64()))
                }
            }

            // Bitwise operations
            "&" => {
                if result_signed {
                    Ok(ExprValue::Integer(lhs_val.as_i64() & rhs_val.as_i64()))
                } else {
                    Ok(ExprValue::UInteger(lhs_val.as_u64() & rhs_val.as_u64()))
                }
            }
            "|" => {
                if result_signed {
                    Ok(ExprValue::Integer(lhs_val.as_i64() | rhs_val.as_i64()))
                } else {
                    Ok(ExprValue::UInteger(lhs_val.as_u64() | rhs_val.as_u64()))
                }
            }
            "^" => {
                if result_signed {
                    Ok(ExprValue::Integer(lhs_val.as_i64() ^ rhs_val.as_i64()))
                } else {
                    Ok(ExprValue::UInteger(lhs_val.as_u64() ^ rhs_val.as_u64()))
                }
            }
            "<<" => {
                // Shift operations promote to unsigned
                if result_signed {
                    let shift = rhs_val.as_i64();
                    if !(0..64).contains(&shift) {
                        return Err(PecosError::Input("Invalid shift amount".to_string()));
                    }
                    let shift_u32 = u32::try_from(shift)
                        .map_err(|_| PecosError::Input("Invalid shift amount".to_string()))?;
                    Ok(ExprValue::Integer(lhs_val.as_i64().wrapping_shl(shift_u32)))
                } else {
                    let shift = rhs_val.as_u64();
                    if shift >= 64 {
                        return Err(PecosError::Input("Invalid shift amount".to_string()));
                    }
                    let shift_u32 = u32::try_from(shift)
                        .map_err(|_| PecosError::Input("Invalid shift amount".to_string()))?;
                    Ok(ExprValue::UInteger(
                        lhs_val.as_u64().wrapping_shl(shift_u32),
                    ))
                }
            }
            ">>" => {
                // Shift operations promote to unsigned
                if result_signed {
                    let shift = rhs_val.as_i64();
                    if !(0..64).contains(&shift) {
                        return Err(PecosError::Input("Invalid shift amount".to_string()));
                    }
                    let shift_u32 = u32::try_from(shift)
                        .map_err(|_| PecosError::Input("Invalid shift amount".to_string()))?;
                    Ok(ExprValue::Integer(lhs_val.as_i64().wrapping_shr(shift_u32)))
                } else {
                    let shift = rhs_val.as_u64();
                    if shift >= 64 {
                        return Err(PecosError::Input("Invalid shift amount".to_string()));
                    }
                    let shift_u32 = u32::try_from(shift)
                        .map_err(|_| PecosError::Input("Invalid shift amount".to_string()))?;
                    Ok(ExprValue::UInteger(
                        lhs_val.as_u64().wrapping_shr(shift_u32),
                    ))
                }
            }

            // Comparison operations (always return boolean)
            "==" => Ok(ExprValue::Boolean(if result_signed {
                lhs_val.as_i64() == rhs_val.as_i64()
            } else {
                lhs_val.as_u64() == rhs_val.as_u64()
            })),
            "!=" => Ok(ExprValue::Boolean(if result_signed {
                lhs_val.as_i64() != rhs_val.as_i64()
            } else {
                lhs_val.as_u64() != rhs_val.as_u64()
            })),
            "<" => Ok(ExprValue::Boolean(if result_signed {
                lhs_val.as_i64() < rhs_val.as_i64()
            } else {
                lhs_val.as_u64() < rhs_val.as_u64()
            })),
            "<=" => Ok(ExprValue::Boolean(if result_signed {
                lhs_val.as_i64() <= rhs_val.as_i64()
            } else {
                lhs_val.as_u64() <= rhs_val.as_u64()
            })),
            ">" => Ok(ExprValue::Boolean(if result_signed {
                lhs_val.as_i64() > rhs_val.as_i64()
            } else {
                lhs_val.as_u64() > rhs_val.as_u64()
            })),
            ">=" => Ok(ExprValue::Boolean(if result_signed {
                lhs_val.as_i64() >= rhs_val.as_i64()
            } else {
                lhs_val.as_u64() >= rhs_val.as_u64()
            })),

            // Logical operations (always return boolean)
            "&&" => Ok(ExprValue::Boolean(lhs_val.as_bool() && rhs_val.as_bool())),
            "||" => Ok(ExprValue::Boolean(lhs_val.as_bool() || rhs_val.as_bool())),

            _ => Err(PecosError::Input(format!(
                "Unsupported binary operation: {op}"
            ))),
        }
    }
}

// Implement Display trait for ExprValue to allow formatting in log messages
impl fmt::Display for ExprValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExprValue::Integer(val) => write!(f, "{val}"),
            ExprValue::UInteger(val) => write!(f, "{val}"),
            ExprValue::Boolean(val) => write!(f, "{val}"),
        }
    }
}

// Implement PartialEq to allow comparing ExprValue with integers
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

impl PartialEq<i32> for ExprValue {
    fn eq(&self, other: &i32) -> bool {
        self.as_i64() == i64::from(*other)
    }
}

impl PartialEq<u32> for ExprValue {
    fn eq(&self, other: &u32) -> bool {
        self.as_u64() == u64::from(*other)
    }
}

impl PartialEq<ExprValue> for i64 {
    fn eq(&self, other: &ExprValue) -> bool {
        *self == other.as_i64()
    }
}

impl PartialEq<ExprValue> for u64 {
    fn eq(&self, other: &ExprValue) -> bool {
        *self == other.as_u64()
    }
}

impl PartialEq<ExprValue> for i32 {
    fn eq(&self, other: &ExprValue) -> bool {
        i64::from(*self) == other.as_i64()
    }
}

impl PartialEq<ExprValue> for u32 {
    fn eq(&self, other: &ExprValue) -> bool {
        u64::from(*self) == other.as_u64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_environment() -> Environment {
        let mut env = Environment::new();

        // Add variables
        env.add_variable("x", DataType::I32, 32).unwrap();
        env.add_variable("y", DataType::U8, 8).unwrap();
        env.add_variable("z", DataType::Bool, 1).unwrap();

        // Set values
        env.set_raw("x", 42).unwrap();
        env.set_raw("y", 255).unwrap();
        env.set_raw("z", 1).unwrap();

        env
    }

    #[test]
    fn test_simple_expressions() {
        let env = setup_environment();
        let mut evaluator = ExpressionEvaluator::new(&env);

        // Test integer literal
        let expr = Expression::Integer(123);
        let result = evaluator.eval_expr(&expr).unwrap();
        assert_eq!(result.as_i64(), 123);

        // Test variable reference
        let expr = Expression::Variable("x".to_string());
        let result = evaluator.eval_expr(&expr).unwrap();
        assert_eq!(result.as_i64(), 42);

        // Test bit access
        let arg = ArgItem::Indexed(("y".to_string(), 0));
        let result = evaluator.eval_arg(&arg).unwrap();
        assert!(result.as_bool()); // 255 has bit 0 set
    }

    #[test]
    fn test_arithmetic_operations() {
        let env = setup_environment();
        let mut evaluator = ExpressionEvaluator::new(&env);

        // Test addition
        let expr = Expression::Operation {
            cop: "+".to_string(),
            args: vec![ArgItem::Simple("x".to_string()), ArgItem::Integer(10)],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert_eq!(result.as_i64(), 52); // 42 + 10

        // Test subtraction
        let expr = Expression::Operation {
            cop: "-".to_string(),
            args: vec![ArgItem::Simple("x".to_string()), ArgItem::Integer(10)],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert_eq!(result.as_i64(), 32); // 42 - 10

        // Test multiplication
        let expr = Expression::Operation {
            cop: "*".to_string(),
            args: vec![ArgItem::Simple("x".to_string()), ArgItem::Integer(2)],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert_eq!(result.as_i64(), 84); // 42 * 2

        // Test division
        let expr = Expression::Operation {
            cop: "/".to_string(),
            args: vec![ArgItem::Simple("x".to_string()), ArgItem::Integer(2)],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert_eq!(result.as_i64(), 21); // 42 / 2
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

        // Test bitwise OR
        let expr = Expression::Operation {
            cop: "|".to_string(),
            args: vec![ArgItem::Simple("x".to_string()), ArgItem::Integer(15)],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert_eq!(result.as_i64(), 47); // 42 | 15 = 0b101010 | 0b1111 = 0b101111 = 47

        // Test bitwise XOR
        let expr = Expression::Operation {
            cop: "^".to_string(),
            args: vec![ArgItem::Simple("x".to_string()), ArgItem::Integer(15)],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert_eq!(result.as_i64(), 37); // 42 ^ 15 = 0b101010 ^ 0b1111 = 0b100101 = 37

        // Test bitwise NOT
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

        // Test equality
        let expr = Expression::Operation {
            cop: "==".to_string(),
            args: vec![ArgItem::Simple("x".to_string()), ArgItem::Integer(42)],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert!(result.as_bool()); // 42 == 42

        // Test inequality
        let expr = Expression::Operation {
            cop: "!=".to_string(),
            args: vec![ArgItem::Simple("x".to_string()), ArgItem::Integer(41)],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert!(result.as_bool()); // 42 != 41

        // Test less than
        let expr = Expression::Operation {
            cop: "<".to_string(),
            args: vec![ArgItem::Simple("x".to_string()), ArgItem::Integer(50)],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert!(result.as_bool()); // 42 < 50

        // Test greater than
        let expr = Expression::Operation {
            cop: ">".to_string(),
            args: vec![ArgItem::Simple("x".to_string()), ArgItem::Integer(10)],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert!(result.as_bool()); // 42 > 10
    }

    #[test]
    fn test_logical_operations() {
        let env = setup_environment();
        let mut evaluator = ExpressionEvaluator::new(&env);

        // Test logical AND
        let expr = Expression::Operation {
            cop: "&&".to_string(),
            args: vec![
                ArgItem::Simple("z".to_string()),
                ArgItem::Simple("z".to_string()),
            ],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert!(result.as_bool()); // true && true

        // Test logical OR
        let expr = Expression::Operation {
            cop: "||".to_string(),
            args: vec![ArgItem::Simple("z".to_string()), ArgItem::Integer(0)],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert!(result.as_bool()); // true || false

        // Test logical NOT
        let expr = Expression::Operation {
            cop: "!".to_string(),
            args: vec![ArgItem::Integer(0)],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert!(result.as_bool()); // !false
    }

    #[test]
    fn test_complex_expressions() {
        let env = setup_environment();
        let mut evaluator = ExpressionEvaluator::new(&env);

        // Test nested expression: (x + 5) * 2
        let expr = Expression::Operation {
            cop: "*".to_string(),
            args: vec![
                ArgItem::Expression(Box::new(Expression::Operation {
                    cop: "+".to_string(),
                    args: vec![ArgItem::Simple("x".to_string()), ArgItem::Integer(5)],
                })),
                ArgItem::Integer(2),
            ],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert_eq!(result.as_i64(), 94); // (42 + 5) * 2 = 94

        // Test complex expression: (x > 40 && y < 10) || z
        let expr = Expression::Operation {
            cop: "||".to_string(),
            args: vec![
                ArgItem::Expression(Box::new(Expression::Operation {
                    cop: "&&".to_string(),
                    args: vec![
                        ArgItem::Expression(Box::new(Expression::Operation {
                            cop: ">".to_string(),
                            args: vec![ArgItem::Simple("x".to_string()), ArgItem::Integer(40)],
                        })),
                        ArgItem::Expression(Box::new(Expression::Operation {
                            cop: "<".to_string(),
                            args: vec![ArgItem::Simple("y".to_string()), ArgItem::Integer(10)],
                        })),
                    ],
                })),
                ArgItem::Simple("z".to_string()),
            ],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert!(result.as_bool()); // (42 > 40 && 255 < 10) || true = (true && false) || true = false || true = true
    }

    #[test]
    fn test_short_circuit_evaluation() {
        let env = setup_environment();
        let mut evaluator = ExpressionEvaluator::new(&env);

        // Test short-circuit AND with false first operand
        let expr = Expression::Operation {
            cop: "&&".to_string(),
            args: vec![
                ArgItem::Integer(0), // false
                ArgItem::Expression(Box::new(Expression::Operation {
                    cop: "/".to_string(),
                    args: vec![
                        ArgItem::Integer(1),
                        ArgItem::Integer(0), // Division by zero, would cause error if evaluated
                    ],
                })),
            ],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert!(!result.as_bool()); // false && (anything) short-circuits to false

        // Test short-circuit OR with true first operand
        let expr = Expression::Operation {
            cop: "||".to_string(),
            args: vec![
                ArgItem::Integer(1), // true
                ArgItem::Expression(Box::new(Expression::Operation {
                    cop: "/".to_string(),
                    args: vec![
                        ArgItem::Integer(1),
                        ArgItem::Integer(0), // Division by zero, would cause error if evaluated
                    ],
                })),
            ],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert!(result.as_bool()); // true || (anything) short-circuits to true
    }

    #[test]
    fn test_to_typed_value_conversions() {
        // Test successful conversions
        let small_int = ExprValue::Integer(42);
        assert!(small_int.to_typed_value(&DataType::I8).is_ok());
        assert!(small_int.to_typed_value(&DataType::I16).is_ok());
        assert!(small_int.to_typed_value(&DataType::I32).is_ok());
        assert!(small_int.to_typed_value(&DataType::I64).is_ok());
        assert!(small_int.to_typed_value(&DataType::U8).is_ok());
        assert!(small_int.to_typed_value(&DataType::U16).is_ok());
        assert!(small_int.to_typed_value(&DataType::U32).is_ok());
        assert!(small_int.to_typed_value(&DataType::U64).is_ok());

        // Test overflow cases
        let large_int = ExprValue::Integer(1000);
        assert!(large_int.to_typed_value(&DataType::I8).is_err());
        assert!(large_int.to_typed_value(&DataType::U8).is_err());
        assert!(large_int.to_typed_value(&DataType::I16).is_ok());
        assert!(large_int.to_typed_value(&DataType::U16).is_ok());

        // Test negative values for unsigned types
        let negative_int = ExprValue::Integer(-1);
        assert!(negative_int.to_typed_value(&DataType::U8).is_err());
        assert!(negative_int.to_typed_value(&DataType::U16).is_err());
        assert!(negative_int.to_typed_value(&DataType::U32).is_err());
        assert!(negative_int.to_typed_value(&DataType::U64).is_err());
        assert!(negative_int.to_typed_value(&DataType::I8).is_ok());
        assert!(negative_int.to_typed_value(&DataType::I16).is_ok());

        // Test boolean conversion
        let bool_val = ExprValue::Boolean(true);
        assert!(bool_val.to_typed_value(&DataType::Bool).is_ok());

        // Test edge cases - max values
        let max_u8 = ExprValue::UInteger(255);
        assert!(max_u8.to_typed_value(&DataType::U8).is_ok());
        assert!(max_u8.to_typed_value(&DataType::U16).is_ok());

        let over_u8 = ExprValue::UInteger(256);
        assert!(over_u8.to_typed_value(&DataType::U8).is_err());
        assert!(over_u8.to_typed_value(&DataType::U16).is_ok());
    }
}
