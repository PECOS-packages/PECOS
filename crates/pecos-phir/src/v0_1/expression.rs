use pecos_core::errors::PecosError;
use std::collections::HashMap;
use std::fmt;
use crate::v0_1::ast::{ArgItem, Expression};
use crate::v0_1::environment::{DataType, Environment, TypedValue};

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
    pub fn as_i64(&self) -> i64 {
        match self {
            ExprValue::Integer(val) => *val,
            ExprValue::UInteger(val) => *val as i64,
            ExprValue::Boolean(val) => if *val { 1 } else { 0 },
        }
    }

    /// Converts the expression value to u64 for calculations
    pub fn as_u64(&self) -> u64 {
        match self {
            ExprValue::Integer(val) => *val as u64,
            ExprValue::UInteger(val) => *val,
            ExprValue::Boolean(val) => if *val { 1 } else { 0 },
        }
    }

    /// Converts the expression value to boolean
    pub fn as_bool(&self) -> bool {
        match self {
            ExprValue::Integer(val) => *val != 0,
            ExprValue::UInteger(val) => *val != 0,
            ExprValue::Boolean(val) => *val,
        }
    }

    /// Converts a TypedValue to an ExprValue
    pub fn from_typed_value(value: TypedValue) -> Self {
        match value {
            TypedValue::I8(val) => ExprValue::Integer(val as i64),
            TypedValue::I16(val) => ExprValue::Integer(val as i64),
            TypedValue::I32(val) => ExprValue::Integer(val as i64),
            TypedValue::I64(val) => ExprValue::Integer(val),
            TypedValue::U8(val) => ExprValue::UInteger(val as u64),
            TypedValue::U16(val) => ExprValue::UInteger(val as u64),
            TypedValue::U32(val) => ExprValue::UInteger(val as u64),
            TypedValue::U64(val) => ExprValue::UInteger(val),
            TypedValue::Bool(val) => ExprValue::Boolean(val),
        }
    }

    /// Converts an ExprValue to a TypedValue with a specific data type
    pub fn to_typed_value(&self, data_type: &DataType) -> TypedValue {
        match data_type {
            DataType::I8 => TypedValue::I8(self.as_i64() as i8),
            DataType::I16 => TypedValue::I16(self.as_i64() as i16),
            DataType::I32 => TypedValue::I32(self.as_i64() as i32),
            DataType::I64 => TypedValue::I64(self.as_i64()),
            DataType::U8 => TypedValue::U8(self.as_u64() as u8),
            DataType::U16 => TypedValue::U16(self.as_u64() as u16),
            DataType::U32 => TypedValue::U32(self.as_u64() as u32),
            DataType::U64 => TypedValue::U64(self.as_u64()),
            DataType::Bool => TypedValue::Bool(self.as_bool()),
            DataType::Qubits => TypedValue::U64(self.as_u64()), // Qubits as u64 for now
        }
    }
}

/// Evaluator for expressions with type information
pub struct ExpressionEvaluator<'a> {
    /// Environment for variable lookups
    environment: &'a Environment,
    /// Cache for variable lookups to improve performance
    var_cache: HashMap<String, ExprValue>,
    /// Cache for expression evaluation results
    expr_cache: HashMap<String, ExprValue>,
}

impl<'a> ExpressionEvaluator<'a> {
    /// Creates a new expression evaluator with the given environment
    pub fn new(environment: &'a Environment) -> Self {
        Self {
            environment,
            var_cache: HashMap::new(),
            expr_cache: HashMap::new(),
        }
    }
    
    /// Creates a new expression evaluator with pre-allocated cache sizes
    pub fn with_capacity(environment: &'a Environment, var_capacity: usize, expr_capacity: usize) -> Self {
        Self {
            environment,
            var_cache: HashMap::with_capacity(var_capacity),
            expr_cache: HashMap::with_capacity(expr_capacity),
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
    fn expr_to_cache_key(&self, expr: &Expression) -> String {
        match expr {
            Expression::Integer(val) => format!("int:{}", val),
            Expression::Variable(name) => format!("var:{}", name),
            Expression::Operation { cop, args } => {
                let mut key = format!("op:{}", cop);
                for arg in args {
                    match arg {
                        ArgItem::Simple(name) => key.push_str(&format!(",simple:{}", name)),
                        ArgItem::Indexed((name, idx)) => key.push_str(&format!(",indexed:{}[{}]", name, idx)),
                        ArgItem::Integer(val) => key.push_str(&format!(",int:{}", val)),
                        ArgItem::Expression(expr) => key.push_str(&format!(",expr:{}", self.expr_to_cache_key(expr))),
                    }
                }
                key
            }
        }
    }
    
    /// Evaluates an expression to an ExprValue with caching
    pub fn eval_expr(&mut self, expr: &Expression) -> Result<ExprValue, PecosError> {
        // For simple expressions, don't bother with caching
        match expr {
            Expression::Integer(val) => {
                // Check if the value fits in i64
                if *val >= 0 {
                    return Ok(ExprValue::Integer(*val));
                } else {
                    // This shouldn't happen as integers are parsed as positive
                    return Ok(ExprValue::Integer(*val));
                }
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
                } else {
                    return Err(PecosError::Input(format!("Variable '{}' not found", name)));
                }
            }
            _ => {}
        }
        
        // For complex expressions, use caching
        let cache_key = self.expr_to_cache_key(expr);
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
                                "Unary operation '{}' requires exactly 1 argument", cop
                            )));
                        }
                        self.eval_unary_op(cop, &args[0])
                    }
                    // Short-circuit logical operations
                    "&&" => {
                        if args.len() != 2 {
                            return Err(PecosError::Input(format!(
                                "Logical AND operation requires exactly 2 arguments"
                            )));
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
                            return Err(PecosError::Input(format!(
                                "Logical OR operation requires exactly 2 arguments"
                            )));
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
                                "Binary operation '{}' requires exactly 2 arguments", cop
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
    
    /// Converts an ExprValue to a bit string of the specified width
    pub fn to_bit_string(&self, value: &ExprValue, width: usize) -> String {
        let bits = match value {
            ExprValue::Integer(val) => format!("{:b}", *val as u64),
            ExprValue::UInteger(val) => format!("{:b}", val),
            ExprValue::Boolean(val) => if *val { "1".to_string() } else { "0".to_string() },
        };
        
        // Pad with zeros to the requested width
        format!("{:0>width$}", bits, width = width)
    }
    
    /// Extract bits from a value as a vector of booleans
    pub fn extract_bits(&self, value: &ExprValue, indices: &[usize]) -> Vec<bool> {
        let value_u64 = value.as_u64();
        indices.iter()
            .map(|&idx| ((value_u64 >> idx) & 1) != 0)
            .collect()
    }

    /// Evaluates an argument to an ExprValue
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
                    Err(PecosError::Input(format!("Variable '{}' not found", name)))
                }
            }
            ArgItem::Indexed((name, idx)) => {
                // Bit access
                if let Ok(bit) = self.environment.get_bit(name, *idx) {
                    Ok(ExprValue::Boolean(bit.0))
                } else {
                    Err(PecosError::Input(format!(
                        "Failed to access bit {}[{}]", name, idx
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
            _ => Err(PecosError::Input(format!("Unsupported unary operation: {}", op)))
        }
    }

    /// Evaluates a binary operation with proper type handling
    fn eval_binary_op(&mut self, op: &str, lhs: &ArgItem, rhs: &ArgItem) -> Result<ExprValue, PecosError> {
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
                    Ok(ExprValue::Integer(lhs_val.as_i64().wrapping_add(rhs_val.as_i64())))
                } else {
                    Ok(ExprValue::UInteger(lhs_val.as_u64().wrapping_add(rhs_val.as_u64())))
                }
            }
            "-" => {
                if result_signed {
                    Ok(ExprValue::Integer(lhs_val.as_i64().wrapping_sub(rhs_val.as_i64())))
                } else {
                    Ok(ExprValue::UInteger(lhs_val.as_u64().wrapping_sub(rhs_val.as_u64())))
                }
            }
            "*" => {
                if result_signed {
                    Ok(ExprValue::Integer(lhs_val.as_i64().wrapping_mul(rhs_val.as_i64())))
                } else {
                    Ok(ExprValue::UInteger(lhs_val.as_u64().wrapping_mul(rhs_val.as_u64())))
                }
            }
            "/" => {
                if result_signed {
                    // Handle division by zero
                    if rhs_val.as_i64() == 0 {
                        return Err(PecosError::Input("Division by zero".to_string()));
                    }
                    Ok(ExprValue::Integer(lhs_val.as_i64().wrapping_div(rhs_val.as_i64())))
                } else {
                    // Handle division by zero
                    if rhs_val.as_u64() == 0 {
                        return Err(PecosError::Input("Division by zero".to_string()));
                    }
                    Ok(ExprValue::UInteger(lhs_val.as_u64().wrapping_div(rhs_val.as_u64())))
                }
            }
            "%" => {
                if result_signed {
                    // Handle modulo by zero
                    if rhs_val.as_i64() == 0 {
                        return Err(PecosError::Input("Modulo by zero".to_string()));
                    }
                    Ok(ExprValue::Integer(lhs_val.as_i64().wrapping_rem(rhs_val.as_i64())))
                } else {
                    // Handle modulo by zero
                    if rhs_val.as_u64() == 0 {
                        return Err(PecosError::Input("Modulo by zero".to_string()));
                    }
                    Ok(ExprValue::UInteger(lhs_val.as_u64().wrapping_rem(rhs_val.as_u64())))
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
                    if shift < 0 || shift >= 64 {
                        return Err(PecosError::Input("Invalid shift amount".to_string()));
                    }
                    Ok(ExprValue::Integer(lhs_val.as_i64().wrapping_shl(shift as u32)))
                } else {
                    let shift = rhs_val.as_u64();
                    if shift >= 64 {
                        return Err(PecosError::Input("Invalid shift amount".to_string()));
                    }
                    Ok(ExprValue::UInteger(lhs_val.as_u64().wrapping_shl(shift as u32)))
                }
            }
            ">>" => {
                // Shift operations promote to unsigned
                if result_signed {
                    let shift = rhs_val.as_i64();
                    if shift < 0 || shift >= 64 {
                        return Err(PecosError::Input("Invalid shift amount".to_string()));
                    }
                    Ok(ExprValue::Integer(lhs_val.as_i64().wrapping_shr(shift as u32)))
                } else {
                    let shift = rhs_val.as_u64();
                    if shift >= 64 {
                        return Err(PecosError::Input("Invalid shift amount".to_string()));
                    }
                    Ok(ExprValue::UInteger(lhs_val.as_u64().wrapping_shr(shift as u32)))
                }
            }
            
            // Comparison operations (always return boolean)
            "==" => Ok(ExprValue::Boolean(
                if result_signed {
                    lhs_val.as_i64() == rhs_val.as_i64()
                } else {
                    lhs_val.as_u64() == rhs_val.as_u64()
                }
            )),
            "!=" => Ok(ExprValue::Boolean(
                if result_signed {
                    lhs_val.as_i64() != rhs_val.as_i64()
                } else {
                    lhs_val.as_u64() != rhs_val.as_u64()
                }
            )),
            "<" => Ok(ExprValue::Boolean(
                if result_signed {
                    lhs_val.as_i64() < rhs_val.as_i64()
                } else {
                    lhs_val.as_u64() < rhs_val.as_u64()
                }
            )),
            "<=" => Ok(ExprValue::Boolean(
                if result_signed {
                    lhs_val.as_i64() <= rhs_val.as_i64()
                } else {
                    lhs_val.as_u64() <= rhs_val.as_u64()
                }
            )),
            ">" => Ok(ExprValue::Boolean(
                if result_signed {
                    lhs_val.as_i64() > rhs_val.as_i64()
                } else {
                    lhs_val.as_u64() > rhs_val.as_u64()
                }
            )),
            ">=" => Ok(ExprValue::Boolean(
                if result_signed {
                    lhs_val.as_i64() >= rhs_val.as_i64()
                } else {
                    lhs_val.as_u64() >= rhs_val.as_u64()
                }
            )),
            
            // Logical operations (always return boolean)
            "&&" => Ok(ExprValue::Boolean(lhs_val.as_bool() && rhs_val.as_bool())),
            "||" => Ok(ExprValue::Boolean(lhs_val.as_bool() || rhs_val.as_bool())),
            
            _ => Err(PecosError::Input(format!("Unsupported binary operation: {}", op)))
        }
    }
}

// Implement Display trait for ExprValue to allow formatting in log messages
impl fmt::Display for ExprValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExprValue::Integer(val) => write!(f, "{}", val),
            ExprValue::UInteger(val) => write!(f, "{}", val),
            ExprValue::Boolean(val) => write!(f, "{}", val),
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
        self.as_i64() == *other as i64
    }
}

impl PartialEq<u32> for ExprValue {
    fn eq(&self, other: &u32) -> bool {
        self.as_u64() == *other as u64
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
        *self as i64 == other.as_i64()
    }
}

impl PartialEq<ExprValue> for u32 {
    fn eq(&self, other: &ExprValue) -> bool {
        *self as u64 == other.as_u64()
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
        assert_eq!(result.as_bool(), true); // 255 has bit 0 set
    }

    #[test]
    fn test_arithmetic_operations() {
        let env = setup_environment();
        let mut evaluator = ExpressionEvaluator::new(&env);
        
        // Test addition
        let expr = Expression::Operation {
            cop: "+".to_string(),
            args: vec![
                ArgItem::Simple("x".to_string()),
                ArgItem::Integer(10),
            ],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert_eq!(result.as_i64(), 52); // 42 + 10
        
        // Test subtraction
        let expr = Expression::Operation {
            cop: "-".to_string(),
            args: vec![
                ArgItem::Simple("x".to_string()),
                ArgItem::Integer(10),
            ],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert_eq!(result.as_i64(), 32); // 42 - 10
        
        // Test multiplication
        let expr = Expression::Operation {
            cop: "*".to_string(),
            args: vec![
                ArgItem::Simple("x".to_string()),
                ArgItem::Integer(2),
            ],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert_eq!(result.as_i64(), 84); // 42 * 2
        
        // Test division
        let expr = Expression::Operation {
            cop: "/".to_string(),
            args: vec![
                ArgItem::Simple("x".to_string()),
                ArgItem::Integer(2),
            ],
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
            args: vec![
                ArgItem::Simple("x".to_string()),
                ArgItem::Integer(15),
            ],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert_eq!(result.as_i64(), 10); // 42 & 15 = 0b101010 & 0b1111 = 0b1010 = 10
        
        // Test bitwise OR
        let expr = Expression::Operation {
            cop: "|".to_string(),
            args: vec![
                ArgItem::Simple("x".to_string()),
                ArgItem::Integer(15),
            ],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert_eq!(result.as_i64(), 47); // 42 | 15 = 0b101010 | 0b1111 = 0b101111 = 47
        
        // Test bitwise XOR
        let expr = Expression::Operation {
            cop: "^".to_string(),
            args: vec![
                ArgItem::Simple("x".to_string()),
                ArgItem::Integer(15),
            ],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert_eq!(result.as_i64(), 37); // 42 ^ 15 = 0b101010 ^ 0b1111 = 0b100101 = 37
        
        // Test bitwise NOT
        let expr = Expression::Operation {
            cop: "~".to_string(),
            args: vec![
                ArgItem::Simple("z".to_string()),
            ],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert_eq!(result.as_bool(), false); // ~true = false
    }

    #[test]
    fn test_comparison_operations() {
        let env = setup_environment();
        let mut evaluator = ExpressionEvaluator::new(&env);
        
        // Test equality
        let expr = Expression::Operation {
            cop: "==".to_string(),
            args: vec![
                ArgItem::Simple("x".to_string()),
                ArgItem::Integer(42),
            ],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert_eq!(result.as_bool(), true); // 42 == 42
        
        // Test inequality
        let expr = Expression::Operation {
            cop: "!=".to_string(),
            args: vec![
                ArgItem::Simple("x".to_string()),
                ArgItem::Integer(41),
            ],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert_eq!(result.as_bool(), true); // 42 != 41
        
        // Test less than
        let expr = Expression::Operation {
            cop: "<".to_string(),
            args: vec![
                ArgItem::Simple("x".to_string()),
                ArgItem::Integer(50),
            ],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert_eq!(result.as_bool(), true); // 42 < 50
        
        // Test greater than
        let expr = Expression::Operation {
            cop: ">".to_string(),
            args: vec![
                ArgItem::Simple("x".to_string()),
                ArgItem::Integer(10),
            ],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert_eq!(result.as_bool(), true); // 42 > 10
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
        assert_eq!(result.as_bool(), true); // true && true
        
        // Test logical OR
        let expr = Expression::Operation {
            cop: "||".to_string(),
            args: vec![
                ArgItem::Simple("z".to_string()),
                ArgItem::Integer(0),
            ],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert_eq!(result.as_bool(), true); // true || false
        
        // Test logical NOT
        let expr = Expression::Operation {
            cop: "!".to_string(),
            args: vec![
                ArgItem::Integer(0),
            ],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert_eq!(result.as_bool(), true); // !false
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
                    args: vec![
                        ArgItem::Simple("x".to_string()),
                        ArgItem::Integer(5),
                    ],
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
                            args: vec![
                                ArgItem::Simple("x".to_string()),
                                ArgItem::Integer(40),
                            ],
                        })),
                        ArgItem::Expression(Box::new(Expression::Operation {
                            cop: "<".to_string(),
                            args: vec![
                                ArgItem::Simple("y".to_string()),
                                ArgItem::Integer(10),
                            ],
                        })),
                    ],
                })),
                ArgItem::Simple("z".to_string()),
            ],
        };
        let result = evaluator.eval_expr(&expr).unwrap();
        assert_eq!(result.as_bool(), true); // (42 > 40 && 255 < 10) || true = (true && false) || true = false || true = true
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
        assert_eq!(result.as_bool(), false); // false && (anything) short-circuits to false
        
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
        assert_eq!(result.as_bool(), true); // true || (anything) short-circuits to true
    }
}