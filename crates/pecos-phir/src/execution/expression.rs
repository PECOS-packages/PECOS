/*!
PHIR Expression Evaluator

Expression evaluation for classical computations in PHIR execution.
This handles arithmetic, logical, and comparison operations on variables.
*/

use super::environment::{Environment, TypedValue};
use crate::error::{PhirError, Result};

/// Expression evaluator for PHIR classical computations
#[derive(Debug, Clone)]
pub struct ExpressionEvaluator {
    environment: Environment,
}

impl ExpressionEvaluator {
    /// Create a new expression evaluator
    #[must_use]
    pub fn new(environment: Environment) -> Self {
        Self { environment }
    }

    /// Evaluate a simple variable reference
    ///
    /// # Errors
    ///
    /// Returns an error if the variable is not found or not initialized
    pub fn evaluate_variable(&self, var_name: &str) -> Result<TypedValue> {
        match self.environment.get_variable(var_name)? {
            Some(value) => Ok(value.clone()),
            None => Err(PhirError::internal(format!(
                "Variable '{var_name}' is not initialized"
            ))),
        }
    }

    /// Evaluate a constant value
    #[must_use]
    pub fn evaluate_constant(&self, value: i64) -> TypedValue {
        // Default to I64 for constants
        TypedValue::I64(value)
    }

    /// Evaluate binary arithmetic operation
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The operation is not supported
    /// - Type mismatch between operands
    /// - The specific operation implementation fails
    pub fn evaluate_binary_op(
        &self,
        left: &TypedValue,
        right: &TypedValue,
        op: &str,
    ) -> Result<TypedValue> {
        match op {
            "+" => Self::add(left, right),
            "-" => Self::subtract(left, right),
            "*" => Self::multiply(left, right),
            "/" => Self::divide(left, right),
            "%" => Self::modulo(left, right),
            "==" => Ok(TypedValue::Bool(Self::equals(left, right))),
            "!=" => Ok(TypedValue::Bool(!Self::equals(left, right))),
            "<" => Ok(TypedValue::Bool(Self::less_than(left, right)?)),
            ">" => Ok(TypedValue::Bool(Self::greater_than(left, right)?)),
            "<=" => Ok(TypedValue::Bool(!Self::greater_than(left, right)?)),
            ">=" => Ok(TypedValue::Bool(!Self::less_than(left, right)?)),
            "&&" => Self::logical_and(left, right),
            "||" => Self::logical_or(left, right),
            "&" => Self::bitwise_and(left, right),
            "|" => Self::bitwise_or(left, right),
            "^" => Self::bitwise_xor(left, right),
            _ => Err(PhirError::internal(format!(
                "Unsupported binary operator: {op}"
            ))),
        }
    }

    /// Add two values
    fn add(left: &TypedValue, right: &TypedValue) -> Result<TypedValue> {
        match (left, right) {
            (TypedValue::I32(a), TypedValue::I32(b)) => Ok(TypedValue::I32(a + b)),
            (TypedValue::I64(a), TypedValue::I64(b)) => Ok(TypedValue::I64(a + b)),
            (TypedValue::U32(a), TypedValue::U32(b)) => Ok(TypedValue::U32(a + b)),
            (TypedValue::U64(a), TypedValue::U64(b)) => Ok(TypedValue::U64(a + b)),
            _ => Err(PhirError::internal("Type mismatch in addition")),
        }
    }

    /// Subtract two values
    fn subtract(left: &TypedValue, right: &TypedValue) -> Result<TypedValue> {
        match (left, right) {
            (TypedValue::I32(a), TypedValue::I32(b)) => Ok(TypedValue::I32(a - b)),
            (TypedValue::I64(a), TypedValue::I64(b)) => Ok(TypedValue::I64(a - b)),
            (TypedValue::U32(a), TypedValue::U32(b)) => Ok(TypedValue::U32(a - b)),
            (TypedValue::U64(a), TypedValue::U64(b)) => Ok(TypedValue::U64(a - b)),
            _ => Err(PhirError::internal("Type mismatch in subtraction")),
        }
    }

    /// Multiply two values
    fn multiply(left: &TypedValue, right: &TypedValue) -> Result<TypedValue> {
        match (left, right) {
            (TypedValue::I32(a), TypedValue::I32(b)) => Ok(TypedValue::I32(a * b)),
            (TypedValue::I64(a), TypedValue::I64(b)) => Ok(TypedValue::I64(a * b)),
            (TypedValue::U32(a), TypedValue::U32(b)) => Ok(TypedValue::U32(a * b)),
            (TypedValue::U64(a), TypedValue::U64(b)) => Ok(TypedValue::U64(a * b)),
            _ => Err(PhirError::internal("Type mismatch in multiplication")),
        }
    }

    /// Divide two values
    fn divide(left: &TypedValue, right: &TypedValue) -> Result<TypedValue> {
        match (left, right) {
            (TypedValue::I32(a), TypedValue::I32(b)) => {
                if *b == 0 {
                    Err(PhirError::internal("Division by zero"))
                } else {
                    Ok(TypedValue::I32(a / b))
                }
            }
            (TypedValue::I64(a), TypedValue::I64(b)) => {
                if *b == 0 {
                    Err(PhirError::internal("Division by zero"))
                } else {
                    Ok(TypedValue::I64(a / b))
                }
            }
            (TypedValue::U32(a), TypedValue::U32(b)) => {
                if *b == 0 {
                    Err(PhirError::internal("Division by zero"))
                } else {
                    Ok(TypedValue::U32(a / b))
                }
            }
            (TypedValue::U64(a), TypedValue::U64(b)) => {
                if *b == 0 {
                    Err(PhirError::internal("Division by zero"))
                } else {
                    Ok(TypedValue::U64(a / b))
                }
            }
            _ => Err(PhirError::internal("Type mismatch in division")),
        }
    }

    /// Modulo operation
    fn modulo(left: &TypedValue, right: &TypedValue) -> Result<TypedValue> {
        match (left, right) {
            (TypedValue::I32(a), TypedValue::I32(b)) => {
                if *b == 0 {
                    Err(PhirError::internal("Modulo by zero"))
                } else {
                    Ok(TypedValue::I32(a % b))
                }
            }
            (TypedValue::I64(a), TypedValue::I64(b)) => {
                if *b == 0 {
                    Err(PhirError::internal("Modulo by zero"))
                } else {
                    Ok(TypedValue::I64(a % b))
                }
            }
            (TypedValue::U32(a), TypedValue::U32(b)) => {
                if *b == 0 {
                    Err(PhirError::internal("Modulo by zero"))
                } else {
                    Ok(TypedValue::U32(a % b))
                }
            }
            (TypedValue::U64(a), TypedValue::U64(b)) => {
                if *b == 0 {
                    Err(PhirError::internal("Modulo by zero"))
                } else {
                    Ok(TypedValue::U64(a % b))
                }
            }
            _ => Err(PhirError::internal("Type mismatch in modulo")),
        }
    }

    /// Check equality
    fn equals(left: &TypedValue, right: &TypedValue) -> bool {
        left == right
    }

    /// Check less than
    fn less_than(left: &TypedValue, right: &TypedValue) -> Result<bool> {
        match (left, right) {
            (TypedValue::I32(a), TypedValue::I32(b)) => Ok(a < b),
            (TypedValue::I64(a), TypedValue::I64(b)) => Ok(a < b),
            (TypedValue::U32(a), TypedValue::U32(b)) => Ok(a < b),
            (TypedValue::U64(a), TypedValue::U64(b)) => Ok(a < b),
            _ => Err(PhirError::internal("Type mismatch in comparison")),
        }
    }

    /// Check greater than
    fn greater_than(left: &TypedValue, right: &TypedValue) -> Result<bool> {
        match (left, right) {
            (TypedValue::I32(a), TypedValue::I32(b)) => Ok(a > b),
            (TypedValue::I64(a), TypedValue::I64(b)) => Ok(a > b),
            (TypedValue::U32(a), TypedValue::U32(b)) => Ok(a > b),
            (TypedValue::U64(a), TypedValue::U64(b)) => Ok(a > b),
            _ => Err(PhirError::internal("Type mismatch in comparison")),
        }
    }

    /// Logical AND
    fn logical_and(left: &TypedValue, right: &TypedValue) -> Result<TypedValue> {
        match (left, right) {
            (TypedValue::Bool(a), TypedValue::Bool(b)) => Ok(TypedValue::Bool(*a && *b)),
            _ => Err(PhirError::internal("Logical AND requires boolean operands")),
        }
    }

    /// Logical OR
    fn logical_or(left: &TypedValue, right: &TypedValue) -> Result<TypedValue> {
        match (left, right) {
            (TypedValue::Bool(a), TypedValue::Bool(b)) => Ok(TypedValue::Bool(*a || *b)),
            _ => Err(PhirError::internal("Logical OR requires boolean operands")),
        }
    }

    /// Bitwise AND
    fn bitwise_and(left: &TypedValue, right: &TypedValue) -> Result<TypedValue> {
        match (left, right) {
            (TypedValue::U32(a), TypedValue::U32(b)) => Ok(TypedValue::U32(a & b)),
            (TypedValue::U64(a), TypedValue::U64(b)) => Ok(TypedValue::U64(a & b)),
            (TypedValue::I32(a), TypedValue::I32(b)) => Ok(TypedValue::I32(a & b)),
            (TypedValue::I64(a), TypedValue::I64(b)) => Ok(TypedValue::I64(a & b)),
            _ => Err(PhirError::internal("Type mismatch in bitwise AND")),
        }
    }

    /// Bitwise OR
    fn bitwise_or(left: &TypedValue, right: &TypedValue) -> Result<TypedValue> {
        match (left, right) {
            (TypedValue::U32(a), TypedValue::U32(b)) => Ok(TypedValue::U32(a | b)),
            (TypedValue::U64(a), TypedValue::U64(b)) => Ok(TypedValue::U64(a | b)),
            (TypedValue::I32(a), TypedValue::I32(b)) => Ok(TypedValue::I32(a | b)),
            (TypedValue::I64(a), TypedValue::I64(b)) => Ok(TypedValue::I64(a | b)),
            _ => Err(PhirError::internal("Type mismatch in bitwise OR")),
        }
    }

    /// Bitwise XOR
    fn bitwise_xor(left: &TypedValue, right: &TypedValue) -> Result<TypedValue> {
        match (left, right) {
            (TypedValue::U32(a), TypedValue::U32(b)) => Ok(TypedValue::U32(a ^ b)),
            (TypedValue::U64(a), TypedValue::U64(b)) => Ok(TypedValue::U64(a ^ b)),
            (TypedValue::I32(a), TypedValue::I32(b)) => Ok(TypedValue::I32(a ^ b)),
            (TypedValue::I64(a), TypedValue::I64(b)) => Ok(TypedValue::I64(a ^ b)),
            _ => Err(PhirError::internal("Type mismatch in bitwise XOR")),
        }
    }
}
