use pecos_core::errors::PecosError;
use crate::v0_1::ast::{ArgItem, Expression};
use crate::v0_1::environment::Environment;

/// Handles expression evaluation for PHIR programs
pub struct ExpressionEvaluator<'a> {
    /// Environment containing variable values
    environment: &'a Environment,
}

impl<'a> ExpressionEvaluator<'a> {
    /// Creates a new expression evaluator with the given environment
    pub fn new(environment: &'a Environment) -> Self {
        Self { environment }
    }

    /// Evaluates an expression to a u64 value
    pub fn eval_expr(&self, expr: &Expression) -> Result<u64, PecosError> {
        match expr {
            Expression::Integer(val) => Ok(*val as u64),
            Expression::Variable(name) => self.environment.get(name)
                .ok_or_else(|| PecosError::Input(format!(
                    "Variable '{}' not found", name
                ))),
            Expression::Operation { cop, args } => self.eval_operation(cop, args),
        }
    }

    /// Evaluates an argument item (which can be an expression, variable, bit reference, etc.)
    pub fn eval_arg(&self, arg: &ArgItem) -> Result<u64, PecosError> {
        match arg {
            ArgItem::Integer(val) => Ok(*val as u64),
            ArgItem::Simple(name) => self.environment.get(name)
                .ok_or_else(|| PecosError::Input(format!(
                    "Variable '{}' not found", name
                ))),
            ArgItem::Indexed((name, idx)) => self.environment.get_bit(name, *idx),
            ArgItem::Expression(expr) => self.eval_expr(expr),
        }
    }

    /// Evaluates an operation with an operator and arguments
    fn eval_operation(&self, op: &str, args: &[ArgItem]) -> Result<u64, PecosError> {
        // Handle unary operations
        if args.len() == 1 {
            return self.eval_unary_op(op, &args[0]);
        }
        
        // Handle binary operations
        if args.len() == 2 {
            return self.eval_binary_op(op, &args[0], &args[1]);
        }
        
        Err(PecosError::Input(format!(
            "Unsupported operation: {} with {} arguments", op, args.len()
        )))
    }

    /// Evaluates a unary operation
    fn eval_unary_op(&self, op: &str, arg: &ArgItem) -> Result<u64, PecosError> {
        let value = self.eval_arg(arg)?;
        
        match op {
            "~" => Ok(!value),
            "-" => Ok(value.wrapping_neg()),
            "!" => Ok(if value == 0 { 1 } else { 0 }),
            _ => Err(PecosError::Input(format!(
                "Unsupported unary operator: {}", op
            ))),
        }
    }

    /// Evaluates a binary operation
    fn eval_binary_op(&self, op: &str, lhs: &ArgItem, rhs: &ArgItem) -> Result<u64, PecosError> {
        let lhs_val = self.eval_arg(lhs)?;
        let rhs_val = self.eval_arg(rhs)?;
        
        match op {
            // Arithmetic operations
            "+" => Ok(lhs_val.wrapping_add(rhs_val)),
            "-" => Ok(lhs_val.wrapping_sub(rhs_val)),
            "*" => Ok(lhs_val.wrapping_mul(rhs_val)),
            "/" => {
                if rhs_val == 0 {
                    return Err(PecosError::RuntimeDivisionByZero);
                }
                Ok(lhs_val.wrapping_div(rhs_val))
            },
            "%" => {
                if rhs_val == 0 {
                    return Err(PecosError::RuntimeDivisionByZero);
                }
                Ok(lhs_val.wrapping_rem(rhs_val))
            },
            
            // Bitwise operations
            "&" => Ok(lhs_val & rhs_val),
            "|" => Ok(lhs_val | rhs_val),
            "^" => Ok(lhs_val ^ rhs_val),
            "<<" => Ok(lhs_val.wrapping_shl(rhs_val as u32)),
            ">>" => Ok(lhs_val.wrapping_shr(rhs_val as u32)),
            
            // Comparison operations (return 1 for true, 0 for false)
            "==" => Ok(if lhs_val == rhs_val { 1 } else { 0 }),
            "!=" => Ok(if lhs_val != rhs_val { 1 } else { 0 }),
            "<" => Ok(if lhs_val < rhs_val { 1 } else { 0 }),
            "<=" => Ok(if lhs_val <= rhs_val { 1 } else { 0 }),
            ">" => Ok(if lhs_val > rhs_val { 1 } else { 0 }),
            ">=" => Ok(if lhs_val >= rhs_val { 1 } else { 0 }),
            
            // Logical operations
            "&&" => Ok(if lhs_val != 0 && rhs_val != 0 { 1 } else { 0 }),
            "||" => Ok(if lhs_val != 0 || rhs_val != 0 { 1 } else { 0 }),
            
            _ => Err(PecosError::Input(format!(
                "Unsupported binary operator: {}", op
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v0_1::environment::DataType;

    #[test]
    fn test_eval_simple_expressions() {
        let mut env = Environment::new();
        env.add_variable("x", DataType::I32, 32).unwrap();
        env.add_variable("y", DataType::I32, 32).unwrap();
        
        env.set("x", 10).unwrap();
        env.set("y", 20).unwrap();
        
        let evaluator = ExpressionEvaluator::new(&env);
        
        // Test integer literal
        let expr_int = Expression::Integer(42);
        assert_eq!(evaluator.eval_expr(&expr_int).unwrap(), 42);
        
        // Test variable reference
        let expr_var = Expression::Variable("x".to_string());
        assert_eq!(evaluator.eval_expr(&expr_var).unwrap(), 10);
        
        // Test simple addition
        let expr_add = Expression::Operation {
            cop: "+".to_string(),
            args: vec![
                ArgItem::Simple("x".to_string()),
                ArgItem::Simple("y".to_string()),
            ],
        };
        assert_eq!(evaluator.eval_expr(&expr_add).unwrap(), 30);
    }

    #[test]
    fn test_eval_complex_expressions() {
        let mut env = Environment::new();
        env.add_variable("a", DataType::I32, 32).unwrap();
        env.add_variable("b", DataType::I32, 32).unwrap();
        env.add_variable("c", DataType::I32, 32).unwrap();
        
        env.set("a", 5).unwrap();
        env.set("b", 3).unwrap();
        env.set("c", 2).unwrap();
        
        let evaluator = ExpressionEvaluator::new(&env);
        
        // Test nested expression: (a + b) * c
        let expr_nested = Expression::Operation {
            cop: "*".to_string(),
            args: vec![
                ArgItem::Expression(Box::new(Expression::Operation {
                    cop: "+".to_string(),
                    args: vec![
                        ArgItem::Simple("a".to_string()),
                        ArgItem::Simple("b".to_string()),
                    ],
                })),
                ArgItem::Simple("c".to_string()),
            ],
        };
        assert_eq!(evaluator.eval_expr(&expr_nested).unwrap(), 16);
        
        // Test bitwise operations
        let expr_bitwise = Expression::Operation {
            cop: "&".to_string(),
            args: vec![
                ArgItem::Simple("a".to_string()), // 5 (0b101)
                ArgItem::Simple("b".to_string()), // 3 (0b011)
            ],
        };
        assert_eq!(evaluator.eval_expr(&expr_bitwise).unwrap(), 1); // 0b001 = 1
    }

    #[test]
    fn test_comparison_operators() {
        let mut env = Environment::new();
        env.add_variable("x", DataType::I32, 32).unwrap();
        env.add_variable("y", DataType::I32, 32).unwrap();
        
        env.set("x", 10).unwrap();
        env.set("y", 20).unwrap();
        
        let evaluator = ExpressionEvaluator::new(&env);
        
        // Test x < y (should be true = 1)
        let expr_lt = Expression::Operation {
            cop: "<".to_string(),
            args: vec![
                ArgItem::Simple("x".to_string()),
                ArgItem::Simple("y".to_string()),
            ],
        };
        assert_eq!(evaluator.eval_expr(&expr_lt).unwrap(), 1);
        
        // Test x == y (should be false = 0)
        let expr_eq = Expression::Operation {
            cop: "==".to_string(),
            args: vec![
                ArgItem::Simple("x".to_string()),
                ArgItem::Simple("y".to_string()),
            ],
        };
        assert_eq!(evaluator.eval_expr(&expr_eq).unwrap(), 0);
    }

    #[test]
    fn test_bit_access() {
        let mut env = Environment::new();
        env.add_variable("bits", DataType::U8, 8).unwrap();
        env.set("bits", 0b10101010).unwrap();
        
        let evaluator = ExpressionEvaluator::new(&env);
        
        // Test accessing individual bits
        let arg_bit0 = ArgItem::Indexed(("bits".to_string(), 0));
        let arg_bit1 = ArgItem::Indexed(("bits".to_string(), 1));
        
        assert_eq!(evaluator.eval_arg(&arg_bit0).unwrap(), 0);
        assert_eq!(evaluator.eval_arg(&arg_bit1).unwrap(), 1);
    }
}