use crate::v0_1::ast::{ArgItem, Expression};
use crate::v0_1::environment::{BitValue, DataType, Environment};
pub use pecos_core::ExprValue;
use pecos_core::errors::PecosError;
use pecos_core::{BitUInt, MIN_EVAL_WIDTH, eval_binary_op, eval_unary_op};

/// Converts a `BitValue` to an `ExprValue`, widening to evaluation width.
///
/// For signed values, sign-extends from the type width (not zero-extends).
/// For example, i8(-1) stored as 0xFF is widened to 0xFFFFFFFFFFFFFFFF.
#[must_use]
fn from_bit_value(value: &BitValue) -> ExprValue {
    if value.is_signed() {
        // Sign-extend via as_i64(), then store at eval width
        let signed_val = value.as_i64();
        ExprValue::signed(signed_val)
    } else {
        let eval_width = MIN_EVAL_WIDTH.max(value.size());
        let raw = value.to_bituint();
        let widened = BitUInt::from_raw_words(eval_width, raw.to_words().into_boxed_slice());
        ExprValue::Unsigned(widened)
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
                        from_bit_value(value)
                    });
                }
                return Err(PecosError::RuntimeUndefinedVariable { name: name.clone() });
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
                    eval_unary_op(cop, self.eval_arg(&args[0])?)
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
                    eval_binary_op(cop, self.eval_arg(&args[0])?, self.eval_arg(&args[1])?)
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
                        from_bit_value(value)
                    })
                } else {
                    Err(PecosError::RuntimeUndefinedVariable { name: name.clone() })
                }
            }
            ArgItem::Indexed((name, idx)) => {
                // Propagate the underlying error so an undefined variable surfaces
                // as `RuntimeUndefinedVariable` (-> KeyError) rather than being
                // flattened into a generic `Input` error.
                let bit = self.environment.get_bit(name, *idx)?;
                Ok(ExprValue::Boolean(bit.0))
            }
            ArgItem::Integer(val) => Ok(ExprValue::signed(*val)),
            ArgItem::UInteger(val) => Ok(ExprValue::unsigned(*val)),
            ArgItem::Expression(expr) => self.eval_expr(expr),
        }
    }

    /// Gets multiple bit values from a variable.
    ///
    /// # Errors
    /// Returns an error if any bit access fails.
    pub fn get_bits(&self, name: &str, indices: &[usize]) -> Result<Vec<bool>, PecosError> {
        let value =
            self.environment
                .get(name)
                .ok_or_else(|| PecosError::RuntimeUndefinedVariable {
                    name: name.to_string(),
                })?;
        let value_u64 = value.as_u64();
        indices
            .iter()
            .map(|&idx| Ok(((value_u64 >> idx) & 1) != 0))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wide_qubits_classical_variable_now_sign_extends_signed_operands() {
        use crate::v0_1::operations::OperationProcessor;

        let mut processor = OperationProcessor::new();
        processor
            .handle_variable_definition("cvar_define", "qubits", "wide", 100)
            .unwrap();
        let mut evaluator = ExpressionEvaluator::new(&processor.environment);
        let result = evaluator
            .eval_expr(&Expression::Variable("wide".into()))
            .unwrap();
        let ExprValue::Unsigned(bits) = result else {
            panic!("expected unsigned");
        };
        assert_eq!(bits.size(), 100);
        assert_eq!(bits.to_words(), [0, 0]);

        let result = evaluator
            .eval_expr(&Expression::Operation {
                cop: "+".into(),
                args: vec![ArgItem::Integer(-1), ArgItem::Simple("wide".into())],
            })
            .unwrap();
        let ExprValue::Unsigned(bits) = result else {
            panic!("expected unsigned");
        };
        assert_eq!(bits.size(), 100);
        assert_eq!(
            bits.to_words(),
            [18_446_744_073_709_551_615, 68_719_476_735]
        );

        processor.environment.set_raw("wide", 1).unwrap();
        let mut evaluator = ExpressionEvaluator::new(&processor.environment);
        let shift = |value| Expression::Operation {
            cop: ">>".into(),
            args: vec![ArgItem::Integer(value), ArgItem::Simple("wide".into())],
        };
        let result = evaluator.eval_expr(&shift(-8)).unwrap();
        let ExprValue::Signed(bits) = result else {
            panic!("expected signed");
        };
        assert_eq!(bits.size(), 100);
        assert_eq!(
            bits.to_words(),
            [18_446_744_073_709_551_612, 68_719_476_735]
        );
        let quotient = Expression::Operation {
            cop: "/".into(),
            args: vec![
                ArgItem::Expression(Box::new(shift(-8))),
                ArgItem::Expression(Box::new(shift(-1))),
            ],
        };
        let result = evaluator.eval_expr(&quotient).unwrap();
        let ExprValue::Signed(bits) = result else {
            panic!("expected signed");
        };
        assert_eq!(bits.size(), 100);
        assert_eq!(bits.to_words(), [4, 0]);

        // The PHIR boundary still uses its lossy accessor for scalar results.
        assert_eq!(processor.evaluate_expression(&shift(-8)).unwrap(), 0);
    }

    fn setup_environment() -> Environment {
        let mut env = Environment::new();
        env.add_variable("x", DataType::I32, 31).unwrap();
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
        env.add_variable("a", DataType::I32, 31).unwrap();
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
