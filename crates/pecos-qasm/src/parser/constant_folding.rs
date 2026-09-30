//! Constant folding optimization for QASM expressions
//!
//! This module provides compile-time evaluation of constant expressions,
//! improving performance and simplifying the AST.

use crate::ast::Expression;
use crate::bitvec_expression::integer_value;
use ::bitvec::prelude::*;
use pecos_core::{ExprValue, bitvec, expression::eval_binary_op};
use std::f64::consts::PI;

/// Fold constants in an expression tree
///
/// This recursively evaluates any sub-expressions that contain only constants,
/// replacing them with their computed values.
#[must_use]
pub fn fold_constants(expr: Expression) -> Expression {
    fold_constants_with_context(expr, false)
}

/// Fold constants in a gate parameter expression
///
/// This is like `fold_constants` but doesn't fold bitwise operations
/// since they're not allowed in gate parameters.
#[must_use]
pub fn fold_constants_gate_param(expr: Expression) -> Expression {
    fold_constants_with_context(expr, true)
}

/// Internal function that handles context-aware constant folding
fn fold_constants_with_context(expr: Expression, is_gate_param: bool) -> Expression {
    match expr {
        // Binary operations
        Expression::BinaryOp { op, left, right } => {
            fold_binary_op_with_context(op, *left, *right, is_gate_param)
        }

        // Unary operations
        Expression::UnaryOp { op, expr } => fold_unary_op_with_context(op, *expr, is_gate_param),

        // Function calls
        Expression::FunctionCall { name, args } => {
            fold_function_call_with_context(name, args, is_gate_param)
        }

        // Leaf nodes remain unchanged
        e @ (Expression::Integer(_)
        | Expression::Float(_)
        | Expression::Pi
        | Expression::Variable(_)
        | Expression::BitId(_, _)) => e,
    }
}

/// Fold binary operations with context awareness
fn fold_binary_op_with_context(
    op: String,
    left: Expression,
    right: Expression,
    is_gate_param: bool,
) -> Expression {
    // First recursively fold the operands
    let left = fold_constants_with_context(left, is_gate_param);
    let right = fold_constants_with_context(right, is_gate_param);

    // Try to evaluate if both operands are constants
    match (&left, &right) {
        // Float arithmetic
        (Expression::Float(l), Expression::Float(r)) => fold_float_binary_op(&op, *l, *r),

        // Integer arithmetic and bitwise operations
        (Expression::Integer(l), Expression::Integer(r)) => {
            fold_integer_binary_op_with_context(&op, l, r, is_gate_param)
        }

        // Mixed float/pi operations
        (Expression::Pi, Expression::Float(r)) => fold_float_binary_op(&op, PI, *r),
        (Expression::Float(l), Expression::Pi) => fold_float_binary_op(&op, *l, PI),
        (Expression::Pi, Expression::Pi) => fold_float_binary_op(&op, PI, PI),

        // Cannot fold - return the operation with folded operands
        _ => Expression::BinaryOp {
            op,
            left: Box::new(left),
            right: Box::new(right),
        },
    }
}

/// Fold binary operations on floats
fn fold_float_binary_op(op: &str, l: f64, r: f64) -> Expression {
    match op {
        "+" => Expression::Float(l + r),
        "-" => Expression::Float(l - r),
        "*" => Expression::Float(l * r),
        "/" => {
            if r == 0.0 {
                // Preserve division by zero as-is for proper error handling
                Expression::BinaryOp {
                    op: "/".to_string(),
                    left: Box::new(Expression::Float(l)),
                    right: Box::new(Expression::Float(r)),
                }
            } else {
                Expression::Float(l / r)
            }
        }
        "**" => Expression::Float(l.powf(r)),
        _ => {
            // Unsupported operation for floats
            Expression::BinaryOp {
                op: op.to_string(),
                left: Box::new(Expression::Float(l)),
                right: Box::new(Expression::Float(r)),
            }
        }
    }
}

/// Fold binary operations on integers with context awareness
fn fold_integer_binary_op_with_context(
    op: &str,
    l: &BitVec<u8, Lsb0>,
    r: &BitVec<u8, Lsb0>,
    is_gate_param: bool,
) -> Expression {
    let expression = Expression::BinaryOp {
        op: op.to_string(),
        left: Box::new(Expression::Integer(l.clone())),
        right: Box::new(Expression::Integer(r.clone())),
    };
    if is_gate_param && matches!(op, "&" | "|" | "^" | "<<" | ">>") {
        return expression;
    }
    let result = integer_value(l)
        .and_then(|left| integer_value(r).and_then(|right| eval_binary_op(op, left, right)));
    result
        .as_ref()
        .ok()
        .and_then(fold_integer_value)
        .unwrap_or(expression)
}

/// Only this subset retains its tag and width when re-read as an AST literal.
fn fold_integer_value(value: &ExprValue) -> Option<Expression> {
    if let ExprValue::Signed(bits) = value
        && bits.size() == 64
        && !bits.get_bit(bits.size() - 1)
    {
        return bitvec::from_expr_value(value, usize::from(bits.size()))
            .ok()
            .map(Expression::Integer);
    }
    None
}

/// Fold unary operations with context awareness
fn fold_unary_op_with_context(op: String, expr: Expression, is_gate_param: bool) -> Expression {
    // First recursively fold the operand
    let expr = fold_constants_with_context(expr, is_gate_param);

    match (&op[..], &expr) {
        // Negation of float
        ("-", Expression::Float(f)) => Expression::Float(-f),
        ("-", Expression::Pi) => Expression::Float(-PI),

        // Cannot fold - return the operation with folded operand
        // Integer unary nodes remain for runtime evaluation with their tag and width.
        _ => Expression::UnaryOp {
            op,
            expr: Box::new(expr),
        },
    }
}

/// Fold function calls with context awareness
fn fold_function_call_with_context(
    name: String,
    args: Vec<Expression>,
    is_gate_param: bool,
) -> Expression {
    // First recursively fold all arguments
    let args: Vec<Expression> = args
        .into_iter()
        .map(|arg| fold_constants_with_context(arg, is_gate_param))
        .collect();

    // Check if all arguments are constants
    let all_float_args: Option<Vec<f64>> = args
        .iter()
        .map(|arg| match arg {
            Expression::Float(f) => Some(*f),
            Expression::Pi => Some(PI),
            _ => None,
        })
        .collect();

    // If all arguments are floats, try to evaluate the function
    if let Some(float_args) = all_float_args {
        match (name.as_str(), float_args.as_slice()) {
            // Single-argument functions
            ("sin", &[x]) => Expression::Float(x.sin()),
            ("cos", &[x]) => Expression::Float(x.cos()),
            ("tan", &[x]) => Expression::Float(x.tan()),
            ("exp", &[x]) => Expression::Float(x.exp()),
            ("ln", &[x]) => {
                if x <= 0.0 {
                    // Preserve invalid ln for error handling
                    Expression::FunctionCall { name, args }
                } else {
                    Expression::Float(x.ln())
                }
            }
            ("sqrt", &[x]) => {
                if x < 0.0 {
                    // Preserve invalid sqrt for error handling
                    Expression::FunctionCall { name, args }
                } else {
                    Expression::Float(x.sqrt())
                }
            }

            // Unknown function or wrong number of arguments
            _ => Expression::FunctionCall { name, args },
        }
    } else {
        // Not all arguments are constants
        Expression::FunctionCall { name, args }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::expressions::parse_integer_to_bitvec;

    #[test]
    fn test_unsigned_division_folds_eight_over_two() {
        let expr = Expression::BinaryOp {
            op: "/".to_string(),
            left: Box::new(Expression::Integer(parse_integer_to_bitvec("8").unwrap())),
            right: Box::new(Expression::Integer(parse_integer_to_bitvec("2").unwrap())),
        };
        // The destination width no longer reaches folding, so there is nothing
        // left to sweep here: the result is 64 bits whatever it is assigned to.
        let Expression::Integer(value) = fold_constants(expr) else {
            panic!("Expected folded integer");
        };
        assert_eq!(value.len(), 64);
        assert_eq!(bitvec::to_decimal_string(&value), "4");
    }

    #[test]
    fn test_unsigned_literals_with_different_widths() {
        let expr = Expression::BinaryOp {
            op: "<".to_string(),
            left: Box::new(Expression::Integer(parse_integer_to_bitvec("8").unwrap())),
            right: Box::new(Expression::Integer(parse_integer_to_bitvec("16").unwrap())),
        };

        assert!(matches!(fold_constants(expr), Expression::BinaryOp { .. }));
    }

    #[test]
    fn test_unsigned_subtraction_with_different_widths() {
        let expr = Expression::BinaryOp {
            op: "-".to_string(),
            left: Box::new(Expression::Integer(parse_integer_to_bitvec("16").unwrap())),
            right: Box::new(Expression::Integer(parse_integer_to_bitvec("8").unwrap())),
        };
        let Expression::Integer(value) = fold_constants(expr) else {
            panic!("Expected folded integer");
        };
        assert_eq!(bitvec::to_decimal_string(&value), "8");
    }

    #[test]
    fn test_negated_literal_comparison_remains_unfolded() {
        for literal in ["0", "1"] {
            let expr = Expression::BinaryOp {
                op: "<".to_string(),
                left: Box::new(Expression::UnaryOp {
                    op: "-".to_string(),
                    expr: Box::new(Expression::Integer(
                        parse_integer_to_bitvec(literal).unwrap(),
                    )),
                }),
                right: Box::new(Expression::Integer(parse_integer_to_bitvec("8").unwrap())),
            };
            let Expression::BinaryOp { op, left, right } = fold_constants(expr) else {
                panic!("Negated literal comparisons must remain unfolded");
            };
            assert_eq!(op, "<");
            assert!(matches!(*left, Expression::UnaryOp { ref op, .. } if op == "-"));
            assert!(matches!(*right, Expression::Integer(_)));
        }
    }

    #[test]
    fn test_integer_not_remains_unfolded() {
        // The complement of a small literal is a negative signed value, which
        // cannot be written back as a literal without changing its tag, so it
        // stays unfolded on both the ordinary and the gate-parameter path.
        for fold in [
            fold_constants as fn(Expression) -> Expression,
            fold_constants_gate_param as fn(Expression) -> Expression,
        ] {
            let expr = Expression::UnaryOp {
                op: "~".to_string(),
                expr: Box::new(Expression::Integer(parse_integer_to_bitvec("1").unwrap())),
            };
            let Expression::UnaryOp { op, expr } = fold(expr) else {
                panic!("Signed negative complement must remain unfolded");
            };
            assert_eq!(op, "~");
            assert!(matches!(*expr, Expression::Integer(_)));
        }
    }

    #[test]
    fn test_float_arithmetic() {
        // Test pi/2
        let expr = Expression::BinaryOp {
            op: "/".to_string(),
            left: Box::new(Expression::Pi),
            right: Box::new(Expression::Float(2.0)),
        };

        match fold_constants(expr) {
            Expression::Float(f) => assert!((f - PI / 2.0).abs() < 1e-10),
            _ => panic!("Expected float result"),
        }

        // Test 2*pi
        let expr = Expression::BinaryOp {
            op: "*".to_string(),
            left: Box::new(Expression::Float(2.0)),
            right: Box::new(Expression::Pi),
        };

        match fold_constants(expr) {
            Expression::Float(f) => assert!((f - 2.0 * PI).abs() < 1e-10),
            _ => panic!("Expected float result"),
        }
    }

    #[test]
    fn test_integer_arithmetic() {
        // Test 5 + 3 = 8
        let expr = Expression::BinaryOp {
            op: "+".to_string(),
            left: Box::new(Expression::Integer(parse_integer_to_bitvec("5").unwrap())),
            right: Box::new(Expression::Integer(parse_integer_to_bitvec("3").unwrap())),
        };

        match fold_constants(expr) {
            Expression::Integer(bv) => {
                assert_eq!(bitvec::to_decimal_string(&bv), "8");
            }
            _ => panic!("Expected integer result"),
        }

        // Test 10 - 7 = 3
        let expr = Expression::BinaryOp {
            op: "-".to_string(),
            left: Box::new(Expression::Integer(parse_integer_to_bitvec("10").unwrap())),
            right: Box::new(Expression::Integer(parse_integer_to_bitvec("7").unwrap())),
        };

        match fold_constants(expr) {
            Expression::Integer(bv) => {
                let result = bitvec::to_decimal_string(&bv);
                // With the fix to bitvec arithmetic, this should now correctly compute 10 - 7 = 3
                assert_eq!(result, "3");
            }
            _ => panic!("Expected integer result"),
        }
    }

    #[test]
    fn test_boolean_operations() {
        for (op, left, right) in [("==", "5", "5"), (">", "3", "5")] {
            let expr = Expression::BinaryOp {
                op: op.to_string(),
                left: Box::new(Expression::Integer(parse_integer_to_bitvec(left).unwrap())),
                right: Box::new(Expression::Integer(parse_integer_to_bitvec(right).unwrap())),
            };
            assert!(matches!(fold_constants(expr), Expression::BinaryOp { .. }));
        }
    }

    #[test]
    fn test_function_folding() {
        // Test sin(pi/2) = 1.0
        let expr = Expression::FunctionCall {
            name: "sin".to_string(),
            args: vec![Expression::BinaryOp {
                op: "/".to_string(),
                left: Box::new(Expression::Pi),
                right: Box::new(Expression::Float(2.0)),
            }],
        };

        match fold_constants(expr) {
            Expression::Float(f) => assert!((f - 1.0).abs() < 1e-10),
            _ => panic!("Expected float result"),
        }
    }

    #[test]
    fn test_nested_folding() {
        // Test (5 + 3) * 2 = 16
        // With proper width handling, the arithmetic should work correctly
        let expr = Expression::BinaryOp {
            op: "*".to_string(),
            left: Box::new(Expression::BinaryOp {
                op: "+".to_string(),
                left: Box::new(Expression::Integer(parse_integer_to_bitvec("5").unwrap())),
                right: Box::new(Expression::Integer(parse_integer_to_bitvec("3").unwrap())),
            }),
            right: Box::new(Expression::Integer(parse_integer_to_bitvec("2").unwrap())),
        };

        match fold_constants(expr) {
            Expression::Integer(bv) => {
                // Shared evaluation uses a minimum of 64 bits.
                assert_eq!(bitvec::to_decimal_string(&bv), "16");
            }
            _ => panic!("Expected integer result"),
        }

        // Test a case that doesn't overflow: (2 + 1) * 2 = 6
        let expr = Expression::BinaryOp {
            op: "*".to_string(),
            left: Box::new(Expression::BinaryOp {
                op: "+".to_string(),
                left: Box::new(Expression::Integer(parse_integer_to_bitvec("2").unwrap())),
                right: Box::new(Expression::Integer(parse_integer_to_bitvec("1").unwrap())),
            }),
            right: Box::new(Expression::Integer(parse_integer_to_bitvec("2").unwrap())),
        };

        match fold_constants(expr) {
            Expression::Integer(bv) => {
                assert_eq!(bitvec::to_decimal_string(&bv), "6");
            }
            _ => panic!("Expected integer result"),
        }
    }
    #[test]
    fn fold_decision_preserves_tag_width_and_sign() {
        use pecos_core::BitUInt;
        assert!(fold_integer_value(&ExprValue::Signed(BitUInt::new(65, 1))).is_none());
        assert!(fold_integer_value(&ExprValue::Signed(BitUInt::new(63, 1))).is_none());
        assert!(fold_integer_value(&ExprValue::unsigned(1)).is_none());
        assert!(fold_integer_value(&ExprValue::signed(-1)).is_none());
        assert!(fold_integer_value(&ExprValue::Boolean(true)).is_none());
        assert!(matches!(
            fold_integer_value(&ExprValue::signed(1)),
            Some(Expression::Integer(_))
        ));
    }

    #[test]
    fn folded_division_is_signed_64_bit_three() {
        let expr = Expression::BinaryOp {
            op: "/".to_string(),
            left: Box::new(Expression::Integer(parse_integer_to_bitvec("7").unwrap())),
            right: Box::new(Expression::Integer(parse_integer_to_bitvec("2").unwrap())),
        };
        let Expression::Integer(bits) = fold_constants(expr) else {
            panic!("7 / 2 must fold");
        };
        let ExprValue::Signed(value) = integer_value(&bits).unwrap() else {
            panic!("Folded literal must re-read as signed");
        };
        assert_eq!(value.size(), 64);
        assert_eq!(value.to_u64(), Some(3));
    }
}
