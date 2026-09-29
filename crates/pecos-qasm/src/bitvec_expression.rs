//! QASM AST adapter for shared classical expression evaluation.

use crate::ast::Expression;
use ::bitvec::prelude::*;
use pecos_core::{
    BitUInt, ExprValue, bitvec,
    errors::PecosError,
    expression::{eval_binary_op, eval_unary_op},
};

/// Register access for classical expression evaluation.
pub trait BitVecExpressionContext {
    /// Get a classical register by name.
    fn get_register(&self, name: &str) -> Option<&BitVec<u8, Lsb0>>;
    /// Get the declared size of a register.
    fn get_register_size(&self, name: &str) -> Option<usize>;
}

/// Read a literal with the shared model's signedness and width.
pub(crate) fn integer_value(bits: &BitVec<u8, Lsb0>) -> Result<ExprValue, PecosError> {
    let value = bitvec::to_bituint(bits)?;
    if let Some(value) = value.to_u64().and_then(|v| i64::try_from(v).ok()) {
        Ok(ExprValue::signed(value))
    } else {
        Ok(ExprValue::Unsigned(value))
    }
}

/// Evaluate classical integer expressions using the shared operator tables.
///
/// Registers are unsigned at their stored width; individual bits are booleans.
/// Small literals are signed 64-bit integers, and larger literals retain their
/// parsed width and are unsigned. The destination never sets arithmetic width.
/// `default_width` is used only for an absent register without a declared size.
/// Function calls require the engine and are handled only at the top level.
///
/// # Errors
/// Returns errors for unsupported expressions, invalid widths, or failed operations.
pub fn evaluate_expression_bitvec(
    expr: &Expression,
    context: &dyn BitVecExpressionContext,
    default_width: usize,
) -> Result<ExprValue, PecosError> {
    match expr {
        Expression::Integer(bits) => integer_value(bits),

        Expression::Float(_) => {
            Err(PecosError::ParseInvalidExpression(
                "Float literals are not allowed in classical register expressions. Use integer literals only.".to_string()
            ))
        }

        Expression::Variable(name) => {
            if let Some(bitvec) = context.get_register(name) {
                Ok(ExprValue::Unsigned(bitvec::to_bituint(bitvec)?))
            } else {
                let width = context.get_register_size(name).unwrap_or(default_width);
                let size = u16::try_from(width).ok().filter(|&size| size != 0).ok_or_else(|| {
                    PecosError::Input(format!("Expression width must be in 1..=65535, got {width}"))
                })?;
                Ok(ExprValue::Unsigned(BitUInt::zero(size)))
            }
        }

        Expression::BitId(reg_name, idx) => {
            let bit_value = context
                .get_register(reg_name)
                .and_then(|bitvec| {
                    bitvec.get(*idx).as_deref().copied()
                })
                .unwrap_or(false);
            Ok(ExprValue::Boolean(bit_value))
        }

        Expression::BinaryOp { op, left, right } => {
            eval_binary_op(
                op,
                evaluate_expression_bitvec(left, context, default_width)?,
                evaluate_expression_bitvec(right, context, default_width)?,
            )
        }

        Expression::UnaryOp { op, expr } => {
            eval_unary_op(op, evaluate_expression_bitvec(expr, context, default_width)?)
        }

        Expression::Pi => {
            Err(PecosError::ParseInvalidExpression(
                "Pi constant is not allowed in classical register expressions. Use integer literals only.".to_string()
            ))
        }

        Expression::FunctionCall { name, args: _ } => {
            // Built-in functions (sin, cos, etc.) return floats and are not allowed
            if crate::BUILTIN_FUNCTIONS.contains(&name.as_str()) {
                Err(PecosError::ParseInvalidExpression(format!(
                    "Built-in function '{name}' returns float and is not allowed in classical register expressions. Use it only in gate parameter expressions."
                )))
            } else {
                // Non-built-in functions (WASM functions) cannot be evaluated here
                // The engine's evaluate_expression_bitvec_with_width will handle them
                Err(PecosError::ParseInvalidExpression(format!(
                    "Function '{name}' cannot be evaluated without engine context"
                )))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::expressions::parse_integer_to_bitvec;

    struct RegisterContext(BitVec<u8, Lsb0>);

    impl BitVecExpressionContext for RegisterContext {
        fn get_register(&self, name: &str) -> Option<&BitVec<u8, Lsb0>> {
            (name == "d").then_some(&self.0)
        }

        fn get_register_size(&self, name: &str) -> Option<usize> {
            self.get_register(name).map(BitVec::len)
        }
    }

    #[test]
    fn unsigned_division_preserves_arbitrary_widths() {
        for width in [4, 64, 65, 128, 129] {
            for dense in [false, true] {
                let mut dividend = BitVec::repeat(dense, width);
                dividend.set(width - 1, true);
                let mut expected = bitvec::shift_right(&dividend, 1);
                expected.resize(width.max(64), false);
                let context = RegisterContext(dividend);
                let expr = binary("/", Expression::Variable("d".to_string()), integer("2"));
                let ExprValue::Unsigned(result) =
                    evaluate_expression_bitvec(&expr, &context, 1).unwrap()
                else {
                    panic!("Register division must be unsigned");
                };
                assert_eq!(usize::from(result.size()), width.max(64));
                assert_eq!(result, bitvec::to_bituint(&expected).unwrap());
            }
        }
    }

    #[test]
    fn unsigned_not_register_ignores_default_width() {
        let context = RegisterContext(bitvec![u8, Lsb0; 1, 1]);
        let expr = Expression::UnaryOp {
            op: "~".to_string(),
            expr: Box::new(Expression::Variable("d".to_string())),
        };
        for destination in [1, 4, 100] {
            let ExprValue::Unsigned(value) =
                evaluate_expression_bitvec(&expr, &context, destination).unwrap()
            else {
                panic!("Unsigned complement");
            };
            assert_eq!(value.size(), 64);
            assert_eq!(value.to_u64(), Some(u64::MAX - 3));
        }
    }

    fn compare_register(op: &str, literal: &str) -> bool {
        let context = RegisterContext(bitvec![u8, Lsb0; 1, 1]);
        let expr = Expression::BinaryOp {
            op: op.to_string(),
            left: Box::new(Expression::Variable("d".to_string())),
            right: Box::new(Expression::Integer(
                parse_integer_to_bitvec(literal).unwrap(),
            )),
        };
        evaluate_expression_bitvec(&expr, &context, 1)
            .unwrap()
            .as_bool()
    }

    #[test]
    fn unsigned_register_eq3_with_width1() {
        assert!(compare_register("==", "3"));
        assert!(!compare_register("==", "2"));
    }

    #[test]
    fn unsigned_register_gt2_with_width1() {
        assert!(compare_register(">", "2"));
        assert!(!compare_register(">", "3"));
    }

    #[test]
    fn unsigned_registers_at_arbitrary_widths() {
        for width in [1, 2, 64, 65, 128, 129] {
            for dense in [false, true] {
                let mut value = BitVec::repeat(dense, width);
                value.set(width - 1, true);
                let context = RegisterContext(value.clone());
                for (op, right, expected) in [
                    ("==", value, true),
                    (">", bitvec![u8, Lsb0; 0], true),
                    ("<", bitvec![u8, Lsb0; 0], false),
                ] {
                    let expr = Expression::BinaryOp {
                        op: op.to_string(),
                        left: Box::new(Expression::Variable("d".to_string())),
                        right: Box::new(Expression::Integer(right)),
                    };
                    assert_eq!(
                        evaluate_expression_bitvec(&expr, &context, 1)
                            .unwrap()
                            .as_bool(),
                        expected,
                        "width={width}, dense={dense}, op={op}",
                    );
                }
            }
        }
    }

    #[test]
    fn negative_comparisons_with_different_widths() {
        let context = RegisterContext(BitVec::new());
        for (op, expected) in [
            ("==", false),
            ("!=", true),
            ("<", false),
            (">", true),
            ("<=", false),
            (">=", true),
        ] {
            let negative = |literal| Expression::UnaryOp {
                op: "-".to_string(),
                expr: Box::new(Expression::Integer(
                    parse_integer_to_bitvec(literal).unwrap(),
                )),
            };
            let expr = Expression::BinaryOp {
                op: op.to_string(),
                left: Box::new(negative("1")),
                right: Box::new(negative("16")),
            };
            assert_eq!(
                evaluate_expression_bitvec(&expr, &context, 1)
                    .unwrap()
                    .as_bool(),
                expected,
                "op={op}",
            );
        }
    }

    fn integer(literal: &str) -> Expression {
        Expression::Integer(parse_integer_to_bitvec(literal).unwrap())
    }

    fn negative(expr: Expression) -> Expression {
        Expression::UnaryOp {
            op: "-".to_string(),
            expr: Box::new(expr),
        }
    }

    fn binary(op: &str, left: Expression, right: Expression) -> Expression {
        Expression::BinaryOp {
            op: op.to_string(),
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    macro_rules! negation_comparison_cases {
        ($($name:ident: ($expr:expr, $expected:expr)),+ $(,)?) => {
            $(
                #[test]
                fn $name() {
                    let context = RegisterContext(BitVec::new());
                    assert_eq!(evaluate_expression_bitvec(&$expr, &context, 1).unwrap().as_bool(), $expected);
                }
            )+
        };
    }

    negation_comparison_cases! {
        negative_zero_lt_zero: (binary("<", negative(integer("0")), integer("0")), false),
        negative_zero_ge_zero: (binary(">=", negative(integer("0")), integer("0")), true),
        negative_zero_eq_zero: (binary("==", negative(integer("0")), integer("0")), true),
        double_negative_one_gt_zero: (binary(">", negative(negative(integer("1"))), integer("0")), true),
        negative_one_lt_eight: (binary("<", negative(integer("1")), integer("8")), true),
        negative_compound_lt_zero: (binary("<", negative(binary("-", integer("1"), integer("3"))), integer("0")), false),
        negative_compound_eq_two: (binary("==", negative(binary("-", integer("1"), integer("3"))), integer("2")), true),
        negative_compound_not_eq_negative_fourteen: (binary("==", negative(binary("-", integer("1"), integer("3"))), negative(integer("14"))), false),
    }

    #[test]
    fn negation_preserves_operand_signedness_and_width() {
        let context = RegisterContext(BitVec::new());
        for (expr, expected) in [
            (binary("-", integer("1"), integer("3")), -2_i64),
            (negative(binary("-", integer("1"), integer("3"))), 2),
            (negative(integer("1")), -1),
            (negative(negative(integer("1"))), 1),
            (negative(integer("14")), -14),
        ] {
            let ExprValue::Signed(value) = evaluate_expression_bitvec(&expr, &context, 1).unwrap()
            else {
                panic!("Expected signed result");
            };
            assert_eq!(value.size(), 64);
            assert_eq!(value.to_u64(), Some(expected.cast_unsigned()));
        }
    }

    #[test]
    fn double_negative_signed_minimum() {
        let context = RegisterContext(BitVec::new());
        let literal = parse_integer_to_bitvec("8").unwrap();
        assert_eq!(literal.len(), 4);
        let expr = negative(negative(Expression::Integer(literal)));
        let ExprValue::Signed(value) = evaluate_expression_bitvec(&expr, &context, 1).unwrap()
        else {
            panic!("Signed negation");
        };
        assert_eq!(value.size(), 64);
        assert_eq!(value.to_u64(), Some(8));
        assert!(
            evaluate_expression_bitvec(&binary("==", expr, integer("8")), &context, 1)
                .unwrap()
                .as_bool()
        );
    }

    #[test]
    fn unsigned_register_shift_count() {
        let context = RegisterContext(bitvec![u8, Lsb0; 1, 1]);
        let expr = binary("<<", integer("1"), Expression::Variable("d".to_string()));
        let ExprValue::Unsigned(value) = evaluate_expression_bitvec(&expr, &context, 1).unwrap()
        else {
            panic!("Mixed shift is unsigned");
        };
        assert_eq!(value.size(), 64);
        assert_eq!(value.to_u64(), Some(8));
    }

    #[test]
    fn unsigned_shift_count_uses_shared_accessor() {
        for (index, left, right) in [(2, 128, 0), (3, 2048, 0), (64, 8, 8), (128, 8, 8)] {
            let mut count = BitVec::repeat(false, 129);
            count.set(index, true);
            let context = RegisterContext(count);
            for (op, expected) in [("<<", left), (">>", right)] {
                let expr = binary(op, integer("8"), Expression::Variable("d".to_string()));
                let result = evaluate_expression_bitvec(&expr, &context, 1).unwrap();
                let (ExprValue::Unsigned(value) | ExprValue::Signed(value)) = result else {
                    panic!("Integer shift");
                };
                assert_eq!(value.size(), 129);
                assert_eq!(value.to_u64(), Some(expected));
            }
        }
    }

    #[test]
    fn shared_shift_boundaries() {
        let context = RegisterContext(BitVec::new());
        for (op, count, expected) in [
            ("<<", "64", 0),
            ("<<", "65536", 1),
            ("<<", "18446744073709551617", 1),
            (">>", "64", 1),
        ] {
            let expr = binary(op, integer("1"), integer(count));
            assert_eq!(
                evaluate_expression_bitvec(&expr, &context, 100)
                    .unwrap()
                    .as_u64(),
                expected
            );
        }
        let expr = binary("<<", integer("1"), integer("9223372036854775808"));
        assert!(
            evaluate_expression_bitvec(&expr, &context, 100)
                .unwrap_err()
                .to_string()
                .contains("Negative shift amount")
        );
    }

    #[test]
    fn shift_counts_follow_values() {
        let context = RegisterContext(BitVec::new());
        for (count, left, right) in [
            (negative(integer("0")), 8, 8),
            (negative(negative(integer("1"))), 16, 4),
        ] {
            for (op, expected) in [("<<", left), (">>", right)] {
                let expr = binary(op, integer("8"), count.clone());
                assert_eq!(
                    evaluate_expression_bitvec(&expr, &context, 1)
                        .unwrap()
                        .as_u64(),
                    expected
                );
            }
        }
        for op in ["<<", ">>"] {
            let expr = binary(op, integer("8"), negative(integer("1")));
            assert!(evaluate_expression_bitvec(&expr, &context, 1).is_err());
        }
    }
    #[test]
    fn runtime_signed_division() {
        let result = evaluate_expression_bitvec(
            &binary("/", negative(integer("7")), integer("2")),
            &RegisterContext(BitVec::new()),
            1,
        )
        .unwrap();
        let ExprValue::Signed(value) = result else {
            panic!("Signed literal division");
        };
        assert_eq!(value.size(), 64);
        assert_eq!(value.to_u64(), Some((-3_i64).cast_unsigned()));
    }

    #[test]
    fn operand_tags_and_missing_registers() {
        struct Missing;
        impl BitVecExpressionContext for Missing {
            fn get_register(&self, _: &str) -> Option<&BitVec<u8, Lsb0>> {
                None
            }
            fn get_register_size(&self, name: &str) -> Option<usize> {
                (name == "declared").then_some(100)
            }
        }
        for (name, width) in [("declared", 100), ("absent", 7)] {
            let ExprValue::Unsigned(bits) =
                evaluate_expression_bitvec(&Expression::Variable(name.to_string()), &Missing, 7)
                    .unwrap()
            else {
                panic!("Unsigned missing register");
            };
            assert_eq!(bits.size(), width);
            assert!(bits.is_zero());
        }
        for width in [0, 65536, usize::MAX] {
            assert!(
                evaluate_expression_bitvec(
                    &Expression::Variable("absent".to_string()),
                    &Missing,
                    width
                )
                .is_err()
            );
        }
        let context = RegisterContext(bitvec![u8, Lsb0; 1]);
        let complement = Expression::UnaryOp {
            op: "~".to_string(),
            expr: Box::new(Expression::BitId("d".to_string(), 0)),
        };
        assert_eq!(
            evaluate_expression_bitvec(&complement, &context, 8).unwrap(),
            ExprValue::Boolean(false)
        );
        for (name, index, expected) in [("absent", 0, false), ("d", 1, false), ("d", 0, true)] {
            assert_eq!(
                evaluate_expression_bitvec(
                    &Expression::BitId(name.to_string(), index),
                    &context,
                    8
                )
                .unwrap(),
                ExprValue::Boolean(expected)
            );
        }
        for (literal, signed, width) in [
            ("9223372036854775807", true, 64),
            ("9223372036854775808", false, 64),
            ("18446744073709551616", false, 65),
        ] {
            let result = integer_value(&parse_integer_to_bitvec(literal).unwrap()).unwrap();
            assert_eq!(matches!(result, ExprValue::Signed(_)), signed);
            let (ExprValue::Signed(bits) | ExprValue::Unsigned(bits)) = result else {
                panic!("Integer");
            };
            assert_eq!(bits.size(), width);
        }
        let result =
            evaluate_expression_bitvec(&binary("==", integer("1"), integer("1")), &context, 100)
                .unwrap();
        let ExprValue::Unsigned(bits) = result else {
            panic!("Comparison must retain unsigned tag");
        };
        assert_eq!(bits.size(), 64);
        assert_eq!(bits, BitUInt::new(64, 1));
    }

    #[test]
    fn unsupported_nodes_and_recursive_calls_still_error() {
        for expr in [
            Expression::Float(1.0),
            Expression::Pi,
            Expression::FunctionCall {
                name: "sin".to_string(),
                args: vec![],
            },
            binary(
                "+",
                integer("1"),
                Expression::FunctionCall {
                    name: "RNGnum".to_string(),
                    args: vec![],
                },
            ),
        ] {
            assert!(matches!(
                evaluate_expression_bitvec(&expr, &RegisterContext(BitVec::new()), 8),
                Err(PecosError::ParseInvalidExpression(_))
            ));
        }
    }
}
