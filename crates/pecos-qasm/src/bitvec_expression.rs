// BitVec-based expression evaluation for arbitrary-precision arithmetic

use crate::ast::Expression;
use crate::parser::comparison::is_negative_expression;
use ::bitvec::prelude::*;
use pecos_core::{bitvec, errors::PecosError};
use std::cmp::Ordering;

/// Result of expression evaluation - can be a `BitVec` or a boolean
#[derive(Debug, Clone)]
pub enum ExpressionValue {
    BitVec(BitVec<u8, Lsb0>),
    Bool(bool),
}

impl ExpressionValue {
    /// Convert to `BitVec`, creating a 1-bit `BitVec` for boolean values
    #[must_use]
    pub fn into_bitvec(self) -> BitVec<u8, Lsb0> {
        match self {
            ExpressionValue::BitVec(bv) => bv,
            ExpressionValue::Bool(b) => {
                let mut bv = BitVec::with_capacity(1);
                bv.push(b);
                bv
            }
        }
    }

    /// Convert to bool, treating any non-zero `BitVec` as true
    #[must_use]
    pub fn into_bool(self) -> bool {
        match self {
            ExpressionValue::Bool(b) => b,
            ExpressionValue::BitVec(bv) => bv.any(),
        }
    }

    /// Get as i64 for compatibility (interprets as signed two's complement)
    #[must_use]
    pub fn as_i64(&self) -> i64 {
        match self {
            ExpressionValue::Bool(b) => i64::from(*b),
            ExpressionValue::BitVec(bv) => bitvec::to_i64(bv),
        }
    }
}

/// Trait for expression evaluation with `BitVec` support
pub trait BitVecExpressionContext {
    /// Get a classical register by name
    fn get_register(&self, name: &str) -> Option<&BitVec<u8, Lsb0>>;

    /// Get the size hint for a register (used for creating result `BitVecs`)
    fn get_register_size(&self, name: &str) -> Option<usize>;
}

/// Evaluate an expression to a `BitVec` value for classical register operations
///
/// This function is used to evaluate expressions in classical register contexts,
/// such as `c = a + b` or `if (c == 5) ...`. It supports:
/// - Integer arithmetic: +, -, *, /
/// - Bitwise operations: &, |, ^, ~, <<, >>
/// - Comparisons: ==, !=, <, >, <=, >=
/// - Integer literals (arbitrary precision via `BitVec`)
/// - Register variables and bit references (reg[idx])
///
/// It does NOT support:
/// - Float literals
/// - Pi constant
/// - Mathematical functions (sin, cos, etc.)
///
/// # Parameters
/// - `expr`: The expression to evaluate
/// - `context`: Provides access to classical register values
/// - `default_width`: The width to use for integer literals (typically the largest register size)
///
/// # Errors
///
/// Returns an error if the expression contains unsupported operations or float values.
pub fn evaluate_expression_bitvec(
    expr: &Expression,
    context: &dyn BitVecExpressionContext,
    default_width: usize,
) -> Result<ExpressionValue, PecosError> {
    match expr {
        Expression::Integer(bitvec) => {
            // Clone the BitVec and resize to default width if needed
            let mut result = bitvec.clone();
            if result.len() < default_width && default_width > 0 {
                // Integer literals are always positive (parsed from decimal strings)
                // Negative numbers remain as UnaryOp nodes and are handled separately
                // So we always zero-extend integer literals
                result.resize(default_width, false);
            }
            Ok(ExpressionValue::BitVec(result))
        }

        Expression::Float(_) => {
            Err(PecosError::ParseInvalidExpression(
                "Float literals are not allowed in classical register expressions. Use integer literals only.".to_string()
            ))
        }

        Expression::Variable(name) => {
            if let Some(bitvec) = context.get_register(name) {
                Ok(ExpressionValue::BitVec(bitvec.clone()))
            } else {
                // Return zero-filled BitVec of appropriate size
                let size = context.get_register_size(name).unwrap_or(default_width);
                Ok(ExpressionValue::BitVec(BitVec::repeat(false, size)))
            }
        }

        Expression::BitId(reg_name, idx) => {
            let bit_value = context
                .get_register(reg_name)
                .and_then(|bitvec| {
                    bitvec.get(*idx).as_deref().copied()
                })
                .unwrap_or(false);
            Ok(ExpressionValue::Bool(bit_value))
        }

        Expression::BinaryOp { op, left, right } => {
            evaluate_binary_op(op, left, right, context, default_width)
        }

        Expression::UnaryOp { op, expr } => {
            evaluate_unary_op(op, expr, context, default_width)
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

/// Evaluate binary operations
fn evaluate_binary_op(
    op: &str,
    left: &Expression,
    right: &Expression,
    context: &dyn BitVecExpressionContext,
    default_width: usize,
) -> Result<ExpressionValue, PecosError> {
    let left_val = evaluate_expression_bitvec(left, context, default_width)?;
    let right_val = evaluate_expression_bitvec(right, context, default_width)?;

    match op {
        // Arithmetic operations
        "+" => {
            let (left_bv, right_bv) = to_same_width_bitvecs(
                left_val,
                right_val,
                is_negative_expression(left),
                is_negative_expression(right),
                default_width,
            );
            Ok(ExpressionValue::BitVec(bitvec::add(&left_bv, &right_bv)))
        }
        "-" => {
            let (left_bv, right_bv) = to_same_width_bitvecs(
                left_val,
                right_val,
                is_negative_expression(left),
                is_negative_expression(right),
                default_width,
            );
            Ok(ExpressionValue::BitVec(bitvec::subtract(
                &left_bv, &right_bv,
            )))
        }
        "*" => {
            let (left_bv, right_bv) = to_same_width_bitvecs(
                left_val,
                right_val,
                is_negative_expression(left),
                is_negative_expression(right),
                default_width,
            );
            Ok(ExpressionValue::BitVec(bitvec::multiply(
                &left_bv, &right_bv,
            )))
        }
        "/" => {
            let (left_bv, right_bv) = to_same_width_bitvecs(
                left_val,
                right_val,
                is_negative_expression(left),
                is_negative_expression(right),
                default_width,
            );
            Ok(ExpressionValue::BitVec(bitvec::divide(&left_bv, &right_bv)))
        }

        // Bitwise operations
        "&" => {
            let (left_bv, right_bv) = to_same_width_bitvecs(
                left_val,
                right_val,
                is_negative_expression(left),
                is_negative_expression(right),
                default_width,
            );
            let mut result = left_bv.clone();
            result &= &right_bv;
            Ok(ExpressionValue::BitVec(result))
        }
        "|" => {
            let (left_bv, right_bv) = to_same_width_bitvecs(
                left_val,
                right_val,
                is_negative_expression(left),
                is_negative_expression(right),
                default_width,
            );
            let mut result = left_bv.clone();
            result |= &right_bv;
            Ok(ExpressionValue::BitVec(result))
        }
        "^" => {
            let (left_bv, right_bv) = to_same_width_bitvecs(
                left_val,
                right_val,
                is_negative_expression(left),
                is_negative_expression(right),
                default_width,
            );
            let mut result = left_bv.clone();
            result ^= &right_bv;
            Ok(ExpressionValue::BitVec(result))
        }

        // Preserve an unsigned operand's top bit by adding a separate sign bit.
        "==" | "!=" | "<" | ">" | "<=" | ">=" => {
            let left_bv = left_val.into_bitvec();
            let right_bv = right_val.into_bitvec();
            let width = left_bv.len().max(right_bv.len()).max(default_width) + 1;
            let left_bv = resize_expression_value(left_bv, is_negative_expression(left), width);
            let right_bv = resize_expression_value(right_bv, is_negative_expression(right), width);
            let result = match op {
                "==" => left_bv == right_bv,
                "!=" => left_bv != right_bv,
                "<" => {
                    debug_assert_eq!(left_bv.len(), right_bv.len());
                    bitvec::compare(&left_bv, &right_bv) == Ordering::Less
                }
                ">" => {
                    debug_assert_eq!(left_bv.len(), right_bv.len());
                    bitvec::compare(&left_bv, &right_bv) == Ordering::Greater
                }
                "<=" => {
                    debug_assert_eq!(left_bv.len(), right_bv.len());
                    bitvec::compare(&left_bv, &right_bv) != Ordering::Greater
                }
                ">=" => {
                    debug_assert_eq!(left_bv.len(), right_bv.len());
                    bitvec::compare(&left_bv, &right_bv) != Ordering::Less
                }
                _ => unreachable!(),
            };
            Ok(ExpressionValue::Bool(result))
        }

        // Shift operations
        "<<" => {
            let left_bv = left_val.into_bitvec();
            let shift_amount = if is_negative_expression(right) {
                0
            } else {
                unsigned_shift_amount(&right_val.into_bitvec(), left_bv.len())
            };
            Ok(ExpressionValue::BitVec(bitvec::shift_left(
                &left_bv,
                shift_amount,
            )))
        }
        ">>" => {
            let left_bv = left_val.into_bitvec();
            let shift_amount = if is_negative_expression(right) {
                0
            } else {
                unsigned_shift_amount(&right_val.into_bitvec(), left_bv.len())
            };
            Ok(ExpressionValue::BitVec(bitvec::shift_right(
                &left_bv,
                shift_amount,
            )))
        }

        _ => Err(PecosError::Processing(format!(
            "Unsupported operation: {op}"
        ))),
    }
}

/// Read an unsigned shift count, saturating at the shifted value's width.
fn unsigned_shift_amount(count: &BitSlice<u8, Lsb0>, width: usize) -> usize {
    let mut amount: usize = 0;
    for bit in count.iter().rev() {
        // Counts beyond the operand width shift every bit out, even if they exceed usize.
        amount = amount.saturating_mul(2).saturating_add(usize::from(*bit));
        if amount >= width {
            return width;
        }
    }
    amount
}

/// Evaluate unary operations
fn evaluate_unary_op(
    op: &str,
    expr: &Expression,
    context: &dyn BitVecExpressionContext,
    default_width: usize,
) -> Result<ExpressionValue, PecosError> {
    let val = evaluate_expression_bitvec(expr, context, default_width)?;

    match op {
        "-" => {
            let bv = val.into_bitvec();
            // Reserve a sign bit without changing the operand's signed or unsigned value.
            let width = bv.len() + 1;
            let bv = resize_expression_value(bv, is_negative_expression(expr), width);
            let result = bitvec::negate(&bv);
            Ok(ExpressionValue::BitVec(result))
        }
        "~" => {
            let mut bv = val.into_bitvec();
            bv = !bv; // Bitwise NOT
            Ok(ExpressionValue::BitVec(bv))
        }
        _ => Err(PecosError::Processing(format!(
            "Unsupported operation: {op}"
        ))),
    }
}

/// Convert two `ExpressionValues` to `BitVecs` of the same width
fn to_same_width_bitvecs(
    left: ExpressionValue,
    right: ExpressionValue,
    left_is_negative: bool,
    right_is_negative: bool,
    default_width: usize,
) -> (BitVec<u8, Lsb0>, BitVec<u8, Lsb0>) {
    let left_bv = left.into_bitvec();
    let right_bv = right.into_bitvec();
    let width = left_bv.len().max(right_bv.len()).max(default_width);
    (
        resize_expression_value(left_bv, left_is_negative, width),
        resize_expression_value(right_bv, right_is_negative, width),
    )
}

/// Resize an expression value, extending its sign only for a negated expression.
/// Register values and other non-negative expressions must retain their unsigned value.
pub(crate) fn resize_expression_value(
    mut value: BitVec<u8, Lsb0>,
    is_negative: bool,
    width: usize,
) -> BitVec<u8, Lsb0> {
    let extension = is_negative && value.last().as_deref().copied().unwrap_or(false);
    value.resize(width, extension);
    value
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
    fn unsigned_and_negative_widening() {
        assert_eq!(
            resize_expression_value(bitvec![u8, Lsb0; 1, 1], false, 5),
            bitvec![u8, Lsb0; 1, 1, 0, 0, 0],
        );
        assert_eq!(
            resize_expression_value(bitvec![u8, Lsb0; 1, 1, 1, 1], true, 5),
            bitvec![u8, Lsb0; 1, 1, 1, 1, 1],
        );
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
            .into_bool()
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
                            .into_bool(),
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
                    .into_bool(),
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
                    assert_eq!(evaluate_expression_bitvec(&$expr, &context, 1).unwrap().into_bool(), $expected);
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
        negative_compound_lt_zero: (binary("<", negative(binary("-", integer("1"), integer("3"))), integer("0")), true),
        negative_compound_ne_two: (binary("==", negative(binary("-", integer("1"), integer("3"))), integer("2")), false),
        negative_compound_eq_negative_fourteen: (binary("==", negative(binary("-", integer("1"), integer("3"))), negative(integer("14"))), true),
    }

    #[test]
    fn negation_preserves_operand_signedness_and_width() {
        // Compound results are unsigned modular values here; value-carried signedness is #869.
        let context = RegisterContext(BitVec::new());
        let evaluate = |expr| {
            evaluate_expression_bitvec(&expr, &context, 1)
                .unwrap()
                .into_bitvec()
        };
        assert_eq!(
            evaluate(binary("-", integer("1"), integer("3"))),
            bitvec![u8, Lsb0; 0, 1, 1, 1]
        );
        assert_eq!(
            evaluate(negative(binary("-", integer("1"), integer("3")))),
            bitvec![u8, Lsb0; 0, 1, 0, 0, 1]
        );
        assert_eq!(evaluate(negative(integer("1"))), bitvec![u8, Lsb0; 1; 5]);
        assert_eq!(
            evaluate(negative(negative(integer("1")))),
            bitvec![u8, Lsb0; 1, 0, 0, 0, 0, 0]
        );
        assert_eq!(evaluate(negative(integer("14"))).len(), 8);
    }

    #[test]
    fn double_negative_signed_minimum() {
        let context = RegisterContext(BitVec::new());
        let literal = parse_integer_to_bitvec("8").unwrap();
        assert_eq!(literal.len(), 4);
        let expr = negative(negative(Expression::Integer(literal)));
        assert_eq!(
            evaluate_expression_bitvec(&expr, &context, 1)
                .unwrap()
                .into_bitvec(),
            bitvec![u8, Lsb0; 0, 0, 0, 1, 0, 0],
        );
        assert!(
            evaluate_expression_bitvec(&binary("==", expr, integer("8")), &context, 1)
                .unwrap()
                .into_bool()
        );
    }

    #[test]
    fn unsigned_register_shift_count() {
        let context = RegisterContext(bitvec![u8, Lsb0; 1, 1]);
        let expr = binary("<<", integer("1"), Expression::Variable("d".to_string()));
        assert_eq!(
            evaluate_expression_bitvec(&expr, &context, 1)
                .unwrap()
                .into_bitvec(),
            bitvec![u8, Lsb0; 0, 0, 0, 1],
        );
    }

    #[test]
    fn unsigned_shift_count_saturates() {
        for index in [2, 3, 64, 128] {
            let mut count = BitVec::repeat(false, 129);
            count.set(index, true);
            let context = RegisterContext(count);
            for op in ["<<", ">>"] {
                let expr = binary(op, integer("8"), Expression::Variable("d".to_string()));
                assert_eq!(
                    evaluate_expression_bitvec(&expr, &context, 1)
                        .unwrap()
                        .into_bitvec(),
                    bitvec![u8, Lsb0; 0; 4],
                    "op={op}, count bit={index}",
                );
            }
        }
    }

    #[test]
    fn unsigned_shift_count_boundaries() {
        for (literal, count) in [
            ("0", 0_usize),
            ("1", 1),
            ("3", 3),
            ("4", 4),
            ("5", 5),
            ("8", 8),
        ] {
            let bits = parse_integer_to_bitvec(literal).unwrap();
            for width in [0, 1, 3, 4, 5] {
                assert_eq!(unsigned_shift_amount(&bits, width), count.min(width));
            }
        }
        assert_eq!(unsigned_shift_amount(&BitVec::new(), 4), 0);
        assert_eq!(
            unsigned_shift_amount(&BitVec::repeat(true, 129), usize::MAX),
            usize::MAX
        );
    }

    #[test]
    fn negative_shape_shift_counts_clamp_to_zero() {
        let context = RegisterContext(BitVec::new());
        for count in [
            negative(integer("0")),
            negative(integer("1")),
            negative(negative(integer("1"))),
        ] {
            for op in ["<<", ">>"] {
                let expr = binary(op, integer("8"), count.clone());
                assert_eq!(
                    evaluate_expression_bitvec(&expr, &context, 1)
                        .unwrap()
                        .into_bitvec(),
                    bitvec![u8, Lsb0; 0, 0, 0, 1],
                );
            }
        }
    }
}
