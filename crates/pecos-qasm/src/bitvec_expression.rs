// BitVec-based expression evaluation for arbitrary-precision arithmetic

use crate::ast::Expression;
use crate::parser::comparison::{
    ComparisonContext, ComparisonResult, analyze_comparison, is_comparison_op,
};
use ::bitvec::prelude::*;
use pecos_core::bitvec::comparison::compare_unsigned;
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
#[allow(clippy::too_many_lines)]
fn evaluate_binary_op(
    op: &str,
    left: &Expression,
    right: &Expression,
    context: &dyn BitVecExpressionContext,
    default_width: usize,
) -> Result<ExpressionValue, PecosError> {
    // For comparison operations, check if we can resolve immediately based on signs
    if is_comparison_op(op) {
        let comparison_context = ComparisonContext {
            left_expr: left,
            right_expr: right,
            register_context: Some(context),
        };

        match analyze_comparison(op, &comparison_context) {
            ComparisonResult::Immediate(result) => {
                return Ok(ExpressionValue::Bool(result));
            }
            ComparisonResult::RequiresEvaluation => {
                // Fall through to normal evaluation
            }
        }
    }

    let left_val = evaluate_expression_bitvec(left, context, default_width)?;
    let right_val = evaluate_expression_bitvec(right, context, default_width)?;

    match op {
        // Arithmetic operations
        "+" => {
            let (left_bv, right_bv) = to_same_width_bitvecs(left_val, right_val, default_width);
            Ok(ExpressionValue::BitVec(bitvec::add(&left_bv, &right_bv)))
        }
        "-" => {
            let (left_bv, right_bv) = to_same_width_bitvecs(left_val, right_val, default_width);
            Ok(ExpressionValue::BitVec(bitvec::subtract(
                &left_bv, &right_bv,
            )))
        }
        "*" => {
            let (left_bv, right_bv) = to_same_width_bitvecs(left_val, right_val, default_width);
            Ok(ExpressionValue::BitVec(bitvec::multiply(
                &left_bv, &right_bv,
            )))
        }
        "/" => {
            let (left_bv, right_bv) = to_same_width_bitvecs(left_val, right_val, default_width);
            Ok(ExpressionValue::BitVec(bitvec::divide(&left_bv, &right_bv)))
        }

        // Bitwise operations
        "&" => {
            let (left_bv, right_bv) = to_same_width_bitvecs(left_val, right_val, default_width);
            let mut result = left_bv.clone();
            result &= &right_bv;
            Ok(ExpressionValue::BitVec(result))
        }
        "|" => {
            let (left_bv, right_bv) = to_same_width_bitvecs(left_val, right_val, default_width);
            let mut result = left_bv.clone();
            result |= &right_bv;
            Ok(ExpressionValue::BitVec(result))
        }
        "^" => {
            let (left_bv, right_bv) = to_same_width_bitvecs(left_val, right_val, default_width);
            let mut result = left_bv.clone();
            result ^= &right_bv;
            Ok(ExpressionValue::BitVec(result))
        }

        // Comparison operations
        "==" => {
            let (left_bv, right_bv) = to_same_width_bitvecs(left_val, right_val, default_width);
            Ok(ExpressionValue::Bool(left_bv == right_bv))
        }
        "!=" => {
            let (left_bv, right_bv) = to_same_width_bitvecs(left_val, right_val, default_width);
            Ok(ExpressionValue::Bool(left_bv != right_bv))
        }
        "<" => {
            let (left_bv, right_bv) = to_same_width_bitvecs(left_val, right_val, default_width);
            // Use unsigned comparison for same-sign numbers
            // (cross-sign cases are handled above)
            Ok(ExpressionValue::Bool(
                compare_unsigned(&left_bv, &right_bv) == Ordering::Less,
            ))
        }
        ">" => {
            let (left_bv, right_bv) = to_same_width_bitvecs(left_val, right_val, default_width);
            // Use unsigned comparison for same-sign numbers
            // (cross-sign cases are handled above)
            Ok(ExpressionValue::Bool(
                compare_unsigned(&left_bv, &right_bv) == Ordering::Greater,
            ))
        }
        "<=" => {
            let (left_bv, right_bv) = to_same_width_bitvecs(left_val, right_val, default_width);
            // Use unsigned comparison for same-sign numbers
            // (cross-sign cases are handled above)
            let cmp = compare_unsigned(&left_bv, &right_bv);
            Ok(ExpressionValue::Bool(
                cmp == Ordering::Less || cmp == Ordering::Equal,
            ))
        }
        ">=" => {
            let (left_bv, right_bv) = to_same_width_bitvecs(left_val, right_val, default_width);
            // Use unsigned comparison for same-sign numbers
            // (cross-sign cases are handled above)
            let cmp = compare_unsigned(&left_bv, &right_bv);
            Ok(ExpressionValue::Bool(
                cmp == Ordering::Greater || cmp == Ordering::Equal,
            ))
        }

        // Shift operations
        "<<" => {
            let left_bv = left_val.into_bitvec();
            let shift_i64 = right_val.as_i64();
            // Clamp negative shifts to 0, and large shifts to the bit width
            let shift_amount = if shift_i64 < 0 {
                0
            } else if let Ok(shift_usize) = usize::try_from(shift_i64) {
                shift_usize.min(left_bv.len())
            } else {
                // Shift amount is too large, shift all bits out
                left_bv.len()
            };
            Ok(ExpressionValue::BitVec(bitvec::shift_left(
                &left_bv,
                shift_amount,
            )))
        }
        ">>" => {
            let left_bv = left_val.into_bitvec();
            let shift_i64 = right_val.as_i64();
            // Clamp negative shifts to 0, and large shifts to the bit width
            let shift_amount = if shift_i64 < 0 {
                0
            } else if let Ok(shift_usize) = usize::try_from(shift_i64) {
                shift_usize.min(left_bv.len())
            } else {
                // Shift amount is too large, shift all bits out
                left_bv.len()
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
    default_width: usize,
) -> (BitVec<u8, Lsb0>, BitVec<u8, Lsb0>) {
    let mut left_bv = left.into_bitvec();
    let mut right_bv = right.into_bitvec();

    bitvec::resize_to_same_width(&mut left_bv, &mut right_bv, default_width);

    (left_bv, right_bv)
}
