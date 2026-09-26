//! Expression-shape classification for signed extension.

use crate::ast::Expression;

/// Check if the outermost expression is a negation.
///
/// Register variables and bit references are unsigned at every width, regardless
/// of their bit pattern. Integer literals are non-negative; only a direct unary
/// minus selects signed extension; its evaluated value may still be zero or positive.
#[must_use]
pub fn is_negative_expression(expr: &Expression) -> bool {
    matches!(expr, Expression::UnaryOp { op, .. } if op == "-")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::expressions::parse_integer_to_bitvec;

    #[test]
    fn test_is_negative_expression_literals() {
        // Integer literals are never negative
        let expr = Expression::Integer(parse_integer_to_bitvec("5").unwrap());
        assert!(!is_negative_expression(&expr));

        // Direct negation is always negative
        let expr = Expression::UnaryOp {
            op: "-".to_string(),
            expr: Box::new(Expression::Integer(parse_integer_to_bitvec("5").unwrap())),
        };
        assert!(is_negative_expression(&expr));
    }

    #[test]
    fn test_signedness_uses_only_outer_expression() {
        assert!(!is_negative_expression(&Expression::Variable(
            "d".to_string()
        )));
        assert!(!is_negative_expression(&Expression::BitId(
            "d".to_string(),
            0
        )));
        let compound = Expression::BinaryOp {
            op: "+".to_string(),
            left: Box::new(Expression::UnaryOp {
                op: "-".to_string(),
                expr: Box::new(Expression::Integer(parse_integer_to_bitvec("1").unwrap())),
            }),
            right: Box::new(Expression::Integer(parse_integer_to_bitvec("0").unwrap())),
        };
        assert!(!is_negative_expression(&compound));
    }
}
