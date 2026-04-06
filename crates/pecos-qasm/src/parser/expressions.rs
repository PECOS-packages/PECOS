use ::bitvec::prelude::*;
use pecos_core::{bitvec, errors::PecosError};
use pest::iterators::Pair;

use crate::ast::Expression;
use crate::parser::Rule;
use crate::parser::constant_folding::{
    fold_constants, fold_constants_gate_param, fold_constants_with_width,
};

/// Parse an arbitrary-length decimal integer string into a `BitVec`
/// This only handles positive integers - negative signs should be handled as unary operations
///
/// # Errors
///
/// Returns an error if the string contains invalid decimal digits
pub fn parse_integer_to_bitvec(s: &str) -> Result<BitVec<u8, Lsb0>, PecosError> {
    bitvec::parse_decimal_string(s).map_err(PecosError::ParseInvalidNumber)
}

/// `UnitaryRep` precedence table (higher number = higher precedence)
fn get_precedence(op: &str) -> Option<i32> {
    match op {
        "|" => Some(1),
        "^" => Some(2),
        "&" => Some(3),
        "==" | "!=" => Some(4),
        "<" | ">" | "<=" | ">=" => Some(5),
        "<<" | ">>" => Some(6),
        "+" | "-" => Some(7),
        "*" | "/" => Some(8),
        "**" => Some(9),
        _ => None,
    }
}

/// Check if operator is right-associative
fn is_right_associative(op: &str) -> bool {
    op == "**"
}

/// Parse expression with precedence climbing (Pratt parsing)
///
/// This function consumes pairs from the beginning of the vector and returns the parsed expression.
/// The pairs vector is modified to remove consumed elements.
///
/// # Errors
///
/// Returns an error if the expression cannot be parsed
pub fn parse_expr_with_precedence(
    pairs: &mut Vec<Pair<Rule>>,
    min_prec: i32,
) -> Result<Expression, PecosError> {
    // Take the first element from pairs
    if pairs.is_empty() {
        return Err(PecosError::ParseInvalidExpression(
            "Expected expression".to_string(),
        ));
    }

    let left_pair = pairs.remove(0);
    let mut left = parse_unary_expr(left_pair)?;

    // Parse binary operations with precedence climbing
    while !pairs.is_empty() {
        // Peek at the next element
        let pair = &pairs[0];

        // Check if this is an operator
        if pair.as_rule() != Rule::binary_op {
            break;
        }

        let op = pair.as_str();
        let prec = get_precedence(op).unwrap_or(0);

        if prec < min_prec {
            break;
        }

        // Consume the operator
        pairs.remove(0);

        if pairs.is_empty() {
            return Err(PecosError::ParseInvalidExpression(
                "Missing right operand".to_string(),
            ));
        }

        // Parse right side recursively with adjusted precedence
        let next_min_prec = if is_right_associative(op) {
            prec
        } else {
            prec + 1
        };
        let right = parse_expr_with_precedence(pairs, next_min_prec)?;

        left = Expression::BinaryOp {
            op: op.to_string(),
            left: Box::new(left),
            right: Box::new(right),
        };
    }

    Ok(left)
}

/// Parse a unary expression
fn parse_unary_expr(pair: Pair<Rule>) -> Result<Expression, PecosError> {
    if pair.as_rule() != Rule::unary_expr {
        return parse_primary_expr(pair);
    }

    let mut pairs = pair.into_inner();
    let mut ops = Vec::new();

    // Collect operators
    while let Some(pair) = pairs.peek() {
        if pair.as_rule() == Rule::unary_op {
            let op_pair = pairs.next().expect("peek confirmed element exists");
            ops.push(op_pair.as_str().to_string());
        } else {
            break;
        }
    }

    // Get operand
    let operand_pair = pairs.next().ok_or_else(|| {
        PecosError::ParseInvalidExpression("Missing operand for unary operation".to_string())
    })?;
    let mut expr = parse_primary_expr(operand_pair)?;

    // Apply operators in reverse order
    for op in ops.iter().rev() {
        expr = Expression::UnaryOp {
            op: op.clone(),
            expr: Box::new(expr),
        };
    }

    Ok(expr)
}

/// Main expression parser
///
/// # Errors
///
/// Returns an error if the expression cannot be parsed
pub fn parse_expr(pair: Pair<Rule>) -> Result<Expression, PecosError> {
    let expr = parse_expr_internal(pair)?;
    // Apply constant folding optimization
    Ok(fold_constants(expr))
}

/// Main expression parser with width context for better constant folding
///
/// # Errors
///
/// Returns an error if the expression cannot be parsed
pub fn parse_expr_with_width(
    pair: Pair<Rule>,
    default_width: usize,
) -> Result<Expression, PecosError> {
    let expr = parse_expr_internal(pair)?;
    // Apply constant folding optimization with width context
    Ok(fold_constants_with_width(expr, default_width))
}

/// Parse a gate parameter expression
///
/// This is like `parse_expr` but:
/// 1. Uses context-aware constant folding that doesn't fold bitwise operations
/// 2. Treats all numeric literals as floats (since gate parameters are angles)
///
/// # Errors
///
/// Returns an error if the expression cannot be parsed
pub fn parse_gate_param_expr(pair: Pair<Rule>) -> Result<Expression, PecosError> {
    let expr = parse_expr_internal_gate_param(pair)?;
    // Apply constant folding optimization with gate parameter context
    Ok(fold_constants_gate_param(expr))
}

/// Internal expression parser for gate parameters (treats numbers as floats)
fn parse_expr_internal_gate_param(pair: Pair<Rule>) -> Result<Expression, PecosError> {
    match pair.as_rule() {
        Rule::expr => {
            // Convert to vector for precedence climbing
            let mut pairs: Vec<_> = pair.into_inner().collect();
            if pairs.is_empty() {
                return Err(PecosError::ParseInvalidExpression(
                    "Empty expression".to_string(),
                ));
            }
            // If we have a single element that's also an expr, unwrap it
            if pairs.len() == 1 && pairs[0].as_rule() == Rule::expr {
                return parse_expr_internal_gate_param(
                    pairs.into_iter().next().expect("length checked to be 1"),
                );
            }
            parse_expr_with_precedence_gate_param(&mut pairs, 1)
        }

        Rule::unary_expr => parse_unary_expr_gate_param(pair),
        _ => parse_primary_expr_gate_param(pair),
    }
}

/// Internal expression parser (without constant folding)
fn parse_expr_internal(pair: Pair<Rule>) -> Result<Expression, PecosError> {
    match pair.as_rule() {
        Rule::expr => {
            // Convert to vector for precedence climbing
            let mut pairs: Vec<_> = pair.into_inner().collect();
            if pairs.is_empty() {
                return Err(PecosError::ParseInvalidExpression(
                    "Empty expression".to_string(),
                ));
            }
            // If we have a single element that's also an expr, unwrap it
            if pairs.len() == 1 && pairs[0].as_rule() == Rule::expr {
                return parse_expr_internal(
                    pairs.into_iter().next().expect("length checked to be 1"),
                );
            }
            parse_expr_with_precedence(&mut pairs, 1)
        }

        // For backwards compatibility with existing code
        Rule::unary_expr => parse_unary_expr(pair),
        _ => parse_primary_expr(pair),
    }
}

/// Parse primary (atomic) expressions
fn parse_primary_expr(pair: Pair<Rule>) -> Result<Expression, PecosError> {
    match pair.as_rule() {
        Rule::primary_expr => {
            let inner = pair.into_inner().next().ok_or_else(|| {
                PecosError::ParseInvalidExpression("Empty primary expression".to_string())
            })?;
            parse_primary_expr(inner)
        }
        Rule::pi_constant => Ok(Expression::Pi),
        Rule::number => {
            let num_str = pair.as_str();
            if num_str.contains('.') || num_str.contains('e') || num_str.contains('E') {
                Ok(Expression::Float(num_str.parse().map_err(|_| {
                    PecosError::ParseInvalidNumber(num_str.to_string())
                })?))
            } else {
                Ok(Expression::Integer(parse_integer_to_bitvec(num_str)?))
            }
        }
        Rule::int => {
            let int_str = pair.as_str();
            Ok(Expression::Integer(parse_integer_to_bitvec(int_str)?))
        }
        Rule::bit_id => {
            let bit_id = pair.as_str();
            let parts: Vec<&str> = bit_id.split('[').collect();
            let name = parts[0].to_string();
            let idx_str = parts[1].trim_end_matches(']');
            let idx = idx_str
                .parse()
                .map_err(|_| PecosError::ParseInvalidNumber(idx_str.to_string()))?;
            Ok(Expression::BitId(name, idx))
        }
        Rule::identifier => Ok(Expression::Variable(pair.as_str().to_string())),
        Rule::function_call => {
            let mut pairs = pair.into_inner();
            let name = pairs
                .next()
                .ok_or_else(|| {
                    PecosError::ParseInvalidExpression(
                        "Missing function name in function call".to_string(),
                    )
                })?
                .as_str()
                .to_string();
            let args: Result<Vec<_>, _> = pairs.map(parse_expr_internal).collect();
            Ok(Expression::FunctionCall { name, args: args? })
        }
        Rule::expr => {
            // Handle nested expr that might come from parentheses
            parse_expr_internal(pair)
        }
        _ => Err(PecosError::ParseInvalidExpression(format!(
            "Unexpected rule in expression: {:?}",
            pair.as_rule()
        ))),
    }
}

/// Parse expression with precedence climbing for gate parameters
fn parse_expr_with_precedence_gate_param(
    pairs: &mut Vec<Pair<Rule>>,
    min_prec: i32,
) -> Result<Expression, PecosError> {
    // Take the first element from pairs
    let first = pairs
        .drain(0..1)
        .next()
        .ok_or_else(|| PecosError::ParseInvalidExpression("Empty expression".to_string()))?;

    let mut left = parse_unary_expr_gate_param(first)?;

    while let Some(op_pair) = pairs.first() {
        if op_pair.as_rule() != Rule::binary_op {
            break;
        }

        let op = op_pair.as_str();
        let prec = get_precedence(op)
            .ok_or_else(|| PecosError::ParseInvalidExpression(format!("Unknown operator: {op}")))?;

        if prec < min_prec {
            break;
        }

        // Consume the operator
        pairs.drain(0..1);

        // Get the right operand
        if pairs.is_empty() {
            return Err(PecosError::ParseInvalidExpression(format!(
                "Missing operand after {op}"
            )));
        }

        let next_min_prec = if is_right_associative(op) {
            prec
        } else {
            prec + 1
        };
        let right = parse_expr_with_precedence_gate_param(pairs, next_min_prec)?;

        left = Expression::BinaryOp {
            op: op.to_string(),
            left: Box::new(left),
            right: Box::new(right),
        };
    }

    Ok(left)
}

/// Parse unary expressions for gate parameters
fn parse_unary_expr_gate_param(pair: Pair<Rule>) -> Result<Expression, PecosError> {
    if pair.as_rule() != Rule::unary_expr {
        return parse_primary_expr_gate_param(pair);
    }

    let mut pairs = pair.into_inner();
    let mut ops = Vec::new();

    // Collect operators
    while let Some(pair) = pairs.peek() {
        if pair.as_rule() == Rule::unary_op {
            let op_pair = pairs.next().expect("peek confirmed element exists");
            ops.push(op_pair.as_str().to_string());
        } else {
            break;
        }
    }

    // Get operand
    let operand_pair = pairs.next().ok_or_else(|| {
        PecosError::ParseInvalidExpression("Missing operand for unary operation".to_string())
    })?;
    let mut expr = parse_primary_expr_gate_param(operand_pair)?;

    // Apply operators in reverse order
    for op in ops.iter().rev() {
        expr = Expression::UnaryOp {
            op: op.clone(),
            expr: Box::new(expr),
        };
    }

    Ok(expr)
}

/// Parse primary expressions for gate parameters (treats all numbers as floats)
fn parse_primary_expr_gate_param(pair: Pair<Rule>) -> Result<Expression, PecosError> {
    match pair.as_rule() {
        Rule::primary_expr => {
            let inner = pair.into_inner().next().ok_or_else(|| {
                PecosError::ParseInvalidExpression("Empty primary expression".to_string())
            })?;
            parse_primary_expr_gate_param(inner)
        }
        Rule::pi_constant => Ok(Expression::Pi),
        Rule::number => {
            // For gate parameters, always parse numbers as floats
            let num_str = pair.as_str();
            Ok(Expression::Float(num_str.parse().map_err(|_| {
                PecosError::ParseInvalidNumber(num_str.to_string())
            })?))
        }
        Rule::int => {
            // For gate parameters, parse integers as floats
            let int_str = pair.as_str();
            Ok(Expression::Float(int_str.parse().map_err(|_| {
                PecosError::ParseInvalidNumber(int_str.to_string())
            })?))
        }
        Rule::bit_id => {
            // Bit references shouldn't appear in gate parameters
            Err(PecosError::ParseInvalidExpression(
                "Bit references are not allowed in gate parameter expressions".to_string(),
            ))
        }
        Rule::identifier => Ok(Expression::Variable(pair.as_str().to_string())),
        Rule::function_call => {
            let mut pairs = pair.into_inner();
            let name = pairs
                .next()
                .ok_or_else(|| {
                    PecosError::ParseInvalidExpression(
                        "Missing function name in function call".to_string(),
                    )
                })?
                .as_str()
                .to_string();
            let args: Result<Vec<_>, _> = pairs.map(parse_expr_internal_gate_param).collect();
            Ok(Expression::FunctionCall { name, args: args? })
        }
        Rule::expr => {
            // Handle nested expr that might come from parentheses
            parse_expr_internal_gate_param(pair)
        }
        _ => Err(PecosError::ParseInvalidExpression(format!(
            "Unexpected rule in expression: {:?}",
            pair.as_rule()
        ))),
    }
}
