use pecos_core::errors::PecosError;
use pest::iterators::Pair;

use crate::ast::{Expression, Operation};
use crate::parser::expressions::parse_expr;
use crate::parser::gates::{parse_gate_definition, parse_opaque_def};
use crate::parser::operations::{parse_classical_operation, parse_if_statement, parse_quantum_op};
use crate::parser::registers::parse_register;
use crate::parser::{Program, QASMParser, Rule};

/// Parse a statement in the QASM program
///
/// This follows recursive descent principles:
/// 1. Each statement type has its own parsing function
/// 2. We directly build and return AST nodes
/// 3. Error messages include context about what we're parsing
///
/// # Errors
///
/// Returns an error if the statement is invalid
pub fn parse_statement(pair: Pair<Rule>, program: &mut Program) -> Result<(), PecosError> {
    let statement_str = pair.as_str();
    let inner = pair
        .into_inner()
        .next()
        .ok_or_else(|| QASMParser::error("Empty statement"))?;

    // Match on the statement type and delegate to specific parsers
    match inner.as_rule() {
        Rule::register_decl => {
            parse_register(inner, program).map_err(|e| {
                add_context(
                    e,
                    &format!(
                        "In register declaration: {}",
                        statement_str.lines().next().unwrap_or("")
                    ),
                )
            })?;
        }
        Rule::quantum_op => {
            let op = parse_quantum_operation(inner, program)?;
            if let Some(operation) = op {
                program.operations.push(operation);
            }
        }
        Rule::classical_op => {
            let op = parse_classical_statement(inner, program)?;
            if let Some(operation) = op {
                program.operations.push(operation);
            }
        }
        Rule::if_stmt => {
            let op = parse_conditional(inner, program)?;
            if let Some(operation) = op {
                program.operations.push(operation);
            }
        }
        Rule::gate_def => {
            parse_gate_definition(inner, program)
                .map_err(|e| add_context(e, "In gate definition"))?;
        }
        Rule::include => {
            return Err(PecosError::ParseSyntax {
                language: "QASM".to_string(),
                message: "Include statements should be preprocessed before parsing".to_string(),
            });
        }
        Rule::opaque_def => {
            let op = parse_opaque_declaration(inner)?;
            if let Some(operation) = op {
                program.operations.push(operation);
            }
        }
        Rule::function_call_stmt => {
            let op = parse_function_call_statement(inner)?;
            if let Some(operation) = op {
                program.operations.push(operation);
            }
        }
        _ => {
            // Unknown statement type - provide helpful error
            return Err(QASMParser::error(format!(
                "Unknown statement type: {:?}",
                inner.as_rule()
            )));
        }
    }
    Ok(())
}

/// Wrapper functions with clearer names following recursive descent style
fn parse_quantum_operation(
    pair: Pair<Rule>,
    program: &Program,
) -> Result<Option<Operation>, PecosError> {
    parse_quantum_op(pair, program)
}

fn parse_classical_statement(
    pair: Pair<Rule>,
    program: &Program,
) -> Result<Option<Operation>, PecosError> {
    parse_classical_operation(pair, program)
}

fn parse_conditional(pair: Pair<Rule>, program: &Program) -> Result<Option<Operation>, PecosError> {
    parse_if_statement(pair, program)
}

fn parse_opaque_declaration(pair: Pair<Rule>) -> Result<Option<Operation>, PecosError> {
    parse_opaque_def(pair)
}

/// Parse a standalone function call statement (for void functions)
fn parse_function_call_statement(pair: Pair<Rule>) -> Result<Option<Operation>, PecosError> {
    let mut inner = pair.into_inner();

    // Get function name
    let func_name = inner
        .next()
        .ok_or_else(|| QASMParser::error("Missing function name"))?
        .as_str()
        .to_string();

    // Parse arguments
    let mut args = Vec::new();
    for arg_pair in inner {
        if arg_pair.as_rule() == Rule::expr {
            args.push(parse_expr(arg_pair)?);
        }
    }

    // Create a function call expression and wrap it in a void function call operation
    let func_expr = Expression::FunctionCall {
        name: func_name,
        args,
    };

    // Create a void function call operation (similar to classical assignment but without target)
    Ok(Some(Operation::VoidFunctionCall {
        expression: func_expr,
    }))
}

/// Add context to error messages
fn add_context(error: PecosError, context: &str) -> PecosError {
    match error {
        PecosError::ParseSyntax { language, message } => PecosError::ParseSyntax {
            language,
            message: format!("{context}: {message}"),
        },
        PecosError::CompileInvalidOperation { operation, reason } => {
            PecosError::CompileInvalidOperation {
                operation,
                reason: format!("{context}: {reason}"),
            }
        }
        _ => error,
    }
}
