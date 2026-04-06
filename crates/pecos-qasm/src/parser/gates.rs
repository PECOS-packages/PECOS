use pecos_core::errors::PecosError;
use pest::iterators::Pair;
use std::collections::BTreeMap;

use crate::ast::{Expression, GateDefinition, GateOperation, Operation};
use crate::parser::expressions::parse_expr;
use crate::parser::{Program, QASMParser, Rule};

/// Parse a gate definition and add it to the program
///
/// # Errors
///
/// Returns an error if the gate definition is invalid
pub fn parse_gate_definition(pair: Pair<Rule>, program: &mut Program) -> Result<(), PecosError> {
    let mut inner = pair.into_inner();

    let name = inner
        .next()
        .ok_or_else(|| QASMParser::error("Missing gate name in gate definition"))?
        .as_str()
        .to_string();

    let mut params = Vec::new();
    let mut qargs = Vec::new();
    let mut body_pairs = Vec::new();

    for inner_pair in inner {
        match inner_pair.as_rule() {
            Rule::param_list => {
                for param in inner_pair.into_inner() {
                    if param.as_rule() == Rule::identifier {
                        params.push(param.as_str().to_string());
                    }
                }
            }
            Rule::identifier_list => {
                for ident in inner_pair.into_inner() {
                    if ident.as_rule() == Rule::identifier {
                        qargs.push(ident.as_str().to_string());
                    }
                }
            }
            Rule::gate_def_statement => {
                body_pairs.push(inner_pair);
            }
            _ => {}
        }
    }

    let mut body = Vec::new();
    for statement_pair in body_pairs {
        if let Some(op) = parse_gate_def_statement(statement_pair)? {
            body.push(op);
        }
    }

    let gate_def = GateDefinition {
        name: name.clone(),
        params,
        qargs,
        body,
    };

    program.gate_definitions.insert(name, gate_def);

    Ok(())
}

/// Parse an opaque gate definition
///
/// # Errors
///
/// Returns an error if the opaque definition is invalid
pub fn parse_opaque_def(pair: Pair<Rule>) -> Result<Option<Operation>, PecosError> {
    let mut inner = pair.into_inner();

    let name = inner
        .next()
        .ok_or_else(|| PecosError::CompileInvalidOperation {
            operation: QASMParser::QASM_OPERATION.to_string(),
            reason: "Missing gate name".to_string(),
        })?
        .as_str()
        .to_string();

    let mut params = Vec::new();
    let mut qargs = Vec::new();

    for part in inner {
        match part.as_rule() {
            Rule::param_list => {
                for param in part.into_inner() {
                    if param.as_rule() == Rule::identifier {
                        params.push(param.as_str().to_string());
                    }
                }
            }
            Rule::identifier_list => {
                for qarg in part.into_inner() {
                    if qarg.as_rule() == Rule::identifier {
                        qargs.push(qarg.as_str().to_string());
                    }
                }
            }
            _ => {}
        }
    }

    Ok(Some(Operation::OpaqueGate {
        name,
        params,
        qargs,
    }))
}

/// Parse a gate definition statement
///
/// # Errors
///
/// Returns an error if the statement is invalid
pub fn parse_gate_def_statement(pair: Pair<Rule>) -> Result<Option<GateOperation>, PecosError> {
    let inner = pair
        .into_inner()
        .next()
        .ok_or_else(|| QASMParser::error("Empty gate definition statement"))?;

    match inner.as_rule() {
        Rule::gate_def_call => {
            let mut parts = inner.into_inner();
            let gate_name = parts
                .next()
                .ok_or_else(|| QASMParser::error("Missing gate name in gate body call"))?
                .as_str();

            let mut params = Vec::new();
            let mut arguments = Vec::new();

            for part in parts {
                match part.as_rule() {
                    Rule::param_values => {
                        for expr_pair in part.into_inner() {
                            let param_expr = parse_expr(expr_pair)?;
                            params.push(param_expr);
                        }
                    }
                    Rule::identifier_list => {
                        for ident in part.into_inner() {
                            if ident.as_rule() == Rule::identifier {
                                arguments.push(ident.as_str().to_string());
                            }
                        }
                    }
                    _ => {}
                }
            }

            Ok(Some(GateOperation {
                name: gate_name.to_string(),
                params,
                qargs: arguments,
            }))
        }
        _ => Ok(None),
    }
}

/// Evaluate a parameter expression within a gate definition
///
/// # Errors
///
/// Returns an error if the expression cannot be evaluated
pub fn evaluate_param_expr(
    expr: &Expression,
    param_map: &BTreeMap<String, f64>,
) -> Result<f64, PecosError> {
    use crate::ast::EvaluationCtx;
    let context = EvaluationCtx {
        params: Some(param_map),
    };
    expr.evaluate(Some(&context))
}
