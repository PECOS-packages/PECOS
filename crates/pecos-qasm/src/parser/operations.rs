use log::debug;
use pecos_core::errors::PecosError;
use pest::iterators::Pair;

use crate::ast::Operation;
use crate::parser::errors::{index_out_of_bounds, register_size_mismatch, unknown_register};
use crate::parser::expressions::{parse_expr, parse_expr_with_width, parse_gate_param_expr};
use crate::parser::registers::parse_indexed_id;
use crate::parser::{Program, QASMParser, Rule};
use pecos_core::prelude::{Gate, GateType, QubitId};

/// Helper to resolve a qubit index to a global ID
fn resolve_qubit_index(reg_name: &str, idx: usize, program: &Program) -> Result<usize, PecosError> {
    let qubit_ids = program
        .quantum_registers
        .get(reg_name)
        .ok_or_else(|| unknown_register("quantum", reg_name))?;

    if idx >= qubit_ids.len() {
        return Err(index_out_of_bounds(reg_name, idx, qubit_ids.len()));
    }

    Ok(qubit_ids[idx])
}

/// Helper to parse qubit references from `any_list`
fn parse_qubit_references(
    any_list: Pair<Rule>,
    program: &Program,
) -> Result<Vec<usize>, PecosError> {
    let mut qubits = Vec::new();

    for item in any_list.into_inner() {
        if item.as_rule() == Rule::any_item {
            let inner = item.into_inner().next().unwrap();
            match inner.as_rule() {
                Rule::identifier => {
                    let reg_name = inner.as_str();
                    let qubit_ids = program
                        .quantum_registers
                        .get(reg_name)
                        .ok_or_else(|| unknown_register("quantum", reg_name))?;
                    qubits.extend(qubit_ids.iter().copied());
                }
                Rule::qubit_id => {
                    let (reg_name, idx) = parse_indexed_id(&inner)?;
                    let qubit_id = resolve_qubit_index(&reg_name, idx, program)?;
                    qubits.push(qubit_id);
                }
                _ => {}
            }
        }
    }

    Ok(qubits)
}

/// Helper to parse qubit operands that tracks register expansion
fn parse_qubit_operands(
    any_list: Pair<Rule>,
    program: &Program,
) -> Result<Vec<(String, Vec<usize>)>, PecosError> {
    let mut operands = Vec::new();

    for item in any_list.into_inner() {
        if item.as_rule() == Rule::any_item {
            let inner = item.into_inner().next().unwrap();
            match inner.as_rule() {
                Rule::identifier => {
                    let reg_name = inner.as_str();
                    let qubit_ids = program
                        .quantum_registers
                        .get(reg_name)
                        .ok_or_else(|| unknown_register("quantum", reg_name))?;
                    operands.push((reg_name.to_string(), qubit_ids.clone()));
                }
                Rule::qubit_id => {
                    let (reg_name, idx) = parse_indexed_id(&inner)?;
                    let qubit_id = resolve_qubit_index(&reg_name, idx, program)?;
                    operands.push((format!("{reg_name}[{idx}]"), vec![qubit_id]));
                }
                _ => {}
            }
        }
    }

    Ok(operands)
}

#[allow(clippy::too_many_lines)]
/// Parse a quantum operation
///
/// # Errors
///
/// Returns an error if the operation is invalid
///
/// # Panics
///
/// Panics if the parser encounters an unexpected structure in the parse tree
pub fn parse_quantum_op(
    pair: Pair<Rule>,
    program: &Program,
) -> Result<Option<Operation>, PecosError> {
    let inner = pair.into_inner().next().unwrap();

    match inner.as_rule() {
        Rule::gate_call => {
            let mut inner_pairs = inner.into_inner();
            let gate_name = inner_pairs.next().unwrap().as_str();

            let mut params = Vec::new();
            let mut register_or_qubits = Vec::new();

            for pair in inner_pairs {
                match pair.as_rule() {
                    Rule::param_values => {
                        for param_expr in pair.into_inner() {
                            if param_expr.as_rule() == Rule::expr {
                                let expr = parse_gate_param_expr(param_expr)?;
                                let value = expr.evaluate(None).map_err(|e| {
                                    PecosError::ParseInvalidExpression(format!(
                                        "Failed to evaluate parameter: {e}"
                                    ))
                                })?;
                                params.push(value);
                            }
                        }
                    }
                    Rule::any_list => {
                        register_or_qubits = parse_qubit_operands(pair, program)?;
                    }
                    _ => {}
                }
            }

            // Now handle the expansion of registers into individual gate operations
            let num_operands = register_or_qubits.len();

            // Check if any of the operands are actually full registers
            let has_register = register_or_qubits
                .iter()
                .any(|(_, qubits)| qubits.len() > 1);

            if !has_register {
                // All operands are individual qubits, no expansion needed
                let mut all_qubits = Vec::new();
                for (_, qubits) in &register_or_qubits {
                    all_qubits.extend(qubits);
                }

                Ok(Some(Operation::Gate {
                    name: gate_name.to_string(),
                    parameters: params,
                    qubits: all_qubits,
                }))
            } else if num_operands == 1 {
                // Single operand that is a register - expand to individual gates
                let (_name, qubits) = &register_or_qubits[0];

                // For phase 2 expansion, create a single gate with multiple qubits
                // PECOS will handle the expansion later
                Ok(Some(Operation::Gate {
                    name: gate_name.to_string(),
                    parameters: params,
                    qubits: qubits.clone(),
                }))
            } else if num_operands == 2 {
                // For two-qubit gates, handle register sizes
                let (_name1, qubits1) = &register_or_qubits[0];
                let (_name2, qubits2) = &register_or_qubits[1];

                // If both are single qubits, no special handling needed
                if qubits1.len() == 1 && qubits2.len() == 1 {
                    Ok(Some(Operation::Gate {
                        name: gate_name.to_string(),
                        parameters: params,
                        qubits: vec![qubits1[0], qubits2[0]],
                    }))
                } else if qubits1.len() == qubits2.len() {
                    // Both are registers of the same size - apply pairwise
                    // For now, we'll create a special marker for this case
                    // that the expansion phase will handle
                    let mut all_qubits = Vec::new();
                    for i in 0..qubits1.len() {
                        all_qubits.push(qubits1[i]);
                        all_qubits.push(qubits2[i]);
                    }

                    Ok(Some(Operation::Gate {
                        name: gate_name.to_string(),
                        parameters: params,
                        qubits: all_qubits,
                    }))
                } else {
                    // Register size mismatch
                    Err(register_size_mismatch(
                        &format!("gate {gate_name}"),
                        &format!(
                            "first operand has {} qubits, second has {}",
                            qubits1.len(),
                            qubits2.len()
                        ),
                    ))
                }
            } else {
                // For gates with more than 2 operands, just collect all qubits
                let mut all_qubits = Vec::new();
                for (_name, qubits) in &register_or_qubits {
                    all_qubits.extend(qubits);
                }

                Ok(Some(Operation::Gate {
                    name: gate_name.to_string(),
                    parameters: params,
                    qubits: all_qubits,
                }))
            }
        }
        Rule::measure => parse_measure(inner, program),
        Rule::reset => parse_reset(inner, program),
        Rule::barrier => parse_barrier(inner, program),
        _ => Ok(None),
    }
}

/// Parse a measurement operation
///
/// # Errors
///
/// Returns an error if the measurement syntax is invalid
pub fn parse_measure(pair: Pair<Rule>, program: &Program) -> Result<Option<Operation>, PecosError> {
    let inner_parts: Vec<_> = pair.into_inner().collect();

    if inner_parts.len() != 2 {
        return Err(QASMParser::error("Invalid measurement syntax"));
    }

    let src = &inner_parts[0];
    let dst = &inner_parts[1];

    match (src.as_rule(), dst.as_rule()) {
        (Rule::qubit_id, Rule::bit_id) => {
            let (q_reg, q_idx) = parse_indexed_id(&src.clone())?;
            let (c_reg, c_idx) = parse_indexed_id(&dst.clone())?;
            let qubit = resolve_qubit_index(&q_reg, q_idx, program)?;

            // Create a Gate with GateType::Measure
            let gate = Gate::new(
                GateType::Measure,
                vec![], // No angles
                vec![], // No parameters
                vec![QubitId(qubit)],
            );

            Ok(Some(Operation::MeasureWithMapping {
                gate,
                c_reg,
                c_index: c_idx,
            }))
        }
        (Rule::identifier, Rule::identifier) => Ok(Some(Operation::RegMeasure {
            q_reg: src.as_str().to_string(),
            c_reg: dst.as_str().to_string(),
        })),
        _ => Err(QASMParser::error("Invalid measurement format")),
    }
}

/// Parse a reset operation
///
/// # Errors
///
/// Returns an error if the reset syntax is invalid
///
/// # Panics
///
/// Panics if the parser encounters an unexpected structure in the parse tree
pub fn parse_reset(pair: Pair<Rule>, program: &Program) -> Result<Option<Operation>, PecosError> {
    let any_item = pair.into_inner().next().unwrap();
    let inner = any_item.into_inner().next().unwrap();

    match inner.as_rule() {
        Rule::qubit_id => {
            // Single qubit - create a single Prep gate
            let (reg_name, idx) = parse_indexed_id(&inner)?;
            let qubit = resolve_qubit_index(&reg_name, idx, program)?;
            let gate = Gate::new(
                GateType::Prep,
                vec![], // No angles
                vec![], // No parameters
                vec![QubitId(qubit)],
            );
            Ok(Some(Operation::NativeGate(gate)))
        }
        Rule::identifier => {
            // Full register - create a Gate operation that will be expanded later
            let reg_name = inner.as_str();
            let qubit_ids = program
                .quantum_registers
                .get(reg_name)
                .ok_or_else(|| unknown_register("quantum", reg_name))?;

            // Return as a Gate operation (not NativeGate) to be expanded during gate expansion phase
            Ok(Some(Operation::Gate {
                name: "reset".to_string(),
                parameters: vec![],
                qubits: qubit_ids.clone(),
            }))
        }
        _ => Err(QASMParser::error("Invalid reset syntax")),
    }
}

/// Parse a barrier operation
///
/// # Errors
///
/// Returns an error if the barrier syntax is invalid
///
/// # Panics
///
/// Panics if the parser encounters an unexpected structure in the parse tree
pub fn parse_barrier(pair: Pair<Rule>, program: &Program) -> Result<Option<Operation>, PecosError> {
    let any_list = pair.into_inner().next().unwrap();
    let qubits = parse_qubit_references(any_list, program)?;
    Ok(Some(Operation::Barrier { qubits }))
}

/// Parse an if statement
///
/// # Errors
///
/// Returns an error if the if statement syntax is invalid
pub fn parse_if_statement(
    pair: Pair<Rule>,
    program: &Program,
) -> Result<Option<Operation>, PecosError> {
    debug!("Parsing if statement: '{}'", pair.as_str());

    let parts: Vec<_> = pair.into_inner().collect();

    if parts.len() < 2 {
        return Err(PecosError::CompileInvalidOperation {
            operation: QASMParser::QASM_OPERATION.to_string(),
            reason: format!(
                "Invalid if statement: expected at least 2 parts, got {}",
                parts.len()
            ),
        });
    }

    let condition_expr_pair = &parts[0];
    let operation_pair = &parts[1];

    let condition = match condition_expr_pair.as_rule() {
        Rule::condition_expr => {
            let expr_pair = condition_expr_pair
                .clone()
                .into_inner()
                .next()
                .ok_or_else(|| PecosError::CompileInvalidOperation {
                    operation: QASMParser::QASM_OPERATION.to_string(),
                    reason: "Empty condition expression".to_string(),
                })?;
            parse_expr(expr_pair)?
        }
        _ => {
            return Err(PecosError::CompileInvalidOperation {
                operation: QASMParser::QASM_OPERATION.to_string(),
                reason: format!(
                    "Invalid rule in if statement, expected condition_expr, got: {:?}",
                    condition_expr_pair.as_rule()
                ),
            });
        }
    };

    let operation = match operation_pair.as_rule() {
        Rule::quantum_op => {
            if let Some(op) = parse_quantum_op(operation_pair.clone(), program)? {
                op
            } else {
                return Err(PecosError::CompileInvalidOperation {
                    operation: QASMParser::QASM_OPERATION.to_string(),
                    reason: "Invalid quantum operation in if statement".to_string(),
                });
            }
        }
        Rule::classical_op => {
            if let Some(op) = parse_classical_operation(operation_pair.clone(), program)? {
                op
            } else {
                return Err(PecosError::CompileInvalidOperation {
                    operation: QASMParser::QASM_OPERATION.to_string(),
                    reason: "Invalid classical operation in if statement".to_string(),
                });
            }
        }
        _ => {
            return Err(PecosError::CompileInvalidOperation {
                operation: QASMParser::QASM_OPERATION.to_string(),
                reason: format!(
                    "Unsupported operation type in if statement: {:?}",
                    operation_pair.as_rule()
                ),
            });
        }
    };

    Ok(Some(Operation::If {
        condition,
        operation: Box::new(operation),
    }))
}

/// Parse a classical operation
///
/// # Errors
///
/// Returns an error if the classical operation syntax is invalid
pub fn parse_classical_operation(
    pair: Pair<Rule>,
    program: &Program,
) -> Result<Option<Operation>, PecosError> {
    let inner_parts: Vec<_> = pair.into_inner().collect();

    if inner_parts.len() >= 2 {
        let target_pair = &inner_parts[0];
        let target: String;
        let is_indexed: bool;
        let index: Option<usize>;

        match target_pair.as_rule() {
            Rule::bit_id => {
                let (reg_name, bit_idx) = parse_indexed_id(target_pair)?;
                target = reg_name;
                is_indexed = true;
                index = Some(bit_idx);
            }
            Rule::identifier => {
                target = target_pair.as_str().to_string();
                is_indexed = false;
                index = None;
            }
            _ => {
                return Err(PecosError::CompileInvalidOperation {
                    operation: QASMParser::QASM_OPERATION.to_string(),
                    reason: format!(
                        "Invalid classical assignment target: {:?}",
                        target_pair.as_rule()
                    ),
                });
            }
        }

        let expr_pair = &inner_parts[1];

        // Get the target register size for width-aware constant folding
        let target_width = program
            .classical_registers
            .get(&target)
            .copied()
            .unwrap_or(0);

        // For width-aware constant folding, we need to determine the maximum width
        // This includes the target register width and any operand widths in the expression
        let default_width = target_width;

        let expression = if default_width > 0 {
            parse_expr_with_width(expr_pair.clone(), default_width)?
        } else {
            parse_expr(expr_pair.clone())?
        };

        return Ok(Some(Operation::ClassicalAssignment {
            target,
            is_indexed,
            index,
            expression,
        }));
    }

    Err(PecosError::CompileInvalidOperation {
        operation: QASMParser::QASM_OPERATION.to_string(),
        reason: "Invalid classical operation".to_string(),
    })
}
