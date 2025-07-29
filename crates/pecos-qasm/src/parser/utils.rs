use pecos_core::errors::PecosError;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

use crate::ast::{GateDefinition, Operation};
use crate::parser::errors::{
    invalid_operation, register_size_mismatch, undefined_gate, unknown_register, wrong_param_count,
    wrong_qubit_count,
};
use crate::parser::gates::evaluate_param_expr;
use crate::parser::native_gates::{canonical_gate_name, is_native_operation, parse_native_gate};
use crate::parser::{Program, QASMParser};
use pecos_core::prelude::{Gate, GateType, QubitId};

/// Expand all gate operations in the program to native gates
///
/// # Errors
///
/// Returns an error if gate expansion fails (e.g., circular dependencies)
pub fn expand_gates(program: &mut Program) -> Result<(), PecosError> {
    let mut expanded_operations = Vec::new();

    for operation in &program.operations {
        match operation {
            Operation::Gate {
                name,
                parameters,
                qubits,
            } => {
                let expanded =
                    expand_gate_operation(name, parameters, qubits, &program.gate_definitions)?;
                expanded_operations.extend(expanded);
            }
            Operation::RegMeasure { q_reg, c_reg } => {
                expand_register_measure(
                    q_reg,
                    c_reg,
                    &program.quantum_registers,
                    &program.classical_registers,
                    &mut expanded_operations,
                )?;
            }
            Operation::If {
                condition,
                operation,
            } => {
                // Recursively expand the operation inside the if statement
                let inner_op = operation.as_ref();
                match inner_op {
                    Operation::Gate {
                        name,
                        parameters,
                        qubits,
                    } => {
                        let expanded_inner = expand_gate_operation(
                            name,
                            parameters,
                            qubits,
                            &program.gate_definitions,
                        )?;
                        // Create separate If operations for each expanded gate
                        for expanded_op in expanded_inner {
                            expanded_operations.push(Operation::If {
                                condition: condition.clone(),
                                operation: Box::new(expanded_op),
                            });
                        }
                    }
                    _ => {
                        // For non-gate operations inside If, just clone
                        expanded_operations.push(Operation::If {
                            condition: condition.clone(),
                            operation: operation.clone(),
                        });
                    }
                }
            }
            _ => expanded_operations.push(operation.clone()),
        }
    }

    program.operations = expanded_operations;
    Ok(())
}

fn expand_gate_operation(
    name: &str,
    parameters: &[f64],
    qubits: &[usize],
    gate_definitions: &BTreeMap<String, GateDefinition>,
) -> Result<Vec<Operation>, PecosError> {
    // First check if this is a user-defined gate
    if let Some(gate_def) = gate_definitions.get(name) {
        // User-defined gate
        return expand_gate_call(gate_def, parameters, qubits, gate_definitions);
    }

    // Check if it's a native gate (case insensitive)
    if let Some(gate_type) = parse_native_gate(name) {
        // Only allow exact uppercase native gates unless there's a definition
        let is_uppercase = name == name.to_uppercase();
        if !is_uppercase && !gate_definitions.contains_key(name) {
            // Lowercase native gate without definition - error
            return Err(undefined_gate(name));
        }

        // Validate parameter count
        let expected_params = gate_type.classical_arity();
        if parameters.len() != expected_params {
            return Err(wrong_param_count(name, expected_params, parameters.len()));
        }

        // Use uppercase name for native gates (no longer needed since we create Gate structs)
        let _native_name = canonical_gate_name(name);

        // Handle register expansion for native gates
        match (gate_type.quantum_arity(), qubits.len()) {
            (1, n) if n > 1 => {
                // Single-qubit gate applied to multiple qubits
                Ok(qubits
                    .iter()
                    .map(|&qubit| {
                        let gate = Gate::new(gate_type, parameters.to_vec(), vec![QubitId(qubit)]);
                        Operation::NativeGate(gate)
                    })
                    .collect())
            }
            (2, n) if n > 2 => {
                // Two-qubit gate applied to multiple qubits
                if n % 2 != 0 {
                    return Err(invalid_operation(format!(
                        "Two-qubit gate '{name}' applied to {n} qubits (must be even number)"
                    )));
                }
                Ok((0..n)
                    .step_by(2)
                    .map(|i| {
                        let gate = Gate::new(
                            gate_type,
                            parameters.to_vec(),
                            vec![QubitId(qubits[i]), QubitId(qubits[i + 1])],
                        );
                        Operation::NativeGate(gate)
                    })
                    .collect())
            }
            (expected, actual) if expected != actual => {
                // Wrong number of qubits
                Err(wrong_qubit_count(name, expected, actual))
            }
            _ => {
                // Correct number of qubits, no expansion needed
                let gate = Gate::new(
                    gate_type,
                    parameters.to_vec(),
                    qubits.iter().map(|&q| QubitId(q)).collect(),
                );
                Ok(vec![Operation::NativeGate(gate)])
            }
        }
    } else if is_native_operation(name) {
        // Other native operations (barrier, reset) - these are handled differently from gates
        match name.to_lowercase().as_str() {
            "barrier" => Ok(vec![Operation::Barrier {
                qubits: qubits.to_vec(),
            }]),
            "reset" => {
                // Create reset operations for each qubit
                Ok(qubits
                    .iter()
                    .map(|&qubit| {
                        let gate = Gate::new(GateType::Prep, vec![], vec![QubitId(qubit)]);
                        Operation::NativeGate(gate)
                    })
                    .collect())
            }
            "measure" => {
                // Measurement operations need classical register mapping, so this should
                // not happen in gate expansion - measurements should be parsed directly
                Err(invalid_operation(
                    "Measure operations require classical register mapping and should not appear in gate expansion".to_string()
                ))
            }
            "opaque" => {
                // Opaque operations are declarations, not executable operations
                Err(invalid_operation(
                    "Opaque is a declaration, not an executable operation".to_string(),
                ))
            }
            _ => {
                // Other native operations should already be handled
                Err(invalid_operation(format!(
                    "Native operation '{name}' should have been handled earlier"
                )))
            }
        }
    } else {
        // Unknown gate
        Err(undefined_gate(name))
    }
}

fn expand_register_measure(
    q_reg: &str,
    c_reg: &str,
    quantum_registers: &BTreeMap<String, Vec<usize>>,
    classical_registers: &BTreeMap<String, usize>,
    expanded_operations: &mut Vec<Operation>,
) -> Result<(), PecosError> {
    let q_qubits = quantum_registers
        .get(q_reg)
        .ok_or_else(|| unknown_register("quantum", q_reg))?;

    let c_size = classical_registers
        .get(c_reg)
        .ok_or_else(|| unknown_register("classical", c_reg))?;

    if q_qubits.len() != *c_size {
        return Err(register_size_mismatch(
            &format!("measure {q_reg} -> {c_reg}"),
            &format!(
                "quantum register {} has {} qubits, classical register {} has {} bits",
                q_reg,
                q_qubits.len(),
                c_reg,
                c_size
            ),
        ));
    }

    // Expand to individual measurements
    for (i, &qubit) in q_qubits.iter().enumerate() {
        // Create a Gate with GateType::Measure
        let gate = Gate::new(
            GateType::Measure,
            vec![], // No parameters
            vec![QubitId(qubit)],
        );

        expanded_operations.push(Operation::MeasureWithMapping {
            gate,
            c_reg: c_reg.to_string(),
            c_index: i,
        });
    }

    Ok(())
}

fn expand_gate_call(
    gate_def: &GateDefinition,
    parameters: &[f64],
    qubits: &[usize],
    all_definitions: &BTreeMap<String, GateDefinition>,
) -> Result<Vec<Operation>, PecosError> {
    // Check if this is a single-qubit gate being applied to multiple qubits
    if gate_def.qargs.len() == 1 && qubits.len() > 1 {
        // Expand to multiple gate calls, one for each qubit
        let mut expanded = Vec::new();
        for &qubit in qubits {
            let single_qubit = vec![qubit];
            let gate_expanded = expand_gate_call_with_stack(
                gate_def,
                parameters,
                &single_qubit,
                all_definitions,
                &mut vec![gate_def.name.clone()],
            )?;
            expanded.extend(gate_expanded);
        }
        Ok(expanded)
    } else if gate_def.qargs.len() == 2 && qubits.len() > 2 {
        // Two-qubit gate applied to multiple qubits - apply pairwise
        if qubits.len() % 2 != 0 {
            return Err(PecosError::CompileInvalidOperation {
                operation: format!("gate '{}'", gate_def.name),
                reason: format!(
                    "Two-qubit gate '{}' applied to {} qubits (must be even number)",
                    gate_def.name,
                    qubits.len()
                ),
            });
        }
        let mut expanded = Vec::new();
        for i in (0..qubits.len()).step_by(2) {
            let pair = vec![qubits[i], qubits[i + 1]];
            let gate_expanded = expand_gate_call_with_stack(
                gate_def,
                parameters,
                &pair,
                all_definitions,
                &mut vec![gate_def.name.clone()],
            )?;
            expanded.extend(gate_expanded);
        }
        Ok(expanded)
    } else {
        // Normal case - single gate call
        expand_gate_call_with_stack(
            gate_def,
            parameters,
            qubits,
            all_definitions,
            &mut vec![gate_def.name.clone()],
        )
    }
}

fn expand_gate_call_with_stack(
    gate_def: &GateDefinition,
    parameters: &[f64],
    qubits: &[usize],
    all_definitions: &BTreeMap<String, GateDefinition>,
    expansion_stack: &mut Vec<String>,
) -> Result<Vec<Operation>, PecosError> {
    let mut expanded = Vec::new();

    // Create parameter mapping
    let mut param_map = BTreeMap::new();
    for (i, param_name) in gate_def.params.iter().enumerate() {
        if i < parameters.len() {
            param_map.insert(param_name.clone(), parameters[i]);
        }
    }

    // Create qubit mapping
    let mut qubit_map = BTreeMap::new();
    for (i, qarg_name) in gate_def.qargs.iter().enumerate() {
        if i < qubits.len() {
            qubit_map.insert(qarg_name.clone(), qubits[i]);
        }
    }

    // Expand each operation in the gate body
    for body_op in &gate_def.body {
        let mapped_name = body_op.name.clone();

        // Substitute parameters
        let mut new_params = Vec::new();
        for param_expr in &body_op.params {
            let value = evaluate_param_expr(param_expr, &param_map)?;
            new_params.push(value);
        }

        // Substitute qubits
        let mut new_qubits = Vec::new();
        for arg_name in &body_op.qargs {
            if let Some(&mapped_qubit) = qubit_map.get(arg_name) {
                new_qubits.push(mapped_qubit);
            }
        }

        let new_op = Operation::Gate {
            name: mapped_name.clone(),
            parameters: new_params.clone(),
            qubits: new_qubits.clone(),
        };

        // Check if this is a user-defined gate first
        if let Some(nested_def) = all_definitions.get(&mapped_name) {
            // User-defined gate - check for circular dependency
            if expansion_stack.contains(&mapped_name) {
                let mut cycle_info = String::new();
                write!(
                    cycle_info,
                    "Circular dependency detected: {} -> {}\n\n",
                    expansion_stack.join(" -> "),
                    mapped_name
                )
                .unwrap();

                cycle_info.push_str("To fix this error:\n");
                cycle_info.push_str("1. Check the gate definitions for circular references\n");
                cycle_info.push_str("2. Ensure no gate directly or indirectly calls itself\n");
                cycle_info.push_str(
                    "3. Consider breaking the cycle by refactoring your gate hierarchy\n\n",
                );
                cycle_info.push_str("The cycle involves these gates:\n");

                for (i, gate) in expansion_stack.iter().enumerate() {
                    write!(cycle_info, "  {}. '{}' calls ", i + 1, gate).unwrap();
                    if i + 1 < expansion_stack.len() {
                        writeln!(cycle_info, "'{}'", expansion_stack[i + 1]).unwrap();
                    } else {
                        writeln!(cycle_info, "'{mapped_name}' (completes the cycle)").unwrap();
                    }
                }

                return Err(PecosError::CompileCircularDependency(cycle_info));
            }

            expansion_stack.push(mapped_name.clone());

            let nested_expanded = expand_gate_call_with_stack(
                nested_def,
                &new_params,
                &new_qubits,
                all_definitions,
                expansion_stack,
            )?;

            expansion_stack.pop();
            expanded.extend(nested_expanded);
        } else if parse_native_gate(&mapped_name).is_some() {
            // Native gate - convert to uppercase
            let native_op = Operation::Gate {
                name: mapped_name.to_uppercase(),
                parameters: new_params.clone(),
                qubits: new_qubits.clone(),
            };
            expanded.push(native_op);
        } else if is_native_operation(&mapped_name) {
            // Other native operations (barrier, reset, etc.) - add directly
            expanded.push(new_op);
        } else {
            // Unknown gate
            return Err(PecosError::CompileInvalidOperation {
                operation: format!("gate '{mapped_name}'"),
                reason: format!(
                    "Undefined gate '{mapped_name}' - gate is neither native nor user-defined. Did you forget to include qelib1.inc?"
                ),
            });
        }
    }

    Ok(expanded)
}

/// Validate that no opaque gates are used in the program
///
/// # Errors
///
/// Returns an error if any opaque gates are used
pub fn validate_no_opaque_gate_usage(program: &Program) -> Result<(), PecosError> {
    let mut opaque_gates = BTreeSet::new();
    let mut gate_usages = Vec::new();

    for operation in &program.operations {
        match operation {
            Operation::OpaqueGate { name, .. } => {
                opaque_gates.insert(name.clone());
            }
            Operation::Gate { name, .. } => {
                gate_usages.push(name.clone());
            }
            _ => {}
        }
    }

    for gate_name in gate_usages {
        if opaque_gates.contains(&gate_name) {
            return Err(PecosError::CompileInvalidOperation {
                operation: QASMParser::QASM_OPERATION.to_string(),
                reason: format!(
                    "Opaque gate '{gate_name}' is used but opaque gates are not yet implemented in PECOS. \
                The gate is declared as opaque but cannot be executed."
                ),
            });
        }
    }

    Ok(())
}
