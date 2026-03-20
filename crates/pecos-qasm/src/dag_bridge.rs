// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Bridge between QASM programs and [`DagCircuit`].
//!
//! Converts between the parsed QASM AST ([`Program`]) and PECOS's circuit IR
//! ([`DagCircuit`]), preserving classical bit mappings, measurement targets,
//! and conditional operations.
//!
//! # Example
//!
//! ```
//! use pecos_qasm::dag_bridge::{qasm_to_dag, dag_to_qasm};
//! use pecos_qasm::parser::QASMParser;
//!
//! let qasm = r#"
//!     OPENQASM 2.0;
//!     include "qelib1.inc";
//!     qreg q[2];
//!     creg c[2];
//!     h q[0];
//!     cx q[0], q[1];
//!     measure q[0] -> c[0];
//!     measure q[1] -> c[1];
//! "#;
//!
//! let program = QASMParser::parse_str(qasm).unwrap();
//! let dag = qasm_to_dag(&program).unwrap();
//! assert_eq!(dag.gate_count(), 4);
//! assert_eq!(dag.num_cbits(), 2);
//!
//! let output = dag_to_qasm(&dag);
//! assert!(output.contains("h q[0]"));
//! ```

use std::collections::BTreeMap;
use std::fmt;

use pecos_core::gate_type::GateType;
use pecos_core::{Angle64, ClassicalBitId, Gate, QubitId};
use pecos_quantum::{Circuit, DagCircuit};

use crate::ast::{Expression, Operation};
use crate::parser::Program;

/// Errors that can occur during QASM bridge conversion.
#[derive(Debug)]
pub enum QasmBridgeError {
    /// An operation type not supported by the bridge.
    UnsupportedOperation(String),
    /// A gate name not recognized by the bridge.
    UnknownGate(String),
    /// A parameter evaluation error.
    ParameterError(String),
}

impl fmt::Display for QasmBridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedOperation(msg) => write!(f, "unsupported operation: {msg}"),
            Self::UnknownGate(name) => write!(f, "unknown gate: {name}"),
            Self::ParameterError(msg) => write!(f, "parameter error: {msg}"),
        }
    }
}

impl std::error::Error for QasmBridgeError {}

/// Convert a parsed QASM [`Program`] to a [`DagCircuit`].
///
/// Walks the program's operations and builds a circuit with auto-wiring.
/// Measurements with classical register mappings are preserved as measurement
/// targets on the DAG. Conditional operations (`if`) are preserved as conditions.
///
/// # Errors
///
/// Returns [`QasmBridgeError`] if an operation cannot be converted.
pub fn qasm_to_dag(program: &Program) -> Result<DagCircuit, QasmBridgeError> {
    let mut dag = DagCircuit::new();

    // Compute total classical bits
    let total_cbits: usize = program.classical_registers.values().sum();
    dag.set_num_cbits(total_cbits);

    // Build a mapping from (register_name, index) -> global classical bit index
    let cbit_map = build_cbit_map(&program.classical_registers);

    for op in &program.operations {
        convert_operation(&mut dag, op, &cbit_map)?;
    }

    Ok(dag)
}

/// Build a mapping from (`register_name`, `bit_index`) to global classical bit index.
fn build_cbit_map(
    classical_registers: &BTreeMap<String, usize>,
) -> BTreeMap<(String, usize), usize> {
    let mut map = BTreeMap::new();
    let mut offset = 0;
    for (name, &size) in classical_registers {
        for i in 0..size {
            map.insert((name.clone(), i), offset + i);
        }
        offset += size;
    }
    map
}

/// Convert a single QASM operation to gates in the `DagCircuit`.
fn convert_operation(
    dag: &mut DagCircuit,
    op: &Operation,
    cbit_map: &BTreeMap<(String, usize), usize>,
) -> Result<(), QasmBridgeError> {
    match op {
        Operation::NativeGate(gate) => {
            dag.add_gate_auto_wire(gate.clone());
            Ok(())
        }

        Operation::Gate {
            name,
            parameters,
            qubits,
        } => {
            let qubit_ids: Vec<QubitId> = qubits.iter().map(|&q| QubitId::from(q)).collect();
            let gate = resolve_gate(name, parameters, &qubit_ids)?;
            dag.add_gate_auto_wire(gate);
            Ok(())
        }

        Operation::MeasureWithMapping {
            gate,
            c_reg,
            c_index,
        } => {
            let node = dag.add_gate_auto_wire(gate.clone());
            if let Some(&global_cbit) = cbit_map.get(&(c_reg.clone(), *c_index)) {
                dag.set_measurement_target(node, ClassicalBitId::new(global_cbit));
            }
            Ok(())
        }

        Operation::RegMeasure { q_reg, c_reg } => {
            // Expand register-level measurement to individual measurements
            // We don't have access to the full register info here, so skip
            Err(QasmBridgeError::UnsupportedOperation(format!(
                "register-level measurement: measure {q_reg} -> {c_reg} (expand before bridging)"
            )))
        }

        Operation::Barrier { .. } => {
            // Barriers don't affect the circuit DAG structure; skip silently
            Ok(())
        }

        Operation::If {
            condition,
            operation,
        } => {
            // Extract the condition: `if (c == val)` where c is a classical register
            let (cbit_id, value) = extract_condition(condition, cbit_map)?;

            // Convert the inner operation, then set the condition on the last added node
            let node_before = dag.last_added_node();
            convert_operation(dag, operation, cbit_map)?;
            let node_after = dag.last_added_node();

            // If a new node was added, set its condition
            if node_after != node_before
                && let Some(node) = node_after
            {
                dag.set_condition(node, cbit_id, value);
            }

            Ok(())
        }

        Operation::ClassicalAssignment { .. } | Operation::VoidFunctionCall { .. } => {
            // Classical-only operations don't produce gates
            Ok(())
        }

        Operation::OpaqueGate { name, .. } => Err(QasmBridgeError::UnsupportedOperation(format!(
            "opaque gate: {name}"
        ))),
    }
}

/// Extract a condition from a QASM `if` expression.
///
/// Handles the common pattern `if (c == val)` where `c` is a classical register
/// and `val` is an integer (0 or 1 for single-bit conditions).
fn extract_condition(
    expr: &Expression,
    cbit_map: &BTreeMap<(String, usize), usize>,
) -> Result<(ClassicalBitId, bool), QasmBridgeError> {
    match expr {
        Expression::BinaryOp { op, left, right } if op == "==" => {
            // Left should be a variable (register name), right should be an integer
            let reg_name = match left.as_ref() {
                Expression::Variable(name) => name.clone(),
                Expression::BitId(name, _idx) => name.clone(),
                _ => {
                    return Err(QasmBridgeError::ParameterError(format!(
                        "expected register name in condition, got: {left}"
                    )));
                }
            };

            let value = match right.as_ref() {
                Expression::Integer(bv) => {
                    // For single-bit: 0 = false, nonzero = true
                    bv.iter().any(|b| *b)
                }
                Expression::Float(f) => *f != 0.0,
                _ => {
                    return Err(QasmBridgeError::ParameterError(format!(
                        "expected integer value in condition, got: {right}"
                    )));
                }
            };

            // For single-bit register, use bit 0
            if let Some(&global_cbit) = cbit_map.get(&(reg_name.clone(), 0)) {
                Ok((ClassicalBitId::new(global_cbit), value))
            } else {
                Err(QasmBridgeError::ParameterError(format!(
                    "unknown classical register in condition: {reg_name}"
                )))
            }
        }
        _ => Err(QasmBridgeError::ParameterError(format!(
            "unsupported condition expression: {expr}"
        ))),
    }
}

/// Resolve a QASM gate name and parameters to a PECOS Gate.
fn resolve_gate(
    name: &str,
    parameters: &[f64],
    qubits: &[QubitId],
) -> Result<Gate, QasmBridgeError> {
    let gate = match name.to_lowercase().as_str() {
        "h" => Gate::h(qubits),
        "x" => Gate::x(qubits),
        "y" => Gate::y(qubits),
        "z" => Gate::z(qubits),
        "s" => Gate::simple(GateType::SZ, qubits.to_vec()),
        "sdg" => Gate::simple(GateType::SZdg, qubits.to_vec()),
        "t" => Gate::simple(GateType::T, qubits.to_vec()),
        "tdg" => Gate::simple(GateType::Tdg, qubits.to_vec()),
        "sx" => Gate::simple(GateType::SX, qubits.to_vec()),
        "sxdg" => Gate::simple(GateType::SXdg, qubits.to_vec()),
        "id" | "i" => Gate::simple(GateType::I, qubits.to_vec()),
        "cx" | "cnot" => {
            if qubits.len() == 2 {
                Gate::cx(&[(qubits[0], qubits[1])])
            } else {
                return Err(QasmBridgeError::ParameterError(format!(
                    "CX gate requires 2 qubits, got {}",
                    qubits.len()
                )));
            }
        }
        "cy" => Gate::simple(GateType::CY, qubits.to_vec()),
        "cz" => Gate::simple(GateType::CZ, qubits.to_vec()),
        "ch" => Gate::simple(GateType::CH, qubits.to_vec()),
        "swap" => Gate::simple(GateType::SWAP, qubits.to_vec()),
        "ccx" | "toffoli" => Gate::simple(GateType::CCX, qubits.to_vec()),
        "rx" => {
            if parameters.len() == 1 {
                Gate::rx(Angle64::from_radians(parameters[0]), qubits)
            } else {
                return Err(QasmBridgeError::ParameterError(
                    "RX gate requires 1 parameter".into(),
                ));
            }
        }
        "ry" => {
            if parameters.len() == 1 {
                Gate::ry(Angle64::from_radians(parameters[0]), qubits)
            } else {
                return Err(QasmBridgeError::ParameterError(
                    "RY gate requires 1 parameter".into(),
                ));
            }
        }
        "rz" => {
            if parameters.len() == 1 {
                Gate::rz(Angle64::from_radians(parameters[0]), qubits)
            } else {
                return Err(QasmBridgeError::ParameterError(
                    "RZ gate requires 1 parameter".into(),
                ));
            }
        }
        "u" | "u3" => {
            if parameters.len() == 3 {
                Gate::with_angles(
                    GateType::U,
                    vec![
                        Angle64::from_radians(parameters[0]),
                        Angle64::from_radians(parameters[1]),
                        Angle64::from_radians(parameters[2]),
                    ],
                    qubits.to_vec(),
                )
            } else {
                return Err(QasmBridgeError::ParameterError(
                    "U gate requires 3 parameters".into(),
                ));
            }
        }
        "crz" => {
            if parameters.len() == 1 && qubits.len() == 2 {
                Gate::with_angles(
                    GateType::CRZ,
                    vec![Angle64::from_radians(parameters[0])],
                    qubits.to_vec(),
                )
            } else {
                return Err(QasmBridgeError::ParameterError(
                    "CRZ gate requires 1 parameter and 2 qubits".into(),
                ));
            }
        }
        "rxx" => {
            if parameters.len() == 1 && qubits.len() == 2 {
                Gate::with_angles(
                    GateType::RXX,
                    vec![Angle64::from_radians(parameters[0])],
                    qubits.to_vec(),
                )
            } else {
                return Err(QasmBridgeError::ParameterError(
                    "RXX gate requires 1 parameter and 2 qubits".into(),
                ));
            }
        }
        "ryy" => {
            if parameters.len() == 1 && qubits.len() == 2 {
                Gate::with_angles(
                    GateType::RYY,
                    vec![Angle64::from_radians(parameters[0])],
                    qubits.to_vec(),
                )
            } else {
                return Err(QasmBridgeError::ParameterError(
                    "RYY gate requires 1 parameter and 2 qubits".into(),
                ));
            }
        }
        "rzz" => {
            if parameters.len() == 1 && qubits.len() == 2 {
                Gate::with_angles(
                    GateType::RZZ,
                    vec![Angle64::from_radians(parameters[0])],
                    qubits.to_vec(),
                )
            } else {
                return Err(QasmBridgeError::ParameterError(
                    "RZZ gate requires 1 parameter and 2 qubits".into(),
                ));
            }
        }
        "measure" => Gate::measure(qubits),
        "reset" => Gate::simple(GateType::PZ, qubits.to_vec()),
        _ => {
            return Err(QasmBridgeError::UnknownGate(name.to_string()));
        }
    };
    Ok(gate)
}

/// Convert a [`DagCircuit`] to a QASM 2.0 string.
///
/// Walks the circuit in topological order and emits QASM statements,
/// including register declarations, gate operations, measurements with
/// classical targets, and conditional operations.
#[must_use]
pub fn dag_to_qasm(dag: &DagCircuit) -> String {
    let mut lines = Vec::new();

    lines.push("OPENQASM 2.0;".to_string());

    // Determine number of qubits
    let num_qubits = if dag.gate_count() > 0 {
        dag.max_qubit() + 1
    } else {
        0
    };
    let num_cbits = dag.num_cbits();

    if num_qubits > 0 {
        lines.push(format!("qreg q[{num_qubits}];"));
    }
    if num_cbits > 0 {
        lines.push(format!("creg c[{num_cbits}];"));
    }

    // Walk in topological order
    for node in dag.topological_order() {
        let Some(gate) = dag.gate(node) else {
            continue;
        };

        let condition = dag.condition(node);
        let meas_target = dag.measurement_target(node);

        let stmt = format_gate_stmt(gate, meas_target, condition);
        lines.push(stmt);
    }

    lines.join("\n") + "\n"
}

/// Format a single gate as a QASM statement.
fn format_gate_stmt(
    gate: &Gate,
    meas_target: Option<ClassicalBitId>,
    condition: Option<(ClassicalBitId, bool)>,
) -> String {
    let mut prefix = String::new();
    if let Some((cbit, value)) = condition {
        let val = i32::from(value);
        prefix = format!("if(c[{}]=={val}) ", cbit.index());
    }

    // Handle measurement with target
    if gate.gate_type == GateType::MZ
        && let Some(cbit) = meas_target
    {
        let qubit_strs: Vec<String> = gate.qubits.iter().map(|q| format!("q[{}]", q.0)).collect();
        return format!(
            "{prefix}measure {} -> c[{}];",
            qubit_strs.join(", "),
            cbit.index()
        );
    }

    let name = gate_type_to_qasm_name(gate.gate_type);

    // Format parameters
    let params = if gate.angles.is_empty() {
        String::new()
    } else {
        let param_strs: Vec<String> = gate
            .angles
            .iter()
            .map(|a| format!("{}", a.to_radians()))
            .collect();
        format!("({})", param_strs.join(", "))
    };

    // Format qubits
    let qubit_strs: Vec<String> = gate.qubits.iter().map(|q| format!("q[{}]", q.0)).collect();

    format!("{prefix}{name}{params} {};", qubit_strs.join(", "))
}

/// Map a PECOS `GateType` to its QASM 2.0 name.
fn gate_type_to_qasm_name(gate_type: GateType) -> &'static str {
    match gate_type {
        GateType::I => "id",
        GateType::X => "x",
        GateType::Y => "y",
        GateType::Z => "z",
        GateType::H => "h",
        GateType::SX => "sx",
        GateType::SXdg => "sxdg",
        GateType::SY => "sy",
        GateType::SYdg => "sydg",
        GateType::SZ => "s",
        GateType::SZdg => "sdg",
        GateType::T => "t",
        GateType::Tdg => "tdg",
        GateType::RX => "rx",
        GateType::RY => "ry",
        GateType::RZ => "rz",
        GateType::U => "u",
        GateType::R1XY => "r1xy",
        GateType::CX => "cx",
        GateType::CY => "cy",
        GateType::CZ => "cz",
        GateType::CH => "ch",
        GateType::SZZ => "szz",
        GateType::SZZdg => "szzdg",
        GateType::SWAP => "swap",
        GateType::CRZ => "crz",
        GateType::RXX => "rxx",
        GateType::RYY => "ryy",
        GateType::RZZ => "rzz",
        GateType::CCX => "ccx",
        GateType::MZ | GateType::MeasureFree => "measure",
        GateType::PZ => "reset",
        GateType::QAlloc => "qalloc",
        GateType::QFree => "qfree",
        GateType::Idle => "idle",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bell_state_dag_to_qasm() {
        let mut dag = DagCircuit::new();
        dag.set_num_cbits(2);
        dag.h(0);
        dag.cx(0, 1);
        dag.mz_to(0, ClassicalBitId::new(0));
        dag.mz_to(1, ClassicalBitId::new(1));

        let qasm = dag_to_qasm(&dag);
        assert!(qasm.contains("OPENQASM 2.0;"));
        assert!(qasm.contains("qreg q[2];"));
        assert!(qasm.contains("creg c[2];"));
        assert!(qasm.contains("h q[0];"));
        assert!(qasm.contains("cx q[0], q[1];"));
        assert!(qasm.contains("measure q[0] -> c[0];"));
        assert!(qasm.contains("measure q[1] -> c[1];"));
    }

    #[test]
    fn test_conditional_dag_to_qasm() {
        let mut dag = DagCircuit::new();
        dag.set_num_cbits(1);
        dag.h(0);
        dag.mz_to(0, ClassicalBitId::new(0));
        dag.if_bit(ClassicalBitId::new(0), true).x(1);

        let qasm = dag_to_qasm(&dag);
        assert!(qasm.contains("if(c[0]==1) x q[1];"));
    }

    #[test]
    fn test_parameterized_gate_dag_to_qasm() {
        let mut dag = DagCircuit::new();
        dag.rz(std::f64::consts::FRAC_PI_4, 0);

        let qasm = dag_to_qasm(&dag);
        assert!(qasm.contains("rz("));
        assert!(qasm.contains("q[0];"));
    }

    #[test]
    fn test_qasm_to_dag_bell_state() {
        let mut program = Program {
            version: "2.0".to_string(),
            total_qubits: 2,
            ..Default::default()
        };
        program
            .quantum_registers
            .insert("q".to_string(), vec![0, 1]);
        program.classical_registers.insert("c".to_string(), 2);
        program.qubit_map.insert(0, ("q".to_string(), 0));
        program.qubit_map.insert(1, ("q".to_string(), 1));

        program
            .operations
            .push(Operation::NativeGate(Gate::h(&[0])));
        program
            .operations
            .push(Operation::NativeGate(Gate::cx(&[(0, 1)])));
        program.operations.push(Operation::MeasureWithMapping {
            gate: Gate::measure(&[QubitId::from(0)]),
            c_reg: "c".to_string(),
            c_index: 0,
        });
        program.operations.push(Operation::MeasureWithMapping {
            gate: Gate::measure(&[QubitId::from(1)]),
            c_reg: "c".to_string(),
            c_index: 1,
        });

        let dag = qasm_to_dag(&program).unwrap();
        assert_eq!(dag.gate_count(), 4);
        assert_eq!(dag.num_cbits(), 2);
        assert_eq!(dag.measurement_targets().len(), 2);
    }

    #[test]
    fn test_resolve_gate_basic() {
        let qubits = vec![QubitId::from(0)];
        let gate = resolve_gate("h", &[], &qubits).unwrap();
        assert_eq!(gate.gate_type, GateType::H);

        let gate = resolve_gate("x", &[], &qubits).unwrap();
        assert_eq!(gate.gate_type, GateType::X);

        let gate = resolve_gate("s", &[], &qubits).unwrap();
        assert_eq!(gate.gate_type, GateType::SZ);

        let gate = resolve_gate("t", &[], &qubits).unwrap();
        assert_eq!(gate.gate_type, GateType::T);
    }

    #[test]
    fn test_resolve_gate_parameterized() {
        let qubits = vec![QubitId::from(0)];
        let gate = resolve_gate("rz", &[std::f64::consts::FRAC_PI_2], &qubits).unwrap();
        assert_eq!(gate.gate_type, GateType::RZ);
    }

    #[test]
    fn test_resolve_gate_two_qubit() {
        let qubits = vec![QubitId::from(0), QubitId::from(1)];
        let gate = resolve_gate("cx", &[], &qubits).unwrap();
        assert_eq!(gate.gate_type, GateType::CX);
    }

    #[test]
    fn test_resolve_gate_unknown() {
        let qubits = vec![QubitId::from(0)];
        assert!(resolve_gate("nonexistent_gate", &[], &qubits).is_err());
    }

    #[test]
    fn test_empty_circuit_to_qasm() {
        let dag = DagCircuit::new();
        let qasm = dag_to_qasm(&dag);
        assert!(qasm.contains("OPENQASM 2.0;"));
        assert!(!qasm.contains("qreg"));
    }

    #[test]
    fn test_barrier_skipped() {
        let mut program = Program {
            version: "2.0".to_string(),
            total_qubits: 1,
            ..Default::default()
        };
        program.quantum_registers.insert("q".to_string(), vec![0]);

        program
            .operations
            .push(Operation::NativeGate(Gate::h(&[0])));
        program
            .operations
            .push(Operation::Barrier { qubits: vec![0] });
        program
            .operations
            .push(Operation::NativeGate(Gate::x(&[0])));

        let dag = qasm_to_dag(&program).unwrap();
        assert_eq!(dag.gate_count(), 2); // Barrier is skipped
    }
}
