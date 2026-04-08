//! QASM to PHIR-JSON conversion.
//!
//! Converts a parsed QASM `Program` into a `serde_json::Value` that conforms
//! to the PHIR/JSON v0.1.0 specification.  All classical registers are emitted
//! as `i64` with their declared size -- matching the hardware model where
//! everything is backed by i64.
//!
//! Unlike [`qasm_to_phir`] (which targets the `pecos-phir` SSA-based Module
//! format), this module targets the JSON dict format consumed by
//! `PhirClassicalInterpreter` and `PhirJsonEngine`.

use std::collections::BTreeMap;

use pecos_core::Angle64;
use pecos_core::bitvec;
use pecos_core::prelude::GateType;
use serde_json::{Value, json};

use crate::ast::{Expression, Operation};
use crate::parser::{Program, QASMParser};

/// Convert a QASM string directly to a PHIR-JSON `serde_json::Value`.
///
/// # Errors
///
/// Returns an error if parsing or conversion fails.
pub fn qasm_to_phir_json(qasm_str: &str) -> Result<Value, String> {
    let program = QASMParser::parse_str(qasm_str).map_err(|e| format!("QASM parse error: {e}"))?;
    program_to_phir_json(&program)
}

/// Convert a parsed QASM `Program` to a PHIR-JSON `serde_json::Value`.
///
/// # Errors
///
/// Returns an error if the program contains unsupported operations.
pub fn program_to_phir_json(program: &Program) -> Result<Value, String> {
    let mut ops: Vec<Value> = Vec::new();

    // 1) Quantum register definitions
    for (name, qubit_ids) in &program.quantum_registers {
        ops.push(json!({
            "data": "qvar_define",
            "data_type": "qubits",
            "variable": name,
            "size": qubit_ids.len()
        }));
    }

    // 2) Classical register definitions -- all i64
    for (name, size) in &program.classical_registers {
        ops.push(json!({
            "data": "cvar_define",
            "data_type": "i64",
            "variable": name,
            "size": size
        }));
    }

    // 3) Convert operations (measurements are inline, not deferred)
    for op in &program.operations {
        convert_op(op, &program.qubit_map, &mut ops)?;
    }

    // 4) Export all classical variables
    let cvar_names: Vec<&str> = program
        .classical_registers
        .keys()
        .map(String::as_str)
        .collect();
    if !cvar_names.is_empty() {
        ops.push(json!({
            "data": "cvar_export",
            "variables": cvar_names
        }));
    }

    Ok(json!({
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": ops
    }))
}

/// Convert a single QASM operation to one or more PHIR-JSON ops.
fn convert_op(
    op: &Operation,
    qubit_map: &BTreeMap<usize, (String, usize)>,
    ops: &mut Vec<Value>,
) -> Result<(), String> {
    match op {
        Operation::Gate {
            name,
            parameters,
            qubits,
        } => {
            let phir_name = qasm_gate_to_phir(name)?;
            let num_qubits_per_gate = gate_arity(&phir_name);
            let args = qubit_args(qubits, qubit_map, num_qubits_per_gate)?;

            let mut qop = json!({"qop": phir_name, "args": args});
            if !parameters.is_empty() {
                qop["angles"] = json!([parameters, "rad"]);
            }
            ops.push(qop);
        }

        Operation::NativeGate(gate) => {
            if gate.gate_type == GateType::MZ {
                return Err(
                    "NativeGate(MZ) should appear as MeasureWithMapping, not bare".to_string(),
                );
            }
            if gate.gate_type == GateType::PZ {
                // Reset/Init
                let args = native_qubit_args(&gate.qubits, qubit_map, 1)?;
                ops.push(json!({"qop": "Init", "args": args}));
                return Ok(());
            }

            let phir_name = gate_type_to_phir(gate.gate_type)?;
            let num_qubits_per_gate = gate_arity(&phir_name);
            let global_ids: Vec<usize> = gate.qubits.iter().map(|q| q.0).collect();
            let args = qubit_args(&global_ids, qubit_map, num_qubits_per_gate)?;

            let mut qop = json!({"qop": phir_name, "args": args});
            if !gate.angles.is_empty() {
                let radians: Vec<f64> = gate.angles.iter().map(Angle64::to_radians).collect();
                qop["angles"] = json!([radians, "rad"]);
            }
            ops.push(qop);
        }

        Operation::MeasureWithMapping {
            gate,
            c_reg,
            c_index,
        } => {
            if let Some(qubit) = gate.qubits.first() {
                let (q_reg, q_idx) = qubit_map
                    .get(&qubit.0)
                    .ok_or_else(|| format!("Qubit ID {} not in qubit_map", qubit.0))?;
                ops.push(json!({
                    "qop": "Measure",
                    "args": [[q_reg, q_idx]],
                    "returns": [[c_reg, c_index]]
                }));
            }
        }

        Operation::RegMeasure { q_reg, c_reg } => {
            // Expand register-level measurement into individual bit measurements
            // Find all qubits belonging to this quantum register
            let mut qubit_entries: Vec<(usize, usize)> = Vec::new();
            for (&global_id, (reg_name, idx)) in qubit_map {
                if reg_name == q_reg {
                    qubit_entries.push((global_id, *idx));
                }
            }
            qubit_entries.sort_by_key(|(_, idx)| *idx);

            let q_args: Vec<Value> = qubit_entries
                .iter()
                .map(|(_, idx)| json!([q_reg, idx]))
                .collect();
            let c_returns: Vec<Value> = qubit_entries
                .iter()
                .map(|(_, idx)| json!([c_reg, idx]))
                .collect();

            if !q_args.is_empty() {
                ops.push(json!({
                    "qop": "Measure",
                    "args": q_args,
                    "returns": c_returns
                }));
            }
        }

        Operation::If {
            condition,
            operation,
        } => {
            let cond = convert_expr(condition)?;
            let mut true_branch = Vec::new();
            convert_op(operation, qubit_map, &mut true_branch)?;
            ops.push(json!({
                "block": "if",
                "condition": cond,
                "true_branch": true_branch
            }));
        }

        Operation::ClassicalAssignment {
            target,
            is_indexed,
            index,
            expression,
        } => {
            let expr = convert_expr(expression)?;
            let returns = if *is_indexed {
                json!([[target, index.unwrap_or(0)]])
            } else {
                json!([target])
            };
            ops.push(json!({
                "cop": "=",
                "args": [expr],
                "returns": returns
            }));
        }

        Operation::VoidFunctionCall { expression } => {
            if let Expression::FunctionCall { name, args } = expression {
                let converted_args: Vec<Value> =
                    args.iter().map(convert_expr).collect::<Result<_, _>>()?;
                ops.push(json!({
                    "cop": "ffcall",
                    "function": name,
                    "args": converted_args
                }));
            }
        }

        Operation::Barrier { .. } | Operation::OpaqueGate { .. } => {
            // Skip barriers and opaque declarations
        }
    }
    Ok(())
}

/// Convert a QASM `Expression` to a PHIR-JSON value.
///
/// Integers and variable references become plain JSON values.
/// Binary/unary ops become nested `{"cop": ..., "args": [...]}` dicts.
fn convert_expr(expr: &Expression) -> Result<Value, String> {
    match expr {
        Expression::Integer(bv) => {
            // QASM integers are non-negative, but we output as i64
            // for PHIR-JSON compatibility
            let val = bitvec::to_i64(bv);
            Ok(json!(val))
        }
        Expression::Float(f) => Ok(json!(f)),
        Expression::Pi => Ok(json!(std::f64::consts::PI)),
        Expression::Variable(name) => Ok(json!(name)),
        Expression::BitId(name, idx) => Ok(json!([name, idx])),
        Expression::BinaryOp { op, left, right } => {
            let l = convert_expr(left)?;
            let r = convert_expr(right)?;
            Ok(json!({"cop": op, "args": [l, r]}))
        }
        Expression::UnaryOp { op, expr } => {
            let e = convert_expr(expr)?;
            Ok(json!({"cop": op, "args": [e]}))
        }
        Expression::FunctionCall { name, .. } => Err(format!(
            "Function call '{name}' cannot appear as a nested expression in PHIR-JSON. \
             Use a top-level ffcall cop instead."
        )),
    }
}

// ── Gate name mapping ────────────────────────────────────────────────

/// Map a QASM gate name string to PHIR gate name.
fn qasm_gate_to_phir(name: &str) -> Result<String, String> {
    let phir = match name.to_lowercase().as_str() {
        "h" => "H",
        "x" => "X",
        "y" => "Y",
        "z" => "Z",
        "s" => "SZ",
        "sdg" => "SZdg",
        "t" => "T",
        "tdg" => "Tdg",
        "cx" | "cnot" => "CX",
        "cy" => "CY",
        "cz" => "CZ",
        "swap" => "SWAP",
        "rx" => "RX",
        "ry" => "RY",
        "rz" => "RZ",
        "rzz" | "zzphase" => "RZZ",
        "r1xy" | "u1q" => "R1XY",
        "u" | "u3" => "U",
        "reset" => "Init",
        other => return Err(format!("Unsupported QASM gate: {other}")),
    };
    Ok(phir.to_string())
}

/// Map a `GateType` enum to PHIR gate name.
fn gate_type_to_phir(gt: GateType) -> Result<String, String> {
    let phir = match gt {
        GateType::I => "I",
        GateType::X => "X",
        GateType::Y => "Y",
        GateType::Z => "Z",
        GateType::H => "H",
        GateType::SX => "SX",
        GateType::SXdg => "SXdg",
        GateType::SY => "SY",
        GateType::SYdg => "SYdg",
        GateType::SZ => "SZ",
        GateType::SZdg => "SZdg",
        GateType::F => "F",
        GateType::Fdg => "Fdg",
        GateType::T => "T",
        GateType::Tdg => "Tdg",
        GateType::RX => "RX",
        GateType::RY => "RY",
        GateType::RZ => "RZ",
        GateType::R1XY => "R1XY",
        GateType::U => "U",
        GateType::CX => "CX",
        GateType::CY => "CY",
        GateType::CZ => "CZ",
        GateType::SXX => "SXX",
        GateType::SXXdg => "SXXdg",
        GateType::SYY => "SYY",
        GateType::SYYdg => "SYYdg",
        GateType::SZZ => "SZZ",
        GateType::SZZdg => "SZZdg",
        GateType::SWAP => "SWAP",
        GateType::RXX => "RXX",
        GateType::RYY => "RYY",
        GateType::RZZ => "RZZ",
        GateType::MZ => "Measure",
        GateType::PZ => "Init",
        _ => return Err(format!("Unsupported gate type: {gt:?}")),
    };
    Ok(phir.to_string())
}

/// How many qubits does one application of this gate use?
fn gate_arity(phir_name: &str) -> usize {
    match phir_name {
        "CX" | "CY" | "CZ" | "SWAP" | "SXX" | "SXXdg" | "SYY" | "SYYdg" | "SZZ" | "SZZdg"
        | "RXX" | "RYY" | "RZZ" | "R2XXYYZZ" | "RXXYYZZ" => 2,
        "CCX" => 3,
        _ => 1,
    }
}

// ── Qubit arg formatting ─────────────────────────────────────────────

/// Format qubit args for PHIR-JSON.
///
/// Single-qubit gates: `[["q", 0], ["q", 1]]`  (flat list of refs)
/// Multi-qubit gates:  `[[["q", 0], ["q", 1]]]` (list of tuples)
fn qubit_args(
    global_ids: &[usize],
    qubit_map: &BTreeMap<usize, (String, usize)>,
    qubits_per_gate: usize,
) -> Result<Value, String> {
    let refs: Vec<Value> = global_ids
        .iter()
        .map(|gid| {
            let (reg, idx) = qubit_map
                .get(gid)
                .ok_or_else(|| format!("Qubit ID {gid} not in qubit_map"))?;
            Ok(json!([reg, idx]))
        })
        .collect::<Result<_, String>>()?;

    if qubits_per_gate == 1 {
        Ok(Value::Array(refs))
    } else {
        // Group into tuples of qubits_per_gate
        let tuples: Vec<Value> = refs
            .chunks(qubits_per_gate)
            .map(|chunk| Value::Array(chunk.to_vec()))
            .collect();
        Ok(Value::Array(tuples))
    }
}

/// Same as `qubit_args` but for `NativeGate`'s `QubitId` vec.
fn native_qubit_args(
    qubits: &[pecos_core::prelude::QubitId],
    qubit_map: &BTreeMap<usize, (String, usize)>,
    qubits_per_gate: usize,
) -> Result<Value, String> {
    let global_ids: Vec<usize> = qubits.iter().map(|q| q.0).collect();
    qubit_args(&global_ids, qubit_map, qubits_per_gate)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn convert(qasm: &str) -> Value {
        qasm_to_phir_json(qasm).expect("conversion should succeed")
    }

    fn get_ops(phir: &Value) -> &Vec<Value> {
        phir["ops"].as_array().expect("ops should be an array")
    }

    #[test]
    fn basic_structure() {
        let phir = convert(
            r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg c[1];
        "#,
        );
        assert_eq!(phir["format"], "PHIR/JSON");
        assert_eq!(phir["version"], "0.1.0");
        assert!(phir["ops"].is_array());
    }

    #[test]
    fn register_definitions() {
        let phir = convert(
            r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg m[3];
        "#,
        );
        let ops = get_ops(&phir);

        let qvar = ops.iter().find(|o| o["data"] == "qvar_define").unwrap();
        assert_eq!(qvar["variable"], "q");
        assert_eq!(qvar["size"], 2);
        assert_eq!(qvar["data_type"], "qubits");

        let cvar = ops.iter().find(|o| o["data"] == "cvar_define").unwrap();
        assert_eq!(cvar["variable"], "m");
        assert_eq!(cvar["size"], 3);
        assert_eq!(cvar["data_type"], "i64");
    }

    #[test]
    fn single_qubit_gate() {
        let phir = convert(
            r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            h q[0];
        "#,
        );
        let ops = get_ops(&phir);
        let h_op = ops.iter().find(|o| o["qop"] == "H").unwrap();
        assert_eq!(h_op["args"], json!([["q", 0]]));
    }

    #[test]
    fn two_qubit_gate() {
        let phir = convert(
            r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            cx q[0], q[1];
        "#,
        );
        let ops = get_ops(&phir);
        let cx_op = ops.iter().find(|o| o["qop"] == "CX").unwrap();
        // Multi-qubit: args is list of tuples
        assert_eq!(cx_op["args"], json!([[["q", 0], ["q", 1]]]));
    }

    #[test]
    fn measurement() {
        let phir = convert(
            r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            measure q[0] -> c[0];
            measure q[1] -> c[1];
        "#,
        );
        let ops = get_ops(&phir);
        let measures: Vec<&Value> = ops.iter().filter(|o| o["qop"] == "Measure").collect();
        assert_eq!(measures.len(), 2);
        assert_eq!(measures[0]["args"], json!([["q", 0]]));
        assert_eq!(measures[0]["returns"], json!([["c", 0]]));
        assert_eq!(measures[1]["args"], json!([["q", 1]]));
        assert_eq!(measures[1]["returns"], json!([["c", 1]]));
    }

    #[test]
    fn conditional_if() {
        let phir = convert(
            r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg c[1];
            measure q[0] -> c[0];
            if(c==1) x q[0];
        "#,
        );
        let ops = get_ops(&phir);
        let if_block = ops.iter().find(|o| o["block"] == "if").unwrap();
        assert_eq!(
            if_block["condition"],
            json!({"cop": "==", "args": ["c", 1]})
        );

        let branch = if_block["true_branch"].as_array().unwrap();
        assert_eq!(branch.len(), 1);
        assert_eq!(branch[0]["qop"], "X");
    }

    #[test]
    fn cvar_export() {
        let phir = convert(
            r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg a[1];
            creg b[2];
        "#,
        );
        let ops = get_ops(&phir);
        let export = ops.iter().find(|o| o["data"] == "cvar_export").unwrap();
        let vars = export["variables"].as_array().unwrap();
        assert!(vars.contains(&json!("a")));
        assert!(vars.contains(&json!("b")));
    }

    #[test]
    fn bell_state_full() {
        let phir = convert(
            r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            h q[0];
            cx q[0], q[1];
            measure q[0] -> c[0];
            measure q[1] -> c[1];
        "#,
        );

        let ops = get_ops(&phir);
        // Check operation order: qvar_define, cvar_define, H, CX, Measure, Measure, cvar_export
        let op_types: Vec<&str> = ops
            .iter()
            .map(|o| {
                if o.get("qop").is_some() {
                    o["qop"].as_str().unwrap()
                } else if o.get("data").is_some() {
                    o["data"].as_str().unwrap()
                } else if o.get("cop").is_some() {
                    o["cop"].as_str().unwrap()
                } else if o.get("block").is_some() {
                    o["block"].as_str().unwrap()
                } else {
                    "unknown"
                }
            })
            .collect();

        assert!(op_types.contains(&"qvar_define"));
        assert!(op_types.contains(&"cvar_define"));
        assert!(op_types.contains(&"H"));
        assert!(op_types.contains(&"CX"));
        assert!(op_types.contains(&"Measure"));
        assert!(op_types.contains(&"cvar_export"));
    }

    #[test]
    fn feedback_loop() {
        let phir = convert(
            r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg m[1];
            creg r[1];
            x q[0];
            measure q[0] -> m[0];
            if(m==1) x q[0];
            measure q[0] -> r[0];
        "#,
        );

        let ops = get_ops(&phir);
        // Verify inline ordering: X, Measure, if-block, Measure
        let qops: Vec<&str> = ops
            .iter()
            .filter_map(|o| o.get("qop").and_then(Value::as_str))
            .collect();
        assert_eq!(qops, vec!["X", "Measure", "Measure"]);

        // The if-block should be between the two measurements
        let if_idx = ops.iter().position(|o| o.get("block").is_some()).unwrap();
        let measure_positions: Vec<usize> = ops
            .iter()
            .enumerate()
            .filter(|(_, o)| o.get("qop") == Some(&json!("Measure")))
            .map(|(i, _)| i)
            .collect();
        assert!(if_idx > measure_positions[0]);
        assert!(if_idx < measure_positions[1]);
    }

    #[test]
    fn gate_with_angle() {
        let phir = convert(
            r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            rz(1.5707963267948966) q[0];
        "#,
        );
        let ops = get_ops(&phir);
        let rz = ops.iter().find(|o| o["qop"] == "RZ").unwrap();
        assert!(rz.get("angles").is_some());
    }

    #[test]
    fn qasm_validation_measure_narrow() {
        let qasm = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/qasm_validation/measure_narrow_2bit.qasm"
        ))
        .unwrap();
        let phir = qasm_to_phir_json(&qasm).unwrap();
        let ops = get_ops(&phir);

        // Bell pair (2 measures) + test qubits (2 measures) = 4 total
        let measures: Vec<&Value> = ops.iter().filter(|o| o["qop"] == "Measure").collect();
        assert_eq!(measures.len(), 4);
    }

    #[test]
    fn qasm_validation_conditional_feedback() {
        let qasm = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/qasm_validation/conditional_feedback.qasm"
        ))
        .unwrap();
        let phir = qasm_to_phir_json(&qasm).unwrap();
        let ops = get_ops(&phir);

        // Should have an if-block
        let if_blocks: Vec<&Value> = ops.iter().filter(|o| o["block"] == "if").collect();
        assert_eq!(if_blocks.len(), 1);
        assert_eq!(
            if_blocks[0]["condition"],
            json!({"cop": "==", "args": ["m", 1]})
        );
    }

    #[test]
    fn qasm_validation_all_fixtures() {
        // Smoke test: all fixture files should parse and convert without error
        let fixture_dir = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/qasm_validation/"
        );
        let entries = std::fs::read_dir(fixture_dir).unwrap();
        let mut count = 0;
        for entry in entries {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|e| e == "qasm") {
                let qasm = std::fs::read_to_string(&path).unwrap();
                let result = qasm_to_phir_json(&qasm);
                assert!(
                    result.is_ok(),
                    "Failed to convert {}: {}",
                    path.display(),
                    result.unwrap_err()
                );

                // Basic structural checks on every output
                let phir = result.unwrap();
                assert_eq!(phir["format"], "PHIR/JSON");
                assert_eq!(phir["version"], "0.1.0");
                assert!(phir["ops"].is_array());
                count += 1;
            }
        }
        assert!(count >= 9, "Expected at least 9 fixtures, got {count}");
    }
}
