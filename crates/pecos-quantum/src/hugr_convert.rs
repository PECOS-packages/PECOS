// Copyright 2025 The PECOS Developers
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

//! HUGR to `DagCircuit` conversion.
//!
//! This module provides conversion between HUGR (Hierarchical Unified Graph
//! Representation) and [`DagCircuit`].
//!
//! Currently supports basic quantum circuits without classical control flow,
//! loops, or conditionals.

use std::collections::{BTreeMap, BTreeSet};

use pecos_core::gate_type::GateType;
use pecos_core::{Angle64, Gate, QubitId};
use tket::TketOp;
use tket::extension::rotation::ConstRotation;
use tket::hugr::builder::{DFGBuilder, Dataflow, DataflowHugr};
use tket::hugr::extension::prelude::qb_t;
use tket::hugr::ops::OpType;
use tket::hugr::types::Signature;
use tket::hugr::{Hugr, HugrView, IncomingPort, Node, NodeIndex, PortIndex, Wire};

use crate::circuit::{Circuit, GateHandle, GateView};
use crate::{Attribute, DagCircuit};

/// Error type for HUGR conversion failures.
#[derive(Debug, Clone)]
pub enum HugrConvertError {
    /// The HUGR contains unsupported structures (loops, nested conditionals).
    UnsupportedStructure(String),
    /// An unknown quantum operation was encountered.
    UnknownOperation(String),
    /// The operation is not from a supported extension.
    UnsupportedExtension(String),
}

impl std::fmt::Display for HugrConvertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HugrConvertError::UnsupportedStructure(msg) => {
                write!(f, "Unsupported HUGR structure: {msg}")
            }
            HugrConvertError::UnknownOperation(op) => {
                write!(f, "Unknown quantum operation: {op}")
            }
            HugrConvertError::UnsupportedExtension(ext) => {
                write!(f, "Unsupported extension: {ext}")
            }
        }
    }
}

impl std::error::Error for HugrConvertError {}

/// Maps HUGR operation names to PECOS `GateType`.
#[must_use]
pub fn hugr_op_to_gate_type(op_name: &str) -> Option<GateType> {
    match op_name {
        // Single-qubit gates
        "H" => Some(GateType::H),
        "X" => Some(GateType::X),
        "Y" => Some(GateType::Y),
        "Z" => Some(GateType::Z),
        "S" => Some(GateType::SZ),
        "Sdg" => Some(GateType::SZdg),
        "T" => Some(GateType::T),
        "Tdg" => Some(GateType::Tdg),
        "V" => Some(GateType::SX),
        "Vdg" => Some(GateType::SXdg),
        "Rx" => Some(GateType::RX),
        "Ry" => Some(GateType::RY),
        "Rz" => Some(GateType::RZ),
        // Two-qubit gates
        "CX" => Some(GateType::CX),
        "CY" => Some(GateType::CY),
        "CZ" => Some(GateType::CZ),
        "CH" => Some(GateType::CH),
        "ZZMax" => Some(GateType::SZZ),
        "SWAP" => Some(GateType::SWAP),
        "CRz" => Some(GateType::CRZ),
        // Three-qubit gates
        "Toffoli" | "CCX" => Some(GateType::CCX),
        // Lifecycle operations
        "QAlloc" => Some(GateType::QAlloc),
        "QFree" => Some(GateType::QFree),
        "Measure" => Some(GateType::Measure),
        "MeasureFree" => Some(GateType::MeasureFree),
        "Reset" => Some(GateType::Prep),
        _ => None,
    }
}

/// Maps PECOS `GateType` to HUGR operation name.
#[must_use]
pub fn gate_type_to_hugr_op(gate_type: GateType) -> Option<&'static str> {
    match gate_type {
        // Single-qubit gates
        GateType::H => Some("H"),
        GateType::X => Some("X"),
        GateType::Y => Some("Y"),
        GateType::Z => Some("Z"),
        GateType::SZ => Some("S"),
        GateType::SZdg => Some("Sdg"),
        GateType::T => Some("T"),
        GateType::Tdg => Some("Tdg"),
        GateType::SX => Some("V"),
        GateType::SXdg => Some("Vdg"),
        GateType::RX => Some("Rx"),
        GateType::RY => Some("Ry"),
        GateType::RZ => Some("Rz"),
        // Two-qubit gates
        GateType::CX => Some("CX"),
        GateType::CY => Some("CY"),
        GateType::CZ => Some("CZ"),
        GateType::CH => Some("CH"),
        GateType::SZZ => Some("ZZMax"),
        GateType::SWAP => Some("SWAP"),
        GateType::CRZ => Some("CRz"),
        // Three-qubit gates
        GateType::CCX => Some("Toffoli"),
        // Lifecycle operations
        GateType::QAlloc => Some("QAlloc"),
        GateType::QFree => Some("QFree"),
        GateType::Measure => Some("Measure"),
        GateType::MeasureFree => Some("MeasureFree"),
        GateType::Prep => Some("Reset"),
        // Unsupported
        _ => None,
    }
}

/// Check if an operation name is a quantum operation we care about.
#[must_use]
pub fn is_quantum_operation(op_name: &str) -> bool {
    hugr_op_to_gate_type(op_name).is_some()
}

/// Maps PECOS `GateType` to tket `TketOp`.
fn gate_type_to_tket_op(gate_type: GateType) -> Option<TketOp> {
    match gate_type {
        GateType::H => Some(TketOp::H),
        GateType::X => Some(TketOp::X),
        GateType::Y => Some(TketOp::Y),
        GateType::Z => Some(TketOp::Z),
        GateType::SZ => Some(TketOp::S),
        GateType::SZdg => Some(TketOp::Sdg),
        GateType::T => Some(TketOp::T),
        GateType::Tdg => Some(TketOp::Tdg),
        GateType::RX => Some(TketOp::Rx),
        GateType::RY => Some(TketOp::Ry),
        GateType::RZ => Some(TketOp::Rz),
        GateType::CX => Some(TketOp::CX),
        GateType::CY => Some(TketOp::CY),
        GateType::CZ => Some(TketOp::CZ),
        GateType::QAlloc => Some(TketOp::QAlloc),
        GateType::QFree => Some(TketOp::QFree),
        GateType::Measure => Some(TketOp::Measure),
        GateType::MeasureFree => Some(TketOp::MeasureFree),
        GateType::Prep => Some(TketOp::Reset),
        _ => None,
    }
}

/// Information about a quantum operation extracted from HUGR.
#[derive(Debug)]
struct QuantumOp {
    /// The HUGR node.
    node: Node,
    /// The original HUGR operation name.
    hugr_op_name: String,
    /// The PECOS gate type.
    gate_type: GateType,
    /// Number of qubit input ports.
    num_qubit_inputs: usize,
    /// Number of qubit output ports.
    num_qubit_outputs: usize,
    /// Extracted rotation parameters (if any).
    params: Vec<f64>,
}

/// Check if a gate type is a rotation gate that takes angle parameters.
#[must_use]
pub fn is_rotation_gate(gate_type: GateType) -> bool {
    matches!(
        gate_type,
        GateType::RX | GateType::RY | GateType::RZ | GateType::CRZ
    )
}

/// Try to extract a constant numeric value from a HUGR Const node.
/// Handles various formats: Const(Tuple(1.0)), Const(4), `ConstInt`, etc.
#[allow(clippy::too_many_lines)]
fn try_extract_const_value(hugr: &Hugr, node: Node) -> Option<f64> {
    let op = hugr.get_optype(node);

    if let OpType::Const(const_op) = op {
        // Use debug representation to extract the value
        let debug_str = format!("{const_op:?}");

        log::trace!(
            "try_extract_const_value: {}",
            &debug_str[..debug_str.len().min(200)]
        );

        // Pattern 1: Const(Tuple(number))
        if let Some(start) = debug_str.find("Tuple(") {
            let rest = &debug_str[start + 6..];
            if let Some(end) = rest.find(')')
                && let Ok(val) = rest[..end].parse::<f64>()
            {
                log::trace!("  -> Tuple pattern matched: {val}");
                return Some(val);
            }
        }

        // Pattern 2: ConstInt { log_width: N, value: V } - integer constant
        if let Some(start) = debug_str.find("ConstInt {") {
            // Find the value field
            if let Some(val_start) = debug_str[start..].find("value:") {
                let rest = &debug_str[start + val_start + 6..];
                // Extract the number (may be followed by comma, space, or brace)
                let mut num_str = String::new();
                for c in rest.trim().chars() {
                    if c.is_ascii_digit() || c == '-' {
                        num_str.push(c);
                    } else if !num_str.is_empty() {
                        break;
                    }
                }
                if !num_str.is_empty()
                    && let Ok(val) = num_str.parse::<i64>()
                {
                    log::trace!("  -> ConstInt pattern matched: {val}");
                    #[allow(clippy::cast_precision_loss)]
                    return Some(val as f64);
                }
            }
        }

        // Pattern 3: F64(number) or float64 value in Extension
        if let Some(start) = debug_str.find("F64(") {
            let rest = &debug_str[start + 4..];
            if let Some(end) = rest.find(')')
                && let Ok(val) = rest[..end].parse::<f64>()
            {
                log::trace!("  -> F64 pattern matched: {val}");
                return Some(val);
            }
        }

        // Pattern 4: ConstF64 { value: V }
        if let Some(start) = debug_str.find("ConstF64 {")
            && let Some(val_start) = debug_str[start..].find("value:")
        {
            let rest = &debug_str[start + val_start + 6..];
            let mut num_str = String::new();
            let mut in_number = false;
            for c in rest.trim().chars() {
                if c.is_ascii_digit() || c == '.' || c == '-' {
                    num_str.push(c);
                    in_number = true;
                } else if in_number && (c == 'e' || c == 'E' || c == '+') {
                    num_str.push(c);
                } else if in_number {
                    break;
                }
            }
            if !num_str.is_empty()
                && let Ok(val) = num_str.parse::<f64>()
            {
                log::trace!("  -> ConstF64 pattern matched: {val}");
                return Some(val);
            }
        }

        // Pattern 5: Look for any float-like number after "value:" or in Sum/Extension
        // Search for patterns like: value: 1.0 or values: [... 1.0 ...]
        for pattern in ["value:", "values:"] {
            if let Some(start) = debug_str.find(pattern) {
                let rest = &debug_str[start + pattern.len()..];
                // Look for a number
                let mut num_str = String::new();
                let mut in_number = false;
                for c in rest.chars() {
                    if c.is_ascii_digit() || c == '.' || c == '-' {
                        num_str.push(c);
                        in_number = true;
                    } else if in_number && (c == 'e' || c == 'E' || c == '+') {
                        num_str.push(c);
                    } else if in_number && num_str.contains('.') {
                        // Only break if we have a decimal (to avoid matching just integers)
                        break;
                    } else if in_number {
                        // Got an integer, check if it's followed by a decimal
                        break;
                    }
                }
                if !num_str.is_empty()
                    && num_str.contains('.')
                    && let Ok(val) = num_str.parse::<f64>()
                {
                    log::trace!("  -> value pattern matched: {val}");
                    return Some(val);
                }
            }
        }

        // Pattern 6: Fallback - search for any reasonable looking float in the entire string
        // This is a last resort for complex nested structures
        let mut best_float: Option<f64> = None;
        let mut i = 0;
        let chars: Vec<char> = debug_str.chars().collect();
        while i < chars.len() {
            if chars[i].is_ascii_digit() || chars[i] == '-' {
                let mut num_str = String::new();
                while i < chars.len() {
                    let c = chars[i];
                    if c.is_ascii_digit() || c == '.' || c == '-' {
                        num_str.push(c);
                        i += 1;
                    } else if (c == 'e' || c == 'E') && i + 1 < chars.len() {
                        num_str.push(c);
                        i += 1;
                        if i < chars.len() && (chars[i] == '+' || chars[i] == '-') {
                            num_str.push(chars[i]);
                            i += 1;
                        }
                    } else {
                        break;
                    }
                }
                if num_str.contains('.')
                    && let Ok(val) = num_str.parse::<f64>()
                {
                    log::trace!("  -> fallback float pattern: {val}");
                    best_float = Some(val);
                    break; // Take the first valid float
                }
            } else {
                i += 1;
            }
        }
        if best_float.is_some() {
            return best_float;
        }

        log::trace!("  -> no pattern matched");
    }

    None
}

/// Recursively trace back through a node to find a constant float value.
/// Returns (value, `is_half_turns`) where `is_half_turns` indicates if we passed through `from_halfturns`.
#[allow(clippy::too_many_lines)]
fn trace_back_for_const(hugr: &Hugr, node: Node, depth: usize) -> Option<(f64, bool)> {
    if depth > 20 {
        log::trace!(
            "{}trace_back_for_const: max depth reached",
            "  ".repeat(depth)
        );
        return None; // Prevent infinite recursion
    }

    let op = hugr.get_optype(node);
    let op_name = format!("{op:?}");
    let op_short = if op_name.len() > 80 {
        format!("{}...", &op_name[..80])
    } else {
        op_name.clone()
    };
    log::trace!(
        "{}trace_back_for_const: node={:?}, op={}",
        "  ".repeat(depth),
        node,
        op_short
    );

    // If it's a Const, extract the value directly
    if matches!(op, OpType::Const(_))
        && let Some(val) = try_extract_const_value(hugr, node)
    {
        return Some((val, false));
    }

    // If it's LoadConstant, look at its input (which should be a Const)
    if matches!(op, OpType::LoadConstant(_)) {
        let const_port = IncomingPort::from(0);
        if let Some((const_node, _)) = hugr.single_linked_output(node, const_port) {
            return trace_back_for_const(hugr, const_node, depth + 1);
        }
    }

    // Check if it's from_halfturns_unchecked extension op
    if let Some(ext_op) = op.as_extension_op() {
        let op_name = ext_op.unqualified_id().to_string();
        if op_name == "from_halfturns_unchecked" {
            // The input is a float in half-turns
            let float_port = IncomingPort::from(0);
            if let Some((src_node, _)) = hugr.single_linked_output(node, float_port)
                && let Some((val, _)) = trace_back_for_const(hugr, src_node, depth + 1)
            {
                // Mark that we found half-turns
                return Some((val, true));
            }
        }

        // Handle float division (fdiv) - used for computing rotation angles
        if op_name == "fdiv" {
            // fdiv has two inputs: numerator (port 0) and denominator (port 1)
            let num_port = IncomingPort::from(0);
            let denom_port = IncomingPort::from(1);

            if let (Some((num_node, _)), Some((denom_node, _))) = (
                hugr.single_linked_output(node, num_port),
                hugr.single_linked_output(node, denom_port),
            ) && let (Some((num_val, _)), Some((denom_val, _))) = (
                trace_back_for_const(hugr, num_node, depth + 1),
                trace_back_for_const(hugr, denom_node, depth + 1),
            ) && denom_val != 0.0
            {
                return Some((num_val / denom_val, false));
            }
        }

        // Handle integer-to-float conversion (convert_s - signed int to float)
        if op_name == "convert_s" || op_name == "convert_u" {
            let input_port = IncomingPort::from(0);
            if let Some((src_node, _)) = hugr.single_linked_output(node, input_port) {
                return trace_back_for_const(hugr, src_node, depth + 1);
            }
        }

        // Handle float negation (fneg) - used for negative rotation angles
        if op_name == "fneg" {
            let input_port = IncomingPort::from(0);
            if let Some((src_node, _)) = hugr.single_linked_output(node, input_port)
                && let Some((val, is_half_turns)) = trace_back_for_const(hugr, src_node, depth + 1)
            {
                return Some((-val, is_half_turns));
            }
        }
    }

    // For UnpackTuple, trace through
    if let OpType::Tag(_) = op {
        // Skip Tag nodes
    } else if format!("{op:?}").contains("UnpackTuple") {
        let input_port = IncomingPort::from(0);
        if let Some((src_node, _)) = hugr.single_linked_output(node, input_port) {
            return trace_back_for_const(hugr, src_node, depth + 1);
        }
    }

    // For Call nodes, try to evaluate if it's an arithmetic operation
    if matches!(op, OpType::Call(_)) {
        // Scan inputs to find:
        // 1. The function being called (FuncDefn) to check the operation type
        // 2. Numeric argument values
        let mut is_division = false;
        let mut is_multiplication = false;
        let mut is_negation = false;
        let mut numeric_values: Vec<(usize, f64, bool)> = Vec::new();

        // Get the number of input ports for this node
        let num_inputs = hugr.num_inputs(node);

        // Check all input ports
        for port_idx in 0..num_inputs {
            let in_port = IncomingPort::from(port_idx);
            if let Some((src_node, _)) = hugr.single_linked_output(node, in_port) {
                let src_op = hugr.get_optype(src_node);

                // Check if this is a FuncDefn to get the function name
                if let OpType::FuncDefn(func_defn) = src_op {
                    let func_name = format!("{func_defn:?}");
                    if func_name.contains("truediv") || func_name.contains("__div__") {
                        is_division = true;
                    }
                    if func_name.contains("__mul__") || func_name.contains("__rmul__") {
                        is_multiplication = true;
                    }
                    if func_name.contains("__neg__") {
                        is_negation = true;
                    }
                }

                // Try to get a numeric value from this input
                if let Some((val, is_half_turns)) = trace_back_for_const(hugr, src_node, depth + 1)
                {
                    numeric_values.push((port_idx, val, is_half_turns));
                }
            }
        }

        // If this is a negation call and we have a numeric value, negate it
        if is_negation && !numeric_values.is_empty() {
            let (_, val, is_half_turns) = numeric_values[0];
            return Some((-val, is_half_turns));
        }

        // If this is a division call and we have two numeric values, compute the result
        if is_division && numeric_values.len() >= 2 {
            // Sort by port index to get correct order (numerator first, denominator second)
            numeric_values.sort_by_key(|(idx, _, _)| *idx);
            let numerator = numeric_values[0].1;
            let denominator = numeric_values[1].1;
            if denominator != 0.0 {
                return Some((numerator / denominator, false));
            }
        }

        // If this is a multiplication call and we have two numeric values, compute the result
        if is_multiplication && numeric_values.len() >= 2 {
            numeric_values.sort_by_key(|(idx, _, _)| *idx);
            let factor1 = numeric_values[0].1;
            let factor2 = numeric_values[1].1;
            return Some((factor1 * factor2, false));
        }

        // For other calls, try to return the first numeric value found
        if let Some((_, val, _)) = numeric_values.first() {
            return Some((*val, false));
        }
    }

    None
}

/// Try to extract rotation angle from a rotation gate's input.
///
/// In HUGR, rotation gates receive their angle as a dataflow input.
/// The tket extension uses half-turns for rotation angles (via `ConstRotation`).
///
/// Returns the angle in **full turns** (compatible with PECOS's `Angle64::from_turns()`).
/// - 0.25 turns = pi/2 radians (quarter turn)
/// - 0.5 turns = pi radians (half turn)
/// - 1.0 turns = 2*pi radians (full turn)
///
/// # Arguments
/// * `hugr` - The HUGR graph
/// * `gate_node` - The rotation gate node
/// * `num_qubit_inputs` - Number of qubit inputs (angle is after these)
pub fn try_extract_rotation_angle(
    hugr: &Hugr,
    gate_node: Node,
    num_qubit_inputs: usize,
) -> Option<f64> {
    // For rotation gates, the angle input is typically after the qubit inputs
    // RZ gate: port 0 = qubit, port 1 = angle (rotation type)
    let angle_port = IncomingPort::from(num_qubit_inputs);

    // Find what's connected to the angle input port
    if let Some((src_node, _src_port)) = hugr.single_linked_output(gate_node, angle_port) {
        // Trace back through the graph to find the value
        if let Some((value, is_half_turns)) = trace_back_for_const(hugr, src_node, 0) {
            if is_half_turns {
                // Value explicitly marked as half-turns, convert to full turns
                // half_turns * 0.5 = full_turns
                return Some(value * 0.5);
            }
            // In tket HUGRs, rotation angles are stored in half-turns via ConstRotation.
            // The value extracted from trace_back_for_const is the raw half-turns value.
            // Convert to full turns: half_turns * 0.5 = full_turns
            return Some(value * 0.5);
        }
    }

    None
}

/// Extract quantum operations from a HUGR.
fn extract_quantum_ops(hugr: &Hugr) -> Vec<QuantumOp> {
    let mut operations = Vec::new();

    for node in hugr.nodes() {
        let op = hugr.get_optype(node);

        // Check if this is an extension operation
        let Some(ext_op) = op.as_extension_op() else {
            continue;
        };

        // Check if it's from the tket.quantum extension
        let ext_id = ext_op.extension_id();
        if ext_id.as_ref() as &str != "tket.quantum" {
            continue;
        }

        let op_name = ext_op.unqualified_id().to_string();

        let Some(gate_type) = hugr_op_to_gate_type(&op_name) else {
            continue;
        };

        // Determine number of qubit inputs/outputs based on gate type
        // Use quantum_arity() for most gates, with special cases for alloc/free
        let (num_qubit_inputs, num_qubit_outputs) = match gate_type {
            // QAlloc: 0 inputs, 1 output (creates a qubit)
            GateType::QAlloc => (0, 1),
            // QFree/MeasureFree: 1 input, 0 qubit outputs (destroys/consumes qubit)
            GateType::QFree | GateType::MeasureFree => (1, 0),
            // All other gates: use quantum_arity for input/output counts
            _ => {
                let arity = gate_type.quantum_arity();
                (arity, arity)
            }
        };

        // Extract rotation parameters if this is a rotation gate
        let params = if is_rotation_gate(gate_type) {
            if let Some(angle) = try_extract_rotation_angle(hugr, node, num_qubit_inputs) {
                vec![angle]
            } else {
                vec![]
            }
        } else {
            vec![]
        };

        operations.push(QuantumOp {
            node,
            hugr_op_name: op_name.clone(),
            gate_type,
            num_qubit_inputs,
            num_qubit_outputs,
            params,
        });
    }

    operations
}

/// Key for tracking qubit wire flow: (node, `output_port_index`)
type WireKey = (Node, usize);

/// Convert a HUGR quantum circuit to a `DagCircuit`.
///
/// # Arguments
///
/// * `hugr` - The HUGR to convert.
///
/// # Returns
///
/// A `DagCircuit` representing the quantum circuit.
///
/// # Errors
///
/// Returns an error if the HUGR contains unsupported structures or unknown operations.
///
/// # Algorithm
///
/// 1. Extract all quantum operations from the HUGR
/// 2. Process operations in topological order (`QAlloc` nodes first)
/// 3. Track qubit identity through wire connections:
///    - `QAlloc` creates a new qubit with a unique ID
///    - Other gates look up their input wires to find qubit IDs
///    - Output wires carry the same qubit IDs (maintaining linear flow)
/// 4. Build the `DagCircuit` with properly identified qubits
#[allow(clippy::too_many_lines)]
pub fn hugr_to_dag_circuit(hugr: &Hugr) -> Result<DagCircuit, HugrConvertError> {
    let operations = extract_quantum_ops(hugr);

    if operations.is_empty() {
        return Ok(DagCircuit::new());
    }

    // Build mapping from HUGR node to operation info
    let node_to_op: BTreeMap<Node, &QuantumOp> =
        operations.iter().map(|op| (op.node, op)).collect();

    // Track which qubit ID flows through each wire (node, output_port) -> qubit_id
    let mut wire_to_qubit: BTreeMap<WireKey, QubitId> = BTreeMap::new();

    // Counter for assigning new qubit IDs
    let mut next_qubit_id: usize = 0;

    // Create the DagCircuit
    let mut dag = DagCircuit::with_capacity(operations.len(), operations.len() * 2);

    // Set circuit-level metadata
    dag.set_attr("source", Attribute::String("hugr".to_string()));

    // Map from HUGR node to DagCircuit node index
    let mut node_to_dag_idx: BTreeMap<Node, usize> = BTreeMap::new();

    // Process operations - we need topological order, but HUGR nodes() doesn't guarantee it.
    // Instead, we'll process by following the wire dependencies.
    // First, find all QAlloc nodes (they have no qubit inputs, so process first)
    let mut processed: std::collections::HashSet<Node> = std::collections::HashSet::new();
    let mut work_queue: Vec<Node> = operations
        .iter()
        .filter(|op| op.gate_type == GateType::QAlloc)
        .map(|op| op.node)
        .collect();

    // Also add any operations that might not have qubit predecessors from quantum ops.
    // This includes:
    // - Operations with 0 qubit inputs
    // - Operations whose qubit inputs come from non-quantum sources (e.g., DFG input node)
    for op in &operations {
        if !work_queue.contains(&op.node) {
            // Check if all qubit predecessors are already ready (not quantum ops or already processed)
            let all_preds_ready = check_predecessors_ready(hugr, op.node, &processed);
            if all_preds_ready {
                work_queue.push(op.node);
            }
        }
    }

    while !work_queue.is_empty() {
        let current_node = work_queue.remove(0);

        if processed.contains(&current_node) {
            continue;
        }

        let Some(op) = node_to_op.get(&current_node) else {
            continue;
        };

        // Determine the qubits this gate acts on
        let qubits: Vec<QubitId> = if op.gate_type == GateType::QAlloc {
            // QAlloc creates a new qubit
            let qubit_id = QubitId::from(next_qubit_id);
            next_qubit_id += 1;

            // Record the qubit flowing out of this node's output port 0
            wire_to_qubit.insert((current_node, 0), qubit_id);

            vec![qubit_id]
        } else {
            // For other gates, look up qubits from input wires
            let mut qubits = Vec::with_capacity(op.num_qubit_inputs);

            for port_idx in 0..op.num_qubit_inputs {
                let in_port = IncomingPort::from(port_idx);

                // Find what's connected to this input port
                if let Some((src_node, src_port)) = hugr.single_linked_output(current_node, in_port)
                {
                    let src_port_idx = src_port.index();
                    let wire_key = (src_node, src_port_idx);

                    if let Some(&qubit_id) = wire_to_qubit.get(&wire_key) {
                        qubits.push(qubit_id);

                        // If this gate has corresponding output (not a terminal op),
                        // propagate the qubit to the output port
                        if port_idx < op.num_qubit_outputs {
                            wire_to_qubit.insert((current_node, port_idx), qubit_id);
                        }
                    } else {
                        // Source qubit not yet known - this shouldn't happen in topological order
                        // but handle gracefully with a fallback
                        let fallback_qubit = QubitId::from(next_qubit_id);
                        next_qubit_id += 1;
                        qubits.push(fallback_qubit);
                        if port_idx < op.num_qubit_outputs {
                            wire_to_qubit.insert((current_node, port_idx), fallback_qubit);
                        }
                    }
                } else {
                    // No linked output found - use fallback
                    let fallback_qubit = QubitId::from(next_qubit_id);
                    next_qubit_id += 1;
                    qubits.push(fallback_qubit);
                }
            }

            qubits
        };

        // Create the gate with proper qubits and extracted angles
        // Convert extracted rotation angles (in full turns) to Angle64
        let angles: Vec<Angle64> = op.params.iter().map(|&p| Angle64::from_turns(p)).collect();
        let gate = Gate::with_angles(op.gate_type, angles, qubits.clone());
        let dag_node_idx = dag.add_gate(gate);
        node_to_dag_idx.insert(current_node, dag_node_idx);

        // Set gate-level metadata from HUGR
        dag.set_gate_attr(
            dag_node_idx,
            "hugr_node",
            Attribute::Int(i64::try_from(current_node.index()).unwrap_or(i64::MAX)),
        );
        dag.set_gate_attr(
            dag_node_idx,
            "hugr_op",
            Attribute::String(op.hugr_op_name.clone()),
        );

        processed.insert(current_node);

        // Find successors (nodes connected to this node's outputs) and add to queue
        // output_neighbours returns nodes that receive data from this node's outputs
        for succ_node in hugr.output_neighbours(current_node) {
            if node_to_op.contains_key(&succ_node) && !processed.contains(&succ_node) {
                // Check if all qubit predecessors are processed
                let all_preds_ready = check_predecessors_ready(hugr, succ_node, &processed);
                if all_preds_ready && !work_queue.contains(&succ_node) {
                    work_queue.push(succ_node);
                }
            }
        }
    }

    // Add edges based on qubit wire connections
    for op in &operations {
        if !node_to_dag_idx.contains_key(&op.node) {
            continue;
        }
        let target_dag_idx = node_to_dag_idx[&op.node];

        for port_idx in 0..op.num_qubit_inputs {
            let in_port = IncomingPort::from(port_idx);

            if let Some((src_node, src_port)) = hugr.single_linked_output(op.node, in_port)
                && let Some(&source_dag_idx) = node_to_dag_idx.get(&src_node)
            {
                let src_port_idx = src_port.index();
                let wire_key = (src_node, src_port_idx);

                if let Some(&qubit_id) = wire_to_qubit.get(&wire_key) {
                    // Add edge connecting these gates on this qubit
                    let _ = dag.connect(source_dag_idx, target_dag_idx, qubit_id);
                }
            }
        }
    }

    Ok(dag)
}

/// Check if all qubit predecessors of a node have been processed.
fn check_predecessors_ready(
    hugr: &Hugr,
    node: Node,
    processed: &std::collections::HashSet<Node>,
) -> bool {
    // Check all input neighbours (nodes that provide data to this node's inputs)
    for pred_node in hugr.input_neighbours(node) {
        // Check if this predecessor is a quantum operation we care about
        let op = hugr.get_optype(pred_node);
        if let Some(ext_op) = op.as_extension_op() {
            let ext_id = ext_op.extension_id();
            if ext_id.as_ref() as &str == "tket.quantum" && !processed.contains(&pred_node) {
                return false;
            }
        }
    }
    true
}

/// Convert a `DagCircuit` back to HUGR.
///
/// # Arguments
///
/// * `dag` - The `DagCircuit` to convert.
///
/// # Returns
///
/// A HUGR representing the quantum circuit.
///
/// # Errors
///
/// Returns an error if:
/// - The circuit contains unsupported gate types
/// - HUGR construction fails
#[allow(clippy::too_many_lines)]
pub fn dag_circuit_to_hugr(dag: &DagCircuit) -> Result<Hugr, HugrConvertError> {
    // Get all qubits used in the circuit
    let qubits = dag.qubits();
    let num_qubits = qubits.len();

    if num_qubits == 0 {
        // Empty circuit - create minimal HUGR
        let builder = DFGBuilder::new(Signature::new(vec![], vec![])).map_err(|e| {
            HugrConvertError::UnsupportedStructure(format!("Failed to create builder: {e}"))
        })?;
        return builder.finish_hugr_with_outputs(vec![]).map_err(|e| {
            HugrConvertError::UnsupportedStructure(format!("Failed to finish HUGR: {e}"))
        });
    }

    // Create qubit type rows for input/output
    let qb_row: Vec<_> = (0..num_qubits).map(|_| qb_t()).collect();
    let signature = Signature::new(qb_row.clone(), qb_row);

    // Create the DFG builder
    let mut builder = DFGBuilder::new(signature).map_err(|e| {
        HugrConvertError::UnsupportedStructure(format!("Failed to create builder: {e}"))
    })?;

    // Get input wires - these represent the initial qubit wires
    let input_wires: Vec<Wire> = builder.input_wires().collect();

    // Map from QubitId to current wire for that qubit
    let mut qubit_wires: BTreeMap<QubitId, Wire> = qubits
        .iter()
        .enumerate()
        .map(|(i, &q)| (q, input_wires[i]))
        .collect();

    // Process gates in topological order
    for node_idx in dag.topological_order() {
        let Some(gate) = dag.gate(node_idx) else {
            continue;
        };

        let Some(tket_op) = gate_type_to_tket_op(gate.gate_type) else {
            return Err(HugrConvertError::UnknownOperation(format!(
                "Unsupported gate type: {:?}",
                gate.gate_type
            )));
        };

        // Collect input wires for this gate
        let gate_input_wires: Vec<Wire> = gate
            .qubits
            .iter()
            .map(|q| {
                qubit_wires.get(q).copied().ok_or_else(|| {
                    HugrConvertError::UnsupportedStructure(format!("Unknown qubit: {q:?}"))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Handle rotation gates - they need an angle parameter
        let output_wires: Vec<Wire> = if is_rotation_gate(gate.gate_type) {
            // Get the angle in half-turns (HUGR uses half-turns, not full turns)
            // Angle::to_radians() returns radians; convert to turns by dividing by TAU
            let angle_turns = gate
                .angles
                .first()
                .map_or(0.0, |a| a.to_radians() / std::f64::consts::TAU);
            let half_turns = angle_turns * 2.0;

            // Create a rotation constant
            let const_rotation = ConstRotation::new(half_turns).map_err(|e| {
                HugrConvertError::UnsupportedStructure(format!("Invalid rotation: {e}"))
            })?;

            // Load the constant
            let rotation_wire = builder.add_load_value(const_rotation);

            // Add the rotation gate with qubit + rotation inputs
            let mut inputs = gate_input_wires;
            inputs.push(rotation_wire);

            builder
                .add_dataflow_op(tket_op, inputs)
                .map_err(|e| {
                    HugrConvertError::UnsupportedStructure(format!("Failed to add gate: {e}"))
                })?
                .outputs()
                .collect()
        } else {
            // Non-rotation gate
            builder
                .add_dataflow_op(tket_op, gate_input_wires)
                .map_err(|e| {
                    HugrConvertError::UnsupportedStructure(format!("Failed to add gate: {e}"))
                })?
                .outputs()
                .collect()
        };

        // Update qubit wire mappings based on gate type
        match gate.gate_type {
            GateType::QAlloc => {
                // QAlloc produces a new qubit
                if let Some(&q) = gate.qubits.first()
                    && let Some(&wire) = output_wires.first()
                {
                    qubit_wires.insert(q, wire);
                }
            }
            GateType::QFree => {
                // QFree consumes a qubit, no output
                if let Some(&q) = gate.qubits.first() {
                    qubit_wires.remove(&q);
                }
            }
            GateType::Measure => {
                // Measure outputs qubit + classical bit
                if let Some(&q) = gate.qubits.first()
                    && let Some(&wire) = output_wires.first()
                {
                    qubit_wires.insert(q, wire);
                }
                // Note: We ignore the classical output for now
            }
            GateType::MeasureFree => {
                // MeasureFree consumes qubit, outputs classical
                if let Some(&q) = gate.qubits.first() {
                    qubit_wires.remove(&q);
                }
            }
            _ => {
                // Regular gates: output wires correspond to input qubits
                for (i, &q) in gate.qubits.iter().enumerate() {
                    if let Some(&wire) = output_wires.get(i) {
                        qubit_wires.insert(q, wire);
                    }
                }
            }
        }
    }

    // Collect final output wires in qubit order
    let output_wires: Vec<Wire> = qubits
        .iter()
        .filter_map(|q| qubit_wires.get(q).copied())
        .collect();

    // Finish the HUGR
    builder
        .finish_hugr_with_outputs(output_wires)
        .map_err(|e| HugrConvertError::UnsupportedStructure(format!("Failed to finish HUGR: {e}")))
}

// ==================== SimpleHugr ====================

/// Information about a gate stored in `SimpleHugr` for fast access.
#[derive(Debug, Clone)]
struct SimpleGate {
    /// The Gate representation.
    gate: Gate,
    /// Predecessor gate indices.
    predecessors: Vec<usize>,
    /// Successor gate indices.
    successors: Vec<usize>,
}

/// A validated wrapper around HUGR that provides a `QuantumCircuit`-like interface.
///
/// `SimpleHugr` validates on construction that the HUGR represents a simple quantum
/// circuit without:
/// - Control flow (conditionals, CFG nodes)
/// - Loops (`TailLoop` nodes)
/// - Other complex structures
///
/// Once validated, it provides efficient access to the quantum operations through
/// the [`Circuit`] trait.
///
/// # Example
///
/// ```
/// use pecos_quantum::DagCircuit;
/// use pecos_quantum::hugr_convert::{SimpleHugr, dag_circuit_to_hugr};
/// use pecos_quantum::Circuit;
///
/// // Create a simple Bell state circuit
/// let mut dag = DagCircuit::new();
/// dag.h(0).cx(0, 1);
///
/// // Convert to HUGR
/// let hugr = dag_circuit_to_hugr(&dag).unwrap();
///
/// // Create SimpleHugr for efficient access
/// let simple = SimpleHugr::try_new(hugr).unwrap();
/// assert_eq!(simple.gate_count(), 2);  // H and CX
/// ```
#[derive(Debug, Clone)]
pub struct SimpleHugr {
    /// The underlying HUGR.
    hugr: Hugr,
    /// Extracted gates with cached structure.
    gates: Vec<SimpleGate>,
    /// All qubits used in the circuit.
    qubits: Vec<QubitId>,
    /// Gates in topological order.
    topo_order: Vec<usize>,
    /// Root gates (no predecessors).
    roots: Vec<usize>,
    /// Leaf gates (no successors).
    leaves: Vec<usize>,
    /// Mapping from qubit to gates acting on it.
    qubit_to_gates: BTreeMap<QubitId, Vec<usize>>,
    /// Circuit-level attributes.
    circuit_attrs: BTreeMap<String, Attribute>,
    /// Gate-level attributes, indexed by gate index.
    gate_attrs: Vec<BTreeMap<String, Attribute>>,
    /// Circuit depth.
    depth: usize,
}

/// Error returned when a HUGR cannot be converted to a `SimpleHugr`.
#[derive(Debug, Clone)]
pub enum NotSimpleError {
    /// The HUGR contains a Conditional node.
    ContainsConditional,
    /// The HUGR contains a `TailLoop` node.
    ContainsLoop,
    /// The HUGR contains a CFG (control flow graph) node.
    ContainsCFG,
    /// The HUGR contains a Case node outside of expected context.
    ContainsCase,
    /// Other unsupported structure.
    UnsupportedStructure(String),
}

impl std::fmt::Display for NotSimpleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NotSimpleError::ContainsConditional => write!(f, "HUGR contains Conditional node"),
            NotSimpleError::ContainsLoop => write!(f, "HUGR contains TailLoop node"),
            NotSimpleError::ContainsCFG => write!(f, "HUGR contains CFG node"),
            NotSimpleError::ContainsCase => write!(f, "HUGR contains Case node"),
            NotSimpleError::UnsupportedStructure(msg) => {
                write!(f, "HUGR contains unsupported structure: {msg}")
            }
        }
    }
}

impl std::error::Error for NotSimpleError {}

impl SimpleHugr {
    /// Creates a new `SimpleHugr` from a HUGR after validating it is a simple quantum circuit.
    ///
    /// # Errors
    ///
    /// Returns [`NotSimpleError`] if the HUGR contains:
    /// - Conditional nodes
    /// - `TailLoop` nodes
    /// - CFG nodes
    /// - Other unsupported structures
    #[allow(clippy::too_many_lines)]
    pub fn try_new(hugr: Hugr) -> Result<Self, NotSimpleError> {
        // Validate that the HUGR is a simple quantum circuit
        Self::validate_simple(&hugr)?;

        Ok(Self::build_from_hugr(hugr))
    }

    /// Creates a new `SimpleHugr` from a HUGR without strict validation.
    ///
    /// This is useful for HUGRs generated by compilers like Guppy that may wrap
    /// quantum circuits in CFG/function structures. The quantum operations are
    /// extracted regardless of the control flow structure.
    ///
    /// # Warning
    ///
    /// This function does not validate that the HUGR represents a simple circuit.
    /// If the HUGR contains actual control flow (conditionals, loops), the behavior
    /// when executing is undefined - measurements may not work correctly with
    /// symbolic simulation.
    #[must_use]
    pub fn new_relaxed(hugr: Hugr) -> Self {
        Self::build_from_hugr(hugr)
    }

    /// Internal method to build `SimpleHugr` from a HUGR.
    #[allow(clippy::too_many_lines)]
    fn build_from_hugr(hugr: Hugr) -> Self {
        // Extract quantum operations
        let quantum_ops = extract_quantum_ops(&hugr);

        if quantum_ops.is_empty() {
            return Self {
                hugr,
                gates: Vec::new(),
                qubits: Vec::new(),
                topo_order: Vec::new(),
                roots: Vec::new(),
                leaves: Vec::new(),
                qubit_to_gates: BTreeMap::new(),
                circuit_attrs: BTreeMap::new(),
                gate_attrs: Vec::new(),
                depth: 0,
            };
        }

        // Build node -> index mapping for the quantum ops
        let node_to_idx: BTreeMap<Node, usize> = quantum_ops
            .iter()
            .enumerate()
            .map(|(i, op)| (op.node, i))
            .collect();

        // Track qubit flow through wires (same logic as hugr_to_dag_circuit)
        let mut wire_to_qubit: BTreeMap<(Node, usize), QubitId> = BTreeMap::new();
        let mut next_qubit_id: usize = 0;

        // Build gates with qubit information
        let mut gates = Vec::with_capacity(quantum_ops.len());
        let mut qubit_set: BTreeSet<QubitId> = BTreeSet::new();
        let mut qubit_to_gates: BTreeMap<QubitId, Vec<usize>> = BTreeMap::new();

        // Process in topological order
        // First, find all QAlloc nodes (they have no qubit inputs, so process first)
        let mut processed: std::collections::HashSet<Node> = std::collections::HashSet::new();
        let mut work_queue: Vec<Node> = quantum_ops
            .iter()
            .filter(|op| op.gate_type == GateType::QAlloc)
            .map(|op| op.node)
            .collect();

        // Also add any operations whose qubit inputs come from non-quantum sources
        // (e.g., DFG input node). This includes gates that are "roots" in the circuit.
        for op in &quantum_ops {
            if !work_queue.contains(&op.node) {
                let all_preds_ready = check_predecessors_ready(&hugr, op.node, &processed);
                if all_preds_ready {
                    work_queue.push(op.node);
                }
            }
        }

        let mut topo_order = Vec::with_capacity(quantum_ops.len());

        while !work_queue.is_empty() {
            let current_node = work_queue.remove(0);

            if processed.contains(&current_node) {
                continue;
            }

            let Some(&op_idx) = node_to_idx.get(&current_node) else {
                continue;
            };
            let op = &quantum_ops[op_idx];

            // Determine qubits for this gate
            let qubits: Vec<QubitId> = if op.gate_type == GateType::QAlloc {
                let qubit_id = QubitId::from(next_qubit_id);
                next_qubit_id += 1;
                wire_to_qubit.insert((current_node, 0), qubit_id);
                vec![qubit_id]
            } else {
                let mut gate_qubits = Vec::with_capacity(op.num_qubit_inputs);
                for port_idx in 0..op.num_qubit_inputs {
                    let in_port = IncomingPort::from(port_idx);
                    if let Some((src_node, src_port)) =
                        hugr.single_linked_output(current_node, in_port)
                    {
                        let wire_key = (src_node, src_port.index());
                        if let Some(&qubit_id) = wire_to_qubit.get(&wire_key) {
                            gate_qubits.push(qubit_id);
                            if port_idx < op.num_qubit_outputs {
                                wire_to_qubit.insert((current_node, port_idx), qubit_id);
                            }
                        } else {
                            let fallback = QubitId::from(next_qubit_id);
                            next_qubit_id += 1;
                            gate_qubits.push(fallback);
                            if port_idx < op.num_qubit_outputs {
                                wire_to_qubit.insert((current_node, port_idx), fallback);
                            }
                        }
                    } else {
                        let fallback = QubitId::from(next_qubit_id);
                        next_qubit_id += 1;
                        gate_qubits.push(fallback);
                    }
                }
                gate_qubits
            };

            // Track qubits
            for &q in &qubits {
                qubit_set.insert(q);
                qubit_to_gates.entry(q).or_default().push(op_idx);
            }

            // Create the gate
            let angles: Vec<Angle64> = op.params.iter().map(|&p| Angle64::from_turns(p)).collect();
            let gate = Gate::with_angles(op.gate_type, angles, qubits);

            gates.push(SimpleGate {
                gate,
                predecessors: Vec::new(), // Will fill in later
                successors: Vec::new(),   // Will fill in later
            });

            topo_order.push(op_idx);
            processed.insert(current_node);

            // Add successors to work queue
            for succ_node in hugr.output_neighbours(current_node) {
                if node_to_idx.contains_key(&succ_node) && !processed.contains(&succ_node) {
                    let all_preds_ready = check_predecessors_ready(&hugr, succ_node, &processed);
                    if all_preds_ready && !work_queue.contains(&succ_node) {
                        work_queue.push(succ_node);
                    }
                }
            }
        }

        // Build predecessor/successor relationships
        for (idx, op) in quantum_ops.iter().enumerate() {
            // Find predecessors (quantum ops connected to our inputs)
            for port_idx in 0..op.num_qubit_inputs {
                let in_port = IncomingPort::from(port_idx);
                if let Some((src_node, _)) = hugr.single_linked_output(op.node, in_port)
                    && let Some(&pred_idx) = node_to_idx.get(&src_node)
                {
                    if !gates[idx].predecessors.contains(&pred_idx) {
                        gates[idx].predecessors.push(pred_idx);
                    }
                    if !gates[pred_idx].successors.contains(&idx) {
                        gates[pred_idx].successors.push(idx);
                    }
                }
            }
        }

        // Find roots and leaves
        let roots: Vec<usize> = gates
            .iter()
            .enumerate()
            .filter(|(_, g)| g.predecessors.is_empty())
            .map(|(i, _)| i)
            .collect();

        let leaves: Vec<usize> = gates
            .iter()
            .enumerate()
            .filter(|(_, g)| g.successors.is_empty())
            .map(|(i, _)| i)
            .collect();

        // Calculate depth
        let depth = Self::calculate_depth(&gates, &roots);

        // Create gate attributes
        let gate_attrs: Vec<BTreeMap<String, Attribute>> = quantum_ops
            .iter()
            .map(|op| {
                let mut attrs = BTreeMap::new();
                attrs.insert(
                    "hugr_node".to_string(),
                    Attribute::Int(i64::try_from(op.node.index()).unwrap_or(i64::MAX)),
                );
                attrs.insert(
                    "hugr_op".to_string(),
                    Attribute::String(op.hugr_op_name.clone()),
                );
                attrs
            })
            .collect();

        // Circuit-level attributes
        let mut circuit_attrs = BTreeMap::new();
        circuit_attrs.insert("source".to_string(), Attribute::String("hugr".to_string()));

        Self {
            hugr,
            gates,
            qubits: qubit_set.into_iter().collect(),
            topo_order,
            roots,
            leaves,
            qubit_to_gates,
            circuit_attrs,
            gate_attrs,
            depth,
        }
    }

    /// Validates that the HUGR is a simple quantum circuit.
    fn validate_simple(hugr: &Hugr) -> Result<(), NotSimpleError> {
        for node in hugr.nodes() {
            let op = hugr.get_optype(node);
            match op {
                OpType::Conditional(_) => return Err(NotSimpleError::ContainsConditional),
                OpType::TailLoop(_) => return Err(NotSimpleError::ContainsLoop),
                OpType::CFG(_) => return Err(NotSimpleError::ContainsCFG),
                OpType::Case(_) => return Err(NotSimpleError::ContainsCase),
                _ => {}
            }
        }
        Ok(())
    }

    /// Calculate the depth of the circuit.
    fn calculate_depth(gates: &[SimpleGate], roots: &[usize]) -> usize {
        if gates.is_empty() {
            return 0;
        }

        let mut depths = vec![0usize; gates.len()];
        let mut max_depth = 0;

        // BFS from roots
        let mut queue: Vec<usize> = roots.to_vec();
        for &root in roots {
            depths[root] = 1;
        }

        while let Some(idx) = queue.pop() {
            let current_depth = depths[idx];
            max_depth = max_depth.max(current_depth);

            for &succ in &gates[idx].successors {
                let new_depth = current_depth + 1;
                if new_depth > depths[succ] {
                    depths[succ] = new_depth;
                    queue.push(succ);
                }
            }
        }

        max_depth
    }

    /// Returns a reference to the underlying HUGR.
    #[must_use]
    pub fn as_hugr(&self) -> &Hugr {
        &self.hugr
    }

    /// Consumes the `SimpleHugr` and returns the underlying HUGR.
    #[must_use]
    pub fn into_hugr(self) -> Hugr {
        self.hugr
    }

    /// Returns the number of wires in the circuit.
    ///
    /// For `SimpleHugr`, this is estimated based on the number of qubit connections.
    fn wire_count_internal(&self) -> usize {
        self.gates.iter().map(|g| g.predecessors.len()).sum()
    }
}

impl Circuit for SimpleHugr {
    fn gate_count(&self) -> usize {
        self.gates.len()
    }

    fn wire_count(&self) -> usize {
        self.wire_count_internal()
    }

    fn qubits(&self) -> Vec<QubitId> {
        self.qubits.clone()
    }

    fn depth(&self) -> usize {
        self.depth
    }

    fn gate(&self, index: GateHandle) -> Option<&Gate> {
        self.gates.get(index).map(|g| &g.gate)
    }

    fn nodes(&self) -> Vec<GateHandle> {
        (0..self.gates.len()).collect()
    }

    fn iter_gates(&self) -> Box<dyn Iterator<Item = GateView<'_>> + '_> {
        Box::new(self.gates.iter().enumerate().map(|(i, g)| GateView {
            gate: &g.gate,
            index: i,
        }))
    }

    fn topological_order(&self) -> Vec<GateHandle> {
        self.topo_order.clone()
    }

    fn iter_gates_topo(&self) -> Box<dyn Iterator<Item = GateView<'_>> + '_> {
        Box::new(self.topo_order.iter().map(|&i| GateView {
            gate: &self.gates[i].gate,
            index: i,
        }))
    }

    fn predecessors(&self, gate: GateHandle) -> Vec<GateHandle> {
        self.gates
            .get(gate)
            .map(|g| g.predecessors.clone())
            .unwrap_or_default()
    }

    fn successors(&self, gate: GateHandle) -> Vec<GateHandle> {
        self.gates
            .get(gate)
            .map(|g| g.successors.clone())
            .unwrap_or_default()
    }

    fn roots(&self) -> Vec<GateHandle> {
        self.roots.clone()
    }

    fn leaves(&self) -> Vec<GateHandle> {
        self.leaves.clone()
    }

    fn gates_on_qubit(&self, qubit: QubitId) -> Vec<GateHandle> {
        self.qubit_to_gates.get(&qubit).cloned().unwrap_or_default()
    }

    fn qubit_timeline(&self, qubit: QubitId) -> Vec<GateHandle> {
        // gates_on_qubit already follows topological order from how we built it
        self.gates_on_qubit(qubit)
    }

    fn circuit_attrs(&self) -> &BTreeMap<String, Attribute> {
        &self.circuit_attrs
    }

    fn gate_attrs(&self, gate: GateHandle) -> Option<&BTreeMap<String, Attribute>> {
        self.gate_attrs.get(gate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hugr_op_to_gate_type() {
        assert_eq!(hugr_op_to_gate_type("H"), Some(GateType::H));
        assert_eq!(hugr_op_to_gate_type("CX"), Some(GateType::CX));
        assert_eq!(hugr_op_to_gate_type("QAlloc"), Some(GateType::QAlloc));
        assert_eq!(hugr_op_to_gate_type("QFree"), Some(GateType::QFree));
        assert_eq!(hugr_op_to_gate_type("Measure"), Some(GateType::Measure));
        assert_eq!(
            hugr_op_to_gate_type("MeasureFree"),
            Some(GateType::MeasureFree)
        );
        assert_eq!(hugr_op_to_gate_type("Reset"), Some(GateType::Prep));
        assert_eq!(hugr_op_to_gate_type("Unknown"), None);
    }

    #[test]
    fn test_gate_type_to_hugr_op() {
        assert_eq!(gate_type_to_hugr_op(GateType::H), Some("H"));
        assert_eq!(gate_type_to_hugr_op(GateType::CX), Some("CX"));
        assert_eq!(gate_type_to_hugr_op(GateType::QAlloc), Some("QAlloc"));
        assert_eq!(gate_type_to_hugr_op(GateType::QFree), Some("QFree"));
        assert_eq!(gate_type_to_hugr_op(GateType::Measure), Some("Measure"));
        assert_eq!(
            gate_type_to_hugr_op(GateType::MeasureFree),
            Some("MeasureFree")
        );
        assert_eq!(gate_type_to_hugr_op(GateType::Prep), Some("Reset"));
    }

    #[test]
    fn test_is_quantum_operation() {
        assert!(is_quantum_operation("H"));
        assert!(is_quantum_operation("CX"));
        assert!(is_quantum_operation("QAlloc"));
        assert!(!is_quantum_operation("Add"));
        assert!(!is_quantum_operation("Unknown"));
    }

    #[test]
    fn test_round_trip_gate_types() {
        // Test that gate types that can be converted to HUGR can be converted back
        let gate_types = [
            GateType::H,
            GateType::X,
            GateType::Y,
            GateType::Z,
            GateType::CX,
            GateType::QAlloc,
            GateType::QFree,
            GateType::Measure,
            GateType::MeasureFree,
            GateType::Prep,
        ];

        for gt in gate_types {
            let hugr_op = gate_type_to_hugr_op(gt);
            assert!(hugr_op.is_some(), "GateType {gt:?} should map to HUGR op");
            let back = hugr_op_to_gate_type(hugr_op.unwrap());
            assert_eq!(back, Some(gt), "Round trip failed for {gt:?}");
        }
    }

    #[test]
    fn test_dag_circuit_to_hugr_empty() {
        // Test conversion of empty DagCircuit
        let dag = DagCircuit::new();
        let hugr = dag_circuit_to_hugr(&dag).expect("Failed to convert empty DagCircuit to HUGR");

        // Verify the HUGR is valid
        assert!(
            hugr.nodes().count() > 0,
            "HUGR should have at least module/function nodes"
        );
    }

    #[test]
    fn test_dag_circuit_to_hugr_single_qubit() {
        // Test conversion of a simple single-qubit circuit: H gate
        let mut dag = DagCircuit::new();
        let q0 = QubitId::from(0);
        dag.add_gate(Gate::with_angles(GateType::H, vec![], vec![q0]));

        let hugr = dag_circuit_to_hugr(&dag).expect("Failed to convert DagCircuit to HUGR");
        assert!(hugr.nodes().count() > 0, "HUGR should have nodes");
    }

    #[test]
    fn test_dag_circuit_to_hugr_bell_state() {
        // Test conversion of a Bell state circuit: H on q0, CX on q0,q1
        let mut dag = DagCircuit::new();
        let q0 = QubitId::from(0);
        let q1 = QubitId::from(1);

        dag.add_gate(Gate::with_angles(GateType::H, vec![], vec![q0]));
        dag.add_gate(Gate::with_angles(GateType::CX, vec![], vec![q0, q1]));

        let hugr =
            dag_circuit_to_hugr(&dag).expect("Failed to convert Bell state DagCircuit to HUGR");
        assert!(hugr.nodes().count() > 0, "HUGR should have nodes");
    }

    #[test]
    fn test_round_trip_dag_circuit_simple() {
        // Test round-trip: DagCircuit -> Hugr -> DagCircuit
        let mut original = DagCircuit::new();
        let q0 = QubitId::from(0);
        let q1 = QubitId::from(1);

        // Build a simple circuit: H(q0), CX(q0,q1), X(q1)
        original.add_gate(Gate::with_angles(GateType::H, vec![], vec![q0]));
        original.add_gate(Gate::with_angles(GateType::CX, vec![], vec![q0, q1]));
        original.add_gate(Gate::with_angles(GateType::X, vec![], vec![q1]));

        // Convert to HUGR
        let hugr = dag_circuit_to_hugr(&original).expect("Failed to convert to HUGR");

        // Convert back to DagCircuit
        let recovered = hugr_to_dag_circuit(&hugr).expect("Failed to convert back to DagCircuit");

        // Verify gate count matches
        assert_eq!(
            original.gate_count(),
            recovered.gate_count(),
            "Gate count should match after round-trip: original={}, recovered={}",
            original.gate_count(),
            recovered.gate_count()
        );

        // Verify qubit count matches
        assert_eq!(
            original.qubits().len(),
            recovered.qubits().len(),
            "Qubit count should match after round-trip: original={}, recovered={}",
            original.qubits().len(),
            recovered.qubits().len()
        );

        // Verify gate types match (collect and compare)
        // Use the inherent iter_gates which returns (usize, &Gate)
        let original_types: Vec<GateType> = original
            .iter_gates()
            .map(|(_, gate)| gate.gate_type)
            .collect();
        let recovered_types: Vec<GateType> = recovered
            .iter_gates()
            .map(|(_, gate)| gate.gate_type)
            .collect();

        assert_eq!(
            original_types.len(),
            recovered_types.len(),
            "Should have same number of gate types"
        );

        // Check that all original gate types appear in recovered (order may differ)
        for gt in &original_types {
            assert!(
                recovered_types.contains(gt),
                "Gate type {gt:?} should appear in recovered circuit"
            );
        }

        // Verify qubit assignments are preserved
        for (_, orig_gate) in original.iter_gates() {
            // Find matching gate in recovered
            let found = recovered.iter_gates().any(|(_, rec_gate)| {
                rec_gate.gate_type == orig_gate.gate_type
                    && rec_gate.qubits.len() == orig_gate.qubits.len()
            });
            assert!(
                found,
                "Gate {:?} with {} qubits should have a match in recovered circuit",
                orig_gate.gate_type,
                orig_gate.qubits.len()
            );
        }
    }

    #[test]
    fn test_round_trip_with_rotation() {
        use pecos_core::Angle64;

        // Test round-trip with a rotation gate
        let mut original = DagCircuit::new();
        let q0 = QubitId::from(0);

        // Add RZ gate with pi/4 rotation (0.125 turns)
        let angle = Angle64::from_turns(0.125);
        original.add_gate(Gate::with_angles(GateType::RZ, vec![angle], vec![q0]));

        // Convert to HUGR and back
        let hugr = dag_circuit_to_hugr(&original).expect("Failed to convert to HUGR");
        let recovered = hugr_to_dag_circuit(&hugr).expect("Failed to convert back");

        assert_eq!(original.gate_count(), recovered.gate_count());

        // Check the rotation angle was preserved
        // Use the inherent iter_gates which returns (usize, &Gate)
        let (_, original_gate) = original.iter_gates().next().unwrap();
        let (_, recovered_gate) = recovered.iter_gates().next().unwrap();

        assert_eq!(original_gate.gate_type, recovered_gate.gate_type);
        assert_eq!(
            original_gate.angles.len(),
            recovered_gate.angles.len(),
            "Should have same number of angles"
        );

        if !original_gate.angles.is_empty() {
            let orig_radians = original_gate.angles[0].to_radians();
            let recov_radians = recovered_gate.angles[0].to_radians();
            assert!(
                (orig_radians - recov_radians).abs() < 1e-10,
                "Rotation angle should be preserved: orig={orig_radians}, recov={recov_radians}"
            );
        }
    }
}
