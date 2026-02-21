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

//! HUGR static analysis and extraction.
//!
//! This module contains functions for analyzing HUGR graphs before execution.
//! These are "preprocessing" functions that extract information from the HUGR
//! structure to guide execution.
//!
//! # Overview
//!
//! The analysis phase extracts:
//! - Control flow structures (Conditionals, CFGs, `TailLoops`, `FuncDefns`)
//! - Quantum operations and their metadata
//! - Classical operations (logic, arithmetic)
//! - Structural information (nodes inside containers, I/O nodes)
//!
//! This information is stored in lookup tables used during execution.

use std::collections::{BTreeMap, BTreeSet};

use log::debug;
use pecos_core::gate_type::GateType;
use pecos_quantum::hugr_convert::{
    hugr_op_to_gate_type, is_rotation_gate, try_extract_rotation_angle,
};
use tket::hugr::ops::OpType;
use tket::hugr::{Hugr, HugrView, Node};

use super::types::{
    CfgInfo, ClassicalOp, ClassicalOpType, ConditionalInfo, ContainerType, DataflowBlockInfo,
    FuncDefnInfo, QuantumOp, TailLoopInfo,
};

// ============================================================================
// Conditional extraction
// ============================================================================

/// Extract Conditional nodes from a HUGR for control flow support.
///
/// Conditionals are HUGR's branching construct. They have:
/// - A control input (Sum type) that selects the branch
/// - Multiple Case children, each containing a dataflow subgraph
/// - Passthrough inputs/outputs for qubits and values
pub fn extract_conditionals(hugr: &Hugr) -> BTreeMap<Node, ConditionalInfo> {
    let mut conditionals = BTreeMap::new();

    for node in hugr.nodes() {
        let op = hugr.get_optype(node);

        if let OpType::Conditional(_cond_op) = op {
            // Find Case children
            let cases: Vec<Node> = hugr.children(node).collect();

            // Count qubit inputs/outputs (simplified - may need refinement)
            // Conditionals pass through qubits, so count port connections
            let num_qubit_inputs = hugr.num_inputs(node).saturating_sub(1); // First input is the control
            let num_qubit_outputs = hugr.num_outputs(node);

            debug!(
                "Found Conditional node {:?} with {} cases, {} qubit inputs, {} qubit outputs",
                node,
                cases.len(),
                num_qubit_inputs,
                num_qubit_outputs
            );

            conditionals.insert(
                node,
                ConditionalInfo {
                    node,
                    cases,
                    num_qubit_inputs,
                    num_qubit_outputs,
                },
            );
        }
    }

    conditionals
}

/// Find all nodes that are inside Case nodes (descendants of Cases).
pub fn find_nodes_inside_cases(
    hugr: &Hugr,
    conditionals: &BTreeMap<Node, ConditionalInfo>,
) -> BTreeSet<Node> {
    let mut inside_cases = BTreeSet::new();

    for cond_info in conditionals.values() {
        for &case_node in &cond_info.cases {
            // Add all descendants of this Case node
            collect_descendants(hugr, case_node, &mut inside_cases);
        }
    }

    inside_cases
}

// ============================================================================
// CFG extraction
// ============================================================================

/// Extract all CFG nodes from the HUGR.
///
/// CFGs (Control Flow Graphs) contain `DataflowBlocks` connected by control edges.
/// Each block can branch to multiple successors based on a Sum output.
pub fn extract_cfgs(hugr: &Hugr) -> BTreeMap<Node, CfgInfo> {
    let mut cfgs = BTreeMap::new();

    for node in hugr.nodes() {
        let op = hugr.get_optype(node);

        if let OpType::CFG(cfg_op) = op {
            let mut entry_block = None;
            let mut exit_block = None;
            let mut blocks = BTreeMap::new();

            // Find all children (DataflowBlocks and ExitBlock)
            for child in hugr.children(node) {
                match hugr.get_optype(child) {
                    OpType::DataflowBlock(dfb) => {
                        let block_info = extract_dataflow_block_info(hugr, child, dfb);
                        // First DataflowBlock is the entry block
                        if entry_block.is_none() {
                            entry_block = Some(child);
                        }
                        blocks.insert(child, block_info);
                    }
                    OpType::ExitBlock(_) => {
                        exit_block = Some(child);
                    }
                    _ => {}
                }
            }

            if let (Some(entry), Some(exit)) = (entry_block, exit_block) {
                let num_inputs = cfg_op.signature.input().len();
                let num_outputs = cfg_op.signature.output().len();

                debug!(
                    "Found CFG node {:?} with {} blocks, entry {:?}, exit {:?}",
                    node,
                    blocks.len(),
                    entry,
                    exit
                );

                cfgs.insert(
                    node,
                    CfgInfo {
                        node,
                        entry_block: entry,
                        exit_block: exit,
                        blocks,
                        num_inputs,
                        num_outputs,
                    },
                );
            }
        }
    }

    cfgs
}

/// Extract information about a `DataflowBlock`.
pub fn extract_dataflow_block_info(
    hugr: &Hugr,
    node: Node,
    dfb: &tket::hugr::ops::DataflowBlock,
) -> DataflowBlockInfo {
    // Number of successors is determined by sum_rows
    let num_successors = dfb.sum_rows.len();
    let num_inputs = dfb.inputs.len();

    // Find Input and Output nodes inside this block
    let (input_node, output_node) = hugr
        .get_io(node)
        .map_or((None, None), |[i, o]| (Some(i), Some(o)));

    // Find successor blocks via control flow edges
    // Each DataflowBlock can have multiple successors based on Sum tag
    let successors = find_block_successors(hugr, node, num_successors);

    // Find all quantum operations inside this block
    let quantum_ops = find_quantum_ops_in_block(hugr, node);

    // Find all Call nodes inside this block
    let call_nodes = find_call_nodes_in_block(hugr, node);

    // Find all Conditional nodes inside this block
    let conditional_nodes = find_conditional_nodes_in_block(hugr, node);

    // Find all tket.bool operation nodes inside this block
    let bool_ops = find_bool_ops_in_block(hugr, node);

    // Find all classical operation nodes inside this block
    let classical_ops = find_classical_ops_in_block(hugr, node);

    // Find extension ops that aren't tracked elsewhere (e.g., tket.result)
    let extension_ops: BTreeSet<Node> = find_extension_ops_in_block(hugr, node)
        .into_iter()
        .filter(|op| {
            !quantum_ops.contains(op) && !bool_ops.contains(op) && !classical_ops.contains(op)
        })
        .collect();

    // Find TailLoop nodes inside this block
    let tailloop_nodes = find_tailloop_nodes_in_block(hugr, node);

    debug!(
        "DataflowBlock {:?}: {} inputs, {} successors, {} quantum ops, {} calls, {} conditionals, {} bool_ops, {} classical_ops, {} extension_ops, {} tailloops",
        node,
        num_inputs,
        num_successors,
        quantum_ops.len(),
        call_nodes.len(),
        conditional_nodes.len(),
        bool_ops.len(),
        classical_ops.len(),
        extension_ops.len(),
        tailloop_nodes.len()
    );

    DataflowBlockInfo {
        node,
        num_inputs,
        num_successors,
        successors,
        quantum_ops,
        call_nodes,
        conditional_nodes,
        bool_ops,
        classical_ops,
        extension_ops,
        tailloop_nodes,
        input_node,
        output_node,
    }
}

/// Find successor blocks for a `DataflowBlock`, ordered by output port.
///
/// This is critical for CFG branching: port 0 corresponds to branch index 0, etc.
/// The Sum type at the block's Output determines which branch is taken:
/// - Sum tag 0 -> successor at port 0
/// - Sum tag 1 -> successor at port 1
/// - etc.
pub fn find_block_successors(hugr: &Hugr, block: Node, num_successors: usize) -> Vec<Node> {
    use tket::hugr::OutgoingPort;

    let mut successors = vec![None; num_successors];

    // Iterate over each output port and find what CFG-related node it connects to
    for (port_idx, successor) in successors.iter_mut().enumerate() {
        let out_port = OutgoingPort::from(port_idx);
        // linked_inputs returns an iterator over (Node, IncomingPort) connected to this output
        for (target_node, _) in hugr.linked_inputs(block, out_port) {
            match hugr.get_optype(target_node) {
                OpType::DataflowBlock(_) | OpType::ExitBlock(_) => {
                    *successor = Some(target_node);
                    break; // Only expect one CFG successor per port
                }
                _ => {}
            }
        }
    }

    // Convert Option<Node> to Node, filtering out None entries
    successors.into_iter().flatten().collect()
}

/// Find all nodes inside CFG blocks (should be deferred until block is active).
pub fn find_nodes_inside_cfg_blocks(hugr: &Hugr, cfgs: &BTreeMap<Node, CfgInfo>) -> BTreeSet<Node> {
    let mut inside_blocks = BTreeSet::new();

    for cfg_info in cfgs.values() {
        for block_info in cfg_info.blocks.values() {
            // Add all descendants of this block
            collect_descendants(hugr, block_info.node, &mut inside_blocks);
        }
    }

    inside_blocks
}

// ============================================================================
// TailLoop extraction
// ============================================================================

/// Extract all `TailLoop` nodes from the HUGR.
///
/// `TailLoops` are HUGR's looping construct. They repeatedly execute their body
/// until the body outputs a "break" Sum variant.
pub fn extract_tailloops(hugr: &Hugr) -> BTreeMap<Node, TailLoopInfo> {
    let mut tailloops = BTreeMap::new();

    for node in hugr.nodes() {
        let op = hugr.get_optype(node);

        if let OpType::TailLoop(tailloop_op) = op {
            // Find Input and Output nodes inside the TailLoop body
            let (input_node, output_node) = hugr
                .get_io(node)
                .map_or((None, None), |[i, o]| (Some(i), Some(o)));

            let Some(input_node) = input_node else {
                debug!("TailLoop {node:?} has no Input node");
                continue;
            };
            let Some(output_node) = output_node else {
                debug!("TailLoop {node:?} has no Output node");
                continue;
            };

            // Calculate port counts from the TailLoop signature
            let just_inputs_count = tailloop_op.just_inputs.len();
            let just_outputs_count = tailloop_op.just_outputs.len();
            let rest_count = tailloop_op.rest.len();

            let num_inputs = just_inputs_count + rest_count;
            let num_outputs = just_outputs_count + rest_count;

            // Find operations inside the TailLoop
            let quantum_ops = find_quantum_ops_in_block(hugr, node);
            let call_nodes = find_call_nodes_in_block(hugr, node);
            let extension_ops: BTreeSet<Node> = find_extension_ops_in_block(hugr, node)
                .into_iter()
                .collect();
            let classical_ops = find_classical_ops_in_block(hugr, node);
            let bool_ops = find_bool_ops_in_block(hugr, node);
            let conditional_nodes = find_conditional_nodes_in_block(hugr, node);

            debug!(
                "Found TailLoop node {:?} with {} inputs, {} outputs, {} quantum ops, {} calls, {} extension ops, {} classical ops, {} bool ops, {} conditionals",
                node,
                num_inputs,
                num_outputs,
                quantum_ops.len(),
                call_nodes.len(),
                extension_ops.len(),
                classical_ops.len(),
                bool_ops.len(),
                conditional_nodes.len()
            );

            tailloops.insert(
                node,
                TailLoopInfo {
                    node,
                    input_node,
                    output_node,
                    just_inputs_count,
                    just_outputs_count,
                    rest_count,
                    quantum_ops,
                    call_nodes,
                    extension_ops,
                    classical_ops,
                    bool_ops,
                    conditional_nodes,
                    num_inputs,
                    num_outputs,
                },
            );
        }
    }

    tailloops
}

/// Find all nodes inside `TailLoop` bodies (should be deferred until loop is active).
pub fn find_nodes_inside_tailloops(
    hugr: &Hugr,
    tailloops: &BTreeMap<Node, TailLoopInfo>,
) -> BTreeSet<Node> {
    let mut inside_tailloops = BTreeSet::new();

    for tailloop_info in tailloops.values() {
        collect_descendants(hugr, tailloop_info.node, &mut inside_tailloops);
    }

    inside_tailloops
}

// ============================================================================
// FuncDefn extraction
// ============================================================================

/// Extract all `FuncDefn` nodes from the HUGR.
///
/// `FuncDefns` are function definitions that can be called via Call nodes.
/// They contain a body dataflow graph and may have nested CFGs.
pub fn extract_func_defns(hugr: &Hugr) -> BTreeMap<Node, FuncDefnInfo> {
    let mut func_defns = BTreeMap::new();

    for node in hugr.nodes() {
        let op = hugr.get_optype(node);

        if let OpType::FuncDefn(func_defn) = op {
            let name = func_defn.func_name().clone();

            // Find Input, Output, and CFG children
            let mut input_node = None;
            let mut output_node = None;
            let mut cfg_node = None;

            for child in hugr.children(node) {
                let child_op = hugr.get_optype(child);
                match child_op {
                    OpType::Input(_) => input_node = Some(child),
                    OpType::Output(_) => output_node = Some(child),
                    OpType::CFG(_) => cfg_node = Some(child),
                    _ => {}
                }
            }

            if let (Some(input_node), Some(output_node)) = (input_node, output_node) {
                let num_inputs = hugr.num_outputs(input_node);
                let num_outputs = hugr.num_inputs(output_node);

                debug!(
                    "Found FuncDefn {node:?} '{name}' with {num_inputs} inputs, {num_outputs} outputs, cfg={cfg_node:?}"
                );

                func_defns.insert(
                    node,
                    FuncDefnInfo {
                        node,
                        name,
                        input_node,
                        output_node,
                        cfg_node,
                        num_inputs,
                        num_outputs,
                    },
                );
            }
        }
    }

    func_defns
}

/// Extract all Call nodes and their target `FuncDefn`.
pub fn extract_call_targets(hugr: &Hugr) -> BTreeMap<Node, Node> {
    let mut call_targets = BTreeMap::new();

    for node in hugr.nodes() {
        let op = hugr.get_optype(node);

        if matches!(op, OpType::Call(_)) {
            // Find the FuncDefn connected to this Call's static port
            // The Call has a static edge from FuncDefn
            for pred in hugr.input_neighbours(node) {
                let pred_op = hugr.get_optype(pred);
                if matches!(pred_op, OpType::FuncDefn(_)) {
                    debug!("Found Call {node:?} targeting FuncDefn {pred:?}");
                    call_targets.insert(node, pred);
                    break;
                }
            }
        }
    }

    call_targets
}

/// Find all nodes inside `FuncDefn` bodies (except the entrypoint).
pub fn find_nodes_inside_func_defns(
    hugr: &Hugr,
    func_defns: &BTreeMap<Node, FuncDefnInfo>,
    call_targets: &BTreeMap<Node, Node>,
) -> BTreeSet<Node> {
    let mut inside_func_defns = BTreeSet::new();

    // Find which FuncDefns are called (not the entrypoint)
    let called_func_defns: BTreeSet<Node> = call_targets.values().copied().collect();

    for &func_defn_node in func_defns.keys() {
        // Only defer nodes inside FuncDefns that are called (not the entrypoint)
        if called_func_defns.contains(&func_defn_node) {
            collect_descendants(hugr, func_defn_node, &mut inside_func_defns);
        }
    }

    inside_func_defns
}

// ============================================================================
// Quantum operation extraction
// ============================================================================

/// Extract all quantum operations from a HUGR.
///
/// This identifies tket.quantum extension operations and extracts their
/// gate type, qubit arity, and rotation parameters.
pub fn extract_quantum_ops(hugr: &Hugr) -> BTreeMap<Node, QuantumOp> {
    let mut operations = BTreeMap::new();

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
            debug!("Unknown quantum operation: {op_name}");
            continue;
        };

        // Determine number of qubit inputs/outputs based on gate type
        // Use quantum_arity() for most gates to correctly handle CRZ, SWAP, CCX, etc.
        let (num_qubit_inputs, num_qubit_outputs) = match gate_type {
            GateType::QAlloc => (0, 1),
            GateType::QFree | GateType::MeasureFree => (1, 0),
            _ => {
                let arity = gate_type.quantum_arity();
                (arity, arity)
            }
        };

        // Extract rotation parameters for RX, RY, RZ gates
        // The angle is returned in full turns, we need radians
        let params = if is_rotation_gate(gate_type) {
            if let Some(angle_turns) = try_extract_rotation_angle(hugr, node, num_qubit_inputs) {
                // Convert from turns to radians: radians = turns * 2 * PI
                let angle_radians = angle_turns * std::f64::consts::TAU;
                debug!("Extracted rotation angle: {angle_turns} turns = {angle_radians} radians");
                vec![angle_radians]
            } else {
                debug!("Could not extract rotation angle for {gate_type:?}");
                vec![]
            }
        } else {
            vec![]
        };

        operations.insert(
            node,
            QuantumOp {
                node,
                gate_type,
                num_qubit_inputs,
                num_qubit_outputs,
                params,
            },
        );
    }

    operations
}

// ============================================================================
// Classical operation extraction
// ============================================================================

/// Extract classical operations from the HUGR (logic, arithmetic, etc.).
///
/// This identifies operations from extensions like:
/// - `logic`: And, Or, Not, Xor, Eq
/// - `arithmetic.int`: iadd, isub, imul, etc.
/// - `arithmetic.float`: fadd, fsub, fmul, etc.
/// - `arithmetic.conversions`: int/float conversions
/// - `prelude`: `MakeTuple`, `UnpackTuple`
#[allow(clippy::too_many_lines)]
pub fn extract_classical_ops(hugr: &Hugr) -> BTreeMap<Node, ClassicalOp> {
    let mut operations = BTreeMap::new();

    for node in hugr.nodes() {
        let op = hugr.get_optype(node);

        // Check if this is an extension operation
        let Some(ext_op) = op.as_extension_op() else {
            continue;
        };

        let ext_id = ext_op.extension_id();
        let ext_name = ext_id.as_ref() as &str;
        let op_name = ext_op.unqualified_id().to_string();

        // Map extension operations to ClassicalOpType
        let (op_type, num_inputs, num_outputs, int_info) = match ext_name {
            // Logic extension
            "logic" => match op_name.as_str() {
                "And" => (ClassicalOpType::And, 2, 1, None),
                "Or" => (ClassicalOpType::Or, 2, 1, None),
                "Not" => (ClassicalOpType::Not, 1, 1, None),
                "Xor" => (ClassicalOpType::Xor, 2, 1, None),
                "Eq" => (ClassicalOpType::Eq, 2, 1, None),
                _ => continue,
            },
            // Integer arithmetic extension
            "arithmetic.int" => {
                // Parse operation name to extract signedness info
                // Operations like "iadd", "isub" are signed; "iadd_u" are unsigned
                let is_signed = !op_name.ends_with("_u");
                match op_name.trim_end_matches("_u").trim_end_matches("_s") {
                    "iadd" => (ClassicalOpType::Iadd, 2, 1, Some((6, is_signed))), // default 64-bit
                    "isub" => (ClassicalOpType::Isub, 2, 1, Some((6, is_signed))),
                    "imul" => (ClassicalOpType::Imul, 2, 1, Some((6, is_signed))),
                    "idiv" | "idiv_checked" => (ClassicalOpType::Idiv, 2, 1, Some((6, is_signed))),
                    "imod" => (ClassicalOpType::Imod, 2, 1, Some((6, is_signed))),
                    "ineg" => (ClassicalOpType::Ineg, 1, 1, Some((6, true))),
                    "iabs" => (ClassicalOpType::Iabs, 1, 1, Some((6, is_signed))),
                    "ieq" => (ClassicalOpType::Ieq, 2, 1, Some((6, is_signed))),
                    "ine" => (ClassicalOpType::Ine, 2, 1, Some((6, is_signed))),
                    "ilt" => (ClassicalOpType::Ilt, 2, 1, Some((6, is_signed))),
                    "ile" => (ClassicalOpType::Ile, 2, 1, Some((6, is_signed))),
                    "igt" => (ClassicalOpType::Igt, 2, 1, Some((6, is_signed))),
                    "ige" => (ClassicalOpType::Ige, 2, 1, Some((6, is_signed))),
                    "iand" => (ClassicalOpType::Iand, 2, 1, Some((6, is_signed))),
                    "ior" => (ClassicalOpType::Ior, 2, 1, Some((6, is_signed))),
                    "ixor" => (ClassicalOpType::Ixor, 2, 1, Some((6, is_signed))),
                    "inot" => (ClassicalOpType::Inot, 1, 1, Some((6, is_signed))),
                    "ishl" => (ClassicalOpType::Ishl, 2, 1, Some((6, is_signed))),
                    "ishr" => (ClassicalOpType::Ishr, 2, 1, Some((6, is_signed))),
                    _ => continue,
                }
            }
            // Float arithmetic extension
            "arithmetic.float" => match op_name.as_str() {
                "fadd" => (ClassicalOpType::Fadd, 2, 1, None),
                "fsub" => (ClassicalOpType::Fsub, 2, 1, None),
                "fmul" => (ClassicalOpType::Fmul, 2, 1, None),
                "fdiv" => (ClassicalOpType::Fdiv, 2, 1, None),
                "fneg" => (ClassicalOpType::Fneg, 1, 1, None),
                "fabs" => (ClassicalOpType::Fabs, 1, 1, None),
                "ffloor" => (ClassicalOpType::Ffloor, 1, 1, None),
                "fceil" => (ClassicalOpType::Fceil, 1, 1, None),
                "feq" => (ClassicalOpType::Feq, 2, 1, None),
                "fne" => (ClassicalOpType::Fne, 2, 1, None),
                "flt" => (ClassicalOpType::Flt, 2, 1, None),
                "fle" => (ClassicalOpType::Fle, 2, 1, None),
                "fgt" => (ClassicalOpType::Fgt, 2, 1, None),
                "fge" => (ClassicalOpType::Fge, 2, 1, None),
                _ => continue,
            },
            // Conversion extension
            "arithmetic.conversions" => match op_name.as_str() {
                "convert_s" | "convert_u" => (ClassicalOpType::ConvertIntToFloat, 1, 1, None),
                "trunc_s" | "trunc_u" => (ClassicalOpType::ConvertFloatToInt, 1, 1, None),
                _ => continue,
            },
            // Prelude extension (tuples, etc.)
            "prelude" => {
                let num_inputs = hugr.num_inputs(node);
                let num_outputs = hugr.num_outputs(node);
                match op_name.as_str() {
                    "MakeTuple" => (ClassicalOpType::MakeTuple, num_inputs, 1, None),
                    "UnpackTuple" => (ClassicalOpType::UnpackTuple, 1, num_outputs, None),
                    _ => continue,
                }
            }
            _ => continue,
        };

        operations.insert(
            node,
            ClassicalOp {
                node,
                op_type,
                num_inputs,
                num_outputs,
                int_info,
                const_value: None,
            },
        );
    }

    operations
}

// ============================================================================
// Block-level analysis helpers
// ============================================================================

/// Find all quantum operations inside a CFG block.
pub fn find_quantum_ops_in_block(hugr: &Hugr, block: Node) -> BTreeSet<Node> {
    let mut ops = BTreeSet::new();
    collect_quantum_ops_recursive(hugr, block, &mut ops);
    ops
}

/// Recursively collect quantum operations in a subtree.
fn collect_quantum_ops_recursive(hugr: &Hugr, node: Node, ops: &mut BTreeSet<Node>) {
    for child in hugr.children(node) {
        let op = hugr.get_optype(child);

        // Check if this is a quantum extension operation
        if let Some(ext_op) = op.as_extension_op() {
            let ext_id = ext_op.extension_id();
            if ext_id.as_ref() as &str == "tket.quantum" {
                let op_name = ext_op.unqualified_id().to_string();
                if hugr_op_to_gate_type(&op_name).is_some() {
                    ops.insert(child);
                }
            }
        }
        // Recurse into nested containers
        collect_quantum_ops_recursive(hugr, child, ops);
    }
}

/// Find all Call nodes inside a CFG block.
pub fn find_call_nodes_in_block(hugr: &Hugr, block: Node) -> BTreeSet<Node> {
    let mut calls = BTreeSet::new();
    collect_call_nodes_recursive(hugr, block, &mut calls);
    calls
}

/// Recursively collect Call nodes in a subtree.
fn collect_call_nodes_recursive(hugr: &Hugr, node: Node, calls: &mut BTreeSet<Node>) {
    for child in hugr.children(node) {
        let op = hugr.get_optype(child);
        if matches!(op, OpType::Call(_)) {
            calls.insert(child);
        }
        // Recurse into nested containers (but not into FuncDefns)
        if !matches!(op, OpType::FuncDefn(_)) {
            collect_call_nodes_recursive(hugr, child, calls);
        }
    }
}

/// Find all Conditional nodes inside a CFG block.
pub fn find_conditional_nodes_in_block(hugr: &Hugr, block: Node) -> BTreeSet<Node> {
    let mut conditionals = BTreeSet::new();
    collect_conditional_nodes_recursive(hugr, block, &mut conditionals);
    conditionals
}

/// Recursively collect Conditional nodes in a subtree.
fn collect_conditional_nodes_recursive(hugr: &Hugr, node: Node, conditionals: &mut BTreeSet<Node>) {
    for child in hugr.children(node) {
        let op = hugr.get_optype(child);
        if matches!(op, OpType::Conditional(_)) {
            conditionals.insert(child);
        }
        // Recurse into nested containers (but not into FuncDefns or Conditionals)
        if !matches!(op, OpType::FuncDefn(_) | OpType::Conditional(_)) {
            collect_conditional_nodes_recursive(hugr, child, conditionals);
        }
    }
}

/// Find all `TailLoop` nodes inside a CFG block.
pub fn find_tailloop_nodes_in_block(hugr: &Hugr, block: Node) -> BTreeSet<Node> {
    let mut tailloops = BTreeSet::new();
    collect_tailloop_nodes_recursive(hugr, block, &mut tailloops);
    tailloops
}

/// Recursively collect `TailLoop` nodes in a subtree.
fn collect_tailloop_nodes_recursive(hugr: &Hugr, node: Node, tailloops: &mut BTreeSet<Node>) {
    for child in hugr.children(node) {
        let op = hugr.get_optype(child);
        if matches!(op, OpType::TailLoop(_)) {
            tailloops.insert(child);
        }
        // Recurse into nested containers (but not into FuncDefns, Conditionals, or TailLoops themselves)
        if !matches!(
            op,
            OpType::FuncDefn(_) | OpType::Conditional(_) | OpType::TailLoop(_)
        ) {
            collect_tailloop_nodes_recursive(hugr, child, tailloops);
        }
    }
}

/// Find all tket.bool operation nodes inside a CFG block.
pub fn find_bool_ops_in_block(hugr: &Hugr, block: Node) -> BTreeSet<Node> {
    let mut bool_ops = BTreeSet::new();
    collect_bool_ops_recursive(hugr, block, &mut bool_ops);
    bool_ops
}

/// Recursively collect tket.bool operation nodes in a subtree.
fn collect_bool_ops_recursive(hugr: &Hugr, node: Node, bool_ops: &mut BTreeSet<Node>) {
    for child in hugr.children(node) {
        let op = hugr.get_optype(child);
        if let Some(ext_op) = op.as_extension_op() {
            let ext_id = ext_op.extension_id();
            if ext_id.as_ref() as &str == "tket.bool" {
                bool_ops.insert(child);
            }
        }
        // Recurse into nested containers (but not into FuncDefns)
        if !matches!(op, OpType::FuncDefn(_)) {
            collect_bool_ops_recursive(hugr, child, bool_ops);
        }
    }
}

/// Find all classical operation nodes inside a CFG block.
/// Classical ops include arithmetic (int/float), logic, and conversions.
pub fn find_classical_ops_in_block(hugr: &Hugr, block: Node) -> BTreeSet<Node> {
    let mut classical_ops = BTreeSet::new();
    collect_classical_ops_recursive(hugr, block, &mut classical_ops);
    classical_ops
}

/// Recursively collect classical operation nodes in a subtree.
fn collect_classical_ops_recursive(hugr: &Hugr, node: Node, classical_ops: &mut BTreeSet<Node>) {
    for child in hugr.children(node) {
        let op = hugr.get_optype(child);
        if let Some(ext_op) = op.as_extension_op() {
            let ext_id = ext_op.extension_id();
            let ext_name = ext_id.as_ref() as &str;
            // Classical ops are from these extensions
            if matches!(
                ext_name,
                "logic" | "arithmetic.int" | "arithmetic.float" | "arithmetic.conversions"
            ) {
                classical_ops.insert(child);
            }
            // Also check prelude for MakeTuple/UnpackTuple
            if ext_name == "prelude" {
                let op_name = ext_op.unqualified_id().to_string();
                if op_name == "MakeTuple" || op_name == "UnpackTuple" {
                    classical_ops.insert(child);
                }
            }
        }
        // Recurse into nested containers (but not into FuncDefns or Conditionals)
        if !matches!(op, OpType::FuncDefn(_) | OpType::Conditional(_)) {
            collect_classical_ops_recursive(hugr, child, classical_ops);
        }
    }
}

/// Find all extension ops inside a CFG block (excluding tket.bool which is tracked separately).
pub fn find_extension_ops_in_block(hugr: &Hugr, block: Node) -> Vec<Node> {
    let mut extension_ops = Vec::new();
    collect_extension_ops_recursive(hugr, block, &mut extension_ops);
    extension_ops
}

fn collect_extension_ops_recursive(hugr: &Hugr, node: Node, extension_ops: &mut Vec<Node>) {
    for child in hugr.children(node) {
        let op = hugr.get_optype(child);
        if let Some(ext_op) = op.as_extension_op() {
            let ext_id = ext_op.extension_id();
            // Skip tket.bool as those are tracked separately
            if ext_id.as_ref() as &str != "tket.bool" {
                extension_ops.push(child);
            }
        }
        // Recurse into nested containers (but not into FuncDefns or Conditionals)
        if !matches!(op, OpType::FuncDefn(_) | OpType::Conditional(_)) {
            collect_extension_ops_recursive(hugr, child, extension_ops);
        }
    }
}

// ============================================================================
// Structural helpers
// ============================================================================

/// Recursively collect all descendants of a node.
pub fn collect_descendants(hugr: &Hugr, node: Node, descendants: &mut BTreeSet<Node>) {
    for child in hugr.children(node) {
        descendants.insert(child);
        collect_descendants(hugr, child, descendants);
    }
}

/// Get the Input and Output nodes for a dataflow container.
/// Uses HUGR's native `get_io()` method which handles different container types properly.
pub fn get_io_nodes(hugr: &Hugr, container: Node) -> Option<(Node, Node)> {
    hugr.get_io(container)
        .map(|[input, output]| (input, output))
}

/// Find the Input node inside a Case (or any dataflow container).
pub fn find_input_node(hugr: &Hugr, container: Node) -> Option<Node> {
    get_io_nodes(hugr, container).map(|(input, _)| input)
}

/// Find the Output node inside a Case (or any dataflow container).
pub fn find_output_node(hugr: &Hugr, container: Node) -> Option<Node> {
    get_io_nodes(hugr, container).map(|(_, output)| output)
}

/// Determine the container type for wire mapping purposes.
pub fn get_container_type(hugr: &Hugr, node: Node) -> ContainerType {
    let op = hugr.get_optype(node);
    match op {
        OpType::DFG(_) => ContainerType::Dfg,
        OpType::Case(_) => ContainerType::Case,
        OpType::Conditional(_) => ContainerType::Conditional,
        OpType::TailLoop(_) => ContainerType::TailLoop,
        OpType::FuncDefn(_) => ContainerType::FuncDefn,
        OpType::Call(_) => ContainerType::Call,
        OpType::CFG(_) => ContainerType::Cfg,
        OpType::DataflowBlock(_) => ContainerType::DataflowBlock,
        _ => ContainerType::Other,
    }
}

/// Check if all quantum predecessors of a node have been processed.
/// This includes quantum operations, Conditionals, CFGs, `TailLoops`, and Call nodes.
pub fn all_predecessors_ready(
    hugr: &Hugr,
    node: Node,
    quantum_ops: &BTreeMap<Node, QuantumOp>,
    conditionals: &BTreeMap<Node, ConditionalInfo>,
    cfgs: &BTreeMap<Node, CfgInfo>,
    processed: &BTreeSet<Node>,
) -> bool {
    for pred_node in hugr.input_neighbours(node) {
        // Check quantum ops
        if quantum_ops.contains_key(&pred_node) && !processed.contains(&pred_node) {
            return false;
        }
        // Check conditionals (they also produce qubit outputs)
        if conditionals.contains_key(&pred_node) && !processed.contains(&pred_node) {
            return false;
        }
        // Check CFG nodes (they also produce qubit outputs)
        if cfgs.contains_key(&pred_node) && !processed.contains(&pred_node) {
            return false;
        }
        // Check Call nodes and TailLoop nodes (they also produce qubit/array outputs)
        let op = hugr.get_optype(pred_node);
        if matches!(op, OpType::Call(_) | OpType::TailLoop(_)) && !processed.contains(&pred_node) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_collect_descendants_empty() {
        // This would require a mock Hugr, so we just test the function signature compiles
        // Real tests require integration with actual HUGR construction
    }
}
