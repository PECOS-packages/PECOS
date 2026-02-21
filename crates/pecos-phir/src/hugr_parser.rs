/*!
HUGR Parser - Direct to PHIR

This module parses HUGR format directly into PHIR structures using tket's hugr re-export,
leveraging PHIR's hierarchical structure to serve as both AST and IR.

Uses flat iteration approach inspired by pecos-hugr-qis to avoid stack overflow
issues with deeply nested structures.
*/

use crate::builtin_ops::FuncOp;
use crate::builtin_ops::ModuleOp;
use crate::error::{PhirError, Result};
use crate::ops::{Operation, QuantumOp};
use crate::phir::{Instruction, SSAValue, Terminator};
use crate::types::{FunctionType, Type};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[cfg(feature = "hugr")]
use tket::hugr::{Hugr, HugrView, Node, NodeIndex, ops::OpType};

/// Parse HUGR bytes directly into PHIR representation
///
/// This handles both JSON and HUGR Package envelope formats
///
/// # Errors
///
/// Returns an error if:
/// - Failed to parse HUGR format
/// - HUGR to PHIR conversion fails
pub fn parse_hugr_bytes_to_phir(hugr_bytes: &[u8]) -> Result<ModuleOp> {
    use tket::hugr::envelope::read_envelope;
    use tket::hugr::extension::{ExtensionRegistry, prelude};
    use tket::hugr::std_extensions::{
        arithmetic::{conversions, float_ops, float_types, int_ops, int_types},
        collections, logic, ptr,
    };
    use tket_qsystem::extension::{futures, gpu, qsystem, result, wasm};

    // Create extension registry with all required extensions including tket-specific ones
    // This matches what pecos-hugr-qis's REGISTRY contains
    let extensions = ExtensionRegistry::new([
        prelude::PRELUDE.clone(),
        int_types::EXTENSION.clone(),
        int_ops::EXTENSION.clone(),
        float_types::EXTENSION.clone(),
        float_ops::EXTENSION.clone(),
        conversions::EXTENSION.clone(),
        logic::EXTENSION.clone(),
        ptr::EXTENSION.clone(),
        collections::list::EXTENSION.clone(),
        collections::array::EXTENSION.clone(),
        collections::static_array::EXTENSION.clone(),
        collections::borrow_array::EXTENSION.clone(),
        futures::EXTENSION.clone(),
        result::EXTENSION.clone(),
        qsystem::EXTENSION.clone(),
        tket::extension::rotation::ROTATION_EXTENSION.clone(),
        tket::extension::TKET_EXTENSION.clone(),
        tket::extension::TKET1_EXTENSION.clone(),
        tket::extension::bool::BOOL_EXTENSION.clone(),
        tket::extension::debug::DEBUG_EXTENSION.clone(),
        gpu::EXTENSION.clone(),
        wasm::EXTENSION.clone(),
    ]);

    if hugr_bytes.is_empty() {
        return Err(PhirError::internal("Empty HUGR input".to_string()));
    }

    // Use read_envelope directly (same approach as pecos-hugr-qis and selene)
    let (_desc, package) = read_envelope(hugr_bytes, &extensions)
        .map_err(|e| PhirError::internal(format!("Failed to read HUGR: {e}")))?;

    let hugr = if let Some(module) = package.modules.first() {
        module.clone()
    } else {
        return Err(PhirError::internal(
            "Package contains no HUGR modules".to_string(),
        ));
    };

    // Convert HUGR to PHIR using flat approach
    Ok(convert_hugr_to_phir_flat(&hugr))
}

/// Parse HUGR string directly into PHIR representation
///
/// Supports HUGR Package envelope format, direct HUGR JSON, and simplified test format
///
/// # Errors
///
/// Returns an error if:
/// - Parsing fails
/// - HUGR format is invalid
/// - Conversion to PHIR fails
pub fn parse_hugr_to_phir(hugr_str: &str) -> Result<ModuleOp> {
    // Try to parse using the bytes parser which handles both envelope and JSON formats
    match parse_hugr_bytes_to_phir(hugr_str.as_bytes()) {
        Ok(module) => Ok(module),
        Err(_) => {
            // If that fails, try to parse as simplified test format
            parse_simplified_hugr_json(hugr_str)
        }
    }
}

/// Convert HUGR to PHIR using flat iteration
fn convert_hugr_to_phir_flat(hugr: &Hugr) -> ModuleOp {
    let mut phir_module = ModuleOp::new("main");

    // First pass: Find all function nodes
    let mut function_nodes = Vec::new();
    for node in hugr.nodes() {
        if let OpType::FuncDefn(func_defn) = hugr.get_optype(node) {
            function_nodes.push((node, func_defn));
        }
    }

    // Second pass: Convert each function
    for (func_node, func_defn) in function_nodes {
        let func = convert_function_flat(hugr, func_node, func_defn);
        phir_module.add_function(func);
    }

    phir_module
}

/// Convert a function using flat iteration
fn convert_function_flat(
    hugr: &Hugr,
    func_node: Node,
    _func_defn: &tket::hugr::ops::FuncDefn,
) -> FuncOp {
    // Name the first function "main" for PECOS compatibility
    let func_name = if func_node.index() == 1 {
        "main".to_string()
    } else {
        format!("func_{}", func_node.index())
    };
    // For now, use a default function type since we can't access the private signature field
    // TODO: Find a way to extract function signature from HUGR
    let func_type = FunctionType {
        inputs: vec![],
        outputs: vec![],
        variadic: false,
    };

    let mut func = FuncOp::new(func_name, func_type);

    // Find all nodes that belong to this function using BFS
    let function_nodes = find_function_nodes(hugr, func_node);

    // Extract operations and build SSA values
    let mut node_values: BTreeMap<Node, Vec<SSAValue>> = BTreeMap::new();
    let mut next_ssa_id = 0;
    let mut instructions = Vec::new();

    // Process nodes in topological order (HUGR should maintain this)
    // First pass: convert all nodes and store their output SSA values
    for node in &function_nodes {
        if let Some(instr) =
            convert_node_to_instruction_flat(hugr, *node, &node_values, &mut next_ssa_id)
        {
            // Store output values for this node before processing the instruction
            // This is important for nodes that reference earlier outputs
            let outputs = instr.results.clone();
            node_values.insert(*node, outputs);

            instructions.push(instr);
        }
    }

    // Build the function body - for now, just a single block
    let mut entry_block = crate::phir::Block::new(None);
    for instr in instructions {
        entry_block.operations.push(instr);
    }

    // Add basic terminator
    entry_block.terminator = Some(Terminator::Return { values: vec![] });

    // Replace the default entry block with our populated one
    // FuncOp::new() creates a function with one region containing one empty entry block
    if func.body.is_empty() {
        func.body.push(crate::phir::Region::new(
            crate::region_kinds::RegionKind::SSACFG,
        ));
        func.body[0].blocks.push(entry_block);
    } else if func.body[0].blocks.is_empty() {
        func.body[0].blocks.push(entry_block);
    } else {
        // Replace the default empty entry block with our populated one
        func.body[0].blocks[0] = entry_block;
    }

    func
}

/// Find all nodes belonging to a function using BFS
fn find_function_nodes(hugr: &Hugr, func_node: Node) -> Vec<Node> {
    let mut nodes = Vec::new();
    let mut visited = BTreeSet::new();
    let mut queue = VecDeque::new();

    // Start with function's children
    for child in hugr.children(func_node) {
        queue.push_back(child);
    }

    while let Some(node) = queue.pop_front() {
        if visited.contains(&node) {
            continue;
        }
        visited.insert(node);
        nodes.push(node);

        // Add children to queue
        for child in hugr.children(node) {
            if !visited.contains(&child) {
                queue.push_back(child);
            }
        }
    }

    nodes
}

/// Convert a single HUGR node to a PHIR instruction
fn convert_node_to_instruction_flat(
    hugr: &Hugr,
    node: Node,
    node_values: &BTreeMap<Node, Vec<SSAValue>>,
    next_ssa_id: &mut u32,
) -> Option<Instruction> {
    let op = hugr.get_optype(node);

    match op {
        OpType::Const(_const_op) => {
            // Handle constants
            // For now, skip - we'd need to extract the actual const value
            None
        }
        OpType::LoadConstant(_load) => {
            // Load constant operation
            // Creates an SSA value from a constant
            let result = SSAValue::new(*next_ssa_id);
            *next_ssa_id += 1;

            Some(Instruction::new(
                Operation::Classical(crate::ops::ClassicalOp::ConstInt(0)), // Placeholder
                vec![],
                vec![result],
                vec![Type::Int(crate::types::IntWidth::I64)],
            ))
        }
        OpType::DFG(_dfg) => {
            // DataFlow Graph node - usually container
            None
        }
        OpType::Input(_) | OpType::Output(_) => {
            // Function input/output nodes
            None
        }
        OpType::Call(_call) => {
            // Function call - would need to resolve the function name
            None
        }
        OpType::CallIndirect(_) => {
            // Indirect call
            None
        }
        OpType::LoadFunction(_) => {
            // Load function reference
            None
        }
        OpType::ExtensionOp(ext_op) => {
            // Extension operation - this is where quantum ops live
            convert_extension_op(ext_op, node, node_values, next_ssa_id, hugr)
        }
        OpType::OpaqueOp(_) => {
            // Opaque operations - similar to extension ops but without full type info
            None
        }
        OpType::CFG(_) | OpType::ExitBlock(_) | OpType::DataflowBlock(_) => {
            // Control flow nodes - handled separately
            None
        }
        OpType::Case(_) | OpType::Conditional(_) | OpType::TailLoop(_) => {
            // Branching/looping constructs
            None
        }
        OpType::Tag(_) => {
            // Data manipulation
            None
        }
        OpType::FuncDefn(_) | OpType::FuncDecl(_) | OpType::Module(_) => {
            // Module-level constructs - handled at higher level
            None
        }
        OpType::AliasDefn(_) | OpType::AliasDecl(_) => {
            // Type aliases
            None
        }
        _ => {
            // Other operations not yet handled
            None
        }
    }
}

/// Convert an extension operation to PHIR
fn convert_extension_op(
    ext_op: &tket::hugr::ops::custom::ExtensionOp,
    _node: Node,
    _node_values: &BTreeMap<Node, Vec<SSAValue>>,
    next_ssa_id: &mut u32,
    _hugr: &Hugr,
) -> Option<Instruction> {
    // Use debug format to extract operation info
    // This is a workaround since the ExtensionOp API isn't clear
    let op_string = format!("{ext_op:?}");

    // Generate operations based on patterns in the debug string
    if op_string.contains("QAlloc") {
        // Quantum allocation
        let result = SSAValue::new(*next_ssa_id);
        *next_ssa_id += 1;

        Some(Instruction::new(
            Operation::Quantum(QuantumOp::Alloc),
            vec![],
            vec![result],
            vec![Type::Qubit],
        ))
    } else if op_string.contains('H') && op_string.contains("quantum") {
        // Hadamard gate
        let qubit = SSAValue::new(0); // Placeholder input
        let result = SSAValue::new(*next_ssa_id);
        *next_ssa_id += 1;

        Some(Instruction::new(
            Operation::Quantum(QuantumOp::H),
            vec![qubit],
            vec![result],
            vec![Type::Qubit],
        ))
    } else if op_string.contains("CX") || op_string.contains("CNOT") {
        // CNOT gate
        let control = SSAValue::new(0);
        let target = SSAValue::new(1);
        let control_result = SSAValue::new(*next_ssa_id);
        *next_ssa_id += 1;
        let target_result = SSAValue::new(*next_ssa_id);
        *next_ssa_id += 1;

        Some(Instruction::new(
            Operation::Quantum(QuantumOp::CX),
            vec![control, target],
            vec![control_result, target_result],
            vec![Type::Qubit, Type::Qubit],
        ))
    } else if op_string.contains("Measure") {
        // Measurement
        let qubit = SSAValue::new(0);
        let result = SSAValue::new(*next_ssa_id);
        *next_ssa_id += 1;

        Some(Instruction::new(
            Operation::Quantum(QuantumOp::Measure),
            vec![qubit],
            vec![result],
            vec![Type::Bool],
        ))
    } else {
        // For now, skip unknown operations
        None
    }
}

#[allow(dead_code)]
fn convert_function_type(_sig: &tket::hugr::types::PolyFuncType) -> FunctionType {
    // Convert HUGR function signature to PHIR function type
    // This would need to properly extract input/output types
    FunctionType {
        inputs: vec![],
        outputs: vec![],
        variadic: false,
    }
}

/// Convert HUGR type to PHIR type
#[allow(dead_code)]
fn convert_hugr_type_to_phir(hugr_type: &tket::hugr::types::Type) -> Type {
    use tket::hugr::extension::prelude::{bool_t, qb_t};

    match hugr_type {
        t if t == &qb_t() => Type::Qubit,
        t if t == &bool_t() => Type::Bool,
        _ => Type::Unknown,
    }
}

/// Parse simplified HUGR JSON format (for testing)
fn parse_simplified_hugr_json(json: &str) -> Result<ModuleOp> {
    // Parse JSON into Value
    let value: Value = serde_json::from_str(json)
        .map_err(|e| PhirError::internal(format!("Invalid JSON: {e}")))?;

    // Create a simple module
    let mut module = ModuleOp::new("main");

    // Look for quantum operations in the JSON
    if let Some(ops) = value["operations"].as_array() {
        let mut func = FuncOp::new(
            "main",
            FunctionType {
                inputs: vec![],
                outputs: vec![],
                variadic: false,
            },
        );

        let mut block = crate::phir::Block::new(None);

        for (i, op) in ops.iter().enumerate() {
            if let Some(op_str) = op["op"].as_str() {
                let instr = match op_str {
                    "H" | "Hadamard" => {
                        let qubit = SSAValue::new(0);
                        Instruction::new(
                            Operation::Quantum(QuantumOp::H),
                            vec![qubit],
                            vec![qubit],
                            vec![Type::Qubit],
                        )
                    }
                    "CNOT" | "CX" => {
                        let control = SSAValue::new(0);
                        let target = SSAValue::new(1);
                        Instruction::new(
                            Operation::Quantum(QuantumOp::CX),
                            vec![control, target],
                            vec![control, target],
                            vec![Type::Qubit, Type::Qubit],
                        )
                    }
                    "Measure" => {
                        let qubit = SSAValue::new(0);
                        let result_id =
                            u32::try_from(i).expect("Operation index too large for u32") + 100;
                        let result = SSAValue::new(result_id);
                        Instruction::new(
                            Operation::Quantum(QuantumOp::Measure),
                            vec![qubit],
                            vec![result],
                            vec![Type::Bool],
                        )
                    }
                    _ => continue,
                };
                block.operations.push(instr);
            }
        }

        block.terminator = Some(Terminator::Return { values: vec![] });
        // Add the block to the function's body region
        if func.body.is_empty() {
            func.body.push(crate::phir::Region::new(
                crate::region_kinds::RegionKind::SSACFG,
            ));
        }
        func.body[0].blocks.push(block);
        module.add_function(func);
    }

    Ok(module)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hugr_parsing_placeholder() {
        // Placeholder test - real tests need valid HUGR data
        // This test exists to validate compilation
        let simple_json = r#"{"operations": []}"#;
        let result = parse_simplified_hugr_json(simple_json);
        assert!(result.is_ok());
    }

    #[test]
    fn test_simplified_json_parsing() {
        let json = r#"
        {
            "operations": [
                {"op": "H", "qubit": 0},
                {"op": "Measure", "qubit": 0}
            ]
        }
        "#;

        let module = parse_simplified_hugr_json(json).unwrap();
        // Should have created a module with one function
        assert_eq!(module.name, "main");
    }
}
