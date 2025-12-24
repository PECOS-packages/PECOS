/*!
PHIR to MLIR Lowering

This module converts PHIR (PECOS High-level IR) to MLIR text format.
The generated MLIR can be processed by MLIR tools (mlir-opt, mlir-translate)
to produce LLVM IR.

TODO: This is currently a stub implementation. Need to implement:
1. PHIR -> MLIR conversion
2. Proper MLIR dialect support
3. Quantum operation mapping
*/

use crate::{
    PhirConfig,
    error::{PhirError, Result},
    phir::Module,
};
use std::fmt;
use std::fmt::Write;

/// MLIR Module representation for text generation
pub struct MlirModule {
    /// Module name
    pub name: String,
    /// MLIR text content
    pub content: String,
}

impl fmt::Display for MlirModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.content)
    }
}

/// Convert PHIR Module to MLIR text
///
/// # Errors
///
/// Returns an error if the conversion fails
pub fn lower_phir_to_mlir(module: &Module, _config: &PhirConfig) -> Result<MlirModule> {
    let mut content = String::new();

    // Always use standard dialect - it will be converted to LLVM by mlir-opt
    writeln!(&mut content, "module @{} {{", module.name).unwrap();

    // Convert module body
    if let Some(block) = module.body.blocks.first() {
        for instruction in &block.operations {
            if let crate::ops::Operation::Builtin(crate::builtin_ops::BuiltinOp::Func(func)) =
                &instruction.operation
            {
                content.push_str(&convert_function_to_mlir(func)?);
                content.push('\n');
            }
        }
    }

    content.push('}');

    Ok(MlirModule {
        name: module.name.clone(),
        content,
    })
}

/// Convert a PHIR function to MLIR text
fn convert_function_to_mlir(func: &crate::builtin_ops::FuncOp) -> Result<String> {
    let mut output = String::new();

    // Add function declarations for QIR intrinsics - using i64 for PECOS compatibility
    output.push_str("  func private @__quantum__qis__h__body(i64)\n");
    output.push_str("  func private @__quantum__qis__cx__body(i64, i64)\n");
    output.push_str("  func private @__quantum__qis__m__body(i64, i64) -> i32\n");
    output.push_str("  func private @__quantum__rt__qubit_allocate() -> i64\n");
    output.push_str("  func private @__quantum__rt__result_allocate() -> i64\n");
    output.push_str("  func private @__quantum__rt__qubit_release(i64)\n");
    output.push('\n');

    // Function signature (using older MLIR syntax for compatibility)
    write!(&mut output, "  func @{}(", func.name).unwrap();

    // Input types - convert qubit types to i64 for PECOS compatibility
    let input_types: Vec<String> = func
        .function_type
        .inputs
        .iter()
        .map(|t| match t {
            crate::types::Type::Qubit => "i64".to_string(),
            _ => type_to_mlir(t),
        })
        .collect();
    output.push_str(&input_types.join(", "));

    output.push_str(") -> (");

    // Output types - convert bool to i32 for QIR compatibility
    let output_types: Vec<String> = func
        .function_type
        .outputs
        .iter()
        .map(|t| match t {
            crate::types::Type::Bool => "i32".to_string(),
            _ => type_to_mlir(t),
        })
        .collect();
    output.push_str(&output_types.join(", "));

    output.push_str(") {\n");

    // Function body
    if let Some(entry_region) = func.entry_region()
        && let Some(block) = entry_region.blocks.first()
    {
        // Track SSA value to qubit mapping
        let mut ssa_to_qubit: std::collections::BTreeMap<u32, u32> =
            std::collections::BTreeMap::new();

        // Convert instructions
        for instruction in &block.operations {
            output.push_str(&convert_instruction_to_mlir_with_mapping(
                instruction,
                &mut ssa_to_qubit,
            )?);
            output.push('\n');
        }

        // Convert terminator
        if let Some(terminator) = &block.terminator {
            output.push_str(&convert_terminator_to_mlir(terminator));
            output.push('\n');
        }
    }

    output.push_str("  }");

    Ok(output)
}

/// Convert PHIR type to MLIR type string
fn type_to_mlir(ty: &crate::types::Type) -> String {
    use crate::types::Type;
    match ty {
        Type::Qubit => "!quantum.qubit".to_string(),
        Type::Bool => "i1".to_string(),
        Type::Int(width) => format!("i{}", width.bits()),
        Type::Float(_) => "f64".to_string(),
        _ => "!unknown".to_string(),
    }
}

/// Convert PHIR instruction to MLIR text with SSA value mapping
fn convert_instruction_to_mlir_with_mapping(
    instruction: &crate::phir::Instruction,
    ssa_to_qubit: &mut std::collections::BTreeMap<u32, u32>,
) -> Result<String> {
    use crate::ops::{Operation, QuantumOp};

    let mut output = String::new();

    // Helper to resolve SSA value to actual qubit
    let resolve_ssa = |ssa_id: u32| -> u32 { ssa_to_qubit.get(&ssa_id).copied().unwrap_or(ssa_id) };

    // Operation
    match &instruction.operation {
        Operation::Quantum(quantum_op) => {
            match quantum_op {
                QuantumOp::Alloc => {
                    // Allocate a new qubit
                    if !instruction.results.is_empty() {
                        let result_id = instruction.results[0].id;
                        write!(
                            &mut output,
                            "    %{result_id} = call @__quantum__rt__qubit_allocate() : () -> i64"
                        )
                        .unwrap();
                        // This SSA value represents an actual qubit
                        ssa_to_qubit.insert(result_id, result_id);
                    }
                }
                QuantumOp::H => {
                    // H gate - operates in-place
                    let operand = instruction
                        .operands
                        .first()
                        .ok_or_else(|| PhirError::internal("H gate missing operand"))?;
                    let qubit_id = resolve_ssa(operand.id);
                    write!(
                        &mut output,
                        "    call @__quantum__qis__h__body(%{qubit_id}) : (i64) -> ()"
                    )
                    .unwrap();

                    // Map output SSA values to the same qubit
                    if !instruction.results.is_empty() {
                        ssa_to_qubit.insert(instruction.results[0].id, qubit_id);
                    }
                }
                QuantumOp::CX => {
                    // CX gate - operates in-place on both qubits
                    if instruction.operands.len() < 2 {
                        return Err(PhirError::internal("CX gate needs 2 operands"));
                    }
                    let control_qubit = resolve_ssa(instruction.operands[0].id);
                    let target_qubit = resolve_ssa(instruction.operands[1].id);

                    write!(&mut output,
                        "    call @__quantum__qis__cx__body(%{control_qubit}, %{target_qubit}) : (i64, i64) -> ()"
                    ).unwrap();

                    // Map output SSA values to the same qubits
                    if !instruction.results.is_empty() {
                        ssa_to_qubit.insert(instruction.results[0].id, control_qubit);
                    }
                    if instruction.results.len() >= 2 {
                        ssa_to_qubit.insert(instruction.results[1].id, target_qubit);
                    }
                }
                QuantumOp::Measure => {
                    // Measurement - QIR requires allocating a result and then measuring
                    let operand = instruction
                        .operands
                        .first()
                        .ok_or_else(|| PhirError::internal("Measure missing operand"))?;
                    let qubit_id = resolve_ssa(operand.id);

                    if !instruction.results.is_empty() {
                        // Allocate a result register
                        let result_reg_id = 900 + instruction.results[0].id; // Use high numbers to avoid conflicts
                        writeln!(&mut output,
                            "    %{result_reg_id} = call @__quantum__rt__result_allocate() : () -> i64"
                        ).unwrap();

                        // Perform measurement
                        write!(
                            &mut output,
                            "    %{} = call @__quantum__qis__m__body(%{}, %{}) : (i64, i64) -> i32",
                            instruction.results[0].id, qubit_id, result_reg_id
                        )
                        .unwrap();
                    }
                }
                _ => {
                    write!(&mut output, "    // TODO: quantum op {quantum_op:?}").unwrap();
                }
            }
        }
        _ => {
            write!(
                &mut output,
                "    // TODO: operation {:?}",
                instruction.operation
            )
            .unwrap();
        }
    }

    Ok(output)
}

/// Convert PHIR terminator to MLIR text
fn convert_terminator_to_mlir(terminator: &crate::phir::Terminator) -> String {
    use crate::phir::Terminator;

    match terminator {
        Terminator::Return { values } => {
            if values.is_empty() {
                "    return".to_string()
            } else {
                let values_str: Vec<String> = values.iter().map(|v| format!("%{}", v.id)).collect();
                // Build the type list based on actual number of values
                // Use i32 for measurement results since that's what QIR returns
                let types: Vec<&str> = values.iter().map(|_| "i32").collect();
                format!(
                    "    return {} : {}",
                    values_str.join(", "),
                    types.join(", ")
                )
            }
        }
        _ => format!("    // TODO: terminator {terminator:?}"),
    }
}

/// Convert PHIR Module to MLIR text string
///
/// This is a convenience wrapper around `lower_phir_to_mlir` that returns the MLIR text directly
///
/// # Errors
///
/// Returns an error if the MLIR lowering fails
pub fn phir_to_mlir(module: &Module, config: &PhirConfig) -> Result<String> {
    let mlir_module = lower_phir_to_mlir(module, config)?;
    Ok(mlir_module.content)
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_mlir_lowering_placeholder() {
        // TODO: Add real tests when implementation is ready
        // Placeholder test to ensure the module compiles
    }
}
