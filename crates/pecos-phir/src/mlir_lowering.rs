/*!
PHIR to MLIR Lowering

This module converts PHIR (PECOS High-level IR) to MLIR text format.
The generated MLIR can be processed by MLIR tools (mlir-opt, mlir-translate)
to produce LLVM IR.

Only allocation, H, CX, measurement and return are supported. Unsupported
operations and control-flow structures return an error instead of losing code.
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

    if module.body.blocks.len() > 1 {
        return Err(PhirError::internal(format!(
            "Module {} has {} blocks; use one module block for MLIR lowering",
            module.name,
            module.body.blocks.len()
        )));
    }
    // Always use standard dialect - it will be converted to LLVM by mlir-opt
    writeln!(&mut content, "module @{} {{", module.name).expect("writing to a String cannot fail");

    // Visit every top-level operation; none may be silently dropped.
    for block in &module.body.blocks {
        if let Some(terminator) = &block.terminator {
            return Err(PhirError::internal(format!(
                "Unsupported module terminator {terminator:?}; put Return in a function body"
            )));
        }
        for instruction in &block.operations {
            if let crate::ops::Operation::Builtin(crate::builtin_ops::BuiltinOp::Func(func)) =
                &instruction.operation
            {
                content.push_str(&convert_function_to_mlir(func)?);
                content.push('\n');
            } else {
                return Err(unsupported(&instruction.operation));
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
    if func.body.len() != 1 || func.body[0].blocks.len() != 1 {
        return Err(PhirError::internal(format!(
            "Function {} has {} regions and block counts {:?}; MLIR lowering requires one region with one block",
            func.name,
            func.body.len(),
            func.body.iter().map(|r| r.blocks.len()).collect::<Vec<_>>()
        )));
    }

    // Add function declarations for QIR intrinsics - using i64 for PECOS compatibility
    output.push_str("  func private @__quantum__qis__h__body(i64)\n");
    output.push_str("  func private @__quantum__qis__cx__body(i64, i64)\n");
    output.push_str("  func private @__quantum__qis__m__body(i64, i64) -> i32\n");
    output.push_str("  func private @__quantum__rt__qubit_allocate() -> i64\n");
    output.push_str("  func private @__quantum__rt__result_allocate() -> i64\n");
    output.push_str("  func private @__quantum__rt__qubit_release(i64)\n");
    output.push('\n');

    // Function signature (using older MLIR syntax for compatibility)
    write!(&mut output, "  func @{}(", func.name).expect("writing to a String cannot fail");

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
            output.push_str(&convert_terminator_to_mlir(terminator)?);
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

    if !instruction.regions.is_empty() {
        return Err(unsupported(&instruction.operation));
    }
    let (operands, max_results, min_results) = match instruction.operation {
        Operation::Quantum(QuantumOp::Alloc) => (0, 1, 1),
        Operation::Quantum(QuantumOp::H) => (1, 1, 0),
        Operation::Quantum(QuantumOp::CX) => (2, 2, 0),
        Operation::Quantum(QuantumOp::Measure) => (1, 1, 1),
        _ => return Err(unsupported(&instruction.operation)),
    };
    if instruction.operands.len() != operands
        || !(min_results..=max_results).contains(&instruction.results.len())
    {
        return Err(PhirError::internal(format!(
            "{} has {} operands and {} results; supply {operands} operands and {min_results}..={max_results} results for MLIR lowering",
            instruction.operation.name(),
            instruction.operands.len(),
            instruction.results.len()
        )));
    }
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
                        .expect("writing to a String cannot fail");
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
                    .expect("writing to a String cannot fail");

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
                    ).expect("writing to a String cannot fail");

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
                        let result_reg_id = instruction.results[0].id.checked_add(900).ok_or_else(|| PhirError::internal(format!("Measure result ID {} is too large; use an ID at most {} for MLIR lowering", instruction.results[0].id, u32::MAX - 900)))?; // Use high numbers to avoid conflicts
                        writeln!(&mut output,
                            "    %{result_reg_id} = call @__quantum__rt__result_allocate() : () -> i64"
                        ).expect("writing to a String cannot fail");

                        // Perform measurement
                        write!(
                            &mut output,
                            "    %{} = call @__quantum__qis__m__body(%{}, %{}) : (i64, i64) -> i32",
                            instruction.results[0].id, qubit_id, result_reg_id
                        )
                        .expect("writing to a String cannot fail");
                    }
                }
                _ => return Err(unsupported(&instruction.operation)),
            }
        }
        _ => return Err(unsupported(&instruction.operation)),
    }

    Ok(output)
}

fn unsupported(operation: &crate::ops::Operation) -> PhirError {
    PhirError::internal(format!(
        "Unsupported operation {} ({operation:?}); use Alloc, H, CX, Measure and Return for MLIR lowering",
        operation.name()
    ))
}

/// Convert PHIR terminator to MLIR text
fn convert_terminator_to_mlir(terminator: &crate::phir::Terminator) -> Result<String> {
    use crate::phir::Terminator;

    Ok(match terminator {
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
        _ => {
            return Err(PhirError::internal(format!(
                "Unsupported terminator {terminator:?}; use Return for MLIR lowering"
            )));
        }
    })
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
