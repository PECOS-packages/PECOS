/*!
RON (Rusty Object Notation) support for PHIR

This module provides serialization and deserialization of PHIR structures to/from RON format.
RON is a human-readable data serialization format similar to JSON but with Rust-like syntax.

RON is used as a debugging format for PHIR, allowing developers to:
1. Inspect PHIR structures in a human-readable format
2. Create test cases by writing PHIR directly in RON
3. Debug transformations by comparing RON outputs
*/

use crate::{Module, PhirError, Result};
use ron::ser::{PrettyConfig, to_string_pretty};
use std::fs;
use std::path::Path;

/// Serialize a PHIR module to RON string
///
/// # Errors
///
/// Returns an error if serialization fails
pub fn to_ron(module: &Module) -> Result<String> {
    let pretty = PrettyConfig::new()
        .depth_limit(4)
        .separate_tuple_members(true)
        .enumerate_arrays(true);

    to_string_pretty(module, pretty)
        .map_err(|e| PhirError::internal(format!("Failed to serialize to RON: {e}")))
}

/// Serialize a PHIR module to a RON file
///
/// # Errors
///
/// Returns an error if serialization or file writing fails
pub fn to_ron_file(module: &Module, path: impl AsRef<Path>) -> Result<()> {
    let ron_string = to_ron(module)?;
    fs::write(path, ron_string)
        .map_err(|e| PhirError::internal(format!("Failed to write RON file: {e}")))
}

/// Deserialize a PHIR module from RON string
///
/// # Errors
///
/// Returns an error if deserialization fails
pub fn from_ron(ron_str: &str) -> Result<Module> {
    ron::from_str(ron_str)
        .map_err(|e| PhirError::internal(format!("Failed to deserialize from RON: {e}")))
}

/// Deserialize a PHIR module from a RON file
///
/// # Errors
///
/// Returns an error if file reading or deserialization fails
pub fn from_ron_file(path: impl AsRef<Path>) -> Result<Module> {
    let ron_string = fs::read_to_string(path)
        .map_err(|e| PhirError::internal(format!("Failed to read RON file: {e}")))?;
    from_ron(&ron_string)
}

/// Extension trait for Module to add RON convenience methods
pub trait ModuleRonExt {
    /// Convert this module to RON string
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails
    fn to_ron(&self) -> Result<String>;

    /// Save this module to a RON file
    ///
    /// # Errors
    ///
    /// Returns an error if serialization or file writing fails
    fn save_ron(&self, path: impl AsRef<Path>) -> Result<()>;
}

impl ModuleRonExt for Module {
    fn to_ron(&self) -> Result<String> {
        to_ron(self)
    }

    fn save_ron(&self, path: impl AsRef<Path>) -> Result<()> {
        to_ron_file(self, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtin_ops::{FuncOp, ModuleOp};
    use crate::ops::{Operation, QuantumOp, SSAValue};
    use crate::phir::{Block, Instruction, Region};
    use crate::region_kinds::RegionKind;
    use crate::types::{FunctionType, bit_type, qubit_type};

    #[test]
    fn test_module_ron_roundtrip() {
        // Create a simple module
        let module = ModuleOp::new("test_module");

        // Convert to RON and back
        let ron_string = to_ron(&module).unwrap();
        let module2 = from_ron(&ron_string).unwrap();

        assert_eq!(module.name, module2.name);
    }

    #[test]
    fn test_complex_module_ron() {
        // Create a module with a function
        let mut module = ModuleOp::new("quantum_module");

        // Create a function
        let signature = FunctionType {
            inputs: vec![qubit_type()],
            outputs: vec![bit_type()],
            variadic: false,
        };

        let mut func = FuncOp::new("measure_qubit", signature);

        // Add a region with a block
        let mut region = Region::new(RegionKind::Graph);
        let mut block = Block::new(None);

        // Add a quantum operation
        let h_gate = Instruction::new(
            Operation::Quantum(QuantumOp::H),
            vec![SSAValue::new(0)],
            vec![SSAValue::new(1)],
            vec![qubit_type()],
        );
        block.add_instruction(h_gate);

        // Add a measurement
        let measure = Instruction::new(
            Operation::Quantum(QuantumOp::Measure),
            vec![SSAValue::new(1)],
            vec![SSAValue::new(2)],
            vec![bit_type()],
        );
        block.add_instruction(measure);

        region.add_block(block);
        func.body.push(region);

        // Add function to module
        let func_inst = Instruction::new(
            Operation::Builtin(crate::builtin_ops::BuiltinOp::Func(func)),
            vec![],
            vec![],
            vec![],
        );
        module.add_operation(func_inst);

        // Convert to RON
        let ron_string = to_ron(&module).unwrap();

        // Should contain our module and function names
        assert!(ron_string.contains("quantum_module"));
        assert!(ron_string.contains("measure_qubit"));

        // Should contain our operations
        assert!(ron_string.contains("Quantum(H)"));
        assert!(ron_string.contains("Quantum(Measure)"));

        // Verify roundtrip
        let module2 = from_ron(&ron_string).unwrap();
        assert_eq!(module.name, module2.name);
        assert_eq!(module.body.blocks.len(), module2.body.blocks.len());
    }

    #[test]
    fn test_ron_pretty_formatting() {
        let module = ModuleOp::new("pretty_test");
        let ron_string = to_ron(&module).unwrap();

        // RON should be nicely formatted with newlines
        assert!(ron_string.contains('\n'));

        // RON starts with parentheses because Module is a type alias
        assert!(ron_string.starts_with('('));
    }

    #[test]
    fn test_qis_pipeline_module_ron_roundtrip() {
        // Parse a real QIS LLVM IR module, then roundtrip through RON
        let ir = r"
declare void @___rxy(i64, double, double)
declare void @___rz(i64, double)
declare void @___rzz(i64, i64, double)
declare i1 @___measure(i64)
declare void @___qalloc(i64)
declare void @___qfree(i64)

define i64 @qmain(i64 %0) {
entry:
  call void @___qalloc(i64 0)
  call void @___qalloc(i64 1)
  call void @___rz(i64 0, double 0x3FF921FB54442D18)
  call void @___rxy(i64 0, double 0x3FF921FB54442D18, double 0.0)
  call void @___rz(i64 0, double 0x3FF921FB54442D18)
  call void @___rxy(i64 1, double 0x3FF921FB54442D18, double 0xBFF921FB54442D18)
  call void @___rzz(i64 0, i64 1, double 0xBFE921FB54442D18)
  call void @___rz(i64 1, double 0xBFF921FB54442D18)
  call void @___rxy(i64 1, double 0x3FF921FB54442D18, double 0x3FF921FB54442D18)
  %m0 = call i1 @___measure(i64 0)
  %m1 = call i1 @___measure(i64 1)
  call void @___qfree(i64 0)
  call void @___qfree(i64 1)
  ret i64 0
}
";
        let module = crate::parse_qis_to_quantum(ir).unwrap();

        // Serialize to RON
        let ron_string = to_ron(&module).unwrap();

        // Verify the RON contains expected quantum ops
        assert!(ron_string.contains("Alloc"), "RON should contain Alloc");
        assert!(ron_string.contains("RZ("), "RON should contain RZ");
        assert!(ron_string.contains("R1XY("), "RON should contain R1XY");
        assert!(ron_string.contains("RZZ("), "RON should contain RZZ");
        assert!(ron_string.contains("Measure"), "RON should contain Measure");
        assert!(ron_string.contains("Dealloc"), "RON should contain Dealloc");

        // Deserialize back
        let module2 = from_ron(&ron_string).unwrap();

        // Verify structural equality
        assert_eq!(module.name, module2.name);
        assert_eq!(module.body.blocks.len(), module2.body.blocks.len());
        assert_eq!(
            module.body.blocks[0].operations.len(),
            module2.body.blocks[0].operations.len(),
        );

        // Verify operation names match
        let ops1: Vec<_> = module.body.blocks[0]
            .operations
            .iter()
            .map(|i| i.operation.name())
            .collect();
        let ops2: Vec<_> = module2.body.blocks[0]
            .operations
            .iter()
            .map(|i| i.operation.name())
            .collect();
        assert_eq!(ops1, ops2);
    }

    #[test]
    fn test_ron_file_roundtrip() {
        let module = ModuleOp::new("file_test");

        let tmp_path = std::env::temp_dir().join("pecos_ron_support_test.ron");

        to_ron_file(&module, &tmp_path).unwrap();
        assert!(tmp_path.exists());

        let module2 = from_ron_file(&tmp_path).unwrap();
        assert_eq!(module.name, module2.name);

        let _ = std::fs::remove_file(&tmp_path);
    }

    #[test]
    fn test_from_ron_invalid_input() {
        let result = from_ron("this is not valid RON");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_ron_file_nonexistent() {
        let result = from_ron_file("/nonexistent/path/module.ron");
        assert!(result.is_err());
    }

    #[test]
    fn test_ron_angle_fidelity_roundtrip() {
        // Verify that f64 angle values survive RON roundtrip exactly
        let ir = r"
declare void @___rxy(i64, double, double)
declare void @___rz(i64, double)
declare void @___rzz(i64, i64, double)
declare i1 @___measure(i64)
declare void @___qalloc(i64)
declare void @___qfree(i64)

define i64 @qmain(i64 %0) {
entry:
  call void @___qalloc(i64 0)
  call void @___qalloc(i64 1)
  call void @___rz(i64 0, double 0x3FF921FB54442D18)
  call void @___rxy(i64 0, double 0x3FF921FB54442D18, double 0.0)
  call void @___rzz(i64 0, i64 1, double 0xBFE921FB54442D18)
  %m0 = call i1 @___measure(i64 0)
  call void @___qfree(i64 0)
  call void @___qfree(i64 1)
  ret i64 0
}
";
        let module = crate::parse_qis_to_quantum(ir).unwrap();

        // Helper to extract angle values from a module (as radians for comparison)
        let extract_angles = |m: &crate::Module| -> Vec<f64> {
            m.body.blocks[0]
                .operations
                .iter()
                .filter_map(|i| match &i.operation {
                    Operation::Quantum(QuantumOp::R1XY(theta, phi)) => {
                        Some(theta.to_radians() + phi.to_radians())
                    }
                    Operation::Quantum(QuantumOp::RZ(angle) | QuantumOp::RZZ(angle)) => {
                        Some(angle.to_radians())
                    }
                    _ => None,
                })
                .collect()
        };

        let angles_before = extract_angles(&module);

        // Roundtrip through RON
        let ron_string = to_ron(&module).unwrap();
        let module2 = from_ron(&ron_string).unwrap();

        let angles_after = extract_angles(&module2);

        assert_eq!(angles_before.len(), angles_after.len());
        for (before, after) in angles_before.iter().zip(angles_after.iter()) {
            assert_eq!(
                before.to_bits(),
                after.to_bits(),
                "angle {before} lost precision in RON roundtrip"
            );
        }
    }
}
