/*!
Builtin operations for PHIR

Following MLIR's design, these are the fundamental operations that structure
the IR. Everything is an Operation - modules, functions, etc.
*/

use crate::ops::Operation;
use crate::phir::{AttributeValue, Attributes, Instruction, Region};
use crate::types::FunctionType;
use std::collections::BTreeMap;
use std::fmt::Write;

/// Builtin operations that define IR structure
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum BuiltinOp {
    /// Module operation - top-level container
    Module(ModuleOp),
    /// Function operation - defines a callable function
    Func(FuncOp),
    /// Return operation - terminates a function
    Return(ReturnOp),
    /// Variable definition operation
    VarDefine(VarDefineOp),
}

/// Module operation - the top-level container
///
/// In MLIR style, a module is just an operation with a single region
/// containing a single block. The module's body contains other operations
/// (typically functions and globals).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ModuleOp {
    /// Module name/symbol
    pub name: String,
    /// Module attributes
    pub attributes: Attributes,
    /// The module body region
    pub body: Region,
}

impl ModuleOp {
    /// Create a new module operation
    pub fn new(name: impl Into<String>) -> Self {
        use crate::region_kinds::RegionKind;

        Self {
            name: name.into(),
            attributes: BTreeMap::new(),
            body: Region::new(RegionKind::SSACFG),
        }
    }

    /// Convert to a generic operation
    #[must_use]
    pub fn to_operation(self) -> Operation {
        Operation::Builtin(BuiltinOp::Module(self))
    }

    /// Add an operation to the module's body
    pub fn add_operation(&mut self, op: Instruction) {
        if let Some(block) = self.body.blocks.first_mut() {
            block.add_instruction(op);
        } else {
            // Create entry block if needed
            let mut block = crate::phir::Block::entry();
            block.add_instruction(op);
            self.body.add_block(block);
        }
    }

    /// Add a function to the module
    pub fn add_function(&mut self, function: FuncOp) {
        let func_inst = Instruction::new(function.to_operation(), vec![], vec![], vec![]);
        self.add_operation(func_inst);
    }

    /// Find a function by name
    #[must_use]
    pub fn find_function(&self, name: &str) -> Option<&FuncOp> {
        if let Some(block) = self.body.blocks.first() {
            for inst in &block.operations {
                if let Operation::Builtin(BuiltinOp::Func(func)) = &inst.operation
                    && func.name == name
                {
                    return Some(func);
                }
            }
        }
        None
    }

    /// Convert to MLIR text representation
    #[must_use]
    pub fn to_mlir_text(&self) -> String {
        builtin_op_to_mlir_text(&BuiltinOp::Module(self.clone()), 0)
    }

    /// Validate module structure
    ///
    /// # Errors
    ///
    /// Returns an error if the module structure is invalid
    pub fn validate(&self) -> crate::error::Result<()> {
        // TODO: Implement validation
        Ok(())
    }

    /// Count quantum operations in module
    #[must_use]
    pub fn count_qubits(&self) -> usize {
        // TODO: Implement by analyzing operations
        0
    }

    /// Count classical operations in module
    #[must_use]
    pub fn count_classical_ops(&self) -> usize {
        // TODO: Implement by counting classical operations
        0
    }
}

/// Function operation - defines a callable function
///
/// A function is an operation with regions for its body.
/// The function signature is encoded in the operation's type.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FuncOp {
    /// Function name/symbol
    pub name: String,
    /// Function signature
    pub function_type: FunctionType,
    /// Function attributes (visibility, etc.)
    pub attributes: Attributes,
    /// Function body regions (usually one)
    pub body: Vec<Region>,
}

impl FuncOp {
    /// Create a new function operation
    pub fn new(name: impl Into<String>, function_type: FunctionType) -> Self {
        use crate::region_kinds::RegionKind;

        // Create a region with an entry block
        let mut region = Region::new(RegionKind::SSACFG);
        region.add_block(crate::phir::Block::entry());

        Self {
            name: name.into(),
            function_type,
            attributes: BTreeMap::new(),
            body: vec![region],
        }
    }

    /// Create a new function with visibility (compatibility)
    pub fn new_with_visibility(
        name: impl Into<String>,
        signature: FunctionType,
        visibility: crate::phir::Visibility,
    ) -> Self {
        let mut func = Self::new(name, signature);
        // Store visibility as an attribute
        func.attributes.insert(
            "visibility".to_string(),
            AttributeValue::String(format!("{visibility:?}")),
        );
        func
    }

    /// Convert to a generic operation
    #[must_use]
    pub fn to_operation(self) -> Operation {
        Operation::Builtin(BuiltinOp::Func(self))
    }

    /// Get the entry region
    #[must_use]
    pub fn entry_region(&self) -> Option<&Region> {
        self.body.first()
    }

    /// Get the entry region mutably
    pub fn entry_region_mut(&mut self) -> Option<&mut Region> {
        self.body.first_mut()
    }

    /// Get function signature (compatibility)
    #[must_use]
    pub fn signature(&self) -> &FunctionType {
        &self.function_type
    }

    /// Get regions (compatibility)
    #[must_use]
    pub fn regions(&self) -> &Vec<Region> {
        &self.body
    }

    /// Convert to MLIR text representation
    #[must_use]
    pub fn to_mlir_text(&self) -> String {
        builtin_op_to_mlir_text(&BuiltinOp::Func(self.clone()), 0)
    }
}

/// Return operation - terminates a function
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ReturnOp {
    /// Values to return
    pub operands: Vec<crate::phir::SSAValue>,
}

impl ReturnOp {
    /// Create a new return operation
    #[must_use]
    pub fn new(operands: Vec<crate::phir::SSAValue>) -> Self {
        Self { operands }
    }

    /// Convert to a generic operation
    #[must_use]
    pub fn to_operation(self) -> Operation {
        Operation::Builtin(BuiltinOp::Return(self))
    }
}

/// Variable definition operation
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct VarDefineOp {
    /// Variable name
    pub name: String,
    /// Variable type ("qubits", "i64", etc.)
    pub var_type: String,
    /// Variable size (array length)
    pub size: usize,
}

impl VarDefineOp {
    /// Create a new variable definition operation
    #[must_use]
    pub fn new(name: String, var_type: String, size: usize) -> Self {
        Self {
            name,
            var_type,
            size,
        }
    }

    /// Convert to a generic operation
    #[must_use]
    pub fn to_operation(self) -> Operation {
        Operation::Builtin(BuiltinOp::VarDefine(self))
    }
}

/// Convert builtin operations to MLIR text
#[must_use]
pub fn builtin_op_to_mlir_text(op: &BuiltinOp, indent: usize) -> String {
    match op {
        BuiltinOp::Module(module_op) => {
            let mut output = String::new();
            writeln!(&mut output, "module @{} {{", module_op.name).unwrap();

            // Module attributes
            if !module_op.attributes.is_empty() {
                output.push_str("  attributes {\n");
                for (key, value) in &module_op.attributes {
                    writeln!(&mut output, "    {key} = {value:?}").unwrap();
                }
                output.push_str("  }\n");
            }

            // Module body
            output.push_str(&module_op.body.to_mlir_text(indent + 1));
            output.push_str("}\n");
            output
        }

        BuiltinOp::Func(func_op) => {
            let mut output = String::new();
            let indent_str = "  ".repeat(indent);

            // Function header
            write!(&mut output, "{}func.func @{}", indent_str, func_op.name).unwrap();

            // Function signature
            output.push('(');
            for (i, input) in func_op.function_type.inputs.iter().enumerate() {
                if i > 0 {
                    output.push_str(", ");
                }
                write!(&mut output, "%arg{i}: {input}").unwrap();
            }
            output.push_str(") -> ");

            if func_op.function_type.outputs.is_empty() {
                output.push_str("()");
            } else if func_op.function_type.outputs.len() == 1 {
                output.push_str(&func_op.function_type.outputs[0].to_string());
            } else {
                output.push('(');
                for (i, output_type) in func_op.function_type.outputs.iter().enumerate() {
                    if i > 0 {
                        output.push_str(", ");
                    }
                    output.push_str(&output_type.to_string());
                }
                output.push(')');
            }

            // Function attributes
            if !func_op.attributes.is_empty() {
                output.push_str(" attributes {");
                for (i, (key, value)) in func_op.attributes.iter().enumerate() {
                    if i > 0 {
                        output.push_str(", ");
                    }
                    write!(&mut output, "{key} = {value:?}").unwrap();
                }
                output.push('}');
            }

            output.push_str(" {\n");

            // Function body
            for region in &func_op.body {
                output.push_str(&region.to_mlir_text(indent + 1));
            }

            writeln!(&mut output, "{indent_str}}}").unwrap();
            output
        }

        BuiltinOp::Return(return_op) => {
            let indent_str = "  ".repeat(indent);
            let mut output = format!("{indent_str}return");

            if !return_op.operands.is_empty() {
                output.push(' ');
                for (i, operand) in return_op.operands.iter().enumerate() {
                    if i > 0 {
                        output.push_str(", ");
                    }
                    output.push_str(&operand.to_string());
                }
            }

            output.push('\n');
            output
        }

        BuiltinOp::VarDefine(var_def) => {
            let indent_str = "  ".repeat(indent);
            format!(
                "{}%{} = phir.var_define {} : {}<{}>",
                indent_str, var_def.name, var_def.var_type, var_def.var_type, var_def.size
            )
        }
    }
}

/// Helper to create a module with functions
///
/// This provides a convenient API while maintaining MLIR's structure
pub struct ModuleBuilder {
    module: ModuleOp,
}

impl ModuleBuilder {
    /// Create a new module builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            module: ModuleOp::new(name),
        }
    }

    /// Add a function to the module
    pub fn add_function(&mut self, func: FuncOp) {
        let func_inst = Instruction::new(func.to_operation(), vec![], vec![], vec![]);
        self.module.add_operation(func_inst);
    }

    /// Build the final module operation
    #[must_use]
    pub fn build(self) -> ModuleOp {
        self.module
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phir::Instruction;
    use crate::types::{FunctionType, Type};

    #[test]
    fn test_module_op() {
        let mut module = ModuleOp::new("test_module");
        assert_eq!(module.name, "test_module");
        assert_eq!(module.body.blocks.len(), 0);

        // Add a function
        let func = FuncOp::new("test_func", FunctionType::default());
        let func_inst = Instruction::new(func.to_operation(), vec![], vec![], vec![]);
        module.add_operation(func_inst);

        assert_eq!(module.body.blocks.len(), 1);
        assert_eq!(module.body.blocks[0].operations.len(), 1);
    }

    #[test]
    fn test_func_op() {
        let func_type = FunctionType {
            inputs: vec![Type::Int(crate::types::IntWidth::I32)],
            outputs: vec![Type::Int(crate::types::IntWidth::I32)],
            variadic: false,
        };

        let func = FuncOp::new("add_one", func_type);
        assert_eq!(func.name, "add_one");
        assert_eq!(func.body.len(), 1);
    }
}
