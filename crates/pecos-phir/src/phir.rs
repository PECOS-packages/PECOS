/*!
PHIR - PECOS High-level Intermediate Representation

MLIR-inspired hierarchical SSA representation that serves as both AST and IR.

Key design principles:
1. MLIR-style hierarchical organization: Module → Function → Region → Block
2. SSA form with explicit use-def chains
3. Unified representation - no separate AST needed
4. Extensible dialect system
5. Region-based organization for control flow
6. Built-in support for quantum-classical hybrid programs

PHIR leverages MLIR's flexibility to handle both parsing and transformations in a single representation.
*/

use std::fmt::Write;

use crate::error::SourceLocation;
use crate::ops::Operation;
pub use crate::ops::SSAValue;
use crate::types::Type;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// PHIR Module - convenience wrapper around `ModuleOp`
///
/// This provides a familiar API while maintaining MLIR's structure where
/// everything is an Operation. The module is actually a `ModuleOp` operation.
pub type Module = crate::builtin_ops::ModuleOp;

/// PHIR Function - convenience wrapper around `FuncOp`
///
/// Functions are operations in MLIR, not separate structures.
pub type Function = crate::builtin_ops::FuncOp;

/// Region containing basic blocks
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Region {
    /// Basic blocks in this region
    pub blocks: Vec<Block>,
    /// Region kind (for optimization hints)
    pub kind: crate::region_kinds::RegionKind,
    /// Region attributes
    pub attributes: Attributes,
}

/// Basic block with operations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Block {
    /// Block label/name
    pub label: Option<String>,
    /// Block arguments (for phi nodes)
    pub arguments: Vec<BlockArgument>,
    /// Operations in this block
    pub operations: Vec<Instruction>,
    /// Block terminator (optional for entry blocks with `NoTerminator` trait)
    pub terminator: Option<Terminator>,
    /// Block attributes
    pub attributes: Attributes,
}

/// Single instruction in PHIR
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Instruction {
    /// The operation being performed
    pub operation: Operation,
    /// SSA operands (inputs)
    pub operands: Vec<SSAValue>,
    /// SSA results (outputs)
    pub results: Vec<SSAValue>,
    /// Result types
    pub result_types: Vec<Type>,
    /// Nested regions (for operations like loops, conditionals, lambdas)
    pub regions: Vec<Region>,
    /// Instruction attributes
    pub attributes: Attributes,
    /// Source location for debugging
    pub location: Option<SourceLocation>,
}

/// Block terminator (control flow)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Terminator {
    /// Return from function
    Return { values: Vec<SSAValue> },
    /// Branch to another block
    Branch {
        target: BlockRef,
        args: Vec<SSAValue>,
    },
    /// Conditional branch
    ConditionalBranch {
        condition: SSAValue,
        true_target: BlockRef,
        true_args: Vec<SSAValue>,
        false_target: BlockRef,
        false_args: Vec<SSAValue>,
    },
    /// Switch statement
    Switch {
        value: SSAValue,
        default_target: BlockRef,
        default_args: Vec<SSAValue>,
        cases: Vec<SwitchCase>,
    },
    /// Unreachable terminator
    Unreachable,
}

/// Global variable or constant
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Global {
    /// Global name
    pub name: String,
    /// Global type
    pub ty: Type,
    /// Initial value (if any)
    pub initial_value: Option<ConstantValue>,
    /// Whether global is mutable
    pub mutable: bool,
    /// Visibility
    pub visibility: Visibility,
    /// Attributes
    pub attributes: Attributes,
}

/// Block argument (for phi nodes and control flow)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlockArgument {
    /// SSA value for this argument
    pub value: SSAValue,
    /// Argument type
    pub ty: Type,
    /// Optional name for debugging
    pub name: Option<String>,
}

/// Block reference
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum BlockRef {
    /// Reference by index within current region
    Index(usize),
    /// Reference by label within current region
    Label(String),
    /// Reference to parent region's continuation
    Parent,
}

/// Switch case
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SwitchCase {
    /// Case value
    pub value: i64,
    /// Target block
    pub target: BlockRef,
    /// Arguments to target block
    pub args: Vec<SSAValue>,
}

/// Function/variable visibility
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Visibility {
    Public,
    Private,
    Internal,
}

/// Constant values in PHIR
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConstantValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Array(Vec<ConstantValue>),
    Complex { real: f64, imag: f64 },
    Unit,
}

/// Attributes (compile-time metadata)
pub type Attributes = BTreeMap<String, AttributeValue>;

/// Attribute values
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AttributeValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Array(Vec<AttributeValue>),
    Dict(BTreeMap<String, AttributeValue>),
}

// Module and Function implementations are now in builtin_ops.rs
// since they are type aliases to ModuleOp and FuncOp

impl Region {
    /// Create a new region
    #[must_use]
    pub fn new(kind: crate::region_kinds::RegionKind) -> Self {
        Self {
            blocks: Vec::new(),
            kind,
            attributes: BTreeMap::new(),
        }
    }

    /// Add a block to the region
    pub fn add_block(&mut self, block: Block) {
        self.blocks.push(block);
    }

    /// Get entry block (first block)
    #[must_use]
    pub fn entry_block(&self) -> Option<&Block> {
        self.blocks.first()
    }

    /// Get entry block mutably
    pub fn entry_block_mut(&mut self) -> Option<&mut Block> {
        self.blocks.first_mut()
    }

    /// Builder-style method to add a block
    #[must_use]
    pub fn with_block(mut self, block: Block) -> Self {
        self.blocks.push(block);
        self
    }

    /// Builder-style method to add an attribute
    #[must_use]
    pub fn with_attr(mut self, key: impl Into<String>, value: AttributeValue) -> Self {
        self.attributes.insert(key.into(), value);
        self
    }

    /// Convert region to MLIR text
    #[must_use]
    pub fn to_mlir_text(&self, indent: usize) -> String {
        let mut output = String::new();
        let _indent_str = "  ".repeat(indent);

        for (i, block) in self.blocks.iter().enumerate() {
            if i > 0 {
                output.push('\n');
            }
            output.push_str(&block.to_mlir_text(indent));
        }

        output
    }
}

impl Block {
    /// Create a new block
    #[must_use]
    pub fn new(label: Option<String>) -> Self {
        Self {
            label,
            arguments: Vec::new(),
            operations: Vec::new(),
            terminator: None,
            attributes: BTreeMap::new(),
        }
    }

    /// Add an instruction to the block
    pub fn add_instruction(&mut self, instruction: Instruction) {
        self.operations.push(instruction);
    }

    /// Set the block terminator
    pub fn set_terminator(&mut self, terminator: Terminator) {
        self.terminator = Some(terminator);
    }

    /// Check if block has a terminator
    #[must_use]
    pub fn has_terminator(&self) -> bool {
        self.terminator.is_some()
    }

    /// Create an entry block (no label, no arguments)
    #[must_use]
    pub fn entry() -> Self {
        Self::new(None)
    }

    /// Builder-style method to add an instruction
    #[must_use]
    pub fn with_instruction(mut self, instruction: Instruction) -> Self {
        self.operations.push(instruction);
        self
    }

    /// Builder-style method to add an attribute
    #[must_use]
    pub fn with_attr(mut self, key: impl Into<String>, value: AttributeValue) -> Self {
        self.attributes.insert(key.into(), value);
        self
    }

    /// Convert block to MLIR text
    #[must_use]
    pub fn to_mlir_text(&self, indent: usize) -> String {
        let mut output = String::new();
        let indent_str = "  ".repeat(indent);

        // Block header with arguments
        if let Some(label) = &self.label {
            write!(output, "{indent_str}^{label}(").unwrap();
        } else {
            write!(output, "{indent_str}^bb0(").unwrap();
        }

        for (i, arg) in self.arguments.iter().enumerate() {
            if i > 0 {
                output.push_str(", ");
            }
            write!(output, "{}: {}", arg.value, arg.ty).unwrap();
        }
        output.push_str("):\n");

        // Instructions
        for instruction in &self.operations {
            output.push_str(&instruction.to_mlir_text(indent + 1));
        }

        // Terminator (if present)
        if let Some(terminator) = &self.terminator {
            output.push_str(&terminator.to_mlir_text(indent + 1));
        }

        output
    }
}

impl Instruction {
    /// Create a new instruction
    #[must_use]
    pub fn new(
        operation: Operation,
        operands: Vec<SSAValue>,
        results: Vec<SSAValue>,
        result_types: Vec<Type>,
    ) -> Self {
        Self {
            operation,
            operands,
            results,
            result_types,
            regions: Vec::new(),
            attributes: BTreeMap::new(),
            location: None,
        }
    }

    /// Create a new instruction with regions
    #[must_use]
    pub fn with_regions(
        operation: Operation,
        operands: Vec<SSAValue>,
        results: Vec<SSAValue>,
        result_types: Vec<Type>,
        regions: Vec<Region>,
    ) -> Self {
        Self {
            operation,
            operands,
            results,
            result_types,
            regions,
            attributes: BTreeMap::new(),
            location: None,
        }
    }

    /// Convert instruction to MLIR text
    #[must_use]
    pub fn to_mlir_text(&self, indent: usize) -> String {
        let indent_str = "  ".repeat(indent);
        let mut output = String::new();

        output.push_str(&indent_str);

        // Results
        if !self.results.is_empty() {
            for (i, result) in self.results.iter().enumerate() {
                if i > 0 {
                    output.push_str(", ");
                }
                write!(output, "{result}").unwrap();
            }
            output.push_str(" = ");
        }

        // For builtin operations, delegate to their specific MLIR text generation
        if let crate::ops::Operation::Builtin(builtin_op) = &self.operation {
            return crate::builtin_ops::builtin_op_to_mlir_text(builtin_op, indent);
        }

        // Operation name
        output.push_str(&self.operation.name());

        // Operands
        if !self.operands.is_empty() {
            output.push('(');
            for (i, operand) in self.operands.iter().enumerate() {
                if i > 0 {
                    output.push_str(", ");
                }
                write!(output, "{operand}").unwrap();
            }
            output.push(')');
        }

        // Result types
        if !self.result_types.is_empty() {
            output.push_str(" : ");
            if !self.operands.is_empty() {
                output.push('(');
                // TODO: Add operand types
                output.push(')');
                output.push_str(" -> ");
            }
            if self.result_types.len() == 1 {
                write!(output, "{}", self.result_types[0]).unwrap();
            } else {
                output.push('(');
                for (i, ty) in self.result_types.iter().enumerate() {
                    if i > 0 {
                        output.push_str(", ");
                    }
                    write!(output, "{ty}").unwrap();
                }
                output.push(')');
            }
        }

        // Nested regions (for operations like loops, conditionals)
        if !self.regions.is_empty() {
            output.push_str(" {\n");
            for (i, region) in self.regions.iter().enumerate() {
                if i > 0 {
                    writeln!(output, "{}}} {{", "  ".repeat(indent + 1)).unwrap();
                }
                output.push_str(&region.to_mlir_text(indent + 1));
            }
            write!(output, "{indent_str}}}").unwrap();
        }

        output.push('\n');
        output
    }
}

impl Terminator {
    /// Convert terminator to MLIR text
    #[must_use]
    pub fn to_mlir_text(&self, indent: usize) -> String {
        let indent_str = "  ".repeat(indent);

        match self {
            Terminator::Return { values } => format_return(&indent_str, values),
            Terminator::Branch { target, args } => format_branch(&indent_str, target, args),
            Terminator::ConditionalBranch {
                condition,
                true_target,
                true_args,
                false_target,
                false_args,
            } => format_conditional_branch(
                &indent_str,
                *condition,
                true_target,
                true_args,
                false_target,
                false_args,
            ),
            Terminator::Switch {
                value,
                default_target,
                default_args,
                cases,
            } => format_switch(&indent_str, *value, default_target, default_args, cases),
            Terminator::Unreachable => {
                format!("{indent_str}unreachable\n")
            }
        }
    }
}

/// Helper function to format arguments list
fn format_args<T: std::fmt::Display>(args: &[T]) -> String {
    let mut output = String::new();
    if !args.is_empty() {
        output.push('(');
        for (i, arg) in args.iter().enumerate() {
            if i > 0 {
                output.push_str(", ");
            }
            write!(output, "{arg}").unwrap();
        }
        output.push(')');
    }
    output
}

/// Format return terminator
fn format_return(indent_str: &str, values: &[SSAValue]) -> String {
    let mut output = format!("{indent_str}return");
    if !values.is_empty() {
        output.push(' ');
        for (i, value) in values.iter().enumerate() {
            if i > 0 {
                output.push_str(", ");
            }
            write!(output, "{value}").unwrap();
        }
    }
    output.push('\n');
    output
}

/// Format branch terminator
fn format_branch(indent_str: &str, target: &BlockRef, args: &[SSAValue]) -> String {
    let mut output = format!("{indent_str}br ");
    output.push_str(&target.to_string());
    output.push_str(&format_args(args));
    output.push('\n');
    output
}

/// Format conditional branch terminator
fn format_conditional_branch(
    indent_str: &str,
    condition: SSAValue,
    true_target: &BlockRef,
    true_args: &[SSAValue],
    false_target: &BlockRef,
    false_args: &[SSAValue],
) -> String {
    let mut output = format!("{indent_str}cond_br {condition}, ");
    output.push_str(&true_target.to_string());
    output.push_str(&format_args(true_args));
    output.push_str(", ");
    output.push_str(&false_target.to_string());
    output.push_str(&format_args(false_args));
    output.push('\n');
    output
}

/// Format switch terminator
fn format_switch(
    indent_str: &str,
    value: SSAValue,
    default_target: &BlockRef,
    default_args: &[SSAValue],
    cases: &[SwitchCase],
) -> String {
    let mut output = format!("{indent_str}switch {value} : i32, ");
    output.push_str(&default_target.to_string());
    output.push_str(&format_args(default_args));
    output.push_str(" [\n");

    for case in cases {
        write!(output, "{}  {}: ", indent_str, case.value).unwrap();
        output.push_str(&case.target.to_string());
        output.push_str(&format_args(&case.args));
        output.push('\n');
    }

    writeln!(output, "{indent_str}]").unwrap();
    output
}

impl BlockRef {
    /// Create a block reference by index
    #[must_use]
    pub fn by_index(index: usize) -> Self {
        Self::Index(index)
    }

    /// Create a block reference by label
    pub fn by_label(label: impl Into<String>) -> Self {
        Self::Label(label.into())
    }
}

impl std::fmt::Display for BlockRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlockRef::Index(idx) => write!(f, "^bb{idx}"),
            BlockRef::Label(label) => write!(f, "^{label}"),
            BlockRef::Parent => write!(f, "^parent"),
        }
    }
}

impl std::fmt::Display for ConstantValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConstantValue::Bool(b) => write!(f, "{b}"),
            ConstantValue::Int(i) => write!(f, "{i}"),
            ConstantValue::Float(fl) => write!(f, "{fl}"),
            ConstantValue::String(s) => write!(f, "\"{s}\""),
            ConstantValue::Array(arr) => {
                write!(f, "[")?;
                for (i, elem) in arr.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{elem}")?;
                }
                write!(f, "]")
            }
            ConstantValue::Complex { real, imag } => {
                write!(f, "({real} + {imag}i)")
            }
            ConstantValue::Unit => write!(f, "()"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::*;
    use crate::types::*;

    #[test]
    fn test_module_creation() {
        let module = Module::new("test_module");
        assert_eq!(module.name, "test_module");
        // Module now has a single region with blocks containing operations
        assert_eq!(module.body.blocks.len(), 0); // No blocks initially
    }

    #[test]
    fn test_function_creation() {
        let signature = FunctionType {
            inputs: vec![qubit_type()],
            outputs: vec![bit_type()],
            variadic: false,
        };

        let function =
            Function::new_with_visibility("test_func", signature.clone(), Visibility::Public);
        assert_eq!(function.name, "test_func");
        assert_eq!(*function.signature(), signature);
        assert_eq!(function.regions().len(), 1);
    }

    #[test]
    fn test_block_creation() {
        let mut block = Block::new(Some("entry".to_string()));
        assert_eq!(block.label, Some("entry".to_string()));
        assert!(block.operations.is_empty());

        let instruction = Instruction::new(
            Operation::Quantum(QuantumOp::H),
            vec![SSAValue::new(1)],
            vec![SSAValue::new(2)],
            vec![qubit_type()],
        );

        block.add_instruction(instruction);
        assert_eq!(block.operations.len(), 1);
    }

    #[test]
    fn test_ssa_values() {
        let val1 = SSAValue::new(42);
        assert_eq!(val1.to_string(), "%42");

        let val2 = SSAValue::with_version(42, 3);
        assert_eq!(val2.to_string(), "%42.3");
    }

    #[test]
    fn test_block_ref() {
        let ref1 = BlockRef::by_index(0);
        assert_eq!(ref1.to_string(), "^bb0");

        let ref2 = BlockRef::by_label("entry");
        assert_eq!(ref2.to_string(), "^entry");
    }

    #[test]
    fn test_mlir_text_generation() {
        use crate::builtin_ops::{BuiltinOp, FuncOp, ModuleOp};
        use crate::ops::Operation;

        let mut module = ModuleOp::new("test");

        let signature = FunctionType {
            inputs: vec![qubit_type()],
            outputs: vec![bit_type()],
            variadic: false,
        };

        let func = FuncOp::new("bell_circuit", signature);
        let func_inst = Instruction::new(
            Operation::Builtin(BuiltinOp::Func(func)),
            vec![],
            vec![],
            vec![],
        );
        module.add_operation(func_inst);

        let mlir_text = crate::builtin_ops::builtin_op_to_mlir_text(&BuiltinOp::Module(module), 0);
        assert!(mlir_text.contains("module @test"));
        assert!(mlir_text.contains("func.func @bell_circuit"));
    }
}
