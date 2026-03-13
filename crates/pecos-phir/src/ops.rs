/*!
Core operation definitions for PHIR

This module defines the complete operation set for PHIR, including:
- Builtin operations (Module, Function, etc.)
- Quantum operations (gates, measurements, state preparation)
- Classical operations (arithmetic, logic, comparisons)
- Control flow operations (branches, loops, calls)
- Memory operations (allocation, load/store)
- Parsing operations (for direct parsing to PHIR)
- Custom/dialect operations

All operations follow MLIR's design where operations can contain nested regions.
*/

use pecos_core::Angle64;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Core operation enum for PHIR
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Operation {
    /// Builtin structural operations (module, func, etc.)
    Builtin(crate::builtin_ops::BuiltinOp),
    /// Quantum operations (gates, measurements, state preparation)
    Quantum(QuantumOp),
    /// Classical arithmetic and logic operations
    Classical(ClassicalOp),
    /// Control flow operations (branches, loops, function calls)
    ControlFlow(ControlFlowOp),
    /// Memory operations (allocation, load, store)
    Memory(MemoryOp),
    /// Custom/extension operations from dialects
    Custom(CustomOp),
    /// Parsing-specific operations (unresolved refs, type inference, etc.)
    Parsing(crate::parsing_ops::ParsingOp),
}

/// Quantum operations
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum QuantumOp {
    // Single-qubit gates
    /// Hadamard gate
    H,
    /// Pauli-X gate
    X,
    /// Pauli-Y gate
    Y,
    /// Pauli-Z gate
    Z,
    /// S gate (phase)
    S,
    /// S† gate
    Sdg,
    /// T gate
    T,
    /// T† gate
    Tdg,

    // Parameterized single-qubit rotations
    /// X-axis rotation
    RX(Angle64),
    /// Y-axis rotation
    RY(Angle64),
    /// Z-axis rotation
    RZ(Angle64),
    /// R1XY rotation (theta, phi) - hardware-native single-qubit gate
    R1XY(Angle64, Angle64),
    /// Arbitrary single-qubit rotation
    U3(Angle64, Angle64, Angle64), // theta, phi, lambda

    // Two-qubit gates
    /// CNOT/CX gate
    CX,
    /// Controlled-Y gate
    CY,
    /// CZ gate
    CZ,
    /// Controlled-Hadamard gate
    CH,
    /// SWAP gate
    SWAP,
    /// Controlled phase
    CPhase(Angle64),
    /// ZZ rotation
    RZZ(Angle64),

    // Multi-qubit gates
    /// Multi-controlled NOT
    MCX(usize), // number of controls
    /// Multi-controlled Z
    MCZ(usize),
    /// Toffoli (CCX)
    Toffoli,
    /// Fredkin (CSWAP)
    Fredkin,

    // Measurements
    /// Computational basis measurement
    Measure,
    /// Pauli basis measurement
    MeasurePauli(PauliBasis),
    /// Expectation value measurement
    MeasureExpectation(String), // observable name

    // State preparation
    /// Initialize qubit to |0⟩
    InitZero,
    /// Initialize qubit to |1⟩
    InitOne,
    /// Initialize qubit to |+⟩
    InitPlus,
    /// Initialize qubit to |-⟩
    InitMinus,
    /// Initialize to arbitrary state
    InitState(Vec<Complex>),

    // Resource management
    /// Allocate fresh qubit
    Alloc,
    /// Deallocate qubit (must be in |0⟩)
    Dealloc,
    /// Reset qubit to |0⟩
    Reset,
}

/// Classical arithmetic and logic operations
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ClassicalOp {
    // Arithmetic
    /// Integer addition
    Add,
    /// Integer subtraction
    Sub,
    /// Integer multiplication
    Mul,
    /// Integer division
    Div,
    /// Modulo operation
    Mod,
    /// Negation
    Neg,

    // Floating point
    /// Float addition
    FAdd,
    /// Float subtraction
    FSub,
    /// Float multiplication
    FMul,
    /// Float division
    FDiv,
    /// Float negation
    FNeg,
    /// Float square root
    Sqrt,
    /// Float power
    Pow,
    /// Trigonometric functions
    Sin,
    Cos,
    Tan,

    // Bitwise operations
    /// Bitwise AND
    And,
    /// Bitwise OR
    Or,
    /// Bitwise XOR
    Xor,
    /// Bitwise NOT
    Not,
    /// Left shift
    Shl(u32),
    /// Right shift
    Shr(u32),

    // Comparisons
    /// Equality
    Eq,
    /// Not equal
    Ne,
    /// Less than
    Lt,
    /// Less than or equal
    Le,
    /// Greater than
    Gt,
    /// Greater than or equal
    Ge,

    // Type conversions
    /// Integer to float
    IntToFloat,
    /// Float to integer
    FloatToInt,
    /// Bitcast (also used for trunc/zext)
    Bitcast,
    /// Select (ternary: `condition`, `true_value`, `false_value`)
    Select,

    // Constants
    /// Integer constant
    ConstInt(i64),
    /// Float constant
    ConstFloat(f64),
    /// Boolean constant
    ConstBool(bool),
    /// String constant
    ConstString(String),
    /// Result operation - maps measurement outcomes to output variables
    Result,
    /// Assignment operation
    Assign,
}

/// Control flow operations
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ControlFlowOp {
    /// Function call
    Call(FunctionCall),
    /// Function return
    Return,
    /// Conditional branch
    Branch(BranchType),
    /// Unconditional jump
    Jump(String), // block name
    /// Loop constructs
    Loop(LoopType),
    /// Parallel execution
    Parallel,
    /// Synchronization barrier
    Barrier,
}

/// Memory management operations
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum MemoryOp {
    /// Allocate memory
    Alloc(AllocType),
    /// Load from memory
    Load,
    /// Store to memory
    Store,
    /// Copy memory
    Copy,
    /// Get array element
    ArrayGet,
    /// Set array element
    ArraySet,
    /// Get array length
    ArrayLen,
    /// Create array from elements
    ArrayCreate,
}

/// Custom operations from dialect extensions
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CustomOp {
    /// Dialect namespace (e.g., "qec", "pulse", "chem")
    pub dialect: String,
    /// Operation name within dialect
    pub name: String,
    /// Operands (for parsing compatibility)
    pub operands: Vec<crate::phir::SSAValue>,
    /// Operation-specific attributes
    pub attributes: BTreeMap<String, crate::phir::AttributeValue>,
}

impl CustomOp {
    /// Create a new custom operation
    #[must_use]
    pub fn new(
        dialect: &str,
        name: &str,
        operands: Vec<crate::phir::SSAValue>,
        attributes: BTreeMap<String, crate::phir::AttributeValue>,
    ) -> Self {
        Self {
            dialect: dialect.to_string(),
            name: name.to_string(),
            operands,
            attributes,
        }
    }

    /// Get the dialect namespace
    #[must_use]
    pub fn dialect(&self) -> &str {
        &self.dialect
    }

    /// Get the operation name
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the operands
    #[must_use]
    pub fn operands(&self) -> &[crate::phir::SSAValue] {
        &self.operands
    }
}

// Supporting types

/// Pauli measurement basis
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum PauliBasis {
    X,
    Y,
    Z,
}

/// Complex number representation
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Complex {
    pub real: f64,
    pub imag: f64,
}

/// Function call details
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub args: Vec<ValueRef>,
}

/// Branch type
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum BranchType {
    /// if-then
    Conditional {
        condition: ValueRef,
        then_block: String,
        else_block: Option<String>,
    },
    /// switch statement
    Switch {
        value: ValueRef,
        cases: Vec<(i64, String)>, // (case_value, block_name)
        default: Option<String>,
    },
}

/// Loop constructs
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum LoopType {
    /// while loop
    While {
        condition: ValueRef,
        body_block: String,
    },
    /// for loop
    For {
        init: ValueRef,
        condition: ValueRef,
        step: ValueRef,
        body_block: String,
    },
    /// Fixed iteration count
    Repeat { count: ValueRef, body_block: String },
}

/// Memory allocation types
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AllocType {
    /// Single value
    Scalar(crate::types::Type),
    /// Array allocation
    Array(crate::types::Type, ValueRef), // type, size
    /// Stack allocation
    Stack(usize), // size in bytes
}

/// Value reference (operand in operations)
#[derive(Clone, Debug, PartialEq, Hash, Serialize, Deserialize)]
pub enum ValueRef {
    /// SSA value reference (for PHIR)
    SSA(SSAValue),
    /// Variable name reference (for parsing operations)
    Variable(String),
    /// Immediate constant
    Constant(ConstantValue),
    /// Block argument
    BlockArg(usize),
}

/// SSA value identifier
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SSAValue {
    pub id: u32,
    pub version: u32, // For phi nodes and versioning
}

/// Constant values
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ConstantValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Array(Vec<ConstantValue>),
}

impl std::hash::Hash for ConstantValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            ConstantValue::Int(i) => i.hash(state),
            ConstantValue::Float(f) => f.to_bits().hash(state), // Hash bit representation
            ConstantValue::Bool(b) => b.hash(state),
            ConstantValue::String(s) => s.hash(state),
            ConstantValue::Array(arr) => arr.hash(state),
        }
    }
}

/// Operation attributes (compile-time metadata)
#[derive(Clone, Debug, PartialEq)]
pub enum Attribute {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Array(Vec<Attribute>),
    Dict(BTreeMap<String, Attribute>),
}

impl Operation {
    /// Get the dialect namespace for this operation
    #[must_use]
    pub fn dialect(&self) -> String {
        match self {
            Operation::Builtin(_) => "builtin".to_string(),
            Operation::Quantum(_) => "quantum".to_string(),
            Operation::Classical(_) => "arith".to_string(),
            Operation::ControlFlow(_) => "control".to_string(),
            Operation::Memory(_) => "memory".to_string(),
            Operation::Custom(op) => op.dialect.clone(),
            Operation::Parsing(_) => "parse".to_string(),
        }
    }

    /// Get the operation name within its dialect
    #[must_use]
    pub fn name(&self) -> String {
        use crate::builtin_ops::BuiltinOp;
        use crate::parsing_ops::ParsingOp;
        match self {
            Operation::Builtin(op) => match op {
                BuiltinOp::Module(_) => "module".to_string(),
                BuiltinOp::Func(_) => "func.func".to_string(),
                BuiltinOp::Return(_) => "return".to_string(),
                BuiltinOp::VarDefine(_) => "var_define".to_string(),
            },
            Operation::Quantum(op) => format!("quantum.{}", op.name()),
            Operation::Classical(op) => format!("arith.{}", op.name()),
            Operation::ControlFlow(op) => format!("control.{}", op.name()),
            Operation::Memory(op) => format!("memory.{}", op.name()),
            Operation::Custom(op) => format!("{}.{}", op.dialect, op.name),
            Operation::Parsing(op) => match op {
                ParsingOp::UnresolvedCall(_) => "parse.unresolved_call".to_string(),
                ParsingOp::UnresolvedRef(_) => "parse.unresolved_ref".to_string(),
                ParsingOp::ForwardDecl(_) => "parse.forward_decl".to_string(),
                ParsingOp::ImplicitCast(_) => "parse.implicit_cast".to_string(),
                ParsingOp::ForLoop(_) => "parse.for_loop".to_string(),
                ParsingOp::IfElse(_) => "parse.if_else".to_string(),
                ParsingOp::InferType(_) => "parse.infer_type".to_string(),
            },
        }
    }

    /// Check if operation has side effects
    #[must_use]
    pub fn has_side_effects(&self) -> bool {
        match self {
            Operation::Builtin(_) | Operation::Classical(_) | Operation::Parsing(_) => false, // Structural, classical, and parsing ops have no side effects
            Operation::Quantum(op) => match op {
                QuantumOp::Measure
                | QuantumOp::MeasurePauli(_)
                | QuantumOp::MeasureExpectation(_)
                | QuantumOp::Alloc
                | QuantumOp::Dealloc
                | QuantumOp::Reset => true,
                _ => false, // Most quantum operations are unitary
            },
            Operation::Memory(_) | Operation::ControlFlow(_) | Operation::Custom(_) => true, // Memory, control flow, and custom ops have side effects (conservative for custom)
        }
    }

    /// Get expected number of operands
    #[must_use]
    pub fn operand_count(&self) -> Option<usize> {
        use crate::builtin_ops::BuiltinOp;
        match self {
            Operation::Builtin(op) => match op {
                BuiltinOp::Return(ret) => Some(ret.operands.len()),
                BuiltinOp::Module(_) | BuiltinOp::Func(_) | BuiltinOp::VarDefine(_) => Some(0),
            },
            Operation::Quantum(op) => op.operand_count(),
            Operation::Classical(op) => op.operand_count(),
            Operation::ControlFlow(op) => op.operand_count(),
            Operation::Memory(op) => op.operand_count(),
            Operation::Custom(_) | Operation::Parsing(_) => None, // Variable
        }
    }
}

impl QuantumOp {
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            QuantumOp::H => "h",
            QuantumOp::X => "x",
            QuantumOp::Y => "y",
            QuantumOp::Z => "z",
            QuantumOp::S => "s",
            QuantumOp::Sdg => "sdg",
            QuantumOp::T => "t",
            QuantumOp::Tdg => "tdg",
            QuantumOp::RX(_) => "rx",
            QuantumOp::RY(_) => "ry",
            QuantumOp::RZ(_) => "rz",
            QuantumOp::R1XY(_, _) => "r1xy",
            QuantumOp::U3(_, _, _) => "u3",
            QuantumOp::CX => "cx",
            QuantumOp::CY => "cy",
            QuantumOp::CZ => "cz",
            QuantumOp::CH => "ch",
            QuantumOp::SWAP => "swap",
            QuantumOp::CPhase(_) => "cp",
            QuantumOp::RZZ(_) => "rzz",
            QuantumOp::MCX(_) => "mcx",
            QuantumOp::MCZ(_) => "mcz",
            QuantumOp::Toffoli => "ccx",
            QuantumOp::Fredkin => "cswap",
            QuantumOp::Measure => "measure",
            QuantumOp::MeasurePauli(_) => "measure_pauli",
            QuantumOp::MeasureExpectation(_) => "measure_expectation",
            QuantumOp::InitZero => "init_zero",
            QuantumOp::InitOne => "init_one",
            QuantumOp::InitPlus => "init_plus",
            QuantumOp::InitMinus => "init_minus",
            QuantumOp::InitState(_) => "init_state",
            QuantumOp::Alloc => "alloc",
            QuantumOp::Dealloc => "dealloc",
            QuantumOp::Reset => "reset",
        }
    }

    #[must_use]
    pub fn operand_count(&self) -> Option<usize> {
        match self {
            // Single-qubit gates
            QuantumOp::H
            | QuantumOp::X
            | QuantumOp::Y
            | QuantumOp::Z
            | QuantumOp::S
            | QuantumOp::Sdg
            | QuantumOp::T
            | QuantumOp::Tdg
            | QuantumOp::RX(_)
            | QuantumOp::RY(_)
            | QuantumOp::RZ(_)
            | QuantumOp::R1XY(_, _)
            | QuantumOp::Measure
            | QuantumOp::MeasurePauli(_)
            | QuantumOp::Reset
            | QuantumOp::Dealloc
            | QuantumOp::U3(_, _, _) => Some(1),

            // Two-qubit gates
            QuantumOp::CX
            | QuantumOp::CY
            | QuantumOp::CZ
            | QuantumOp::CH
            | QuantumOp::SWAP
            | QuantumOp::CPhase(_)
            | QuantumOp::RZZ(_) => Some(2),
            QuantumOp::Toffoli | QuantumOp::Fredkin => Some(3),

            // Multi-qubit gates (variable)
            QuantumOp::MCX(n) | QuantumOp::MCZ(n) => Some(*n + 1),

            // No operands
            QuantumOp::Alloc
            | QuantumOp::InitZero
            | QuantumOp::InitOne
            | QuantumOp::InitPlus
            | QuantumOp::InitMinus => Some(0),

            // Variable operands
            QuantumOp::InitState(_) | QuantumOp::MeasureExpectation(_) => None,
        }
    }

    /// Check if operation is unitary (reversible)
    #[must_use]
    pub fn is_unitary(&self) -> bool {
        !matches!(
            self,
            QuantumOp::Measure
                | QuantumOp::MeasurePauli(_)
                | QuantumOp::MeasureExpectation(_)
                | QuantumOp::Reset
                | QuantumOp::Alloc
                | QuantumOp::Dealloc
                | QuantumOp::InitZero
                | QuantumOp::InitOne
                | QuantumOp::InitPlus
                | QuantumOp::InitMinus
                | QuantumOp::InitState(_)
        )
    }
}

impl ClassicalOp {
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            ClassicalOp::Add => "add",
            ClassicalOp::Sub => "sub",
            ClassicalOp::Mul => "mul",
            ClassicalOp::Div => "div",
            ClassicalOp::Mod => "mod",
            ClassicalOp::Neg => "neg",
            ClassicalOp::FAdd => "fadd",
            ClassicalOp::FSub => "fsub",
            ClassicalOp::FMul => "fmul",
            ClassicalOp::FDiv => "fdiv",
            ClassicalOp::FNeg => "fneg",
            ClassicalOp::Sqrt => "sqrt",
            ClassicalOp::Pow => "pow",
            ClassicalOp::Sin => "sin",
            ClassicalOp::Cos => "cos",
            ClassicalOp::Tan => "tan",
            ClassicalOp::And => "and",
            ClassicalOp::Or => "or",
            ClassicalOp::Xor => "xor",
            ClassicalOp::Not => "not",
            ClassicalOp::Shl(_) => "shl",
            ClassicalOp::Shr(_) => "shr",
            ClassicalOp::Eq => "eq",
            ClassicalOp::Ne => "ne",
            ClassicalOp::Lt => "lt",
            ClassicalOp::Le => "le",
            ClassicalOp::Gt => "gt",
            ClassicalOp::Ge => "ge",
            ClassicalOp::IntToFloat => "int_to_float",
            ClassicalOp::FloatToInt => "float_to_int",
            ClassicalOp::Bitcast => "bitcast",
            ClassicalOp::Select => "select",
            ClassicalOp::ConstInt(_) => "const_int",
            ClassicalOp::ConstFloat(_) => "const_float",
            ClassicalOp::ConstBool(_) => "const_bool",
            ClassicalOp::ConstString(_) => "const_string",
            ClassicalOp::Result => "result",
            ClassicalOp::Assign => "assign",
        }
    }

    #[must_use]
    pub fn operand_count(&self) -> Option<usize> {
        match self {
            // Binary operations
            ClassicalOp::Add
            | ClassicalOp::Sub
            | ClassicalOp::Mul
            | ClassicalOp::Div
            | ClassicalOp::Mod
            | ClassicalOp::FAdd
            | ClassicalOp::FSub
            | ClassicalOp::FMul
            | ClassicalOp::FDiv
            | ClassicalOp::Pow
            | ClassicalOp::And
            | ClassicalOp::Or
            | ClassicalOp::Xor
            | ClassicalOp::Eq
            | ClassicalOp::Ne
            | ClassicalOp::Lt
            | ClassicalOp::Le
            | ClassicalOp::Gt
            | ClassicalOp::Ge => Some(2),

            // Unary operations
            // Unary operations
            ClassicalOp::Neg
            | ClassicalOp::FNeg
            | ClassicalOp::Not
            | ClassicalOp::Sqrt
            | ClassicalOp::Sin
            | ClassicalOp::Cos
            | ClassicalOp::Tan
            | ClassicalOp::IntToFloat
            | ClassicalOp::FloatToInt
            | ClassicalOp::Bitcast
            | ClassicalOp::Shl(_)
            | ClassicalOp::Shr(_)
            | ClassicalOp::Assign => Some(1),

            // Constants (no operands)
            ClassicalOp::ConstInt(_)
            | ClassicalOp::ConstFloat(_)
            | ClassicalOp::ConstBool(_)
            | ClassicalOp::ConstString(_) => Some(0),

            // Ternary (condition + two values)
            ClassicalOp::Select => Some(3),

            // Result operation (variable number of operands)
            ClassicalOp::Result => None,
        }
    }
}

impl ControlFlowOp {
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            ControlFlowOp::Call(_) => "call",
            ControlFlowOp::Return => "return",
            ControlFlowOp::Branch(_) => "branch",
            ControlFlowOp::Jump(_) => "jump",
            ControlFlowOp::Loop(_) => "loop",
            ControlFlowOp::Parallel => "parallel",
            ControlFlowOp::Barrier => "barrier",
        }
    }

    #[must_use]
    pub fn operand_count(&self) -> Option<usize> {
        match self {
            ControlFlowOp::Call(call) => Some(call.args.len()),
            ControlFlowOp::Return | ControlFlowOp::Loop(_) => None, // Variable
            ControlFlowOp::Branch(_) => Some(1),                    // Condition
            ControlFlowOp::Jump(_) | ControlFlowOp::Parallel | ControlFlowOp::Barrier => Some(0),
        }
    }
}

impl MemoryOp {
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            MemoryOp::Alloc(_) => "alloc",
            MemoryOp::Load => "load",
            MemoryOp::Store => "store",
            MemoryOp::Copy => "copy",
            MemoryOp::ArrayGet => "array_get",
            MemoryOp::ArraySet => "array_set",
            MemoryOp::ArrayLen => "array_len",
            MemoryOp::ArrayCreate => "array_create",
        }
    }

    #[must_use]
    pub fn operand_count(&self) -> Option<usize> {
        match self {
            MemoryOp::Alloc(_) => Some(0),
            MemoryOp::Load | MemoryOp::ArrayLen => Some(1), // address/array
            MemoryOp::Store | MemoryOp::ArrayGet => Some(2), // address+value/array+index
            MemoryOp::Copy | MemoryOp::ArraySet => Some(3), // src+dst+size/array+index+value
            MemoryOp::ArrayCreate => None,                  // Variable number of elements
        }
    }
}

impl SSAValue {
    #[must_use]
    pub fn new(id: u32) -> Self {
        Self { id, version: 0 }
    }

    #[must_use]
    pub fn with_version(id: u32, version: u32) -> Self {
        Self { id, version }
    }
}

impl std::fmt::Display for SSAValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.version == 0 {
            write!(f, "%{}", self.id)
        } else {
            write!(f, "%{}.{}", self.id, self.version)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operation_names() {
        assert_eq!(Operation::Quantum(QuantumOp::H).name(), "quantum.h");
        assert_eq!(Operation::Classical(ClassicalOp::Add).name(), "arith.add");
        assert_eq!(
            Operation::ControlFlow(ControlFlowOp::Return).name(),
            "control.return"
        );
    }

    #[test]
    fn test_quantum_op_properties() {
        assert!(QuantumOp::H.is_unitary());
        assert!(!QuantumOp::Measure.is_unitary());

        assert_eq!(QuantumOp::CX.operand_count(), Some(2));
        assert_eq!(QuantumOp::Toffoli.operand_count(), Some(3));
    }

    #[test]
    fn test_ssa_value_display() {
        let val1 = SSAValue::new(42);
        assert_eq!(val1.to_string(), "%42");

        let val2 = SSAValue::with_version(42, 3);
        assert_eq!(val2.to_string(), "%42.3");
    }
}
