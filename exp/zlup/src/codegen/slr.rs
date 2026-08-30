//! SLR-AST code generation for Zluppy.
//!
//! This module generates SLR-AST JSON from Zluppy AST. The output maps directly
//! to Python's frozen dataclass structure for seamless interop with PECOS.
//!
//! ## Design Philosophy
//!
//! Low-level but safe. The JSON output is:
//! - Explicit: Every node has a `type` field - no magic
//! - Simple: Flat structure, predictable format
//! - Constrained: Only valid SLR-AST constructs can be generated
//!
//! ## Output Format
//!
//! The generated JSON maps 1:1 to Python's SLR-AST dataclasses:
//!
//! ```json
//! {
//!   "type": "Program",
//!   "name": "main",
//!   "allocator": {"type": "AllocatorDecl", "name": "q", "capacity": 2},
//!   "declarations": [...],
//!   "body": [
//!     {"type": "GateOp", "gate": "H", "targets": [{"type": "SlotRef", "allocator": "q", "index": 0}]
//!   ],
//!   "returns": []
//! }
//! ```

use std::collections::BTreeMap;
use thiserror::Error;

use crate::ast::{
    BinaryOp, Binding, Block, CallExpr, ElseBranch, Expr, FnDecl, ForRange, ForStmt, IfStmt,
    IndexExpr, Program, Stmt, TopLevelDecl,
};

// =============================================================================
// Errors
// =============================================================================

/// SLR-AST code generation errors.
#[derive(Debug, Error)]
pub enum SlrError {
    #[error("unknown gate '{name}'")]
    UnknownGate { name: String },

    #[error("undefined allocator '{name}'")]
    UndefinedAllocator { name: String },

    #[error(
        "qubit index {index} out of bounds for allocator '{allocator}' with capacity {capacity}"
    )]
    QubitIndexOutOfBounds {
        allocator: String,
        index: usize,
        capacity: usize,
    },

    #[error("expected {expected} arguments for gate '{gate}', got {got}")]
    WrongArgumentCount {
        gate: String,
        expected: usize,
        got: usize,
    },

    #[error("unsupported expression in SLR codegen")]
    UnsupportedExpression,

    #[error("invalid rotation angle")]
    InvalidAngle,

    #[error("JSON serialization error: {0}")]
    JsonError(String),
}

/// Result type for SLR-AST code generation.
pub type SlrResult<T> = Result<T, SlrError>;

// =============================================================================
// SLR-AST Node Types
// =============================================================================

/// SLR-AST Program node.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SlrProgram {
    #[serde(rename = "type")]
    pub node_type: &'static str,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allocator: Option<SlrAllocatorDecl>,
    pub declarations: Vec<SlrDeclaration>,
    /// External function declarations (FFI)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub externs: Vec<SlrExternDecl>,
    pub body: Vec<SlrStatement>,
    pub returns: Vec<SlrTypeExpr>,
}

impl SlrProgram {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            node_type: "Program",
            name: name.into(),
            allocator: None,
            declarations: Vec::new(),
            externs: Vec::new(),
            body: Vec::new(),
            returns: Vec::new(),
        }
    }
}

/// SLR-AST allocator declaration.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SlrAllocatorDecl {
    #[serde(rename = "type")]
    pub node_type: &'static str,
    pub name: String,
    pub capacity: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
}

impl SlrAllocatorDecl {
    pub fn new(name: impl Into<String>, capacity: usize) -> Self {
        Self {
            node_type: "AllocatorDecl",
            name: name.into(),
            capacity,
            parent: None,
        }
    }

    pub fn with_parent(mut self, parent: impl Into<String>) -> Self {
        self.parent = Some(parent.into());
        self
    }
}

/// SLR-AST register declaration (classical bits).
#[derive(Debug, Clone, serde::Serialize)]
pub struct SlrRegisterDecl {
    #[serde(rename = "type")]
    pub node_type: &'static str,
    pub name: String,
    pub size: usize,
    pub is_result: bool,
}

impl SlrRegisterDecl {
    pub fn new(name: impl Into<String>, size: usize) -> Self {
        Self {
            node_type: "RegisterDecl",
            name: name.into(),
            size,
            is_result: true,
        }
    }
}

/// SLR-AST declaration (allocator or register).
#[derive(Debug, Clone, serde::Serialize)]
#[serde(untagged)]
pub enum SlrDeclaration {
    Allocator(SlrAllocatorDecl),
    Register(SlrRegisterDecl),
}

/// SLR-AST slot reference (qubit in allocator).
#[derive(Debug, Clone, serde::Serialize)]
pub struct SlrSlotRef {
    #[serde(rename = "type")]
    pub node_type: &'static str,
    pub allocator: String,
    pub index: usize,
}

impl SlrSlotRef {
    pub fn new(allocator: impl Into<String>, index: usize) -> Self {
        Self {
            node_type: "SlotRef",
            allocator: allocator.into(),
            index,
        }
    }
}

/// SLR-AST bit reference (classical bit in register).
#[derive(Debug, Clone, serde::Serialize)]
pub struct SlrBitRef {
    #[serde(rename = "type")]
    pub node_type: &'static str,
    pub register: String,
    pub index: usize,
}

impl SlrBitRef {
    pub fn new(register: impl Into<String>, index: usize) -> Self {
        Self {
            node_type: "BitRef",
            register: register.into(),
            index,
        }
    }
}

/// SLR-AST gate operation.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SlrGateOp {
    #[serde(rename = "type")]
    pub node_type: &'static str,
    pub gate: &'static str,
    pub targets: Vec<SlrSlotRef>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub params: Vec<SlrExpression>,
    /// Attributes for this gate (e.g., syndrome type, layer info)
    #[serde(skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub attrs: std::collections::BTreeMap<String, SlrAttributeValue>,
}

impl SlrGateOp {
    pub fn new(gate: &'static str, targets: Vec<SlrSlotRef>) -> Self {
        Self {
            node_type: "GateOp",
            gate,
            targets,
            params: Vec::new(),
            attrs: std::collections::BTreeMap::new(),
        }
    }

    pub fn with_params(mut self, params: Vec<SlrExpression>) -> Self {
        self.params = params;
        self
    }

    pub fn with_attrs(
        mut self,
        attrs: std::collections::BTreeMap<String, SlrAttributeValue>,
    ) -> Self {
        self.attrs = attrs;
        self
    }
}

/// SLR-AST prepare operation (reset qubits).
#[derive(Debug, Clone, serde::Serialize)]
pub struct SlrPrepareOp {
    #[serde(rename = "type")]
    pub node_type: &'static str,
    pub allocator: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slots: Option<Vec<usize>>,
}

impl SlrPrepareOp {
    pub fn all(allocator: impl Into<String>) -> Self {
        Self {
            node_type: "PrepareOp",
            allocator: allocator.into(),
            slots: None,
        }
    }

    pub fn slots(allocator: impl Into<String>, slots: Vec<usize>) -> Self {
        Self {
            node_type: "PrepareOp",
            allocator: allocator.into(),
            slots: Some(slots),
        }
    }
}

/// SLR-AST measure operation.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SlrMeasureOp {
    #[serde(rename = "type")]
    pub node_type: &'static str,
    pub targets: Vec<SlrSlotRef>,
    pub results: Vec<SlrBitRef>,
    /// Result type for each measurement (u1, u8, or u64)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_type: Option<String>,
}

impl SlrMeasureOp {
    pub fn new(targets: Vec<SlrSlotRef>, results: Vec<SlrBitRef>) -> Self {
        Self {
            node_type: "MeasureOp",
            targets,
            results,
            result_type: None,
        }
    }

    pub fn with_result_type(
        targets: Vec<SlrSlotRef>,
        results: Vec<SlrBitRef>,
        result_type: &str,
    ) -> Self {
        Self {
            node_type: "MeasureOp",
            targets,
            results,
            result_type: Some(result_type.to_string()),
        }
    }
}

/// SLR-AST barrier operation.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SlrBarrierOp {
    #[serde(rename = "type")]
    pub node_type: &'static str,
    pub allocators: Vec<String>,
}

impl SlrBarrierOp {
    pub fn new(allocators: Vec<String>) -> Self {
        Self {
            node_type: "BarrierOp",
            allocators,
        }
    }
}

/// SLR-AST swap operation.
/// Swaps two values in place: @swap(&a, &b)
#[derive(Debug, Clone, serde::Serialize)]
pub struct SlrSwapOp {
    #[serde(rename = "type")]
    pub node_type: &'static str,
    pub a: SlrExpression,
    pub b: SlrExpression,
}

impl SlrSwapOp {
    pub fn new(a: SlrExpression, b: SlrExpression) -> Self {
        Self {
            node_type: "SwapOp",
            a,
            b,
        }
    }
}

/// SLR-AST if statement.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SlrIfStmt {
    #[serde(rename = "type")]
    pub node_type: &'static str,
    pub condition: SlrExpression,
    pub then_body: Vec<SlrStatement>,
    pub else_body: Vec<SlrStatement>,
}

impl SlrIfStmt {
    pub fn new(condition: SlrExpression, then_body: Vec<SlrStatement>) -> Self {
        Self {
            node_type: "IfStmt",
            condition,
            then_body,
            else_body: Vec::new(),
        }
    }

    pub fn with_else(mut self, else_body: Vec<SlrStatement>) -> Self {
        self.else_body = else_body;
        self
    }
}

/// SLR-AST for statement (bounded iteration - NASA Power of 10 compliant).
#[derive(Debug, Clone, serde::Serialize)]
pub struct SlrForStmt {
    #[serde(rename = "type")]
    pub node_type: &'static str,
    pub variable: String,
    pub start: SlrExpression,
    pub end: SlrExpression,
    pub body: Vec<SlrStatement>,
}

impl SlrForStmt {
    pub fn new(
        variable: impl Into<String>,
        start: SlrExpression,
        end: SlrExpression,
        body: Vec<SlrStatement>,
    ) -> Self {
        Self {
            node_type: "ForStmt",
            variable: variable.into(),
            start,
            end,
            body,
        }
    }
}

/// SLR-AST repeat statement (fixed iteration count).
#[derive(Debug, Clone, serde::Serialize)]
pub struct SlrRepeatStmt {
    #[serde(rename = "type")]
    pub node_type: &'static str,
    pub count: usize,
    pub body: Vec<SlrStatement>,
}

impl SlrRepeatStmt {
    pub fn new(count: usize, body: Vec<SlrStatement>) -> Self {
        Self {
            node_type: "RepeatStmt",
            count,
            body,
        }
    }
}

/// SLR-AST attribute value.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(untagged)]
pub enum SlrAttributeValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
}

/// SLR-AST tick statement (parallel gate layer).
/// Represents a time slice where all gates execute in parallel.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SlrTickStmt {
    #[serde(rename = "type")]
    pub node_type: &'static str,
    /// Optional label for the tick (e.g., "syndrome_round_1")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Attributes for this tick (e.g., round number, tick type)
    /// Using BTreeMap for deterministic ordering
    #[serde(skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub attrs: std::collections::BTreeMap<String, SlrAttributeValue>,
    /// Gates/operations within this tick (execute in parallel)
    pub body: Vec<SlrStatement>,
}

impl SlrTickStmt {
    pub fn new(body: Vec<SlrStatement>) -> Self {
        Self {
            node_type: "TickStmt",
            label: None,
            attrs: std::collections::BTreeMap::new(),
            body,
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn with_attrs(
        mut self,
        attrs: std::collections::BTreeMap<String, SlrAttributeValue>,
    ) -> Self {
        self.attrs = attrs;
        self
    }
}

/// SLR-AST log statement for debugging and tracing.
///
/// Log statements are controlled by ZLUP_LOG environment variable at runtime.
/// In release builds, they can be elided entirely at compile time.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SlrLogStmt {
    #[serde(rename = "type")]
    pub node_type: &'static str,
    /// Log level: "trace", "debug", "info", "warn", "error", or numeric
    pub level: SlrLogLevel,
    /// Namespace for filtering (module path + optional sub-namespace)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
    /// Message expression (usually an f-string)
    pub message: SlrExpression,
    /// Optional structured data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<SlrExpression>,
}

impl SlrLogStmt {
    pub fn new(level: SlrLogLevel, message: SlrExpression) -> Self {
        Self {
            node_type: "LogStmt",
            level,
            namespace: None,
            message,
            data: None,
        }
    }

    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }

    pub fn with_data(mut self, data: SlrExpression) -> Self {
        self.data = Some(data);
        self
    }
}

/// Log level for SLR log statements.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(untagged)]
pub enum SlrLogLevel {
    /// Standard named level
    Standard(String),
    /// Custom numeric level
    Numeric(i64),
}

/// SLR-AST send statement for out-of-band communication.
///
/// Unified type for `result(key, value)` and `sim.send(key, value)`.
/// The channel determines how the message is handled:
/// - "result": Program output (never elided)
/// - "sim": Simulator control (elided for hardware, emits barrier by default)
#[derive(Debug, Clone, serde::Serialize)]
pub struct SlrSendStmt {
    #[serde(rename = "type")]
    pub node_type: &'static str,
    /// Channel: "result", "sim"
    pub channel: String,
    /// Key identifying the message (e.g., "counts", "noise_enable")
    pub key: String,
    /// Optional value expression
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<SlrExpression>,
}

impl SlrSendStmt {
    pub fn new(channel: impl Into<String>, key: impl Into<String>) -> Self {
        Self {
            node_type: "SendStmt",
            channel: channel.into(),
            key: key.into(),
            value: None,
        }
    }

    pub fn with_value(mut self, value: SlrExpression) -> Self {
        self.value = Some(value);
        self
    }

    /// Create a result send (program output).
    pub fn result(key: impl Into<String>, value: SlrExpression) -> Self {
        Self::new("result", key).with_value(value)
    }

    /// Create a sim send (simulator control).
    pub fn sim(key: impl Into<String>) -> Self {
        Self::new("sim", key)
    }

    /// Create a sim send with value.
    pub fn sim_with_value(key: impl Into<String>, value: SlrExpression) -> Self {
        Self::new("sim", key).with_value(value)
    }
}

/// SLR-AST statement (union of all statement types).
#[derive(Debug, Clone, serde::Serialize)]
#[serde(untagged)]
pub enum SlrStatement {
    Gate(SlrGateOp),
    Prepare(SlrPrepareOp),
    Measure(SlrMeasureOp),
    Barrier(SlrBarrierOp),
    If(SlrIfStmt),
    For(SlrForStmt),
    Repeat(SlrRepeatStmt),
    Tick(SlrTickStmt),
    ExternCall(SlrExternCall),
    Log(SlrLogStmt),
    Send(SlrSendStmt),
    Swap(SlrSwapOp),
}

/// SLR-AST external function declaration (FFI).
/// Declares an external function that can be called from Zlup code.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SlrExternDecl {
    #[serde(rename = "type")]
    pub node_type: &'static str,
    /// Function name
    pub name: String,
    /// Library to link against (e.g., "libdecoder")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub library: Option<String>,
    /// Calling convention ("C" or "Rust")
    pub calling_convention: String,
    /// Parameter declarations with C-compatible type info
    pub params: Vec<SlrExternParam>,
    /// Return type in C-compatible format
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_type: Option<SlrCType>,
}

impl SlrExternDecl {
    pub fn new(
        name: String,
        library: Option<String>,
        calling_convention: String,
        params: Vec<SlrExternParam>,
        return_type: Option<SlrCType>,
    ) -> Self {
        Self {
            node_type: "ExternDecl",
            name,
            library,
            calling_convention,
            params,
            return_type,
        }
    }
}

/// Parameter for an external function.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SlrExternParam {
    pub name: String,
    pub ctype: SlrCType,
}

/// C-compatible type representation for FFI.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "kind")]
pub enum SlrCType {
    /// Primitive integer types
    #[serde(rename = "int")]
    Int { bits: u8, signed: bool },
    /// Floating point types
    #[serde(rename = "float")]
    Float { bits: u8 },
    /// Fixed-point angle type (maps to PECOS Angle64)
    /// Angles are represented as fractions of a full turn.
    #[serde(rename = "angle")]
    Angle { bits: u8 },
    /// Pointer type
    #[serde(rename = "pointer")]
    Pointer {
        element: Box<SlrCType>,
        is_const: bool,
    },
    /// Void type (for return)
    #[serde(rename = "void")]
    Void,
    /// Opaque type (user-defined struct, passed by name)
    #[serde(rename = "opaque")]
    Opaque { name: String },
}

/// SLR-AST external function call.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SlrExternCall {
    #[serde(rename = "type")]
    pub node_type: &'static str,
    /// Name of the external function to call
    pub function: String,
    /// Arguments to pass
    pub args: Vec<SlrExpression>,
    /// Optional: variable to store result
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
}

impl SlrExternCall {
    pub fn new(function: String, args: Vec<SlrExpression>, result: Option<String>) -> Self {
        Self {
            node_type: "ExternCall",
            function,
            args,
            result,
        }
    }
}

/// SLR-AST expression.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(untagged)]
pub enum SlrExpression {
    Literal(SlrLiteralExpr),
    Var(SlrVarExpr),
    Bit(SlrBitExpr),
    Binary(SlrBinaryExpr),
    Unary(SlrUnaryExpr),
    FString(SlrFStringExpr),
    ExternCall(Box<SlrExternCall>),
}

/// SLR-AST literal expression.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SlrLiteralExpr {
    #[serde(rename = "type")]
    pub node_type: &'static str,
    pub value: SlrLiteralValue,
}

/// Literal value variants.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(untagged)]
pub enum SlrLiteralValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    /// String value
    String(String),
    /// Angle value in turns (for a64 type)
    Angle(f64),
}

impl SlrLiteralExpr {
    pub fn int(value: i64) -> Self {
        Self {
            node_type: "LiteralExpr",
            value: SlrLiteralValue::Int(value),
        }
    }

    pub fn float(value: f64) -> Self {
        Self {
            node_type: "LiteralExpr",
            value: SlrLiteralValue::Float(value),
        }
    }

    pub fn bool(value: bool) -> Self {
        Self {
            node_type: "LiteralExpr",
            value: SlrLiteralValue::Bool(value),
        }
    }

    /// Create an angle literal (value in turns)
    pub fn angle(turns: f64) -> Self {
        Self {
            node_type: "LiteralExpr",
            value: SlrLiteralValue::Angle(turns),
        }
    }

    pub fn string(value: impl Into<String>) -> Self {
        Self {
            node_type: "LiteralExpr",
            value: SlrLiteralValue::String(value.into()),
        }
    }
}

/// SLR-AST f-string expression (string interpolation).
#[derive(Debug, Clone, serde::Serialize)]
pub struct SlrFStringExpr {
    #[serde(rename = "type")]
    pub node_type: &'static str,
    /// Parts of the f-string (text and expression parts)
    pub parts: Vec<SlrFStringPart>,
}

impl SlrFStringExpr {
    pub fn new(parts: Vec<SlrFStringPart>) -> Self {
        Self {
            node_type: "FStringExpr",
            parts,
        }
    }
}

/// Part of an f-string.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "kind")]
pub enum SlrFStringPart {
    /// Literal text
    #[serde(rename = "text")]
    Text { value: String },
    /// Interpolated expression
    #[serde(rename = "expr")]
    Expr {
        value: Box<SlrExpression>,
        #[serde(skip_serializing_if = "Option::is_none")]
        format: Option<String>,
    },
}

/// SLR-AST variable expression.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SlrVarExpr {
    #[serde(rename = "type")]
    pub node_type: &'static str,
    pub name: String,
}

impl SlrVarExpr {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            node_type: "VarExpr",
            name: name.into(),
        }
    }
}

/// SLR-AST bit expression (for conditions).
#[derive(Debug, Clone, serde::Serialize)]
pub struct SlrBitExpr {
    #[serde(rename = "type")]
    pub node_type: &'static str,
    pub register: String,
    pub index: usize,
}

impl SlrBitExpr {
    pub fn new(register: impl Into<String>, index: usize) -> Self {
        Self {
            node_type: "BitExpr",
            register: register.into(),
            index,
        }
    }
}

/// SLR-AST binary expression.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SlrBinaryExpr {
    #[serde(rename = "type")]
    pub node_type: &'static str,
    pub op: &'static str,
    pub left: Box<SlrExpression>,
    pub right: Box<SlrExpression>,
}

impl SlrBinaryExpr {
    pub fn new(op: &'static str, left: SlrExpression, right: SlrExpression) -> Self {
        Self {
            node_type: "BinaryExpr",
            op,
            left: Box::new(left),
            right: Box::new(right),
        }
    }
}

/// SLR-AST unary expression.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SlrUnaryExpr {
    #[serde(rename = "type")]
    pub node_type: &'static str,
    pub op: &'static str,
    pub operand: Box<SlrExpression>,
}

impl SlrUnaryExpr {
    pub fn new(op: &'static str, operand: SlrExpression) -> Self {
        Self {
            node_type: "UnaryExpr",
            op,
            operand: Box::new(operand),
        }
    }
}

/// SLR-AST type expression (for return types).
#[derive(Debug, Clone, serde::Serialize)]
#[serde(untagged)]
pub enum SlrTypeExpr {
    Qubit(SlrQubitType),
    Bit(SlrBitType),
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SlrQubitType {
    #[serde(rename = "type")]
    pub node_type: &'static str,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SlrBitType {
    #[serde(rename = "type")]
    pub node_type: &'static str,
}

// =============================================================================
// Gate Mapping
// =============================================================================

/// Gate information for SLR-AST.
struct GateInfo {
    /// Gate name in SLR-AST (matches GateKind enum).
    name: &'static str,
    /// Number of qubit targets.
    arity: usize,
    /// Whether this gate takes parameters.
    parameterized: bool,
}

/// Get the arity (number of qubit targets) for a gate kind.
fn gate_kind_arity(kind: &crate::ast::GateKind) -> usize {
    use crate::ast::GateKind;
    match kind {
        // Single-qubit gates
        GateKind::X
        | GateKind::Y
        | GateKind::Z
        | GateKind::H
        | GateKind::T
        | GateKind::Tdg
        | GateKind::SX
        | GateKind::SY
        | GateKind::SZ
        | GateKind::SXdg
        | GateKind::SYdg
        | GateKind::SZdg
        | GateKind::RX
        | GateKind::RY
        | GateKind::RZ
        | GateKind::F
        | GateKind::Fdg
        | GateKind::F4
        | GateKind::F4dg
        | GateKind::PZ => 1,
        // Two-qubit gates
        GateKind::CX
        | GateKind::CY
        | GateKind::CZ
        | GateKind::CH
        | GateKind::SWAP
        | GateKind::ISWAP
        | GateKind::SXX
        | GateKind::SYY
        | GateKind::SZZ
        | GateKind::SXXdg
        | GateKind::SYYdg
        | GateKind::SZZdg
        | GateKind::CRZ
        | GateKind::RZZ => 2,
        // Three-qubit gates
        GateKind::CCX => 3,
    }
}

/// Maps Zluppy gate names to SLR-AST gate info.
///
/// Zluppy uses lowercase gate names only.
/// The output name in SLR-AST remains uppercase for compatibility with downstream tools.
fn get_gate_info(name: &str) -> Option<GateInfo> {
    match name {
        // Single-qubit Pauli gates (lowercase only)
        "x" => Some(GateInfo {
            name: "X",
            arity: 1,
            parameterized: false,
        }),
        "y" => Some(GateInfo {
            name: "Y",
            arity: 1,
            parameterized: false,
        }),
        "z" => Some(GateInfo {
            name: "Z",
            arity: 1,
            parameterized: false,
        }),

        // Hadamard
        "h" => Some(GateInfo {
            name: "H",
            arity: 1,
            parameterized: false,
        }),

        // Square root gates (sx = sqrt(X), sy = sqrt(Y), sz = sqrt(Z))
        "sx" => Some(GateInfo {
            name: "SX",
            arity: 1,
            parameterized: false,
        }),
        "sy" => Some(GateInfo {
            name: "SY",
            arity: 1,
            parameterized: false,
        }),
        "sz" => Some(GateInfo {
            name: "SZ",
            arity: 1,
            parameterized: false,
        }),
        "sxdg" => Some(GateInfo {
            name: "SXdg",
            arity: 1,
            parameterized: false,
        }),
        "sydg" => Some(GateInfo {
            name: "SYdg",
            arity: 1,
            parameterized: false,
        }),
        "szdg" => Some(GateInfo {
            name: "SZdg",
            arity: 1,
            parameterized: false,
        }),

        // T gates (fourth root of Z)
        "t" => Some(GateInfo {
            name: "T",
            arity: 1,
            parameterized: false,
        }),
        "tdg" => Some(GateInfo {
            name: "Tdg",
            arity: 1,
            parameterized: false,
        }),

        // Rotation gates (parameterized)
        "rx" => Some(GateInfo {
            name: "RX",
            arity: 1,
            parameterized: true,
        }),
        "ry" => Some(GateInfo {
            name: "RY",
            arity: 1,
            parameterized: true,
        }),
        "rz" => Some(GateInfo {
            name: "RZ",
            arity: 1,
            parameterized: true,
        }),

        // Two-qubit Clifford
        "cx" => Some(GateInfo {
            name: "CX",
            arity: 2,
            parameterized: false,
        }),
        "cy" => Some(GateInfo {
            name: "CY",
            arity: 2,
            parameterized: false,
        }),
        "cz" => Some(GateInfo {
            name: "CZ",
            arity: 2,
            parameterized: false,
        }),
        "ch" => Some(GateInfo {
            name: "CH",
            arity: 2,
            parameterized: false,
        }),

        // Two-qubit rotation (parameterized)
        "rzz" => Some(GateInfo {
            name: "RZZ",
            arity: 2,
            parameterized: true,
        }),

        // Two-qubit Ising
        "sxx" => Some(GateInfo {
            name: "SXX",
            arity: 2,
            parameterized: false,
        }),
        "syy" => Some(GateInfo {
            name: "SYY",
            arity: 2,
            parameterized: false,
        }),
        "szz" => Some(GateInfo {
            name: "SZZ",
            arity: 2,
            parameterized: false,
        }),

        // Face rotations
        "f" => Some(GateInfo {
            name: "F",
            arity: 1,
            parameterized: false,
        }),
        "fdg" => Some(GateInfo {
            name: "Fdg",
            arity: 1,
            parameterized: false,
        }),
        "f4" => Some(GateInfo {
            name: "F4",
            arity: 1,
            parameterized: false,
        }),
        "f4dg" => Some(GateInfo {
            name: "F4dg",
            arity: 1,
            parameterized: false,
        }),

        // Two-qubit controlled rotation (parameterized)
        "crz" => Some(GateInfo {
            name: "CRZ",
            arity: 2,
            parameterized: true,
        }),

        // Swap gates
        "swap" => Some(GateInfo {
            name: "SWAP",
            arity: 2,
            parameterized: false,
        }),
        "iswap" => Some(GateInfo {
            name: "iSWAP",
            arity: 2,
            parameterized: false,
        }),

        // Three-qubit gates
        "ccx" => Some(GateInfo {
            name: "CCX",
            arity: 3,
            parameterized: false,
        }),

        // Two-qubit Ising dagger gates
        "sxxdg" => Some(GateInfo {
            name: "SXXdg",
            arity: 2,
            parameterized: false,
        }),
        "syydg" => Some(GateInfo {
            name: "SYYdg",
            arity: 2,
            parameterized: false,
        }),
        "szzdg" => Some(GateInfo {
            name: "SZZdg",
            arity: 2,
            parameterized: false,
        }),

        _ => None,
    }
}

// =============================================================================
// Binary/Unary Operator Mapping
// =============================================================================

fn binary_op_to_slr(op: &BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "ADD",
        BinaryOp::Sub => "SUB",
        BinaryOp::Mul => "MUL",
        BinaryOp::Div => "DIV",
        BinaryOp::Mod => "MOD",
        BinaryOp::Eq => "EQ",
        BinaryOp::Ne => "NE",
        BinaryOp::Lt => "LT",
        BinaryOp::Le => "LE",
        BinaryOp::Gt => "GT",
        BinaryOp::Ge => "GE",
        BinaryOp::And => "AND",
        BinaryOp::Or => "OR",
        BinaryOp::Orelse => "ORELSE",
        BinaryOp::BitAnd => "AND",
        BinaryOp::BitOr => "OR",
        BinaryOp::BitXor => "XOR",
        BinaryOp::Shl => "LSHIFT",
        BinaryOp::Shr => "RSHIFT",
        // Set membership operators - these are handled specially, not as SLR ops
        BinaryOp::In => "IN",
        BinaryOp::NotIn => "NOT_IN",
        // Error handling - catch is control flow, not typically a direct SLR op
        BinaryOp::Catch => "CATCH",
    }
}

// =============================================================================
// Code Generator
// =============================================================================

/// Tracks an allocator during codegen.
#[derive(Debug, Clone)]
struct AllocatorInfo {
    name: String,
    capacity: usize,
    parent: Option<String>,
}

/// Tracks a register during codegen.
#[derive(Debug, Clone)]
struct RegisterInfo {
    name: String,
    size: usize,
}

/// Log level threshold for compile-time elision.
///
/// Standard levels are spaced 100 apart for custom levels in between:
/// - trace=0, debug=100, info=200, warn=300, error=400
///
/// Set to higher values to elide more logs at compile time.
#[derive(Debug, Clone, Copy, Default)]
pub struct LogElisionLevel(pub Option<u32>);

impl LogElisionLevel {
    /// No elision - emit all logs
    pub const NONE: Self = Self(None);
    /// Elide trace logs (keep debug and above)
    pub const DEBUG: Self = Self(Some(100));
    /// Elide trace and debug logs (keep info and above)
    pub const INFO: Self = Self(Some(200));
    /// Elide trace, debug, and info logs (keep warn and above)
    pub const WARN: Self = Self(Some(300));
    /// Elide everything except errors
    pub const ERROR: Self = Self(Some(400));
    /// Elide all logs (for release builds)
    pub const ALL: Self = Self(Some(u32::MAX));

    /// Check if a log level should be elided.
    pub fn should_elide(&self, level: u32) -> bool {
        match self.0 {
            Some(threshold) => level < threshold,
            None => false,
        }
    }
}

/// SLR-AST code generator.
///
/// Walks a Zluppy AST and produces SLR-AST JSON.
pub struct SlrCodegen {
    /// Allocators by name.
    allocators: BTreeMap<String, AllocatorInfo>,
    /// Registers by name.
    registers: BTreeMap<String, RegisterInfo>,
    /// Auto-generated register counter.
    register_counter: usize,
    /// Names of external functions (for call lookup).
    extern_fns: std::collections::BTreeSet<String>,
    /// Current module path for log namespace.
    current_module: Option<String>,
    /// Minimum log level to emit (for compile-time elision).
    log_elision: LogElisionLevel,
    /// How to handle sim commands for non-simulator targets.
    sim_mode: SimMode,
}

/// How simulator commands are handled during code generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SimMode {
    /// Emit actual SimStmt (simulator target).
    #[default]
    Emit,
    /// Emit a barrier/no-op to preserve ordering (hardware default).
    Barrier,
    /// Completely elide - no output at all (explicit opt-in).
    Elide,
}

impl SlrCodegen {
    /// Create a new SLR-AST code generator.
    pub fn new() -> Self {
        Self {
            allocators: BTreeMap::new(),
            registers: BTreeMap::new(),
            register_counter: 0,
            extern_fns: std::collections::BTreeSet::new(),
            current_module: None,
            log_elision: LogElisionLevel::NONE,
            sim_mode: SimMode::Emit,
        }
    }

    /// Set the log elision level for compile-time removal of log statements.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut codegen = SlrCodegen::new();
    /// // Elide all logs below info level
    /// codegen.set_log_elision(LogElisionLevel::INFO);
    /// ```
    pub fn set_log_elision(&mut self, level: LogElisionLevel) {
        self.log_elision = level;
    }

    /// Set how sim commands are handled.
    ///
    /// - `SimMode::Emit` - Output actual SimStmt (simulator target)
    /// - `SimMode::Barrier` - Output barrier to preserve ordering (hardware default)
    /// - `SimMode::Elide` - Completely remove (explicit opt-in for max optimization)
    pub fn set_sim_mode(&mut self, mode: SimMode) {
        self.sim_mode = mode;
    }

    /// Create a new SLR-AST code generator with log elision for release builds.
    pub fn new_release() -> Self {
        let mut codegen = Self::new();
        codegen.log_elision = LogElisionLevel::ALL;
        codegen
    }

    /// Set the current module path for automatic log namespacing.
    ///
    /// The module path is used as the default namespace for log statements
    /// that don't specify an explicit namespace. Sub-namespaces specified
    /// in log statements are appended to this module path.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut codegen = SlrCodegen::new();
    /// codegen.set_module("myproject::syndromes");
    ///
    /// // log.debug(f"msg") -> namespace: "myproject::syndromes"
    /// // log.debug("round", f"msg") -> namespace: "myproject::syndromes::round"
    /// ```
    pub fn set_module(&mut self, module: impl Into<String>) {
        self.current_module = Some(module.into());
    }

    /// Compile a Zluppy program to SLR-AST.
    pub fn compile(&mut self, program: &Program) -> SlrResult<SlrProgram> {
        let mut slr_program = SlrProgram::new("main");

        // First pass: collect allocators and registers
        for decl in &program.declarations {
            self.collect_decl(decl)?;
        }

        // Build declarations
        for alloc in self.allocators.values() {
            let mut decl = SlrAllocatorDecl::new(&alloc.name, alloc.capacity);
            if let Some(ref parent) = alloc.parent {
                decl = decl.with_parent(parent);
            }
            // Set base allocator if this is the first one without a parent
            if alloc.parent.is_none() && slr_program.allocator.is_none() {
                slr_program.allocator = Some(decl.clone());
            }
            slr_program
                .declarations
                .push(SlrDeclaration::Allocator(decl));
        }

        for reg in self.registers.values() {
            slr_program
                .declarations
                .push(SlrDeclaration::Register(SlrRegisterDecl::new(
                    &reg.name, reg.size,
                )));
        }

        // Collect extern function declarations
        for decl in &program.declarations {
            if let TopLevelDecl::ExternFn(extern_fn) = decl {
                self.extern_fns.insert(extern_fn.name.clone());
                slr_program.externs.push(self.convert_extern_fn(extern_fn)?);
            }
        }

        // Second pass: convert statements
        for decl in &program.declarations {
            if let TopLevelDecl::Fn(fn_decl) = decl
                && fn_decl.name == "main"
            {
                let body = self.convert_block(&fn_decl.body)?;
                slr_program.body = body;
            }
        }

        Ok(slr_program)
    }

    /// Compile a function to SLR-AST.
    pub fn compile_function(&mut self, fn_decl: &FnDecl) -> SlrResult<SlrProgram> {
        // Collect from function body first
        self.collect_block(&fn_decl.body)?;

        let mut slr_program = SlrProgram::new(&fn_decl.name);

        // Build declarations
        for alloc in self.allocators.values() {
            let mut decl = SlrAllocatorDecl::new(&alloc.name, alloc.capacity);
            if let Some(ref parent) = alloc.parent {
                decl = decl.with_parent(parent);
            }
            if alloc.parent.is_none() && slr_program.allocator.is_none() {
                slr_program.allocator = Some(decl.clone());
            }
            slr_program
                .declarations
                .push(SlrDeclaration::Allocator(decl));
        }

        for reg in self.registers.values() {
            slr_program
                .declarations
                .push(SlrDeclaration::Register(SlrRegisterDecl::new(
                    &reg.name, reg.size,
                )));
        }

        // Convert body
        slr_program.body = self.convert_block(&fn_decl.body)?;

        Ok(slr_program)
    }

    /// Convert to JSON string.
    pub fn to_json(&self, program: &SlrProgram) -> SlrResult<String> {
        serde_json::to_string_pretty(program).map_err(|e| SlrError::JsonError(e.to_string()))
    }

    /// Convert to compact JSON string.
    pub fn to_json_compact(&self, program: &SlrProgram) -> SlrResult<String> {
        serde_json::to_string(program).map_err(|e| SlrError::JsonError(e.to_string()))
    }

    // =========================================================================
    // Collection Phase
    // =========================================================================

    fn collect_decl(&mut self, decl: &TopLevelDecl) -> SlrResult<()> {
        match decl {
            TopLevelDecl::Fn(fn_decl) if fn_decl.name == "main" => {
                self.collect_block(&fn_decl.body)?;
            }
            TopLevelDecl::Binding(binding) => self.collect_binding(binding)?,
            _ => {}
        }
        Ok(())
    }

    fn collect_binding(&mut self, binding: &Binding) -> SlrResult<()> {
        if let Some(ref value) = binding.value {
            if let Some(capacity) = self.try_extract_allocator(value) {
                self.allocators.insert(
                    binding.name.clone(),
                    AllocatorInfo {
                        name: binding.name.clone(),
                        capacity,
                        parent: None,
                    },
                );
            } else if let Some((parent, size)) = self.try_extract_child_allocator(value) {
                self.allocators.insert(
                    binding.name.clone(),
                    AllocatorInfo {
                        name: binding.name.clone(),
                        capacity: size,
                        parent: Some(parent),
                    },
                );
            }
        }
        Ok(())
    }

    fn collect_block(&mut self, block: &Block) -> SlrResult<()> {
        for stmt in &block.statements {
            self.collect_stmt(stmt)?;
        }
        Ok(())
    }

    fn collect_stmt(&mut self, stmt: &Stmt) -> SlrResult<()> {
        match stmt {
            Stmt::Binding(binding) => self.collect_binding(binding)?,
            Stmt::If(if_stmt) => {
                self.collect_block(&if_stmt.then_body)?;
                if let Some(else_branch) = &if_stmt.else_body {
                    self.collect_else_branch(else_branch)?;
                }
            }
            Stmt::For(for_stmt) => self.collect_block(&for_stmt.body)?,
            Stmt::Block(block) => self.collect_block(block)?,
            Stmt::Tick(tick_stmt) => {
                // Collect allocators from tick body
                for inner_stmt in &tick_stmt.body {
                    self.collect_stmt(inner_stmt)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn collect_else_branch(&mut self, branch: &ElseBranch) -> SlrResult<()> {
        match branch {
            ElseBranch::Else(block) => self.collect_block(block)?,
            ElseBranch::ElseIf(if_stmt) => {
                self.collect_block(&if_stmt.then_body)?;
                if let Some(else_branch) = &if_stmt.else_body {
                    self.collect_else_branch(else_branch)?;
                }
            }
        }
        Ok(())
    }

    // =========================================================================
    // Conversion Phase
    // =========================================================================

    fn convert_block(&mut self, block: &Block) -> SlrResult<Vec<SlrStatement>> {
        let mut stmts = Vec::new();
        for stmt in &block.statements {
            stmts.extend(self.convert_stmt(stmt)?);
        }
        Ok(stmts)
    }

    fn convert_stmt(&mut self, stmt: &Stmt) -> SlrResult<Vec<SlrStatement>> {
        match stmt {
            Stmt::Expr(expr_stmt) => self.convert_expr_stmt(expr_stmt),
            Stmt::If(if_stmt) => self.convert_if(if_stmt).map(|s| vec![s]),
            Stmt::For(for_stmt) => self.convert_for(for_stmt).map(|s| vec![s]),
            Stmt::Block(block) => {
                // Nested block - flatten into parent
                self.convert_block(block)
            }
            Stmt::Tick(tick_stmt) => self.convert_tick(tick_stmt).map(|s| vec![s]),
            // Quantum operations
            Stmt::Gate(gate_op) => self.convert_gate_op(gate_op).map(|s| vec![s]),
            Stmt::Prepare(prepare_op) => self.convert_prepare_op(prepare_op).map(|s| vec![s]),
            Stmt::Measure(measure_op) => self.convert_measure_op(measure_op).map(|s| vec![s]),
            Stmt::Barrier(barrier_op) => self.convert_barrier_op(barrier_op).map(|s| vec![s]),
            // Handle declarations - check for measurement calls in values
            Stmt::Binding(binding) => {
                if let Some(ref value) = binding.value {
                    self.convert_decl_value(value)
                } else {
                    Ok(vec![])
                }
            }
            _ => Ok(vec![]),
        }
    }

    /// Convert a declaration value, extracting any measurement/gate calls.
    fn convert_decl_value(&mut self, expr: &Expr) -> SlrResult<Vec<SlrStatement>> {
        // Check for measure syntax: mz(T) targets
        if let Expr::Measure(measure) = expr {
            return self.convert_measure_expr(measure);
        }
        // Old call syntax mz(T, targets) is no longer supported
        // Use mz(T) targets instead
        Ok(vec![])
    }

    /// Convert new measure syntax: mz(T) targets
    fn convert_measure_expr(
        &mut self,
        measure: &crate::ast::MeasureExpr,
    ) -> SlrResult<Vec<SlrStatement>> {
        let mut results = Vec::new();
        let mut targets = Vec::new();

        // Extract result type from the type expression
        let result_type = self.extract_measurement_result_type_from_type_expr(&measure.result_type);

        // Extract targets from the expression
        match &measure.targets {
            Expr::BracketArray(arr) => {
                // Inline array: mz(u1) [q[0], q[1], q[2]]
                for elem in &arr.elements {
                    let slot_ref = self.extract_slot_ref(elem)?;
                    let reg_name = format!("c{}", self.register_counter);
                    self.register_counter += 1;

                    if !self.registers.contains_key(&reg_name) {
                        self.registers.insert(
                            reg_name.clone(),
                            RegisterInfo {
                                name: reg_name.clone(),
                                size: 1,
                            },
                        );
                    }

                    results.push(SlrBitRef::new(&reg_name, 0));
                    targets.push(slot_ref);
                }
            }
            Expr::Index(_) => {
                // Single qubit: mz(u1) q[0]
                let slot_ref = self.extract_slot_ref(&measure.targets)?;
                let reg_name = format!("c{}", self.register_counter);
                self.register_counter += 1;

                if !self.registers.contains_key(&reg_name) {
                    self.registers.insert(
                        reg_name.clone(),
                        RegisterInfo {
                            name: reg_name.clone(),
                            size: 1,
                        },
                    );
                }

                results.push(SlrBitRef::new(&reg_name, 0));
                targets.push(slot_ref);
            }
            _ => {
                // Variable or slice - try to extract slot refs
                return Err(SlrError::UnsupportedExpression);
            }
        }

        Ok(vec![SlrStatement::Measure(SlrMeasureOp::with_result_type(
            targets,
            results,
            &result_type,
        ))])
    }

    fn convert_expr_stmt(
        &mut self,
        expr_stmt: &crate::ast::ExprStmt,
    ) -> SlrResult<Vec<SlrStatement>> {
        // Convert attributes for gates
        let attrs = self.convert_attributes(&expr_stmt.attrs);

        match &expr_stmt.expr {
            Expr::Call(call) => self.convert_call_with_attrs(call, attrs),
            Expr::BatchApply(batch) => self.convert_batch_apply_with_attrs(batch, attrs),
            Expr::Gate(gate) => self.convert_gate_expr_with_attrs(gate, attrs),
            Expr::Result(result) => self.convert_result_expr(result),
            Expr::Channel(channel) => self.convert_channel_expr(channel),
            Expr::Builtin(builtin) => self.convert_builtin_expr(builtin),
            _ => Ok(vec![]),
        }
    }

    /// Convert builtin expressions like @swap.
    fn convert_builtin_expr(
        &self,
        builtin: &crate::ast::BuiltinExpr,
    ) -> SlrResult<Vec<SlrStatement>> {
        match builtin.name.as_str() {
            "swap" => {
                // @swap(&a, &b) - emit as SwapOp
                if builtin.args.len() != 2 {
                    return Err(SlrError::WrongArgumentCount {
                        gate: "swap".to_string(),
                        expected: 2,
                        got: builtin.args.len(),
                    });
                }
                // Extract the underlying expressions (handling &x -> x)
                let a = self.convert_swap_arg(&builtin.args[0])?;
                let b = self.convert_swap_arg(&builtin.args[1])?;
                Ok(vec![SlrStatement::Swap(SlrSwapOp::new(a, b))])
            }
            _ => Ok(vec![]), // Other builtins may not produce SLR statements
        }
    }

    /// Convert a swap argument, unwrapping address-of if present.
    fn convert_swap_arg(&self, expr: &Expr) -> SlrResult<SlrExpression> {
        match expr {
            Expr::Unary(unary) => {
                use crate::ast::UnaryOp;
                match unary.op {
                    UnaryOp::AddrOf => {
                        // &x -> just use x
                        self.convert_expression(&unary.operand)
                    }
                    _ => self.convert_expression(expr),
                }
            }
            _ => self.convert_expression(expr),
        }
    }

    /// Convert a result expression to SLR send statement.
    ///
    /// Result sends are never elided - they represent the actual program output.
    fn convert_result_expr(&self, result: &crate::ast::ResultExpr) -> SlrResult<Vec<SlrStatement>> {
        let value = self.convert_expression(&result.value)?;
        Ok(vec![SlrStatement::Send(SlrSendStmt::result(
            &result.tag,
            value,
        ))])
    }

    /// Convert a channel expression to SLR statements.
    ///
    /// Handles all @emit.channel.command(...) expressions:
    /// - @emit.log.*: Logging with elision support
    /// - @emit.sim.*: Simulator control with barrier/elide modes
    /// - @emit.hw.*: Hardware messages (elided for simulator)
    /// - Custom channels: Configurable behavior
    fn convert_channel_expr(
        &mut self,
        channel: &crate::ast::ChannelExpr,
    ) -> SlrResult<Vec<SlrStatement>> {
        match channel.channel.as_str() {
            "log" => self.convert_log_channel(channel),
            "sim" => self.convert_sim_channel(channel),
            "hw" => self.convert_hw_channel(channel),
            _ => self.convert_custom_channel(channel),
        }
    }

    /// Convert @emit.log.* channel expressions.
    fn convert_log_channel(
        &self,
        channel: &crate::ast::ChannelExpr,
    ) -> SlrResult<Vec<SlrStatement>> {
        // Map command to log level
        let (level, numeric_level, level_consumes_arg) = match channel.command.as_str() {
            "trace" => (SlrLogLevel::Standard("trace".to_string()), 0u32, false),
            "debug" => (SlrLogLevel::Standard("debug".to_string()), 100, false),
            "info" => (SlrLogLevel::Standard("info".to_string()), 200, false),
            "warn" => (SlrLogLevel::Standard("warn".to_string()), 300, false),
            "error" => (SlrLogLevel::Standard("error".to_string()), 400, false),
            "at" => {
                // @emit.log.at(level, message) or @emit.log.at(level, ns, message) - first arg is level
                if let Some(level_arg) = channel.args.first() {
                    if let Ok(SlrExpression::Literal(SlrLiteralExpr {
                        value: SlrLiteralValue::Int(n),
                        ..
                    })) = self.convert_expression(level_arg.value())
                    {
                        (SlrLogLevel::Numeric(n), n as u32, true)
                    } else {
                        (SlrLogLevel::Standard("custom".to_string()), u32::MAX, true)
                    }
                } else {
                    return Ok(vec![]); // Invalid, skip
                }
            }
            _ => return Ok(vec![]), // Unknown log command
        };

        // Check if this log should be elided
        if self.log_elision.should_elide(numeric_level) {
            return Ok(vec![]);
        }

        // Start index after the level argument (if "at" command)
        let start_idx = if level_consumes_arg { 1 } else { 0 };

        // Determine namespace and message
        // If first positional is string literal AND there's another arg, first is namespace
        let (sub_namespace, message) = {
            let remaining: Vec<_> = channel
                .args
                .iter()
                .skip(start_idx)
                .filter(|arg| arg.name() != Some("data"))
                .collect();

            if remaining.len() >= 2 {
                // Check if first is plain string literal (namespace)
                if let crate::ast::Expr::StringLit(s) = remaining[0].value() {
                    let ns = Some(s.value.clone());
                    let msg = self.convert_expression(remaining[1].value())?;
                    (ns, msg)
                } else {
                    (None, self.convert_expression(remaining[0].value())?)
                }
            } else if !remaining.is_empty() {
                (None, self.convert_expression(remaining[0].value())?)
            } else {
                return Ok(vec![]); // No message
            }
        };

        let mut stmt = SlrLogStmt::new(level, message);

        // Combine module namespace with sub-namespace
        let full_namespace = match (&self.current_module, &sub_namespace) {
            (Some(module), Some(sub)) => Some(format!("{}::{}", module, sub)),
            (Some(module), None) => Some(module.clone()),
            (None, Some(sub)) => Some(sub.clone()),
            (None, None) => None,
        };
        if let Some(ns) = full_namespace {
            stmt = stmt.with_namespace(ns);
        }

        // Check for named "data" argument
        for arg in &channel.args {
            if arg.name() == Some("data") {
                stmt = stmt.with_data(self.convert_expression(arg.value())?);
            }
        }

        Ok(vec![SlrStatement::Log(stmt)])
    }

    /// Convert @emit.sim.* channel expressions.
    fn convert_sim_channel(
        &mut self,
        channel: &crate::ast::ChannelExpr,
    ) -> SlrResult<Vec<SlrStatement>> {
        match self.sim_mode {
            SimMode::Elide => Ok(vec![]),
            SimMode::Barrier => {
                let scope_allocators: Vec<String> = self.allocators.keys().cloned().collect();
                Ok(vec![SlrStatement::Barrier(SlrBarrierOp::new(
                    scope_allocators,
                ))])
            }
            SimMode::Emit => {
                // Handle @emit.sim.send(key, value) specially - key comes from first arg
                if channel.command == "send" {
                    if channel.args.len() >= 2 {
                        // Extract key from first arg (should be string literal)
                        let key_expr = channel.args[0].value();
                        let key = if let crate::ast::Expr::StringLit(s) = key_expr {
                            s.value.clone()
                        } else {
                            return Err(SlrError::UnsupportedExpression);
                        };
                        let value = self.convert_expression(channel.args[1].value())?;
                        return Ok(vec![SlrStatement::Send(SlrSendStmt::sim_with_value(
                            &key, value,
                        ))]);
                    } else if channel.args.len() == 1 {
                        // Just a key, no value
                        let key_expr = channel.args[0].value();
                        let key = if let crate::ast::Expr::StringLit(s) = key_expr {
                            s.value.clone()
                        } else {
                            return Err(SlrError::UnsupportedExpression);
                        };
                        return Ok(vec![SlrStatement::Send(SlrSendStmt::sim(&key))]);
                    }
                }

                // For other commands like @emit.sim.noise_enable(), @emit.sim.noise_disable()
                // the command name becomes the key
                let key = channel.command.clone();
                if let Some(arg) = channel.args.first() {
                    let value = self.convert_expression(arg.value())?;
                    Ok(vec![SlrStatement::Send(SlrSendStmt::sim_with_value(
                        &key, value,
                    ))])
                } else {
                    Ok(vec![SlrStatement::Send(SlrSendStmt::sim(&key))])
                }
            }
        }
    }

    /// Convert @emit.hw.* channel expressions.
    fn convert_hw_channel(
        &mut self,
        channel: &crate::ast::ChannelExpr,
    ) -> SlrResult<Vec<SlrStatement>> {
        // hw channel is opposite of sim: active for hardware, elided for simulator
        match self.sim_mode {
            SimMode::Emit => {
                // Simulator target - elide hw messages
                Ok(vec![])
            }
            SimMode::Barrier | SimMode::Elide => {
                // Hardware target - emit hw messages
                let key = channel.command.clone();
                if let Some(arg) = channel.args.first() {
                    let value = self.convert_expression(arg.value())?;
                    Ok(vec![SlrStatement::Send(
                        SlrSendStmt::new("hw", &key).with_value(value),
                    )])
                } else {
                    Ok(vec![SlrStatement::Send(SlrSendStmt::new("hw", &key))])
                }
            }
        }
    }

    /// Convert custom channel expressions.
    fn convert_custom_channel(
        &mut self,
        channel: &crate::ast::ChannelExpr,
    ) -> SlrResult<Vec<SlrStatement>> {
        // Custom channels emit as SendStmt with channel name
        // They act as barriers (sticky)
        let key = channel.command.clone();
        if let Some(arg) = channel.args.first() {
            let value = self.convert_expression(arg.value())?;
            Ok(vec![SlrStatement::Send(
                SlrSendStmt::new(&channel.channel, &key).with_value(value),
            )])
        } else {
            Ok(vec![SlrStatement::Send(SlrSendStmt::new(
                &channel.channel,
                &key,
            ))])
        }
    }

    /// Convert a gate expression: h q[0], rx(0.123) q[0], h {q[0], q[1]}
    fn convert_gate_expr_with_attrs(
        &mut self,
        gate: &crate::ast::GateExpr,
        attrs: std::collections::BTreeMap<String, SlrAttributeValue>,
    ) -> SlrResult<Vec<SlrStatement>> {
        use crate::ast::GateKind;

        // Handle PZ (prepare) specially
        if gate.kind == GateKind::PZ {
            return self.convert_prepare_from_gate_expr(gate);
        }

        // Map GateKind to SLR gate name
        let gate_name: &'static str = match gate.kind {
            GateKind::X => "X",
            GateKind::Y => "Y",
            GateKind::Z => "Z",
            GateKind::H => "H",
            GateKind::T => "T",
            GateKind::Tdg => "Tdg",
            GateKind::SX => "SX",
            GateKind::SY => "SY",
            GateKind::SZ => "SZ",
            GateKind::SXdg => "SXdg",
            GateKind::SYdg => "SYdg",
            GateKind::SZdg => "SZdg",
            GateKind::RX => "RX",
            GateKind::RY => "RY",
            GateKind::RZ => "RZ",
            GateKind::CX => "CX",
            GateKind::CY => "CY",
            GateKind::CZ => "CZ",
            GateKind::CH => "CH",
            GateKind::SWAP => "SWAP",
            GateKind::ISWAP => "ISWAP",
            GateKind::SXX => "SXX",
            GateKind::SYY => "SYY",
            GateKind::SZZ => "SZZ",
            GateKind::SXXdg => "SXXdg",
            GateKind::SYYdg => "SYYdg",
            GateKind::SZZdg => "SZZdg",
            GateKind::CRZ => "CRZ",
            GateKind::RZZ => "RZZ",
            GateKind::CCX => "CCX",
            GateKind::F => "F",
            GateKind::Fdg => "Fdg",
            GateKind::F4 => "F4",
            GateKind::F4dg => "F4dg",
            GateKind::PZ => unreachable!(), // Handled above
        };

        // Convert parameters
        let params: Vec<SlrExpression> = gate
            .params
            .iter()
            .map(|p| self.convert_expression(p))
            .collect::<SlrResult<Vec<_>>>()?;

        // Convert target based on its type
        match &gate.target {
            // Single qubit: q[0]
            Expr::Index(_) => {
                let slot_ref = self.extract_slot_ref(&gate.target)?;
                Ok(vec![SlrStatement::Gate(
                    SlrGateOp::new(gate_name, vec![slot_ref])
                        .with_params(params)
                        .with_attrs(attrs),
                )])
            }
            // Batch with set: {q[0], q[1]}
            Expr::Set(set) => {
                let arity = gate_kind_arity(&gate.kind);
                let mut statements = Vec::new();
                for elem in &set.elements {
                    match (arity, elem) {
                        // Single qubit in batch (arity 1 only)
                        (1, Expr::Index(_)) => {
                            let slot_ref = self.extract_slot_ref(elem)?;
                            statements.push(SlrStatement::Gate(
                                SlrGateOp::new(gate_name, vec![slot_ref])
                                    .with_params(params.clone())
                                    .with_attrs(attrs.clone()),
                            ));
                        }
                        // Tuple for multi-qubit gates: (q[0], q[1], ...)
                        (expected, Expr::Tuple(tuple)) if tuple.elements.len() == expected => {
                            let targets = tuple
                                .elements
                                .iter()
                                .map(|element| self.extract_slot_ref(element))
                                .collect::<SlrResult<Vec<_>>>()?;
                            statements.push(SlrStatement::Gate(
                                SlrGateOp::new(gate_name, targets)
                                    .with_params(params.clone())
                                    .with_attrs(attrs.clone()),
                            ));
                        }
                        // Wrong arity: single qubit for a multi-qubit gate
                        (expected, Expr::Index(_)) => {
                            return Err(SlrError::WrongArgumentCount {
                                gate: gate_name.to_string(),
                                expected,
                                got: 1,
                            });
                        }
                        // Wrong arity: tuple has the wrong number of qubits
                        (expected, Expr::Tuple(tuple)) => {
                            return Err(SlrError::WrongArgumentCount {
                                gate: gate_name.to_string(),
                                expected,
                                got: tuple.elements.len(),
                            });
                        }
                        (_, _) => return Err(SlrError::UnsupportedExpression),
                    }
                }
                Ok(statements)
            }
            // Multi-qubit gate with tuple: (q[0], q[1], ...)
            Expr::Tuple(tuple) => {
                let expected = gate_kind_arity(&gate.kind);
                if tuple.elements.len() != expected {
                    return Err(SlrError::WrongArgumentCount {
                        gate: gate_name.to_string(),
                        expected,
                        got: tuple.elements.len(),
                    });
                }
                let targets = tuple
                    .elements
                    .iter()
                    .map(|element| self.extract_slot_ref(element))
                    .collect::<SlrResult<Vec<_>>>()?;
                Ok(vec![SlrStatement::Gate(
                    SlrGateOp::new(gate_name, targets)
                        .with_params(params)
                        .with_attrs(attrs),
                )])
            }
            // Batch with bracket array: [q[0], q[1]]
            Expr::BracketArray(arr) => {
                let arity = gate_kind_arity(&gate.kind);
                let mut statements = Vec::new();
                for elem in &arr.elements {
                    match (arity, elem) {
                        (1, Expr::Index(_)) => {
                            let slot_ref = self.extract_slot_ref(elem)?;
                            statements.push(SlrStatement::Gate(
                                SlrGateOp::new(gate_name, vec![slot_ref])
                                    .with_params(params.clone())
                                    .with_attrs(attrs.clone()),
                            ));
                        }
                        (expected, Expr::Tuple(tuple)) if tuple.elements.len() == expected => {
                            let targets = tuple
                                .elements
                                .iter()
                                .map(|element| self.extract_slot_ref(element))
                                .collect::<SlrResult<Vec<_>>>()?;
                            statements.push(SlrStatement::Gate(
                                SlrGateOp::new(gate_name, targets)
                                    .with_params(params.clone())
                                    .with_attrs(attrs.clone()),
                            ));
                        }
                        (expected, Expr::Index(_)) => {
                            return Err(SlrError::WrongArgumentCount {
                                gate: gate_name.to_string(),
                                expected,
                                got: 1,
                            });
                        }
                        (expected, Expr::Tuple(tuple)) => {
                            return Err(SlrError::WrongArgumentCount {
                                gate: gate_name.to_string(),
                                expected,
                                got: tuple.elements.len(),
                            });
                        }
                        (_, _) => return Err(SlrError::UnsupportedExpression),
                    }
                }
                Ok(statements)
            }
            // Allocator reference for "apply to all": h q (single-qubit gates only)
            Expr::Ident(ident) => {
                // Get allocator info to determine capacity
                let alloc = self.allocators.get(&ident.name).ok_or_else(|| {
                    SlrError::UndefinedAllocator {
                        name: ident.name.clone(),
                    }
                })?;
                let capacity = alloc.capacity;
                let alloc_name = alloc.name.clone();

                // Apply gate to all qubits in allocator
                let mut statements = Vec::new();
                for i in 0..capacity {
                    statements.push(SlrStatement::Gate(
                        SlrGateOp::new(gate_name, vec![SlrSlotRef::new(&alloc_name, i)])
                            .with_params(params.clone())
                            .with_attrs(attrs.clone()),
                    ));
                }
                Ok(statements)
            }
            _ => Err(SlrError::UnsupportedExpression),
        }
    }

    /// Convert pz (prepare) gate expression to prepare operation
    fn convert_prepare_from_gate_expr(
        &mut self,
        gate: &crate::ast::GateExpr,
    ) -> SlrResult<Vec<SlrStatement>> {
        match &gate.target {
            // pz q - prepare all qubits in allocator
            Expr::Ident(ident) => Ok(vec![SlrStatement::Prepare(SlrPrepareOp::all(&ident.name))]),
            // pz q[0] - prepare single qubit
            Expr::Index(_) => {
                let slot_ref = self.extract_slot_ref(&gate.target)?;
                Ok(vec![SlrStatement::Prepare(SlrPrepareOp::slots(
                    slot_ref.allocator,
                    vec![slot_ref.index],
                ))])
            }
            // pz {q[0], q[1]} - prepare batch (set)
            Expr::Set(set) => {
                let mut slots_by_alloc: std::collections::BTreeMap<String, Vec<usize>> =
                    std::collections::BTreeMap::new();
                for elem in &set.elements {
                    let slot_ref = self.extract_slot_ref(elem)?;
                    slots_by_alloc
                        .entry(slot_ref.allocator)
                        .or_default()
                        .push(slot_ref.index);
                }
                let mut statements = Vec::new();
                for (alloc, slots) in slots_by_alloc {
                    statements.push(SlrStatement::Prepare(SlrPrepareOp::slots(alloc, slots)));
                }
                Ok(statements)
            }
            // pz [q[0], q[1]] - prepare batch (array)
            Expr::BracketArray(arr) => {
                let mut slots_by_alloc: std::collections::BTreeMap<String, Vec<usize>> =
                    std::collections::BTreeMap::new();
                for elem in &arr.elements {
                    let slot_ref = self.extract_slot_ref(elem)?;
                    slots_by_alloc
                        .entry(slot_ref.allocator)
                        .or_default()
                        .push(slot_ref.index);
                }
                let mut statements = Vec::new();
                for (alloc, slots) in slots_by_alloc {
                    statements.push(SlrStatement::Prepare(SlrPrepareOp::slots(alloc, slots)));
                }
                Ok(statements)
            }
            _ => Err(SlrError::UnsupportedExpression),
        }
    }

    /// Convert a batch apply expression: h { q[0], q[1] } or rz(pi/4) { q[0], q[1] }
    fn convert_batch_apply_with_attrs(
        &mut self,
        batch: &crate::ast::BatchApplyExpr,
        attrs: std::collections::BTreeMap<String, SlrAttributeValue>,
    ) -> SlrResult<Vec<SlrStatement>> {
        // Extract gate info and params from the operation
        let (gate_name, params) = match &batch.operation {
            Expr::Ident(ident) => (ident.name.clone(), Vec::new()),
            Expr::Call(call) => {
                if let Expr::Ident(ident) = &call.callee {
                    let mut params = Vec::new();
                    for arg in &call.args {
                        params.push(self.convert_expression(arg)?);
                    }
                    (ident.name.clone(), params)
                } else {
                    return Err(SlrError::UnsupportedExpression);
                }
            }
            _ => return Err(SlrError::UnsupportedExpression),
        };

        // Get gate info
        let Some(gate_info) = get_gate_info(&gate_name) else {
            return Err(SlrError::UnsupportedExpression);
        };

        // Convert batch targets to statements
        self.convert_batch_gate_with_attrs(gate_info, &batch.targets, params, attrs)
    }

    /// Convert AST attributes to SLR attributes.
    fn convert_attributes(
        &self,
        attrs: &[crate::ast::Attribute],
    ) -> std::collections::BTreeMap<String, SlrAttributeValue> {
        let mut result = std::collections::BTreeMap::new();
        for attr in attrs {
            let value = match &attr.value {
                Some(crate::ast::AttributeValue::Bool(b)) => SlrAttributeValue::Bool(*b),
                Some(crate::ast::AttributeValue::Int(i)) => SlrAttributeValue::Int(*i),
                Some(crate::ast::AttributeValue::Float(f)) => SlrAttributeValue::Float(*f),
                Some(crate::ast::AttributeValue::String(s)) => SlrAttributeValue::String(s.clone()),
                Some(crate::ast::AttributeValue::Ident(s)) => SlrAttributeValue::String(s.clone()),
                None => SlrAttributeValue::Bool(true), // Flag attributes default to true
            };
            result.insert(attr.name.clone(), value);
        }
        result
    }

    fn convert_call(&mut self, call: &CallExpr) -> SlrResult<Vec<SlrStatement>> {
        self.convert_call_with_attrs(call, std::collections::BTreeMap::new())
    }

    fn convert_call_with_attrs(
        &mut self,
        call: &CallExpr,
        attrs: std::collections::BTreeMap<String, SlrAttributeValue>,
    ) -> SlrResult<Vec<SlrStatement>> {
        let name = self.extract_call_name(&call.callee)?;

        // Check for special operations (lowercase only)
        match name.as_str() {
            // mz uses new syntax: mz(T) targets - handled via Expr::Measure
            // Old mz(T, target) call syntax is no longer supported
            // pz = prepare +Z eigenstate (reset)
            "pz" => return self.convert_reset(call),
            "barrier" => return self.convert_barrier(call),
            _ => {}
        }

        // Check for external function calls
        if self.extern_fns.contains(&name) {
            let args: Result<Vec<SlrExpression>, SlrError> = call
                .args
                .iter()
                .map(|arg| self.convert_expression(arg))
                .collect();
            return Ok(vec![SlrStatement::ExternCall(SlrExternCall::new(
                name, args?, None, // Result variable set later during assignment handling
            ))]);
        }

        // Check for gate calls
        let Some(gate_info) = get_gate_info(&name) else {
            return Ok(vec![]);
        };

        // For parameterized gates: angle comes first, then qubits
        // rz(1.57, q[0]) or rz(1.57, [q[0], q[1])
        let (params, qubit_args) = if gate_info.parameterized {
            if call.args.is_empty() {
                return Err(SlrError::WrongArgumentCount {
                    gate: name,
                    expected: gate_info.arity + 1,
                    got: 0,
                });
            }
            let param = self.convert_expression(&call.args[0])?;
            (vec![param], &call.args[1..])
        } else {
            (Vec::new(), &call.args[..])
        };

        // Check if qubit argument is a Set or BracketArray (batch operation)
        if !qubit_args.is_empty() {
            if let Expr::Set(set_expr) = &qubit_args[0] {
                return self.convert_batch_gate_with_attrs(
                    gate_info,
                    &set_expr.elements,
                    params,
                    attrs,
                );
            }
            if let Expr::BracketArray(arr_expr) = &qubit_args[0] {
                return self.convert_batch_gate_with_attrs(
                    gate_info,
                    &arr_expr.elements,
                    params,
                    attrs,
                );
            }
        }

        // Standard single-target gate call
        // Validate argument count
        if qubit_args.len() != gate_info.arity {
            return Err(SlrError::WrongArgumentCount {
                gate: name,
                expected: if gate_info.parameterized {
                    gate_info.arity + 1
                } else {
                    gate_info.arity
                },
                got: call.args.len(),
            });
        }

        // Extract qubit targets
        let mut targets = Vec::with_capacity(gate_info.arity);
        for arg in qubit_args.iter().take(gate_info.arity) {
            let slot_ref = self.extract_slot_ref(arg)?;
            targets.push(slot_ref);
        }

        Ok(vec![SlrStatement::Gate(
            SlrGateOp::new(gate_info.name, targets)
                .with_params(params)
                .with_attrs(attrs),
        )])
    }

    /// Convert a batch gate operation into multiple SLR gates.
    fn convert_batch_gate(
        &mut self,
        gate_info: GateInfo,
        elements: &[Expr],
        params: Vec<SlrExpression>,
    ) -> SlrResult<Vec<SlrStatement>> {
        self.convert_batch_gate_with_attrs(
            gate_info,
            elements,
            params,
            std::collections::BTreeMap::new(),
        )
    }

    /// Convert a batch gate operation with attributes into multiple SLR gates.
    /// Works with both Set and BracketArray expressions.
    fn convert_batch_gate_with_attrs(
        &mut self,
        gate_info: GateInfo,
        elements: &[Expr],
        params: Vec<SlrExpression>,
        attrs: std::collections::BTreeMap<String, SlrAttributeValue>,
    ) -> SlrResult<Vec<SlrStatement>> {
        let mut statements = Vec::new();

        if gate_info.arity == 1 {
            // Single-qubit gate: each element is a qubit
            for elem in elements {
                let slot_ref = self.extract_slot_ref(elem)?;
                statements.push(SlrStatement::Gate(
                    SlrGateOp::new(gate_info.name, vec![slot_ref])
                        .with_params(params.clone())
                        .with_attrs(attrs.clone()),
                ));
            }
        } else if gate_info.arity == 2 {
            // Two-qubit gate: each element is a tuple (control, target)
            for elem in elements {
                if let Expr::Tuple(tuple) = elem {
                    if tuple.elements.len() == 2 {
                        let control = self.extract_slot_ref(&tuple.elements[0])?;
                        let target = self.extract_slot_ref(&tuple.elements[1])?;
                        statements.push(SlrStatement::Gate(
                            SlrGateOp::new(gate_info.name, vec![control, target])
                                .with_params(params.clone())
                                .with_attrs(attrs.clone()),
                        ));
                    } else {
                        return Err(SlrError::UnsupportedExpression);
                    }
                } else {
                    return Err(SlrError::UnsupportedExpression);
                }
            }
        } else {
            return Err(SlrError::UnsupportedExpression);
        }

        Ok(statements)
    }

    fn convert_measure(&mut self, call: &CallExpr) -> SlrResult<Vec<SlrStatement>> {
        // Typed measurement: mz(type, target) where:
        // - type is u1, u8, u64, []u1, []u8, []u64
        // - target is q[0] or &[q[0], q[1], ...]

        // Check for typed measurement syntax (2 args: type + target)
        if call.args.len() == 2 {
            return self.convert_typed_measure(call);
        }

        // Legacy: measure(q[0]) or measure(q[0], q[1], ...)
        let mut targets = Vec::new();
        let mut results = Vec::new();

        for arg in &call.args {
            let slot_ref = self.extract_slot_ref(arg)?;

            // Auto-generate result register if needed
            let reg_name = format!("c{}", self.register_counter);
            self.register_counter += 1;

            if !self.registers.contains_key(&reg_name) {
                self.registers.insert(
                    reg_name.clone(),
                    RegisterInfo {
                        name: reg_name.clone(),
                        size: 1,
                    },
                );
            }

            results.push(SlrBitRef::new(&reg_name, 0));
            targets.push(slot_ref);
        }

        Ok(vec![SlrStatement::Measure(SlrMeasureOp::new(
            targets, results,
        ))])
    }

    /// Convert a typed measurement call: mz(type, target)
    fn convert_typed_measure(&mut self, call: &CallExpr) -> SlrResult<Vec<SlrStatement>> {
        // First argument is the type
        let result_type = self.extract_measurement_result_type(&call.args[0]);
        // Second argument is the target(s)
        let target_arg = &call.args[1];

        let mut targets = Vec::new();
        let mut results = Vec::new();

        // Extract targets from second argument
        match target_arg {
            // Single qubit: q[0]
            Expr::Index(_) => {
                let slot_ref = self.extract_slot_ref(target_arg)?;
                let reg_name = format!("c{}", self.register_counter);
                self.register_counter += 1;

                if !self.registers.contains_key(&reg_name) {
                    self.registers.insert(
                        reg_name.clone(),
                        RegisterInfo {
                            name: reg_name.clone(),
                            size: 1,
                        },
                    );
                }

                results.push(SlrBitRef::new(&reg_name, 0));
                targets.push(slot_ref);
            }

            // Address-of array: &[q[0], q[1], ...]
            Expr::Unary(unary) => {
                if let crate::ast::UnaryOp::AddrOf = unary.op {
                    if let Expr::BracketArray(arr) = &unary.operand {
                        for elem in &arr.elements {
                            let slot_ref = self.extract_slot_ref(elem)?;
                            let reg_name = format!("c{}", self.register_counter);
                            self.register_counter += 1;

                            if !self.registers.contains_key(&reg_name) {
                                self.registers.insert(
                                    reg_name.clone(),
                                    RegisterInfo {
                                        name: reg_name.clone(),
                                        size: 1,
                                    },
                                );
                            }

                            results.push(SlrBitRef::new(&reg_name, 0));
                            targets.push(slot_ref);
                        }
                    } else {
                        return Err(SlrError::UnsupportedExpression);
                    }
                } else {
                    return Err(SlrError::UnsupportedExpression);
                }
            }

            _ => return Err(SlrError::UnsupportedExpression),
        }

        Ok(vec![SlrStatement::Measure(SlrMeasureOp::with_result_type(
            targets,
            results,
            &result_type,
        ))])
    }

    /// Extract measurement result type from a type expression.
    fn extract_measurement_result_type(&self, expr: &Expr) -> String {
        match expr {
            Expr::Ident(ident) => ident.name.clone(),
            // For slice types []u1, []u8, []u64 - extract the element type
            Expr::SlotRef(slot_ref) => slot_ref.allocator.clone(),
            _ => "u1".to_string(), // Default to u1
        }
    }

    /// Extract measurement result type from a TypeExpr (for new mz(T) target syntax).
    fn extract_measurement_result_type_from_type_expr(
        &self,
        type_expr: &crate::ast::TypeExpr,
    ) -> String {
        use crate::ast::{PrimitiveType, TypeExpr};
        match type_expr {
            TypeExpr::Primitive(prim) => match prim {
                PrimitiveType::UInt { bits } => format!("u{bits}"),
                PrimitiveType::IInt { bits } => format!("i{bits}"),
                _ => "u1".to_string(), // Default for other types
            },
            TypeExpr::Named(path) => {
                // For named types, return the path as string
                path.segments.join("::")
            }
            TypeExpr::Array(arr) => {
                // For array types like []u1, extract the element type
                self.extract_measurement_result_type_from_type_expr(&arr.element)
            }
            _ => "u1".to_string(), // Default to u1
        }
    }

    fn convert_reset(&mut self, call: &CallExpr) -> SlrResult<Vec<SlrStatement>> {
        // reset(q[0]) or reset with allocator
        if call.args.is_empty() {
            return Ok(vec![]);
        }

        let slot_ref = self.extract_slot_ref(&call.args[0])?;
        Ok(vec![SlrStatement::Prepare(SlrPrepareOp::slots(
            slot_ref.allocator,
            vec![slot_ref.index],
        ))])
    }

    fn convert_barrier(&mut self, call: &CallExpr) -> SlrResult<Vec<SlrStatement>> {
        let mut allocators = Vec::new();
        for arg in &call.args {
            if let Expr::Ident(ident) = arg {
                allocators.push(ident.name.clone());
            }
        }
        Ok(vec![SlrStatement::Barrier(SlrBarrierOp::new(allocators))])
    }

    // Conversion functions for AST quantum statement types

    fn convert_gate_op(&mut self, gate_op: &crate::ast::GateOp) -> SlrResult<SlrStatement> {
        use crate::ast::GateKind;

        // Map AST GateKind to SLR gate name
        let gate_name: &'static str = match gate_op.kind {
            GateKind::X => "X",
            GateKind::Y => "Y",
            GateKind::Z => "Z",
            GateKind::H => "H",
            GateKind::T => "T",
            GateKind::Tdg => "Tdg",
            GateKind::SX => "SX",
            GateKind::SY => "SY",
            GateKind::SZ => "SZ",
            GateKind::SXdg => "SXdg",
            GateKind::SYdg => "SYdg",
            GateKind::SZdg => "SZdg",
            GateKind::RX => "RX",
            GateKind::RY => "RY",
            GateKind::RZ => "RZ",
            GateKind::CX => "CX",
            GateKind::CY => "CY",
            GateKind::CZ => "CZ",
            GateKind::CH => "CH",
            GateKind::SWAP => "SWAP",
            GateKind::ISWAP => "ISWAP",
            GateKind::SXX => "SXX",
            GateKind::SYY => "SYY",
            GateKind::SZZ => "SZZ",
            GateKind::SXXdg => "SXXdg",
            GateKind::SYYdg => "SYYdg",
            GateKind::SZZdg => "SZZdg",
            GateKind::CRZ => "CRZ",
            GateKind::RZZ => "RZZ",
            GateKind::CCX => "CCX",
            GateKind::F => "F",
            GateKind::Fdg => "Fdg",
            GateKind::F4 => "F4",
            GateKind::F4dg => "F4dg",
            GateKind::PZ => "PZ", // Prepare operation, handled specially
        };

        // Convert targets
        let targets: Vec<SlrSlotRef> = gate_op
            .targets
            .iter()
            .map(|slot_ref| {
                let index = self.extract_const_index(&slot_ref.index).unwrap_or(0);
                SlrSlotRef::new(&slot_ref.allocator, index)
            })
            .collect();

        // Convert parameters
        let params: Vec<SlrExpression> = gate_op
            .params
            .iter()
            .map(|expr| self.convert_expression(expr))
            .collect::<SlrResult<Vec<_>>>()?;

        Ok(SlrStatement::Gate(
            SlrGateOp::new(gate_name, targets).with_params(params),
        ))
    }

    fn convert_prepare_op(
        &mut self,
        prepare_op: &crate::ast::PrepareOp,
    ) -> SlrResult<SlrStatement> {
        if let Some(ref slots) = prepare_op.slots {
            let slot_indices: Vec<usize> = slots.iter().map(|&s| s as usize).collect();
            Ok(SlrStatement::Prepare(SlrPrepareOp::slots(
                &prepare_op.allocator,
                slot_indices,
            )))
        } else {
            Ok(SlrStatement::Prepare(SlrPrepareOp::all(
                &prepare_op.allocator,
            )))
        }
    }

    fn convert_measure_op(
        &mut self,
        measure_op: &crate::ast::MeasureOp,
    ) -> SlrResult<SlrStatement> {
        let mut targets = Vec::new();
        let mut results = Vec::new();

        for slot_ref in &measure_op.targets {
            let index = self.extract_const_index(&slot_ref.index).unwrap_or(0);
            targets.push(SlrSlotRef::new(&slot_ref.allocator, index));

            // Auto-generate result register if needed
            let reg_name = format!("c{}", self.register_counter);
            self.register_counter += 1;

            if !self.registers.contains_key(&reg_name) {
                self.registers.insert(
                    reg_name.clone(),
                    RegisterInfo {
                        name: reg_name.clone(),
                        size: 1,
                    },
                );
            }

            results.push(SlrBitRef::new(&reg_name, 0));
        }

        Ok(SlrStatement::Measure(SlrMeasureOp::new(targets, results)))
    }

    fn convert_barrier_op(
        &mut self,
        barrier_op: &crate::ast::BarrierOp,
    ) -> SlrResult<SlrStatement> {
        Ok(SlrStatement::Barrier(SlrBarrierOp::new(
            barrier_op.allocators.clone(),
        )))
    }

    fn convert_if(&mut self, if_stmt: &IfStmt) -> SlrResult<SlrStatement> {
        let condition = self.convert_expression(&if_stmt.condition)?;
        let then_body = self.convert_block(&if_stmt.then_body)?;

        let else_body = if let Some(ref else_branch) = if_stmt.else_body {
            self.convert_else_branch(else_branch)?
        } else {
            Vec::new()
        };

        Ok(SlrStatement::If(
            SlrIfStmt::new(condition, then_body).with_else(else_body),
        ))
    }

    fn convert_else_branch(&mut self, branch: &ElseBranch) -> SlrResult<Vec<SlrStatement>> {
        match branch {
            ElseBranch::Else(block) => self.convert_block(block),
            ElseBranch::ElseIf(if_stmt) => {
                let condition = self.convert_expression(&if_stmt.condition)?;
                let then_body = self.convert_block(&if_stmt.then_body)?;
                let else_body = if let Some(ref else_branch) = if_stmt.else_body {
                    self.convert_else_branch(else_branch)?
                } else {
                    Vec::new()
                };
                Ok(vec![SlrStatement::If(
                    SlrIfStmt::new(condition, then_body).with_else(else_body),
                )])
            }
        }
    }

    fn convert_tick(&mut self, tick_stmt: &crate::ast::TickStmt) -> SlrResult<SlrStatement> {
        // Convert all statements within the tick
        let mut body = Vec::new();
        for stmt in &tick_stmt.body {
            body.extend(self.convert_stmt(stmt)?);
        }

        // Create tick statement with optional label
        let mut slr_tick = SlrTickStmt::new(body);
        if let Some(ref label) = tick_stmt.label {
            slr_tick = slr_tick.with_label(label.clone());
        }

        // Convert attributes
        if !tick_stmt.attrs.is_empty() {
            let mut attrs = std::collections::BTreeMap::new();
            for attr in &tick_stmt.attrs {
                let value = match &attr.value {
                    Some(crate::ast::AttributeValue::Bool(b)) => SlrAttributeValue::Bool(*b),
                    Some(crate::ast::AttributeValue::Int(i)) => SlrAttributeValue::Int(*i),
                    Some(crate::ast::AttributeValue::Float(f)) => SlrAttributeValue::Float(*f),
                    Some(crate::ast::AttributeValue::String(s)) => {
                        SlrAttributeValue::String(s.clone())
                    }
                    Some(crate::ast::AttributeValue::Ident(s)) => {
                        SlrAttributeValue::String(s.clone())
                    }
                    None => SlrAttributeValue::Bool(true), // Flag attributes default to true
                };
                attrs.insert(attr.name.clone(), value);
            }
            slr_tick = slr_tick.with_attrs(attrs);
        }

        Ok(SlrStatement::Tick(slr_tick))
    }

    fn convert_for(&mut self, for_stmt: &ForStmt) -> SlrResult<SlrStatement> {
        // For SLR, convert bounded for loops to repeat statements
        // For unbounded, use ForStmt
        if let Some(count) = self.try_extract_repeat_count(for_stmt) {
            let body = self.convert_block(&for_stmt.body)?;
            return Ok(SlrStatement::Repeat(SlrRepeatStmt::new(count, body)));
        }

        // General for loop - extract from ForRange
        let (start, end) = match &for_stmt.range {
            ForRange::Range { start, end } => (
                self.convert_expression(start)?,
                self.convert_expression(end)?,
            ),
            ForRange::Collection(expr) => {
                // For collection iteration, use the expression as both start and end placeholder
                let expr_converted = self.convert_expression(expr)?;
                (expr_converted.clone(), expr_converted)
            }
        };

        // Get the iteration variable from captures (first capture is the loop variable)
        let variable = for_stmt
            .captures
            .first()
            .cloned()
            .unwrap_or_else(|| "_".to_string());

        let body = self.convert_block(&for_stmt.body)?;
        Ok(SlrStatement::For(SlrForStmt::new(
            variable, start, end, body,
        )))
    }

    fn try_extract_repeat_count(&self, for_stmt: &ForStmt) -> Option<usize> {
        // Check for patterns like: for _ in 0..10 { }
        if let ForRange::Range { start, end } = &for_stmt.range
            && let Expr::IntLit(start_lit) = start
            && start_lit.value == 0
            && let Expr::IntLit(end_lit) = end
        {
            return Some(end_lit.value as usize);
        }
        None
    }

    fn convert_expression(&self, expr: &Expr) -> SlrResult<SlrExpression> {
        match expr {
            Expr::IntLit(lit) => Ok(SlrExpression::Literal(SlrLiteralExpr::int(
                lit.value as i64,
            ))),
            Expr::FloatLit(lit) => Ok(SlrExpression::Literal(SlrLiteralExpr::float(lit.value))),
            Expr::BoolLit(lit) => Ok(SlrExpression::Literal(SlrLiteralExpr::bool(lit.value))),
            Expr::Ident(ident) => {
                // Check for built-in angle constants
                match ident.name.as_str() {
                    "pi" | "PI" => {
                        return Ok(SlrExpression::Literal(SlrLiteralExpr::float(
                            std::f64::consts::PI,
                        )));
                    }
                    "tau" | "TAU" => {
                        return Ok(SlrExpression::Literal(SlrLiteralExpr::float(
                            std::f64::consts::TAU,
                        )));
                    }
                    "e" | "E" => {
                        return Ok(SlrExpression::Literal(SlrLiteralExpr::float(
                            std::f64::consts::E,
                        )));
                    }
                    _ => {}
                }
                Ok(SlrExpression::Var(SlrVarExpr::new(&ident.name)))
            }
            Expr::Binary(binary) => {
                let op = binary_op_to_slr(&binary.op);
                let left = self.convert_expression(&binary.left)?;
                let right = self.convert_expression(&binary.right)?;
                Ok(SlrExpression::Binary(SlrBinaryExpr::new(op, left, right)))
            }
            Expr::Index(index) => {
                // Check if this is a bit reference (register[index])
                let name = self.extract_identifier(&index.object)?;
                if self.registers.contains_key(&name) {
                    let idx = self.extract_integer(&index.index)?;
                    return Ok(SlrExpression::Bit(SlrBitExpr::new(name, idx)));
                }
                Err(SlrError::UnsupportedExpression)
            }
            Expr::AngleLit(angle) => {
                use crate::ast::AngleUnit;

                // For radians, try to recognize common pi-based patterns for exact conversion
                if let AngleUnit::Rad = angle.unit
                    && let Some(exact_turns) = recognize_exact_radian_pattern(&angle.value)
                {
                    return Ok(SlrExpression::Literal(SlrLiteralExpr::angle(exact_turns)));
                }

                // Fall back to floating-point evaluation
                use crate::comptime::ComptimeEvaluator;
                let mut eval = ComptimeEvaluator::new();
                let value = eval
                    .eval_expr(&angle.value)
                    .map_err(|_| SlrError::InvalidAngle)?;
                let numeric = match value {
                    crate::comptime::ComptimeValue::Float(f) => f,
                    crate::comptime::ComptimeValue::Int(i) => i as f64,
                    crate::comptime::ComptimeValue::Uint(u) => u as f64,
                    _ => return Err(SlrError::InvalidAngle),
                };
                // Convert to turns (the native unit for angles)
                let turns = angle.unit.to_turns(numeric);
                // Output as angle literal with type information
                Ok(SlrExpression::Literal(SlrLiteralExpr::angle(turns)))
            }
            Expr::TypeAscription(asc) => {
                // For type ascription, evaluate the expression
                // The type information is used for semantic checking, but the value is what matters for codegen
                self.convert_expression(&asc.value)
            }
            Expr::StringLit(lit) => Ok(SlrExpression::Literal(SlrLiteralExpr::string(&lit.value))),
            Expr::FString(fstr) => {
                use crate::ast::FStringPart;
                let parts = fstr
                    .parts
                    .iter()
                    .map(|part| match part {
                        FStringPart::Text(text) => Ok(SlrFStringPart::Text {
                            value: text.clone(),
                        }),
                        FStringPart::Expr { expr, format } => Ok(SlrFStringPart::Expr {
                            value: Box::new(self.convert_expression(expr)?),
                            format: format.clone(),
                        }),
                    })
                    .collect::<SlrResult<Vec<_>>>()?;
                Ok(SlrExpression::FString(SlrFStringExpr::new(parts)))
            }
            _ => Err(SlrError::UnsupportedExpression),
        }
    }

    // =========================================================================
    // Extraction Helpers
    // =========================================================================

    fn try_extract_allocator(&self, expr: &Expr) -> Option<usize> {
        if let Expr::Call(call) = expr {
            let name = self.extract_call_name(&call.callee).ok()?;
            if name == "qalloc" && call.args.len() == 1 {
                return self.extract_integer(&call.args[0]).ok();
            }
        }
        None
    }

    fn try_extract_child_allocator(&self, expr: &Expr) -> Option<(String, usize)> {
        if let Expr::Call(call) = expr
            && let Expr::Field(field) = &call.callee
            && field.field == "child"
            && call.args.len() == 1
        {
            let parent = self.extract_identifier(&field.object).ok()?;
            let size = self.extract_integer(&call.args[0]).ok()?;
            return Some((parent, size));
        }
        None
    }

    fn extract_call_name(&self, callee: &Expr) -> SlrResult<String> {
        match callee {
            Expr::Ident(ident) => Ok(ident.name.clone()),
            Expr::Field(field) => Ok(field.field.clone()),
            _ => Err(SlrError::UnsupportedExpression),
        }
    }

    fn extract_identifier(&self, expr: &Expr) -> SlrResult<String> {
        match expr {
            Expr::Ident(ident) => Ok(ident.name.clone()),
            _ => Err(SlrError::UnsupportedExpression),
        }
    }

    fn extract_slot_ref(&self, expr: &Expr) -> SlrResult<SlrSlotRef> {
        match expr {
            Expr::Index(index) => self.extract_slot_from_index(index),
            _ => Err(SlrError::UnsupportedExpression),
        }
    }

    fn extract_slot_from_index(&self, index: &IndexExpr) -> SlrResult<SlrSlotRef> {
        let allocator = self.extract_identifier(&index.object)?;
        let idx = self.extract_integer(&index.index)?;

        // Validate allocator exists
        let alloc =
            self.allocators
                .get(&allocator)
                .ok_or_else(|| SlrError::UndefinedAllocator {
                    name: allocator.clone(),
                })?;

        // Validate index bounds
        if idx >= alloc.capacity {
            return Err(SlrError::QubitIndexOutOfBounds {
                allocator: allocator.clone(),
                index: idx,
                capacity: alloc.capacity,
            });
        }

        Ok(SlrSlotRef::new(allocator, idx))
    }

    fn extract_integer(&self, expr: &Expr) -> SlrResult<usize> {
        match expr {
            Expr::IntLit(lit) => Ok(lit.value as usize),
            _ => Err(SlrError::UnsupportedExpression),
        }
    }

    fn extract_const_index(&self, expr: &Expr) -> Option<usize> {
        match expr {
            Expr::IntLit(lit) => Some(lit.value as usize),
            _ => None,
        }
    }

    /// Convert an ExternFnDecl to SLR-AST representation.
    fn convert_extern_fn(&self, extern_fn: &crate::ast::ExternFnDecl) -> SlrResult<SlrExternDecl> {
        let params: Vec<SlrExternParam> = extern_fn
            .params
            .iter()
            .map(|p| SlrExternParam {
                name: p.name.clone(),
                ctype: self.convert_type_to_ctype(&p.ty),
            })
            .collect();

        let return_type = extern_fn
            .return_type
            .as_ref()
            .map(|t| self.convert_type_to_ctype(t));

        Ok(SlrExternDecl::new(
            extern_fn.name.clone(),
            extern_fn.library.clone(),
            extern_fn.calling_convention.clone(),
            params,
            return_type,
        ))
    }

    /// Convert a Zlup type expression to a C-compatible type.
    fn convert_type_to_ctype(&self, ty: &crate::ast::TypeExpr) -> SlrCType {
        use crate::ast::{PrimitiveType, TypeExpr};

        match ty {
            // Primitive types (parsed from u8, u32, i32, f32, etc.)
            TypeExpr::Primitive(prim) => match prim {
                PrimitiveType::UInt { bits } => SlrCType::Int {
                    bits: *bits as u8,
                    signed: false,
                },
                PrimitiveType::IInt { bits } => SlrCType::Int {
                    bits: *bits as u8,
                    signed: true,
                },
                PrimitiveType::Usize => SlrCType::Int {
                    bits: 64,
                    signed: false,
                }, // Assume 64-bit
                PrimitiveType::Isize => SlrCType::Int {
                    bits: 64,
                    signed: true,
                },
                PrimitiveType::F16 => SlrCType::Float { bits: 16 },
                PrimitiveType::F32 => SlrCType::Float { bits: 32 },
                PrimitiveType::F64 => SlrCType::Float { bits: 64 },
                PrimitiveType::F128 => SlrCType::Float { bits: 128 },
                PrimitiveType::A64 => SlrCType::Angle { bits: 64 }, // PECOS Angle64
                PrimitiveType::Bool => SlrCType::Int {
                    bits: 8,
                    signed: false,
                },
            },
            // Named types (for fallback or custom types)
            TypeExpr::Named(path) => {
                let name = path.segments.join("::");
                match name.as_str() {
                    "u8" => SlrCType::Int {
                        bits: 8,
                        signed: false,
                    },
                    "u16" => SlrCType::Int {
                        bits: 16,
                        signed: false,
                    },
                    "u32" => SlrCType::Int {
                        bits: 32,
                        signed: false,
                    },
                    "u64" => SlrCType::Int {
                        bits: 64,
                        signed: false,
                    },
                    "usize" => SlrCType::Int {
                        bits: 64,
                        signed: false,
                    }, // Assume 64-bit
                    "i8" => SlrCType::Int {
                        bits: 8,
                        signed: true,
                    },
                    "i16" => SlrCType::Int {
                        bits: 16,
                        signed: true,
                    },
                    "i32" => SlrCType::Int {
                        bits: 32,
                        signed: true,
                    },
                    "i64" => SlrCType::Int {
                        bits: 64,
                        signed: true,
                    },
                    "isize" => SlrCType::Int {
                        bits: 64,
                        signed: true,
                    },
                    "f32" => SlrCType::Float { bits: 32 },
                    "f64" => SlrCType::Float { bits: 64 },
                    "bool" => SlrCType::Int {
                        bits: 8,
                        signed: false,
                    },
                    "unit" | "void" => SlrCType::Void,
                    _ => SlrCType::Opaque { name },
                }
            }
            // Pointer types: *T, [*]T, [*:0]T
            TypeExpr::Pointer(ptr) => {
                let element = self.convert_type_to_ctype(&ptr.pointee);
                SlrCType::Pointer {
                    element: Box::new(element),
                    is_const: ptr.is_const,
                }
            }
            // Array types used as pointers in C
            TypeExpr::Array(arr) => {
                let element = self.convert_type_to_ctype(&arr.element);
                SlrCType::Pointer {
                    element: Box::new(element),
                    is_const: false,
                }
            }
            // Unit type
            TypeExpr::Unit => SlrCType::Void,
            // Default to opaque for unknown types
            _ => SlrCType::Opaque {
                name: format!("{:?}", ty),
            },
        }
    }
}

impl Default for SlrCodegen {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Angle Precision Helpers
// =============================================================================

/// Recognize common pi-based radian expressions and return exact turn fractions.
///
/// This preserves precision by pattern-matching the AST before floating-point evaluation.
/// For example:
/// - `pi / 2` → 0.25 turns (exact)
/// - `pi / 4` → 0.125 turns (exact)
/// - `3 * pi / 4` → 0.375 turns (exact)
fn recognize_exact_radian_pattern(expr: &Expr) -> Option<f64> {
    // Pattern: pi (just pi = 1/2 turn)
    if is_pi_reference(expr) {
        return Some(0.5);
    }

    // Pattern: pi / N (pi divided by integer)
    if let Expr::Binary(binary) = expr {
        if binary.op == crate::ast::BinaryOp::Div
            && is_pi_reference(&binary.left)
            && let Some(n) = extract_integer_value(&binary.right)
            && n > 0
        {
            // pi / n radians = 1 / (2*n) turns
            return Some(1.0 / (2.0 * n as f64));
        }

        // Pattern: N * pi / M or (N * pi) / M
        if binary.op == crate::ast::BinaryOp::Div
            && let Some((num, denom)) = extract_pi_fraction(&binary.left, &binary.right)
        {
            // (num * pi) / denom radians = num / (2 * denom) turns
            return Some(num as f64 / (2.0 * denom as f64));
        }

        // Pattern: pi * N / M (reordered)
        if binary.op == crate::ast::BinaryOp::Mul {
            // Check for pi * (N / M) - less common but possible
            if is_pi_reference(&binary.left)
                && let Expr::Binary(inner) = &binary.right
                && inner.op == crate::ast::BinaryOp::Div
                && let (Some(num), Some(denom)) = (
                    extract_integer_value(&inner.left),
                    extract_integer_value(&inner.right),
                )
                && denom > 0
            {
                return Some(num as f64 / (2.0 * denom as f64));
            }
        }
    }

    None
}

/// Check if an expression is a reference to pi (std.f64.pi, pi, PI, etc.)
fn is_pi_reference(expr: &Expr) -> bool {
    match expr {
        Expr::Ident(ident) => {
            matches!(ident.name.as_str(), "pi" | "PI")
        }
        Expr::Field(field) => {
            // Check for std.f64.pi or similar
            field.field == "pi" || field.field == "PI"
        }
        _ => false,
    }
}

/// Extract an integer value from an expression (literal or simple expression)
fn extract_integer_value(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::IntLit(lit) => Some(lit.value as i64),
        // Handle negative integers
        Expr::Unary(unary) if unary.op == crate::ast::UnaryOp::Neg => {
            extract_integer_value(&unary.operand).map(|v| -v)
        }
        _ => None,
    }
}

/// Extract numerator and denominator from expressions like N * pi / M
fn extract_pi_fraction(left: &Expr, right: &Expr) -> Option<(i64, i64)> {
    // Check if left is N * pi
    if let Expr::Binary(mul) = left
        && mul.op == crate::ast::BinaryOp::Mul
    {
        // N * pi
        if let Some(n) = extract_integer_value(&mul.left)
            && is_pi_reference(&mul.right)
            && let Some(m) = extract_integer_value(right)
            && m > 0
        {
            return Some((n, m));
        }
        // pi * N
        if is_pi_reference(&mul.left)
            && let Some(n) = extract_integer_value(&mul.right)
            && let Some(m) = extract_integer_value(right)
            && m > 0
        {
            return Some((n, m));
        }
    }
    None
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::GateKind;
    use crate::parse;

    fn compile_to_slr(source: &str) -> SlrResult<SlrProgram> {
        let program = parse(source).expect("parse failed");
        let mut codegen = SlrCodegen::new();
        codegen.compile(&program)
    }

    fn to_json(source: &str) -> String {
        let program = parse(source).expect("parse failed");
        let mut codegen = SlrCodegen::new();
        let slr = codegen.compile(&program).expect("compile failed");
        codegen.to_json(&slr).expect("json failed")
    }

    #[test]
    fn test_gate_info_covers_every_lowered_gate_kind() {
        for kind in GateKind::ALL {
            if *kind == GateKind::PZ {
                // PZ becomes an SLR Prepare statement before the gate-info lookup.
                assert!(get_gate_info(kind.keyword()).is_none());
                continue;
            }

            let info = get_gate_info(kind.keyword())
                .unwrap_or_else(|| panic!("missing SLR gate info for {}", kind.keyword()));
            assert_eq!(info.arity, kind.arity(), "gate: {}", kind.keyword());
            assert_eq!(
                info.parameterized,
                kind.is_parameterized(),
                "gate: {}",
                kind.keyword()
            );
        }
    }

    #[test]
    fn test_empty_program() {
        let slr = compile_to_slr("").unwrap();
        assert_eq!(slr.name, "main");
        assert!(slr.body.is_empty());
    }

    #[test]
    fn test_single_qubit_gate() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(1);
                h q[0];
            }
        "#;

        let slr = compile_to_slr(source).unwrap();
        assert_eq!(slr.body.len(), 1);

        if let SlrStatement::Gate(gate) = &slr.body[0] {
            assert_eq!(gate.gate, "H"); // Output remains uppercase
            assert_eq!(gate.targets.len(), 1);
            assert_eq!(gate.targets[0].allocator, "q");
            assert_eq!(gate.targets[0].index, 0);
        } else {
            panic!("Expected gate operation");
        }
    }

    #[test]
    fn test_bell_state() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(2);
                h q[0];
                cx (q[0], q[1]);
            }
        "#;

        let slr = compile_to_slr(source).unwrap();
        assert_eq!(slr.body.len(), 2);

        // First gate: h (output is uppercase H)
        if let SlrStatement::Gate(gate) = &slr.body[0] {
            assert_eq!(gate.gate, "H");
        } else {
            panic!("Expected H gate");
        }

        // Second gate: cx (output is uppercase CX)
        if let SlrStatement::Gate(gate) = &slr.body[1] {
            assert_eq!(gate.gate, "CX");
            assert_eq!(gate.targets.len(), 2);
        } else {
            panic!("Expected CX gate");
        }
    }

    #[test]
    fn test_three_qubit_gate() {
        let source = r#"
            pub fn main() -> unit {
                q := qalloc(3);
                ccx (q[0], q[1], q[2]);
                return unit;
            }
        "#;

        let slr = compile_to_slr(source).unwrap();
        let SlrStatement::Gate(gate) = &slr.body[0] else {
            panic!("Expected CCX gate operation");
        };
        assert_eq!(gate.gate, "CCX");
        assert_eq!(gate.targets.len(), 3);
    }

    #[test]
    fn test_rotation_gate() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(1);
                rz(1.57, q[0]);
            }
        "#;

        let slr = compile_to_slr(source).unwrap();
        assert_eq!(slr.body.len(), 1);

        if let SlrStatement::Gate(gate) = &slr.body[0] {
            assert_eq!(gate.gate, "RZ");
            assert_eq!(gate.params.len(), 1);
        } else {
            panic!("Expected rz gate");
        }
    }

    #[test]
    fn test_angle_precision_preservation() {
        // Test that pi/N rad patterns are converted to exact turn fractions
        let source = r#"
            pi: f64 = 3.14159265358979323846;
            pub fn main() -> unit {
                mut q := qalloc(2);
                rz(pi/4 rad) q[0];   // Should be exactly 0.125 turns
                rz(pi/2 rad) q[1];   // Should be exactly 0.25 turns
            }
        "#;

        let slr = compile_to_slr(source).unwrap();
        assert_eq!(slr.body.len(), 2);

        // Check first gate: pi/4 rad = 0.125 turns
        if let SlrStatement::Gate(gate) = &slr.body[0] {
            assert_eq!(gate.gate, "RZ");
            if let SlrExpression::Literal(lit) = &gate.params[0] {
                if let SlrLiteralValue::Angle(turns) = &lit.value {
                    assert!(
                        (*turns - 0.125).abs() < 1e-15,
                        "pi/4 rad should be exactly 0.125 turns, got {}",
                        turns
                    );
                } else {
                    panic!("Expected angle literal");
                }
            }
        }

        // Check second gate: pi/2 rad = 0.25 turns
        if let SlrStatement::Gate(gate) = &slr.body[1] {
            assert_eq!(gate.gate, "RZ");
            if let SlrExpression::Literal(lit) = &gate.params[0] {
                if let SlrLiteralValue::Angle(turns) = &lit.value {
                    assert!(
                        (*turns - 0.25).abs() < 1e-15,
                        "pi/2 rad should be exactly 0.25 turns, got {}",
                        turns
                    );
                } else {
                    panic!("Expected angle literal");
                }
            }
        }
    }

    #[test]
    fn test_allocator_decl() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(4);
            }
        "#;

        let slr = compile_to_slr(source).unwrap();
        assert!(slr.allocator.is_some());

        let alloc = slr.allocator.unwrap();
        assert_eq!(alloc.name, "q");
        assert_eq!(alloc.capacity, 4);
    }

    #[test]
    fn test_child_allocator() {
        let source = r#"
            pub fn main() -> unit {
                mut base := qalloc(4);
                mut q := base.child(2);
                h q[0];
            }
        "#;

        let slr = compile_to_slr(source).unwrap();

        // Should have both allocators in declarations
        let alloc_count = slr
            .declarations
            .iter()
            .filter(|d| matches!(d, SlrDeclaration::Allocator(_)))
            .count();
        assert_eq!(alloc_count, 2);
    }

    #[test]
    fn test_json_output() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(2);
                h q[0];
                cx (q[0], q[1]);
            }
        "#;

        let json = to_json(source);

        // Verify it's valid JSON
        let value: serde_json::Value = serde_json::from_str(&json).expect("invalid JSON");

        // Check structure
        assert_eq!(value["type"], "Program");
        assert_eq!(value["name"], "main");
        assert!(value["allocator"].is_object());
        assert!(value["body"].is_array());
        assert_eq!(value["body"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_if_statement() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(1);
                mut x := 1;
                if (x == 1) {
                    h q[0];
                }
            }
        "#;

        let slr = compile_to_slr(source).unwrap();

        // Find the if statement
        let has_if = slr.body.iter().any(|s| matches!(s, SlrStatement::If(_)));
        assert!(has_if, "Expected if statement in body");
    }

    #[test]
    fn test_wrong_argument_count() {
        // CX requires pairs of qubits - passing individual qubits in a batch is wrong
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(2);
                cx {q[0], q[1]};
            }
        "#;

        let result = compile_to_slr(source);
        assert!(matches!(result, Err(SlrError::WrongArgumentCount { .. })));
    }

    #[test]
    fn test_batch_gate_rejects_non_qubit_elements_without_fake_arity() {
        for source in [
            r#"
                pub fn main() -> unit {
                    mut q := qalloc(1);
                    h {q[0], 42};
                }
            "#,
            r#"
                pub fn main() -> unit {
                    mut q := qalloc(1);
                    h [q[0], 42];
                }
            "#,
        ] {
            assert!(matches!(
                compile_to_slr(source),
                Err(SlrError::UnsupportedExpression)
            ));
        }
    }

    #[test]
    fn test_qubit_out_of_bounds() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(2);
                h q[5];
            }
        "#;

        let result = compile_to_slr(source);
        assert!(matches!(
            result,
            Err(SlrError::QubitIndexOutOfBounds { .. })
        ));
    }

    #[test]
    fn test_tick_block() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(4);
                tick {
                    h q[0];
                    h q[1];
                }
            }
        "#;

        let slr = compile_to_slr(source).unwrap();
        assert_eq!(slr.body.len(), 1);

        if let SlrStatement::Tick(tick) = &slr.body[0] {
            assert!(tick.label.is_none());
            assert_eq!(tick.body.len(), 2);
        } else {
            panic!("Expected tick statement");
        }
    }

    #[test]
    fn test_tick_block_with_label() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(4);
                tick syndrome_round {
                    h q[0];
                    cx (q[0], q[1]);
                }
            }
        "#;

        let slr = compile_to_slr(source).unwrap();
        assert_eq!(slr.body.len(), 1);

        if let SlrStatement::Tick(tick) = &slr.body[0] {
            assert_eq!(tick.label.as_ref().unwrap(), "syndrome_round");
            assert_eq!(tick.body.len(), 2);
        } else {
            panic!("Expected tick statement");
        }
    }

    #[test]
    fn test_tick_block_with_string_label() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(4);
                tick "layer_1" {
                    h q[0];
                }
            }
        "#;

        let slr = compile_to_slr(source).unwrap();
        assert_eq!(slr.body.len(), 1);

        if let SlrStatement::Tick(tick) = &slr.body[0] {
            assert_eq!(tick.label.as_ref().unwrap(), "layer_1");
        } else {
            panic!("Expected tick statement");
        }
    }

    #[test]
    fn test_nested_tick_blocks() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(4);
                tick outer {
                    tick inner1 {
                        h q[0];
                    }
                    tick inner2 {
                        x(q[1]);
                    }
                }
            }
        "#;

        let slr = compile_to_slr(source).unwrap();
        assert_eq!(slr.body.len(), 1);

        if let SlrStatement::Tick(outer) = &slr.body[0] {
            assert_eq!(outer.label.as_ref().unwrap(), "outer");
            assert_eq!(outer.body.len(), 2);

            // Check nested ticks
            if let SlrStatement::Tick(inner1) = &outer.body[0] {
                assert_eq!(inner1.label.as_ref().unwrap(), "inner1");
            } else {
                panic!("Expected nested tick");
            }
        } else {
            panic!("Expected tick statement");
        }
    }

    #[test]
    fn test_tick_json_output() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(2);
                tick layer1 {
                    h q[0];
                    h q[1];
                }
            }
        "#;

        let json = to_json(source);
        let value: serde_json::Value = serde_json::from_str(&json).expect("invalid JSON");

        // Check tick structure in JSON
        let tick = &value["body"][0];
        assert_eq!(tick["type"], "TickStmt");
        assert_eq!(tick["label"], "layer1");
        assert!(tick["body"].is_array());
        assert_eq!(tick["body"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_tick_with_inline_attributes() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(2);
                tick @attrs({round: 0, kind: "syndrome"}) syndrome_round {
                    h q[0];
                }
            }
        "#;

        let slr = compile_to_slr(source).unwrap();
        assert_eq!(slr.body.len(), 1);

        if let SlrStatement::Tick(tick) = &slr.body[0] {
            assert_eq!(tick.label.as_ref().unwrap(), "syndrome_round");
            assert_eq!(tick.attrs.len(), 2);
            assert!(matches!(
                tick.attrs.get("round"),
                Some(SlrAttributeValue::Int(0))
            ));
            assert!(
                matches!(tick.attrs.get("kind"), Some(SlrAttributeValue::String(s)) if s == "syndrome")
            );
        } else {
            panic!("Expected tick statement");
        }
    }

    #[test]
    fn test_tick_with_prefix_attributes() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(2);
                @attr(noisy, true)
                tick {
                    h q[0];
                }
            }
        "#;

        let slr = compile_to_slr(source).unwrap();
        assert_eq!(slr.body.len(), 1);

        if let SlrStatement::Tick(tick) = &slr.body[0] {
            assert!(tick.label.is_none());
            assert_eq!(tick.attrs.len(), 1);
            assert!(matches!(
                tick.attrs.get("noisy"),
                Some(SlrAttributeValue::Bool(true))
            ));
        } else {
            panic!("Expected tick statement");
        }
    }

    #[test]
    fn test_tick_with_mixed_attributes() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(2);
                @attrs({error_rate: 0.001, round: 5})
                tick layer {
                    h q[0];
                }
            }
        "#;

        let slr = compile_to_slr(source).unwrap();

        if let SlrStatement::Tick(tick) = &slr.body[0] {
            assert_eq!(tick.label.as_ref().unwrap(), "layer");
            assert_eq!(tick.attrs.len(), 2);
            assert!(matches!(
                tick.attrs.get("round"),
                Some(SlrAttributeValue::Int(5))
            ));
            // Check float attribute
            if let Some(SlrAttributeValue::Float(f)) = tick.attrs.get("error_rate") {
                assert!((*f - 0.001).abs() < 0.0001);
            } else {
                panic!("Expected float attribute");
            }
        } else {
            panic!("Expected tick statement");
        }
    }

    #[test]
    fn test_tick_attributes_json_output() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(2);
                tick @attrs({round: 0, kind: "syndrome"}) syndrome {
                    h q[0];
                }
            }
        "#;

        let json = to_json(source);
        let value: serde_json::Value = serde_json::from_str(&json).expect("invalid JSON");

        let tick = &value["body"][0];
        assert_eq!(tick["type"], "TickStmt");
        assert_eq!(tick["label"], "syndrome");
        assert_eq!(tick["attrs"]["round"], 0);
        assert_eq!(tick["attrs"]["kind"], "syndrome");
    }

    // =========================================================================
    // Gate Attribute Tests
    // =========================================================================

    #[test]
    fn test_gate_with_attributes() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(2);
                @attr(syndrome, "X")
                cx (q[0], q[1]);
            }
        "#;

        let slr = compile_to_slr(source).unwrap();
        assert_eq!(slr.body.len(), 1);

        if let SlrStatement::Gate(gate) = &slr.body[0] {
            assert_eq!(gate.gate, "CX");
            assert_eq!(gate.attrs.len(), 1);
            assert!(
                matches!(gate.attrs.get("syndrome"), Some(SlrAttributeValue::String(s)) if s == "X")
            );
        } else {
            panic!("Expected gate statement");
        }
    }

    #[test]
    fn test_gate_with_multiple_attributes() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(2);
                @attrs({syndrome: "Z", layer: 1})
                h q[0];
            }
        "#;

        let slr = compile_to_slr(source).unwrap();

        if let SlrStatement::Gate(gate) = &slr.body[0] {
            assert_eq!(gate.gate, "H");
            assert_eq!(gate.attrs.len(), 2);
            assert!(
                matches!(gate.attrs.get("syndrome"), Some(SlrAttributeValue::String(s)) if s == "Z")
            );
            assert!(matches!(
                gate.attrs.get("layer"),
                Some(SlrAttributeValue::Int(1))
            ));
        } else {
            panic!("Expected gate statement");
        }
    }

    #[test]
    fn test_batch_gate_with_attributes() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(4);
                @attr(syndrome, "X")
                h([q[0], q[1]]);
            }
        "#;

        let slr = compile_to_slr(source).unwrap();
        // Batch gate expands to 2 gates, each should have the attribute
        assert_eq!(slr.body.len(), 2);

        for stmt in &slr.body {
            if let SlrStatement::Gate(gate) = stmt {
                assert_eq!(gate.gate, "H");
                assert_eq!(gate.attrs.len(), 1);
                assert!(
                    matches!(gate.attrs.get("syndrome"), Some(SlrAttributeValue::String(s)) if s == "X")
                );
            } else {
                panic!("Expected gate statement");
            }
        }
    }

    #[test]
    fn test_gate_attributes_json_output() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(2);
                @attrs({syndrome: "X", ancilla: true})
                cx (q[0], q[1]);
            }
        "#;

        let json = to_json(source);
        let value: serde_json::Value = serde_json::from_str(&json).expect("invalid JSON");

        let gate = &value["body"][0];
        assert_eq!(gate["type"], "GateOp");
        assert_eq!(gate["gate"], "CX");
        assert_eq!(gate["attrs"]["syndrome"], "X");
        assert_eq!(gate["attrs"]["ancilla"], true);
    }

    #[test]
    fn test_gate_without_attributes_no_attrs_field() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(1);
                h q[0];
            }
        "#;

        let json = to_json(source);
        let value: serde_json::Value = serde_json::from_str(&json).expect("invalid JSON");

        let gate = &value["body"][0];
        assert_eq!(gate["type"], "GateOp");
        // attrs field should be absent (skip_serializing_if = is_empty)
        assert!(gate.get("attrs").is_none());
    }

    // =========================================================================
    // Typed Measurement Tests
    // =========================================================================

    #[test]
    fn test_typed_measurement_single_qubit() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(2);
                r := mz(u1) q[0];
            }
        "#;

        let slr = compile_to_slr(source).unwrap();
        assert_eq!(slr.body.len(), 1);

        if let SlrStatement::Measure(measure) = &slr.body[0] {
            assert_eq!(measure.targets.len(), 1);
            assert_eq!(measure.targets[0].allocator, "q");
            assert_eq!(measure.targets[0].index, 0);
            assert_eq!(measure.results.len(), 1);
        } else {
            panic!("Expected measure operation");
        }
    }

    #[test]
    fn test_typed_measurement_array() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(4);
                results := mz(u1) [q[0], q[1], q[2]];
            }
        "#;

        let slr = compile_to_slr(source).unwrap();
        assert_eq!(slr.body.len(), 1);

        if let SlrStatement::Measure(measure) = &slr.body[0] {
            assert_eq!(measure.targets.len(), 3);
            assert_eq!(measure.targets[0].allocator, "q");
            assert_eq!(measure.targets[0].index, 0);
            assert_eq!(measure.targets[1].index, 1);
            assert_eq!(measure.targets[2].index, 2);
            assert_eq!(measure.results.len(), 3);
        } else {
            panic!("Expected measure operation");
        }
    }

    #[test]
    fn test_typed_measurement_json_output() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(2);
                r := mz(u1) q[0];
            }
        "#;

        let json = to_json(source);
        let value: serde_json::Value = serde_json::from_str(&json).expect("invalid JSON");

        let measure = &value["body"][0];
        assert_eq!(measure["type"], "MeasureOp");
        assert!(measure["targets"].is_array());
        assert_eq!(measure["targets"].as_array().unwrap().len(), 1);
        assert!(measure["results"].is_array());
        assert_eq!(measure["results"].as_array().unwrap().len(), 1);
        assert_eq!(measure["result_type"], "u1");
    }

    #[test]
    fn test_measurement_result_type_u8() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(2);
                r := mz(u8) q[0];
            }
        "#;

        let json = to_json(source);
        let value: serde_json::Value = serde_json::from_str(&json).expect("invalid JSON");

        let measure = &value["body"][0];
        assert_eq!(measure["type"], "MeasureOp");
        assert_eq!(measure["result_type"], "u8");
    }

    #[test]
    fn test_measurement_result_type_u64() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(2);
                r := mz(u64) q[0];
            }
        "#;

        let json = to_json(source);
        let value: serde_json::Value = serde_json::from_str(&json).expect("invalid JSON");

        let measure = &value["body"][0];
        assert_eq!(measure["type"], "MeasureOp");
        assert_eq!(measure["result_type"], "u64");
    }

    #[test]
    fn test_extern_fn_declaration() {
        let source = r#"
            extern "C" fn decode(data: [*]u8, len: usize) -> i32;

            pub fn main() -> unit {
                q := qalloc(1);
                pz q;
            }
        "#;

        let json = to_json(source);
        let value: serde_json::Value = serde_json::from_str(&json).expect("invalid JSON");

        // Check that externs array exists and has one entry
        let externs = &value["externs"];
        assert!(externs.is_array(), "externs should be an array");
        assert_eq!(externs.as_array().unwrap().len(), 1);

        // Check extern function properties
        let extern_fn = &externs[0];
        assert_eq!(extern_fn["type"], "ExternDecl");
        assert_eq!(extern_fn["name"], "decode");
        assert_eq!(extern_fn["calling_convention"], "C");

        // Check params
        let params = &extern_fn["params"];
        assert_eq!(params.as_array().unwrap().len(), 2);
        assert_eq!(params[0]["name"], "data");
        assert_eq!(params[1]["name"], "len");
    }

    #[test]
    fn test_extern_fn_with_library() {
        let source = r#"
            @link("libdecoder")
            extern "C" fn mwpm_decode(syndrome: [*]u8, n: u32) -> i32;

            pub fn main() -> unit {
                q := qalloc(1);
                pz q;
            }
        "#;

        let json = to_json(source);
        let value: serde_json::Value = serde_json::from_str(&json).expect("invalid JSON");

        let extern_fn = &value["externs"][0];
        assert_eq!(extern_fn["name"], "mwpm_decode");
        assert_eq!(extern_fn["library"], "libdecoder");
        assert_eq!(extern_fn["calling_convention"], "C");

        // Check params
        let first_param = &extern_fn["params"][0];
        assert_eq!(first_param["name"], "syndrome");
        assert_eq!(first_param["ctype"]["kind"], "pointer");
        assert_eq!(first_param["ctype"]["element"]["kind"], "int");
        assert_eq!(first_param["ctype"]["element"]["bits"], 8);
    }

    #[test]
    fn test_extern_fn_rust_abi() {
        let source = r#"
            extern "Rust" fn pecos_simulate(circuit: *const u8) -> u64;

            pub fn main() -> unit {
                q := qalloc(1);
                pz q;
            }
        "#;

        let json = to_json(source);
        let value: serde_json::Value = serde_json::from_str(&json).expect("invalid JSON");

        let extern_fn = &value["externs"][0];
        assert_eq!(extern_fn["name"], "pecos_simulate");
        assert_eq!(extern_fn["calling_convention"], "Rust");
    }

    #[test]
    fn test_extern_fn_call() {
        let source = r#"
            extern "C" fn simple_decode(value: u32) -> i32;

            pub fn main() -> unit {
                q := qalloc(2);
                pz q;
                h q[0];
                cx (q[0], q[1]);

                // Call external decoder with a simple literal
                simple_decode(42);
            }
        "#;

        let json = to_json(source);
        let value: serde_json::Value = serde_json::from_str(&json).expect("invalid JSON");

        // Check that extern declaration is present
        assert_eq!(value["externs"][0]["name"], "simple_decode");

        // Check that extern call is in the body
        let body = &value["body"];
        let extern_call = body
            .as_array()
            .unwrap()
            .iter()
            .find(|stmt| stmt["type"] == "ExternCall");
        assert!(extern_call.is_some(), "Expected ExternCall in body");

        let call = extern_call.unwrap();
        assert_eq!(call["function"], "simple_decode");
        assert_eq!(call["args"].as_array().unwrap().len(), 1);
        // The argument should be a literal value
        assert_eq!(call["args"][0]["value"], 42);
    }
}
