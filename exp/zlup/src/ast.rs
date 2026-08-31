//! AST node definitions for Zluppy programs.
//!
//! This module defines the Abstract Syntax Tree (AST) nodes for representing
//! Zluppy quantum programs. The design mirrors the Python SLR-AST for easy
//! conversion while supporting Zig-inspired language features.
//!
//! Design principles:
//! - Immutable data structures
//! - Explicit source location tracking
//! - Direct mapping to SLR-AST where applicable
//! - Support for comptime evaluation

use serde::{Deserialize, Serialize};
use std::fmt;

// =============================================================================
// Source Location
// =============================================================================

/// Source location for error reporting.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SourceLocation {
    pub line: u32,
    pub column: u32,
    pub end_line: u32,
    pub end_column: u32,
    pub file: Option<String>,
}

impl SourceLocation {
    pub fn new(line: u32, column: u32) -> Self {
        Self {
            line,
            column,
            end_line: line,
            end_column: column + 1,
            file: None,
        }
    }

    pub fn with_end(line: u32, column: u32, end_line: u32, end_column: u32) -> Self {
        Self {
            line,
            column,
            end_line,
            end_column,
            file: None,
        }
    }

    pub fn with_file(line: u32, column: u32, file: impl Into<String>) -> Self {
        Self {
            line,
            column,
            end_line: line,
            end_column: column + 1,
            file: Some(file.into()),
        }
    }
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.file {
            Some(file) => write!(f, "{}:{}:{}", file, self.line, self.column),
            None => write!(f, "{}:{}", self.line, self.column),
        }
    }
}

// =============================================================================
// Attributes (Metadata)
// =============================================================================

/// Attribute value types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AttributeValue {
    /// Boolean flag: @attr(noisy, true)
    Bool(bool),
    /// Integer value: @attr(round, 0)
    Int(i64),
    /// Float value: @attr(error_rate, 0.001)
    Float(f64),
    /// String value: @attr(kind, "syndrome")
    String(String),
    /// Identifier value: @attr(gate_type, Hadamard)
    Ident(String),
}

/// An attribute attached to a statement or construct.
/// Examples: @attr(round, 0), @attr(kind, "syndrome"), @attrs({round: 0, kind: "syndrome"})
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attribute {
    /// Attribute name (e.g., "round", "type", "noisy")
    pub name: String,
    /// Attribute value (None for boolean flags like @noisy)
    pub value: Option<AttributeValue>,
    pub location: Option<SourceLocation>,
}

impl Attribute {
    /// Create a boolean flag attribute (e.g., @noisy)
    pub fn flag(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            value: None,
            location: None,
        }
    }

    /// Create an attribute with a value
    pub fn with_value(name: impl Into<String>, value: AttributeValue) -> Self {
        Self {
            name: name.into(),
            value: Some(value),
            location: None,
        }
    }
}

// =============================================================================
// Program
// =============================================================================

/// Root node representing a Zluppy program.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Program {
    pub name: String,
    pub declarations: Vec<TopLevelDecl>,
    pub location: Option<SourceLocation>,
}

/// Top-level declarations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TopLevelDecl {
    Binding(Binding),
    Fn(FnDecl),
    ExternFn(ExternFnDecl),
    Struct(StructDecl),
    Enum(EnumDecl),
    Union(UnionDecl),
    ErrorSet(ErrorSetDecl),
    FaultSet(FaultSetDecl),
    Test(TestDecl),
    DeclareGate(TargetGateDecl),
    Gate(CompositeGateDecl),
}

// =============================================================================
// Declarations
// =============================================================================

/// Binding declaration (unified const/var with Pascal/Go syntax)
/// Immutable: `x := value;` or `x: T = value;`
/// Mutable: `mut x := value;` or `mut x: T = value;`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Binding {
    pub name: String,
    pub ty: Option<TypeExpr>,
    pub value: Option<Expr>, // None means undefined
    pub is_mutable: bool,    // true if `mut` keyword present
    pub is_pub: bool,
    pub doc_comment: Option<String>,
    pub location: Option<SourceLocation>,
}

/// Alias binding - creates a named view into existing data.
/// `alias name := slice_expr;`
///
/// Aliases are immutable views with overlap checking.
/// Unlike regular bindings, aliases track their source for overlap detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AliasBinding {
    /// Name of the alias
    pub name: String,
    /// The source expression (must be a slice/range expression)
    pub source: Expr,
    /// Source location
    pub location: Option<SourceLocation>,
}

/// Function declaration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FnDecl {
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: Option<TypeExpr>,
    pub body: Block,
    pub is_pub: bool,
    pub is_inline: bool,
    /// Error handling mode for the function body.
    /// `try` = collect all errors (QEC pattern)
    /// `try!` = stop on first error (traditional)
    /// None = no automatic error handling
    pub error_mode: Option<TryMode>,
    pub doc_comment: Option<String>,
    pub location: Option<SourceLocation>,
}

/// Function parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Param {
    pub name: String,
    pub ty: TypeExpr,
    pub is_comptime: bool,
    pub location: Option<SourceLocation>,
}

/// External function declaration (FFI).
/// `@link("libdecoder") extern "C" fn decode(data: [*]u8, len: usize) -> i32;`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternFnDecl {
    pub name: String,
    /// Library to link against (e.g., "libdecoder", "pecos_runtime")
    pub library: Option<String>,
    /// Calling convention (e.g., "C", "Rust")
    pub calling_convention: String,
    pub params: Vec<Param>,
    pub return_type: Option<TypeExpr>,
    pub is_pub: bool,
    pub doc_comment: Option<String>,
    pub location: Option<SourceLocation>,
}

/// Struct declaration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructDecl {
    pub name: String,
    pub fields: Vec<StructField>,
    pub methods: Vec<FnDecl>,
    /// Associated constants defined within the struct
    pub associated_consts: Vec<Binding>,
    pub is_pub: bool,
    pub is_packed: bool,
    pub doc_comment: Option<String>,
    pub location: Option<SourceLocation>,
}

/// Struct field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructField {
    pub name: String,
    pub ty: TypeExpr,
    pub default: Option<Expr>,
    pub doc_comment: Option<String>,
    pub location: Option<SourceLocation>,
}

/// Enum declaration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnumDecl {
    pub name: String,
    pub tag_type: Option<TypeExpr>,
    pub variants: Vec<EnumVariant>,
    pub is_pub: bool,
    pub doc_comment: Option<String>,
    pub location: Option<SourceLocation>,
}

/// Enum variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnumVariant {
    pub name: String,
    pub value: Option<Expr>,
    pub location: Option<SourceLocation>,
}

/// Union declaration (tagged union / sum type).
/// Example: `const Value = union(enum) { Int: i32, Float: f64, None }`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnionDecl {
    pub name: String,
    /// The tag type: None = untagged, Some(None) = auto-tagged (enum), Some(Some(ty)) = external tag
    pub tag: Option<Option<TypeExpr>>,
    pub fields: Vec<UnionField>,
    pub is_pub: bool,
    pub doc_comment: Option<String>,
    pub location: Option<SourceLocation>,
}

/// Union field (variant with optional payload type).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnionField {
    pub name: String,
    /// The payload type for this variant. None = no payload (like an enum variant)
    pub ty: Option<TypeExpr>,
    pub location: Option<SourceLocation>,
}

/// Error set declaration - classical/logical errors that crash if unhandled.
/// `DecodeError := error { SyndromeAmbiguous, WeightTooHigh };`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorSetDecl {
    pub name: String,
    /// The error variants in this set
    pub variants: Vec<ErrorVariant>,
    pub is_pub: bool,
    pub doc_comment: Option<String>,
    pub location: Option<SourceLocation>,
}

/// Fault set declaration - quantum/physical faults, collected in try blocks.
/// `QuantumFault := fault { Leakage, QubitLoss, GateFailure };`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultSetDecl {
    pub name: String,
    /// The fault variants in this set
    pub variants: Vec<ErrorVariant>, // Reuse ErrorVariant structure
    pub is_pub: bool,
    pub doc_comment: Option<String>,
    pub location: Option<SourceLocation>,
}

/// An error/fault variant within an error or fault set.
/// Can optionally have associated data: `Leakage: struct { gate: []const u8, qubit: usize }`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorVariant {
    pub name: String,
    /// Optional associated data type
    pub data_type: Option<TypeExpr>,
    pub location: Option<SourceLocation>,
}

/// Test declaration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestDecl {
    pub name: String,
    pub body: Block,
    pub location: Option<SourceLocation>,
}

// =============================================================================
// Custom Gate Declarations
// =============================================================================

/// A gate parameter (angle or classical parameter).
/// Example: `theta` in `declare gate rx(theta)(q);`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateParam {
    pub name: String,
    pub ty: Option<TypeExpr>,
    pub location: Option<SourceLocation>,
}

/// A qubit parameter for gate declarations.
/// Example: `q` in `declare gate h()(q);`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QubitParam {
    pub name: String,
    pub location: Option<SourceLocation>,
}

/// Target gate declaration — declares a gate that will be provided by the backend.
/// Example: `declare gate rx(theta: a64)(q);`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetGateDecl {
    pub name: String,
    pub params: Vec<GateParam>,
    pub qubits: Vec<QubitParam>,
    pub is_pub: bool,
    pub doc_comment: Option<String>,
    pub location: Option<SourceLocation>,
}

/// Composite gate declaration — a gate defined in terms of other gates.
/// Example: `gate bell()(q0, q1) { h q0; cx (q0, q1); }`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeGateDecl {
    pub name: String,
    pub params: Vec<GateParam>,
    pub qubits: Vec<QubitParam>,
    pub body: Block,
    pub is_pub: bool,
    pub doc_comment: Option<String>,
    pub location: Option<SourceLocation>,
}

// =============================================================================
// Allocator Declarations (Quantum-specific)
// =============================================================================

/// Qubit allocator declaration (maps to SLR AllocatorDecl).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocatorDecl {
    pub name: String,
    pub capacity: u32,
    pub parent: Option<String>,
    pub location: Option<SourceLocation>,
}

/// Classical register declaration (maps to SLR RegisterDecl).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterDecl {
    pub name: String,
    pub size: u32,
    pub is_result: bool,
    pub location: Option<SourceLocation>,
}

// =============================================================================
// Statements
// =============================================================================

/// Statement types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Stmt {
    Binding(Binding),
    Alias(AliasBinding),
    Assign(AssignStmt),
    If(IfStmt),
    For(ForStmt),
    Switch(SwitchStmt),
    Tick(TickStmt),
    TryBlock(TryBlockStmt),
    Return(ReturnStmt),
    Break(BreakStmt),
    Continue(ContinueStmt),
    Defer(DeferStmt),
    Errdefer(ErrDeferStmt),
    Block(Block),
    Expr(ExprStmt),

    // Quantum operations (map directly to SLR)
    Gate(GateOp),
    Prepare(PrepareOp),
    Measure(MeasureOp),
    Barrier(BarrierOp),
}

/// Try block statement for error handling.
/// `try { }` - collect all errors (QEC pattern), returns []E!T
/// `try! { }` - stop on first error (traditional), returns E!T
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TryBlockStmt {
    /// The error handling mode
    pub mode: TryMode,
    /// The block body
    pub body: Block,
    /// Optional catch clause: catch |err| { ... }
    pub catch_clause: Option<CatchClause>,
    pub location: Option<SourceLocation>,
}

/// Error handling mode for try blocks and functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TryMode {
    /// `try` - collect all errors, continue executing (QEC pattern)
    Collect,
    /// `try!` - stop on first error, propagate immediately (traditional)
    Propagate,
}

/// Catch clause for try blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatchClause {
    /// The error capture variable name
    pub capture: String,
    /// The catch body (block or expression)
    pub body: Expr,
    pub location: Option<SourceLocation>,
}

/// Tick statement - a time slice of parallel quantum gates.
/// Maps to PECOS TickCircuit's tick concept.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickStmt {
    /// Optional label for the tick (for QEC rounds, debugging, etc.)
    pub label: Option<String>,
    /// Attributes attached to this tick (e.g., @round(0), @type("syndrome"))
    /// Using Vec to preserve order of declaration
    pub attrs: Vec<Attribute>,
    /// Statements within this tick (gates execute in parallel)
    pub body: Vec<Stmt>,
    pub location: Option<SourceLocation>,
}

/// Assignment statement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssignStmt {
    pub target: Expr, // LValue
    pub op: AssignOp,
    pub value: Expr,
    pub location: Option<SourceLocation>,
}

/// Assignment operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssignOp {
    Assign,    // =
    AddAssign, // +=
    SubAssign, // -=
    MulAssign, // *=
    DivAssign, // /=
    AndAssign, // &=
    OrAssign,  // |=
    XorAssign, // ^=
}

/// If statement.
/// Supports optional unwrapping: if (opt) |value| { ... }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IfStmt {
    pub condition: Expr,
    /// Optional capture variable for unwrapping optionals: if (opt) |value| { ... }
    pub capture: Option<String>,
    pub then_body: Block,
    pub else_body: Option<ElseBranch>,
    pub location: Option<SourceLocation>,
}

/// Else branch (can be else-if or else).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ElseBranch {
    ElseIf(Box<IfStmt>),
    Else(Block),
}

/// For statement (bounded iteration - NASA Power of 10 compliant).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForStmt {
    pub label: Option<String>,
    pub is_inline: bool,
    pub range: ForRange,
    pub captures: Vec<String>,
    pub body: Block,
    pub location: Option<SourceLocation>,
}

/// For loop range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ForRange {
    /// `0..n` or `start..end`
    Range { start: Expr, end: Expr },
    /// Iterate over a collection
    Collection(Expr),
}

/// Switch statement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwitchStmt {
    pub value: Expr,
    pub prongs: Vec<SwitchProng>,
    pub location: Option<SourceLocation>,
}

/// Switch prong.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwitchProng {
    pub cases: Vec<SwitchCase>,
    pub is_else: bool,
    pub body: Expr,
    pub location: Option<SourceLocation>,
}

/// Switch case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwitchCase {
    pub value: Expr,
    pub end: Option<Expr>, // For ranges
    pub location: Option<SourceLocation>,
}

/// Return statement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReturnStmt {
    pub value: Option<Expr>,
    pub location: Option<SourceLocation>,
}

/// Break statement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreakStmt {
    pub label: Option<String>,
    pub value: Option<Expr>,
    pub location: Option<SourceLocation>,
}

/// Continue statement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinueStmt {
    pub label: Option<String>,
    pub location: Option<SourceLocation>,
}

/// Defer statement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeferStmt {
    pub body: Box<Stmt>,
    pub location: Option<SourceLocation>,
}

/// Errdefer statement - executes on error return.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrDeferStmt {
    pub body: Box<Stmt>,
    /// Optional capture name for the error value: errdefer |err| { ... }
    pub capture: Option<String>,
    pub location: Option<SourceLocation>,
}

/// Block of statements.
/// Can have an optional trailing expression that is the block's return value (like Rust).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    pub label: Option<String>,
    /// Attributes attached to this block (e.g., @attr(kind, "syndrome"))
    pub attrs: Vec<Attribute>,
    pub statements: Vec<Stmt>,
    /// Optional trailing expression (block's return value)
    pub trailing_expr: Option<Box<Expr>>,
    pub location: Option<SourceLocation>,
}

/// Expression statement.
/// Can have prefix attributes for annotating gate calls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExprStmt {
    pub expr: Expr,
    /// Attributes attached to this statement (e.g., @syndrome("X") for gates)
    pub attrs: Vec<Attribute>,
    pub location: Option<SourceLocation>,
}

// =============================================================================
// Quantum Operations (map to SLR nodes)
// =============================================================================

/// Gate operation (maps to SLR GateOp).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateOp {
    pub kind: GateKind,
    pub targets: Vec<SlotRef>,
    pub params: Vec<Expr>,
    /// Attributes attached to this gate (e.g., @preserve, @round(0))
    pub attrs: Vec<Attribute>,
    pub location: Option<SourceLocation>,
}

macro_rules! gate_kinds {
    ($($variant:ident => $keyword:literal),+ $(,)?) => {
        /// Gate types (matches SLR GateKind).
        /// Note: Gate names like SXX, RZZ use uppercase for clarity with quantum conventions.
        #[allow(clippy::upper_case_acronyms)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
        pub enum GateKind {
            $($variant),+
        }

        impl GateKind {
            /// Every variant in declaration order.
            pub const ALL: &[GateKind] = &[$(GateKind::$variant),+];

            /// Canonical zlup source keyword for this gate.
            pub fn keyword(&self) -> &'static str {
                match self {
                    $(GateKind::$variant => $keyword),+
                }
            }
        }
    };
}

gate_kinds! {
    // Single-qubit Paulis
    X => "x",
    Y => "y",
    Z => "z",

    // Hadamard
    H => "h",

    // T gates (fourth root of Z)
    T => "t",
    Tdg => "tdg",

    // Square root gates (SZ is the S gate / sqrt(Z))
    SX => "sx",
    SY => "sy",
    SZ => "sz",
    SXdg => "sxdg",
    SYdg => "sydg",
    SZdg => "szdg",

    // Rotation gates (parameterized)
    RX => "rx",
    RY => "ry",
    RZ => "rz",

    // Two-qubit gates
    CX => "cx",
    CY => "cy",
    CZ => "cz",
    CH => "ch",
    SWAP => "swap",
    ISWAP => "iswap",

    // Two-qubit rotation gates
    SXX => "sxx",
    SYY => "syy",
    SZZ => "szz",
    SXXdg => "sxxdg",
    SYYdg => "syydg",
    SZZdg => "szzdg",
    CRZ => "crz",
    RZZ => "rzz",

    // Three-qubit gates
    CCX => "ccx", // Toffoli gate

    // Face rotations
    F => "f",
    Fdg => "fdg",
    F4 => "f4",
    F4dg => "f4dg",

    // Prepare/reset operations (pz = prepare Z, reset to |0⟩)
    PZ => "pz",
}

impl GateKind {
    /// Number of qubit arguments required.
    pub fn arity(&self) -> usize {
        use GateKind::*;
        match self {
            CCX => 3,
            CX | CY | CZ | CH | SWAP | ISWAP | SXX | SYY | SZZ | SXXdg | SYYdg | SZZdg | CRZ
            | RZZ => 2,
            _ => 1,
        }
    }

    /// Whether this gate takes angle parameters.
    pub fn is_parameterized(&self) -> bool {
        use GateKind::*;
        matches!(self, RX | RY | RZ | CRZ | RZZ)
    }

    /// Whether this gate is a preparation/reset operation.
    /// PZ resets qubits to |0⟩ and can be applied to unprepared qubits.
    pub fn is_prepare(&self) -> bool {
        matches!(self, GateKind::PZ)
    }
}

/// Prepare operation (maps to SLR PrepareOp).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrepareOp {
    pub allocator: String,
    pub slots: Option<Vec<u32>>, // None means prepare_all
    pub location: Option<SourceLocation>,
}

/// Measure operation (maps to SLR MeasureOp).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasureOp {
    pub targets: Vec<SlotRef>,
    pub results: Vec<BitRef>,
    pub location: Option<SourceLocation>,
}

/// Barrier operation (maps to SLR BarrierOp).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarrierOp {
    pub allocators: Vec<String>,
    pub location: Option<SourceLocation>,
}

/// Reference to a qubit slot (maps to SLR SlotRef).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotRef {
    pub allocator: String,
    pub index: Box<Expr>, // Can be comptime or runtime
    pub location: Option<SourceLocation>,
}

/// Reference to a classical bit (maps to SLR BitRef).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitRef {
    pub register: String,
    pub index: Box<Expr>,
    pub location: Option<SourceLocation>,
}

// =============================================================================
// Expressions
// =============================================================================

/// Expression types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Expr {
    // Literals
    IntLit(IntLit),
    FloatLit(FloatLit),
    AngleLit(Box<AngleLit>), // 0.25 turns, pi/4 rad - angle with explicit unit
    TypeAscription(Box<TypeAscription>), // 42 u32, 1/4 f64 - expression with type suffix
    BoolLit(BoolLit),
    StringLit(StringLit),
    FString(Box<FStringExpr>), // f"Hello {name}!" - Python-style interpolation
    CharLit(CharLit),
    Null(NullLit),
    Undefined(UndefinedLit),
    Unit(UnitLit),

    // Identifiers and references
    Ident(Ident),
    SlotRef(Box<SlotRef>),
    BitRef(Box<BitRef>),

    // Operators
    Binary(Box<BinaryExpr>),
    Unary(Box<UnaryExpr>),

    // Access
    Field(Box<FieldExpr>),
    Index(Box<IndexExpr>),
    Call(Box<CallExpr>),
    BatchApply(Box<BatchApplyExpr>), // h { q[0], q[1] } - batch gate apply

    // Special
    If(Box<IfExpr>),
    Block(Box<BlockExpr>),
    Comptime(Box<ComptimeExpr>),
    Builtin(Box<BuiltinExpr>),
    AnonStruct(Box<AnonStructExpr>), // struct { x: i32, y: i32 } - anonymous struct type
    StructInit(Box<StructInitExpr>),
    ArrayInit(Box<ArrayInitExpr>),
    BracketArray(Box<BracketArrayExpr>), // [a, b, c] literal
    Tuple(Box<TupleExpr>),               // (a, b) tuple
    Set(Box<SetExpr>),                   // {a, b, c} set literal
    Range(Box<RangeExpr>),
    Measure(Box<MeasureExpr>), // mz(T) targets - measurement
    Gate(Box<GateExpr>),       // h q[0], rx(0.123) q[0] - quantum gate

    // Error/fault handling
    ErrorValue(Box<ErrorValueExpr>), // error.Name literal
    FaultValue(Box<FaultValueExpr>), // fault.Name literal
    Catch(Box<CatchExpr>),           // a catch |err| b
    TryBlock(Box<TryBlockExpr>),     // try { } or try! { } as expression

    // Function literal (for comptime type constructors)
    FnLit(Box<FnDecl>), // fn(params) -> ret { body }

    // Result emission (program output channel - special, never elided)
    Result(Box<ResultExpr>), // result("tag", value) - emit to caller

    // Side-channel communication (sticky/barrier semantics)
    Channel(Box<ChannelExpr>), // @emit.channel.command(...) - log, sim, hw, custom
}

/// Try block as expression.
/// `errors := try { ... };`
/// `result := try! { ... } catch |err| { default };`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TryBlockExpr {
    /// The error handling mode
    pub mode: TryMode,
    /// The block body
    pub body: Block,
    /// Optional catch clause
    pub catch_clause: Option<CatchClause>,
    pub location: Option<SourceLocation>,
}

impl Expr {
    /// Get the source location of this expression, if available.
    pub fn get_location(&self) -> Option<SourceLocation> {
        match self {
            Expr::IntLit(lit) => lit.location.clone(),
            Expr::FloatLit(lit) => lit.location.clone(),
            Expr::AngleLit(lit) => lit.location.clone(),
            Expr::TypeAscription(asc) => asc.location.clone(),
            Expr::BoolLit(lit) => lit.location.clone(),
            Expr::StringLit(lit) => lit.location.clone(),
            Expr::FString(fstr) => fstr.location.clone(),
            Expr::CharLit(lit) => lit.location.clone(),
            Expr::Null(lit) => lit.location.clone(),
            Expr::Undefined(lit) => lit.location.clone(),
            Expr::Unit(lit) => lit.location.clone(),
            Expr::Ident(ident) => ident.location.clone(),
            Expr::SlotRef(slot) => slot.location.clone(),
            Expr::BitRef(bit) => bit.location.clone(),
            Expr::Binary(binary) => binary.location.clone(),
            Expr::Unary(unary) => unary.location.clone(),
            Expr::Field(field) => field.location.clone(),
            Expr::Index(index) => index.location.clone(),
            Expr::Call(call) => call.location.clone(),
            Expr::BatchApply(batch) => batch.location.clone(),
            Expr::If(if_expr) => if_expr.location.clone(),
            Expr::Block(block) => block.location.clone(),
            Expr::Comptime(comptime) => comptime.location.clone(),
            Expr::Builtin(builtin) => builtin.location.clone(),
            Expr::AnonStruct(anon) => anon.location.clone(),
            Expr::StructInit(init) => init.location.clone(),
            Expr::ArrayInit(init) => init.location.clone(),
            Expr::BracketArray(arr) => arr.location.clone(),
            Expr::Tuple(tuple) => tuple.location.clone(),
            Expr::Set(set) => set.location.clone(),
            Expr::Range(range) => range.location.clone(),
            Expr::Measure(measure) => measure.location.clone(),
            Expr::Gate(gate) => gate.location.clone(),
            Expr::ErrorValue(err) => err.location.clone(),
            Expr::FaultValue(fault) => fault.location.clone(),
            Expr::Catch(catch) => catch.location.clone(),
            Expr::TryBlock(try_block) => try_block.location.clone(),
            Expr::FnLit(func) => func.location.clone(),
            Expr::Result(result) => result.location.clone(),
            Expr::Channel(channel) => channel.location.clone(),
        }
    }
}

/// Integer literal with optional type suffix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntLit {
    pub value: i128,
    /// Type suffix (e.g., "u32", "i8", "usize")
    pub suffix: Option<String>,
    pub location: Option<SourceLocation>,
}

/// Float literal with optional type suffix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FloatLit {
    pub value: f64,
    /// Type suffix (e.g., "f32", "f64")
    pub suffix: Option<String>,
    pub location: Option<SourceLocation>,
}

/// Angle literal with explicit unit: `0.25 turns` or `pi/4 rad`
/// Units: turns (native, 1 turn = full rotation), rad (radians)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AngleLit {
    /// The numeric value expression (can be a literal or arithmetic like pi/4)
    pub value: Expr,
    /// The angle unit
    pub unit: AngleUnit,
    pub location: Option<SourceLocation>,
}

/// Type ascription: expression with explicit type suffix
/// Examples: `42 u32`, `1/4 f64`, `(a + b) i64`
/// Allows type annotation on expressions with space for readability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeAscription {
    /// The expression to type
    pub value: Expr,
    /// The type name as a string (e.g., "u32", "f64", "a64")
    pub type_name: String,
    pub location: Option<SourceLocation>,
}

/// Angle unit specifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AngleUnit {
    /// Turns - native unit, 1 turn = full rotation (360 degrees)
    /// Common values: 0.25 = quarter turn, 0.5 = half turn, 0.125 = T gate
    Turns,
    /// Radians - mathematical convention
    /// pi/2 = quarter turn, pi = half turn
    Rad,
}

impl AngleUnit {
    /// Convert a value in this unit to turns (the native unit)
    pub fn to_turns(&self, value: f64) -> f64 {
        match self {
            AngleUnit::Turns => value,
            AngleUnit::Rad => value / (2.0 * std::f64::consts::PI),
        }
    }

    /// Convert a value in turns to this unit
    pub fn from_turns(&self, turns: f64) -> f64 {
        match self {
            AngleUnit::Turns => turns,
            AngleUnit::Rad => turns * 2.0 * std::f64::consts::PI,
        }
    }
}

/// Boolean literal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoolLit {
    pub value: bool,
    pub location: Option<SourceLocation>,
}

/// String literal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StringLit {
    pub value: String,
    pub location: Option<SourceLocation>,
}

/// F-string (Python-style interpolated string): f"Hello {name}!"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FStringExpr {
    pub parts: Vec<FStringPart>,
    pub location: Option<SourceLocation>,
}

/// A part of an f-string - either literal text or an interpolated expression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FStringPart {
    /// Literal text portion
    Text(String),
    /// Interpolated expression with optional format spec: {expr} or {expr:.2f}
    Expr {
        expr: Expr,
        /// Optional format specifier (e.g., ".2f", ">10", "08d")
        format: Option<String>,
    },
}

/// Character literal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharLit {
    pub value: char,
    pub location: Option<SourceLocation>,
}

/// Null literal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NullLit {
    pub location: Option<SourceLocation>,
}

/// Undefined literal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UndefinedLit {
    pub location: Option<SourceLocation>,
}

/// Unit literal - the single value of the unit type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnitLit {
    pub location: Option<SourceLocation>,
}

/// Identifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ident {
    pub name: String,
    pub location: Option<SourceLocation>,
}

/// Binary expression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryExpr {
    pub op: BinaryOp,
    pub left: Expr,
    pub right: Expr,
    pub location: Option<SourceLocation>,
}

/// Binary operators (matches SLR BinaryOp).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Mod,

    // Comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    // Membership (for sets)
    In,
    NotIn,

    // Logical
    And,
    Or,

    // Optional
    Orelse, // a orelse b - returns a if not null, else b

    // Error handling
    Catch, // a catch |err| b - unwrap error union or handle error

    // Bitwise
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

/// Unary expression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnaryExpr {
    pub op: UnaryOp,
    pub operand: Expr,
    pub location: Option<SourceLocation>,
}

/// Unary operators (matches SLR UnaryOp).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOp {
    Neg,            // -
    Not,            // !
    BitNot,         // ~
    AddrOf,         // &
    Deref,          // *
    OptionalUnwrap, // .?
    ErrorUnwrap,    // .!
    Try,            // try - error propagation
}

/// Field access expression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldExpr {
    pub object: Expr,
    pub field: String,
    pub location: Option<SourceLocation>,
}

/// Index expression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexExpr {
    pub object: Expr,
    pub index: Expr,
    pub location: Option<SourceLocation>,
}

/// Call expression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallExpr {
    pub callee: Expr,
    pub args: Vec<Expr>,
    pub location: Option<SourceLocation>,
}

/// Batch apply expression: h { q[0], q[1] } or rz(pi/4) { q[0], q[1] }
/// For gates where application order doesn't matter (set semantics).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchApplyExpr {
    /// The gate/operation being applied (may include params, e.g., rz(pi/4))
    pub operation: Expr,
    /// The targets (qubits or qubit pairs) to apply to
    pub targets: Vec<Expr>,
    pub location: Option<SourceLocation>,
}

/// Measurement expression: mz(T) targets or mz(pack T) targets
///
/// Per-qubit mode (pack=false): Each qubit produces one T, count must match exactly.
/// Pack mode (pack=true): Bits fill T sequentially, T must have enough capacity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasureExpr {
    /// The result type (e.g., u1, [4]u1, u8, Syndrome)
    pub result_type: TypeExpr,
    /// Whether to pack bits into the type (vs per-qubit results)
    pub pack: bool,
    /// The targets to measure (array literal, variable, or slice)
    pub targets: Expr,
    pub location: Option<SourceLocation>,
}

/// Gate expression: gate target or gate(params) target
/// Consistent DSL-like syntax for quantum gates.
///
/// Examples:
/// - `h q[0]` - single qubit gate
/// - `cx (q[0], q[1])` - two-qubit gate with tuple
/// - `rx(0.123) q[0]` - parameterized gate
/// - `h {q[0], q[1]}` - batch apply (set semantics)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateExpr {
    /// The gate kind (H, X, CX, RZ, etc.)
    pub kind: GateKind,
    /// Parameters for parameterized gates (e.g., rotation angle)
    pub params: Vec<Expr>,
    /// The target(s) - single qubit, tuple, array, or set
    pub target: Expr,
    pub location: Option<SourceLocation>,
}

/// If expression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IfExpr {
    pub condition: Expr,
    pub then_expr: Expr,
    pub else_expr: Expr,
    pub location: Option<SourceLocation>,
}

/// Block expression (labeled block that returns a value).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockExpr {
    pub label: String,
    /// Attributes attached to this block (e.g., @attr(kind, "syndrome"))
    pub attrs: Vec<Attribute>,
    pub statements: Vec<Stmt>,
    /// Optional trailing expression (block's return value)
    pub trailing_expr: Option<Box<Expr>>,
    pub location: Option<SourceLocation>,
}

/// Comptime expression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComptimeExpr {
    pub inner: Expr,
    pub location: Option<SourceLocation>,
}

/// Builtin call (@import, @This, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuiltinExpr {
    pub name: String, // Without the @
    pub args: Vec<Expr>,
    pub location: Option<SourceLocation>,
}

/// Anonymous struct type definition.
/// `struct { x: i32, y: i32 }` creates an anonymous struct type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnonStructExpr {
    pub fields: Vec<StructField>,
    pub is_packed: bool,
    pub location: Option<SourceLocation>,
}

/// Struct initialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructInitExpr {
    pub ty: Option<TypeExpr>,
    pub fields: Vec<FieldInit>,
    pub location: Option<SourceLocation>,
}

/// Field initializer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldInit {
    pub name: String,
    pub value: Expr,
    pub location: Option<SourceLocation>,
}

/// Array initialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArrayInitExpr {
    pub ty: Option<TypeExpr>,
    pub elements: Vec<Expr>,
    pub location: Option<SourceLocation>,
}

/// Bracket array literal: [a, b, c]
/// Used for batch quantum operations like h(&[q[0], q[1], q[2]])
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BracketArrayExpr {
    pub elements: Vec<Expr>,
    pub location: Option<SourceLocation>,
}

/// Tuple expression: (a, b) or (a, b, c)
/// Used for two-qubit gate pairs like cx(&[(q[0], q[1]), (q[2], q[3])])
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TupleExpr {
    pub elements: Vec<Expr>,
    pub location: Option<SourceLocation>,
}

/// Set expression: {a, b, c}
/// Unique unordered elements, backed by BTreeSet at runtime
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetExpr {
    pub elements: Vec<Expr>,
    pub element_type: Option<TypeExpr>, // For empty_set: Set(T){}
    pub location: Option<SourceLocation>,
}

/// Range expression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeExpr {
    pub start: Option<Expr>,
    pub end: Option<Expr>,
    pub location: Option<SourceLocation>,
}

/// Error value literal: error.OutOfMemory, error.InvalidArgument, etc.
/// Represents a specific error value from an error set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorValueExpr {
    pub name: String,
    pub location: Option<SourceLocation>,
}

/// Fault value literal: fault.Leakage, fault.QubitLoss, etc.
/// Represents a specific fault value from a fault set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultValueExpr {
    pub name: String,
    pub location: Option<SourceLocation>,
}

/// Catch expression: `expr catch |err| handler`
/// Unwraps an error union, returning the payload if successful,
/// or evaluating the handler with the error if it fails.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatchExpr {
    pub operand: Expr,
    /// Optional capture variable for the error value
    pub capture: Option<String>,
    pub handler: Expr,
    pub location: Option<SourceLocation>,
}

// =============================================================================
// Channel Expression (Unified Side-Channel Communication)
// =============================================================================

/// Channel expression for side-channel communication.
///
/// Unified syntax: `@emit.channel.command(args)`
///
/// All channel expressions use the `@emit` prefix and have sticky/barrier semantics.
/// The behavior (elision, barriers) depends on target and channel configuration.
///
/// Built-in channels:
/// - `@emit.log.*` - Logging (trace, debug, info, warn, error, at)
/// - `@emit.sim.*` - Simulator control (send, noise_enable, noise_disable)
/// - `@emit.hw.*` - Hardware communication (send, calibrate, etc.)
///
/// Custom channels can be defined for instrumentation, timing, debugging, etc.
///
/// Examples:
/// - `@emit.log.trace(f"detailed message")`
/// - `@emit.log.debug("namespace", f"message")`
/// - `@emit.log.info(f"msg", data: obj)`
/// - `@emit.sim.send("seed", 42)`
/// - `@emit.sim.noise_disable()`
/// - `@emit.hw.send("calibration", params)`
/// - `@emit.timing.send("checkpoint", t)`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelExpr {
    /// Channel name (log, sim, hw, timing, debug, etc.)
    pub channel: String,
    /// Command name (send, trace, debug, noise_enable, etc.)
    pub command: String,
    /// Arguments (positional and/or named)
    pub args: Vec<ChannelArg>,
    pub location: Option<SourceLocation>,
}

/// Argument to a channel command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChannelArg {
    /// Positional argument
    Positional(Expr),
    /// Named argument (name: value)
    Named { name: String, value: Expr },
}

impl ChannelArg {
    /// Get the expression value of this argument.
    pub fn value(&self) -> &Expr {
        match self {
            ChannelArg::Positional(e) => e,
            ChannelArg::Named { value, .. } => value,
        }
    }

    /// Get the name if this is a named argument.
    pub fn name(&self) -> Option<&str> {
        match self {
            ChannelArg::Positional(_) => None,
            ChannelArg::Named { name, .. } => Some(name),
        }
    }
}

// =============================================================================
// Result Expression (Program Output Channel)
// =============================================================================

/// Result emission expression.
///
/// `result(tag, value)` emits a tagged value as program output.
/// This is the primary way to return structured data from quantum programs
/// back to the caller/orchestrator. Unlike logs, results are NEVER elided.
///
/// Examples:
/// - `result("measurement", m)` - simple result
/// - `result("qec/syndrome", syndrome)` - namespaced with / convention
/// - `result("round_1/parity", parity)` - hierarchical naming
///
/// The tag must be a compile-time string literal (like Guppy).
/// Value can be any serializable type: int, bool, float, arrays.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultExpr {
    /// The tag/key for this result (compile-time string literal)
    pub tag: String,
    /// The value to emit
    pub value: Expr,
    pub location: Option<SourceLocation>,
}

// =============================================================================
// Types
// =============================================================================

/// Type expressions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TypeExpr {
    // Primitive types
    Primitive(PrimitiveType),

    // Quantum types
    Qubit,
    Bit,
    QAlloc(Option<Box<Expr>>), // qalloc or qalloc(N)

    // Compound types
    Array(Box<ArrayType>),
    Pointer(Box<PointerType>),
    Optional(Box<TypeExpr>),
    ErrorUnion(Box<ErrorUnionType>), // E!T - single error, either/or
    CollectedErrors(Box<CollectedErrorsType>), // []E!T - collected errors, both
    Fn(Box<FnType>),
    Tuple(Vec<TypeExpr>),
    Set(Box<TypeExpr>), // Set(T) - unordered unique elements

    // Inline/anonymous struct type: struct { x: i32, y: i32 }
    Struct(Box<InlineStructType>),
    // Inline/anonymous enum type: enum { a, b, c }
    Enum(Box<InlineEnumType>),

    // Named type
    Named(TypePath),

    // Special
    Type,    // The type `type`
    AnyType, // anytype for generic params
    Unit,    // unit type - has exactly one value
}

/// Primitive types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrimitiveType {
    // Arbitrary-width integers (like Zig: u1, u4, u7, u128, etc.)
    UInt { bits: u16 }, // Unsigned integer with N bits
    IInt { bits: u16 }, // Signed integer with N bits
    Usize,              // Platform-dependent unsigned size
    Isize,              // Platform-dependent signed size
    // Floating point
    F16,
    F32,
    F64,
    F128,
    A64, // Angle type (64-bit, maps to PECOS Angle64)
    Bool,
}

/// Array type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArrayType {
    pub element: TypeExpr,
    pub size: Option<Expr>, // None for slices
    pub sentinel: Option<Expr>,
}

/// Pointer type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointerType {
    pub pointee: TypeExpr,
    pub is_const: bool,
    pub is_many: bool, // [*] vs *
    pub sentinel: Option<Expr>,
}

/// Error union type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorUnionType {
    pub error_type: TypeExpr,
    pub payload_type: TypeExpr,
}

/// Collected errors type: []E!T
/// Represents "array of error E, with value T" (both, not either/or).
/// Used for QEC-style error collection where all operations execute
/// and errors are collected rather than stopping on first error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectedErrorsType {
    /// The error type (e.g., QuantumError)
    pub error_type: TypeExpr,
    /// The payload type (e.g., void or u1)
    pub payload_type: TypeExpr,
}

/// Function type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FnType {
    pub params: Vec<TypeExpr>,
    pub return_type: Option<TypeExpr>,
}

/// Inline/anonymous struct type: struct { x: i32, y: i32 }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InlineStructType {
    pub fields: Vec<StructField>,
    pub is_packed: bool,
}

/// Inline/anonymous enum type: enum { a, b, c }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InlineEnumType {
    pub variants: Vec<EnumVariant>,
    pub tag_type: Option<TypeExpr>,
}

/// Type path (e.g., `std.mem.Allocator`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypePath {
    pub segments: Vec<String>,
    pub location: Option<SourceLocation>,
}

// =============================================================================
// Convenience Implementations
// =============================================================================

impl From<&str> for Ident {
    fn from(name: &str) -> Self {
        Ident {
            name: name.to_string(),
            location: None,
        }
    }
}

impl From<i128> for Expr {
    fn from(value: i128) -> Self {
        Expr::IntLit(IntLit {
            value,
            suffix: None,
            location: None,
        })
    }
}

impl From<f64> for Expr {
    fn from(value: f64) -> Self {
        Expr::FloatLit(FloatLit {
            value,
            suffix: None,
            location: None,
        })
    }
}

impl From<bool> for Expr {
    fn from(value: bool) -> Self {
        Expr::BoolLit(BoolLit {
            value,
            location: None,
        })
    }
}

impl From<String> for Expr {
    fn from(value: String) -> Self {
        Expr::StringLit(StringLit {
            value,
            location: None,
        })
    }
}
