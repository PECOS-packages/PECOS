//! Guppy AST - Clean Rust representation of Guppy programs.
//!
//! This module defines a Rust AST for Guppy quantum programs. It provides
//! a clean, typed representation that is independent of the Python parser.

use serde::{Deserialize, Serialize};

/// A source span for error reporting.
#[derive(Debug, Clone, Copy, Default)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }
}

/// A Guppy module (compilation unit).
#[derive(Debug, Clone)]
pub struct Module {
    pub functions: Vec<Function>,
    pub span: Span,
}

/// A function definition.
#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: Option<Type>,
    pub body: Vec<Stmt>,
    pub decorators: Vec<String>,
    pub span: Span,
}

/// A function parameter.
#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub ty: Option<Type>,
    pub span: Span,
}

/// Type expressions.
#[derive(Debug, Clone)]
pub enum Type {
    /// Primitive types: int, float, bool, str, None
    Primitive(PrimitiveType),
    /// Qubit register: qubit[n]
    Qubit { size: Option<Box<Expr>> },
    /// Array/list type: list[T]
    Array { element: Box<Type> },
    /// Optional type: Optional[T]
    Optional { inner: Box<Type> },
    /// Named/custom type
    Named { name: String },
    /// Tuple type
    Tuple { elements: Vec<Type> },
    /// Unknown/unresolved type
    Unknown,
}

/// Primitive types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrimitiveType {
    Int,
    Float,
    Bool,
    Str,
    None,
}

/// Statements in Guppy.
#[derive(Debug, Clone)]
pub enum Stmt {
    /// Qubit allocation: q = qubit[n]
    Qalloc {
        name: String,
        size: Expr,
        span: Span,
    },

    /// Quantum gate application: h(q[0]), cx(q[0], q[1])
    Gate {
        gate: GateKind,
        targets: Vec<Expr>,
        params: Vec<Expr>,
        span: Span,
    },

    /// Measurement: m = measure(q)
    Measure {
        target: Expr,
        result: Option<String>,
        span: Span,
    },

    /// For loop: for i in range(n)
    For {
        var: String,
        iter: ForIter,
        body: Vec<Stmt>,
        orelse: Vec<Stmt>,
        span: Span,
    },

    /// While loop: while cond
    While {
        test: Expr,
        body: Vec<Stmt>,
        orelse: Vec<Stmt>,
        span: Span,
    },

    /// If statement
    If {
        test: Expr,
        body: Vec<Stmt>,
        orelse: Vec<Stmt>,
        span: Span,
    },

    /// Assignment: x = expr
    Assign {
        target: AssignTarget,
        value: Expr,
        span: Span,
    },

    /// Annotated assignment: x: T = expr
    AnnAssign {
        target: String,
        annotation: Type,
        value: Option<Expr>,
        span: Span,
    },

    /// Augmented assignment: x += expr
    AugAssign {
        target: AssignTarget,
        op: BinOp,
        value: Expr,
        span: Span,
    },

    /// Return statement
    Return { value: Option<Expr>, span: Span },

    /// Expression statement
    Expr { value: Expr, span: Span },

    /// Pass statement
    Pass { span: Span },

    /// Break statement
    Break { span: Span },

    /// Continue statement
    Continue { span: Span },

    /// Barrier (quantum synchronization)
    Barrier { qubits: Vec<Expr>, span: Span },

    /// Try/except block
    Try {
        body: Vec<Stmt>,
        handlers: Vec<ExceptHandler>,
        orelse: Vec<Stmt>,
        finalbody: Vec<Stmt>,
        span: Span,
    },

    /// Assert statement
    Assert {
        test: Expr,
        msg: Option<Expr>,
        span: Span,
    },

    /// With statement (context manager)
    With {
        items: Vec<WithItem>,
        body: Vec<Stmt>,
        span: Span,
    },
}

/// Exception handler.
#[derive(Debug, Clone)]
pub struct ExceptHandler {
    pub ty: Option<Expr>,
    pub name: Option<String>,
    pub body: Vec<Stmt>,
    pub span: Span,
}

/// With item (context manager).
#[derive(Debug, Clone)]
pub struct WithItem {
    pub context: Expr,
    pub target: Option<AssignTarget>,
    pub span: Span,
}

/// Assignment target.
#[derive(Debug, Clone)]
pub enum AssignTarget {
    /// Simple name: x
    Name { name: String, span: Span },
    /// Subscript: x[i]
    Subscript {
        value: Box<Expr>,
        slice: Box<Expr>,
        span: Span,
    },
    /// Attribute: x.attr
    Attribute {
        value: Box<Expr>,
        attr: String,
        span: Span,
    },
    /// Tuple unpacking: (a, b)
    Tuple { elts: Vec<AssignTarget>, span: Span },
}

/// For loop iterator.
#[derive(Debug, Clone)]
pub enum ForIter {
    /// range(end) or range(start, end) or range(start, end, step)
    Range {
        start: Option<Box<Expr>>,
        end: Box<Expr>,
        step: Option<Box<Expr>>,
    },
    /// Arbitrary iterable
    Iter(Box<Expr>),
}

/// Expressions in Guppy.
#[derive(Debug, Clone)]
pub enum Expr {
    /// Integer literal
    IntLit { value: i64, span: Span },

    /// Float literal
    FloatLit { value: f64, span: Span },

    /// String literal
    StrLit { value: String, span: Span },

    /// Boolean literal
    BoolLit { value: bool, span: Span },

    /// None literal
    NoneLit { span: Span },

    /// Identifier/name
    Name { name: String, span: Span },

    /// Subscript: a[i]
    Subscript {
        value: Box<Expr>,
        slice: Box<Expr>,
        span: Span,
    },

    /// Attribute access: a.b
    Attribute {
        value: Box<Expr>,
        attr: String,
        span: Span,
    },

    /// Binary operation: a + b
    BinOp {
        left: Box<Expr>,
        op: BinOp,
        right: Box<Expr>,
        span: Span,
    },

    /// Unary operation: -a, not a
    UnaryOp {
        op: UnaryOp,
        operand: Box<Expr>,
        span: Span,
    },

    /// Comparison: a < b, a == b
    Compare {
        left: Box<Expr>,
        ops: Vec<CmpOp>,
        comparators: Vec<Expr>,
        span: Span,
    },

    /// Boolean operation: a and b, a or b
    BoolOp {
        op: BoolOpKind,
        values: Vec<Expr>,
        span: Span,
    },

    /// Function call: f(a, b)
    Call {
        func: Box<Expr>,
        args: Vec<Expr>,
        keywords: Vec<Keyword>,
        span: Span,
    },

    /// Conditional expression: a if cond else b
    IfExp {
        test: Box<Expr>,
        body: Box<Expr>,
        orelse: Box<Expr>,
        span: Span,
    },

    /// List literal: [a, b, c]
    List { elts: Vec<Expr>, span: Span },

    /// Tuple literal: (a, b, c)
    Tuple { elts: Vec<Expr>, span: Span },

    /// Dict literal: {a: b, c: d}
    Dict {
        keys: Vec<Option<Expr>>,
        values: Vec<Expr>,
        span: Span,
    },

    /// Set literal: {a, b, c}
    Set { elts: Vec<Expr>, span: Span },

    /// List comprehension: [x for x in xs if cond]
    ListComp {
        elt: Box<Expr>,
        generators: Vec<Comprehension>,
        span: Span,
    },

    /// Dict comprehension: {k: v for k, v in items}
    DictComp {
        key: Box<Expr>,
        value: Box<Expr>,
        generators: Vec<Comprehension>,
        span: Span,
    },

    /// Set comprehension: {x for x in xs}
    SetComp {
        elt: Box<Expr>,
        generators: Vec<Comprehension>,
        span: Span,
    },

    /// Generator expression: (x for x in xs)
    GeneratorExp {
        elt: Box<Expr>,
        generators: Vec<Comprehension>,
        span: Span,
    },

    /// Lambda: lambda x: x + 1
    Lambda {
        params: Vec<String>,
        body: Box<Expr>,
        span: Span,
    },
}

impl Expr {
    /// Get the span of this expression.
    pub fn span(&self) -> Span {
        match self {
            Expr::IntLit { span, .. }
            | Expr::FloatLit { span, .. }
            | Expr::StrLit { span, .. }
            | Expr::BoolLit { span, .. }
            | Expr::NoneLit { span }
            | Expr::Name { span, .. }
            | Expr::Subscript { span, .. }
            | Expr::Attribute { span, .. }
            | Expr::BinOp { span, .. }
            | Expr::UnaryOp { span, .. }
            | Expr::Compare { span, .. }
            | Expr::BoolOp { span, .. }
            | Expr::Call { span, .. }
            | Expr::IfExp { span, .. }
            | Expr::List { span, .. }
            | Expr::Tuple { span, .. }
            | Expr::Dict { span, .. }
            | Expr::Set { span, .. }
            | Expr::ListComp { span, .. }
            | Expr::DictComp { span, .. }
            | Expr::SetComp { span, .. }
            | Expr::GeneratorExp { span, .. }
            | Expr::Lambda { span, .. } => *span,
        }
    }
}

/// Keyword argument.
#[derive(Debug, Clone)]
pub struct Keyword {
    pub name: Option<String>,
    pub value: Expr,
}

/// Comprehension clause.
#[derive(Debug, Clone)]
pub struct Comprehension {
    pub target: AssignTarget,
    pub iter: Expr,
    pub ifs: Vec<Expr>,
    pub is_async: bool,
}

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinOp {
    Add,
    Sub,
    Mult,
    Div,
    FloorDiv,
    Mod,
    Pow,
    LShift,
    RShift,
    BitOr,
    BitXor,
    BitAnd,
    MatMult,
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Invert, // ~
    Not,    // not
    UAdd,   // +
    USub,   // -
}

/// Comparison operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Eq,
    NotEq,
    Lt,
    LtE,
    Gt,
    GtE,
    Is,
    IsNot,
    In,
    NotIn,
}

/// Boolean operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoolOpKind {
    And,
    Or,
}

/// Quantum gate kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateKind {
    // Single-qubit gates
    H,
    X,
    Y,
    Z,
    T,
    Tdg,
    S,
    Sdg,
    Sx,
    Sy,
    Sz,
    // Parameterized single-qubit gates
    Rx,
    Ry,
    Rz,
    // Two-qubit gates
    Cx,
    Cy,
    Cz,
    Swap,
    Iswap,
    // Three-qubit gates
    Ccx,
    // Other operations
    Pz, // Reset
    // Generic/unknown gate
    Custom(u32), // Index into a name table
}

impl GateKind {
    /// Parse a gate name into a GateKind.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "h" | "hadamard" => Some(GateKind::H),
            "x" | "pauli_x" => Some(GateKind::X),
            "y" | "pauli_y" => Some(GateKind::Y),
            "z" | "pauli_z" => Some(GateKind::Z),
            "t" => Some(GateKind::T),
            "tdg" | "t_dagger" => Some(GateKind::Tdg),
            "s" => Some(GateKind::S),
            "sdg" | "s_dagger" => Some(GateKind::Sdg),
            "sx" | "sqrt_x" => Some(GateKind::Sx),
            "sy" | "sqrt_y" => Some(GateKind::Sy),
            "sz" | "sqrt_z" => Some(GateKind::Sz),
            "rx" => Some(GateKind::Rx),
            "ry" => Some(GateKind::Ry),
            "rz" => Some(GateKind::Rz),
            "cx" | "cnot" => Some(GateKind::Cx),
            "cy" => Some(GateKind::Cy),
            "cz" => Some(GateKind::Cz),
            "swap" => Some(GateKind::Swap),
            "iswap" => Some(GateKind::Iswap),
            "ccx" | "toffoli" | "ccnot" => Some(GateKind::Ccx),
            "pz" | "reset" => Some(GateKind::Pz),
            _ => None,
        }
    }

    /// Check if this gate is parameterized.
    pub fn is_parameterized(&self) -> bool {
        matches!(self, GateKind::Rx | GateKind::Ry | GateKind::Rz)
    }

    /// Get the number of qubits this gate operates on.
    pub fn num_qubits(&self) -> usize {
        match self {
            GateKind::H
            | GateKind::X
            | GateKind::Y
            | GateKind::Z
            | GateKind::T
            | GateKind::Tdg
            | GateKind::S
            | GateKind::Sdg
            | GateKind::Sx
            | GateKind::Sy
            | GateKind::Sz
            | GateKind::Rx
            | GateKind::Ry
            | GateKind::Rz
            | GateKind::Pz => 1,
            GateKind::Cx | GateKind::Cy | GateKind::Cz | GateKind::Swap | GateKind::Iswap => 2,
            GateKind::Ccx => 3,
            GateKind::Custom(_) => 0, // Unknown
        }
    }

    /// Get the name of this gate.
    pub fn name(&self) -> &'static str {
        match self {
            GateKind::H => "h",
            GateKind::X => "x",
            GateKind::Y => "y",
            GateKind::Z => "z",
            GateKind::T => "t",
            GateKind::Tdg => "tdg",
            GateKind::S => "s",
            GateKind::Sdg => "sdg",
            GateKind::Sx => "sx",
            GateKind::Sy => "sy",
            GateKind::Sz => "sz",
            GateKind::Rx => "rx",
            GateKind::Ry => "ry",
            GateKind::Rz => "rz",
            GateKind::Cx => "cx",
            GateKind::Cy => "cy",
            GateKind::Cz => "cz",
            GateKind::Swap => "swap",
            GateKind::Iswap => "iswap",
            GateKind::Ccx => "ccx",
            GateKind::Pz => "pz",
            GateKind::Custom(_) => "custom",
        }
    }
}
