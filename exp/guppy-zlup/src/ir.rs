//! Guppy IR types and emitter.
//!
//! This module defines the intermediate representation for validated Guppy programs
//! and provides functions to emit IR from Python source code.

use rustpython_parser::ast::{self, Constant, Expr as PyExpr, Mod, Stmt as PyStmt};
use serde::{Deserialize, Serialize};

pub const IR_VERSION: &str = "0.1.0";

/// Expression kinds in the IR.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExprKind {
    Literal,
    Ident,
    Index,
    Binary,
    Unary,
    Call,
    Field,
    Tuple,
}

/// Statement kinds in the IR.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StmtKind {
    Qalloc,
    Gate,
    Measure,
    For,
    While,
    If,
    Assign,
    Return,
    Expr,
    Binding,
    Barrier,
    Break,
    Continue,
    /// Result emission: result(tag, value)
    Result,
}

/// Gate types.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum GateKind {
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
    Rx,
    Ry,
    Rz,
    Cx,
    Cy,
    Cz,
    Swap,
    Iswap,
    Ccx,
    Pz,
}

impl GateKind {
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "h" => Some(GateKind::H),
            "x" => Some(GateKind::X),
            "y" => Some(GateKind::Y),
            "z" => Some(GateKind::Z),
            "t" => Some(GateKind::T),
            "tdg" => Some(GateKind::Tdg),
            "s" => Some(GateKind::S),
            "sdg" => Some(GateKind::Sdg),
            "sx" => Some(GateKind::Sx),
            "sy" => Some(GateKind::Sy),
            "sz" => Some(GateKind::Sz),
            "rx" => Some(GateKind::Rx),
            "ry" => Some(GateKind::Ry),
            "rz" => Some(GateKind::Rz),
            "cx" | "cnot" => Some(GateKind::Cx),
            "cy" => Some(GateKind::Cy),
            "cz" => Some(GateKind::Cz),
            "swap" => Some(GateKind::Swap),
            "iswap" => Some(GateKind::Iswap),
            "ccx" | "toffoli" => Some(GateKind::Ccx),
            "pz" | "reset" => Some(GateKind::Pz),
            _ => None,
        }
    }

    pub fn to_zlup_name(&self) -> &'static str {
        match self {
            GateKind::H => "h",
            GateKind::X => "x",
            GateKind::Y => "y",
            GateKind::Z => "z",
            GateKind::T => "t",
            GateKind::Tdg => "tdg",
            GateKind::S => "sz",
            GateKind::Sdg => "szdg",
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
        }
    }

    pub fn arity(&self) -> usize {
        match self {
            GateKind::Ccx => 3,
            GateKind::Cx | GateKind::Cy | GateKind::Cz | GateKind::Swap | GateKind::Iswap => 2,
            _ => 1,
        }
    }

    pub fn is_parameterized(&self) -> bool {
        matches!(self, GateKind::Rx | GateKind::Ry | GateKind::Rz)
    }
}

/// Source location.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceLocation {
    pub line: u32,
    pub column: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_line: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_column: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
}

/// Type expression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeExpr {
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub element: Option<Box<TypeExpr>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<Box<Expr>>,
    /// For tuple types: the element types
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub elements: Vec<TypeExpr>,
}

/// Expression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Expr {
    pub kind: ExprKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub array: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<Box<Expr>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub op: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub left: Option<Box<Expr>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub right: Option<Box<Expr>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operand: Option<Box<Expr>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub callee: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<Expr>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub object: Option<Box<Expr>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub field: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<SourceLocation>,
}

/// Range expression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeExpr {
    pub start: Expr,
    pub end: Expr,
}

/// Statement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stmt {
    pub kind: StmtKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<Expr>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gate: Option<GateKind>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub targets: Vec<Expr>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub params: Vec<Expr>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub results: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub var: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub range: Option<RangeExpr>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condition: Option<Expr>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub then_body: Vec<Stmt>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub else_body: Vec<Stmt>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<Expr>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<Expr>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_value: Option<Expr>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expr: Option<Expr>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub ty: Option<TypeExpr>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_mutable: Option<bool>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub body: Vec<Stmt>,
    /// Tag for result statements: result(tag, value)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tag: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<SourceLocation>,
}

/// Function parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Param {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: TypeExpr,
}

/// Function definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub params: Vec<Param>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_type: Option<TypeExpr>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub body: Vec<Stmt>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_pub: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<SourceLocation>,
}

/// Root IR node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuppyIR {
    pub version: String,
    pub functions: Vec<Function>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_file: Option<String>,
}

impl GuppyIR {
    pub fn new() -> Self {
        Self {
            version: IR_VERSION.to_string(),
            functions: Vec::new(),
            source_file: None,
        }
    }
}

impl Default for GuppyIR {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// IR Validation
// =============================================================================

/// IR validation error.
#[derive(Debug, Clone)]
pub enum ValidationError {
    /// Missing required field for statement kind.
    MissingField {
        kind: StmtKind,
        field: &'static str,
        location: Option<SourceLocation>,
    },
    /// Missing required field for expression kind.
    MissingExprField { kind: ExprKind, field: &'static str },
    /// Undefined variable reference.
    UndefinedVariable {
        name: String,
        location: Option<SourceLocation>,
    },
    /// Undefined allocator (qubit register) reference.
    UndefinedAllocator {
        name: String,
        location: Option<SourceLocation>,
    },
    /// Gate used before qalloc.
    GateBeforeAlloc {
        allocator: String,
        location: Option<SourceLocation>,
    },
    /// Invalid gate arity.
    InvalidGateArity {
        gate: GateKind,
        expected: usize,
        actual: usize,
        location: Option<SourceLocation>,
    },
    /// Unknown operator.
    UnknownOperator {
        op: String,
        location: Option<SourceLocation>,
    },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::MissingField { kind, field, .. } => write!(
                f,
                "missing required field '{}' for {:?} statement",
                field, kind
            ),
            ValidationError::MissingExprField { kind, field } => write!(
                f,
                "missing required field '{}' for {:?} expression",
                field, kind
            ),
            ValidationError::UndefinedVariable { name, .. } => {
                write!(f, "undefined variable: {}", name)
            }
            ValidationError::UndefinedAllocator { name, .. } => {
                write!(f, "undefined qubit allocator: {}", name)
            }
            ValidationError::GateBeforeAlloc { allocator, .. } => write!(
                f,
                "gate uses allocator '{}' before it was allocated",
                allocator
            ),
            ValidationError::InvalidGateArity {
                gate,
                expected,
                actual,
                ..
            } => write!(
                f,
                "gate {:?} expects {} targets, got {}",
                gate, expected, actual
            ),
            ValidationError::UnknownOperator { op, .. } => write!(f, "unknown operator: {}", op),
        }
    }
}

impl std::error::Error for ValidationError {}

/// IR validation result.
#[derive(Debug, Clone, Default)]
pub struct ValidationResult {
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<String>,
}

impl ValidationResult {
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    pub fn add_error(&mut self, error: ValidationError) {
        self.errors.push(error);
    }

    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }
}

/// IR validator with semantic tracking.
#[derive(Debug, Default)]
pub struct IrValidator {
    /// Known variables in scope.
    variables: std::collections::BTreeSet<String>,
    /// Known qubit allocators.
    allocators: std::collections::BTreeSet<String>,
    /// Validation results.
    result: ValidationResult,
}

impl IrValidator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate the entire IR.
    pub fn validate(&mut self, ir: &GuppyIR) -> ValidationResult {
        for func in &ir.functions {
            self.validate_function(func);
        }
        std::mem::take(&mut self.result)
    }

    /// Validate a function.
    fn validate_function(&mut self, func: &Function) {
        // Clear per-function state
        self.variables.clear();
        self.allocators.clear();

        // Add parameters to variables
        for param in &func.params {
            self.variables.insert(param.name.clone());
        }

        // Validate body
        for stmt in &func.body {
            self.validate_stmt(stmt);
        }
    }

    /// Validate a statement.
    fn validate_stmt(&mut self, stmt: &Stmt) {
        match stmt.kind {
            StmtKind::Qalloc => {
                // Schema: requires name
                if let Some(name) = &stmt.name {
                    self.allocators.insert(name.clone());
                    self.variables.insert(name.clone());
                } else {
                    self.result.add_error(ValidationError::MissingField {
                        kind: stmt.kind.clone(),
                        field: "name",
                        location: stmt.location.clone(),
                    });
                }
            }

            StmtKind::Gate => {
                // Schema: requires gate
                if let Some(gate) = &stmt.gate {
                    // Check arity
                    if stmt.targets.len() != gate.arity() {
                        self.result.add_error(ValidationError::InvalidGateArity {
                            gate: *gate,
                            expected: gate.arity(),
                            actual: stmt.targets.len(),
                            location: stmt.location.clone(),
                        });
                    }
                } else {
                    self.result.add_error(ValidationError::MissingField {
                        kind: stmt.kind.clone(),
                        field: "gate",
                        location: stmt.location.clone(),
                    });
                }
                // Validate targets reference known allocators
                for target in &stmt.targets {
                    self.validate_qubit_ref(target, &stmt.location);
                }
                // Validate params are valid expressions
                for param in &stmt.params {
                    self.validate_expr(param);
                }
            }

            StmtKind::Measure => {
                // Validate targets
                for target in &stmt.targets {
                    self.validate_qubit_ref(target, &stmt.location);
                }
                // Add results to variables
                for result in &stmt.results {
                    self.variables.insert(result.clone());
                }
            }

            StmtKind::For => {
                // Schema: requires var and range
                if stmt.var.is_none() {
                    self.result.add_error(ValidationError::MissingField {
                        kind: stmt.kind.clone(),
                        field: "var",
                        location: stmt.location.clone(),
                    });
                }
                if stmt.range.is_none() {
                    self.result.add_error(ValidationError::MissingField {
                        kind: stmt.kind.clone(),
                        field: "range",
                        location: stmt.location.clone(),
                    });
                }
                // Add loop var to scope and validate body
                if let Some(var) = &stmt.var {
                    self.variables.insert(var.clone());
                }
                for s in &stmt.body {
                    self.validate_stmt(s);
                }
            }

            StmtKind::While => {
                // Schema: requires condition
                if let Some(condition) = &stmt.condition {
                    self.validate_expr(condition);
                } else {
                    self.result.add_error(ValidationError::MissingField {
                        kind: stmt.kind.clone(),
                        field: "condition",
                        location: stmt.location.clone(),
                    });
                }
                for s in &stmt.body {
                    self.validate_stmt(s);
                }
            }

            StmtKind::If => {
                // Schema: requires condition
                if let Some(condition) = &stmt.condition {
                    self.validate_expr(condition);
                } else {
                    self.result.add_error(ValidationError::MissingField {
                        kind: stmt.kind.clone(),
                        field: "condition",
                        location: stmt.location.clone(),
                    });
                }
                for s in &stmt.then_body {
                    self.validate_stmt(s);
                }
                for s in &stmt.else_body {
                    self.validate_stmt(s);
                }
            }

            StmtKind::Assign => {
                // Validate target and value
                if let Some(target) = &stmt.target {
                    // For simple idents, don't require prior definition (could be new var)
                    self.validate_expr_allow_new_var(target);
                }
                if let Some(value) = &stmt.value {
                    self.validate_expr(value);
                }
                // If target is a simple ident, add to variables
                if let Some(target) = &stmt.target
                    && target.kind == ExprKind::Ident
                    && let Some(name) = &target.name
                {
                    self.variables.insert(name.clone());
                }
            }

            StmtKind::Binding => {
                // Schema: requires name
                if stmt.name.is_none() {
                    self.result.add_error(ValidationError::MissingField {
                        kind: stmt.kind.clone(),
                        field: "name",
                        location: stmt.location.clone(),
                    });
                } else {
                    self.variables.insert(stmt.name.clone().unwrap());
                }
                if let Some(value) = &stmt.value {
                    self.validate_expr(value);
                }
            }

            StmtKind::Return => {
                if let Some(value) = &stmt.return_value {
                    self.validate_expr(value);
                }
            }

            StmtKind::Expr => {
                if let Some(expr) = &stmt.expr {
                    self.validate_expr(expr);
                }
            }

            StmtKind::Result => {
                if let Some(value) = &stmt.value {
                    self.validate_expr(value);
                }
            }

            StmtKind::Break | StmtKind::Continue | StmtKind::Barrier => {
                // No additional validation needed
            }
        }
    }

    /// Validate a qubit reference (index into allocator).
    fn validate_qubit_ref(&mut self, expr: &Expr, stmt_location: &Option<SourceLocation>) {
        match &expr.kind {
            ExprKind::Index => {
                if let Some(array) = &expr.array
                    && !self.allocators.contains(array)
                {
                    self.result.add_error(ValidationError::UndefinedAllocator {
                        name: array.clone(),
                        location: stmt_location.clone(),
                    });
                }
            }
            ExprKind::Ident => {
                // Single qubit reference
                if let Some(name) = &expr.name
                    && !self.variables.contains(name)
                {
                    self.result.add_error(ValidationError::UndefinedVariable {
                        name: name.clone(),
                        location: stmt_location.clone(),
                    });
                }
            }
            _ => {}
        }
    }

    /// Validate an expression (recursively).
    fn validate_expr(&mut self, expr: &Expr) {
        match &expr.kind {
            ExprKind::Ident => {
                if let Some(name) = &expr.name
                    && !self.variables.contains(name)
                {
                    self.result.add_error(ValidationError::UndefinedVariable {
                        name: name.clone(),
                        location: expr.location.clone(),
                    });
                }
            }
            ExprKind::Index => {
                if let Some(array) = &expr.array
                    && !self.variables.contains(array)
                    && !self.allocators.contains(array)
                {
                    self.result.add_error(ValidationError::UndefinedVariable {
                        name: array.clone(),
                        location: expr.location.clone(),
                    });
                }
                if let Some(index) = &expr.index {
                    self.validate_expr(index);
                }
            }
            ExprKind::Binary => {
                // Validate operator
                if let Some(op) = &expr.op {
                    let valid_ops = [
                        "add", "sub", "mul", "div", "floordiv", "mod", "eq", "ne", "lt", "le",
                        "gt", "ge", "and", "or", "bitand", "bitor", "bitxor", "shl", "shr",
                    ];
                    if !valid_ops.contains(&op.as_str()) {
                        self.result.add_error(ValidationError::UnknownOperator {
                            op: op.clone(),
                            location: expr.location.clone(),
                        });
                    }
                }
                if let Some(left) = &expr.left {
                    self.validate_expr(left);
                }
                if let Some(right) = &expr.right {
                    self.validate_expr(right);
                }
            }
            ExprKind::Unary => {
                if let Some(op) = &expr.op {
                    let valid_ops = ["neg", "not", "bitnot"];
                    if !valid_ops.contains(&op.as_str()) {
                        self.result.add_error(ValidationError::UnknownOperator {
                            op: op.clone(),
                            location: expr.location.clone(),
                        });
                    }
                }
                if let Some(operand) = &expr.operand {
                    self.validate_expr(operand);
                }
            }
            ExprKind::Call => {
                for arg in &expr.args {
                    self.validate_expr(arg);
                }
            }
            ExprKind::Field => {
                if let Some(object) = &expr.object {
                    self.validate_expr(object);
                }
            }
            ExprKind::Tuple => {
                for arg in &expr.args {
                    self.validate_expr(arg);
                }
            }
            ExprKind::Literal => {
                // No validation needed for literals
            }
        }
    }

    /// Validate expression but allow undefined identifiers (for assignment targets).
    fn validate_expr_allow_new_var(&mut self, expr: &Expr) {
        match &expr.kind {
            ExprKind::Ident => {
                // Allow undefined - this might be a new variable
            }
            ExprKind::Index => {
                // Array must exist, but we check elsewhere
                if let Some(index) = &expr.index {
                    self.validate_expr(index);
                }
            }
            _ => self.validate_expr(expr),
        }
    }
}

/// Validate IR and return result.
pub fn validate_ir(ir: &GuppyIR) -> ValidationResult {
    let mut validator = IrValidator::new();
    validator.validate(ir)
}

// =============================================================================
// IR Emission
// =============================================================================

/// Error during IR emission.
#[derive(Debug, Clone)]
pub enum EmitError {
    /// Parse error from rustpython.
    Parse(String),
    /// The input was not a module.
    NotAModule,
    /// Unsupported Python construct.
    Unsupported(String),
    /// Skip this node (internal).
    Skip,
}

impl std::fmt::Display for EmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmitError::Parse(msg) => write!(f, "parse error: {}", msg),
            EmitError::NotAModule => write!(f, "expected a module"),
            EmitError::Unsupported(msg) => write!(f, "unsupported: {}", msg),
            EmitError::Skip => write!(f, "skip"),
        }
    }
}

impl std::error::Error for EmitError {}

/// Emit IR from Python source.
pub fn emit_ir(source: &str, filename: Option<&str>) -> Result<GuppyIR, EmitError> {
    let parsed = rustpython_parser::parse(
        source,
        rustpython_parser::Mode::Module,
        filename.unwrap_or("<stdin>"),
    )
    .map_err(|e| EmitError::Parse(e.to_string()))?;

    let mut ir = GuppyIR::new();
    ir.source_file = filename.map(String::from);

    let Mod::Module(module) = parsed else {
        return Err(EmitError::NotAModule);
    };

    for stmt in module.body {
        if let PyStmt::FunctionDef(func) = stmt {
            ir.functions.push(convert_function(&func, source)?);
        }
    }

    Ok(ir)
}

fn convert_function(func: &ast::StmtFunctionDef, source: &str) -> Result<Function, EmitError> {
    let params = func
        .args
        .args
        .iter()
        .map(|arg| {
            let ty = arg
                .def
                .annotation
                .as_ref()
                .map(|ann| convert_type_annotation(ann))
                .unwrap_or_else(|| TypeExpr {
                    kind: "primitive".to_string(),
                    name: Some("unknown".to_string()),
                    element: None,
                    size: None,
                    elements: Vec::new(),
                });
            Param {
                name: arg.def.arg.to_string(),
                ty,
            }
        })
        .collect();

    let return_type = func
        .returns
        .as_ref()
        .map(|ann| convert_type_annotation(ann));

    let body = func
        .body
        .iter()
        .filter_map(|stmt| convert_stmt(stmt).ok())
        .collect();

    let (line, column) = offset_to_line_col(source, func.range.start().into());
    let (end_line, end_column) = offset_to_line_col(source, func.range.end().into());

    Ok(Function {
        name: func.name.to_string(),
        params,
        return_type,
        body,
        is_pub: None,
        location: Some(SourceLocation {
            line: line as u32,
            column: column as u32,
            end_line: Some(end_line as u32),
            end_column: Some(end_column as u32),
            file: None,
        }),
    })
}

fn offset_to_line_col(source: &str, offset: usize) -> (usize, usize) {
    let mut line = 1;
    let mut col = 1;
    for (i, ch) in source.char_indices() {
        if i >= offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    (line, col)
}

fn convert_type_annotation(ann: &PyExpr) -> TypeExpr {
    match ann {
        PyExpr::Name(name) => {
            let name_str = name.id.as_str();
            match name_str {
                "int" | "float" | "bool" | "str" | "None" => TypeExpr {
                    kind: "primitive".to_string(),
                    name: Some(name_str.to_string()),
                    element: None,
                    size: None,
                    elements: Vec::new(),
                },
                "qubit" => TypeExpr {
                    kind: "qalloc".to_string(),
                    name: None,
                    element: None,
                    size: None,
                    elements: Vec::new(),
                },
                _ => TypeExpr {
                    kind: "named".to_string(),
                    name: Some(name_str.to_string()),
                    element: None,
                    size: None,
                    elements: Vec::new(),
                },
            }
        }
        PyExpr::Subscript(sub) => {
            if let PyExpr::Name(name) = sub.value.as_ref() {
                let name_str = name.id.as_str();
                match name_str {
                    "list" | "List" => TypeExpr {
                        kind: "array".to_string(),
                        name: None,
                        element: Some(Box::new(convert_type_annotation(&sub.slice))),
                        size: None,
                        elements: Vec::new(),
                    },
                    "Optional" => TypeExpr {
                        kind: "optional".to_string(),
                        name: None,
                        element: Some(Box::new(convert_type_annotation(&sub.slice))),
                        size: None,
                        elements: Vec::new(),
                    },
                    "qubit" => TypeExpr {
                        kind: "qalloc".to_string(),
                        name: None,
                        element: None,
                        size: convert_expr(&sub.slice).ok().map(Box::new),
                        elements: Vec::new(),
                    },
                    "tuple" => {
                        // tuple[T1, T2, ...] - the slice is a Tuple of types
                        let elements = if let PyExpr::Tuple(tuple) = sub.slice.as_ref() {
                            tuple.elts.iter().map(convert_type_annotation).collect()
                        } else {
                            // Single element tuple
                            vec![convert_type_annotation(&sub.slice)]
                        };
                        TypeExpr {
                            kind: "tuple".to_string(),
                            name: None,
                            element: None,
                            size: None,
                            elements,
                        }
                    }
                    _ => TypeExpr {
                        kind: "named".to_string(),
                        name: Some(name_str.to_string()),
                        element: None,
                        size: None,
                        elements: Vec::new(),
                    },
                }
            } else {
                TypeExpr {
                    kind: "primitive".to_string(),
                    name: Some("unknown".to_string()),
                    element: None,
                    size: None,
                    elements: Vec::new(),
                }
            }
        }
        PyExpr::Constant(c) if matches!(c.value, Constant::None) => TypeExpr {
            kind: "primitive".to_string(),
            name: Some("None".to_string()),
            element: None,
            size: None,
            elements: Vec::new(),
        },
        _ => TypeExpr {
            kind: "primitive".to_string(),
            name: Some("unknown".to_string()),
            element: None,
            size: None,
            elements: Vec::new(),
        },
    }
}

fn convert_stmt(stmt: &PyStmt) -> Result<Stmt, EmitError> {
    match stmt {
        PyStmt::Assign(assign) => {
            if assign.targets.len() != 1 {
                return Err(EmitError::Unsupported("multiple assignment targets".into()));
            }

            // Check for qalloc: q = qubit[4]
            if let PyExpr::Subscript(sub) = assign.value.as_ref()
                && let PyExpr::Name(name) = sub.value.as_ref()
                && (name.id.as_str() == "qubit" || name.id.as_str() == "qalloc")
                && let PyExpr::Name(target) = &assign.targets[0]
            {
                return Ok(Stmt {
                    kind: StmtKind::Qalloc,
                    name: Some(target.id.to_string()),
                    size: convert_expr(&sub.slice).ok(),
                    gate: None,
                    targets: Vec::new(),
                    params: Vec::new(),
                    results: Vec::new(),
                    var: None,
                    range: None,
                    condition: None,
                    then_body: Vec::new(),
                    else_body: Vec::new(),
                    target: None,
                    value: None,
                    return_value: None,
                    expr: None,
                    ty: None,
                    is_mutable: None,
                    body: Vec::new(),
                    tag: None,
                    location: None,
                });
            }

            // Check for single qubit alloc: q = qubit()
            if let PyExpr::Call(call) = assign.value.as_ref()
                && let Some(callee_name) = get_callee_name(&call.func)
                && (callee_name == "qubit" || callee_name == "qalloc")
                && let PyExpr::Name(target) = &assign.targets[0]
            {
                // Single qubit allocation - size is 1
                return Ok(Stmt {
                    kind: StmtKind::Qalloc,
                    name: Some(target.id.to_string()),
                    size: Some(Expr {
                        kind: ExprKind::Literal,
                        value: Some(serde_json::json!(1)),
                        name: None,
                        array: None,
                        index: None,
                        op: None,
                        left: None,
                        right: None,
                        operand: None,
                        callee: None,
                        args: Vec::new(),
                        object: None,
                        field: None,
                        location: None,
                    }),
                    gate: None,
                    targets: Vec::new(),
                    params: Vec::new(),
                    results: Vec::new(),
                    var: None,
                    range: None,
                    condition: None,
                    then_body: Vec::new(),
                    else_body: Vec::new(),
                    target: None,
                    value: None,
                    return_value: None,
                    expr: None,
                    ty: None,
                    is_mutable: None,
                    body: Vec::new(),
                    tag: None,
                    location: None,
                });
            }

            // Check for measure: m = measure(q)
            if let PyExpr::Call(call) = assign.value.as_ref()
                && let Some(callee_name) = get_callee_name(&call.func)
                && (callee_name == "measure" || callee_name == "mz" || callee_name == "measure_all")
                && let PyExpr::Name(result_var) = &assign.targets[0]
            {
                let targets = call
                    .args
                    .iter()
                    .filter_map(|arg| convert_expr(arg).ok())
                    .collect();

                return Ok(Stmt {
                    kind: StmtKind::Measure,
                    name: None,
                    size: None,
                    gate: None,
                    targets,
                    params: Vec::new(),
                    results: vec![result_var.id.to_string()],
                    var: None,
                    range: None,
                    condition: None,
                    then_body: Vec::new(),
                    else_body: Vec::new(),
                    target: None,
                    value: None,
                    return_value: None,
                    expr: None,
                    ty: None,
                    is_mutable: None,
                    body: Vec::new(),
                    tag: None,
                    location: None,
                });
            }

            let target = convert_expr(&assign.targets[0])?;
            let value = convert_expr(&assign.value)?;

            Ok(Stmt {
                kind: StmtKind::Assign,
                name: None,
                size: None,
                gate: None,
                targets: Vec::new(),
                params: Vec::new(),
                results: Vec::new(),
                var: None,
                range: None,
                condition: None,
                then_body: Vec::new(),
                else_body: Vec::new(),
                target: Some(target),
                value: Some(value),
                return_value: None,
                expr: None,
                ty: None,
                is_mutable: None,
                body: Vec::new(),
                tag: None,
                location: None,
            })
        }

        PyStmt::Expr(expr_stmt) => {
            // Check for gate calls
            if let PyExpr::Call(call) = expr_stmt.value.as_ref() {
                let callee_name = get_callee_name(&call.func);
                if let Some(name) = &callee_name {
                    if let Some(gate) = GateKind::from_name(name) {
                        let mut targets: Vec<Expr> = call
                            .args
                            .iter()
                            .filter_map(|arg| convert_expr(arg).ok())
                            .collect();

                        // For parameterized gates like rx(q, angle), the last argument is the angle
                        let mut params = Vec::new();
                        if gate.is_parameterized() && !targets.is_empty() {
                            params.push(targets.pop().unwrap());
                        }

                        return Ok(Stmt {
                            kind: StmtKind::Gate,
                            name: None,
                            size: None,
                            gate: Some(gate),
                            targets,
                            params,
                            results: Vec::new(),
                            var: None,
                            range: None,
                            condition: None,
                            then_body: Vec::new(),
                            else_body: Vec::new(),
                            target: None,
                            value: None,
                            return_value: None,
                            expr: None,
                            ty: None,
                            is_mutable: None,
                            body: Vec::new(),
                            tag: None,
                            location: None,
                        });
                    }

                    // Check for measure
                    if name == "measure" || name == "mz" || name == "measure_all" {
                        let targets = call
                            .args
                            .iter()
                            .filter_map(|arg| convert_expr(arg).ok())
                            .collect();

                        return Ok(Stmt {
                            kind: StmtKind::Measure,
                            name: None,
                            size: None,
                            gate: None,
                            targets,
                            params: Vec::new(),
                            results: Vec::new(),
                            var: None,
                            range: None,
                            condition: None,
                            then_body: Vec::new(),
                            else_body: Vec::new(),
                            target: None,
                            value: None,
                            return_value: None,
                            expr: None,
                            ty: None,
                            is_mutable: None,
                            body: Vec::new(),
                            tag: None,
                            location: None,
                        });
                    }

                    // Check for result: result(tag, value)
                    if name == "result" && call.args.len() >= 2 {
                        let tag = if let PyExpr::Constant(c) = &call.args[0] {
                            if let Constant::Str(s) = &c.value {
                                Some(s.to_string())
                            } else {
                                None
                            }
                        } else {
                            None
                        };

                        let value = convert_expr(&call.args[1]).ok();

                        return Ok(Stmt {
                            kind: StmtKind::Result,
                            name: None,
                            size: None,
                            gate: None,
                            targets: Vec::new(),
                            params: Vec::new(),
                            results: Vec::new(),
                            var: None,
                            range: None,
                            condition: None,
                            then_body: Vec::new(),
                            else_body: Vec::new(),
                            target: None,
                            value,
                            return_value: None,
                            expr: None,
                            ty: None,
                            is_mutable: None,
                            body: Vec::new(),
                            tag,
                            location: None,
                        });
                    }
                }
            }

            // Skip docstrings (string constant expressions)
            if let PyExpr::Constant(c) = expr_stmt.value.as_ref()
                && matches!(c.value, Constant::Str(_))
            {
                return Err(EmitError::Skip);
            }

            let expr = convert_expr(&expr_stmt.value)?;
            Ok(Stmt {
                kind: StmtKind::Expr,
                name: None,
                size: None,
                gate: None,
                targets: Vec::new(),
                params: Vec::new(),
                results: Vec::new(),
                var: None,
                range: None,
                condition: None,
                then_body: Vec::new(),
                else_body: Vec::new(),
                target: None,
                value: None,
                return_value: None,
                expr: Some(expr),
                ty: None,
                is_mutable: None,
                body: Vec::new(),
                tag: None,
                location: None,
            })
        }

        PyStmt::For(for_stmt) => {
            let var = if let PyExpr::Name(name) = for_stmt.target.as_ref() {
                name.id.to_string()
            } else {
                return Err(EmitError::Unsupported("complex for target".into()));
            };

            let range = if let PyExpr::Call(call) = for_stmt.iter.as_ref() {
                if let PyExpr::Name(name) = call.func.as_ref() {
                    if name.id.as_str() == "range" {
                        let args: Vec<_> = call
                            .args
                            .iter()
                            .filter_map(|a| convert_expr(a).ok())
                            .collect();

                        if args.len() == 1 {
                            Some(RangeExpr {
                                start: Expr {
                                    kind: ExprKind::Literal,
                                    value: Some(serde_json::json!(0)),
                                    name: None,
                                    array: None,
                                    index: None,
                                    op: None,
                                    left: None,
                                    right: None,
                                    operand: None,
                                    callee: None,
                                    args: Vec::new(),
                                    object: None,
                                    field: None,
                                    location: None,
                                },
                                end: args.into_iter().next().unwrap(),
                            })
                        } else if args.len() >= 2 {
                            let mut iter = args.into_iter();
                            Some(RangeExpr {
                                start: iter.next().unwrap(),
                                end: iter.next().unwrap(),
                            })
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            let body = for_stmt
                .body
                .iter()
                .filter_map(|s| convert_stmt(s).ok())
                .collect();

            Ok(Stmt {
                kind: StmtKind::For,
                name: None,
                size: None,
                gate: None,
                targets: Vec::new(),
                params: Vec::new(),
                results: Vec::new(),
                var: Some(var),
                range,
                condition: None,
                then_body: Vec::new(),
                else_body: Vec::new(),
                target: None,
                value: None,
                return_value: None,
                expr: None,
                ty: None,
                is_mutable: None,
                body,
                tag: None,
                location: None,
            })
        }

        PyStmt::While(while_stmt) => {
            let condition = convert_expr(&while_stmt.test)?;
            let body = while_stmt
                .body
                .iter()
                .filter_map(|s| convert_stmt(s).ok())
                .collect();

            Ok(Stmt {
                kind: StmtKind::While,
                name: None,
                size: None,
                gate: None,
                targets: Vec::new(),
                params: Vec::new(),
                results: Vec::new(),
                var: None,
                range: None,
                condition: Some(condition),
                then_body: Vec::new(),
                else_body: Vec::new(),
                target: None,
                value: None,
                return_value: None,
                expr: None,
                ty: None,
                is_mutable: None,
                body,
                tag: None,
                location: None,
            })
        }

        PyStmt::Break(_) => Ok(Stmt {
            kind: StmtKind::Break,
            name: None,
            size: None,
            gate: None,
            targets: Vec::new(),
            params: Vec::new(),
            results: Vec::new(),
            var: None,
            range: None,
            condition: None,
            then_body: Vec::new(),
            else_body: Vec::new(),
            target: None,
            value: None,
            return_value: None,
            expr: None,
            ty: None,
            is_mutable: None,
            body: Vec::new(),
            tag: None,
            location: None,
        }),

        PyStmt::Continue(_) => Ok(Stmt {
            kind: StmtKind::Continue,
            name: None,
            size: None,
            gate: None,
            targets: Vec::new(),
            params: Vec::new(),
            results: Vec::new(),
            var: None,
            range: None,
            condition: None,
            then_body: Vec::new(),
            else_body: Vec::new(),
            target: None,
            value: None,
            return_value: None,
            expr: None,
            ty: None,
            is_mutable: None,
            body: Vec::new(),
            tag: None,
            location: None,
        }),

        PyStmt::If(if_stmt) => {
            let condition = convert_expr(&if_stmt.test)?;
            let then_body = if_stmt
                .body
                .iter()
                .filter_map(|s| convert_stmt(s).ok())
                .collect();
            let else_body = if_stmt
                .orelse
                .iter()
                .filter_map(|s| convert_stmt(s).ok())
                .collect();

            Ok(Stmt {
                kind: StmtKind::If,
                name: None,
                size: None,
                gate: None,
                targets: Vec::new(),
                params: Vec::new(),
                results: Vec::new(),
                var: None,
                range: None,
                condition: Some(condition),
                then_body,
                else_body,
                target: None,
                value: None,
                return_value: None,
                expr: None,
                ty: None,
                is_mutable: None,
                body: Vec::new(),
                tag: None,
                location: None,
            })
        }

        PyStmt::Return(ret) => {
            let return_value = ret.value.as_ref().and_then(|v| convert_expr(v).ok());

            Ok(Stmt {
                kind: StmtKind::Return,
                name: None,
                size: None,
                gate: None,
                targets: Vec::new(),
                params: Vec::new(),
                results: Vec::new(),
                var: None,
                range: None,
                condition: None,
                then_body: Vec::new(),
                else_body: Vec::new(),
                target: None,
                value: None,
                return_value,
                expr: None,
                ty: None,
                is_mutable: None,
                body: Vec::new(),
                tag: None,
                location: None,
            })
        }

        PyStmt::Pass(_) => Err(EmitError::Skip),

        PyStmt::AnnAssign(ann) => {
            // Handle annotated assignment: x: int = 5
            let name = if let PyExpr::Name(name) = ann.target.as_ref() {
                name.id.to_string()
            } else {
                return Err(EmitError::Unsupported("complex annotated target".into()));
            };

            // Check for qalloc: q: qubit[4] or q: qubit[4] = ...
            if let PyExpr::Subscript(sub) = ann.annotation.as_ref()
                && let PyExpr::Name(type_name) = sub.value.as_ref()
                && (type_name.id.as_str() == "qubit" || type_name.id.as_str() == "qalloc")
            {
                return Ok(Stmt {
                    kind: StmtKind::Qalloc,
                    name: Some(name),
                    size: convert_expr(&sub.slice).ok(),
                    gate: None,
                    targets: Vec::new(),
                    params: Vec::new(),
                    results: Vec::new(),
                    var: None,
                    range: None,
                    condition: None,
                    then_body: Vec::new(),
                    else_body: Vec::new(),
                    target: None,
                    value: None,
                    return_value: None,
                    expr: None,
                    ty: None,
                    is_mutable: None,
                    body: Vec::new(),
                    tag: None,
                    location: None,
                });
            }

            // Regular annotated assignment -> Binding
            let ty = convert_type_annotation(&ann.annotation);
            let value = ann.value.as_ref().and_then(|v| convert_expr(v).ok());

            Ok(Stmt {
                kind: StmtKind::Binding,
                name: Some(name),
                size: None,
                gate: None,
                targets: Vec::new(),
                params: Vec::new(),
                results: Vec::new(),
                var: None,
                range: None,
                condition: None,
                then_body: Vec::new(),
                else_body: Vec::new(),
                target: None,
                value,
                return_value: None,
                expr: None,
                ty: Some(ty),
                is_mutable: Some(true), // Assume mutable by default
                body: Vec::new(),
                tag: None,
                location: None,
            })
        }

        // Augmented assignment: i += 1, x -= 2, etc.
        PyStmt::AugAssign(aug) => {
            let target = convert_expr(&aug.target)?;
            let value = convert_expr(&aug.value)?;

            // Convert augmented assignment to regular assignment with binary op
            // i += 1 becomes i = i + 1
            let op = match aug.op {
                ast::Operator::Add => "add",
                ast::Operator::Sub => "sub",
                ast::Operator::Mult => "mul",
                ast::Operator::Div => "div",
                ast::Operator::Mod => "mod",
                ast::Operator::BitAnd => "bitand",
                ast::Operator::BitOr => "bitor",
                ast::Operator::BitXor => "bitxor",
                ast::Operator::LShift => "shl",
                ast::Operator::RShift => "shr",
                _ => {
                    return Err(EmitError::Unsupported(format!(
                        "augmented operator: {:?}",
                        aug.op
                    )));
                }
            };

            Ok(Stmt {
                kind: StmtKind::Assign,
                name: None,
                size: None,
                gate: None,
                targets: Vec::new(),
                params: Vec::new(),
                results: Vec::new(),
                var: None,
                range: None,
                condition: None,
                then_body: Vec::new(),
                else_body: Vec::new(),
                target: Some(target.clone()),
                value: Some(Expr {
                    kind: ExprKind::Binary,
                    value: None,
                    name: None,
                    array: None,
                    index: None,
                    op: Some(op.to_string()),
                    left: Some(Box::new(target)),
                    right: Some(Box::new(value)),
                    operand: None,
                    callee: None,
                    args: Vec::new(),
                    object: None,
                    field: None,
                    location: None,
                }),
                return_value: None,
                expr: None,
                ty: None,
                is_mutable: None,
                body: Vec::new(),
                tag: None,
                location: None,
            })
        }

        _ => Err(EmitError::Unsupported(format!(
            "statement type: {:?}",
            std::mem::discriminant(stmt)
        ))),
    }
}

fn convert_expr(expr: &PyExpr) -> Result<Expr, EmitError> {
    match expr {
        PyExpr::Constant(c) => {
            let value = match &c.value {
                Constant::Int(i) => serde_json::json!(i.to_string().parse::<i64>().unwrap_or(0)),
                Constant::Float(f) => serde_json::json!(f),
                Constant::Str(s) => serde_json::json!(s),
                Constant::Bool(b) => serde_json::json!(b),
                Constant::None => serde_json::Value::Null,
                _ => serde_json::json!(null),
            };
            Ok(Expr {
                kind: ExprKind::Literal,
                value: Some(value),
                name: None,
                array: None,
                index: None,
                op: None,
                left: None,
                right: None,
                operand: None,
                callee: None,
                args: Vec::new(),
                object: None,
                field: None,
                location: None,
            })
        }

        PyExpr::Name(name) => Ok(Expr {
            kind: ExprKind::Ident,
            value: None,
            name: Some(name.id.to_string()),
            array: None,
            index: None,
            op: None,
            left: None,
            right: None,
            operand: None,
            callee: None,
            args: Vec::new(),
            object: None,
            field: None,
            location: None,
        }),

        PyExpr::Subscript(sub) => {
            if let PyExpr::Name(name) = sub.value.as_ref() {
                let index = convert_expr(&sub.slice)?;
                Ok(Expr {
                    kind: ExprKind::Index,
                    value: None,
                    name: None,
                    array: Some(name.id.to_string()),
                    index: Some(Box::new(index)),
                    op: None,
                    left: None,
                    right: None,
                    operand: None,
                    callee: None,
                    args: Vec::new(),
                    object: None,
                    field: None,
                    location: None,
                })
            } else {
                Err(EmitError::Unsupported("complex subscript".into()))
            }
        }

        PyExpr::BinOp(binop) => {
            let left = convert_expr(&binop.left)?;
            let right = convert_expr(&binop.right)?;
            let op = match binop.op {
                ast::Operator::Add => "add",
                ast::Operator::Sub => "sub",
                ast::Operator::Mult => "mul",
                ast::Operator::Div => "div",
                ast::Operator::FloorDiv => "floordiv",
                ast::Operator::Mod => "mod",
                ast::Operator::BitAnd => "bitand",
                ast::Operator::BitOr => "bitor",
                ast::Operator::BitXor => "bitxor",
                ast::Operator::LShift => "shl",
                ast::Operator::RShift => "shr",
                other => {
                    return Err(EmitError::Unsupported(format!(
                        "binary operator: {:?}",
                        other
                    )));
                }
            };

            Ok(Expr {
                kind: ExprKind::Binary,
                value: None,
                name: None,
                array: None,
                index: None,
                op: Some(op.to_string()),
                left: Some(Box::new(left)),
                right: Some(Box::new(right)),
                operand: None,
                callee: None,
                args: Vec::new(),
                object: None,
                field: None,
                location: None,
            })
        }

        PyExpr::UnaryOp(unary) => {
            let operand = convert_expr(&unary.operand)?;
            let op = match unary.op {
                ast::UnaryOp::UAdd => "pos",
                ast::UnaryOp::USub => "neg",
                ast::UnaryOp::Not => "not",
                ast::UnaryOp::Invert => "bitnot",
            };

            Ok(Expr {
                kind: ExprKind::Unary,
                value: None,
                name: None,
                array: None,
                index: None,
                op: Some(op.to_string()),
                left: None,
                right: None,
                operand: Some(Box::new(operand)),
                callee: None,
                args: Vec::new(),
                object: None,
                field: None,
                location: None,
            })
        }

        PyExpr::Compare(cmp) => {
            let left = convert_expr(&cmp.left)?;
            if let Some((cmpop, right_expr)) = cmp.ops.first().zip(cmp.comparators.first()) {
                let right = convert_expr(right_expr)?;
                let op = match cmpop {
                    ast::CmpOp::Eq => "eq",
                    ast::CmpOp::NotEq => "ne",
                    ast::CmpOp::Lt => "lt",
                    ast::CmpOp::LtE => "le",
                    ast::CmpOp::Gt => "gt",
                    ast::CmpOp::GtE => "ge",
                    other => {
                        return Err(EmitError::Unsupported(format!(
                            "comparison operator: {:?}",
                            other
                        )));
                    }
                };

                Ok(Expr {
                    kind: ExprKind::Binary,
                    value: None,
                    name: None,
                    array: None,
                    index: None,
                    op: Some(op.to_string()),
                    left: Some(Box::new(left)),
                    right: Some(Box::new(right)),
                    operand: None,
                    callee: None,
                    args: Vec::new(),
                    object: None,
                    field: None,
                    location: None,
                })
            } else {
                Ok(left)
            }
        }

        PyExpr::BoolOp(boolop) => {
            // Handle `and` and `or` operators
            // BoolOp can have multiple values: a and b and c
            // We convert to nested binary expressions: (a and b) and c
            let op = match boolop.op {
                ast::BoolOp::And => "and",
                ast::BoolOp::Or => "or",
            };

            if boolop.values.len() < 2 {
                return Err(EmitError::Unsupported("boolop with < 2 values".into()));
            }

            let mut iter = boolop.values.iter();
            let first = convert_expr(iter.next().unwrap())?;
            let second = convert_expr(iter.next().unwrap())?;

            let mut result = Expr {
                kind: ExprKind::Binary,
                value: None,
                name: None,
                array: None,
                index: None,
                op: Some(op.to_string()),
                left: Some(Box::new(first)),
                right: Some(Box::new(second)),
                operand: None,
                callee: None,
                args: Vec::new(),
                object: None,
                field: None,
                location: None,
            };

            // Handle chained operators: a and b and c -> (a and b) and c
            for val in iter {
                let right = convert_expr(val)?;
                result = Expr {
                    kind: ExprKind::Binary,
                    value: None,
                    name: None,
                    array: None,
                    index: None,
                    op: Some(op.to_string()),
                    left: Some(Box::new(result)),
                    right: Some(Box::new(right)),
                    operand: None,
                    callee: None,
                    args: Vec::new(),
                    object: None,
                    field: None,
                    location: None,
                };
            }

            Ok(result)
        }

        PyExpr::Call(call) => {
            let callee = get_callee_name(&call.func);
            let args = call
                .args
                .iter()
                .filter_map(|a| convert_expr(a).ok())
                .collect();

            Ok(Expr {
                kind: ExprKind::Call,
                value: None,
                name: None,
                array: None,
                index: None,
                op: None,
                left: None,
                right: None,
                operand: None,
                callee,
                args,
                object: None,
                field: None,
                location: None,
            })
        }

        PyExpr::Attribute(attr) => {
            let object = convert_expr(&attr.value)?;
            Ok(Expr {
                kind: ExprKind::Field,
                value: None,
                name: None,
                array: None,
                index: None,
                op: None,
                left: None,
                right: None,
                operand: None,
                callee: None,
                args: Vec::new(),
                object: Some(Box::new(object)),
                field: Some(attr.attr.to_string()),
                location: None,
            })
        }

        PyExpr::Tuple(tuple) => {
            let elements: Vec<Expr> = tuple
                .elts
                .iter()
                .filter_map(|e| convert_expr(e).ok())
                .collect();

            Ok(Expr {
                kind: ExprKind::Tuple,
                value: None,
                name: None,
                array: None,
                index: None,
                op: None,
                left: None,
                right: None,
                operand: None,
                callee: None,
                args: elements,
                object: None,
                field: None,
                location: None,
            })
        }

        _ => Err(EmitError::Unsupported(format!(
            "expression type: {:?}",
            std::mem::discriminant(expr)
        ))),
    }
}

fn get_callee_name(expr: &PyExpr) -> Option<String> {
    match expr {
        PyExpr::Name(name) => Some(name.id.to_string()),
        PyExpr::Attribute(attr) => Some(attr.attr.to_string()),
        _ => None,
    }
}

#[cfg(test)]
mod validation_tests {
    use super::*;

    fn make_ir(body: Vec<Stmt>) -> GuppyIR {
        GuppyIR {
            version: IR_VERSION.to_string(),
            functions: vec![Function {
                name: "test".to_string(),
                params: vec![],
                return_type: None,
                body,
                is_pub: None,
                location: None,
            }],
            source_file: None,
        }
    }

    #[test]
    fn test_valid_qalloc_gate() {
        let ir = make_ir(vec![
            Stmt {
                kind: StmtKind::Qalloc,
                name: Some("q".to_string()),
                size: Some(Expr {
                    kind: ExprKind::Literal,
                    value: Some(serde_json::json!(4)),
                    name: None,
                    array: None,
                    index: None,
                    op: None,
                    left: None,
                    right: None,
                    operand: None,
                    callee: None,
                    args: vec![],
                    object: None,
                    field: None,
                    location: None,
                }),
                gate: None,
                targets: vec![],
                params: vec![],
                results: vec![],
                var: None,
                range: None,
                condition: None,
                then_body: vec![],
                else_body: vec![],
                target: None,
                value: None,
                return_value: None,
                expr: None,
                ty: None,
                is_mutable: None,
                body: vec![],
                tag: None,
                location: None,
            },
            Stmt {
                kind: StmtKind::Gate,
                name: None,
                size: None,
                gate: Some(GateKind::H),
                targets: vec![Expr {
                    kind: ExprKind::Index,
                    value: None,
                    name: None,
                    array: Some("q".to_string()),
                    index: Some(Box::new(Expr {
                        kind: ExprKind::Literal,
                        value: Some(serde_json::json!(0)),
                        name: None,
                        array: None,
                        index: None,
                        op: None,
                        left: None,
                        right: None,
                        operand: None,
                        callee: None,
                        args: vec![],
                        object: None,
                        field: None,
                        location: None,
                    })),
                    op: None,
                    left: None,
                    right: None,
                    operand: None,
                    callee: None,
                    args: vec![],
                    object: None,
                    field: None,
                    location: None,
                }],
                params: vec![],
                results: vec![],
                var: None,
                range: None,
                condition: None,
                then_body: vec![],
                else_body: vec![],
                target: None,
                value: None,
                return_value: None,
                expr: None,
                ty: None,
                is_mutable: None,
                body: vec![],
                tag: None,
                location: None,
            },
        ]);

        let result = validate_ir(&ir);
        assert!(
            result.is_valid(),
            "Expected valid IR, got errors: {:?}",
            result.errors
        );
    }

    #[test]
    fn test_undefined_allocator() {
        let ir = make_ir(vec![
            // Use q[0] without allocating q first
            Stmt {
                kind: StmtKind::Gate,
                name: None,
                size: None,
                gate: Some(GateKind::H),
                targets: vec![Expr {
                    kind: ExprKind::Index,
                    value: None,
                    name: None,
                    array: Some("q".to_string()),
                    index: Some(Box::new(Expr {
                        kind: ExprKind::Literal,
                        value: Some(serde_json::json!(0)),
                        name: None,
                        array: None,
                        index: None,
                        op: None,
                        left: None,
                        right: None,
                        operand: None,
                        callee: None,
                        args: vec![],
                        object: None,
                        field: None,
                        location: None,
                    })),
                    op: None,
                    left: None,
                    right: None,
                    operand: None,
                    callee: None,
                    args: vec![],
                    object: None,
                    field: None,
                    location: None,
                }],
                params: vec![],
                results: vec![],
                var: None,
                range: None,
                condition: None,
                then_body: vec![],
                else_body: vec![],
                target: None,
                value: None,
                return_value: None,
                expr: None,
                ty: None,
                is_mutable: None,
                body: vec![],
                tag: None,
                location: None,
            },
        ]);

        let result = validate_ir(&ir);
        assert!(
            !result.is_valid(),
            "Expected validation error for undefined allocator"
        );
        assert!(
            result
                .errors
                .iter()
                .any(|e| matches!(e, ValidationError::UndefinedAllocator { .. }))
        );
    }

    #[test]
    fn test_missing_gate_field() {
        let ir = make_ir(vec![Stmt {
            kind: StmtKind::Gate,
            name: None,
            size: None,
            gate: None, // Missing required gate field
            targets: vec![],
            params: vec![],
            results: vec![],
            var: None,
            range: None,
            condition: None,
            then_body: vec![],
            else_body: vec![],
            target: None,
            value: None,
            return_value: None,
            expr: None,
            ty: None,
            is_mutable: None,
            body: vec![],
            tag: None,
            location: None,
        }]);

        let result = validate_ir(&ir);
        assert!(
            !result.is_valid(),
            "Expected validation error for missing gate field"
        );
        assert!(
            result
                .errors
                .iter()
                .any(|e| matches!(e, ValidationError::MissingField { field: "gate", .. }))
        );
    }

    #[test]
    fn test_invalid_gate_arity() {
        let ir = make_ir(vec![
            Stmt {
                kind: StmtKind::Qalloc,
                name: Some("q".to_string()),
                size: Some(Expr {
                    kind: ExprKind::Literal,
                    value: Some(serde_json::json!(2)),
                    name: None,
                    array: None,
                    index: None,
                    op: None,
                    left: None,
                    right: None,
                    operand: None,
                    callee: None,
                    args: vec![],
                    object: None,
                    field: None,
                    location: None,
                }),
                gate: None,
                targets: vec![],
                params: vec![],
                results: vec![],
                var: None,
                range: None,
                condition: None,
                then_body: vec![],
                else_body: vec![],
                target: None,
                value: None,
                return_value: None,
                expr: None,
                ty: None,
                is_mutable: None,
                body: vec![],
                tag: None,
                location: None,
            },
            Stmt {
                kind: StmtKind::Gate,
                name: None,
                size: None,
                gate: Some(GateKind::H), // H is 1-qubit gate
                targets: vec![
                    // But we provide 2 targets
                    Expr {
                        kind: ExprKind::Index,
                        value: None,
                        name: None,
                        array: Some("q".to_string()),
                        index: Some(Box::new(Expr {
                            kind: ExprKind::Literal,
                            value: Some(serde_json::json!(0)),
                            name: None,
                            array: None,
                            index: None,
                            op: None,
                            left: None,
                            right: None,
                            operand: None,
                            callee: None,
                            args: vec![],
                            object: None,
                            field: None,
                            location: None,
                        })),
                        op: None,
                        left: None,
                        right: None,
                        operand: None,
                        callee: None,
                        args: vec![],
                        object: None,
                        field: None,
                        location: None,
                    },
                    Expr {
                        kind: ExprKind::Index,
                        value: None,
                        name: None,
                        array: Some("q".to_string()),
                        index: Some(Box::new(Expr {
                            kind: ExprKind::Literal,
                            value: Some(serde_json::json!(1)),
                            name: None,
                            array: None,
                            index: None,
                            op: None,
                            left: None,
                            right: None,
                            operand: None,
                            callee: None,
                            args: vec![],
                            object: None,
                            field: None,
                            location: None,
                        })),
                        op: None,
                        left: None,
                        right: None,
                        operand: None,
                        callee: None,
                        args: vec![],
                        object: None,
                        field: None,
                        location: None,
                    },
                ],
                params: vec![],
                results: vec![],
                var: None,
                range: None,
                condition: None,
                then_body: vec![],
                else_body: vec![],
                target: None,
                value: None,
                return_value: None,
                expr: None,
                ty: None,
                is_mutable: None,
                body: vec![],
                tag: None,
                location: None,
            },
        ]);

        let result = validate_ir(&ir);
        assert!(
            !result.is_valid(),
            "Expected validation error for invalid gate arity"
        );
        assert!(
            result
                .errors
                .iter()
                .any(|e| matches!(e, ValidationError::InvalidGateArity { .. }))
        );
    }

    #[test]
    fn test_unknown_operator() {
        let ir = make_ir(vec![Stmt {
            kind: StmtKind::Return,
            name: None,
            size: None,
            gate: None,
            targets: vec![],
            params: vec![],
            results: vec![],
            var: None,
            range: None,
            condition: None,
            then_body: vec![],
            else_body: vec![],
            target: None,
            value: None,
            return_value: Some(Expr {
                kind: ExprKind::Binary,
                value: None,
                name: None,
                array: None,
                index: None,
                op: Some("invalid_op".to_string()), // Unknown operator
                left: Some(Box::new(Expr {
                    kind: ExprKind::Literal,
                    value: Some(serde_json::json!(1)),
                    name: None,
                    array: None,
                    index: None,
                    op: None,
                    left: None,
                    right: None,
                    operand: None,
                    callee: None,
                    args: vec![],
                    object: None,
                    field: None,
                    location: None,
                })),
                right: Some(Box::new(Expr {
                    kind: ExprKind::Literal,
                    value: Some(serde_json::json!(2)),
                    name: None,
                    array: None,
                    index: None,
                    op: None,
                    left: None,
                    right: None,
                    operand: None,
                    callee: None,
                    args: vec![],
                    object: None,
                    field: None,
                    location: None,
                })),
                operand: None,
                callee: None,
                args: vec![],
                object: None,
                field: None,
                location: None,
            }),
            expr: None,
            ty: None,
            is_mutable: None,
            body: vec![],
            tag: None,
            location: None,
        }]);

        let result = validate_ir(&ir);
        assert!(
            !result.is_valid(),
            "Expected validation error for unknown operator"
        );
        assert!(
            result
                .errors
                .iter()
                .any(|e| matches!(e, ValidationError::UnknownOperator { .. }))
        );
    }

    #[test]
    fn test_undefined_variable() {
        let ir = make_ir(vec![Stmt {
            kind: StmtKind::Return,
            name: None,
            size: None,
            gate: None,
            targets: vec![],
            params: vec![],
            results: vec![],
            var: None,
            range: None,
            condition: None,
            then_body: vec![],
            else_body: vec![],
            target: None,
            value: None,
            return_value: Some(Expr {
                kind: ExprKind::Ident,
                value: None,
                name: Some("undefined_var".to_string()),
                array: None,
                index: None,
                op: None,
                left: None,
                right: None,
                operand: None,
                callee: None,
                args: vec![],
                object: None,
                field: None,
                location: None,
            }),
            expr: None,
            ty: None,
            is_mutable: None,
            body: vec![],
            tag: None,
            location: None,
        }]);

        let result = validate_ir(&ir);
        assert!(
            !result.is_valid(),
            "Expected validation error for undefined variable"
        );
        assert!(
            result
                .errors
                .iter()
                .any(|e| matches!(e, ValidationError::UndefinedVariable { .. }))
        );
    }

    #[test]
    fn test_variable_from_binding() {
        let ir = make_ir(vec![
            Stmt {
                kind: StmtKind::Binding,
                name: Some("x".to_string()),
                size: None,
                gate: None,
                targets: vec![],
                params: vec![],
                results: vec![],
                var: None,
                range: None,
                condition: None,
                then_body: vec![],
                else_body: vec![],
                target: None,
                value: Some(Expr {
                    kind: ExprKind::Literal,
                    value: Some(serde_json::json!(42)),
                    name: None,
                    array: None,
                    index: None,
                    op: None,
                    left: None,
                    right: None,
                    operand: None,
                    callee: None,
                    args: vec![],
                    object: None,
                    field: None,
                    location: None,
                }),
                return_value: None,
                expr: None,
                ty: None,
                is_mutable: None,
                body: vec![],
                tag: None,
                location: None,
            },
            Stmt {
                kind: StmtKind::Return,
                name: None,
                size: None,
                gate: None,
                targets: vec![],
                params: vec![],
                results: vec![],
                var: None,
                range: None,
                condition: None,
                then_body: vec![],
                else_body: vec![],
                target: None,
                value: None,
                return_value: Some(Expr {
                    kind: ExprKind::Ident,
                    value: None,
                    name: Some("x".to_string()),
                    array: None,
                    index: None,
                    op: None,
                    left: None,
                    right: None,
                    operand: None,
                    callee: None,
                    args: vec![],
                    object: None,
                    field: None,
                    location: None,
                }),
                expr: None,
                ty: None,
                is_mutable: None,
                body: vec![],
                tag: None,
                location: None,
            },
        ]);

        let result = validate_ir(&ir);
        assert!(
            result.is_valid(),
            "Expected valid IR with defined variable, got errors: {:?}",
            result.errors
        );
    }
}
