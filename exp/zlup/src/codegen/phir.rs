//! PHIR/JSON code generation for Zlup.
//!
//! This module generates **PHIR/JSON** (the JSON serialization of PHIR) from Zlup AST.
//!
//! ## PHIR vs PHIR/JSON
//!
//! - **PHIR** (PECOS High-level Intermediate Representation): The abstract IR for
//!   representing hybrid quantum-classical programs. Defined in the `pecos-phir` crate.
//! - **PHIR/JSON**: The JSON serialization format for PHIR programs, as specified in
//!   the `pecos-phir-json` crate (v0.1.0). This is what this module generates.
//!
//! ## Design Philosophy
//!
//! PHIR/JSON provides:
//! - Explicit variable definitions (quantum and classical)
//! - Quantum operations with qubit references
//! - Classical operations with AST-style expressions
//! - Control flow via if/else blocks
//! - Parallel execution via qparallel blocks
//!
//! ## Output Format
//!
//! The output conforms to PHIR/JSON specification v0.1.0:
//!
//! ```json
//! {
//!   "format": "PHIR/JSON",
//!   "version": "0.1.0",
//!   "metadata": {"program_name": "main"},
//!   "ops": [
//!     {"data": "qvar_define", "variable": "q", "size": 2},
//!     {"qop": "H", "args": [["q", 0]]},
//!     {"qop": "CX", "args": [[["q", 0], ["q", 1]]]}
//!   ]
//! }
//! ```

use std::collections::BTreeMap;
use thiserror::Error;

use crate::ast::{
    BinaryOp, Binding, Block, CallExpr, ElseBranch, Expr, FnDecl, ForRange, GateKind, GateOp,
    IfStmt, IntLit, MeasureOp, PrepareOp, Program, Stmt, TickStmt, TopLevelDecl, UnaryOp,
};

// =============================================================================
// Errors
// =============================================================================

/// PHIR/JSON code generation errors.
#[derive(Debug, Error)]
pub enum PhirJsonError {
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

    #[error("unsupported expression in PHIR codegen")]
    UnsupportedExpression,

    #[error("invalid rotation angle")]
    InvalidAngle,

    #[error("JSON serialization error: {0}")]
    JsonError(String),

    #[error("unsupported statement in PHIR codegen: {0}")]
    UnsupportedStatement(String),
}

/// Result type for PHIR/JSON code generation.
pub type PhirJsonResult<T> = Result<T, PhirJsonError>;

// =============================================================================
// PHIR/JSON Node Types
// =============================================================================

/// Top-level PHIR/JSON program structure.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PhirJsonProgram {
    pub format: &'static str,
    pub version: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<PhirJsonMetadata>,
    pub ops: Vec<PhirJsonOp>,
}

impl Default for PhirJsonProgram {
    fn default() -> Self {
        Self::new()
    }
}

impl PhirJsonProgram {
    pub fn new() -> Self {
        Self {
            format: "PHIR/JSON",
            version: "0.1.0",
            metadata: None,
            ops: Vec::new(),
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.metadata = Some(PhirJsonMetadata {
            program_name: Some(name.into()),
            description: None,
            strict_parallelism: None,
        });
        self
    }
}

/// PHIR/JSON metadata.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PhirJsonMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub program_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict_parallelism: Option<String>,
}

/// PHIR/JSON operation - can be data, qop, cop, mop, or block.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(untagged)]
pub enum PhirJsonOp {
    Comment(PhirJsonComment),
    QvarDefine(PhirJsonQvarDefine),
    CvarDefine(PhirJsonCvarDefine),
    CvarExport(PhirJsonCvarExport),
    Qop(PhirJsonQop),
    Cop(PhirJsonCop),
    Block(PhirJsonBlock),
    Barrier(PhirJsonBarrier),
}

/// PHIR/JSON comment.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PhirJsonComment {
    #[serde(rename = "//")]
    pub comment: String,
}

/// PHIR/JSON quantum variable definition.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PhirJsonQvarDefine {
    pub data: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_type: Option<&'static str>,
    pub variable: String,
    pub size: usize,
}

impl PhirJsonQvarDefine {
    pub fn new(variable: impl Into<String>, size: usize) -> Self {
        Self {
            data: "qvar_define",
            data_type: Some("qubits"),
            variable: variable.into(),
            size,
        }
    }
}

/// PHIR/JSON classical variable definition.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PhirJsonCvarDefine {
    pub data: &'static str,
    pub data_type: String,
    pub variable: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<usize>,
}

impl PhirJsonCvarDefine {
    pub fn new(variable: impl Into<String>, size: usize) -> Self {
        Self {
            data: "cvar_define",
            data_type: "i64".to_string(),
            variable: variable.into(),
            size: Some(size),
        }
    }
}

/// PHIR/JSON classical variable export.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PhirJsonCvarExport {
    pub data: &'static str,
    pub variables: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to: Option<Vec<String>>,
}

impl PhirJsonCvarExport {
    pub fn new(variables: Vec<String>) -> Self {
        Self {
            data: "cvar_export",
            variables,
            to: None,
        }
    }
}

/// PHIR/JSON quantum operation.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PhirJsonQop {
    pub qop: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub angles: Option<(Vec<f64>, String)>,
    pub args: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub returns: Option<serde_json::Value>,
}

impl PhirJsonQop {
    /// Create a single-qubit gate operation.
    pub fn single_qubit(gate: impl Into<String>, qubits: Vec<(String, usize)>) -> Self {
        let args: Vec<serde_json::Value> = qubits
            .into_iter()
            .map(|(name, idx)| serde_json::json!([name, idx]))
            .collect();
        Self {
            qop: gate.into(),
            angles: None,
            args: serde_json::Value::Array(args),
            returns: None,
        }
    }

    /// Create a two-qubit gate operation.
    pub fn two_qubit(
        gate: impl Into<String>,
        pairs: Vec<((String, usize), (String, usize))>,
    ) -> Self {
        let args: Vec<serde_json::Value> = pairs
            .into_iter()
            .map(|((n1, i1), (n2, i2))| serde_json::json!([[n1, i1], [n2, i2]]))
            .collect();
        Self {
            qop: gate.into(),
            angles: None,
            args: serde_json::Value::Array(args),
            returns: None,
        }
    }

    /// Create a single-qubit rotation.
    pub fn rotation(
        gate: impl Into<String>,
        angle: f64,
        unit: &str,
        qubits: Vec<(String, usize)>,
    ) -> Self {
        let args: Vec<serde_json::Value> = qubits
            .into_iter()
            .map(|(name, idx)| serde_json::json!([name, idx]))
            .collect();
        Self {
            qop: gate.into(),
            angles: Some((vec![angle], unit.to_string())),
            args: serde_json::Value::Array(args),
            returns: None,
        }
    }

    /// Create a measurement operation.
    pub fn measure(qubits: Vec<(String, usize)>, results: Vec<(String, usize)>) -> Self {
        let args: Vec<serde_json::Value> = qubits
            .into_iter()
            .map(|(name, idx)| serde_json::json!([name, idx]))
            .collect();
        let rets: Vec<serde_json::Value> = results
            .into_iter()
            .map(|(name, idx)| serde_json::json!([name, idx]))
            .collect();
        Self {
            qop: "Measure".to_string(),
            angles: None,
            args: serde_json::Value::Array(args),
            returns: Some(serde_json::Value::Array(rets)),
        }
    }

    /// Create an Init operation.
    pub fn init(qubits: Vec<(String, usize)>) -> Self {
        let args: Vec<serde_json::Value> = qubits
            .into_iter()
            .map(|(name, idx)| serde_json::json!([name, idx]))
            .collect();
        Self {
            qop: "Init".to_string(),
            angles: None,
            args: serde_json::Value::Array(args),
            returns: None,
        }
    }
}

/// PHIR/JSON classical operation.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PhirJsonCop {
    pub cop: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub returns: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<String>,
}

impl PhirJsonCop {
    /// Create an assignment operation.
    pub fn assign(value: serde_json::Value, target: serde_json::Value) -> Self {
        Self {
            cop: "=".to_string(),
            args: Some(serde_json::Value::Array(vec![value])),
            returns: Some(serde_json::Value::Array(vec![target])),
            function: None,
        }
    }

    /// Create a Result export operation.
    pub fn result(sources: Vec<String>, targets: Vec<String>) -> Self {
        Self {
            cop: "Result".to_string(),
            args: Some(serde_json::Value::Array(
                sources.into_iter().map(serde_json::Value::String).collect(),
            )),
            returns: Some(serde_json::Value::Array(
                targets.into_iter().map(serde_json::Value::String).collect(),
            )),
            function: None,
        }
    }
}

/// PHIR/JSON block (sequence, qparallel, if).
#[derive(Debug, Clone, serde::Serialize)]
pub struct PhirJsonBlock {
    pub block: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ops: Option<Vec<PhirJsonOp>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condition: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub true_branch: Option<Vec<PhirJsonOp>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub false_branch: Option<Vec<PhirJsonOp>>,
}

impl PhirJsonBlock {
    /// Create a qparallel block.
    pub fn qparallel(ops: Vec<PhirJsonOp>) -> Self {
        Self {
            block: "qparallel".to_string(),
            ops: Some(ops),
            condition: None,
            true_branch: None,
            false_branch: None,
        }
    }

    /// Create an if block.
    pub fn if_block(
        condition: serde_json::Value,
        true_branch: Vec<PhirJsonOp>,
        false_branch: Option<Vec<PhirJsonOp>>,
    ) -> Self {
        Self {
            block: "if".to_string(),
            ops: None,
            condition: Some(condition),
            true_branch: Some(true_branch),
            false_branch,
        }
    }
}

/// PHIR/JSON barrier meta instruction.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PhirJsonBarrier {
    pub meta: &'static str,
    pub args: serde_json::Value,
}

impl PhirJsonBarrier {
    pub fn new(qubits: Vec<(String, usize)>) -> Self {
        let args: Vec<serde_json::Value> = qubits
            .into_iter()
            .map(|(name, idx)| serde_json::json!([name, idx]))
            .collect();
        Self {
            meta: "barrier",
            args: serde_json::Value::Array(args),
        }
    }
}

// =============================================================================
// Gate Information
// =============================================================================

/// Gate information for PHIR/JSON output.
struct GateInfo {
    /// Gate name in PHIR/JSON.
    phir_name: &'static str,
    /// Number of qubits (1 or 2).
    num_qubits: usize,
    /// Number of angle parameters.
    num_angles: usize,
}

/// Get gate info from GateKind.
fn get_gate_info(kind: GateKind) -> GateInfo {
    match kind {
        // Single-qubit gates
        GateKind::H => GateInfo {
            phir_name: "H",
            num_qubits: 1,
            num_angles: 0,
        },
        GateKind::X => GateInfo {
            phir_name: "X",
            num_qubits: 1,
            num_angles: 0,
        },
        GateKind::Y => GateInfo {
            phir_name: "Y",
            num_qubits: 1,
            num_angles: 0,
        },
        GateKind::Z => GateInfo {
            phir_name: "Z",
            num_qubits: 1,
            num_angles: 0,
        },
        GateKind::T => GateInfo {
            phir_name: "T",
            num_qubits: 1,
            num_angles: 0,
        },
        GateKind::Tdg => GateInfo {
            phir_name: "Tdg",
            num_qubits: 1,
            num_angles: 0,
        },
        GateKind::SX => GateInfo {
            phir_name: "SX",
            num_qubits: 1,
            num_angles: 0,
        },
        GateKind::SXdg => GateInfo {
            phir_name: "SXdg",
            num_qubits: 1,
            num_angles: 0,
        },
        GateKind::SY => GateInfo {
            phir_name: "SY",
            num_qubits: 1,
            num_angles: 0,
        },
        GateKind::SYdg => GateInfo {
            phir_name: "SYdg",
            num_qubits: 1,
            num_angles: 0,
        },
        GateKind::SZ => GateInfo {
            phir_name: "SZ",
            num_qubits: 1,
            num_angles: 0,
        },
        GateKind::SZdg => GateInfo {
            phir_name: "SZdg",
            num_qubits: 1,
            num_angles: 0,
        },
        GateKind::F => GateInfo {
            phir_name: "F",
            num_qubits: 1,
            num_angles: 0,
        },
        GateKind::Fdg => GateInfo {
            phir_name: "Fdg",
            num_qubits: 1,
            num_angles: 0,
        },
        GateKind::F4 => GateInfo {
            phir_name: "F4",
            num_qubits: 1,
            num_angles: 0,
        },
        GateKind::F4dg => GateInfo {
            phir_name: "F4dg",
            num_qubits: 1,
            num_angles: 0,
        },

        // Single-qubit rotations
        GateKind::RX => GateInfo {
            phir_name: "RX",
            num_qubits: 1,
            num_angles: 1,
        },
        GateKind::RY => GateInfo {
            phir_name: "RY",
            num_qubits: 1,
            num_angles: 1,
        },
        GateKind::RZ => GateInfo {
            phir_name: "RZ",
            num_qubits: 1,
            num_angles: 1,
        },

        // Two-qubit gates
        GateKind::CX => GateInfo {
            phir_name: "CX",
            num_qubits: 2,
            num_angles: 0,
        },
        GateKind::CY => GateInfo {
            phir_name: "CY",
            num_qubits: 2,
            num_angles: 0,
        },
        GateKind::CZ => GateInfo {
            phir_name: "CZ",
            num_qubits: 2,
            num_angles: 0,
        },
        GateKind::CH => GateInfo {
            phir_name: "CH",
            num_qubits: 2,
            num_angles: 0,
        },
        GateKind::SWAP => GateInfo {
            phir_name: "SWAP",
            num_qubits: 2,
            num_angles: 0,
        },
        GateKind::ISWAP => GateInfo {
            phir_name: "ISWAP",
            num_qubits: 2,
            num_angles: 0,
        },
        GateKind::SXX => GateInfo {
            phir_name: "SXX",
            num_qubits: 2,
            num_angles: 0,
        },
        GateKind::SXXdg => GateInfo {
            phir_name: "SXXdg",
            num_qubits: 2,
            num_angles: 0,
        },
        GateKind::SYY => GateInfo {
            phir_name: "SYY",
            num_qubits: 2,
            num_angles: 0,
        },
        GateKind::SYYdg => GateInfo {
            phir_name: "SYYdg",
            num_qubits: 2,
            num_angles: 0,
        },
        GateKind::SZZ => GateInfo {
            phir_name: "SZZ",
            num_qubits: 2,
            num_angles: 0,
        },
        GateKind::SZZdg => GateInfo {
            phir_name: "SZZdg",
            num_qubits: 2,
            num_angles: 0,
        },
        GateKind::RZZ => GateInfo {
            phir_name: "RZZ",
            num_qubits: 2,
            num_angles: 1,
        },

        // Three-qubit gates
        GateKind::CCX => GateInfo {
            phir_name: "CCX",
            num_qubits: 3,
            num_angles: 0,
        },

        // Prepare operations (treated as Init)
        GateKind::PZ => GateInfo {
            phir_name: "Init",
            num_qubits: 1,
            num_angles: 0,
        },
    }
}

// =============================================================================
// Allocator Tracking
// =============================================================================

/// Tracks an allocator during codegen.
#[derive(Debug, Clone)]
struct AllocatorInfo {
    name: String,
    capacity: usize,
}

/// Tracks a classical register during codegen.
#[derive(Debug, Clone)]
struct RegisterInfo {
    name: String,
    size: usize,
}

// =============================================================================
// PHIR/JSON Code Generator
// =============================================================================

/// PHIR/JSON code generator.
///
/// Walks a Zlup AST and produces PHIR/JSON output.
pub struct PhirJsonCodegen {
    /// Allocators by name.
    allocators: BTreeMap<String, AllocatorInfo>,
    /// Classical registers by name.
    registers: BTreeMap<String, RegisterInfo>,
    /// Auto-generated register counter.
    register_counter: usize,
}

impl Default for PhirJsonCodegen {
    fn default() -> Self {
        Self::new()
    }
}

impl PhirJsonCodegen {
    /// Create a new PHIR/JSON code generator.
    pub fn new() -> Self {
        Self {
            allocators: BTreeMap::new(),
            registers: BTreeMap::new(),
            register_counter: 0,
        }
    }

    /// Compile a Zlup program to PHIR/JSON.
    pub fn compile(&mut self, program: &Program) -> PhirJsonResult<PhirJsonProgram> {
        let mut phir = PhirJsonProgram::new().with_name("main");

        // First pass: collect allocators
        for decl in &program.declarations {
            self.collect_decl(decl)?;
        }

        // Add quantum variable definitions
        for alloc in self.allocators.values() {
            phir.ops
                .push(PhirJsonOp::QvarDefine(PhirJsonQvarDefine::new(
                    &alloc.name,
                    alloc.capacity,
                )));
        }

        // Add classical variable definitions
        for reg in self.registers.values() {
            phir.ops
                .push(PhirJsonOp::CvarDefine(PhirJsonCvarDefine::new(
                    &reg.name, reg.size,
                )));
        }

        // Second pass: convert main function body
        for decl in &program.declarations {
            if let TopLevelDecl::Fn(fn_decl) = decl
                && fn_decl.name == "main"
            {
                let ops = self.convert_block(&fn_decl.body)?;
                phir.ops.extend(ops);
            }
        }

        // Export all classical variables
        if !self.registers.is_empty() {
            let vars: Vec<String> = self.registers.keys().cloned().collect();
            phir.ops
                .push(PhirJsonOp::CvarExport(PhirJsonCvarExport::new(vars)));
        }

        Ok(phir)
    }

    /// Compile a function to PHIR/JSON.
    pub fn compile_function(&mut self, fn_decl: &FnDecl) -> PhirJsonResult<PhirJsonProgram> {
        // Collect from function body
        self.collect_block(&fn_decl.body)?;

        let mut phir = PhirJsonProgram::new().with_name(&fn_decl.name);

        // Add definitions
        for alloc in self.allocators.values() {
            phir.ops
                .push(PhirJsonOp::QvarDefine(PhirJsonQvarDefine::new(
                    &alloc.name,
                    alloc.capacity,
                )));
        }

        for reg in self.registers.values() {
            phir.ops
                .push(PhirJsonOp::CvarDefine(PhirJsonCvarDefine::new(
                    &reg.name, reg.size,
                )));
        }

        // Convert body
        let ops = self.convert_block(&fn_decl.body)?;
        phir.ops.extend(ops);

        Ok(phir)
    }

    /// Convert to JSON string.
    pub fn to_json(&self, program: &PhirJsonProgram) -> PhirJsonResult<String> {
        serde_json::to_string_pretty(program).map_err(|e| PhirJsonError::JsonError(e.to_string()))
    }

    /// Convert to compact JSON string.
    pub fn to_json_compact(&self, program: &PhirJsonProgram) -> PhirJsonResult<String> {
        serde_json::to_string(program).map_err(|e| PhirJsonError::JsonError(e.to_string()))
    }

    // =========================================================================
    // Collection Phase
    // =========================================================================

    fn collect_decl(&mut self, decl: &TopLevelDecl) -> PhirJsonResult<()> {
        match decl {
            TopLevelDecl::Fn(fn_decl) => {
                if fn_decl.name == "main" {
                    self.collect_block(&fn_decl.body)?;
                }
            }
            TopLevelDecl::Binding(binding) => self.collect_binding(binding)?,
            _ => {}
        }
        Ok(())
    }

    fn collect_block(&mut self, block: &Block) -> PhirJsonResult<()> {
        for stmt in &block.statements {
            self.collect_stmt(stmt)?;
        }
        Ok(())
    }

    fn collect_stmt(&mut self, stmt: &Stmt) -> PhirJsonResult<()> {
        match stmt {
            Stmt::Binding(binding) => self.collect_binding(binding)?,
            Stmt::If(if_stmt) => {
                self.collect_block(&if_stmt.then_body)?;
                if let Some(else_branch) = &if_stmt.else_body {
                    match else_branch {
                        ElseBranch::Else(block) => self.collect_block(block)?,
                        ElseBranch::ElseIf(nested_if) => {
                            self.collect_stmt(&Stmt::If(*nested_if.clone()))?;
                        }
                    }
                }
            }
            Stmt::For(for_stmt) => {
                self.collect_block(&for_stmt.body)?;
            }
            Stmt::Tick(tick_stmt) => {
                for stmt in &tick_stmt.body {
                    self.collect_stmt(stmt)?;
                }
            }
            Stmt::Block(block) => {
                self.collect_block(block)?;
            }
            _ => {}
        }
        Ok(())
    }

    fn collect_binding(&mut self, binding: &Binding) -> PhirJsonResult<()> {
        if let Some(ref init) = binding.value {
            // Check for qalloc
            if let Expr::Call(call) = init
                && self.get_callee_name(call) == Some("qalloc".to_string())
                && let Some(Expr::IntLit(IntLit { value, .. })) = call.args.first()
            {
                self.allocators.insert(
                    binding.name.clone(),
                    AllocatorInfo {
                        name: binding.name.clone(),
                        capacity: *value as usize,
                    },
                );
            }

            // Check for typed measurement (creates a register)
            if let Expr::Call(call) = init
                && let Some(name) = self.get_callee_name(call)
                && (name == "mz" || name == "mx" || name == "my")
            {
                let size = call.args.len().max(1);
                self.registers.insert(
                    binding.name.clone(),
                    RegisterInfo {
                        name: binding.name.clone(),
                        size,
                    },
                );
            }
        }
        Ok(())
    }

    // =========================================================================
    // Conversion Phase
    // =========================================================================

    fn convert_block(&mut self, block: &Block) -> PhirJsonResult<Vec<PhirJsonOp>> {
        let mut ops = Vec::new();
        for stmt in &block.statements {
            ops.extend(self.convert_stmt(stmt)?);
        }
        Ok(ops)
    }

    fn convert_stmt(&mut self, stmt: &Stmt) -> PhirJsonResult<Vec<PhirJsonOp>> {
        match stmt {
            Stmt::Binding(binding) => self.convert_binding(binding),
            Stmt::Expr(expr_stmt) => self.convert_expr_stmt(&expr_stmt.expr),
            Stmt::If(if_stmt) => self.convert_if(if_stmt),
            Stmt::For(for_stmt) => self.convert_for(for_stmt),
            Stmt::Tick(tick_stmt) => self.convert_tick(tick_stmt),
            Stmt::Return(_) => Ok(vec![]),
            Stmt::Block(block) => self.convert_block(block),
            Stmt::Gate(gate_op) => self.convert_gate(gate_op),
            Stmt::Prepare(prepare_op) => self.convert_prepare(prepare_op),
            Stmt::Measure(measure_op) => self.convert_measure(measure_op),
            Stmt::Barrier(barrier_op) => {
                let qubits: Vec<(String, usize)> = barrier_op
                    .allocators
                    .iter()
                    .flat_map(|alloc| {
                        self.allocators
                            .get(alloc)
                            .map(|info| {
                                (0..info.capacity)
                                    .map(|i| (info.name.clone(), i))
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_default()
                    })
                    .collect();
                Ok(vec![PhirJsonOp::Barrier(PhirJsonBarrier::new(qubits))])
            }
            Stmt::Break(_) | Stmt::Continue(_) => Ok(vec![]),
            _ => Ok(vec![]),
        }
    }

    fn convert_binding(&mut self, binding: &Binding) -> PhirJsonResult<Vec<PhirJsonOp>> {
        let mut ops = Vec::new();

        if let Some(ref init) = binding.value {
            // Check for qalloc - already handled in collection phase
            if let Expr::Call(call) = init
                && self.get_callee_name(call) == Some("qalloc".to_string())
            {
                return Ok(vec![]);
            }

            // Check for measurement call (mz(...) [targets])
            if let Expr::Call(call) = init
                && let Some(name) = self.get_callee_name(call)
                && (name == "mz" || name == "mx" || name == "my")
            {
                let qubits = self.extract_qubits_from_args(&call.args)?;
                let results: Vec<(String, usize)> = qubits
                    .iter()
                    .enumerate()
                    .map(|(i, _)| (binding.name.clone(), i))
                    .collect();
                ops.push(PhirJsonOp::Qop(PhirJsonQop::measure(qubits, results)));
                return Ok(ops);
            }

            // Check for measurement expression (mz(T) targets)
            if let Expr::Measure(measure_expr) = init {
                let qubits = self.extract_qubits_from_target(&measure_expr.targets)?;
                let results: Vec<(String, usize)> = qubits
                    .iter()
                    .enumerate()
                    .map(|(i, _)| (binding.name.clone(), i))
                    .collect();
                ops.push(PhirJsonOp::Qop(PhirJsonQop::measure(qubits, results)));
                return Ok(ops);
            }

            // Try to convert to a value for assignment - skip unsupported expressions
            match self.convert_expr_to_value(init) {
                Ok(value) => {
                    ops.push(PhirJsonOp::Cop(PhirJsonCop::assign(
                        value,
                        serde_json::Value::String(binding.name.clone()),
                    )));
                }
                Err(PhirJsonError::UnsupportedExpression) => {
                    // Skip unsupported expressions silently - they may be quantum ops
                }
                Err(e) => return Err(e),
            }
        }

        Ok(ops)
    }

    fn convert_expr_stmt(&mut self, expr: &Expr) -> PhirJsonResult<Vec<PhirJsonOp>> {
        match expr {
            Expr::Call(call) => self.convert_call(call),
            Expr::Gate(gate_expr) => self.convert_gate_expr(gate_expr),
            Expr::Measure(measure_expr) => self.convert_measure_expr(measure_expr),
            _ => Ok(vec![]),
        }
    }

    fn convert_gate_expr(
        &self,
        gate_expr: &crate::ast::GateExpr,
    ) -> PhirJsonResult<Vec<PhirJsonOp>> {
        let gate_info = get_gate_info(gate_expr.kind);

        // Handle prepare operations
        if gate_info.phir_name == "Init" {
            let qubits = self.extract_qubits_from_target(&gate_expr.target)?;
            if qubits.is_empty() {
                // Prepare all qubits in the allocator
                if let Expr::Ident(ident) = &gate_expr.target
                    && let Some(alloc) = self.allocators.get(&ident.name)
                {
                    let all_qubits: Vec<(String, usize)> = (0..alloc.capacity)
                        .map(|i| (alloc.name.clone(), i))
                        .collect();
                    return Ok(vec![PhirJsonOp::Qop(PhirJsonQop::init(all_qubits))]);
                }
            }
            return Ok(vec![PhirJsonOp::Qop(PhirJsonQop::init(qubits))]);
        }

        let qubits = self.extract_qubits_from_target(&gate_expr.target)?;

        if gate_info.num_qubits == 1 {
            if gate_info.num_angles > 0 {
                // Rotation gate - get angle from params
                let angle = gate_expr
                    .params
                    .first()
                    .and_then(|p| self.eval_expr_to_float(p))
                    .unwrap_or(0.0);
                Ok(vec![PhirJsonOp::Qop(PhirJsonQop::rotation(
                    gate_info.phir_name,
                    angle * std::f64::consts::TAU,
                    "rad",
                    qubits,
                ))])
            } else {
                Ok(vec![PhirJsonOp::Qop(PhirJsonQop::single_qubit(
                    gate_info.phir_name,
                    qubits,
                ))])
            }
        } else {
            // Two-qubit gate - pair up qubits
            if qubits.len() % 2 != 0 && gate_info.num_qubits == 2 {
                return Err(PhirJsonError::WrongArgumentCount {
                    gate: gate_info.phir_name.to_string(),
                    expected: 2,
                    got: qubits.len(),
                });
            }
            let pairs: Vec<_> = qubits
                .chunks(2)
                .map(|chunk| (chunk[0].clone(), chunk[1].clone()))
                .collect();
            Ok(vec![PhirJsonOp::Qop(PhirJsonQop::two_qubit(
                gate_info.phir_name,
                pairs,
            ))])
        }
    }

    fn convert_measure_expr(
        &mut self,
        measure_expr: &crate::ast::MeasureExpr,
    ) -> PhirJsonResult<Vec<PhirJsonOp>> {
        let qubits = self.extract_qubits_from_target(&measure_expr.targets)?;
        let reg_name = format!("m{}", self.register_counter);
        self.register_counter += 1;
        let results: Vec<(String, usize)> = qubits
            .iter()
            .enumerate()
            .map(|(i, _)| (reg_name.clone(), i))
            .collect();
        Ok(vec![PhirJsonOp::Qop(PhirJsonQop::measure(qubits, results))])
    }

    fn extract_qubits_from_target(&self, target: &Expr) -> PhirJsonResult<Vec<(String, usize)>> {
        let mut qubits = Vec::new();
        match target {
            Expr::Index(idx_expr) => {
                if let Expr::Ident(ident) = &idx_expr.object
                    && let Some(idx) = self.eval_index(&idx_expr.index)
                {
                    qubits.push((ident.name.clone(), idx));
                }
            }
            Expr::Tuple(tuple_expr) => {
                for elem in &tuple_expr.elements {
                    qubits.extend(self.extract_qubits_from_target(elem)?);
                }
            }
            Expr::BracketArray(arr) => {
                for elem in &arr.elements {
                    qubits.extend(self.extract_qubits_from_target(elem)?);
                }
            }
            Expr::Set(set_expr) => {
                for elem in &set_expr.elements {
                    qubits.extend(self.extract_qubits_from_target(elem)?);
                }
            }
            Expr::SlotRef(slot_ref) => {
                if let Some(idx) = self.eval_index(&slot_ref.index) {
                    qubits.push((slot_ref.allocator.clone(), idx));
                }
            }
            Expr::Ident(_) => {
                // This is a bare allocator name - handled by caller
            }
            _ => {}
        }
        Ok(qubits)
    }

    fn convert_call(&mut self, call: &CallExpr) -> PhirJsonResult<Vec<PhirJsonOp>> {
        let name = self.get_callee_name(call).unwrap_or_default();

        // Check for prepare operations
        if name == "pz" || name == "px" || name == "py" {
            let qubits = self.extract_qubits_from_args(&call.args)?;
            if qubits.is_empty() {
                // Prepare all qubits in the allocator
                if let Some(Expr::Ident(ident)) = call.args.first()
                    && let Some(alloc) = self.allocators.get(&ident.name)
                {
                    let all_qubits: Vec<(String, usize)> = (0..alloc.capacity)
                        .map(|i| (alloc.name.clone(), i))
                        .collect();
                    return Ok(vec![PhirJsonOp::Qop(PhirJsonQop::init(all_qubits))]);
                }
            }
            return Ok(vec![PhirJsonOp::Qop(PhirJsonQop::init(qubits))]);
        }

        // Check for measurement
        if name == "mz" || name == "mx" || name == "my" {
            let qubits = self.extract_qubits_from_args(&call.args)?;
            let reg_name = format!("c{}", self.register_counter);
            self.register_counter += 1;
            let results: Vec<(String, usize)> = qubits
                .iter()
                .enumerate()
                .map(|(i, _)| (reg_name.clone(), i))
                .collect();
            return Ok(vec![PhirJsonOp::Qop(PhirJsonQop::measure(qubits, results))]);
        }

        Ok(vec![])
    }

    fn convert_gate(&self, gate_op: &GateOp) -> PhirJsonResult<Vec<PhirJsonOp>> {
        let gate_info = get_gate_info(gate_op.kind);

        // Handle prepare operations
        if gate_info.phir_name == "Init" {
            let qubits = self.convert_slot_refs(&gate_op.targets);
            return Ok(vec![PhirJsonOp::Qop(PhirJsonQop::init(qubits))]);
        }

        // Handle measurement operations
        if gate_info.phir_name == "Measure" {
            let qubits = self.convert_slot_refs(&gate_op.targets);
            let reg_name = format!("m{}", qubits.len());
            let results: Vec<(String, usize)> = qubits
                .iter()
                .enumerate()
                .map(|(i, _)| (reg_name.clone(), i))
                .collect();
            return Ok(vec![PhirJsonOp::Qop(PhirJsonQop::measure(qubits, results))]);
        }

        let qubits = self.convert_slot_refs(&gate_op.targets);

        if gate_info.num_qubits == 1 {
            if gate_info.num_angles > 0 {
                // Rotation gate - get angle from params
                let angle = gate_op
                    .params
                    .first()
                    .and_then(|p| self.eval_expr_to_float(p))
                    .unwrap_or(0.0);
                Ok(vec![PhirJsonOp::Qop(PhirJsonQop::rotation(
                    gate_info.phir_name,
                    angle * std::f64::consts::TAU,
                    "rad",
                    qubits,
                ))])
            } else {
                Ok(vec![PhirJsonOp::Qop(PhirJsonQop::single_qubit(
                    gate_info.phir_name,
                    qubits,
                ))])
            }
        } else {
            // Two-qubit gate - pair up qubits
            if !qubits.len().is_multiple_of(2) {
                return Err(PhirJsonError::WrongArgumentCount {
                    gate: gate_info.phir_name.to_string(),
                    expected: 2,
                    got: qubits.len(),
                });
            }
            let pairs: Vec<_> = qubits
                .chunks(2)
                .map(|chunk| (chunk[0].clone(), chunk[1].clone()))
                .collect();
            Ok(vec![PhirJsonOp::Qop(PhirJsonQop::two_qubit(
                gate_info.phir_name,
                pairs,
            ))])
        }
    }

    fn convert_prepare(&self, prepare_op: &PrepareOp) -> PhirJsonResult<Vec<PhirJsonOp>> {
        let alloc = self.allocators.get(&prepare_op.allocator).ok_or_else(|| {
            PhirJsonError::UndefinedAllocator {
                name: prepare_op.allocator.clone(),
            }
        })?;

        let qubits: Vec<(String, usize)> = if let Some(ref slots) = prepare_op.slots {
            slots
                .iter()
                .map(|&i| (alloc.name.clone(), i as usize))
                .collect()
        } else {
            (0..alloc.capacity)
                .map(|i| (alloc.name.clone(), i))
                .collect()
        };

        Ok(vec![PhirJsonOp::Qop(PhirJsonQop::init(qubits))])
    }

    fn convert_measure(&mut self, measure_op: &MeasureOp) -> PhirJsonResult<Vec<PhirJsonOp>> {
        let qubits = self.convert_slot_refs(&measure_op.targets);
        let results: Vec<(String, usize)> = measure_op
            .results
            .iter()
            .filter_map(|br| {
                self.eval_index(&br.index)
                    .map(|idx| (br.register.clone(), idx))
            })
            .collect();

        Ok(vec![PhirJsonOp::Qop(PhirJsonQop::measure(qubits, results))])
    }

    fn convert_if(&mut self, if_stmt: &IfStmt) -> PhirJsonResult<Vec<PhirJsonOp>> {
        let condition = self.convert_expr_to_value(&if_stmt.condition)?;
        let true_branch = self.convert_block(&if_stmt.then_body)?;

        let false_branch = if let Some(ref else_branch) = if_stmt.else_body {
            match else_branch {
                ElseBranch::Else(block) => Some(self.convert_block(block)?),
                ElseBranch::ElseIf(nested_if) => Some(self.convert_if(nested_if)?),
            }
        } else {
            None
        };

        Ok(vec![PhirJsonOp::Block(PhirJsonBlock::if_block(
            condition,
            true_branch,
            false_branch,
        ))])
    }

    fn convert_for(&mut self, for_stmt: &crate::ast::ForStmt) -> PhirJsonResult<Vec<PhirJsonOp>> {
        let mut ops = Vec::new();

        if let ForRange::Range { start, end, .. } = &for_stmt.range
            && let (Some(start_val), Some(end_val)) =
                (self.try_eval_const(start), self.try_eval_const(end))
        {
            for _ in start_val..end_val {
                ops.extend(self.convert_block(&for_stmt.body)?);
            }
            return Ok(ops);
        }

        Err(PhirJsonError::UnsupportedStatement(
            "for loops with non-constant bounds".to_string(),
        ))
    }

    fn convert_tick(&mut self, tick_stmt: &TickStmt) -> PhirJsonResult<Vec<PhirJsonOp>> {
        let mut qops = Vec::new();
        for stmt in &tick_stmt.body {
            let converted = self.convert_stmt(stmt)?;
            for op in converted {
                if matches!(op, PhirJsonOp::Qop(_)) {
                    qops.push(op);
                }
            }
        }

        if qops.is_empty() {
            Ok(vec![])
        } else {
            Ok(vec![PhirJsonOp::Block(PhirJsonBlock::qparallel(qops))])
        }
    }

    // =========================================================================
    // Helper Methods
    // =========================================================================

    fn get_callee_name(&self, call: &CallExpr) -> Option<String> {
        match &call.callee {
            Expr::Ident(ident) => Some(ident.name.clone()),
            _ => None,
        }
    }

    fn convert_slot_refs(&self, targets: &[crate::ast::SlotRef]) -> Vec<(String, usize)> {
        targets
            .iter()
            .filter_map(|slot| {
                self.eval_index(&slot.index)
                    .map(|idx| (slot.allocator.clone(), idx))
            })
            .collect()
    }

    fn extract_qubits_from_args(&self, args: &[Expr]) -> PhirJsonResult<Vec<(String, usize)>> {
        let mut qubits = Vec::new();
        for arg in args {
            match arg {
                Expr::Index(idx_expr) => {
                    if let Expr::Ident(ident) = &idx_expr.object
                        && let Some(idx) = self.eval_index(&idx_expr.index)
                    {
                        qubits.push((ident.name.clone(), idx));
                    }
                }
                Expr::Tuple(tuple_expr) => {
                    for elem in &tuple_expr.elements {
                        if let Expr::Index(idx_expr) = elem
                            && let Expr::Ident(ident) = &idx_expr.object
                            && let Some(idx) = self.eval_index(&idx_expr.index)
                        {
                            qubits.push((ident.name.clone(), idx));
                        }
                    }
                }
                Expr::BracketArray(arr) => {
                    for elem in &arr.elements {
                        if let Expr::Index(idx_expr) = elem
                            && let Expr::Ident(ident) = &idx_expr.object
                            && let Some(idx) = self.eval_index(&idx_expr.index)
                        {
                            qubits.push((ident.name.clone(), idx));
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(qubits)
    }

    fn eval_index(&self, expr: &Expr) -> Option<usize> {
        match expr {
            Expr::IntLit(IntLit { value, .. }) => Some(*value as usize),
            _ => None,
        }
    }

    fn eval_expr_to_float(&self, expr: &Expr) -> Option<f64> {
        match expr {
            Expr::IntLit(IntLit { value, .. }) => Some(*value as f64),
            Expr::FloatLit(fl) => Some(fl.value),
            Expr::AngleLit(angle) => {
                // Evaluate the angle value and convert to turns
                let value = self.eval_expr_to_float(&angle.value)?;
                Some(angle.unit.to_turns(value))
            }
            _ => None,
        }
    }

    fn try_eval_const(&self, expr: &Expr) -> Option<i64> {
        match expr {
            Expr::IntLit(IntLit { value, .. }) => Some(*value as i64),
            _ => None,
        }
    }

    fn convert_expr_to_value(&self, expr: &Expr) -> PhirJsonResult<serde_json::Value> {
        match expr {
            Expr::IntLit(IntLit { value, .. }) => {
                Ok(serde_json::Value::Number((*value as i64).into()))
            }
            Expr::FloatLit(fl) => Ok(serde_json::json!(fl.value)),
            Expr::BoolLit(bl) => Ok(serde_json::Value::Number(
                if bl.value { 1 } else { 0 }.into(),
            )),
            Expr::Ident(ident) => Ok(serde_json::Value::String(ident.name.clone())),
            Expr::Binary(bin) => {
                let left = self.convert_expr_to_value(&bin.left)?;
                let right = self.convert_expr_to_value(&bin.right)?;
                let op = match bin.op {
                    BinaryOp::Add => "+",
                    BinaryOp::Sub => "-",
                    BinaryOp::Mul => "*",
                    BinaryOp::Div => "/",
                    BinaryOp::Mod => "%",
                    BinaryOp::Eq => "==",
                    BinaryOp::Ne => "!=",
                    BinaryOp::Lt => "<",
                    BinaryOp::Le => "<=",
                    BinaryOp::Gt => ">",
                    BinaryOp::Ge => ">=",
                    BinaryOp::BitAnd => "&",
                    BinaryOp::BitOr => "|",
                    BinaryOp::BitXor => "^",
                    BinaryOp::Shl => "<<",
                    BinaryOp::Shr => ">>",
                    BinaryOp::And => "&",
                    BinaryOp::Or => "|",
                    _ => return Err(PhirJsonError::UnsupportedExpression),
                };
                Ok(serde_json::json!({"cop": op, "args": [left, right]}))
            }
            Expr::Unary(un) => {
                let operand = self.convert_expr_to_value(&un.operand)?;
                let op = match un.op {
                    UnaryOp::Neg => "-",
                    UnaryOp::Not => "~",
                    _ => return Err(PhirJsonError::UnsupportedExpression),
                };
                Ok(serde_json::json!({"cop": op, "args": [operand]}))
            }
            Expr::Index(idx) => {
                if let Expr::Ident(ident) = &idx.object
                    && let Some(i) = self.eval_index(&idx.index)
                {
                    return Ok(serde_json::json!([ident.name, i]));
                }
                Err(PhirJsonError::UnsupportedExpression)
            }
            _ => Err(PhirJsonError::UnsupportedExpression),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bell_state() {
        let source = r#"
pub fn main() -> unit {
    q := qalloc(2);
    pz q;
    h q[0];
    cx (q[0], q[1]);
    results: [2]u1 = mz([2]u1) [q[0], q[1]];
    return unit;
}
"#;
        let ast = crate::parse(source).unwrap();
        let mut codegen = PhirJsonCodegen::new();
        let phir = codegen.compile(&ast).unwrap();
        let json = codegen.to_json(&phir).unwrap();

        assert!(json.contains("\"format\": \"PHIR/JSON\""));
        assert!(json.contains("\"version\": \"0.1.0\""));
        assert!(json.contains("\"qvar_define\""));
        assert!(json.contains("\"H\""));
        assert!(json.contains("\"CX\""));
        assert!(json.contains("\"Measure\""));
    }

    #[test]
    fn test_ghz_state() {
        let source = r#"
pub fn main() -> unit {
    q := qalloc(4);
    pz q;
    h q[0];
    cx (q[0], q[1]);
    cx (q[0], q[2]);
    cx (q[0], q[3]);
    results: [4]u1 = mz([4]u1) [q[0], q[1], q[2], q[3]];
    return unit;
}
"#;
        let ast = crate::parse(source).unwrap();
        let mut codegen = PhirJsonCodegen::new();
        let phir = codegen.compile(&ast).unwrap();
        let json = codegen.to_json(&phir).unwrap();

        assert!(json.contains("\"size\": 4"));
        assert!(json.contains("\"CX\""));
    }

    #[test]
    fn test_single_qubit_gates() {
        let source = r#"
pub fn main() -> unit {
    q := qalloc(1);
    pz q;
    h q[0];
    x q[0];
    y q[0];
    z q[0];
    sz q[0];
    t q[0];
    return unit;
}
"#;
        let ast = crate::parse(source).unwrap();
        let mut codegen = PhirJsonCodegen::new();
        let phir = codegen.compile(&ast).unwrap();
        let json = codegen.to_json(&phir).unwrap();

        assert!(json.contains("\"H\""));
        assert!(json.contains("\"X\""));
        assert!(json.contains("\"Y\""));
        assert!(json.contains("\"Z\""));
        assert!(json.contains("\"SZ\""));
        assert!(json.contains("\"T\""));
    }

    #[test]
    fn test_to_json_format() {
        let source = r#"
pub fn main() -> unit {
    q := qalloc(2);
    pz q;
    h q[0];
    return unit;
}
"#;
        let ast = crate::parse(source).unwrap();
        let mut codegen = PhirJsonCodegen::new();
        let phir = codegen.compile(&ast).unwrap();
        let json = codegen.to_json(&phir).unwrap();

        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["format"], "PHIR/JSON");
        assert_eq!(parsed["version"], "0.1.0");
        assert!(parsed["ops"].is_array());
    }
}
