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

use std::cell::RefCell;
use std::collections::BTreeMap;
use thiserror::Error;

use crate::ast::{
    BinaryOp, Binding, Block, CallExpr, ElseBranch, Expr, FnDecl, ForRange, GateKind, GateOp,
    IfStmt, IntLit, MeasureOp, PrepareOp, Program, Stmt, TickStmt, TopLevelDecl, UnaryOp,
};
use crate::comptime::angle_expression_name;
use crate::comptime::{
    ComptimeEvaluator, ComptimeValue, angle_evaluator, define_comptime_binding, resolve_angle_turns,
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

    #[error("classical variable '{name}' has no runtime declaration or assignment")]
    UndefinedClassicalVariable { name: String },

    #[error("qubit or register index must be a nonnegative compile-time integer")]
    NonConstantIndex,

    #[error("allocation capacity must be a nonnegative compile-time integer")]
    InvalidAllocationCapacity,

    #[error("rotation angle '{expression}' is not known at compile time: {reason}")]
    RuntimeAngle { expression: String, reason: String },

    #[error("rotation gate is missing its angle")]
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

    /// Create a two-qubit rotation operation.
    pub fn two_qubit_rotation(
        gate: impl Into<String>,
        angle: f64,
        unit: &str,
        pairs: Vec<((String, usize), (String, usize))>,
    ) -> Self {
        let args: Vec<serde_json::Value> = pairs
            .into_iter()
            .map(|((n1, i1), (n2, i2))| serde_json::json!([[n1, i1], [n2, i2]]))
            .collect();
        Self {
            qop: gate.into(),
            angles: Some((vec![angle], unit.to_string())),
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
        GateKind::CRZ => GateInfo {
            phir_name: "CRZ",
            num_qubits: 2,
            num_angles: 1,
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
    /// Number of active statically unrolled loop bodies.
    unrolled_loop_depth: usize,
    tick_depth: usize,
    all_allocators: BTreeMap<String, AllocatorInfo>,
    /// Classical registers by name.
    registers: BTreeMap<String, RegisterInfo>,
    register_bindings: BTreeMap<String, String>,
    /// Auto-generated register counter.
    register_counter: usize,
    /// Compile-time constants used to resolve unit-bearing gate angles.
    angle_evaluator: RefCell<ComptimeEvaluator>,
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
            unrolled_loop_depth: 0,
            tick_depth: 0,
            all_allocators: BTreeMap::new(),
            registers: BTreeMap::new(),
            register_bindings: BTreeMap::new(),
            register_counter: 0,
            angle_evaluator: RefCell::new(angle_evaluator()),
        }
    }

    /// Compile a Zlup program to PHIR/JSON.
    pub fn compile(&mut self, program: &Program) -> PhirJsonResult<PhirJsonProgram> {
        *self = Self::new();
        for decl in &program.declarations {
            if let TopLevelDecl::Binding(binding) = decl {
                if !define_comptime_binding(&mut self.angle_evaluator.borrow_mut(), binding) {
                    self.angle_evaluator
                        .borrow_mut()
                        .context
                        .define(&binding.name, ComptimeValue::Undefined);
                }
                self.collect_binding(binding)?;
            }
        }
        let mut body = Vec::new();
        for decl in &program.declarations {
            if let TopLevelDecl::Fn(function) = decl
                && function.name == "main"
            {
                body.extend(self.convert_function_body(function)?);
            }
        }
        let mut phir = self.program_with_definitions("main");
        phir.ops.extend(body);
        if !self.registers.is_empty() {
            phir.ops
                .push(PhirJsonOp::CvarExport(PhirJsonCvarExport::new(
                    self.registers.keys().cloned().collect(),
                )));
        }
        Ok(phir)
    }

    /// Compile a function to PHIR/JSON.
    pub fn compile_function(&mut self, fn_decl: &FnDecl) -> PhirJsonResult<PhirJsonProgram> {
        *self = Self::new();
        let body = self.convert_function_body(fn_decl)?;
        let mut phir = self.program_with_definitions(&fn_decl.name);
        phir.ops.extend(body);
        Ok(phir)
    }

    fn convert_function_body(&mut self, fn_decl: &FnDecl) -> PhirJsonResult<Vec<PhirJsonOp>> {
        self.angle_evaluator.borrow_mut().context.push_scope();
        let allocators = self.allocators.clone();
        let register_bindings = self.register_bindings.clone();
        for param in &fn_decl.params {
            self.angle_evaluator
                .borrow_mut()
                .context
                .define(&param.name, ComptimeValue::Undefined);
            self.allocators.remove(&param.name);
            self.register_bindings.remove(&param.name);
        }
        let result = self.convert_block(&fn_decl.body);
        self.allocators = allocators;
        self.register_bindings = register_bindings;
        self.angle_evaluator.borrow_mut().context.pop_scope();
        result
    }

    fn program_with_definitions(&self, name: &str) -> PhirJsonProgram {
        let mut phir = PhirJsonProgram::new().with_name(name);
        for alloc in self
            .all_allocators
            .values()
            .filter(|alloc| alloc.capacity > 0)
        {
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
        phir
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

    fn collect_binding(&mut self, binding: &Binding) -> PhirJsonResult<()> {
        if let Some(Expr::Call(call)) = &binding.value
            && self.get_callee_name(call).as_deref() == Some("qalloc")
        {
            let capacity = self.eval_capacity(
                call.args
                    .first()
                    .ok_or(PhirJsonError::UnsupportedExpression)?,
            )?;
            let name = if self.all_allocators.contains_key(&binding.name) {
                format!("{}#{}", binding.name, self.all_allocators.len())
            } else {
                binding.name.clone()
            };
            let allocator = AllocatorInfo {
                name: name.clone(),
                capacity,
            };
            self.all_allocators.insert(name, allocator.clone());
            self.allocators.insert(binding.name.clone(), allocator);
        } else {
            self.allocators.remove(&binding.name);
        }
        Ok(())
    }

    // =========================================================================
    // Conversion Phase
    // =========================================================================

    fn convert_block(&mut self, block: &Block) -> PhirJsonResult<Vec<PhirJsonOp>> {
        self.angle_evaluator.borrow_mut().context.push_scope();
        let allocators = self.allocators.clone();
        let register_bindings = self.register_bindings.clone();
        let result = (|| {
            let mut ops = Vec::new();
            for stmt in &block.statements {
                ops.extend(self.convert_stmt(stmt)?);
            }

            Ok(ops)
        })();
        self.allocators = allocators;
        self.register_bindings = register_bindings;
        self.angle_evaluator.borrow_mut().context.pop_scope();
        result
    }

    fn convert_stmt(&mut self, stmt: &Stmt) -> PhirJsonResult<Vec<PhirJsonOp>> {
        if self.unrolled_loop_depth > 0 {
            let statement = match stmt {
                Stmt::Switch(_) => Some("switch"),
                Stmt::TryBlock(_) => Some("try block"),
                Stmt::Defer(_) => Some("defer"),
                Stmt::Errdefer(_) => Some("errdefer"),
                Stmt::Binding(binding) if matches!(&binding.value, Some(Expr::Unary(unary)) if matches!(unary.op, crate::ast::UnaryOp::Try)) => {
                    Some("try propagation")
                }
                _ => None,
            };
            if let Some(statement) = statement {
                return Err(PhirJsonError::UnsupportedStatement(format!(
                    "{statement} in an unrolled loop"
                )));
            }
        }
        match stmt {
            Stmt::Binding(binding) => {
                if self.tick_depth > 0
                    && define_comptime_binding(&mut self.angle_evaluator.borrow_mut(), binding)
                {
                    self.collect_binding(binding)?;
                    self.register_bindings.remove(&binding.name);
                    return Ok(Vec::new());
                }
                // Retain assignments outside ticks, as runtime expressions can
                // refer to these bindings. Evaluate before defining the new name.
                let result = self.convert_binding(binding)?;
                self.collect_binding(binding)?;
                if !define_comptime_binding(&mut self.angle_evaluator.borrow_mut(), binding) {
                    self.angle_evaluator
                        .borrow_mut()
                        .context
                        .define(&binding.name, ComptimeValue::Undefined);
                }
                Ok(result)
            }
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
            Stmt::Break(_) => Err(PhirJsonError::UnsupportedStatement(
                "break in an unrolled loop".to_string(),
            )),
            Stmt::Continue(_) => Err(PhirJsonError::UnsupportedStatement(
                "continue in an unrolled loop".to_string(),
            )),
            _ => Ok(vec![]),
        }
    }

    fn convert_binding(&mut self, binding: &Binding) -> PhirJsonResult<Vec<PhirJsonOp>> {
        let mut ops = Vec::new();

        if let Some(ref init) = binding.value {
            // Allocation declarations are collected after converting the initializer.
            if let Expr::Call(call) = init
                && self.get_callee_name(call) == Some("qalloc".to_string())
            {
                self.register_bindings.remove(&binding.name);
                return Ok(vec![]);
            }

            // Check for measurement call (mz(...) [targets])
            if let Expr::Call(call) = init
                && let Some(name) = self.get_callee_name(call)
                && (name == "mz" || name == "mx" || name == "my")
            {
                let qubits = self.extract_qubits_from_args(&call.args)?;
                let name = self.define_register(&binding.name, qubits.len());
                let results: Vec<(String, usize)> = qubits
                    .iter()
                    .enumerate()
                    .map(|(i, _)| (name.clone(), i))
                    .collect();
                ops.push(PhirJsonOp::Qop(PhirJsonQop::measure(qubits, results)));
                return Ok(ops);
            }

            // Check for measurement expression (mz(T) targets)
            if let Expr::Measure(measure_expr) = init {
                let qubits = self.extract_qubits_from_target(&measure_expr.targets)?;
                let name = self.define_register(&binding.name, qubits.len());
                let results: Vec<(String, usize)> = qubits
                    .iter()
                    .enumerate()
                    .map(|(i, _)| (name.clone(), i))
                    .collect();
                ops.push(PhirJsonOp::Qop(PhirJsonQop::measure(qubits, results)));
                return Ok(ops);
            }

            // Try to convert to a value for assignment - skip unsupported expressions
            match self.convert_expr_to_value(init) {
                Ok(value) => {
                    let name = self.define_register(&binding.name, 1);
                    ops.push(PhirJsonOp::Cop(PhirJsonCop::assign(
                        value,
                        serde_json::Value::String(name),
                    )));
                    return Ok(ops);
                }
                Err(PhirJsonError::UnsupportedExpression) => {
                    // Skip unsupported expressions silently - they may be quantum ops
                }
                Err(e) => return Err(e),
            }
        }

        self.register_bindings.remove(&binding.name);
        Ok(ops)
    }

    fn define_register(&mut self, source_name: &str, size: usize) -> String {
        let name = if self.registers.contains_key(source_name) {
            format!("{}#{}", source_name, self.registers.len())
        } else {
            source_name.to_string()
        };
        self.registers.insert(
            name.clone(),
            RegisterInfo {
                name: name.clone(),
                size,
            },
        );
        self.register_bindings
            .insert(source_name.to_string(), name.clone());
        name
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
        if gate_expr.kind == GateKind::CRZ {
            let theta = self.eval_angle_turns(
                gate_expr
                    .params
                    .first()
                    .ok_or(PhirJsonError::InvalidAngle)?,
            )? * std::f64::consts::TAU;
            let qubits = self.extract_qubits_from_target(&gate_expr.target)?;
            if qubits.len() != 2 {
                return Err(PhirJsonError::WrongArgumentCount {
                    gate: "CRZ".to_string(),
                    expected: 2,
                    got: qubits.len(),
                });
            }
            let pair = (qubits[0].clone(), qubits[1].clone());
            return Ok(vec![
                PhirJsonOp::Qop(PhirJsonQop::two_qubit_rotation(
                    "RZZ",
                    -theta / 2.0,
                    "rad",
                    vec![pair],
                )),
                PhirJsonOp::Qop(PhirJsonQop::rotation(
                    "RZ",
                    theta / 2.0,
                    "rad",
                    vec![qubits[1].clone()],
                )),
            ]);
        }
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
                let angle = self.eval_angle_turns(
                    gate_expr
                        .params
                        .first()
                        .ok_or(PhirJsonError::InvalidAngle)?,
                )?;
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
        match target {
            Expr::Index(index) => {
                let Expr::Ident(ident) = &index.object else {
                    return Err(PhirJsonError::UnsupportedExpression);
                };
                Ok(vec![self.resolve_qubit(&ident.name, &index.index)?])
            }
            Expr::SlotRef(slot) => Ok(vec![self.resolve_qubit(&slot.allocator, &slot.index)?]),
            Expr::Tuple(tuple) => self.extract_qubits_from_args(&tuple.elements),
            Expr::BracketArray(array) => self.extract_qubits_from_args(&array.elements),
            Expr::Set(set) => self.extract_qubits_from_args(&set.elements),
            Expr::Unary(unary) if matches!(unary.op, UnaryOp::AddrOf) => {
                self.extract_qubits_from_target(&unary.operand)
            }
            Expr::Ident(ident) => {
                let alloc = self.allocators.get(&ident.name).ok_or_else(|| {
                    PhirJsonError::UndefinedAllocator {
                        name: ident.name.clone(),
                    }
                })?;
                Ok((0..alloc.capacity)
                    .map(|index| (alloc.name.clone(), index))
                    .collect())
            }
            _ => Err(PhirJsonError::UnsupportedExpression),
        }
    }

    fn resolve_qubit(&self, name: &str, index: &Expr) -> PhirJsonResult<(String, usize)> {
        let index = self.eval_index(index)?;
        // PHIR retains symbolic slot names for child bindings, as before loop
        // unrolling. Only allocations owned here need a generated scoped name.
        if let Some(alloc) = self.allocators.get(name) {
            if index >= alloc.capacity {
                return Err(PhirJsonError::QubitIndexOutOfBounds {
                    allocator: name.to_string(),
                    index,
                    capacity: alloc.capacity,
                });
            }
            return Ok((alloc.name.clone(), index));
        }
        Ok((name.to_string(), index))
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
        if gate_op.kind == GateKind::CRZ {
            let theta = self
                .eval_angle_turns(gate_op.params.first().ok_or(PhirJsonError::InvalidAngle)?)?
                * std::f64::consts::TAU;
            let qubits = self.convert_slot_refs(&gate_op.targets)?;
            if qubits.len() != 2 {
                return Err(PhirJsonError::WrongArgumentCount {
                    gate: "CRZ".to_string(),
                    expected: 2,
                    got: qubits.len(),
                });
            }
            let pair = (qubits[0].clone(), qubits[1].clone());
            return Ok(vec![
                PhirJsonOp::Qop(PhirJsonQop::two_qubit_rotation(
                    "RZZ",
                    -theta / 2.0,
                    "rad",
                    vec![pair],
                )),
                PhirJsonOp::Qop(PhirJsonQop::rotation(
                    "RZ",
                    theta / 2.0,
                    "rad",
                    vec![qubits[1].clone()],
                )),
            ]);
        }
        let gate_info = get_gate_info(gate_op.kind);

        // Handle prepare operations
        if gate_info.phir_name == "Init" {
            let qubits = self.convert_slot_refs(&gate_op.targets)?;
            return Ok(vec![PhirJsonOp::Qop(PhirJsonQop::init(qubits))]);
        }

        // Handle measurement operations
        if gate_info.phir_name == "Measure" {
            let qubits = self.convert_slot_refs(&gate_op.targets)?;
            let reg_name = format!("m{}", qubits.len());
            let results: Vec<(String, usize)> = qubits
                .iter()
                .enumerate()
                .map(|(i, _)| (reg_name.clone(), i))
                .collect();
            return Ok(vec![PhirJsonOp::Qop(PhirJsonQop::measure(qubits, results))]);
        }

        let qubits = self.convert_slot_refs(&gate_op.targets)?;

        if gate_info.num_qubits == 1 {
            if gate_info.num_angles > 0 {
                // Rotation gate - get angle from params
                let angle = self
                    .eval_angle_turns(gate_op.params.first().ok_or(PhirJsonError::InvalidAngle)?)?;
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
        let qubits = self.convert_slot_refs(&measure_op.targets)?;
        let results: Vec<(String, usize)> = measure_op
            .results
            .iter()
            .map(|br| {
                self.eval_index(&br.index).map(|idx| {
                    (
                        self.register_bindings
                            .get(&br.register)
                            .unwrap_or(&br.register)
                            .clone(),
                        idx,
                    )
                })
            })
            .collect::<PhirJsonResult<_>>()?;

        Ok(vec![PhirJsonOp::Qop(PhirJsonQop::measure(qubits, results))])
    }

    fn convert_if(&mut self, if_stmt: &IfStmt) -> PhirJsonResult<Vec<PhirJsonOp>> {
        let known = self
            .angle_evaluator
            .borrow_mut()
            .eval_expr(&if_stmt.condition)
            .ok()
            .and_then(|value| value.as_bool());
        if let Some(condition) = known {
            return if condition {
                self.convert_block(&if_stmt.then_body)
            } else {
                match &if_stmt.else_body {
                    Some(ElseBranch::Else(block)) => self.convert_block(block),
                    Some(ElseBranch::ElseIf(nested)) => self.convert_if(nested),
                    None => Ok(Vec::new()),
                }
            };
        }
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
        if let Some(statement) = super::block_control(&for_stmt.body) {
            return Err(PhirJsonError::UnsupportedStatement(format!(
                "{statement} in an unrolled loop"
            )));
        }
        let ForRange::Range { start, end } = &for_stmt.range else {
            return Err(PhirJsonError::UnsupportedStatement(
                "collection for loop".to_string(),
            ));
        };
        let [capture] = for_stmt.captures.as_slice() else {
            return Err(PhirJsonError::UnsupportedStatement(
                "range for loop requires exactly one capture".to_string(),
            ));
        };
        let (Some(start), Some(end)) = (self.try_eval_const(start), self.try_eval_const(end))
        else {
            return Err(PhirJsonError::UnsupportedStatement(
                "for loop bounds must be compile-time integers".to_string(),
            ));
        };
        let mut ops = Vec::new();
        for value in start..end {
            self.angle_evaluator.borrow_mut().context.push_scope();
            self.angle_evaluator
                .borrow_mut()
                .context
                .define(capture, ComptimeValue::Int(value));
            let allocators = self.allocators.clone();
            let register_bindings = self.register_bindings.clone();
            self.allocators.remove(capture);
            self.register_bindings.remove(capture);
            self.unrolled_loop_depth += 1;
            let result = self.convert_block(&for_stmt.body);
            self.unrolled_loop_depth -= 1;
            self.allocators = allocators;
            self.register_bindings = register_bindings;
            self.angle_evaluator.borrow_mut().context.pop_scope();
            ops.extend(result?);
        }
        Ok(ops)
    }

    fn convert_tick(&mut self, tick_stmt: &TickStmt) -> PhirJsonResult<Vec<PhirJsonOp>> {
        // A tick groups operations without introducing a lexical scope.
        self.tick_depth += 1;
        let result = (|| {
            let mut qops = Vec::new();
            for stmt in &tick_stmt.body {
                let converted = self.convert_stmt(stmt)?;
                for op in converted {
                    if matches!(op, PhirJsonOp::Qop(_)) {
                        qops.push(op);
                    } else {
                        return Err(PhirJsonError::UnsupportedStatement(
                            "non-quantum operation in a tick block".to_string(),
                        ));
                    }
                }
            }

            if qops.is_empty() {
                Ok(vec![])
            } else {
                Ok(vec![PhirJsonOp::Block(PhirJsonBlock::qparallel(qops))])
            }
        })();
        self.tick_depth -= 1;
        result
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

    fn convert_slot_refs(
        &self,
        targets: &[crate::ast::SlotRef],
    ) -> PhirJsonResult<Vec<(String, usize)>> {
        targets
            .iter()
            .map(|slot| self.resolve_qubit(&slot.allocator, &slot.index))
            .collect()
    }

    fn extract_qubits_from_args(&self, args: &[Expr]) -> PhirJsonResult<Vec<(String, usize)>> {
        let mut qubits = Vec::new();
        for arg in args {
            qubits.extend(self.extract_qubits_from_target(arg)?);
        }
        Ok(qubits)
    }

    fn eval_capacity(&self, expr: &Expr) -> PhirJsonResult<usize> {
        self.angle_evaluator
            .borrow_mut()
            .eval_expr(expr)
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(PhirJsonError::InvalidAllocationCapacity)
    }

    fn eval_index(&self, expr: &Expr) -> PhirJsonResult<usize> {
        self.angle_evaluator
            .borrow_mut()
            .eval_expr(expr)
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(PhirJsonError::NonConstantIndex)
    }

    fn eval_angle_turns(&self, expr: &Expr) -> PhirJsonResult<f64> {
        resolve_angle_turns(&mut self.angle_evaluator.borrow_mut(), expr).map_err(|error| {
            PhirJsonError::RuntimeAngle {
                expression: angle_expression_name(expr),
                reason: error.to_string(),
            }
        })
    }

    fn try_eval_const(&self, expr: &Expr) -> Option<i64> {
        match self.angle_evaluator.borrow_mut().eval_expr(expr) {
            Ok(ComptimeValue::Int(value)) => Some(value),
            Ok(ComptimeValue::Uint(value)) => i64::try_from(value).ok(),
            _ => None,
        }
    }

    fn classical_name(&self, source_name: &str) -> PhirJsonResult<&str> {
        self.register_bindings
            .get(source_name)
            .map(String::as_str)
            .ok_or_else(|| PhirJsonError::UndefinedClassicalVariable {
                name: source_name.to_string(),
            })
    }

    fn convert_expr_to_value(&self, expr: &Expr) -> PhirJsonResult<serde_json::Value> {
        match self.angle_evaluator.borrow_mut().eval_expr(expr) {
            Ok(ComptimeValue::Int(value)) => return Ok(serde_json::json!(value)),
            Ok(ComptimeValue::Uint(value)) => return Ok(serde_json::json!(value)),
            Ok(ComptimeValue::Bool(value)) => return Ok(serde_json::json!(u8::from(value))),
            Ok(value @ (ComptimeValue::Float(_) | ComptimeValue::Rational(_))) => {
                return Ok(serde_json::json!(value.as_float()));
            }
            _ => {}
        }
        match expr {
            Expr::IntLit(IntLit { value, .. }) => {
                Ok(serde_json::Value::Number((*value as i64).into()))
            }
            Expr::FloatLit(fl) => Ok(serde_json::json!(fl.value)),
            Expr::BoolLit(bl) => Ok(serde_json::Value::Number(
                if bl.value { 1 } else { 0 }.into(),
            )),
            Expr::Ident(ident) => Ok(serde_json::Value::String(
                self.classical_name(&ident.name)?.to_string(),
            )),
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
                if let Expr::Ident(ident) = &idx.object {
                    return Ok(serde_json::json!([
                        self.classical_name(&ident.name)?,
                        self.eval_index(&idx.index)?
                    ]));
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
    fn test_round4_transfer_containers() {
        for body in [
            "if true { h q[0]; } else if (blk: { return unit; false }) { h q[0]; }",
            "switch (i) { 0 => (blk: { return unit; 0 }), else => 0, }",
            "switch ((blk: { return unit; 0 })) { 0 => 0, else => 0, }",
            "try! { if false { return unit; } }",
            "defer { if false { return unit; } }",
            "errdefer { if false { return unit; } }",
            "a := try! { return unit; };",
            "a := try! { 0 } catch |e| (blk: { return unit; 0 });",
            "for j in (blk: { return unit; [0] }) { h q[0]; }",
            "tick { if false { return unit; } }",
            "a := [(blk: { return unit; 0 })];",
            "a := 1 + (blk: { return unit; 0 });",
            "a: [1:(blk: { return unit; 0 })]u1 = undefined;",
        ] {
            let source = format!(
                "pub fn main() -> unit {{ mut q := qalloc(1); for i in 0..1 {{ {body} }} return unit; }}"
            );
            let error = PhirJsonCodegen::new()
                .compile(&crate::parse(&source).unwrap())
                .expect_err("nested transfer must be rejected");
            assert!(error.to_string().contains("return"), "{body}: {error}");
        }
    }

    // Compatibility: only the loop's return boundary is forbidden.
    #[test]
    fn test_round4_compat_terminal_return() {
        let source =
            "pub fn main() -> unit { mut q := qalloc(1); for i in 0..1 { h q[0]; } return unit; }";
        PhirJsonCodegen::new()
            .compile(&crate::parse(source).unwrap())
            .unwrap();
    }

    #[test]
    fn test_round4_elseif_condition() {
        let source = "pub fn main() -> unit { mut q := qalloc(2); pz q; for i in 0..3 { if false { h q[0]; } else if (blk: { return unit; true }) { x q[0]; } } return unit; }";
        let error = PhirJsonCodegen::new()
            .compile(&crate::parse(source).unwrap())
            .expect_err("transfer in loop syntax must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    #[test]
    fn test_round4_nested_trailing() {
        let source = "pub fn main() -> unit { mut q := qalloc(2); pz q; for i in 0..3 { a := blk: { mut n := false; v := blk: { for j in 0..1 { (blk: { n = true; unit }) } unit }; if n { return unit; } 0.125 }; rx(a turns) q[0]; } return unit; }";
        let error = PhirJsonCodegen::new()
            .compile(&crate::parse(source).unwrap())
            .expect_err("transfer in loop syntax must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    #[test]
    fn test_round4_zero_iterations() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..0 { return unit; } return unit; }";
        let error = PhirJsonCodegen::new()
            .compile(&crate::parse(source).unwrap())
            .expect_err("transfer in loop syntax must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    #[test]
    fn test_round4_dead_break() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..1 { if false { break; } } return unit; }";
        let error = PhirJsonCodegen::new()
            .compile(&crate::parse(source).unwrap())
            .expect_err("transfer in loop syntax must be rejected");
        assert!(error.to_string().contains("break"), "{error}");
    }
    #[test]
    fn test_round4_dead_continue() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..1 { if false { continue; } } return unit; }";
        let error = PhirJsonCodegen::new()
            .compile(&crate::parse(source).unwrap())
            .expect_err("transfer in loop syntax must be rejected");
        assert!(error.to_string().contains("continue"), "{error}");
    }
    #[test]
    fn test_round4_numeric_eq() {
        for condition in ["1/2 == 0.5", "1 == 1.0", "1 == 1u64"] {
            let source = format!(
                "pub fn main() -> unit {{ mut q := qalloc(1); if {condition} {{ x q[0]; }} else {{ h q[0]; }} return unit; }}"
            );
            let output = PhirJsonCodegen::new()
                .compile(&crate::parse(&source).unwrap())
                .unwrap();
            let gates: Vec<_> = output
                .ops
                .iter()
                .filter_map(|op| match op {
                    PhirJsonOp::Qop(gate) => Some(gate.qop.as_str()),
                    _ => None,
                })
                .collect();
            assert_eq!(gates, ["X"], "{condition}");
        }
    }
    #[test]
    fn test_round4_numeric_ne() {
        for condition in ["1/2 != 0.5", "1 != 1.0", "1 != 1u64"] {
            let source = format!(
                "pub fn main() -> unit {{ mut q := qalloc(1); if {condition} {{ h q[0]; }} else {{ x q[0]; }} return unit; }}"
            );
            let output = PhirJsonCodegen::new()
                .compile(&crate::parse(&source).unwrap())
                .unwrap();
            let gates: Vec<_> = output
                .ops
                .iter()
                .filter_map(|op| match op {
                    PhirJsonOp::Qop(gate) => Some(gate.qop.as_str()),
                    _ => None,
                })
                .collect();
            assert_eq!(gates, ["X"], "{condition}");
        }
    }
    #[test]
    fn test_round4_numeric_lt() {
        for condition in ["1/2 < 0.75", "1 < 1.5", "1 < 2u64"] {
            let source = format!(
                "pub fn main() -> unit {{ mut q := qalloc(1); if {condition} {{ x q[0]; }} else {{ h q[0]; }} return unit; }}"
            );
            let output = PhirJsonCodegen::new()
                .compile(&crate::parse(&source).unwrap())
                .unwrap();
            let gates: Vec<_> = output
                .ops
                .iter()
                .filter_map(|op| match op {
                    PhirJsonOp::Qop(gate) => Some(gate.qop.as_str()),
                    _ => None,
                })
                .collect();
            assert_eq!(gates, ["X"], "{condition}");
        }
    }
    #[test]
    fn test_round4_numeric_le() {
        for condition in ["1/2 <= 0.5", "1 <= 1.0", "1 <= 1u64"] {
            let source = format!(
                "pub fn main() -> unit {{ mut q := qalloc(1); if {condition} {{ x q[0]; }} else {{ h q[0]; }} return unit; }}"
            );
            let output = PhirJsonCodegen::new()
                .compile(&crate::parse(&source).unwrap())
                .unwrap();
            let gates: Vec<_> = output
                .ops
                .iter()
                .filter_map(|op| match op {
                    PhirJsonOp::Qop(gate) => Some(gate.qop.as_str()),
                    _ => None,
                })
                .collect();
            assert_eq!(gates, ["X"], "{condition}");
        }
    }
    #[test]
    fn test_round4_numeric_gt() {
        for condition in ["1/2 > 0.25", "1 > 0.5", "1 > 0u64"] {
            let source = format!(
                "pub fn main() -> unit {{ mut q := qalloc(1); if {condition} {{ x q[0]; }} else {{ h q[0]; }} return unit; }}"
            );
            let output = PhirJsonCodegen::new()
                .compile(&crate::parse(&source).unwrap())
                .unwrap();
            let gates: Vec<_> = output
                .ops
                .iter()
                .filter_map(|op| match op {
                    PhirJsonOp::Qop(gate) => Some(gate.qop.as_str()),
                    _ => None,
                })
                .collect();
            assert_eq!(gates, ["X"], "{condition}");
        }
    }
    #[test]
    fn test_round4_numeric_ge() {
        for condition in ["1/2 >= 0.5", "1 >= 1.0", "1 >= 1u64"] {
            let source = format!(
                "pub fn main() -> unit {{ mut q := qalloc(1); if {condition} {{ x q[0]; }} else {{ h q[0]; }} return unit; }}"
            );
            let output = PhirJsonCodegen::new()
                .compile(&crate::parse(&source).unwrap())
                .unwrap();
            let gates: Vec<_> = output
                .ops
                .iter()
                .filter_map(|op| match op {
                    PhirJsonOp::Qop(gate) => Some(gate.qop.as_str()),
                    _ => None,
                })
                .collect();
            assert_eq!(gates, ["X"], "{condition}");
        }
    }

    #[test]
    fn test_round3_undeclared_condition_name() {
        let source =
            "pub fn main(c: bool) -> unit { mut q := qalloc(1); if c { x q[0]; } return unit; }";
        let error = PhirJsonCodegen::new()
            .compile(&crate::parse(source).unwrap())
            .expect_err("a runtime condition cannot reference an undeclared name");
        assert!(
            error.to_string().contains("classical variable 'c'"),
            "{error}"
        );
    }

    #[test]
    fn test_round3_angle_dead_branch() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..1 { rx((blk: { if false { return unit; } 0.125 }) turns) q[0]; } return unit; }";
        let error = PhirJsonCodegen::new()
            .compile(&crate::parse(source).unwrap())
            .expect_err("syntactically present return must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    #[test]
    fn test_round3_if_expression_dead_branch() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..1 { theta := if (false) { return unit; 0.25 } else { 0.125 }; rx(theta turns) q[0]; } return unit; }";
        let error = PhirJsonCodegen::new()
            .compile(&crate::parse(source).unwrap())
            .expect_err("syntactically present return must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    #[test]
    fn test_round3_nested_capture_dead_branch() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..1 { theta := blk: { for j in i..i + 2 { if j > 8 { return unit; } } 0.125 }; rx(theta turns) q[0]; } return unit; }";
        let error = PhirJsonCodegen::new()
            .compile(&crate::parse(source).unwrap())
            .expect_err("syntactically present return must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    // Compatibility: a possibly reached transfer must still be rejected.
    #[test]
    fn test_round3_compat_unknown_path() {
        let source = "pub fn main(c: bool) -> unit { mut q := qalloc(1); for i in 0..1 { a := blk: { if c { return unit; } 0.125 }; h q[0]; } return unit; }";
        let error = PhirJsonCodegen::new()
            .compile(&crate::parse(source).unwrap())
            .expect_err("reachable return must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    // Compatibility: a possibly reached transfer must still be rejected.
    #[test]
    fn test_round3_compat_later_iteration() {
        let source = "pub fn main(c: bool) -> unit { mut q := qalloc(1); for i in 0..1 { a := blk: { mut n := 0; for j in 0..2 { if n == 1 { return unit; } n = n + 1; } 0.125 }; h q[0]; } return unit; }";
        let error = PhirJsonCodegen::new()
            .compile(&crate::parse(source).unwrap())
            .expect_err("reachable return must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    // Compatibility: a possibly reached transfer must still be rejected.
    #[test]
    fn test_round3_compat_unknown_assignment() {
        let source = "pub fn main(c: bool) -> unit { mut q := qalloc(1); for i in 0..1 { a := blk: { mut n := 0; if c { n = 1; } if n == 1 { return unit; } 0.125 }; h q[0]; } return unit; }";
        let error = PhirJsonCodegen::new()
            .compile(&crate::parse(source).unwrap())
            .expect_err("reachable return must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    // Compatibility: a nested function has its own return boundary.
    #[test]
    fn test_round3_compat_function_boundary() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..1 { get_n := fn() -> i64 { return 1; }; n := get_n(); h q[0]; } return unit; }";
        PhirJsonCodegen::new()
            .compile(&crate::parse(source).unwrap())
            .unwrap();
    }

    #[test]
    fn test_round3_trailing_return() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); pz q; for i in 0..3 { rx(0.125 turns) q[0]; (blk: { return unit; }) } return unit; }";
        let error = PhirJsonCodegen::new()
            .compile(&crate::parse(source).unwrap())
            .expect_err("trailing return must fail loudly");
        assert!(error.to_string().contains("return"), "{error}");
    }
    #[test]
    fn test_round3_trailing_break() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); pz q; for i in 0..3 { rx(0.125 turns) q[0]; (blk: { break; }) } return unit; }";
        let error = PhirJsonCodegen::new()
            .compile(&crate::parse(source).unwrap())
            .expect_err("trailing break must fail loudly");
        assert!(error.to_string().contains("break"), "{error}");
    }
    #[test]
    fn test_round3_trailing_continue() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); pz q; for i in 0..3 { rx(0.125 turns) q[0]; (blk: { continue; }) } return unit; }";
        let error = PhirJsonCodegen::new()
            .compile(&crate::parse(source).unwrap())
            .expect_err("trailing continue must fail loudly");
        assert!(error.to_string().contains("continue"), "{error}");
    }
    #[test]
    fn test_round3_dead_branch() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..1 { theta := blk: { if false { return unit; } 0.125 }; rx(theta turns) q[0]; } return unit; }";
        let error = PhirJsonCodegen::new()
            .compile(&crate::parse(source).unwrap())
            .expect_err("syntactically present return must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    #[test]
    fn test_round3_empty_range() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..1 { theta := blk: { for j in 0..0 { return unit; } 0.125 }; rx(theta turns) q[0]; } return unit; }";
        let error = PhirJsonCodegen::new()
            .compile(&crate::parse(source).unwrap())
            .expect_err("syntactically present return must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    #[test]
    fn test_round3_capture_branch() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..1 { theta := blk: { n := i + 1; if n == 9 { return unit; } 0.125 }; rx(theta turns) q[0]; } return unit; }";
        let error = PhirJsonCodegen::new()
            .compile(&crate::parse(source).unwrap())
            .expect_err("syntactically present return must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    #[test]
    fn test_round3_capture_range() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..1 { theta := blk: { for j in i..i { return unit; } 0.125 }; rx(theta turns) q[0]; } return unit; }";
        let error = PhirJsonCodegen::new()
            .compile(&crate::parse(source).unwrap())
            .expect_err("syntactically present return must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    #[test]
    fn test_round3_child_references() {
        let source = "pub fn main() -> unit { mut q := qalloc(2); data := q.child(1); ancilla := q.child(1); pz q; cx (data[0], ancilla[0]); return unit; }";
        let output = PhirJsonCodegen::new()
            .compile(&crate::parse(source).unwrap())
            .unwrap();
        let gate = output
            .ops
            .iter()
            .find_map(|op| match op {
                PhirJsonOp::Qop(gate) if gate.qop == "CX" => Some(gate),
                _ => None,
            })
            .unwrap();
        assert_eq!(
            serde_json::to_value(gate).unwrap(),
            serde_json::json!({"qop": "CX", "args": [[["data", 0], ["ancilla", 0]]]})
        );
    }

    #[test]
    fn test_round3_runtime_float_binding() {
        for binding in ["cutoff := 0.5;", "tick { cutoff := 0.5; }"] {
            let source = format!(
                "pub fn main() -> unit {{ mut q := qalloc(2); pz q; {binding} mut m := mz(u1) q[0]; if m > cutoff {{ x q[1]; }} return unit; }}"
            );
            let output = PhirJsonCodegen::new()
                .compile(&crate::parse(&source).unwrap())
                .unwrap();
            let assignments: Vec<_> = output
                .ops
                .iter()
                .filter_map(|op| match op {
                    PhirJsonOp::Cop(op) => Some(serde_json::to_value(op).unwrap()),
                    _ => None,
                })
                .collect();
            if binding.starts_with("tick") {
                assert!(assignments.is_empty());
            } else {
                assert_eq!(
                    assignments,
                    vec![serde_json::json!({"cop": "=", "args": [0.5], "returns": ["cutoff"]})]
                );
            }
            let condition = output
                .ops
                .iter()
                .find_map(|op| match op {
                    PhirJsonOp::Block(block) if block.block == "if" => {
                        Some(serde_json::to_value(block).unwrap())
                    }
                    _ => None,
                })
                .unwrap();
            assert_eq!(
                condition["condition"],
                serde_json::json!({"cop": ">", "args": ["m", 0.5]})
            );
        }
    }

    // The conservative rule rejects unreachable transfers inside loop bodies.
    #[test]
    fn test_review_unreachable_and_terminal_returns() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..3 { if i == 9 { return unit; } h q[0]; } for j in 1..1 { return unit; } return unit; }";
        let error = PhirJsonCodegen::new()
            .compile(&crate::parse(source).unwrap())
            .expect_err("syntactically present return must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }

    #[test]
    fn test_review_loop_control_expression() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..3 { a := blk: { return unit; }; } return unit; }".to_string();
        let error = PhirJsonCodegen::new()
            .compile(&crate::parse(&source).unwrap())
            .expect_err("control transfer in loop must fail loudly");
        assert!(error.to_string().contains("return"), "{error}");
    }

    #[test]
    fn test_review_loop_control_switch() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..3 { switch (i) { 0 => 0, else => 1, } } return unit; }".to_string();
        let error = PhirJsonCodegen::new()
            .compile(&crate::parse(&source).unwrap())
            .expect_err("control transfer in loop must fail loudly");
        assert!(error.to_string().contains("switch"), "{error}");
    }

    #[test]
    fn test_review_loop_control_try_block() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..3 { try! { return unit; } } return unit; }".to_string();
        let error = PhirJsonCodegen::new()
            .compile(&crate::parse(&source).unwrap())
            .expect_err("control transfer in loop must fail loudly");
        assert!(error.to_string().contains("return"), "{error}");
    }

    #[test]
    fn test_review_loop_control_defer() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..3 { defer { return unit; } } return unit; }".to_string();
        let error = PhirJsonCodegen::new()
            .compile(&crate::parse(&source).unwrap())
            .expect_err("control transfer in loop must fail loudly");
        assert!(error.to_string().contains("return"), "{error}");
    }

    #[test]
    fn test_review_loop_control_propagation() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..3 { a := try missing; } return unit; }".to_string();
        let error = PhirJsonCodegen::new()
            .compile(&crate::parse(&source).unwrap())
            .expect_err("control transfer in loop must fail loudly");
        assert!(error.to_string().contains("try"), "{error}");
    }
    #[test]
    fn test_review_failed_comptime_scope() {
        let source = "pub fn main(c: bool) -> unit { mut q := qalloc(1); pz q; n := 0.125; a := blk: { n := 0.25; c }; rz(n turns) q[0]; return unit; }";
        let program = crate::parse(source).unwrap();
        let phir = PhirJsonCodegen::new().compile(&program).unwrap();
        let angle = phir
            .ops
            .iter()
            .find_map(|op| match op {
                PhirJsonOp::Qop(gate) if gate.qop == "RZ" => {
                    Some(gate.angles.as_ref().unwrap().0[0])
                }
                _ => None,
            })
            .unwrap();
        assert_eq!(angle, std::f64::consts::FRAC_PI_4);
    }

    #[test]
    fn test_review_loop_return_direct() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); pz q; for i in 0..3 { rx(0.125 turns) q[0]; return unit; } return unit; }".to_string();
        let error = PhirJsonCodegen::new()
            .compile(&crate::parse(&source).unwrap())
            .expect_err("return in unrolled loop must fail loudly");
        assert!(error.to_string().contains("return"), "{error}");
    }

    #[test]
    fn test_review_loop_return_block() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); pz q; for i in 0..3 { rx(0.125 turns) q[0]; { return unit; } } return unit; }".to_string();
        let error = PhirJsonCodegen::new()
            .compile(&crate::parse(&source).unwrap())
            .expect_err("return in unrolled loop must fail loudly");
        assert!(error.to_string().contains("return"), "{error}");
    }

    #[test]
    fn test_review_loop_return_if() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); pz q; for i in 0..3 { rx(0.125 turns) q[0]; if i == 0 { return unit; } } return unit; }".to_string();
        let error = PhirJsonCodegen::new()
            .compile(&crate::parse(&source).unwrap())
            .expect_err("return in unrolled loop must fail loudly");
        assert!(error.to_string().contains("return"), "{error}");
    }

    #[test]
    fn test_review_loop_return_tick() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); pz q; for i in 0..3 { rx(0.125 turns) q[0]; tick { return unit; } } return unit; }".to_string();
        let error = PhirJsonCodegen::new()
            .compile(&crate::parse(&source).unwrap())
            .expect_err("return in unrolled loop must fail loudly");
        assert!(error.to_string().contains("return"), "{error}");
    }

    #[test]
    fn test_review_loop_return_nested_loop() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); pz q; for i in 0..3 { rx(0.125 turns) q[0]; for j in 0..1 { return unit; } } return unit; }".to_string();
        let error = PhirJsonCodegen::new()
            .compile(&crate::parse(&source).unwrap())
            .expect_err("return in unrolled loop must fail loudly");
        assert!(error.to_string().contains("return"), "{error}");
    }

    #[test]
    fn test_review_tick_binding_visibility() {
        let source = "pub fn main() -> unit { tick { mut q := qalloc(1); theta := 0.25; pz q; } rz(theta turns) q[0]; return unit; }";
        let phir = PhirJsonCodegen::new()
            .compile(&crate::parse(source).unwrap())
            .unwrap();
        let gate = phir
            .ops
            .iter()
            .find_map(|op| match op {
                PhirJsonOp::Qop(gate) if gate.qop == "RZ" => Some(gate),
                _ => None,
            })
            .unwrap();
        assert_eq!(
            gate.angles.as_ref().unwrap().0,
            vec![std::f64::consts::FRAC_PI_2]
        );
    }

    #[test]
    fn test_review_tick_comptime_binding() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); pz q; tick { theta := 0.25; rz(theta turns) q[0]; } return unit; }";
        let phir = PhirJsonCodegen::new()
            .compile(&crate::parse(source).unwrap())
            .unwrap();
        assert!(
            !phir
                .ops
                .iter()
                .any(|op| matches!(op, PhirJsonOp::Cop(_) | PhirJsonOp::CvarDefine(_)))
        );
        let block = phir
            .ops
            .iter()
            .find_map(|op| {
                if let PhirJsonOp::Block(block) = op {
                    Some(block)
                } else {
                    None
                }
            })
            .unwrap();
        assert_eq!(block.block, "qparallel");
        let [PhirJsonOp::Qop(gate)] = block.ops.as_ref().unwrap().as_slice() else {
            panic!("expected single rotation");
        };
        assert_eq!(gate.qop, "RZ");
        assert_eq!(
            gate.angles.as_ref().unwrap().0,
            vec![std::f64::consts::FRAC_PI_2]
        );
    }

    #[test]
    fn test_review_capacity_diagnostic() {
        let source = "pub fn main(n: u32) -> unit { mut q := qalloc(n); return unit; }";
        let error = PhirJsonCodegen::new()
            .compile(&crate::parse(source).unwrap())
            .unwrap_err();
        assert!(error.to_string().contains("allocation capacity"), "{error}");
    }
    #[test]
    fn test_control_flow_local_allocations_and_measurement_scopes() {
        let program = crate::parse(
            "pub fn main() -> unit {
            mut q := qalloc(1); mut c := mz(u1) q[0];
            for i in 0..2 { mut q := qalloc(i + 1); h q[i]; mut c := mz(u1) q[i]; if c { x q[i]; } }
            for i in 0..1 { mut c := 0; }
            if c { h q[0]; }
        }",
        )
        .unwrap();
        let phir = PhirJsonCodegen::new().compile(&program).unwrap();
        let gates: Vec<_> = phir
            .ops
            .iter()
            .filter_map(|op| {
                if let PhirJsonOp::Qop(gate) = op {
                    Some(serde_json::to_value(gate).unwrap())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(
            gates[1],
            serde_json::json!({"qop": "H", "args": [["q#1", 0]]})
        );
        assert_eq!(
            gates[3],
            serde_json::json!({"qop": "H", "args": [["q#2", 1]]})
        );
        let assignment = phir
            .ops
            .iter()
            .find_map(|op| {
                if let PhirJsonOp::Cop(cop) = op {
                    Some(serde_json::to_value(cop).unwrap())
                } else {
                    None
                }
            })
            .unwrap();
        assert_eq!(assignment["returns"], serde_json::json!(["c#3"]));
        let conditions: Vec<_> = phir
            .ops
            .iter()
            .filter_map(|op| {
                if let PhirJsonOp::Block(block) = op {
                    block.condition.as_ref()
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(
            conditions,
            vec![
                &serde_json::json!("c#1"),
                &serde_json::json!("c#2"),
                &serde_json::json!("c")
            ]
        );
    }

    #[test]
    fn test_control_flow_nested_bounds_and_selected_branch() {
        let program = crate::parse(
            "n := 3; pub fn main() -> unit {
            mut q := qalloc(n);
            for n in 0..n { for j in n..n + 1 { if j == 1 { h q[j]; } } }
            h q[n - 1];
        }",
        )
        .unwrap();
        let phir = PhirJsonCodegen::new().compile(&program).unwrap();
        let gates: Vec<_> = phir
            .ops
            .iter()
            .filter_map(|op| {
                if let PhirJsonOp::Qop(gate) = op {
                    Some(serde_json::to_value(gate).unwrap())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(
            gates,
            vec![
                serde_json::json!({"qop": "H", "args": [["q", 1]]}),
                serde_json::json!({"qop": "H", "args": [["q", 2]]}),
            ]
        );
    }

    #[test]
    fn test_control_flow_unresolved_targets_fail() {
        for target in ["q[i]", "(q[0], q[i])", "[q[0], q[i]]", "q[0 - 1]", "q[8]"] {
            let source =
                format!("pub fn main(i: int) -> unit {{ mut q := qalloc(2); h {target}; }}");
            let program = crate::parse(&source).unwrap();
            assert!(
                PhirJsonCodegen::new().compile(&program).is_err(),
                "{target}"
            );
        }
    }

    #[test]
    fn test_control_flow_builder_target_and_call_reject_unknown_index() {
        let program = crate::parse("pub fn main() -> unit { mut q := qalloc(2); }").unwrap();
        let mut codegen = PhirJsonCodegen::new();
        let TopLevelDecl::Fn(function) = &program.declarations[0] else {
            panic!("expected function")
        };
        let Stmt::Binding(binding) = &function.body.statements[0] else {
            panic!("expected binding")
        };
        codegen.collect_binding(binding).unwrap();
        let parsed = crate::parse("pub fn main(i: int) -> unit { h q[i]; }").unwrap();
        let TopLevelDecl::Fn(function) = &parsed.declarations[0] else {
            panic!("expected function")
        };
        let Stmt::Expr(stmt) = &function.body.statements[0] else {
            panic!("expected expression")
        };
        let Expr::Gate(gate) = &stmt.expr else {
            panic!("expected gate")
        };
        let Expr::Index(index) = &gate.target else {
            panic!("expected index")
        };
        let slot = crate::ast::SlotRef {
            allocator: "q".to_string(),
            index: Box::new(index.index.clone()),
            location: None,
        };
        assert!(matches!(
            codegen.convert_slot_refs(&[slot]),
            Err(PhirJsonError::NonConstantIndex)
        ));
        assert!(matches!(
            codegen.extract_qubits_from_args(std::slice::from_ref(&gate.target)),
            Err(PhirJsonError::NonConstantIndex)
        ));
    }

    #[test]
    fn test_control_flow_indexed_target() {
        let ast = crate::parse(
            "pub fn main() -> unit {
            mut q := qalloc(3);
            for i in 0..3 { h q[i]; }
        }",
        )
        .unwrap();
        let phir = PhirJsonCodegen::new().compile(&ast).unwrap();
        let gates: Vec<_> = phir
            .ops
            .iter()
            .filter_map(|op| {
                if let PhirJsonOp::Qop(gate) = op {
                    Some(serde_json::to_value(gate).unwrap())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(
            gates,
            vec![
                serde_json::json!({"qop": "H", "args": [["q", 0]]}),
                serde_json::json!({"qop": "H", "args": [["q", 1]]}),
                serde_json::json!({"qop": "H", "args": [["q", 2]]}),
            ]
        );
    }

    #[test]
    fn empty_quantum_allocator_is_not_declared() {
        let ast = crate::parse("pub fn main() -> unit { q := qalloc(0); return unit; }").unwrap();
        assert!(PhirJsonCodegen::new().compile(&ast).unwrap().ops.is_empty());
        let TopLevelDecl::Fn(function) = &ast.declarations[0] else {
            panic!("expected main function");
        };
        assert!(
            PhirJsonCodegen::new()
                .compile_function(function)
                .unwrap()
                .ops
                .is_empty()
        );
    }

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
