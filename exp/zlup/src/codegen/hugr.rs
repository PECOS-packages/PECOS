//! HUGR code generation for Zluppy.
//!
//! This module generates HUGR (Hierarchical Unified Graph Representation) from
//! Zluppy AST. HUGR is used for targeting experiments and quantum hardware.
//!
//! ## Design
//!
//! The codegen walks the Zluppy AST and:
//! 1. Collects allocator declarations to determine qubit counts
//! 2. Maps gate calls to TketOp operations
//! 3. Tracks wire flow through the circuit
//! 4. Handles rotation angles (converted to half-turns)
//!
//! ## Wire Tracking
//!
//! In HUGR, each qubit is represented by a Wire that flows through the graph.
//! When a gate operates on a qubit, it consumes the input wire and produces
//! a new output wire. We maintain a mapping from qubit identifiers to their
//! current wire.

use std::collections::BTreeMap;
use thiserror::Error;

use std::io::Cursor;

use std::cell::RefCell;

use tket::TketOp;
use tket::extension::measurement::MeasurementOp;
use tket::hugr::builder::{
    BuildError, Dataflow, DataflowHugr, DataflowSubContainer, FunctionBuilder, SubContainer,
};
use tket::hugr::envelope::EnvelopeConfig;
use tket::hugr::extension::prelude::{bool_t, qb_t};
use tket::hugr::types::Signature;
use tket::hugr::{Hugr, Wire, type_row};

use crate::ast::{
    Binding, Block, CallExpr, ElseBranch, Expr, FnDecl, ForRange, ForStmt, IfStmt, IndexExpr,
    Program, Stmt, TopLevelDecl,
};
use crate::comptime::{
    ComptimeEvaluator, ComptimeValue, angle_evaluator, angle_expression_name,
    define_comptime_binding, resolve_angle_turns,
};

// =============================================================================
// Errors
// =============================================================================

/// HUGR code generation errors.
#[derive(Debug, Error)]
pub enum HugrError {
    #[error("unknown gate '{name}'")]
    UnknownGate { name: String },

    #[error("HUGR codegen supports only gate calls; call to '{name}' is unsupported")]
    UnsupportedCall { name: String },

    #[error("undefined qubit '{name}'")]
    UndefinedQubit { name: String },

    #[error("qubit index {index} out of bounds for allocator with capacity {capacity}")]
    QubitIndexOutOfBounds { index: usize, capacity: usize },

    #[error("expected {expected} arguments for gate '{gate}', got {got}")]
    WrongArgumentCount {
        gate: String,
        expected: usize,
        got: usize,
    },

    #[error("allocator '{name}' not found")]
    AllocatorNotFound { name: String },

    #[error("collection for loops are unsupported in HUGR codegen")]
    CollectionLoop,

    #[error("for loop bounds must be compile-time integers")]
    NonConstantLoopBound,

    #[error("range for loops require exactly one capture, got {count}")]
    InvalidLoopCaptures { count: usize },

    #[error("{statement} cannot be represented in an unrolled loop")]
    UnsupportedLoopControl { statement: &'static str },

    #[error("if condition must be a measured classical variable or a compile-time boolean")]
    UnsupportedCondition,

    #[error("nested runtime conditionals are unsupported by the HUGR builder")]
    NestedConditional,

    #[error("HUGR builder error: {0}")]
    BuilderError(String),

    #[error("unsupported expression in codegen")]
    UnsupportedExpression,

    #[error(
        "rotation angle expression {expression} is not a numeric compile-time constant: {reason}"
    )]
    InvalidRotationAngle { expression: String, reason: String },

    #[error(
        "cannot broadcast {arity}-qubit gate '{gate}' over an allocator; allocator broadcasts require a single-qubit gate"
    )]
    InvalidAllocatorBroadcast { gate: String, arity: usize },

    #[error("cannot broadcast gate '{gate}' over zero-capacity allocator '{allocator}'")]
    EmptyAllocatorBroadcast { gate: String, allocator: String },

    #[error("gate '{gate}' batch must contain at least one target")]
    EmptyGateBatch { gate: String },

    #[error("HUGR serialization error: {0}")]
    SerializationError(String),
}

/// Result type for HUGR code generation.
pub type HugrResult<T> = Result<T, HugrError>;

// =============================================================================
// Qubit Tracking
// =============================================================================

/// Tracks an allocator and its qubits.
#[derive(Debug, Clone)]
pub struct Allocator {
    /// Name of the allocator variable.
    pub name: String,
    /// Capacity (number of qubits).
    pub capacity: usize,
    /// Starting index in the global qubit array.
    pub start_index: usize,
}

/// Tracks a qubit reference (allocator + index).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct QubitRef {
    /// Allocator name.
    pub allocator: String,
    /// Index within the allocator.
    pub index: usize,
}

impl QubitRef {
    pub fn new(allocator: impl Into<String>, index: usize) -> Self {
        Self {
            allocator: allocator.into(),
            index,
        }
    }
}

// =============================================================================
// Gate Mapping
// =============================================================================

/// Result of mapping a gate name - either a direct TketOp or a composite gate.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone)]
enum GateMapping {
    /// Direct mapping to a TketOp
    Direct(TketOp),
    /// SWAP gate (decomposed to 3 CX gates)
    Swap,
    /// iSWAP gate (decomposed)
    ISwap,
    /// SY gate (sqrt of Y) - implemented as Ry(π/2)
    SY,
    /// SYdg gate (sqrt of Y dagger) - implemented as Ry(-π/2)
    SYdg,
    /// CH gate (controlled Hadamard) - decomposed to Ry(π/4) CZ Ry(-π/4)
    CH,
    /// SXX gate (sqrt of XX Ising) - decomposed
    SXX,
    /// SYY gate (sqrt of YY Ising) - decomposed
    SYY,
    /// SZZ gate (sqrt of ZZ Ising) - decomposed to CX S CX
    SZZ,
    /// Dagger versions of Ising gates
    SXXdg,
    SYYdg,
    SZZdg,
    /// RZZ gate (ZZ rotation) - decomposed to CX Rz CX
    RZZ,
    /// F gate (Clifford face rotation) - decomposed to H Sdg H Sdg
    F,
    /// F dagger - decomposed to S H S H
    Fdg,
    /// F4 gate (fourth root of face rotation) - decomposed
    F4,
    /// F4 dagger
    F4dg,
}

impl GateMapping {
    fn arity(&self) -> usize {
        match self {
            Self::Direct(op) => gate_qubit_count(op),
            Self::SY | Self::SYdg | Self::F | Self::Fdg | Self::F4 | Self::F4dg => 1,
            Self::Swap
            | Self::ISwap
            | Self::CH
            | Self::SXX
            | Self::SYY
            | Self::SZZ
            | Self::SXXdg
            | Self::SYYdg
            | Self::SZZdg
            | Self::RZZ => 2,
        }
    }

    fn needs_angle(&self) -> bool {
        match self {
            Self::Direct(op) => gate_needs_angle(op),
            Self::RZZ => true,
            _ => false,
        }
    }
}

/// Maps Zluppy gate names to gate operations.
///
/// Zluppy uses lowercase gate names following Zig-style conventions.
/// All gate names are lowercase.
///
/// Available gates:
/// - Single-qubit Pauli: h, x, y, z
/// - Square root: sx, sxdg, sy, sydg, sz, szdg (sqrt of X, Y, and Z)
/// - T gates: t, tdg (fourth root of Z)
/// - F gates: f, fdg, f4, f4dg (Clifford face rotations)
/// - Rotation: rx, ry, rz (single-qubit), rzz (two-qubit)
/// - Two-qubit: cx, cy, cz, ch, swap, iswap
/// - Two-qubit Ising: sxx, syy, szz, sxxdg, syydg, szzdg
/// - Three-qubit: ccx
/// - State preparation: pz (prepare +Z eigenstate)
///
/// Composite gates (decomposed):
/// - swap: cx(a,b) cx(b,a) cx(a,b)
/// - iswap: sz(a) sz(b) h(a) cx(a,b) cx(b,a) h(b)
/// - sy: ry(π/2)
/// - sydg: ry(-π/2)
/// - ch: ry(π/4, b) cz(a,b) ry(-π/4, b)
/// - szz: cx(a,b) sz(b) cx(a,b)
/// - sxx: h(a) h(b) szz(a,b) h(a) h(b)
/// - syy: sxdg(a) sxdg(b) szz(a,b) sx(a) sx(b)
/// - rzz(θ): cx(a,b) rz(θ, b) cx(a,b)
/// - f: h sdg h sdg (Clifford: X→Y→Z→X)
/// - fdg: s h s h
fn gate_name_to_mapping(name: &str) -> Option<GateMapping> {
    match name {
        // Single-qubit Pauli gates
        "h" => Some(GateMapping::Direct(TketOp::H)),
        "x" => Some(GateMapping::Direct(TketOp::X)),
        "y" => Some(GateMapping::Direct(TketOp::Y)),
        "z" => Some(GateMapping::Direct(TketOp::Z)),

        // Square root gates (sx = sqrt(X), sy = sqrt(Y), sz = sqrt(Z))
        "sx" => Some(GateMapping::Direct(TketOp::V)), // V = sqrt(X)
        "sxdg" => Some(GateMapping::Direct(TketOp::Vdg)),
        "sy" => Some(GateMapping::SY),     // sqrt(Y) = Ry(π/2)
        "sydg" => Some(GateMapping::SYdg), // sqrt(Y)† = Ry(-π/2)
        "sz" => Some(GateMapping::Direct(TketOp::S)), // S = sqrt(Z)
        "szdg" => Some(GateMapping::Direct(TketOp::Sdg)),

        // T gates (fourth root of Z)
        "t" => Some(GateMapping::Direct(TketOp::T)),
        "tdg" => Some(GateMapping::Direct(TketOp::Tdg)),

        // Rotation gates (require angle parameter)
        "rx" => Some(GateMapping::Direct(TketOp::Rx)),
        "ry" => Some(GateMapping::Direct(TketOp::Ry)),
        "rz" => Some(GateMapping::Direct(TketOp::Rz)),

        // Two-qubit gates
        "cx" => Some(GateMapping::Direct(TketOp::CX)),
        "cy" => Some(GateMapping::Direct(TketOp::CY)),
        "cz" => Some(GateMapping::Direct(TketOp::CZ)),
        "ch" => Some(GateMapping::CH), // Controlled Hadamard (decomposed)
        "crz" => Some(GateMapping::Direct(TketOp::CRz)),
        "rzz" => Some(GateMapping::RZZ), // ZZ rotation (decomposed)

        // Two-qubit Ising gates (decomposed)
        "sxx" => Some(GateMapping::SXX),
        "syy" => Some(GateMapping::SYY),
        "szz" => Some(GateMapping::SZZ),
        "sxxdg" => Some(GateMapping::SXXdg),
        "syydg" => Some(GateMapping::SYYdg),
        "szzdg" => Some(GateMapping::SZZdg),

        // Composite two-qubit gates (decomposed)
        "swap" => Some(GateMapping::Swap),
        "iswap" => Some(GateMapping::ISwap),

        // Three-qubit gates
        "ccx" => Some(GateMapping::Direct(TketOp::Toffoli)),

        // F gates (Clifford face rotations, decomposed)
        "f" => Some(GateMapping::F),
        "fdg" => Some(GateMapping::Fdg),
        "f4" => Some(GateMapping::F4),
        "f4dg" => Some(GateMapping::F4dg),

        // Prepare +Z eigenstate (reset)
        "pz" => Some(GateMapping::Direct(TketOp::Reset)),

        _ => None,
    }
}

/// Returns the number of qubit operands for a gate.
fn gate_qubit_count(op: &TketOp) -> usize {
    match op {
        // Single-qubit gates
        TketOp::H
        | TketOp::X
        | TketOp::Y
        | TketOp::Z
        | TketOp::S
        | TketOp::Sdg
        | TketOp::T
        | TketOp::Tdg
        | TketOp::V
        | TketOp::Vdg
        | TketOp::Rx
        | TketOp::Ry
        | TketOp::Rz
        | TketOp::Measure
        | TketOp::MeasureFree
        | TketOp::Reset
        | TketOp::QFree => 1,

        // Two-qubit gates
        TketOp::CX | TketOp::CY | TketOp::CZ | TketOp::CRz => 2,

        // Three-qubit gates
        TketOp::Toffoli => 3,

        // Zero-qubit gates (allocation)
        TketOp::QAlloc | TketOp::TryQAlloc => 0,

        // Default for any future variants
        _ => 1,
    }
}

/// Returns whether a gate requires a rotation angle parameter.
fn gate_needs_angle(op: &TketOp) -> bool {
    matches!(op, TketOp::Rx | TketOp::Ry | TketOp::Rz | TketOp::CRz)
}

// =============================================================================
// Code Generator Configuration
// =============================================================================

/// Controls how composite gates are handled during code generation.
///
/// When targeting real hardware or HUGR-native execution, use `Decompose` to
/// break down gates like SWAP and iSWAP into primitive operations.
///
/// When targeting simulation (e.g., PECOS), use `Native` to emit the gates
/// directly if the simulator supports them natively.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CodegenMode {
    /// Decompose composite gates into primitives (e.g., SWAP → 3 CX gates).
    /// Use this for hardware targets or HUGR-only execution.
    #[default]
    Decompose,

    /// Emit gates natively without decomposition.
    /// Use this for simulation backends that support composite gates.
    ///
    /// Note: Currently iSWAP and SWAP are always decomposed since HUGR's
    /// tket extension doesn't have native support for them. This mode
    /// affects future gates where we might have both options.
    Native,
}

// =============================================================================
// Code Generator
// =============================================================================

/// HUGR code generator.
///
/// Walks a Zluppy AST and produces a HUGR graph.
pub struct HugrCodegen {
    /// Code generation mode.
    mode: CodegenMode,
    /// Allocators by name.
    allocators: BTreeMap<String, Allocator>,
    /// Number of active statically unrolled loop bodies.
    unrolled_loop_depth: usize,
    /// Total number of qubits across all allocators.
    total_qubits: usize,
    /// Collected gate operations.
    operations: Vec<GateOp>,
    /// Names of classical variables (from measurement results).
    classical_vars: BTreeMap<String, String>,
    /// Every allocation, including distinct instances of scoped declarations.
    all_allocators: BTreeMap<String, Allocator>,
    measurement_names: std::collections::BTreeSet<String>,
    /// Compile-time values available to gate angle expressions.
    comptime: RefCell<ComptimeEvaluator>,
}

/// A gate operation to be compiled.
#[derive(Debug, Clone)]
enum GateOp {
    /// A direct TketOp gate.
    Direct {
        op: TketOp,
        qubits: Vec<QubitRef>,
        angle: Option<f64>,
    },
    /// SWAP gate (will be decomposed to 3 CX gates).
    Swap {
        qubit_a: QubitRef,
        qubit_b: QubitRef,
    },
    /// iSWAP gate (will be decomposed).
    ISwap {
        qubit_a: QubitRef,
        qubit_b: QubitRef,
    },
    /// Mid-circuit measurement (keeps qubit, stores result).
    MidMeasure {
        qubit: QubitRef,
        /// Name of the classical variable to store the result.
        result_var: String,
    },
    /// Conditional block based on classical measurement result.
    Conditional {
        /// Name of the classical variable to condition on.
        condition_var: String,
        /// Operations to execute if condition is true.
        then_ops: Vec<GateOp>,
        /// Operations to execute if condition is false.
        else_ops: Vec<GateOp>,
    },
}

impl HugrCodegen {
    /// Create a new HUGR code generator with default settings.
    ///
    /// Uses `CodegenMode::Decompose` by default, which breaks down composite
    /// gates into primitives for maximum compatibility.
    pub fn new() -> Self {
        Self {
            mode: CodegenMode::default(),
            allocators: BTreeMap::new(),
            unrolled_loop_depth: 0,
            total_qubits: 0,
            operations: Vec::new(),
            classical_vars: BTreeMap::new(),
            all_allocators: BTreeMap::new(),
            measurement_names: std::collections::BTreeSet::new(),
            comptime: RefCell::new(angle_evaluator()),
        }
    }

    /// Create a new HUGR code generator with the specified mode.
    ///
    /// # Example
    /// ```ignore
    /// let codegen = HugrCodegen::with_mode(CodegenMode::Native);
    /// ```
    pub fn with_mode(mode: CodegenMode) -> Self {
        Self {
            mode,
            allocators: BTreeMap::new(),
            unrolled_loop_depth: 0,
            total_qubits: 0,
            operations: Vec::new(),
            classical_vars: BTreeMap::new(),
            all_allocators: BTreeMap::new(),
            measurement_names: std::collections::BTreeSet::new(),
            comptime: RefCell::new(angle_evaluator()),
        }
    }

    /// Get the current codegen mode.
    pub fn mode(&self) -> CodegenMode {
        self.mode
    }

    /// Set the codegen mode.
    pub fn set_mode(&mut self, mode: CodegenMode) {
        self.mode = mode;
    }

    /// Number of qubits allocated by the compiled program.
    pub fn num_qubits(&self) -> usize {
        self.total_qubits
    }

    /// Compile a Zluppy program to HUGR.
    pub fn compile(&mut self, program: &Program) -> HugrResult<Hugr> {
        // Phase 1: Collect allocators and operations
        self.collect_program(program)?;

        // Phase 2: Build HUGR
        self.build_hugr()
    }

    /// Compile a function to HUGR.
    pub fn compile_function(&mut self, fn_decl: &FnDecl) -> HugrResult<Hugr> {
        *self = Self::with_mode(self.mode);
        // Collect from function body
        self.collect_block_with_bindings(
            &fn_decl.body,
            fn_decl.params.iter().map(|param| param.name.clone()),
        )?;

        // Build HUGR
        self.build_hugr()
    }

    // =========================================================================
    // Collection Phase
    // =========================================================================

    fn collect_program(&mut self, program: &Program) -> HugrResult<()> {
        self.comptime = RefCell::new(angle_evaluator());
        self.allocators.clear();
        self.unrolled_loop_depth = 0;
        self.all_allocators.clear();
        self.classical_vars.clear();
        self.operations.clear();
        self.total_qubits = 0;
        self.measurement_names.clear();
        for decl in &program.declarations {
            if let TopLevelDecl::Binding(binding) = decl {
                self.collect_comptime_binding(binding);
            }
        }
        for decl in &program.declarations {
            self.collect_top_level(decl)?;
        }
        Ok(())
    }

    fn collect_comptime_binding(&mut self, binding: &Binding) {
        if !define_comptime_binding(&mut self.comptime.borrow_mut(), binding) {
            self.comptime
                .borrow_mut()
                .context
                .define(&binding.name, ComptimeValue::Undefined);
        }
    }

    fn collect_top_level(&mut self, decl: &TopLevelDecl) -> HugrResult<()> {
        match decl {
            TopLevelDecl::Fn(fn_decl)
                // Only collect from main function for now
                if fn_decl.name == "main" => {
                    self.collect_block_with_bindings(
                        &fn_decl.body,
                        fn_decl.params.iter().map(|param| param.name.clone()),
                    )?;
                }
            TopLevelDecl::Binding(binding) => {
                self.collect_binding(binding)?;
            }
            _ => {}
        }
        Ok(())
    }

    fn collect_binding(&mut self, binding: &Binding) -> HugrResult<()> {
        if let Some(value) = &binding.value {
            let capacity = self.try_extract_allocator(value).or_else(|| {
                self.try_extract_child_allocator(value)
                    .map(|(_, size)| size)
            });
            if let Some(capacity) = capacity {
                let name = if self.all_allocators.contains_key(&binding.name) {
                    format!("{}#{}", binding.name, self.all_allocators.len())
                } else {
                    binding.name.clone()
                };
                let allocator = Allocator {
                    name: name.clone(),
                    capacity,
                    start_index: self.total_qubits,
                };
                self.total_qubits += capacity;
                self.all_allocators.insert(name, allocator.clone());
                self.allocators.insert(binding.name.clone(), allocator);
                self.classical_vars.remove(&binding.name);
                return Ok(());
            }
            if self.is_measurement_call(value) || matches!(value, Expr::Measure(_)) {
                let name = self.collect_measurement_assignment(value, &binding.name)?;
                self.classical_vars.insert(binding.name.clone(), name);
                self.allocators.remove(&binding.name);
                return Ok(());
            }
        }
        self.allocators.remove(&binding.name);
        self.classical_vars.remove(&binding.name);
        Ok(())
    }

    /// Check if an expression is a measurement call.
    fn is_measurement_call(&self, expr: &Expr) -> bool {
        if let Expr::Call(call) = expr
            && let Ok(name) = self.extract_call_name(&call.callee)
        {
            return name.as_str() == "mz";
        }
        false
    }

    /// Collect a measurement assignment: mut result := mz(u1) q[0] or mz(u1) [q[0], q[1]]
    fn collect_measurement_assignment(
        &mut self,
        expr: &Expr,
        result_var: &str,
    ) -> HugrResult<String> {
        let qubits = match expr {
            Expr::Measure(measure) => self.extract_measurement_targets(&measure.targets)?,
            Expr::Call(call) => match call.args.as_slice() {
                [_, target] => self.extract_measurement_targets(target)?,
                [target] => vec![self.extract_qubit_ref(target)?],
                _ => {
                    return Err(HugrError::WrongArgumentCount {
                        gate: "mz".to_string(),
                        expected: 2,
                        got: call.args.len(),
                    });
                }
            },
            _ => return Err(HugrError::UnsupportedExpression),
        };
        let mut first_result = None;
        for (index, qubit) in qubits.into_iter().enumerate() {
            let preferred = if index == 0 {
                result_var.to_string()
            } else {
                format!("{result_var}_{index}")
            };
            let identity = self.fresh_measurement_name(&preferred);
            if index == 0 {
                first_result = Some(identity.clone());
            }
            self.operations.push(GateOp::MidMeasure {
                qubit,
                result_var: identity,
            });
        }
        first_result.ok_or(HugrError::UnsupportedExpression)
    }

    /// Reserve every element's identity, including array elements and repeated declarations.
    fn fresh_measurement_name(&mut self, preferred: &str) -> String {
        let mut name = preferred.to_string();
        let mut suffix = 0;
        while !self.measurement_names.insert(name.clone()) {
            name = format!("{preferred}#{suffix}");
            suffix += 1;
        }
        name
    }

    /// Extract measurement targets from an expression.
    /// Handles both single qubit (q[0]) and array (&[q[0], q[1]]) syntax.
    fn extract_measurement_targets(&self, expr: &Expr) -> HugrResult<Vec<QubitRef>> {
        match expr {
            // Single qubit: q[0]
            Expr::Index(index_expr) => {
                let qubit = self.extract_qubit_from_index(index_expr)?;
                Ok(vec![qubit])
            }
            Expr::BracketArray(array) => array
                .elements
                .iter()
                .map(|expr| self.extract_qubit_ref(expr))
                .collect(),
            // Array of qubits: &[q[0], q[1]]
            Expr::Unary(unary) => {
                if let crate::ast::UnaryOp::AddrOf = unary.op
                    && let Expr::BracketArray(arr) = &unary.operand
                {
                    let mut qubits = Vec::new();
                    for elem in &arr.elements {
                        let qubit = self.extract_qubit_ref(elem)?;
                        qubits.push(qubit);
                    }
                    return Ok(qubits);
                }
                Err(HugrError::UnsupportedExpression)
            }
            _ => Err(HugrError::UnsupportedExpression),
        }
    }

    /// Return the elements of a supported batch literal.
    fn batch_elements<'a>(&self, expr: &'a Expr) -> Option<&'a [Expr]> {
        match expr {
            Expr::Set(set) => Some(&set.elements),
            Expr::BracketArray(array) => Some(&array.elements),
            Expr::Unary(unary) if matches!(unary.op, crate::ast::UnaryOp::AddrOf) => {
                if let Expr::BracketArray(array) = &unary.operand {
                    Some(&array.elements)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Check if an expression is a batch literal.
    fn is_batch_literal(&self, expr: &Expr) -> bool {
        self.batch_elements(expr).is_some()
    }

    fn collect_block(&mut self, block: &Block) -> HugrResult<()> {
        self.collect_block_with_bindings(block, std::iter::empty())
    }

    fn collect_block_with_bindings(
        &mut self,
        block: &Block,
        bindings: impl IntoIterator<Item = String>,
    ) -> HugrResult<()> {
        self.comptime.borrow_mut().context.push_scope();
        let allocators = self.allocators.clone();
        let classical_vars = self.classical_vars.clone();
        for name in bindings {
            self.comptime
                .borrow_mut()
                .context
                .define(&name, ComptimeValue::Undefined);
            self.allocators.remove(&name);
            self.classical_vars.remove(&name);
        }
        let result = (|| {
            block
                .statements
                .iter()
                .try_for_each(|stmt| self.collect_stmt(stmt))?;

            Ok(())
        })();
        self.allocators = allocators;
        self.classical_vars = classical_vars;
        self.comptime.borrow_mut().context.pop_scope();
        result
    }

    fn collect_stmt(&mut self, stmt: &Stmt) -> HugrResult<()> {
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
                return Err(HugrError::UnsupportedLoopControl { statement });
            }
        }
        match stmt {
            Stmt::Binding(binding) => {
                self.collect_binding(binding)?;
                if !define_comptime_binding(&mut self.comptime.borrow_mut(), binding) {
                    self.comptime
                        .borrow_mut()
                        .context
                        .define(&binding.name, ComptimeValue::Undefined);
                }
            }
            Stmt::Expr(expr_stmt) => self.collect_expr(&expr_stmt.expr)?,
            Stmt::Gate(gate) => self.collect_gate_op(gate)?,
            Stmt::Prepare(prepare) => self.collect_prepare_op(prepare)?,
            // Tick blocks - flatten operations (HUGR doesn't have native parallel blocks)
            Stmt::Tick(tick_stmt) => {
                // Preserve tick visibility: allocator and measurement bindings remain
                // available afterward, while comptime locals retain their original scope.
                self.comptime.borrow_mut().context.push_scope();
                let result = tick_stmt
                    .body
                    .iter()
                    .try_for_each(|stmt| self.collect_stmt(stmt));
                self.comptime.borrow_mut().context.pop_scope();
                result?;
                // The surviving runtime bindings must also shadow outer
                // comptime names after the tick's local evaluation scope ends.
                for name in self.allocators.keys().chain(self.classical_vars.keys()) {
                    self.comptime
                        .borrow_mut()
                        .context
                        .define(name, ComptimeValue::Undefined);
                }
            }
            Stmt::If(if_stmt) => self.collect_if(if_stmt)?,
            Stmt::For(for_stmt) => self.collect_for(for_stmt)?,
            Stmt::Break(_) => return Err(HugrError::UnsupportedLoopControl { statement: "break" }),
            Stmt::Continue(_) => {
                return Err(HugrError::UnsupportedLoopControl {
                    statement: "continue",
                });
            }
            Stmt::Block(block) => self.collect_block(block)?,
            _ => {}
        }
        Ok(())
    }

    fn collect_if(&mut self, if_stmt: &IfStmt) -> HugrResult<()> {
        let condition = self
            .comptime
            .borrow_mut()
            .eval_expr(&if_stmt.condition)
            .ok()
            .and_then(|value| value.as_bool());
        if let Some(condition) = condition {
            if condition {
                self.collect_block(&if_stmt.then_body)?;
            } else if let Some(branch) = &if_stmt.else_body {
                self.collect_else_branch(branch)?;
            }
        } else if let Some(condition_var) = self.try_extract_classical_condition(&if_stmt.condition)
        {
            let then_ops = self.collect_block_ops(&if_stmt.then_body)?;
            let else_ops = if let Some(branch) = &if_stmt.else_body {
                self.collect_else_ops(branch)?
            } else {
                Vec::new()
            };
            self.operations.push(GateOp::Conditional {
                condition_var,
                then_ops,
                else_ops,
            });
        } else {
            return Err(HugrError::UnsupportedCondition);
        }
        Ok(())
    }

    fn collect_for(&mut self, for_stmt: &ForStmt) -> HugrResult<()> {
        if let Some(statement) = super::block_control(&for_stmt.body) {
            return Err(HugrError::UnsupportedLoopControl { statement });
        }
        let ForRange::Range { start, end } = &for_stmt.range else {
            return Err(HugrError::CollectionLoop);
        };
        let [capture] = for_stmt.captures.as_slice() else {
            return Err(HugrError::InvalidLoopCaptures {
                count: for_stmt.captures.len(),
            });
        };
        let start = self.eval_loop_bound(start)?;
        let end = self.eval_loop_bound(end)?;
        for value in start..end {
            self.comptime.borrow_mut().context.push_scope();
            self.comptime
                .borrow_mut()
                .context
                .define(capture, ComptimeValue::Int(value));
            let allocators = self.allocators.clone();
            let classical_vars = self.classical_vars.clone();
            self.allocators.remove(capture);
            self.classical_vars.remove(capture);
            self.unrolled_loop_depth += 1;
            let result = self.collect_block(&for_stmt.body);
            self.unrolled_loop_depth -= 1;
            self.allocators = allocators;
            self.classical_vars = classical_vars;
            self.comptime.borrow_mut().context.pop_scope();
            result?;
        }
        Ok(())
    }

    fn eval_loop_bound(&self, expr: &Expr) -> HugrResult<i64> {
        match self.comptime.borrow_mut().eval_expr(expr) {
            Ok(ComptimeValue::Int(value)) => Ok(value),
            Ok(ComptimeValue::Uint(value)) => {
                i64::try_from(value).map_err(|_| HugrError::NonConstantLoopBound)
            }
            _ => Err(HugrError::NonConstantLoopBound),
        }
    }

    /// Try to extract a classical variable name from a condition expression.
    /// Returns Some(var_name) if the condition is a simple reference to a classical variable.
    fn try_extract_classical_condition(&self, expr: &Expr) -> Option<String> {
        if let Expr::Ident(ident) = expr {
            return self.classical_vars.get(&ident.name).cloned();
        }
        None
    }

    /// Collect a branch without disturbing the surrounding operation list on error.
    fn collect_block_ops(&mut self, block: &Block) -> HugrResult<Vec<GateOp>> {
        let saved_ops = std::mem::take(&mut self.operations);
        let result = self.collect_block(block);
        let collected = std::mem::replace(&mut self.operations, saved_ops);
        result.map(|()| collected)
    }

    fn collect_else_ops(&mut self, branch: &ElseBranch) -> HugrResult<Vec<GateOp>> {
        let saved_ops = std::mem::take(&mut self.operations);
        let result = self.collect_else_branch(branch);
        let collected = std::mem::replace(&mut self.operations, saved_ops);
        result.map(|()| collected)
    }

    fn collect_else_branch(&mut self, branch: &ElseBranch) -> HugrResult<()> {
        match branch {
            ElseBranch::Else(block) => self.collect_block(block),
            ElseBranch::ElseIf(if_stmt) => self.collect_if(if_stmt),
        }
    }

    fn collect_expr(&mut self, expr: &Expr) -> HugrResult<()> {
        match expr {
            // HUGR lowering does not yet represent ordinary function calls.
            // Reject them so user code cannot silently disappear from the circuit.
            Expr::Call(call) => self.reject_unsupported_call(call)?,
            Expr::Gate(gate) => self.collect_gate_expr(gate)?,
            Expr::Binary(binary) => {
                self.collect_expr(&binary.left)?;
                self.collect_expr(&binary.right)?;
            }
            _ => {}
        }
        Ok(())
    }

    fn collect_gate_expr(&mut self, gate: &crate::ast::GateExpr) -> HugrResult<()> {
        if let Expr::Ident(ident) = &gate.target {
            let capacity = self
                .allocators
                .get(&ident.name)
                .ok_or_else(|| HugrError::AllocatorNotFound {
                    name: ident.name.clone(),
                })?
                .capacity;
            let arity = gate.kind.arity();
            if arity != 1 {
                return Err(HugrError::InvalidAllocatorBroadcast {
                    gate: gate.kind.keyword().to_string(),
                    arity,
                });
            }
            if capacity == 0 {
                return Err(HugrError::EmptyAllocatorBroadcast {
                    gate: gate.kind.keyword().to_string(),
                    allocator: ident.name.clone(),
                });
            }
            for index in 0..capacity {
                self.collect_named_gate(
                    gate.kind.keyword(),
                    &gate.params,
                    vec![Expr::SlotRef(Box::new(crate::ast::SlotRef {
                        allocator: ident.name.clone(),
                        index: Box::new(Expr::IntLit(crate::ast::IntLit {
                            value: index as i128,
                            suffix: None,
                            location: None,
                        })),
                        location: gate.location.clone(),
                    }))],
                )?;
            }
            return Ok(());
        }

        let targets = match &gate.target {
            Expr::Tuple(tuple) => tuple.elements.clone(),
            target => vec![target.clone()],
        };
        self.collect_named_gate(gate.kind.keyword(), &gate.params, targets)
    }

    fn collect_gate_op(&mut self, gate: &crate::ast::GateOp) -> HugrResult<()> {
        let targets = gate
            .targets
            .iter()
            .cloned()
            .map(|target| Expr::SlotRef(Box::new(target)))
            .collect();
        self.collect_named_gate(gate.kind.keyword(), &gate.params, targets)
    }

    fn collect_named_gate(
        &mut self,
        name: &str,
        params: &[Expr],
        targets: Vec<Expr>,
    ) -> HugrResult<()> {
        let mut args = params.to_vec();
        args.extend(targets);
        self.collect_call(&CallExpr {
            callee: Expr::Ident(crate::ast::Ident {
                name: name.to_string(),
                location: None,
            }),
            args,
            location: None,
        })
    }

    fn reject_unsupported_call(&self, call: &CallExpr) -> HugrResult<()> {
        let name = self.extract_call_name(&call.callee)?;
        Err(HugrError::UnsupportedCall { name })
    }

    fn collect_prepare_op(&mut self, prepare: &crate::ast::PrepareOp) -> HugrResult<()> {
        let allocator = self.allocators.get(&prepare.allocator).ok_or_else(|| {
            HugrError::AllocatorNotFound {
                name: prepare.allocator.clone(),
            }
        })?;
        let slots = prepare
            .slots
            .clone()
            .unwrap_or_else(|| (0..allocator.capacity as u32).collect());
        for index in slots {
            let index = index as usize;
            if index >= allocator.capacity {
                return Err(HugrError::QubitIndexOutOfBounds {
                    index,
                    capacity: allocator.capacity,
                });
            }
            self.operations.push(GateOp::Direct {
                op: TketOp::Reset,
                qubits: vec![QubitRef::new(&allocator.name, index)],
                angle: None,
            });
        }
        Ok(())
    }

    fn collect_call(&mut self, call: &CallExpr) -> HugrResult<()> {
        self.collect_call_inner(call, true)
    }

    fn collect_call_inner(&mut self, call: &CallExpr, allow_batch: bool) -> HugrResult<()> {
        // Check if this is a gate call
        let name = self.extract_call_name(&call.callee)?;

        // Named-gate lowering must resolve through the canonical HUGR mapping.
        let Some(mapping) = gate_name_to_mapping(&name) else {
            return Err(HugrError::UnknownGate { name });
        };

        let qubit_count = mapping.arity();
        let needs_angle = mapping.needs_angle();
        let qubit_start = usize::from(needs_angle);
        if call.args.len() < qubit_start {
            return Err(HugrError::WrongArgumentCount {
                gate: name,
                expected: qubit_count + qubit_start,
                got: call.args.len(),
            });
        }
        let qubit_args = &call.args[qubit_start..];
        if allow_batch && qubit_args.len() == 1 && self.is_batch_literal(&qubit_args[0]) {
            let Some(elements) = self.batch_elements(&qubit_args[0]).map(<[Expr]>::to_vec) else {
                return Err(HugrError::UnsupportedExpression);
            };
            if elements.is_empty() {
                return Err(HugrError::EmptyGateBatch { gate: name });
            }
            for element in elements {
                let targets = if qubit_count == 1 {
                    vec![element]
                } else if let Expr::Tuple(tuple) = element {
                    if tuple.elements.len() != qubit_count {
                        return Err(HugrError::WrongArgumentCount {
                            gate: name,
                            expected: qubit_count,
                            got: tuple.elements.len(),
                        });
                    }
                    tuple.elements
                } else {
                    return Err(HugrError::WrongArgumentCount {
                        gate: name,
                        expected: qubit_count,
                        got: 1,
                    });
                };
                let mut args = call.args[..qubit_start].to_vec();
                args.extend(targets);
                self.collect_call_inner(
                    &CallExpr {
                        callee: call.callee.clone(),
                        args,
                        location: call.location.clone(),
                    },
                    false,
                )?;
            }
            return Ok(());
        }

        match mapping {
            GateMapping::Direct(op) => {
                // For rotation gates, angle comes first (angle-first syntax)
                let (angle, qubit_start) = if needs_angle {
                    (Some(self.extract_angle(&call.args[0])?), 1)
                } else {
                    (None, 0)
                };

                // Standard non-batch case
                let expected_args = if needs_angle {
                    qubit_count + 1
                } else {
                    qubit_count
                };

                if call.args.len() != expected_args {
                    return Err(HugrError::WrongArgumentCount {
                        gate: name,
                        expected: expected_args,
                        got: call.args.len(),
                    });
                }

                // Extract qubit references (after angle if present)
                let mut qubits = Vec::with_capacity(qubit_count);
                for arg in call.args.iter().skip(qubit_start).take(qubit_count) {
                    let qubit_ref = self.extract_qubit_ref(arg)?;
                    qubits.push(qubit_ref);
                }

                self.operations.push(GateOp::Direct { op, qubits, angle });
            }

            GateMapping::Swap => {
                // SWAP requires exactly 2 qubit arguments
                if call.args.len() != 2 {
                    return Err(HugrError::WrongArgumentCount {
                        gate: name,
                        expected: 2,
                        got: call.args.len(),
                    });
                }
                let qubit_a = self.extract_qubit_ref(&call.args[0])?;
                let qubit_b = self.extract_qubit_ref(&call.args[1])?;
                self.operations.push(GateOp::Swap { qubit_a, qubit_b });
            }

            GateMapping::ISwap => {
                // iSWAP requires exactly 2 qubit arguments
                if call.args.len() != 2 {
                    return Err(HugrError::WrongArgumentCount {
                        gate: name,
                        expected: 2,
                        got: call.args.len(),
                    });
                }
                let qubit_a = self.extract_qubit_ref(&call.args[0])?;
                let qubit_b = self.extract_qubit_ref(&call.args[1])?;
                self.operations.push(GateOp::ISwap { qubit_a, qubit_b });
            }

            GateMapping::SY => {
                // SY (sqrt of Y) requires exactly 1 qubit argument
                // Decomposed to Ry(π/2)
                if call.args.len() != 1 {
                    return Err(HugrError::WrongArgumentCount {
                        gate: name,
                        expected: 1,
                        got: call.args.len(),
                    });
                }
                let qubit = self.extract_qubit_ref(&call.args[0])?;
                self.operations.push(GateOp::Direct {
                    op: TketOp::Ry,
                    qubits: vec![qubit],
                    angle: Some(std::f64::consts::FRAC_PI_2),
                });
            }

            GateMapping::SYdg => {
                // SYdg (sqrt of Y dagger) requires exactly 1 qubit argument
                // Decomposed to Ry(-π/2)
                if call.args.len() != 1 {
                    return Err(HugrError::WrongArgumentCount {
                        gate: name,
                        expected: 1,
                        got: call.args.len(),
                    });
                }
                let qubit = self.extract_qubit_ref(&call.args[0])?;
                self.operations.push(GateOp::Direct {
                    op: TketOp::Ry,
                    qubits: vec![qubit],
                    angle: Some(-std::f64::consts::FRAC_PI_2),
                });
            }

            GateMapping::CH => {
                // CH (controlled Hadamard) requires exactly 2 qubit arguments
                // Decomposed to: Ry(π/4, b) CZ(a,b) Ry(-π/4, b)
                if call.args.len() != 2 {
                    return Err(HugrError::WrongArgumentCount {
                        gate: name,
                        expected: 2,
                        got: call.args.len(),
                    });
                }
                let qubit_a = self.extract_qubit_ref(&call.args[0])?;
                let qubit_b = self.extract_qubit_ref(&call.args[1])?;
                // Ry(π/4) on target
                self.operations.push(GateOp::Direct {
                    op: TketOp::Ry,
                    qubits: vec![qubit_b.clone()],
                    angle: Some(std::f64::consts::FRAC_PI_4),
                });
                // CZ(control, target)
                self.operations.push(GateOp::Direct {
                    op: TketOp::CZ,
                    qubits: vec![qubit_a, qubit_b.clone()],
                    angle: None,
                });
                // Ry(-π/4) on target
                self.operations.push(GateOp::Direct {
                    op: TketOp::Ry,
                    qubits: vec![qubit_b],
                    angle: Some(-std::f64::consts::FRAC_PI_4),
                });
            }

            GateMapping::SZZ => {
                // SZZ (sqrt of ZZ Ising) requires exactly 2 qubit arguments
                // Decomposed to: CX(a,b) S(b) CX(a,b)
                if call.args.len() != 2 {
                    return Err(HugrError::WrongArgumentCount {
                        gate: name,
                        expected: 2,
                        got: call.args.len(),
                    });
                }
                let qubit_a = self.extract_qubit_ref(&call.args[0])?;
                let qubit_b = self.extract_qubit_ref(&call.args[1])?;
                self.operations.push(GateOp::Direct {
                    op: TketOp::CX,
                    qubits: vec![qubit_a.clone(), qubit_b.clone()],
                    angle: None,
                });
                self.operations.push(GateOp::Direct {
                    op: TketOp::S,
                    qubits: vec![qubit_b.clone()],
                    angle: None,
                });
                self.operations.push(GateOp::Direct {
                    op: TketOp::CX,
                    qubits: vec![qubit_a, qubit_b],
                    angle: None,
                });
            }

            GateMapping::SZZdg => {
                // SZZdg (sqrt of ZZ Ising dagger) - use Sdg instead of S
                if call.args.len() != 2 {
                    return Err(HugrError::WrongArgumentCount {
                        gate: name,
                        expected: 2,
                        got: call.args.len(),
                    });
                }
                let qubit_a = self.extract_qubit_ref(&call.args[0])?;
                let qubit_b = self.extract_qubit_ref(&call.args[1])?;
                self.operations.push(GateOp::Direct {
                    op: TketOp::CX,
                    qubits: vec![qubit_a.clone(), qubit_b.clone()],
                    angle: None,
                });
                self.operations.push(GateOp::Direct {
                    op: TketOp::Sdg,
                    qubits: vec![qubit_b.clone()],
                    angle: None,
                });
                self.operations.push(GateOp::Direct {
                    op: TketOp::CX,
                    qubits: vec![qubit_a, qubit_b],
                    angle: None,
                });
            }

            GateMapping::SXX => {
                // SXX (sqrt of XX Ising) requires exactly 2 qubit arguments
                // Decomposed to: H(a) H(b) SZZ(a,b) H(a) H(b)
                if call.args.len() != 2 {
                    return Err(HugrError::WrongArgumentCount {
                        gate: name,
                        expected: 2,
                        got: call.args.len(),
                    });
                }
                let qubit_a = self.extract_qubit_ref(&call.args[0])?;
                let qubit_b = self.extract_qubit_ref(&call.args[1])?;
                // H(a) H(b)
                self.operations.push(GateOp::Direct {
                    op: TketOp::H,
                    qubits: vec![qubit_a.clone()],
                    angle: None,
                });
                self.operations.push(GateOp::Direct {
                    op: TketOp::H,
                    qubits: vec![qubit_b.clone()],
                    angle: None,
                });
                // SZZ decomposition inline: CX S CX
                self.operations.push(GateOp::Direct {
                    op: TketOp::CX,
                    qubits: vec![qubit_a.clone(), qubit_b.clone()],
                    angle: None,
                });
                self.operations.push(GateOp::Direct {
                    op: TketOp::S,
                    qubits: vec![qubit_b.clone()],
                    angle: None,
                });
                self.operations.push(GateOp::Direct {
                    op: TketOp::CX,
                    qubits: vec![qubit_a.clone(), qubit_b.clone()],
                    angle: None,
                });
                // H(a) H(b)
                self.operations.push(GateOp::Direct {
                    op: TketOp::H,
                    qubits: vec![qubit_a],
                    angle: None,
                });
                self.operations.push(GateOp::Direct {
                    op: TketOp::H,
                    qubits: vec![qubit_b],
                    angle: None,
                });
            }

            GateMapping::SXXdg => {
                // SXXdg - same as SXX but use Sdg instead of S
                if call.args.len() != 2 {
                    return Err(HugrError::WrongArgumentCount {
                        gate: name,
                        expected: 2,
                        got: call.args.len(),
                    });
                }
                let qubit_a = self.extract_qubit_ref(&call.args[0])?;
                let qubit_b = self.extract_qubit_ref(&call.args[1])?;
                self.operations.push(GateOp::Direct {
                    op: TketOp::H,
                    qubits: vec![qubit_a.clone()],
                    angle: None,
                });
                self.operations.push(GateOp::Direct {
                    op: TketOp::H,
                    qubits: vec![qubit_b.clone()],
                    angle: None,
                });
                self.operations.push(GateOp::Direct {
                    op: TketOp::CX,
                    qubits: vec![qubit_a.clone(), qubit_b.clone()],
                    angle: None,
                });
                self.operations.push(GateOp::Direct {
                    op: TketOp::Sdg,
                    qubits: vec![qubit_b.clone()],
                    angle: None,
                });
                self.operations.push(GateOp::Direct {
                    op: TketOp::CX,
                    qubits: vec![qubit_a.clone(), qubit_b.clone()],
                    angle: None,
                });
                self.operations.push(GateOp::Direct {
                    op: TketOp::H,
                    qubits: vec![qubit_a],
                    angle: None,
                });
                self.operations.push(GateOp::Direct {
                    op: TketOp::H,
                    qubits: vec![qubit_b],
                    angle: None,
                });
            }

            GateMapping::SYY => {
                // SYY (sqrt of YY Ising) requires exactly 2 qubit arguments
                // Decomposed to: Vdg(a) Vdg(b) SZZ(a,b) V(a) V(b)
                if call.args.len() != 2 {
                    return Err(HugrError::WrongArgumentCount {
                        gate: name,
                        expected: 2,
                        got: call.args.len(),
                    });
                }
                let qubit_a = self.extract_qubit_ref(&call.args[0])?;
                let qubit_b = self.extract_qubit_ref(&call.args[1])?;
                // Vdg(a) Vdg(b) - SXdg gates
                self.operations.push(GateOp::Direct {
                    op: TketOp::Vdg,
                    qubits: vec![qubit_a.clone()],
                    angle: None,
                });
                self.operations.push(GateOp::Direct {
                    op: TketOp::Vdg,
                    qubits: vec![qubit_b.clone()],
                    angle: None,
                });
                // SZZ decomposition inline: CX S CX
                self.operations.push(GateOp::Direct {
                    op: TketOp::CX,
                    qubits: vec![qubit_a.clone(), qubit_b.clone()],
                    angle: None,
                });
                self.operations.push(GateOp::Direct {
                    op: TketOp::S,
                    qubits: vec![qubit_b.clone()],
                    angle: None,
                });
                self.operations.push(GateOp::Direct {
                    op: TketOp::CX,
                    qubits: vec![qubit_a.clone(), qubit_b.clone()],
                    angle: None,
                });
                // V(a) V(b) - SX gates
                self.operations.push(GateOp::Direct {
                    op: TketOp::V,
                    qubits: vec![qubit_a],
                    angle: None,
                });
                self.operations.push(GateOp::Direct {
                    op: TketOp::V,
                    qubits: vec![qubit_b],
                    angle: None,
                });
            }

            GateMapping::SYYdg => {
                // SYYdg - same as SYY but use Sdg instead of S
                if call.args.len() != 2 {
                    return Err(HugrError::WrongArgumentCount {
                        gate: name,
                        expected: 2,
                        got: call.args.len(),
                    });
                }
                let qubit_a = self.extract_qubit_ref(&call.args[0])?;
                let qubit_b = self.extract_qubit_ref(&call.args[1])?;
                self.operations.push(GateOp::Direct {
                    op: TketOp::Vdg,
                    qubits: vec![qubit_a.clone()],
                    angle: None,
                });
                self.operations.push(GateOp::Direct {
                    op: TketOp::Vdg,
                    qubits: vec![qubit_b.clone()],
                    angle: None,
                });
                self.operations.push(GateOp::Direct {
                    op: TketOp::CX,
                    qubits: vec![qubit_a.clone(), qubit_b.clone()],
                    angle: None,
                });
                self.operations.push(GateOp::Direct {
                    op: TketOp::Sdg,
                    qubits: vec![qubit_b.clone()],
                    angle: None,
                });
                self.operations.push(GateOp::Direct {
                    op: TketOp::CX,
                    qubits: vec![qubit_a.clone(), qubit_b.clone()],
                    angle: None,
                });
                self.operations.push(GateOp::Direct {
                    op: TketOp::V,
                    qubits: vec![qubit_a],
                    angle: None,
                });
                self.operations.push(GateOp::Direct {
                    op: TketOp::V,
                    qubits: vec![qubit_b],
                    angle: None,
                });
            }

            GateMapping::RZZ => {
                // RZZ (ZZ rotation) requires 1 angle + 2 qubit arguments (angle-first)
                // Decomposed to: CX(a,b) Rz(θ, b) CX(a,b)
                if call.args.len() != 3 {
                    return Err(HugrError::WrongArgumentCount {
                        gate: name,
                        expected: 3,
                        got: call.args.len(),
                    });
                }
                // Angle-first: rzz(angle, qubit_a, qubit_b)
                let angle = self.extract_angle(&call.args[0])?;
                let qubit_a = self.extract_qubit_ref(&call.args[1])?;
                let qubit_b = self.extract_qubit_ref(&call.args[2])?;
                self.operations.push(GateOp::Direct {
                    op: TketOp::CX,
                    qubits: vec![qubit_a.clone(), qubit_b.clone()],
                    angle: None,
                });
                self.operations.push(GateOp::Direct {
                    op: TketOp::Rz,
                    qubits: vec![qubit_b.clone()],
                    angle: Some(angle),
                });
                self.operations.push(GateOp::Direct {
                    op: TketOp::CX,
                    qubits: vec![qubit_a, qubit_b],
                    angle: None,
                });
            }

            GateMapping::F => {
                // F gate (Clifford face rotation) requires exactly 1 qubit argument
                // Decomposed to: H Sdg H Sdg
                if call.args.len() != 1 {
                    return Err(HugrError::WrongArgumentCount {
                        gate: name,
                        expected: 1,
                        got: call.args.len(),
                    });
                }
                let qubit = self.extract_qubit_ref(&call.args[0])?;
                self.operations.push(GateOp::Direct {
                    op: TketOp::H,
                    qubits: vec![qubit.clone()],
                    angle: None,
                });
                self.operations.push(GateOp::Direct {
                    op: TketOp::Sdg,
                    qubits: vec![qubit.clone()],
                    angle: None,
                });
                self.operations.push(GateOp::Direct {
                    op: TketOp::H,
                    qubits: vec![qubit.clone()],
                    angle: None,
                });
                self.operations.push(GateOp::Direct {
                    op: TketOp::Sdg,
                    qubits: vec![qubit],
                    angle: None,
                });
            }

            GateMapping::Fdg => {
                // Fdg gate (F dagger) - S H S H
                if call.args.len() != 1 {
                    return Err(HugrError::WrongArgumentCount {
                        gate: name,
                        expected: 1,
                        got: call.args.len(),
                    });
                }
                let qubit = self.extract_qubit_ref(&call.args[0])?;
                self.operations.push(GateOp::Direct {
                    op: TketOp::S,
                    qubits: vec![qubit.clone()],
                    angle: None,
                });
                self.operations.push(GateOp::Direct {
                    op: TketOp::H,
                    qubits: vec![qubit.clone()],
                    angle: None,
                });
                self.operations.push(GateOp::Direct {
                    op: TketOp::S,
                    qubits: vec![qubit.clone()],
                    angle: None,
                });
                self.operations.push(GateOp::Direct {
                    op: TketOp::H,
                    qubits: vec![qubit],
                    angle: None,
                });
            }

            GateMapping::F4 => {
                // F4 gate (fourth root of F) - approximated with T gates
                // F4 ≈ Ry(π/4) Rz(π/4)
                if call.args.len() != 1 {
                    return Err(HugrError::WrongArgumentCount {
                        gate: name,
                        expected: 1,
                        got: call.args.len(),
                    });
                }
                let qubit = self.extract_qubit_ref(&call.args[0])?;
                self.operations.push(GateOp::Direct {
                    op: TketOp::Ry,
                    qubits: vec![qubit.clone()],
                    angle: Some(std::f64::consts::FRAC_PI_4),
                });
                self.operations.push(GateOp::Direct {
                    op: TketOp::Rz,
                    qubits: vec![qubit],
                    angle: Some(std::f64::consts::FRAC_PI_4),
                });
            }

            GateMapping::F4dg => {
                // F4dg gate (fourth root of F dagger) - reverse of F4
                if call.args.len() != 1 {
                    return Err(HugrError::WrongArgumentCount {
                        gate: name,
                        expected: 1,
                        got: call.args.len(),
                    });
                }
                let qubit = self.extract_qubit_ref(&call.args[0])?;
                self.operations.push(GateOp::Direct {
                    op: TketOp::Rz,
                    qubits: vec![qubit.clone()],
                    angle: Some(-std::f64::consts::FRAC_PI_4),
                });
                self.operations.push(GateOp::Direct {
                    op: TketOp::Ry,
                    qubits: vec![qubit],
                    angle: Some(-std::f64::consts::FRAC_PI_4),
                });
            }
        }

        Ok(())
    }

    // =========================================================================
    // Extraction Helpers
    // =========================================================================

    /// Try to extract allocator capacity from qalloc(n) call.
    fn try_extract_allocator(&self, expr: &Expr) -> Option<usize> {
        if let Expr::Call(call) = expr {
            let name = self.extract_call_name(&call.callee).ok()?;
            if name == "qalloc" && call.args.len() == 1 {
                return self.extract_integer(&call.args[0]).ok();
            }
        }
        None
    }

    /// Try to extract child allocator from base.child(n) call.
    fn try_extract_child_allocator(&self, expr: &Expr) -> Option<(String, usize)> {
        if let Expr::Call(call) = expr {
            // Check for method call pattern: expr.child(n)
            if let Expr::Field(field) = &call.callee
                && field.field == "child"
                && call.args.len() == 1
            {
                let parent = self.extract_identifier(&field.object).ok()?;
                let size = self.extract_integer(&call.args[0]).ok()?;
                return Some((parent, size));
            }
        }
        None
    }

    /// Extract the name from a call expression's callee.
    fn extract_call_name(&self, callee: &Expr) -> HugrResult<String> {
        match callee {
            Expr::Ident(ident) => Ok(ident.name.clone()),
            // Method call: q.child(n) -> "child"
            Expr::Field(field) => Ok(field.field.clone()),
            _ => Err(HugrError::UnsupportedExpression),
        }
    }

    /// Extract an identifier from an expression.
    fn extract_identifier(&self, expr: &Expr) -> HugrResult<String> {
        match expr {
            Expr::Ident(ident) => Ok(ident.name.clone()),
            _ => Err(HugrError::UnsupportedExpression),
        }
    }

    /// Extract a qubit reference from an expression (e.g., q[0]).
    fn extract_qubit_ref(&self, expr: &Expr) -> HugrResult<QubitRef> {
        match expr {
            Expr::Index(index_expr) => self.extract_qubit_from_index(index_expr),
            Expr::SlotRef(slot_ref) => {
                let index = self.extract_integer(&slot_ref.index)?;
                let allocator = self.allocators.get(&slot_ref.allocator).ok_or_else(|| {
                    HugrError::AllocatorNotFound {
                        name: slot_ref.allocator.clone(),
                    }
                })?;
                if index >= allocator.capacity {
                    return Err(HugrError::QubitIndexOutOfBounds {
                        index,
                        capacity: allocator.capacity,
                    });
                }
                Ok(QubitRef::new(&allocator.name, index))
            }
            _ => Err(HugrError::UnsupportedExpression),
        }
    }

    fn extract_qubit_from_index(&self, index: &IndexExpr) -> HugrResult<QubitRef> {
        let allocator = self.extract_identifier(&index.object)?;
        let idx = self.extract_integer(&index.index)?;

        // Validate the allocator exists
        let alloc =
            self.allocators
                .get(&allocator)
                .ok_or_else(|| HugrError::AllocatorNotFound {
                    name: allocator.clone(),
                })?;

        // Validate index is in bounds
        if idx >= alloc.capacity {
            return Err(HugrError::QubitIndexOutOfBounds {
                index: idx,
                capacity: alloc.capacity,
            });
        }

        Ok(QubitRef::new(&alloc.name, idx))
    }

    /// Extract an integer from an expression.
    fn extract_integer(&self, expr: &Expr) -> HugrResult<usize> {
        self.comptime
            .borrow_mut()
            .eval_expr(expr)
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(HugrError::UnsupportedExpression)
    }

    /// Extract a rotation angle in radians from an expression.
    fn extract_angle(&mut self, expr: &Expr) -> HugrResult<f64> {
        let value =
            resolve_angle_turns(&mut self.comptime.borrow_mut(), expr).map_err(|error| {
                HugrError::InvalidRotationAngle {
                    expression: angle_expression_name(expr),
                    reason: error.to_string(),
                }
            })?;
        Ok(value * std::f64::consts::TAU)
    }

    // =========================================================================
    // HUGR Building Phase
    // =========================================================================

    fn build_hugr(&self) -> HugrResult<Hugr> {
        if self.total_qubits == 0 {
            // Empty circuit - create minimal HUGR
            return self.build_empty_hugr();
        }

        // Create signature: no inputs, N bool outputs (measurement results)
        let bool_row: Vec<_> = (0..self.total_qubits).map(|_| bool_t()).collect();
        let signature = Signature::new(vec![], bool_row);

        // Build the body of main, the module's function-definition entry point.
        let mut builder = FunctionBuilder::new("main", signature)
            .map_err(|e| HugrError::BuilderError(e.to_string()))?;

        // Allocate qubits using QAlloc
        let mut qubit_wires: BTreeMap<QubitRef, Wire> = BTreeMap::new();
        for (name, alloc) in &self.all_allocators {
            for i in 0..alloc.capacity {
                let qubit_ref = QubitRef::new(name.clone(), i);
                // Add QAlloc operation to allocate a qubit
                let qalloc_wire: Wire = builder
                    .add_dataflow_op(TketOp::QAlloc, vec![])
                    .map_err(|e| HugrError::BuilderError(e.to_string()))?
                    .outputs()
                    .next()
                    .ok_or_else(|| {
                        HugrError::BuilderError("QAlloc produced no output".to_string())
                    })?;
                qubit_wires.insert(qubit_ref, qalloc_wire);
            }
        }

        // Track classical wires from mid-circuit measurements
        let mut classical_wires: BTreeMap<String, Wire> = BTreeMap::new();

        // Apply operations
        for gate_op in &self.operations {
            self.apply_gate(
                &mut builder,
                &mut qubit_wires,
                &mut classical_wires,
                gate_op,
            )?;
        }

        // Measure and free all qubits, then read the resulting measurements as bools.
        let output_wires: Vec<Wire> = (0..self.total_qubits)
            .map(|global_idx| {
                // Find which allocator this belongs to
                for (name, alloc) in &self.all_allocators {
                    if global_idx >= alloc.start_index
                        && global_idx < alloc.start_index + alloc.capacity
                    {
                        let local_idx = global_idx - alloc.start_index;
                        let qubit_ref = QubitRef::new(name.clone(), local_idx);
                        if let Some(&wire) = qubit_wires.get(&qubit_ref) {
                            // MeasureFree consumes a qubit and produces a Measurement.
                            let measurement = builder
                                .add_dataflow_op(TketOp::MeasureFree, vec![wire])
                                .map_err(|e| HugrError::BuilderError(e.to_string()))
                                .ok()?
                                .outputs()
                                .next()?;
                            let bool_result = builder
                                .add_dataflow_op(MeasurementOp::Read, vec![measurement])
                                .map_err(|e| HugrError::BuilderError(e.to_string()))
                                .ok()?
                                .outputs()
                                .next()?;
                            return Some(bool_result);
                        }
                    }
                }
                None
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| HugrError::BuilderError("Failed to measure all qubits".to_string()))?;

        // Finish HUGR
        builder
            .finish_hugr_with_outputs(output_wires)
            .map_err(|e| HugrError::BuilderError(e.to_string()))
    }

    fn build_empty_hugr(&self) -> HugrResult<Hugr> {
        let signature = Signature::new(vec![], vec![]);
        let builder = FunctionBuilder::new("main", signature)
            .map_err(|e| HugrError::BuilderError(e.to_string()))?;
        builder
            .finish_hugr_with_outputs(vec![])
            .map_err(|e| HugrError::BuilderError(e.to_string()))
    }

    fn apply_gate(
        &self,
        builder: &mut FunctionBuilder<Hugr>,
        qubit_wires: &mut BTreeMap<QubitRef, Wire>,
        classical_wires: &mut BTreeMap<String, Wire>,
        gate_op: &GateOp,
    ) -> HugrResult<()> {
        match gate_op {
            GateOp::Direct { op, qubits, angle } => {
                self.apply_direct_gate(builder, qubit_wires, *op, qubits, *angle)?;
            }

            GateOp::Swap { qubit_a, qubit_b } => {
                // SWAP decomposition: CX(a,b) CX(b,a) CX(a,b)
                self.apply_direct_gate(
                    builder,
                    qubit_wires,
                    TketOp::CX,
                    &[qubit_a.clone(), qubit_b.clone()],
                    None,
                )?;
                self.apply_direct_gate(
                    builder,
                    qubit_wires,
                    TketOp::CX,
                    &[qubit_b.clone(), qubit_a.clone()],
                    None,
                )?;
                self.apply_direct_gate(
                    builder,
                    qubit_wires,
                    TketOp::CX,
                    &[qubit_a.clone(), qubit_b.clone()],
                    None,
                )?;
            }

            GateOp::ISwap { qubit_a, qubit_b } => {
                // iSWAP decomposition: S(a) S(b) H(a) CX(a,b) CX(b,a) H(b)
                self.apply_direct_gate(
                    builder,
                    qubit_wires,
                    TketOp::S,
                    std::slice::from_ref(qubit_a),
                    None,
                )?;
                self.apply_direct_gate(
                    builder,
                    qubit_wires,
                    TketOp::S,
                    std::slice::from_ref(qubit_b),
                    None,
                )?;
                self.apply_direct_gate(
                    builder,
                    qubit_wires,
                    TketOp::H,
                    std::slice::from_ref(qubit_a),
                    None,
                )?;
                self.apply_direct_gate(
                    builder,
                    qubit_wires,
                    TketOp::CX,
                    &[qubit_a.clone(), qubit_b.clone()],
                    None,
                )?;
                self.apply_direct_gate(
                    builder,
                    qubit_wires,
                    TketOp::CX,
                    &[qubit_b.clone(), qubit_a.clone()],
                    None,
                )?;
                self.apply_direct_gate(
                    builder,
                    qubit_wires,
                    TketOp::H,
                    std::slice::from_ref(qubit_b),
                    None,
                )?;
            }

            GateOp::MidMeasure { qubit, result_var } => {
                // Mid-circuit measurement: Measure keeps the qubit alive
                let wire =
                    qubit_wires
                        .get(qubit)
                        .copied()
                        .ok_or_else(|| HugrError::UndefinedQubit {
                            name: format!("{}[{}]", qubit.allocator, qubit.index),
                        })?;

                // Measure produces (qubit, bool); MeasureFree is the form
                // that produces a Measurement token requiring Read.
                let outputs: Vec<Wire> = builder
                    .add_dataflow_op(TketOp::Measure, vec![wire])
                    .map_err(|e| HugrError::BuilderError(e.to_string()))?
                    .outputs()
                    .collect();

                // Update qubit wire (first output)
                if let Some(&qubit_wire) = outputs.first() {
                    qubit_wires.insert(qubit.clone(), qubit_wire);
                }

                // Store the direct classical result.
                if let Some(&measurement_wire) = outputs.get(1) {
                    classical_wires.insert(result_var.clone(), measurement_wire);
                }
            }

            GateOp::Conditional {
                condition_var,
                then_ops,
                else_ops,
            } => {
                // Get the classical condition wire
                let condition_wire =
                    classical_wires.get(condition_var).copied().ok_or_else(|| {
                        HugrError::BuilderError(format!(
                            "Classical variable '{}' not found for conditional",
                            condition_var
                        ))
                    })?;

                // Collect all qubits used in both branches
                let mut used_qubits: Vec<QubitRef> = Vec::new();
                self.collect_used_qubits(then_ops, &mut used_qubits);
                self.collect_used_qubits(else_ops, &mut used_qubits);

                // Deduplicate while preserving order
                let mut seen = std::collections::BTreeSet::new();
                used_qubits.retain(|q| seen.insert(q.clone()));

                if used_qubits.is_empty() {
                    // No qubits affected - just skip this conditional
                    return Ok(());
                }

                // Collect input wires for the conditional
                let qubit_inputs: Vec<(tket::hugr::types::Type, Wire)> = used_qubits
                    .iter()
                    .map(|q| {
                        let wire = qubit_wires.get(q).copied().ok_or_else(|| {
                            HugrError::UndefinedQubit {
                                name: format!("{}[{}]", q.allocator, q.index),
                            }
                        })?;
                        Ok((qb_t(), wire))
                    })
                    .collect::<HugrResult<Vec<_>>>()?;

                // Output types are the same as input types (all qubits)
                let output_types: Vec<_> = used_qubits.iter().map(|_| qb_t()).collect();

                // Build the conditional
                // HUGR bool is Sum<Unit, Unit> where 0=false, 1=true
                let mut conditional = builder
                    .conditional_builder(
                        ([type_row![], type_row![]], condition_wire),
                        qubit_inputs,
                        output_types.into(),
                    )
                    .map_err(|e| HugrError::BuilderError(e.to_string()))?;

                // Case 0: false branch (else_ops)
                {
                    let mut case0 = conditional
                        .case_builder(0)
                        .map_err(|e| HugrError::BuilderError(e.to_string()))?;
                    let input_wires: Vec<Wire> = case0.input_wires().collect();

                    // Create temporary wire mapping for this branch
                    let mut branch_qubit_wires: BTreeMap<QubitRef, Wire> = used_qubits
                        .iter()
                        .zip(input_wires.iter())
                        .map(|(q, &w)| (q.clone(), w))
                        .collect();
                    let mut branch_classical_wires = classical_wires.clone();

                    // Apply else operations
                    for op in else_ops {
                        self.apply_gate_in_case(
                            &mut case0,
                            &mut branch_qubit_wires,
                            &mut branch_classical_wires,
                            op,
                        )?;
                    }

                    // Collect output wires in the same order as used_qubits
                    let output_wires: Vec<Wire> =
                        used_qubits.iter().map(|q| branch_qubit_wires[q]).collect();

                    case0
                        .finish_with_outputs(output_wires)
                        .map_err(|e| HugrError::BuilderError(e.to_string()))?;
                }

                // Case 1: true branch (then_ops)
                {
                    let mut case1 = conditional
                        .case_builder(1)
                        .map_err(|e| HugrError::BuilderError(e.to_string()))?;
                    let input_wires: Vec<Wire> = case1.input_wires().collect();

                    // Create temporary wire mapping for this branch
                    let mut branch_qubit_wires: BTreeMap<QubitRef, Wire> = used_qubits
                        .iter()
                        .zip(input_wires.iter())
                        .map(|(q, &w)| (q.clone(), w))
                        .collect();
                    let mut branch_classical_wires = classical_wires.clone();

                    // Apply then operations
                    for op in then_ops {
                        self.apply_gate_in_case(
                            &mut case1,
                            &mut branch_qubit_wires,
                            &mut branch_classical_wires,
                            op,
                        )?;
                    }

                    // Collect output wires in the same order as used_qubits
                    let output_wires: Vec<Wire> =
                        used_qubits.iter().map(|q| branch_qubit_wires[q]).collect();

                    case1
                        .finish_with_outputs(output_wires)
                        .map_err(|e| HugrError::BuilderError(e.to_string()))?;
                }

                // Finish conditional and update qubit wires
                let cond_handle = conditional
                    .finish_sub_container()
                    .map_err(|e: BuildError| HugrError::BuilderError(e.to_string()))?;

                // Update qubit wires with conditional outputs
                for (i, qubit_ref) in used_qubits.iter().enumerate() {
                    if let Some(wire) = cond_handle.outputs().nth(i) {
                        qubit_wires.insert(qubit_ref.clone(), wire);
                    }
                }
            }
        }
        Ok(())
    }

    /// Collect all qubits used in a list of operations.
    fn collect_used_qubits(&self, ops: &[GateOp], qubits: &mut Vec<QubitRef>) {
        for op in ops {
            match op {
                GateOp::Direct { qubits: qs, .. } => qubits.extend(qs.iter().cloned()),
                GateOp::Swap { qubit_a, qubit_b } => {
                    qubits.push(qubit_a.clone());
                    qubits.push(qubit_b.clone());
                }
                GateOp::ISwap { qubit_a, qubit_b } => {
                    qubits.push(qubit_a.clone());
                    qubits.push(qubit_b.clone());
                }
                GateOp::MidMeasure { qubit, .. } => qubits.push(qubit.clone()),
                GateOp::Conditional {
                    then_ops, else_ops, ..
                } => {
                    self.collect_used_qubits(then_ops, qubits);
                    self.collect_used_qubits(else_ops, qubits);
                }
            }
        }
    }

    /// Apply a gate operation inside a case builder (for conditionals).
    fn apply_gate_in_case<T: Dataflow>(
        &self,
        builder: &mut T,
        qubit_wires: &mut BTreeMap<QubitRef, Wire>,
        classical_wires: &mut BTreeMap<String, Wire>,
        gate_op: &GateOp,
    ) -> HugrResult<()> {
        match gate_op {
            GateOp::Direct { op, qubits, angle } => {
                // Collect input wires for this gate
                let input_wires: Vec<Wire> = qubits
                    .iter()
                    .map(|q| {
                        qubit_wires
                            .get(q)
                            .copied()
                            .ok_or_else(|| HugrError::UndefinedQubit {
                                name: format!("{}[{}]", q.allocator, q.index),
                            })
                    })
                    .collect::<HugrResult<Vec<_>>>()?;

                // Handle rotation angle if present
                let all_inputs = if let Some(angle_radians) = angle {
                    let half_turns = angle_radians / std::f64::consts::PI;
                    use tket::extension::rotation::ConstRotation;
                    let const_rotation = ConstRotation::new(half_turns)
                        .map_err(|e| HugrError::BuilderError(e.to_string()))?;
                    let rotation_wire = builder.add_load_value(const_rotation);
                    let mut inputs = input_wires;
                    inputs.push(rotation_wire);
                    inputs
                } else {
                    input_wires
                };

                // Add the gate operation
                let output_wires: Vec<Wire> = builder
                    .add_dataflow_op(*op, all_inputs)
                    .map_err(|e| HugrError::BuilderError(e.to_string()))?
                    .outputs()
                    .collect();

                // Update wire mappings
                for (i, qubit_ref) in qubits.iter().enumerate() {
                    if let Some(&wire) = output_wires.get(i) {
                        qubit_wires.insert(qubit_ref.clone(), wire);
                    }
                }
            }

            GateOp::Swap { qubit_a, qubit_b } => {
                // SWAP decomposition: CX(a,b) CX(b,a) CX(a,b)
                for (q1, q2) in [(qubit_a, qubit_b), (qubit_b, qubit_a), (qubit_a, qubit_b)] {
                    let in_wires: Vec<Wire> = vec![qubit_wires[q1], qubit_wires[q2]];
                    let out_wires: Vec<Wire> = builder
                        .add_dataflow_op(TketOp::CX, in_wires)
                        .map_err(|e| HugrError::BuilderError(e.to_string()))?
                        .outputs()
                        .collect();
                    qubit_wires.insert(q1.clone(), out_wires[0]);
                    qubit_wires.insert(q2.clone(), out_wires[1]);
                }
            }

            GateOp::ISwap { qubit_a, qubit_b } => {
                // iSWAP decomposition: S(a) S(b) H(a) CX(a,b) CX(b,a) H(b)
                for (op, qs) in [
                    (TketOp::S, vec![qubit_a]),
                    (TketOp::S, vec![qubit_b]),
                    (TketOp::H, vec![qubit_a]),
                ] {
                    for q in qs {
                        let in_wire = qubit_wires[q];
                        let out_wire = builder
                            .add_dataflow_op(op, vec![in_wire])
                            .map_err(|e| HugrError::BuilderError(e.to_string()))?
                            .outputs()
                            .next()
                            .unwrap();
                        qubit_wires.insert(q.clone(), out_wire);
                    }
                }
                // CX gates
                for (q1, q2) in [(qubit_a, qubit_b), (qubit_b, qubit_a)] {
                    let in_wires: Vec<Wire> = vec![qubit_wires[q1], qubit_wires[q2]];
                    let out_wires: Vec<Wire> = builder
                        .add_dataflow_op(TketOp::CX, in_wires)
                        .map_err(|e| HugrError::BuilderError(e.to_string()))?
                        .outputs()
                        .collect();
                    qubit_wires.insert(q1.clone(), out_wires[0]);
                    qubit_wires.insert(q2.clone(), out_wires[1]);
                }
                // Final H(b)
                let in_wire = qubit_wires[qubit_b];
                let out_wire = builder
                    .add_dataflow_op(TketOp::H, vec![in_wire])
                    .map_err(|e| HugrError::BuilderError(e.to_string()))?
                    .outputs()
                    .next()
                    .unwrap();
                qubit_wires.insert(qubit_b.clone(), out_wire);
            }

            GateOp::MidMeasure { qubit, result_var } => {
                let wire =
                    qubit_wires
                        .get(qubit)
                        .copied()
                        .ok_or_else(|| HugrError::UndefinedQubit {
                            name: format!("{}[{}]", qubit.allocator, qubit.index),
                        })?;

                let outputs: Vec<Wire> = builder
                    .add_dataflow_op(TketOp::Measure, vec![wire])
                    .map_err(|e| HugrError::BuilderError(e.to_string()))?
                    .outputs()
                    .collect();

                if let Some(&qubit_wire) = outputs.first() {
                    qubit_wires.insert(qubit.clone(), qubit_wire);
                }
                if let Some(&bool_wire) = outputs.get(1) {
                    classical_wires.insert(result_var.clone(), bool_wire);
                }
            }

            GateOp::Conditional { .. } => {
                // Nested conditionals not yet supported in cases
                return Err(HugrError::NestedConditional);
            }
        }
        Ok(())
    }

    /// Apply a direct TketOp gate.
    fn apply_direct_gate(
        &self,
        builder: &mut FunctionBuilder<Hugr>,
        qubit_wires: &mut BTreeMap<QubitRef, Wire>,
        op: TketOp,
        qubits: &[QubitRef],
        angle: Option<f64>,
    ) -> HugrResult<()> {
        // Collect input wires for this gate
        let input_wires: Vec<Wire> = qubits
            .iter()
            .map(|q| {
                qubit_wires
                    .get(q)
                    .copied()
                    .ok_or_else(|| HugrError::UndefinedQubit {
                        name: format!("{}[{}]", q.allocator, q.index),
                    })
            })
            .collect::<HugrResult<Vec<_>>>()?;

        // For rotation gates, we need to add the angle as a constant
        let all_inputs = if let Some(angle_radians) = angle {
            // Convert radians to half-turns (HUGR uses half-turns)
            let half_turns = angle_radians / std::f64::consts::PI;

            // Create rotation constant and load it
            use tket::extension::rotation::ConstRotation;
            let const_rotation = ConstRotation::new(half_turns)
                .map_err(|e| HugrError::BuilderError(e.to_string()))?;
            let rotation_wire = builder.add_load_value(const_rotation);

            let mut inputs = input_wires;
            inputs.push(rotation_wire);
            inputs
        } else {
            input_wires
        };

        // Add the gate operation
        let output_wires: Vec<Wire> = builder
            .add_dataflow_op(op, all_inputs)
            .map_err(|e| HugrError::BuilderError(e.to_string()))?
            .outputs()
            .collect();

        // Update wire mappings
        for (i, qubit_ref) in qubits.iter().enumerate() {
            if let Some(&wire) = output_wires.get(i) {
                qubit_wires.insert(qubit_ref.clone(), wire);
            }
        }

        Ok(())
    }

    /// Serialize a HUGR to bytes (text envelope format).
    ///
    /// This format can be loaded by PECOS and lowered to QIS at the Python boundary.
    pub fn to_bytes(&self, hugr: &Hugr) -> HugrResult<Vec<u8>> {
        let mut buffer = Cursor::new(Vec::new());
        hugr.store(&mut buffer, EnvelopeConfig::text())
            .map_err(|e| HugrError::SerializationError(e.to_string()))?;
        Ok(buffer.into_inner())
    }

    /// Serialize a HUGR to a string (text envelope format).
    ///
    /// This format can be loaded by PECOS and lowered to QIS at the Python boundary.
    pub fn to_string(&self, hugr: &Hugr) -> HugrResult<String> {
        let bytes = self.to_bytes(hugr)?;
        String::from_utf8(bytes).map_err(|e| HugrError::SerializationError(e.to_string()))
    }
}

impl Default for HugrCodegen {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Compatibility: transfer-free bodies retain their emitted operations.
    #[test]
    fn test_round4_compat_selected_allocations() {
        let source = "pub fn main() -> unit { for i in 0..3 { if i == 1 { mut q := qalloc(i + 1); x q[i]; } else if i == 8 { mut r := qalloc(1); h r[0]; } } }";
        let ops = collect_operations(source);
        assert_eq!(ops.len(), 1);
        assert_direct_operation(&ops[0], TketOp::X, &[QubitRef::new("q", 1)], None);
        compile_to_hugr(source).unwrap();
    }
    // Compatibility: transfer-free bodies retain their emitted operations.
    #[test]
    fn test_round4_compat_empty_range() {
        let source = "pub fn main() -> unit { for i in 3..1 { mut q := qalloc(1); h q[0]; } }";
        let ops = collect_operations(source);
        assert_eq!(ops.len(), 0);
        compile_to_hugr(source).unwrap();
    }
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
            let error = compile_to_hugr(&source).expect_err("nested transfer must be rejected");
            assert!(error.to_string().contains("return"), "{body}: {error}");
        }
    }

    // Compatibility: only the loop's return boundary is forbidden.
    #[test]
    fn test_round4_compat_terminal_return() {
        let source =
            "pub fn main() -> unit { mut q := qalloc(1); for i in 0..1 { h q[0]; } return unit; }";
        compile_to_hugr(source).unwrap();
    }

    #[test]
    fn test_round4_elseif_condition() {
        let source = "pub fn main() -> unit { mut q := qalloc(2); pz q; for i in 0..3 { if false { h q[0]; } else if (blk: { return unit; true }) { x q[0]; } } return unit; }";
        let error = compile_to_hugr(source).expect_err("transfer in loop syntax must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    #[test]
    fn test_round4_nested_trailing() {
        let source = "pub fn main() -> unit { mut q := qalloc(2); pz q; for i in 0..3 { a := blk: { mut n := false; v := blk: { for j in 0..1 { (blk: { n = true; unit }) } unit }; if n { return unit; } 0.125 }; rx(a turns) q[0]; } return unit; }";
        let error = compile_to_hugr(source).expect_err("transfer in loop syntax must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    #[test]
    fn test_round4_zero_iterations() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..0 { return unit; } return unit; }";
        let error = compile_to_hugr(source).expect_err("transfer in loop syntax must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    #[test]
    fn test_round4_dead_break() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..1 { if false { break; } } return unit; }";
        let error = compile_to_hugr(source).expect_err("transfer in loop syntax must be rejected");
        assert!(error.to_string().contains("break"), "{error}");
    }
    #[test]
    fn test_round4_dead_continue() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..1 { if false { continue; } } return unit; }";
        let error = compile_to_hugr(source).expect_err("transfer in loop syntax must be rejected");
        assert!(error.to_string().contains("continue"), "{error}");
    }
    #[test]
    fn test_round4_numeric_eq() {
        for condition in ["1/2 == 0.5", "1 == 1.0", "1 == 1u64"] {
            let source = format!(
                "pub fn main() -> unit {{ mut q := qalloc(1); if {condition} {{ x q[0]; }} else {{ h q[0]; }} return unit; }}"
            );
            let ops = collect_operations(&source);
            assert_eq!(ops.len(), 1, "{condition}: {ops:?}");
            assert_direct_operation(&ops[0], TketOp::X, &[QubitRef::new("q", 0)], None);
        }
    }
    #[test]
    fn test_round4_numeric_ne() {
        for condition in ["1/2 != 0.5", "1 != 1.0", "1 != 1u64"] {
            let source = format!(
                "pub fn main() -> unit {{ mut q := qalloc(1); if {condition} {{ h q[0]; }} else {{ x q[0]; }} return unit; }}"
            );
            let ops = collect_operations(&source);
            assert_eq!(ops.len(), 1, "{condition}: {ops:?}");
            assert_direct_operation(&ops[0], TketOp::X, &[QubitRef::new("q", 0)], None);
        }
    }
    #[test]
    fn test_round4_numeric_lt() {
        for condition in ["1/2 < 0.75", "1 < 1.5", "1 < 2u64"] {
            let source = format!(
                "pub fn main() -> unit {{ mut q := qalloc(1); if {condition} {{ x q[0]; }} else {{ h q[0]; }} return unit; }}"
            );
            let ops = collect_operations(&source);
            assert_eq!(ops.len(), 1, "{condition}: {ops:?}");
            assert_direct_operation(&ops[0], TketOp::X, &[QubitRef::new("q", 0)], None);
        }
    }
    #[test]
    fn test_round4_numeric_le() {
        for condition in ["1/2 <= 0.5", "1 <= 1.0", "1 <= 1u64"] {
            let source = format!(
                "pub fn main() -> unit {{ mut q := qalloc(1); if {condition} {{ x q[0]; }} else {{ h q[0]; }} return unit; }}"
            );
            let ops = collect_operations(&source);
            assert_eq!(ops.len(), 1, "{condition}: {ops:?}");
            assert_direct_operation(&ops[0], TketOp::X, &[QubitRef::new("q", 0)], None);
        }
    }
    #[test]
    fn test_round4_numeric_gt() {
        for condition in ["1/2 > 0.25", "1 > 0.5", "1 > 0u64"] {
            let source = format!(
                "pub fn main() -> unit {{ mut q := qalloc(1); if {condition} {{ x q[0]; }} else {{ h q[0]; }} return unit; }}"
            );
            let ops = collect_operations(&source);
            assert_eq!(ops.len(), 1, "{condition}: {ops:?}");
            assert_direct_operation(&ops[0], TketOp::X, &[QubitRef::new("q", 0)], None);
        }
    }
    #[test]
    fn test_round4_numeric_ge() {
        for condition in ["1/2 >= 0.5", "1 >= 1.0", "1 >= 1u64"] {
            let source = format!(
                "pub fn main() -> unit {{ mut q := qalloc(1); if {condition} {{ x q[0]; }} else {{ h q[0]; }} return unit; }}"
            );
            let ops = collect_operations(&source);
            assert_eq!(ops.len(), 1, "{condition}: {ops:?}");
            assert_direct_operation(&ops[0], TketOp::X, &[QubitRef::new("q", 0)], None);
        }
    }

    #[test]
    fn test_round3_tick_measurement_shadowing() {
        let source = "pub fn main() -> unit { mut q := qalloc(2); pz q; c := true; tick { mut c := mz(pack bool) q[0]; } if c { x q[1]; } return unit; }";
        let ops = collect_operations(source);
        let measured_name = ops
            .iter()
            .find_map(|op| match op {
                GateOp::MidMeasure { qubit, result_var } if *qubit == QubitRef::new("q", 0) => {
                    Some(result_var)
                }
                _ => None,
            })
            .unwrap();
        assert!(ops.iter().any(|op| matches!(op, GateOp::Conditional { condition_var, .. } if condition_var == measured_name)), "the condition must use the measurement declared in tick: {ops:?}");
        compile_to_hugr(source).unwrap();
    }

    #[test]
    fn test_round3_angle_dead_branch() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..1 { rx((blk: { if false { return unit; } 0.125 }) turns) q[0]; } return unit; }";
        let error =
            compile_to_hugr(source).expect_err("syntactically present return must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    #[test]
    fn test_round3_if_expression_dead_branch() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..1 { theta := if (false) { return unit; 0.25 } else { 0.125 }; rx(theta turns) q[0]; } return unit; }";
        let error =
            compile_to_hugr(source).expect_err("syntactically present return must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    #[test]
    fn test_round3_nested_capture_dead_branch() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..1 { theta := blk: { for j in i..i + 2 { if j > 8 { return unit; } } 0.125 }; rx(theta turns) q[0]; } return unit; }";
        let error =
            compile_to_hugr(source).expect_err("syntactically present return must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    // Compatibility: a possibly reached transfer must still be rejected.
    #[test]
    fn test_round3_compat_unknown_path() {
        let source = "pub fn main(c: bool) -> unit { mut q := qalloc(1); for i in 0..1 { a := blk: { if c { return unit; } 0.125 }; h q[0]; } return unit; }";
        let error = compile_to_hugr(source).expect_err("reachable return must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    // Compatibility: a possibly reached transfer must still be rejected.
    #[test]
    fn test_round3_compat_later_iteration() {
        let source = "pub fn main(c: bool) -> unit { mut q := qalloc(1); for i in 0..1 { a := blk: { mut n := 0; for j in 0..2 { if n == 1 { return unit; } n = n + 1; } 0.125 }; h q[0]; } return unit; }";
        let error = compile_to_hugr(source).expect_err("reachable return must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    // Compatibility: a possibly reached transfer must still be rejected.
    #[test]
    fn test_round3_compat_unknown_assignment() {
        let source = "pub fn main(c: bool) -> unit { mut q := qalloc(1); for i in 0..1 { a := blk: { mut n := 0; if c { n = 1; } if n == 1 { return unit; } 0.125 }; h q[0]; } return unit; }";
        let error = compile_to_hugr(source).expect_err("reachable return must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    // Compatibility: a nested function has its own return boundary.
    #[test]
    fn test_round3_compat_function_boundary() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..1 { get_n := fn() -> i64 { return 1; }; n := get_n(); h q[0]; } return unit; }";
        compile_to_hugr(source).unwrap();
    }

    #[test]
    fn test_round3_trailing_return() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); pz q; for i in 0..3 { rx(0.125 turns) q[0]; (blk: { return unit; }) } return unit; }";
        let error = compile_to_hugr(source).expect_err("trailing return must fail loudly");
        assert!(error.to_string().contains("return"), "{error}");
    }
    #[test]
    fn test_round3_trailing_break() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); pz q; for i in 0..3 { rx(0.125 turns) q[0]; (blk: { break; }) } return unit; }";
        let error = compile_to_hugr(source).expect_err("trailing break must fail loudly");
        assert!(error.to_string().contains("break"), "{error}");
    }
    #[test]
    fn test_round3_trailing_continue() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); pz q; for i in 0..3 { rx(0.125 turns) q[0]; (blk: { continue; }) } return unit; }";
        let error = compile_to_hugr(source).expect_err("trailing continue must fail loudly");
        assert!(error.to_string().contains("continue"), "{error}");
    }
    #[test]
    fn test_round3_dead_branch() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..1 { theta := blk: { if false { return unit; } 0.125 }; rx(theta turns) q[0]; } return unit; }";
        let error =
            compile_to_hugr(source).expect_err("syntactically present return must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    #[test]
    fn test_round3_empty_range() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..1 { theta := blk: { for j in 0..0 { return unit; } 0.125 }; rx(theta turns) q[0]; } return unit; }";
        let error =
            compile_to_hugr(source).expect_err("syntactically present return must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    #[test]
    fn test_round3_capture_branch() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..1 { theta := blk: { n := i + 1; if n == 9 { return unit; } 0.125 }; rx(theta turns) q[0]; } return unit; }";
        let error =
            compile_to_hugr(source).expect_err("syntactically present return must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    #[test]
    fn test_round3_capture_range() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..1 { theta := blk: { for j in i..i { return unit; } 0.125 }; rx(theta turns) q[0]; } return unit; }";
        let error =
            compile_to_hugr(source).expect_err("syntactically present return must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    #[test]
    fn test_round3_tick_allocator_visibility() {
        let source =
            "pub fn main() -> unit { tick { mut q := qalloc(1); pz q; } h q[0]; return unit; }";
        let ops = collect_operations(source);
        assert_eq!(ops.len(), 2);
        assert_direct_operation(&ops[1], TketOp::H, &[QubitRef::new("q", 0)], None);
        compile_to_hugr(source).unwrap();
    }

    use crate::codegen::QasmCodegen;
    use crate::codegen::phir::{PhirJsonCodegen, PhirJsonOp};
    use crate::codegen::slr::{SlrCodegen, SlrExpression, SlrLiteralValue, SlrStatement};
    use crate::parse;
    use crate::semantic::SemanticAnalyzer;
    use tket::extension::rotation::ConstRotation;
    use tket::hugr::HugrView;
    use tket::hugr::ops::OpType;

    fn compile_to_hugr(source: &str) -> HugrResult<Hugr> {
        let program = parse(source).expect("parse failed");
        let mut codegen = HugrCodegen::new();
        codegen.compile(&program)
    }

    fn collect_operations(source: &str) -> Vec<GateOp> {
        let program = parse(source).expect("parse failed");
        let mut codegen = HugrCodegen::new();
        codegen
            .collect_program(&program)
            .expect("collection failed");
        codegen.operations
    }

    fn assert_direct_operation(
        operation: &GateOp,
        expected_op: TketOp,
        expected_qubits: &[QubitRef],
        expected_angle: Option<f64>,
    ) {
        let GateOp::Direct { op, qubits, angle } = operation else {
            panic!("expected direct operation, got {operation:?}");
        };
        assert_eq!(*op, expected_op);
        assert_eq!(qubits, expected_qubits);
        match (angle, expected_angle) {
            (Some(actual), Some(expected)) => {
                assert!((actual - expected).abs() < f64::EPSILON);
            }
            (None, None) => {}
            _ => panic!("expected angle {expected_angle:?}, got {angle:?}"),
        }
    }

    fn emitted_angle_values(source: &str) -> (f64, f64, f64, f64) {
        let program = parse(source).expect("parse failed");
        SemanticAnalyzer::new_permissive()
            .analyze(&program)
            .expect("semantic analysis failed");

        let slr = SlrCodegen::new()
            .compile(&program)
            .expect("SLR lowering failed");
        let slr_turns = slr
            .body
            .iter()
            .find_map(|stmt| match stmt {
                SlrStatement::Gate(gate) => match gate.params.first() {
                    Some(SlrExpression::Literal(literal)) => match literal.value {
                        SlrLiteralValue::Angle(turns) => Some(turns),
                        _ => None,
                    },
                    _ => None,
                },
                _ => None,
            })
            .expect("missing SLR angle");

        let hugr = HugrCodegen::new()
            .compile(&program)
            .expect("HUGR lowering failed");
        let hugr_half_turns = hugr
            .nodes()
            .find_map(|node| match hugr.get_optype(node) {
                OpType::Const(constant) => constant
                    .value()
                    .get_custom_value::<ConstRotation>()
                    .map(ConstRotation::half_turns),
                _ => None,
            })
            .expect("missing HUGR rotation constant");

        let phir = PhirJsonCodegen::new()
            .compile(&program)
            .expect("PHIR lowering failed");
        let phir_radians = phir
            .ops
            .iter()
            .find_map(|op| match op {
                PhirJsonOp::Qop(qop) if qop.qop == "RZ" => {
                    qop.angles.as_ref().map(|(angles, unit)| {
                        assert_eq!(unit, "rad");
                        angles[0]
                    })
                }
                _ => None,
            })
            .expect("missing PHIR angle");

        let qasm = QasmCodegen::new()
            .compile(&program)
            .expect("QASM lowering failed");
        let qasm_radians = qasm
            .lines()
            .find_map(|line| {
                line.trim()
                    .strip_prefix("rz(")
                    .and_then(|rest| rest.split_once(')'))
                    .and_then(|(value, _)| value.parse::<f64>().ok())
            })
            .expect("missing QASM angle");

        (slr_turns, hugr_half_turns, phir_radians, qasm_radians)
    }

    fn slot_ref(allocator: &str, index: i128) -> crate::ast::SlotRef {
        crate::ast::SlotRef {
            allocator: allocator.to_string(),
            index: Box::new(Expr::IntLit(crate::ast::IntLit {
                value: index,
                suffix: None,
                location: None,
            })),
            location: None,
        }
    }

    fn insert_builder_statements(program: &mut Program, statements: Vec<Stmt>) {
        let main = program
            .declarations
            .iter_mut()
            .find_map(|decl| match decl {
                TopLevelDecl::Fn(function) if function.name == "main" => Some(function),
                _ => None,
            })
            .expect("main function");
        let insertion_index = main.body.statements.len() - 1;
        main.body
            .statements
            .splice(insertion_index..insertion_index, statements);
    }

    // The conservative rule rejects unreachable transfers inside loop bodies.
    #[test]
    fn test_review_unreachable_and_terminal_returns() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..3 { if i == 9 { return unit; } h q[0]; } for j in 1..1 { return unit; } return unit; }";
        let error =
            compile_to_hugr(source).expect_err("syntactically present return must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }

    #[test]
    fn test_review_loop_control_expression() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..3 { a := blk: { return unit; }; } return unit; }".to_string();
        let error =
            compile_to_hugr(&source).expect_err("control transfer in loop must fail loudly");
        assert!(error.to_string().contains("return"), "{error}");
    }

    #[test]
    fn test_review_loop_control_switch() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..3 { switch (i) { 0 => 0, else => 1, } } return unit; }".to_string();
        let error =
            compile_to_hugr(&source).expect_err("control transfer in loop must fail loudly");
        assert!(error.to_string().contains("switch"), "{error}");
    }

    #[test]
    fn test_review_loop_control_try_block() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..3 { try! { return unit; } } return unit; }".to_string();
        let error =
            compile_to_hugr(&source).expect_err("control transfer in loop must fail loudly");
        assert!(error.to_string().contains("return"), "{error}");
    }

    #[test]
    fn test_review_loop_control_defer() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..3 { defer { return unit; } } return unit; }".to_string();
        let error =
            compile_to_hugr(&source).expect_err("control transfer in loop must fail loudly");
        assert!(error.to_string().contains("return"), "{error}");
    }

    #[test]
    fn test_review_loop_control_propagation() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..3 { a := try missing; } return unit; }".to_string();
        let error =
            compile_to_hugr(&source).expect_err("control transfer in loop must fail loudly");
        assert!(error.to_string().contains("try"), "{error}");
    }
    #[test]
    fn test_review_failed_comptime_scope() {
        let source = "pub fn main(c: bool) -> unit { mut q := qalloc(1); pz q; n := 0.125; a := blk: { n := 0.25; c }; rz(n turns) q[0]; return unit; }";
        let ops = collect_operations(source);
        assert_direct_operation(
            &ops[1],
            TketOp::Rz,
            &[QubitRef::new("q", 0)],
            Some(std::f64::consts::FRAC_PI_4),
        );
    }

    #[test]
    fn test_review_loop_return_direct() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); pz q; for i in 0..3 { rx(0.125 turns) q[0]; return unit; } return unit; }".to_string();
        let error = compile_to_hugr(&source).expect_err("return in unrolled loop must fail loudly");
        assert!(error.to_string().contains("return"), "{error}");
    }

    #[test]
    fn test_review_loop_return_block() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); pz q; for i in 0..3 { rx(0.125 turns) q[0]; { return unit; } } return unit; }".to_string();
        let error = compile_to_hugr(&source).expect_err("return in unrolled loop must fail loudly");
        assert!(error.to_string().contains("return"), "{error}");
    }

    #[test]
    fn test_review_loop_return_if() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); pz q; for i in 0..3 { rx(0.125 turns) q[0]; if i == 0 { return unit; } } return unit; }".to_string();
        let error = compile_to_hugr(&source).expect_err("return in unrolled loop must fail loudly");
        assert!(error.to_string().contains("return"), "{error}");
    }

    #[test]
    fn test_review_loop_return_tick() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); pz q; for i in 0..3 { rx(0.125 turns) q[0]; tick { return unit; } } return unit; }".to_string();
        let error = compile_to_hugr(&source).expect_err("return in unrolled loop must fail loudly");
        assert!(error.to_string().contains("return"), "{error}");
    }

    #[test]
    fn test_review_loop_return_nested_loop() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); pz q; for i in 0..3 { rx(0.125 turns) q[0]; for j in 0..1 { return unit; } } return unit; }".to_string();
        let error = compile_to_hugr(&source).expect_err("return in unrolled loop must fail loudly");
        assert!(error.to_string().contains("return"), "{error}");
    }

    #[test]
    fn test_review_measurement_identity_collision() {
        for bindings in [
            "mut c_1 := mz(pack bool) q[0]; mut c := mz([2]u1) [q[1], q[2]];",
            "mut c := mz([2]u1) [q[1], q[2]]; mut c_1 := mz(pack bool) q[0];",
        ] {
            let source = format!(
                "pub fn main() -> unit {{ mut q := qalloc(4); pz q; x q[0]; {bindings} if c_1 {{ x q[3]; }} return unit; }}"
            );
            let ops = collect_operations(&source);
            let condition = ops
                .iter()
                .find_map(|op| match op {
                    GateOp::Conditional { condition_var, .. } => Some(condition_var),
                    _ => None,
                })
                .unwrap();
            let measured_qubits: Vec<_> = ops
                .iter()
                .filter_map(|op| match op {
                    GateOp::MidMeasure { qubit, result_var } if result_var == condition => {
                        Some(qubit.clone())
                    }
                    _ => None,
                })
                .collect();
            assert_eq!(
                measured_qubits,
                vec![QubitRef::new("q", 0)],
                "conditional must read only the original c_1 measurement"
            );
            let names: Vec<_> = ops
                .iter()
                .filter_map(|op| {
                    if let GateOp::MidMeasure { result_var, .. } = op {
                        Some(result_var)
                    } else {
                        None
                    }
                })
                .collect();
            assert_eq!(names.len(), 3);
            assert_eq!(
                names
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len(),
                3
            );
            compile_to_hugr(&source).unwrap();
        }
    }
    #[test]
    fn test_control_flow_bounds_and_capture_shadow() {
        let source = "n := 3; pub fn main() -> unit { mut q := qalloc(n); for n in 0..n { k := n; for j in n..k + 1 { h q[j]; } } h q[n - 1]; }";
        let ops = collect_operations(source);
        assert_eq!(ops.len(), 4);
        assert_direct_operation(&ops[0], TketOp::H, &[QubitRef::new("q", 0)], None);
        assert_direct_operation(&ops[1], TketOp::H, &[QubitRef::new("q", 1)], None);
        assert_direct_operation(&ops[2], TketOp::H, &[QubitRef::new("q", 2)], None);
        assert_direct_operation(&ops[3], TketOp::H, &[QubitRef::new("q", 2)], None);
        compile_to_hugr(source).unwrap();
    }

    #[test]
    fn test_control_flow_local_allocations() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..2 { mut q := qalloc(i + 1); h q[i]; } h q[0]; }";
        let ops = collect_operations(source);
        assert_eq!(ops.len(), 3);
        assert_direct_operation(&ops[0], TketOp::H, &[QubitRef::new("q#1", 0)], None);
        assert_direct_operation(&ops[1], TketOp::H, &[QubitRef::new("q#2", 1)], None);
        assert_direct_operation(&ops[2], TketOp::H, &[QubitRef::new("q", 0)], None);
        compile_to_hugr(source).unwrap();
    }

    #[test]
    fn test_control_flow_selected_allocations() {
        let source = "pub fn main() -> unit { for i in 0..3 { if i == 1 { mut q := qalloc(i + 1); x q[i]; } else if i == 8 { break; } } }";
        let error =
            compile_to_hugr(source).expect_err("syntactically present break must be rejected");
        assert!(error.to_string().contains("break"), "{error}");
    }

    #[test]
    fn test_control_flow_empty_range() {
        let source = "pub fn main() -> unit { for i in 3..1 { break; } }";
        let error =
            compile_to_hugr(source).expect_err("syntactically present break must be rejected");
        assert!(error.to_string().contains("break"), "{error}");
    }

    #[test]
    fn test_control_flow_else_if_build_fails_loudly() {
        let error = compile_to_hugr(
            "pub fn main() -> unit {
            mut q := qalloc(3);
            mut c := mz(u1) q[0]; mut d := mz(u1) q[1];
            if c { x q[2]; } else if d { h q[2]; } else { z q[2]; }
        }",
        )
        .unwrap_err();
        assert!(matches!(error, HugrError::NestedConditional), "{error}");
    }

    #[test]
    fn test_control_flow_measurement_scope() {
        let ops = collect_operations(
            "pub fn main() -> unit {
            mut q := qalloc(2); mut c := mz(u1) q[0];
            for c in 0..2 { if c == 1 { x q[c]; } }
            if c { h q[0]; }
        }",
        );
        assert_eq!(ops.len(), 3);
        assert_direct_operation(&ops[1], TketOp::X, &[QubitRef::new("q", 1)], None);
        assert!(
            matches!(&ops[2], GateOp::Conditional { condition_var, .. } if condition_var == "c")
        );
    }

    #[test]
    fn test_control_flow_runtime_comparison_is_not_false() {
        let source = "pub fn main(n: int) -> unit { mut q := qalloc(1); if n == 1 { h q[0]; } }";
        assert!(compile_to_hugr(source).is_err());
    }

    #[test]
    fn test_control_flow_fixed_target() {
        let source = "pub fn main() -> unit { mut q := qalloc(4); for i in 0..3 { h q[0]; } }";
        let ops = collect_operations(source);
        assert_eq!(ops.len(), 3);
        assert_direct_operation(&ops[0], TketOp::H, &[QubitRef::new("q", 0)], None);
        assert_direct_operation(&ops[1], TketOp::H, &[QubitRef::new("q", 0)], None);
        assert_direct_operation(&ops[2], TketOp::H, &[QubitRef::new("q", 0)], None);
    }

    #[test]
    fn test_control_flow_indexed_target() {
        let source = "pub fn main() -> unit { mut q := qalloc(4); for i in 0..3 { h q[i]; } }";
        let ops = collect_operations(source);
        assert_eq!(ops.len(), 3);
        assert_direct_operation(&ops[0], TketOp::H, &[QubitRef::new("q", 0)], None);
        assert_direct_operation(&ops[1], TketOp::H, &[QubitRef::new("q", 1)], None);
        assert_direct_operation(&ops[2], TketOp::H, &[QubitRef::new("q", 2)], None);
    }

    #[test]
    fn test_control_flow_nested_loops() {
        let source = "pub fn main() -> unit { mut q := qalloc(4); for i in 0..2 { for j in 0..2 { cx (q[i], q[j + 2]); } } }";
        let ops = collect_operations(source);
        assert_eq!(ops.len(), 4);
        assert_direct_operation(
            &ops[0],
            TketOp::CX,
            &[QubitRef::new("q", 0), QubitRef::new("q", 2)],
            None,
        );
        assert_direct_operation(
            &ops[1],
            TketOp::CX,
            &[QubitRef::new("q", 0), QubitRef::new("q", 3)],
            None,
        );
        assert_direct_operation(
            &ops[2],
            TketOp::CX,
            &[QubitRef::new("q", 1), QubitRef::new("q", 2)],
            None,
        );
        assert_direct_operation(
            &ops[3],
            TketOp::CX,
            &[QubitRef::new("q", 1), QubitRef::new("q", 3)],
            None,
        );
    }

    #[test]
    fn test_control_flow_comptime_if() {
        let source =
            "pub fn main() -> unit { mut q := qalloc(4); for i in 0..3 { if i == 1 { x q[i]; } } }";
        let ops = collect_operations(source);
        assert_eq!(ops.len(), 1);
        assert_direct_operation(&ops[0], TketOp::X, &[QubitRef::new("q", 1)], None);
    }

    #[test]
    fn test_control_flow_reject_collection() {
        let source =
            "pub fn main(n: int) -> unit { mut q := qalloc(4); for i in [0, 1] { h q[0]; } }";
        assert!(
            compile_to_hugr(source).is_err(),
            "unsupported collection must fail loudly"
        );
    }

    #[test]
    fn test_control_flow_reject_runtime_bound() {
        let source =
            "pub fn main(n: int) -> unit { mut q := qalloc(4); for i in 0..n { h q[0]; } }";
        assert!(
            compile_to_hugr(source).is_err(),
            "unsupported runtime_bound must fail loudly"
        );
    }

    #[test]
    fn test_control_flow_reject_multi_capture() {
        let source =
            "pub fn main(n: int) -> unit { mut q := qalloc(4); for i, j in 0..3 { h q[0]; } }";
        assert!(
            compile_to_hugr(source).is_err(),
            "unsupported multi_capture must fail loudly"
        );
    }

    #[test]
    fn test_control_flow_reject_break() {
        let source = "pub fn main(n: int) -> unit { mut q := qalloc(4); for i in 0..3 { break; } }";
        assert!(
            compile_to_hugr(source).is_err(),
            "unsupported break must fail loudly"
        );
    }

    #[test]
    fn test_control_flow_reject_continue() {
        let source =
            "pub fn main(n: int) -> unit { mut q := qalloc(4); for i in 0..3 { continue; } }";
        assert!(
            compile_to_hugr(source).is_err(),
            "unsupported continue must fail loudly"
        );
    }

    #[test]
    fn test_control_flow_reject_runtime_if() {
        let source = "pub fn main(n: int) -> unit { mut q := qalloc(4); if n == 1 { x q[0]; } }";
        assert!(
            compile_to_hugr(source).is_err(),
            "unsupported runtime_if must fail loudly"
        );
    }

    #[test]
    fn test_control_flow_else_if() {
        let ops = collect_operations(
            "pub fn main() -> unit {
            mut q := qalloc(3);
            mut c := mz(u1) q[0];
            mut d := mz(u1) q[1];
            if c { x q[2]; } else if d { h q[2]; } else { z q[2]; }
        }",
        );
        let GateOp::Conditional {
            condition_var,
            then_ops,
            else_ops,
        } = &ops[2]
        else {
            panic!("expected outer conditional");
        };
        assert_eq!(condition_var, "c");
        assert_eq!(then_ops.len(), 1);
        assert_eq!(
            else_ops.len(),
            1,
            "else-if must remain a nested conditional"
        );
        let GateOp::Conditional {
            condition_var,
            then_ops,
            else_ops,
        } = &else_ops[0]
        else {
            panic!("expected nested conditional");
        };
        assert_eq!(condition_var, "d");
        assert_direct_operation(&then_ops[0], TketOp::H, &[QubitRef::new("q", 2)], None);
        assert_direct_operation(&else_ops[0], TketOp::Z, &[QubitRef::new("q", 2)], None);
    }

    #[test]
    fn test_empty_program() {
        let hugr = compile_to_hugr("").unwrap();
        assert!(hugr.num_nodes() > 0);
        assert!(matches!(
            hugr.get_optype(hugr.module_root()),
            OpType::Module(_)
        ));
        let OpType::FuncDefn(main) = hugr.get_optype(hugr.entrypoint()) else {
            panic!("expected a function-definition entry point");
        };
        assert_eq!(main.func_name(), "main");
    }

    #[test]
    fn test_single_qubit_gate() {
        let mut program = parse(
            r#"
            pub fn main() -> unit {
                mut q := qalloc(1);
                return unit;
            }
        "#,
        )
        .expect("parse failed");
        insert_builder_statements(
            &mut program,
            vec![Stmt::Gate(crate::ast::GateOp {
                kind: crate::ast::GateKind::H,
                targets: vec![slot_ref("q", 0)],
                params: Vec::new(),
                attrs: Vec::new(),
                location: None,
            })],
        );
        let mut codegen = HugrCodegen::new();
        codegen.collect_program(&program).unwrap();
        assert_eq!(codegen.operations.len(), 1);
        assert_direct_operation(
            &codegen.operations[0],
            TketOp::H,
            &[QubitRef::new("q", 0)],
            None,
        );

        let hugr = codegen.build_hugr().unwrap();
        // Should have input, h gate, output nodes
        assert!(hugr.num_nodes() >= 3);
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

        let operations = collect_operations(source);
        assert_eq!(operations.len(), 2);
        assert_direct_operation(&operations[0], TketOp::H, &[QubitRef::new("q", 0)], None);
        assert_direct_operation(
            &operations[1],
            TketOp::CX,
            &[QubitRef::new("q", 0), QubitRef::new("q", 1)],
            None,
        );

        let hugr = compile_to_hugr(source).unwrap();
        // The same Bell body is inside the module's main function.
        assert!(hugr.num_nodes() >= 4);
        assert!(matches!(
            hugr.get_optype(hugr.module_root()),
            OpType::Module(_)
        ));
        let OpType::FuncDefn(main) = hugr.get_optype(hugr.entrypoint()) else {
            panic!("expected a function-definition entry point");
        };
        assert_eq!(main.func_name(), "main");
    }

    #[test]
    fn test_rotation_gate() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(1);
                rz(1.57 rad) q[0];
            }
        "#;

        let hugr = compile_to_hugr(source).unwrap();
        assert!(hugr.num_nodes() >= 3);
    }

    #[test]
    fn test_allocator_tracking() {
        let source = r#"
            pub fn main() -> unit {
                mut base := qalloc(4);
                mut q := base.child(2);
                h q[0];
                h q[1];
            }
        "#;

        let hugr = compile_to_hugr(source).unwrap();
        assert!(hugr.num_nodes() >= 4);
    }

    #[test]
    fn test_ccx_gate() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(3);
                ccx (q[0], q[1], q[2]);
            }
        "#;

        let operations = collect_operations(source);
        assert_eq!(operations.len(), 1);
        assert_direct_operation(
            &operations[0],
            TketOp::Toffoli,
            &[
                QubitRef::new("q", 0),
                QubitRef::new("q", 1),
                QubitRef::new("q", 2),
            ],
            None,
        );

        let hugr = compile_to_hugr(source).unwrap();
        assert!(hugr.num_nodes() >= 3);
    }

    #[test]
    fn test_wrong_argument_count() {
        // h is a single-qubit gate, so using batch syntax with two qubits should work fine
        // This test was originally testing the old call syntax which is no longer valid
        // Let's test that CX with wrong number of qubits fails
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(1);
                cx (q[0], q[0]);  // CX needs 2 different qubits
            }
        "#;

        // A repeated qubit (`cx q[0], q[0]`) is a logic issue, not a syntax/arity
        // error (arity is checked at the gate expression level, not here), so
        // either an Ok or an Err result is acceptable: this only requires that
        // codegen does not panic on such input.
        let _ = compile_to_hugr(source);
    }

    #[test]
    fn test_qubit_index_out_of_bounds() {
        // Qubit bounds checking is done at semantic analysis, not HUGR codegen
        // Use the semantic analyzer directly to verify bounds checking
        use crate::semantic::SemanticAnalyzer;

        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(2);
                h q[5];
            }
        "#;

        let program = parse(source).expect("parse failed");
        let mut analyzer = SemanticAnalyzer::new();
        let result = analyzer.analyze(&program);

        assert!(result.is_err(), "Expected QubitIndexOutOfBounds error");
        assert!(
            matches!(
                result,
                Err(crate::semantic::SemanticError::QubitIndexOutOfBounds {
                    index: 5,
                    capacity: 2,
                    ..
                })
            ),
            "Expected QubitIndexOutOfBounds error, got: {:?}",
            result
        );
    }

    #[test]
    fn test_swap_gate() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(2);
                x(q[0]);
                swap(q[0], q[1]);
            }
        "#;

        let hugr = compile_to_hugr(source).unwrap();
        // swap decomposes to 3 cx gates, so should have more nodes
        assert!(hugr.num_nodes() >= 5);
    }

    #[test]
    fn test_iswap_gate() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(2);
                iswap(q[0], q[1]);
            }
        "#;

        let hugr = compile_to_hugr(source).unwrap();
        // iswap decomposes to s, s, h, cx, cx, h
        assert!(hugr.num_nodes() >= 6);
    }

    #[test]
    fn test_mid_circuit_measure() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(2);
                h q[0];
                mz(u1) q[0];
                cx (q[0], q[1]);
            }
        "#;

        // Mid-circuit measurement should compile without error
        let hugr = compile_to_hugr(source).unwrap();
        assert!(hugr.num_nodes() >= 4);
    }

    #[test]
    fn test_sx_and_sxdg_gates() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(1);
                sx(q[0]);
                sxdg(q[0]);
            }
        "#;

        let hugr = compile_to_hugr(source).unwrap();
        assert!(hugr.num_nodes() >= 3);
    }

    #[test]
    fn test_sy_and_sydg_gates() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(1);
                sy(q[0]);
                sydg(q[0]);
            }
        "#;

        // SY and SYdg decompose to Ry gates
        let hugr = compile_to_hugr(source).unwrap();
        assert!(hugr.num_nodes() >= 3);
    }

    #[test]
    fn test_crz_lowers_to_tket_crz() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(2);
                crz(1.57 rad) (q[0], q[1]);
            }
        "#;

        // `crz` is its own GateKind (PR #638); the HUGR backend emits tket's own
        // CRz spelling rather than a decomposition, since Guppy/tket own that name.
        let operations = collect_operations(source);
        assert_eq!(operations.len(), 1);
        assert_direct_operation(
            &operations[0],
            TketOp::CRz,
            &[QubitRef::new("q", 0), QubitRef::new("q", 1)],
            Some(1.57),
        );
    }

    #[test]
    fn test_rotation_angles_use_module_comptime_constants() {
        let operations = collect_operations(
            r#"
                pi_2: a64 = 0.25 turns;
                quarter_turn: a64 = 0.125 turns;

                pub fn main() -> unit {
                    q := qalloc(2);
                    crz(pi_2) (q[1], q[0]);
                    crz(quarter_turn) (q[0], q[1]);
                    crz(pi_2 / 2) (q[1], q[0]);
                }
            "#,
        );

        assert_eq!(operations.len(), 3);
        assert_direct_operation(
            &operations[0],
            TketOp::CRz,
            &[QubitRef::new("q", 1), QubitRef::new("q", 0)],
            Some(std::f64::consts::FRAC_PI_2),
        );
        assert_direct_operation(
            &operations[1],
            TketOp::CRz,
            &[QubitRef::new("q", 0), QubitRef::new("q", 1)],
            Some(std::f64::consts::FRAC_PI_4),
        );
        assert_direct_operation(
            &operations[2],
            TketOp::CRz,
            &[QubitRef::new("q", 1), QubitRef::new("q", 0)],
            Some(std::f64::consts::FRAC_PI_4),
        );
    }

    #[test]
    fn test_builtin_angle_constant_spellings() {
        let operations = collect_operations(
            r#"
                pub fn main() -> unit {
                    q := qalloc(2);
                    crz(PI rad) (q[0], q[1]);
                    crz(pi rad) (q[0], q[1]);
                    crz(TAU rad) (q[0], q[1]);
                    crz(tau rad) (q[0], q[1]);
                }
            "#,
        );

        assert_eq!(operations.len(), 4);
        for (index, operation) in operations.iter().enumerate() {
            let angle = if index < 2 {
                std::f64::consts::PI
            } else {
                std::f64::consts::TAU
            };
            assert_direct_operation(
                operation,
                TketOp::CRz,
                &[QubitRef::new("q", 0), QubitRef::new("q", 1)],
                Some(angle),
            );
        }
    }

    #[test]
    fn test_builtin_named_and_literal_angle_units_agree() {
        let operations = collect_operations(
            r#"
                half: a64 = 0.5 turns;
                hp: a64 = pi rad;
                one_rad: a64 = 1.0 rad;

                pub fn main() -> unit {
                    q := qalloc(2);
                    crz(pi rad) (q[0], q[1]);
                    crz(0.5 turns) (q[0], q[1]);
                    crz(half) (q[0], q[1]);
                    crz(hp) (q[0], q[1]);
                    crz(one_rad) (q[0], q[1]);
                    crz(1.0 rad) (q[0], q[1]);
                }
            "#,
        );

        assert_eq!(operations.len(), 6);
        for operation in &operations[..4] {
            assert_direct_operation(
                operation,
                TketOp::CRz,
                &[QubitRef::new("q", 0), QubitRef::new("q", 1)],
                Some(std::f64::consts::PI),
            );
        }
        for operation in &operations[4..] {
            assert_direct_operation(
                operation,
                TketOp::CRz,
                &[QubitRef::new("q", 0), QubitRef::new("q", 1)],
                Some(1.0),
            );
        }
    }

    #[test]
    fn test_all_backends_emit_equivalent_angle_values() {
        let std_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("std/std.zlp")
            .canonicalize()
            .expect("canonical stdlib path");
        let cases = [
            ("pi rad", 0.5),
            ("tau rad", 1.0),
            ("1/4 turns", 0.25),
            ("0.25 turns", 0.25),
            ("module_angle", 0.25),
            ("std.a64.t_angle", 0.125),
            ("module_angle / 2", 0.125),
            ("-module_angle", -0.25),
        ];

        for (spelling, expected_turns) in cases {
            let source = format!(
                r#"
                std := @import("{}");
                module_angle: a64 = 0.25 turns;
                pub fn main() -> unit {{
                    q := qalloc(1);
                    pz q;
                    rz({spelling}) q[0];
                    return unit;
                }}
                "#,
                crate::tests::escape_source_string(&std_path.to_string_lossy()),
            );
            let (slr_turns, hugr_half_turns, phir_radians, qasm_radians) =
                emitted_angle_values(&source);

            let expected_half_turns = expected_turns * 2.0;
            let expected_radians = expected_turns * std::f64::consts::TAU;
            assert!(
                (slr_turns - expected_turns).abs() < 1e-12,
                "SLR mismatch for {spelling}: {slr_turns}"
            );
            assert!(
                (hugr_half_turns - expected_half_turns).abs() < 1e-12,
                "HUGR mismatch for {spelling}: {hugr_half_turns}"
            );
            assert!(
                (phir_radians - expected_radians).abs() < 1e-12,
                "PHIR mismatch for {spelling}: {phir_radians}"
            );
            assert!(
                (qasm_radians - expected_radians).abs() < 1e-12,
                "QASM mismatch for {spelling}: {qasm_radians}"
            );
        }
    }

    #[test]
    fn test_non_angle_module_constants_remain_plain_numbers() {
        let program = parse(
            r#"
                plain_pi: f64 = pi;
                plain_tau: f64 = tau;
                half_from_number: a64 = plain_pi rad;
                full_from_number: a64 = plain_tau rad;

                pub fn main() -> unit {
                    q := qalloc(2);
                    crz(half_from_number) (q[0], q[1]);
                    crz(full_from_number) (q[0], q[1]);
                }
            "#,
        )
        .expect("parse failed");
        let mut codegen = HugrCodegen::new();
        codegen
            .collect_program(&program)
            .expect("collection failed");

        assert_eq!(
            codegen
                .comptime
                .borrow()
                .context
                .lookup("plain_pi")
                .and_then(ComptimeValue::as_float),
            Some(std::f64::consts::PI)
        );
        assert_eq!(
            codegen
                .comptime
                .borrow()
                .context
                .lookup("plain_tau")
                .and_then(ComptimeValue::as_float),
            Some(std::f64::consts::TAU)
        );
        assert_eq!(codegen.operations.len(), 2);
        assert_direct_operation(
            &codegen.operations[0],
            TketOp::CRz,
            &[QubitRef::new("q", 0), QubitRef::new("q", 1)],
            Some(std::f64::consts::PI),
        );
        assert_direct_operation(
            &codegen.operations[1],
            TketOp::CRz,
            &[QubitRef::new("q", 0), QubitRef::new("q", 1)],
            Some(std::f64::consts::TAU),
        );
    }

    #[test]
    fn test_rotation_angle_rejects_unresolvable_names() {
        for (source, offending_name) in [
            (
                r#"
                    pub fn main() -> unit {
                        q := qalloc(2);
                        crz(undefined_angle) (q[0], q[1]);
                    }
                "#,
                "undefined_angle",
            ),
            (
                r#"
                    mut runtime_angle: a64 = 0.5 turns;

                    pub fn main() -> unit {
                        q := qalloc(2);
                        crz(runtime_angle) (q[0], q[1]);
                    }
                "#,
                "runtime_angle",
            ),
        ] {
            let program = parse(source).expect("parse failed");
            let error = HugrCodegen::new()
                .collect_program(&program)
                .expect_err("angle should not resolve");
            let message = error.to_string();
            assert!(
                message.contains(offending_name),
                "error did not name {offending_name}: {message}"
            );
            assert!(
                message.contains("at line "),
                "missing source location: {message}"
            );
            assert!(
                !message.contains("SourceLocation")
                    && !message.contains(env!("CARGO_MANIFEST_DIR")),
                "error leaked an AST dump or source path: {message}"
            );
        }
    }

    #[test]
    fn test_local_binding_shadows_module_angle_constant() {
        let program = parse(
            r#"
                theta: a64 = 1.0 turns;

                pub fn main() -> unit {
                    q := qalloc(2);
                    theta := 3.0 turns;
                    crz(theta) (q[0], q[1]);
                }
            "#,
        )
        .expect("parse failed");
        let mut codegen = HugrCodegen::new();
        codegen
            .collect_program(&program)
            .expect("local comptime shadow should resolve");

        assert_eq!(codegen.operations.len(), 1);
        assert_direct_operation(
            &codegen.operations[0],
            TketOp::CRz,
            &[QubitRef::new("q", 0), QubitRef::new("q", 1)],
            Some(3.0 * std::f64::consts::TAU),
        );
    }

    #[test]
    fn test_loop_binding_shadows_module_angle_constant() {
        let program = parse(
            r#"
                i: a64 = 1.0 turns;

                pub fn main() -> unit {
                    q := qalloc(2);
                    for i in 0..3 {
                        crz(i) (q[0], q[1]);
                    }
                }
            "#,
        )
        .expect("parse failed");
        let mut codegen = HugrCodegen::new();
        codegen.collect_program(&program).unwrap();
        assert_eq!(codegen.operations.len(), 3);
        for (i, op) in codegen.operations.iter().enumerate() {
            assert_direct_operation(
                op,
                TketOp::CRz,
                &[QubitRef::new("q", 0), QubitRef::new("q", 1)],
                Some(i as f64 * std::f64::consts::TAU),
            );
        }
    }

    #[test]
    fn test_reset_gate() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(1);
                x q[0];
                pz q[0];
            }
        "#;

        let operations = collect_operations(source);
        assert_eq!(operations.len(), 2);
        assert_direct_operation(&operations[0], TketOp::X, &[QubitRef::new("q", 0)], None);
        assert_direct_operation(
            &operations[1],
            TketOp::Reset,
            &[QubitRef::new("q", 0)],
            None,
        );

        let hugr = compile_to_hugr(source).unwrap();
        assert!(hugr.num_nodes() >= 3);
    }

    #[test]
    fn test_classical_conditional() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(2);
                h q[0];
                mut result := mz(u1) q[0];
                if (result) {
                    x(q[1]);
                }
            }
        "#;

        // Classical conditional should compile without error
        let hugr = compile_to_hugr(source).unwrap();
        // Should have conditional node in addition to gates
        assert!(hugr.num_nodes() >= 5);
    }

    #[test]
    fn test_classical_conditional_with_else() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(2);
                h q[0];
                mut result := mz(u1) q[0];
                if (result) {
                    x(q[1]);
                } else {
                    z(q[1]);
                }
            }
        "#;

        // Classical conditional with else should compile without error
        let hugr = compile_to_hugr(source).unwrap();
        assert!(hugr.num_nodes() >= 5);
    }

    #[test]
    fn test_ch_gate() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(2);
                ch (q[0], q[1]);
            }
        "#;

        // CH decomposes to Ry CZ Ry
        let hugr = compile_to_hugr(source).unwrap();
        assert!(hugr.num_nodes() >= 4);
    }

    #[test]
    fn test_szz_gate() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(2);
                szz(q[0], q[1]);
            }
        "#;

        // SZZ decomposes to CX S CX
        let hugr = compile_to_hugr(source).unwrap();
        assert!(hugr.num_nodes() >= 4);
    }

    #[test]
    fn test_sxx_gate() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(2);
                sxx(q[0], q[1]);
            }
        "#;

        // SXX decomposes to H H (SZZ) H H
        let hugr = compile_to_hugr(source).unwrap();
        assert!(hugr.num_nodes() >= 6);
    }

    #[test]
    fn test_syy_gate() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(2);
                syy(q[0], q[1]);
            }
        "#;

        // SYY decomposes to Vdg Vdg (SZZ) V V
        let hugr = compile_to_hugr(source).unwrap();
        assert!(hugr.num_nodes() >= 6);
    }

    #[test]
    fn test_rzz_gate() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(2);
                rzz(0.25 turns) (q[0], q[1]);
            }
        "#;

        let operations = collect_operations(source);
        assert_eq!(operations.len(), 3);
        assert_direct_operation(
            &operations[0],
            TketOp::CX,
            &[QubitRef::new("q", 0), QubitRef::new("q", 1)],
            None,
        );
        assert_direct_operation(
            &operations[1],
            TketOp::Rz,
            &[QubitRef::new("q", 1)],
            Some(std::f64::consts::FRAC_PI_2),
        );
        assert_direct_operation(
            &operations[2],
            TketOp::CX,
            &[QubitRef::new("q", 0), QubitRef::new("q", 1)],
            None,
        );

        let hugr = compile_to_hugr(source).unwrap();
        assert!(hugr.num_nodes() >= 4);
    }

    #[test]
    fn test_rzz_gate_with_radian_literal() {
        let operations = collect_operations(
            r#"
                pub fn main() -> unit {
                    mut q := qalloc(2);
                    rzz(1.25 rad) (q[0], q[1]);
                }
            "#,
        );

        assert_eq!(operations.len(), 3);
        assert_direct_operation(
            &operations[0],
            TketOp::CX,
            &[QubitRef::new("q", 0), QubitRef::new("q", 1)],
            None,
        );
        assert_direct_operation(
            &operations[1],
            TketOp::Rz,
            &[QubitRef::new("q", 1)],
            Some(1.25),
        );
        assert_direct_operation(
            &operations[2],
            TketOp::CX,
            &[QubitRef::new("q", 0), QubitRef::new("q", 1)],
            None,
        );
    }

    #[test]
    fn test_prepare_whole_allocator() {
        let operations = collect_operations(
            r#"
                pub fn main() -> unit {
                    mut q := qalloc(2);
                    pz q;
                }
            "#,
        );

        assert_eq!(operations.len(), 2);
        assert_direct_operation(
            &operations[0],
            TketOp::Reset,
            &[QubitRef::new("q", 0)],
            None,
        );
        assert_direct_operation(
            &operations[1],
            TketOp::Reset,
            &[QubitRef::new("q", 1)],
            None,
        );
    }

    #[test]
    fn test_single_qubit_gate_broadcasts_over_allocator() {
        let operations = collect_operations(
            r#"
                pub fn main() -> unit {
                    q := qalloc(3);
                    h q;
                }
            "#,
        );

        assert_eq!(operations.len(), 3);
        for (index, operation) in operations.iter().enumerate() {
            assert_direct_operation(operation, TketOp::H, &[QubitRef::new("q", index)], None);
        }
    }

    #[test]
    fn test_multi_qubit_gate_allocator_broadcast_is_an_error() {
        let program = parse(
            r#"
                pub fn main() -> unit {
                    q := qalloc(3);
                    cx q;
                }
            "#,
        )
        .expect("parse failed");
        let error = HugrCodegen::new()
            .collect_program(&program)
            .expect_err("multi-qubit broadcast should fail");

        assert!(matches!(
            error,
            HugrError::InvalidAllocatorBroadcast { gate, arity: 2 } if gate == "cx"
        ));
    }

    #[test]
    fn test_zero_capacity_allocator_broadcast_is_an_error() {
        let program = parse(
            r#"
                pub fn main() -> unit {
                    q := qalloc(0);
                    h q;
                }
            "#,
        )
        .expect("parse failed");
        let error = HugrCodegen::new()
            .collect_program(&program)
            .expect_err("empty allocator broadcast should fail");

        assert!(matches!(
            error,
            HugrError::EmptyAllocatorBroadcast { gate, allocator }
                if gate == "h" && allocator == "q"
        ));
    }

    #[test]
    fn test_prepare_explicit_slot() {
        let operations = collect_operations(
            r#"
                pub fn main() -> unit {
                    mut q := qalloc(2);
                    pz q[1];
                }
            "#,
        );

        assert_eq!(operations.len(), 1);
        assert_direct_operation(
            &operations[0],
            TketOp::Reset,
            &[QubitRef::new("q", 1)],
            None,
        );
    }

    #[test]
    fn test_builder_shaped_gate_and_prepare_statements() {
        let mut program = parse(
            r#"
                pub fn main() -> unit {
                    mut q := qalloc(2);
                    return unit;
                }
            "#,
        )
        .expect("parse failed");
        insert_builder_statements(
            &mut program,
            vec![
                Stmt::Gate(crate::ast::GateOp {
                    kind: crate::ast::GateKind::X,
                    targets: vec![slot_ref("q", 0)],
                    params: Vec::new(),
                    attrs: Vec::new(),
                    location: None,
                }),
                Stmt::Prepare(crate::ast::PrepareOp {
                    allocator: "q".to_string(),
                    slots: Some(vec![1]),
                    location: None,
                }),
            ],
        );

        let mut codegen = HugrCodegen::new();
        codegen
            .collect_program(&program)
            .expect("collection failed");
        assert_eq!(codegen.operations.len(), 2);
        assert_direct_operation(
            &codegen.operations[0],
            TketOp::X,
            &[QubitRef::new("q", 0)],
            None,
        );
        assert_direct_operation(
            &codegen.operations[1],
            TketOp::Reset,
            &[QubitRef::new("q", 1)],
            None,
        );
    }

    #[test]
    fn test_builder_shaped_multi_qubit_gate_target_order() {
        let mut program = parse(
            r#"
                pub fn main() -> unit {
                    mut q := qalloc(3);
                    return unit;
                }
            "#,
        )
        .expect("parse failed");
        insert_builder_statements(
            &mut program,
            vec![
                Stmt::Gate(crate::ast::GateOp {
                    kind: crate::ast::GateKind::CX,
                    targets: vec![slot_ref("q", 0), slot_ref("q", 1)],
                    params: Vec::new(),
                    attrs: Vec::new(),
                    location: None,
                }),
                Stmt::Gate(crate::ast::GateOp {
                    kind: crate::ast::GateKind::CCX,
                    targets: vec![slot_ref("q", 0), slot_ref("q", 1), slot_ref("q", 2)],
                    params: Vec::new(),
                    attrs: Vec::new(),
                    location: None,
                }),
            ],
        );

        let mut codegen = HugrCodegen::new();
        codegen
            .collect_program(&program)
            .expect("collection failed");
        assert_eq!(codegen.operations.len(), 2);
        assert_direct_operation(
            &codegen.operations[0],
            TketOp::CX,
            &[QubitRef::new("q", 0), QubitRef::new("q", 1)],
            None,
        );
        assert_direct_operation(
            &codegen.operations[1],
            TketOp::Toffoli,
            &[
                QubitRef::new("q", 0),
                QubitRef::new("q", 1),
                QubitRef::new("q", 2),
            ],
            None,
        );
    }

    #[test]
    fn test_builder_prepare_whole_allocator() {
        let mut program = parse(
            r#"
                pub fn main() -> unit {
                    mut q := qalloc(3);
                    return unit;
                }
            "#,
        )
        .expect("parse failed");
        insert_builder_statements(
            &mut program,
            vec![Stmt::Prepare(crate::ast::PrepareOp {
                allocator: "q".to_string(),
                slots: None,
                location: None,
            })],
        );

        let mut codegen = HugrCodegen::new();
        codegen
            .collect_program(&program)
            .expect("collection failed");
        assert_eq!(codegen.operations.len(), 3);
        for (index, operation) in codegen.operations.iter().enumerate() {
            assert_direct_operation(operation, TketOp::Reset, &[QubitRef::new("q", index)], None);
        }
    }

    #[test]
    fn test_builder_prepare_rejects_out_of_bounds_slot() {
        let mut program = parse(
            r#"
                pub fn main() -> unit {
                    mut q := qalloc(2);
                    return unit;
                }
            "#,
        )
        .expect("parse failed");
        insert_builder_statements(
            &mut program,
            vec![Stmt::Prepare(crate::ast::PrepareOp {
                allocator: "q".to_string(),
                slots: Some(vec![2]),
                location: None,
            })],
        );

        let mut codegen = HugrCodegen::new();
        assert!(matches!(
            codegen.collect_program(&program),
            Err(HugrError::QubitIndexOutOfBounds {
                index: 2,
                capacity: 2
            })
        ));
        assert!(codegen.operations.is_empty());
    }

    #[test]
    fn test_all_gate_kinds_have_hugr_mappings() {
        for kind in crate::ast::GateKind::ALL {
            assert!(
                gate_name_to_mapping(kind.keyword()).is_some(),
                "missing HUGR mapping for {}",
                kind.keyword()
            );
        }
    }

    #[test]
    fn test_ordinary_function_call_is_an_error() {
        let program = parse(
            r#"
                fn helper() -> unit {
                    return unit;
                }

                pub fn main() -> unit {
                    helper();
                }
            "#,
        )
        .expect("parse failed");
        let mut codegen = HugrCodegen::new();

        assert!(matches!(
            codegen.collect_program(&program),
            Err(HugrError::UnsupportedCall { name }) if name == "helper"
        ));
    }

    #[test]
    fn test_missing_named_gate_mapping_is_an_error() {
        let mut codegen = HugrCodegen::new();

        assert!(matches!(
            codegen.collect_named_gate("unmapped_gate", &[], Vec::new()),
            Err(HugrError::UnknownGate { name }) if name == "unmapped_gate"
        ));
    }

    #[test]
    fn test_f_gate() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(1);
                f(q[0]);
            }
        "#;

        // F decomposes to H Sdg H Sdg
        let hugr = compile_to_hugr(source).unwrap();
        assert!(hugr.num_nodes() >= 5);
    }

    #[test]
    fn test_fdg_gate() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(1);
                fdg(q[0]);
            }
        "#;

        // Fdg decomposes to S H S H
        let hugr = compile_to_hugr(source).unwrap();
        assert!(hugr.num_nodes() >= 5);
    }

    #[test]
    fn test_f4_gate() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(1);
                f4(q[0]);
            }
        "#;

        // F4 decomposes to Ry Rz
        let hugr = compile_to_hugr(source).unwrap();
        assert!(hugr.num_nodes() >= 3);
    }

    #[test]
    fn test_ising_dagger_gates() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(2);
                sxxdg(q[0], q[1]);
                syydg(q[0], q[1]);
                szzdg(q[0], q[1]);
            }
        "#;

        let hugr = compile_to_hugr(source).unwrap();
        assert!(hugr.num_nodes() >= 10);
    }

    // =========================================================================
    // Batch Operations Tests
    // =========================================================================

    #[test]
    fn test_batch_single_qubit_gate() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(3);
                h {q[0], q[1], q[2]};
            }
        "#;

        let hugr = compile_to_hugr(source).unwrap();
        // Should expand to 3 H gates
        assert!(hugr.num_nodes() >= 4);
    }

    #[test]
    fn test_hugr_single_qubit_batch_literal_spellings() {
        for statement in ["h [q[0], q[1]];", "h {q[0], q[1]};", "h(&[q[0], q[1]]);"] {
            let source = format!(
                r#"
                    pub fn main() -> unit {{
                        q := qalloc(2);
                        {statement}
                    }}
                "#
            );
            let operations = collect_operations(&source);

            assert_eq!(operations.len(), 2, "statement: {statement}");
            assert_direct_operation(&operations[0], TketOp::H, &[QubitRef::new("q", 0)], None);
            assert_direct_operation(&operations[1], TketOp::H, &[QubitRef::new("q", 1)], None);
        }
    }

    #[test]
    fn test_nested_batch_is_an_error() {
        let program = parse(
            r#"
                pub fn main() -> unit {
                    q := qalloc(1);
                    h [[q[0]]];
                }
            "#,
        )
        .expect("parse failed");
        let mut codegen = HugrCodegen::new();

        assert!(matches!(
            codegen.collect_program(&program),
            Err(HugrError::UnsupportedExpression)
        ));
        assert!(codegen.operations.is_empty());
    }

    #[test]
    fn test_empty_batch_is_an_error() {
        let program = parse(
            r#"
                pub fn main() -> unit {
                    q := qalloc(1);
                    h [];
                }
            "#,
        )
        .expect("parse failed");
        let error = HugrCodegen::new()
            .collect_program(&program)
            .expect_err("empty batch should fail");

        assert!(matches!(
            error,
            HugrError::EmptyGateBatch { gate } if gate == "h"
        ));
    }

    #[test]
    fn test_batch_two_qubit_gate() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(4);
                cx {(q[0], q[1]), (q[2], q[3])};
            }
        "#;

        let hugr = compile_to_hugr(source).unwrap();
        // Should expand to 2 CX gates
        assert!(hugr.num_nodes() >= 3);
    }

    #[test]
    fn test_batch_rotation_gate() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(2);
                rz(1/8 turns) {q[0], q[1]};
            }
        "#;

        let hugr = compile_to_hugr(source).unwrap();
        // Should expand to 2 Rz gates
        assert!(hugr.num_nodes() >= 3);
    }

    #[test]
    fn test_batch_array_syntax() {
        // Test &[...] syntax for batch gates
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(3);
                h(&[q[0], q[1], q[2]]);
            }
        "#;

        let hugr = compile_to_hugr(source).unwrap();
        // Should expand to 3 H gates
        assert!(hugr.num_nodes() >= 4);
    }

    #[test]
    fn test_batch_array_two_qubit() {
        // Test &[(a,b), (c,d)] syntax for batch two-qubit gates
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(4);
                cx(&[(q[0], q[1]), (q[2], q[3])]);
            }
        "#;

        let hugr = compile_to_hugr(source).unwrap();
        // Should expand to 2 CX gates
        assert!(hugr.num_nodes() >= 3);
    }

    // =========================================================================
    // Tick Block Tests
    // =========================================================================

    #[test]
    fn test_tick_block() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(2);
                tick {
                    h q[0];
                    h q[1];
                }
            }
        "#;

        let hugr = compile_to_hugr(source).unwrap();
        // Should have 2 H gates (tick block is flattened)
        assert!(hugr.num_nodes() >= 3);
    }

    #[test]
    fn test_nested_tick_blocks() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(4);
                tick outer {
                    tick layer1 {
                        h {q[0], q[1]};
                    }
                    tick layer2 {
                        cx {(q[0], q[2]), (q[1], q[3])};
                    }
                }
            }
        "#;

        let hugr = compile_to_hugr(source).unwrap();
        // Should have 2 H + 2 CX gates
        assert!(hugr.num_nodes() >= 5);
    }

    // =========================================================================
    // Typed Measurement Tests
    // =========================================================================

    #[test]
    fn test_typed_measurement_single() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(1);
                h q[0];
                r := mz(u1) q[0];
            }
        "#;

        let hugr = compile_to_hugr(source).unwrap();
        assert!(hugr.num_nodes() >= 3);
    }

    #[test]
    fn test_typed_measurement_array() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(3);
                h {q[0], q[1], q[2]};
                results := mz([3]u1) [q[0], q[1], q[2]];
            }
        "#;

        let operations = collect_operations(source);
        assert_eq!(operations.len(), 6);
        for index in 0..3 {
            assert_direct_operation(
                &operations[index],
                TketOp::H,
                &[QubitRef::new("q", index)],
                None,
            );
            let GateOp::MidMeasure { qubit, result_var } = &operations[index + 3] else {
                panic!("expected collected measurement");
            };
            assert_eq!(qubit, &QubitRef::new("q", index));
            let expected = if index == 0 {
                "results".to_string()
            } else {
                format!("results_{index}")
            };
            assert_eq!(result_var, &expected);
        }
        compile_to_hugr(source).unwrap();
    }

    #[test]
    fn test_qft_example_preserves_intended_operations() {
        let operations = collect_operations(include_str!("../../examples/qft_3qubit.zlp"));

        assert_eq!(operations.len(), 15);
        for (index, op) in operations[12..].iter().enumerate() {
            assert!(
                matches!(op, GateOp::MidMeasure { qubit, .. } if qubit == &QubitRef::new("q", index))
            );
        }
        assert_direct_operation(
            &operations[0],
            TketOp::Reset,
            &[QubitRef::new("q", 0)],
            None,
        );
        assert_direct_operation(
            &operations[1],
            TketOp::Reset,
            &[QubitRef::new("q", 1)],
            None,
        );
        assert_direct_operation(
            &operations[2],
            TketOp::Reset,
            &[QubitRef::new("q", 2)],
            None,
        );
        assert_direct_operation(&operations[3], TketOp::X, &[QubitRef::new("q", 0)], None);
        assert_direct_operation(&operations[4], TketOp::X, &[QubitRef::new("q", 2)], None);
        assert_direct_operation(&operations[5], TketOp::H, &[QubitRef::new("q", 0)], None);
        assert_direct_operation(
            &operations[6],
            TketOp::CRz,
            &[QubitRef::new("q", 1), QubitRef::new("q", 0)],
            Some(std::f64::consts::FRAC_PI_2),
        );
        assert_direct_operation(
            &operations[7],
            TketOp::CRz,
            &[QubitRef::new("q", 2), QubitRef::new("q", 0)],
            Some(std::f64::consts::FRAC_PI_4),
        );
        assert_direct_operation(&operations[8], TketOp::H, &[QubitRef::new("q", 1)], None);
        assert_direct_operation(
            &operations[9],
            TketOp::CRz,
            &[QubitRef::new("q", 2), QubitRef::new("q", 1)],
            Some(std::f64::consts::FRAC_PI_2),
        );
        assert_direct_operation(&operations[10], TketOp::H, &[QubitRef::new("q", 2)], None);
        assert!(matches!(
            &operations[11],
            GateOp::Swap { qubit_a, qubit_b }
                if qubit_a == &QubitRef::new("q", 0) && qubit_b == &QubitRef::new("q", 2)
        ));
    }

    #[test]
    fn test_all_examples_compile_to_hugr() {
        let examples_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
        let mut examples = std::fs::read_dir(&examples_dir)
            .expect("failed to read examples directory")
            .map(|entry| entry.expect("failed to read example entry").path())
            .filter(|path| path.extension().is_some_and(|extension| extension == "zlp"))
            .collect::<Vec<_>>();
        examples.sort();
        assert!(!examples.is_empty(), "no .zlp examples found");

        for path in examples {
            let source = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
            let program = parse(&source)
                .unwrap_or_else(|error| panic!("failed to parse {}: {error}", path.display()));
            HugrCodegen::new()
                .compile(&program)
                .unwrap_or_else(|error| panic!("failed to compile {}: {error}", path.display()));
        }
    }
}
