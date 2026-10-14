//! OpenQASM 2.0 code generation for Zlup.
//!
//! This module generates OpenQASM 2.0 from Zlup AST, enabling execution on
//! simulators and hardware that support the QASM format.
//!
//! ## Output Format
//!
//! ```qasm
//! OPENQASM 2.0;
//! include "qelib1.inc";
//!
//! qreg q[4];
//! creg c[4];
//!
//! h q[0];
//! cx q[0], q[1];
//! measure q[0] -> c[0];
//! ```

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt::Write;
use thiserror::Error;

use crate::ast::{
    BinaryOp, Binding, Block, CallExpr, ElseBranch, Expr, FnDecl, ForRange, ForStmt, IfStmt,
    Program, Stmt, TopLevelDecl,
};
use crate::comptime::{
    ComptimeEvaluator, ComptimeValue, angle_evaluator, angle_expression_name,
    define_comptime_binding, resolve_angle_turns,
};

// =============================================================================
// Errors
// =============================================================================

/// QASM code generation errors.
#[derive(Debug, Error)]
pub enum QasmError {
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

    #[error("unsupported expression in QASM codegen")]
    UnsupportedExpression,

    #[error("rotation angle '{expression}' is not known at compile time: {reason}")]
    RuntimeAngle { expression: String, reason: String },

    #[error("unsupported control flow in QASM 2.0")]
    UnsupportedControlFlow,

    #[error("collection for loops are unsupported in QASM codegen")]
    CollectionLoop,

    #[error("for loop bounds must be compile-time integers")]
    NonConstantLoopBound,

    #[error("range for loops require exactly one capture, got {count}")]
    InvalidLoopCaptures { count: usize },

    #[error("{statement} cannot be represented in an unrolled loop")]
    UnsupportedLoopControl { statement: &'static str },

    #[error("if condition must be a compile-time boolean in QASM codegen")]
    NonConstantCondition,

    #[error("formatting error: {0}")]
    FormatError(std::fmt::Error),
}

impl From<std::fmt::Error> for QasmError {
    fn from(e: std::fmt::Error) -> Self {
        QasmError::FormatError(e)
    }
}

/// Result type for QASM code generation.
pub type QasmResult<T> = Result<T, QasmError>;

// =============================================================================
// Gate Mapping
// =============================================================================

/// Gate information for QASM.
struct GateInfo {
    /// Gate name in QASM.
    name: &'static str,
    /// Number of qubit targets.
    arity: usize,
    /// Whether this gate takes parameters.
    parameterized: bool,
}

/// Maps Zlup gate names to QASM gate info.
fn get_gate_info(name: &str) -> Option<GateInfo> {
    match name {
        // Single-qubit Pauli gates
        "x" => Some(GateInfo {
            name: "x",
            arity: 1,
            parameterized: false,
        }),
        "y" => Some(GateInfo {
            name: "y",
            arity: 1,
            parameterized: false,
        }),
        "z" => Some(GateInfo {
            name: "z",
            arity: 1,
            parameterized: false,
        }),

        // Hadamard
        "h" => Some(GateInfo {
            name: "h",
            arity: 1,
            parameterized: false,
        }),

        // S gates (zlup uses sz/szdg, QASM uses s/sdg)
        "sz" => Some(GateInfo {
            name: "s",
            arity: 1,
            parameterized: false,
        }),
        "szdg" => Some(GateInfo {
            name: "sdg",
            arity: 1,
            parameterized: false,
        }),

        // T gates
        "t" => Some(GateInfo {
            name: "t",
            arity: 1,
            parameterized: false,
        }),
        "tdg" => Some(GateInfo {
            name: "tdg",
            arity: 1,
            parameterized: false,
        }),

        // Square root gates
        "sx" => Some(GateInfo {
            name: "sx",
            arity: 1,
            parameterized: false,
        }),

        // Rotation gates
        "rx" => Some(GateInfo {
            name: "rx",
            arity: 1,
            parameterized: true,
        }),
        "ry" => Some(GateInfo {
            name: "ry",
            arity: 1,
            parameterized: true,
        }),
        "rz" => Some(GateInfo {
            name: "rz",
            arity: 1,
            parameterized: true,
        }),

        // U gates (parameterized)
        "u1" => Some(GateInfo {
            name: "u1",
            arity: 1,
            parameterized: true,
        }),
        "u2" => Some(GateInfo {
            name: "u2",
            arity: 1,
            parameterized: true,
        }),
        "u3" => Some(GateInfo {
            name: "u3",
            arity: 1,
            parameterized: true,
        }),

        // Two-qubit gates
        "cx" => Some(GateInfo {
            name: "cx",
            arity: 2,
            parameterized: false,
        }),
        "cy" => Some(GateInfo {
            name: "cy",
            arity: 2,
            parameterized: false,
        }),
        "cz" => Some(GateInfo {
            name: "cz",
            arity: 2,
            parameterized: false,
        }),
        "ch" => Some(GateInfo {
            name: "ch",
            arity: 2,
            parameterized: false,
        }),
        "swap" => Some(GateInfo {
            name: "swap",
            arity: 2,
            parameterized: false,
        }),

        // Two-qubit rotation
        "rzz" => Some(GateInfo {
            name: "rzz",
            arity: 2,
            parameterized: true,
        }),

        // Three-qubit gates
        "ccx" => Some(GateInfo {
            name: "ccx",
            arity: 3,
            parameterized: false,
        }),

        _ => None,
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
    /// Global offset for this allocator in the flat qubit register
    offset: usize,
}

/// QASM code generator.
///
/// Walks a Zlup AST and produces OpenQASM 2.0.
pub struct QasmCodegen {
    /// Allocators by name.
    allocators: BTreeMap<String, AllocatorInfo>,
    /// Number of active statically unrolled loop bodies.
    unrolled_loop_depth: usize,
    /// Total qubit count.
    total_qubits: usize,
    /// Classical register counter.
    creg_counter: usize,
    /// Output buffer.
    output: String,
    /// Compile-time constants used to inline unit-bearing gate angles.
    angle_evaluator: RefCell<ComptimeEvaluator>,
}

impl QasmCodegen {
    /// Create a new QASM code generator.
    pub fn new() -> Self {
        Self {
            allocators: BTreeMap::new(),
            unrolled_loop_depth: 0,
            total_qubits: 0,
            creg_counter: 0,
            output: String::new(),
            angle_evaluator: RefCell::new(angle_evaluator()),
        }
    }

    /// Compile a Zlup program to OpenQASM 2.0.
    pub fn compile(&mut self, program: &Program) -> QasmResult<String> {
        // Reset state
        self.allocators.clear();
        self.unrolled_loop_depth = 0;
        self.total_qubits = 0;
        self.creg_counter = 0;
        self.output.clear();
        self.angle_evaluator = RefCell::new(angle_evaluator());
        for decl in &program.declarations {
            if let TopLevelDecl::Binding(binding) = decl
                && !define_comptime_binding(&mut self.angle_evaluator.borrow_mut(), binding)
            {
                self.angle_evaluator
                    .borrow_mut()
                    .context
                    .define(&binding.name, ComptimeValue::Undefined);
            }
        }

        // Module allocations are visible while converting main. Local allocations
        // are collected as their declarations are visited, including each iteration.
        for decl in &program.declarations {
            if let TopLevelDecl::Binding(binding) = decl {
                self.collect_binding(binding)?;
            }
        }

        // Convert statements and count measurements before writing the header
        let mut body_output = String::new();
        let mut measurement_count = 0;
        for decl in &program.declarations {
            if let TopLevelDecl::Fn(fn_decl) = decl
                && fn_decl.name == "main"
            {
                self.angle_evaluator.borrow_mut().context.push_scope();
                for param in &fn_decl.params {
                    self.allocators.remove(&param.name);
                    self.angle_evaluator
                        .borrow_mut()
                        .context
                        .define(&param.name, crate::comptime::ComptimeValue::Undefined);
                }
                let converted = self.convert_block(&fn_decl.body);
                self.angle_evaluator.borrow_mut().context.pop_scope();
                let (body, mcount) = converted?;
                body_output = body;
                measurement_count = mcount;
            }
        }

        self.write_header()?;

        // Write classical register if measurements exist
        if measurement_count > 0 {
            writeln!(self.output, "creg c[{}];", measurement_count)?;
        }

        writeln!(self.output)?;

        // Write body
        self.output.push_str(&body_output);

        Ok(self.output.clone())
    }

    /// Compile a function to OpenQASM 2.0.
    pub fn compile_function(&mut self, fn_decl: &FnDecl) -> QasmResult<String> {
        // Reset state
        self.allocators.clear();
        self.unrolled_loop_depth = 0;
        self.total_qubits = 0;
        self.creg_counter = 0;
        self.output.clear();
        self.angle_evaluator = RefCell::new(angle_evaluator());

        // Convert body
        self.angle_evaluator.borrow_mut().context.push_scope();
        for param in &fn_decl.params {
            self.angle_evaluator
                .borrow_mut()
                .context
                .define(&param.name, crate::comptime::ComptimeValue::Undefined);
        }
        let converted = self.convert_block(&fn_decl.body);
        self.angle_evaluator.borrow_mut().context.pop_scope();
        let (body, measurement_count) = converted?;

        self.write_header()?;

        // Write classical register
        if measurement_count > 0 {
            writeln!(self.output, "creg c[{}];", measurement_count)?;
        }

        writeln!(self.output)?;
        self.output.push_str(&body);

        Ok(self.output.clone())
    }

    // =========================================================================
    // Collection Phase
    // =========================================================================

    fn write_header(&mut self) -> QasmResult<()> {
        writeln!(self.output, "OPENQASM 2.0;")?;
        writeln!(self.output, "include \"qelib1.inc\";")?;
        writeln!(self.output)?;
        if self.total_qubits > 0 {
            writeln!(self.output, "qreg q[{}];", self.total_qubits)?;
        }
        Ok(())
    }

    fn collect_binding(&mut self, binding: &Binding) -> QasmResult<()> {
        if let Some(value) = &binding.value {
            if let Some(capacity) = self.try_extract_allocator(value) {
                let offset = self.total_qubits;
                self.allocators.insert(
                    binding.name.clone(),
                    AllocatorInfo {
                        name: binding.name.clone(),
                        capacity,
                        offset,
                    },
                );
                self.total_qubits += capacity;
                return Ok(());
            }
            if let Some((parent, size)) = self.try_extract_child_allocator(value) {
                let parent_info =
                    self.allocators
                        .get(&parent)
                        .ok_or_else(|| QasmError::UndefinedAllocator {
                            name: parent.clone(),
                        })?;
                if size > parent_info.capacity {
                    return Err(QasmError::QubitIndexOutOfBounds {
                        allocator: parent,
                        index: size - 1,
                        capacity: parent_info.capacity,
                    });
                }
                let offset = parent_info.offset;
                self.allocators.insert(
                    binding.name.clone(),
                    AllocatorInfo {
                        name: binding.name.clone(),
                        capacity: size,
                        offset,
                    },
                );
                return Ok(());
            }
        }
        self.allocators.remove(&binding.name);
        Ok(())
    }

    // =========================================================================
    // Conversion Phase
    // =========================================================================

    /// Convert a block, returning (output, measurement_count)
    fn convert_block(&mut self, block: &Block) -> QasmResult<(String, usize)> {
        self.angle_evaluator.borrow_mut().context.push_scope();
        let allocators = self.allocators.clone();
        let result = self.convert_block_in_scope(block);
        self.allocators = allocators;
        self.angle_evaluator.borrow_mut().context.pop_scope();
        result
    }

    fn convert_block_in_scope(&mut self, block: &Block) -> QasmResult<(String, usize)> {
        let mut output = String::new();
        let mut measurement_count = 0;

        for stmt in &block.statements {
            let (stmt_output, mcount) = self.convert_stmt(stmt)?;
            output.push_str(&stmt_output);
            measurement_count += mcount;
        }

        Ok((output, measurement_count))
    }

    /// Convert a statement, returning (output, measurement_count)
    fn convert_stmt(&mut self, stmt: &Stmt) -> QasmResult<(String, usize)> {
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
                return Err(QasmError::UnsupportedLoopControl { statement });
            }
        }
        match stmt {
            Stmt::Expr(expr_stmt) => self.convert_expr_stmt(expr_stmt),
            Stmt::Tick(tick_stmt) => {
                // A tick groups operations without introducing a lexical scope.
                let mut output = String::new();
                let mut measurements = 0;
                if !tick_stmt.body.is_empty() {
                    writeln!(
                        output,
                        "// tick{}",
                        tick_stmt
                            .label
                            .as_ref()
                            .map(|label| format!(" {label}"))
                            .unwrap_or_default()
                    )?;
                }
                for stmt in &tick_stmt.body {
                    let (body, count) = self.convert_stmt(stmt)?;
                    output.push_str(&body);
                    measurements += count;
                }
                Ok((output, measurements))
            }
            Stmt::Block(block) => self.convert_block(block),
            // Handle declarations - check for measurement calls
            Stmt::Binding(binding) => {
                let result = if let Some(ref value) = binding.value {
                    self.convert_decl_value(value)
                } else {
                    Ok((String::new(), 0))
                };
                let result = result?;
                self.collect_binding(binding)?;
                if !define_comptime_binding(&mut self.angle_evaluator.borrow_mut(), binding) {
                    self.angle_evaluator
                        .borrow_mut()
                        .context
                        .define(&binding.name, crate::comptime::ComptimeValue::Undefined);
                }
                Ok(result)
            }
            Stmt::If(if_stmt) => self.convert_if(if_stmt),
            Stmt::For(for_stmt) => self.convert_for(for_stmt),
            Stmt::Break(_) => Err(QasmError::UnsupportedLoopControl { statement: "break" }),
            Stmt::Continue(_) => Err(QasmError::UnsupportedLoopControl {
                statement: "continue",
            }),
            _ => Ok((String::new(), 0)),
        }
    }

    fn convert_if(&mut self, if_stmt: &IfStmt) -> QasmResult<(String, usize)> {
        let condition = self
            .angle_evaluator
            .borrow_mut()
            .eval_expr(&if_stmt.condition)
            .ok()
            .and_then(|value| value.as_bool())
            .ok_or(QasmError::NonConstantCondition)?;
        if condition {
            self.convert_block(&if_stmt.then_body)
        } else {
            match &if_stmt.else_body {
                Some(ElseBranch::Else(block)) => self.convert_block(block),
                Some(ElseBranch::ElseIf(nested)) => self.convert_if(nested),
                None => Ok((String::new(), 0)),
            }
        }
    }

    fn convert_for(&mut self, for_stmt: &ForStmt) -> QasmResult<(String, usize)> {
        if let Some(statement) = super::block_control(&for_stmt.body) {
            return Err(QasmError::UnsupportedLoopControl { statement });
        }
        let ForRange::Range { start, end } = &for_stmt.range else {
            return Err(QasmError::CollectionLoop);
        };
        let [capture] = for_stmt.captures.as_slice() else {
            return Err(QasmError::InvalidLoopCaptures {
                count: for_stmt.captures.len(),
            });
        };
        // Bounds are evaluated before introducing the capture, which may shadow them.
        let start = self.eval_loop_bound(start)?;
        let end = self.eval_loop_bound(end)?;
        let mut output = String::new();
        let mut measurements = 0;
        for value in start..end {
            self.angle_evaluator.borrow_mut().context.push_scope();
            self.angle_evaluator
                .borrow_mut()
                .context
                .define(capture, ComptimeValue::Int(value));
            let allocators = self.allocators.clone();
            self.allocators.remove(capture);
            self.unrolled_loop_depth += 1;
            let result = self.convert_block(&for_stmt.body);
            self.unrolled_loop_depth -= 1;
            self.allocators = allocators;
            self.angle_evaluator.borrow_mut().context.pop_scope();
            let (body, count) = result?;
            output.push_str(&body);
            measurements += count;
        }
        Ok((output, measurements))
    }

    fn eval_loop_bound(&self, expr: &Expr) -> QasmResult<i64> {
        match self.angle_evaluator.borrow_mut().eval_expr(expr) {
            Ok(ComptimeValue::Int(value)) => Ok(value),
            Ok(ComptimeValue::Uint(value)) => {
                i64::try_from(value).map_err(|_| QasmError::NonConstantLoopBound)
            }
            _ => Err(QasmError::NonConstantLoopBound),
        }
    }

    fn convert_decl_value(&mut self, expr: &Expr) -> QasmResult<(String, usize)> {
        match expr {
            Expr::Call(call) => {
                let name = self.extract_call_name(&call.callee)?;
                if name == "mz" {
                    return self.convert_measure(call);
                }
                Ok((String::new(), 0))
            }
            Expr::Measure(measure) => self.convert_measure_expr(measure),
            _ => Ok((String::new(), 0)),
        }
    }

    fn convert_expr_stmt(
        &mut self,
        expr_stmt: &crate::ast::ExprStmt,
    ) -> QasmResult<(String, usize)> {
        match &expr_stmt.expr {
            Expr::Call(call) => self.convert_call(call),
            Expr::Gate(gate) => self.convert_gate_expr(gate),
            Expr::Measure(measure) => self.convert_measure_expr(measure),
            _ => Ok((String::new(), 0)),
        }
    }

    fn convert_gate_expr(&mut self, gate: &crate::ast::GateExpr) -> QasmResult<(String, usize)> {
        use crate::ast::GateKind;
        use std::fmt::Write;

        // Map GateKind to QASM gate name (lowercase)
        let gate_name: &str = match gate.kind {
            GateKind::X => "x",
            GateKind::Y => "y",
            GateKind::Z => "z",
            GateKind::H => "h",
            GateKind::T => "t",
            GateKind::Tdg => "tdg",
            GateKind::SX => "sx",
            GateKind::SY => "sy",
            GateKind::SZ => "s", // QASM uses "s" for S gate
            GateKind::SXdg => "sxdg",
            GateKind::SYdg => "sydg",
            GateKind::SZdg => "sdg", // QASM uses "sdg" for S-dagger
            GateKind::RX => "rx",
            GateKind::RY => "ry",
            GateKind::RZ => "rz",
            GateKind::CX => "cx",
            GateKind::CY => "cy",
            GateKind::CZ => "cz",
            GateKind::CH => "ch",
            GateKind::SWAP => "swap",
            GateKind::ISWAP => "iswap",
            GateKind::SXX => "sxx",
            GateKind::SYY => "syy",
            GateKind::SZZ => "szz",
            GateKind::SXXdg => "sxxdg",
            GateKind::SYYdg => "syydg",
            GateKind::SZZdg => "szzdg",
            GateKind::CRZ => "crz",
            GateKind::RZZ => "rzz",
            GateKind::CCX => "ccx",
            GateKind::F => "f",
            GateKind::Fdg => "fdg",
            GateKind::F4 => "f4",
            GateKind::F4dg => "f4dg",
            GateKind::PZ => return Ok((String::new(), 0)), // Prepare is implicit in QASM
        };

        let mut output = String::new();

        // Handle batch targets (sets)
        if let Expr::Set(set_expr) = &gate.target {
            let gate_info = get_gate_info(gate_name).ok_or_else(|| QasmError::UnknownGate {
                name: gate_name.to_string(),
            })?;
            let params: Vec<String> = gate
                .params
                .iter()
                .map(|p| self.convert_angle_expression(p))
                .collect::<Result<_, _>>()?;
            return self.convert_batch_gate(&gate_info, &set_expr.elements, &params);
        }

        // Convert parameters
        let params: Vec<String> = gate
            .params
            .iter()
            .map(|p| self.convert_angle_expression(p))
            .collect::<Result<_, _>>()?;

        // Convert target(s)
        let targets = self.extract_gate_qubit_targets(&gate.target)?;

        // Format gate with optional parameters
        if params.is_empty() {
            write!(output, "{} ", gate_name)?;
        } else {
            write!(output, "{}({}) ", gate_name, params.join(", "))?;
        }

        // Format targets
        writeln!(output, "{};", targets.join(", "))?;

        Ok((output, 0))
    }

    fn convert_measure_expr(
        &mut self,
        measure: &crate::ast::MeasureExpr,
    ) -> QasmResult<(String, usize)> {
        use std::fmt::Write;

        let mut output = String::new();
        let targets = self.extract_gate_qubit_targets(&measure.targets)?;

        for (i, target) in targets.iter().enumerate() {
            let bit_idx = self.creg_counter + i;
            writeln!(output, "measure {} -> c[{}];", target, bit_idx)?;
        }

        let count = targets.len();
        self.creg_counter += count;
        Ok((output, count))
    }

    fn extract_gate_qubit_targets(&self, expr: &Expr) -> QasmResult<Vec<String>> {
        match expr {
            Expr::Index(idx) => {
                let allocator = self.extract_identifier(&idx.object)?;
                let index = self.extract_integer(&idx.index)?;
                let global_index = self.get_global_qubit_index(&allocator, index)?;
                Ok(vec![format!("q[{}]", global_index)])
            }
            Expr::Tuple(tuple) => {
                let mut targets = Vec::new();
                for elem in &tuple.elements {
                    targets.extend(self.extract_gate_qubit_targets(elem)?);
                }
                Ok(targets)
            }
            Expr::BracketArray(arr) => {
                let mut targets = Vec::new();
                for elem in &arr.elements {
                    targets.extend(self.extract_gate_qubit_targets(elem)?);
                }
                Ok(targets)
            }
            _ => Err(QasmError::UnsupportedExpression),
        }
    }

    fn convert_call(&mut self, call: &CallExpr) -> QasmResult<(String, usize)> {
        let name = self.extract_call_name(&call.callee)?;

        // Check for special operations
        match name.as_str() {
            "mz" => return self.convert_measure(call),
            "barrier" => return self.convert_barrier(call),
            _ => {}
        }

        // Check for gate calls
        let Some(gate_info) = get_gate_info(&name) else {
            return Ok((String::new(), 0));
        };

        let mut output = String::new();

        // For parameterized gates: qubits come first, then angle
        // e.g., rz(q[0], 1.5708) or rz(&[q[0], q[1]], 1.5708)
        let (params, qubit_args): (Vec<String>, &[Expr]) = if gate_info.parameterized {
            if call.args.len() < 2 {
                return Err(QasmError::WrongArgumentCount {
                    gate: name,
                    expected: gate_info.arity + 1,
                    got: call.args.len(),
                });
            }
            // Last argument is the parameter (angle)
            let param = self.convert_angle_expression(call.args.last().unwrap())?;
            // All but last are qubit args
            (vec![param], &call.args[..call.args.len() - 1])
        } else {
            (Vec::new(), &call.args[..])
        };

        // Check for batch operations (set literal or address-of array)
        if !qubit_args.is_empty() {
            // Set literal: h([q[0], q[1])
            if let Expr::Set(set_expr) = &qubit_args[0] {
                return self.convert_batch_gate(&gate_info, &set_expr.elements, &params);
            }
            // Address-of array: h(&[q[0], q[1]])
            if let Expr::Unary(unary) = &qubit_args[0]
                && let crate::ast::UnaryOp::AddrOf = unary.op
                && let Expr::BracketArray(arr) = &unary.operand
            {
                return self.convert_batch_gate(&gate_info, &arr.elements, &params);
            }
        }

        // Standard gate call
        if qubit_args.len() != gate_info.arity {
            return Err(QasmError::WrongArgumentCount {
                gate: name,
                expected: if gate_info.parameterized {
                    gate_info.arity + 1
                } else {
                    gate_info.arity
                },
                got: call.args.len(),
            });
        }

        // Build gate string
        write!(output, "{}", gate_info.name)?;

        // Add parameters
        if !params.is_empty() {
            write!(output, "({})", params.join(", "))?;
        }

        // Add qubit targets
        write!(output, " ")?;
        for (i, arg) in qubit_args.iter().enumerate() {
            if i > 0 {
                write!(output, ", ")?;
            }
            let (alloc, idx) = self.extract_qubit_ref(arg)?;
            let global_idx = self.get_global_qubit_index(&alloc, idx)?;
            write!(output, "q[{}]", global_idx)?;
        }
        writeln!(output, ";")?;

        Ok((output, 0))
    }

    fn convert_batch_gate(
        &mut self,
        gate_info: &GateInfo,
        elements: &[Expr],
        params: &[String],
    ) -> QasmResult<(String, usize)> {
        let mut output = String::new();

        if gate_info.arity == 1 {
            // Single-qubit gate on multiple qubits
            for elem in elements {
                let (alloc, idx) = self.extract_qubit_ref(elem)?;
                let global_idx = self.get_global_qubit_index(&alloc, idx)?;

                write!(output, "{}", gate_info.name)?;
                if !params.is_empty() {
                    write!(output, "({})", params.join(", "))?;
                }
                writeln!(output, " q[{}];", global_idx)?;
            }
        } else if gate_info.arity == 2 {
            // Two-qubit gate with tuple pairs
            for elem in elements {
                if let Expr::Tuple(tuple) = elem {
                    if tuple.elements.len() == 2 {
                        let (alloc1, idx1) = self.extract_qubit_ref(&tuple.elements[0])?;
                        let (alloc2, idx2) = self.extract_qubit_ref(&tuple.elements[1])?;
                        let global1 = self.get_global_qubit_index(&alloc1, idx1)?;
                        let global2 = self.get_global_qubit_index(&alloc2, idx2)?;

                        write!(output, "{}", gate_info.name)?;
                        if !params.is_empty() {
                            write!(output, "({})", params.join(", "))?;
                        }
                        writeln!(output, " q[{}], q[{}];", global1, global2)?;
                    } else {
                        return Err(QasmError::UnsupportedExpression);
                    }
                } else {
                    return Err(QasmError::UnsupportedExpression);
                }
            }
        } else {
            return Err(QasmError::UnsupportedExpression);
        }

        Ok((output, 0))
    }

    fn convert_measure(&mut self, call: &CallExpr) -> QasmResult<(String, usize)> {
        let mut output = String::new();
        let mut measurement_count = 0;

        // Typed measurement: mz(type, target)
        if call.args.len() == 2 {
            let target_arg = &call.args[1];

            match target_arg {
                // Single qubit: q[0]
                Expr::Index(_) => {
                    let (alloc, idx) = self.extract_qubit_ref(target_arg)?;
                    let global_idx = self.get_global_qubit_index(&alloc, idx)?;
                    let creg_idx = self.creg_counter;
                    self.creg_counter += 1;
                    writeln!(output, "measure q[{}] -> c[{}];", global_idx, creg_idx)?;
                    measurement_count = 1;
                }
                // Address-of array: &[q[0], q[1], ...]
                Expr::Unary(unary) => {
                    if let crate::ast::UnaryOp::AddrOf = unary.op
                        && let Expr::BracketArray(arr) = &unary.operand
                    {
                        for elem in &arr.elements {
                            let (alloc, idx) = self.extract_qubit_ref(elem)?;
                            let global_idx = self.get_global_qubit_index(&alloc, idx)?;
                            let creg_idx = self.creg_counter;
                            self.creg_counter += 1;
                            writeln!(output, "measure q[{}] -> c[{}];", global_idx, creg_idx)?;
                            measurement_count += 1;
                        }
                    }
                }
                _ => return Err(QasmError::UnsupportedExpression),
            }
        } else {
            // Legacy: mz(q[0])
            for arg in &call.args {
                let (alloc, idx) = self.extract_qubit_ref(arg)?;
                let global_idx = self.get_global_qubit_index(&alloc, idx)?;
                let creg_idx = self.creg_counter;
                self.creg_counter += 1;
                writeln!(output, "measure q[{}] -> c[{}];", global_idx, creg_idx)?;
                measurement_count += 1;
            }
        }

        Ok((output, measurement_count))
    }

    fn convert_barrier(&mut self, call: &CallExpr) -> QasmResult<(String, usize)> {
        let mut output = String::new();

        if call.args.is_empty() {
            // Barrier on all qubits
            writeln!(output, "barrier q;")?;
        } else {
            // Barrier on specific allocators
            write!(output, "barrier ")?;
            let mut first = true;
            for arg in &call.args {
                if let Expr::Ident(ident) = arg
                    && let Some(alloc) = self.allocators.get(&ident.name)
                {
                    for i in 0..alloc.capacity {
                        if !first {
                            write!(output, ", ")?;
                        }
                        first = false;
                        write!(output, "q[{}]", alloc.offset + i)?;
                    }
                }
            }
            writeln!(output, ";")?;
        }

        Ok((output, 0))
    }

    fn convert_expression(&self, expr: &Expr) -> QasmResult<String> {
        match expr {
            Expr::IntLit(lit) => Ok(lit.value.to_string()),
            Expr::FloatLit(lit) => Ok(format!("{}", lit.value)),
            Expr::Ident(ident) => {
                // Check for built-in constants
                match ident.name.as_str() {
                    "pi" | "PI" => Ok("pi".to_string()),
                    "tau" | "TAU" => Ok("2*pi".to_string()),
                    _ => Ok(ident.name.clone()),
                }
            }
            Expr::Binary(binary) => {
                let left = self.convert_expression(&binary.left)?;
                let right = self.convert_expression(&binary.right)?;
                let op = match binary.op {
                    BinaryOp::Add => "+",
                    BinaryOp::Sub => "-",
                    BinaryOp::Mul => "*",
                    BinaryOp::Div => "/",
                    _ => return Err(QasmError::UnsupportedExpression),
                };
                Ok(format!("({} {} {})", left, op, right))
            }
            Expr::Unary(unary) => {
                let operand = self.convert_expression(&unary.operand)?;
                match unary.op {
                    crate::ast::UnaryOp::Neg => Ok(format!("-{}", operand)),
                    _ => Err(QasmError::UnsupportedExpression),
                }
            }
            _ => Err(QasmError::UnsupportedExpression),
        }
    }

    fn convert_angle_expression(&self, expr: &Expr) -> QasmResult<String> {
        let turns =
            resolve_angle_turns(&mut self.angle_evaluator.borrow_mut(), expr).map_err(|error| {
                QasmError::RuntimeAngle {
                    expression: angle_expression_name(expr),
                    reason: error.to_string(),
                }
            })?;
        Ok(format!("{}", turns * std::f64::consts::TAU))
    }

    // =========================================================================
    // Helpers
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

    fn extract_call_name(&self, callee: &Expr) -> QasmResult<String> {
        match callee {
            Expr::Ident(ident) => Ok(ident.name.clone()),
            Expr::Field(field) => Ok(field.field.clone()),
            _ => Err(QasmError::UnsupportedExpression),
        }
    }

    fn extract_identifier(&self, expr: &Expr) -> QasmResult<String> {
        match expr {
            Expr::Ident(ident) => Ok(ident.name.clone()),
            _ => Err(QasmError::UnsupportedExpression),
        }
    }

    fn extract_qubit_ref(&self, expr: &Expr) -> QasmResult<(String, usize)> {
        match expr {
            Expr::Index(index) => {
                let allocator = self.extract_identifier(&index.object)?;
                let idx = self.extract_integer(&index.index)?;
                Ok((allocator, idx))
            }
            _ => Err(QasmError::UnsupportedExpression),
        }
    }

    fn get_global_qubit_index(&self, allocator: &str, index: usize) -> QasmResult<usize> {
        let alloc =
            self.allocators
                .get(allocator)
                .ok_or_else(|| QasmError::UndefinedAllocator {
                    name: allocator.to_string(),
                })?;

        if index >= alloc.capacity {
            return Err(QasmError::QubitIndexOutOfBounds {
                allocator: allocator.to_string(),
                index,
                capacity: alloc.capacity,
            });
        }

        Ok(alloc.offset + index)
    }

    fn extract_integer(&self, expr: &Expr) -> QasmResult<usize> {
        self.angle_evaluator
            .borrow_mut()
            .eval_expr(expr)
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(QasmError::UnsupportedExpression)
    }
}

impl Default for QasmCodegen {
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
        let output = compile_to_qasm(source).unwrap();
        let gates: Vec<_> = output
            .lines()
            .filter(|line| line.starts_with("x "))
            .collect();
        assert_eq!(gates, ["x q[1];"]);
        assert!(output.contains("qreg q[2];"));
    }
    // Compatibility: transfer-free bodies retain their emitted operations.
    #[test]
    fn test_round4_compat_empty_range() {
        let source = "pub fn main() -> unit { for i in 3..1 { mut q := qalloc(1); h q[0]; } }";
        let output = compile_to_qasm(source).unwrap();
        let gates: Vec<_> = output
            .lines()
            .filter(|line| line.starts_with("h "))
            .collect();
        assert_eq!(gates, Vec::<&str>::new());
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
            let error = compile_to_qasm(&source).expect_err("nested transfer must be rejected");
            assert!(error.to_string().contains("return"), "{body}: {error}");
        }
    }

    // Compatibility: only the loop's return boundary is forbidden.
    #[test]
    fn test_round4_compat_terminal_return() {
        let source =
            "pub fn main() -> unit { mut q := qalloc(1); for i in 0..1 { h q[0]; } return unit; }";
        compile_to_qasm(source).unwrap();
    }

    #[test]
    fn test_round4_elseif_condition() {
        let source = "pub fn main() -> unit { mut q := qalloc(2); pz q; for i in 0..3 { if false { h q[0]; } else if (blk: { return unit; true }) { x q[0]; } } return unit; }";
        let error = compile_to_qasm(source).expect_err("transfer in loop syntax must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    #[test]
    fn test_round4_nested_trailing() {
        let source = "pub fn main() -> unit { mut q := qalloc(2); pz q; for i in 0..3 { a := blk: { mut n := false; v := blk: { for j in 0..1 { (blk: { n = true; unit }) } unit }; if n { return unit; } 0.125 }; rx(a turns) q[0]; } return unit; }";
        let error = compile_to_qasm(source).expect_err("transfer in loop syntax must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    #[test]
    fn test_round4_zero_iterations() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..0 { return unit; } return unit; }";
        let error = compile_to_qasm(source).expect_err("transfer in loop syntax must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    #[test]
    fn test_round4_dead_break() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..1 { if false { break; } } return unit; }";
        let error = compile_to_qasm(source).expect_err("transfer in loop syntax must be rejected");
        assert!(error.to_string().contains("break"), "{error}");
    }
    #[test]
    fn test_round4_dead_continue() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..1 { if false { continue; } } return unit; }";
        let error = compile_to_qasm(source).expect_err("transfer in loop syntax must be rejected");
        assert!(error.to_string().contains("continue"), "{error}");
    }
    #[test]
    fn test_round4_numeric_eq() {
        for condition in ["1/2 == 0.5", "1 == 1.0", "1 == 1u64"] {
            let source = format!(
                "pub fn main() -> unit {{ mut q := qalloc(1); if {condition} {{ x q[0]; }} else {{ h q[0]; }} return unit; }}"
            );
            let output = compile_to_qasm(&source).unwrap();
            assert_eq!(
                output.lines().filter(|line| *line == "x q[0];").count(),
                1,
                "{condition}: {output}"
            );
            assert!(!output.contains("h q[0];"));
        }
    }
    #[test]
    fn test_round4_numeric_ne() {
        for condition in ["1/2 != 0.5", "1 != 1.0", "1 != 1u64"] {
            let source = format!(
                "pub fn main() -> unit {{ mut q := qalloc(1); if {condition} {{ h q[0]; }} else {{ x q[0]; }} return unit; }}"
            );
            let output = compile_to_qasm(&source).unwrap();
            assert_eq!(
                output.lines().filter(|line| *line == "x q[0];").count(),
                1,
                "{condition}: {output}"
            );
            assert!(!output.contains("h q[0];"));
        }
    }
    #[test]
    fn test_round4_numeric_lt() {
        for condition in ["1/2 < 0.75", "1 < 1.5", "1 < 2u64"] {
            let source = format!(
                "pub fn main() -> unit {{ mut q := qalloc(1); if {condition} {{ x q[0]; }} else {{ h q[0]; }} return unit; }}"
            );
            let output = compile_to_qasm(&source).unwrap();
            assert_eq!(
                output.lines().filter(|line| *line == "x q[0];").count(),
                1,
                "{condition}: {output}"
            );
            assert!(!output.contains("h q[0];"));
        }
    }
    #[test]
    fn test_round4_numeric_le() {
        for condition in ["1/2 <= 0.5", "1 <= 1.0", "1 <= 1u64"] {
            let source = format!(
                "pub fn main() -> unit {{ mut q := qalloc(1); if {condition} {{ x q[0]; }} else {{ h q[0]; }} return unit; }}"
            );
            let output = compile_to_qasm(&source).unwrap();
            assert_eq!(
                output.lines().filter(|line| *line == "x q[0];").count(),
                1,
                "{condition}: {output}"
            );
            assert!(!output.contains("h q[0];"));
        }
    }
    #[test]
    fn test_round4_numeric_gt() {
        for condition in ["1/2 > 0.25", "1 > 0.5", "1 > 0u64"] {
            let source = format!(
                "pub fn main() -> unit {{ mut q := qalloc(1); if {condition} {{ x q[0]; }} else {{ h q[0]; }} return unit; }}"
            );
            let output = compile_to_qasm(&source).unwrap();
            assert_eq!(
                output.lines().filter(|line| *line == "x q[0];").count(),
                1,
                "{condition}: {output}"
            );
            assert!(!output.contains("h q[0];"));
        }
    }
    #[test]
    fn test_round4_numeric_ge() {
        for condition in ["1/2 >= 0.5", "1 >= 1.0", "1 >= 1u64"] {
            let source = format!(
                "pub fn main() -> unit {{ mut q := qalloc(1); if {condition} {{ x q[0]; }} else {{ h q[0]; }} return unit; }}"
            );
            let output = compile_to_qasm(&source).unwrap();
            assert_eq!(
                output.lines().filter(|line| *line == "x q[0];").count(),
                1,
                "{condition}: {output}"
            );
            assert!(!output.contains("h q[0];"));
        }
    }

    #[test]
    fn test_round3_angle_dead_branch() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..1 { rx((blk: { if false { return unit; } 0.125 }) turns) q[0]; } return unit; }";
        let error =
            compile_to_qasm(source).expect_err("syntactically present return must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    #[test]
    fn test_round3_if_expression_dead_branch() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..1 { theta := if (false) { return unit; 0.25 } else { 0.125 }; rx(theta turns) q[0]; } return unit; }";
        let error =
            compile_to_qasm(source).expect_err("syntactically present return must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    #[test]
    fn test_round3_nested_capture_dead_branch() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..1 { theta := blk: { for j in i..i + 2 { if j > 8 { return unit; } } 0.125 }; rx(theta turns) q[0]; } return unit; }";
        let error =
            compile_to_qasm(source).expect_err("syntactically present return must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    // Compatibility: a possibly reached transfer must still be rejected.
    #[test]
    fn test_round3_compat_unknown_path() {
        let source = "pub fn main(c: bool) -> unit { mut q := qalloc(1); for i in 0..1 { a := blk: { if c { return unit; } 0.125 }; h q[0]; } return unit; }";
        let error = compile_to_qasm(source).expect_err("reachable return must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    // Compatibility: a possibly reached transfer must still be rejected.
    #[test]
    fn test_round3_compat_later_iteration() {
        let source = "pub fn main(c: bool) -> unit { mut q := qalloc(1); for i in 0..1 { a := blk: { mut n := 0; for j in 0..2 { if n == 1 { return unit; } n = n + 1; } 0.125 }; h q[0]; } return unit; }";
        let error = compile_to_qasm(source).expect_err("reachable return must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    // Compatibility: a possibly reached transfer must still be rejected.
    #[test]
    fn test_round3_compat_unknown_assignment() {
        let source = "pub fn main(c: bool) -> unit { mut q := qalloc(1); for i in 0..1 { a := blk: { mut n := 0; if c { n = 1; } if n == 1 { return unit; } 0.125 }; h q[0]; } return unit; }";
        let error = compile_to_qasm(source).expect_err("reachable return must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    // Compatibility: a nested function has its own return boundary.
    #[test]
    fn test_round3_compat_function_boundary() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..1 { get_n := fn() -> i64 { return 1; }; n := get_n(); h q[0]; } return unit; }";
        compile_to_qasm(source).unwrap();
    }

    #[test]
    fn test_round3_trailing_return() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); pz q; for i in 0..3 { rx(0.125 turns) q[0]; (blk: { return unit; }) } return unit; }";
        let error = compile_to_qasm(source).expect_err("trailing return must fail loudly");
        assert!(error.to_string().contains("return"), "{error}");
    }
    #[test]
    fn test_round3_trailing_break() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); pz q; for i in 0..3 { rx(0.125 turns) q[0]; (blk: { break; }) } return unit; }";
        let error = compile_to_qasm(source).expect_err("trailing break must fail loudly");
        assert!(error.to_string().contains("break"), "{error}");
    }
    #[test]
    fn test_round3_trailing_continue() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); pz q; for i in 0..3 { rx(0.125 turns) q[0]; (blk: { continue; }) } return unit; }";
        let error = compile_to_qasm(source).expect_err("trailing continue must fail loudly");
        assert!(error.to_string().contains("continue"), "{error}");
    }
    #[test]
    fn test_round3_dead_branch() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..1 { theta := blk: { if false { return unit; } 0.125 }; rx(theta turns) q[0]; } return unit; }";
        let error =
            compile_to_qasm(source).expect_err("syntactically present return must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    #[test]
    fn test_round3_empty_range() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..1 { theta := blk: { for j in 0..0 { return unit; } 0.125 }; rx(theta turns) q[0]; } return unit; }";
        let error =
            compile_to_qasm(source).expect_err("syntactically present return must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    #[test]
    fn test_round3_capture_branch() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..1 { theta := blk: { n := i + 1; if n == 9 { return unit; } 0.125 }; rx(theta turns) q[0]; } return unit; }";
        let error =
            compile_to_qasm(source).expect_err("syntactically present return must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }
    #[test]
    fn test_round3_capture_range() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..1 { theta := blk: { for j in i..i { return unit; } 0.125 }; rx(theta turns) q[0]; } return unit; }";
        let error =
            compile_to_qasm(source).expect_err("syntactically present return must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }

    use crate::parse;

    fn compile_to_qasm(source: &str) -> QasmResult<String> {
        let program = parse(source).expect("parse failed");
        let mut codegen = QasmCodegen::new();
        codegen.compile(&program)
    }

    // The conservative rule rejects unreachable transfers inside loop bodies.
    #[test]
    fn test_review_unreachable_and_terminal_returns() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..3 { if i == 9 { return unit; } h q[0]; } for j in 1..1 { return unit; } return unit; }";
        let error =
            compile_to_qasm(source).expect_err("syntactically present return must be rejected");
        assert!(error.to_string().contains("return"), "{error}");
    }

    #[test]
    fn test_review_loop_control_expression() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..3 { a := blk: { return unit; }; } return unit; }".to_string();
        let error =
            compile_to_qasm(&source).expect_err("control transfer in loop must fail loudly");
        assert!(error.to_string().contains("return"), "{error}");
    }

    #[test]
    fn test_review_loop_control_switch() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..3 { switch (i) { 0 => 0, else => 1, } } return unit; }".to_string();
        let error =
            compile_to_qasm(&source).expect_err("control transfer in loop must fail loudly");
        assert!(error.to_string().contains("switch"), "{error}");
    }

    #[test]
    fn test_review_loop_control_try_block() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..3 { try! { return unit; } } return unit; }".to_string();
        let error =
            compile_to_qasm(&source).expect_err("control transfer in loop must fail loudly");
        assert!(error.to_string().contains("return"), "{error}");
    }

    #[test]
    fn test_review_loop_control_defer() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..3 { defer { return unit; } } return unit; }".to_string();
        let error =
            compile_to_qasm(&source).expect_err("control transfer in loop must fail loudly");
        assert!(error.to_string().contains("return"), "{error}");
    }

    #[test]
    fn test_review_loop_control_propagation() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..3 { a := try missing; } return unit; }".to_string();
        let error =
            compile_to_qasm(&source).expect_err("control transfer in loop must fail loudly");
        assert!(error.to_string().contains("try"), "{error}");
    }
    #[test]
    fn test_review_failed_comptime_scope() {
        let source = "pub fn main(c: bool) -> unit { mut q := qalloc(1); pz q; n := 0.125; a := blk: { n := 0.25; c }; rz(n turns) q[0]; return unit; }";
        let output = compile_to_qasm(source).unwrap();
        assert!(output.contains("rz(0.7853981633974483) q[0];"), "{output}");
    }

    #[test]
    fn test_review_loop_return_direct() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); pz q; for i in 0..3 { rx(0.125 turns) q[0]; return unit; } return unit; }".to_string();
        let error = compile_to_qasm(&source).expect_err("return in unrolled loop must fail loudly");
        assert!(error.to_string().contains("return"), "{error}");
    }

    #[test]
    fn test_review_loop_return_block() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); pz q; for i in 0..3 { rx(0.125 turns) q[0]; { return unit; } } return unit; }".to_string();
        let error = compile_to_qasm(&source).expect_err("return in unrolled loop must fail loudly");
        assert!(error.to_string().contains("return"), "{error}");
    }

    #[test]
    fn test_review_loop_return_if() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); pz q; for i in 0..3 { rx(0.125 turns) q[0]; if i == 0 { return unit; } } return unit; }".to_string();
        let error = compile_to_qasm(&source).expect_err("return in unrolled loop must fail loudly");
        assert!(error.to_string().contains("return"), "{error}");
    }

    #[test]
    fn test_review_loop_return_tick() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); pz q; for i in 0..3 { rx(0.125 turns) q[0]; tick { return unit; } } return unit; }".to_string();
        let error = compile_to_qasm(&source).expect_err("return in unrolled loop must fail loudly");
        assert!(error.to_string().contains("return"), "{error}");
    }

    #[test]
    fn test_review_loop_return_nested_loop() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); pz q; for i in 0..3 { rx(0.125 turns) q[0]; for j in 0..1 { return unit; } } return unit; }".to_string();
        let error = compile_to_qasm(&source).expect_err("return in unrolled loop must fail loudly");
        assert!(error.to_string().contains("return"), "{error}");
    }

    #[test]
    fn test_review_tick_binding_visibility() {
        let source = "pub fn main() -> unit { tick { mut q := qalloc(1); theta := 0.25; pz q; } rz(theta turns) q[0]; return unit; }";
        let output = compile_to_qasm(source).unwrap();
        assert!(output.contains("rz(1.5707963267948966) q[0];"));
    }
    #[test]
    fn test_control_flow_angles_and_compile_function() {
        let program = parse(
            "pub fn main() -> unit {
            mut q := qalloc(1);
            for i in 0..3 { theta := i; rz(theta) q[0]; }
        }",
        )
        .unwrap();
        let TopLevelDecl::Fn(function) = &program.declarations[0] else {
            panic!("expected function")
        };
        let output = QasmCodegen::new().compile_function(function).unwrap();
        let rotations: Vec<_> = output
            .lines()
            .filter(|line| line.starts_with("rz("))
            .collect();
        assert_eq!(rotations.len(), 3);
        assert_eq!(rotations[0], "rz(0) q[0];");
        assert_eq!(rotations[1], format!("rz({}) q[0];", std::f64::consts::TAU));
        assert_eq!(
            rotations[2],
            format!("rz({}) q[0];", 2.0 * std::f64::consts::TAU)
        );
    }

    #[test]
    fn test_control_flow_bounds_and_capture_shadow() {
        let source = "n := 3; pub fn main() -> unit { mut q := qalloc(n); for n in 0..n { k := n; for j in n..k + 1 { h q[j]; } } h q[n - 1]; }";
        let output = compile_to_qasm(source).unwrap();
        let gates: Vec<_> = output
            .lines()
            .filter(|line| line.starts_with("h "))
            .collect();
        assert_eq!(gates, ["h q[0];", "h q[1];", "h q[2];", "h q[2];"]);
    }

    #[test]
    fn test_control_flow_local_allocations() {
        let source = "pub fn main() -> unit { mut q := qalloc(1); for i in 0..2 { mut q := qalloc(i + 1); h q[i]; } h q[0]; }";
        let output = compile_to_qasm(source).unwrap();
        let gates: Vec<_> = output
            .lines()
            .filter(|line| line.starts_with("h "))
            .collect();
        assert_eq!(gates, ["h q[1];", "h q[3];", "h q[0];"]);
    }

    #[test]
    fn test_control_flow_selected_allocations() {
        let source = "pub fn main() -> unit { for i in 0..3 { if i == 1 { mut q := qalloc(i + 1); x q[i]; } else if i == 8 { break; } } }";
        let error =
            compile_to_qasm(source).expect_err("syntactically present break must be rejected");
        assert!(error.to_string().contains("break"), "{error}");
    }

    #[test]
    fn test_control_flow_empty_range() {
        let source = "pub fn main() -> unit { for i in 3..1 { break; } }";
        let error =
            compile_to_qasm(source).expect_err("syntactically present break must be rejected");
        assert!(error.to_string().contains("break"), "{error}");
    }

    #[test]
    fn test_control_flow_runtime_comparison_is_not_false() {
        let source = "pub fn main(n: int) -> unit { mut q := qalloc(1); if n == 1 { h q[0]; } }";
        assert!(compile_to_qasm(source).is_err());
    }

    #[test]
    fn test_control_flow_fixed_target() {
        let source = "pub fn main() -> unit { mut q := qalloc(4); for i in 0..3 { h q[0]; } }";
        let output = compile_to_qasm(source).unwrap();
        let gates: Vec<_> = output
            .lines()
            .filter(|line| line.starts_with("h "))
            .collect();
        assert_eq!(gates, ["h q[0];", "h q[0];", "h q[0];"]);
    }

    #[test]
    fn test_control_flow_indexed_target() {
        let source = "pub fn main() -> unit { mut q := qalloc(4); for i in 0..3 { h q[i]; } }";
        let output = compile_to_qasm(source).unwrap();
        let gates: Vec<_> = output
            .lines()
            .filter(|line| line.starts_with("h "))
            .collect();
        assert_eq!(gates, ["h q[0];", "h q[1];", "h q[2];"]);
    }

    #[test]
    fn test_control_flow_nested_loops() {
        let source = "pub fn main() -> unit { mut q := qalloc(4); for i in 0..2 { for j in 0..2 { cx (q[i], q[j + 2]); } } }";
        let output = compile_to_qasm(source).unwrap();
        let gates: Vec<_> = output
            .lines()
            .filter(|line| line.starts_with("cx "))
            .collect();
        assert_eq!(
            gates,
            [
                "cx q[0], q[2];",
                "cx q[0], q[3];",
                "cx q[1], q[2];",
                "cx q[1], q[3];"
            ]
        );
    }

    #[test]
    fn test_control_flow_comptime_if() {
        let source =
            "pub fn main() -> unit { mut q := qalloc(4); for i in 0..3 { if i == 1 { x q[i]; } } }";
        let output = compile_to_qasm(source).unwrap();
        let gates: Vec<_> = output
            .lines()
            .filter(|line| line.starts_with("x "))
            .collect();
        assert_eq!(gates, ["x q[1];"]);
    }

    #[test]
    fn test_control_flow_reject_collection() {
        let source =
            "pub fn main(n: int) -> unit { mut q := qalloc(4); for i in [0, 1] { h q[0]; } }";
        assert!(
            compile_to_qasm(source).is_err(),
            "unsupported collection must fail loudly"
        );
    }

    #[test]
    fn test_control_flow_reject_runtime_bound() {
        let source =
            "pub fn main(n: int) -> unit { mut q := qalloc(4); for i in 0..n { h q[0]; } }";
        assert!(
            compile_to_qasm(source).is_err(),
            "unsupported runtime_bound must fail loudly"
        );
    }

    #[test]
    fn test_control_flow_reject_multi_capture() {
        let source =
            "pub fn main(n: int) -> unit { mut q := qalloc(4); for i, j in 0..3 { h q[0]; } }";
        assert!(
            compile_to_qasm(source).is_err(),
            "unsupported multi_capture must fail loudly"
        );
    }

    #[test]
    fn test_control_flow_reject_break() {
        let source = "pub fn main(n: int) -> unit { mut q := qalloc(4); for i in 0..3 { break; } }";
        assert!(
            compile_to_qasm(source).is_err(),
            "unsupported break must fail loudly"
        );
    }

    #[test]
    fn test_control_flow_reject_continue() {
        let source =
            "pub fn main(n: int) -> unit { mut q := qalloc(4); for i in 0..3 { continue; } }";
        assert!(
            compile_to_qasm(source).is_err(),
            "unsupported continue must fail loudly"
        );
    }

    #[test]
    fn test_control_flow_reject_runtime_if() {
        let source = "pub fn main(n: int) -> unit { mut q := qalloc(4); mut c := mz(u1) q[0]; if c { x q[0]; } }";
        assert!(
            compile_to_qasm(source).is_err(),
            "unsupported runtime_if must fail loudly"
        );
    }

    #[test]
    fn test_empty_program() {
        let qasm = compile_to_qasm("").unwrap();
        assert!(qasm.contains("OPENQASM 2.0;"));
    }

    #[test]
    fn test_single_qubit_gate() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(1);
                h q[0];
            }
        "#;

        let qasm = compile_to_qasm(source).unwrap();
        assert!(qasm.contains("qreg q[1];"));
        assert!(qasm.contains("h q[0];"));
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

        let qasm = compile_to_qasm(source).unwrap();
        assert!(qasm.contains("qreg q[2];"));
        assert!(qasm.contains("h q[0];"));
        assert!(qasm.contains("cx q[0], q[1];"));
    }

    #[test]
    fn test_rotation_gate() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(1);
                rz(1.57 rad) q[0];
            }
        "#;

        let qasm = compile_to_qasm(source).unwrap();
        assert!(qasm.contains("rz(1.57) q[0];"));
    }

    #[test]
    fn test_measurement() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(2);
                h q[0];
                r := mz(u1) q[0];
            }
        "#;

        let qasm = compile_to_qasm(source).unwrap();
        assert!(qasm.contains("creg c[1];"));
        assert!(qasm.contains("measure q[0] -> c[0];"));
    }

    #[test]
    fn test_batch_gate() {
        // Use new batch gate syntax: h {targets}
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(3);
                h {q[0], q[1], q[2]};
            }
        "#;

        let qasm = compile_to_qasm(source).unwrap();
        assert!(qasm.contains("h q[0];"));
        assert!(qasm.contains("h q[1];"));
        assert!(qasm.contains("h q[2];"));
    }

    #[test]
    fn test_batch_cx() {
        // Use new batch gate syntax: cx {(ctrl, target), ...}
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(4);
                cx {(q[0], q[1]), (q[2], q[3])};
            }
        "#;

        let qasm = compile_to_qasm(source).unwrap();
        assert!(qasm.contains("cx q[0], q[1];"));
        assert!(qasm.contains("cx q[2], q[3];"));
    }

    #[test]
    fn test_pi_constant() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(1);
                rz((pi / 4) rad) q[0];
            }
        "#;

        let qasm = compile_to_qasm(source).unwrap();
        assert!(qasm.contains("rz(0.7853981633974483) q[0];"));
    }

    #[test]
    fn test_tick_flattened() {
        let source = r#"
            pub fn main() -> unit {
                mut q := qalloc(2);
                tick {
                    h q[0];
                    h q[1];
                }
            }
        "#;

        let qasm = compile_to_qasm(source).unwrap();
        // Tick blocks are flattened in QASM
        assert!(qasm.contains("h q[0];"));
        assert!(qasm.contains("h q[1];"));
    }

    #[test]
    fn test_multiple_allocators() {
        let source = r#"
            pub fn main() -> unit {
                mut data := qalloc(2);
                mut ancilla := qalloc(1);
                h data[0];
                cx (data[0], ancilla[0]);
            }
        "#;

        let qasm = compile_to_qasm(source).unwrap();
        // Total qubits = 2 + 1 = 3
        assert!(qasm.contains("qreg q[3];"));
        // data[0] = q[0], ancilla[0] = q[2]
        assert!(qasm.contains("h q[0];"));
        assert!(qasm.contains("cx q[0], q[2];"));
    }
}
