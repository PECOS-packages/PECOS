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

use std::collections::BTreeMap;
use std::fmt::Write;
use thiserror::Error;

use crate::ast::{
    BinaryOp, Binding, Block, CallExpr, ElseBranch, Expr, FnDecl, Program, Stmt, TopLevelDecl,
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

    #[error("unsupported control flow in QASM 2.0")]
    UnsupportedControlFlow,

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
    /// Total qubit count.
    total_qubits: usize,
    /// Classical register counter.
    creg_counter: usize,
    /// Output buffer.
    output: String,
}

impl QasmCodegen {
    /// Create a new QASM code generator.
    pub fn new() -> Self {
        Self {
            allocators: BTreeMap::new(),
            total_qubits: 0,
            creg_counter: 0,
            output: String::new(),
        }
    }

    /// Compile a Zlup program to OpenQASM 2.0.
    pub fn compile(&mut self, program: &Program) -> QasmResult<String> {
        // Reset state
        self.allocators.clear();
        self.total_qubits = 0;
        self.creg_counter = 0;
        self.output.clear();

        // First pass: collect allocators
        for decl in &program.declarations {
            self.collect_decl(decl)?;
        }

        // Write header
        writeln!(self.output, "OPENQASM 2.0;")?;
        writeln!(self.output, "include \"qelib1.inc\";")?;
        writeln!(self.output)?;

        // Write qubit register
        if self.total_qubits > 0 {
            writeln!(self.output, "qreg q[{}];", self.total_qubits)?;
        }

        // Second pass: convert statements and count measurements
        let mut body_output = String::new();
        let mut measurement_count = 0;
        for decl in &program.declarations {
            if let TopLevelDecl::Fn(fn_decl) = decl
                && fn_decl.name == "main"
            {
                let (body, mcount) = self.convert_block(&fn_decl.body)?;
                body_output = body;
                measurement_count = mcount;
            }
        }

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
        self.total_qubits = 0;
        self.creg_counter = 0;
        self.output.clear();

        // Collect from function body
        self.collect_block(&fn_decl.body)?;

        // Write header
        writeln!(self.output, "OPENQASM 2.0;")?;
        writeln!(self.output, "include \"qelib1.inc\";")?;
        writeln!(self.output)?;

        // Write qubit register
        if self.total_qubits > 0 {
            writeln!(self.output, "qreg q[{}];", self.total_qubits)?;
        }

        // Convert body
        let (body, measurement_count) = self.convert_block(&fn_decl.body)?;

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

    fn collect_decl(&mut self, decl: &TopLevelDecl) -> QasmResult<()> {
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

    fn collect_binding(&mut self, binding: &Binding) -> QasmResult<()> {
        if let Some(ref value) = binding.value {
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
            } else if let Some((parent, size)) = self.try_extract_child_allocator(value) {
                // Child allocator shares parent's qubits
                if let Some(parent_info) = self.allocators.get(&parent) {
                    let offset = parent_info.offset;
                    self.allocators.insert(
                        binding.name.clone(),
                        AllocatorInfo {
                            name: binding.name.clone(),
                            capacity: size,
                            offset,
                        },
                    );
                }
            }
        }
        Ok(())
    }

    fn collect_block(&mut self, block: &Block) -> QasmResult<()> {
        for stmt in &block.statements {
            self.collect_stmt(stmt)?;
        }
        Ok(())
    }

    fn collect_stmt(&mut self, stmt: &Stmt) -> QasmResult<()> {
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
                for inner_stmt in &tick_stmt.body {
                    self.collect_stmt(inner_stmt)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn collect_else_branch(&mut self, branch: &ElseBranch) -> QasmResult<()> {
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

    /// Convert a block, returning (output, measurement_count)
    fn convert_block(&mut self, block: &Block) -> QasmResult<(String, usize)> {
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
        match stmt {
            Stmt::Expr(expr_stmt) => self.convert_expr_stmt(expr_stmt),
            Stmt::Tick(tick_stmt) => {
                // Tick blocks are flattened - operations within are just sequential in QASM
                let mut output = String::new();
                let mut measurement_count = 0;

                // Add barrier to mark tick boundary (optional but useful)
                if !tick_stmt.body.is_empty() {
                    writeln!(
                        output,
                        "// tick{}",
                        tick_stmt
                            .label
                            .as_ref()
                            .map(|l| format!(" {}", l))
                            .unwrap_or_default()
                    )?;
                }

                for inner_stmt in &tick_stmt.body {
                    let (stmt_output, mcount) = self.convert_stmt(inner_stmt)?;
                    output.push_str(&stmt_output);
                    measurement_count += mcount;
                }

                Ok((output, measurement_count))
            }
            Stmt::Block(block) => self.convert_block(block),
            // Handle declarations - check for measurement calls
            Stmt::Binding(binding) => {
                if let Some(ref value) = binding.value {
                    self.convert_decl_value(value)
                } else {
                    Ok((String::new(), 0))
                }
            }
            // Skip unsupported control flow in QASM 2.0
            // (QASM 3.0 would support these)
            Stmt::If(_) | Stmt::For(_) => {
                // For now, skip control flow - could emit warning
                Ok((String::new(), 0))
            }
            _ => Ok((String::new(), 0)),
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
                .map(|p| self.convert_expression(p))
                .collect::<Result<_, _>>()?;
            return self.convert_batch_gate(&gate_info, &set_expr.elements, &params);
        }

        // Convert parameters
        let params: Vec<String> = gate
            .params
            .iter()
            .map(|p| self.convert_expression(p))
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
            let param = self.convert_expression(call.args.last().unwrap())?;
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
        match expr {
            Expr::IntLit(lit) => Ok(lit.value as usize),
            _ => Err(QasmError::UnsupportedExpression),
        }
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
    use crate::parse;

    fn compile_to_qasm(source: &str) -> QasmResult<String> {
        let program = parse(source).expect("parse failed");
        let mut codegen = QasmCodegen::new();
        codegen.compile(&program)
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
                rz(1.57) q[0];
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
                rz(pi / 4) q[0];
            }
        "#;

        let qasm = compile_to_qasm(source).unwrap();
        assert!(qasm.contains("rz((pi / 4)) q[0];"));
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
