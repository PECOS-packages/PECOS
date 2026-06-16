//! Optimization passes for Zlup AST.
//!
//! This module provides various optimization passes that transform the AST
//! to produce more efficient code. All optimizations preserve program semantics.
//!
//! ## Available Passes
//!
//! - **Constant Folding**: Evaluate constant expressions at compile time
//! - **Dead Code Elimination**: Remove unreachable code
//! - **Unused Binding Elimination**: Remove unused variable declarations
//! - **Gate Cancellation**: Cancel adjacent inverse quantum gates (X X = I, H H = I)
//! - **Identity Removal**: Remove gates that have no effect
//!
//! ## Usage
//!
//! ```rust
//! use zlup::optimize::Optimizer;
//!
//! let source = "fn main() -> unit { x := 1 + 2; return unit; }";
//! let program = zlup::parse(source).expect("parse failed");
//! let mut optimizer = Optimizer::new();
//! let optimized = optimizer.optimize(program);
//! // optimized has constant expressions folded (1 + 2 -> 3)
//! ```

use std::collections::BTreeSet;

use crate::ast::{
    Attribute, BinaryExpr, BinaryOp, Binding, Block, BoolLit, ElseBranch, Expr, FloatLit,
    ForRange, ForStmt, GateKind, GateOp, IfStmt, IntLit, Program, SlotRef, Stmt, TopLevelDecl,
    UnaryExpr, UnaryOp,
};
use crate::comptime::{ComptimeEvaluator, ComptimeValue};

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for the optimizer.
#[derive(Debug, Clone)]
pub struct OptimizeConfig {
    /// Enable constant folding
    pub constant_folding: bool,
    /// Enable dead code elimination
    pub dead_code_elimination: bool,
    /// Enable unused binding elimination
    pub unused_binding_elimination: bool,
    /// Enable gate cancellation
    pub gate_cancellation: bool,
    /// Enable identity gate removal
    pub identity_removal: bool,
    /// Enable inline for loop unrolling
    pub inline_for_unrolling: bool,
    /// Maximum number of optimization iterations
    pub max_iterations: usize,
    /// Maximum number of iterations for inline for unrolling (safety limit)
    pub max_inline_for_iterations: usize,
}

impl Default for OptimizeConfig {
    fn default() -> Self {
        Self {
            constant_folding: true,
            dead_code_elimination: true,
            unused_binding_elimination: true,
            gate_cancellation: true,
            identity_removal: true,
            inline_for_unrolling: true,
            max_iterations: 10,
            max_inline_for_iterations: 1024,
        }
    }
}

impl OptimizeConfig {
    /// Create a config with all optimizations enabled.
    pub fn all() -> Self {
        Self::default()
    }

    /// Create a config with no optimizations.
    pub fn none() -> Self {
        Self {
            constant_folding: false,
            dead_code_elimination: false,
            unused_binding_elimination: false,
            gate_cancellation: false,
            identity_removal: false,
            inline_for_unrolling: false,
            max_iterations: 0,
            max_inline_for_iterations: 0,
        }
    }
}

// =============================================================================
// Statistics
// =============================================================================

/// Statistics about optimizations performed.
#[derive(Debug, Clone, Default)]
pub struct OptimizeStats {
    /// Number of constants folded
    pub constants_folded: usize,
    /// Number of dead code blocks removed
    pub dead_code_removed: usize,
    /// Number of unused bindings removed
    pub unused_bindings_removed: usize,
    /// Number of gate pairs cancelled
    pub gates_cancelled: usize,
    /// Number of identity gates removed
    pub identities_removed: usize,
    /// Number of inline for loops unrolled
    pub inline_for_unrolled: usize,
    /// Total number of statements generated from inline for unrolling
    pub inline_for_statements_generated: usize,
    /// Number of optimization iterations performed
    pub iterations: usize,
}

// =============================================================================
// Optimizer
// =============================================================================

/// AST optimizer.
pub struct Optimizer {
    config: OptimizeConfig,
    stats: OptimizeStats,
    evaluator: ComptimeEvaluator,
}

impl Default for Optimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimizer {
    /// Create a new optimizer with default configuration.
    pub fn new() -> Self {
        Self {
            config: OptimizeConfig::default(),
            evaluator: ComptimeEvaluator::new(),
            stats: OptimizeStats::default(),
        }
    }

    /// Create a new optimizer with specific configuration.
    pub fn with_config(config: OptimizeConfig) -> Self {
        Self {
            config,
            evaluator: ComptimeEvaluator::new(),
            stats: OptimizeStats::default(),
        }
    }

    /// Get optimization statistics.
    pub fn stats(&self) -> &OptimizeStats {
        &self.stats
    }

    /// Optimize a program.
    pub fn optimize(&mut self, program: Program) -> Program {
        let mut result = program;

        for iteration in 0..self.config.max_iterations {
            let before_stats = self.stats.clone();

            result = self.optimize_pass(result);

            self.stats.iterations = iteration + 1;

            // Check if any optimizations were performed
            if self.stats.constants_folded == before_stats.constants_folded
                && self.stats.dead_code_removed == before_stats.dead_code_removed
                && self.stats.unused_bindings_removed == before_stats.unused_bindings_removed
                && self.stats.gates_cancelled == before_stats.gates_cancelled
                && self.stats.identities_removed == before_stats.identities_removed
            {
                // No changes, stop iterating
                break;
            }
        }

        result
    }

    /// Run one optimization pass.
    fn optimize_pass(&mut self, program: Program) -> Program {
        let mut declarations = Vec::new();

        for decl in program.declarations {
            declarations.push(self.optimize_decl(decl));
        }

        Program {
            name: program.name,
            declarations,
            location: program.location,
        }
    }

    /// Optimize a top-level declaration.
    fn optimize_decl(&mut self, decl: TopLevelDecl) -> TopLevelDecl {
        match decl {
            TopLevelDecl::Fn(mut fn_decl) => {
                fn_decl.body = self.optimize_block(fn_decl.body);
                TopLevelDecl::Fn(fn_decl)
            }
            TopLevelDecl::Binding(binding) => {
                TopLevelDecl::Binding(self.optimize_binding(binding))
            }
            other => other,
        }
    }

    /// Optimize a block of statements.
    fn optimize_block(&mut self, block: Block) -> Block {
        let mut statements = Vec::new();

        // First pass: optimize individual statements
        for stmt in block.statements {
            if let Some(optimized) = self.optimize_stmt(stmt) {
                statements.push(optimized);
            }
        }

        // Gate cancellation pass
        if self.config.gate_cancellation {
            statements = self.cancel_gates(statements);
        }

        // Unused binding elimination
        if self.config.unused_binding_elimination {
            statements = self.eliminate_unused_bindings(statements);
        }

        Block {
            label: block.label,
            attrs: block.attrs,
            statements,
            trailing_expr: block.trailing_expr,
            location: block.location,
        }
    }

    /// Optimize a statement. Returns None if the statement should be removed.
    fn optimize_stmt(&mut self, stmt: Stmt) -> Option<Stmt> {
        match stmt {
            Stmt::Binding(binding) => Some(Stmt::Binding(self.optimize_binding(binding))),

            Stmt::Expr(expr_stmt) => {
                // Check for identity gates (e.g., rz(0)) before optimization
                if self.config.identity_removal
                    && let Expr::Gate(ref gate) = expr_stmt.expr
                        && let Some(angle) = gate.params.first()
                            && self.is_zero_angle(angle) {
                                self.stats.identities_removed += 1;
                                return None; // Remove the identity gate
                            }
                let optimized = self.optimize_expr(expr_stmt.expr);
                Some(Stmt::Expr(crate::ast::ExprStmt {
                    expr: optimized,
                    attrs: expr_stmt.attrs,
                    location: expr_stmt.location,
                }))
            }

            Stmt::If(if_stmt) => self.optimize_if(if_stmt),

            Stmt::For(for_stmt) => {
                // Try to unroll inline for loops
                if self.config.inline_for_unrolling && for_stmt.is_inline {
                    if let Some(unrolled) = self.try_unroll_inline_for(&for_stmt) {
                        // Return the unrolled statements as a block
                        return Some(Stmt::Block(Block {
                            label: for_stmt.label.clone(),
                            attrs: vec![],
                            statements: unrolled,
                            trailing_expr: None,
                            location: for_stmt.location.clone(),
                        }));
                    }
                }
                Some(Stmt::For(self.optimize_for(for_stmt)))
            }

            Stmt::Block(block) => Some(Stmt::Block(self.optimize_block(block))),

            Stmt::Gate(gate_op) => self.optimize_gate_stmt(gate_op),

            Stmt::Return(ret) => {
                let mut optimized = ret;
                if let Some(expr) = optimized.value {
                    optimized.value = Some(self.optimize_expr(expr));
                }
                Some(Stmt::Return(optimized))
            }

            Stmt::Tick(tick_stmt) => {
                // Optimize statements within the tick block individually
                // Note: The tick block itself acts as a barrier for cross-block optimization,
                // but gates WITHIN the same tick can still cancel each other
                let optimized_stmts: Vec<Stmt> = tick_stmt
                    .body
                    .into_iter()
                    .filter_map(|stmt| self.optimize_stmt(stmt))
                    .collect();

                // Apply gate cancellation within the tick
                let optimized_stmts = if self.config.gate_cancellation {
                    self.cancel_gates(optimized_stmts)
                } else {
                    optimized_stmts
                };

                Some(Stmt::Tick(crate::ast::TickStmt {
                    label: tick_stmt.label,
                    body: optimized_stmts,
                    attrs: tick_stmt.attrs,
                    location: tick_stmt.location,
                }))
            }

            // Pass through other statements
            other => Some(other),
        }
    }

    /// Optimize a binding.
    fn optimize_binding(&mut self, binding: Binding) -> Binding {
        let mut optimized = binding;
        if let Some(value) = optimized.value {
            optimized.value = Some(self.optimize_expr(value));
        }
        optimized
    }

    /// Optimize an expression.
    fn optimize_expr(&mut self, expr: Expr) -> Expr {
        // Try constant folding first
        if self.config.constant_folding
            && let Some(folded) = self.try_fold_constant(&expr) {
                self.stats.constants_folded += 1;
                return folded;
            }

        // Recursively optimize subexpressions
        match expr {
            Expr::Binary(bin) => {
                let left = self.optimize_expr(bin.left.clone());
                let right = self.optimize_expr(bin.right.clone());

                // Try folding after optimizing children
                let new_bin = BinaryExpr {
                    op: bin.op,
                    left,
                    right,
                    location: bin.location,
                };

                if self.config.constant_folding
                    && let Some(folded) = self.try_fold_binary(&new_bin) {
                        self.stats.constants_folded += 1;
                        return folded;
                    }

                Expr::Binary(Box::new(new_bin))
            }

            Expr::Unary(un) => {
                let operand = self.optimize_expr(un.operand.clone());

                let new_un = UnaryExpr {
                    op: un.op,
                    operand,
                    location: un.location,
                };

                if self.config.constant_folding
                    && let Some(folded) = self.try_fold_unary(&new_un) {
                        self.stats.constants_folded += 1;
                        return folded;
                    }

                Expr::Unary(Box::new(new_un))
            }

            Expr::Call(mut call) => {
                call.args = call.args.into_iter().map(|a| self.optimize_expr(a)).collect();
                Expr::Call(call)
            }

            Expr::Index(mut idx) => {
                idx.object = self.optimize_expr(idx.object);
                idx.index = self.optimize_expr(idx.index);
                Expr::Index(idx)
            }

            Expr::Field(mut field) => {
                field.object = self.optimize_expr(field.object);
                Expr::Field(field)
            }

            Expr::Tuple(mut tuple) => {
                tuple.elements = tuple.elements.into_iter().map(|e| self.optimize_expr(e)).collect();
                Expr::Tuple(tuple)
            }

            Expr::BracketArray(mut arr) => {
                arr.elements = arr.elements.into_iter().map(|e| self.optimize_expr(e)).collect();
                Expr::BracketArray(arr)
            }

            // Pass through other expressions
            other => other,
        }
    }

    /// Try to fold an expression to a constant.
    fn try_fold_constant(&mut self, expr: &Expr) -> Option<Expr> {
        match self.evaluator.eval_expr(expr) {
            Ok(value) => self.comptime_to_expr(&value, expr),
            Err(_) => None,
        }
    }

    /// Try to fold a binary expression.
    fn try_fold_binary(&mut self, bin: &BinaryExpr) -> Option<Expr> {
        // Check for identity operations
        match (&bin.left, bin.op, &bin.right) {
            // x + 0 = x, x - 0 = x
            (_, BinaryOp::Add | BinaryOp::Sub, Expr::IntLit(IntLit { value: 0, .. })) => {
                return Some(bin.left.clone());
            }
            // 0 + x = x
            (Expr::IntLit(IntLit { value: 0, .. }), BinaryOp::Add, _) => {
                return Some(bin.right.clone());
            }
            // x * 1 = x, x / 1 = x
            (_, BinaryOp::Mul | BinaryOp::Div, Expr::IntLit(IntLit { value: 1, .. })) => {
                return Some(bin.left.clone());
            }
            // 1 * x = x
            (Expr::IntLit(IntLit { value: 1, .. }), BinaryOp::Mul, _) => {
                return Some(bin.right.clone());
            }
            // x * 0 = 0
            (_, BinaryOp::Mul, Expr::IntLit(IntLit { value: 0, .. })) => {
                return Some(Expr::IntLit(IntLit {
                    value: 0,
                    suffix: None,
                    location: bin.location.clone(),
                }));
            }
            // 0 * x = 0
            (Expr::IntLit(IntLit { value: 0, .. }), BinaryOp::Mul, _) => {
                return Some(Expr::IntLit(IntLit {
                    value: 0,
                    suffix: None,
                    location: bin.location.clone(),
                }));
            }
            // x && true = x, x || false = x
            (_, BinaryOp::And, Expr::BoolLit(BoolLit { value: true, .. })) => {
                return Some(bin.left.clone());
            }
            (_, BinaryOp::Or, Expr::BoolLit(BoolLit { value: false, .. })) => {
                return Some(bin.left.clone());
            }
            // true && x = x, false || x = x
            (Expr::BoolLit(BoolLit { value: true, .. }), BinaryOp::And, _) => {
                return Some(bin.right.clone());
            }
            (Expr::BoolLit(BoolLit { value: false, .. }), BinaryOp::Or, _) => {
                return Some(bin.right.clone());
            }
            // x && false = false, x || true = true
            (_, BinaryOp::And, Expr::BoolLit(BoolLit { value: false, .. })) => {
                return Some(Expr::BoolLit(BoolLit {
                    value: false,
                    location: bin.location.clone(),
                }));
            }
            (_, BinaryOp::Or, Expr::BoolLit(BoolLit { value: true, .. })) => {
                return Some(Expr::BoolLit(BoolLit {
                    value: true,
                    location: bin.location.clone(),
                }));
            }
            _ => {}
        }

        // Try full constant evaluation
        self.try_fold_constant(&Expr::Binary(Box::new(bin.clone())))
    }

    /// Try to fold a unary expression.
    fn try_fold_unary(&mut self, un: &UnaryExpr) -> Option<Expr> {
        // Double negation: --x = x, !!x = x
        if let Expr::Unary(inner) = &un.operand
            && un.op == inner.op && matches!(un.op, UnaryOp::Neg | UnaryOp::Not) {
                return Some(inner.operand.clone());
            }

        // Try full constant evaluation
        self.try_fold_constant(&Expr::Unary(Box::new(un.clone())))
    }

    /// Convert a comptime value to an AST expression.
    fn comptime_to_expr(&self, value: &ComptimeValue, original: &Expr) -> Option<Expr> {
        let location = match original {
            Expr::Binary(b) => b.location.clone(),
            Expr::Unary(u) => u.location.clone(),
            Expr::IntLit(i) => i.location.clone(),
            Expr::FloatLit(f) => f.location.clone(),
            Expr::BoolLit(b) => b.location.clone(),
            _ => None,
        };

        match value {
            ComptimeValue::Int(v) => Some(Expr::IntLit(IntLit {
                value: *v as i128,
                suffix: None,
                location,
            })),
            ComptimeValue::Uint(v) => Some(Expr::IntLit(IntLit {
                value: *v as i128,
                suffix: None,
                location,
            })),
            ComptimeValue::Float(v) => Some(Expr::FloatLit(FloatLit {
                value: *v,
                suffix: None,
                location,
            })),
            ComptimeValue::Bool(v) => Some(Expr::BoolLit(BoolLit {
                value: *v,
                location,
            })),
            _ => None, // Don't fold complex types
        }
    }

    /// Optimize an if statement.
    fn optimize_if(&mut self, if_stmt: IfStmt) -> Option<Stmt> {
        let condition = self.optimize_expr(if_stmt.condition);

        // Check for constant condition
        if self.config.dead_code_elimination
            && let Expr::BoolLit(BoolLit { value, .. }) = &condition {
                self.stats.dead_code_removed += 1;
                if *value {
                    // Condition is always true - keep then branch
                    return Some(Stmt::Block(self.optimize_block(if_stmt.then_body)));
                } else {
                    // Condition is always false - keep else branch or remove
                    return match if_stmt.else_body {
                        Some(ElseBranch::Else(block)) => {
                            Some(Stmt::Block(self.optimize_block(block)))
                        }
                        Some(ElseBranch::ElseIf(nested)) => self.optimize_if(*nested),
                        None => None,
                    };
                }
            }

        let then_body = self.optimize_block(if_stmt.then_body);
        let else_body = match if_stmt.else_body {
            Some(ElseBranch::Else(block)) => Some(ElseBranch::Else(self.optimize_block(block))),
            Some(ElseBranch::ElseIf(nested)) => {
                if let Some(Stmt::If(optimized)) = self.optimize_if(*nested) {
                    Some(ElseBranch::ElseIf(Box::new(optimized)))
                } else {
                    None
                }
            }
            None => None,
        };

        Some(Stmt::If(IfStmt {
            condition,
            capture: if_stmt.capture,
            then_body,
            else_body,
            location: if_stmt.location,
        }))
    }

    /// Optimize a for statement.
    fn optimize_for(&mut self, for_stmt: ForStmt) -> ForStmt {
        let body = self.optimize_block(for_stmt.body);

        // Optimize range bounds
        let range = match for_stmt.range {
            ForRange::Range { start, end } => ForRange::Range {
                start: self.optimize_expr(start),
                end: self.optimize_expr(end),
            },
            ForRange::Collection(expr) => ForRange::Collection(self.optimize_expr(expr)),
        };

        ForStmt {
            label: for_stmt.label,
            is_inline: for_stmt.is_inline,
            range,
            captures: for_stmt.captures,
            body,
            location: for_stmt.location,
        }
    }

    // =========================================================================
    // Inline For Loop Unrolling
    // =========================================================================

    /// Try to unroll an inline for loop.
    ///
    /// Returns `Some(unrolled_statements)` if the loop can be unrolled, `None` otherwise.
    /// Unrolling is possible when:
    /// - The loop is marked as `inline`
    /// - The range bounds are comptime-evaluable
    /// - The iteration count is within the safety limit
    fn try_unroll_inline_for(&mut self, for_stmt: &ForStmt) -> Option<Vec<Stmt>> {
        if !for_stmt.is_inline {
            return None;
        }

        // Evaluate range bounds at comptime
        let (start, end) = match &for_stmt.range {
            ForRange::Range { start, end } => {
                let start_val = self.evaluator.eval_expr(start).ok()?.as_int()?;
                let end_val = self.evaluator.eval_expr(end).ok()?.as_int()?;
                (start_val, end_val)
            }
            ForRange::Collection(expr) => {
                // For collections, evaluate and get the length
                let val = self.evaluator.eval_expr(expr).ok()?;
                match val {
                    ComptimeValue::Array(arr) => (0, arr.len() as i64),
                    _ => return None,
                }
            }
        };

        // Check iteration count against safety limit
        let iteration_count = (end - start).max(0) as usize;
        if iteration_count > self.config.max_inline_for_iterations {
            return None;
        }

        // Get the capture variable name
        let capture_name = for_stmt.captures.first()?;

        // Generate unrolled statements
        let mut unrolled = Vec::with_capacity(iteration_count * for_stmt.body.statements.len());
        for i in start..end {
            // Substitute the loop variable with the concrete value in each statement
            for stmt in &for_stmt.body.statements {
                let substituted = self.substitute_in_stmt(stmt, capture_name, i);
                // Recursively optimize the substituted statement (handles nested inline for)
                if let Some(optimized) = self.optimize_stmt(substituted) {
                    // If the optimized statement is a block (from nested unrolling), flatten it
                    match optimized {
                        Stmt::Block(block) => unrolled.extend(block.statements),
                        other => unrolled.push(other),
                    }
                }
            }
        }

        self.stats.inline_for_unrolled += 1;
        self.stats.inline_for_statements_generated += unrolled.len();

        Some(unrolled)
    }

    /// Substitute a variable with a concrete integer value in a statement.
    fn substitute_in_stmt(&self, stmt: &Stmt, var_name: &str, value: i64) -> Stmt {
        match stmt {
            Stmt::Binding(binding) => Stmt::Binding(Binding {
                name: binding.name.clone(),
                ty: binding.ty.clone(),
                value: binding.value.as_ref().map(|e| self.substitute_in_expr(e, var_name, value)),
                is_mutable: binding.is_mutable,
                is_pub: binding.is_pub,
                doc_comment: binding.doc_comment.clone(),
                location: binding.location.clone(),
            }),

            Stmt::Expr(expr_stmt) => Stmt::Expr(crate::ast::ExprStmt {
                expr: self.substitute_in_expr(&expr_stmt.expr, var_name, value),
                attrs: expr_stmt.attrs.clone(),
                location: expr_stmt.location.clone(),
            }),

            Stmt::Gate(gate_op) => Stmt::Gate(GateOp {
                kind: gate_op.kind,
                targets: gate_op
                    .targets
                    .iter()
                    .map(|t| self.substitute_in_slot_ref(t, var_name, value))
                    .collect(),
                params: gate_op
                    .params
                    .iter()
                    .map(|e| self.substitute_in_expr(e, var_name, value))
                    .collect(),
                attrs: gate_op.attrs.clone(),
                location: gate_op.location.clone(),
            }),

            Stmt::If(if_stmt) => Stmt::If(IfStmt {
                condition: self.substitute_in_expr(&if_stmt.condition, var_name, value),
                capture: if_stmt.capture.clone(),
                then_body: self.substitute_in_block(&if_stmt.then_body, var_name, value),
                else_body: if_stmt.else_body.as_ref().map(|eb| match eb {
                    ElseBranch::Else(block) => {
                        ElseBranch::Else(self.substitute_in_block(block, var_name, value))
                    }
                    ElseBranch::ElseIf(nested) => {
                        if let Stmt::If(nested_if) = self.substitute_in_stmt(&Stmt::If(*nested.clone()), var_name, value) {
                            ElseBranch::ElseIf(Box::new(nested_if))
                        } else {
                            eb.clone()
                        }
                    }
                }),
                location: if_stmt.location.clone(),
            }),

            Stmt::For(for_stmt) => Stmt::For(ForStmt {
                label: for_stmt.label.clone(),
                is_inline: for_stmt.is_inline,
                range: match &for_stmt.range {
                    ForRange::Range { start, end } => ForRange::Range {
                        start: self.substitute_in_expr(start, var_name, value),
                        end: self.substitute_in_expr(end, var_name, value),
                    },
                    ForRange::Collection(expr) => {
                        ForRange::Collection(self.substitute_in_expr(expr, var_name, value))
                    }
                },
                captures: for_stmt.captures.clone(),
                body: self.substitute_in_block(&for_stmt.body, var_name, value),
                location: for_stmt.location.clone(),
            }),

            Stmt::Block(block) => Stmt::Block(self.substitute_in_block(block, var_name, value)),

            Stmt::Return(ret) => Stmt::Return(crate::ast::ReturnStmt {
                value: ret.value.as_ref().map(|e| self.substitute_in_expr(e, var_name, value)),
                location: ret.location.clone(),
            }),

            Stmt::Assign(assign) => Stmt::Assign(crate::ast::AssignStmt {
                target: self.substitute_in_expr(&assign.target, var_name, value),
                op: assign.op,
                value: self.substitute_in_expr(&assign.value, var_name, value),
                location: assign.location.clone(),
            }),

            Stmt::Tick(tick) => Stmt::Tick(crate::ast::TickStmt {
                label: tick.label.clone(),
                attrs: tick.attrs.clone(),
                body: tick.body.iter().map(|s| self.substitute_in_stmt(s, var_name, value)).collect(),
                location: tick.location.clone(),
            }),

            // Pass through statements that don't contain expressions
            other => other.clone(),
        }
    }

    /// Substitute a variable with a concrete integer value in a block.
    fn substitute_in_block(&self, block: &Block, var_name: &str, value: i64) -> Block {
        Block {
            label: block.label.clone(),
            attrs: block.attrs.clone(),
            statements: block
                .statements
                .iter()
                .map(|s| self.substitute_in_stmt(s, var_name, value))
                .collect(),
            trailing_expr: block
                .trailing_expr
                .as_ref()
                .map(|e| Box::new(self.substitute_in_expr(e, var_name, value))),
            location: block.location.clone(),
        }
    }

    /// Substitute a variable with a concrete integer value in an expression.
    fn substitute_in_expr(&self, expr: &Expr, var_name: &str, value: i64) -> Expr {
        match expr {
            Expr::Ident(ident) if ident.name == var_name => {
                // Replace identifier with the concrete value
                Expr::IntLit(IntLit {
                    value: value as i128,
                    suffix: None,
                    location: ident.location.clone(),
                })
            }

            Expr::Binary(bin) => Expr::Binary(Box::new(BinaryExpr {
                op: bin.op,
                left: self.substitute_in_expr(&bin.left, var_name, value),
                right: self.substitute_in_expr(&bin.right, var_name, value),
                location: bin.location.clone(),
            })),

            Expr::Unary(un) => Expr::Unary(Box::new(UnaryExpr {
                op: un.op,
                operand: self.substitute_in_expr(&un.operand, var_name, value),
                location: un.location.clone(),
            })),

            Expr::Index(idx) => Expr::Index(Box::new(crate::ast::IndexExpr {
                object: self.substitute_in_expr(&idx.object, var_name, value),
                index: self.substitute_in_expr(&idx.index, var_name, value),
                location: idx.location.clone(),
            })),

            Expr::Field(field) => Expr::Field(Box::new(crate::ast::FieldExpr {
                object: self.substitute_in_expr(&field.object, var_name, value),
                field: field.field.clone(),
                location: field.location.clone(),
            })),

            Expr::Call(call) => Expr::Call(Box::new(crate::ast::CallExpr {
                callee: self.substitute_in_expr(&call.callee, var_name, value),
                args: call.args.iter().map(|a| self.substitute_in_expr(a, var_name, value)).collect(),
                location: call.location.clone(),
            })),

            Expr::Tuple(tuple) => Expr::Tuple(Box::new(crate::ast::TupleExpr {
                elements: tuple.elements.iter().map(|e| self.substitute_in_expr(e, var_name, value)).collect(),
                location: tuple.location.clone(),
            })),

            Expr::BracketArray(arr) => Expr::BracketArray(Box::new(crate::ast::BracketArrayExpr {
                elements: arr.elements.iter().map(|e| self.substitute_in_expr(e, var_name, value)).collect(),
                location: arr.location.clone(),
            })),

            Expr::Gate(gate) => Expr::Gate(Box::new(crate::ast::GateExpr {
                kind: gate.kind,
                params: gate.params.iter().map(|p| self.substitute_in_expr(p, var_name, value)).collect(),
                target: self.substitute_in_expr(&gate.target, var_name, value),
                location: gate.location.clone(),
            })),

            Expr::SlotRef(slot) => Expr::SlotRef(Box::new(self.substitute_in_slot_ref(slot, var_name, value))),

            Expr::If(if_expr) => Expr::If(Box::new(crate::ast::IfExpr {
                condition: self.substitute_in_expr(&if_expr.condition, var_name, value),
                then_expr: self.substitute_in_expr(&if_expr.then_expr, var_name, value),
                else_expr: self.substitute_in_expr(&if_expr.else_expr, var_name, value),
                location: if_expr.location.clone(),
            })),

            // Pass through expressions that don't contain the variable
            other => other.clone(),
        }
    }

    /// Substitute a variable in a slot reference.
    fn substitute_in_slot_ref(&self, slot: &SlotRef, var_name: &str, value: i64) -> SlotRef {
        SlotRef {
            allocator: slot.allocator.clone(),
            index: Box::new(self.substitute_in_expr(&slot.index, var_name, value)),
            location: slot.location.clone(),
        }
    }

    /// Optimize a gate statement.
    fn optimize_gate_stmt(&mut self, gate_op: GateOp) -> Option<Stmt> {
        // Remove identity rotations (rotation by 0)
        if self.config.identity_removal
            && let Some(angle) = gate_op.params.first()
                && self.is_zero_angle(angle) {
                    self.stats.identities_removed += 1;
                    return None;
                }

        Some(Stmt::Gate(gate_op))
    }

    /// Check if an expression represents a zero angle.
    fn is_zero_angle(&self, expr: &Expr) -> bool {
        match expr {
            Expr::IntLit(IntLit { value: 0, .. }) => true,
            Expr::FloatLit(FloatLit { value, .. }) if *value == 0.0 => true,
            Expr::AngleLit(angle_lit) => self.is_zero_angle(&angle_lit.value),
            _ => false,
        }
    }

    /// Cancel adjacent inverse gates.
    ///
    /// Respects optimization barriers:
    /// - `@attr(preserve, ...)` attribute on gates prevents cancellation
    /// - `@attr(round, n)` attribute changes act as barriers (different rounds don't cancel)
    /// - `@attr(timing, ...)` attribute preserves gates for timing purposes
    /// - `@attr(identity, ...)` explicitly marks intentional identity operations
    /// - Blocks with `@attr(noopt, ...)` act as optimization barriers
    fn cancel_gates(&mut self, statements: Vec<Stmt>) -> Vec<Stmt> {
        let mut result = Vec::new();
        let mut current_round: Option<i64> = None;

        let mut i = 0;
        while i < statements.len() {
            let stmt = &statements[i];

            // Check for optimization barriers
            if self.is_optimization_barrier(stmt) {
                result.push(statements[i].clone());
                i += 1;
                continue;
            }

            // Track round changes - different rounds don't cancel across
            if let Some(round) = self.get_round_attr(stmt) {
                if current_round.is_some() && current_round != Some(round) {
                    // Round changed - this is a barrier
                    current_round = Some(round);
                    result.push(statements[i].clone());
                    i += 1;
                    continue;
                }
                current_round = Some(round);
            }

            // Check if this and the next statement are inverse gates that can cancel
            // Gates can be either Stmt::Gate(GateOp) or Stmt::Expr(ExprStmt { expr: Expr::Gate(...) })
            if i + 1 < statements.len()
                && let (Some((g1, attrs1, no_opt1)), Some((g2, attrs2, no_opt2))) =
                    (self.extract_gate_info(stmt), self.extract_gate_info(&statements[i + 1]))
                {
                    // Don't cancel if either gate is wrapped in @no_optimize or has preserve attr
                    if !no_opt1
                        && !no_opt2
                        && !self.has_preserve_attr(&attrs1)
                        && !self.has_preserve_attr(&attrs2)
                        && !self.is_optimization_barrier(&statements[i + 1])
                        && self.are_inverse_gate_exprs(&g1, &g2)
                    {
                        self.stats.gates_cancelled += 2;
                        i += 2; // Skip both gates
                        continue;
                    }
                }

            result.push(statements[i].clone());
            i += 1;
        }

        result
    }

    /// Extract gate info from a statement (handles both Stmt::Gate and Stmt::Expr with Expr::Gate).
    /// Also handles @no_optimize(gate_expr) wrapped gates.
    fn extract_gate_info(&self, stmt: &Stmt) -> Option<(crate::ast::GateExpr, Vec<Attribute>, bool)> {
        match stmt {
            Stmt::Gate(gate_op) => {
                // Convert GateOp to GateExpr-like structure
                // For GateOp, we need to convert targets to a single Expr
                let target = if gate_op.targets.len() == 1 {
                    Expr::SlotRef(Box::new(gate_op.targets[0].clone()))
                } else {
                    // Multiple targets - create a tuple
                    Expr::Tuple(Box::new(crate::ast::TupleExpr {
                        elements: gate_op
                            .targets
                            .iter()
                            .map(|t| Expr::SlotRef(Box::new(t.clone())))
                            .collect(),
                        location: gate_op.location.clone(),
                    }))
                };
                Some((
                    crate::ast::GateExpr {
                        kind: gate_op.kind,
                        params: gate_op.params.clone(),
                        target,
                        location: gate_op.location.clone(),
                    },
                    gate_op.attrs.clone(),
                    false, // not wrapped in @no_optimize
                ))
            }
            Stmt::Expr(expr_stmt) => {
                // Check for @no_optimize(gate_expr) builtin
                if let Expr::Builtin(builtin) = &expr_stmt.expr
                    && builtin.name == "no_optimize" && builtin.args.len() == 1
                        && let Expr::Gate(gate_expr) = &builtin.args[0] {
                            return Some((
                                *gate_expr.clone(),
                                expr_stmt.attrs.clone(),
                                true, // wrapped in @no_optimize - should not be cancelled
                            ));
                        }
                // Regular gate expression
                if let Expr::Gate(gate_expr) = &expr_stmt.expr {
                    Some((*gate_expr.clone(), expr_stmt.attrs.clone(), false))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Check if two gate expressions are inverses (cancel each other).
    fn are_inverse_gate_exprs(
        &self,
        g1: &crate::ast::GateExpr,
        g2: &crate::ast::GateExpr,
    ) -> bool {
        // Must have same target
        if !self.same_gate_target(&g1.target, &g2.target) {
            return false;
        }

        // Check for self-inverse gates
        match (g1.kind, g2.kind) {
            // H H = I, X X = I, Y Y = I, Z Z = I
            (GateKind::H, GateKind::H)
            | (GateKind::X, GateKind::X)
            | (GateKind::Y, GateKind::Y)
            | (GateKind::Z, GateKind::Z) => true,

            // T Tdg = I, Tdg T = I
            (GateKind::T, GateKind::Tdg) | (GateKind::Tdg, GateKind::T) => true,

            // SX SXdg = I, SZ SZdg = I (SZ is the S gate), etc.
            (GateKind::SX, GateKind::SXdg) | (GateKind::SXdg, GateKind::SX) => true,
            (GateKind::SY, GateKind::SYdg) | (GateKind::SYdg, GateKind::SY) => true,
            (GateKind::SZ, GateKind::SZdg) | (GateKind::SZdg, GateKind::SZ) => true,

            // F Fdg = I, etc.
            (GateKind::F, GateKind::Fdg) | (GateKind::Fdg, GateKind::F) => true,
            (GateKind::F4, GateKind::F4dg) | (GateKind::F4dg, GateKind::F4) => true,

            // Two-qubit self-inverse: CX CX = I, CZ CZ = I, SWAP SWAP = I
            (GateKind::CX, GateKind::CX)
            | (GateKind::CY, GateKind::CY)
            | (GateKind::CZ, GateKind::CZ)
            | (GateKind::CH, GateKind::CH)
            | (GateKind::SWAP, GateKind::SWAP) => true,

            // Rotation cancellation: RX(a) RX(-a) = I
            (GateKind::RX, GateKind::RX)
            | (GateKind::RY, GateKind::RY)
            | (GateKind::RZ, GateKind::RZ) => {
                self.are_inverse_rotations(&g1.params, &g2.params)
            }

            _ => false,
        }
    }

    /// Check if two gate targets are the same.
    fn same_gate_target(&self, t1: &Expr, t2: &Expr) -> bool {
        match (t1, t2) {
            // Single qubit: compare SlotRef
            (Expr::SlotRef(s1), Expr::SlotRef(s2)) => {
                self.same_slot_ref(s1, s2)
            }
            // Index expressions (q[0], q[1], etc.)
            (Expr::Index(idx1), Expr::Index(idx2)) => {
                self.same_index_expr(idx1, idx2)
            }
            // Tuple targets (for multi-qubit gates like cx (q[0], q[1]))
            (Expr::Tuple(tup1), Expr::Tuple(tup2)) => {
                if tup1.elements.len() != tup2.elements.len() {
                    return false;
                }
                tup1.elements.iter().zip(tup2.elements.iter()).all(|(a, b)| {
                    self.same_gate_target(a, b)
                })
            }
            _ => false,
        }
    }

    /// Check if two index expressions are the same (e.g., q[0] == q[0]).
    fn same_index_expr(&self, idx1: &crate::ast::IndexExpr, idx2: &crate::ast::IndexExpr) -> bool {
        // Compare the object (e.g., 'q' in q[0])
        let same_object = match (&idx1.object, &idx2.object) {
            (Expr::Ident(id1), Expr::Ident(id2)) => id1.name == id2.name,
            _ => false,
        };
        if !same_object {
            return false;
        }
        // Compare the index (e.g., '0' in q[0])
        match (&idx1.index, &idx2.index) {
            (Expr::IntLit(lit1), Expr::IntLit(lit2)) => lit1.value == lit2.value,
            _ => false,
        }
    }

    /// Check if two SlotRefs are the same.
    fn same_slot_ref(&self, s1: &SlotRef, s2: &SlotRef) -> bool {
        if s1.allocator != s2.allocator {
            return false;
        }
        // Compare indices
        match (&*s1.index, &*s2.index) {
            (
                Expr::IntLit(IntLit { value: v1, .. }),
                Expr::IntLit(IntLit { value: v2, .. }),
            ) => v1 == v2,
            _ => false,
        }
    }

    /// Check if a statement acts as an optimization barrier.
    ///
    /// Use `@preserve {}` blocks or `tick {}` to prevent optimization across boundaries.
    fn is_optimization_barrier(&self, stmt: &Stmt) -> bool {
        match stmt {
            Stmt::Gate(gate) => self.has_preserve_attr(&gate.attrs),
            Stmt::Block(block) => self.has_preserve_attr(&block.attrs),
            // Tick blocks always act as barriers - they represent time slices
            // Gates in different ticks shouldn't cancel across the boundary
            Stmt::Tick(_) => true,
            _ => false,
        }
    }

    /// Check if attributes contain a preserve marker.
    ///
    /// Recognized preserve attributes:
    /// - `@preserve` - explicit preserve
    /// - `@timing` - preserved for timing purposes
    /// - `@identity` - intentional identity operation
    /// - `@noopt` - no optimization
    ///
    /// Note: `tick {}` blocks are handled separately as always being barriers.
    fn has_preserve_attr(&self, attrs: &[Attribute]) -> bool {
        attrs.iter().any(|attr| {
            matches!(
                attr.name.as_str(),
                "preserve" | "timing" | "identity" | "noopt"
            )
        })
    }

    /// Get the round number from a @round(n) attribute, if present.
    fn get_round_attr(&self, stmt: &Stmt) -> Option<i64> {
        let attrs = match stmt {
            Stmt::Gate(gate) => &gate.attrs,
            Stmt::Block(block) => &block.attrs,
            Stmt::Tick(tick) => &tick.attrs,
            _ => return None,
        };

        for attr in attrs {
            if attr.name == "round"
                && let Some(crate::ast::AttributeValue::Int(n)) = &attr.value {
                    return Some(*n);
                }
        }
        None
    }

    /// Check if two gates are inverses (cancel each other).
    fn are_inverse_gates(&self, g1: &GateOp, g2: &GateOp) -> bool {
        // Must have same targets
        if !self.same_targets(&g1.targets, &g2.targets) {
            return false;
        }

        // Check for self-inverse gates
        match (g1.kind, g2.kind) {
            // H H = I, X X = I, Y Y = I, Z Z = I
            (GateKind::H, GateKind::H)
            | (GateKind::X, GateKind::X)
            | (GateKind::Y, GateKind::Y)
            | (GateKind::Z, GateKind::Z) => true,

            // T Tdg = I, Tdg T = I
            (GateKind::T, GateKind::Tdg) | (GateKind::Tdg, GateKind::T) => true,

            // SX SXdg = I, SZ SZdg = I (SZ is the S gate), etc.
            (GateKind::SX, GateKind::SXdg) | (GateKind::SXdg, GateKind::SX) => true,
            (GateKind::SY, GateKind::SYdg) | (GateKind::SYdg, GateKind::SY) => true,
            (GateKind::SZ, GateKind::SZdg) | (GateKind::SZdg, GateKind::SZ) => true,

            // F Fdg = I, etc.
            (GateKind::F, GateKind::Fdg) | (GateKind::Fdg, GateKind::F) => true,
            (GateKind::F4, GateKind::F4dg) | (GateKind::F4dg, GateKind::F4) => true,

            // Two-qubit self-inverse: CX CX = I, CZ CZ = I, SWAP SWAP = I
            (GateKind::CX, GateKind::CX)
            | (GateKind::CY, GateKind::CY)
            | (GateKind::CZ, GateKind::CZ)
            | (GateKind::CH, GateKind::CH)
            | (GateKind::SWAP, GateKind::SWAP) => true,

            // Rotation cancellation: RX(a) RX(-a) = I
            (GateKind::RX, GateKind::RX)
            | (GateKind::RY, GateKind::RY)
            | (GateKind::RZ, GateKind::RZ) => {
                self.are_inverse_rotations(&g1.params, &g2.params)
            }

            _ => false,
        }
    }

    /// Check if rotation parameters are inverses.
    fn are_inverse_rotations(&self, params1: &[Expr], params2: &[Expr]) -> bool {
        if params1.len() != 1 || params2.len() != 1 {
            return false;
        }

        // Simple check: if one is the negation of the other
        match (&params1[0], &params2[0]) {
            (Expr::IntLit(IntLit { value: a, .. }), Expr::Unary(un)) => {
                if un.op == UnaryOp::Neg
                    && let Expr::IntLit(IntLit { value: b, .. }) = &un.operand {
                        return *a == *b;
                    }
                false
            }
            (Expr::Unary(un), Expr::IntLit(IntLit { value: b, .. })) => {
                if un.op == UnaryOp::Neg
                    && let Expr::IntLit(IntLit { value: a, .. }) = &un.operand {
                        return *a == *b;
                    }
                false
            }
            (
                Expr::FloatLit(FloatLit { value: a, .. }),
                Expr::FloatLit(FloatLit { value: b, .. }),
            ) => (*a + *b).abs() < 1e-10,
            _ => false,
        }
    }

    /// Check if two target lists are the same.
    fn same_targets(&self, t1: &[SlotRef], t2: &[SlotRef]) -> bool {
        if t1.len() != t2.len() {
            return false;
        }
        for (a, b) in t1.iter().zip(t2.iter()) {
            if a.allocator != b.allocator {
                return false;
            }
            // Compare indices
            match (&*a.index, &*b.index) {
                (Expr::IntLit(IntLit { value: v1, .. }), Expr::IntLit(IntLit { value: v2, .. })) => {
                    if v1 != v2 {
                        return false;
                    }
                }
                _ => return false,
            }
        }
        true
    }

    /// Eliminate unused bindings from a block.
    fn eliminate_unused_bindings(&mut self, statements: Vec<Stmt>) -> Vec<Stmt> {
        // Collect all used identifiers
        let mut used: BTreeSet<String> = BTreeSet::new();
        for stmt in &statements {
            self.collect_used_identifiers(stmt, &mut used);
        }

        // Filter out unused bindings (but keep those with side effects)
        let mut result = Vec::new();
        for stmt in statements {
            match &stmt {
                Stmt::Binding(binding) => {
                    if used.contains(&binding.name) || self.has_side_effects(&binding.value) {
                        result.push(stmt);
                    } else {
                        self.stats.unused_bindings_removed += 1;
                    }
                }
                _ => result.push(stmt),
            }
        }

        result
    }

    /// Collect all identifiers used in a statement.
    fn collect_used_identifiers(&self, stmt: &Stmt, used: &mut BTreeSet<String>) {
        match stmt {
            Stmt::Expr(expr_stmt) => self.collect_used_in_expr(&expr_stmt.expr, used),
            Stmt::Binding(binding) => {
                if let Some(value) = &binding.value {
                    self.collect_used_in_expr(value, used);
                }
            }
            Stmt::If(if_stmt) => {
                self.collect_used_in_expr(&if_stmt.condition, used);
                for s in &if_stmt.then_body.statements {
                    self.collect_used_identifiers(s, used);
                }
                if let Some(else_branch) = &if_stmt.else_body {
                    match else_branch {
                        ElseBranch::Else(block) => {
                            for s in &block.statements {
                                self.collect_used_identifiers(s, used);
                            }
                        }
                        ElseBranch::ElseIf(nested) => {
                            self.collect_used_identifiers(&Stmt::If(*nested.clone()), used);
                        }
                    }
                }
            }
            Stmt::For(for_stmt) => {
                match &for_stmt.range {
                    ForRange::Range { start, end } => {
                        self.collect_used_in_expr(start, used);
                        self.collect_used_in_expr(end, used);
                    }
                    ForRange::Collection(expr) => self.collect_used_in_expr(expr, used),
                }
                for s in &for_stmt.body.statements {
                    self.collect_used_identifiers(s, used);
                }
            }
            Stmt::Return(ret) => {
                if let Some(value) = &ret.value {
                    self.collect_used_in_expr(value, used);
                }
            }
            Stmt::Block(block) => {
                for s in &block.statements {
                    self.collect_used_identifiers(s, used);
                }
            }
            Stmt::Gate(gate) => {
                for target in &gate.targets {
                    used.insert(target.allocator.clone());
                }
                for param in &gate.params {
                    self.collect_used_in_expr(param, used);
                }
            }
            _ => {}
        }
    }

    /// Collect identifiers used in an expression.
    fn collect_used_in_expr(&self, expr: &Expr, used: &mut BTreeSet<String>) {
        match expr {
            Expr::Ident(ident) => {
                used.insert(ident.name.clone());
            }
            Expr::Binary(bin) => {
                self.collect_used_in_expr(&bin.left, used);
                self.collect_used_in_expr(&bin.right, used);
            }
            Expr::Unary(un) => {
                self.collect_used_in_expr(&un.operand, used);
            }
            Expr::Call(call) => {
                self.collect_used_in_expr(&call.callee, used);
                for arg in &call.args {
                    self.collect_used_in_expr(arg, used);
                }
            }
            Expr::Index(idx) => {
                self.collect_used_in_expr(&idx.object, used);
                self.collect_used_in_expr(&idx.index, used);
            }
            Expr::Field(field) => {
                self.collect_used_in_expr(&field.object, used);
            }
            Expr::Tuple(tuple) => {
                for elem in &tuple.elements {
                    self.collect_used_in_expr(elem, used);
                }
            }
            Expr::BracketArray(arr) => {
                for elem in &arr.elements {
                    self.collect_used_in_expr(elem, used);
                }
            }
            Expr::SlotRef(slot) => {
                used.insert(slot.allocator.clone());
                self.collect_used_in_expr(&slot.index, used);
            }
            _ => {}
        }
    }

    /// Check if an expression might have side effects.
    fn has_side_effects(&self, expr: &Option<Expr>) -> bool {
        match expr {
            None => false,
            Some(Expr::Call(_)) => true, // Function calls may have side effects
            Some(Expr::Gate(_)) => true, // Gate expressions have side effects
            Some(Expr::Measure(_)) => true, // Measurements have side effects
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;

    // =========================================================================
    // Constant Folding Tests
    // =========================================================================

    #[test]
    fn test_constant_folding_arithmetic() {
        let source = "x := 2 + 3;";
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let optimized = optimizer.optimize(ast);

        // The binding should have a constant value of 5
        if let TopLevelDecl::Binding(binding) = &optimized.declarations[0] {
            if let Some(Expr::IntLit(lit)) = &binding.value {
                assert_eq!(lit.value, 5);
            } else {
                panic!("Expected constant folded to IntLit");
            }
        }
        assert!(optimizer.stats.constants_folded > 0);
    }

    #[test]
    fn test_constant_folding_subtraction() {
        let source = "x := 10 - 3;";
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let optimized = optimizer.optimize(ast);

        if let TopLevelDecl::Binding(binding) = &optimized.declarations[0] {
            if let Some(Expr::IntLit(lit)) = &binding.value {
                assert_eq!(lit.value, 7);
            } else {
                panic!("Expected constant folded to IntLit");
            }
        }
    }

    #[test]
    fn test_constant_folding_multiplication() {
        let source = "x := 4 * 5;";
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let optimized = optimizer.optimize(ast);

        if let TopLevelDecl::Binding(binding) = &optimized.declarations[0] {
            if let Some(Expr::IntLit(lit)) = &binding.value {
                assert_eq!(lit.value, 20);
            } else {
                panic!("Expected constant folded to IntLit");
            }
        }
    }

    #[test]
    fn test_constant_folding_division() {
        let source = "x := 20 / 4;";
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let optimized = optimizer.optimize(ast);

        if let TopLevelDecl::Binding(binding) = &optimized.declarations[0] {
            if let Some(Expr::IntLit(lit)) = &binding.value {
                assert_eq!(lit.value, 5);
            } else {
                panic!("Expected constant folded to IntLit");
            }
        }
    }

    #[test]
    fn test_constant_folding_nested_arithmetic() {
        let source = "x := (2 + 3) * (4 - 1);";
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let optimized = optimizer.optimize(ast);

        if let TopLevelDecl::Binding(binding) = &optimized.declarations[0] {
            if let Some(Expr::IntLit(lit)) = &binding.value {
                assert_eq!(lit.value, 15); // (2+3) * (4-1) = 5 * 3 = 15
            } else {
                panic!("Expected constant folded to IntLit");
            }
        }
    }

    #[test]
    fn test_constant_folding_boolean_and() {
        // Zlup uses 'and' keyword (not &&)
        let source = "pub fn test() -> bool { x := true; return x and false; }\n";
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let optimized = optimizer.optimize(ast);

        // Verify the optimizer runs without error
        if let TopLevelDecl::Fn(fn_decl) = &optimized.declarations[0] {
            assert_eq!(fn_decl.name, "test");
        }
    }

    #[test]
    fn test_constant_folding_boolean_or() {
        // Zlup uses 'or' keyword (not ||)
        let source = "pub fn test() -> bool { x := false; return x or true; }\n";
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let optimized = optimizer.optimize(ast);

        if let TopLevelDecl::Fn(fn_decl) = &optimized.declarations[0] {
            assert_eq!(fn_decl.name, "test");
        }
    }

    #[test]
    fn test_constant_folding_comparison_eq() {
        let source = "x := 5 == 5;";
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let optimized = optimizer.optimize(ast);

        if let TopLevelDecl::Binding(binding) = &optimized.declarations[0] {
            if let Some(Expr::BoolLit(lit)) = &binding.value {
                assert!(lit.value, "5 == 5 should be true");
            } else {
                panic!("Expected constant folded to BoolLit");
            }
        }
    }

    #[test]
    fn test_constant_folding_comparison_ne() {
        let source = "x := 5 != 3;";
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let optimized = optimizer.optimize(ast);

        if let TopLevelDecl::Binding(binding) = &optimized.declarations[0] {
            if let Some(Expr::BoolLit(lit)) = &binding.value {
                assert!(lit.value, "5 != 3 should be true");
            } else {
                panic!("Expected constant folded to BoolLit");
            }
        }
    }

    #[test]
    fn test_constant_folding_comparison_lt() {
        let source = "x := 3 < 5;";
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let optimized = optimizer.optimize(ast);

        if let TopLevelDecl::Binding(binding) = &optimized.declarations[0] {
            if let Some(Expr::BoolLit(lit)) = &binding.value {
                assert!(lit.value, "3 < 5 should be true");
            } else {
                panic!("Expected constant folded to BoolLit");
            }
        }
    }

    #[test]
    fn test_constant_folding_comparison_gt() {
        let source = "x := 5 > 3;";
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let optimized = optimizer.optimize(ast);

        if let TopLevelDecl::Binding(binding) = &optimized.declarations[0] {
            if let Some(Expr::BoolLit(lit)) = &binding.value {
                assert!(lit.value, "5 > 3 should be true");
            } else {
                panic!("Expected constant folded to BoolLit");
            }
        }
    }

    // =========================================================================
    // Identity Simplification Tests
    // =========================================================================

    #[test]
    fn test_identity_simplification() {
        let source = "x := y + 0;";
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let optimized = optimizer.optimize(ast);

        if let TopLevelDecl::Binding(binding) = &optimized.declarations[0] {
            if let Some(Expr::Ident(ident)) = &binding.value {
                assert_eq!(ident.name, "y");
            } else {
                panic!("Expected simplified to just 'y'");
            }
        }
    }

    #[test]
    fn test_identity_add_zero_left() {
        let source = "x := 0 + y;";
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let optimized = optimizer.optimize(ast);

        if let TopLevelDecl::Binding(binding) = &optimized.declarations[0] {
            if let Some(Expr::Ident(ident)) = &binding.value {
                assert_eq!(ident.name, "y", "0 + y should simplify to y");
            } else {
                panic!("Expected simplified to just 'y'");
            }
        }
    }

    #[test]
    fn test_identity_subtract_zero() {
        let source = "x := y - 0;";
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let optimized = optimizer.optimize(ast);

        if let TopLevelDecl::Binding(binding) = &optimized.declarations[0] {
            if let Some(Expr::Ident(ident)) = &binding.value {
                assert_eq!(ident.name, "y", "y - 0 should simplify to y");
            } else {
                panic!("Expected simplified to just 'y'");
            }
        }
    }

    #[test]
    fn test_identity_multiply_one_right() {
        let source = "x := y * 1;";
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let optimized = optimizer.optimize(ast);

        if let TopLevelDecl::Binding(binding) = &optimized.declarations[0] {
            if let Some(Expr::Ident(ident)) = &binding.value {
                assert_eq!(ident.name, "y", "y * 1 should simplify to y");
            } else {
                panic!("Expected simplified to just 'y'");
            }
        }
    }

    #[test]
    fn test_identity_multiply_one_left() {
        let source = "x := 1 * y;";
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let optimized = optimizer.optimize(ast);

        if let TopLevelDecl::Binding(binding) = &optimized.declarations[0] {
            if let Some(Expr::Ident(ident)) = &binding.value {
                assert_eq!(ident.name, "y", "1 * y should simplify to y");
            } else {
                panic!("Expected simplified to just 'y'");
            }
        }
    }

    #[test]
    fn test_identity_divide_one() {
        let source = "x := y / 1;";
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let optimized = optimizer.optimize(ast);

        if let TopLevelDecl::Binding(binding) = &optimized.declarations[0] {
            if let Some(Expr::Ident(ident)) = &binding.value {
                assert_eq!(ident.name, "y", "y / 1 should simplify to y");
            } else {
                panic!("Expected simplified to just 'y'");
            }
        }
    }

    #[test]
    fn test_identity_multiply_zero_right() {
        let source = "x := y * 0;";
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let optimized = optimizer.optimize(ast);

        if let TopLevelDecl::Binding(binding) = &optimized.declarations[0] {
            if let Some(Expr::IntLit(lit)) = &binding.value {
                assert_eq!(lit.value, 0, "y * 0 should simplify to 0");
            } else {
                panic!("Expected simplified to 0");
            }
        }
    }

    #[test]
    fn test_identity_multiply_zero_left() {
        let source = "x := 0 * y;";
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let optimized = optimizer.optimize(ast);

        if let TopLevelDecl::Binding(binding) = &optimized.declarations[0] {
            if let Some(Expr::IntLit(lit)) = &binding.value {
                assert_eq!(lit.value, 0, "0 * y should simplify to 0");
            } else {
                panic!("Expected simplified to 0");
            }
        }
    }

    #[test]
    fn test_identity_and_true_right() {
        // Zlup uses 'and' keyword, not '&&'
        let source = "x := y and true;";
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let optimized = optimizer.optimize(ast);

        if let TopLevelDecl::Binding(binding) = &optimized.declarations[0] {
            if let Some(Expr::Ident(ident)) = &binding.value {
                assert_eq!(ident.name, "y", "y and true should simplify to y");
            } else {
                panic!("Expected simplified to just 'y'");
            }
        }
    }

    #[test]
    fn test_identity_and_true_left() {
        // Zlup uses 'and' keyword, not '&&'
        let source = "x := true and y;";
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let optimized = optimizer.optimize(ast);

        if let TopLevelDecl::Binding(binding) = &optimized.declarations[0] {
            if let Some(Expr::Ident(ident)) = &binding.value {
                assert_eq!(ident.name, "y", "true and y should simplify to y");
            } else {
                panic!("Expected simplified to just 'y'");
            }
        }
    }

    #[test]
    fn test_identity_or_false_right() {
        // Zlup uses 'or' keyword, not '||'
        let source = "x := y or false;";
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let optimized = optimizer.optimize(ast);

        if let TopLevelDecl::Binding(binding) = &optimized.declarations[0] {
            if let Some(Expr::Ident(ident)) = &binding.value {
                assert_eq!(ident.name, "y", "y or false should simplify to y");
            } else {
                panic!("Expected simplified to just 'y'");
            }
        }
    }

    #[test]
    fn test_identity_or_false_left() {
        // Zlup uses 'or' keyword, not '||'
        let source = "x := false or y;";
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let optimized = optimizer.optimize(ast);

        if let TopLevelDecl::Binding(binding) = &optimized.declarations[0] {
            if let Some(Expr::Ident(ident)) = &binding.value {
                assert_eq!(ident.name, "y", "false or y should simplify to y");
            } else {
                panic!("Expected simplified to just 'y'");
            }
        }
    }

    #[test]
    fn test_identity_and_false_short_circuit() {
        // Zlup uses 'and' keyword, not '&&'
        let source = "x := y and false;";
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let optimized = optimizer.optimize(ast);

        if let TopLevelDecl::Binding(binding) = &optimized.declarations[0] {
            if let Some(Expr::BoolLit(lit)) = &binding.value {
                assert!(!lit.value, "y and false should simplify to false");
            } else {
                panic!("Expected simplified to false");
            }
        }
    }

    #[test]
    fn test_identity_or_true_short_circuit() {
        // Zlup uses 'or' keyword, not '||'
        let source = "x := y or true;";
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let optimized = optimizer.optimize(ast);

        if let TopLevelDecl::Binding(binding) = &optimized.declarations[0] {
            if let Some(Expr::BoolLit(lit)) = &binding.value {
                assert!(lit.value, "y or true should simplify to true");
            } else {
                panic!("Expected simplified to true");
            }
        }
    }

    // =========================================================================
    // Dead Code Elimination Tests
    // =========================================================================

    #[test]
    fn test_dead_code_elimination() {
        let source = r#"
pub fn main() -> unit {
    if false {
        x := 1;
    }
    return unit;
}
"#;
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let optimized = optimizer.optimize(ast);

        // The if statement should be removed
        assert!(optimizer.stats.dead_code_removed > 0);

        // Verify the function body no longer contains an if statement
        if let TopLevelDecl::Fn(fn_decl) = &optimized.declarations[0] {
            let has_if = fn_decl.body.statements.iter().any(|s| matches!(s, Stmt::If(_)));
            assert!(!has_if, "if statement should have been eliminated");
        }
    }

    #[test]
    fn test_dead_code_if_true_keeps_then_branch() {
        let source = r#"
pub fn main() -> unit {
    if true {
        x := 42;
    } else {
        y := 0;
    }
    return unit;
}
"#;
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let optimized = optimizer.optimize(ast);

        assert!(optimizer.stats.dead_code_removed > 0);

        // The if should be replaced by just the then body block
        if let TopLevelDecl::Fn(fn_decl) = &optimized.declarations[0] {
            let has_if = fn_decl.body.statements.iter().any(|s| matches!(s, Stmt::If(_)));
            assert!(!has_if, "if true should be replaced by then block");
        }
    }

    #[test]
    fn test_dead_code_if_false_keeps_else_branch() {
        let source = r#"
pub fn main() -> unit {
    if false {
        x := 42;
    } else {
        y := 0;
    }
    return unit;
}
"#;
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let optimized = optimizer.optimize(ast);

        assert!(optimizer.stats.dead_code_removed > 0);

        // The if should be replaced by just the else body block
        if let TopLevelDecl::Fn(fn_decl) = &optimized.declarations[0] {
            let has_if = fn_decl.body.statements.iter().any(|s| matches!(s, Stmt::If(_)));
            assert!(!has_if, "if false should be replaced by else block");
        }
    }

    // =========================================================================
    // Double Negation Tests
    // =========================================================================

    #[test]
    fn test_double_negation() {
        let source = "x := --5;";
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let optimized = optimizer.optimize(ast);

        if let TopLevelDecl::Binding(binding) = &optimized.declarations[0] {
            if let Some(Expr::IntLit(lit)) = &binding.value {
                assert_eq!(lit.value, 5);
            } else {
                panic!("Expected double negation folded");
            }
        }
    }

    #[test]
    fn test_double_not() {
        let source = "x := !!true;";
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let optimized = optimizer.optimize(ast);

        if let TopLevelDecl::Binding(binding) = &optimized.declarations[0] {
            if let Some(Expr::BoolLit(lit)) = &binding.value {
                assert!(lit.value, "!!true should be true");
            } else {
                panic!("Expected double not folded to BoolLit");
            }
        }
    }

    // =========================================================================
    // Gate Cancellation Tests
    // =========================================================================

    #[test]
    fn test_gate_cancellation_h_h() {
        let source = r#"
pub fn main() -> unit {
    q := qalloc(2);
    pz q;
    h q[0];
    h q[0];
    return unit;
}
"#;
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let optimized = optimizer.optimize(ast);

        assert_eq!(optimizer.stats.gates_cancelled, 2, "H H should cancel");

        // Verify no H gates remain
        if let TopLevelDecl::Fn(fn_decl) = &optimized.declarations[0] {
            let h_count = fn_decl.body.statements.iter().filter(|s| {
                matches!(s, Stmt::Gate(g) if g.kind == GateKind::H)
            }).count();
            assert_eq!(h_count, 0, "Both H gates should be cancelled");
        }
    }

    #[test]
    fn test_gate_cancellation_x_x() {
        let source = r#"
pub fn main() -> unit {
    q := qalloc(2);
    pz q;
    x q[0];
    x q[0];
    return unit;
}
"#;
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let _optimized = optimizer.optimize(ast);

        assert_eq!(optimizer.stats.gates_cancelled, 2, "X X should cancel");
    }

    #[test]
    fn test_gate_cancellation_y_y() {
        let source = r#"
pub fn main() -> unit {
    q := qalloc(2);
    pz q;
    y q[0];
    y q[0];
    return unit;
}
"#;
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let _optimized = optimizer.optimize(ast);

        assert_eq!(optimizer.stats.gates_cancelled, 2, "Y Y should cancel");
    }

    #[test]
    fn test_gate_cancellation_z_z() {
        let source = r#"
pub fn main() -> unit {
    q := qalloc(2);
    pz q;
    z q[0];
    z q[0];
    return unit;
}
"#;
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let _optimized = optimizer.optimize(ast);

        assert_eq!(optimizer.stats.gates_cancelled, 2, "Z Z should cancel");
    }

    #[test]
    fn test_gate_cancellation_sz_szdg() {
        let source = r#"
pub fn main() -> unit {
    q := qalloc(2);
    pz q;
    sz q[0];
    szdg q[0];
    return unit;
}
"#;
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let _optimized = optimizer.optimize(ast);

        assert_eq!(optimizer.stats.gates_cancelled, 2, "SZ SZdg should cancel");
    }

    #[test]
    fn test_gate_cancellation_t_tdg() {
        let source = r#"
pub fn main() -> unit {
    q := qalloc(2);
    pz q;
    t q[0];
    tdg q[0];
    return unit;
}
"#;
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let _optimized = optimizer.optimize(ast);

        assert_eq!(optimizer.stats.gates_cancelled, 2, "T Tdg should cancel");
    }

    #[test]
    fn test_gate_cancellation_cx_cx() {
        let source = r#"
pub fn main() -> unit {
    q := qalloc(2);
    pz q;
    cx (q[0], q[1]);
    cx (q[0], q[1]);
    return unit;
}
"#;
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let _optimized = optimizer.optimize(ast);

        assert_eq!(optimizer.stats.gates_cancelled, 2, "CX CX should cancel");
    }

    #[test]
    fn test_gate_cancellation_cz_cz() {
        let source = r#"
pub fn main() -> unit {
    q := qalloc(2);
    pz q;
    cz (q[0], q[1]);
    cz (q[0], q[1]);
    return unit;
}
"#;
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let _optimized = optimizer.optimize(ast);

        assert_eq!(optimizer.stats.gates_cancelled, 2, "CZ CZ should cancel");
    }

    #[test]
    fn test_gate_cancellation_swap_swap() {
        let source = r#"
pub fn main() -> unit {
    q := qalloc(2);
    pz q;
    swap (q[0], q[1]);
    swap (q[0], q[1]);
    return unit;
}
"#;
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let _optimized = optimizer.optimize(ast);

        assert_eq!(optimizer.stats.gates_cancelled, 2, "SWAP SWAP should cancel");
    }

    #[test]
    fn test_gate_no_cancel_different_qubits() {
        let source = r#"
pub fn main() -> unit {
    q := qalloc(2);
    pz q;
    h q[0];
    h q[1];
    return unit;
}
"#;
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let optimized = optimizer.optimize(ast);

        // Should NOT cancel - different qubits
        assert_eq!(optimizer.stats.gates_cancelled, 0, "H on different qubits should not cancel");

        // Verify both H gates remain
        // Note: Gates are represented as Stmt::Expr containing Expr::Gate
        if let TopLevelDecl::Fn(fn_decl) = &optimized.declarations[0] {
            let h_count = fn_decl.body.statements.iter().filter(|s| {
                if let Stmt::Expr(expr_stmt) = s {
                    if let Expr::Gate(gate) = &expr_stmt.expr {
                        return gate.kind == GateKind::H;
                    }
                }
                false
            }).count();
            assert_eq!(h_count, 2, "Both H gates on different qubits should remain");
        }
    }

    #[test]
    fn test_gate_no_cancel_different_types() {
        let source = r#"
pub fn main() -> unit {
    q := qalloc(2);
    pz q;
    h q[0];
    x q[0];
    return unit;
}
"#;
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let _optimized = optimizer.optimize(ast);

        // Should NOT cancel - different gate types
        assert_eq!(optimizer.stats.gates_cancelled, 0, "H X should not cancel");
    }

    // =========================================================================
    // Optimization Barrier Tests
    // =========================================================================

    #[test]
    fn test_has_preserve_attr() {
        use crate::ast::Attribute;

        let optimizer = Optimizer::new();

        // Test @preserve
        let attrs = vec![Attribute::flag("preserve")];
        assert!(optimizer.has_preserve_attr(&attrs));

        // Test @timing
        let attrs = vec![Attribute::flag("timing")];
        assert!(optimizer.has_preserve_attr(&attrs));

        // Test @identity
        let attrs = vec![Attribute::flag("identity")];
        assert!(optimizer.has_preserve_attr(&attrs));

        // Test @noopt
        let attrs = vec![Attribute::flag("noopt")];
        assert!(optimizer.has_preserve_attr(&attrs));

        // Test non-preserve attr
        let attrs = vec![Attribute::flag("other")];
        assert!(!optimizer.has_preserve_attr(&attrs));

        // Test empty attrs
        let attrs: Vec<Attribute> = vec![];
        assert!(!optimizer.has_preserve_attr(&attrs));
    }

    #[test]
    fn test_no_optimize_prevents_gate_cancellation() {
        // @no_optimize(expr) builtin prevents cancellation
        let source = r#"
pub fn main() -> unit {
    q := qalloc(2);
    pz q;
    @no_optimize(h q[0]);
    h q[0];
    return unit;
}
"#;
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let _optimized = optimizer.optimize(ast);

        // The first H is wrapped in @no_optimize, so cancellation shouldn't happen
        assert_eq!(optimizer.stats.gates_cancelled, 0, "@no_optimize should prevent cancellation");
    }

    #[test]
    fn test_no_optimize_on_second_gate_prevents_cancellation() {
        // @no_optimize on the second gate also prevents cancellation
        let source = r#"
pub fn main() -> unit {
    q := qalloc(2);
    pz q;
    h q[0];
    @no_optimize(h q[0]);
    return unit;
}
"#;
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let _optimized = optimizer.optimize(ast);

        assert_eq!(optimizer.stats.gates_cancelled, 0, "@no_optimize on second gate should prevent cancellation");
    }

    #[test]
    fn test_no_optimize_both_gates_no_cancellation() {
        // Both gates wrapped in @no_optimize
        let source = r#"
pub fn main() -> unit {
    q := qalloc(2);
    pz q;
    @no_optimize(h q[0]);
    @no_optimize(h q[0]);
    return unit;
}
"#;
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let _optimized = optimizer.optimize(ast);

        assert_eq!(optimizer.stats.gates_cancelled, 0, "Both @no_optimize should prevent cancellation");
    }

    #[test]
    fn test_tick_blocks_are_barriers() {
        let source = r#"
pub fn main() -> unit {
    q := qalloc(2);
    pz q;
    h q[0];
    tick {
        x q[1];
    }
    h q[0];
    return unit;
}
"#;
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let _optimized = optimizer.optimize(ast);

        // tick {} is a barrier, so the two H gates shouldn't cancel
        assert_eq!(optimizer.stats.gates_cancelled, 0, "tick should act as barrier");
    }

    #[test]
    fn test_gates_cancel_within_same_tick() {
        let source = r#"
pub fn main() -> unit {
    q := qalloc(2);
    pz q;
    tick {
        h q[0];
        h q[0];
    }
    return unit;
}
"#;
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let _optimized = optimizer.optimize(ast);

        // Within the same tick, gates CAN cancel
        assert_eq!(optimizer.stats.gates_cancelled, 2, "H H within same tick should cancel");
    }

    // =========================================================================
    // @round(n) Attribute Tests
    // =========================================================================

    #[test]
    fn test_get_round_attr() {
        use crate::ast::{Attribute, AttributeValue, Block};

        let optimizer = Optimizer::new();

        // Create a block with @round(0)
        let block = Block {
            label: None,
            attrs: vec![Attribute::with_value("round", AttributeValue::Int(0))],
            statements: vec![],
            trailing_expr: None,
            location: None,
        };

        let stmt = Stmt::Block(block);
        assert_eq!(optimizer.get_round_attr(&stmt), Some(0));

        // Create a block with @round(5)
        let block = Block {
            label: None,
            attrs: vec![Attribute::with_value("round", AttributeValue::Int(5))],
            statements: vec![],
            trailing_expr: None,
            location: None,
        };

        let stmt = Stmt::Block(block);
        assert_eq!(optimizer.get_round_attr(&stmt), Some(5));

        // Block without round attr
        let block = Block {
            label: None,
            attrs: vec![],
            statements: vec![],
            trailing_expr: None,
            location: None,
        };

        let stmt = Stmt::Block(block);
        assert_eq!(optimizer.get_round_attr(&stmt), None);
    }

    // =========================================================================
    // Unused Binding Elimination Tests
    // =========================================================================

    #[test]
    fn test_unused_binding_eliminated() {
        let source = r#"
pub fn main() -> unit {
    unused := 42;
    return unit;
}
"#;
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let optimized = optimizer.optimize(ast);

        assert!(optimizer.stats.unused_bindings_removed > 0, "unused binding should be removed");

        // Verify the binding was removed
        if let TopLevelDecl::Fn(fn_decl) = &optimized.declarations[0] {
            let has_binding = fn_decl.body.statements.iter().any(|s| {
                matches!(s, Stmt::Binding(b) if b.name == "unused")
            });
            assert!(!has_binding, "unused binding should be eliminated");
        }
    }

    #[test]
    fn test_used_binding_kept() {
        let source = r#"
pub fn main() -> i32 {
    used := 42;
    return used;
}
"#;
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let optimized = optimizer.optimize(ast);

        // Verify the binding was NOT removed
        if let TopLevelDecl::Fn(fn_decl) = &optimized.declarations[0] {
            let has_binding = fn_decl.body.statements.iter().any(|s| {
                matches!(s, Stmt::Binding(b) if b.name == "used")
            });
            assert!(has_binding, "used binding should be kept");
        }
    }

    // =========================================================================
    // Zero-Angle Identity Removal Tests
    // =========================================================================

    #[test]
    fn test_zero_rotation_removed() {
        let source = r#"
pub fn main() -> unit {
    q := qalloc(2);
    pz q;
    rz(0) q[0];
    return unit;
}
"#;
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let optimized = optimizer.optimize(ast);

        assert!(optimizer.stats.identities_removed > 0, "rz(0) should be removed as identity");

        // Verify the gate was removed
        // Note: Gates are represented as Stmt::Expr containing Expr::Gate
        if let TopLevelDecl::Fn(fn_decl) = &optimized.declarations[0] {
            let has_rz = fn_decl.body.statements.iter().any(|s| {
                if let Stmt::Expr(expr_stmt) = s {
                    if let Expr::Gate(gate) = &expr_stmt.expr {
                        return gate.kind == GateKind::RZ;
                    }
                }
                false
            });
            assert!(!has_rz, "rz(0) identity gate should be removed");
        }
    }

    // =========================================================================
    // Optimizer Configuration Tests
    // =========================================================================

    #[test]
    fn test_disable_constant_folding() {
        let source = "x := 2 + 3;";
        let ast = parse(source).unwrap();
        let mut config = OptimizeConfig::all();
        config.constant_folding = false;
        let mut optimizer = Optimizer::with_config(config);
        let optimized = optimizer.optimize(ast);

        // Constant folding disabled - should still have binary expression
        if let TopLevelDecl::Binding(binding) = &optimized.declarations[0] {
            assert!(matches!(&binding.value, Some(Expr::Binary(_))),
                "with constant_folding disabled, should keep binary expr");
        }
        assert_eq!(optimizer.stats.constants_folded, 0);
    }

    #[test]
    fn test_disable_gate_cancellation() {
        let source = r#"
pub fn main() -> unit {
    q := qalloc(2);
    pz q;
    h q[0];
    h q[0];
    return unit;
}
"#;
        let ast = parse(source).unwrap();
        let mut config = OptimizeConfig::all();
        config.gate_cancellation = false;
        let mut optimizer = Optimizer::with_config(config);
        let optimized = optimizer.optimize(ast);

        assert_eq!(optimizer.stats.gates_cancelled, 0, "gate_cancellation disabled");

        // Both H gates should remain
        // Note: Gates are represented as Stmt::Expr containing Expr::Gate
        if let TopLevelDecl::Fn(fn_decl) = &optimized.declarations[0] {
            let h_count = fn_decl.body.statements.iter().filter(|s| {
                if let Stmt::Expr(expr_stmt) = s {
                    if let Expr::Gate(gate) = &expr_stmt.expr {
                        return gate.kind == GateKind::H;
                    }
                }
                false
            }).count();
            assert_eq!(h_count, 2, "Both H gates should remain when cancellation disabled");
        }
    }

    #[test]
    fn test_disable_all_optimizations() {
        let source = r#"
pub fn main() -> unit {
    q := qalloc(2);
    pz q;
    unused := 2 + 3;
    h q[0];
    h q[0];
    if false {
        x q[0];
    }
    return unit;
}
"#;
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::with_config(OptimizeConfig::none());
        let _optimized = optimizer.optimize(ast);

        assert_eq!(optimizer.stats.constants_folded, 0);
        assert_eq!(optimizer.stats.dead_code_removed, 0);
        assert_eq!(optimizer.stats.unused_bindings_removed, 0);
        assert_eq!(optimizer.stats.gates_cancelled, 0);
        assert_eq!(optimizer.stats.identities_removed, 0);
        assert_eq!(optimizer.stats.iterations, 0);
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[test]
    fn test_multiple_gate_pairs_cancel() {
        let source = r#"
pub fn main() -> unit {
    q := qalloc(2);
    pz q;
    h q[0];
    h q[0];
    x q[0];
    x q[0];
    z q[0];
    z q[0];
    return unit;
}
"#;
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let _optimized = optimizer.optimize(ast);

        assert_eq!(optimizer.stats.gates_cancelled, 6, "All 3 pairs should cancel");
    }

    #[test]
    fn test_interleaved_gates_no_cancel() {
        let source = r#"
pub fn main() -> unit {
    q := qalloc(2);
    pz q;
    h q[0];
    x q[0];
    h q[0];
    return unit;
}
"#;
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let _optimized = optimizer.optimize(ast);

        // H and H are not adjacent, so should not cancel
        assert_eq!(optimizer.stats.gates_cancelled, 0, "Non-adjacent gates should not cancel");
    }

    #[test]
    fn test_optimizer_stats() {
        let source = r#"
pub fn main() -> unit {
    q := qalloc(2);
    pz q;
    x := 2 + 3;
    h q[0];
    h q[0];
    return unit;
}
"#;
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let _optimized = optimizer.optimize(ast);

        let stats = optimizer.stats();
        assert!(stats.constants_folded > 0);
        assert!(stats.gates_cancelled >= 2);
        assert!(stats.iterations >= 1);
    }

    // =========================================================================
    // Inline For Loop Unrolling Tests
    // =========================================================================

    #[test]
    fn test_inline_for_unrolling_basic() {
        // inline for i in 0..4 { h q[i]; } should produce 4 h gates
        let source = r#"
pub fn main() -> unit {
    q := qalloc(4);
    pz q;
    inline for i in 0..4 {
        h q[i];
    }
    return unit;
}
"#;
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let optimized = optimizer.optimize(ast);

        // Verify unrolling happened
        assert!(
            optimizer.stats.inline_for_unrolled > 0,
            "Expected inline for to be unrolled, stats: {:?}",
            optimizer.stats
        );
        assert_eq!(
            optimizer.stats.inline_for_statements_generated, 4,
            "Expected 4 statements generated from inline for"
        );

        // Verify the function body has 4 h gates
        // Note: Gates can be represented as either Stmt::Gate or Stmt::Expr(Expr::Gate)
        fn count_gates_recursive(stmts: &[Stmt]) -> usize {
            let mut count = 0;
            for stmt in stmts {
                match stmt {
                    Stmt::Gate(_) => count += 1,
                    Stmt::Expr(e) => {
                        if matches!(e.expr, Expr::Gate(_)) {
                            count += 1;
                        }
                    }
                    Stmt::Block(b) => count += count_gates_recursive(&b.statements),
                    _ => {}
                }
            }
            count
        }
        if let TopLevelDecl::Fn(fn_decl) = &optimized.declarations[0] {
            let gate_count = count_gates_recursive(&fn_decl.body.statements);
            // Should have pz + 4 h gates
            // At minimum, we should have 4 h gates from unrolling (pz may or may not count)
            assert!(gate_count >= 4, "Expected at least 4 gates, got {}", gate_count);
        }
    }

    #[test]
    fn test_inline_for_nested() {
        // Nested 2x3 inline for should produce 6 statements
        let source = r#"
pub fn main() -> unit {
    q := qalloc(6);
    pz q;
    inline for i in 0..2 {
        inline for j in 0..3 {
            h q[i * 3 + j];
        }
    }
    return unit;
}
"#;
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let _optimized = optimizer.optimize(ast);

        // Should unroll both loops: outer produces 2 iterations,
        // each containing an inner loop that produces 3 statements
        assert!(
            optimizer.stats.inline_for_unrolled >= 2,
            "Expected at least 2 inline for loops unrolled"
        );
        // 2 outer iterations * 3 inner statements = 6 statements
        // Plus the 2 inner loops themselves produce statements
        assert!(
            optimizer.stats.inline_for_statements_generated >= 6,
            "Expected at least 6 statements generated, got {}",
            optimizer.stats.inline_for_statements_generated
        );
    }

    #[test]
    fn test_inline_for_with_comptime_bounds() {
        // Bounds using comptime expression (2 * 2)
        let source = r#"
pub fn main() -> unit {
    q := qalloc(4);
    pz q;
    inline for i in 0..(2 * 2) {
        h q[i];
    }
    return unit;
}
"#;
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let _optimized = optimizer.optimize(ast);

        assert!(
            optimizer.stats.inline_for_unrolled > 0,
            "Expected inline for with comptime bounds to be unrolled"
        );
        assert_eq!(
            optimizer.stats.inline_for_statements_generated, 4,
            "Expected 4 statements from 2*2=4"
        );
    }

    #[test]
    fn test_inline_for_empty_range() {
        // 0..0 should produce 0 statements
        let source = r#"
pub fn main() -> unit {
    q := qalloc(4);
    pz q;
    inline for i in 0..0 {
        h q[i];
    }
    return unit;
}
"#;
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let _optimized = optimizer.optimize(ast);

        // Empty range still counts as unrolled (just produces 0 statements)
        assert_eq!(
            optimizer.stats.inline_for_statements_generated, 0,
            "Expected 0 statements from empty range"
        );
    }

    #[test]
    fn test_regular_for_not_unrolled() {
        // Regular for loop (without inline) should not be unrolled
        let source = r#"
pub fn main() -> unit {
    q := qalloc(4);
    pz q;
    for i in 0..4 {
        h q[i];
    }
    return unit;
}
"#;
        let ast = parse(source).unwrap();
        let mut optimizer = Optimizer::new();
        let _optimized = optimizer.optimize(ast);

        assert_eq!(
            optimizer.stats.inline_for_unrolled, 0,
            "Regular for loop should not be unrolled"
        );
    }
}
