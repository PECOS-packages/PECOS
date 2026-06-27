//! Linter for Zlup code quality and NASA Power of 10 compliance.
//!
//! Provides static analysis checks beyond semantic correctness:
//! - NASA Power of 10 rules
//! - Code style and naming conventions
//! - Complexity metrics
//! - Best practices for quantum code
//!
//! ## Usage
//!
//! ```rust
//! use zlup::linter::{Linter, LintConfig};
//!
//! let source = "fn main() -> unit { return unit; }";
//! let program = zlup::parse(source).expect("parse failed");
//! let linter = Linter::new(LintConfig::strict());
//! let diagnostics = linter.lint(&program);
//! // diagnostics contains any lint warnings/errors found
//! ```

use crate::ast::{self, Expr, Program, Stmt, TopLevelDecl};
use std::collections::BTreeMap;

// =============================================================================
// Lint Configuration
// =============================================================================

/// Lint severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Informational suggestion.
    Hint,
    /// Style warning (doesn't affect correctness).
    Warning,
    /// Should be fixed (violates best practices).
    Error,
    /// Must be fixed (violates safety rules).
    Deny,
}

/// Safety level for an auto-fix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixSafety {
    /// Safe fix - can be applied automatically without risk.
    /// These fixes preserve semantics and are guaranteed correct.
    Safe,
    /// Unsafe fix - probably safe but not guaranteed.
    /// These may change behavior in edge cases or require manual verification.
    Unsafe,
}

/// A potential fix for a lint diagnostic.
#[derive(Debug, Clone)]
pub struct LintFix {
    /// Start byte offset in source.
    pub start: usize,
    /// End byte offset in source.
    pub end: usize,
    /// Replacement text.
    pub replacement: String,
    /// Safety level.
    pub safety: FixSafety,
}

/// Configuration for individual lint rules.
#[derive(Debug, Clone)]
pub struct LintRule {
    /// Whether this rule is enabled.
    pub enabled: bool,
    /// Severity level for violations.
    pub severity: Severity,
}

impl LintRule {
    pub fn enabled(severity: Severity) -> Self {
        Self {
            enabled: true,
            severity,
        }
    }

    pub fn disabled() -> Self {
        Self {
            enabled: false,
            severity: Severity::Warning,
        }
    }
}

/// Linter configuration.
#[derive(Debug, Clone)]
pub struct LintConfig {
    // NASA Power of 10 Rules
    /// Maximum function body lines (NASA PoT Rule 4: ≤60 lines).
    pub max_function_lines: usize,
    /// Rule for function size violations.
    pub function_too_long: LintRule,

    /// Minimum assertions per function on average (NASA PoT Rule 5).
    pub min_assertions_per_function: f64,
    /// Rule for low assertion density.
    pub low_assertion_density: LintRule,

    /// Maximum nesting depth (related to NASA PoT Rule 1: simple control flow).
    pub max_nesting_depth: usize,
    /// Rule for deep nesting.
    pub deep_nesting: LintRule,

    // Naming Conventions
    /// Rule for function naming (snake_case).
    pub function_naming: LintRule,
    /// Rule for variable naming (snake_case).
    pub variable_naming: LintRule,
    /// Rule for constant naming (SCREAMING_SNAKE_CASE).
    pub constant_naming: LintRule,
    /// Rule for type naming (PascalCase).
    pub type_naming: LintRule,

    // Code Quality
    /// Rule for unused variables.
    pub unused_variable: LintRule,
    /// Rule for unused functions.
    pub unused_function: LintRule,
    /// Rule for missing documentation on public items.
    pub missing_docs: LintRule,
    /// Rule for TODO/FIXME comments.
    pub todo_comments: LintRule,

    // Quantum-Specific
    /// Rule for measurement without using result.
    pub unused_measurement: LintRule,
    /// Rule for potentially inefficient gate sequences.
    pub redundant_gates: LintRule,
    /// Rule for missing barrier/tick between non-commuting gates.
    pub missing_barrier: LintRule,

    // Angle Precision
    /// Rule for using radians when exact turn fractions exist.
    pub prefer_turns_over_radians: LintRule,
    /// Rule for using decimal turns when exact fractions exist.
    pub prefer_fraction_turns: LintRule,
    /// Rule for using float literals where exact fractions/constants exist.
    pub prefer_exact_angles: LintRule,
}

impl Default for LintConfig {
    fn default() -> Self {
        // Default is strict - this is a safety-critical language
        Self::strict()
    }
}

impl LintConfig {
    /// Relaxed configuration (warnings instead of errors).
    pub fn relaxed() -> Self {
        Self {
            // NASA Power of 10
            max_function_lines: 60,
            function_too_long: LintRule::disabled(),
            min_assertions_per_function: 2.0,
            low_assertion_density: LintRule::enabled(Severity::Hint),
            max_nesting_depth: 4,
            deep_nesting: LintRule::enabled(Severity::Warning),

            // Naming
            function_naming: LintRule::enabled(Severity::Warning),
            variable_naming: LintRule::enabled(Severity::Hint),
            constant_naming: LintRule::enabled(Severity::Hint),
            type_naming: LintRule::enabled(Severity::Warning),

            // Code Quality
            unused_variable: LintRule::enabled(Severity::Warning),
            unused_function: LintRule::enabled(Severity::Hint),
            missing_docs: LintRule::disabled(),
            todo_comments: LintRule::enabled(Severity::Hint),

            // Quantum
            unused_measurement: LintRule::enabled(Severity::Warning),
            redundant_gates: LintRule::enabled(Severity::Hint),
            missing_barrier: LintRule::disabled(),

            // Angle Precision
            prefer_turns_over_radians: LintRule::enabled(Severity::Hint),
            prefer_fraction_turns: LintRule::enabled(Severity::Hint),
            prefer_exact_angles: LintRule::enabled(Severity::Hint),
        }
    }

    /// Strict configuration for safety-critical code (DEFAULT).
    pub fn strict() -> Self {
        Self {
            // NASA Power of 10 - enforced (except line count which is arbitrary)
            max_function_lines: 60,
            function_too_long: LintRule::disabled(),
            min_assertions_per_function: 2.0,
            low_assertion_density: LintRule::enabled(Severity::Warning),
            max_nesting_depth: 4,
            deep_nesting: LintRule::enabled(Severity::Error),

            // Naming - enforced
            function_naming: LintRule::enabled(Severity::Error),
            variable_naming: LintRule::enabled(Severity::Warning),
            constant_naming: LintRule::enabled(Severity::Warning),
            type_naming: LintRule::enabled(Severity::Error),

            // Code Quality - enforced
            unused_variable: LintRule::enabled(Severity::Error),
            unused_function: LintRule::enabled(Severity::Warning),
            missing_docs: LintRule::enabled(Severity::Warning),
            todo_comments: LintRule::enabled(Severity::Warning),

            // Quantum - enforced
            unused_measurement: LintRule::enabled(Severity::Error),
            redundant_gates: LintRule::enabled(Severity::Warning),
            missing_barrier: LintRule::enabled(Severity::Hint),

            // Angle Precision - enforced
            prefer_turns_over_radians: LintRule::enabled(Severity::Warning),
            prefer_fraction_turns: LintRule::enabled(Severity::Warning),
            prefer_exact_angles: LintRule::enabled(Severity::Warning),
        }
    }

    /// Minimal configuration (only critical issues).
    pub fn minimal() -> Self {
        Self {
            function_too_long: LintRule::disabled(),
            low_assertion_density: LintRule::disabled(),
            deep_nesting: LintRule::disabled(),
            function_naming: LintRule::disabled(),
            variable_naming: LintRule::disabled(),
            constant_naming: LintRule::disabled(),
            type_naming: LintRule::disabled(),
            unused_variable: LintRule::enabled(Severity::Warning),
            unused_function: LintRule::disabled(),
            missing_docs: LintRule::disabled(),
            todo_comments: LintRule::disabled(),
            unused_measurement: LintRule::enabled(Severity::Warning),
            redundant_gates: LintRule::disabled(),
            missing_barrier: LintRule::disabled(),
            prefer_turns_over_radians: LintRule::disabled(),
            prefer_fraction_turns: LintRule::disabled(),
            prefer_exact_angles: LintRule::disabled(),
            ..Default::default()
        }
    }
}

// =============================================================================
// Lint Diagnostics
// =============================================================================

/// A lint diagnostic.
#[derive(Debug, Clone)]
pub struct LintDiagnostic {
    /// Lint rule that was violated.
    pub rule: &'static str,
    /// Human-readable message.
    pub message: String,
    /// Severity level.
    pub severity: Severity,
    /// Source location (if available).
    pub location: Option<ast::SourceLocation>,
    /// Suggested fix (if available).
    pub suggestion: Option<String>,
    /// Auto-fix (if available).
    pub fix: Option<LintFix>,
}

impl LintDiagnostic {
    pub fn new(rule: &'static str, message: impl Into<String>, severity: Severity) -> Self {
        Self {
            rule,
            message: message.into(),
            severity,
            location: None,
            suggestion: None,
            fix: None,
        }
    }

    pub fn with_location(mut self, location: ast::SourceLocation) -> Self {
        self.location = Some(location);
        self
    }

    pub fn with_location_opt(mut self, location: Option<ast::SourceLocation>) -> Self {
        self.location = location;
        self
    }

    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }

    pub fn with_fix(mut self, fix: LintFix) -> Self {
        self.fix = Some(fix);
        self
    }

    /// Check if this diagnostic has a safe fix available.
    pub fn has_safe_fix(&self) -> bool {
        matches!(&self.fix, Some(f) if f.safety == FixSafety::Safe)
    }

    /// Check if this diagnostic has any fix available.
    pub fn has_fix(&self) -> bool {
        self.fix.is_some()
    }
}

impl std::fmt::Display for LintDiagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let severity_str = match self.severity {
            Severity::Hint => "hint",
            Severity::Warning => "warning",
            Severity::Error => "error",
            Severity::Deny => "deny",
        };

        if let Some(ref loc) = self.location {
            write!(
                f,
                "{}: {} [{}] ({}:{})",
                severity_str, self.message, self.rule, loc.line, loc.column
            )?;
        } else {
            write!(f, "{}: {} [{}]", severity_str, self.message, self.rule)?;
        }

        if let Some(ref suggestion) = self.suggestion {
            write!(f, "\n  suggestion: {}", suggestion)?;
        }

        Ok(())
    }
}

// =============================================================================
// Linter
// =============================================================================

/// The linter engine.
pub struct Linter {
    config: LintConfig,
    diagnostics: Vec<LintDiagnostic>,
    /// Track variable usage for unused detection.
    variable_uses: BTreeMap<String, usize>,
    /// Track function usage for unused detection.
    function_uses: BTreeMap<String, usize>,
    /// Track defined variables.
    defined_variables: BTreeMap<String, ast::SourceLocation>,
    /// Track defined functions.
    defined_functions: BTreeMap<String, (ast::SourceLocation, bool)>, // (location, is_pub)
    /// Current nesting depth.
    current_depth: usize,
    /// Assertion count for current function.
    assertion_count: usize,
    /// Total functions analyzed.
    function_count: usize,
    /// Total assertions across all functions.
    total_assertions: usize,
    /// Source code for computing fix offsets.
    source: Option<String>,
}

impl Linter {
    pub fn new(config: LintConfig) -> Self {
        Self {
            config,
            diagnostics: Vec::new(),
            variable_uses: BTreeMap::new(),
            function_uses: BTreeMap::new(),
            defined_variables: BTreeMap::new(),
            defined_functions: BTreeMap::new(),
            current_depth: 0,
            assertion_count: 0,
            function_count: 0,
            total_assertions: 0,
            source: None,
        }
    }

    /// Set the source code for computing fix offsets.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Lint a program and return diagnostics.
    pub fn lint(mut self, program: &Program) -> Vec<LintDiagnostic> {
        // First pass: collect definitions
        for decl in &program.declarations {
            self.collect_definitions(decl);
        }

        // Second pass: analyze usage
        for decl in &program.declarations {
            self.analyze_decl(decl);
        }

        // Check for unused definitions
        self.check_unused();

        // Check assertion density
        self.check_assertion_density();

        self.diagnostics
    }

    fn collect_definitions(&mut self, decl: &TopLevelDecl) {
        match decl {
            TopLevelDecl::Fn(fn_decl) => {
                let location = fn_decl.location.clone().unwrap_or_default();
                self.defined_functions
                    .insert(fn_decl.name.clone(), (location, fn_decl.is_pub));
            }
            TopLevelDecl::Binding(binding) => {
                if let Some(ref loc) = binding.location {
                    self.defined_variables
                        .insert(binding.name.clone(), loc.clone());
                }
            }
            TopLevelDecl::Struct(struct_decl) => {
                self.check_type_name(&struct_decl.name, struct_decl.location.as_ref());
            }
            TopLevelDecl::Enum(enum_decl) => {
                self.check_type_name(&enum_decl.name, enum_decl.location.as_ref());
            }
            TopLevelDecl::Union(union_decl) => {
                self.check_type_name(&union_decl.name, union_decl.location.as_ref());
            }
            _ => {}
        }
    }

    fn analyze_decl(&mut self, decl: &TopLevelDecl) {
        if let TopLevelDecl::Fn(fn_decl) = decl {
            self.analyze_function(fn_decl);
        }
    }

    fn analyze_function(&mut self, fn_decl: &ast::FnDecl) {
        self.function_count += 1;
        self.assertion_count = 0;

        // Check function naming
        self.check_function_name(&fn_decl.name, fn_decl.location.as_ref());

        // Check function length
        self.check_function_length(fn_decl);

        // Check parameters
        for param in &fn_decl.params {
            self.check_variable_name(&param.name, param.location.as_ref());
            if let Some(ref loc) = param.location {
                self.defined_variables
                    .insert(param.name.clone(), loc.clone());
            }
        }

        // Analyze body
        self.current_depth = 0;
        self.analyze_block(&fn_decl.body);

        // Record assertions for this function
        self.total_assertions += self.assertion_count;
    }

    fn analyze_block(&mut self, block: &ast::Block) {
        self.current_depth += 1;
        self.check_nesting_depth(block.location.as_ref());

        for stmt in &block.statements {
            self.analyze_stmt(stmt);
        }

        if let Some(ref trailing) = block.trailing_expr {
            self.analyze_expr(trailing);
        }

        self.current_depth -= 1;
    }

    fn analyze_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Binding(binding) => {
                // Check variable naming
                self.check_variable_name(&binding.name, binding.location.as_ref());
                if let Some(ref loc) = binding.location {
                    self.defined_variables
                        .insert(binding.name.clone(), loc.clone());
                }
                if let Some(ref value) = binding.value {
                    self.analyze_expr(value);
                }
            }
            Stmt::Expr(expr_stmt) => {
                self.analyze_expr(&expr_stmt.expr);
            }
            Stmt::If(if_stmt) => {
                self.analyze_expr(&if_stmt.condition);
                self.analyze_block(&if_stmt.then_body);
                if let Some(ref else_branch) = if_stmt.else_body {
                    self.analyze_else_branch(else_branch);
                }
            }
            Stmt::For(for_stmt) => {
                // Check capture variable naming
                for capture in &for_stmt.captures {
                    self.check_variable_name(capture, for_stmt.location.as_ref());
                }
                self.analyze_for_range(&for_stmt.range);
                self.analyze_block(&for_stmt.body);
            }
            Stmt::Block(block) => {
                self.analyze_block(block);
            }
            Stmt::Return(ret_stmt) => {
                if let Some(ref value) = ret_stmt.value {
                    self.analyze_expr(value);
                }
            }
            Stmt::Tick(tick_stmt) => {
                for inner in &tick_stmt.body {
                    self.analyze_stmt(inner);
                }
            }
            Stmt::Assign(assign) => {
                self.analyze_expr(&assign.target);
                self.analyze_expr(&assign.value);
            }
            Stmt::Gate(gate_op) => {
                // Track allocator usage
                for target in &gate_op.targets {
                    *self
                        .variable_uses
                        .entry(target.allocator.clone())
                        .or_insert(0) += 1;
                    self.analyze_expr(&target.index);
                }
            }
            Stmt::Measure(measure_op) => {
                // Track allocator usage
                for target in &measure_op.targets {
                    *self
                        .variable_uses
                        .entry(target.allocator.clone())
                        .or_insert(0) += 1;
                    self.analyze_expr(&target.index);
                }
            }
            _ => {}
        }
    }

    fn analyze_else_branch(&mut self, else_branch: &ast::ElseBranch) {
        match else_branch {
            ast::ElseBranch::ElseIf(if_stmt) => {
                self.analyze_expr(&if_stmt.condition);
                self.analyze_block(&if_stmt.then_body);
                if let Some(ref inner_else) = if_stmt.else_body {
                    self.analyze_else_branch(inner_else);
                }
            }
            ast::ElseBranch::Else(block) => {
                self.analyze_block(block);
            }
        }
    }

    fn analyze_for_range(&mut self, range: &ast::ForRange) {
        match range {
            ast::ForRange::Range { start, end } => {
                self.analyze_expr(start);
                self.analyze_expr(end);
            }
            ast::ForRange::Collection(expr) => {
                self.analyze_expr(expr);
            }
        }
    }

    fn analyze_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Ident(ident) => {
                // Track variable usage
                *self.variable_uses.entry(ident.name.clone()).or_insert(0) += 1;
            }
            Expr::Call(call) => {
                // Track function usage
                if let Expr::Ident(ident) = &call.callee {
                    *self.function_uses.entry(ident.name.clone()).or_insert(0) += 1;

                    // Check for assertions
                    if ident.name == "assert" || ident.name == "debug_assert" {
                        self.assertion_count += 1;
                    }
                }

                // Analyze callee and arguments
                self.analyze_expr(&call.callee);
                for arg in &call.args {
                    self.analyze_expr(arg);
                }
            }
            Expr::Binary(binary) => {
                self.analyze_expr(&binary.left);
                self.analyze_expr(&binary.right);
            }
            Expr::Unary(unary) => {
                self.analyze_expr(&unary.operand);
            }
            Expr::Index(index) => {
                self.analyze_expr(&index.object);
                self.analyze_expr(&index.index);
            }
            Expr::Field(field) => {
                self.analyze_expr(&field.object);
            }
            Expr::If(if_expr) => {
                self.analyze_expr(&if_expr.condition);
                self.analyze_expr(&if_expr.then_expr);
                self.analyze_expr(&if_expr.else_expr);
            }
            Expr::Block(block_expr) => {
                self.current_depth += 1;
                self.check_nesting_depth(block_expr.location.as_ref());
                for stmt in &block_expr.statements {
                    self.analyze_stmt(stmt);
                }
                if let Some(ref trailing) = block_expr.trailing_expr {
                    self.analyze_expr(trailing);
                }
                self.current_depth -= 1;
            }
            Expr::Tuple(tuple) => {
                for elem in &tuple.elements {
                    self.analyze_expr(elem);
                }
            }
            Expr::Gate(gate) => {
                for param in &gate.params {
                    self.analyze_expr(param);
                }
                self.analyze_expr(&gate.target);
            }
            Expr::Measure(measure) => {
                self.analyze_expr(&measure.targets);
            }
            Expr::AngleLit(angle) => {
                self.check_angle_precision(angle);
                self.analyze_expr(&angle.value);
            }
            _ => {}
        }
    }

    // =========================================================================
    // Check Functions
    // =========================================================================

    fn check_function_name(&mut self, name: &str, location: Option<&ast::SourceLocation>) {
        if !self.config.function_naming.enabled {
            return;
        }

        // Skip main and special names
        if name == "main" || name.starts_with('_') {
            return;
        }

        if !is_snake_case(name) {
            let mut diag = LintDiagnostic::new(
                "function_naming",
                format!("function `{}` should be snake_case", name),
                self.config.function_naming.severity,
            )
            .with_suggestion(format!("rename to `{}`", to_snake_case(name)));

            if let Some(loc) = location {
                diag = diag.with_location(loc.clone());
            }

            self.diagnostics.push(diag);
        }
    }

    fn check_variable_name(&mut self, name: &str, location: Option<&ast::SourceLocation>) {
        if !self.config.variable_naming.enabled {
            return;
        }

        // Skip underscore-prefixed names (intentionally unused)
        if name.starts_with('_') {
            return;
        }

        // Common short names are allowed
        if matches!(
            name,
            "i" | "j" | "k" | "n" | "x" | "y" | "z" | "q" | "r" | "c"
        ) {
            return;
        }

        if !is_snake_case(name) {
            let snake_name = to_snake_case(name);
            let mut diag = LintDiagnostic::new(
                "variable_naming",
                format!("variable `{}` should be snake_case", name),
                self.config.variable_naming.severity,
            )
            .with_suggestion(format!("rename to `{}`", snake_name));

            if let Some(loc) = location {
                diag = diag.with_location(loc.clone());
                // Note: Renaming is unsafe because we'd need to rename all uses
                // For now, we don't provide an auto-fix for renames
            }

            self.diagnostics.push(diag);
        }
    }

    fn check_type_name(&mut self, name: &str, location: Option<&ast::SourceLocation>) {
        if !self.config.type_naming.enabled {
            return;
        }

        if !is_pascal_case(name) {
            let mut diag = LintDiagnostic::new(
                "type_naming",
                format!("type `{}` should be PascalCase", name),
                self.config.type_naming.severity,
            )
            .with_suggestion(format!("rename to `{}`", to_pascal_case(name)));

            if let Some(loc) = location {
                diag = diag.with_location(loc.clone());
            }

            self.diagnostics.push(diag);
        }
    }

    fn check_function_length(&mut self, fn_decl: &ast::FnDecl) {
        if !self.config.function_too_long.enabled {
            return;
        }

        let line_count = count_block_lines(&fn_decl.body);

        if line_count > self.config.max_function_lines {
            let mut diag = LintDiagnostic::new(
                "function_too_long",
                format!(
                    "function `{}` has {} lines, exceeds maximum of {} (NASA PoT Rule 4)",
                    fn_decl.name, line_count, self.config.max_function_lines
                ),
                self.config.function_too_long.severity,
            )
            .with_suggestion("consider breaking into smaller functions");

            if let Some(ref loc) = fn_decl.location {
                diag = diag.with_location(loc.clone());
            }

            self.diagnostics.push(diag);
        }
    }

    fn check_nesting_depth(&mut self, location: Option<&ast::SourceLocation>) {
        if !self.config.deep_nesting.enabled {
            return;
        }

        if self.current_depth > self.config.max_nesting_depth {
            let mut diag = LintDiagnostic::new(
                "deep_nesting",
                format!(
                    "nesting depth {} exceeds maximum of {} (NASA PoT Rule 1)",
                    self.current_depth, self.config.max_nesting_depth
                ),
                self.config.deep_nesting.severity,
            )
            .with_suggestion("consider extracting to a separate function or simplifying logic");

            if let Some(loc) = location {
                diag = diag.with_location(loc.clone());
            }

            self.diagnostics.push(diag);
        }
    }

    fn check_unused(&mut self) {
        // Check unused variables
        if self.config.unused_variable.enabled {
            for (name, location) in &self.defined_variables {
                if name.starts_with('_') {
                    continue; // Intentionally unused
                }
                if self.variable_uses.get(name).copied().unwrap_or(0) == 0 {
                    let mut diag = LintDiagnostic::new(
                        "unused_variable",
                        format!("unused variable `{}`", name),
                        self.config.unused_variable.severity,
                    )
                    .with_location(location.clone())
                    .with_suggestion(format!("prefix with underscore: `_{}`", name));

                    // Generate safe fix: prefix with underscore
                    if let Some(ref source) = self.source {
                        let start = location_to_offset(source, location.line, location.column);
                        let end = start + name.len();
                        diag = diag.with_fix(LintFix {
                            start,
                            end,
                            replacement: format!("_{}", name),
                            safety: FixSafety::Safe,
                        });
                    }

                    self.diagnostics.push(diag);
                }
            }
        }

        // Check unused functions
        if self.config.unused_function.enabled {
            for (name, (location, is_pub)) in &self.defined_functions {
                if name == "main" || *is_pub {
                    continue; // main and pub functions are considered used
                }
                if self.function_uses.get(name).copied().unwrap_or(0) == 0 {
                    self.diagnostics.push(
                        LintDiagnostic::new(
                            "unused_function",
                            format!("unused function `{}`", name),
                            self.config.unused_function.severity,
                        )
                        .with_location(location.clone())
                        .with_suggestion("remove or add `pub` if intended for export"),
                    );
                    // Note: No auto-fix for unused functions - removing code is always unsafe
                }
            }
        }
    }

    fn check_assertion_density(&mut self) {
        if !self.config.low_assertion_density.enabled || self.function_count == 0 {
            return;
        }

        let density = self.total_assertions as f64 / self.function_count as f64;

        if density < self.config.min_assertions_per_function {
            self.diagnostics.push(LintDiagnostic::new(
                "low_assertion_density",
                format!(
                    "assertion density is {:.1} per function, below minimum of {:.1} (NASA PoT Rule 5)",
                    density, self.config.min_assertions_per_function
                ),
                self.config.low_assertion_density.severity,
            )
            .with_suggestion("add `assert()` calls to verify invariants and preconditions"));
        }
    }

    fn check_angle_precision(&mut self, angle: &ast::AngleLit) {
        use crate::ast::AngleUnit;
        use crate::rational::Rational;

        // Check if using radians when exact turn fractions exist
        if self.config.prefer_turns_over_radians.enabled
            && let AngleUnit::Rad = angle.unit
        {
            // Try to evaluate the expression to see if it's a common angle
            if let Some(suggestion) = self.suggest_turn_equivalent(angle) {
                self.diagnostics.push(
                    LintDiagnostic::new(
                        "prefer_turns_over_radians",
                        "radians can lose precision; consider using turns for exact representation"
                            .to_string(),
                        self.config.prefer_turns_over_radians.severity,
                    )
                    .with_location_opt(angle.location.clone())
                    .with_suggestion(&suggestion),
                );
            }
        }

        // Check if using decimal turns when fractions exist
        if self.config.prefer_fraction_turns.enabled
            && let AngleUnit::Turns = angle.unit
            && let Expr::FloatLit(lit) = &angle.value
            && let Some(fraction_str) = suggest_fraction_for_decimal(lit.value)
        {
            let mut diag = LintDiagnostic::new(
                "prefer_fraction_turns",
                format!(
                    "decimal `{}` can be expressed exactly as `{}`",
                    lit.value, fraction_str
                ),
                self.config.prefer_fraction_turns.severity,
            )
            .with_location_opt(angle.location.clone())
            .with_suggestion(format!(
                "use `{} turns` for exact representation",
                fraction_str
            ));

            // Generate safe fix: replace the float literal with the fraction
            if let (Some(source), Some(lit_loc)) = (&self.source, &lit.location) {
                let start = location_to_offset(source, lit_loc.line, lit_loc.column);
                let end = location_to_offset(source, lit_loc.end_line, lit_loc.end_column);
                diag = diag.with_fix(LintFix {
                    start,
                    end,
                    replacement: fraction_str.clone(),
                    safety: FixSafety::Safe,
                });
            }

            self.diagnostics.push(diag);
        }

        // Check if using float literals that could be exact fractions or std constants
        if self.config.prefer_exact_angles.enabled
            && let Expr::FloatLit(lit) = &angle.value
        {
            let value = lit.value;

            // Check if this float could be a rational fraction
            if let Some(r) = Rational::from_f64_common(value) {
                // Only suggest if it's a "nice" fraction (small denominator)
                if r.denominator() <= 16 && !r.is_integer() {
                    let suggestion = if r.numerator() == 1 {
                        format!("1/{}", r.denominator())
                    } else {
                        format!("{}/{}", r.numerator(), r.denominator())
                    };

                    self.diagnostics.push(
                        LintDiagnostic::new(
                            "prefer_exact_angles",
                            format!(
                                "float literal `{}` can be expressed exactly as fraction `{}`",
                                value, suggestion
                            ),
                            self.config.prefer_exact_angles.severity,
                        )
                        .with_location_opt(angle.location.clone())
                        .with_suggestion(format!(
                            "use `{} {}` for exact representation",
                            suggestion,
                            match angle.unit {
                                AngleUnit::Turns => "turns",
                                AngleUnit::Rad => "rad",
                            }
                        )),
                    );
                }
            }

            // For radians, also check if it's a pi multiple (like 3.14159...)
            if let AngleUnit::Rad = angle.unit
                && let Some((n, d)) = Rational::from_f64_pi_multiple(value)
            {
                let pi_suggestion = if n == 1 && d == 1 {
                    "std.f64.pi".to_string()
                } else if n == 1 {
                    format!("std.f64.pi/{}", d)
                } else if d == 1 {
                    format!("{}*std.f64.pi", n)
                } else {
                    format!("{}*std.f64.pi/{}", n, d)
                };

                // Also suggest the turns equivalent
                let turns_fraction = Rational::new(n, 2 * d as i64);
                let turns_suggestion = if turns_fraction.numerator() == 1 {
                    format!("1/{}", turns_fraction.denominator())
                } else {
                    format!(
                        "{}/{}",
                        turns_fraction.numerator(),
                        turns_fraction.denominator()
                    )
                };

                self.diagnostics.push(
                    LintDiagnostic::new(
                        "prefer_exact_angles",
                        format!(
                            "float `{}` appears to be {}*pi/{}; use exact form or turns",
                            value, n, d
                        ),
                        self.config.prefer_exact_angles.severity,
                    )
                    .with_location_opt(angle.location.clone())
                    .with_suggestion(format!(
                        "use `{} rad` or `{} turns` for exact representation",
                        pi_suggestion, turns_suggestion
                    )),
                );
            }
        }
    }

    fn suggest_turn_equivalent(&self, angle: &ast::AngleLit) -> Option<String> {
        use crate::comptime::ComptimeEvaluator;
        use crate::rational::Rational;

        let mut eval = ComptimeEvaluator::new();
        let value = eval.eval_expr(&angle.value).ok()?;
        let radians = match value {
            crate::comptime::ComptimeValue::Float(f) => f,
            crate::comptime::ComptimeValue::Int(i) => i as f64,
            crate::comptime::ComptimeValue::Rational(r) => r.to_f64(),
            _ => return None,
        };

        // Use Rational to detect pi multiples and convert to turns
        if let Some(turns_rational) = Rational::radians_to_turns(radians)
            && (!turns_rational.is_integer() || turns_rational.numerator() != 0)
        {
            let suggestion = if turns_rational.numerator() == 1 {
                format!("1/{}", turns_rational.denominator())
            } else {
                format!(
                    "{}/{}",
                    turns_rational.numerator(),
                    turns_rational.denominator()
                )
            };
            return Some(format!("use `{} turns` instead", suggestion));
        }

        None
    }
}

/// Suggest a fraction representation for a decimal turn value
fn suggest_fraction_for_decimal(value: f64) -> Option<String> {
    const TOLERANCE: f64 = 1e-10;
    let common_fractions = [
        (0.5, "1/2"),
        (0.25, "1/4"),
        (0.125, "1/8"),
        (0.0625, "1/16"),
        (1.0 / 3.0, "1/3"),
        (1.0 / 6.0, "1/6"),
        (0.75, "3/4"),
        (0.375, "3/8"),
        (2.0 / 3.0, "2/3"),
        (0.1, "1/10"),
        (0.2, "1/5"),
    ];

    for (frac_val, frac_str) in common_fractions {
        if (value - frac_val).abs() < TOLERANCE {
            return Some(frac_str.to_string());
        }
    }

    None
}

// =============================================================================
// Naming Convention Helpers
// =============================================================================

fn is_snake_case(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }

    let mut chars = s.chars().peekable();

    // First char must be lowercase or underscore
    match chars.next() {
        Some(c) if c.is_ascii_lowercase() || c == '_' => {}
        _ => return false,
    }

    // Rest must be lowercase, digits, or underscores
    for c in chars {
        if !c.is_ascii_lowercase() && !c.is_ascii_digit() && c != '_' {
            return false;
        }
    }

    // No double underscores
    !s.contains("__")
}

fn is_pascal_case(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }

    let mut chars = s.chars();

    // First char must be uppercase
    match chars.next() {
        Some(c) if c.is_ascii_uppercase() => {}
        _ => return false,
    }

    // No underscores allowed and must have at least one lowercase letter
    // (to distinguish from SCREAMING_SNAKE_CASE)
    !s.contains('_') && s.chars().any(|c| c.is_ascii_lowercase())
}

fn to_snake_case(s: &str) -> String {
    let mut result = String::new();

    for (i, c) in s.chars().enumerate() {
        if c.is_ascii_uppercase() {
            if i > 0 {
                result.push('_');
            }
            result.push(c.to_ascii_lowercase());
        } else {
            result.push(c);
        }
    }

    result
}

fn to_pascal_case(s: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = true;

    for c in s.chars() {
        if c == '_' {
            capitalize_next = true;
        } else if capitalize_next {
            result.push(c.to_ascii_uppercase());
            capitalize_next = false;
        } else {
            result.push(c);
        }
    }

    result
}

/// Count approximate lines in a block (for function length checking).
fn count_block_lines(block: &ast::Block) -> usize {
    if let (Some(start), Some(end)) = (&block.location, block.statements.last())
        && let Some(end_loc) = get_stmt_location(end)
    {
        return (end_loc.line.saturating_sub(start.line) + 1) as usize;
    }

    // Fallback: count statements
    block.statements.len()
}

fn get_stmt_location(stmt: &Stmt) -> Option<ast::SourceLocation> {
    match stmt {
        Stmt::Binding(b) => b.location.clone(),
        Stmt::Expr(e) => e.location.clone(),
        Stmt::If(i) => i.location.clone(),
        Stmt::For(f) => f.location.clone(),
        Stmt::Return(r) => r.location.clone(),
        Stmt::Block(b) => b.location.clone(),
        Stmt::Tick(t) => t.location.clone(),
        Stmt::Assign(a) => a.location.clone(),
        Stmt::Gate(g) => g.location.clone(),
        Stmt::Measure(m) => m.location.clone(),
        _ => None,
    }
}

// =============================================================================
// Public API
// =============================================================================

/// Lint a program with default configuration.
pub fn lint(program: &Program) -> Vec<LintDiagnostic> {
    Linter::new(LintConfig::default()).lint(program)
}

/// Lint a program with strict configuration.
pub fn lint_strict(program: &Program) -> Vec<LintDiagnostic> {
    Linter::new(LintConfig::strict()).lint(program)
}

// =============================================================================
// Fix Application
// =============================================================================

/// Result of applying fixes.
#[derive(Debug)]
pub struct FixResult {
    /// The modified source code.
    pub source: String,
    /// Number of safe fixes applied.
    pub safe_fixes_applied: usize,
    /// Number of unsafe fixes applied.
    pub unsafe_fixes_applied: usize,
    /// Number of fixes skipped (due to safety or conflicts).
    pub fixes_skipped: usize,
}

/// Apply fixes from diagnostics to source code.
///
/// # Arguments
/// * `source` - Original source code
/// * `diagnostics` - Lint diagnostics with potential fixes
/// * `include_unsafe` - Whether to apply unsafe fixes
///
/// # Returns
/// The modified source code and statistics about fixes applied.
pub fn apply_fixes(
    source: &str,
    diagnostics: &[LintDiagnostic],
    include_unsafe: bool,
) -> FixResult {
    // Collect all applicable fixes
    let mut fixes: Vec<&LintFix> = diagnostics
        .iter()
        .filter_map(|d| d.fix.as_ref())
        .filter(|f| include_unsafe || f.safety == FixSafety::Safe)
        .collect();

    // Sort by start position (descending) so we can apply from end to start
    // This prevents offset shifts from affecting earlier fixes
    fixes.sort_by(|a, b| b.start.cmp(&a.start));

    // Check for overlapping fixes and remove conflicts
    let mut result = source.to_string();
    let mut safe_applied = 0;
    let mut unsafe_applied = 0;
    let mut skipped = 0;
    let mut last_start = usize::MAX;

    for fix in fixes {
        // Skip if this fix overlaps with a previously applied fix
        if fix.end > last_start {
            skipped += 1;
            continue;
        }

        // Apply the fix
        if fix.start <= result.len() && fix.end <= result.len() {
            result.replace_range(fix.start..fix.end, &fix.replacement);
            last_start = fix.start;

            match fix.safety {
                FixSafety::Safe => safe_applied += 1,
                FixSafety::Unsafe => unsafe_applied += 1,
            }
        } else {
            skipped += 1;
        }
    }

    // Count skipped fixes (those not matching safety criteria)
    let total_fixes = diagnostics.iter().filter(|d| d.fix.is_some()).count();
    let not_applied = total_fixes - safe_applied - unsafe_applied;

    FixResult {
        source: result,
        safe_fixes_applied: safe_applied,
        unsafe_fixes_applied: unsafe_applied,
        fixes_skipped: skipped + not_applied,
    }
}

/// Compute byte offset from line/column (1-indexed).
pub fn location_to_offset(source: &str, line: u32, column: u32) -> usize {
    source
        .lines()
        .take(line.saturating_sub(1) as usize)
        .map(|l| l.len() + 1) // +1 for newline
        .sum::<usize>()
        + column.saturating_sub(1) as usize
}

/// Compute byte offset for end of a source location.
pub fn location_end_offset(source: &str, loc: &ast::SourceLocation) -> usize {
    location_to_offset(source, loc.end_line, loc.end_column)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse;

    fn lint_source(source: &str) -> Vec<LintDiagnostic> {
        let program = parse(source).expect("parse failed");
        Linter::new(LintConfig::default()).lint(&program)
    }

    fn lint_source_strict(source: &str) -> Vec<LintDiagnostic> {
        let program = parse(source).expect("parse failed");
        Linter::new(LintConfig::strict()).lint(&program)
    }

    #[test]
    fn test_snake_case_detection() {
        assert!(is_snake_case("foo"));
        assert!(is_snake_case("foo_bar"));
        assert!(is_snake_case("foo_bar_baz"));
        assert!(is_snake_case("_foo"));
        assert!(is_snake_case("foo2"));

        assert!(!is_snake_case("Foo"));
        assert!(!is_snake_case("fooBar"));
        assert!(!is_snake_case("FOO"));
        assert!(!is_snake_case("foo__bar"));
    }

    #[test]
    fn test_pascal_case_detection() {
        assert!(is_pascal_case("Foo"));
        assert!(is_pascal_case("FooBar"));
        assert!(is_pascal_case("FooBarBaz"));

        assert!(!is_pascal_case("foo"));
        assert!(!is_pascal_case("foo_bar"));
        assert!(!is_pascal_case("FOO"));
    }

    #[test]
    fn test_function_naming_lint() {
        let diags = lint_source(
            r#"
            fn badName() -> unit {
                return unit;
            }
        "#,
        );

        assert!(
            diags.iter().any(|d| d.rule == "function_naming"),
            "expected function_naming lint"
        );
    }

    #[test]
    fn test_good_function_naming() {
        let diags = lint_source(
            r#"
            fn good_name() -> unit {
                return unit;
            }
        "#,
        );

        assert!(
            !diags.iter().any(|d| d.rule == "function_naming"),
            "unexpected function_naming lint"
        );
    }

    #[test]
    fn test_type_naming_lint() {
        // Note: The struct syntax `Name := struct {}` parses as a Binding in the current grammar.
        // Type naming is checked on TopLevelDecl::Struct, so this test would need
        // the grammar to be updated. For now, just verify the is_pascal_case function works.
        assert!(!is_pascal_case("bad_name"));
        assert!(is_pascal_case("GoodName"));
    }

    #[test]
    fn test_unused_variable_lint() {
        let diags = lint_source(
            r#"
            fn main() -> unit {
                x := 5;
                return unit;
            }
        "#,
        );

        assert!(
            diags.iter().any(|d| d.rule == "unused_variable"),
            "expected unused_variable lint"
        );
    }

    #[test]
    fn test_underscore_unused_ok() {
        let diags = lint_source(
            r#"
            fn main() -> unit {
                _x := 5;
                return unit;
            }
        "#,
        );

        assert!(
            !diags.iter().any(|d| d.rule == "unused_variable"),
            "underscore prefix should suppress unused warning"
        );
    }

    #[test]
    fn test_deep_nesting_lint() {
        let diags = lint_source_strict(
            r#"
            fn main() -> unit {
                if (true) {
                    if (true) {
                        if (true) {
                            if (true) {
                                if (true) {
                                    x := 1;
                                }
                            }
                        }
                    }
                }
                return unit;
            }
        "#,
        );

        assert!(
            diags.iter().any(|d| d.rule == "deep_nesting"),
            "expected deep_nesting lint"
        );
    }

    #[test]
    fn test_severity_display() {
        let diag = LintDiagnostic::new("test_rule", "test message", Severity::Warning);
        let s = diag.to_string();
        assert!(s.contains("warning"));
        assert!(s.contains("test message"));
        assert!(s.contains("test_rule"));
    }

    // =========================================================================
    // Assertion Density Tests
    // =========================================================================

    #[test]
    fn test_low_assertion_density() {
        let diags = lint_source_strict(
            r#"
            fn foo() -> unit {
                x := 1;
                return unit;
            }
            fn bar() -> unit {
                y := 2;
                return unit;
            }
        "#,
        );

        assert!(
            diags.iter().any(|d| d.rule == "low_assertion_density"),
            "expected low_assertion_density lint when no assertions present"
        );
    }

    #[test]
    fn test_good_assertion_density() {
        let diags = lint_source_strict(
            r#"
            fn foo() -> unit {
                x := 1;
                assert(x > 0);
                assert(x < 10);
                return unit;
            }
        "#,
        );

        assert!(
            !diags.iter().any(|d| d.rule == "low_assertion_density"),
            "should not warn when assertion density is good"
        );
    }

    // =========================================================================
    // Unused Function Tests
    // =========================================================================

    #[test]
    fn test_unused_function_lint() {
        let diags = lint_source_strict(
            r#"
            fn helper() -> unit {
                return unit;
            }
            pub fn main() -> unit {
                return unit;
            }
        "#,
        );

        assert!(
            diags
                .iter()
                .any(|d| d.rule == "unused_function" && d.message.contains("helper")),
            "expected unused_function lint for helper"
        );
    }

    #[test]
    fn test_used_function_no_lint() {
        let diags = lint_source_strict(
            r#"
            fn helper() -> unit {
                return unit;
            }
            pub fn main() -> unit {
                helper();
                return unit;
            }
        "#,
        );

        assert!(
            !diags
                .iter()
                .any(|d| d.rule == "unused_function" && d.message.contains("helper")),
            "should not lint used function"
        );
    }

    #[test]
    fn test_pub_function_not_unused() {
        let diags = lint_source_strict(
            r#"
            pub fn exported() -> unit {
                return unit;
            }
        "#,
        );

        assert!(
            !diags
                .iter()
                .any(|d| d.rule == "unused_function" && d.message.contains("exported")),
            "pub functions should not be considered unused"
        );
    }

    #[test]
    fn test_main_not_unused() {
        let diags = lint_source_strict(
            r#"
            fn main() -> unit {
                return unit;
            }
        "#,
        );

        assert!(
            !diags
                .iter()
                .any(|d| d.rule == "unused_function" && d.message.contains("main")),
            "main should never be considered unused"
        );
    }

    // =========================================================================
    // Variable Usage Tests
    // =========================================================================

    #[test]
    fn test_used_variable_no_lint() {
        let diags = lint_source(
            r#"
            fn main() -> unit {
                x := 5;
                y := x + 1;
                return unit;
            }
        "#,
        );

        assert!(
            !diags
                .iter()
                .any(|d| d.rule == "unused_variable" && d.message.contains("`x`")),
            "used variable should not be flagged"
        );
    }

    #[test]
    fn test_multiple_unused_variables() {
        let diags = lint_source(
            r#"
            fn main() -> unit {
                a := 1;
                b := 2;
                c := 3;
                return unit;
            }
        "#,
        );

        let unused_count = diags.iter().filter(|d| d.rule == "unused_variable").count();
        assert!(
            unused_count >= 3,
            "expected at least 3 unused variable lints, got {}",
            unused_count
        );
    }

    // =========================================================================
    // Naming Convention Tests
    // =========================================================================

    #[test]
    fn test_camel_case_function_lint() {
        let diags = lint_source(
            r#"
            fn myFunction() -> unit {
                return unit;
            }
        "#,
        );

        assert!(
            diags.iter().any(|d| d.rule == "function_naming"),
            "camelCase function should trigger lint"
        );
    }

    #[test]
    fn test_screaming_snake_case_function_lint() {
        let diags = lint_source(
            r#"
            fn MY_FUNCTION() -> unit {
                return unit;
            }
        "#,
        );

        assert!(
            diags.iter().any(|d| d.rule == "function_naming"),
            "SCREAMING_SNAKE_CASE function should trigger lint"
        );
    }

    #[test]
    fn test_single_letter_variable_ok() {
        let diags = lint_source(
            r#"
            fn main() -> unit {
                x := 1;
                y := 2;
                q := x + y;
                return unit;
            }
        "#,
        );

        // Single letter names (x, y, q, etc.) should not trigger naming lint
        assert!(
            !diags.iter().any(|d| d.rule == "variable_naming"),
            "single letter variables should be allowed"
        );
    }

    // =========================================================================
    // Nesting Depth Tests
    // =========================================================================

    #[test]
    fn test_acceptable_nesting() {
        let diags = lint_source_strict(
            r#"
            fn main() -> unit {
                if (true) {
                    if (true) {
                        x := 1;
                    }
                }
                return unit;
            }
        "#,
        );

        assert!(
            !diags.iter().any(|d| d.rule == "deep_nesting"),
            "nesting depth of 2-3 should be acceptable"
        );
    }

    #[test]
    fn test_for_loop_nesting_counts() {
        let diags = lint_source_strict(
            r#"
            fn main() -> unit {
                for i in 0..10 {
                    for j in 0..10 {
                        for k in 0..10 {
                            for l in 0..10 {
                                for m in 0..10 {
                                    x := 1;
                                }
                            }
                        }
                    }
                }
                return unit;
            }
        "#,
        );

        assert!(
            diags.iter().any(|d| d.rule == "deep_nesting"),
            "deeply nested for loops should trigger lint"
        );
    }

    // =========================================================================
    // Configuration Tests
    // =========================================================================

    #[test]
    fn test_minimal_config_fewer_warnings() {
        let program = parse(
            r#"
            fn badName() -> unit {
                x := 5;
                return unit;
            }
        "#,
        )
        .expect("parse failed");

        let strict_diags = Linter::new(LintConfig::strict()).lint(&program);
        let minimal_diags = Linter::new(LintConfig::minimal()).lint(&program);

        assert!(
            minimal_diags.len() < strict_diags.len(),
            "minimal config should produce fewer diagnostics"
        );
    }

    #[test]
    fn test_relaxed_config_warnings_not_errors() {
        let program = parse(
            r#"
            fn badName() -> unit {
                return unit;
            }
        "#,
        )
        .expect("parse failed");

        let relaxed_diags = Linter::new(LintConfig::relaxed()).lint(&program);

        // In relaxed mode, function naming should be a warning, not error
        let naming_diag = relaxed_diags.iter().find(|d| d.rule == "function_naming");
        if let Some(diag) = naming_diag {
            assert!(
                matches!(diag.severity, Severity::Warning | Severity::Hint),
                "relaxed config should use warnings, not errors"
            );
        }
    }

    // =========================================================================
    // Suggestion Tests
    // =========================================================================

    #[test]
    fn test_function_naming_suggestion() {
        let diags = lint_source(
            r#"
            fn badName() -> unit {
                return unit;
            }
        "#,
        );

        let naming_diag = diags.iter().find(|d| d.rule == "function_naming");
        assert!(naming_diag.is_some(), "should have function_naming lint");

        let suggestion = &naming_diag.unwrap().suggestion;
        assert!(suggestion.is_some(), "should have a suggestion");
        assert!(
            suggestion.as_ref().unwrap().contains("bad_name"),
            "suggestion should include snake_case version"
        );
    }

    #[test]
    fn test_unused_variable_suggestion() {
        let diags = lint_source(
            r#"
            fn main() -> unit {
                myVar := 5;
                return unit;
            }
        "#,
        );

        let unused_diag = diags.iter().find(|d| d.rule == "unused_variable");
        assert!(unused_diag.is_some(), "should have unused_variable lint");

        let suggestion = &unused_diag.unwrap().suggestion;
        assert!(suggestion.is_some(), "should have a suggestion");
        assert!(
            suggestion.as_ref().unwrap().contains("_myVar"),
            "suggestion should recommend underscore prefix"
        );
    }

    // =========================================================================
    // Quantum-Specific Tests
    // =========================================================================

    #[test]
    fn test_gate_operations_track_usage() {
        let diags = lint_source(
            r#"
            pub fn main() -> unit {
                mut q := qalloc(2);
                pz q;
                h q[0];
                cx (q[0], q[1]);
                return unit;
            }
        "#,
        );

        // q should be considered used because of gate operations
        assert!(
            !diags
                .iter()
                .any(|d| d.rule == "unused_variable" && d.message.contains("`q`")),
            "allocator used in gates should not be flagged as unused"
        );
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[test]
    fn test_empty_program() {
        let diags = lint_source("");
        // Should not crash, might have assertion density warning
        assert!(diags.iter().all(|d| d.rule != "unused_variable"));
    }

    #[test]
    fn test_program_with_only_comments() {
        // Comments are stripped during parsing, so this is effectively empty
        let result = parse("// just a comment");
        // Parser may or may not accept this - either way, linter shouldn't crash
        if let Ok(program) = result {
            let _diags = Linter::new(LintConfig::default()).lint(&program);
        }
    }

    #[test]
    fn test_to_snake_case_conversion() {
        assert_eq!(to_snake_case("myFunction"), "my_function");
        assert_eq!(to_snake_case("MyClass"), "my_class");
        assert_eq!(to_snake_case("XMLParser"), "x_m_l_parser");
        assert_eq!(to_snake_case("already_snake"), "already_snake");
    }

    #[test]
    fn test_to_pascal_case_conversion() {
        assert_eq!(to_pascal_case("my_type"), "MyType");
        assert_eq!(to_pascal_case("some_struct"), "SomeStruct");
        assert_eq!(to_pascal_case("already"), "Already");
    }

    // =========================================================================
    // Critical Edge Cases
    // =========================================================================

    #[test]
    fn test_variable_used_in_return() {
        let diags = lint_source(
            r#"
            fn add(a: u32, b: u32) -> u32 {
                result := a + b;
                return result;
            }
        "#,
        );

        assert!(
            !diags
                .iter()
                .any(|d| d.rule == "unused_variable" && d.message.contains("result")),
            "variable used in return should not be flagged"
        );
    }

    #[test]
    fn test_variable_used_in_condition() {
        let diags = lint_source(
            r#"
            fn main() -> unit {
                flag := true;
                if (flag) {
                    x := 1;
                }
                return unit;
            }
        "#,
        );

        assert!(
            !diags
                .iter()
                .any(|d| d.rule == "unused_variable" && d.message.contains("flag")),
            "variable used in condition should not be flagged"
        );
    }

    #[test]
    fn test_parameter_unused() {
        let diags = lint_source(
            r#"
            fn ignore_param(x: u32) -> unit {
                return unit;
            }
        "#,
        );

        assert!(
            diags
                .iter()
                .any(|d| d.rule == "unused_variable" && d.message.contains("x")),
            "unused parameter should be flagged"
        );
    }

    #[test]
    fn test_parameter_underscore_prefix_ok() {
        let diags = lint_source(
            r#"
            fn ignore_param(_x: u32) -> unit {
                return unit;
            }
        "#,
        );

        assert!(
            !diags
                .iter()
                .any(|d| d.rule == "unused_variable" && d.message.contains("_x")),
            "underscore-prefixed parameter should not be flagged"
        );
    }

    #[test]
    fn test_shadowed_variable() {
        let diags = lint_source(
            r#"
            fn main() -> unit {
                x := 1;
                x := 2;
                y := x;
                return unit;
            }
        "#,
        );

        // The first x is shadowed but that's valid - just check we don't crash
        assert!(diags.iter().filter(|d| d.rule == "unused_variable").count() <= 2);
    }

    #[test]
    fn test_recursive_function_call() {
        let diags = lint_source(
            r#"
            fn factorial(n: u32) -> u32 {
                if (n <= 1) {
                    return 1;
                }
                return n * factorial(n - 1);
            }
        "#,
        );

        // factorial calls itself, so it should be considered used
        assert!(
            !diags
                .iter()
                .any(|d| d.rule == "unused_function" && d.message.contains("factorial")),
            "recursive function should not be flagged as unused"
        );
    }

    #[test]
    fn test_mutual_recursion() {
        let diags = lint_source(
            r#"
            fn is_even(n: u32) -> bool {
                if (n == 0) { return true; }
                return is_odd(n - 1);
            }
            fn is_odd(n: u32) -> bool {
                if (n == 0) { return false; }
                return is_even(n - 1);
            }
            pub fn main() -> unit {
                r := is_even(10);
                return unit;
            }
        "#,
        );

        // Both functions are used via mutual recursion
        assert!(
            !diags
                .iter()
                .any(|d| d.rule == "unused_function" && d.message.contains("is_even")),
            "mutually recursive function should not be flagged"
        );
        assert!(
            !diags
                .iter()
                .any(|d| d.rule == "unused_function" && d.message.contains("is_odd")),
            "mutually recursive function should not be flagged"
        );
    }

    #[test]
    fn test_nested_blocks_variable_scope() {
        let diags = lint_source(
            r#"
            fn main() -> unit {
                {
                    inner := 5;
                }
                return unit;
            }
        "#,
        );

        assert!(
            diags
                .iter()
                .any(|d| d.rule == "unused_variable" && d.message.contains("inner")),
            "variable unused in nested block should be flagged"
        );
    }

    #[test]
    fn test_tick_block_operations() {
        let diags = lint_source(
            r#"
            pub fn main() -> unit {
                mut q := qalloc(2);
                pz q;
                tick {
                    h q[0];
                    h q[1];
                }
                return unit;
            }
        "#,
        );

        assert!(
            !diags
                .iter()
                .any(|d| d.rule == "unused_variable" && d.message.contains("q")),
            "allocator used in tick block should not be flagged"
        );
    }

    #[test]
    fn test_measurement_result_unused() {
        // This would test unused_measurement rule if implemented
        let diags = lint_source(
            r#"
            pub fn main() -> unit {
                mut q := qalloc(1);
                pz q;
                h q[0];
                mz(u1) q[0];
                return unit;
            }
        "#,
        );

        // Currently we don't track measurement results specially
        // Just verify it doesn't crash
        let _ = diags;
    }

    #[test]
    fn test_complex_expression_usage() {
        let diags = lint_source(
            r#"
            fn main() -> unit {
                a := 1;
                b := 2;
                c := 3;
                result := (a + b) * c;
                return unit;
            }
        "#,
        );

        // a, b, c are all used in the expression
        assert!(
            !diags
                .iter()
                .any(|d| d.rule == "unused_variable" && d.message.contains("`a`")),
            "variable used in expression should not be flagged"
        );
        assert!(
            !diags
                .iter()
                .any(|d| d.rule == "unused_variable" && d.message.contains("`b`")),
            "variable used in expression should not be flagged"
        );
        assert!(
            !diags
                .iter()
                .any(|d| d.rule == "unused_variable" && d.message.contains("`c`")),
            "variable used in expression should not be flagged"
        );
    }

    #[test]
    fn test_all_severities() {
        // Test that all severity levels display correctly
        for (sev, expected) in [
            (Severity::Hint, "hint"),
            (Severity::Warning, "warning"),
            (Severity::Error, "error"),
            (Severity::Deny, "deny"),
        ] {
            let diag = LintDiagnostic::new("test", "msg", sev);
            assert!(diag.to_string().contains(expected));
        }
    }

    #[test]
    fn test_diagnostic_with_location() {
        let loc = crate::ast::SourceLocation {
            line: 10,
            column: 5,
            end_line: 10,
            end_column: 15,
            file: Some("test.zlp".to_string()),
        };
        let diag = LintDiagnostic::new("test", "msg", Severity::Error).with_location(loc);
        let s = diag.to_string();
        assert!(s.contains("10"));
        assert!(s.contains("5"));
    }

    #[test]
    fn test_diagnostic_with_suggestion() {
        let diag = LintDiagnostic::new("test", "msg", Severity::Warning)
            .with_suggestion("try this instead");
        let s = diag.to_string();
        assert!(s.contains("try this instead"));
    }

    #[test]
    fn test_prefer_fraction_turns() {
        // Test that decimal turns trigger a suggestion for fractions
        let source = r#"
            fn main() -> unit {
                mut q := qalloc(1);
                rz(0.25 turns) q[0];
                return unit;
            }
        "#;
        let program = crate::parse(source).unwrap();
        let linter = Linter::new(LintConfig::strict());
        let diagnostics = linter.lint(&program);

        // Should have a suggestion to use 1/4 instead of 0.25
        let fraction_diags: Vec<_> = diagnostics
            .iter()
            .filter(|d| d.rule == "prefer_fraction_turns")
            .collect();
        assert!(
            !fraction_diags.is_empty(),
            "Expected prefer_fraction_turns diagnostic for 0.25 turns"
        );
    }

    #[test]
    fn test_fraction_turns_no_warning() {
        // Test that fraction turns don't trigger warnings
        let source = r#"
            fn main() -> unit {
                mut q := qalloc(1);
                rz(1/4 turns) q[0];
                return unit;
            }
        "#;
        let program = crate::parse(source).unwrap();
        let linter = Linter::new(LintConfig::strict());
        let diagnostics = linter.lint(&program);

        // Should NOT have prefer_fraction_turns diagnostic
        let fraction_diags: Vec<_> = diagnostics
            .iter()
            .filter(|d| d.rule == "prefer_fraction_turns")
            .collect();
        assert!(
            fraction_diags.is_empty(),
            "Should not warn about fraction turns: {:?}",
            fraction_diags
        );
    }

    // =========================================================================
    // Auto-fix Tests
    // =========================================================================

    #[test]
    fn test_fix_unused_variable() {
        let source = r#"fn main() -> unit {
    x := 5;
    return unit;
}"#;
        let program = crate::parse(source).unwrap();
        let diagnostics = Linter::new(LintConfig::strict())
            .with_source(source)
            .lint(&program);

        // Should have an unused_variable diagnostic with a fix
        let unused_diag = diagnostics.iter().find(|d| d.rule == "unused_variable");
        assert!(unused_diag.is_some(), "Expected unused_variable diagnostic");

        let diag = unused_diag.unwrap();
        assert!(diag.fix.is_some(), "Expected fix to be available");

        let fix = diag.fix.as_ref().unwrap();
        assert_eq!(fix.safety, FixSafety::Safe);
        assert_eq!(fix.replacement, "_x");
    }

    #[test]
    fn test_apply_unused_variable_fix() {
        let source = r#"fn main() -> unit {
    x := 5;
    return unit;
}"#;
        let program = crate::parse(source).unwrap();
        let diagnostics = Linter::new(LintConfig::strict())
            .with_source(source)
            .lint(&program);

        let result = apply_fixes(source, &diagnostics, false);

        assert_eq!(result.safe_fixes_applied, 1);
        assert!(
            result.source.contains("_x := 5"),
            "Expected fix to prefix with underscore"
        );
    }

    #[test]
    fn test_apply_multiple_fixes() {
        let source = r#"fn main() -> unit {
    a := 1;
    b := 2;
    c := 3;
    return unit;
}"#;
        let program = crate::parse(source).unwrap();
        let diagnostics = Linter::new(LintConfig::strict())
            .with_source(source)
            .lint(&program);

        let result = apply_fixes(source, &diagnostics, false);

        // Should fix all unused variables
        assert!(
            result.safe_fixes_applied >= 3,
            "Expected at least 3 fixes, got {}",
            result.safe_fixes_applied
        );
        assert!(result.source.contains("_a := 1"));
        assert!(result.source.contains("_b := 2"));
        assert!(result.source.contains("_c := 3"));
    }

    #[test]
    fn test_fix_decimal_turns() {
        let source = r#"fn main() -> unit {
    mut q := qalloc(1);
    rz(0.25 turns) q[0];
    return unit;
}"#;
        let program = crate::parse(source).unwrap();
        let diagnostics = Linter::new(LintConfig::strict())
            .with_source(source)
            .lint(&program);

        // Should have a prefer_fraction_turns diagnostic with a fix
        let fraction_diag = diagnostics
            .iter()
            .find(|d| d.rule == "prefer_fraction_turns");
        assert!(
            fraction_diag.is_some(),
            "Expected prefer_fraction_turns diagnostic"
        );

        let diag = fraction_diag.unwrap();
        assert!(diag.fix.is_some(), "Expected fix to be available");

        let fix = diag.fix.as_ref().unwrap();
        assert_eq!(fix.safety, FixSafety::Safe);
        assert_eq!(fix.replacement, "1/4");
    }

    #[test]
    fn test_apply_decimal_turns_fix() {
        let source = r#"fn main() -> unit {
    mut q := qalloc(1);
    rz(0.25 turns) q[0];
    return unit;
}"#;
        let program = crate::parse(source).unwrap();
        let diagnostics = Linter::new(LintConfig::strict())
            .with_source(source)
            .lint(&program);

        let result = apply_fixes(source, &diagnostics, false);

        assert!(result.safe_fixes_applied >= 1);
        assert!(
            result.source.contains("1/4 turns"),
            "Expected 0.25 to be replaced with 1/4"
        );
    }

    #[test]
    fn test_has_safe_fix() {
        let diag_with_safe_fix =
            LintDiagnostic::new("test", "msg", Severity::Warning).with_fix(LintFix {
                start: 0,
                end: 1,
                replacement: "x".to_string(),
                safety: FixSafety::Safe,
            });
        assert!(diag_with_safe_fix.has_safe_fix());
        assert!(diag_with_safe_fix.has_fix());

        let diag_with_unsafe_fix =
            LintDiagnostic::new("test", "msg", Severity::Warning).with_fix(LintFix {
                start: 0,
                end: 1,
                replacement: "x".to_string(),
                safety: FixSafety::Unsafe,
            });
        assert!(!diag_with_unsafe_fix.has_safe_fix());
        assert!(diag_with_unsafe_fix.has_fix());

        let diag_no_fix = LintDiagnostic::new("test", "msg", Severity::Warning);
        assert!(!diag_no_fix.has_safe_fix());
        assert!(!diag_no_fix.has_fix());
    }

    #[test]
    fn test_no_fixes_to_apply() {
        let source = r#"fn main() -> unit {
    return unit;
}"#;
        let program = crate::parse(source).unwrap();
        let diagnostics = Linter::new(LintConfig::strict())
            .with_source(source)
            .lint(&program);

        let result = apply_fixes(source, &diagnostics, false);

        assert_eq!(result.safe_fixes_applied, 0);
        assert_eq!(result.unsafe_fixes_applied, 0);
    }
}
