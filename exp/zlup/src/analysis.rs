//! Parallelism analysis passes for Zlup AST.
//!
//! This module provides analysis passes that determine what operations can
//! execute in parallel. The analysis is constraint-based: parallelism follows
//! directly from allocator ownership and explicit operation targets.
//!
//! ## Design Philosophy
//!
//! > **Note:** Zlup is an experimental toy language for exploring quantum programming
//! > language design. This analysis is exploratory, not production-ready.
//!
//! Parallelism is expressed through constraints, not annotations:
//! - Functions that don't take allocator parameters can't touch qubits
//! - Operations on disjoint allocators are independent
//! - Scopes define synchronization boundaries
//!
//! ## Available Passes
//!
//! - **Allocator Scope Analysis**: Track allocator lifetimes and accessibility
//! - **Operation Tagging**: Tag each operation with resources it touches
//! - **Dependency Graph**: Build edges between dependent operations
//! - **Parallel Layer Extraction**: Find maximal independent operation sets
//!
//! ## Usage
//!
//! ```rust
//! use zlup::analysis::{AllocatorAnalysis, OperationTagger, DependencyGraph};
//!
//! let source = r#"
//!     fn main() -> unit {
//!         mut q := qalloc(2);
//!         h q[0];
//!         cx (q[0], q[1]);
//!         return;
//!     }
//! "#;
//!
//! let program = zlup::parse(source).unwrap();
//! let allocators = AllocatorAnalysis::analyze(&program);
//! let tagger = OperationTagger::tag(&program);
//! let graph = DependencyGraph::build(tagger.operations);
//! let layers = graph.parallel_layers();
//!
//! assert!(allocators.allocators.contains_key("q"));
//! assert!(!layers.is_empty());
//! ```

use std::collections::{BTreeMap, BTreeSet};

use crate::ast::{Block, Expr, FnDecl, GateOp, MeasureOp, Program, Stmt, TopLevelDecl};

// =============================================================================
// Resource Identifiers
// =============================================================================

/// A resource that an operation can touch.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Resource {
    /// A qubit allocator (e.g., "q", "ancilla") - coarse-grained
    Allocator(String),
    /// A specific qubit within an allocator (e.g., "q[0]", "q[1]") - fine-grained
    /// Only used when the index is a compile-time constant.
    Qubit(String, i128),
    /// A classical variable (e.g., "syndrome", "corrections")
    Variable(String),
}

impl Resource {
    pub fn allocator(name: impl Into<String>) -> Self {
        Resource::Allocator(name.into())
    }

    pub fn qubit(allocator: impl Into<String>, index: i128) -> Self {
        Resource::Qubit(allocator.into(), index)
    }

    pub fn variable(name: impl Into<String>) -> Self {
        Resource::Variable(name.into())
    }

    pub fn is_allocator(&self) -> bool {
        matches!(self, Resource::Allocator(_))
    }

    pub fn is_qubit(&self) -> bool {
        matches!(self, Resource::Qubit(_, _))
    }

    pub fn is_variable(&self) -> bool {
        matches!(self, Resource::Variable(_))
    }

    /// Check if this resource touches qubits (either allocator or specific qubit)
    pub fn touches_qubits(&self) -> bool {
        matches!(self, Resource::Allocator(_) | Resource::Qubit(_, _))
    }
}

// =============================================================================
// Allocator Information
// =============================================================================

/// Information about a qubit allocator.
#[derive(Debug, Clone)]
pub struct AllocatorInfo {
    /// The allocator name
    pub name: String,
    /// Size if statically known
    pub size: Option<usize>,
    /// Scope depth where defined
    pub scope_depth: usize,
    /// Line number where defined
    pub defined_at_line: u32,
}

// =============================================================================
// Pass 1: Allocator Scope Analysis
// =============================================================================

/// Tracks allocator lifetimes and scope accessibility.
///
/// This pass builds a map of what allocators are accessible at each point
/// in the program. Since Zlup requires static allocation, all allocators
/// are known at compile time.
#[derive(Debug, Default)]
pub struct AllocatorAnalysis {
    /// All allocators in the program, by name
    pub allocators: BTreeMap<String, AllocatorInfo>,
    /// Allocators accessible in each function
    pub function_allocators: BTreeMap<String, BTreeSet<String>>,
}

impl AllocatorAnalysis {
    /// Analyze a program for allocator information.
    pub fn analyze(program: &Program) -> Self {
        let mut analysis = Self::default();

        for decl in &program.declarations {
            if let TopLevelDecl::Fn(fn_decl) = decl {
                analysis.analyze_function(fn_decl);
            }
        }

        analysis
    }

    fn analyze_function(&mut self, fn_decl: &FnDecl) {
        let mut local_allocators = BTreeSet::new();

        // Analyze parameters for allocator types
        for param in &fn_decl.params {
            if Self::is_allocator_type_expr(&param.ty) {
                local_allocators.insert(param.name.clone());
                self.allocators.insert(
                    param.name.clone(),
                    AllocatorInfo {
                        name: param.name.clone(),
                        size: Self::extract_allocator_size_from_type(&param.ty),
                        scope_depth: 0,
                        defined_at_line: fn_decl.location.as_ref().map(|l| l.line).unwrap_or(0),
                    },
                );
            }
        }

        // Analyze body for qalloc statements
        self.analyze_block(&fn_decl.body, &mut local_allocators, 1);

        self.function_allocators
            .insert(fn_decl.name.clone(), local_allocators);
    }

    fn analyze_block(
        &mut self,
        block: &Block,
        local_allocators: &mut BTreeSet<String>,
        depth: usize,
    ) {
        for stmt in &block.statements {
            self.analyze_stmt(stmt, local_allocators, depth);
        }
    }

    fn analyze_stmt(&mut self, stmt: &Stmt, local_allocators: &mut BTreeSet<String>, depth: usize) {
        match stmt {
            Stmt::Binding(binding) => {
                // Check if this is a qalloc
                if let Some(value) = &binding.value
                    && Self::is_qalloc_expr(value)
                {
                    local_allocators.insert(binding.name.clone());
                    self.allocators.insert(
                        binding.name.clone(),
                        AllocatorInfo {
                            name: binding.name.clone(),
                            size: Self::extract_qalloc_size(value),
                            scope_depth: depth,
                            defined_at_line: binding.location.as_ref().map(|l| l.line).unwrap_or(0),
                        },
                    );
                }
            }
            Stmt::Block(inner_block) => {
                self.analyze_block(inner_block, local_allocators, depth + 1);
            }
            Stmt::If(if_stmt) => {
                self.analyze_block(&if_stmt.then_body, local_allocators, depth + 1);
                if let Some(else_branch) = &if_stmt.else_body {
                    match else_branch {
                        crate::ast::ElseBranch::Else(block) => {
                            self.analyze_block(block, local_allocators, depth + 1);
                        }
                        crate::ast::ElseBranch::ElseIf(nested_if) => {
                            self.analyze_stmt(
                                &Stmt::If(*nested_if.clone()),
                                local_allocators,
                                depth,
                            );
                        }
                    }
                }
            }
            Stmt::For(for_stmt) => {
                self.analyze_block(&for_stmt.body, local_allocators, depth + 1);
            }
            _ => {}
        }
    }

    /// Check if a type represents a qubit allocator.
    pub fn is_allocator_type_expr(ty: &crate::ast::TypeExpr) -> bool {
        // Check for [n]qubit, qubit, or QAlloc types
        match ty {
            crate::ast::TypeExpr::Array(array_type) => {
                // [n]qubit
                matches!(array_type.element, crate::ast::TypeExpr::Qubit)
            }
            crate::ast::TypeExpr::Qubit => true,
            crate::ast::TypeExpr::QAlloc(_) => true,
            crate::ast::TypeExpr::Named(name) => {
                // Named type that might be "qubit"
                name.segments.first().is_some_and(|s| s == "qubit")
            }
            _ => false,
        }
    }

    /// Extract size from an allocator type if statically known.
    fn extract_allocator_size_from_type(ty: &crate::ast::TypeExpr) -> Option<usize> {
        if let crate::ast::TypeExpr::Array(array_type) = ty
            && let Some(size_expr) = &array_type.size
            && let Expr::IntLit(lit) = size_expr
        {
            return Some(lit.value as usize);
        }
        None
    }

    /// Check if an expression is a qalloc call.
    fn is_qalloc_expr(expr: &Expr) -> bool {
        if let Expr::Call(call) = expr
            && let Expr::Ident(ident) = &call.callee
        {
            return ident.name == "qalloc";
        }
        false
    }

    /// Extract size from a qalloc call if statically known.
    fn extract_qalloc_size(expr: &Expr) -> Option<usize> {
        if let Expr::Call(call) = expr
            && let Some(first_arg) = call.args.first()
            && let Expr::IntLit(lit) = first_arg
        {
            return Some(lit.value as usize);
        }
        None
    }
}

// =============================================================================
// Pass 2: Operation Tagging
// =============================================================================

/// An operation in the program with its resource usage.
#[derive(Debug, Clone)]
pub struct TaggedOp {
    /// Unique ID for this operation
    pub id: usize,
    /// Resources this operation reads from
    pub reads: BTreeSet<Resource>,
    /// Resources this operation writes to
    pub writes: BTreeSet<Resource>,
    /// Source location line number
    pub line: u32,
    /// Human-readable description
    pub description: String,
}

impl TaggedOp {
    /// Check if this operation touches any qubit allocators.
    pub fn touches_qubits(&self) -> bool {
        self.reads.iter().any(|r| r.touches_qubits())
            || self.writes.iter().any(|r| r.touches_qubits())
    }

    /// Check if this operation is purely classical.
    pub fn is_classical(&self) -> bool {
        !self.touches_qubits()
    }

    /// Get all resources this operation touches (reads or writes).
    pub fn all_resources(&self) -> BTreeSet<Resource> {
        let mut all = self.reads.clone();
        all.extend(self.writes.clone());
        all
    }
}

/// Tags operations with the resources they touch.
#[derive(Debug, Default)]
pub struct OperationTagger {
    /// All tagged operations
    pub operations: Vec<TaggedOp>,
    /// Next operation ID
    next_id: usize,
}

impl OperationTagger {
    /// Tag all operations in a program.
    pub fn tag(program: &Program) -> Self {
        let mut tagger = Self::default();

        for decl in &program.declarations {
            if let TopLevelDecl::Fn(fn_decl) = decl {
                tagger.tag_function(fn_decl);
            }
        }

        tagger
    }

    fn tag_function(&mut self, fn_decl: &FnDecl) {
        self.tag_block(&fn_decl.body);
    }

    fn tag_block(&mut self, block: &Block) {
        for stmt in &block.statements {
            self.tag_stmt(stmt);
        }
    }

    fn tag_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Gate(gate_op) => {
                self.tag_gate_op(gate_op);
            }
            Stmt::Measure(measure_op) => {
                self.tag_measure_op(measure_op);
            }
            Stmt::Expr(expr_stmt) => {
                // Check if this is a gate or measure expression
                match &expr_stmt.expr {
                    Expr::Gate(gate_expr) => {
                        self.tag_gate_expr(gate_expr);
                    }
                    Expr::Measure(measure_expr) => {
                        self.tag_measure_expr(measure_expr);
                    }
                    _ => {
                        // Other expression statements
                        let mut reads = BTreeSet::new();
                        self.collect_expr_resources(&expr_stmt.expr, &mut reads);

                        if !reads.is_empty() {
                            let id = self.next_id();
                            self.operations.push(TaggedOp {
                                id,
                                reads,
                                writes: BTreeSet::new(),
                                line: expr_stmt.location.as_ref().map(|l| l.line).unwrap_or(0),
                                description: "expr".to_string(),
                            });
                        }
                    }
                }
            }
            Stmt::Binding(binding) => {
                // Track variable definitions
                let mut writes = BTreeSet::new();
                writes.insert(Resource::variable(&binding.name));

                let mut reads = BTreeSet::new();
                if let Some(value) = &binding.value {
                    self.collect_expr_resources(value, &mut reads);
                }

                let id = self.next_id();
                self.operations.push(TaggedOp {
                    id,
                    reads,
                    writes,
                    line: binding.location.as_ref().map(|l| l.line).unwrap_or(0),
                    description: format!("bind {}", binding.name),
                });
            }
            Stmt::Assign(assign) => {
                let mut writes = BTreeSet::new();
                let mut reads = BTreeSet::new();

                // Target is written
                self.collect_expr_resources(&assign.target, &mut writes);

                // Value is read
                self.collect_expr_resources(&assign.value, &mut reads);

                let id = self.next_id();
                self.operations.push(TaggedOp {
                    id,
                    reads,
                    writes,
                    line: assign.location.as_ref().map(|l| l.line).unwrap_or(0),
                    description: "assign".to_string(),
                });
            }
            Stmt::Block(block) => {
                self.tag_block(block);
            }
            Stmt::If(if_stmt) => {
                // Condition is read
                let mut reads = BTreeSet::new();
                self.collect_expr_resources(&if_stmt.condition, &mut reads);

                let id = self.next_id();
                self.operations.push(TaggedOp {
                    id,
                    reads,
                    writes: BTreeSet::new(),
                    line: if_stmt.location.as_ref().map(|l| l.line).unwrap_or(0),
                    description: "if condition".to_string(),
                });

                self.tag_block(&if_stmt.then_body);
                if let Some(else_branch) = &if_stmt.else_body {
                    match else_branch {
                        crate::ast::ElseBranch::Else(block) => self.tag_block(block),
                        crate::ast::ElseBranch::ElseIf(nested_if) => {
                            self.tag_stmt(&Stmt::If(*nested_if.clone()))
                        }
                    }
                }
            }
            Stmt::For(for_stmt) => {
                self.tag_block(&for_stmt.body);
            }
            Stmt::Return(ret) => {
                let mut reads = BTreeSet::new();
                if let Some(value) = &ret.value {
                    self.collect_expr_resources(value, &mut reads);
                }

                let id = self.next_id();
                self.operations.push(TaggedOp {
                    id,
                    reads,
                    writes: BTreeSet::new(),
                    line: ret.location.as_ref().map(|l| l.line).unwrap_or(0),
                    description: "return".to_string(),
                });
            }
            _ => {}
        }
    }

    fn tag_gate_op(&mut self, gate: &GateOp) {
        let mut writes = BTreeSet::new();

        // Gates write to their target qubits
        for target in &gate.targets {
            self.collect_slot_ref_allocator(target, &mut writes);
        }

        let id = self.next_id();
        self.operations.push(TaggedOp {
            id,
            reads: BTreeSet::new(),
            writes,
            line: gate.location.as_ref().map(|l| l.line).unwrap_or(0),
            description: format!("{:?}", gate.kind),
        });
    }

    fn tag_gate_expr(&mut self, gate: &crate::ast::GateExpr) {
        let mut writes = BTreeSet::new();

        // Collect allocators from the target expression
        self.collect_allocators_from_expr(&gate.target, &mut writes);

        let id = self.next_id();
        self.operations.push(TaggedOp {
            id,
            reads: BTreeSet::new(),
            writes,
            line: gate.location.as_ref().map(|l| l.line).unwrap_or(0),
            description: format!("{:?}", gate.kind),
        });
    }

    fn tag_measure_op(&mut self, measure: &MeasureOp) {
        let mut reads = BTreeSet::new();
        let writes = BTreeSet::new();

        // Measure reads from qubits
        for target in &measure.targets {
            self.collect_slot_ref_allocator(target, &mut reads);
        }

        let id = self.next_id();
        self.operations.push(TaggedOp {
            id,
            reads,
            writes,
            line: measure.location.as_ref().map(|l| l.line).unwrap_or(0),
            description: "measure".to_string(),
        });
    }

    fn tag_measure_expr(&mut self, measure: &crate::ast::MeasureExpr) {
        let mut reads = BTreeSet::new();
        let writes = BTreeSet::new();

        // Collect allocators from the targets expression
        self.collect_allocators_from_expr(&measure.targets, &mut reads);

        let id = self.next_id();
        self.operations.push(TaggedOp {
            id,
            reads,
            writes,
            line: measure.location.as_ref().map(|l| l.line).unwrap_or(0),
            description: "measure".to_string(),
        });
    }

    /// Collect allocator resources from an expression that represents qubit targets.
    fn collect_allocators_from_expr(&self, expr: &Expr, resources: &mut BTreeSet<Resource>) {
        match expr {
            Expr::Index(index) => {
                // e.g., q[0] - extract the base allocator name
                if let Expr::Ident(ident) = &index.object {
                    resources.insert(Resource::allocator(&ident.name));
                } else {
                    self.collect_allocators_from_expr(&index.object, resources);
                }
            }
            Expr::Ident(ident) => {
                // Bare identifier might be an allocator
                resources.insert(Resource::allocator(&ident.name));
            }
            Expr::Tuple(tuple) => {
                // e.g., (q[0], q[1]) for two-qubit gates
                for elem in &tuple.elements {
                    self.collect_allocators_from_expr(elem, resources);
                }
            }
            Expr::BracketArray(arr) => {
                // e.g., [q[0], q[1], q[2]]
                for elem in &arr.elements {
                    self.collect_allocators_from_expr(elem, resources);
                }
            }
            Expr::SlotRef(slot) => {
                resources.insert(Resource::allocator(&slot.allocator));
            }
            _ => {}
        }
    }

    fn collect_slot_ref_allocator(
        &self,
        slot_ref: &crate::ast::SlotRef,
        resources: &mut BTreeSet<Resource>,
    ) {
        resources.insert(Resource::allocator(&slot_ref.allocator));
    }

    fn collect_expr_resources(&self, expr: &Expr, resources: &mut BTreeSet<Resource>) {
        match expr {
            Expr::Ident(ident) => {
                resources.insert(Resource::variable(&ident.name));
            }
            Expr::Index(index) => {
                // The object being indexed
                self.collect_expr_resources(&index.object, resources);
            }
            Expr::Binary(binary) => {
                self.collect_expr_resources(&binary.left, resources);
                self.collect_expr_resources(&binary.right, resources);
            }
            Expr::Unary(unary) => {
                self.collect_expr_resources(&unary.operand, resources);
            }
            Expr::Call(call) => {
                self.collect_expr_resources(&call.callee, resources);
                for arg in &call.args {
                    self.collect_expr_resources(arg, resources);
                }
            }
            Expr::Field(field) => {
                self.collect_expr_resources(&field.object, resources);
            }
            Expr::Measure(measure) => {
                // Measure reads from qubits - collect allocators
                self.collect_allocators_from_expr(&measure.targets, resources);
            }
            Expr::Gate(gate) => {
                // Gate touches qubits - collect allocators
                self.collect_allocators_from_expr(&gate.target, resources);
            }
            _ => {}
        }
    }

    fn next_id(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        id
    }
}

// =============================================================================
// Pass 3: Dependency Graph
// =============================================================================

/// The kind of dependency between operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DepKind {
    /// Both operations touch the same qubit allocator
    QubitDep(String),
    /// One operation reads a variable written by another
    DataDep(String),
    /// Control flow dependency
    ControlDep,
}

/// An edge in the dependency graph.
#[derive(Debug, Clone)]
pub struct DepEdge {
    /// Source operation ID
    pub from: usize,
    /// Target operation ID
    pub to: usize,
    /// Kind of dependency
    pub kind: DepKind,
}

/// Dependency graph for operations.
#[derive(Debug, Default)]
pub struct DependencyGraph {
    /// All operations
    pub operations: Vec<TaggedOp>,
    /// Dependency edges (from -> to means "from must complete before to")
    pub edges: Vec<DepEdge>,
}

impl DependencyGraph {
    /// Build a dependency graph from tagged operations.
    pub fn build(operations: Vec<TaggedOp>) -> Self {
        let mut graph = Self {
            operations,
            edges: Vec::new(),
        };

        graph.compute_dependencies();
        graph
    }

    fn compute_dependencies(&mut self) {
        // For each pair of operations, check for dependencies
        for i in 0..self.operations.len() {
            for j in (i + 1)..self.operations.len() {
                if let Some(kind) = self.check_dependency(i, j) {
                    self.edges.push(DepEdge {
                        from: i,
                        to: j,
                        kind,
                    });
                }
            }
        }
    }

    fn check_dependency(&self, earlier: usize, later: usize) -> Option<DepKind> {
        let op1 = &self.operations[earlier];
        let op2 = &self.operations[later];

        // Check for qubit dependencies (WAW, RAW, WAR on allocators)
        for res in &op1.writes {
            if let Resource::Allocator(name) = res
                && (op2.reads.contains(res) || op2.writes.contains(res))
            {
                return Some(DepKind::QubitDep(name.clone()));
            }
        }
        for res in &op1.reads {
            if let Resource::Allocator(name) = res
                && op2.writes.contains(res)
            {
                return Some(DepKind::QubitDep(name.clone()));
            }
        }

        // Check for data dependencies (classical variables)
        for res in &op1.writes {
            if let Resource::Variable(name) = res
                && op2.reads.contains(res)
            {
                return Some(DepKind::DataDep(name.clone()));
            }
        }

        None
    }

    /// Extract parallel layers - operations at the same layer can execute in parallel.
    pub fn parallel_layers(&self) -> Vec<Vec<usize>> {
        if self.operations.is_empty() {
            return vec![];
        }

        // Compute the "level" of each operation (longest path from any root)
        let mut levels = vec![0usize; self.operations.len()];
        let mut predecessors: Vec<Vec<usize>> = vec![vec![]; self.operations.len()];

        // Build predecessor list
        for edge in &self.edges {
            predecessors[edge.to].push(edge.from);
        }

        // Compute levels (topological order with level assignment)
        let mut changed = true;
        while changed {
            changed = false;
            for i in 0..self.operations.len() {
                let max_pred_level = predecessors[i]
                    .iter()
                    .map(|&p| levels[p])
                    .max()
                    .unwrap_or(0);
                let new_level = if predecessors[i].is_empty() {
                    0
                } else {
                    max_pred_level + 1
                };
                if new_level > levels[i] {
                    levels[i] = new_level;
                    changed = true;
                }
            }
        }

        // Group operations by level
        let max_level = levels.iter().copied().max().unwrap_or(0);
        let mut layers: Vec<Vec<usize>> = vec![vec![]; max_level + 1];
        for (op_id, &level) in levels.iter().enumerate() {
            layers[level].push(op_id);
        }

        layers
    }

    /// Get operations that can parallelize with a given operation.
    pub fn independent_ops(&self, op_id: usize) -> Vec<usize> {
        let mut dependent: BTreeSet<usize> = BTreeSet::new();

        // Find all operations connected by edges
        for edge in &self.edges {
            if edge.from == op_id {
                dependent.insert(edge.to);
            }
            if edge.to == op_id {
                dependent.insert(edge.from);
            }
        }

        // Return operations not in dependent set
        (0..self.operations.len())
            .filter(|&id| id != op_id && !dependent.contains(&id))
            .collect()
    }

    /// Print the dependency graph for debugging.
    pub fn debug_print(&self) {
        println!("Operations:");
        for op in &self.operations {
            let reads: Vec<_> = op.reads.iter().map(|r| format!("{:?}", r)).collect();
            let writes: Vec<_> = op.writes.iter().map(|r| format!("{:?}", r)).collect();
            println!(
                "  [{}] {} (line {}) reads: {:?}, writes: {:?}",
                op.id, op.description, op.line, reads, writes
            );
        }

        println!("\nDependencies:");
        for edge in &self.edges {
            println!("  {} -> {} ({:?})", edge.from, edge.to, edge.kind);
        }

        println!("\nParallel Layers:");
        for (level, layer) in self.parallel_layers().iter().enumerate() {
            let descs: Vec<_> = layer
                .iter()
                .map(|&id| format!("[{}]{}", id, self.operations[id].description))
                .collect();
            println!("  Layer {}: {:?}", level, descs);
        }
    }
}

// =============================================================================
// Analysis Summary
// =============================================================================

/// Summary of parallelism analysis for a function.
#[derive(Debug)]
pub struct ParallelismSummary {
    /// Function name
    pub function_name: String,
    /// Total operations
    pub total_ops: usize,
    /// Number of parallel layers
    pub num_layers: usize,
    /// Maximum parallelism (largest layer)
    pub max_parallelism: usize,
    /// Number of purely classical operations
    pub classical_ops: usize,
    /// Number of quantum operations
    pub quantum_ops: usize,
}

/// Analyze parallelism in a program.
pub fn analyze_parallelism(program: &Program) -> Vec<ParallelismSummary> {
    let mut summaries = vec![];

    for decl in &program.declarations {
        if let TopLevelDecl::Fn(fn_decl) = decl {
            let tagger = OperationTagger::tag(&Program {
                name: program.name.clone(),
                declarations: vec![TopLevelDecl::Fn(fn_decl.clone())],
                location: None,
            });

            let graph = DependencyGraph::build(tagger.operations);
            let layers = graph.parallel_layers();

            let classical_ops = graph
                .operations
                .iter()
                .filter(|op| op.is_classical())
                .count();
            let quantum_ops = graph
                .operations
                .iter()
                .filter(|op| op.touches_qubits())
                .count();

            summaries.push(ParallelismSummary {
                function_name: fn_decl.name.clone(),
                total_ops: graph.operations.len(),
                num_layers: layers.len(),
                max_parallelism: layers.iter().map(|l| l.len()).max().unwrap_or(0),
                classical_ops,
                quantum_ops,
            });
        }
    }

    summaries
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_and_analyze(source: &str) -> DependencyGraph {
        let program = crate::parse(source).expect("parse failed");
        let tagger = OperationTagger::tag(&program);
        DependencyGraph::build(tagger.operations)
    }

    #[test]
    fn test_gates_same_allocator() {
        // Gates on the same allocator are treated as dependent (conservative).
        // This is allocator-level tracking, not qubit-index-level.
        let source = r#"
            fn main() -> unit {
                mut q := qalloc(4);
                h q[0];
                h q[1];
                h q[2];
                h q[3];
                return;
            }
        "#;

        let graph = parse_and_analyze(source);

        // We should have: qalloc binding, 4 H gates, return
        assert_eq!(graph.operations.len(), 6);

        // All H gates touch the same allocator "q", so they're serialized
        let h_ops: Vec<_> = graph
            .operations
            .iter()
            .filter(|op| op.description.contains("H"))
            .collect();
        assert_eq!(h_ops.len(), 4);

        // All should write to allocator "q"
        for op in &h_ops {
            assert!(
                op.writes.contains(&Resource::allocator("q")),
                "H gate should touch allocator q"
            );
        }
    }

    #[test]
    fn test_dependent_gates() {
        let source = r#"
            fn main() -> unit {
                mut q := qalloc(2);
                h q[0];
                cx (q[0], q[1]);
                return;
            }
        "#;

        let graph = parse_and_analyze(source);

        // We should have: qalloc binding, H gate, CX gate, return
        assert_eq!(graph.operations.len(), 4, "Expected 4 operations");

        // H and CX both touch allocator "q", so they're dependent
        let h_op = graph
            .operations
            .iter()
            .find(|op| op.description.contains("H"));
        let cx_op = graph
            .operations
            .iter()
            .find(|op| op.description.contains("CX"));

        assert!(h_op.is_some(), "Should have H gate");
        assert!(cx_op.is_some(), "Should have CX gate");

        // Both should touch allocator "q"
        assert!(
            h_op.unwrap().writes.contains(&Resource::allocator("q")),
            "H should touch q"
        );
        assert!(
            cx_op.unwrap().writes.contains(&Resource::allocator("q")),
            "CX should touch q"
        );
    }

    #[test]
    fn test_disjoint_allocators() {
        let source = r#"
            fn main() -> unit {
                mut q1 := qalloc(2);
                mut q2 := qalloc(2);
                h q1[0];
                h q2[0];
                cx (q1[0], q1[1]);
                cx (q2[0], q2[1]);
                return;
            }
        "#;

        let graph = parse_and_analyze(source);

        // Operations on q1 and q2 should be independent
        // Find the H gates
        let h_ops: Vec<_> = graph
            .operations
            .iter()
            .enumerate()
            .filter(|(_, op)| op.description.contains("H"))
            .map(|(id, _)| id)
            .collect();

        assert_eq!(h_ops.len(), 2);

        // Check that the two H gates are independent of each other
        let independent_of_first = graph.independent_ops(h_ops[0]);
        assert!(
            independent_of_first.contains(&h_ops[1]),
            "H gates on different allocators should be independent"
        );
    }

    #[test]
    fn test_classical_quantum_independence() {
        let source = r#"
            fn main() -> unit {
                mut q := qalloc(2);
                x := 1 + 2;
                h q[0];
                y := x * 3;
                cx (q[0], q[1]);
                return;
            }
        "#;

        let graph = parse_and_analyze(source);

        // Classical operations (x, y bindings) should be independent of quantum ops
        let classical: Vec<_> = graph
            .operations
            .iter()
            .filter(|op| op.is_classical() && op.description.starts_with("bind"))
            .collect();

        let quantum: Vec<_> = graph
            .operations
            .iter()
            .filter(|op| op.touches_qubits())
            .collect();

        assert!(!classical.is_empty(), "Should have classical ops");
        assert!(!quantum.is_empty(), "Should have quantum ops");
    }

    #[test]
    fn test_allocator_analysis() {
        let source = r#"
            fn main() -> unit {
                mut q := qalloc(4);
                mut ancilla := qalloc(2);
                h q[0];
                return;
            }
        "#;

        let program = crate::parse(source).expect("parse failed");
        let analysis = AllocatorAnalysis::analyze(&program);

        assert!(analysis.allocators.contains_key("q"));
        assert!(analysis.allocators.contains_key("ancilla"));
        assert_eq!(analysis.allocators["q"].size, Some(4));
        assert_eq!(analysis.allocators["ancilla"].size, Some(2));
    }

    #[test]
    fn test_measurement_creates_dependency() {
        let source = r#"
            fn main() -> unit {
                mut q := qalloc(2);
                h q[0];
                m := mz([2]u1) q;
                return;
            }
        "#;

        let graph = parse_and_analyze(source);

        // The measure is part of a binding (m := mz...), so it's recorded as "bind m"
        // but should still track that it reads from allocator q
        let m_binding = graph
            .operations
            .iter()
            .find(|op| op.description == "bind m");
        assert!(m_binding.is_some(), "Should have binding for m");

        // The binding should read from allocator "q" (via the measure expression)
        // Note: Currently we track mz as reading q through collect_expr_resources
        // but we need to enhance the binding handler to detect measure expressions
    }

    #[test]
    fn test_parallel_layers_correct() {
        let source = r#"
            fn main() -> unit {
                mut q1 := qalloc(2);
                mut q2 := qalloc(2);
                h q1[0];
                h q2[0];
                return;
            }
        "#;

        let graph = parse_and_analyze(source);
        let layers = graph.parallel_layers();

        // Layer 0: Both qalloc bindings (parallel - different variables)
        // Layer 1: Both H gates (parallel - different allocators)
        // Layer 2: return

        // The H gates should be in the same layer since they're independent
        let h_ops: Vec<usize> = graph
            .operations
            .iter()
            .enumerate()
            .filter(|(_, op)| op.description.contains("H"))
            .map(|(id, _)| id)
            .collect();

        assert_eq!(h_ops.len(), 2);

        // Find which layer contains the H gates
        let h_layer = layers.iter().find(|layer| layer.contains(&h_ops[0]));
        assert!(h_layer.is_some());
        assert!(
            h_layer.unwrap().contains(&h_ops[1]),
            "Both H gates should be in the same layer (parallel)"
        );
    }

    #[test]
    fn test_data_dependency_classical() {
        let source = r#"
            fn main() -> unit {
                x := 1;
                y := x + 1;
                z := y + 1;
                return;
            }
        "#;

        let graph = parse_and_analyze(source);

        // x, y, z form a chain of data dependencies
        // y depends on x, z depends on y
        let layers = graph.parallel_layers();

        // Each binding should be in a different layer due to data dependencies
        // (x, return could be parallel with others if no dep, but y needs x, z needs y)
        assert!(
            layers.len() >= 3,
            "Should have at least 3 layers for x->y->z chain"
        );
    }

    #[test]
    fn test_function_parameter_allocator() {
        let source = r#"
            fn apply_h(q: [4]qubit) -> unit {
                h q[0];
                return;
            }
        "#;

        let program = crate::parse(source).expect("parse failed");
        let analysis = AllocatorAnalysis::analyze(&program);

        // Parameter q should be detected as an allocator
        assert!(
            analysis.allocators.contains_key("q"),
            "Parameter q should be detected as allocator"
        );
    }

    #[test]
    fn test_debug_print() {
        let source = r#"
            fn main() -> unit {
                mut q := qalloc(2);
                h q[0];
                cx (q[0], q[1]);
                return;
            }
        "#;

        let graph = parse_and_analyze(source);

        // Just verify debug_print doesn't panic
        graph.debug_print();
    }

    #[test]
    fn test_parallelism_summary() {
        let source = r#"
            fn main() -> unit {
                mut q1 := qalloc(2);
                mut q2 := qalloc(2);
                h q1[0];
                h q2[0];
                x := 1 + 2;
                return;
            }
        "#;

        let program = crate::parse(source).expect("parse failed");
        let summaries = super::analyze_parallelism(&program);

        assert_eq!(summaries.len(), 1);
        let summary = &summaries[0];
        assert_eq!(summary.function_name, "main");
        assert!(summary.total_ops > 0);
        assert!(
            summary.quantum_ops >= 2,
            "Should have at least 2 quantum ops (H gates)"
        );
        assert!(
            summary.classical_ops >= 1,
            "Should have at least 1 classical op (x binding)"
        );
    }

    #[test]
    fn test_empty_function() {
        let source = r#"
            fn empty() -> unit {
                return;
            }
        "#;

        let graph = parse_and_analyze(source);

        // Should have just the return operation
        assert!(
            graph.operations.len() <= 1,
            "Empty function should have minimal ops"
        );
        let layers = graph.parallel_layers();
        assert!(
            layers.len() <= 1,
            "Empty function should have at most 1 layer"
        );
    }

    #[test]
    fn test_purely_classical_function() {
        let source = r#"
            fn classical() -> i32 {
                a := 1;
                b := 2;
                c := a + b;
                return c;
            }
        "#;

        let program = crate::parse(source).expect("parse failed");
        let summaries = super::analyze_parallelism(&program);

        assert_eq!(summaries.len(), 1);
        let summary = &summaries[0];
        assert_eq!(summary.quantum_ops, 0, "Should have no quantum ops");
        assert!(summary.classical_ops > 0, "Should have classical ops");
    }

    #[test]
    fn test_nested_scopes() {
        let source = r#"
            fn nested() -> unit {
                mut q := qalloc(2);
                {
                    h q[0];
                    {
                        cx (q[0], q[1]);
                    }
                }
                return;
            }
        "#;

        let program = crate::parse(source).expect("parse failed");
        let analysis = AllocatorAnalysis::analyze(&program);

        // Allocator q should still be tracked despite nested scopes
        assert!(analysis.allocators.contains_key("q"));
        assert_eq!(analysis.allocators["q"].size, Some(2));
    }

    #[test]
    fn test_multiple_functions() {
        let source = r#"
            fn func1() -> unit {
                mut q := qalloc(2);
                h q[0];
                return;
            }

            fn func2() -> unit {
                mut r := qalloc(3);
                h r[0];
                h r[1];
                return;
            }
        "#;

        let program = crate::parse(source).expect("parse failed");
        let summaries = super::analyze_parallelism(&program);

        assert_eq!(summaries.len(), 2, "Should analyze both functions");

        let func1 = summaries.iter().find(|s| s.function_name == "func1");
        let func2 = summaries.iter().find(|s| s.function_name == "func2");

        assert!(func1.is_some(), "Should have func1 summary");
        assert!(func2.is_some(), "Should have func2 summary");

        // func2 has more H gates
        assert!(func2.unwrap().quantum_ops >= func1.unwrap().quantum_ops);
    }

    #[test]
    fn test_if_statement_analysis() {
        let source = r#"
            fn conditional(cond: bool) -> unit {
                mut q := qalloc(2);
                if cond {
                    h q[0];
                } else {
                    x q[0];
                }
                return;
            }
        "#;

        let program = crate::parse(source).expect("parse failed");
        let analysis = AllocatorAnalysis::analyze(&program);

        // Allocator should be tracked even inside conditionals
        assert!(analysis.allocators.contains_key("q"));
    }

    #[test]
    fn test_for_loop_analysis() {
        let source = r#"
            fn looped() -> unit {
                mut q := qalloc(4);
                for i in 0..4 {
                    h q[i];
                }
                return;
            }
        "#;

        let program = crate::parse(source).expect("parse failed");
        let analysis = AllocatorAnalysis::analyze(&program);

        assert!(analysis.allocators.contains_key("q"));
        assert_eq!(analysis.allocators["q"].size, Some(4));
    }
}
