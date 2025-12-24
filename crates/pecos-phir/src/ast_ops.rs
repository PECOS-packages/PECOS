//! AST-like operations for source language representation
//!
//! Inspired by pliron's approach: operations are just data with traits for behavior

use std::fmt;
use crate::{
    Attribute, Identifier, Type, Value, Region, Operation,
    OpImpl, Verify, OperationLike,
};

/// Macro to define AST operations with minimal boilerplate
/// Inspired by pliron's def_op!
macro_rules! def_ast_op {
    (
        $name:ident {
            $($field:ident : $type:ty),* $(,)?
        }
        $(regions: $num_regions:expr)?
        $(operands: $num_operands:expr)?
        $(results: $num_results:expr)?
    ) => {
        #[derive(Debug, Clone)]
        pub struct $name {
            $(pub $field: $type,)*
            pub regions: Vec<Region>,
            pub operands: Vec<Value>,
            pub results: Vec<Type>,
            pub attributes: Attributes,
        }

        impl $name {
            pub fn new($($field: $type),*) -> Self {
                Self {
                    $($field,)*
                    regions: vec![Region::new(); def_ast_op!(@count_regions $($num_regions)?)],
                    operands: vec![],
                    results: vec![],
                    attributes: Attributes::new(),
                }
            }
        }

        impl Operation for $name {
            fn name(&self) -> &'static str {
                concat!("ast.", stringify!($name))
            }

            fn regions(&self) -> &[Region] {
                &self.regions
            }

            fn regions_mut(&mut self) -> &mut Vec<Region> {
                &mut self.regions
            }

            fn operands(&self) -> &[Value] {
                &self.operands
            }

            fn results(&self) -> &[Type] {
                &self.results
            }

            fn attributes(&self) -> &Attributes {
                &self.attributes
            }
        }
    };

    (@count_regions) => { 0 };
    (@count_regions $n:expr) => { $n };
}

/// Common attributes storage
#[derive(Debug, Clone, Default)]
pub struct Attributes(std::collections::BTreeMap<String, Attribute>);

impl Attributes {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<Attribute>) {
        self.0.insert(key.into(), value.into());
    }

    pub fn get(&self, key: &str) -> Option<&Attribute> {
        self.0.get(key)
    }
}

// ============================================================================
// Parse Dialect - AST-like operations for source language capture
// ============================================================================

/// Unresolved variable reference (before name resolution)
def_ast_op! {
    UnresolvedRef {
        name: String,
        scope_hint: Option<String>,
    }
    results: 1
}

/// Variable declaration with optional type and initializer
def_ast_op! {
    VarDecl {
        name: String,
        type_expr: Option<String>,  // Type as string before resolution
    }
    regions: 1  // Initializer expression
    results: 1
}

/// For loop with init, condition, update, and body
def_ast_op! {
    ForLoop {
        // No fields - everything is in regions
    }
    regions: 4  // init, condition, update, body
}

/// If-else statement
def_ast_op! {
    IfElse {
        // No fields - condition, then, else are regions
    }
    regions: 3  // condition, then, else
}

/// Function definition
def_ast_op! {
    FunctionDef {
        name: String,
    }
    regions: 2  // parameters, body
}

/// Function call (unresolved)
def_ast_op! {
    UnresolvedCall {
        name: String,
    }
    operands: 1  // variable number via operands vec
}

// ============================================================================
// Source-Specific Dialects
// ============================================================================

/// QASM-specific operations
pub mod qasm3 {
    use super::*;

    def_ast_op! {
        GateCall {
            gate: String,
            // Qubits as strings before resolution
        }
        operands: 1  // Will be populated with qubit refs
    }

    def_ast_op! {
        QregDecl {
            name: String,
            size: u32,
        }
    }

    def_ast_op! {
        ForRange {
            var: String,
            start: i32,
            stop: i32,
            step: i32,
        }
        regions: 1  // body
    }

    impl GateCall {
        pub fn with_qubits(mut self, qubits: Vec<String>) -> Self {
            // Store qubit names in attributes before resolution
            self.attributes.insert("qubit_names", qubits);
            self
        }
    }
}

/// Guppy-specific operations
pub mod guppy {
    use super::*;

    def_ast_op! {
        ListComp {
            target: String,
        }
        regions: 3  // element expr, iterator, filter (optional)
    }

    def_ast_op! {
        TupleAssign {
            // Target names stored in attributes
        }
        regions: 1  // RHS expression
    }

    def_ast_op! {
        Decorator {
            name: String,
        }
        regions: 1  // decorated item
    }

    impl TupleAssign {
        pub fn with_targets(mut self, targets: Vec<String>) -> Self {
            self.attributes.insert("targets", targets);
            self
        }
    }
}

/// HUGR-specific operations
pub mod hugr {
    use super::*;

    def_ast_op! {
        Node {
            node_id: String,
            op_type: String,
        }
        operands: 1  // Variable inputs
        results: 1   // Variable outputs
    }

    def_ast_op! {
        Edge {
            source: String,
            target: String,
        }
    }

    def_ast_op! {
        FuncDefn {
            signature: String,  // Type signature as string
        }
        regions: 1  // body containing nodes
    }
}

// ============================================================================
// Builder API - Inspired by pliron's approach
// ============================================================================

pub struct AstBuilder {
    current_region: Vec<Box<dyn Operation>>,
    region_stack: Vec<Vec<Box<dyn Operation>>>,
}

impl AstBuilder {
    pub fn new() -> Self {
        Self {
            current_region: vec![],
            region_stack: vec![],
        }
    }

    /// Build a for loop with closures for each region
    pub fn for_loop(
        &mut self,
        init: impl FnOnce(&mut Self),
        cond: impl FnOnce(&mut Self) -> Value,
        update: impl FnOnce(&mut Self),
        body: impl FnOnce(&mut Self),
    ) -> &mut Self {
        let mut for_op = ForLoop::new();

        // Build each region
        self.with_region(|b| init(b));
        for_op.regions[0] = self.take_region();

        self.with_region(|b| { cond(b); });
        for_op.regions[1] = self.take_region();

        self.with_region(|b| update(b));
        for_op.regions[2] = self.take_region();

        self.with_region(|b| body(b));
        for_op.regions[3] = self.take_region();

        self.push_op(for_op);
        self
    }

    /// Build variable declaration
    pub fn var_decl(&mut self, name: &str, type_expr: Option<&str>) -> Value {
        let mut op = VarDecl::new(
            name.to_string(),
            type_expr.map(|s| s.to_string()),
        );

        let value = Value::new_ssa();
        op.results = vec![Type::Unknown]; // Will be resolved later

        self.push_op(op);
        value
    }

    /// Build unresolved reference
    pub fn var_ref(&mut self, name: &str) -> Value {
        let mut op = UnresolvedRef::new(
            name.to_string(),
            None, // Could add scope hint
        );

        let value = Value::new_ssa();
        op.results = vec![Type::Unknown];

        self.push_op(op);
        value
    }

    /// QASM-specific: gate call
    pub fn qasm_gate(&mut self, gate: &str, qubits: Vec<&str>) -> &mut Self {
        let op = qasm3::GateCall::new(gate.to_string())
            .with_qubits(qubits.into_iter().map(|s| s.to_string()).collect());

        self.push_op(op);
        self
    }

    /// Guppy-specific: list comprehension
    pub fn guppy_list_comp(
        &mut self,
        target: &str,
        element: impl FnOnce(&mut Self) -> Value,
        iterator: impl FnOnce(&mut Self) -> Value,
    ) -> Value {
        let mut op = guppy::ListComp::new(target.to_string());

        // Build element expression
        self.with_region(|b| { element(b); });
        op.regions[0] = self.take_region();

        // Build iterator
        self.with_region(|b| { iterator(b); });
        op.regions[1] = self.take_region();

        let value = Value::new_ssa();
        op.results = vec![Type::Unknown]; // List type

        self.push_op(op);
        value
    }

    // Internal helpers
    fn push_op(&mut self, op: impl Operation + 'static) {
        self.current_region.push(Box::new(op));
    }

    fn with_region(&mut self, f: impl FnOnce(&mut Self)) {
        self.region_stack.push(std::mem::take(&mut self.current_region));
        f(self);
    }

    fn take_region(&mut self) -> Region {
        let ops = std::mem::take(&mut self.current_region);
        self.current_region = self.region_stack.pop().unwrap_or_default();
        Region::from_ops(ops)
    }
}

// ============================================================================
// Lowering Infrastructure - Pattern-based like pliron
// ============================================================================

pub trait LoweringPattern {
    fn matches(&self, op: &dyn Operation) -> bool;
    fn rewrite(&self, op: &dyn Operation) -> Box<dyn Operation>;
}

pub struct ResolveNames {
    symbol_table: SymbolTable,
}

impl LoweringPattern for ResolveNames {
    fn matches(&self, op: &dyn Operation) -> bool {
        op.name() == "ast.UnresolvedRef"
    }

    fn rewrite(&self, op: &dyn Operation) -> Box<dyn Operation> {
        let unresolved = op.downcast_ref::<UnresolvedRef>().unwrap();
        let symbol = self.symbol_table.lookup(&unresolved.name).unwrap();

        // Create resolved reference
        Box::new(ValueRef {
            value: symbol.value,
            attributes: op.attributes().clone(),
        })
    }
}

// ============================================================================
// Usage Example
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ast_building() {
        let mut builder = AstBuilder::new();

        // Build a simple for loop AST
        builder.for_loop(
            |b| { b.var_decl("i", Some("int")); },
            |b| b.var_ref("i"), // Would actually build comparison
            |b| { /* i++ */ },
            |b| {
                b.qasm_gate("h", vec!["q[i]"]);
            }
        );

        // Build Guppy list comprehension
        let results = builder.guppy_list_comp(
            "results",
            |b| b.var_ref("x"),
            |b| b.var_ref("qubits"),
        );
    }
}

// Type placeholders - would be defined elsewhere
#[derive(Debug, Clone)]
pub struct SymbolTable;
impl SymbolTable {
    pub fn lookup(&self, _name: &str) -> Option<Symbol> { None }
}

#[derive(Debug, Clone)]
pub struct Symbol {
    pub value: Value,
}

#[derive(Debug, Clone)]
pub struct ValueRef {
    pub value: Value,
    pub attributes: Attributes,
}

impl Operation for ValueRef {
    fn name(&self) -> &'static str { "core.value_ref" }
    fn regions(&self) -> &[Region] { &[] }
    fn regions_mut(&mut self) -> &mut Vec<Region> { unimplemented!() }
    fn operands(&self) -> &[Value] { &[] }
    fn results(&self) -> &[Type] { &[] }
    fn attributes(&self) -> &Attributes { &self.attributes }
}