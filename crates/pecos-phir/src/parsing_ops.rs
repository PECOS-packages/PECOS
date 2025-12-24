/*!
Parsing-specific operations for PHIR

These operations help us parse directly to PHIR without needing a separate AST.
They handle forward references, unresolved names, and gradual type checking.
*/

use crate::ops::{SSAValue, ValueRef};
use crate::phir::Region;
use crate::types::Type;
use std::collections::BTreeMap;

/// Parsing-specific operations that get resolved/lowered later
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ParsingOp {
    /// Unresolved function call (before name resolution)
    UnresolvedCall(UnresolvedCall),

    /// Unresolved variable reference
    UnresolvedRef(UnresolvedRef),

    /// Forward declaration placeholder
    ForwardDecl(ForwardDecl),

    /// Implicit cast (inserted during type checking)
    ImplicitCast(ImplicitCast),

    /// High-level for loop (before CFG lowering)
    ForLoop(ForLoop),

    /// High-level if-else (before CFG lowering)
    IfElse(IfElse),

    /// Type to be inferred
    InferType(InferType),
}

/// Unresolved function call
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct UnresolvedCall {
    /// Function name (unresolved)
    pub name: String,
    /// Arguments (may have unresolved types)
    pub args: Vec<ValueRef>,
    /// Expected return type (if known)
    pub expected_type: Option<Type>,
    /// Source location for error reporting
    pub location: crate::error::SourceLocation,
}

/// Unresolved variable/symbol reference
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct UnresolvedRef {
    /// Symbol name
    pub name: String,
    /// Scope hint (local, global, etc.)
    pub scope_hint: ScopeHint,
    /// Expected type (if known from context)
    pub expected_type: Option<Type>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ScopeHint {
    Local,
    Global,
    Function,
    Type,
    Unknown,
}

/// Forward declaration placeholder
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ForwardDecl {
    /// Symbol being forward declared
    pub name: String,
    /// Kind of declaration
    pub kind: DeclKind,
    /// Partial type info (if available)
    pub partial_type: Option<Type>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum DeclKind {
    Function,
    Type,
    Global,
}

/// Implicit cast operation (inserted during type checking)
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ImplicitCast {
    /// Value to cast
    pub value: ValueRef,
    /// Source type
    pub from_type: Type,
    /// Target type
    pub to_type: Type,
    /// Kind of cast
    pub cast_kind: CastKind,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum CastKind {
    /// Numeric widening (i32 -> i64)
    NumericWiden,
    /// Numeric narrowing (i64 -> i32)
    NumericNarrow,
    /// Float to int
    FloatToInt,
    /// Int to float
    IntToFloat,
    /// Array coercion
    ArrayCoercion,
    /// Quantum state preparation
    QuantumPrep,
}

/// High-level for loop (before lowering to CFG)
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ForLoop {
    /// Loop variable
    pub induction_var: String,
    /// Start value
    pub start: ValueRef,
    /// End value (exclusive)
    pub end: ValueRef,
    /// Step value (default 1)
    pub step: Option<ValueRef>,
    /// Loop body region
    pub body: Region,
}

/// High-level if-else (before lowering to CFG)
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct IfElse {
    /// Condition
    pub condition: ValueRef,
    /// Then region
    pub then_region: Region,
    /// Else region (optional)
    pub else_region: Option<Region>,
    /// Phi outputs (values that flow out)
    pub outputs: Vec<Type>,
}

/// Type to be inferred
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InferType {
    /// Type variable ID
    pub type_var: u32,
    /// Constraints on the type
    pub constraints: Vec<TypeConstraint>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum TypeConstraint {
    /// Must be numeric
    Numeric,
    /// Must be quantum
    Quantum,
    /// Must be classical
    Classical,
    /// Must unify with another type
    UnifyWith(Type),
    /// Must be callable with given signature
    Callable(Vec<Type>, Vec<Type>),
}

/// Name resolution context
#[allow(dead_code)]
pub struct NameResolver {
    /// Symbol tables for each scope
    scopes: Vec<SymbolTable>,
    /// Type inference context
    type_context: TypeContext,
    /// Forward declarations waiting to be resolved
    forward_decls: BTreeMap<String, ForwardDecl>,
}

/// Symbol table for a scope
#[allow(dead_code)]
pub struct SymbolTable {
    /// Symbols in this scope
    symbols: BTreeMap<String, Symbol>,
    /// Parent scope (if any)
    parent: Option<usize>,
}

/// Resolved symbol information
pub struct Symbol {
    pub name: String,
    pub kind: SymbolKind,
    pub ty: Type,
}

/// Kind of symbol
pub enum SymbolKind {
    Local(SSAValue),
    Global(String),
    Function(String),
    Type(Type),
}

/// Type inference context
#[allow(dead_code)]
pub struct TypeContext {
    /// Type variables
    type_vars: BTreeMap<u32, Option<Type>>,
    /// Type constraints
    constraints: Vec<(u32, TypeConstraint)>,
}

impl Default for NameResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl NameResolver {
    #[must_use]
    pub fn new() -> Self {
        Self {
            scopes: vec![SymbolTable {
                symbols: BTreeMap::new(),
                parent: None,
            }],
            type_context: TypeContext {
                type_vars: BTreeMap::new(),
                constraints: Vec::new(),
            },
            forward_decls: BTreeMap::new(),
        }
    }

    pub fn push_scope(&mut self) {
        let parent = self.scopes.len() - 1;
        self.scopes.push(SymbolTable {
            symbols: BTreeMap::new(),
            parent: Some(parent),
        });
    }

    pub fn pop_scope(&mut self) {
        if self.scopes.len() > 1 {
            self.scopes.pop();
        }
    }
}

/// Example: Parsing a function with forward references
///
/// ```text
/// func @factorial(%n: i32) -> i32 {
///     %cond = cmpi "eq", %n, %zero : i32
///     cond_br %cond, ^base, ^recursive
///
///   ^base:
///     return %one : i32
///
///   ^recursive:
///     %n_minus_1 = subi %n, %one : i32
///     %rec = call @factorial(%n_minus_1) : (i32) -> i32  // Forward ref!
///     %result = muli %n, %rec : i32
///     return %result : i32
/// }
/// ```
///
/// During parsing:
/// 1. Create func op with regions
/// 2. Use `UnresolvedCall` for the recursive call
/// 3. After the function is complete, resolve the call
/// 4. Type check and verify
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unresolved_call() {
        let call = UnresolvedCall {
            name: "unknown_func".to_string(),
            args: vec![],
            expected_type: Some(crate::types::Type::Int(crate::types::IntWidth::I32)),
            location: crate::error::SourceLocation {
                file: "test.pmir".to_string(),
                line: 10,
                column: 5,
                span: crate::error::Span {
                    start: 100,
                    end: 115,
                },
            },
        };

        let op = ParsingOp::UnresolvedCall(call);

        // This would be resolved during a resolution pass
        match op {
            ParsingOp::UnresolvedCall(c) => {
                assert_eq!(c.name, "unknown_func");
            }
            _ => panic!("Wrong op type"),
        }
    }
}
