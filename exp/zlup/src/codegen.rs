//! Code generation backends for Zluppy.
//!
//! Zluppy compiles to multiple targets:
//! - **HUGR**: Hierarchical Unified Graph Representation for experiments/hardware
//! - **SLR-AST**: JSON bridge to Python/PECOS for integration
//! - **PHIR/JSON**: JSON serialization of PECOS High-level IR for simulator targeting
//! - **QASM**: OpenQASM 2.0 for hardware execution
//!
//! ## Design Philosophy
//!
//! Same problems as Guppy, simpler idioms:
//! - Explicit over implicit
//! - Low-level but safe
//! - Predictable, bounded output

#[cfg(feature = "hugr")]
pub mod hugr;
pub mod phir;
pub mod qasm;
pub mod slr;

#[cfg(feature = "hugr")]
pub use hugr::{CodegenMode, HugrCodegen};
pub use phir::{PhirJsonCodegen, PhirJsonError, PhirJsonProgram};
pub use qasm::{QasmCodegen, QasmError};
pub use slr::SlrCodegen;

/// Unrolling cannot preserve escaping control transfers. Reject their syntax
/// before evaluating bounds or selecting branches, including empty ranges.
/// Function literals own a separate return boundary and are not traversed.
fn block_control(block: &crate::ast::Block) -> Option<&'static str> {
    statements_control(&block.statements, block.trailing_expr.as_deref())
}

fn statements_control(
    statements: &[crate::ast::Stmt],
    trailing: Option<&crate::ast::Expr>,
) -> Option<&'static str> {
    statements
        .iter()
        .find_map(statement_control)
        .or_else(|| trailing.and_then(expression_control))
}

fn if_control(stmt: &crate::ast::IfStmt) -> Option<&'static str> {
    expression_control(&stmt.condition)
        .or_else(|| block_control(&stmt.then_body))
        .or_else(|| {
            stmt.else_body.as_ref().and_then(|branch| match branch {
                crate::ast::ElseBranch::Else(block) => block_control(block),
                crate::ast::ElseBranch::ElseIf(stmt) => if_control(stmt),
            })
        })
}

fn statement_control(stmt: &crate::ast::Stmt) -> Option<&'static str> {
    use crate::ast::{ForRange, Stmt};
    match stmt {
        Stmt::Return(_) => Some("return"),
        Stmt::Break(_) => Some("break"),
        Stmt::Continue(_) => Some("continue"),
        Stmt::Binding(binding) => binding
            .ty
            .as_ref()
            .and_then(type_control)
            .or_else(|| binding.value.as_ref().and_then(expression_control)),
        Stmt::Alias(alias) => expression_control(&alias.source),
        Stmt::Assign(assign) => {
            expression_control(&assign.target).or_else(|| expression_control(&assign.value))
        }
        Stmt::Expr(stmt) => expression_control(&stmt.expr),
        Stmt::If(stmt) => if_control(stmt),
        Stmt::For(stmt) => (match &stmt.range {
            ForRange::Range { start, end } => {
                expression_control(start).or_else(|| expression_control(end))
            }
            ForRange::Collection(expr) => expression_control(expr),
        })
        .or_else(|| block_control(&stmt.body)),
        Stmt::Switch(stmt) => expression_control(&stmt.value).or_else(|| {
            stmt.prongs.iter().find_map(|prong| {
                prong
                    .cases
                    .iter()
                    .find_map(|case| {
                        expression_control(&case.value)
                            .or_else(|| case.end.as_ref().and_then(expression_control))
                    })
                    .or_else(|| expression_control(&prong.body))
            })
        }),
        Stmt::Tick(stmt) => statements_control(&stmt.body, None),
        Stmt::TryBlock(stmt) => block_control(&stmt.body).or_else(|| {
            stmt.catch_clause
                .as_ref()
                .and_then(|clause| expression_control(&clause.body))
        }),
        Stmt::Defer(stmt) => statement_control(&stmt.body),
        Stmt::Errdefer(stmt) => statement_control(&stmt.body),
        Stmt::Block(block) => block_control(block),
        Stmt::Gate(gate) => gate.params.iter().find_map(expression_control).or_else(|| {
            gate.targets
                .iter()
                .find_map(|slot| expression_control(&slot.index))
        }),
        Stmt::Measure(measure) => measure
            .targets
            .iter()
            .find_map(|slot| expression_control(&slot.index))
            .or_else(|| {
                measure
                    .results
                    .iter()
                    .find_map(|bit| expression_control(&bit.index))
            }),
        Stmt::Prepare(_) | Stmt::Barrier(_) => None,
    }
}

fn expression_control(expr: &crate::ast::Expr) -> Option<&'static str> {
    use crate::ast::{Expr, FStringPart};
    match expr {
        Expr::Unary(unary) => expression_control(&unary.operand),
        Expr::TryBlock(stmt) => block_control(&stmt.body).or_else(|| {
            stmt.catch_clause
                .as_ref()
                .and_then(|clause| expression_control(&clause.body))
        }),
        Expr::Block(block) => statements_control(&block.statements, block.trailing_expr.as_deref()),
        Expr::Binary(binary) => {
            expression_control(&binary.left).or_else(|| expression_control(&binary.right))
        }
        Expr::If(expr) => expression_control(&expr.condition)
            .or_else(|| expression_control(&expr.then_expr))
            .or_else(|| expression_control(&expr.else_expr)),
        Expr::AngleLit(angle) => expression_control(&angle.value),
        Expr::TypeAscription(expr) => expression_control(&expr.value),
        Expr::Comptime(expr) => expression_control(&expr.inner),
        Expr::FString(string) => string.parts.iter().find_map(|part| match part {
            FStringPart::Expr { expr, .. } => expression_control(expr),
            FStringPart::Text(_) => None,
        }),
        Expr::SlotRef(slot) => expression_control(&slot.index),
        Expr::BitRef(bit) => expression_control(&bit.index),
        Expr::Field(field) => expression_control(&field.object),
        Expr::Index(index) => {
            expression_control(&index.object).or_else(|| expression_control(&index.index))
        }
        Expr::Call(call) => expression_control(&call.callee)
            .or_else(|| call.args.iter().find_map(expression_control)),
        Expr::BatchApply(batch) => expression_control(&batch.operation)
            .or_else(|| batch.targets.iter().find_map(expression_control)),
        Expr::Builtin(builtin) => builtin.args.iter().find_map(expression_control),
        Expr::StructInit(init) => init.ty.as_ref().and_then(type_control).or_else(|| {
            init.fields
                .iter()
                .find_map(|field| expression_control(&field.value))
        }),
        Expr::ArrayInit(array) => array
            .ty
            .as_ref()
            .and_then(type_control)
            .or_else(|| array.elements.iter().find_map(expression_control)),
        Expr::BracketArray(array) => array.elements.iter().find_map(expression_control),
        Expr::Tuple(tuple) => tuple.elements.iter().find_map(expression_control),
        Expr::Set(set) => set.elements.iter().find_map(expression_control),
        Expr::Range(range) => range
            .start
            .as_ref()
            .and_then(expression_control)
            .or_else(|| range.end.as_ref().and_then(expression_control)),
        Expr::Measure(measure) => {
            type_control(&measure.result_type).or_else(|| expression_control(&measure.targets))
        }
        Expr::Gate(gate) => gate
            .params
            .iter()
            .find_map(expression_control)
            .or_else(|| expression_control(&gate.target)),
        Expr::Catch(catch) => {
            expression_control(&catch.operand).or_else(|| expression_control(&catch.handler))
        }
        Expr::Result(result) => expression_control(&result.value),
        Expr::Channel(channel) => channel
            .args
            .iter()
            .find_map(|arg| expression_control(arg.value())),
        Expr::AnonStruct(structure) => structure.fields.iter().find_map(field_control),
        // A function literal defines its own return boundary; it is not executed here.
        Expr::FnLit(_)
        | Expr::IntLit(_)
        | Expr::FloatLit(_)
        | Expr::BoolLit(_)
        | Expr::StringLit(_)
        | Expr::CharLit(_)
        | Expr::Null(_)
        | Expr::Undefined(_)
        | Expr::Unit(_)
        | Expr::Ident(_)
        | Expr::ErrorValue(_)
        | Expr::FaultValue(_) => None,
    }
}

fn field_control(field: &crate::ast::StructField) -> Option<&'static str> {
    type_control(&field.ty).or_else(|| field.default.as_ref().and_then(expression_control))
}

fn type_control(ty: &crate::ast::TypeExpr) -> Option<&'static str> {
    use crate::ast::TypeExpr;
    match ty {
        TypeExpr::QAlloc(size) => size.as_deref().and_then(expression_control),
        TypeExpr::Array(array) => type_control(&array.element)
            .or_else(|| array.size.as_ref().and_then(expression_control))
            .or_else(|| array.sentinel.as_ref().and_then(expression_control)),
        TypeExpr::Pointer(pointer) => type_control(&pointer.pointee)
            .or_else(|| pointer.sentinel.as_ref().and_then(expression_control)),
        TypeExpr::Optional(ty) | TypeExpr::Set(ty) => type_control(ty),
        TypeExpr::ErrorUnion(ty) => {
            type_control(&ty.error_type).or_else(|| type_control(&ty.payload_type))
        }
        TypeExpr::CollectedErrors(ty) => {
            type_control(&ty.error_type).or_else(|| type_control(&ty.payload_type))
        }
        TypeExpr::Tuple(types) => types.iter().find_map(type_control),
        TypeExpr::Struct(ty) => ty.fields.iter().find_map(field_control),
        TypeExpr::Enum(ty) => ty.tag_type.as_ref().and_then(type_control).or_else(|| {
            ty.variants
                .iter()
                .find_map(|variant| variant.value.as_ref().and_then(expression_control))
        }),
        TypeExpr::Fn(ty) => ty
            .params
            .iter()
            .find_map(type_control)
            .or_else(|| ty.return_type.as_ref().and_then(type_control)),
        TypeExpr::Primitive(_)
        | TypeExpr::Qubit
        | TypeExpr::Bit
        | TypeExpr::Named(_)
        | TypeExpr::Type
        | TypeExpr::AnyType
        | TypeExpr::Unit => None,
    }
}
