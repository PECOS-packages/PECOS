//! ZLUP003: Detect dynamic allocation inside loops.

use rustpython_parser::ast::{self, Expr, Mod, Stmt};

use super::super::diagnostic::{Diagnostic, Severity};
use super::{LintRule, make_location};

/// Functions/types that perform allocation.
const ALLOCATION_FUNCTIONS: &[&str] = &[
    "qalloc",
    "qubit",
    "list",
    "dict",
    "set",
    "bytearray",
    "array",
];

/// Methods that may cause allocation.
const ALLOCATION_METHODS: &[&str] = &["append", "extend", "insert", "copy"];

/// Detects dynamic allocation inside loop bodies.
pub struct ZLUP003DynamicAllocation;

impl LintRule for ZLUP003DynamicAllocation {
    fn id(&self) -> &'static str {
        "ZLUP003"
    }

    fn name(&self) -> &'static str {
        "dynamic-allocation"
    }

    fn description(&self) -> &'static str {
        "Dynamic allocation inside loops is prohibited. All allocations \
         should be performed at initialization time with fixed sizes."
    }

    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn check(&self, parsed: &Mod, filename: &str, source: &str) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();

        let Mod::Module(module) = parsed else {
            return diagnostics;
        };

        for stmt in &module.body {
            check_stmt(stmt, filename, source, 0, &mut diagnostics);
        }

        diagnostics
    }
}

fn check_stmt(
    stmt: &Stmt,
    filename: &str,
    source: &str,
    loop_depth: usize,
    diagnostics: &mut Vec<Diagnostic>,
) {
    match stmt {
        Stmt::For(for_stmt) => {
            for s in &for_stmt.body {
                check_stmt(s, filename, source, loop_depth + 1, diagnostics);
            }
            for s in &for_stmt.orelse {
                check_stmt(s, filename, source, loop_depth, diagnostics);
            }
        }
        Stmt::While(while_stmt) => {
            for s in &while_stmt.body {
                check_stmt(s, filename, source, loop_depth + 1, diagnostics);
            }
            for s in &while_stmt.orelse {
                check_stmt(s, filename, source, loop_depth, diagnostics);
            }
        }
        Stmt::FunctionDef(func) => {
            for s in &func.body {
                check_stmt(s, filename, source, 0, diagnostics);
            }
        }
        Stmt::AsyncFunctionDef(func) => {
            for s in &func.body {
                check_stmt(s, filename, source, 0, diagnostics);
            }
        }
        Stmt::ClassDef(class) => {
            for s in &class.body {
                check_stmt(s, filename, source, 0, diagnostics);
            }
        }
        Stmt::If(if_stmt) => {
            for s in &if_stmt.body {
                check_stmt(s, filename, source, loop_depth, diagnostics);
            }
            for s in &if_stmt.orelse {
                check_stmt(s, filename, source, loop_depth, diagnostics);
            }
        }
        Stmt::With(with_stmt) => {
            for s in &with_stmt.body {
                check_stmt(s, filename, source, loop_depth, diagnostics);
            }
        }
        Stmt::Try(try_stmt) => {
            for s in &try_stmt.body {
                check_stmt(s, filename, source, loop_depth, diagnostics);
            }
            for handler in &try_stmt.handlers {
                let ast::ExceptHandler::ExceptHandler(h) = handler;
                for s in &h.body {
                    check_stmt(s, filename, source, loop_depth, diagnostics);
                }
            }
            for s in &try_stmt.orelse {
                check_stmt(s, filename, source, loop_depth, diagnostics);
            }
            for s in &try_stmt.finalbody {
                check_stmt(s, filename, source, loop_depth, diagnostics);
            }
        }
        Stmt::Expr(expr_stmt) if loop_depth > 0 => {
            check_expr(&expr_stmt.value, filename, source, diagnostics);
        }
        Stmt::Assign(assign) if loop_depth > 0 => {
            check_expr(&assign.value, filename, source, diagnostics);
        }
        Stmt::AugAssign(aug) if loop_depth > 0 => {
            check_expr(&aug.value, filename, source, diagnostics);
        }
        Stmt::AnnAssign(ann) => {
            if loop_depth > 0
                && let Some(value) = &ann.value
            {
                check_expr(value, filename, source, diagnostics);
            }
        }
        _ => {}
    }
}

fn check_expr(expr: &Expr, filename: &str, source: &str, diagnostics: &mut Vec<Diagnostic>) {
    match expr {
        Expr::Call(call) => {
            // Check for direct allocation calls: list(), qalloc(), etc.
            if let Expr::Name(name) = call.func.as_ref() {
                let func_name = name.id.as_str();
                if ALLOCATION_FUNCTIONS.contains(&func_name) {
                    diagnostics.push(
                        Diagnostic::error(
                            "ZLUP003",
                            format!("'{}()' allocates memory inside a loop", func_name),
                            make_location(call.range, filename, source),
                        )
                        .with_suggestion(
                            "Move allocation outside the loop and reuse or pre-allocate with a fixed size",
                        )
                        .with_source_context(source),
                    );
                }
            }

            // Check for allocation methods: list.append(), etc.
            if let Expr::Attribute(attr) = call.func.as_ref() {
                let method_name = attr.attr.as_str();
                if ALLOCATION_METHODS.contains(&method_name) {
                    diagnostics.push(
                        Diagnostic::error(
                            "ZLUP003",
                            format!("'.{}()' may allocate memory inside a loop", method_name),
                            make_location(call.range, filename, source),
                        )
                        .with_suggestion(
                            "Pre-allocate the collection with sufficient capacity before the loop",
                        )
                        .with_source_context(source),
                    );
                }
            }

            // Recurse into arguments
            for arg in &call.args {
                check_expr(arg, filename, source, diagnostics);
            }
        }

        Expr::Subscript(sub) => {
            // Check for qubit[n] allocation
            if let Expr::Name(name) = sub.value.as_ref()
                && name.id.as_str() == "qubit"
            {
                diagnostics.push(
                    Diagnostic::error(
                        "ZLUP003",
                        "'qubit[...]' allocates qubits inside a loop",
                        make_location(sub.range, filename, source),
                    )
                    .with_suggestion("Allocate qubits outside the loop")
                    .with_source_context(source),
                );
            }
        }

        Expr::ListComp(comp) => {
            diagnostics.push(
                Diagnostic::error(
                    "ZLUP003",
                    "List comprehension allocates memory inside a loop",
                    make_location(comp.range, filename, source),
                )
                .with_suggestion("Pre-compute the list outside the loop")
                .with_source_context(source),
            );
        }

        Expr::DictComp(comp) => {
            diagnostics.push(
                Diagnostic::error(
                    "ZLUP003",
                    "Dict comprehension allocates memory inside a loop",
                    make_location(comp.range, filename, source),
                )
                .with_suggestion("Pre-compute the dict outside the loop")
                .with_source_context(source),
            );
        }

        Expr::SetComp(comp) => {
            diagnostics.push(
                Diagnostic::error(
                    "ZLUP003",
                    "Set comprehension allocates memory inside a loop",
                    make_location(comp.range, filename, source),
                )
                .with_suggestion("Pre-compute the set outside the loop")
                .with_source_context(source),
            );
        }

        // Recurse into sub-expressions
        Expr::BinOp(binop) => {
            check_expr(&binop.left, filename, source, diagnostics);
            check_expr(&binop.right, filename, source, diagnostics);
        }
        Expr::UnaryOp(unary) => {
            check_expr(&unary.operand, filename, source, diagnostics);
        }
        Expr::IfExp(ifexp) => {
            check_expr(&ifexp.test, filename, source, diagnostics);
            check_expr(&ifexp.body, filename, source, diagnostics);
            check_expr(&ifexp.orelse, filename, source, diagnostics);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustpython_parser::{Mode, parse};

    fn check_source(source: &str) -> Vec<Diagnostic> {
        let parsed = parse(source, Mode::Module, "<test>").unwrap();
        ZLUP003DynamicAllocation.check(&parsed, "<test>", source)
    }

    #[test]
    fn test_list_in_loop() {
        let diagnostics = check_source(
            r#"
for i in range(10):
    x = list()
"#,
        );
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].message.contains("list()"));
    }

    #[test]
    fn test_append_in_loop() {
        let diagnostics = check_source(
            r#"
result = []
for i in range(10):
    result.append(i)
"#,
        );
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].message.contains(".append()"));
    }

    #[test]
    fn test_list_comprehension_in_loop() {
        let diagnostics = check_source(
            r#"
for i in range(10):
    x = [j for j in range(5)]
"#,
        );
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].message.contains("List comprehension"));
    }

    #[test]
    fn test_allocation_outside_loop() {
        let diagnostics = check_source(
            r#"
x = list()
for i in range(10):
    pass
"#,
        );
        assert!(diagnostics.is_empty());
    }
}
