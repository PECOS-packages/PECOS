//! ZLUP008: Detect excessive call depth (deeply nested function calls).

use rustpython_parser::ast::{Expr, Mod, Stmt};

use super::super::diagnostic::{Diagnostic, Severity};
use super::{LintRule, make_location};

/// Maximum allowed call depth (function calls nested within other calls).
const DEFAULT_MAX_CALL_DEPTH: u32 = 4;

/// Detects excessive call depth that makes code hard to verify.
pub struct ZLUP008CallDepth {
    max_depth: u32,
}

impl ZLUP008CallDepth {
    pub fn new(max_depth: u32) -> Self {
        Self { max_depth }
    }
}

impl Default for ZLUP008CallDepth {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_CALL_DEPTH)
    }
}

impl LintRule for ZLUP008CallDepth {
    fn id(&self) -> &'static str {
        "ZLUP008"
    }

    fn name(&self) -> &'static str {
        "call-depth"
    }

    fn description(&self) -> &'static str {
        "Deeply nested function calls make code hard to verify and debug. \
         Keep call chains shallow for better analyzability."
    }

    fn severity(&self) -> Severity {
        Severity::Warning
    }

    fn check(&self, parsed: &Mod, filename: &str, source: &str) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();

        let Mod::Module(module) = parsed else {
            return diagnostics;
        };

        for stmt in &module.body {
            check_stmt(stmt, filename, source, self.max_depth, &mut diagnostics);
        }

        diagnostics
    }
}

fn check_stmt(
    stmt: &Stmt,
    filename: &str,
    source: &str,
    max_depth: u32,
    diagnostics: &mut Vec<Diagnostic>,
) {
    match stmt {
        Stmt::FunctionDef(func) => {
            for s in &func.body {
                check_stmt(s, filename, source, max_depth, diagnostics);
            }
        }
        Stmt::AsyncFunctionDef(func) => {
            for s in &func.body {
                check_stmt(s, filename, source, max_depth, diagnostics);
            }
        }
        Stmt::ClassDef(class) => {
            for s in &class.body {
                check_stmt(s, filename, source, max_depth, diagnostics);
            }
        }
        Stmt::For(for_stmt) => {
            check_expr(&for_stmt.iter, filename, source, max_depth, 0, diagnostics);
            for s in &for_stmt.body {
                check_stmt(s, filename, source, max_depth, diagnostics);
            }
            for s in &for_stmt.orelse {
                check_stmt(s, filename, source, max_depth, diagnostics);
            }
        }
        Stmt::While(while_stmt) => {
            check_expr(
                &while_stmt.test,
                filename,
                source,
                max_depth,
                0,
                diagnostics,
            );
            for s in &while_stmt.body {
                check_stmt(s, filename, source, max_depth, diagnostics);
            }
            for s in &while_stmt.orelse {
                check_stmt(s, filename, source, max_depth, diagnostics);
            }
        }
        Stmt::If(if_stmt) => {
            check_expr(&if_stmt.test, filename, source, max_depth, 0, diagnostics);
            for s in &if_stmt.body {
                check_stmt(s, filename, source, max_depth, diagnostics);
            }
            for s in &if_stmt.orelse {
                check_stmt(s, filename, source, max_depth, diagnostics);
            }
        }
        Stmt::With(with_stmt) => {
            for s in &with_stmt.body {
                check_stmt(s, filename, source, max_depth, diagnostics);
            }
        }
        Stmt::Try(try_stmt) => {
            for s in &try_stmt.body {
                check_stmt(s, filename, source, max_depth, diagnostics);
            }
            for s in &try_stmt.orelse {
                check_stmt(s, filename, source, max_depth, diagnostics);
            }
            for s in &try_stmt.finalbody {
                check_stmt(s, filename, source, max_depth, diagnostics);
            }
        }
        Stmt::Expr(expr_stmt) => {
            check_expr(
                &expr_stmt.value,
                filename,
                source,
                max_depth,
                0,
                diagnostics,
            );
        }
        Stmt::Assign(assign) => {
            check_expr(&assign.value, filename, source, max_depth, 0, diagnostics);
        }
        Stmt::AugAssign(aug) => {
            check_expr(&aug.value, filename, source, max_depth, 0, diagnostics);
        }
        Stmt::AnnAssign(ann) => {
            if let Some(value) = &ann.value {
                check_expr(value, filename, source, max_depth, 0, diagnostics);
            }
        }
        Stmt::Return(ret) => {
            if let Some(value) = &ret.value {
                check_expr(value, filename, source, max_depth, 0, diagnostics);
            }
        }
        _ => {}
    }
}

fn check_expr(
    expr: &Expr,
    filename: &str,
    source: &str,
    max_depth: u32,
    current_depth: u32,
    diagnostics: &mut Vec<Diagnostic>,
) {
    match expr {
        Expr::Call(call) => {
            let new_depth = current_depth + 1;

            if new_depth > max_depth {
                diagnostics.push(
                    Diagnostic::warning(
                        "ZLUP008",
                        format!(
                            "Call depth {} exceeds maximum {} - consider breaking into intermediate variables",
                            new_depth, max_depth
                        ),
                        make_location(call.range, filename, source),
                    )
                    .with_suggestion("Extract nested calls into named intermediate values")
                    .with_source_context(source),
                );
            }

            // Check the function being called
            check_expr(
                &call.func,
                filename,
                source,
                max_depth,
                new_depth,
                diagnostics,
            );

            // Check arguments
            for arg in &call.args {
                check_expr(arg, filename, source, max_depth, new_depth, diagnostics);
            }
        }

        // Recurse into sub-expressions
        Expr::BinOp(binop) => {
            check_expr(
                &binop.left,
                filename,
                source,
                max_depth,
                current_depth,
                diagnostics,
            );
            check_expr(
                &binop.right,
                filename,
                source,
                max_depth,
                current_depth,
                diagnostics,
            );
        }
        Expr::UnaryOp(unary) => {
            check_expr(
                &unary.operand,
                filename,
                source,
                max_depth,
                current_depth,
                diagnostics,
            );
        }
        Expr::Compare(cmp) => {
            check_expr(
                &cmp.left,
                filename,
                source,
                max_depth,
                current_depth,
                diagnostics,
            );
            for comparator in &cmp.comparators {
                check_expr(
                    comparator,
                    filename,
                    source,
                    max_depth,
                    current_depth,
                    diagnostics,
                );
            }
        }
        Expr::BoolOp(boolop) => {
            for value in &boolop.values {
                check_expr(
                    value,
                    filename,
                    source,
                    max_depth,
                    current_depth,
                    diagnostics,
                );
            }
        }
        Expr::IfExp(ifexp) => {
            check_expr(
                &ifexp.test,
                filename,
                source,
                max_depth,
                current_depth,
                diagnostics,
            );
            check_expr(
                &ifexp.body,
                filename,
                source,
                max_depth,
                current_depth,
                diagnostics,
            );
            check_expr(
                &ifexp.orelse,
                filename,
                source,
                max_depth,
                current_depth,
                diagnostics,
            );
        }
        Expr::List(list) => {
            for elt in &list.elts {
                check_expr(elt, filename, source, max_depth, current_depth, diagnostics);
            }
        }
        Expr::Tuple(tuple) => {
            for elt in &tuple.elts {
                check_expr(elt, filename, source, max_depth, current_depth, diagnostics);
            }
        }
        Expr::Subscript(sub) => {
            check_expr(
                &sub.value,
                filename,
                source,
                max_depth,
                current_depth,
                diagnostics,
            );
            check_expr(
                &sub.slice,
                filename,
                source,
                max_depth,
                current_depth,
                diagnostics,
            );
        }
        Expr::Attribute(attr) => {
            check_expr(
                &attr.value,
                filename,
                source,
                max_depth,
                current_depth,
                diagnostics,
            );
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustpython_parser::{Mode, parse};

    fn check_source_with_max(source: &str, max: u32) -> Vec<Diagnostic> {
        let parsed = parse(source, Mode::Module, "<test>").unwrap();
        ZLUP008CallDepth::new(max).check(&parsed, "<test>", source)
    }

    #[test]
    fn test_simple_call() {
        let diagnostics = check_source_with_max("result = foo()", 4);
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn test_nested_calls() {
        let diagnostics = check_source_with_max("result = foo(bar())", 4);
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn test_deeply_nested_calls() {
        let diagnostics = check_source_with_max("result = a(b(c(d(e()))))", 3);
        assert!(!diagnostics.is_empty());
        assert!(diagnostics[0].message.contains("Call depth"));
    }

    #[test]
    fn test_call_in_arguments() {
        let diagnostics = check_source_with_max("result = foo(bar(baz(qux())))", 2);
        assert!(!diagnostics.is_empty());
    }
}
