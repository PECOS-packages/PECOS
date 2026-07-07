//! ZLUP001: Detect unbounded loops.

use rustpython_parser::ast::{self, Constant, Expr, Mod, Stmt};

use super::super::diagnostic::{Diagnostic, Severity};
use super::{LintRule, make_location};

/// Detects unbounded loops that violate NASA Power of 10 rules.
pub struct ZLUP001UnboundedLoops;

impl LintRule for ZLUP001UnboundedLoops {
    fn id(&self) -> &'static str {
        "ZLUP001"
    }

    fn name(&self) -> &'static str {
        "unbounded-loops"
    }

    fn description(&self) -> &'static str {
        "All loops must have a fixed upper bound. Unbounded loops \
         (while True, while with non-constant condition) are prohibited."
    }

    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn check(&self, parsed: &Mod, filename: &str, source: &str) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();

        if let Mod::Module(module) = parsed {
            for stmt in &module.body {
                check_stmt(stmt, filename, source, &mut diagnostics);
            }
        }

        diagnostics
    }
}

fn check_stmt(stmt: &Stmt, filename: &str, source: &str, diagnostics: &mut Vec<Diagnostic>) {
    match stmt {
        Stmt::While(while_stmt) => {
            check_while_loop(while_stmt, filename, source, diagnostics);
            // Recursively check body
            for s in &while_stmt.body {
                check_stmt(s, filename, source, diagnostics);
            }
            for s in &while_stmt.orelse {
                check_stmt(s, filename, source, diagnostics);
            }
        }
        Stmt::For(for_stmt) => {
            for s in &for_stmt.body {
                check_stmt(s, filename, source, diagnostics);
            }
            for s in &for_stmt.orelse {
                check_stmt(s, filename, source, diagnostics);
            }
        }
        Stmt::FunctionDef(func) => {
            for s in &func.body {
                check_stmt(s, filename, source, diagnostics);
            }
        }
        Stmt::AsyncFunctionDef(func) => {
            for s in &func.body {
                check_stmt(s, filename, source, diagnostics);
            }
        }
        Stmt::ClassDef(class) => {
            for s in &class.body {
                check_stmt(s, filename, source, diagnostics);
            }
        }
        Stmt::If(if_stmt) => {
            for s in &if_stmt.body {
                check_stmt(s, filename, source, diagnostics);
            }
            for s in &if_stmt.orelse {
                check_stmt(s, filename, source, diagnostics);
            }
        }
        Stmt::With(with_stmt) => {
            for s in &with_stmt.body {
                check_stmt(s, filename, source, diagnostics);
            }
        }
        Stmt::Try(try_stmt) => {
            for s in &try_stmt.body {
                check_stmt(s, filename, source, diagnostics);
            }
            for handler in &try_stmt.handlers {
                let ast::ExceptHandler::ExceptHandler(h) = handler;
                for s in &h.body {
                    check_stmt(s, filename, source, diagnostics);
                }
            }
            for s in &try_stmt.orelse {
                check_stmt(s, filename, source, diagnostics);
            }
            for s in &try_stmt.finalbody {
                check_stmt(s, filename, source, diagnostics);
            }
        }
        _ => {}
    }
}

fn check_while_loop(
    while_stmt: &ast::StmtWhile,
    filename: &str,
    source: &str,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let test = &while_stmt.test;

    // Check for `while True:`
    if let Expr::Constant(c) = test.as_ref() {
        if matches!(c.value, Constant::Bool(true)) {
            // Only report if there's no unconditional break
            if !has_unconditional_break(&while_stmt.body) {
                diagnostics.push(
                    Diagnostic::error(
                        "ZLUP001",
                        "'while True' creates an unbounded loop",
                        make_location(while_stmt.range, filename, source),
                    )
                    .with_suggestion("Use a for loop with a fixed upper bound instead")
                    .with_source_context(source),
                );
            }
            return;
        }

        // Check for `while 1:`
        if let Constant::Int(ref i) = c.value
            && (i.to_u32_digits().1 == [1] || i.to_string() == "1")
        {
            // Only report if there's no unconditional break
            if !has_unconditional_break(&while_stmt.body) {
                diagnostics.push(
                    Diagnostic::error(
                        "ZLUP001",
                        "'while 1' creates an unbounded loop",
                        make_location(while_stmt.range, filename, source),
                    )
                    .with_suggestion("Use a for loop with a fixed upper bound instead")
                    .with_source_context(source),
                );
            }
            return;
        }
    }

    // Check for while loops without a clear termination condition
    if !has_bounded_condition(test) && !has_unconditional_break(&while_stmt.body) {
        diagnostics.push(
            Diagnostic::error(
                "ZLUP001",
                "while loop may be unbounded",
                make_location(while_stmt.range, filename, source),
            )
            .with_suggestion(
                "Consider using a for loop with range() or add a fixed iteration limit",
            )
            .with_source_context(source),
        );
    }
}

fn has_bounded_condition(test: &Expr) -> bool {
    match test {
        // Comparisons against variables that might change are acceptable
        Expr::Compare(_) => true,
        // Boolean variables are acceptable (might become False)
        Expr::Name(_) => true,
        // Binary boolean operations
        Expr::BoolOp(op) => op.values.iter().all(has_bounded_condition),
        // Unary not
        Expr::UnaryOp(unary) if matches!(unary.op, ast::UnaryOp::Not) => {
            has_bounded_condition(&unary.operand)
        }
        _ => false,
    }
}

fn has_unconditional_break(body: &[Stmt]) -> bool {
    for stmt in body {
        if matches!(stmt, Stmt::Break(_)) {
            return true;
        }
        // Check for guaranteed break in all branches of an if
        if let Stmt::If(if_stmt) = stmt
            && has_unconditional_break(&if_stmt.body)
            && has_unconditional_break(&if_stmt.orelse)
        {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustpython_parser::{Mode, parse};

    fn check_source(source: &str) -> Vec<Diagnostic> {
        let parsed = parse(source, Mode::Module, "<test>").unwrap();
        ZLUP001UnboundedLoops.check(&parsed, "<test>", source)
    }

    #[test]
    fn test_while_true() {
        let diagnostics = check_source(
            r#"
while True:
    pass
"#,
        );
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].message.contains("while True"));
    }

    #[test]
    fn test_while_one() {
        let diagnostics = check_source(
            r#"
while 1:
    pass
"#,
        );
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].message.contains("while 1"));
    }

    #[test]
    fn test_while_with_comparison() {
        let diagnostics = check_source(
            r#"
x = 10
while x > 0:
    x -= 1
"#,
        );
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn test_while_with_break() {
        let diagnostics = check_source(
            r#"
while True:
    if condition:
        break
    else:
        break
"#,
        );
        // Has unconditional break in both branches
        assert!(diagnostics.is_empty());
    }
}
