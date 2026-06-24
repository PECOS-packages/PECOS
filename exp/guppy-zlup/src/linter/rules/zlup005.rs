//! ZLUP005: Detect unchecked error conditions.

use rustpython_parser::ast::{self, Constant, Expr, Mod, Stmt};

use super::{make_location, LintRule};
use super::super::diagnostic::{Diagnostic, Severity};

/// Functions that may raise exceptions and should be wrapped in try/except.
const FUNCTIONS_THAT_MAY_RAISE: &[(&str, &str)] = &[
    ("open", "may raise FileNotFoundError or PermissionError"),
    ("int", "may raise ValueError"),
    ("float", "may raise ValueError"),
];

/// Detects potentially unchecked error conditions.
pub struct ZLUP005UncheckedErrors;

impl LintRule for ZLUP005UncheckedErrors {
    fn id(&self) -> &'static str {
        "ZLUP005"
    }

    fn name(&self) -> &'static str {
        "unchecked-errors"
    }

    fn description(&self) -> &'static str {
        "Operations that may fail should have explicit error handling. \
         Assertions should be used to validate preconditions."
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
            check_stmt(stmt, filename, source, false, &mut diagnostics);
        }

        diagnostics
    }
}

fn check_stmt(
    stmt: &Stmt,
    filename: &str,
    source: &str,
    in_try: bool,
    diagnostics: &mut Vec<Diagnostic>,
) {
    match stmt {
        Stmt::Try(try_stmt) => {
            // Inside try block, errors are handled
            for s in &try_stmt.body {
                check_stmt(s, filename, source, true, diagnostics);
            }
            for handler in &try_stmt.handlers {
                let ast::ExceptHandler::ExceptHandler(h) = handler;
                for s in &h.body {
                    check_stmt(s, filename, source, false, diagnostics);
                }
            }
            for s in &try_stmt.orelse {
                check_stmt(s, filename, source, false, diagnostics);
            }
            for s in &try_stmt.finalbody {
                check_stmt(s, filename, source, false, diagnostics);
            }
        }
        Stmt::FunctionDef(func) => {
            for s in &func.body {
                check_stmt(s, filename, source, false, diagnostics);
            }
        }
        Stmt::AsyncFunctionDef(func) => {
            for s in &func.body {
                check_stmt(s, filename, source, false, diagnostics);
            }
        }
        Stmt::ClassDef(class) => {
            for s in &class.body {
                check_stmt(s, filename, source, false, diagnostics);
            }
        }
        Stmt::For(for_stmt) => {
            check_expr(&for_stmt.iter, filename, source, in_try, diagnostics);
            for s in &for_stmt.body {
                check_stmt(s, filename, source, in_try, diagnostics);
            }
            for s in &for_stmt.orelse {
                check_stmt(s, filename, source, in_try, diagnostics);
            }
        }
        Stmt::While(while_stmt) => {
            check_expr(&while_stmt.test, filename, source, in_try, diagnostics);
            for s in &while_stmt.body {
                check_stmt(s, filename, source, in_try, diagnostics);
            }
            for s in &while_stmt.orelse {
                check_stmt(s, filename, source, in_try, diagnostics);
            }
        }
        Stmt::If(if_stmt) => {
            check_expr(&if_stmt.test, filename, source, in_try, diagnostics);
            for s in &if_stmt.body {
                check_stmt(s, filename, source, in_try, diagnostics);
            }
            for s in &if_stmt.orelse {
                check_stmt(s, filename, source, in_try, diagnostics);
            }
        }
        Stmt::With(with_stmt) => {
            for s in &with_stmt.body {
                check_stmt(s, filename, source, in_try, diagnostics);
            }
        }
        Stmt::Expr(expr_stmt) => {
            check_expr(&expr_stmt.value, filename, source, in_try, diagnostics);
        }
        Stmt::Assign(assign) => {
            check_expr(&assign.value, filename, source, in_try, diagnostics);
        }
        Stmt::AugAssign(aug) => {
            check_expr(&aug.value, filename, source, in_try, diagnostics);
        }
        Stmt::AnnAssign(ann) => {
            if let Some(value) = &ann.value {
                check_expr(value, filename, source, in_try, diagnostics);
            }
        }
        Stmt::Return(ret) => {
            if let Some(value) = &ret.value {
                check_expr(value, filename, source, in_try, diagnostics);
            }
        }
        _ => {}
    }
}

fn check_expr(
    expr: &Expr,
    filename: &str,
    source: &str,
    in_try: bool,
    diagnostics: &mut Vec<Diagnostic>,
) {
    if in_try {
        // Inside try block, don't report
        return;
    }

    match expr {
        Expr::BinOp(binop) => {
            // Check for division
            if matches!(
                binop.op,
                ast::Operator::Div | ast::Operator::FloorDiv | ast::Operator::Mod
            ) {
                // Check if divisor is a non-zero literal
                let is_safe = if let Expr::Constant(c) = binop.right.as_ref() {
                    match &c.value {
                        Constant::Int(i) => {
                            // Check if not zero
                            i.to_u32_digits().1.first() != Some(&0) || !i.to_u32_digits().1.is_empty()
                        }
                        Constant::Float(f) => *f != 0.0,
                        _ => false,
                    }
                } else {
                    false
                };

                if !is_safe {
                    let op_name = match binop.op {
                        ast::Operator::Div => "division",
                        ast::Operator::FloorDiv => "floor division",
                        ast::Operator::Mod => "modulo",
                        _ => "operation",
                    };
                    diagnostics.push(
                        Diagnostic::warning(
                            "ZLUP005",
                            format!("Unchecked {} may raise ZeroDivisionError", op_name),
                            make_location(binop.range, filename, source),
                        )
                        .with_suggestion("Add an assertion or check that divisor is non-zero")
                        .with_source_context(source),
                    );
                }
            }

            check_expr(&binop.left, filename, source, in_try, diagnostics);
            check_expr(&binop.right, filename, source, in_try, diagnostics);
        }

        Expr::Call(call) => {
            // Check for calls to functions that may raise
            if let Expr::Name(name) = call.func.as_ref() {
                let func_name = name.id.as_str();
                if let Some((_, reason)) =
                    FUNCTIONS_THAT_MAY_RAISE.iter().find(|(n, _)| *n == func_name)
                {
                    diagnostics.push(
                        Diagnostic::warning(
                            "ZLUP005",
                            format!("'{}()' {}", func_name, reason),
                            make_location(call.range, filename, source),
                        )
                        .with_suggestion("Wrap in try/except or validate input first")
                        .with_source_context(source),
                    );
                }
            }

            // Recurse into arguments
            for arg in &call.args {
                check_expr(arg, filename, source, in_try, diagnostics);
            }
        }

        // Recurse into sub-expressions
        Expr::UnaryOp(unary) => {
            check_expr(&unary.operand, filename, source, in_try, diagnostics);
        }
        Expr::Compare(cmp) => {
            check_expr(&cmp.left, filename, source, in_try, diagnostics);
            for comparator in &cmp.comparators {
                check_expr(comparator, filename, source, in_try, diagnostics);
            }
        }
        Expr::BoolOp(boolop) => {
            for value in &boolop.values {
                check_expr(value, filename, source, in_try, diagnostics);
            }
        }
        Expr::IfExp(ifexp) => {
            check_expr(&ifexp.test, filename, source, in_try, diagnostics);
            check_expr(&ifexp.body, filename, source, in_try, diagnostics);
            check_expr(&ifexp.orelse, filename, source, in_try, diagnostics);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustpython_parser::{parse, Mode};

    fn check_source(source: &str) -> Vec<Diagnostic> {
        let parsed = parse(source, Mode::Module, "<test>").unwrap();
        ZLUP005UncheckedErrors.check(&parsed, "<test>", source)
    }

    #[test]
    fn test_division() {
        let diagnostics = check_source(
            r#"
result = a / b
"#,
        );
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].message.contains("division"));
    }

    #[test]
    fn test_division_by_literal() {
        let diagnostics = check_source(
            r#"
result = a / 2
"#,
        );
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn test_int_conversion() {
        let diagnostics = check_source(
            r#"
x = int(user_input)
"#,
        );
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].message.contains("int()"));
    }

    #[test]
    fn test_in_try_block() {
        let diagnostics = check_source(
            r#"
try:
    x = int(user_input)
except ValueError:
    x = 0
"#,
        );
        assert!(diagnostics.is_empty());
    }
}
