//! ZLUP006: Detect missing type annotations.

use rustpython_parser::ast::{Mod, Stmt};

use super::super::diagnostic::{Diagnostic, Severity};
use super::{LintRule, make_location};

/// Detects missing type annotations on function signatures.
pub struct ZLUP006MissingTypes;

impl LintRule for ZLUP006MissingTypes {
    fn id(&self) -> &'static str {
        "ZLUP006"
    }

    fn name(&self) -> &'static str {
        "missing-types"
    }

    fn description(&self) -> &'static str {
        "All function parameters and return types should have explicit \
         type annotations for Zlup compilation."
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
            check_stmt(stmt, filename, source, &mut diagnostics);
        }

        diagnostics
    }
}

fn check_stmt(stmt: &Stmt, filename: &str, source: &str, diagnostics: &mut Vec<Diagnostic>) {
    match stmt {
        Stmt::FunctionDef(func) => {
            check_function(func, filename, source, diagnostics);
            // Check nested functions
            for s in &func.body {
                check_stmt(s, filename, source, diagnostics);
            }
        }
        Stmt::AsyncFunctionDef(func) => {
            check_async_function(func, filename, source, diagnostics);
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
        Stmt::For(for_stmt) => {
            for s in &for_stmt.body {
                check_stmt(s, filename, source, diagnostics);
            }
        }
        Stmt::While(while_stmt) => {
            for s in &while_stmt.body {
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
        }
        _ => {}
    }
}

fn check_function(
    func: &rustpython_parser::ast::StmtFunctionDef,
    filename: &str,
    source: &str,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let func_name = func.name.as_str();

    // Skip dunder methods and test functions
    if func_name.starts_with("__") || func_name.starts_with("test_") {
        return;
    }

    // Check parameters (func.args instead of func.parameters)
    for arg in &func.args.args {
        let arg_name = arg.def.arg.as_str();

        // Skip 'self' and 'cls'
        if arg_name == "self" || arg_name == "cls" {
            continue;
        }

        if arg.def.annotation.is_none() {
            diagnostics.push(
                Diagnostic::warning(
                    "ZLUP006",
                    format!("Parameter '{}' is missing type annotation", arg_name),
                    make_location(func.range, filename, source),
                )
                .with_suggestion(format!("Add type annotation: {}: <type>", arg_name))
                .with_source_context(source),
            );
        }
    }

    // Check return type
    if func.returns.is_none() && !is_trivial_body(&func.body) {
        diagnostics.push(
            Diagnostic::warning(
                "ZLUP006",
                format!("Function '{}' is missing return type annotation", func_name),
                make_location(func.range, filename, source),
            )
            .with_suggestion(format!(
                "Add return type: def {}(...) -> <type>:",
                func_name
            ))
            .with_source_context(source),
        );
    }
}

fn check_async_function(
    func: &rustpython_parser::ast::StmtAsyncFunctionDef,
    filename: &str,
    source: &str,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let func_name = func.name.as_str();

    // Skip dunder methods and test functions
    if func_name.starts_with("__") || func_name.starts_with("test_") {
        return;
    }

    // Check parameters (func.args instead of func.parameters)
    for arg in &func.args.args {
        let arg_name = arg.def.arg.as_str();

        if arg_name == "self" || arg_name == "cls" {
            continue;
        }

        if arg.def.annotation.is_none() {
            diagnostics.push(
                Diagnostic::warning(
                    "ZLUP006",
                    format!("Parameter '{}' is missing type annotation", arg_name),
                    make_location(func.range, filename, source),
                )
                .with_suggestion(format!("Add type annotation: {}: <type>", arg_name))
                .with_source_context(source),
            );
        }
    }

    // Check return type
    if func.returns.is_none() && !is_trivial_body(&func.body) {
        diagnostics.push(
            Diagnostic::warning(
                "ZLUP006",
                format!(
                    "Async function '{}' is missing return type annotation",
                    func_name
                ),
                make_location(func.range, filename, source),
            )
            .with_suggestion(format!(
                "Add return type: async def {}(...) -> <type>:",
                func_name
            ))
            .with_source_context(source),
        );
    }
}

fn is_trivial_body(body: &[Stmt]) -> bool {
    if body.is_empty() {
        return true;
    }
    if body.len() == 1 {
        match &body[0] {
            Stmt::Pass(_) => return true,
            Stmt::Expr(expr_stmt) => {
                // Check for docstring or ellipsis
                if let rustpython_parser::ast::Expr::Constant(c) = expr_stmt.value.as_ref() {
                    match &c.value {
                        rustpython_parser::ast::Constant::Str(_) => return true,
                        rustpython_parser::ast::Constant::Ellipsis => return true,
                        _ => {}
                    }
                }
            }
            _ => {}
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
        ZLUP006MissingTypes.check(&parsed, "<test>", source)
    }

    #[test]
    fn test_missing_param_type() {
        let diagnostics = check_source(
            r#"
def foo(x):
    return x + 1
"#,
        );
        assert!(!diagnostics.is_empty());
        assert!(
            diagnostics
                .iter()
                .any(|d| d.message.contains("Parameter 'x'"))
        );
    }

    #[test]
    fn test_missing_return_type() {
        let diagnostics = check_source(
            r#"
def foo(x: int):
    return x + 1
"#,
        );
        assert!(!diagnostics.is_empty());
        assert!(
            diagnostics
                .iter()
                .any(|d| d.message.contains("return type"))
        );
    }

    #[test]
    fn test_fully_typed() {
        let diagnostics = check_source(
            r#"
def foo(x: int) -> int:
    return x + 1
"#,
        );
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn test_stub_function() {
        let diagnostics = check_source(
            r#"
def foo(x):
    pass
"#,
        );
        // Still warns about parameter but not return (trivial body)
        assert!(diagnostics.iter().any(|d| d.message.contains("Parameter")));
        assert!(
            !diagnostics
                .iter()
                .any(|d| d.message.contains("return type"))
        );
    }

    #[test]
    fn test_skip_self() {
        let diagnostics = check_source(
            r#"
class Foo:
    def bar(self, x: int) -> int:
        return x
"#,
        );
        assert!(diagnostics.is_empty());
    }
}
