//! ZLUP009: Require assertions in non-trivial functions.

use rustpython_parser::ast::{Mod, Stmt};

use super::super::diagnostic::{Diagnostic, Severity};
use super::LintRule;

/// Minimum number of statements in a function body to require assertions.
const MIN_STATEMENTS_FOR_ASSERTION: usize = 5;

/// Detects functions without assertions that should have them.
pub struct ZLUP009AssertionDensity;

impl LintRule for ZLUP009AssertionDensity {
    fn id(&self) -> &'static str {
        "ZLUP009"
    }

    fn name(&self) -> &'static str {
        "assertion-density"
    }

    fn description(&self) -> &'static str {
        "Non-trivial functions should contain assertions to validate preconditions, \
         postconditions, and invariants. This helps catch errors early."
    }

    fn severity(&self) -> Severity {
        Severity::Info
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
            check_function(&func.name, &func.body, func.range, filename, source, diagnostics);
            // Check nested functions
            for s in &func.body {
                check_stmt(s, filename, source, diagnostics);
            }
        }
        Stmt::AsyncFunctionDef(func) => {
            check_function(&func.name, &func.body, func.range, filename, source, diagnostics);
            for s in &func.body {
                check_stmt(s, filename, source, diagnostics);
            }
        }
        Stmt::ClassDef(class) => {
            for s in &class.body {
                check_stmt(s, filename, source, diagnostics);
            }
        }
        _ => {}
    }
}

fn check_function(
    name: &str,
    body: &[Stmt],
    range: rustpython_parser::text_size::TextRange,
    filename: &str,
    source: &str,
    diagnostics: &mut Vec<Diagnostic>,
) {
    // Skip test functions, dunder methods, and trivial functions
    if name.starts_with("test_") || name.starts_with("__") {
        return;
    }

    // Count statements (flattening simple control flow)
    let stmt_count = count_statements(body);

    if stmt_count < MIN_STATEMENTS_FOR_ASSERTION {
        return;
    }

    // Check if function has any assertions
    let has_assertion = has_assertion_in_body(body);

    if !has_assertion {
        diagnostics.push(
            Diagnostic::new(
                "ZLUP009",
                format!(
                    "Function '{}' has {} statements but no assertions",
                    name, stmt_count
                ),
                Severity::Info,
                super::make_location(range, filename, source),
            )
            .with_suggestion(
                "Add assert statements to validate preconditions, postconditions, or invariants",
            )
            .with_source_context(source),
        );
    }
}

fn count_statements(body: &[Stmt]) -> usize {
    let mut count = 0;

    for stmt in body {
        count += 1;

        // Count statements in nested blocks
        match stmt {
            Stmt::If(if_stmt) => {
                count += count_statements(&if_stmt.body);
                count += count_statements(&if_stmt.orelse);
            }
            Stmt::For(for_stmt) => {
                count += count_statements(&for_stmt.body);
                count += count_statements(&for_stmt.orelse);
            }
            Stmt::While(while_stmt) => {
                count += count_statements(&while_stmt.body);
                count += count_statements(&while_stmt.orelse);
            }
            Stmt::With(with_stmt) => {
                count += count_statements(&with_stmt.body);
            }
            Stmt::Try(try_stmt) => {
                count += count_statements(&try_stmt.body);
                count += count_statements(&try_stmt.orelse);
                count += count_statements(&try_stmt.finalbody);
            }
            _ => {}
        }
    }

    count
}

fn has_assertion_in_body(body: &[Stmt]) -> bool {
    for stmt in body {
        if matches!(stmt, Stmt::Assert(_)) {
            return true;
        }

        // Check nested blocks
        let has_nested = match stmt {
            Stmt::If(if_stmt) => {
                has_assertion_in_body(&if_stmt.body) || has_assertion_in_body(&if_stmt.orelse)
            }
            Stmt::For(for_stmt) => {
                has_assertion_in_body(&for_stmt.body) || has_assertion_in_body(&for_stmt.orelse)
            }
            Stmt::While(while_stmt) => {
                has_assertion_in_body(&while_stmt.body) || has_assertion_in_body(&while_stmt.orelse)
            }
            Stmt::With(with_stmt) => has_assertion_in_body(&with_stmt.body),
            Stmt::Try(try_stmt) => {
                has_assertion_in_body(&try_stmt.body)
                    || has_assertion_in_body(&try_stmt.orelse)
                    || has_assertion_in_body(&try_stmt.finalbody)
            }
            Stmt::FunctionDef(_) | Stmt::AsyncFunctionDef(_) => {
                // Don't look into nested function definitions
                false
            }
            _ => false,
        };

        if has_nested {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustpython_parser::{parse, Mode};

    fn check_source(source: &str) -> Vec<Diagnostic> {
        let parsed = parse(source, Mode::Module, "<test>").unwrap();
        ZLUP009AssertionDensity.check(&parsed, "<test>", source)
    }

    #[test]
    fn test_small_function() {
        let diagnostics = check_source(
            r#"
def foo():
    x = 1
    return x
"#,
        );
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn test_large_function_without_assertion() {
        let diagnostics = check_source(
            r#"
def process(data):
    x = 1
    y = 2
    z = 3
    a = x + y
    b = y + z
    return a + b
"#,
        );
        assert!(!diagnostics.is_empty());
        assert!(diagnostics[0].message.contains("no assertions"));
    }

    #[test]
    fn test_large_function_with_assertion() {
        let diagnostics = check_source(
            r#"
def process(data):
    assert data is not None
    x = 1
    y = 2
    z = 3
    a = x + y
    b = y + z
    return a + b
"#,
        );
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn test_skip_test_functions() {
        let diagnostics = check_source(
            r#"
def test_something():
    x = 1
    y = 2
    z = 3
    a = x + y
    b = y + z
    return a + b
"#,
        );
        assert!(diagnostics.is_empty());
    }
}
