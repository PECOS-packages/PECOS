//! ZLUP010: Restrict mutable global state.

use rustpython_parser::ast::{Mod, Stmt};

use super::super::diagnostic::{Diagnostic, Severity};
use super::{make_location, LintRule};

/// Names that are allowed as module-level constants (typically UPPER_CASE).
fn is_constant_name(name: &str) -> bool {
    // Allow UPPER_CASE names as constants
    name.chars().all(|c| c.is_uppercase() || c == '_' || c.is_numeric())
        && name.chars().next().is_some_and(|c| c.is_alphabetic() || c == '_')
}

/// Known safe module-level definitions.
fn is_safe_assignment(name: &str) -> bool {
    // Type aliases and protocol definitions
    name.ends_with("Type")
        || name.ends_with("Protocol")
        || name.ends_with("T")  // Generic type vars
        || name == "__all__"
        || name == "__version__"
        || name == "__author__"
}

/// Detects mutable global state that can cause issues in concurrent/quantum code.
pub struct ZLUP010GlobalState;

impl LintRule for ZLUP010GlobalState {
    fn id(&self) -> &'static str {
        "ZLUP010"
    }

    fn name(&self) -> &'static str {
        "global-state"
    }

    fn description(&self) -> &'static str {
        "Mutable global state is prohibited. Use constants (UPPER_CASE names) \
         or pass state explicitly through function parameters."
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
            check_module_level_stmt(stmt, filename, source, &mut diagnostics);
        }

        diagnostics
    }
}

fn check_module_level_stmt(
    stmt: &Stmt,
    filename: &str,
    source: &str,
    diagnostics: &mut Vec<Diagnostic>,
) {
    match stmt {
        // Check for module-level variable assignments
        Stmt::Assign(assign) => {
            for target in &assign.targets {
                if let rustpython_parser::ast::Expr::Name(name) = target {
                    let var_name = name.id.as_str();

                    // Allow constants and safe assignments
                    if !is_constant_name(var_name) && !is_safe_assignment(var_name) {
                        diagnostics.push(
                            Diagnostic::warning(
                                "ZLUP010",
                                format!(
                                    "Module-level variable '{}' creates mutable global state",
                                    var_name
                                ),
                                make_location(assign.range, filename, source),
                            )
                            .with_suggestion(format!(
                                "Use UPPER_CASE for constants (e.g., {}) or pass as function parameter",
                                var_name.to_uppercase()
                            ))
                            .with_source_context(source),
                        );
                    }
                }
            }
        }

        // Check for annotated assignments at module level
        Stmt::AnnAssign(ann) => {
            if let rustpython_parser::ast::Expr::Name(name) = ann.target.as_ref() {
                let var_name = name.id.as_str();

                if !is_constant_name(var_name) && !is_safe_assignment(var_name) {
                    diagnostics.push(
                        Diagnostic::warning(
                            "ZLUP010",
                            format!(
                                "Module-level variable '{}' creates mutable global state",
                                var_name
                            ),
                            make_location(ann.range, filename, source),
                        )
                        .with_suggestion(format!(
                            "Use UPPER_CASE for constants (e.g., {}) or pass as function parameter",
                            var_name.to_uppercase()
                        ))
                        .with_source_context(source),
                    );
                }
            }
        }

        // Check for use of 'global' keyword in functions
        Stmt::FunctionDef(func) => {
            check_global_in_function(&func.body, filename, source, diagnostics);
        }
        Stmt::AsyncFunctionDef(func) => {
            check_global_in_function(&func.body, filename, source, diagnostics);
        }

        // Skip class definitions, imports, etc.
        _ => {}
    }
}

fn check_global_in_function(
    body: &[Stmt],
    filename: &str,
    source: &str,
    diagnostics: &mut Vec<Diagnostic>,
) {
    for stmt in body {
        match stmt {
            Stmt::Global(global_stmt) => {
                for name in &global_stmt.names {
                    diagnostics.push(
                        Diagnostic::warning(
                            "ZLUP010",
                            format!("Use of 'global {}' creates hidden state dependencies", name),
                            make_location(global_stmt.range, filename, source),
                        )
                        .with_suggestion("Pass the value as a function parameter instead")
                        .with_source_context(source),
                    );
                }
            }
            Stmt::If(if_stmt) => {
                check_global_in_function(&if_stmt.body, filename, source, diagnostics);
                check_global_in_function(&if_stmt.orelse, filename, source, diagnostics);
            }
            Stmt::For(for_stmt) => {
                check_global_in_function(&for_stmt.body, filename, source, diagnostics);
            }
            Stmt::While(while_stmt) => {
                check_global_in_function(&while_stmt.body, filename, source, diagnostics);
            }
            Stmt::With(with_stmt) => {
                check_global_in_function(&with_stmt.body, filename, source, diagnostics);
            }
            Stmt::Try(try_stmt) => {
                check_global_in_function(&try_stmt.body, filename, source, diagnostics);
                check_global_in_function(&try_stmt.orelse, filename, source, diagnostics);
                check_global_in_function(&try_stmt.finalbody, filename, source, diagnostics);
            }
            Stmt::FunctionDef(func) => {
                // Check nested functions
                check_global_in_function(&func.body, filename, source, diagnostics);
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustpython_parser::{parse, Mode};

    fn check_source(source: &str) -> Vec<Diagnostic> {
        let parsed = parse(source, Mode::Module, "<test>").unwrap();
        ZLUP010GlobalState.check(&parsed, "<test>", source)
    }

    #[test]
    fn test_constant_allowed() {
        let diagnostics = check_source("MAX_SIZE = 100");
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn test_lowercase_variable_flagged() {
        let diagnostics = check_source("counter = 0");
        assert!(!diagnostics.is_empty());
        assert!(diagnostics[0].message.contains("mutable global state"));
    }

    #[test]
    fn test_global_keyword_flagged() {
        let diagnostics = check_source(
            r#"
counter = 0

def increment():
    global counter
    counter += 1
"#,
        );
        // Should flag both the module-level variable and the global statement
        assert!(diagnostics.len() >= 2);
    }

    #[test]
    fn test_dunder_allowed() {
        let diagnostics = check_source("__all__ = ['foo', 'bar']");
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn test_function_def_ok() {
        let diagnostics = check_source(
            r#"
def foo():
    x = 1
    return x
"#,
        );
        assert!(diagnostics.is_empty());
    }
}
