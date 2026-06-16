//! ZLUP004: Detect dynamic dispatch and runtime evaluation.

use rustpython_parser::ast::{Expr, Mod, Stmt};

use super::{make_location, LintRule};
use super::super::diagnostic::{Diagnostic, Severity};

/// Functions that enable dynamic dispatch or code execution.
const DYNAMIC_FUNCTIONS: &[(&str, &str)] = &[
    ("eval", "executes arbitrary code at runtime"),
    ("exec", "executes arbitrary code at runtime"),
    ("compile", "compiles code at runtime"),
    ("getattr", "dynamically resolves attributes"),
    ("setattr", "dynamically sets attributes"),
    ("delattr", "dynamically deletes attributes"),
    ("hasattr", "dynamically checks attributes"),
    ("__import__", "dynamically imports modules"),
];

/// Detects dynamic dispatch and runtime code execution.
pub struct ZLUP004DynamicDispatch;

impl LintRule for ZLUP004DynamicDispatch {
    fn id(&self) -> &'static str {
        "ZLUP004"
    }

    fn name(&self) -> &'static str {
        "dynamic-dispatch"
    }

    fn description(&self) -> &'static str {
        "Dynamic dispatch and runtime code execution are prohibited. \
         All function calls and attribute accesses must be statically determinable."
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
            check_stmt(stmt, filename, source, &mut diagnostics);
        }

        diagnostics
    }
}

fn check_stmt(stmt: &Stmt, filename: &str, source: &str, diagnostics: &mut Vec<Diagnostic>) {
    match stmt {
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
        Stmt::For(for_stmt) => {
            check_expr(&for_stmt.iter, filename, source, diagnostics);
            for s in &for_stmt.body {
                check_stmt(s, filename, source, diagnostics);
            }
            for s in &for_stmt.orelse {
                check_stmt(s, filename, source, diagnostics);
            }
        }
        Stmt::While(while_stmt) => {
            check_expr(&while_stmt.test, filename, source, diagnostics);
            for s in &while_stmt.body {
                check_stmt(s, filename, source, diagnostics);
            }
            for s in &while_stmt.orelse {
                check_stmt(s, filename, source, diagnostics);
            }
        }
        Stmt::If(if_stmt) => {
            check_expr(&if_stmt.test, filename, source, diagnostics);
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
            for s in &try_stmt.orelse {
                check_stmt(s, filename, source, diagnostics);
            }
            for s in &try_stmt.finalbody {
                check_stmt(s, filename, source, diagnostics);
            }
        }
        Stmt::Expr(expr_stmt) => {
            check_expr(&expr_stmt.value, filename, source, diagnostics);
        }
        Stmt::Assign(assign) => {
            check_expr(&assign.value, filename, source, diagnostics);
        }
        Stmt::AugAssign(aug) => {
            check_expr(&aug.value, filename, source, diagnostics);
        }
        Stmt::AnnAssign(ann) => {
            if let Some(value) = &ann.value {
                check_expr(value, filename, source, diagnostics);
            }
        }
        Stmt::Return(ret) => {
            if let Some(value) = &ret.value {
                check_expr(value, filename, source, diagnostics);
            }
        }
        _ => {}
    }
}

fn check_expr(expr: &Expr, filename: &str, source: &str, diagnostics: &mut Vec<Diagnostic>) {
    match expr {
        Expr::Call(call) => {
            // Check for calls to dynamic functions
            if let Expr::Name(name) = call.func.as_ref() {
                let func_name = name.id.as_str();
                if let Some((_, reason)) = DYNAMIC_FUNCTIONS.iter().find(|(n, _)| *n == func_name) {
                    let suggestion = get_suggestion(func_name);
                    diagnostics.push(
                        Diagnostic::error(
                            "ZLUP004",
                            format!("'{}()' {}", func_name, reason),
                            make_location(call.range, filename, source),
                        )
                        .with_suggestion(suggestion)
                        .with_source_context(source),
                    );
                }
            }

            // Check for calling subscripted expressions (dynamic dispatch)
            if let Expr::Subscript(_) = call.func.as_ref() {
                diagnostics.push(
                    Diagnostic::error(
                        "ZLUP004",
                        "Calling a subscripted expression creates dynamic dispatch",
                        make_location(call.range, filename, source),
                    )
                    .with_suggestion("Use explicit function calls or match statements instead")
                    .with_source_context(source),
                );
            }

            // Recurse into callee and arguments
            check_expr(&call.func, filename, source, diagnostics);
            for arg in &call.args {
                check_expr(arg, filename, source, diagnostics);
            }
        }

        // Recurse into sub-expressions
        Expr::BinOp(binop) => {
            check_expr(&binop.left, filename, source, diagnostics);
            check_expr(&binop.right, filename, source, diagnostics);
        }
        Expr::UnaryOp(unary) => {
            check_expr(&unary.operand, filename, source, diagnostics);
        }
        Expr::Compare(cmp) => {
            check_expr(&cmp.left, filename, source, diagnostics);
            for comparator in &cmp.comparators {
                check_expr(comparator, filename, source, diagnostics);
            }
        }
        Expr::BoolOp(boolop) => {
            for value in &boolop.values {
                check_expr(value, filename, source, diagnostics);
            }
        }
        Expr::IfExp(ifexp) => {
            check_expr(&ifexp.test, filename, source, diagnostics);
            check_expr(&ifexp.body, filename, source, diagnostics);
            check_expr(&ifexp.orelse, filename, source, diagnostics);
        }
        Expr::List(list) => {
            for elt in &list.elts {
                check_expr(elt, filename, source, diagnostics);
            }
        }
        Expr::Tuple(tuple) => {
            for elt in &tuple.elts {
                check_expr(elt, filename, source, diagnostics);
            }
        }
        Expr::Dict(dict) => {
            for key in dict.keys.iter().flatten() {
                check_expr(key, filename, source, diagnostics);
            }
            for value in &dict.values {
                check_expr(value, filename, source, diagnostics);
            }
        }
        Expr::Subscript(sub) => {
            check_expr(&sub.value, filename, source, diagnostics);
            check_expr(&sub.slice, filename, source, diagnostics);
        }
        Expr::Attribute(attr) => {
            check_expr(&attr.value, filename, source, diagnostics);
        }
        _ => {}
    }
}

fn get_suggestion(func_name: &str) -> &'static str {
    match func_name {
        "eval" | "exec" | "compile" => "Use direct function calls or predefined operations",
        "getattr" => "Use direct attribute access (obj.attr) or a dictionary lookup",
        "setattr" => "Use direct attribute assignment (obj.attr = value)",
        "delattr" => "Use direct attribute deletion (del obj.attr)",
        "hasattr" => "Use direct attribute access with try/except or a dictionary",
        "__import__" => "Use regular import statements",
        _ => "Avoid dynamic dispatch",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustpython_parser::{parse, Mode};

    fn check_source(source: &str) -> Vec<Diagnostic> {
        let parsed = parse(source, Mode::Module, "<test>").unwrap();
        ZLUP004DynamicDispatch.check(&parsed, "<test>", source)
    }

    #[test]
    fn test_eval() {
        let diagnostics = check_source(
            r#"
result = eval("1 + 2")
"#,
        );
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].message.contains("eval()"));
    }

    #[test]
    fn test_getattr() {
        let diagnostics = check_source(
            r#"
value = getattr(obj, "attr")
"#,
        );
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].message.contains("getattr()"));
    }

    #[test]
    fn test_subscript_call() {
        let diagnostics = check_source(
            r#"
func_table = {"a": func_a, "b": func_b}
func_table["a"]()
"#,
        );
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].message.contains("subscripted expression"));
    }

    #[test]
    fn test_normal_function_call() {
        let diagnostics = check_source(
            r#"
def foo():
    pass

foo()
"#,
        );
        assert!(diagnostics.is_empty());
    }
}
