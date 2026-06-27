//! ZLUP002: Detect recursive function calls.

use std::collections::{HashMap, HashSet};

use rustpython_parser::ast::{self, Expr, Mod, Stmt};

use super::super::diagnostic::{Diagnostic, Severity};
use super::{LintRule, make_location};

/// Detects recursive function calls (call graph cycles).
pub struct ZLUP002Recursion;

impl LintRule for ZLUP002Recursion {
    fn id(&self) -> &'static str {
        "ZLUP002"
    }

    fn name(&self) -> &'static str {
        "recursion"
    }

    fn description(&self) -> &'static str {
        "Recursive function calls are prohibited. All call graphs must be acyclic \
         to ensure bounded stack usage and predictable execution."
    }

    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn check(&self, parsed: &Mod, filename: &str, source: &str) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();

        let Mod::Module(module) = parsed else {
            return diagnostics;
        };

        // Collect function definitions
        let mut functions: HashMap<String, &ast::StmtFunctionDef> = HashMap::new();
        for stmt in &module.body {
            if let Stmt::FunctionDef(func) = stmt {
                functions.insert(func.name.to_string(), func);
            }
        }

        // Build call graph
        let mut call_graph: HashMap<String, HashSet<String>> = HashMap::new();
        for (name, func) in &functions {
            let calls = collect_calls(&func.body, &functions);
            call_graph.insert(name.clone(), calls);
        }

        // Detect cycles
        let mut reported: HashSet<String> = HashSet::new();
        for func_name in functions.keys() {
            if reported.contains(func_name) {
                continue;
            }

            if let Some(cycle) = find_cycle(func_name, &call_graph) {
                let func = functions.get(func_name).unwrap();

                if cycle.len() == 1 {
                    // Direct recursion
                    diagnostics.push(
                        Diagnostic::error(
                            "ZLUP002",
                            format!("function '{}' calls itself (direct recursion)", func_name),
                            make_location(func.range, filename, source),
                        )
                        .with_suggestion(
                            "Replace recursion with iteration using a loop with a fixed upper bound",
                        )
                        .with_source_context(source),
                    );
                } else {
                    // Indirect recursion
                    let cycle_str = cycle
                        .iter()
                        .chain(std::iter::once(&cycle[0]))
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(" -> ");

                    diagnostics.push(
                        Diagnostic::error(
                            "ZLUP002",
                            format!(
                                "function '{}' is part of a recursive cycle: {}",
                                func_name, cycle_str
                            ),
                            make_location(func.range, filename, source),
                        )
                        .with_suggestion(
                            "Break the cycle by restructuring the call graph or using iteration",
                        )
                        .with_source_context(source),
                    );
                }

                // Mark all functions in the cycle as reported
                for name in &cycle {
                    reported.insert(name.clone());
                }
            }
        }

        diagnostics
    }
}

fn collect_calls(
    body: &[Stmt],
    known_funcs: &HashMap<String, &ast::StmtFunctionDef>,
) -> HashSet<String> {
    let mut calls = HashSet::new();

    for stmt in body {
        collect_calls_from_stmt(stmt, known_funcs, &mut calls);
    }

    calls
}

fn collect_calls_from_stmt(
    stmt: &Stmt,
    known_funcs: &HashMap<String, &ast::StmtFunctionDef>,
    calls: &mut HashSet<String>,
) {
    match stmt {
        Stmt::Expr(expr_stmt) => {
            collect_calls_from_expr(&expr_stmt.value, known_funcs, calls);
        }
        Stmt::Return(ret) => {
            if let Some(value) = &ret.value {
                collect_calls_from_expr(value, known_funcs, calls);
            }
        }
        Stmt::Assign(assign) => {
            collect_calls_from_expr(&assign.value, known_funcs, calls);
        }
        Stmt::AugAssign(aug) => {
            collect_calls_from_expr(&aug.value, known_funcs, calls);
        }
        Stmt::AnnAssign(ann) => {
            if let Some(value) = &ann.value {
                collect_calls_from_expr(value, known_funcs, calls);
            }
        }
        Stmt::For(for_stmt) => {
            collect_calls_from_expr(&for_stmt.iter, known_funcs, calls);
            for s in &for_stmt.body {
                collect_calls_from_stmt(s, known_funcs, calls);
            }
            for s in &for_stmt.orelse {
                collect_calls_from_stmt(s, known_funcs, calls);
            }
        }
        Stmt::While(while_stmt) => {
            collect_calls_from_expr(&while_stmt.test, known_funcs, calls);
            for s in &while_stmt.body {
                collect_calls_from_stmt(s, known_funcs, calls);
            }
            for s in &while_stmt.orelse {
                collect_calls_from_stmt(s, known_funcs, calls);
            }
        }
        Stmt::If(if_stmt) => {
            collect_calls_from_expr(&if_stmt.test, known_funcs, calls);
            for s in &if_stmt.body {
                collect_calls_from_stmt(s, known_funcs, calls);
            }
            for s in &if_stmt.orelse {
                collect_calls_from_stmt(s, known_funcs, calls);
            }
        }
        Stmt::With(with_stmt) => {
            for s in &with_stmt.body {
                collect_calls_from_stmt(s, known_funcs, calls);
            }
        }
        Stmt::Try(try_stmt) => {
            for s in &try_stmt.body {
                collect_calls_from_stmt(s, known_funcs, calls);
            }
            for s in &try_stmt.orelse {
                collect_calls_from_stmt(s, known_funcs, calls);
            }
            for s in &try_stmt.finalbody {
                collect_calls_from_stmt(s, known_funcs, calls);
            }
        }
        _ => {}
    }
}

fn collect_calls_from_expr(
    expr: &Expr,
    known_funcs: &HashMap<String, &ast::StmtFunctionDef>,
    calls: &mut HashSet<String>,
) {
    match expr {
        Expr::Call(call) => {
            if let Expr::Name(name) = call.func.as_ref() {
                let func_name = name.id.to_string();
                if known_funcs.contains_key(&func_name) {
                    calls.insert(func_name);
                }
            }
            // Recurse into arguments
            for arg in &call.args {
                collect_calls_from_expr(arg, known_funcs, calls);
            }
        }
        Expr::BinOp(binop) => {
            collect_calls_from_expr(&binop.left, known_funcs, calls);
            collect_calls_from_expr(&binop.right, known_funcs, calls);
        }
        Expr::UnaryOp(unary) => {
            collect_calls_from_expr(&unary.operand, known_funcs, calls);
        }
        Expr::Compare(cmp) => {
            collect_calls_from_expr(&cmp.left, known_funcs, calls);
            for comparator in &cmp.comparators {
                collect_calls_from_expr(comparator, known_funcs, calls);
            }
        }
        Expr::BoolOp(boolop) => {
            for value in &boolop.values {
                collect_calls_from_expr(value, known_funcs, calls);
            }
        }
        Expr::IfExp(ifexp) => {
            collect_calls_from_expr(&ifexp.test, known_funcs, calls);
            collect_calls_from_expr(&ifexp.body, known_funcs, calls);
            collect_calls_from_expr(&ifexp.orelse, known_funcs, calls);
        }
        Expr::List(list) => {
            for elt in &list.elts {
                collect_calls_from_expr(elt, known_funcs, calls);
            }
        }
        Expr::Tuple(tuple) => {
            for elt in &tuple.elts {
                collect_calls_from_expr(elt, known_funcs, calls);
            }
        }
        Expr::Subscript(sub) => {
            collect_calls_from_expr(&sub.value, known_funcs, calls);
            collect_calls_from_expr(&sub.slice, known_funcs, calls);
        }
        _ => {}
    }
}

fn find_cycle(start: &str, graph: &HashMap<String, HashSet<String>>) -> Option<Vec<String>> {
    let mut visited: HashSet<String> = HashSet::new();
    let mut path: Vec<String> = Vec::new();

    fn dfs(
        node: &str,
        graph: &HashMap<String, HashSet<String>>,
        visited: &mut HashSet<String>,
        path: &mut Vec<String>,
    ) -> Option<Vec<String>> {
        if let Some(pos) = path.iter().position(|n| n == node) {
            // Found a cycle
            return Some(path[pos..].to_vec());
        }

        if visited.contains(node) {
            return None;
        }

        visited.insert(node.to_string());
        path.push(node.to_string());

        if let Some(neighbors) = graph.get(node) {
            for neighbor in neighbors {
                if let Some(cycle) = dfs(neighbor, graph, visited, path) {
                    return Some(cycle);
                }
            }
        }

        path.pop();
        None
    }

    dfs(start, graph, &mut visited, &mut path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustpython_parser::{Mode, parse};

    fn check_source(source: &str) -> Vec<Diagnostic> {
        let parsed = parse(source, Mode::Module, "<test>").unwrap();
        ZLUP002Recursion.check(&parsed, "<test>", source)
    }

    #[test]
    fn test_direct_recursion() {
        let diagnostics = check_source(
            r#"
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"#,
        );
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].message.contains("direct recursion"));
    }

    #[test]
    fn test_indirect_recursion() {
        let diagnostics = check_source(
            r#"
def foo():
    bar()

def bar():
    foo()
"#,
        );
        // Should detect a cycle
        assert!(!diagnostics.is_empty());
        assert!(diagnostics[0].message.contains("recursive cycle"));
    }

    #[test]
    fn test_no_recursion() {
        let diagnostics = check_source(
            r#"
def foo():
    pass

def bar():
    foo()

def baz():
    bar()
"#,
        );
        assert!(diagnostics.is_empty());
    }
}
