//! ZLUP007: Detect overly complex control flow.

use rustpython_parser::ast::{self, Expr, Mod, Stmt};

use super::super::diagnostic::{Diagnostic, Severity};
use super::{LintRule, make_location};

/// Detects functions with overly complex control flow.
pub struct ZLUP007ComplexControlFlow {
    max_complexity: u32,
}

impl ZLUP007ComplexControlFlow {
    pub fn new(max_complexity: u32) -> Self {
        Self { max_complexity }
    }
}

impl LintRule for ZLUP007ComplexControlFlow {
    fn id(&self) -> &'static str {
        "ZLUP007"
    }

    fn name(&self) -> &'static str {
        "complex-control-flow"
    }

    fn description(&self) -> &'static str {
        "Functions should have a cyclomatic complexity below the threshold. \
         High complexity makes code harder to analyze and test."
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
            check_stmt(
                stmt,
                filename,
                source,
                self.max_complexity,
                &mut diagnostics,
            );
        }

        diagnostics
    }
}

fn check_stmt(
    stmt: &Stmt,
    filename: &str,
    source: &str,
    max_complexity: u32,
    diagnostics: &mut Vec<Diagnostic>,
) {
    match stmt {
        Stmt::FunctionDef(func) => {
            let complexity = calculate_complexity(&func.body);
            if complexity > max_complexity {
                diagnostics.push(
                    Diagnostic::warning(
                        "ZLUP007",
                        format!(
                            "Function '{}' has cyclomatic complexity {} (max: {})",
                            func.name, complexity, max_complexity
                        ),
                        make_location(func.range, filename, source),
                    )
                    .with_suggestion(
                        "Break the function into smaller functions or simplify the control flow",
                    )
                    .with_source_context(source),
                );
            }
            // Check nested functions
            for s in &func.body {
                check_stmt(s, filename, source, max_complexity, diagnostics);
            }
        }
        Stmt::AsyncFunctionDef(func) => {
            let complexity = calculate_complexity(&func.body);
            if complexity > max_complexity {
                diagnostics.push(
                    Diagnostic::warning(
                        "ZLUP007",
                        format!(
                            "Async function '{}' has cyclomatic complexity {} (max: {})",
                            func.name, complexity, max_complexity
                        ),
                        make_location(func.range, filename, source),
                    )
                    .with_suggestion(
                        "Break the function into smaller functions or simplify the control flow",
                    )
                    .with_source_context(source),
                );
            }
            for s in &func.body {
                check_stmt(s, filename, source, max_complexity, diagnostics);
            }
        }
        Stmt::ClassDef(class) => {
            for s in &class.body {
                check_stmt(s, filename, source, max_complexity, diagnostics);
            }
        }
        _ => {}
    }
}

/// Calculate cyclomatic complexity of a function body.
///
/// Complexity = 1 + number of decision points
///
/// Decision points:
/// - if/elif
/// - for/while loops
/// - and/or in boolean expressions
/// - except handlers
/// - assert statements
/// - ternary expressions (if expressions)
/// - comprehensions with if clauses
/// - match cases
fn calculate_complexity(body: &[Stmt]) -> u32 {
    let mut decision_points = 0;

    for stmt in body {
        decision_points += count_decision_points_stmt(stmt);
    }

    1 + decision_points
}

fn count_decision_points_stmt(stmt: &Stmt) -> u32 {
    let mut count = 0;

    match stmt {
        Stmt::If(if_stmt) => {
            count += 1; // The if itself
            count += count_decision_points_expr(&if_stmt.test);
            for s in &if_stmt.body {
                count += count_decision_points_stmt(s);
            }
            for s in &if_stmt.orelse {
                count += count_decision_points_stmt(s);
            }
        }
        Stmt::For(for_stmt) => {
            count += 1; // The for loop
            for s in &for_stmt.body {
                count += count_decision_points_stmt(s);
            }
            for s in &for_stmt.orelse {
                count += count_decision_points_stmt(s);
            }
        }
        Stmt::While(while_stmt) => {
            count += 1; // The while loop
            count += count_decision_points_expr(&while_stmt.test);
            for s in &while_stmt.body {
                count += count_decision_points_stmt(s);
            }
            for s in &while_stmt.orelse {
                count += count_decision_points_stmt(s);
            }
        }
        Stmt::Try(try_stmt) => {
            count += try_stmt.handlers.len() as u32; // Each handler is a decision point
            for s in &try_stmt.body {
                count += count_decision_points_stmt(s);
            }
            for handler in &try_stmt.handlers {
                let ast::ExceptHandler::ExceptHandler(h) = handler;
                for s in &h.body {
                    count += count_decision_points_stmt(s);
                }
            }
            for s in &try_stmt.orelse {
                count += count_decision_points_stmt(s);
            }
            for s in &try_stmt.finalbody {
                count += count_decision_points_stmt(s);
            }
        }
        Stmt::Assert(_) => {
            count += 1; // Assert is a decision point
        }
        Stmt::Match(match_stmt) => {
            count += match_stmt.cases.len() as u32; // Each case is a decision point
            for case in &match_stmt.cases {
                for s in &case.body {
                    count += count_decision_points_stmt(s);
                }
            }
        }
        Stmt::With(with_stmt) => {
            for s in &with_stmt.body {
                count += count_decision_points_stmt(s);
            }
        }
        Stmt::Expr(expr_stmt) => {
            count += count_decision_points_expr(&expr_stmt.value);
        }
        Stmt::Assign(assign) => {
            count += count_decision_points_expr(&assign.value);
        }
        Stmt::AugAssign(aug) => {
            count += count_decision_points_expr(&aug.value);
        }
        Stmt::AnnAssign(ann) => {
            if let Some(value) = &ann.value {
                count += count_decision_points_expr(value);
            }
        }
        Stmt::Return(ret) => {
            if let Some(value) = &ret.value {
                count += count_decision_points_expr(value);
            }
        }
        _ => {}
    }

    count
}

fn count_decision_points_expr(expr: &Expr) -> u32 {
    let mut count = 0;

    match expr {
        Expr::BoolOp(boolop) => {
            // Each 'and' or 'or' adds (len(values) - 1) decision points
            count += (boolop.values.len() - 1) as u32;
            for value in &boolop.values {
                count += count_decision_points_expr(value);
            }
        }
        Expr::IfExp(_) => {
            count += 1; // Ternary expression
        }
        Expr::ListComp(comp) => {
            for generator in &comp.generators {
                count += generator.ifs.len() as u32;
            }
        }
        Expr::SetComp(comp) => {
            for generator in &comp.generators {
                count += generator.ifs.len() as u32;
            }
        }
        Expr::DictComp(comp) => {
            for generator in &comp.generators {
                count += generator.ifs.len() as u32;
            }
        }
        Expr::GeneratorExp(gen_expr) => {
            for generator in &gen_expr.generators {
                count += generator.ifs.len() as u32;
            }
        }
        Expr::Call(call) => {
            for arg in &call.args {
                count += count_decision_points_expr(arg);
            }
        }
        Expr::BinOp(binop) => {
            count += count_decision_points_expr(&binop.left);
            count += count_decision_points_expr(&binop.right);
        }
        Expr::UnaryOp(unary) => {
            count += count_decision_points_expr(&unary.operand);
        }
        Expr::Compare(cmp) => {
            count += count_decision_points_expr(&cmp.left);
            for comparator in &cmp.comparators {
                count += count_decision_points_expr(comparator);
            }
        }
        _ => {}
    }

    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustpython_parser::{Mode, parse};

    fn check_source_with_max(source: &str, max: u32) -> Vec<Diagnostic> {
        let parsed = parse(source, Mode::Module, "<test>").unwrap();
        ZLUP007ComplexControlFlow::new(max).check(&parsed, "<test>", source)
    }

    #[test]
    fn test_simple_function() {
        let diagnostics = check_source_with_max(
            r#"
def foo():
    return 1
"#,
            10,
        );
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn test_complex_function() {
        let diagnostics = check_source_with_max(
            r#"
def foo(x):
    if x > 0:
        if x > 10:
            if x > 100:
                return 3
            return 2
        return 1
    elif x < 0:
        if x < -10:
            return -2
        return -1
    else:
        return 0
"#,
            3,
        );
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].message.contains("cyclomatic complexity"));
    }

    #[test]
    fn test_boolean_operators() {
        let diagnostics = check_source_with_max(
            r#"
def foo(a, b, c, d):
    if a and b and c and d:
        return 1
    return 0
"#,
            3,
        );
        // 1 (base) + 1 (if) + 3 (and operators) = 5 > 3
        assert_eq!(diagnostics.len(), 1);
    }

    #[test]
    fn test_comprehension_with_if() {
        let diagnostics = check_source_with_max(
            r#"
def foo(items):
    return [x for x in items if x > 0 if x < 100]
"#,
            2,
        );
        // 1 (base) + 2 (if clauses in comprehension) = 3 > 2
        assert_eq!(diagnostics.len(), 1);
    }
}
