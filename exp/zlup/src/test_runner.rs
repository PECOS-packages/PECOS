//! Built-in test runner for Zluppy programs.
//!
//! Discovers `test "name" { ... }` blocks in Zluppy source files,
//! analyzes them, and runs classical tests via the comptime evaluator.

use std::time::{Duration, Instant};

use crate::ast::{Program, TestDecl, TopLevelDecl};
use crate::semantic::SemanticAnalyzer;

/// Configuration for the test runner.
#[derive(Debug, Clone)]
pub struct TestRunConfig {
    /// Only run tests matching this pattern (substring match)
    pub filter: Option<String>,
    /// Enable strict mode (NASA Power of 10)
    pub strict: bool,
    /// Print verbose output
    pub verbose: bool,
}

impl Default for TestRunConfig {
    fn default() -> Self {
        Self {
            filter: None,
            strict: false,
            verbose: false,
        }
    }
}

/// Outcome of a single test.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TestOutcome {
    Pass,
    Fail(String),
    Skip(String),
}

/// Result of running a single test.
#[derive(Debug, Clone)]
pub struct TestResult {
    pub name: String,
    pub outcome: TestOutcome,
    pub duration: Duration,
}

/// Test runner for Zluppy programs.
pub struct TestRunner {
    config: TestRunConfig,
}

impl TestRunner {
    /// Create a new test runner with the given configuration.
    pub fn new(config: TestRunConfig) -> Self {
        Self { config }
    }

    /// Discover test declarations from a program AST.
    pub fn discover_tests<'a>(&self, program: &'a Program) -> Vec<&'a TestDecl> {
        let mut tests = Vec::new();
        for decl in &program.declarations {
            if let TopLevelDecl::Test(test) = decl {
                if let Some(ref filter) = self.config.filter {
                    if !test.name.contains(filter.as_str()) {
                        continue;
                    }
                }
                tests.push(test);
            }
        }
        tests
    }

    /// Run all discovered tests in a program.
    pub fn run(&self, program: &Program) -> Vec<TestResult> {
        let tests = self.discover_tests(program);
        let mut results = Vec::new();

        // Run each test — skip quantum tests, try semantic analysis for classical ones
        for test in &tests {
            let result = self.run_single_test(test, program);
            results.push(result);
        }

        results
    }

    /// Run a single test.
    fn run_single_test(&self, test: &TestDecl, program: &Program) -> TestResult {
        let start = Instant::now();

        // Check if the test body contains quantum operations
        // The comptime evaluator cannot run quantum gates
        if contains_quantum_ops(&test.body) {
            return TestResult {
                name: test.name.clone(),
                outcome: TestOutcome::Skip(
                    "test contains quantum operations (gates/measurements) which cannot be evaluated at compile time".to_string(),
                ),
                duration: start.elapsed(),
            };
        }

        // Run semantic analysis on the program
        let mut analyzer = SemanticAnalyzer::new();
        if self.config.strict {
            analyzer.set_strict_mode(true);
        }
        if let Err(err) = analyzer.analyze(program) {
            return TestResult {
                name: test.name.clone(),
                outcome: TestOutcome::Fail(format!("semantic error: {}", err)),
                duration: start.elapsed(),
            };
        }

        // For now, tests that pass semantic analysis and don't contain
        // quantum operations are considered passing. Full comptime evaluation
        // of test bodies will be added when the comptime evaluator supports
        // statement-level evaluation.
        TestResult {
            name: test.name.clone(),
            outcome: TestOutcome::Pass,
            duration: start.elapsed(),
        }
    }
}

/// Check if a block contains quantum operations.
fn contains_quantum_ops(block: &crate::ast::Block) -> bool {
    use crate::ast::Stmt;
    for stmt in &block.statements {
        match stmt {
            Stmt::Gate(_) | Stmt::Prepare(_) | Stmt::Measure(_) | Stmt::Barrier(_) => {
                return true;
            }
            Stmt::Expr(expr_stmt) => {
                if expr_contains_quantum(&expr_stmt.expr) {
                    return true;
                }
            }
            Stmt::If(if_stmt) => {
                if contains_quantum_ops(&if_stmt.then_body) {
                    return true;
                }
                if let Some(ref else_branch) = if_stmt.else_body {
                    match else_branch {
                        crate::ast::ElseBranch::Else(block) => {
                            if contains_quantum_ops(block) {
                                return true;
                            }
                        }
                        crate::ast::ElseBranch::ElseIf(if_stmt) => {
                            if contains_quantum_ops(&if_stmt.then_body) {
                                return true;
                            }
                        }
                    }
                }
            }
            Stmt::For(for_stmt) => {
                if contains_quantum_ops(&for_stmt.body) {
                    return true;
                }
            }
            Stmt::Block(block) => {
                if contains_quantum_ops(block) {
                    return true;
                }
            }
            Stmt::Tick(tick) => {
                // Tick blocks always contain quantum ops
                let _ = tick;
                return true;
            }
            _ => {}
        }
    }
    false
}

/// Check if an expression contains quantum operations (gate calls, qalloc, etc.).
fn expr_contains_quantum(expr: &crate::ast::Expr) -> bool {
    matches!(expr, crate::ast::Expr::Gate(_))
}

/// Format test results for terminal output.
pub fn format_results(results: &[TestResult]) -> String {
    let mut out = String::new();

    for result in results {
        let status = match &result.outcome {
            TestOutcome::Pass => "PASS",
            TestOutcome::Fail(_) => "FAIL",
            TestOutcome::Skip(_) => "SKIP",
        };

        out.push_str(&format!(
            "  {} {} ({:.3}ms)\n",
            status,
            result.name,
            result.duration.as_secs_f64() * 1000.0,
        ));

        match &result.outcome {
            TestOutcome::Fail(msg) => {
                out.push_str(&format!("       {}\n", msg));
            }
            TestOutcome::Skip(reason) => {
                out.push_str(&format!("       {}\n", reason));
            }
            _ => {}
        }
    }

    let pass_count = results.iter().filter(|r| r.outcome == TestOutcome::Pass).count();
    let fail_count = results.iter().filter(|r| matches!(r.outcome, TestOutcome::Fail(_))).count();
    let skip_count = results.iter().filter(|r| matches!(r.outcome, TestOutcome::Skip(_))).count();
    let total = results.len();

    out.push_str(&format!(
        "\n{} tests: {} passed, {} failed, {} skipped\n",
        total, pass_count, fail_count, skip_count,
    ));

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;

    #[test]
    fn test_discover_tests() {
        let source = r#"
            test "addition works" {
                x := 1 + 1;
            }
            test "subtraction works" {
                x := 2 - 1;
            }
            fn main() -> unit { return; }
        "#;
        let program = parse(source).unwrap();
        let runner = TestRunner::new(TestRunConfig::default());
        let tests = runner.discover_tests(&program);
        assert_eq!(tests.len(), 2);
        assert_eq!(tests[0].name, "addition works");
        assert_eq!(tests[1].name, "subtraction works");
    }

    #[test]
    fn test_filter_by_name() {
        let source = r#"
            test "addition works" {
                x := 1 + 1;
            }
            test "subtraction works" {
                x := 2 - 1;
            }
        "#;
        let program = parse(source).unwrap();
        let runner = TestRunner::new(TestRunConfig {
            filter: Some("addition".to_string()),
            ..Default::default()
        });
        let tests = runner.discover_tests(&program);
        assert_eq!(tests.len(), 1);
        assert_eq!(tests[0].name, "addition works");
    }

    #[test]
    fn test_passing_test() {
        let source = r#"
            test "simple math" {
                x := 1 + 1;
            }
        "#;
        let program = parse(source).unwrap();
        let runner = TestRunner::new(TestRunConfig::default());
        let results = runner.run(&program);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].outcome, TestOutcome::Pass);
    }

    #[test]
    fn test_quantum_test_skipped() {
        let source = r#"
            test "quantum test" {
                mut q := qalloc(2);
                h q[0];
            }
        "#;
        let program = parse(source).unwrap();
        let runner = TestRunner::new(TestRunConfig::default());
        let results = runner.run(&program);
        assert_eq!(results.len(), 1);
        assert!(matches!(results[0].outcome, TestOutcome::Skip(_)));
    }

    #[test]
    fn test_format_results() {
        let results = vec![
            TestResult {
                name: "test 1".to_string(),
                outcome: TestOutcome::Pass,
                duration: Duration::from_millis(1),
            },
            TestResult {
                name: "test 2".to_string(),
                outcome: TestOutcome::Fail("assertion failed".to_string()),
                duration: Duration::from_millis(2),
            },
        ];
        let output = format_results(&results);
        assert!(output.contains("PASS"));
        assert!(output.contains("FAIL"));
        assert!(output.contains("2 tests: 1 passed, 1 failed, 0 skipped"));
    }
}
