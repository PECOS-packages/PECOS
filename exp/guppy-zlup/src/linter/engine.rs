//! Main linter engine for guppy-zlup.

use std::fmt;

use rustpython_parser::{parse, Mode};

use super::config::Config;
use super::diagnostic::{Diagnostic, Severity, SourceLocation};
use super::noqa;
use super::rules::{self, LintRule};

/// Result of linting a file.
#[derive(Debug, Clone, Default)]
pub struct LintResult {
    pub diagnostics: Vec<Diagnostic>,
    pub has_errors: bool,
    pub has_warnings: bool,
}

impl LintResult {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, diagnostic: Diagnostic) {
        match diagnostic.severity {
            Severity::Error => self.has_errors = true,
            Severity::Warning => self.has_warnings = true,
            _ => {}
        }
        self.diagnostics.push(diagnostic);
    }

    pub fn is_ok(&self, treat_warnings_as_errors: bool) -> bool {
        if self.has_errors {
            return false;
        }
        if treat_warnings_as_errors && self.has_warnings {
            return false;
        }
        true
    }

    pub fn merge(&mut self, other: LintResult) {
        self.has_errors |= other.has_errors;
        self.has_warnings |= other.has_warnings;
        self.diagnostics.extend(other.diagnostics);
    }
}

impl fmt::Display for LintResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.diagnostics.is_empty() {
            return write!(f, "No issues found.");
        }
        for (i, diag) in self.diagnostics.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "{}", diag)?;
        }
        Ok(())
    }
}

/// Main linter for Guppy programs.
pub struct Linter {
    config: Config,
    rules: Vec<Box<dyn LintRule>>,
}

impl Linter {
    pub fn new(config: Config) -> Self {
        let rules = Self::load_rules(&config);
        Self { config, rules }
    }

    fn load_rules(config: &Config) -> Vec<Box<dyn LintRule>> {
        let mut rules: Vec<Box<dyn LintRule>> = Vec::new();

        if config.is_rule_enabled("ZLUP001") {
            rules.push(Box::new(rules::ZLUP001UnboundedLoops));
        }
        if config.is_rule_enabled("ZLUP002") {
            rules.push(Box::new(rules::ZLUP002Recursion));
        }
        if config.is_rule_enabled("ZLUP003") {
            rules.push(Box::new(rules::ZLUP003DynamicAllocation));
        }
        if config.is_rule_enabled("ZLUP004") {
            rules.push(Box::new(rules::ZLUP004DynamicDispatch));
        }
        if config.is_rule_enabled("ZLUP005") {
            rules.push(Box::new(rules::ZLUP005UncheckedErrors));
        }
        if config.is_rule_enabled("ZLUP006") {
            rules.push(Box::new(rules::ZLUP006MissingTypes));
        }
        if config.is_rule_enabled("ZLUP007") {
            rules.push(Box::new(rules::ZLUP007ComplexControlFlow::new(config.max_complexity)));
        }
        if config.is_rule_enabled("ZLUP008") {
            rules.push(Box::new(rules::ZLUP008CallDepth::default()));
        }
        if config.is_rule_enabled("ZLUP009") {
            rules.push(Box::new(rules::ZLUP009AssertionDensity));
        }
        if config.is_rule_enabled("ZLUP010") {
            rules.push(Box::new(rules::ZLUP010GlobalState));
        }

        rules
    }

    pub fn lint_source(&self, source: &str, filename: &str) -> LintResult {
        let mut result = LintResult::new();

        // Parse noqa directives first
        let noqa_directives = noqa::parse_noqa(source);

        // Parse Python source
        let parsed = match parse(source, Mode::Module, filename) {
            Ok(p) => p,
            Err(e) => {
                result.add(Diagnostic::error(
                    "PARSE",
                    format!("Syntax error: {}", e),
                    SourceLocation::new(1, 0).with_file(filename),
                ));
                return result;
            }
        };

        // Run each lint rule
        for rule in &self.rules {
            let diagnostics = rule.check(&parsed, filename, source);
            for diag in diagnostics {
                // Filter out diagnostics that are suppressed by noqa comments
                if !noqa_directives.is_suppressed(diag.location.line, &diag.rule_id) {
                    result.add(diag);
                }
            }
        }

        result
    }

    pub fn lint_file(&self, path: &str) -> Result<LintResult, std::io::Error> {
        let source = std::fs::read_to_string(path)?;
        Ok(self.lint_source(&source, path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lint_result_merge() {
        let mut r1 = LintResult::new();
        r1.add(Diagnostic::error("TEST", "error", SourceLocation::new(1, 0)));

        let mut r2 = LintResult::new();
        r2.add(Diagnostic::warning("TEST", "warning", SourceLocation::new(2, 0)));

        r1.merge(r2);
        assert!(r1.has_errors);
        assert!(r1.has_warnings);
        assert_eq!(r1.diagnostics.len(), 2);
    }

    #[test]
    fn test_lint_syntax_error() {
        let config = Config::default();
        let linter = Linter::new(config);
        let result = linter.lint_source("def foo(", "test.py");

        assert!(result.has_errors);
        assert!(result.diagnostics.iter().any(|d| d.rule_id == "PARSE"));
    }
}
