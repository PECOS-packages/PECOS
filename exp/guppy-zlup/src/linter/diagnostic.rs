//! Diagnostic types for lint results.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Source location for error reporting.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceLocation {
    pub line: u32,
    pub column: u32,
    pub end_line: Option<u32>,
    pub end_column: Option<u32>,
    pub file: Option<String>,
}

impl SourceLocation {
    pub fn new(line: u32, column: u32) -> Self {
        Self {
            line,
            column,
            end_line: None,
            end_column: None,
            file: None,
        }
    }

    pub fn with_end(line: u32, column: u32, end_line: u32, end_column: u32) -> Self {
        Self {
            line,
            column,
            end_line: Some(end_line),
            end_column: Some(end_column),
            file: None,
        }
    }

    pub fn with_file(mut self, file: impl Into<String>) -> Self {
        self.file = Some(file.into());
        self
    }
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.file {
            Some(file) => write!(f, "{}:{}:{}", file, self.line, self.column),
            None => write!(f, "{}:{}", self.line, self.column),
        }
    }
}

/// Diagnostic severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Error,
    Warning,
    Info,
    Hint,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Error => write!(f, "error"),
            Severity::Warning => write!(f, "warning"),
            Severity::Info => write!(f, "info"),
            Severity::Hint => write!(f, "hint"),
        }
    }
}

/// A lint diagnostic/violation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    pub rule_id: String,
    pub message: String,
    pub severity: Severity,
    pub location: SourceLocation,
    pub suggestion: Option<String>,
    /// The source code line(s) for context (not serialized)
    #[serde(skip)]
    pub source_context: Option<String>,
}

impl Diagnostic {
    pub fn new(
        rule_id: impl Into<String>,
        message: impl Into<String>,
        severity: Severity,
        location: SourceLocation,
    ) -> Self {
        Self {
            rule_id: rule_id.into(),
            message: message.into(),
            severity,
            location,
            suggestion: None,
            source_context: None,
        }
    }

    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }

    pub fn with_source_context(mut self, source: &str) -> Self {
        let line_num = self.location.line as usize;
        if line_num > 0
            && let Some(line) = source.lines().nth(line_num - 1) {
                self.source_context = Some(line.to_string());
            }
        self
    }

    pub fn error(rule_id: impl Into<String>, message: impl Into<String>, location: SourceLocation) -> Self {
        Self::new(rule_id, message, Severity::Error, location)
    }

    pub fn warning(rule_id: impl Into<String>, message: impl Into<String>, location: SourceLocation) -> Self {
        Self::new(rule_id, message, Severity::Warning, location)
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{}: [{}] {}",
            self.severity, self.rule_id, self.message
        )?;
        writeln!(f, "  --> {}", self.location)?;

        // Show source context if available
        if let Some(ref source_line) = self.source_context {
            let line_num = self.location.line;
            let col = self.location.column.saturating_sub(1) as usize;
            let line_num_width = line_num.to_string().len();

            // Empty line number gutter
            writeln!(f, "{:width$} |", "", width = line_num_width)?;
            // Source line
            writeln!(f, "{} | {}", line_num, source_line)?;
            // Underline pointing to the location
            let underline_len = if let Some(end_col) = self.location.end_column {
                (end_col.saturating_sub(self.location.column) as usize).max(1)
            } else {
                1
            };
            writeln!(
                f,
                "{:width$} | {:>col$}{}",
                "",
                "",
                "^".repeat(underline_len),
                width = line_num_width,
                col = col
            )?;
        }

        if let Some(suggestion) = &self.suggestion {
            writeln!(f, "  help: {}", suggestion)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostic_display() {
        let diag = Diagnostic::error(
            "ZLUP001",
            "'while True' creates an unbounded loop",
            SourceLocation::new(5, 4).with_file("test.py"),
        )
        .with_suggestion("Use a for loop with a fixed upper bound instead");

        let output = format!("{}", diag);
        assert!(output.contains("error"));
        assert!(output.contains("ZLUP001"));
        assert!(output.contains("while True"));
        assert!(output.contains("test.py:5:4"));
    }
}
