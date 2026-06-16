//! Lint rules for guppy-zlup.

mod zlup001;
mod zlup002;
mod zlup003;
mod zlup004;
mod zlup005;
mod zlup006;
mod zlup007;
mod zlup008;
mod zlup009;
mod zlup010;

pub use zlup001::ZLUP001UnboundedLoops;
pub use zlup002::ZLUP002Recursion;
pub use zlup003::ZLUP003DynamicAllocation;
pub use zlup004::ZLUP004DynamicDispatch;
pub use zlup005::ZLUP005UncheckedErrors;
pub use zlup006::ZLUP006MissingTypes;
pub use zlup007::ZLUP007ComplexControlFlow;
pub use zlup008::ZLUP008CallDepth;
pub use zlup009::ZLUP009AssertionDensity;
pub use zlup010::ZLUP010GlobalState;

use rustpython_parser::ast::Mod;
use rustpython_parser::text_size::TextRange;

use super::diagnostic::{Diagnostic, Severity, SourceLocation};

/// Trait for lint rules.
pub trait LintRule: Send + Sync {
    /// Rule identifier (e.g., "ZLUP001").
    fn id(&self) -> &'static str;

    /// Human-readable rule name.
    fn name(&self) -> &'static str;

    /// Full description of the rule.
    fn description(&self) -> &'static str;

    /// Default severity level.
    fn severity(&self) -> Severity;

    /// Check an AST for violations of this rule.
    fn check(&self, parsed: &Mod, filename: &str, source: &str) -> Vec<Diagnostic>;
}

/// Convert a byte offset to (line, column) using the source text.
fn offset_to_line_col(source: &str, offset: usize) -> (usize, usize) {
    let mut line = 1;
    let mut col = 1;
    for (i, ch) in source.char_indices() {
        if i >= offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    (line, col)
}

/// Helper to create a source location from a TextRange.
pub fn make_location(range: TextRange, filename: &str, source: &str) -> SourceLocation {
    let (line, column) = offset_to_line_col(source, range.start().into());
    let (end_line, end_column) = offset_to_line_col(source, range.end().into());
    SourceLocation {
        line: line as u32,
        column: column as u32,
        end_line: Some(end_line as u32),
        end_column: Some(end_column as u32),
        file: Some(filename.to_string()),
    }
}
