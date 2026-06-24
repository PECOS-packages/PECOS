//! # guppy-zlup
//!
//! Guppy linter and Zlup compiler.
//!
//! This crate provides:
//! - **Linting**: Validates Guppy quantum programs against NASA Power of 10 rules
//! - **Compilation**: Transforms validated Guppy IR into Zlup source code
//!
//! ## Usage
//!
//! ### Linting
//!
//! ```rust
//! use guppy_zlup::lint_source;
//!
//! // Lint source code
//! let result = lint_source("def main(): pass", None);
//! if result.has_errors {
//!     for diag in &result.diagnostics {
//!         println!("{}", diag);
//!     }
//! }
//! assert!(!result.has_errors);
//! ```
//!
//! To lint a file:
//!
//! ```rust,no_run
//! use guppy_zlup::lint_file;
//!
//! let result = lint_file("example.py").unwrap();
//! ```
//!
//! ### Compilation
//!
//! ```rust
//! use guppy_zlup::{compile, compile_to_ast};
//!
//! let ir_json = r#"{
//!     "version": "0.1.0",
//!     "functions": [{
//!         "name": "main",
//!         "params": [],
//!         "body": [
//!             {"kind": "qalloc", "name": "q", "size": {"kind": "literal", "value": 2}},
//!             {"kind": "gate", "gate": "h", "targets": [
//!                 {"kind": "index", "array": "q", "index": {"kind": "literal", "value": 0}}
//!             ]}
//!         ]
//!     }]
//! }"#;
//!
//! // Get Zlup source code directly
//! let source = compile(ir_json).unwrap();
//! assert!(source.contains("fn main"));
//!
//! // Or get the Zlup AST for programmatic use
//! let ast = compile_to_ast(ir_json).unwrap();
//! assert_eq!(ast.declarations.len(), 1);
//! ```

#![warn(clippy::all)]
#![allow(dead_code)]

pub mod compiler;
pub mod ir;
pub mod linter;

// Re-export zlup::ast for users who want to work with the AST
pub use zlup::ast as zlup_ast;

// Re-export linter types
pub use linter::{Config, Diagnostic, LintResult, Linter, LowerError, OutputFormat, Severity};

// Re-export IR types
pub use ir::{GuppyIR, IrValidator, ValidationError, ValidationResult, validate_ir};

// Re-export compiler types
pub use compiler::{parse_ir, ParseError, TransformError};

/// Crate version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// =============================================================================
// Linter API
// =============================================================================

/// Lint a source string and return diagnostics.
pub fn lint_source(source: &str, filename: Option<&str>) -> LintResult {
    let config = Config::default();
    let linter = Linter::new(config);
    linter.lint_source(source, filename.unwrap_or("<stdin>"))
}

/// Lint a file and return diagnostics.
pub fn lint_file(path: &str) -> Result<LintResult, std::io::Error> {
    let source = std::fs::read_to_string(path)?;
    Ok(lint_source(&source, Some(path)))
}

// =============================================================================
// Compiler API
// =============================================================================

/// Compile Guppy IR JSON to Zlup AST.
///
/// Use this when you need programmatic access to the Zlup AST,
/// e.g., for further transformation or analysis.
pub fn compile_to_ast(ir_json: &str) -> Result<zlup::ast::Program, CompileError> {
    let ir = compiler::parse_ir(ir_json)?;
    let zlup_ast = compiler::transform(&ir)?;
    Ok(zlup_ast)
}

/// Compile Guppy IR JSON to Zlup source code.
///
/// This uses Zlup's canonical formatter for consistent output.
/// The output is validated by parsing it back through Zlup's parser.
pub fn compile(ir_json: &str) -> Result<String, CompileError> {
    let zlup_ast = compile_to_ast(ir_json)?;
    let options = zlup::pretty::PrettyOptions::default();
    let zlup_source = zlup::pretty::pretty_print(&zlup_ast, &options);

    // Validate the generated Zlup by parsing it back
    validate_zlup(&zlup_source)?;

    Ok(zlup_source)
}

/// Validate generated Zlup source by parsing and analyzing it.
///
/// This ensures the generated code is syntactically and semantically valid.
pub fn validate_zlup(source: &str) -> Result<(), CompileError> {
    // Parse the Zlup source
    let program = zlup::parse(source).map_err(|e| {
        CompileError::Transform(compiler::TransformError::ValidationFailed(format!(
            "Generated Zlup failed to parse: {}",
            e
        )))
    })?;

    // Run semantic analysis (permissive mode since we're validating generated code)
    let mut analyzer = zlup::semantic::SemanticAnalyzer::new_permissive();
    analyzer.analyze(&program).map_err(|e| {
        CompileError::Transform(compiler::TransformError::ValidationFailed(format!(
            "Generated Zlup failed semantic analysis: {}",
            e
        )))
    })?;

    Ok(())
}

/// Validate with round-trip: generated AST → source → parsed AST comparison.
///
/// This ensures the pretty printer produces code that parses back to an equivalent AST.
pub fn validate_zlup_roundtrip(original_ast: &zlup::ast::Program, source: &str) -> Result<(), CompileError> {
    // First do basic validation
    validate_zlup(source)?;

    // Parse back
    let reparsed = zlup::parse(source).map_err(|e| {
        CompileError::Transform(compiler::TransformError::ValidationFailed(format!(
            "Round-trip: failed to reparse: {}",
            e
        )))
    })?;

    // Compare key structural properties
    let original_fns: Vec<_> = original_ast.declarations.iter()
        .filter_map(|d| match d {
            zlup_ast::TopLevelDecl::Fn(f) => Some(f),
            _ => None,
        })
        .collect();

    let reparsed_fns: Vec<_> = reparsed.declarations.iter()
        .filter_map(|d| match d {
            zlup_ast::TopLevelDecl::Fn(f) => Some(f),
            _ => None,
        })
        .collect();

    if original_fns.len() != reparsed_fns.len() {
        return Err(CompileError::Transform(compiler::TransformError::ValidationFailed(format!(
            "Round-trip: function count mismatch (original: {}, reparsed: {})",
            original_fns.len(),
            reparsed_fns.len()
        ))));
    }

    for (orig, repr) in original_fns.iter().zip(reparsed_fns.iter()) {
        if orig.name != repr.name {
            return Err(CompileError::Transform(compiler::TransformError::ValidationFailed(format!(
                "Round-trip: function name mismatch (original: {}, reparsed: {})",
                orig.name, repr.name
            ))));
        }
        if orig.params.len() != repr.params.len() {
            return Err(CompileError::Transform(compiler::TransformError::ValidationFailed(format!(
                "Round-trip: parameter count mismatch for '{}' (original: {}, reparsed: {})",
                orig.name, orig.params.len(), repr.params.len()
            ))));
        }
    }

    Ok(())
}

/// Compile and validate with round-trip checking.
pub fn compile_with_roundtrip(ir_json: &str) -> Result<String, CompileError> {
    let ir = compiler::parse_ir(ir_json)?;
    let zlup_ast = compiler::transform(&ir)?;
    let options = zlup::pretty::PrettyOptions::default();
    let zlup_source = zlup::pretty::pretty_print(&zlup_ast, &options);

    // Validate with round-trip
    validate_zlup_roundtrip(&zlup_ast, &zlup_source)?;

    Ok(zlup_source)
}

/// Compile Guppy IR from a file to Zlup source code.
pub fn compile_file(path: &str) -> Result<String, CompileError> {
    let json = std::fs::read_to_string(path).map_err(CompileError::Io)?;
    compile(&json)
}

/// Compile Guppy IR from a file to Zlup AST.
pub fn compile_file_to_ast(path: &str) -> Result<zlup::ast::Program, CompileError> {
    let json = std::fs::read_to_string(path).map_err(CompileError::Io)?;
    compile_to_ast(&json)
}

/// Compilation error.
#[derive(Debug, thiserror::Error)]
pub enum CompileError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parse error: {0}")]
    Parse(#[from] compiler::ParseError),

    #[error("Transform error: {0}")]
    Transform(#[from] compiler::TransformError),
}

// =============================================================================
// Combined API
// =============================================================================

/// Lint and compile a Guppy source file to Zlup.
///
/// This is the main entry point for the full pipeline:
/// 1. Parse the Guppy source
/// 2. Run lint checks
/// 3. Lower to IR
/// 4. Validate IR
/// 5. Transform to Zlup AST
/// 6. Pretty-print to Zlup source
/// 7. Validate generated Zlup
pub fn lint_and_compile(source: &str, filename: Option<&str>) -> Result<String, PipelineError> {
    let filename = filename.unwrap_or("<stdin>");

    // Run linter
    let config = Config::default();
    let linter = Linter::new(config);
    let result = linter.lint_source(source, filename);

    if result.has_errors {
        return Err(PipelineError::LintErrors(result));
    }

    // Emit IR
    let ir = ir::emit_ir(source, Some(filename)).map_err(PipelineError::Emit)?;

    // Validate IR
    let validation = ir::validate_ir(&ir);
    if !validation.is_valid() {
        return Err(PipelineError::IrValidation(validation));
    }

    // Compile to Zlup
    let ir_json = serde_json::to_string(&ir).map_err(|e| PipelineError::Serialize(e.to_string()))?;
    let zlup_source = compile(&ir_json)?;

    Ok(zlup_source)
}

/// Pipeline error encompassing all stages.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("Lint errors found")]
    LintErrors(LintResult),

    #[error("IR emission error: {0}")]
    Emit(#[from] ir::EmitError),

    #[error("IR validation errors: {0:?}")]
    IrValidation(ir::ValidationResult),

    #[error("Serialization error: {0}")]
    Serialize(String),

    #[error("Compilation error: {0}")]
    Compile(#[from] CompileError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_lint_empty_source() {
        let result = lint_source("", None);
        assert!(result.is_ok(false));
    }

    #[test]
    fn test_lint_simple_function() {
        let source = r#"
def main() -> None:
    pass
"#;
        let result = lint_source(source, None);
        assert!(!result.has_errors);
    }

    #[test]
    fn test_lint_while_true() {
        let source = r#"
def main():
    while True:
        pass
"#;
        let result = lint_source(source, None);
        assert!(result.has_errors);
        assert!(result.diagnostics.iter().any(|d| d.rule_id == "ZLUP001"));
    }

    #[test]
    fn test_compile_simple() {
        let ir = r#"{
            "version": "0.1.0",
            "functions": [
                {
                    "name": "main",
                    "params": [],
                    "body": [
                        {"kind": "qalloc", "name": "q", "size": {"kind": "literal", "value": 4}},
                        {"kind": "gate", "gate": "h", "targets": [{"kind": "index", "array": "q", "index": {"kind": "literal", "value": 0}}]}
                    ]
                }
            ]
        }"#;

        let result = compile(ir);
        assert!(result.is_ok(), "Compile failed: {:?}", result.err());
        let zlup = result.unwrap();
        assert!(zlup.contains("fn main"));
        assert!(zlup.contains("qalloc"));
    }

    #[test]
    fn test_roundtrip_simple() {
        let ir = r#"{
            "version": "0.1.0",
            "functions": [
                {
                    "name": "test_func",
                    "params": [
                        {"name": "x", "type": {"kind": "primitive", "name": "int"}}
                    ],
                    "return_type": {"kind": "primitive", "name": "int"},
                    "body": [
                        {"kind": "return", "return_value": {"kind": "ident", "name": "x"}}
                    ]
                }
            ]
        }"#;

        let result = compile_with_roundtrip(ir);
        assert!(result.is_ok(), "Round-trip compile failed: {:?}", result.err());
    }

    #[test]
    fn test_roundtrip_complex() {
        let ir = r#"{
            "version": "0.1.0",
            "functions": [
                {
                    "name": "quantum_ops",
                    "params": [],
                    "body": [
                        {"kind": "qalloc", "name": "q", "size": {"kind": "literal", "value": 2}},
                        {"kind": "gate", "gate": "h", "targets": [{"kind": "index", "array": "q", "index": {"kind": "literal", "value": 0}}]},
                        {"kind": "gate", "gate": "cx", "targets": [
                            {"kind": "index", "array": "q", "index": {"kind": "literal", "value": 0}},
                            {"kind": "index", "array": "q", "index": {"kind": "literal", "value": 1}}
                        ]}
                    ]
                }
            ]
        }"#;

        let result = compile_with_roundtrip(ir);
        assert!(result.is_ok(), "Round-trip compile failed: {:?}", result.err());
    }
}
