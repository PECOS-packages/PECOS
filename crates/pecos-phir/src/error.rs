/*!
Error handling for PECOS PHIR

This module provides comprehensive error handling for all PHIR operations including:
- Parse errors from various input formats
- Type checking and validation errors
- Runtime execution errors
- Compilation and optimization errors
- QEC-specific errors

Uses the `thiserror` crate for ergonomic error handling.
*/

use thiserror::Error;

/// Main error type for PHIR operations
#[derive(Debug, Clone, Error)]
pub enum PhirError {
    /// Parsing errors from input formats
    #[error("Parse error: {0}")]
    Parse(#[from] Box<ParseError>),

    /// Type system errors
    #[error("Type error: {0}")]
    Type(#[from] Box<TypeError>),

    /// Validation errors (semantic analysis)
    #[error("Validation error: {0}")]
    Validation(#[from] Box<ValidationError>),

    /// Runtime execution errors
    #[error("Runtime error: {0}")]
    Runtime(#[from] Box<RuntimeError>),

    /// Compilation/optimization errors
    #[error("Compilation error: {0}")]
    Compilation(#[from] Box<CompilationError>),

    /// I/O errors
    #[error("I/O error: {0}")]
    IO(String),

    /// Internal errors (bugs)
    #[error("Internal error: {0}")]
    Internal(String),
}

/// Parsing errors from various input formats
#[derive(Debug, Clone, Error)]
pub enum ParseError {
    /// Syntax error in input
    #[error("Syntax error at {location}: {message}")]
    Syntax {
        message: String,
        location: SourceLocation,
        expected: Option<String>,
        found: Option<String>,
    },

    /// Unsupported feature in input format
    #[error("Unsupported feature '{feature}' in {format} at {location}")]
    Unsupported {
        feature: String,
        format: String,
        location: SourceLocation,
    },

    /// Invalid structure (e.g., malformed HUGR)
    #[error("Invalid structure at {location}: {message}")]
    InvalidStructure {
        message: String,
        location: SourceLocation,
    },

    /// JSON/serialization errors
    #[error("Serialization error in {format}: {message}")]
    Serialization { message: String, format: String },

    /// File I/O errors during parsing
    #[error("File I/O error for '{path}': {message}")]
    FileIO { path: String, message: String },
}

/// Type system errors
#[derive(Debug, Clone, Error)]
pub enum TypeError {
    /// Type mismatch
    #[error("Type mismatch at {location}: expected {expected:?}, found {found:?}")]
    Mismatch {
        expected: crate::types::Type,
        found: crate::types::Type,
        location: SourceLocation,
    },

    /// Undefined type
    #[error("Undefined type '{type_name}' at {location}")]
    Undefined {
        type_name: String,
        location: SourceLocation,
    },

    /// Incompatible types in operation
    #[error("Incompatible types for operation '{op_name}' at {location}: {types:?}")]
    Incompatible {
        op_name: String,
        types: Vec<crate::types::Type>,
        location: SourceLocation,
    },

    /// Type inference failure
    #[error("Type inference failed at {location}: {message}")]
    InferenceFailed {
        message: String,
        location: SourceLocation,
    },

    /// Quantum no-cloning violation
    #[error("Quantum no-cloning violation for variable '{variable}' at {location}")]
    NoCloning {
        variable: String,
        location: SourceLocation,
    },

    /// Invalid type parameters
    #[error("Invalid type parameters for '{type_name}' at {location}: {message}")]
    InvalidParameters {
        type_name: String,
        message: String,
        location: SourceLocation,
    },
}

/// Semantic validation errors
#[derive(Debug, Clone, Error)]
pub enum ValidationError {
    /// Undefined variable or function
    #[error("Undefined {kind:?} '{name}' at {location}")]
    Undefined {
        name: String,
        kind: DefinitionKind,
        location: SourceLocation,
    },

    /// Duplicate definition
    #[error("Duplicate definition of '{name}' at {location}")]
    DuplicateDefinition {
        name: String,
        location: SourceLocation,
        previous: SourceLocation,
    },

    /// Invalid structure (e.g., CFG violations)
    #[error("Invalid structure at {location}: {message}")]
    InvalidStructure {
        message: String,
        location: SourceLocation,
    },

    /// Missing required component
    #[error("Missing {component} at {location}")]
    MissingComponent {
        component: String,
        location: SourceLocation,
    },

    /// SSA violation
    #[error("SSA violation for variable '{variable}' at {location}")]
    SSAViolation {
        variable: String,
        location: SourceLocation,
    },

    /// Ownership/borrowing violation
    #[error("Ownership violation for '{resource}' at {location}: {message}")]
    OwnershipViolation {
        resource: String,
        message: String,
        location: SourceLocation,
    },

    /// Variable used before definition
    #[error("Variable '{variable}' used before definition at {use_location}")]
    UseBeforeDefine {
        variable: String,
        use_location: SourceLocation,
        define_location: Option<SourceLocation>,
    },

    /// Multiple definitions of same name
    #[error(
        "Redefinition of {kind:?} '{name}' at {second_location} (first defined at {first_location})"
    )]
    Redefinition {
        name: String,
        kind: DefinitionKind,
        first_location: SourceLocation,
        second_location: SourceLocation,
    },

    /// Invalid control flow
    #[error("Invalid control flow at {location}: {message}")]
    ControlFlow {
        message: String,
        location: SourceLocation,
    },

    /// Quantum circuit violations
    #[error("Quantum violation '{rule}' at {location}: {message}")]
    QuantumViolation {
        rule: String,
        message: String,
        location: SourceLocation,
    },

    /// Unknown dialect
    #[error("Unknown dialect: {0}")]
    UnknownDialect(String),

    /// Unknown operation
    #[error("Unknown operation: {0}")]
    UnknownOperation(String),
}

/// Runtime execution errors
#[derive(Debug, Clone, Error)]
pub enum RuntimeError {
    /// Division by zero
    #[error("Division by zero at {location}")]
    DivisionByZero { location: SourceLocation },

    /// Index out of bounds
    #[error("Index {index} out of bounds for array of size {size} at {location}")]
    IndexOutOfBounds {
        index: usize,
        size: usize,
        location: SourceLocation,
    },

    /// External function call failed
    #[error("External function '{function}' failed at {location}: {message}")]
    ExternalCall {
        function: String,
        message: String,
        location: SourceLocation,
    },

    /// Resource exhausted (e.g., memory, qubits)
    #[error("Resource exhausted at {location}: {resource}")]
    ResourceExhausted {
        resource: String,
        location: SourceLocation,
    },

    /// Execution failed with custom message
    #[error("Execution failed at {location}: {message}")]
    ExecutionFailed {
        message: String,
        location: SourceLocation,
    },
}

/// Compilation and optimization errors
#[derive(Debug, Clone, Error)]
pub enum CompilationError {
    /// Optimization pass failed
    #[error("Optimization pass '{pass}' failed: {message}")]
    OptimizationFailed { pass: String, message: String },

    /// Code generation failed
    #[error("Code generation failed for target '{target}': {message}")]
    CodeGenFailed { target: String, message: String },

    /// Resource estimation exceeded limits
    #[error("Resource estimation failed: {message}")]
    ResourceEstimation { message: String },

    /// Circuit routing failed
    #[error("Circuit routing failed for topology '{topology}': {message}")]
    RoutingFailed { topology: String, message: String },
}

/// Kind of definition for validation errors
#[derive(Debug, Clone)]
pub enum DefinitionKind {
    Variable,
    Function,
    Type,
    Module,
    Block,
}

/// Source location for error reporting
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SourceLocation {
    /// Source file path
    pub file: String,
    /// Line number (1-based)
    pub line: usize,
    /// Column number (1-based)
    pub column: usize,
    /// Character span in source
    pub span: Span,
}

/// Character span in source text
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Span {
    /// Start position (0-based)
    pub start: usize,
    /// End position (0-based, exclusive)
    pub end: usize,
}

/// Result type alias for PHIR operations
pub type Result<T> = std::result::Result<T, PhirError>;

// Helper constructors
impl PhirError {
    /// Create a parse error
    pub fn parse_error(message: impl Into<String>, location: SourceLocation) -> Self {
        Box::new(ParseError::Syntax {
            message: message.into(),
            location,
            expected: None,
            found: None,
        })
        .into()
    }

    /// Create a type error
    #[must_use]
    pub fn type_error(
        expected: crate::types::Type,
        found: crate::types::Type,
        location: SourceLocation,
    ) -> Self {
        Box::new(TypeError::Mismatch {
            expected,
            found,
            location,
        })
        .into()
    }

    /// Create a validation error
    pub fn undefined_variable(name: impl Into<String>, location: SourceLocation) -> Self {
        Box::new(ValidationError::Undefined {
            name: name.into(),
            kind: DefinitionKind::Variable,
            location,
        })
        .into()
    }

    /// Create a runtime error
    pub fn runtime_error(message: impl Into<String>, location: SourceLocation) -> Self {
        Box::new(RuntimeError::ExternalCall {
            function: "unknown".to_string(),
            message: message.into(),
            location,
        })
        .into()
    }

    /// Create an internal error (for bugs)
    pub fn internal(message: impl Into<String>) -> Self {
        PhirError::Internal(message.into())
    }

    /// Create an I/O error
    pub fn io_error(message: impl Into<String>) -> Self {
        PhirError::IO(message.into())
    }
}

impl SourceLocation {
    /// Create an unknown source location
    #[must_use]
    pub fn unknown() -> Self {
        Self {
            file: "<unknown>".to_string(),
            line: 0,
            column: 0,
            span: Span { start: 0, end: 0 },
        }
    }

    /// Create a source location from file, line, and column
    #[must_use]
    pub fn new(file: impl Into<String>, line: usize, column: usize) -> Self {
        Self {
            file: file.into(),
            line,
            column,
            span: Span { start: 0, end: 0 },
        }
    }
}

impl std::fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}:{}", self.file, self.line, self.column)
    }
}

impl std::fmt::Display for DefinitionKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DefinitionKind::Variable => write!(f, "variable"),
            DefinitionKind::Function => write!(f, "function"),
            DefinitionKind::Type => write!(f, "type"),
            DefinitionKind::Module => write!(f, "module"),
            DefinitionKind::Block => write!(f, "block"),
        }
    }
}

// Convert from std::io::Error
impl From<std::io::Error> for PhirError {
    fn from(err: std::io::Error) -> Self {
        PhirError::IO(err.to_string())
    }
}

// Convert to PecosError for interoperability with other PECOS crates
impl From<PhirError> for pecos_core::errors::PecosError {
    fn from(err: PhirError) -> Self {
        use pecos_core::errors::PecosError;

        match err {
            PhirError::Parse(e) => PecosError::ParseSyntax {
                language: "PHIR".to_string(),
                message: e.to_string(),
            },
            PhirError::Type(e) => PecosError::Compilation(format!("Type error: {e}")),
            PhirError::Validation(e) => match e.as_ref() {
                ValidationError::Undefined { name, kind, .. } => {
                    PecosError::CompileUndefinedReference {
                        kind: format!("{kind:?}"),
                        name: name.clone(),
                    }
                }
                ValidationError::UnknownDialect(d) => {
                    PecosError::Compilation(format!("Unknown dialect: {d}"))
                }
                ValidationError::UnknownOperation(op) => {
                    PecosError::Compilation(format!("Unknown operation: {op}"))
                }
                ValidationError::ControlFlow { message, .. } => {
                    PecosError::ValidationInvalidCircuitStructure(message.clone())
                }
                _ => PecosError::Compilation(format!("Validation error: {e}")),
            },
            PhirError::Runtime(e) => match e.as_ref() {
                RuntimeError::DivisionByZero { .. } => PecosError::RuntimeDivisionByZero,
                RuntimeError::IndexOutOfBounds { index, size, .. } => {
                    PecosError::RuntimeIndexOutOfBounds {
                        index: *index,
                        length: *size,
                    }
                }
                _ => PecosError::Processing(format!("Runtime error: {e}")),
            },
            PhirError::Compilation(e) => PecosError::Compilation(e.to_string()),
            PhirError::IO(msg) => PecosError::Resource(msg),
            PhirError::Internal(msg) => PecosError::Generic(format!("Internal PHIR error: {msg}")),
        }
    }
}

// Allow converting PecosError to PhirError when needed
impl From<pecos_core::errors::PecosError> for PhirError {
    fn from(err: pecos_core::errors::PecosError) -> Self {
        // For now, wrap it as an Internal error with the message
        PhirError::Internal(err.to_string())
    }
}
