//! Semantic analysis for Zluppy programs.
//!
//! This module performs:
//! - Name resolution and scope management
//! - Type checking and inference
//! - Comptime evaluation
//! - Quantum resource validation (for HUGR codegen)
//! - **Qubit state tracking** (prepared/unprepared lifecycle)
//! - **Allocator capacity validation** (bounds checking)
//! - **Loop bound checking** (NASA Power of 10 compliance)
//!
//! ## Qubit State Tracking
//!
//! Zluppy tracks qubit states at compile time. Every qubit slot has exactly
//! two states:
//!
//! ```text
//! ┌────────────┐  prepare()  ┌──────────┐
//! │ unprepared │ ──────────> │ prepared │
//! └────────────┘             └──────────┘
//!       ^                          │
//!       │         measure()        │
//!       └──────────────────────────┘
//! ```
//!
//! - **unprepared**: Initial state, or after measurement
//! - **prepared**: Ready for gate operations
//!
//! Gates on unprepared qubits are compile-time errors. This is the allocator
//! model's answer to Guppy's linear types - simpler, explicit, same safety.
//!
//! The semantic analysis produces a typed AST and symbol table that
//! can be used by code generators (HUGR, SLR-AST, etc.).

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use thiserror::Error;

use crate::ast::{
    self, BinaryOp, Binding, Block, ElseBranch, Expr, FStringPart, FnDecl, ForRange, PrimitiveType,
    Program, SourceLocation, Stmt, StructDecl, TopLevelDecl, TypeExpr, UnaryOp,
};
use crate::comptime::{ComptimeEvaluator, ComptimeValue};
use crate::module::{ExportedSymbol, ModuleLoader};

// =============================================================================
// Semantic Errors
// =============================================================================

/// Semantic analysis errors.
#[derive(Debug, Clone, Error)]
pub enum SemanticError {
    #[error("undefined symbol '{name}'")]
    UndefinedSymbol {
        name: String,
        location: SourceLocation,
    },

    #[error("symbol '{name}' already defined")]
    DuplicateSymbol {
        name: String,
        location: SourceLocation,
    },

    #[error("type mismatch: expected {expected}, found {found}")]
    TypeMismatch {
        expected: String,
        found: String,
        location: SourceLocation,
    },

    #[error("cannot infer type for '{name}'")]
    CannotInferType {
        name: String,
        location: SourceLocation,
    },

    #[error(
        "empty array literal requires explicit type annotation: use `[]: [0]T` or provide elements"
    )]
    EmptyArrayNeedsType { location: SourceLocation },

    #[error(
        "empty set literal requires explicit type annotation: use `set{{}} as Set(T)` or provide elements"
    )]
    EmptySetNeedsType { location: SourceLocation },

    #[error("invalid integer bit width {bits}: must be between 1 and 128")]
    InvalidBitWidth { bits: u16, location: SourceLocation },

    #[error("gate '{gate}' requires {expected} qubits, got {found}")]
    GateArityMismatch {
        gate: String,
        expected: usize,
        found: usize,
        location: SourceLocation,
    },

    #[error(
        "ambiguous target for multi-qubit gate '{gate}': use explicit qubit pairs like '{gate} (q[0], q[1])' or batch '{gate} {{(q[0], q[1]), ...}}'"
    )]
    AmbiguousGateTarget {
        gate: String,
        location: SourceLocation,
    },

    #[error("invalid gate syntax: use '{gate} {hint}' instead of '{gate}(...)'")]
    InvalidGateSyntax {
        gate: String,
        hint: String,
        location: SourceLocation,
    },

    #[error("invalid qubit reference")]
    InvalidQubitRef { location: SourceLocation },

    #[error("allocator '{name}' not found")]
    AllocatorNotFound {
        name: String,
        location: SourceLocation,
    },

    #[error("comptime evaluation failed: {message}")]
    ComptimeError {
        message: String,
        location: SourceLocation,
    },

    #[error("function '{name}' not found")]
    FunctionNotFound {
        name: String,
        location: SourceLocation,
    },

    #[error("cannot call non-function type")]
    NotCallable { location: SourceLocation },

    #[error("wrong number of arguments: expected {expected}, got {found}")]
    ArgumentCountMismatch {
        expected: usize,
        found: usize,
        location: SourceLocation,
    },

    #[error("{message}")]
    Other { message: String },

    #[error("module error: {message}")]
    ModuleError {
        message: String,
        location: SourceLocation,
    },

    // =========================================================================
    // Qubit State Errors (Zluppy-specific safety)
    // =========================================================================
    #[error("qubit '{allocator}[{index}]' is not prepared - call prepare() first")]
    QubitNotPrepared {
        allocator: String,
        index: usize,
        location: SourceLocation,
    },

    #[error("qubit '{allocator}[{index}]' is already prepared")]
    QubitAlreadyPrepared {
        allocator: String,
        index: usize,
        location: SourceLocation,
    },

    #[error("qubit index {index} out of bounds for allocator '{allocator}' (capacity: {capacity})")]
    QubitIndexOutOfBounds {
        allocator: String,
        index: usize,
        capacity: usize,
        location: SourceLocation,
    },

    #[error("array index {index} out of bounds for array of size {size}")]
    ArrayIndexOutOfBounds {
        index: usize,
        size: u64,
        location: SourceLocation,
    },

    #[error(
        "cannot call .child() on immutable allocator '{name}' - declare with 'mut' to partition: mut {name} := qalloc(...)"
    )]
    ChildRequiresMutableParent {
        name: String,
        location: SourceLocation,
    },

    #[error(
        "cannot assign to immutable variable '{name}' - declare with 'mut' to allow modification: mut {name} := ..."
    )]
    ImmutableAssignment {
        name: String,
        location: SourceLocation,
    },

    // =========================================================================
    // NASA Power of 10 Errors
    // =========================================================================
    #[error("unbounded loop detected - use bounded 'for' loops instead")]
    UnboundedLoop { location: SourceLocation },

    #[error("loop bound too large ({bound}) - maximum allowed is {max}")]
    LoopBoundTooLarge {
        bound: usize,
        max: usize,
        location: SourceLocation,
    },

    #[error("recursion detected in function '{name}' - recursion is not allowed")]
    RecursionDetected {
        name: String,
        location: SourceLocation,
    },

    #[error("invalid measurement type '{ty}' - expected u1, u8, u64, []u1, []u8, or []u64")]
    InvalidMeasurementType {
        ty: String,
        location: SourceLocation,
    },

    #[error("measurement requires type and target arguments")]
    MeasurementMissingArgs { location: SourceLocation },

    #[error("deprecated measurement syntax: use 'mz(T) target' instead of 'mz(T, target)'")]
    DeprecatedMeasurementSyntax { location: SourceLocation },

    #[error("deprecated syntax: use '{new}' instead of '{old}'")]
    DeprecatedSyntax {
        old: String,
        new: String,
        location: SourceLocation,
    },

    #[error(
        "measurement type mismatch: declared [{declared}]{element} but measuring {actual} qubit(s)"
    )]
    MeasurementSizeMismatch {
        declared: String,
        element: String,
        actual: usize,
        location: SourceLocation,
    },

    #[error("single qubit measurement requires scalar type (e.g., 'mz(u1) q[0]'), not array type")]
    MeasurementScalarExpected { location: SourceLocation },

    #[error("multiple qubit measurement requires array type (e.g., 'mz([2]u1) [q[0], q[1]]')")]
    MeasurementArrayExpected { location: SourceLocation },

    #[error("pack mode: type {ty} has {capacity} bits but measuring {qubits} qubit(s)")]
    MeasurementPackCapacity {
        ty: String,
        capacity: usize,
        qubits: usize,
        location: SourceLocation,
    },

    #[error(
        "pack mode requires compile-time verifiable type size, but '{ty}' has unknown bit capacity"
    )]
    MeasurementPackUnknownSize {
        ty: String,
        location: SourceLocation,
    },

    #[error(
        "qubit '{allocator}[{index}]' used multiple times within tick block - parallel operations cannot target the same qubit"
    )]
    DuplicateQubitInTick {
        allocator: String,
        index: usize,
        location: SourceLocation,
    },

    #[error("nested tick blocks are not allowed - a tick is an atomic time slice")]
    NestedTick { location: SourceLocation },

    #[error("duplicate qubit in measurement: {allocator}[{index}]")]
    DuplicateQubitInMeasurement {
        allocator: String,
        index: usize,
        location: SourceLocation,
    },

    #[error("{keyword} outside of loop")]
    BreakContinueOutsideLoop {
        keyword: String,
        location: SourceLocation,
    },

    #[error(
        "inline for range must be comptime-evaluable, but '{expr}' cannot be evaluated at compile time"
    )]
    InlineForRangeNotComptime {
        expr: String,
        location: SourceLocation,
    },

    #[error("'break' is not allowed in inline for loops - inline for is unrolled at compile time")]
    BreakInInlineFor { location: SourceLocation },

    #[error(
        "'continue' is not allowed in inline for loops - inline for is unrolled at compile time"
    )]
    ContinueInInlineFor { location: SourceLocation },

    #[error(
        "alias '{}' overlaps with existing alias '{}' on source '{}'",
        .0.new_alias, .0.existing_alias, .0.source_var
    )]
    // Boxed because this is the only variant whose inline payload (four `String`s
    // plus a `SourceLocation`) pushes `SemanticError` over clippy's 128-byte
    // `result_large_err` threshold; boxing keeps the enum (and every
    // `SemanticResult`) small.
    OverlappingAlias(Box<OverlappingAliasError>),

    #[error("alias source must be a slice expression (e.g., arr[0..4]), found '{found}'")]
    AliasSourceNotSlice {
        found: String,
        location: SourceLocation,
    },

    #[error(
        "alias range must be comptime-evaluable for overlap checking, but '{expr}' cannot be evaluated at compile time"
    )]
    AliasRangeNotComptime {
        expr: String,
        location: SourceLocation,
    },

    #[error(
        "missing return statement in function '{name}' - all code paths must have explicit returns (use 'return unit;' for unit functions)"
    )]
    MissingReturn {
        name: String,
        location: SourceLocation,
    },

    #[error(
        "'return;' without a value is only allowed in functions that return unit, but this function returns '{expected}'"
    )]
    ReturnWithoutValue {
        expected: String,
        location: SourceLocation,
    },

    #[error("'catch' can only be used on error union types (T!E), found '{found}'")]
    CatchOnNonErrorType {
        found: String,
        location: SourceLocation,
    },

    #[error("undefined type '{name}'")]
    UndefinedType {
        name: String,
        location: SourceLocation,
    },

    #[error("type could not be fully resolved: '{ty}' contains unresolved type variables")]
    UnresolvedType {
        ty: String,
        context: String,
        location: SourceLocation,
    },

    #[error("symbol table limit exceeded: {count} symbols exceeds maximum of {max}")]
    SymbolTableLimitExceeded { count: usize, max: usize },

    #[error("scope nesting limit exceeded: {depth} levels exceeds maximum of {max}")]
    ScopeNestingLimitExceeded { depth: usize, max: usize },

    #[error("duplicate case value '{value}' in switch statement")]
    DuplicateSwitchCase {
        value: String,
        location: SourceLocation,
    },

    // =========================================================================
    // Reference Safety Errors (safe-by-constraint memory model)
    // =========================================================================
    #[error(
        "cannot return reference to local variable '{name}' - local variables are deallocated when the function returns"
    )]
    ReturnReferenceToLocal {
        name: String,
        location: SourceLocation,
    },

    #[error(
        "cannot return slice of local array '{name}' - local arrays are deallocated when the function returns"
    )]
    ReturnSliceOfLocal {
        name: String,
        location: SourceLocation,
    },

    #[error(
        "cannot store reference to local '{name}' in outer scope - would create dangling reference"
    )]
    ReferenceEscapesScope {
        name: String,
        location: SourceLocation,
    },
}

/// Boxed payload for [`SemanticError::OverlappingAlias`].
///
/// Kept behind a `Box` so the large set of fields does not inflate
/// `SemanticError` (and therefore every `SemanticResult`) past clippy's
/// `result_large_err` size threshold.
#[derive(Debug, Clone)]
pub struct OverlappingAliasError {
    pub new_alias: String,
    pub existing_alias: String,
    pub source_var: String,
    pub overlap_range: String,
    pub location: SourceLocation,
}

impl SemanticError {
    /// Get the source location of the error, if available.
    pub fn location(&self) -> Option<&SourceLocation> {
        match self {
            Self::UndefinedSymbol { location, .. } => Some(location),
            Self::DuplicateSymbol { location, .. } => Some(location),
            Self::TypeMismatch { location, .. } => Some(location),
            Self::CannotInferType { location, .. } => Some(location),
            Self::EmptyArrayNeedsType { location } => Some(location),
            Self::EmptySetNeedsType { location } => Some(location),
            Self::InvalidBitWidth { location, .. } => Some(location),
            Self::GateArityMismatch { location, .. } => Some(location),
            Self::AmbiguousGateTarget { location, .. } => Some(location),
            Self::InvalidGateSyntax { location, .. } => Some(location),
            Self::InvalidQubitRef { location } => Some(location),
            Self::AllocatorNotFound { location, .. } => Some(location),
            Self::ComptimeError { location, .. } => Some(location),
            Self::FunctionNotFound { location, .. } => Some(location),
            Self::NotCallable { location } => Some(location),
            Self::ArgumentCountMismatch { location, .. } => Some(location),
            Self::QubitNotPrepared { location, .. } => Some(location),
            Self::QubitAlreadyPrepared { location, .. } => Some(location),
            Self::QubitIndexOutOfBounds { location, .. } => Some(location),
            Self::ArrayIndexOutOfBounds { location, .. } => Some(location),
            Self::ChildRequiresMutableParent { location, .. } => Some(location),
            Self::ImmutableAssignment { location, .. } => Some(location),
            Self::UnboundedLoop { location } => Some(location),
            Self::LoopBoundTooLarge { location, .. } => Some(location),
            Self::RecursionDetected { location, .. } => Some(location),
            Self::InvalidMeasurementType { location, .. } => Some(location),
            Self::MeasurementMissingArgs { location } => Some(location),
            Self::DeprecatedMeasurementSyntax { location } => Some(location),
            Self::DeprecatedSyntax { location, .. } => Some(location),
            Self::MeasurementSizeMismatch { location, .. } => Some(location),
            Self::MeasurementScalarExpected { location } => Some(location),
            Self::MeasurementArrayExpected { location } => Some(location),
            Self::MeasurementPackCapacity { location, .. } => Some(location),
            Self::MeasurementPackUnknownSize { location, .. } => Some(location),
            Self::DuplicateQubitInTick { location, .. } => Some(location),
            Self::NestedTick { location } => Some(location),
            Self::DuplicateQubitInMeasurement { location, .. } => Some(location),
            Self::BreakContinueOutsideLoop { location, .. } => Some(location),
            Self::InlineForRangeNotComptime { location, .. } => Some(location),
            Self::BreakInInlineFor { location } => Some(location),
            Self::ContinueInInlineFor { location } => Some(location),
            Self::OverlappingAlias(e) => Some(&e.location),
            Self::AliasSourceNotSlice { location, .. } => Some(location),
            Self::AliasRangeNotComptime { location, .. } => Some(location),
            Self::MissingReturn { location, .. } => Some(location),
            Self::ReturnWithoutValue { location, .. } => Some(location),
            Self::ModuleError { location, .. } => Some(location),
            Self::CatchOnNonErrorType { location, .. } => Some(location),
            Self::UndefinedType { location, .. } => Some(location),
            Self::UnresolvedType { location, .. } => Some(location),
            Self::SymbolTableLimitExceeded { .. } => None,
            Self::ScopeNestingLimitExceeded { .. } => None,
            Self::DuplicateSwitchCase { location, .. } => Some(location),
            Self::ReturnReferenceToLocal { location, .. } => Some(location),
            Self::ReturnSliceOfLocal { location, .. } => Some(location),
            Self::ReferenceEscapesScope { location, .. } => Some(location),
            Self::Other { .. } => None,
        }
    }
}

pub type SemanticResult<T> = Result<T, SemanticError>;

/// Multiple semantic errors collected during analysis.
///
/// This type is returned by `analyze_collecting_errors` to provide all
/// errors at once, allowing developers to fix multiple issues in one pass.
#[derive(Debug, Clone)]
pub struct SemanticErrors {
    errors: Vec<SemanticError>,
}

impl SemanticErrors {
    /// Create a new collection of semantic errors.
    pub fn new(errors: Vec<SemanticError>) -> Self {
        Self { errors }
    }

    /// Get the number of errors.
    pub fn len(&self) -> usize {
        self.errors.len()
    }

    /// Check if there are no errors.
    pub fn is_empty(&self) -> bool {
        self.errors.is_empty()
    }

    /// Get an iterator over the errors.
    pub fn iter(&self) -> impl Iterator<Item = &SemanticError> {
        self.errors.iter()
    }

    /// Get the first error, if any.
    pub fn first(&self) -> Option<&SemanticError> {
        self.errors.first()
    }

    /// Convert to a Vec of errors.
    pub fn into_vec(self) -> Vec<SemanticError> {
        self.errors
    }
}

impl std::fmt::Display for SemanticErrors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Found {} error(s):", self.errors.len())?;
        for (i, err) in self.errors.iter().enumerate() {
            writeln!(f, "  {}. {}", i + 1, err)?;
        }
        Ok(())
    }
}

impl std::error::Error for SemanticErrors {}

impl IntoIterator for SemanticErrors {
    type Item = SemanticError;
    type IntoIter = std::vec::IntoIter<SemanticError>;

    fn into_iter(self) -> Self::IntoIter {
        self.errors.into_iter()
    }
}

impl<'a> IntoIterator for &'a SemanticErrors {
    type Item = &'a SemanticError;
    type IntoIter = std::slice::Iter<'a, SemanticError>;

    fn into_iter(self) -> Self::IntoIter {
        self.errors.iter()
    }
}

// =============================================================================
// Input Size Limits
// =============================================================================

/// Maximum number of symbols allowed in the symbol table.
/// This prevents memory exhaustion from programs with excessive declarations.
pub const MAX_SYMBOL_COUNT: usize = 100_000;

/// Maximum scope nesting depth.
/// This prevents stack overflow from deeply nested scopes.
pub const MAX_SCOPE_DEPTH: usize = 256;

// =============================================================================
// Resolved Types
// =============================================================================

/// A validated bit width for integer types.
///
/// Guarantees the value is in the valid range 1-128 (matching Rust's max integer size).
/// Once constructed, a BitWidth is always valid - invalid values are unrepresentable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BitWidth(u16);

impl BitWidth {
    /// Minimum valid bit width.
    pub const MIN: u16 = 1;
    /// Maximum valid bit width (matches Rust's i128/u128).
    pub const MAX: u16 = 128;

    /// Create a new BitWidth if the value is valid (1-128).
    /// Returns None for invalid values like 0 or 129+.
    pub const fn new(bits: u16) -> Option<Self> {
        if bits >= Self::MIN && bits <= Self::MAX {
            Some(Self(bits))
        } else {
            None
        }
    }

    /// Create a BitWidth without validation.
    /// # Safety
    /// The caller must ensure bits is in range 1-128.
    pub const unsafe fn new_unchecked(bits: u16) -> Self {
        Self(bits)
    }

    /// Common bit width constants.
    pub const BITS_1: Self = Self(1);
    pub const BITS_8: Self = Self(8);
    pub const BITS_16: Self = Self(16);
    pub const BITS_32: Self = Self(32);
    pub const BITS_64: Self = Self(64);
    pub const BITS_128: Self = Self(128);

    /// Get the bit width value.
    pub const fn get(self) -> u16 {
        self.0
    }

    /// Create a BitWidth, panicking if invalid.
    /// Use only for compile-time known values.
    #[track_caller]
    pub const fn must(bits: u16) -> Self {
        match Self::new(bits) {
            Some(bw) => bw,
            None => panic!("invalid bit width"),
        }
    }
}

impl std::fmt::Display for BitWidth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl TryFrom<u16> for BitWidth {
    type Error = &'static str;

    fn try_from(bits: u16) -> Result<Self, Self::Error> {
        Self::new(bits).ok_or("bit width must be between 1 and 128")
    }
}

/// Resolved types after semantic analysis.
///
/// Unlike `TypeExpr` which is a syntactic representation, `Type` is the
/// semantic representation with all information resolved.
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    // Primitives
    Bool,
    // Arbitrary-width integers (like Zig: u1, u4, u7, u128, etc.)
    UInt {
        bits: BitWidth,
    }, // Unsigned integer with N bits
    IInt {
        bits: BitWidth,
    }, // Signed integer with N bits
    Usize, // Platform-dependent unsigned size
    Isize, // Platform-dependent signed size
    // Floating point
    F16,
    F32,
    F64,
    F128,
    A64, // Angle type (maps to PECOS Angle64)

    // Quantum types
    Qubit,
    Bit,
    /// Allocator with known capacity
    Allocator {
        capacity: Option<u64>,
    },

    // Compound types
    Array {
        element: Box<Type>,
        size: Option<u64>,
    },
    Slice {
        element: Box<Type>,
    },
    Set {
        element: Box<Type>,
    },
    Pointer {
        pointee: Box<Type>,
        is_const: bool,
        is_many: bool,
    },
    Optional {
        inner: Box<Type>,
    },
    ErrorUnion {
        error: Box<Type>,
        payload: Box<Type>,
    },
    /// Collected errors: []E!T - array of errors E with value T (both, not either/or)
    /// Used for QEC-style error collection
    CollectedErrors {
        error: Box<Type>,
        payload: Box<Type>,
    },
    /// Tuple type: (T1, T2, ...)
    Tuple {
        elements: Vec<Type>,
    },

    // Function type
    Function {
        params: Vec<Type>,
        return_type: Box<Type>,
    },

    // Named/user-defined types
    Struct {
        name: String,
        fields: Vec<(String, Type)>,
    },
    Enum {
        name: String,
        variants: Vec<String>,
    },
    Union {
        name: String,
        /// Fields of the union: (name, optional payload type)
        fields: Vec<(String, Option<Type>)>,
        /// Is this union tagged (has an auto-generated or external tag)?
        is_tagged: bool,
    },
    /// Classical error set - crashes if unhandled
    /// Each variant is (name, optional_associated_data_type)
    ErrorSet {
        name: String,
        errors: Vec<(String, Option<Box<Type>>)>,
    },
    /// Quantum fault set - collected in try blocks
    /// Each variant is (name, optional_associated_data_type)
    FaultSet {
        name: String,
        faults: Vec<(String, Option<Box<Type>>)>,
    },
    /// The `anyerror` type - represents any error type
    AnyError,
    /// The `anyfault` type - represents any fault type
    AnyFault,

    // Special types
    Unit,                // Unit type - has exactly one value
    Type,                // The metatype (type of types)
    Comptime(Box<Type>), // Comptime-known value of this type
    Never,               // Bottom type (for functions that don't return)

    /// Imported module type
    Module {
        /// Absolute path to the module file
        path: String,
        /// Exported symbols: name -> (kind, type)
        exports: std::collections::BTreeMap<String, (ModuleExportKind, Type)>,
    },

    // Unknown (for type inference)
    Unknown,
}

/// Kind of exported symbol from a module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModuleExportKind {
    Function,
    Const,
    Type,
    ErrorSet,
    FaultSet,
}

impl Type {
    /// Check if this type is numeric.
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            Type::UInt { .. }
                | Type::IInt { .. }
                | Type::Usize
                | Type::Isize
                | Type::F16
                | Type::F32
                | Type::F64
                | Type::F128
                | Type::A64
        )
    }

    /// Check if this type is an integer.
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            Type::UInt { .. } | Type::IInt { .. } | Type::Usize | Type::Isize
        )
    }

    /// Check if this type is a floating point.
    pub fn is_float(&self) -> bool {
        matches!(self, Type::F16 | Type::F32 | Type::F64 | Type::F128)
    }

    /// Check if this type is a quantum type.
    pub fn is_quantum(&self) -> bool {
        matches!(self, Type::Qubit | Type::Bit | Type::Allocator { .. })
    }

    /// Get the display name for error messages.
    pub fn display_name(&self) -> String {
        match self {
            Type::Bool => "bool".to_string(),
            Type::UInt { bits } => format!("u{bits}"),
            Type::IInt { bits } => format!("i{bits}"),
            Type::Usize => "usize".to_string(),
            Type::Isize => "isize".to_string(),
            Type::F16 => "f16".to_string(),
            Type::F32 => "f32".to_string(),
            Type::F64 => "f64".to_string(),
            Type::F128 => "f128".to_string(),
            Type::A64 => "a64".to_string(),
            Type::Qubit => "qubit".to_string(),
            Type::Bit => "bit".to_string(),
            Type::Allocator { capacity: Some(n) } => format!("qalloc({n})"),
            Type::Allocator { capacity: None } => "qalloc".to_string(),
            Type::Array {
                element,
                size: Some(n),
            } => format!("[{n}]{}", element.display_name()),
            Type::Array {
                element,
                size: None,
            } => format!("[_]{}", element.display_name()),
            Type::Slice { element } => format!("[]{}", element.display_name()),
            Type::Pointer {
                pointee, is_const, ..
            } => {
                if *is_const {
                    format!("*const {}", pointee.display_name())
                } else {
                    format!("*{}", pointee.display_name())
                }
            }
            Type::Optional { inner } => format!("?{}", inner.display_name()),
            Type::ErrorUnion { error, payload } => {
                // Syntax is E!T where E is error type and T is payload type
                format!("{}!{}", error.display_name(), payload.display_name())
            }
            Type::CollectedErrors { error, payload } => {
                // Syntax is []E!T where E is error type and T is payload type
                format!("[]{}!{}", error.display_name(), payload.display_name())
            }
            Type::Tuple { elements } => {
                let elem_strs: Vec<_> = elements.iter().map(|e| e.display_name()).collect();
                format!("({})", elem_strs.join(", "))
            }
            Type::Function {
                params,
                return_type,
            } => {
                let params_str: Vec<_> = params.iter().map(|p| p.display_name()).collect();
                format!(
                    "fn({}) {}",
                    params_str.join(", "),
                    return_type.display_name()
                )
            }
            Type::Struct { name, .. } => name.clone(),
            Type::Enum { name, .. } => name.clone(),
            Type::Union { name, .. } => name.clone(),
            Type::ErrorSet { name, .. } => format!("error.{}", name),
            Type::FaultSet { name, .. } => format!("fault.{}", name),
            Type::AnyError => "anyerror".to_string(),
            Type::AnyFault => "anyfault".to_string(),
            Type::Unit => "unit".to_string(),
            Type::Type => "type".to_string(),
            Type::Comptime(inner) => format!("comptime {}", inner.display_name()),
            Type::Never => "noreturn".to_string(),
            Type::Unknown => "unknown".to_string(),
            Type::Set { element } => format!("Set({})", element.display_name()),
            Type::Module { path, .. } => format!("module({})", path),
        }
    }

    /// Check if this type contains `Unknown` anywhere in its structure.
    ///
    /// Returns `true` if this type is `Unknown` or contains `Unknown` in any
    /// nested position (e.g., `Array { element: Unknown, .. }`).
    pub fn contains_unknown(&self) -> bool {
        match self {
            Type::Unknown => true,

            // Primitives never contain Unknown
            Type::Bool
            | Type::UInt { .. }
            | Type::IInt { .. }
            | Type::Usize
            | Type::Isize
            | Type::F16
            | Type::F32
            | Type::F64
            | Type::F128
            | Type::A64
            | Type::Qubit
            | Type::Bit
            | Type::Unit
            | Type::Type
            | Type::Never
            | Type::AnyError
            | Type::AnyFault => false,

            // Check allocator (no nested types)
            Type::Allocator { .. } => false,

            // Container types - check nested types
            Type::Array { element, .. } => element.contains_unknown(),
            Type::Slice { element } => element.contains_unknown(),
            Type::Set { element } => element.contains_unknown(),
            Type::Pointer { pointee, .. } => pointee.contains_unknown(),
            Type::Optional { inner } => inner.contains_unknown(),
            Type::Comptime(inner) => inner.contains_unknown(),

            // Compound types with two nested types
            Type::ErrorUnion { error, payload } => {
                error.contains_unknown() || payload.contains_unknown()
            }
            Type::CollectedErrors { error, payload } => {
                error.contains_unknown() || payload.contains_unknown()
            }

            // Collections of types
            Type::Tuple { elements } => elements.iter().any(|e| e.contains_unknown()),
            Type::Function {
                params,
                return_type,
            } => params.iter().any(|p| p.contains_unknown()) || return_type.contains_unknown(),

            // Named types with fields
            Type::Struct { fields, .. } => fields.iter().any(|(_, ty)| ty.contains_unknown()),
            Type::Union { fields, .. } => fields
                .iter()
                .any(|(_, opt_ty)| opt_ty.as_ref().is_some_and(|ty| ty.contains_unknown())),

            // Named types without nested types to check
            Type::Enum { .. } | Type::ErrorSet { .. } | Type::FaultSet { .. } => false,

            // Module exports
            Type::Module { exports, .. } => exports.values().any(|(_, ty)| ty.contains_unknown()),
        }
    }

    /// Try to resolve this type, returning `Some(ResolvedType)` if it contains
    /// no `Unknown` anywhere in its structure, or `None` otherwise.
    pub fn resolve(&self) -> Option<ResolvedType> {
        if self.contains_unknown() {
            None
        } else {
            Some(ResolvedType(self.clone()))
        }
    }

    /// Check if this type is fully resolved (contains no Unknown).
    pub fn is_resolved(&self) -> bool {
        !self.contains_unknown()
    }
}

/// A type that is guaranteed to contain no `Unknown` anywhere in its structure.
///
/// This wrapper provides type-level safety at API boundaries where we need to
/// guarantee that type inference has completed. Use `Type::resolve()` to create
/// a `ResolvedType` from a `Type`.
///
/// # Example
///
/// ```rust
/// use zlup::semantic::Type;
///
/// // ResolvedType guarantees no Unknown variants
/// let ty = Type::Bool.resolve().expect("Bool is resolved");
/// assert!(matches!(ty.as_type(), Type::Bool));
///
/// // Unknown types cannot become ResolvedType
/// let unknown = Type::Unknown.resolve();
/// assert!(unknown.is_none());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedType(Type);

impl ResolvedType {
    /// Get the underlying `Type`.
    ///
    /// The returned type is guaranteed to not contain `Unknown`.
    pub fn as_type(&self) -> &Type {
        &self.0
    }

    /// Unwrap into the underlying `Type`.
    ///
    /// The returned type is guaranteed to not contain `Unknown`.
    pub fn into_type(self) -> Type {
        self.0
    }

    /// Get the display name for error messages.
    pub fn display_name(&self) -> String {
        self.0.display_name()
    }

    /// Check if this type is numeric.
    pub fn is_numeric(&self) -> bool {
        self.0.is_numeric()
    }

    /// Check if this type is an integer.
    pub fn is_integer(&self) -> bool {
        self.0.is_integer()
    }

    /// Check if this type is a floating point.
    pub fn is_float(&self) -> bool {
        self.0.is_float()
    }

    /// Check if this type is a quantum type.
    pub fn is_quantum(&self) -> bool {
        self.0.is_quantum()
    }
}

impl std::fmt::Display for ResolvedType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.display_name())
    }
}

// =============================================================================
// Symbol Table
// =============================================================================

/// Symbol kinds.
#[derive(Debug, Clone)]
pub enum SymbolKind {
    /// Local or global variable
    Variable {
        ty: Type,
        is_const: bool,
        is_comptime: bool,
    },
    /// Function
    Function {
        params: Vec<(String, Type)>,
        return_type: Type,
        is_pub: bool,
        /// Which parameter indices are comptime (for generic functions)
        comptime_param_indices: Vec<usize>,
        /// Original function declaration (for instantiation of generics)
        original_decl: Option<Box<crate::ast::FnDecl>>,
    },
    /// Type definition (struct, enum)
    TypeDef { ty: Type },
    /// Qubit allocator
    Allocator { capacity: Option<u64> },
    /// Parameter
    Parameter { ty: Type, is_comptime: bool },
}

/// A symbol in the symbol table.
#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
    pub kind: SymbolKind,
    pub location: Option<SourceLocation>,
}

/// A scope containing symbols.
#[derive(Debug)]
pub struct Scope {
    /// Symbols defined in this scope
    symbols: BTreeMap<String, Symbol>,
    /// Parent scope index (None for global scope)
    parent: Option<usize>,
    /// Scope kind for context
    kind: ScopeKind,
}

/// Scope kinds for different contexts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeKind {
    Global,
    Function,
    Block,
    Loop,
    Struct,
}

/// Symbol table with nested scopes.
#[derive(Debug)]
pub struct SymbolTable {
    scopes: Vec<Scope>,
    current_scope: usize,
}

impl SymbolTable {
    /// Create a new symbol table with a global scope.
    pub fn new() -> Self {
        let global_scope = Scope {
            symbols: BTreeMap::new(),
            parent: None,
            kind: ScopeKind::Global,
        };
        Self {
            scopes: vec![global_scope],
            current_scope: 0,
        }
    }

    /// Push a new scope.
    pub fn push_scope(&mut self, kind: ScopeKind) -> SemanticResult<()> {
        // Check scope nesting limit
        let depth = self.scope_depth();
        if depth >= MAX_SCOPE_DEPTH {
            return Err(SemanticError::ScopeNestingLimitExceeded {
                depth: depth + 1,
                max: MAX_SCOPE_DEPTH,
            });
        }

        let new_scope = Scope {
            symbols: BTreeMap::new(),
            parent: Some(self.current_scope),
            kind,
        };
        self.scopes.push(new_scope);
        self.current_scope = self.scopes.len() - 1;
        Ok(())
    }

    /// Calculate current scope nesting depth.
    fn scope_depth(&self) -> usize {
        let mut depth = 0;
        let mut scope_idx = Some(self.current_scope);
        while let Some(idx) = scope_idx {
            depth += 1;
            scope_idx = self.scopes[idx].parent;
        }
        depth
    }

    /// Pop the current scope.
    pub fn pop_scope(&mut self) {
        if let Some(parent) = self.scopes[self.current_scope].parent {
            self.current_scope = parent;
        }
    }

    /// Define a symbol in the current scope.
    pub fn define(&mut self, symbol: Symbol) -> SemanticResult<()> {
        // Check symbol table size limit
        let total_symbols: usize = self.scopes.iter().map(|s| s.symbols.len()).sum();
        if total_symbols >= MAX_SYMBOL_COUNT {
            return Err(SemanticError::SymbolTableLimitExceeded {
                count: total_symbols + 1,
                max: MAX_SYMBOL_COUNT,
            });
        }

        let scope = &mut self.scopes[self.current_scope];
        if scope.symbols.contains_key(&symbol.name) {
            return Err(SemanticError::DuplicateSymbol {
                name: symbol.name.clone(),
                location: symbol.location.clone().unwrap_or_default(),
            });
        }
        scope.symbols.insert(symbol.name.clone(), symbol);
        Ok(())
    }

    /// Look up a symbol, searching parent scopes.
    pub fn lookup(&self, name: &str) -> Option<&Symbol> {
        let mut scope_idx = Some(self.current_scope);
        while let Some(idx) = scope_idx {
            let scope = &self.scopes[idx];
            if let Some(symbol) = scope.symbols.get(name) {
                return Some(symbol);
            }
            scope_idx = scope.parent;
        }
        None
    }

    /// Look up a symbol only in the current scope.
    pub fn lookup_current(&self, name: &str) -> Option<&Symbol> {
        self.scopes[self.current_scope].symbols.get(name)
    }

    /// Get the current scope kind.
    pub fn current_scope_kind(&self) -> ScopeKind {
        self.scopes[self.current_scope].kind
    }

    /// Check if we're inside a loop.
    pub fn in_loop(&self) -> bool {
        let mut scope_idx = Some(self.current_scope);
        while let Some(idx) = scope_idx {
            if self.scopes[idx].kind == ScopeKind::Loop {
                return true;
            }
            scope_idx = self.scopes[idx].parent;
        }
        false
    }

    /// Find an error set containing the given variant name.
    /// Returns the Type::ErrorSet if found.
    pub fn find_error_set_by_variant(&self, variant_name: &str) -> Option<Type> {
        let mut scope_idx = Some(self.current_scope);
        while let Some(idx) = scope_idx {
            let scope = &self.scopes[idx];
            for symbol in scope.symbols.values() {
                if let SymbolKind::TypeDef { ty } = &symbol.kind
                    && let Type::ErrorSet { name, errors } = ty
                    && errors.iter().any(|(n, _)| n == variant_name)
                {
                    return Some(Type::ErrorSet {
                        name: name.clone(),
                        errors: errors.clone(),
                    });
                }
            }
            scope_idx = scope.parent;
        }
        None
    }

    /// Find a fault set containing the given variant name.
    /// Returns the Type::FaultSet if found.
    pub fn find_fault_set_by_variant(&self, variant_name: &str) -> Option<Type> {
        let mut scope_idx = Some(self.current_scope);
        while let Some(idx) = scope_idx {
            let scope = &self.scopes[idx];
            for symbol in scope.symbols.values() {
                if let SymbolKind::TypeDef { ty } = &symbol.kind
                    && let Type::FaultSet { name, faults } = ty
                    && faults.iter().any(|(n, _)| n == variant_name)
                {
                    return Some(Type::FaultSet {
                        name: name.clone(),
                        faults: faults.clone(),
                    });
                }
            }
            scope_idx = scope.parent;
        }
        None
    }
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Qubit State Tracking
// =============================================================================

/// Qubit slot state - exactly two states as per Zluppy design.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QubitState {
    /// Initial state, or after measurement. Cannot apply gates.
    Unprepared,
    /// Ready for gate operations. Result of prepare().
    Prepared,
}

impl QubitState {
    /// Check if gates can be applied to this qubit.
    pub fn can_apply_gate(&self) -> bool {
        matches!(self, QubitState::Prepared)
    }
}

/// Information about an allocator for tracking.
#[derive(Debug, Clone)]
pub struct AllocatorInfo {
    /// Name of the allocator variable.
    pub name: String,
    /// Known capacity (if comptime-known).
    pub capacity: Option<usize>,
    /// Parent allocator (for child allocators).
    pub parent: Option<String>,
    /// State of each qubit slot.
    pub slot_states: Vec<QubitState>,
}

impl AllocatorInfo {
    /// Create a new allocator with given capacity.
    pub fn new(name: impl Into<String>, capacity: usize) -> Self {
        Self {
            name: name.into(),
            capacity: Some(capacity),
            parent: None,
            slot_states: vec![QubitState::Unprepared; capacity],
        }
    }

    /// Create a child allocator.
    pub fn child(name: impl Into<String>, parent: impl Into<String>, capacity: usize) -> Self {
        Self {
            name: name.into(),
            capacity: Some(capacity),
            parent: Some(parent.into()),
            slot_states: vec![QubitState::Unprepared; capacity],
        }
    }

    /// Create an allocator with unknown capacity.
    pub fn unknown(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            capacity: None,
            parent: None,
            slot_states: Vec::new(),
        }
    }

    /// Prepare a specific slot.
    pub fn prepare_slot(&mut self, index: usize) -> Result<(), (usize, QubitState)> {
        if let Some(state) = self.slot_states.get_mut(index) {
            if *state == QubitState::Prepared {
                return Err((index, *state));
            }
            *state = QubitState::Prepared;
            Ok(())
        } else if self.capacity.is_none() {
            // Unknown capacity - assume valid
            Ok(())
        } else {
            Err((index, QubitState::Unprepared))
        }
    }

    /// Prepare all slots.
    pub fn prepare_all(&mut self) {
        for state in &mut self.slot_states {
            *state = QubitState::Prepared;
        }
    }

    /// Measure a specific slot (transitions to unprepared).
    pub fn measure_slot(&mut self, index: usize) {
        if let Some(state) = self.slot_states.get_mut(index) {
            *state = QubitState::Unprepared;
        }
    }

    /// Get the state of a slot.
    pub fn get_state(&self, index: usize) -> Option<QubitState> {
        self.slot_states.get(index).copied()
    }

    /// Check if an index is in bounds.
    pub fn is_in_bounds(&self, index: usize) -> bool {
        match self.capacity {
            Some(cap) => index < cap,
            None => true, // Unknown capacity - assume valid
        }
    }
}

/// Tracks qubit states across the program.
#[derive(Debug, Default)]
pub struct QubitStateTracker {
    /// Allocators by name.
    allocators: BTreeMap<String, AllocatorInfo>,
}

impl QubitStateTracker {
    /// Create a new tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an allocator.
    pub fn register_allocator(&mut self, info: AllocatorInfo) {
        self.allocators.insert(info.name.clone(), info);
    }

    /// Get an allocator by name.
    pub fn get_allocator(&self, name: &str) -> Option<&AllocatorInfo> {
        self.allocators.get(name)
    }

    /// Get a mutable allocator by name.
    pub fn get_allocator_mut(&mut self, name: &str) -> Option<&mut AllocatorInfo> {
        self.allocators.get_mut(name)
    }

    /// Check if a qubit slot is prepared for gate operations.
    pub fn is_prepared(&self, allocator: &str, index: usize) -> Option<bool> {
        self.allocators
            .get(allocator)
            .and_then(|a| a.get_state(index))
            .map(|s| s == QubitState::Prepared)
    }

    /// Validate a qubit reference for gate operations.
    pub fn validate_for_gate(
        &self,
        allocator: &str,
        index: usize,
        location: &SourceLocation,
    ) -> SemanticResult<()> {
        let alloc =
            self.allocators
                .get(allocator)
                .ok_or_else(|| SemanticError::AllocatorNotFound {
                    name: allocator.to_string(),
                    location: location.clone(),
                })?;

        // Check bounds
        if !alloc.is_in_bounds(index) {
            return Err(SemanticError::QubitIndexOutOfBounds {
                allocator: allocator.to_string(),
                index,
                capacity: alloc.capacity.unwrap_or(0),
                location: location.clone(),
            });
        }

        // Check state (only if capacity is known)
        if alloc.capacity.is_some()
            && let Some(state) = alloc.get_state(index)
            && !state.can_apply_gate()
        {
            return Err(SemanticError::QubitNotPrepared {
                allocator: allocator.to_string(),
                index,
                location: location.clone(),
            });
        }

        Ok(())
    }
}

// =============================================================================
// NASA Power of 10 Checks
// =============================================================================

/// Maximum allowed loop bound (NASA Power of 10 Rule 2).
pub const MAX_LOOP_BOUND: usize = 1_000_000;

/// Tracks function calls to detect recursion.
#[derive(Debug, Default)]
pub struct RecursionTracker {
    /// Currently active function call stack.
    call_stack: BTreeSet<String>,
}

impl RecursionTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Enter a function. Returns error if already in the call stack.
    pub fn enter_function(&mut self, name: &str, location: &SourceLocation) -> SemanticResult<()> {
        if self.call_stack.contains(name) {
            return Err(SemanticError::RecursionDetected {
                name: name.to_string(),
                location: location.clone(),
            });
        }
        self.call_stack.insert(name.to_string());
        Ok(())
    }

    /// Exit a function.
    pub fn exit_function(&mut self, name: &str) {
        self.call_stack.remove(name);
    }

    /// Check if a function is in the call stack.
    pub fn is_in_call_stack(&self, name: &str) -> bool {
        self.call_stack.contains(name)
    }
}

// =============================================================================
// Semantic Analyzer
// =============================================================================

/// Semantic analyzer for Zluppy programs.
pub struct SemanticAnalyzer {
    /// Symbol table
    pub symbols: SymbolTable,
    /// Qubit state tracker
    pub qubit_states: QubitStateTracker,
    /// Recursion tracker (NASA Power of 10)
    pub recursion_tracker: RecursionTracker,
    /// Current function return type (for return statement checking)
    current_return_type: Option<Type>,
    /// Current function name (for call tracking)
    current_function: Option<String>,
    /// Whether to enforce strict qubit state checking
    strict_mode: bool,
    /// Errors collected during analysis
    errors: Vec<SemanticError>,
    /// Comptime evaluator for compile-time expression evaluation
    comptime: ComptimeEvaluator,
    /// Storage for comptime-evaluated values by expression location
    comptime_values: BTreeMap<String, ComptimeValue>,
    /// Module loader for handling @import
    module_loader: ModuleLoader,
    /// Current file path (for resolving relative imports)
    current_file: Option<std::path::PathBuf>,
    /// Loop nesting depth (for validating break/continue)
    loop_depth: usize,
    /// Inline for nesting depth (to disallow break/continue in inline for)
    inline_for_depth: usize,
    /// Tick nesting depth (to disallow nested ticks)
    tick_depth: usize,
    /// Call graph for mutual recursion detection (strict mode)
    /// Maps function name -> set of functions it calls
    call_graph: BTreeMap<String, BTreeSet<String>>,
    /// Set of user-defined function names (for call graph filtering)
    user_functions: BTreeSet<String>,
    /// Cache of generic function instantiations.
    /// Key: (original function name, serialized comptime arg values)
    /// Value: mangled name of the specialized function
    generic_instantiations: BTreeMap<(String, String), String>,
    /// Storage for specialized function declarations generated from generics
    specialized_functions: Vec<crate::ast::FnDecl>,
    /// Alias tracking for overlap detection
    /// Key: alias name, Value: AliasInfo
    aliases: BTreeMap<String, AliasInfo>,
    /// Gate registry for custom gate declarations
    gate_registry: BTreeMap<String, GateSignature>,
}

/// Where a registered gate came from. Determines which redeclarations are
/// allowed: built-ins may be redeclared only with their exact signature, while a
/// user gate (declared or defined) may not be redeclared at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateOrigin {
    /// A built-in gate provided by the language/backend.
    Builtin,
    /// `declare gate name(...)(...);` -- an opaque target/backend gate.
    TargetDeclared,
    /// `gate name(...)(...) { ... }` -- a composite gate defined inline.
    CompositeDefined,
}

/// Signature of a registered gate (built-in or custom).
#[derive(Debug, Clone)]
pub struct GateSignature {
    pub name: String,
    pub num_params: usize,
    pub num_qubits: usize,
    pub origin: GateOrigin,
}

/// Information about an alias for overlap detection.
#[derive(Debug, Clone)]
pub struct AliasInfo {
    /// Name of the alias
    pub name: String,
    /// Name of the source variable
    pub source: String,
    /// Static range if known (start..end)
    pub range: Option<(i64, i64)>,
    /// Source location for error reporting
    pub location: SourceLocation,
}

impl SemanticAnalyzer {
    /// Create a new semantic analyzer with strict mode enabled (the default).
    ///
    /// Strict mode enforces safety guarantees required for quantum programs:
    /// - Qubit state checking (gates on unprepared qubits are errors)
    /// - Loop bounds checked against MAX_LOOP_BOUND
    /// - Recursion is prohibited (use FFI with Rust if needed)
    /// - Explicit return statements required
    ///
    /// This is the recommended mode for quantum programs.
    pub fn new() -> Self {
        let mut analyzer = Self {
            symbols: SymbolTable::new(),
            qubit_states: QubitStateTracker::new(),
            recursion_tracker: RecursionTracker::new(),
            current_return_type: None,
            current_function: None,
            strict_mode: true, // Strict by default for quantum safety
            errors: Vec::new(),
            comptime: ComptimeEvaluator::new(),
            comptime_values: BTreeMap::new(),
            module_loader: ModuleLoader::new(),
            current_file: None,
            loop_depth: 0,
            inline_for_depth: 0,
            tick_depth: 0,
            call_graph: BTreeMap::new(),
            user_functions: BTreeSet::new(),
            generic_instantiations: BTreeMap::new(),
            specialized_functions: Vec::new(),
            aliases: BTreeMap::new(),
            gate_registry: BTreeMap::new(),
        };
        analyzer.define_builtins();
        analyzer.populate_builtin_gates();
        analyzer
    }

    /// Set the current file path for resolving relative imports.
    pub fn set_current_file(&mut self, path: impl Into<std::path::PathBuf>) {
        self.current_file = Some(path.into());
    }

    /// Create a new semantic analyzer with permissive mode (strict mode disabled).
    ///
    /// Permissive mode relaxes some safety checks:
    /// - Qubit state checking is not enforced
    /// - Loop bounds are not checked
    ///
    /// Note: Recursion is always prohibited (use FFI with Rust if needed).
    ///
    /// Use this mode only when interfacing with external code or for testing.
    /// For production quantum programs, use `new()` (strict mode).
    pub fn new_permissive() -> Self {
        let mut analyzer = Self::new();
        analyzer.strict_mode = false;
        analyzer
    }

    /// Enable or disable strict mode.
    pub fn set_strict_mode(&mut self, strict: bool) {
        self.strict_mode = strict;
    }

    /// Define built-in functions and types.
    fn define_builtins(&mut self) {
        // Built-in quantum gates are recognized by name during call analysis
        // Built-in types are handled in resolve_type

        // Define qalloc as a built-in function that returns an allocator
        let _ = self.symbols.define(Symbol {
            name: "qalloc".to_string(),
            kind: SymbolKind::Function {
                params: vec![(
                    "capacity".to_string(),
                    Type::UInt {
                        bits: BitWidth::BITS_32,
                    },
                )],
                return_type: Type::Allocator { capacity: None },
                is_pub: true,
                comptime_param_indices: vec![],
                original_decl: None,
            },
            location: None,
        });

        // Define measure as a built-in
        let _ = self.symbols.define(Symbol {
            name: "measure".to_string(),
            kind: SymbolKind::Function {
                params: vec![("target".to_string(), Type::Qubit)],
                return_type: Type::Bit,
                is_pub: true,
                comptime_param_indices: vec![],
                original_decl: None,
            },
            location: None,
        });
    }

    /// Populate the gate registry with all built-in gates.
    fn populate_builtin_gates(&mut self) {
        let builtin_gates: &[(&str, usize, usize)] = &[
            // (name, num_params, num_qubits)
            ("x", 0, 1),
            ("y", 0, 1),
            ("z", 0, 1),
            ("h", 0, 1),
            ("t", 0, 1),
            ("tdg", 0, 1),
            ("sx", 0, 1),
            ("sy", 0, 1),
            ("sz", 0, 1),
            ("sxdg", 0, 1),
            ("sydg", 0, 1),
            ("szdg", 0, 1),
            ("rx", 1, 1),
            ("ry", 1, 1),
            ("rz", 1, 1),
            ("cx", 0, 2),
            ("cy", 0, 2),
            ("cz", 0, 2),
            ("ch", 0, 2),
            ("sxx", 0, 2),
            ("syy", 0, 2),
            ("szz", 0, 2),
            ("sxxdg", 0, 2),
            ("syydg", 0, 2),
            ("szzdg", 0, 2),
            ("rzz", 1, 2),
            ("swap", 0, 2),
            ("iswap", 0, 2),
            ("ccx", 0, 3),
            ("f", 0, 1),
            ("fdg", 0, 1),
            ("f4", 0, 1),
            ("f4dg", 0, 1),
            ("pz", 0, 1),
        ];

        for &(name, num_params, num_qubits) in builtin_gates {
            self.gate_registry.insert(
                name.to_string(),
                GateSignature {
                    name: name.to_string(),
                    num_params,
                    num_qubits,
                    origin: GateOrigin::Builtin,
                },
            );
        }
    }

    /// Analyze a program.
    pub fn analyze(&mut self, program: &Program) -> SemanticResult<()> {
        log::debug!(
            "Analyzing program with {} declarations (strict={})",
            program.declarations.len(),
            self.strict_mode
        );

        // First pass: collect all top-level declarations
        log::trace!("Pass 1: collecting top-level declarations");
        for decl in &program.declarations {
            self.collect_top_level(decl)?;
        }

        // Second pass: analyze bodies
        log::trace!("Pass 2: analyzing declaration bodies");
        for decl in &program.declarations {
            self.analyze_top_level(decl)?;
        }

        // Check for mutual recursion in strict mode
        if self.strict_mode {
            log::trace!("Pass 3: checking call graph for cycles");
            self.check_call_graph_cycles()?;
        }

        // Validation pass: ensure no unresolved types remain in symbol table
        // This is a safety net to catch any Unknown types that slip through
        log::trace!("Pass 4: validating resolved types");
        self.validate_types_resolved();

        if self.errors.is_empty() {
            log::debug!("Semantic analysis completed successfully");
            Ok(())
        } else {
            log::debug!("Semantic analysis found {} error(s)", self.errors.len());
            // Return the first error for backward compatibility
            Err(self.errors.remove(0))
        }
    }

    /// Analyze a program and return all collected errors.
    ///
    /// Unlike `analyze()` which returns the first error, this method
    /// returns all errors found during analysis, allowing developers
    /// to fix multiple issues in one pass.
    ///
    /// # Example
    ///
    /// ```rust
    /// use zlup::semantic::SemanticAnalyzer;
    ///
    /// // Program with multiple errors
    /// let source = "fn main() -> unit { x := undefined; y := also_undefined; return unit; }";
    /// let program = zlup::parse(source).expect("parse failed");
    /// let mut analyzer = SemanticAnalyzer::new();
    /// match analyzer.analyze_collecting_errors(&program) {
    ///     Ok(()) => println!("No errors"),
    ///     Err(errors) => {
    ///         // All errors collected, not just the first one
    ///         assert!(errors.len() >= 2);
    ///     }
    /// }
    /// ```
    pub fn analyze_collecting_errors(&mut self, program: &Program) -> Result<(), SemanticErrors> {
        // First pass: collect all top-level declarations
        for decl in &program.declarations {
            if let Err(e) = self.collect_top_level(decl) {
                self.errors.push(e);
            }
        }

        // Second pass: analyze bodies
        for decl in &program.declarations {
            if let Err(e) = self.analyze_top_level(decl) {
                self.errors.push(e);
            }
        }

        // Validation pass: ensure no unresolved types remain in symbol table
        self.validate_types_resolved();

        if self.errors.is_empty() {
            Ok(())
        } else {
            Err(SemanticErrors::new(std::mem::take(&mut self.errors)))
        }
    }

    /// Get the number of errors collected so far.
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    /// Get all collected errors without consuming them.
    pub fn errors(&self) -> &[SemanticError] {
        &self.errors
    }

    /// Take all collected errors, leaving the error list empty.
    pub fn take_errors(&mut self) -> Vec<SemanticError> {
        std::mem::take(&mut self.errors)
    }

    /// Check for cycles in the call graph (mutual recursion detection).
    /// Uses DFS to detect back edges in the call graph.
    fn check_call_graph_cycles(&self) -> SemanticResult<()> {
        let mut visited = BTreeSet::new();
        let mut rec_stack = BTreeSet::new();

        for func in &self.user_functions {
            if !visited.contains(func)
                && let Some(cycle_func) = self.dfs_detect_cycle(func, &mut visited, &mut rec_stack)
            {
                return Err(SemanticError::RecursionDetected {
                    name: cycle_func,
                    location: SourceLocation::default(),
                });
            }
        }
        Ok(())
    }

    /// DFS helper for cycle detection. Returns the name of a function in a cycle if found.
    fn dfs_detect_cycle(
        &self,
        func: &str,
        visited: &mut BTreeSet<String>,
        rec_stack: &mut BTreeSet<String>,
    ) -> Option<String> {
        visited.insert(func.to_string());
        rec_stack.insert(func.to_string());

        if let Some(callees) = self.call_graph.get(func) {
            for callee in callees {
                // If callee is in the current recursion stack, we found a cycle
                if rec_stack.contains(callee) {
                    return Some(callee.clone());
                }
                // If not visited, recurse
                if !visited.contains(callee)
                    && let Some(cycle) = self.dfs_detect_cycle(callee, visited, rec_stack)
                {
                    return Some(cycle);
                }
            }
        }

        rec_stack.remove(func);
        None
    }

    /// Validate that no Unknown types remain in the symbol table.
    ///
    /// This is a safety net that runs after semantic analysis to ensure
    /// all types are fully resolved before code generation.
    fn validate_types_resolved(&mut self) {
        for scope in &self.symbols.scopes {
            for symbol in scope.symbols.values() {
                let (ty, context) = match &symbol.kind {
                    SymbolKind::Variable { ty, .. } => (ty, "variable"),
                    SymbolKind::Function { return_type, .. } => {
                        (return_type, "function return type")
                    }
                    SymbolKind::TypeDef { ty } => (ty, "type definition"),
                    SymbolKind::Parameter { ty, .. } => (ty, "parameter"),
                    SymbolKind::Allocator { .. } => continue,
                };

                // Skip Module types - they can have Unknown in exports due to
                // incomplete type extraction from imported modules
                if matches!(ty, Type::Module { .. }) {
                    continue;
                }

                if ty.contains_unknown() {
                    self.errors.push(SemanticError::UnresolvedType {
                        ty: ty.display_name(),
                        context: format!("{} '{}'", context, symbol.name),
                        location: symbol.location.clone().unwrap_or_default(),
                    });
                }

                // Also check function parameter types
                if let SymbolKind::Function { params, .. } = &symbol.kind {
                    for (param_name, param_ty) in params {
                        if param_ty.contains_unknown() {
                            self.errors.push(SemanticError::UnresolvedType {
                                ty: param_ty.display_name(),
                                context: format!(
                                    "parameter '{}' of function '{}'",
                                    param_name, symbol.name
                                ),
                                location: symbol.location.clone().unwrap_or_default(),
                            });
                        }
                    }
                }
            }
        }
    }

    /// Register a user-declared gate (target declaration or composite definition).
    ///
    /// Rejects a duplicate user gate (the same name declared/defined twice).
    /// `declare gate` introduces an opaque target/backend gate and `gate ... {}`
    /// a composite definition; PECOS treats either as a complete declaration, so
    /// a second one of the same name -- including declare-then-define -- is a
    /// duplicate, not a forward declaration.
    ///
    /// A built-in gate may only be redeclared with its exact signature (a
    /// harmless no-op); shadowing a built-in with a different arity/parameter
    /// count is rejected, because built-in names are parsed with a fixed
    /// parameterization and a mismatched redeclaration would be uncallable.
    fn register_user_gate(
        &mut self,
        name: &str,
        num_params: usize,
        num_qubits: usize,
        origin: GateOrigin,
    ) -> SemanticResult<()> {
        if let Some(existing) = self.gate_registry.get(name) {
            match existing.origin {
                GateOrigin::Builtin => {
                    if existing.num_params != num_params || existing.num_qubits != num_qubits {
                        return Err(SemanticError::Other {
                            message: format!(
                                "cannot redeclare built-in gate '{name}' with a different \
                                 signature: built-in '{name}' takes {} parameter(s) and {} \
                                 qubit(s)",
                                existing.num_params, existing.num_qubits
                            ),
                        });
                    }
                    // Exact-signature redeclaration of a built-in: no-op.
                    return Ok(());
                }
                GateOrigin::TargetDeclared => {
                    return Err(SemanticError::Other {
                        message: format!(
                            "gate '{name}' is already declared as a target gate; \
                             `declare gate` is an opaque backend gate, not a forward declaration"
                        ),
                    });
                }
                GateOrigin::CompositeDefined => {
                    return Err(SemanticError::Other {
                        message: format!("gate '{name}' is already defined"),
                    });
                }
            }
        }
        self.gate_registry.insert(
            name.to_string(),
            GateSignature {
                name: name.to_string(),
                num_params,
                num_qubits,
                origin,
            },
        );
        Ok(())
    }

    /// Collect top-level declarations (forward declaration pass).
    fn collect_top_level(&mut self, decl: &TopLevelDecl) -> SemanticResult<()> {
        match decl {
            TopLevelDecl::Fn(fn_decl) => {
                let params: Vec<(String, Type)> = fn_decl
                    .params
                    .iter()
                    .map(|p| (p.name.clone(), self.resolve_type(&p.ty)))
                    .collect();
                let return_type = fn_decl
                    .return_type
                    .as_ref()
                    .map(|t| self.resolve_type(t))
                    .unwrap_or(Type::Unit);

                // Collect indices of comptime parameters (for generic instantiation)
                let comptime_param_indices: Vec<usize> = fn_decl
                    .params
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| p.is_comptime)
                    .map(|(i, _)| i)
                    .collect();

                // Store original declaration if function has comptime params (is generic)
                let original_decl = if !comptime_param_indices.is_empty() {
                    Some(Box::new(fn_decl.clone()))
                } else {
                    None
                };

                self.symbols.define(Symbol {
                    name: fn_decl.name.clone(),
                    kind: SymbolKind::Function {
                        params,
                        return_type,
                        is_pub: fn_decl.is_pub,
                        comptime_param_indices,
                        original_decl,
                    },
                    location: fn_decl.location.clone(),
                })?;

                // Track user-defined functions for call graph analysis
                self.user_functions.insert(fn_decl.name.clone());
            }
            TopLevelDecl::Struct(struct_decl) => {
                let fields: Vec<(String, Type)> = struct_decl
                    .fields
                    .iter()
                    .map(|f| (f.name.clone(), self.resolve_type(&f.ty)))
                    .collect();

                self.symbols.define(Symbol {
                    name: struct_decl.name.clone(),
                    kind: SymbolKind::TypeDef {
                        ty: Type::Struct {
                            name: struct_decl.name.clone(),
                            fields,
                        },
                    },
                    location: struct_decl.location.clone(),
                })?;
            }
            TopLevelDecl::Enum(enum_decl) => {
                let variants: Vec<String> =
                    enum_decl.variants.iter().map(|v| v.name.clone()).collect();

                self.symbols.define(Symbol {
                    name: enum_decl.name.clone(),
                    kind: SymbolKind::TypeDef {
                        ty: Type::Enum {
                            name: enum_decl.name.clone(),
                            variants,
                        },
                    },
                    location: enum_decl.location.clone(),
                })?;
            }
            TopLevelDecl::Binding(binding) => {
                // Binding declarations are collected but not fully analyzed yet
                // For struct/enum type bindings, also register as TypeDef
                let ty = if let Some(type_expr) = &binding.ty {
                    self.resolve_type(type_expr)
                } else if let Some(value) = &binding.value {
                    // Try to infer type from value (for struct { } bindings)
                    match self.analyze_expr(value) {
                        Ok(ty) => ty,
                        Err(e) => {
                            self.errors.push(e);
                            Type::Unknown
                        }
                    }
                } else {
                    Type::Unknown // Will be inferred
                };

                // If binding a struct/enum type, register as TypeDef
                if let Type::Struct { fields, .. } = &ty {
                    self.symbols.define(Symbol {
                        name: binding.name.clone(),
                        kind: SymbolKind::TypeDef {
                            ty: Type::Struct {
                                name: binding.name.clone(),
                                fields: fields.clone(),
                            },
                        },
                        location: binding.location.clone(),
                    })?;
                } else if let Type::Enum { variants, .. } = &ty {
                    self.symbols.define(Symbol {
                        name: binding.name.clone(),
                        kind: SymbolKind::TypeDef {
                            ty: Type::Enum {
                                name: binding.name.clone(),
                                variants: variants.clone(),
                            },
                        },
                        location: binding.location.clone(),
                    })?;
                } else {
                    self.symbols.define(Symbol {
                        name: binding.name.clone(),
                        kind: SymbolKind::Variable {
                            ty,
                            is_const: !binding.is_mutable,
                            is_comptime: !binding.is_mutable, // Top-level immutable bindings are comptime
                        },
                        location: binding.location.clone(),
                    })?;
                }
            }
            TopLevelDecl::Test(_) => {
                // Tests don't declare symbols
            }
            TopLevelDecl::DeclareGate(gate) => {
                // Register an opaque target gate (reject duplicates)
                self.register_user_gate(
                    &gate.name,
                    gate.params.len(),
                    gate.qubits.len(),
                    GateOrigin::TargetDeclared,
                )?;
            }
            TopLevelDecl::Gate(gate) => {
                // Register a composite gate definition (reject duplicates)
                self.register_user_gate(
                    &gate.name,
                    gate.params.len(),
                    gate.qubits.len(),
                    GateOrigin::CompositeDefined,
                )?;
            }
            TopLevelDecl::ErrorSet(error_set) => {
                // Error sets define a type containing the error values with optional associated data
                let errors: Vec<(String, Option<Box<Type>>)> = error_set
                    .variants
                    .iter()
                    .map(|v| {
                        let data_type = v
                            .data_type
                            .as_ref()
                            .map(|ty| Box::new(self.resolve_type(ty)));
                        (v.name.clone(), data_type)
                    })
                    .collect();

                self.symbols.define(Symbol {
                    name: error_set.name.clone(),
                    kind: SymbolKind::TypeDef {
                        ty: Type::ErrorSet {
                            name: error_set.name.clone(),
                            errors,
                        },
                    },
                    location: error_set.location.clone(),
                })?;
            }
            TopLevelDecl::FaultSet(fault_set) => {
                // Fault sets define a type containing the fault values with optional associated data
                let faults: Vec<(String, Option<Box<Type>>)> = fault_set
                    .variants
                    .iter()
                    .map(|v| {
                        let data_type = v
                            .data_type
                            .as_ref()
                            .map(|ty| Box::new(self.resolve_type(ty)));
                        (v.name.clone(), data_type)
                    })
                    .collect();

                self.symbols.define(Symbol {
                    name: fault_set.name.clone(),
                    kind: SymbolKind::TypeDef {
                        ty: Type::FaultSet {
                            name: fault_set.name.clone(),
                            faults,
                        },
                    },
                    location: fault_set.location.clone(),
                })?;
            }
            TopLevelDecl::Union(union_decl) => {
                // Union defines a tagged union type
                let fields: Vec<(String, Option<Type>)> = union_decl
                    .fields
                    .iter()
                    .map(|f| (f.name.clone(), f.ty.as_ref().map(|t| self.resolve_type(t))))
                    .collect();

                // tag: None = untagged, Some(None) = auto-tagged, Some(Some(_)) = external tag
                let is_tagged = union_decl.tag.is_some();

                self.symbols.define(Symbol {
                    name: union_decl.name.clone(),
                    kind: SymbolKind::TypeDef {
                        ty: Type::Union {
                            name: union_decl.name.clone(),
                            fields,
                            is_tagged,
                        },
                    },
                    location: union_decl.location.clone(),
                })?;
            }
            TopLevelDecl::ExternFn(extern_fn) => {
                // External functions are registered like regular functions
                let params: Vec<(String, Type)> = extern_fn
                    .params
                    .iter()
                    .map(|p| (p.name.clone(), self.resolve_type(&p.ty)))
                    .collect();
                let return_type = extern_fn
                    .return_type
                    .as_ref()
                    .map(|t| self.resolve_type(t))
                    .unwrap_or(Type::Unit);

                self.symbols.define(Symbol {
                    name: extern_fn.name.clone(),
                    kind: SymbolKind::Function {
                        params,
                        return_type,
                        is_pub: extern_fn.is_pub,
                        comptime_param_indices: vec![], // Extern functions don't support comptime params
                        original_decl: None,
                    },
                    location: extern_fn.location.clone(),
                })?;
            }
        }
        Ok(())
    }

    /// Analyze a top-level declaration (full analysis pass).
    fn analyze_top_level(&mut self, decl: &TopLevelDecl) -> SemanticResult<()> {
        match decl {
            TopLevelDecl::Fn(fn_decl) => self.analyze_fn(fn_decl),
            TopLevelDecl::Struct(struct_decl) => self.analyze_struct(struct_decl),
            TopLevelDecl::Binding(binding) => self.analyze_binding(binding),
            TopLevelDecl::Test(test_decl) => self.analyze_block(&test_decl.body),
            TopLevelDecl::Enum(_) => Ok(()), // Enums are fully analyzed in collect pass
            TopLevelDecl::Union(_) => Ok(()), // Unions are fully analyzed in collect pass
            TopLevelDecl::ErrorSet(_) => Ok(()), // Error sets are fully analyzed in collect pass
            TopLevelDecl::FaultSet(_) => Ok(()), // Fault sets are fully analyzed in collect pass
            TopLevelDecl::ExternFn(_) => Ok(()), // Extern functions are fully analyzed in collect pass
            TopLevelDecl::DeclareGate(_) => Ok(()), // Fully analyzed in collect pass
            TopLevelDecl::Gate(gate) => {
                // Analyze composite gate body in a scope with qubit/param bindings
                self.symbols.push_scope(ScopeKind::Function)?;
                for qp in &gate.qubits {
                    self.symbols.define(Symbol {
                        name: qp.name.clone(),
                        kind: SymbolKind::Variable {
                            ty: Type::Qubit,
                            is_const: false,
                            is_comptime: false,
                        },
                        location: qp.location.clone(),
                    })?;
                }
                for gp in &gate.params {
                    self.symbols.define(Symbol {
                        name: gp.name.clone(),
                        kind: SymbolKind::Variable {
                            ty: Type::A64, // Gate params are angles by default
                            is_const: false,
                            is_comptime: false,
                        },
                        location: gp.location.clone(),
                    })?;
                }
                let result = self.analyze_block(&gate.body);
                self.symbols.pop_scope();
                result
            }
        }
    }

    /// Analyze a function declaration.
    fn analyze_fn(&mut self, fn_decl: &FnDecl) -> SemanticResult<()> {
        self.symbols.push_scope(ScopeKind::Function)?;

        // Set current function context
        let prev_function = self.current_function.take();
        self.current_function = Some(fn_decl.name.clone());

        // Track function in call stack for recursion detection (always enforced)
        self.recursion_tracker
            .enter_function(&fn_decl.name, &fn_decl.location.clone().unwrap_or_default())?;

        // Define parameters
        for param in &fn_decl.params {
            let ty = self.resolve_type(&param.ty);
            self.symbols.define(Symbol {
                name: param.name.clone(),
                kind: SymbolKind::Parameter {
                    ty: ty.clone(),
                    is_comptime: param.is_comptime,
                },
                location: param.location.clone(),
            })?;
        }

        // Set return type for return statement checking
        let return_type = fn_decl.return_type.as_ref().map(|t| self.resolve_type(t));
        self.current_return_type = return_type.clone();

        // Analyze body
        self.analyze_block(&fn_decl.body)?;

        // Check that all functions have explicit returns on all paths
        // (NASA Power of 10: explicit control flow)
        // - Unit functions must have explicit `return unit;`
        // - Non-unit functions must have explicit `return expr;`
        // - Never functions are exempt (they never return normally)
        let is_never = matches!(&return_type, Some(Type::Never));
        if !is_never && !self.block_always_returns(&fn_decl.body) {
            return Err(SemanticError::MissingReturn {
                name: fn_decl.name.clone(),
                location: fn_decl.location.clone().unwrap_or_default(),
            });
        }

        // Exit recursion tracker (always enforced)
        self.recursion_tracker.exit_function(&fn_decl.name);

        // Restore previous function context
        self.current_return_type = None;
        self.current_function = prev_function;
        self.symbols.pop_scope();
        Ok(())
    }

    /// Analyze a struct declaration.
    fn analyze_struct(&mut self, struct_decl: &StructDecl) -> SemanticResult<()> {
        // Analyze default field values
        for field in &struct_decl.fields {
            if let Some(default) = &field.default {
                let field_ty = self.resolve_type(&field.ty);
                let expr_ty = self.analyze_expr(default)?;
                self.check_assignable(&field_ty, &expr_ty, field.location.clone())?;
            }
        }

        // Analyze methods
        for method in &struct_decl.methods {
            self.analyze_fn(method)?;
        }

        Ok(())
    }

    /// Check if a block always returns (all code paths have explicit return).
    /// This enforces explicit returns - trailing expressions don't count.
    fn block_always_returns(&self, block: &Block) -> bool {
        // Check each statement - if any always returns, the block returns
        for stmt in &block.statements {
            if self.stmt_always_returns(stmt) {
                return true;
            }
        }
        // Also check trailing expression - if it always returns, block returns
        // (e.g., `if (cond) { return 1; } else { return 2; }` as trailing expr)
        if let Some(trailing) = &block.trailing_expr
            && self.expr_always_returns(trailing)
        {
            return true;
        }
        false
    }

    /// Check if a statement always returns.
    fn stmt_always_returns(&self, stmt: &Stmt) -> bool {
        match stmt {
            Stmt::Return(_) => true,

            Stmt::If(if_stmt) => {
                // If with else - both branches must return
                if let Some(else_branch) = &if_stmt.else_body {
                    let then_returns = self.block_always_returns(&if_stmt.then_body);
                    let else_returns = self.else_branch_always_returns(else_branch);
                    then_returns && else_returns
                } else {
                    // If without else - doesn't guarantee return
                    false
                }
            }

            Stmt::Block(block) => self.block_always_returns(block),

            Stmt::Switch(switch_stmt) => {
                // All prongs must return, and there must be an else prong
                let has_else = switch_stmt.prongs.iter().any(|p| p.is_else);
                if !has_else {
                    return false;
                }
                switch_stmt
                    .prongs
                    .iter()
                    .all(|p| self.expr_always_returns(&p.body))
            }

            Stmt::For(for_stmt) => {
                // For loop body might not execute at all
                // Even if body returns, loop might have 0 iterations
                // But if body has unreachable return, we can't reach after loop
                // For simplicity, assume for loops don't guarantee return
                let _ = for_stmt;
                false
            }

            // Expression statements might contain if expressions that return
            Stmt::Expr(expr_stmt) => self.expr_always_returns(&expr_stmt.expr),

            // Other statements don't return
            Stmt::Binding(_)
            | Stmt::Alias(_)
            | Stmt::Assign(_)
            | Stmt::Tick(_)
            | Stmt::TryBlock(_)
            | Stmt::Break(_)
            | Stmt::Continue(_)
            | Stmt::Defer(_)
            | Stmt::Errdefer(_)
            | Stmt::Gate(_)
            | Stmt::Prepare(_)
            | Stmt::Measure(_)
            | Stmt::Barrier(_) => false,
        }
    }

    /// Check if an else branch always returns.
    fn else_branch_always_returns(&self, else_branch: &ElseBranch) -> bool {
        match else_branch {
            ElseBranch::ElseIf(nested_if) => {
                // Recursively check the nested if statement
                if let Some(else_body) = &nested_if.else_body {
                    let then_returns = self.block_always_returns(&nested_if.then_body);
                    let else_returns = self.else_branch_always_returns(else_body);
                    then_returns && else_returns
                } else {
                    // else if without else - doesn't guarantee return
                    false
                }
            }
            ElseBranch::Else(block) => self.block_always_returns(block),
        }
    }

    /// Check if an expression always returns.
    /// This handles block expressions that might contain return statements.
    fn expr_always_returns(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Block(block_expr) => {
                // Check statements in the block expression
                for stmt in &block_expr.statements {
                    if self.stmt_always_returns(stmt) {
                        return true;
                    }
                }
                // Trailing expression doesn't count as return
                false
            }
            Expr::If(if_expr) => {
                // If expression always has both branches (ternary form)
                // Both must return for the expression to always return
                self.expr_always_returns(&if_expr.then_expr)
                    && self.expr_always_returns(&if_expr.else_expr)
            }
            // Most expressions don't contain return statements
            _ => false,
        }
    }

    /// Analyze a binding declaration.
    fn analyze_binding(&mut self, binding: &Binding) -> SemanticResult<()> {
        if let Some(value) = &binding.value {
            let expr_ty = self.analyze_expr(value)?;

            if let Some(type_expr) = &binding.ty {
                let declared_ty = self.resolve_type(type_expr);
                self.check_assignable(&declared_ty, &expr_ty, binding.location.clone())?;
            }
        }

        Ok(())
    }

    /// Analyze a block.
    fn analyze_block(&mut self, block: &Block) -> SemanticResult<()> {
        self.symbols.push_scope(ScopeKind::Block)?;

        for stmt in &block.statements {
            self.analyze_stmt(stmt)?;
        }

        self.symbols.pop_scope();
        Ok(())
    }

    /// Analyze a statement.
    fn analyze_stmt(&mut self, stmt: &Stmt) -> SemanticResult<()> {
        match stmt {
            Stmt::Binding(binding) => {
                let ty = if let Some(value) = &binding.value {
                    let expr_ty = self.analyze_expr(value)?;

                    // Register allocator if this is a qalloc() call
                    // (gates don't require mut, only .child() does)
                    if let Some(capacity) = self.try_extract_allocator_capacity(value) {
                        self.qubit_states
                            .register_allocator(AllocatorInfo::new(&binding.name, capacity));
                    }
                    // Register child allocator if this is base.child(n)
                    else if let Some((parent, capacity)) = self.try_extract_child_allocator(value)
                    {
                        // Check that the parent allocator is mutable
                        if let Some(symbol) = self.symbols.lookup(&parent)
                            && let SymbolKind::Variable { is_const: true, .. } = &symbol.kind
                        {
                            return Err(SemanticError::ChildRequiresMutableParent {
                                name: parent,
                                location: binding.location.clone().unwrap_or_default(),
                            });
                        }
                        self.qubit_states.register_allocator(AllocatorInfo::child(
                            &binding.name,
                            parent,
                            capacity,
                        ));
                    }

                    // Try to evaluate immutable bindings at comptime
                    if !binding.is_mutable {
                        let mut evaluator = ComptimeEvaluator::new();
                        // Populate evaluator context with existing comptime values
                        for (name, comptime_val) in &self.comptime_values {
                            evaluator.context.define(name, comptime_val.clone());
                        }
                        if let Ok(comptime_val) = evaluator.eval_expr(value) {
                            self.comptime_values
                                .insert(binding.name.clone(), comptime_val);
                        }
                    }

                    if let Some(type_expr) = &binding.ty {
                        let declared_ty = self.resolve_type(type_expr);
                        self.check_assignable(&declared_ty, &expr_ty, binding.location.clone())?;
                        declared_ty
                    } else {
                        expr_ty
                    }
                } else if let Some(type_expr) = &binding.ty {
                    self.resolve_type(type_expr)
                } else {
                    return Err(SemanticError::CannotInferType {
                        name: binding.name.clone(),
                        location: binding.location.clone().unwrap_or_default(),
                    });
                };

                // If binding a struct/enum type, also register as TypeDef for type resolution
                // This allows: Syndrome := struct { x: u8 }; mz(pack Syndrome) [...]
                if let Type::Struct { fields, .. } = &ty {
                    self.symbols.define(Symbol {
                        name: binding.name.clone(),
                        kind: SymbolKind::TypeDef {
                            ty: Type::Struct {
                                name: binding.name.clone(),
                                fields: fields.clone(),
                            },
                        },
                        location: binding.location.clone(),
                    })?;
                } else if let Type::Enum { variants, .. } = &ty {
                    self.symbols.define(Symbol {
                        name: binding.name.clone(),
                        kind: SymbolKind::TypeDef {
                            ty: Type::Enum {
                                name: binding.name.clone(),
                                variants: variants.clone(),
                            },
                        },
                        location: binding.location.clone(),
                    })?;
                } else {
                    self.symbols.define(Symbol {
                        name: binding.name.clone(),
                        kind: SymbolKind::Variable {
                            ty,
                            is_const: !binding.is_mutable,
                            is_comptime: false,
                        },
                        location: binding.location.clone(),
                    })?;
                }
            }
            Stmt::Alias(alias) => {
                self.analyze_alias(alias)?;
            }
            Stmt::Assign(assign) => {
                // Check mutability before allowing assignment
                self.check_assignment_target_mutable(&assign.target, &assign.location)?;

                let target_ty = self.analyze_expr(&assign.target)?;
                let value_ty = self.analyze_expr(&assign.value)?;
                self.check_assignable(&target_ty, &value_ty, assign.location.clone())?;
            }
            Stmt::If(if_stmt) => {
                let cond_ty = self.analyze_expr(&if_stmt.condition)?;

                // Check if this is an optional unwrap pattern: if (opt) |value| { ... }
                if let Some(capture_name) = &if_stmt.capture {
                    // Condition must be an optional type
                    if let Type::Optional { inner } = &cond_ty {
                        // Create a new scope for the then block with the capture variable
                        self.symbols.push_scope(ScopeKind::Block)?;
                        self.symbols.define(Symbol {
                            name: capture_name.clone(),
                            kind: SymbolKind::Variable {
                                ty: *inner.clone(),
                                is_const: true,
                                is_comptime: false,
                            },
                            location: if_stmt.location.clone(),
                        })?;
                        self.analyze_block(&if_stmt.then_body)?;
                        self.symbols.pop_scope();
                    } else {
                        return Err(SemanticError::TypeMismatch {
                            expected: "optional type (?T)".to_string(),
                            found: cond_ty.display_name(),
                            location: if_stmt.location.clone().unwrap_or_default(),
                        });
                    }
                } else {
                    // Regular if statement - condition must be bool
                    self.check_assignable(&Type::Bool, &cond_ty, if_stmt.location.clone())?;
                    self.analyze_block(&if_stmt.then_body)?;
                }

                // Analyze else branch (no capture variable here)
                if let Some(else_branch) = &if_stmt.else_body {
                    match else_branch {
                        ast::ElseBranch::ElseIf(else_if) => {
                            self.analyze_stmt(&Stmt::If(*else_if.clone()))?;
                        }
                        ast::ElseBranch::Else(block) => {
                            self.analyze_block(block)?;
                        }
                    }
                }
            }
            Stmt::For(for_stmt) => {
                self.symbols.push_scope(ScopeKind::Loop)?;
                self.loop_depth += 1;

                // Inline for loops require comptime-evaluable ranges
                if for_stmt.is_inline {
                    self.inline_for_depth += 1;
                    // Validate that range bounds are comptime-evaluable
                    match &for_stmt.range {
                        ForRange::Range { start, end } => {
                            let mut evaluator = ComptimeEvaluator::new();
                            // Populate evaluator with known comptime values
                            for (name, val) in &self.comptime_values {
                                evaluator.context.define(name, val.clone());
                            }
                            if evaluator.eval_expr(start).is_err() {
                                return Err(SemanticError::InlineForRangeNotComptime {
                                    expr: format!("{:?}", start),
                                    location: for_stmt.location.clone().unwrap_or_default(),
                                });
                            }
                            if evaluator.eval_expr(end).is_err() {
                                return Err(SemanticError::InlineForRangeNotComptime {
                                    expr: format!("{:?}", end),
                                    location: for_stmt.location.clone().unwrap_or_default(),
                                });
                            }
                        }
                        ForRange::Collection(expr) => {
                            let mut evaluator = ComptimeEvaluator::new();
                            for (name, val) in &self.comptime_values {
                                evaluator.context.define(name, val.clone());
                            }
                            if evaluator.eval_expr(expr).is_err() {
                                return Err(SemanticError::InlineForRangeNotComptime {
                                    expr: format!("{:?}", expr),
                                    location: for_stmt.location.clone().unwrap_or_default(),
                                });
                            }
                        }
                    }
                }

                // Infer type from range expression
                let capture_type = self.infer_for_range_type(&for_stmt.range)?;

                // Define capture variables with inferred type
                for capture in &for_stmt.captures {
                    self.symbols.define(Symbol {
                        name: capture.clone(),
                        kind: SymbolKind::Variable {
                            ty: capture_type.clone(),
                            is_const: true,
                            is_comptime: for_stmt.is_inline,
                        },
                        location: for_stmt.location.clone(),
                    })?;
                }

                self.analyze_block(&for_stmt.body)?;

                if for_stmt.is_inline {
                    self.inline_for_depth -= 1;
                }
                self.loop_depth -= 1;
                self.symbols.pop_scope();
            }
            Stmt::Switch(switch_stmt) => {
                let _value_ty = self.analyze_expr(&switch_stmt.value)?;

                // Track seen case values for duplicate detection
                let mut seen_cases: std::collections::BTreeSet<String> =
                    std::collections::BTreeSet::new();

                for prong in &switch_stmt.prongs {
                    for case in &prong.cases {
                        self.analyze_expr(&case.value)?;

                        // Try to get a string representation for duplicate detection
                        // For literals and simple expressions, use their string form
                        let case_key = self.case_value_key(&case.value);
                        if let Some(key) = case_key
                            && !seen_cases.insert(key.clone())
                        {
                            return Err(SemanticError::DuplicateSwitchCase {
                                value: key,
                                location: case.location.clone().unwrap_or_default(),
                            });
                        }
                    }
                    self.analyze_expr(&prong.body)?;
                }
            }
            Stmt::Return(ret) => {
                if let Some(value) = &ret.value {
                    let value_ty = self.analyze_expr(value)?;
                    if let Some(expected) = &self.current_return_type {
                        self.check_assignable(expected, &value_ty, ret.location.clone())?;
                    }
                    // Check for escaping references to local variables
                    // This is always enforced (safe-by-constraint memory model)
                    self.check_no_local_escape(value, ret.location.clone().unwrap_or_default())?;
                } else {
                    // `return;` without a value is only allowed for unit functions
                    // (equivalent to `return unit;`)
                    if let Some(expected) = &self.current_return_type
                        && !matches!(expected, Type::Unit)
                    {
                        return Err(SemanticError::ReturnWithoutValue {
                            expected: expected.display_name(),
                            location: ret.location.clone().unwrap_or_default(),
                        });
                    }
                    // If no return type specified, unit is implied - return; is valid
                }
            }
            Stmt::Break(break_stmt) => {
                if self.loop_depth == 0 {
                    return Err(SemanticError::BreakContinueOutsideLoop {
                        keyword: "break".to_string(),
                        location: break_stmt.location.clone().unwrap_or_default(),
                    });
                }
                // break is not allowed in inline for loops
                if self.inline_for_depth > 0 {
                    return Err(SemanticError::BreakInInlineFor {
                        location: break_stmt.location.clone().unwrap_or_default(),
                    });
                }
                // Analyze break value if present
                if let Some(value) = &break_stmt.value {
                    self.analyze_expr(value)?;
                }
            }
            Stmt::Continue(continue_stmt) => {
                if self.loop_depth == 0 {
                    return Err(SemanticError::BreakContinueOutsideLoop {
                        keyword: "continue".to_string(),
                        location: continue_stmt.location.clone().unwrap_or_default(),
                    });
                }
                // continue is not allowed in inline for loops
                if self.inline_for_depth > 0 {
                    return Err(SemanticError::ContinueInInlineFor {
                        location: continue_stmt.location.clone().unwrap_or_default(),
                    });
                }
            }
            Stmt::Defer(defer) => {
                self.analyze_stmt(&defer.body)?;
            }
            Stmt::Errdefer(errdefer) => {
                // If there's a capture, add it to scope for the body
                if let Some(capture_name) = &errdefer.capture {
                    self.symbols.push_scope(ScopeKind::Block)?;
                    // The captured error has type anyerror (could be any error type)
                    self.symbols.define(Symbol {
                        name: capture_name.clone(),
                        kind: SymbolKind::Variable {
                            ty: Type::AnyError,
                            is_const: true,
                            is_comptime: false,
                        },
                        location: errdefer.location.clone(),
                    })?;
                    self.analyze_stmt(&errdefer.body)?;
                    self.symbols.pop_scope();
                } else {
                    self.analyze_stmt(&errdefer.body)?;
                }
            }
            Stmt::Block(block) => {
                self.analyze_block(block)?;
            }
            Stmt::Expr(expr_stmt) => {
                self.analyze_expr(&expr_stmt.expr)?;
            }
            Stmt::Gate(gate_op) => {
                // Validate gate arity
                if gate_op.targets.len() != gate_op.kind.arity() {
                    return Err(SemanticError::GateArityMismatch {
                        gate: format!("{:?}", gate_op.kind),
                        expected: gate_op.kind.arity(),
                        found: gate_op.targets.len(),
                        location: gate_op.location.clone().unwrap_or_default(),
                    });
                }

                // Validate targets are valid qubit references and check state
                for target in &gate_op.targets {
                    self.validate_qubit_ref(target)?;

                    // In strict mode, verify qubits are prepared
                    if self.strict_mode {
                        // Try to extract constant index for state tracking
                        if let Some(index) = self.try_extract_constant_usize(&target.index) {
                            let location = gate_op.location.clone().unwrap_or_default();
                            self.qubit_states.validate_for_gate(
                                &target.allocator,
                                index,
                                &location,
                            )?;
                        }
                        // If index is not constant, we can't track state at compile time
                        // (runtime checking would be needed)
                    }
                }

                // Validate parameters for parameterized gates
                for param in &gate_op.params {
                    let param_ty = self.analyze_expr(param)?;
                    if !param_ty.is_float() && param_ty != Type::Unknown {
                        self.errors.push(SemanticError::TypeMismatch {
                            expected: "float".to_string(),
                            found: param_ty.display_name(),
                            location: gate_op.location.clone().unwrap_or_default(),
                        });
                    }
                }
            }
            Stmt::Prepare(prepare_op) => {
                // Validate allocator exists
                if self.symbols.lookup(&prepare_op.allocator).is_none() {
                    return Err(SemanticError::AllocatorNotFound {
                        name: prepare_op.allocator.clone(),
                        location: prepare_op.location.clone().unwrap_or_default(),
                    });
                }

                // Track state transitions
                if let Some(alloc) = self.qubit_states.get_allocator_mut(&prepare_op.allocator) {
                    if let Some(slots) = &prepare_op.slots {
                        // Prepare specific slots
                        for &slot in slots {
                            let slot_usize = slot as usize;
                            if self.strict_mode {
                                if let Err((idx, _state)) = alloc.prepare_slot(slot_usize) {
                                    // Slot out of bounds or already prepared
                                    if !alloc.is_in_bounds(idx) {
                                        return Err(SemanticError::QubitIndexOutOfBounds {
                                            allocator: prepare_op.allocator.clone(),
                                            index: idx,
                                            capacity: alloc.capacity.unwrap_or(0),
                                            location: prepare_op
                                                .location
                                                .clone()
                                                .unwrap_or_default(),
                                        });
                                    }
                                }
                            } else {
                                let _ = alloc.prepare_slot(slot_usize);
                            }
                        }
                    } else {
                        // Prepare all slots
                        alloc.prepare_all();
                    }
                }
            }
            Stmt::Measure(measure_op) => {
                for target in &measure_op.targets {
                    self.validate_qubit_ref(target)?;

                    // Try to extract constant index for state tracking
                    if let Some(index) = self.try_extract_constant_usize(&target.index) {
                        // In strict mode, verify qubits are prepared before measurement
                        if self.strict_mode {
                            let location = target.location.clone().unwrap_or_default();
                            self.qubit_states.validate_for_gate(
                                &target.allocator,
                                index,
                                &location,
                            )?;
                        }

                        // Transition to unprepared after measurement
                        if let Some(alloc) = self.qubit_states.get_allocator_mut(&target.allocator)
                        {
                            alloc.measure_slot(index);
                        }
                    }
                    // If index is not constant, we can't track state at compile time
                }
            }
            Stmt::Barrier(_) => {
                // Barriers are always valid
            }
            Stmt::Tick(tick_stmt) => {
                // Tick blocks represent parallel gate layers - no nesting allowed
                if self.tick_depth > 0 {
                    return Err(SemanticError::NestedTick {
                        location: tick_stmt.location.clone().unwrap_or_default(),
                    });
                }

                // In strict mode, validate that no qubit is used twice within a tick
                if self.strict_mode {
                    self.check_duplicate_qubits_in_tick(&tick_stmt.body, &tick_stmt.location)?;
                }

                // Track tick depth and analyze statements
                self.tick_depth += 1;
                for stmt in &tick_stmt.body {
                    self.analyze_stmt(stmt)?;
                }
                self.tick_depth -= 1;
            }
            Stmt::TryBlock(try_block) => {
                // Analyze statements within the try block body
                for stmt in &try_block.body.statements {
                    self.analyze_stmt(stmt)?;
                }
                // Analyze trailing expression if present
                if let Some(trailing) = &try_block.body.trailing_expr {
                    self.analyze_expr(trailing)?;
                }
                // Analyze catch clause if present
                if let Some(catch_clause) = &try_block.catch_clause {
                    // The catch variable is in scope for the catch body
                    // For now, just analyze the body expression
                    self.analyze_expr(&catch_clause.body)?;
                }
            }
        }
        Ok(())
    }

    /// Analyze an expression and return its type.
    fn analyze_expr(&mut self, expr: &Expr) -> SemanticResult<Type> {
        match expr {
            Expr::IntLit(lit) => {
                // Use suffix type if present, otherwise default to i64
                if let Some(suffix) = &lit.suffix {
                    Ok(int_suffix_to_type(suffix))
                } else {
                    Ok(Type::IInt {
                        bits: BitWidth::BITS_64,
                    })
                }
            }
            Expr::FloatLit(lit) => {
                // Use suffix type if present, otherwise default to f64
                if let Some(suffix) = &lit.suffix {
                    Ok(float_suffix_to_type(suffix))
                } else {
                    Ok(Type::F64)
                }
            }
            Expr::AngleLit(angle) => {
                // Analyze the inner value expression
                let inner_type = self.analyze_expr(&angle.value)?;
                // Inner must be numeric
                if !matches!(
                    inner_type,
                    Type::IInt { .. }
                        | Type::UInt { .. }
                        | Type::F32
                        | Type::F64
                        | Type::F16
                        | Type::F128
                ) {
                    return Err(SemanticError::TypeMismatch {
                        expected: "numeric".to_string(),
                        found: format!("{:?}", inner_type),
                        location: angle.location.clone().unwrap_or_default(),
                    });
                }
                // Angle literals have type a64
                Ok(Type::A64)
            }
            Expr::TypeAscription(asc) => {
                // Analyze the inner value expression
                let _inner_type = self.analyze_expr(&asc.value)?;
                // Parse the type name and return that type
                match asc.type_name.as_str() {
                    "f16" => Ok(Type::F16),
                    "f32" => Ok(Type::F32),
                    "f64" => Ok(Type::F64),
                    "f128" => Ok(Type::F128),
                    "a64" => Ok(Type::A64),
                    "u8" => Ok(Type::UInt { bits: BitWidth::BITS_8 }),
                    "u16" => Ok(Type::UInt { bits: BitWidth::BITS_16 }),
                    "u32" => Ok(Type::UInt { bits: BitWidth::BITS_32 }),
                    "u64" => Ok(Type::UInt { bits: BitWidth::BITS_64 }),
                    "u128" => Ok(Type::UInt { bits: BitWidth::BITS_128 }),
                    "usize" => Ok(Type::Usize),
                    "i8" => Ok(Type::IInt { bits: BitWidth::BITS_8 }),
                    "i16" => Ok(Type::IInt { bits: BitWidth::BITS_16 }),
                    "i32" => Ok(Type::IInt { bits: BitWidth::BITS_32 }),
                    "i64" => Ok(Type::IInt { bits: BitWidth::BITS_64 }),
                    "i128" => Ok(Type::IInt { bits: BitWidth::BITS_128 }),
                    "isize" => Ok(Type::Isize),
                    _ => Err(SemanticError::TypeMismatch {
                        expected: "valid numeric type suffix (u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, a64, etc.)".to_string(),
                        found: asc.type_name.clone(),
                        location: asc.location.clone().unwrap_or_default(),
                    }),
                }
            }
            Expr::BoolLit(_) => Ok(Type::Bool),
            Expr::StringLit(_) => Ok(Type::Slice {
                element: Box::new(Type::UInt {
                    bits: BitWidth::BITS_8,
                }),
            }),
            Expr::FString(fstr) => {
                // Analyze all interpolated expressions for errors
                for part in &fstr.parts {
                    if let FStringPart::Expr { expr, format: _ } = part {
                        self.analyze_expr(expr)?;
                    }
                }
                // F-strings produce string slices
                Ok(Type::Slice {
                    element: Box::new(Type::UInt {
                        bits: BitWidth::BITS_8,
                    }),
                })
            }
            Expr::CharLit(_) => Ok(Type::UInt {
                bits: BitWidth::BITS_8,
            }),
            Expr::Null(_) => Ok(Type::Optional {
                inner: Box::new(Type::Unknown),
            }),
            Expr::Undefined(_) => Ok(Type::Unknown),
            Expr::Unit(_) => Ok(Type::Unit),

            Expr::Ident(ident) => {
                if let Some(symbol) = self.symbols.lookup(&ident.name) {
                    match &symbol.kind {
                        SymbolKind::Variable { ty, .. } => Ok(ty.clone()),
                        SymbolKind::Parameter { ty, .. } => Ok(ty.clone()),
                        SymbolKind::Function {
                            params,
                            return_type,
                            ..
                        } => Ok(Type::Function {
                            params: params.iter().map(|(_, t)| t.clone()).collect(),
                            return_type: Box::new(return_type.clone()),
                        }),
                        SymbolKind::TypeDef { ty } => Ok(ty.clone()),
                        SymbolKind::Allocator { capacity } => Ok(Type::Allocator {
                            capacity: *capacity,
                        }),
                    }
                } else {
                    // Check if it's a built-in constant
                    if is_builtin_constant(&ident.name) {
                        Ok(get_builtin_constant_type(&ident.name))
                    // Check if it's a built-in gate name
                    } else if is_gate_name(&ident.name) {
                        Ok(Type::Function {
                            params: vec![Type::Qubit],
                            return_type: Box::new(Type::Unit),
                        })
                    // Check if it's a built-in type name (for comptime type values)
                    } else if resolve_builtin_type_name(&ident.name).is_some() {
                        Ok(Type::Type) // The expression evaluates to a type value
                    } else {
                        Err(SemanticError::UndefinedSymbol {
                            name: ident.name.clone(),
                            location: ident.location.clone().unwrap_or_default(),
                        })
                    }
                }
            }

            Expr::Binary(binary) => {
                let left_ty = self.analyze_expr(&binary.left)?;
                let right_ty = self.analyze_expr(&binary.right)?;
                self.check_binary_op(binary.op, &left_ty, &right_ty, binary.location.clone())
            }

            Expr::Unary(unary) => {
                let operand_ty = self.analyze_expr(&unary.operand)?;
                self.check_unary_op(unary.op, &operand_ty, unary.location.clone())
            }

            Expr::Call(call) => {
                let callee_ty = self.analyze_expr(&call.callee)?;

                // Reject gate names used with call syntax - must use gate expression syntax
                if let Expr::Ident(ident) = &call.callee {
                    if ident.name == "mz" {
                        // Reject old syntax: mz(type, target) - should use mz(T) target
                        return Err(SemanticError::DeprecatedMeasurementSyntax {
                            location: call.location.clone().unwrap_or_default(),
                        });
                    }
                    if let Some(info) = get_gate_info(&ident.name) {
                        // Gate names cannot be called with function syntax
                        // Generate helpful hint based on gate type
                        let hint = if info.parameterized {
                            match info.arity {
                                1 => "(<angle>) q[i]".to_string(),
                                2 => "(<angle>) (q[i], q[j])".to_string(),
                                _ => "(<angle>) (q[...])".to_string(),
                            }
                        } else {
                            match info.arity {
                                1 => "q[i]".to_string(),
                                2 => "(q[i], q[j])".to_string(),
                                3 => "(q[i], q[j], q[k])".to_string(),
                                _ => "(q[...])".to_string(),
                            }
                        };
                        return Err(SemanticError::InvalidGateSyntax {
                            gate: ident.name.clone(),
                            hint,
                            location: call.location.clone().unwrap_or_default(),
                        });
                    }

                    // Recursion is never allowed (NASA Power of 10 compliance)
                    // Use FFI with Rust if complex recursive algorithms are needed
                    // Check for both direct and mutual recursion using the call stack
                    if self.recursion_tracker.is_in_call_stack(&ident.name) {
                        return Err(SemanticError::RecursionDetected {
                            name: ident.name.clone(),
                            location: call.location.clone().unwrap_or_default(),
                        });
                    }

                    // Record call in call graph for mutual recursion detection
                    if let Some(ref caller) = self.current_function
                        && self.user_functions.contains(&ident.name)
                    {
                        self.call_graph
                            .entry(caller.clone())
                            .or_default()
                            .insert(ident.name.clone());
                    }

                    // Check if this is a generic function that needs instantiation
                    // First, extract all needed data from the immutable borrow
                    let generic_info = self.symbols.lookup(&ident.name).and_then(|symbol| {
                        if let SymbolKind::Function {
                            comptime_param_indices,
                            original_decl,
                            return_type: fn_return_type,
                            ..
                        } = &symbol.kind
                            && !comptime_param_indices.is_empty()
                            && let Some(original) = original_decl
                        {
                            return Some((
                                comptime_param_indices.clone(),
                                *original.clone(),
                                fn_return_type.clone(),
                            ));
                        }
                        None
                    });

                    // Now we can use the extracted data with a mutable borrow
                    if let Some((comptime_param_indices, original_decl, fn_return_type)) =
                        generic_info
                    {
                        // Evaluate comptime arguments
                        let mut comptime_args = Vec::new();
                        for &idx in &comptime_param_indices {
                            if idx < call.args.len() {
                                // Try to evaluate the argument at comptime
                                match self.comptime.eval_expr(&call.args[idx]) {
                                    Ok(val) => comptime_args.push(val),
                                    Err(e) => {
                                        return Err(SemanticError::ComptimeError {
                                            message: format!(
                                                "comptime argument {} must be evaluable at compile time: {}",
                                                idx, e
                                            ),
                                            location: call.args[idx]
                                                .get_location()
                                                .unwrap_or_default(),
                                        });
                                    }
                                }
                            }
                        }

                        // Instantiate the generic function
                        let _mangled_name = self.instantiate_generic_function(
                            &ident.name,
                            &comptime_args,
                            &comptime_param_indices,
                            &original_decl,
                        )?;

                        // For now, return the original return type
                        // (Full specialization would substitute types too)
                        return Ok(fn_return_type);
                    }
                }

                match callee_ty {
                    Type::Function {
                        params,
                        return_type,
                    } => {
                        // Validate argument count
                        if call.args.len() != params.len() {
                            return Err(SemanticError::ArgumentCountMismatch {
                                expected: params.len(),
                                found: call.args.len(),
                                location: call.location.clone().unwrap_or_default(),
                            });
                        }

                        // Validate argument types
                        for (arg, param_ty) in call.args.iter().zip(params.iter()) {
                            let arg_ty = self.analyze_expr(arg)?;
                            if !self.types_compatible(&arg_ty, param_ty) {
                                return Err(SemanticError::TypeMismatch {
                                    expected: param_ty.display_name(),
                                    found: arg_ty.display_name(),
                                    location: arg.get_location().unwrap_or_else(|| {
                                        call.location.clone().unwrap_or_default()
                                    }),
                                });
                            }
                        }

                        Ok(*return_type)
                    }
                    _ => Err(SemanticError::NotCallable {
                        location: call.location.clone().unwrap_or_default(),
                    }),
                }
            }

            Expr::BatchApply(batch) => {
                // Batch apply: h { q[0], q[1] } or rz(pi/4) { q[0], q[1] }
                // Analyze the operation and targets
                self.analyze_expr(&batch.operation)?;

                // Extract gate name to check arity
                let gate_name = match &batch.operation {
                    Expr::Ident(ident) => Some(ident.name.clone()),
                    Expr::Call(call) => {
                        if let Expr::Ident(ident) = &call.callee {
                            Some(ident.name.clone())
                        } else {
                            None
                        }
                    }
                    _ => None,
                };

                // Validate target arity if we know the gate
                if let Some(ref name) = gate_name
                    && let Some(info) = get_gate_info(name)
                {
                    for target in &batch.targets {
                        let target_arity = self.count_target_elements(target);
                        if target_arity != info.arity {
                            return Err(SemanticError::GateArityMismatch {
                                gate: name.clone(),
                                expected: info.arity,
                                found: target_arity,
                                location: batch.location.clone().unwrap_or_default(),
                            });
                        }
                    }
                }

                for target in &batch.targets {
                    self.analyze_expr(target)?;
                }
                // Batch gate operations return unit
                Ok(Type::Unit)
            }

            Expr::Measure(measure) => {
                // mz(T) target - per-qubit mode, count must match
                // mz(pack T) target - pack mode, type must have enough bits
                let location = measure.location.clone().unwrap_or_default();
                let result_type = self.resolve_type(&measure.result_type);

                // Count targets
                let target_count = self.count_measurement_targets(&measure.targets)?;

                if measure.pack {
                    // Pack mode: bits fill the type, must have enough capacity
                    let bit_capacity = self.type_bit_size(&result_type);

                    match bit_capacity {
                        Some(capacity) => {
                            if capacity < target_count {
                                return Err(SemanticError::MeasurementPackCapacity {
                                    ty: result_type.display_name(),
                                    capacity,
                                    qubits: target_count,
                                    location,
                                });
                            }
                        }
                        None => {
                            // Pack mode requires compile-time verifiable bit size
                            return Err(SemanticError::MeasurementPackUnknownSize {
                                ty: result_type.display_name(),
                                location,
                            });
                        }
                    }

                    self.analyze_expr(&measure.targets)?;
                    self.validate_gate_target_bounds(&measure.targets)?;
                    Ok(result_type)
                } else {
                    // Per-qubit mode: count must match exactly
                    match &result_type {
                        // Scalar type: must have exactly 1 target
                        Type::UInt { .. } => {
                            if target_count != 1 {
                                return Err(SemanticError::MeasurementArrayExpected { location });
                            }
                            self.analyze_expr(&measure.targets)?;
                            self.validate_gate_target_bounds(&measure.targets)?;
                            Ok(result_type)
                        }
                        // Array type: size must be explicit and match target count
                        Type::Array { element, size } => {
                            // Validate element type is a valid measurement type (unsigned integer)
                            let is_valid_element = matches!(**element, Type::UInt { .. });
                            if !is_valid_element {
                                return Err(SemanticError::InvalidMeasurementType {
                                    ty: element.display_name(),
                                    location,
                                });
                            }

                            // Size must be explicit (no [_]T inference for measurements)
                            let declared_size = match size {
                                Some(s) => *s,
                                None => {
                                    return Err(SemanticError::InvalidMeasurementType {
                                        ty: format!(
                                            "[_]{} - use explicit size like [{}]{}",
                                            element.display_name(),
                                            target_count,
                                            element.display_name()
                                        ),
                                        location,
                                    });
                                }
                            };

                            // Check size matches target count
                            if declared_size as usize != target_count {
                                return Err(SemanticError::MeasurementSizeMismatch {
                                    declared: declared_size.to_string(),
                                    element: element.display_name(),
                                    actual: target_count,
                                    location,
                                });
                            }

                            self.analyze_expr(&measure.targets)?;
                            self.validate_gate_target_bounds(&measure.targets)?;
                            Ok(Type::Array {
                                element: element.clone(),
                                size: Some(declared_size),
                            })
                        }
                        // Struct type in per-qubit mode: each qubit produces one struct
                        Type::Struct { .. } => {
                            if target_count != 1 {
                                return Err(SemanticError::MeasurementArrayExpected { location });
                            }
                            self.analyze_expr(&measure.targets)?;
                            self.validate_gate_target_bounds(&measure.targets)?;
                            Ok(result_type)
                        }
                        _ => Err(SemanticError::InvalidMeasurementType {
                            ty: result_type.display_name(),
                            location,
                        }),
                    }
                }
            }

            Expr::Gate(gate) => {
                // gate target or gate(params) target
                // Validate parameters for parameterized gates
                let gate_kind = gate.kind;
                if gate_kind.is_parameterized() {
                    if gate.params.is_empty() {
                        return Err(SemanticError::GateArityMismatch {
                            gate: format!("{:?}", gate_kind),
                            expected: 1,
                            found: 0,
                            location: gate.location.clone().unwrap_or_default(),
                        });
                    }
                    // Analyze parameter expressions
                    for param in &gate.params {
                        self.analyze_expr(param)?;
                    }
                }

                // For multi-qubit gates (arity > 1), reject bare allocator targets
                // e.g., `cx q` is ambiguous - use `cx (q[0], q[1])` or `cx {(q[0], q[1]), ...}`
                if gate_kind.arity() > 1
                    && let Expr::Ident(ident) = &gate.target
                {
                    // Check if this is an allocator
                    if let Some(symbol) = self.symbols.lookup(&ident.name) {
                        let is_allocator = match &symbol.kind {
                            SymbolKind::Variable { ty, .. } => matches!(ty, Type::Allocator { .. }),
                            SymbolKind::Allocator { .. } => true,
                            _ => false,
                        };
                        if is_allocator {
                            return Err(SemanticError::AmbiguousGateTarget {
                                gate: format!("{:?}", gate_kind).to_lowercase(),
                                location: gate.location.clone().unwrap_or_default(),
                            });
                        }
                    }
                }

                // Check gate target arity
                let expected_arity = gate_kind.arity();
                match &gate.target {
                    // Set literal: each element must have correct arity
                    Expr::Set(set) => {
                        for element in &set.elements {
                            let found_arity = self.count_target_elements(element);
                            if found_arity != expected_arity {
                                return Err(SemanticError::GateArityMismatch {
                                    gate: format!("{:?}", gate_kind).to_lowercase(),
                                    expected: expected_arity,
                                    found: found_arity,
                                    location: gate.location.clone().unwrap_or_default(),
                                });
                            }
                        }
                    }
                    // Single target (qubit ref or tuple)
                    target => {
                        let found_arity = self.count_target_elements(target);
                        if found_arity != expected_arity {
                            return Err(SemanticError::GateArityMismatch {
                                gate: format!("{:?}", gate_kind).to_lowercase(),
                                expected: expected_arity,
                                found: found_arity,
                                location: gate.location.clone().unwrap_or_default(),
                            });
                        }
                    }
                }

                // Analyze target expression
                self.analyze_expr(&gate.target)?;

                // Validate qubit bounds for gate targets
                self.validate_gate_target_bounds(&gate.target)?;

                // Handle prepare gates specially (PZ resets qubits to |0⟩)
                if gate_kind.is_prepare() {
                    // Prepare gates can be applied to unprepared qubits
                    // and transition them to the Prepared state
                    self.prepare_gate_targets(&gate.target);
                } else if self.strict_mode {
                    // In strict mode, verify qubits are prepared before non-prepare gates
                    self.validate_gate_target_states(
                        &gate.target,
                        &gate.location.clone().unwrap_or_default(),
                    )?;
                }

                // Gate operations are statements, return unit
                Ok(Type::Unit)
            }

            Expr::Field(field) => {
                let object_ty = self.analyze_expr(&field.object)?;
                match object_ty {
                    Type::Struct { fields, .. } => {
                        if let Some((_, ty)) = fields.iter().find(|(n, _)| n == &field.field) {
                            Ok(ty.clone())
                        } else {
                            Err(SemanticError::UndefinedSymbol {
                                name: field.field.clone(),
                                location: field.location.clone().unwrap_or_default(),
                            })
                        }
                    }
                    Type::Array { element, .. } => {
                        // Array properties
                        match field.field.as_str() {
                            "len" => Ok(Type::Usize), // Compile-time known length
                            "ptr" => Ok(Type::Pointer {
                                pointee: element,
                                is_const: false,
                                is_many: true,
                            }),
                            _ => Err(SemanticError::UndefinedSymbol {
                                name: field.field.clone(),
                                location: field.location.clone().unwrap_or_default(),
                            }),
                        }
                    }
                    Type::Slice { element } => {
                        // Slice properties
                        match field.field.as_str() {
                            "len" => Ok(Type::Usize), // Dynamic length
                            "ptr" => Ok(Type::Pointer {
                                pointee: element,
                                is_const: false,
                                is_many: true,
                            }),
                            _ => Err(SemanticError::UndefinedSymbol {
                                name: field.field.clone(),
                                location: field.location.clone().unwrap_or_default(),
                            }),
                        }
                    }
                    Type::Allocator { .. } => {
                        // Allocator methods
                        match field.field.as_str() {
                            "child" => Ok(Type::Function {
                                params: vec![Type::UInt {
                                    bits: BitWidth::BITS_32,
                                }],
                                return_type: Box::new(Type::Allocator { capacity: None }),
                            }),
                            "release" => Ok(Type::Function {
                                params: vec![],
                                return_type: Box::new(Type::Unit),
                            }),
                            // Deprecated: use `pz q` or `pz {q[0], q[1]}` instead
                            "prepare_all" | "prepare" => Err(SemanticError::DeprecatedSyntax {
                                old: format!(
                                    "{}.{}()",
                                    if let Expr::Ident(id) = &field.object {
                                        &id.name
                                    } else {
                                        "allocator"
                                    },
                                    field.field
                                ),
                                new: if field.field == "prepare_all" {
                                    "pz <allocator>".to_string()
                                } else {
                                    "pz {q[i], q[j], ...}".to_string()
                                },
                                location: field.location.clone().unwrap_or_default(),
                            }),
                            _ => Ok(Type::Unknown),
                        }
                    }
                    Type::Module { exports, .. } => {
                        // Module field access - look up exported symbol
                        if let Some((_, ty)) = exports.get(&field.field) {
                            Ok(ty.clone())
                        } else {
                            Err(SemanticError::UndefinedSymbol {
                                name: field.field.clone(),
                                location: field.location.clone().unwrap_or_default(),
                            })
                        }
                    }
                    _ => Ok(Type::Unknown),
                }
            }

            Expr::Index(index) => {
                let object_ty = self.analyze_expr(&index.object)?;
                let _index_ty = self.analyze_expr(&index.index)?;

                // Check if index is a range expression (slicing)
                let is_slice_op = matches!(&index.index, Expr::Range(_));

                match object_ty {
                    Type::Array { element, size } => {
                        if is_slice_op {
                            // arr[0..2] returns a slice
                            Ok(Type::Slice { element })
                        } else {
                            // Bounds check: if both size and index are known at compile time
                            if let Some(n) = size
                                && let Some(idx) = self.try_extract_constant_usize(&index.index)
                                && idx >= n as usize
                            {
                                return Err(SemanticError::ArrayIndexOutOfBounds {
                                    index: idx,
                                    size: n,
                                    location: index.location.clone().unwrap_or_default(),
                                });
                            }
                            // arr[0] returns an element
                            Ok(*element)
                        }
                    }
                    Type::Slice { element } => {
                        if is_slice_op {
                            // slice[0..2] returns a slice (re-slicing)
                            Ok(Type::Slice { element })
                        } else {
                            // slice[0] returns an element
                            Ok(*element)
                        }
                    }
                    Type::Allocator { .. } => Ok(Type::Qubit),
                    _ => Ok(Type::Unknown),
                }
            }

            Expr::If(if_expr) => {
                let cond_ty = self.analyze_expr(&if_expr.condition)?;
                self.check_assignable(&Type::Bool, &cond_ty, if_expr.location.clone())?;

                let then_ty = self.analyze_expr(&if_expr.then_expr)?;
                let else_ty = self.analyze_expr(&if_expr.else_expr)?;

                if then_ty == else_ty {
                    Ok(then_ty)
                } else {
                    Ok(Type::Unknown) // Could be improved with type unification
                }
            }

            Expr::Block(block) => {
                self.symbols.push_scope(ScopeKind::Block)?;
                for stmt in &block.statements {
                    self.analyze_stmt(stmt)?;
                }
                self.symbols.pop_scope();
                Ok(Type::Unknown) // Block expression type depends on break value
            }

            Expr::Comptime(comptime) => {
                // First, analyze the inner expression to get its type
                let inner_ty = self.analyze_expr(&comptime.inner)?;

                // Evaluate the expression at compile time
                match self.comptime.eval_expr(&comptime.inner) {
                    Ok(value) => {
                        // Store the evaluated value by location key
                        if let Some(loc) = &comptime.location {
                            let key = format!("{}:{}", loc.line, loc.column);
                            self.comptime_values.insert(key, value);
                        }
                        Ok(Type::Comptime(Box::new(inner_ty)))
                    }
                    Err(e) => {
                        // Comptime evaluation failed
                        let location = comptime
                            .location
                            .clone()
                            .unwrap_or_else(|| SourceLocation::new(0, 0));
                        Err(SemanticError::ComptimeError {
                            message: e.message,
                            location,
                        })
                    }
                }
            }

            Expr::Builtin(builtin) => {
                match builtin.name.as_str() {
                    "import" => self.analyze_import(builtin),
                    "This" => Ok(Type::Type),
                    "sizeOf" => Ok(Type::Usize),
                    "typeInfo" => Ok(Type::Type),
                    "typeName" => Ok(Type::Slice {
                        element: Box::new(Type::UInt {
                            bits: BitWidth::BITS_8,
                        }),
                    }),
                    "swap" => {
                        // @swap(&a, &b) - swap two values in place
                        // Requires exactly 2 pointer arguments of the same type
                        if builtin.args.len() != 2 {
                            return Err(SemanticError::ArgumentCountMismatch {
                                expected: 2,
                                found: builtin.args.len(),
                                location: builtin.location.clone().unwrap_or_default(),
                            });
                        }
                        let ty1 = self.analyze_expr(&builtin.args[0])?;
                        let ty2 = self.analyze_expr(&builtin.args[1])?;
                        // Both must be pointers to the same type
                        match (&ty1, &ty2) {
                            (
                                Type::Pointer { pointee: e1, .. },
                                Type::Pointer { pointee: e2, .. },
                            ) => {
                                if e1 != e2 {
                                    return Err(SemanticError::TypeMismatch {
                                        expected: format!("*{:?}", e1),
                                        found: format!("*{:?}", e2),
                                        location: builtin.location.clone().unwrap_or_default(),
                                    });
                                }
                            }
                            (Type::Pointer { .. }, _) => {
                                return Err(SemanticError::TypeMismatch {
                                    expected: "pointer".to_string(),
                                    found: format!("{:?}", ty2),
                                    location: builtin.location.clone().unwrap_or_default(),
                                });
                            }
                            (_, Type::Pointer { .. }) => {
                                return Err(SemanticError::TypeMismatch {
                                    expected: "pointer".to_string(),
                                    found: format!("{:?}", ty1),
                                    location: builtin.location.clone().unwrap_or_default(),
                                });
                            }
                            _ => {
                                return Err(SemanticError::TypeMismatch {
                                    expected: "pointer".to_string(),
                                    found: format!("{:?}", ty1),
                                    location: builtin.location.clone().unwrap_or_default(),
                                });
                            }
                        }
                        Ok(Type::Unit)
                    }
                    _ => Ok(Type::Unknown),
                }
            }

            Expr::AnonStruct(anon) => {
                // Anonymous struct type definition: struct { x: i32, y: i32 }
                // This creates a type, not a value
                let fields: Vec<(String, Type)> = anon
                    .fields
                    .iter()
                    .map(|f| (f.name.clone(), self.resolve_type(&f.ty)))
                    .collect();
                Ok(Type::Struct {
                    name: "<anonymous>".to_string(),
                    fields,
                })
            }

            Expr::StructInit(init) => {
                if let Some(ty) = &init.ty {
                    Ok(self.resolve_type(ty))
                } else {
                    // Anonymous struct initialization
                    let mut fields: Vec<(String, Type)> = Vec::with_capacity(init.fields.len());
                    for f in &init.fields {
                        let ty = match self.analyze_expr(&f.value) {
                            Ok(ty) => ty,
                            Err(e) => {
                                self.errors.push(e);
                                Type::Unknown
                            }
                        };
                        fields.push((f.name.clone(), ty));
                    }
                    Ok(Type::Struct {
                        name: "<anonymous>".to_string(),
                        fields,
                    })
                }
            }

            Expr::ArrayInit(init) => {
                let element_type = if let Some(elem) = init.elements.first() {
                    self.analyze_expr(elem)?
                } else {
                    Type::Unknown
                };
                Ok(Type::Array {
                    element: Box::new(element_type),
                    size: Some(init.elements.len() as u64),
                })
            }

            Expr::Range(_) => Ok(Type::Unknown), // Range type
            Expr::SlotRef(_) => Ok(Type::Qubit),
            Expr::BitRef(_) => Ok(Type::Bit),

            Expr::BracketArray(arr) => {
                // Bracket array [a, b, c] - infer element type from first element
                if arr.elements.is_empty() {
                    // In strict mode, empty arrays require explicit type annotation
                    if self.strict_mode {
                        return Err(SemanticError::EmptyArrayNeedsType {
                            location: arr.location.clone().unwrap_or_default(),
                        });
                    }
                    Ok(Type::Slice {
                        element: Box::new(Type::Unknown),
                    })
                } else {
                    let element_ty = self.analyze_expr(&arr.elements[0])?;
                    // Analyze all elements for side effects/validation
                    for elem in arr.elements.iter().skip(1) {
                        let _ = self.analyze_expr(elem)?;
                    }
                    Ok(Type::Array {
                        element: Box::new(element_ty),
                        size: Some(arr.elements.len() as u64),
                    })
                }
            }

            Expr::Tuple(tuple) => {
                // Tuple (a, b) - analyze each element and build tuple type
                let element_types: Result<Vec<Type>, SemanticError> = tuple
                    .elements
                    .iter()
                    .map(|elem| self.analyze_expr(elem))
                    .collect();
                Ok(Type::Tuple {
                    elements: element_types?,
                })
            }

            Expr::Set(set_expr) => {
                // Set literal {a, b, c} - infer element type from first element
                if set_expr.elements.is_empty() {
                    // Empty set - check if we have an explicit element type
                    if let Some(type_expr) = &set_expr.element_type {
                        let element_ty = self.resolve_type(type_expr);
                        Ok(Type::Set {
                            element: Box::new(element_ty),
                        })
                    } else {
                        // In strict mode, empty sets require explicit type annotation
                        if self.strict_mode {
                            return Err(SemanticError::EmptySetNeedsType {
                                location: set_expr.location.clone().unwrap_or_default(),
                            });
                        }
                        Ok(Type::Set {
                            element: Box::new(Type::Unknown),
                        })
                    }
                } else {
                    let element_ty = self.analyze_expr(&set_expr.elements[0])?;
                    // Analyze all elements for side effects/validation
                    for elem in set_expr.elements.iter().skip(1) {
                        let _ = self.analyze_expr(elem)?;
                    }
                    Ok(Type::Set {
                        element: Box::new(element_ty),
                    })
                }
            }

            Expr::ErrorValue(err) => {
                // Look up which error set contains this variant
                if let Some(error_type) = self.symbols.find_error_set_by_variant(&err.name) {
                    Ok(error_type)
                } else {
                    Err(SemanticError::UndefinedSymbol {
                        name: format!("error.{}", err.name),
                        location: err.location.clone().unwrap_or_default(),
                    })
                }
            }

            Expr::FaultValue(fault) => {
                // Look up which fault set contains this variant
                if let Some(fault_type) = self.symbols.find_fault_set_by_variant(&fault.name) {
                    Ok(fault_type)
                } else {
                    Err(SemanticError::UndefinedSymbol {
                        name: format!("fault.{}", fault.name),
                        location: fault.location.clone().unwrap_or_default(),
                    })
                }
            }

            Expr::Catch(catch) => {
                // catch expression: operand catch |err| handler
                // Type is the payload type of the error union operand
                let operand_ty = self.analyze_expr(&catch.operand)?;
                let _handler_ty = self.analyze_expr(&catch.handler)?;

                // If operand is an error union T!E, the result is T
                match operand_ty {
                    Type::ErrorUnion { payload, .. } => Ok(*payload),
                    Type::Unknown => Ok(Type::Unknown), // Allow Unknown for error recovery
                    _ => {
                        // Non-error-union with catch - this is an error
                        Err(SemanticError::CatchOnNonErrorType {
                            found: operand_ty.display_name(),
                            location: catch.location.clone().unwrap_or_default(),
                        })
                    }
                }
            }

            Expr::TryBlock(try_block) => {
                // Analyze the try block body
                for stmt in &try_block.body.statements {
                    self.analyze_stmt(stmt)?;
                }

                // Get the type of trailing expression (if any)
                let body_type = if let Some(trailing) = &try_block.body.trailing_expr {
                    self.analyze_expr(trailing)?
                } else {
                    Type::Unit
                };

                // Analyze catch clause if present
                let catch_type = if let Some(catch_clause) = &try_block.catch_clause {
                    Some(self.analyze_expr(&catch_clause.body)?)
                } else {
                    None
                };

                // Return type depends on mode and whether there's a catch clause
                use crate::ast::TryMode;
                match try_block.mode {
                    TryMode::Collect => {
                        // try {} (collect mode) -> []AnyError!T
                        // Collects all errors that occur during execution
                        if catch_type.is_some() {
                            // With catch, errors are handled - return array of results
                            Ok(Type::Slice {
                                element: Box::new(body_type),
                            })
                        } else {
                            // Without catch, return error union array
                            Ok(Type::Slice {
                                element: Box::new(Type::ErrorUnion {
                                    error: Box::new(Type::AnyError),
                                    payload: Box::new(body_type),
                                }),
                            })
                        }
                    }
                    TryMode::Propagate => {
                        // try! {} (propagate mode) -> E!T or T (if catch handles it)
                        if let Some(catch_ty) = catch_type {
                            // With catch, the catch provides the fallback value
                            // Type is union of body_type and catch_type
                            if body_type == catch_ty {
                                Ok(body_type)
                            } else {
                                // Types must be compatible
                                Ok(body_type)
                            }
                        } else {
                            // Without catch, return error union
                            Ok(Type::ErrorUnion {
                                error: Box::new(Type::AnyError),
                                payload: Box::new(body_type),
                            })
                        }
                    }
                }
            }

            Expr::FnLit(func) => {
                // Function literal - return function type
                // At comptime, these can return types (type constructors)
                let param_types: Vec<Type> = func
                    .params
                    .iter()
                    .map(|p| self.resolve_type(&p.ty))
                    .collect();
                let return_type = func
                    .return_type
                    .as_ref()
                    .map(|ty| self.resolve_type(ty))
                    .unwrap_or(Type::Unit);
                Ok(Type::Function {
                    params: param_types,
                    return_type: Box::new(return_type),
                })
            }

            Expr::Result(result) => {
                // Result expressions - emit tagged values to caller
                // Tag is compile-time string (already validated by parser)
                // Analyze the value expression
                self.analyze_expr(&result.value)?;

                // Result expressions evaluate to unit
                Ok(Type::Unit)
            }

            Expr::Channel(channel) => {
                // Channel expressions - >channel.command(args)
                // Analyze all argument expressions
                for arg in &channel.args {
                    self.analyze_expr(arg.value())?;
                }

                // Channel expressions evaluate to unit
                Ok(Type::Unit)
            }
        }
    }

    /// Resolve a type expression to a semantic type.
    fn resolve_type(&mut self, type_expr: &TypeExpr) -> Type {
        match type_expr {
            TypeExpr::Primitive(prim) => match prim {
                PrimitiveType::Bool => Type::Bool,
                PrimitiveType::UInt { bits } => Type::UInt {
                    bits: BitWidth::new(*bits).unwrap_or(BitWidth::BITS_64),
                },
                PrimitiveType::IInt { bits } => Type::IInt {
                    bits: BitWidth::new(*bits).unwrap_or(BitWidth::BITS_64),
                },
                PrimitiveType::Usize => Type::Usize,
                PrimitiveType::Isize => Type::Isize,
                PrimitiveType::F16 => Type::F16,
                PrimitiveType::F32 => Type::F32,
                PrimitiveType::F64 => Type::F64,
                PrimitiveType::F128 => Type::F128,
                PrimitiveType::A64 => Type::A64,
            },
            TypeExpr::Qubit => Type::Qubit,
            TypeExpr::Bit => Type::Bit,
            TypeExpr::QAlloc(_) => Type::Allocator { capacity: None },
            TypeExpr::Array(array) => {
                let element = self.resolve_type(&array.element);
                // Evaluate size expression at comptime if present
                if let Some(size_expr) = &array.size {
                    let mut evaluator = ComptimeEvaluator::new();
                    // Populate evaluator context with stored comptime values (for const propagation)
                    for (name, value) in &self.comptime_values {
                        evaluator.context.define(name, value.clone());
                    }
                    let size = if let Ok(value) = evaluator.eval_expr(size_expr) {
                        value.to_usize().map(|n| n as u64)
                    } else {
                        None
                    };
                    Type::Array {
                        element: Box::new(element),
                        size,
                    }
                } else {
                    // []T with no size is a slice type
                    Type::Slice {
                        element: Box::new(element),
                    }
                }
            }
            TypeExpr::Pointer(ptr) => {
                let pointee = self.resolve_type(&ptr.pointee);
                Type::Pointer {
                    pointee: Box::new(pointee),
                    is_const: ptr.is_const,
                    is_many: ptr.is_many,
                }
            }
            TypeExpr::Optional(inner) => Type::Optional {
                inner: Box::new(self.resolve_type(inner)),
            },
            TypeExpr::ErrorUnion(eu) => Type::ErrorUnion {
                error: Box::new(self.resolve_type(&eu.error_type)),
                payload: Box::new(self.resolve_type(&eu.payload_type)),
            },
            TypeExpr::CollectedErrors(ce) => Type::CollectedErrors {
                error: Box::new(self.resolve_type(&ce.error_type)),
                payload: Box::new(self.resolve_type(&ce.payload_type)),
            },
            TypeExpr::Tuple(elements) => {
                let resolved: Vec<Type> = elements.iter().map(|t| self.resolve_type(t)).collect();
                Type::Tuple { elements: resolved }
            }
            TypeExpr::Fn(fn_type) => {
                let params: Vec<Type> = fn_type
                    .params
                    .iter()
                    .map(|t| self.resolve_type(t))
                    .collect();
                let return_type = fn_type
                    .return_type
                    .as_ref()
                    .map(|t| self.resolve_type(t))
                    .unwrap_or(Type::Unit);
                Type::Function {
                    params,
                    return_type: Box::new(return_type),
                }
            }
            TypeExpr::Named(path) => {
                let name = path.segments.join(".");
                if let Some(symbol) = self.symbols.lookup(&name)
                    && let SymbolKind::TypeDef { ty } = &symbol.kind
                {
                    return ty.clone();
                }
                // Report error for undefined type name
                self.errors.push(SemanticError::UndefinedType {
                    name: name.clone(),
                    location: path.location.clone().unwrap_or_default(),
                });
                Type::Unknown
            }
            TypeExpr::Type => Type::Type,
            TypeExpr::AnyType => Type::Unknown,
            TypeExpr::Unit => Type::Unit,
            TypeExpr::Set(element_type) => Type::Set {
                element: Box::new(self.resolve_type(element_type)),
            },
            TypeExpr::Struct(s) => {
                // Anonymous struct type
                let fields = s
                    .fields
                    .iter()
                    .map(|f| (f.name.clone(), self.resolve_type(&f.ty)))
                    .collect();
                Type::Struct {
                    name: String::new(), // Anonymous
                    fields,
                }
            }
            TypeExpr::Enum(e) => {
                // Anonymous enum type
                let variants = e.variants.iter().map(|v| v.name.clone()).collect();
                Type::Enum {
                    name: String::new(), // Anonymous
                    variants,
                }
            }
        }
    }

    /// Check if an assignment target is mutable.
    /// Returns an error if trying to assign to an immutable variable.
    fn check_assignment_target_mutable(
        &self,
        target: &Expr,
        location: &Option<SourceLocation>,
    ) -> SemanticResult<()> {
        match target {
            // Direct variable assignment: x = value
            Expr::Ident(ident) => {
                if let Some(symbol) = self.symbols.lookup(&ident.name) {
                    match &symbol.kind {
                        SymbolKind::Variable { is_const: true, .. } => {
                            return Err(SemanticError::ImmutableAssignment {
                                name: ident.name.clone(),
                                location: location.clone().unwrap_or_default(),
                            });
                        }
                        SymbolKind::Parameter { .. } => {
                            // Parameters are always immutable
                            return Err(SemanticError::ImmutableAssignment {
                                name: ident.name.clone(),
                                location: location.clone().unwrap_or_default(),
                            });
                        }
                        _ => {}
                    }
                }
                Ok(())
            }
            // Field assignment: obj.field = value - check root object mutability
            Expr::Field(field) => self.check_assignment_target_mutable(&field.object, location),
            // Index assignment: arr[i] = value - check root object mutability
            Expr::Index(index) => self.check_assignment_target_mutable(&index.object, location),
            // Dereference assignment: *ptr = value - allowed if pointer is valid
            Expr::Unary(unary) if unary.op == ast::UnaryOp::Deref => Ok(()),
            // Other expressions (like function calls) can't be assigned to
            _ => Ok(()),
        }
    }

    /// Get a string key for a case value expression for duplicate detection.
    /// Returns Some(key) for literals and simple expressions that can be compared.
    fn case_value_key(&self, expr: &Expr) -> Option<String> {
        match expr {
            Expr::IntLit(lit) => Some(lit.value.to_string()),
            Expr::FloatLit(lit) => Some(lit.value.to_string()),
            Expr::BoolLit(lit) => Some(lit.value.to_string()),
            Expr::StringLit(lit) => Some(format!("\"{}\"", lit.value)),
            Expr::CharLit(lit) => Some(format!("'{}'", lit.value)),
            Expr::Ident(ident) => Some(ident.name.clone()),
            Expr::Field(fa) => {
                // For enum variants like Color.Red
                if let Expr::Ident(ident) = &fa.object {
                    Some(format!("{}.{}", ident.name, fa.field))
                } else {
                    None
                }
            }
            // For complex expressions, we can't easily detect duplicates
            _ => None,
        }
    }

    /// Check if a value type is assignable to a target type.
    fn check_assignable(
        &self,
        target: &Type,
        value: &Type,
        location: Option<SourceLocation>,
    ) -> SemanticResult<()> {
        // Same type is always ok
        if target == value {
            return Ok(());
        }

        // Unknown types are compatible with anything (for inference)
        if *target == Type::Unknown || *value == Type::Unknown {
            return Ok(());
        }

        // Allow null (?unknown) to be assigned to any optional type ?T
        if let (Type::Optional { .. }, Type::Optional { inner }) = (target, value)
            && **inner == Type::Unknown
        {
            // null (which is ?unknown) can be assigned to any ?T
            return Ok(());
        }

        // Allow numeric coercion between numeric types only
        // (but NOT from numeric to bool or vice versa)
        if target.is_numeric() && value.is_numeric() {
            return Ok(());
        }

        // Allow T to be assigned to T!E (returning success from error union function)
        if let Type::ErrorUnion { payload, .. } = target
            && self
                .check_assignable(payload.as_ref(), value, location.clone())
                .is_ok()
        {
            return Ok(());
        }

        // Allow error value to be assigned to T!E (returning error from error union function)
        if let Type::ErrorUnion { error, .. } = target {
            // Check if value is an error type that's compatible with the expected error type
            match value {
                // Same error set - always compatible
                Type::ErrorSet {
                    name: value_name,
                    errors: value_errors,
                } => {
                    if let Type::ErrorSet {
                        name: expected_name,
                        errors: expected_errors,
                    } = error.as_ref()
                    {
                        // Exact match
                        if value_name == expected_name {
                            return Ok(());
                        }
                        // Value's errors are a subset of expected's errors (union compatibility)
                        if value_errors.iter().all(|e| expected_errors.contains(e)) {
                            return Ok(());
                        }
                    }
                    // Also allow if expected is AnyError
                    if *error.as_ref() == Type::AnyError {
                        return Ok(());
                    }
                }
                // AnyError can be returned from any error union
                Type::AnyError => return Ok(()),
                _ => {}
            }
        }

        // Allow fault value to be assigned to T!F (returning fault from fault union function)
        if let Type::ErrorUnion { error, .. } = target
            && let Type::FaultSet {
                name: value_name,
                faults: value_faults,
            } = value
        {
            if let Type::FaultSet {
                name: expected_name,
                faults: expected_faults,
            } = error.as_ref()
            {
                // Exact match
                if value_name == expected_name {
                    return Ok(());
                }
                // Value's faults are a subset of expected's faults (union compatibility)
                if value_faults.iter().all(|f| expected_faults.contains(f)) {
                    return Ok(());
                }
            }
            // Also allow if expected is AnyFault
            if *error.as_ref() == Type::AnyFault {
                return Ok(());
            }
        }

        Err(SemanticError::TypeMismatch {
            expected: target.display_name(),
            found: value.display_name(),
            location: location.unwrap_or_default(),
        })
    }

    /// Check if two types are compatible (for function argument checking).
    /// Returns true if `value` can be passed where `expected` is required.
    fn types_compatible(&self, value: &Type, expected: &Type) -> bool {
        // Same type is always compatible
        if value == expected {
            return true;
        }

        // Unknown types are compatible with anything (for inference)
        if *value == Type::Unknown || *expected == Type::Unknown {
            return true;
        }

        // Allow numeric coercion between numeric types
        if value.is_numeric() && expected.is_numeric() {
            return true;
        }

        // Allow T to be passed where ?T is expected
        if let Type::Optional { inner } = expected
            && self.types_compatible(value, inner)
        {
            return true;
        }

        false
    }

    /// Infer the element type from a for loop range.
    /// Returns the type that the loop variable should have.
    fn infer_for_range_type(&mut self, range: &ForRange) -> SemanticResult<Type> {
        match range {
            ForRange::Range { start, end } => {
                // Analyze start and end expressions to get their types
                let start_ty = self.analyze_expr(start)?;
                let end_ty = self.analyze_expr(end)?;

                // For numeric ranges, prefer the start type if both are numeric
                // If one is Unknown, use the other
                if start_ty == Type::Unknown {
                    if end_ty == Type::Unknown {
                        // Both unknown, default to usize for indices
                        Ok(Type::Usize)
                    } else {
                        Ok(end_ty)
                    }
                } else if end_ty == Type::Unknown || start_ty == end_ty {
                    Ok(start_ty)
                } else if start_ty.is_numeric() && end_ty.is_numeric() {
                    // Both numeric but different - use start type
                    Ok(start_ty)
                } else {
                    // Mismatched types - default to usize
                    Ok(Type::Usize)
                }
            }
            ForRange::Collection(expr) => {
                // Analyze the collection expression
                let coll_ty = self.analyze_expr(expr)?;

                // Extract element type from collection
                match coll_ty {
                    Type::Array { element, .. } => Ok(*element),
                    Type::Slice { element } => Ok(*element),
                    Type::Set { element } => Ok(*element),
                    Type::Allocator { .. } => Ok(Type::Qubit), // Iterating over qubit allocator
                    _ => {
                        // For other types (including Pointer), default to usize
                        Ok(Type::Usize)
                    }
                }
            }
        }
    }

    /// Check that an expression doesn't escape a reference to a local variable.
    /// This prevents returning pointers/slices to stack-allocated data.
    fn check_no_local_escape(&self, expr: &Expr, location: SourceLocation) -> SemanticResult<()> {
        match expr {
            // &x - check if x is a local variable
            Expr::Unary(unary) if unary.op == UnaryOp::AddrOf => {
                if let Some(name) = self.get_local_var_name(&unary.operand) {
                    return Err(SemanticError::ReturnReferenceToLocal { name, location });
                }
            }
            // arr[start..end] - check if arr is a local array (slice creation)
            Expr::Range(range) => {
                // Range expressions in return context could be slices
                // For now, we check if the operands reference locals
                if let Some(start) = &range.start {
                    self.check_no_local_escape(start, location.clone())?;
                }
                if let Some(end) = &range.end {
                    self.check_no_local_escape(end, location.clone())?;
                }
            }
            // Index with range: arr[0..n]
            Expr::Index(index) => {
                // Check if this is a slice (index is a range) of a local array
                if matches!(index.index, Expr::Range(_))
                    && let Some(name) = self.get_local_var_name(&index.object)
                {
                    return Err(SemanticError::ReturnSliceOfLocal { name, location });
                }
            }
            // Tuple/struct with references inside - check each element
            Expr::Tuple(tuple) => {
                for elem in &tuple.elements {
                    self.check_no_local_escape(elem, location.clone())?;
                }
            }
            Expr::StructInit(init) => {
                for field in &init.fields {
                    self.check_no_local_escape(&field.value, location.clone())?;
                }
            }
            Expr::BracketArray(arr) => {
                for elem in &arr.elements {
                    self.check_no_local_escape(elem, location.clone())?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Get the name of a local variable if the expression is a simple identifier
    /// referring to a variable defined in the current function scope (not a parameter).
    fn get_local_var_name(&self, expr: &Expr) -> Option<String> {
        if let Expr::Ident(ident) = expr {
            // Check if this identifier is a local variable (not a parameter or global)
            if let Some(symbol) = self.symbols.lookup(&ident.name) {
                match &symbol.kind {
                    SymbolKind::Variable { .. } => {
                        // It's a variable - check if it's in function scope (local)
                        // For now, we consider all variables in function scope as local
                        // Parameters are tracked separately as SymbolKind::Parameter
                        return Some(ident.name.clone());
                    }
                    SymbolKind::Parameter { .. } => {
                        // Parameters are borrowed from caller, so returning &param is OK
                        // (the caller owns the data, not us)
                        return None;
                    }
                    _ => return None,
                }
            }
        }
        None
    }

    /// Check binary operator types.
    fn check_binary_op(
        &self,
        op: BinaryOp,
        left: &Type,
        right: &Type,
        location: Option<SourceLocation>,
    ) -> SemanticResult<Type> {
        match op {
            BinaryOp::Add | BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod => {
                if left.is_numeric() && right.is_numeric() {
                    Ok(left.clone())
                } else {
                    Err(SemanticError::TypeMismatch {
                        expected: "numeric".to_string(),
                        found: format!("{} and {}", left.display_name(), right.display_name()),
                        location: location.unwrap_or_default(),
                    })
                }
            }
            BinaryOp::Sub => {
                // - works for numeric (subtraction) and Set (difference)
                if left.is_numeric() && right.is_numeric() {
                    Ok(left.clone())
                } else if let (Type::Set { element: l_elem }, Type::Set { element: r_elem }) =
                    (left, right)
                {
                    // Set difference returns a set of the same element type
                    let _ = r_elem; // Both sets should have compatible element types
                    Ok(Type::Set {
                        element: l_elem.clone(),
                    })
                } else {
                    Err(SemanticError::TypeMismatch {
                        expected: "numeric or Set".to_string(),
                        found: format!("{} and {}", left.display_name(), right.display_name()),
                        location: location.unwrap_or_default(),
                    })
                }
            }
            BinaryOp::Eq | BinaryOp::Ne => Ok(Type::Bool),
            BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                // These work for numeric (comparison) and Set (subset/superset)
                if left.is_numeric() && right.is_numeric() {
                    Ok(Type::Bool)
                } else if matches!((left, right), (Type::Set { .. }, Type::Set { .. })) {
                    // Set comparisons: < (proper subset), <= (subset), > (proper superset), >= (superset)
                    Ok(Type::Bool)
                } else {
                    Err(SemanticError::TypeMismatch {
                        expected: "numeric or Set".to_string(),
                        found: format!("{} and {}", left.display_name(), right.display_name()),
                        location: location.unwrap_or_default(),
                    })
                }
            }
            BinaryOp::And | BinaryOp::Or => {
                if *left == Type::Bool && *right == Type::Bool {
                    Ok(Type::Bool)
                } else {
                    Err(SemanticError::TypeMismatch {
                        expected: "bool".to_string(),
                        found: format!("{} and {}", left.display_name(), right.display_name()),
                        location: location.unwrap_or_default(),
                    })
                }
            }
            BinaryOp::Orelse => {
                // orelse: ?T orelse T -> T
                // Left must be optional, right must be assignable to inner type
                if let Type::Optional { inner } = left {
                    // Check if right is assignable to inner type (allows numeric coercion)
                    self.check_assignable(inner, right, location.clone())?;
                    Ok(*inner.clone())
                } else {
                    Err(SemanticError::TypeMismatch {
                        expected: "optional type (?T)".to_string(),
                        found: left.display_name(),
                        location: location.unwrap_or_default(),
                    })
                }
            }
            BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor => {
                // Bitwise ops work for integers, and also for sets:
                // & = intersection, | = union, ^ = symmetric difference
                if left.is_integer() && right.is_integer() {
                    Ok(left.clone())
                } else if let (Type::Set { element: l_elem }, Type::Set { element: _ }) =
                    (left, right)
                {
                    // Set operations return a set of the same element type
                    Ok(Type::Set {
                        element: l_elem.clone(),
                    })
                } else if let (
                    Type::ErrorSet {
                        name: l_name,
                        errors: l_errors,
                    },
                    Type::ErrorSet {
                        name: r_name,
                        errors: r_errors,
                    },
                ) = (left, right)
                {
                    // Error set union: ErrorA || ErrorB
                    // Only BitOr makes sense for error sets (union)
                    if op != BinaryOp::BitOr {
                        return Err(SemanticError::TypeMismatch {
                            expected: "|| (union) operator for error sets".to_string(),
                            found: format!("{:?}", op),
                            location: location.unwrap_or_default(),
                        });
                    }
                    // Combine error variants, deduplicating
                    let mut combined_errors = l_errors.clone();
                    for err in r_errors {
                        if !combined_errors.contains(err) {
                            combined_errors.push(err.clone());
                        }
                    }
                    Ok(Type::ErrorSet {
                        name: format!("{}||{}", l_name, r_name),
                        errors: combined_errors,
                    })
                } else if let (
                    Type::FaultSet {
                        name: l_name,
                        faults: l_faults,
                    },
                    Type::FaultSet {
                        name: r_name,
                        faults: r_faults,
                    },
                ) = (left, right)
                {
                    // Fault set union: FaultA || FaultB
                    if op != BinaryOp::BitOr {
                        return Err(SemanticError::TypeMismatch {
                            expected: "|| (union) operator for fault sets".to_string(),
                            found: format!("{:?}", op),
                            location: location.unwrap_or_default(),
                        });
                    }
                    let mut combined_faults = l_faults.clone();
                    for fault in r_faults {
                        if !combined_faults.contains(fault) {
                            combined_faults.push(fault.clone());
                        }
                    }
                    Ok(Type::FaultSet {
                        name: format!("{}||{}", l_name, r_name),
                        faults: combined_faults,
                    })
                } else {
                    Err(SemanticError::TypeMismatch {
                        expected: "integer, Set, or error/fault set".to_string(),
                        found: format!("{} and {}", left.display_name(), right.display_name()),
                        location: location.unwrap_or_default(),
                    })
                }
            }
            BinaryOp::Shl | BinaryOp::Shr => {
                if left.is_integer() {
                    Ok(left.clone())
                } else {
                    Err(SemanticError::TypeMismatch {
                        expected: "integer".to_string(),
                        found: left.display_name(),
                        location: location.unwrap_or_default(),
                    })
                }
            }

            BinaryOp::In | BinaryOp::NotIn => {
                // Membership operators: element in Set(element) -> bool
                if let Type::Set { element: set_elem } = right {
                    // Check that left type matches the set's element type
                    // For now, just return bool - more strict checking can be added later
                    let _ = set_elem; // Acknowledge we have the element type
                    Ok(Type::Bool)
                } else {
                    Err(SemanticError::TypeMismatch {
                        expected: "Set".to_string(),
                        found: right.display_name(),
                        location: location.unwrap_or_default(),
                    })
                }
            }

            BinaryOp::Catch => {
                // catch: T!E catch handler -> T
                // Left should be error union, right is handler that returns T
                match left {
                    Type::ErrorUnion { payload, .. } => {
                        // Handler should return payload type (or be compatible)
                        // For now, just check that handler produces a value
                        let _ = right; // Handler type - could validate more strictly
                        Ok(*payload.clone())
                    }
                    _ => Err(SemanticError::TypeMismatch {
                        expected: "error union (T!E)".to_string(),
                        found: left.display_name(),
                        location: location.unwrap_or_default(),
                    }),
                }
            }
        }
    }

    /// Check unary operator types.
    fn check_unary_op(
        &self,
        op: UnaryOp,
        operand: &Type,
        location: Option<SourceLocation>,
    ) -> SemanticResult<Type> {
        match op {
            UnaryOp::Neg => {
                if operand.is_numeric() {
                    Ok(operand.clone())
                } else {
                    Err(SemanticError::TypeMismatch {
                        expected: "numeric".to_string(),
                        found: operand.display_name(),
                        location: location.unwrap_or_default(),
                    })
                }
            }
            UnaryOp::Not => {
                if *operand == Type::Bool {
                    Ok(Type::Bool)
                } else {
                    Err(SemanticError::TypeMismatch {
                        expected: "bool".to_string(),
                        found: operand.display_name(),
                        location: location.unwrap_or_default(),
                    })
                }
            }
            UnaryOp::BitNot => {
                if operand.is_integer() {
                    Ok(operand.clone())
                } else {
                    Err(SemanticError::TypeMismatch {
                        expected: "integer".to_string(),
                        found: operand.display_name(),
                        location: location.unwrap_or_default(),
                    })
                }
            }
            UnaryOp::AddrOf => Ok(Type::Pointer {
                pointee: Box::new(operand.clone()),
                is_const: false,
                is_many: false,
            }),
            UnaryOp::Deref => match operand {
                Type::Pointer { pointee, .. } => Ok(*pointee.clone()),
                _ => Err(SemanticError::TypeMismatch {
                    expected: "pointer".to_string(),
                    found: operand.display_name(),
                    location: location.unwrap_or_default(),
                }),
            },
            UnaryOp::OptionalUnwrap => match operand {
                Type::Optional { inner } => Ok(*inner.clone()),
                _ => Err(SemanticError::TypeMismatch {
                    expected: "optional".to_string(),
                    found: operand.display_name(),
                    location: location.unwrap_or_default(),
                }),
            },
            UnaryOp::ErrorUnwrap => {
                // For error unions, unwrap returns the success type
                match operand {
                    Type::ErrorUnion { payload, .. } => Ok(*payload.clone()),
                    _ => Err(SemanticError::TypeMismatch {
                        expected: "error union (T!E)".to_string(),
                        found: operand.display_name(),
                        location: location.unwrap_or_default(),
                    }),
                }
            }
            UnaryOp::Try => {
                // try: T!E -> T, propagates E to caller
                match operand {
                    Type::ErrorUnion { payload, .. } => Ok(*payload.clone()),
                    _ => Err(SemanticError::TypeMismatch {
                        expected: "error union (T!E)".to_string(),
                        found: operand.display_name(),
                        location: location.unwrap_or_default(),
                    }),
                }
            }
        }
    }

    /// Validate a qubit reference.
    fn validate_qubit_ref(&self, slot_ref: &ast::SlotRef) -> SemanticResult<()> {
        // First check that the allocator exists in the symbol table
        let is_allocator = if let Some(symbol) = self.symbols.lookup(&slot_ref.allocator) {
            match &symbol.kind {
                SymbolKind::Variable { ty, .. } | SymbolKind::Parameter { ty, .. } => {
                    matches!(ty, Type::Allocator { .. })
                }
                SymbolKind::Allocator { .. } => true,
                _ => false,
            }
        } else {
            false
        };

        if !is_allocator {
            return Err(SemanticError::AllocatorNotFound {
                name: slot_ref.allocator.clone(),
                location: slot_ref.location.clone().unwrap_or_default(),
            });
        }

        // Check bounds if both index and capacity are known at compile time
        // Get capacity from qubit_states (where qalloc capacity is tracked)
        if let Some(alloc_info) = self.qubit_states.get_allocator(&slot_ref.allocator)
            && let Some(capacity) = alloc_info.capacity
            && let Some(index) = self.try_extract_constant_usize(&slot_ref.index)
            && index >= capacity
        {
            return Err(SemanticError::QubitIndexOutOfBounds {
                allocator: slot_ref.allocator.clone(),
                index,
                capacity,
                location: slot_ref.location.clone().unwrap_or_default(),
            });
        }

        Ok(())
    }

    /// Validate qubit bounds for gate target expressions.
    /// This handles Index expressions (q[5]), tuples, sets, etc.
    fn validate_gate_target_bounds(&self, target: &Expr) -> SemanticResult<()> {
        match target {
            Expr::Index(index_expr) => {
                // Check if this is an allocator index access (q[5])
                if let Expr::Ident(ident) = &index_expr.object {
                    let allocator_name = &ident.name;

                    // Check if this is an allocator
                    let is_allocator = if let Some(symbol) = self.symbols.lookup(allocator_name) {
                        matches!(
                            &symbol.kind,
                            SymbolKind::Variable {
                                ty: Type::Allocator { .. },
                                ..
                            } | SymbolKind::Allocator { .. }
                        )
                    } else {
                        false
                    };

                    if is_allocator {
                        // Get capacity from qubit_states
                        if let Some(alloc_info) = self.qubit_states.get_allocator(allocator_name)
                            && let Some(capacity) = alloc_info.capacity
                            && let Some(index) = self.try_extract_constant_usize(&index_expr.index)
                            && index >= capacity
                        {
                            return Err(SemanticError::QubitIndexOutOfBounds {
                                allocator: allocator_name.clone(),
                                index,
                                capacity,
                                location: index_expr.location.clone().unwrap_or_default(),
                            });
                        }
                    }
                }
                Ok(())
            }
            Expr::Tuple(tuple) => {
                // Validate each element in the tuple (e.g., (q[0], q[1]))
                for elem in &tuple.elements {
                    self.validate_gate_target_bounds(elem)?;
                }
                Ok(())
            }
            Expr::Set(set_expr) => {
                // Validate each element in the set (e.g., {q[0], q[1], q[2]})
                for elem in &set_expr.elements {
                    self.validate_gate_target_bounds(elem)?;
                }
                Ok(())
            }
            Expr::BracketArray(array) => {
                // Validate each element in the array (e.g., [q[0], q[1], q[2]])
                for elem in &array.elements {
                    self.validate_gate_target_bounds(elem)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Validate qubit states for gate target expressions (strict mode only).
    /// This ensures qubits are prepared before gates are applied.
    fn validate_gate_target_states(
        &self,
        target: &Expr,
        location: &SourceLocation,
    ) -> SemanticResult<()> {
        // Extract qubit IDs from the target expression
        let qubit_ids = self.extract_qubit_ids_from_arg(target);

        // Validate each qubit is prepared
        for (allocator, index) in qubit_ids {
            self.qubit_states
                .validate_for_gate(&allocator, index, location)?;
        }

        Ok(())
    }

    /// Prepare qubits targeted by a prepare gate (PZ).
    /// Transitions targeted qubits to the Prepared state.
    fn prepare_gate_targets(&mut self, target: &Expr) {
        // Extract qubit IDs from the target expression
        let qubit_ids = self.extract_qubit_ids_from_arg(target);

        // Transition each qubit to Prepared state
        for (allocator, index) in qubit_ids {
            if let Some(alloc) = self.qubit_states.get_allocator_mut(&allocator) {
                let _ = alloc.prepare_slot(index);
            }
        }

        // Also handle the case where the target is a bare allocator (pz q; prepares all)
        if let Expr::Ident(ident) = target
            && let Some(alloc) = self.qubit_states.get_allocator_mut(&ident.name)
        {
            alloc.prepare_all();
        }
    }

    /// Check for duplicate qubit usage within a tick block.
    /// In quantum computing, parallel operations cannot target the same qubit.
    fn check_duplicate_qubits_in_tick(
        &self,
        statements: &[ast::Stmt],
        tick_location: &Option<SourceLocation>,
    ) -> SemanticResult<()> {
        // Collect all qubit identifications: (allocator_name, constant_index)
        let mut seen_qubits: BTreeSet<(String, usize)> = BTreeSet::new();

        for stmt in statements {
            let qubits = self.collect_qubit_ids_from_stmt(stmt);

            for (allocator, index) in qubits {
                let key = (allocator.clone(), index);
                if !seen_qubits.insert(key) {
                    // Duplicate found
                    return Err(SemanticError::DuplicateQubitInTick {
                        allocator,
                        index,
                        location: tick_location.clone().unwrap_or_default(),
                    });
                }
            }
        }

        Ok(())
    }

    /// Collect qubit identifications (allocator, index) from a statement.
    fn collect_qubit_ids_from_stmt(&self, stmt: &ast::Stmt) -> Vec<(String, usize)> {
        match stmt {
            // Stmt::Gate uses SlotRef
            ast::Stmt::Gate(gate_op) => gate_op
                .targets
                .iter()
                .filter_map(|slot_ref| {
                    self.try_extract_constant_usize(&slot_ref.index)
                        .map(|idx| (slot_ref.allocator.clone(), idx))
                })
                .collect(),
            // Stmt::Measure uses SlotRef
            ast::Stmt::Measure(measure_op) => measure_op
                .targets
                .iter()
                .filter_map(|slot_ref| {
                    self.try_extract_constant_usize(&slot_ref.index)
                        .map(|idx| (slot_ref.allocator.clone(), idx))
                })
                .collect(),
            // PrepareOp doesn't target individual qubits in tick context
            ast::Stmt::Prepare(_) => Vec::new(),
            // Nested tick blocks recursively check their contents
            ast::Stmt::Tick(tick_stmt) => tick_stmt
                .body
                .iter()
                .flat_map(|s| self.collect_qubit_ids_from_stmt(s))
                .collect(),
            // Expression statements can contain gate calls (h(q[0]), cx(q[0], q[1]), etc.)
            ast::Stmt::Expr(expr_stmt) => self.collect_qubit_ids_from_expr(&expr_stmt.expr),
            // Other statements don't contain qubit references
            _ => Vec::new(),
        }
    }

    /// Collect qubit identifications from an expression (for gate calls and measurements).
    fn collect_qubit_ids_from_expr(&self, expr: &Expr) -> Vec<(String, usize)> {
        match expr {
            // Gate expression: h q[0], cx (q[0], q[1]), rx(angle) q[0], etc.
            Expr::Gate(gate) => {
                // Extract qubit IDs from the gate target
                self.extract_qubit_ids_from_arg(&gate.target)
            }
            // Measure expression: mz(u1) q[0], mz(u8) q[0..8], etc.
            Expr::Measure(measure) => {
                // Extract qubit IDs from the measurement target
                self.extract_qubit_ids_from_arg(&measure.targets)
            }
            // Direct function call: h(q[0]), cx(q[0], q[1]), etc.
            Expr::Call(call) => {
                // Check if this is a gate call
                if let Expr::Ident(ident) = &call.callee
                    && is_gate_name(&ident.name)
                {
                    // Collect qubit IDs from arguments
                    return call
                        .args
                        .iter()
                        .flat_map(|arg| self.extract_qubit_ids_from_arg(arg))
                        .collect();
                }
                Vec::new()
            }
            // Batch apply: h { q[0], q[1] } or rz(pi/4) { q[0], q[1] }
            Expr::BatchApply(batch) => {
                // Extract gate name from operation
                let gate_name = match &batch.operation {
                    Expr::Ident(ident) => Some(&ident.name),
                    Expr::Call(call) => {
                        if let Expr::Ident(ident) = &call.callee {
                            Some(&ident.name)
                        } else {
                            None
                        }
                    }
                    _ => None,
                };
                if let Some(name) = gate_name
                    && is_gate_name(name)
                {
                    return batch
                        .targets
                        .iter()
                        .flat_map(|target| self.extract_qubit_ids_from_arg(target))
                        .collect();
                }
                Vec::new()
            }
            _ => Vec::new(),
        }
    }

    /// Extract qubit (allocator, index) from a gate argument expression.
    fn extract_qubit_ids_from_arg(&self, expr: &Expr) -> Vec<(String, usize)> {
        match expr {
            // Index expression: q[0]
            Expr::Index(index_expr) => {
                if let Expr::Ident(ident) = &index_expr.object
                    && let Some(idx) = self.try_extract_constant_usize(&index_expr.index)
                {
                    return vec![(ident.name.clone(), idx)];
                }
                Vec::new()
            }
            // Tuple of qubits: (q[0], q[1]) for two-qubit gates
            Expr::Tuple(tuple) => tuple
                .elements
                .iter()
                .flat_map(|e| self.extract_qubit_ids_from_arg(e))
                .collect(),
            // Address-of array: &[q[0], q[1]]
            Expr::Unary(unary) if unary.op == ast::UnaryOp::AddrOf => {
                self.extract_qubit_ids_from_arg(&unary.operand)
            }
            // Bracket array: [q[0], q[1]]
            Expr::BracketArray(arr) => arr
                .elements
                .iter()
                .flat_map(|e| self.extract_qubit_ids_from_arg(e))
                .collect(),
            // Set literal: [q[0], q[1]
            Expr::Set(set) => set
                .elements
                .iter()
                .flat_map(|e| self.extract_qubit_ids_from_arg(e))
                .collect(),
            _ => Vec::new(),
        }
    }

    // =========================================================================
    // Allocator Extraction Helpers
    // =========================================================================

    /// Try to extract allocator capacity from a qalloc(n) call.
    fn try_extract_allocator_capacity(&self, expr: &Expr) -> Option<usize> {
        if let Expr::Call(call) = expr
            && let Expr::Ident(ident) = &call.callee
            && ident.name == "qalloc"
            && call.args.len() == 1
        {
            return self.try_extract_constant_usize(&call.args[0]);
        }
        None
    }

    /// Try to extract child allocator info from base.child(n) call.
    fn try_extract_child_allocator(&self, expr: &Expr) -> Option<(String, usize)> {
        if let Expr::Call(call) = expr
            && let Expr::Field(field) = &call.callee
            && field.field == "child"
            && call.args.len() == 1
            && let Expr::Ident(parent_ident) = &field.object
        {
            let capacity = self.try_extract_constant_usize(&call.args[0])?;
            return Some((parent_ident.name.clone(), capacity));
        }
        None
    }

    /// Try to extract a constant usize from an expression.
    fn try_extract_constant_usize(&self, expr: &Expr) -> Option<usize> {
        match expr {
            Expr::IntLit(lit) => Some(lit.value as usize),
            Expr::Ident(ident) => {
                // Try to look up a comptime constant in our stored values
                if let Some(symbol) = self.symbols.lookup(&ident.name)
                    && let SymbolKind::Variable {
                        ty: Type::Comptime(_),
                        is_const: true,
                        ..
                    } = &symbol.kind
                {
                    // Look up the value in the comptime evaluator context
                    if let Some(val) = self.comptime.context.lookup(&ident.name) {
                        return val.to_usize();
                    }
                }
                None
            }
            // For other expressions, try comptime evaluation
            _ => {
                // Create a temporary evaluator to try evaluation
                let mut evaluator = ComptimeEvaluator::new();
                if let Ok(value) = evaluator.eval_expr(expr) {
                    value.to_usize()
                } else {
                    None
                }
            }
        }
    }

    /// Count the number of elements in a gate target expression.
    /// Used for batch gate arity checking.
    /// - Single qubit ref (q[0]) => 1
    /// - Tuple of 2 (q[0], q[1]) => 2
    /// - Tuple of 3 (q[0], q[1], q[2]) => 3
    fn count_target_elements(&self, expr: &Expr) -> usize {
        match expr {
            Expr::Tuple(tuple) => tuple.elements.len(),
            // Single element (qubit ref, identifier, etc.)
            _ => 1,
        }
    }

    // =========================================================================
    // NASA Power of 10 Helpers
    // =========================================================================

    /// Check if a loop has a valid bound (NASA Power of 10 Rule 2).
    fn check_loop_bound(&self, bound: usize, location: &SourceLocation) -> SemanticResult<()> {
        if bound > MAX_LOOP_BOUND {
            return Err(SemanticError::LoopBoundTooLarge {
                bound,
                max: MAX_LOOP_BOUND,
                location: location.clone(),
            });
        }
        Ok(())
    }

    // =========================================================================
    // Typed Measurement Analysis
    // =========================================================================

    /// Analyze a typed measurement call: mz(T) target
    ///
    /// Examples:
    /// - `mz(u1) q[0]` - single qubit, returns u1
    /// - `mz([]u1, &[q[0], q[1]])` - multiple qubits, returns []u1
    fn analyze_typed_measurement(&mut self, call: &ast::CallExpr) -> SemanticResult<Type> {
        let location = call.location.clone().unwrap_or_default();

        // Must have exactly 2 arguments: type and target
        if call.args.len() != 2 {
            return Err(SemanticError::MeasurementMissingArgs { location });
        }

        // Extract measurement result type from first argument
        let result_type = self.extract_measurement_type(&call.args[0])?;

        // Extract and validate targets from second argument
        let targets = self.extract_measurement_targets(&call.args[1])?;

        // Check for duplicate qubits (array elements must be unique but ordered)
        self.check_measurement_uniqueness(&targets, &location)?;

        // Validate qubit states if in strict mode
        for (allocator, index) in &targets {
            if self.strict_mode {
                self.qubit_states
                    .validate_for_gate(allocator, *index, &location)?;
            }
            // Transition to unprepared after measurement
            if let Some(alloc) = self.qubit_states.get_allocator_mut(allocator) {
                alloc.measure_slot(*index);
            }
        }

        // Return type depends on whether targets is single or multiple
        // For single target, return scalar; for multiple, return slice
        if targets.len() == 1 {
            // Single target: mz(u1) q[0] returns u1
            Ok(result_type)
        } else {
            // Multiple targets: mz([]u1, &[q[0], q[1]]) returns []u1
            // The result_type should already be a slice type
            Ok(result_type)
        }
    }

    /// Extract measurement result type from the type argument.
    ///
    /// Valid types: u1, u8, u64, []u1, []u8, []u64
    fn extract_measurement_type(&self, expr: &Expr) -> SemanticResult<Type> {
        let location = expr.get_location().unwrap_or_default();

        match expr {
            // Simple type: u1, u8, u64, etc. (arbitrary bit-width)
            Expr::Ident(ident) => {
                // Parse arbitrary unsigned integer type: u<bits>
                if let Some(bits_str) = ident.name.strip_prefix('u')
                    && let Ok(bits) = bits_str.parse::<u16>()
                    && let Some(bw) = BitWidth::new(bits)
                {
                    return Ok(Type::UInt { bits: bw });
                }
                Err(SemanticError::InvalidMeasurementType {
                    ty: ident.name.clone(),
                    location,
                })
            }

            // Array type: []u1, []u8, []u64 - parsed as array_type_expr
            Expr::ArrayInit(arr) => {
                // Empty array init [] with element type
                // This is how []u1 might be parsed - check the type
                if arr.elements.is_empty() {
                    // Need to get element type from context
                    Ok(Type::Slice {
                        element: Box::new(Type::UInt {
                            bits: BitWidth::BITS_1,
                        }), // Default to u1
                    })
                } else {
                    Err(SemanticError::InvalidMeasurementType {
                        ty: "array literal".to_string(),
                        location,
                    })
                }
            }

            // SlotRef for []type syntax (array type expression)
            // The parser might produce this for []u1
            Expr::SlotRef(slot_ref) => {
                // This might be []u1 parsed as a slot ref with allocator="u1"
                // Parse arbitrary unsigned integer type from allocator name
                if let Some(bits_str) = slot_ref.allocator.strip_prefix('u')
                    && let Ok(bits) = bits_str.parse::<u16>()
                    && let Some(bw) = BitWidth::new(bits)
                {
                    return Ok(Type::Slice {
                        element: Box::new(Type::UInt { bits: bw }),
                    });
                }
                Err(SemanticError::InvalidMeasurementType {
                    ty: format!("[]{}", slot_ref.allocator),
                    location,
                })
            }

            _ => Err(SemanticError::InvalidMeasurementType {
                ty: "unknown".to_string(),
                location,
            }),
        }
    }

    /// Count measurement targets from the target expression.
    ///
    /// Accepts:
    /// - Single qubit: q[0] → 1
    /// - Bracket array: [q[0], q[1]] → element count
    /// - Allocator: q → allocator capacity
    fn count_measurement_targets(&self, expr: &Expr) -> SemanticResult<usize> {
        match expr {
            // Single qubit: q[0]
            Expr::Index(_) => Ok(1),

            // Bracket array: [q[0], q[1]]
            Expr::BracketArray(arr) => Ok(arr.elements.len()),

            // Allocator: q (measure all qubits)
            Expr::Ident(ident) => {
                if let Some(alloc) = self.qubit_states.get_allocator(&ident.name) {
                    if let Some(capacity) = alloc.capacity {
                        Ok(capacity)
                    } else {
                        // Unknown capacity - can't validate at compile time
                        Ok(0) // Will be validated at runtime
                    }
                } else {
                    Ok(0) // Not an allocator, will be caught by analyze_expr
                }
            }

            _ => Ok(0), // Will be caught by analyze_expr
        }
    }

    /// Calculate the bit size of a type for pack mode validation.
    ///
    /// Returns None if the size cannot be determined at compile time.
    fn type_bit_size(&self, ty: &Type) -> Option<usize> {
        match ty {
            // Arbitrary-width integers
            Type::UInt { bits } | Type::IInt { bits } => Some(bits.get() as usize),
            Type::Bool => Some(1),
            Type::Array { element, size } => {
                if let (Some(elem_bits), Some(arr_size)) = (self.type_bit_size(element), size) {
                    Some(elem_bits * (*arr_size as usize))
                } else {
                    None
                }
            }
            Type::Struct { fields, .. } => {
                let mut total = 0;
                for (_, field_ty) in fields {
                    if let Some(bits) = self.type_bit_size(field_ty) {
                        total += bits;
                    } else {
                        return None;
                    }
                }
                Some(total)
            }
            Type::Tuple { elements } => {
                let mut total = 0;
                for elem_ty in elements {
                    if let Some(bits) = self.type_bit_size(elem_ty) {
                        total += bits;
                    } else {
                        return None;
                    }
                }
                Some(total)
            }
            _ => None, // Unknown size for other types
        }
    }

    /// Extract measurement targets from the target argument (legacy).
    ///
    /// Accepts:
    /// - Single qubit: q[0]
    /// - Array of qubits: &[q[0], q[1]]
    fn extract_measurement_targets(&self, expr: &Expr) -> SemanticResult<Vec<(String, usize)>> {
        let location = expr.get_location().unwrap_or_default();

        match expr {
            // Single qubit: q[0]
            Expr::Index(index) => {
                let (allocator, idx) = self.extract_qubit_from_index(index)?;
                Ok(vec![(allocator, idx)])
            }

            // Address-of array: &[q[0], q[1]]
            Expr::Unary(unary) if matches!(unary.op, UnaryOp::AddrOf) => match &unary.operand {
                Expr::BracketArray(arr) => {
                    let mut targets = Vec::new();
                    for elem in &arr.elements {
                        if let Expr::Index(index) = elem {
                            let (allocator, idx) = self.extract_qubit_from_index(index)?;
                            targets.push((allocator, idx));
                        } else {
                            return Err(SemanticError::InvalidQubitRef { location });
                        }
                    }
                    Ok(targets)
                }
                _ => Err(SemanticError::InvalidQubitRef { location }),
            },

            _ => Err(SemanticError::InvalidQubitRef { location }),
        }
    }

    /// Extract allocator name and index from an index expression.
    fn extract_qubit_from_index(&self, index: &ast::IndexExpr) -> SemanticResult<(String, usize)> {
        let location = index.location.clone().unwrap_or_default();

        // Get allocator name
        let allocator = match &index.object {
            Expr::Ident(ident) => ident.name.clone(),
            _ => return Err(SemanticError::InvalidQubitRef { location }),
        };

        // Get index (must be comptime-known for uniqueness checking)
        let idx = self
            .try_extract_constant_usize(&index.index)
            .ok_or(SemanticError::InvalidQubitRef { location })?;

        Ok((allocator, idx))
    }

    /// Check that all qubits in measurement are unique.
    fn check_measurement_uniqueness(
        &self,
        targets: &[(String, usize)],
        location: &SourceLocation,
    ) -> SemanticResult<()> {
        let mut seen = BTreeSet::new();
        for (allocator, index) in targets {
            let key = (allocator.clone(), *index);
            if !seen.insert(key) {
                return Err(SemanticError::DuplicateQubitInMeasurement {
                    allocator: allocator.clone(),
                    index: *index,
                    location: location.clone(),
                });
            }
        }
        Ok(())
    }

    /// Analyze an @import builtin expression.
    fn analyze_import(&mut self, builtin: &ast::BuiltinExpr) -> SemanticResult<Type> {
        let location = builtin.location.clone().unwrap_or_default();

        // Extract the import path from the first argument
        if builtin.args.is_empty() {
            return Err(SemanticError::TypeMismatch {
                expected: "string literal".to_string(),
                found: "no arguments".to_string(),
                location,
            });
        }

        let import_path = match &builtin.args[0] {
            Expr::StringLit(s) => s.value.clone(),
            _ => {
                return Err(SemanticError::TypeMismatch {
                    expected: "string literal".to_string(),
                    found: "non-string expression".to_string(),
                    location,
                });
            }
        };

        // Try to load the module
        let from_file = self.current_file.as_deref();
        match self.module_loader.load(&import_path, from_file) {
            Ok(module) => {
                // Clone the exports and path to release the borrow on module_loader
                let module_exports = module.exports.clone();
                let module_path = module.path.display().to_string();
                // module reference is now released after cloning

                // Build exports map for the type
                let mut exports = std::collections::BTreeMap::new();
                for (name, export) in &module_exports {
                    let (kind, ty) = match export {
                        ExportedSymbol::Function {
                            params,
                            return_type,
                            ..
                        } => {
                            // Extract function signature from AST
                            let param_types: Vec<Type> =
                                params.iter().map(|(_, ty)| self.resolve_type(ty)).collect();
                            let ret_type = return_type
                                .as_ref()
                                .map(|t| self.resolve_type(t))
                                .unwrap_or(Type::Unit);
                            (
                                ModuleExportKind::Function,
                                Type::Function {
                                    params: param_types,
                                    return_type: Box::new(ret_type),
                                },
                            )
                        }
                        ExportedSymbol::Const { .. } => (ModuleExportKind::Const, Type::Unknown),
                        ExportedSymbol::Type { .. } => (ModuleExportKind::Type, Type::Type),
                        ExportedSymbol::ErrorSet { variants, .. } => {
                            // Imported error sets don't carry associated data types
                            let errors: Vec<(String, Option<Box<Type>>)> =
                                variants.iter().map(|v| (v.clone(), None)).collect();
                            (
                                ModuleExportKind::ErrorSet,
                                Type::ErrorSet {
                                    name: name.clone(),
                                    errors,
                                },
                            )
                        }
                        ExportedSymbol::FaultSet { variants, .. } => {
                            // Imported fault sets don't carry associated data types
                            let faults: Vec<(String, Option<Box<Type>>)> =
                                variants.iter().map(|v| (v.clone(), None)).collect();
                            (
                                ModuleExportKind::FaultSet,
                                Type::FaultSet {
                                    name: name.clone(),
                                    faults,
                                },
                            )
                        }
                    };
                    exports.insert(name.clone(), (kind, ty));
                }

                Ok(Type::Module {
                    path: module_path,
                    exports,
                })
            }
            Err(e) => {
                // Module loading failed - report as semantic error
                Err(SemanticError::ModuleError {
                    message: e.to_string(),
                    location,
                })
            }
        }
    }

    // =========================================================================
    // Alias Analysis
    // =========================================================================

    /// Analyze an alias statement.
    /// Validates that the source is a slice expression and checks for overlaps.
    fn analyze_alias(&mut self, alias: &crate::ast::AliasBinding) -> SemanticResult<()> {
        let location = alias.location.clone().unwrap_or_default();

        // Extract source variable and range from the alias source expression
        let (source_name, range) = self.extract_alias_source_info(&alias.source, &location)?;

        // Check for overlaps with existing aliases on the same source
        for (existing_name, existing_info) in &self.aliases {
            if existing_info.source == source_name
                && let (Some(new_range), Some(existing_range)) = (range, existing_info.range)
                && Self::ranges_overlap(new_range, existing_range)
            {
                return Err(SemanticError::OverlappingAlias(Box::new(
                    OverlappingAliasError {
                        new_alias: alias.name.clone(),
                        existing_alias: existing_name.clone(),
                        source_var: source_name.clone(),
                        overlap_range: format!(
                            "{}..{} overlaps with {}..{}",
                            new_range.0, new_range.1, existing_range.0, existing_range.1
                        ),
                        location,
                    },
                )));
            }
        }

        // Analyze the source expression for type checking
        let source_ty = self.analyze_expr(&alias.source)?;

        // Store alias info for future overlap checks
        self.aliases.insert(
            alias.name.clone(),
            AliasInfo {
                name: alias.name.clone(),
                source: source_name.clone(),
                range,
                location: location.clone(),
            },
        );

        // Register the alias as a variable in the symbol table
        self.symbols.define(Symbol {
            name: alias.name.clone(),
            kind: SymbolKind::Variable {
                ty: source_ty,
                is_const: true, // Aliases are always immutable in MVP
                is_comptime: false,
            },
            location: Some(location),
        })?;

        Ok(())
    }

    /// Extract the source variable name and static range from an alias source expression.
    fn extract_alias_source_info(
        &self,
        expr: &crate::ast::Expr,
        location: &SourceLocation,
    ) -> SemanticResult<(String, Option<(i64, i64)>)> {
        // The source must be a slice expression: source[start..end]
        if let crate::ast::Expr::Index(index) = expr {
            // Get the base name
            let source_name = self.extract_base_name(&index.object)?;

            // Get the range if it's a RangeExpr
            if let crate::ast::Expr::Range(range_expr) = &index.index {
                // Try to evaluate bounds at comptime
                let start = if let Some(start_expr) = &range_expr.start {
                    self.try_eval_comptime_int(start_expr)
                } else {
                    Some(0) // Default start is 0
                };

                let end = if let Some(end_expr) = &range_expr.end {
                    self.try_eval_comptime_int(end_expr)
                } else {
                    None // Open-ended range
                };

                if let (Some(s), Some(e)) = (start, end) {
                    return Ok((source_name, Some((s, e))));
                } else {
                    // Range is not fully comptime - allow but skip overlap checking
                    return Ok((source_name, None));
                }
            }
        }

        // Not a valid slice expression
        Err(SemanticError::AliasSourceNotSlice {
            found: format!("{:?}", expr),
            location: location.clone(),
        })
    }

    /// Extract the base variable name from an expression.
    fn extract_base_name(&self, expr: &crate::ast::Expr) -> SemanticResult<String> {
        match expr {
            crate::ast::Expr::Ident(ident) => Ok(ident.name.clone()),
            crate::ast::Expr::Index(index) => self.extract_base_name(&index.object),
            crate::ast::Expr::Field(field) => self.extract_base_name(&field.object),
            _ => Ok("<complex>".to_string()),
        }
    }

    /// Try to evaluate an expression as a comptime integer.
    fn try_eval_comptime_int(&self, expr: &crate::ast::Expr) -> Option<i64> {
        let mut evaluator = ComptimeEvaluator::new();
        // Populate with known comptime values
        for (name, val) in &self.comptime_values {
            evaluator.context.define(name, val.clone());
        }
        match evaluator.eval_expr(expr) {
            Ok(ComptimeValue::Int(n)) => Some(n),
            Ok(ComptimeValue::Uint(n)) => Some(n as i64),
            _ => None,
        }
    }

    /// Check if two ranges overlap.
    fn ranges_overlap(a: (i64, i64), b: (i64, i64)) -> bool {
        // Ranges [a.0, a.1) and [b.0, b.1) overlap if:
        // a.0 < b.1 && b.0 < a.1
        a.0 < b.1 && b.0 < a.1
    }

    // =========================================================================
    // Generic Type Instantiation
    // =========================================================================

    /// Serialize comptime values to a string key for caching.
    fn serialize_comptime_args(args: &[ComptimeValue]) -> String {
        args.iter()
            .map(|v| format!("{}", v))
            .collect::<Vec<_>>()
            .join("_")
    }

    /// Generate a mangled name for a specialized function.
    fn mangle_generic_name(base_name: &str, comptime_args: &[ComptimeValue]) -> String {
        let args_suffix = comptime_args
            .iter()
            .map(|v| match v {
                ComptimeValue::Int(n) => format!("{}", n),
                ComptimeValue::Uint(n) => format!("{}", n),
                ComptimeValue::Bool(b) => if *b { "true" } else { "false" }.to_string(),
                ComptimeValue::Type(t) => t.display_name().replace(' ', "_"),
                ComptimeValue::String(s) => s.replace(' ', "_"),
                _ => format!("{:?}", v),
            })
            .collect::<Vec<_>>()
            .join("__");
        format!("{}__CT__{}", base_name, args_suffix)
    }

    /// Instantiate a generic function with concrete comptime arguments.
    /// Returns the mangled name of the specialized function.
    pub fn instantiate_generic_function(
        &mut self,
        fn_name: &str,
        comptime_args: &[ComptimeValue],
        comptime_param_indices: &[usize],
        original_decl: &crate::ast::FnDecl,
    ) -> SemanticResult<String> {
        // Create cache key
        let args_key = Self::serialize_comptime_args(comptime_args);
        let cache_key = (fn_name.to_string(), args_key.clone());

        // Check if already instantiated
        if let Some(mangled_name) = self.generic_instantiations.get(&cache_key) {
            return Ok(mangled_name.clone());
        }

        // Generate mangled name for the specialized function
        let mangled_name = Self::mangle_generic_name(fn_name, comptime_args);

        // Clone the original declaration and substitute comptime params
        let mut specialized = original_decl.clone();
        specialized.name = mangled_name.clone();

        // Build a mapping from comptime param names to their concrete values
        let mut comptime_bindings: BTreeMap<String, ComptimeValue> = BTreeMap::new();
        for (i, &param_idx) in comptime_param_indices.iter().enumerate() {
            if param_idx < original_decl.params.len() && i < comptime_args.len() {
                let param_name = &original_decl.params[param_idx].name;
                comptime_bindings.insert(param_name.clone(), comptime_args[i].clone());
            }
        }

        // Remove comptime parameters from the specialized function
        // (they become concrete values, not parameters)
        specialized.params = original_decl
            .params
            .iter()
            .enumerate()
            .filter(|(i, _)| !comptime_param_indices.contains(i))
            .map(|(_, p)| p.clone())
            .collect();

        // Store the comptime bindings for use during analysis of the specialized function
        for (name, value) in &comptime_bindings {
            self.comptime_values.insert(name.clone(), value.clone());
            self.comptime.context.define(name, value.clone());
        }

        // Register the specialized function in the symbol table
        let params: Vec<(String, Type)> = specialized
            .params
            .iter()
            .map(|p| (p.name.clone(), self.resolve_type(&p.ty)))
            .collect();
        let return_type = specialized
            .return_type
            .as_ref()
            .map(|t| self.resolve_type(t))
            .unwrap_or(Type::Unit);

        self.symbols.define(Symbol {
            name: mangled_name.clone(),
            kind: SymbolKind::Function {
                params,
                return_type,
                is_pub: false,                  // Specialized functions are internal
                comptime_param_indices: vec![], // No longer generic
                original_decl: None,
            },
            location: specialized.location.clone(),
        })?;

        // Store the specialized function for later codegen
        self.specialized_functions.push(specialized);

        // Cache the instantiation
        self.generic_instantiations
            .insert(cache_key, mangled_name.clone());

        Ok(mangled_name)
    }

    /// Get the list of specialized functions generated during analysis.
    pub fn get_specialized_functions(&self) -> &[crate::ast::FnDecl] {
        &self.specialized_functions
    }
}

impl Default for SemanticAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Gate information for semantic analysis.
struct GateInfo {
    /// Number of qubit arguments (arity)
    arity: usize,
    /// Whether the gate takes angle parameters
    parameterized: bool,
}

/// Get gate information by name.
/// Returns None if the name is not a recognized gate.
fn get_gate_info(name: &str) -> Option<GateInfo> {
    match name {
        // Single-qubit Pauli gates (non-parameterized, arity 1)
        "h" | "x" | "y" | "z" => Some(GateInfo {
            arity: 1,
            parameterized: false,
        }),
        // Square root gates (sx, sy, sz and their daggers)
        // Note: S gate is "sz" not "s", Sdg is "szdg" not "sdg"
        "sx" | "sy" | "sz" | "sxdg" | "sydg" | "szdg" => Some(GateInfo {
            arity: 1,
            parameterized: false,
        }),
        // T gates (fourth root of Z)
        "t" | "tdg" => Some(GateInfo {
            arity: 1,
            parameterized: false,
        }),
        // F gates
        "f" | "fdg" | "f4" | "f4dg" => Some(GateInfo {
            arity: 1,
            parameterized: false,
        }),
        // Rotation gates (parameterized, arity 1)
        "rx" | "ry" | "rz" => Some(GateInfo {
            arity: 1,
            parameterized: true,
        }),
        // Two-qubit gates (non-parameterized, arity 2)
        "cx" | "cy" | "cz" | "ch" => Some(GateInfo {
            arity: 2,
            parameterized: false,
        }),
        "swap" | "iswap" => Some(GateInfo {
            arity: 2,
            parameterized: false,
        }),
        // Square-root two-qubit gates
        "sxx" | "syy" | "szz" | "sxxdg" | "syydg" | "szzdg" => Some(GateInfo {
            arity: 2,
            parameterized: false,
        }),
        // Controlled rotation (parameterized, arity 2)
        "crz" | "rzz" => Some(GateInfo {
            arity: 2,
            parameterized: true,
        }),
        // Three-qubit gates
        "ccx" => Some(GateInfo {
            arity: 3,
            parameterized: false,
        }),
        // Special operations (handled separately but recognized as gates)
        "mz" | "pz" => Some(GateInfo {
            arity: 1,
            parameterized: false,
        }),
        _ => None,
    }
}

/// Check if a name is a built-in gate.
/// All gate names must be lowercase.
fn is_gate_name(name: &str) -> bool {
    get_gate_info(name).is_some()
}

/// Check if a name is a built-in constant.
fn is_builtin_constant(name: &str) -> bool {
    matches!(name, "pi" | "tau" | "e")
}

/// Get the type of a built-in constant.
fn get_builtin_constant_type(name: &str) -> Type {
    match name {
        "pi" | "tau" | "e" => Type::A64,
        _ => Type::Unknown,
    }
}

/// Maximum valid bit width for integer types (matches Rust's i128/u128).
const MAX_INT_BITS: u16 = 128;

/// Validate that a bit width is valid (1-128).
fn is_valid_bit_width(bits: u16) -> bool {
    (1..=MAX_INT_BITS).contains(&bits)
}

/// Resolve a built-in type name to a Type (for comptime type values).
/// Returns Some(Type) if the name is a built-in type, None otherwise.
/// Returns None for invalid bit widths (e.g., u0, u9999).
fn resolve_builtin_type_name(name: &str) -> Option<Type> {
    // Special cases first
    match name {
        "bool" => return Some(Type::Bool),
        "usize" => return Some(Type::Usize),
        "isize" => return Some(Type::Isize),
        "f16" => return Some(Type::F16),
        "f32" => return Some(Type::F32),
        "f64" => return Some(Type::F64),
        "f128" => return Some(Type::F128),
        "a64" => return Some(Type::A64),
        "type" => return Some(Type::Type),
        "unit" => return Some(Type::Unit),
        "qubit" => return Some(Type::Qubit),
        "bit" => return Some(Type::Bit),
        _ => {}
    }

    // Arbitrary-width integers: u<bits> or i<bits>
    if let Some(bits_str) = name.strip_prefix('u') {
        if let Ok(bits) = bits_str.parse::<u16>() {
            if let Some(bw) = BitWidth::new(bits) {
                return Some(Type::UInt { bits: bw });
            }
            // Invalid bit width - return None to trigger error
            return None;
        }
    } else if let Some(bits_str) = name.strip_prefix('i')
        && let Ok(bits) = bits_str.parse::<u16>()
    {
        if let Some(bw) = BitWidth::new(bits) {
            return Some(Type::IInt { bits: bw });
        }
        // Invalid bit width - return None to trigger error
        return None;
    }

    None
}

/// Convert an integer type suffix to a Type.
/// Supports arbitrary bit-width integers: u1, u4, u7, u128, i32, etc.
fn int_suffix_to_type(suffix: &str) -> Type {
    // Handle optional underscore prefix
    let s = suffix.strip_prefix('_').unwrap_or(suffix);

    // Special cases
    match s {
        "usize" => return Type::Usize,
        "isize" => return Type::Isize,
        _ => {}
    }

    // Arbitrary-width integers: u<bits> or i<bits>
    if let Some(bits_str) = s.strip_prefix('u') {
        if let Ok(bits) = bits_str.parse::<u16>()
            && let Some(bw) = BitWidth::new(bits)
        {
            return Type::UInt { bits: bw };
        }
        // Invalid bit width - fall through to default
    } else if let Some(bits_str) = s.strip_prefix('i')
        && let Ok(bits) = bits_str.parse::<u16>()
        && let Some(bw) = BitWidth::new(bits)
    {
        return Type::IInt { bits: bw };
    }
    // Invalid bit width - fall through to default

    Type::IInt {
        bits: BitWidth::BITS_64,
    } // Default fallback
}

/// Convert a float type suffix to a Type.
fn float_suffix_to_type(suffix: &str) -> Type {
    // Handle optional underscore prefix
    let s = suffix.strip_prefix('_').unwrap_or(suffix);
    match s {
        "f16" => Type::F16,
        "f32" => Type::F32,
        "f64" => Type::F64,
        "f128" => Type::F128,
        "a64" => Type::A64,
        _ => Type::F64, // Default fallback
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;

    fn analyze(source: &str) -> SemanticResult<()> {
        let program = parse(source).expect("parse failed");
        let mut analyzer = SemanticAnalyzer::new();
        analyzer.analyze(&program)
    }

    #[test]
    fn test_const_declaration() {
        assert!(analyze("x: u32 = 42;").is_ok());
    }

    #[test]
    fn test_function_declaration() {
        assert!(analyze("fn add(a: u32, b: u32) -> u32 { return a + b; }").is_ok());
    }

    #[test]
    fn test_type_mismatch() {
        // First check that parsing works
        let program1 = parse("x: u32 = 42;").unwrap();
        assert!(
            !program1.declarations.is_empty(),
            "u32 version should have declarations"
        );

        let program2 = parse("x: bool = 42;").unwrap();
        assert!(
            !program2.declarations.is_empty(),
            "bool version should have declarations: got {:?}",
            program2
        );

        // bool is not numeric, so int can't be assigned
        let result = analyze("x: bool = 42;");
        assert!(result.is_err(), "Expected type mismatch error");
    }

    #[test]
    fn test_undefined_symbol() {
        // Note: "test" is a keyword, so use "run" instead
        // Also "y" is now a gate name, so use "foo" instead
        let result = analyze("fn run() -> unit { x := foo; }");
        if result.is_ok() {
            panic!("Expected undefined symbol error but got ok");
        }
    }

    #[test]
    fn test_quantum_alloc() {
        assert!(
            analyze(
                r#"
            fn main() -> unit {
                mut q := qalloc(2);

                return unit;            }
            "#
            )
            .is_ok()
        );
    }

    // =========================================================================
    // Qubit State Tracking Tests
    // =========================================================================

    fn analyze_strict(source: &str) -> SemanticResult<()> {
        // Note: new() is now strict by default, so this is equivalent to analyze()
        let program = parse(source).expect("parse failed");
        let mut analyzer = SemanticAnalyzer::new();
        analyzer.analyze(&program)
    }

    fn analyze_permissive(source: &str) -> SemanticResult<()> {
        let program = parse(source).expect("parse failed");
        let mut analyzer = SemanticAnalyzer::new_permissive();
        analyzer.analyze(&program)
    }

    #[test]
    fn test_allocator_info_new() {
        let alloc = AllocatorInfo::new("q", 4);
        assert_eq!(alloc.name, "q");
        assert_eq!(alloc.capacity, Some(4));
        assert_eq!(alloc.slot_states.len(), 4);
        assert!(
            alloc
                .slot_states
                .iter()
                .all(|s| *s == QubitState::Unprepared)
        );
    }

    #[test]
    fn test_allocator_info_prepare_slot() {
        let mut alloc = AllocatorInfo::new("q", 4);

        // Prepare slot 0
        assert!(alloc.prepare_slot(0).is_ok());
        assert_eq!(alloc.get_state(0), Some(QubitState::Prepared));
        assert_eq!(alloc.get_state(1), Some(QubitState::Unprepared));

        // Can't prepare already prepared slot
        assert!(alloc.prepare_slot(0).is_err());
    }

    #[test]
    fn test_allocator_info_prepare_all() {
        let mut alloc = AllocatorInfo::new("q", 4);
        alloc.prepare_all();

        assert!(alloc.slot_states.iter().all(|s| *s == QubitState::Prepared));
    }

    #[test]
    fn test_allocator_info_measure_slot() {
        let mut alloc = AllocatorInfo::new("q", 4);
        alloc.prepare_all();
        alloc.measure_slot(1);

        assert_eq!(alloc.get_state(0), Some(QubitState::Prepared));
        assert_eq!(alloc.get_state(1), Some(QubitState::Unprepared));
        assert_eq!(alloc.get_state(2), Some(QubitState::Prepared));
    }

    #[test]
    fn test_allocator_info_bounds() {
        let alloc = AllocatorInfo::new("q", 4);
        assert!(alloc.is_in_bounds(0));
        assert!(alloc.is_in_bounds(3));
        assert!(!alloc.is_in_bounds(4));
        assert!(!alloc.is_in_bounds(100));
    }

    #[test]
    fn test_qubit_state_tracker() {
        let mut tracker = QubitStateTracker::new();
        tracker.register_allocator(AllocatorInfo::new("q", 2));

        assert!(tracker.get_allocator("q").is_some());
        assert!(tracker.get_allocator("x").is_none());

        assert_eq!(tracker.is_prepared("q", 0), Some(false));

        if let Some(alloc) = tracker.get_allocator_mut("q") {
            alloc.prepare_all();
        }

        assert_eq!(tracker.is_prepared("q", 0), Some(true));
    }

    #[test]
    fn test_recursion_tracker() {
        let mut tracker = RecursionTracker::new();
        let loc = SourceLocation::default();

        // First call should succeed
        assert!(tracker.enter_function("foo", &loc).is_ok());
        assert!(tracker.is_in_call_stack("foo"));

        // Recursive call should fail
        let result = tracker.enter_function("foo", &loc);
        assert!(matches!(
            result,
            Err(SemanticError::RecursionDetected { .. })
        ));

        // Exit and re-enter should work
        tracker.exit_function("foo");
        assert!(!tracker.is_in_call_stack("foo"));
        assert!(tracker.enter_function("foo", &loc).is_ok());
    }

    #[test]
    fn test_loop_bound_check() {
        let analyzer = SemanticAnalyzer::new();
        let loc = SourceLocation::default();

        // Small bound should pass
        assert!(analyzer.check_loop_bound(100, &loc).is_ok());
        assert!(analyzer.check_loop_bound(MAX_LOOP_BOUND, &loc).is_ok());

        // Large bound should fail
        let result = analyzer.check_loop_bound(MAX_LOOP_BOUND + 1, &loc);
        assert!(matches!(
            result,
            Err(SemanticError::LoopBoundTooLarge { .. })
        ));
    }

    // =========================================================================
    // Semantic Analyzer Integration Tests
    // =========================================================================

    #[test]
    fn test_allocator_registration() {
        let source = r#"
            fn main() -> unit {
                mut q := qalloc(4);

                return unit;            }
        "#;

        let program = parse(source).expect("parse failed");
        let mut analyzer = SemanticAnalyzer::new();
        analyzer.analyze(&program).expect("analysis failed");

        // Check allocator was registered
        assert!(analyzer.qubit_states.get_allocator("q").is_some());
        let alloc = analyzer.qubit_states.get_allocator("q").unwrap();
        assert_eq!(alloc.capacity, Some(4));
    }

    #[test]
    fn test_child_allocator_registration() {
        let source = r#"
            fn main() -> unit {
                mut base := qalloc(8);
                mut q := base.child(4);

                return unit;            }
        "#;

        let program = parse(source).expect("parse failed");
        let mut analyzer = SemanticAnalyzer::new();
        analyzer.analyze(&program).expect("analysis failed");

        // Check both allocators were registered
        assert!(analyzer.qubit_states.get_allocator("base").is_some());
        assert!(analyzer.qubit_states.get_allocator("q").is_some());

        let child = analyzer.qubit_states.get_allocator("q").unwrap();
        assert_eq!(child.capacity, Some(4));
        assert_eq!(child.parent, Some("base".to_string()));
    }

    #[test]
    fn test_immutable_allocator_for_gates() {
        // Allocators don't need mut when just applying gates
        let source = r#"
            fn main() -> unit {
                q := qalloc(4);
                pz q;
                h q[0];
                cx (q[0], q[1]);
                return unit;
            }
        "#;

        let program = parse(source).expect("parse failed");
        let mut analyzer = SemanticAnalyzer::new();
        analyzer
            .analyze(&program)
            .expect("analysis should succeed for immutable allocator with gates");

        // Check allocator was registered
        assert!(analyzer.qubit_states.get_allocator("q").is_some());
    }

    #[test]
    fn test_child_requires_mutable_parent() {
        // .child() requires the parent to be mutable
        let source = r#"
            fn main() -> unit {
                base := qalloc(8);
                q := base.child(4);
                return unit;
            }
        "#;

        let program = parse(source).expect("parse failed");
        let mut analyzer = SemanticAnalyzer::new();
        let result = analyzer.analyze(&program);

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(&err, SemanticError::ChildRequiresMutableParent { name, .. } if name == "base"),
            "expected ChildRequiresMutableParent error for 'base', got {:?}",
            err
        );
    }

    #[test]
    fn test_strict_mode_toggle() {
        // new() is now strict by default
        let analyzer = SemanticAnalyzer::new();
        assert!(analyzer.strict_mode);

        // new_permissive() disables strict mode
        let permissive_analyzer = SemanticAnalyzer::new_permissive();
        assert!(!permissive_analyzer.strict_mode);

        // Can toggle strict mode off
        let mut toggled = SemanticAnalyzer::new();
        toggled.set_strict_mode(false);
        assert!(!toggled.strict_mode);
    }

    #[test]
    fn test_gate_on_unprepared_qubit_rejected_strict() {
        // In strict mode, gates on unprepared qubits should fail
        let result = analyze_strict(
            r#"
            fn main() -> unit {
                mut q := qalloc(4);
                h q[0];  // No pz q; first - qubit is unprepared
                return unit;
            }
            "#,
        );
        assert!(result.is_err(), "Expected QubitNotPrepared error");
        assert!(
            matches!(result.unwrap_err(), SemanticError::QubitNotPrepared { .. }),
            "Expected QubitNotPrepared error"
        );
    }

    #[test]
    fn test_pz_prepares_qubits() {
        // PZ (prepare Z) can be applied to unprepared qubits and prepares them
        let result = analyze_strict(
            r#"
            fn main() -> unit {
                mut q := qalloc(4);
                pz q;   // Prepare all qubits
                h q[0]; // Now this should succeed
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "PZ should prepare qubits for subsequent gates: {:?}",
            result
        );
    }

    #[test]
    fn test_pz_on_specific_qubit() {
        // PZ can prepare specific qubits
        let result = analyze_strict(
            r#"
            fn main() -> unit {
                mut q := qalloc(4);
                pz q[0];  // Prepare only q[0]
                h q[0];   // This should succeed
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "PZ should prepare specific qubit: {:?}",
            result
        );
    }

    // =========================================================================
    // Typed Measurement Tests
    // =========================================================================

    #[test]
    fn test_typed_measurement_single_qubit() {
        // Single qubit measurement: mz(u1) q[0]
        let result = analyze(
            r#"
            fn main() -> unit {
                q := qalloc(2);
                pz q;
                r := mz(u1) q[0];
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected typed measurement to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_typed_measurement_array() {
        // Array measurement with explicit size: mz([2]u1) [q[0], q[1]]
        let result = analyze(
            r#"
            fn main() -> unit {
                q := qalloc(4);
                pz q;
                results := mz([2]u1) [q[0], q[1]];
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected typed array measurement to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_typed_measurement_size_mismatch() {
        // Size mismatch: declared [3]u1 but only 2 targets
        let result = analyze(
            r#"
            fn main() -> unit {
                q := qalloc(4);
                pz q;
                results := mz([3]u1) [q[0], q[1]];
                return unit;
            }
            "#,
        );
        assert!(
            matches!(result, Err(SemanticError::MeasurementSizeMismatch { .. })),
            "Expected MeasurementSizeMismatch error, got: {:?}",
            result
        );
    }

    #[test]
    fn test_typed_measurement_scalar_with_multiple_targets() {
        // Scalar type with multiple targets should fail
        let result = analyze(
            r#"
            fn main() -> unit {
                q := qalloc(4);
                pz q;
                results := mz(u1) [q[0], q[1]];
                return unit;
            }
            "#,
        );
        assert!(
            matches!(result, Err(SemanticError::MeasurementArrayExpected { .. })),
            "Expected MeasurementArrayExpected error, got: {:?}",
            result
        );
    }

    #[test]
    fn test_typed_measurement_missing_args() {
        // Old call syntax should be rejected
        let result = analyze(
            r#"
            fn main() -> unit {
                q := qalloc(2);
                r := mz();
                return unit;
            }
            "#,
        );
        assert!(
            matches!(
                result,
                Err(SemanticError::DeprecatedMeasurementSyntax { .. })
            ),
            "Expected DeprecatedMeasurementSyntax error for old mz() call syntax"
        );
    }

    #[test]
    fn test_typed_measurement_invalid_type() {
        // Invalid type (f64) should fail
        let result = analyze(
            r#"
            fn main() -> unit {
                q := qalloc(2);
                pz q;
                r := mz(f64) q[0];
                return unit;
            }
            "#,
        );
        assert!(
            matches!(result, Err(SemanticError::InvalidMeasurementType { .. })),
            "Expected InvalidMeasurementType error, got: {:?}",
            result
        );
    }

    #[test]
    fn test_measurement_pack_u8() {
        // Pack 8 qubits into u8
        let result = analyze(
            r#"
            fn main() -> unit {
                q := qalloc(8);
                pz q;
                bits := mz(pack u8) [q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]];
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected pack measurement to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_measurement_pack_fewer_qubits() {
        // Pack 4 qubits into u8 (has room for 8)
        let result = analyze(
            r#"
            fn main() -> unit {
                q := qalloc(4);
                pz q;
                bits := mz(pack u8) [q[0], q[1], q[2], q[3]];
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected pack with extra capacity to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_measurement_pack_capacity_error() {
        // Try to pack 10 qubits into u8 (only 8 bits) - should fail
        let result = analyze(
            r#"
            fn main() -> unit {
                q := qalloc(10);
                pz q;
                bits := mz(pack u8) [q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9]];
                return unit;
            }
            "#,
        );
        assert!(
            matches!(result, Err(SemanticError::MeasurementPackCapacity { .. })),
            "Expected MeasurementPackCapacity error, got: {:?}",
            result
        );
    }

    #[test]
    fn test_measurement_pack_array() {
        // Pack 16 qubits into [2]u8
        let result = analyze(
            r#"
            fn main() -> unit {
                q := qalloc(16);
                pz q;
                bits := mz(pack [2]u8) q;
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected pack into array to pass: {:?}",
            result
        );
    }

    // =========================================================================
    // Optional Type Tests
    // =========================================================================

    #[test]
    fn test_optional_type_declaration() {
        // Optional type declaration
        let result = analyze(
            r#"
            fn main() -> unit {
                mut x: ?u32 = none;

                return unit;            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected optional type to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_orelse_operator() {
        // orelse operator: ?T orelse T -> T
        let result = analyze(
            r#"
            fn main() -> unit {
                mut x: ?u32 = none;
                y: u32 = x orelse 42;

                return unit;            }
            "#,
        );
        assert!(result.is_ok(), "Expected orelse to pass: {:?}", result);
    }

    #[test]
    fn test_orelse_type_mismatch() {
        // orelse with wrong default type should fail
        let result = analyze(
            r#"
            fn main() -> unit {
                mut x: ?u32 = none;
                y := x orelse true;

                return unit;            }
            "#,
        );
        assert!(
            matches!(result, Err(SemanticError::TypeMismatch { .. })),
            "Expected TypeMismatch error for orelse"
        );
    }

    #[test]
    fn test_orelse_non_optional() {
        // orelse on non-optional should fail
        let result = analyze(
            r#"
            fn main() -> unit {
                mut x: u32 = 10;
                y := x orelse 42;

                return unit;            }
            "#,
        );
        assert!(
            matches!(result, Err(SemanticError::TypeMismatch { .. })),
            "Expected TypeMismatch error for non-optional"
        );
    }

    #[test]
    fn test_optional_unwrap() {
        // .? unwrap operator
        let result = analyze(
            r#"
            fn main() -> unit {
                mut x: ?u32 = none;
                y := x.?;

                return unit;            }
            "#,
        );
        assert!(result.is_ok(), "Expected .? unwrap to pass: {:?}", result);
    }

    #[test]
    fn test_if_unwrap_optional() {
        // if-unwrap syntax: if value := opt { ... } (walrus operator)
        let result = analyze(
            r#"
            fn find() -> ?u32 {
                return none;
            }
            fn main() -> unit {
                opt := find();
                if value := opt {
                    x: u32 = value;
                }
                return unit;
            }
            "#,
        );
        assert!(result.is_ok(), "Expected if-unwrap to pass: {:?}", result);
    }

    #[test]
    fn test_if_unwrap_non_optional() {
        // if-unwrap on non-optional should fail
        let result = analyze(
            r#"
            fn main() -> unit {
                x: u32 = 42;
                if value := x {
                    y := value;
                }
            }
            "#,
        );
        assert!(
            matches!(result, Err(SemanticError::TypeMismatch { .. })),
            "Expected TypeMismatch error for if-unwrap on non-optional"
        );
    }

    // =========================================================================
    // Comptime Tests
    // =========================================================================

    #[test]
    fn test_comptime_expression() {
        // Comptime expression should be evaluated
        let result = analyze(
            r#"
            fn main() -> unit {
                x := comptime 2 + 3;

                return unit;            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected comptime expression to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_comptime_block() {
        // Comptime block expression should be evaluated
        // Note: We test a simpler block expression to avoid parser edge cases
        let result = analyze(
            r#"
            fn main() -> unit {
                y := comptime (10 + 20);

                return unit;            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected comptime expression to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_comptime_type_returns_comptime() {
        // Comptime expression should have Comptime type wrapper
        let source = r#"
            fn main() -> unit {
                x := comptime 42;

                return unit;            }
        "#;
        let program = crate::parse(source).expect("parse failed");
        let mut analyzer = SemanticAnalyzer::new();
        let result = analyzer.analyze(&program);
        assert!(result.is_ok(), "Expected comptime to pass: {:?}", result);
    }

    #[test]
    fn test_comptime_function_parameter() {
        // Comptime parameters should be accepted
        let result = analyze(
            r#"
            fn make_array(comptime size: u32) -> unit {
                mut x: u32 = size;
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected comptime param to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_comptime_parameter_in_expression() {
        // Comptime parameters should be usable in expressions
        let result = analyze(
            r#"
            fn compute(comptime n: u32) -> u32 {
                return n * 2;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected comptime param in expr: {:?}",
            result
        );
    }

    #[test]
    fn test_error_set_definition() {
        // Error set definitions should be analyzed correctly
        let result = analyze(
            r#"
            MyError := error {
                OutOfMemory,
                InvalidInput,
            };
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected error set definition to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_error_union_type() {
        // Error union type should work in function signatures
        let result = analyze(
            r#"
            MyError := error { Failed };

            fn risky() -> MyError!u32 {
                x: u32 = 42;
                return x;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected error union type to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_error_value_type_tracking() {
        // Error values from defined error sets should have correct type
        let result = analyze(
            r#"
            MyError := error { OutOfMemory, InvalidInput };

            fn risky() -> MyError!u32 {
                if true {
                    return error.OutOfMemory;
                }
                return 42;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected error value return to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_error_value_wrong_set() {
        // Error values from different error sets should not be assignable
        let result = analyze(
            r#"
            MyError := error { Failed };
            OtherError := error { NotFound };

            fn risky() -> MyError!u32 {
                return error.NotFound;
            }
            "#,
        );
        // This should fail because NotFound is from OtherError, not MyError
        assert!(result.is_err(), "Expected mismatched error set to fail");
    }

    #[test]
    fn test_error_set_union() {
        // Error set union using | operator should combine errors from both sets
        let result = analyze(
            r#"
            IoError := error { FileNotFound, PermissionDenied };
            NetworkError := error { Timeout, ConnectionRefused };

            fn combined_errors() -> unit {
                combined := IoError | NetworkError;
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected error set union to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_error_set_with_associated_data() {
        // Error sets can have associated data types for their variants
        let result = analyze(
            r#"
            FileError := error {
                NotFound: struct { path: []u8 },
                PermissionDenied: struct { path: []u8, mode: u32 },
                IoError,
            };
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected error set with associated data to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_fault_set_with_associated_data() {
        // Fault sets can also have associated data types
        let result = analyze(
            r#"
            QuantumFault := fault {
                Leakage: struct { qubit_id: u32, gate: []u8 },
                BitFlip,
                PhaseError: struct { angle: f64 },
            };
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected fault set with associated data to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_fault_set_definition_basic() {
        // Basic fault set definition should work
        let result = analyze("GateFaults := fault { Leakage, Depolarization };");
        assert!(
            result.is_ok(),
            "Expected fault set definition to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_fault_set_as_value() {
        // Fault set should be usable as a value (like error sets)
        let result = analyze(
            r#"
            GateFaults := fault { Leakage };

            fn test_fault() -> unit {
                x := GateFaults;
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected fault set as value to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_fault_set_union() {
        // Fault set union using | operator should combine faults from both sets
        let result = analyze(
            r#"
            GateFaults := fault { Leakage, Depolarization };
            MeasurementFaults := fault { BitFlip, ReadoutError };

            fn combined_faults() -> unit {
                combined := GateFaults | MeasurementFaults;
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected fault set union to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_try_block_collect_mode() {
        // try {} (collect mode) should analyze correctly
        let result = analyze(
            r#"
            fn risky_collect() -> unit {
                errors := try {
                    x := 42;
                };
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected try collect block to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_try_block_propagate_mode() {
        // try! {} (propagate mode) should analyze correctly
        let result = analyze(
            r#"
            fn risky_propagate() -> unit {
                result := try! {
                    x := 42;
                };
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected try! propagate block to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_try_block_with_catch() {
        // try! {} with catch clause should analyze correctly
        let result = analyze(
            r#"
            fn risky_with_catch() -> unit {
                result := try! {
                    x := 42;
                    x
                } catch |err| {
                    0
                };
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected try! with catch to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_errdefer_basic() {
        // Basic errdefer without capture should pass semantic analysis
        let result = analyze(
            r#"
            fn risky() -> unit {
                errdefer cleanup();
                return unit;
            }

            fn cleanup() -> unit { return unit; }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected basic errdefer to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_errdefer_with_capture() {
        // Errdefer with capture should provide the error variable in scope
        let result = analyze(
            r#"
            fn risky() -> unit {
                errdefer |err| {
                    x := err;
                }
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected errdefer with capture to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_union_tagged() {
        // Tagged union with auto-enum should pass semantic analysis
        let result = analyze(
            r#"
            Value := union(enum) {
                Int: i32,
                Float: f64,
                None,
            };
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected tagged union to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_union_untagged() {
        // Untagged union should pass semantic analysis
        let result = analyze(
            r#"
            RawValue := union {
                Int: i32,
                Float: f64,
            };
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected untagged union to pass: {:?}",
            result
        );
    }

    // =========================================================================
    // Module Import Tests
    // =========================================================================

    #[test]
    fn test_module_import_not_found() {
        // Importing a non-existent module should fail
        let result = analyze(r#"utils := @import("nonexistent.zlp");"#);
        assert!(result.is_err(), "Expected module not found error");
        if let Err(e) = result {
            assert!(
                matches!(e, SemanticError::ModuleError { .. }),
                "Expected ModuleError, got {:?}",
                e
            );
        }
    }

    #[test]
    fn test_module_import_with_file() {
        use std::io::Write;
        use tempfile::TempDir;

        // Create a temporary module file
        let temp_dir = TempDir::new().unwrap();
        let utils_path = temp_dir.path().join("utils.zlp");
        let mut file = std::fs::File::create(&utils_path).unwrap();
        writeln!(file, "pub fn helper() -> unit {{ return unit; }}").unwrap();
        writeln!(file, "pub VALUE: u32 = 42;").unwrap();

        // Create main file that imports utils
        let main_path = temp_dir.path().join("main.zlp");
        let mut file = std::fs::File::create(&main_path).unwrap();
        writeln!(file, "utils := @import(\"utils.zlp\");").unwrap();
        writeln!(file, "fn main() -> unit {{ return unit; }}").unwrap();

        // Parse and analyze main file
        let source = std::fs::read_to_string(&main_path).unwrap();
        let program = crate::parse_file(&source, main_path.display().to_string()).unwrap();
        let mut analyzer = SemanticAnalyzer::new();
        analyzer.set_current_file(&main_path);
        let result = analyzer.analyze(&program);
        assert!(result.is_ok(), "Expected import to succeed: {:?}", result);
    }

    #[test]
    fn test_module_import_call_function() {
        use std::io::Write;
        use tempfile::TempDir;

        // Create a temporary module file with a function
        let temp_dir = TempDir::new().unwrap();
        let utils_path = temp_dir.path().join("utils.zlp");
        let mut file = std::fs::File::create(&utils_path).unwrap();
        writeln!(
            file,
            "pub fn add(a: u32, b: u32) -> u32 {{ return a + b; }}"
        )
        .unwrap();

        // Create main file that imports and calls the function
        let main_path = temp_dir.path().join("main.zlp");
        let mut file = std::fs::File::create(&main_path).unwrap();
        writeln!(file, "utils := @import(\"utils.zlp\");").unwrap();
        writeln!(file, "fn main() -> unit {{").unwrap();
        writeln!(file, "    result := utils.add(1, 2);").unwrap();
        writeln!(file, "    return unit;").unwrap();
        writeln!(file, "}}").unwrap();

        // Parse and analyze main file
        let source = std::fs::read_to_string(&main_path).unwrap();
        let program = crate::parse_file(&source, main_path.display().to_string()).unwrap();
        let mut analyzer = SemanticAnalyzer::new();
        analyzer.set_current_file(&main_path);
        let result = analyzer.analyze(&program);
        assert!(
            result.is_ok(),
            "Expected module function call to succeed: {:?}",
            result
        );
    }

    #[test]
    fn test_module_import_wrong_arg_count() {
        use std::io::Write;
        use tempfile::TempDir;

        // Create a temporary module file with a function
        let temp_dir = TempDir::new().unwrap();
        let utils_path = temp_dir.path().join("utils.zlp");
        let mut file = std::fs::File::create(&utils_path).unwrap();
        writeln!(
            file,
            "pub fn add(a: u32, b: u32) -> u32 {{ return a + b; }}"
        )
        .unwrap();

        // Create main file that calls with wrong number of arguments
        let main_path = temp_dir.path().join("main.zlp");
        let mut file = std::fs::File::create(&main_path).unwrap();
        writeln!(file, "utils := @import(\"utils.zlp\");").unwrap();
        writeln!(file, "fn main() -> unit {{").unwrap();
        writeln!(file, "    result := utils.add(1);").unwrap(); // Missing second arg
        writeln!(file, "    return unit;").unwrap();
        writeln!(file, "}}").unwrap();

        // Parse and analyze main file
        let source = std::fs::read_to_string(&main_path).unwrap();
        let program = crate::parse_file(&source, main_path.display().to_string()).unwrap();
        let mut analyzer = SemanticAnalyzer::new();
        analyzer.set_current_file(&main_path);
        let result = analyzer.analyze(&program);
        // This should fail due to argument count mismatch
        assert!(result.is_err(), "Expected error for wrong argument count");
    }

    // =========================================================================
    // Tuple Type Tests
    // =========================================================================

    #[test]
    fn test_tuple_type_inference() {
        // Tuple literal should have correct type
        let result = analyze(
            r#"
            fn main() -> unit {
                pair := (1, 2);

                return unit;            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected tuple literal to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_tuple_mixed_types() {
        // Tuple can contain mixed types
        let result = analyze(
            r#"
            fn main() -> unit {
                mixed := (42, true);

                return unit;            }
            "#,
        );
        assert!(result.is_ok(), "Expected mixed tuple to pass: {:?}", result);
    }

    #[test]
    fn test_tuple_with_qubits() {
        // Tuple of qubit references (for two-qubit gates)
        let result = analyze(
            r#"
            fn main() -> unit {
                mut q := qalloc(4);
                pz q;
                cx (q[0], q[1]);

                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected tuple with qubits to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_tuple_type_annotation() {
        // Tuple with explicit type annotation (i64 is default int type)
        let result = analyze(
            r#"
            fn main() -> unit {
                mut pair: (i64, bool) = (42, true);

                return unit;            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected tuple type annotation to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_tuple_type_annotation_triple() {
        // Triple tuple with explicit type annotation
        let result = analyze(
            r#"
            fn main() -> unit {
                triple: (i64, i64, i64) = (1, 2, 3);

                return unit;            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected triple tuple type to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_tuple_type_mismatch() {
        // Tuple type mismatch should fail
        let result = analyze(
            r#"
            fn main() -> unit {
                mut pair: (u32, bool) = (42, true);

                return unit;            }
            "#,
        );
        // Should fail because 42 is i64, not u32
        assert!(result.is_err(), "Expected tuple type mismatch error");
    }

    // =========================================================================
    // Numeric Type Suffix Tests
    // =========================================================================

    #[test]
    fn test_int_suffix_u32() {
        // Integer with u32 suffix should have u32 type
        let result = analyze(
            r#"
            fn main() -> unit {
                x: u32 = 42u32;

                return unit;            }
            "#,
        );
        assert!(result.is_ok(), "Expected u32 suffix to pass: {:?}", result);
    }

    #[test]
    fn test_int_suffix_with_underscore() {
        // Integer with underscore separator before suffix
        let result = analyze(
            r#"
            fn main() -> unit {
                x: u64 = 1000_u64;

                return unit;            }
            "#,
        );
        assert!(result.is_ok(), "Expected _u64 suffix to pass: {:?}", result);
    }

    #[test]
    fn test_float_suffix_f32() {
        // Float with f32 suffix
        let result = analyze(
            r#"
            fn main() -> unit {
                x: f32 = 3.14f32;

                return unit;            }
            "#,
        );
        assert!(result.is_ok(), "Expected f32 suffix to pass: {:?}", result);
    }

    #[test]
    fn test_suffix_tuple_match() {
        // Suffixes allow tuple type to match
        let result = analyze(
            r#"
            fn main() -> unit {
                pair: (u32, bool) = (42u32, true);

                return unit;            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected suffixed tuple to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_suffix_mismatch() {
        // Suffix type should not match different declared type
        // bool cannot be assigned to an integer type
        let result = analyze(
            r#"
            fn main() -> unit {
                x: u32 = true;

                return unit;            }
            "#,
        );
        assert!(result.is_err(), "Expected type mismatch with bool and u32");
    }

    #[test]
    fn test_suffix_default_type_vs_suffix() {
        // Without suffix, 42 is i64 which won't match (u32, bool) tuple
        // With suffix, 42u32 matches (u32, bool) tuple
        let result_no_suffix = analyze(
            r#"
            fn main() -> unit {
                pair: (u32, bool) = (42, true);

                return unit;            }
            "#,
        );
        let result_with_suffix = analyze(
            r#"
            fn main() -> unit {
                pair: (u32, bool) = (42u32, true);

                return unit;            }
            "#,
        );
        // Tuple elements require exact type match (no numeric coercion)
        // Unsuffixed 42 is i64, so (i64, bool) doesn't match (u32, bool)
        assert!(
            result_no_suffix.is_err(),
            "Expected unsuffixed tuple to fail type check"
        );
        // With suffix, types match exactly
        assert!(
            result_with_suffix.is_ok(),
            "Expected suffixed tuple to pass: {:?}",
            result_with_suffix
        );
    }

    // =========================================================================
    // Tick Block Duplicate Qubit Tests
    // =========================================================================

    #[test]
    fn test_tick_no_duplicate_qubits() {
        // Valid: different qubits in parallel
        let result = analyze_strict(
            r#"
            fn main() -> unit {
                mut q := qalloc(4);
                pz q;
                tick {
                    h q[0];
                    h q[1];
                }
                return unit;
            }
            "#,
        );
        assert!(result.is_ok(), "Expected no duplicate error: {:?}", result);
    }

    #[test]
    fn test_tick_duplicate_qubit_same_gate() {
        // Invalid: same qubit used twice in parallel
        let result = analyze_strict(
            r#"
            fn main() -> unit {
                mut q := qalloc(4);
                pz q;
                tick {
                    h q[0];
                    x q[0];
                }
            }
            "#,
        );
        assert!(
            matches!(result, Err(SemanticError::DuplicateQubitInTick { ref allocator, index, .. }) if allocator == "q" && index == 0),
            "Expected DuplicateQubitInTick error for q[0]: {:?}",
            result
        );
    }

    #[test]
    fn test_tick_duplicate_in_two_qubit_gate() {
        // Invalid: qubit used in both h and cx
        let result = analyze_strict(
            r#"
            fn main() -> unit {
                mut q := qalloc(4);
                pz q;
                tick {
                    h q[0];
                    cx (q[0], q[1]);
                }
            }
            "#,
        );
        assert!(
            matches!(result, Err(SemanticError::DuplicateQubitInTick { .. })),
            "Expected DuplicateQubitInTick error: {:?}",
            result
        );
    }

    #[test]
    fn test_tick_no_duplicate_permissive() {
        // In permissive mode, duplicates are allowed (no error)
        let result = analyze_permissive(
            r#"
            fn main() -> unit {
                mut q := qalloc(4);
                pz q;
                tick {
                    h q[0];
                    x q[0];
                }
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Permissive mode should allow duplicate qubits: {:?}",
            result
        );
    }

    #[test]
    fn test_nested_tick_error() {
        // Invalid: nested tick blocks
        let result = analyze(
            r#"
            fn main() -> unit {
                q := qalloc(2);
                pz q;
                tick {
                    h q[0];
                    tick {
                        x q[1];
                    }
                }
                return unit;
            }
            "#,
        );
        assert!(result.is_err(), "Expected nested tick error");
        let err = result.unwrap_err();
        assert!(
            matches!(err, SemanticError::NestedTick { .. }),
            "Expected NestedTick error, got: {:?}",
            err
        );
    }

    #[test]
    fn test_sequential_ticks_ok() {
        // Valid: sequential tick blocks (not nested)
        let result = analyze(
            r#"
            fn main() -> unit {
                q := qalloc(2);
                pz q;
                tick { h q[0]; }
                tick { h q[1]; }
                tick { cx (q[0], q[1]); }
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Sequential ticks should be valid: {:?}",
            result
        );
    }

    // =========================================================================
    // Break/Continue Validation Tests
    // =========================================================================

    #[test]
    fn test_break_inside_loop() {
        // Valid: break inside a for loop
        let result = analyze(
            r#"
            fn main() -> unit {
                for i in 0..10 {
                    if (i == 5) {
                        break;
                    }
                }
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected break inside loop to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_continue_inside_loop() {
        // Valid: continue inside a for loop
        let result = analyze(
            r#"
            fn main() -> unit {
                for i in 0..10 {
                    if (i == 5) {
                        continue;
                    }
                }
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected continue inside loop to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_break_outside_loop() {
        // Invalid: break outside of any loop
        let result = analyze(
            r#"
            fn main() -> unit {
                break;

                return unit;            }
            "#,
        );
        assert!(
            matches!(result, Err(SemanticError::BreakContinueOutsideLoop { ref keyword, .. }) if keyword == "break"),
            "Expected BreakContinueOutsideLoop error for break: {:?}",
            result
        );
    }

    #[test]
    fn test_continue_outside_loop() {
        // Invalid: continue outside of any loop
        let result = analyze(
            r#"
            fn main() -> unit {
                continue;

                return unit;            }
            "#,
        );
        assert!(
            matches!(result, Err(SemanticError::BreakContinueOutsideLoop { ref keyword, .. }) if keyword == "continue"),
            "Expected BreakContinueOutsideLoop error for continue: {:?}",
            result
        );
    }

    #[test]
    fn test_break_in_nested_loop() {
        // Valid: break inside nested loops
        let result = analyze(
            r#"
            fn main() -> unit {
                for i in 0..10 {
                    for j in 0..10 {
                        if (j == 5) {
                            break;
                        }
                    }
                }
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected break in nested loop to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_for_loop_range_type_inference_default() {
        // For loop with plain integer literals defaults to i64
        let result = analyze(
            r#"
            fn main() -> unit {
                for i in 0..10 {
                    x: i64 = i;  // Should work since i is i64
                }
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected for loop with i64 range to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_for_loop_range_type_inference_u32() {
        // For loop with u32 suffix should infer u32 loop variable
        let result = analyze(
            r#"
            fn main() -> unit {
                for i in 0u32..10u32 {
                    x: u32 = i;  // Should work since i is u32
                }
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected for loop with u32 range to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_for_loop_range_type_inference_usize() {
        // For loop with usize suffix should infer usize loop variable
        let result = analyze(
            r#"
            fn main() -> unit {
                for i in 0_usize..10_usize {
                    x: usize = i;  // Should work since i is usize
                }
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected for loop with usize range to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_array_type_size_literal() {
        // Array type with literal size should analyze correctly
        let result = analyze(
            r#"
            fn main() -> unit {
                mut arr: [10]u32 = undefined;

                return unit;            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected array with literal size to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_array_type_size_hex_literal() {
        // Array type with hex literal size should analyze correctly
        let result = analyze(
            r#"
            fn main() -> unit {
                mut arr: [0x10]u8 = undefined;

                return unit;            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected array with hex size to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_array_slice_type() {
        // Array slice (no size) should analyze correctly
        let result = analyze(
            r#"
            fn get_slice() -> []u8 {
                return undefined;
            }
            "#,
        );
        assert!(result.is_ok(), "Expected slice type to pass: {:?}", result);
    }

    #[test]
    fn test_const_propagation_array_size() {
        // Const propagation: use const value as array size
        let result = analyze(
            r#"
            fn main() -> unit {
                N := 10;
                mut arr: [N]u32 = undefined;

                return unit;            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected const propagation for array size to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_const_propagation_array_size_with_type() {
        // Const propagation with explicit type annotation
        let result = analyze(
            r#"
            fn main() -> unit {
                SIZE: usize = 5;
                mut buffer: [SIZE]u8 = undefined;

                return unit;            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected typed const propagation to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_const_propagation_chained() {
        // Chained propagation: const derived from another const
        let result = analyze(
            r#"
            fn main() -> unit {
                A := 4;
                B := A;
                mut arr: [B]u32 = undefined;

                return unit;            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected chained const propagation to pass: {:?}",
            result
        );
    }

    // =========================================================================
    // Array and Slice Property Tests
    // =========================================================================

    #[test]
    fn test_array_len_property() {
        // Array .len returns the compile-time known length
        let result = analyze(
            r#"
            fn main() -> unit {
                arr: [10]u32 = undefined;
                len := arr.len;

                return unit;
            }
            "#,
        );
        assert!(result.is_ok(), "Expected array .len to pass: {:?}", result);
    }

    #[test]
    fn test_array_ptr_property() {
        // Array .ptr returns a pointer to the first element
        let result = analyze(
            r#"
            fn main() -> unit {
                arr: [10]u32 = undefined;
                ptr := arr.ptr;

                return unit;
            }
            "#,
        );
        assert!(result.is_ok(), "Expected array .ptr to pass: {:?}", result);
    }

    #[test]
    fn test_slice_len_property() {
        // Slice .len returns the dynamic length
        let result = analyze(
            r#"
            fn process(data: []u32) -> unit {
                len := data.len;

                return unit;
            }
            "#,
        );
        assert!(result.is_ok(), "Expected slice .len to pass: {:?}", result);
    }

    #[test]
    fn test_slice_ptr_property() {
        // Slice .ptr returns a pointer to the first element
        let result = analyze(
            r#"
            fn process(data: []u32) -> unit {
                ptr := data.ptr;

                return unit;
            }
            "#,
        );
        assert!(result.is_ok(), "Expected slice .ptr to pass: {:?}", result);
    }

    // =========================================================================
    // Gate Syntax Tests
    // =========================================================================

    #[test]
    fn test_valid_gate_syntax() {
        // Correct gate syntax: h q[0], cx (q[0], q[1])
        let result = analyze(
            r#"
            fn main() -> unit {
                q := qalloc(2);
                pz q;
                h q[0];
                cx (q[0], q[1]);
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected valid gate syntax to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_qubit_index_out_of_bounds() {
        // Access q[5] when allocator only has 2 qubits
        let result = analyze(
            r#"
            fn main() -> unit {
                q := qalloc(2);
                pz q;
                h q[5];
                return unit;
            }
            "#,
        );
        assert!(
            matches!(result, Err(SemanticError::QubitIndexOutOfBounds { ref allocator, index: 5, capacity: 2, .. }) if allocator == "q"),
            "Expected QubitIndexOutOfBounds error, got: {:?}",
            result
        );
    }

    #[test]
    fn test_qubit_index_at_boundary() {
        // Access q[1] when allocator has 2 qubits (valid, 0-indexed)
        let result = analyze(
            r#"
            fn main() -> unit {
                q := qalloc(2);
                pz q;
                h q[1];
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected index at boundary to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_qubit_index_exactly_at_capacity() {
        // Access q[2] when allocator has 2 qubits (invalid, 0-indexed means max is 1)
        let result = analyze(
            r#"
            fn main() -> unit {
                q := qalloc(2);
                pz q;
                h q[2];
                return unit;
            }
            "#,
        );
        assert!(
            matches!(
                result,
                Err(SemanticError::QubitIndexOutOfBounds {
                    index: 2,
                    capacity: 2,
                    ..
                })
            ),
            "Expected QubitIndexOutOfBounds error for index at capacity, got: {:?}",
            result
        );
    }

    #[test]
    fn test_measurement_index_out_of_bounds() {
        // Measurement should also catch out-of-bounds
        let result = analyze(
            r#"
            fn main() -> unit {
                q := qalloc(2);
                pz q;
                r := mz(u1) q[5];
                return unit;
            }
            "#,
        );
        assert!(
            matches!(result, Err(SemanticError::QubitIndexOutOfBounds { ref allocator, index: 5, capacity: 2, .. }) if allocator == "q"),
            "Expected QubitIndexOutOfBounds error for measurement, got: {:?}",
            result
        );
    }

    #[test]
    fn test_measurement_array_index_out_of_bounds() {
        // Array measurement with one out-of-bounds index
        let result = analyze(
            r#"
            fn main() -> unit {
                q := qalloc(2);
                pz q;
                r := mz([2]u1) [q[0], q[5]];
                return unit;
            }
            "#,
        );
        assert!(
            matches!(result, Err(SemanticError::QubitIndexOutOfBounds { ref allocator, index: 5, capacity: 2, .. }) if allocator == "q"),
            "Expected QubitIndexOutOfBounds error for array measurement, got: {:?}",
            result
        );
    }

    #[test]
    fn test_pz_index_out_of_bounds() {
        // Prepare specific slots with out-of-bounds index
        let result = analyze(
            r#"
            fn main() -> unit {
                q := qalloc(2);
                pz {q[0], q[5]};
                return unit;
            }
            "#,
        );
        assert!(
            matches!(result, Err(SemanticError::QubitIndexOutOfBounds { ref allocator, index: 5, capacity: 2, .. }) if allocator == "q"),
            "Expected QubitIndexOutOfBounds error for pz, got: {:?}",
            result
        );
    }

    #[test]
    fn test_paren_gate_syntax_valid() {
        // h(q[0]) is now valid - parentheses are just grouping, equivalent to h q[0]
        // This is more intuitive and consistent with how most languages work
        let result = analyze(
            r#"
            fn main() -> unit {
                q := qalloc(1);
                pz q;
                h(q[0]);
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "h(q[0]) should be valid (parens are grouping): {:?}",
            result
        );
    }

    #[test]
    fn test_cx_tuple_syntax_valid() {
        // cx(q[0], q[1]) is valid - tuple target for two-qubit gate
        let result = analyze(
            r#"
            fn main() -> unit {
                q := qalloc(2);
                pz q;
                cx(q[0], q[1]);
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "cx(q[0], q[1]) should be valid: {:?}",
            result
        );
    }

    #[test]
    fn test_rx_without_angle_error() {
        // rx is a parameterized gate, it REQUIRES an angle parameter
        // rx q[0] without angle should fail at parse time
        let source = r#"
            fn main() -> unit {
                q := qalloc(1);
                pz q;
                rx q[0];
                return unit;
            }
        "#;
        let result = parse(source);
        assert!(
            result.is_err(),
            "rx without angle should fail at parse time"
        );
    }

    // =========================================================================
    // Type Ascription Tests (space-separated: `42 u32`, `1/4 f64`)
    // =========================================================================

    #[test]
    fn test_type_ascription_simple() {
        // Type ascription with space: `42 u32`
        let result = analyze(
            r#"
            fn main() -> unit {
                x: u32 = 42 u32;
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected type ascription `42 u32` to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_type_ascription_float() {
        // Type ascription with float: `3.14 f64`
        let result = analyze(
            r#"
            fn main() -> unit {
                x: f64 = 3.14 f64;
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected type ascription `3.14 f64` to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_type_ascription_expression() {
        // Type ascription on expression: `1/4 f64` should be 0.25
        let result = analyze(
            r#"
            fn main() -> unit {
                x: f64 = 1/4 f64;
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected type ascription `1/4 f64` to pass: {:?}",
            result
        );
    }

    // =========================================================================
    // Angle Literal Tests (`0.25 turns`, `pi/4 rad`)
    // =========================================================================

    #[test]
    fn test_angle_literal_turns() {
        // Angle literal with turns unit
        let result = analyze(
            r#"
            fn main() -> unit {
                angle: a64 = 0.25 turns;
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected angle literal `0.25 turns` to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_angle_literal_half_turn() {
        // Half turn
        let result = analyze(
            r#"
            fn main() -> unit {
                angle: a64 = 0.5 turns;
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected angle literal `0.5 turns` to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_angle_literal_type_is_a64() {
        // Angle literals should be type a64
        let result = analyze(
            r#"
            fn main() -> unit {
                angle := 0.25 turns;
                check: a64 = angle;
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected angle literal to be a64 type: {:?}",
            result
        );
    }

    // =========================================================================
    // Error Reporting Tests - Ensure errors are not silently swallowed
    // =========================================================================

    #[test]
    fn test_undefined_symbol_in_binding_reports_error() {
        // Undefined symbols in bindings should report an error, not silently use Type::Unknown
        let result = analyze("x := undefined_symbol;");
        assert!(result.is_err(), "Expected undefined symbol error");
        if let Err(e) = result {
            assert!(
                matches!(e, SemanticError::UndefinedSymbol { .. }),
                "Expected UndefinedSymbol, got {:?}",
                e
            );
        }
    }

    #[test]
    fn test_undefined_symbol_in_struct_init_reports_error() {
        // Undefined symbols in struct field values should report an error
        let result = analyze(
            r#"
            fn main() -> unit {
                s := .{ x: undefined_symbol };
                return unit;
            }
            "#,
        );
        assert!(
            result.is_err(),
            "Expected undefined symbol error in struct init"
        );
        if let Err(e) = result {
            assert!(
                matches!(e, SemanticError::UndefinedSymbol { .. }),
                "Expected UndefinedSymbol, got {:?}",
                e
            );
        }
    }

    #[test]
    fn test_direct_recursion_rejected() {
        // Direct recursion should be rejected in strict mode (NASA Power of 10 compliance)
        let result = analyze_strict(
            r#"
            fn factorial(n: u32) -> u32 {
                if n == 0 {
                    return 1;
                }
                return n * factorial(n - 1);
            }
            "#,
        );
        assert!(result.is_err(), "Expected recursion error");
        if let Err(e) = result {
            assert!(
                matches!(e, SemanticError::RecursionDetected { .. }),
                "Expected RecursionDetected, got {:?}",
                e
            );
        }
    }

    #[test]
    fn test_recursion_rejected_even_in_permissive_mode() {
        // Recursion is always rejected (safe-by-constraint, no escape hatch)
        // Use FFI with Rust if recursive algorithms are needed
        let result = analyze_permissive(
            r#"
            fn factorial(n: u32) -> u32 {
                if n == 0 {
                    return 1;
                }
                return n * factorial(n - 1);
            }
            "#,
        );
        assert!(
            result.is_err(),
            "Recursion should be rejected even in permissive mode"
        );
        if let Err(e) = result {
            assert!(
                matches!(e, SemanticError::RecursionDetected { .. }),
                "Expected RecursionDetected, got {:?}",
                e
            );
        }
    }

    #[test]
    fn test_non_recursive_function_allowed() {
        // Non-recursive functions should be allowed
        let result = analyze(
            r#"
            fn helper() -> u32 {
                return 42;
            }
            fn main() -> u32 {
                return helper();
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected non-recursive call to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_catch_on_non_error_type_rejected() {
        // catch on non-error types should be rejected
        let result = analyze(
            r#"
            fn main() -> u32 {
                x := 42 catch 0;
                return x;
            }
            "#,
        );
        assert!(result.is_err(), "Expected catch on non-error type to fail");
        if let Err(e) = result {
            assert!(
                matches!(e, SemanticError::CatchOnNonErrorType { .. }),
                "Expected CatchOnNonErrorType, got {:?}",
                e
            );
        }
    }

    #[test]
    fn test_catch_on_error_union_allowed() {
        // catch on error union types should be allowed
        let result = analyze(
            r#"
            MyError := error { Fail };
            fn might_fail() -> MyError!u32 {
                return 42;
            }
            fn main() -> u32 {
                x := might_fail() catch 0;
                return x;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected catch on error union to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_batch_gate_single_qubit_correct_arity() {
        // Single qubit gate with single qubit targets should pass
        let result = analyze(
            r#"
            fn main() -> unit {
                mut q := qalloc(4);
                pz q;
                h { q[0], q[1], q[2] };
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected single qubit batch gate to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_batch_gate_two_qubit_correct_arity() {
        // Two qubit gate with pair targets should pass
        let result = analyze(
            r#"
            fn main() -> unit {
                mut q := qalloc(4);
                pz q;
                cx { (q[0], q[1]), (q[2], q[3]) };
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected two qubit batch gate to pass: {:?}",
            result
        );
    }

    #[test]
    fn test_batch_gate_two_qubit_wrong_arity() {
        // Two qubit gate with single qubit target should fail
        let result = analyze(
            r#"
            fn main() -> unit {
                mut q := qalloc(4);
                pz q;
                cx { q[0] };
                return unit;
            }
            "#,
        );
        assert!(
            result.is_err(),
            "Expected two qubit gate with single qubit to fail"
        );
        if let Err(e) = result {
            assert!(
                matches!(
                    e,
                    SemanticError::GateArityMismatch {
                        expected: 2,
                        found: 1,
                        ..
                    }
                ),
                "Expected GateArityMismatch, got {:?}",
                e
            );
        }
    }

    #[test]
    fn test_batch_gate_single_qubit_wrong_arity() {
        // Single qubit gate with pair target should fail
        let result = analyze(
            r#"
            fn main() -> unit {
                mut q := qalloc(4);
                pz q;
                h { (q[0], q[1]) };
                return unit;
            }
            "#,
        );
        assert!(
            result.is_err(),
            "Expected single qubit gate with pair to fail"
        );
        if let Err(e) = result {
            assert!(
                matches!(
                    e,
                    SemanticError::GateArityMismatch {
                        expected: 1,
                        found: 2,
                        ..
                    }
                ),
                "Expected GateArityMismatch, got {:?}",
                e
            );
        }
    }

    // =========================================================================
    // ResolvedType and contains_unknown Tests
    // =========================================================================

    #[test]
    fn test_contains_unknown_primitive_types() {
        // Primitives never contain Unknown
        assert!(!Type::Bool.contains_unknown());
        assert!(
            !Type::UInt {
                bits: BitWidth::BITS_32
            }
            .contains_unknown()
        );
        assert!(
            !Type::IInt {
                bits: BitWidth::BITS_64
            }
            .contains_unknown()
        );
        assert!(!Type::F64.contains_unknown());
        assert!(!Type::Qubit.contains_unknown());
        assert!(!Type::Unit.contains_unknown());
        assert!(!Type::Never.contains_unknown());
    }

    #[test]
    fn test_contains_unknown_direct() {
        // Unknown itself contains unknown
        assert!(Type::Unknown.contains_unknown());
    }

    #[test]
    fn test_contains_unknown_nested_in_array() {
        // Array with Unknown element contains unknown
        let arr_unknown = Type::Array {
            element: Box::new(Type::Unknown),
            size: Some(10),
        };
        assert!(arr_unknown.contains_unknown());

        // Array with concrete element doesn't contain unknown
        let arr_u32 = Type::Array {
            element: Box::new(Type::UInt {
                bits: BitWidth::BITS_32,
            }),
            size: Some(10),
        };
        assert!(!arr_u32.contains_unknown());
    }

    #[test]
    fn test_contains_unknown_nested_in_tuple() {
        // Tuple with Unknown element
        let tuple_with_unknown = Type::Tuple {
            elements: vec![
                Type::UInt {
                    bits: BitWidth::BITS_32,
                },
                Type::Unknown,
                Type::Bool,
            ],
        };
        assert!(tuple_with_unknown.contains_unknown());

        // Tuple without Unknown
        let tuple_concrete = Type::Tuple {
            elements: vec![
                Type::UInt {
                    bits: BitWidth::BITS_32,
                },
                Type::Bool,
            ],
        };
        assert!(!tuple_concrete.contains_unknown());
    }

    #[test]
    fn test_contains_unknown_deeply_nested() {
        // Unknown deeply nested: Array(Optional(Unknown))
        let deeply_nested = Type::Array {
            element: Box::new(Type::Optional {
                inner: Box::new(Type::Unknown),
            }),
            size: Some(5),
        };
        assert!(deeply_nested.contains_unknown());
    }

    #[test]
    fn test_contains_unknown_error_union() {
        // Unknown in error position
        let unknown_error = Type::ErrorUnion {
            error: Box::new(Type::Unknown),
            payload: Box::new(Type::UInt {
                bits: BitWidth::BITS_32,
            }),
        };
        assert!(unknown_error.contains_unknown());

        // Unknown in payload position
        let unknown_payload = Type::ErrorUnion {
            error: Box::new(Type::AnyError),
            payload: Box::new(Type::Unknown),
        };
        assert!(unknown_payload.contains_unknown());

        // Neither contains Unknown
        let concrete_union = Type::ErrorUnion {
            error: Box::new(Type::AnyError),
            payload: Box::new(Type::UInt {
                bits: BitWidth::BITS_32,
            }),
        };
        assert!(!concrete_union.contains_unknown());
    }

    #[test]
    fn test_resolve_success() {
        // Concrete type resolves successfully
        let ty = Type::UInt {
            bits: BitWidth::BITS_32,
        };
        let resolved = ty.resolve();
        assert!(resolved.is_some());
        assert_eq!(resolved.unwrap().display_name(), "u32");
    }

    #[test]
    fn test_resolve_failure_direct() {
        // Unknown type fails to resolve
        let ty = Type::Unknown;
        assert!(ty.resolve().is_none());
    }

    #[test]
    fn test_resolve_failure_nested() {
        // Type containing Unknown fails to resolve
        let ty = Type::Optional {
            inner: Box::new(Type::Unknown),
        };
        assert!(ty.resolve().is_none());
    }

    #[test]
    fn test_resolved_type_methods() {
        let ty = Type::UInt {
            bits: BitWidth::BITS_64,
        };
        let resolved = ty.resolve().unwrap();

        // Check wrapper methods work correctly
        assert!(resolved.is_numeric());
        assert!(resolved.is_integer());
        assert!(!resolved.is_float());
        assert!(!resolved.is_quantum());
        assert_eq!(resolved.display_name(), "u64");
    }

    #[test]
    fn test_is_resolved() {
        assert!(Type::Bool.is_resolved());
        assert!(Type::F64.is_resolved());
        assert!(!Type::Unknown.is_resolved());

        let nested = Type::Array {
            element: Box::new(Type::Unknown),
            size: None,
        };
        assert!(!nested.is_resolved());
    }

    // =========================================================================
    // Undefined Type Error Tests
    // =========================================================================

    #[test]
    fn test_undefined_type_in_variable_declaration() {
        // Using undefined type should produce an error
        let result = analyze(
            r#"
            fn main() -> unit {
                x: UndefinedType = 42;
                return unit;
            }
            "#,
        );
        assert!(
            matches!(result, Err(SemanticError::UndefinedType { ref name, .. }) if name == "UndefinedType"),
            "Expected UndefinedType error, got: {:?}",
            result
        );
    }

    #[test]
    fn test_undefined_type_in_function_param() {
        // Using undefined type in function parameter should produce an error
        let result = analyze(
            r#"
            fn foo(x: NonexistentType) -> unit {
                return unit;
            }
            "#,
        );
        assert!(
            matches!(result, Err(SemanticError::UndefinedType { ref name, .. }) if name == "NonexistentType"),
            "Expected UndefinedType error, got: {:?}",
            result
        );
    }

    #[test]
    fn test_undefined_type_in_return_type() {
        // Using undefined type in return type should produce an error
        let result = analyze(
            r#"
            fn foo() -> MissingType {
                return 42;
            }
            "#,
        );
        assert!(
            matches!(result, Err(SemanticError::UndefinedType { ref name, .. }) if name == "MissingType"),
            "Expected UndefinedType error, got: {:?}",
            result
        );
    }

    #[test]
    fn test_defined_type_works() {
        // Defined types should work correctly
        let result = analyze(
            r#"
            MyStruct := struct { x: u32 };
            fn main() -> unit {
                s: MyStruct = MyStruct { x: 42 };
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected defined type to work: {:?}",
            result
        );
    }

    // =========================================================================
    // Input Size Limit Tests
    // =========================================================================

    #[test]
    fn test_scope_nesting_within_limit() {
        // Normal scope nesting should work
        let result = analyze(
            r#"
            fn main() -> unit {
                {
                    {
                        {
                            x := 42;
                        }
                    }
                }
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected normal nesting to work: {:?}",
            result
        );
    }

    #[test]
    fn test_symbol_table_normal_usage() {
        // Normal symbol usage should work
        let result = analyze(
            r#"
            fn main() -> unit {
                a := 1;
                b := 2;
                c := 3;
                d := 4;
                e := 5;
                return unit;
            }
            "#,
        );
        assert!(
            result.is_ok(),
            "Expected normal symbol usage to work: {:?}",
            result
        );
    }

    #[test]
    fn test_max_scope_depth_constant() {
        // Verify the constant is reasonable (checked at compile time).
        const {
            assert!(
                MAX_SCOPE_DEPTH >= 64,
                "MAX_SCOPE_DEPTH should be at least 64"
            );
            assert!(
                MAX_SCOPE_DEPTH <= 1024,
                "MAX_SCOPE_DEPTH should not be excessive"
            );
        }
    }

    #[test]
    fn test_max_symbol_count_constant() {
        // Verify the constant is reasonable (checked at compile time).
        const {
            assert!(
                MAX_SYMBOL_COUNT >= 10_000,
                "MAX_SYMBOL_COUNT should be at least 10000"
            );
            assert!(
                MAX_SYMBOL_COUNT <= 10_000_000,
                "MAX_SYMBOL_COUNT should not be excessive"
            );
        }
    }

    // =========================================================================
    // Multi-Error Collection Tests
    // =========================================================================

    #[test]
    fn test_analyze_collecting_errors_multiple() {
        // Program with multiple errors should collect them all
        let source = r#"
            fn foo() -> UndefinedType1 {
                return 42;
            }
            fn bar() -> UndefinedType2 {
                return 42;
            }
        "#;
        let program = crate::parse(source).unwrap();
        let mut analyzer = SemanticAnalyzer::new();
        let result = analyzer.analyze_collecting_errors(&program);

        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(
            errors.len() >= 2,
            "Expected at least 2 errors, got {}",
            errors.len()
        );

        // Check that both undefined types are reported
        let error_names: Vec<_> = errors
            .iter()
            .filter_map(|e| {
                if let SemanticError::UndefinedType { name, .. } = e {
                    Some(name.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert!(
            error_names.contains(&"UndefinedType1"),
            "Should report UndefinedType1"
        );
        assert!(
            error_names.contains(&"UndefinedType2"),
            "Should report UndefinedType2"
        );
    }

    #[test]
    fn test_analyze_collecting_errors_none() {
        // Valid program should have no errors
        let source = r#"
            fn main() -> unit {
                return unit;
            }
        "#;
        let program = crate::parse(source).unwrap();
        let mut analyzer = SemanticAnalyzer::new();
        let result = analyzer.analyze_collecting_errors(&program);

        assert!(result.is_ok(), "Expected no errors: {:?}", result);
    }

    #[test]
    fn test_semantic_errors_display() {
        // Test the Display impl for SemanticErrors
        let errors = SemanticErrors::new(vec![
            SemanticError::UndefinedType {
                name: "Foo".to_string(),
                location: SourceLocation::default(),
            },
            SemanticError::UndefinedType {
                name: "Bar".to_string(),
                location: SourceLocation::default(),
            },
        ]);

        let display = format!("{}", errors);
        assert!(display.contains("2 error(s)"));
        assert!(display.contains("Foo"));
        assert!(display.contains("Bar"));
    }

    #[test]
    fn test_semantic_errors_iter() {
        let errors = SemanticErrors::new(vec![
            SemanticError::UndefinedType {
                name: "A".to_string(),
                location: SourceLocation::default(),
            },
            SemanticError::UndefinedType {
                name: "B".to_string(),
                location: SourceLocation::default(),
            },
        ]);

        assert_eq!(errors.len(), 2);
        assert!(!errors.is_empty());

        let names: Vec<_> = errors
            .iter()
            .filter_map(|e| {
                if let SemanticError::UndefinedType { name, .. } = e {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(names, vec!["A", "B"]);
    }

    #[test]
    fn test_error_count_and_take() {
        let source = r#"
            fn foo(x: UndefinedType) -> unit {
                return unit;
            }
        "#;
        let program = crate::parse(source).unwrap();
        let mut analyzer = SemanticAnalyzer::new();
        let _ = analyzer.analyze(&program); // Ignore result

        // Errors should have been collected
        assert!(analyzer.error_count() > 0);

        // Take errors should empty the list
        let taken = analyzer.take_errors();
        assert!(!taken.is_empty());
        assert_eq!(analyzer.error_count(), 0);
    }

    #[test]
    fn test_slice_element_access() {
        // Test that indexing a slice parameter with an integer returns element type
        // NOTE: Variable name must NOT be a gate name (s, h, x, y, z, t are gates)
        let source = r#"
            fn get_element(data: []i32) -> i32 {
                return data[0];
            }
        "#;
        let result = analyze(source);
        assert!(
            result.is_ok(),
            "Slice element access should work: {:?}",
            result.err()
        );
    }

    // =========================================================================
    // Inline For Loop Validation Tests
    // =========================================================================

    #[test]
    fn test_inline_for_valid() {
        // Valid inline for with comptime range
        let source = r#"
            fn main() -> unit {
                q := qalloc(4);
                pz q;
                inline for i in 0..4 {
                    h q[i];
                }
                return unit;
            }
        "#;
        let result = analyze(source);
        assert!(
            result.is_ok(),
            "Valid inline for should pass: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_inline_for_with_comptime_expr() {
        // Valid inline for with comptime expression bound
        let source = r#"
            fn main() -> unit {
                q := qalloc(4);
                pz q;
                inline for i in 0..(2 * 2) {
                    h q[i];
                }
                return unit;
            }
        "#;
        let result = analyze(source);
        assert!(
            result.is_ok(),
            "Inline for with comptime expr should pass: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_inline_for_error_break() {
        // break is not allowed in inline for
        let source = r#"
            fn main() -> unit {
                q := qalloc(4);
                pz q;
                inline for i in 0..4 {
                    h q[i];
                    break;
                }
                return unit;
            }
        "#;
        let result = analyze(source);
        assert!(
            matches!(result, Err(SemanticError::BreakInInlineFor { .. })),
            "Expected BreakInInlineFor error, got {:?}",
            result
        );
    }

    #[test]
    fn test_inline_for_error_continue() {
        // continue is not allowed in inline for
        let source = r#"
            fn main() -> unit {
                q := qalloc(4);
                pz q;
                inline for i in 0..4 {
                    h q[i];
                    continue;
                }
                return unit;
            }
        "#;
        let result = analyze(source);
        assert!(
            matches!(result, Err(SemanticError::ContinueInInlineFor { .. })),
            "Expected ContinueInInlineFor error, got {:?}",
            result
        );
    }

    #[test]
    fn test_inline_for_error_non_comptime_range() {
        // Runtime variable in range is not allowed
        let source = r#"
            fn main() -> unit {
                q := qalloc(4);
                pz q;
                mut n := 4;
                inline for i in 0..n {
                    h q[i];
                }
                return unit;
            }
        "#;
        let result = analyze(source);
        assert!(
            matches!(result, Err(SemanticError::InlineForRangeNotComptime { .. })),
            "Expected InlineForRangeNotComptime error, got {:?}",
            result
        );
    }

    #[test]
    fn test_inline_for_nested_break_in_regular_for() {
        // break in a regular for inside inline for should be allowed
        let source = r#"
            fn main() -> unit {
                q := qalloc(4);
                pz q;
                inline for i in 0..2 {
                    for j in 0..10 {
                        if j == 5 {
                            break;
                        }
                    }
                }
                return unit;
            }
        "#;
        let result = analyze(source);
        // This should fail because we're still inside an inline for
        // and break applies to the inner regular for
        assert!(
            matches!(result, Err(SemanticError::BreakInInlineFor { .. })),
            "Break in nested regular for inside inline for should fail: {:?}",
            result
        );
    }

    // =========================================================================
    // Generic Type Instantiation Tests
    // =========================================================================

    #[test]
    fn test_mangle_generic_name() {
        use crate::comptime::ComptimeValue;

        let args = vec![
            ComptimeValue::Type(Type::UInt {
                bits: BitWidth::must(32),
            }),
            ComptimeValue::Uint(4),
        ];
        let mangled = SemanticAnalyzer::mangle_generic_name("make_array", &args);
        assert!(mangled.starts_with("make_array__CT__"));
        assert!(mangled.contains("u32") || mangled.contains("UInt"));
        assert!(mangled.contains("4"));
    }

    #[test]
    fn test_serialize_comptime_args() {
        use crate::comptime::ComptimeValue;

        let args = vec![ComptimeValue::Int(42), ComptimeValue::Bool(true)];
        let serialized = SemanticAnalyzer::serialize_comptime_args(&args);
        assert!(serialized.contains("42"));
        assert!(serialized.contains("true"));
    }

    #[test]
    fn test_generic_function_detection() {
        // Test that functions with comptime params are detected
        // Use a comptime integer parameter since comptime type params need more work
        let source = r#"
            fn repeat(comptime N: u32, val: i32) -> i32 {
                return val;
            }
        "#;
        let program = crate::parse(source).unwrap();

        // First verify the parser detected the comptime param
        if let crate::ast::TopLevelDecl::Fn(fn_decl) = &program.declarations[0] {
            assert_eq!(fn_decl.params.len(), 2);
            assert!(
                fn_decl.params[0].is_comptime,
                "First param should be comptime (parser should set this). Got: {:?}",
                fn_decl.params[0]
            );
            assert!(
                !fn_decl.params[1].is_comptime,
                "Second param should not be comptime"
            );
        } else {
            panic!("Expected function declaration");
        }

        let mut analyzer = SemanticAnalyzer::new();
        let result = analyzer.analyze(&program);
        assert!(
            result.is_ok(),
            "Generic function should parse: {:?}",
            result.err()
        );

        // Check that the function was registered with comptime param info
        if let Some(symbol) = analyzer.symbols.lookup("repeat") {
            if let SymbolKind::Function {
                comptime_param_indices,
                original_decl,
                ..
            } = &symbol.kind
            {
                assert_eq!(
                    comptime_param_indices.len(),
                    1,
                    "Should have 1 comptime param"
                );
                assert_eq!(
                    comptime_param_indices[0], 0,
                    "First param should be comptime"
                );
                assert!(
                    original_decl.is_some(),
                    "Should store original decl for generic"
                );
            } else {
                panic!("Expected function symbol");
            }
        } else {
            panic!("Function not found in symbol table");
        }
    }

    #[test]
    fn test_non_generic_function_no_original_decl() {
        // Test that regular functions don't store original_decl
        let source = r#"
            fn add(a: i32, b: i32) -> i32 {
                return a + b;
            }
        "#;
        let program = crate::parse(source).unwrap();
        let mut analyzer = SemanticAnalyzer::new();
        let _ = analyzer.analyze(&program);

        if let Some(symbol) = analyzer.symbols.lookup("add")
            && let SymbolKind::Function {
                comptime_param_indices,
                original_decl,
                ..
            } = &symbol.kind
        {
            assert!(
                comptime_param_indices.is_empty(),
                "Should have no comptime params"
            );
            assert!(
                original_decl.is_none(),
                "Should not store original decl for non-generic"
            );
        }
    }

    // =========================================================================
    // Alias Tests
    // =========================================================================

    #[test]
    fn test_alias_basic() {
        // Basic alias should parse and analyze
        let source = r#"
            pub fn main() -> unit {
                arr: [8]u32 = undefined;
                alias view := arr[0..4];
                return;
            }
        "#;
        let program = crate::parse(source).unwrap();
        let mut analyzer = SemanticAnalyzer::new();
        let result = analyzer.analyze(&program);
        assert!(result.is_ok(), "Basic alias should analyze: {:?}", result);
    }

    #[test]
    fn test_alias_non_overlapping() {
        // Non-overlapping aliases on same source should work
        let source = r#"
            pub fn main() -> unit {
                arr: [8]u32 = undefined;
                alias first := arr[0..4];
                alias second := arr[4..8];
                return;
            }
        "#;
        let program = crate::parse(source).unwrap();
        let mut analyzer = SemanticAnalyzer::new();
        let result = analyzer.analyze(&program);
        assert!(
            result.is_ok(),
            "Non-overlapping aliases should work: {:?}",
            result
        );
    }

    #[test]
    fn test_alias_overlapping_error() {
        // Overlapping aliases should be an error
        let source = r#"
            pub fn main() -> unit {
                arr: [8]u32 = undefined;
                alias first := arr[0..4];
                alias second := arr[2..6];
                return;
            }
        "#;
        let program = crate::parse(source).unwrap();
        let mut analyzer = SemanticAnalyzer::new();
        let result = analyzer.analyze(&program);
        assert!(
            matches!(result, Err(SemanticError::OverlappingAlias(_))),
            "Overlapping aliases should error: {:?}",
            result
        );
    }

    #[test]
    fn test_alias_adjacent_ranges() {
        // Adjacent ranges [0..4) and [4..8) should not overlap
        let source = r#"
            pub fn main() -> unit {
                arr: [8]u32 = undefined;
                alias a := arr[0..4];
                alias b := arr[4..8];
                return;
            }
        "#;
        let program = crate::parse(source).unwrap();
        let mut analyzer = SemanticAnalyzer::new();
        let result = analyzer.analyze(&program);
        assert!(
            result.is_ok(),
            "Adjacent ranges should not overlap: {:?}",
            result
        );
    }

    #[test]
    fn test_alias_different_sources() {
        // Aliases on different sources can have overlapping ranges
        let source = r#"
            pub fn main() -> unit {
                arr1: [8]u32 = undefined;
                arr2: [8]u32 = undefined;
                alias a := arr1[0..4];
                alias b := arr2[0..4];
                return;
            }
        "#;
        let program = crate::parse(source).unwrap();
        let mut analyzer = SemanticAnalyzer::new();
        let result = analyzer.analyze(&program);
        assert!(
            result.is_ok(),
            "Different sources can have same ranges: {:?}",
            result
        );
    }

    #[test]
    fn test_alias_usable_as_value() {
        // Alias should be usable like a slice
        let source = r#"
            fn take_slice(data: []u32) -> unit {
                return;
            }
            pub fn main() -> unit {
                arr: [8]u32 = undefined;
                alias view := arr[0..4];
                take_slice(view);
                return;
            }
        "#;
        let program = crate::parse(source).unwrap();
        let mut analyzer = SemanticAnalyzer::new();
        let result = analyzer.analyze(&program);
        assert!(
            result.is_ok(),
            "Alias should be usable as slice: {:?}",
            result
        );
    }

    #[test]
    fn test_ranges_overlap_function() {
        // Test the ranges_overlap helper
        assert!(SemanticAnalyzer::ranges_overlap((0, 4), (2, 6))); // Overlap
        assert!(SemanticAnalyzer::ranges_overlap((2, 6), (0, 4))); // Overlap (reversed)
        assert!(!SemanticAnalyzer::ranges_overlap((0, 4), (4, 8))); // Adjacent, no overlap
        assert!(!SemanticAnalyzer::ranges_overlap((0, 2), (5, 8))); // Disjoint
        assert!(SemanticAnalyzer::ranges_overlap((0, 10), (5, 6))); // Contained
    }

    // =========================================================================
    // Array bounds checking
    // =========================================================================

    #[test]
    fn test_array_index_out_of_bounds() {
        let result = analyze(
            r#"
            fn main() -> unit {
                arr: [3]i64 = [1, 2, 3];
                x := arr[5];
                return;
            }
        "#,
        );
        assert!(result.is_err(), "Expected ArrayIndexOutOfBounds error");
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("out of bounds"),
            "Error message should mention out of bounds, got: {err}",
        );
    }

    #[test]
    fn test_array_index_at_boundary() {
        // arr[2] on [3]i64 is valid (indices 0, 1, 2)
        let result = analyze(
            r#"
            fn main() -> unit {
                arr: [3]i64 = [1, 2, 3];
                x := arr[2];
                return;
            }
        "#,
        );
        assert!(
            result.is_ok(),
            "arr[2] on [3]i64 should be valid, got: {:?}",
            result
        );
    }

    #[test]
    fn test_array_index_literal_at_size() {
        // arr[3] on [3]i64 is out of bounds (valid indices are 0, 1, 2)
        let result = analyze(
            r#"
            fn main() -> unit {
                arr: [3]i64 = [1, 2, 3];
                x := arr[3];
                return;
            }
        "#,
        );
        assert!(
            result.is_err(),
            "Expected ArrayIndexOutOfBounds for index == size"
        );
    }

    #[test]
    fn test_array_dynamic_index_no_error() {
        // Dynamic index should not produce a compile-time error
        assert!(
            analyze(
                r#"
            fn get(arr: [3]i64, i: i64) -> i64 {
                return arr[i];
            }
        "#
            )
            .is_ok()
        );
    }

    // =========================================================================
    // Custom gate declarations
    // =========================================================================

    #[test]
    fn test_declare_gate_registered() {
        // declare gate should be registered in the gate registry
        assert!(
            analyze(
                r#"
            declare gate my_rx(theta)(q);
        "#
            )
            .is_ok()
        );
    }

    #[test]
    fn test_declare_gate_no_params() {
        assert!(
            analyze(
                r#"
            declare gate my_x()(q);
        "#
            )
            .is_ok()
        );
    }

    #[test]
    fn test_declare_gate_multi_qubit() {
        assert!(
            analyze(
                r#"
            declare gate cnot()(control, target);
        "#
            )
            .is_ok()
        );
    }

    #[test]
    fn test_composite_gate_basic() {
        assert!(
            analyze(
                r#"
            gate my_h()(q) {
                h q;
            }
        "#
            )
            .is_ok()
        );
    }

    #[test]
    fn test_composite_gate_multi_qubit() {
        assert!(
            analyze(
                r#"
            gate bell()(q0, q1) {
                h q0;
                cx (q0, q1);
            }
        "#
            )
            .is_ok()
        );
    }

    #[test]
    fn test_declare_gate_duplicate_rejected() {
        // Defining same gate name twice should fail
        let result = analyze(
            r#"
            declare gate my_gate()(q);
            declare gate my_gate()(q);
        "#,
        );
        assert!(result.is_err(), "Duplicate gate declaration should fail");
    }

    #[test]
    fn test_declare_then_define_same_gate_rejected() {
        // `declare gate` is an opaque target gate, not a forward declaration:
        // declaring then defining the same name is a duplicate, not a definition.
        let err = analyze(
            r#"
            declare gate foo()(q);
            gate foo()(q) { h q; }
        "#,
        )
        .expect_err("declare-then-define of the same gate should fail");
        assert!(
            err.to_string().contains("target gate"),
            "error should explain the gate is already a target declaration: {err}"
        );
    }

    #[test]
    fn test_define_then_define_same_gate_rejected() {
        // Two composite definitions of the same name collide.
        let err = analyze(
            r#"
            gate foo()(q) { h q; }
            gate foo()(q) { x q; }
        "#,
        )
        .expect_err("defining the same gate twice should fail");
        assert!(
            err.to_string().contains("already defined"),
            "error should report the gate is already defined: {err}"
        );
    }

    #[test]
    fn test_declare_gate_builtin_exact_signature_allowed() {
        // Redeclaring a built-in with its exact signature is a harmless no-op.
        // `rz` is a 1-parameter, 1-qubit built-in.
        assert!(
            analyze(
                r#"
            declare gate rz(angle)(q);
        "#
            )
            .is_ok()
        );
    }

    #[test]
    fn test_declare_gate_builtin_mismatched_signature_rejected() {
        // `rz` is a 1-parameter built-in; redeclaring it with no parameters
        // would be uncallable under the fixed built-in parameterization.
        assert!(
            analyze(
                r#"
            declare gate rz()(q);
        "#
            )
            .is_err()
        );
    }

    #[test]
    fn test_builtin_gates_still_work() {
        // Built-in gates should still work alongside custom gate declarations
        assert!(
            analyze(
                r#"
            declare gate custom_rx(theta)(q);

            fn apply(q: qubit) -> unit {
                h q;
                x q;
                return;
            }
        "#
            )
            .is_ok()
        );
    }
}
