//! Trait for QIS program execution interfaces
//!
//! This module defines the `QisInterface` trait that different implementations
//! (JIT, Helios, etc.) must implement to execute quantum programs and collect operations.

use pecos_qis_ffi_types::OperationCollector;
use std::collections::BTreeMap;

/// Program format for loading
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgramFormat {
    /// LLVM IR text
    LlvmIrText,
    /// LLVM bitcode
    LlvmBitcode,
    /// HUGR bytes
    HugrBytes,
    /// QIS bitcode (Selene format)
    QisBitcode,
}

/// Error type for interface operations
///
/// This is kept minimal to avoid circular dependencies with pecos-core.
/// Implementations can convert to `PecosError` as needed.
#[derive(Debug, Clone)]
pub enum InterfaceError {
    /// Program loading error
    LoadError(String),
    /// Execution error
    ExecutionError(String),
    /// Invalid program format
    InvalidFormat(String),
    /// Other error
    Other(String),
}

impl std::fmt::Display for InterfaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LoadError(msg) => write!(f, "Load error: {msg}"),
            Self::ExecutionError(msg) => write!(f, "Execution error: {msg}"),
            Self::InvalidFormat(msg) => write!(f, "Invalid format: {msg}"),
            Self::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for InterfaceError {}

/// Trait for QIS interface implementations
///
/// A `QisInterface` implementation is responsible for executing a quantum program and
/// collecting the quantum operations that need to be performed.
///
/// Different implementations:
/// - `pecos_qis_jit::QisJitInterface` - Uses LLVM JIT compilation
/// - `pecos_qis_selene::QisHeliosInterface` - Links with Selene's Helios compiler
/// - `SimpleQisInterface` - Pre-built operations list
pub trait QisInterface: Send + Sync {
    /// Load a program into the interface
    ///
    /// The format depends on the implementation:
    /// - JIT: LLVM IR text or bitcode
    /// - Helios: QIS bitcode or HUGR bytes
    ///
    /// # Errors
    /// Returns an error if the program cannot be loaded or parsed.
    fn load_program(
        &mut self,
        program_bytes: &[u8],
        format: ProgramFormat,
    ) -> Result<(), InterfaceError>;

    /// Execute the program to collect operations
    ///
    /// This runs the program in "collection mode" to discover all quantum
    /// operations without actually performing quantum simulation.
    ///
    /// # Errors
    /// Returns an error if the program execution fails.
    fn collect_operations(&mut self) -> Result<OperationCollector, InterfaceError>;

    /// Execute with measurement results
    ///
    /// This runs the program with specific measurement results to handle
    /// conditional execution paths correctly.
    ///
    /// # Errors
    /// Returns an error if the program execution fails.
    fn execute_with_measurements(
        &mut self,
        measurements: BTreeMap<usize, bool>,
    ) -> Result<OperationCollector, InterfaceError>;

    /// Get metadata about the implementation
    fn metadata(&self) -> BTreeMap<String, String> {
        BTreeMap::new()
    }

    /// Get the name of this implementation
    fn name(&self) -> &'static str;

    /// Reset the interface for a new execution
    ///
    /// # Errors
    /// Returns an error if the reset operation fails.
    fn reset(&mut self) -> Result<(), InterfaceError>;
}

/// Box type for interface implementations
pub type BoxedInterface = Box<dyn QisInterface>;
