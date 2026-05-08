//! Program abstraction for QIS Classical Control Engine
//!
//! Unified program interface that allows different
//! program types (`Qis`, HUGR, raw `QisInterface`) to be used with
//! the `QisEngine` through a consistent `.program()` API.
//!
//! Default implementations use Selene-based interfaces with explicit
//! error handling - no silent fallbacks are provided.

use pecos_core::errors::PecosError;
use pecos_programs::{Hugr, Qis};
use pecos_qis_ffi_types::OperationCollector;

/// A trait for types that can be converted into a `QisInterface`
///
/// This allows the `QisEngine` builder to accept different program types
/// through a unified `.program()` method, similar to how `QASMEngine` works.
///
/// Default implementations use Selene-based interfaces (Helios for QIS/HUGR programs).
/// If the default is not available, explicit error messages guide users to alternatives.
pub trait IntoQisInterface {
    /// Convert this program into a `QisInterface`
    ///
    /// # Errors
    /// Returns an error directing users to use explicit implementation crates.
    fn into_qis_interface(self) -> Result<OperationCollector, PecosError>;
}

/// Program type classification for interface provider selection
#[derive(Debug, Clone, PartialEq)]
pub enum ProgramType {
    /// LLVM IR text format
    LlvmIr,
    /// QIS bitcode format
    QisBitcode,
    /// HUGR bytes format
    HugrBytes,
}

/// Implement `IntoQisInterface` for `OperationCollector` itself (identity conversion)
impl IntoQisInterface for OperationCollector {
    fn into_qis_interface(self) -> Result<OperationCollector, PecosError> {
        Ok(self)
    }
}

/// Trait for building `QisInterface` instances from programs
///
/// This trait allows different compilation strategies (e.g., Helios)
/// to be plugged into the `QisEngineBuilder` through the .`interface()` method.
pub trait QisInterfaceBuilder: Send + Sync + dyn_clone::DynClone {
    /// Build a `QisInterface` from the given program using this builder's strategy
    ///
    /// Since we can't call sized methods on trait objects, each implementation
    /// needs to handle the program type directly
    ///
    /// # Errors
    /// Returns an error if the program cannot be built into an interface.
    fn build_from_qis_program(&self, program: Qis) -> Result<OperationCollector, PecosError>;

    /// Build from HUGR program
    ///
    /// # Errors
    /// Returns an error if the program cannot be built into an interface.
    fn build_from_hugr_program(&self, program: Hugr) -> Result<OperationCollector, PecosError>;

    /// Build from pre-built interface
    ///
    /// # Errors
    /// Returns an error if the interface cannot be processed.
    fn build_from_interface(
        &self,
        interface: OperationCollector,
    ) -> Result<OperationCollector, PecosError>;

    /// Get a descriptive name for this builder
    fn name(&self) -> &'static str;

    /// Create a boxed interface for dynamic execution (without collecting operations)
    ///
    /// This is used when dynamic circuit execution is enabled. Instead of
    /// pre-collecting all operations, it creates an interface that can run
    /// the program dynamically and coordinate with the quantum simulator.
    ///
    /// # Errors
    /// Returns an error if the interface cannot be created.
    fn create_dynamic_interface_from_qis(
        &self,
        program: Qis,
    ) -> Result<crate::qis_interface::BoxedInterface, PecosError> {
        // Default implementation: not supported
        let _ = program;
        Err(PecosError::Processing(format!(
            "Interface builder '{}' does not support dynamic execution.\n\
            Dynamic execution requires an interface that can run LLVM programs incrementally.",
            self.name()
        )))
    }

    /// Create a boxed interface for dynamic execution from HUGR program
    ///
    /// # Errors
    /// Returns an error if the interface cannot be created.
    fn create_dynamic_interface_from_hugr(
        &self,
        program: Hugr,
    ) -> Result<crate::qis_interface::BoxedInterface, PecosError> {
        // Default implementation: not supported
        let _ = program;
        Err(PecosError::Processing(format!(
            "Interface builder '{}' does not support dynamic HUGR execution.\n\
            Dynamic execution requires an interface that can run LLVM programs incrementally.",
            self.name()
        )))
    }
}

// Implement dyn_clone for the trait
dyn_clone::clone_trait_object!(QisInterfaceBuilder);

/// Enum to specify which interface builder to use (for backwards compatibility)
#[derive(Debug, Clone)]
pub enum InterfaceChoice {
    /// Auto-select (returns error, user must choose explicit implementation)
    Auto,
}

/// Implement `IntoQisInterface` for `Qis`
///
/// Users must explicitly specify runtime and interface using the builder API.
impl IntoQisInterface for Qis {
    fn into_qis_interface(self) -> Result<OperationCollector, PecosError> {
        Err(PecosError::Processing(
            "No default QIS interface implementation available.\n\
            Please explicitly specify a runtime and interface when building the engine:\n\n\
            use pecos::qis_engine;\n\
            use pecos::{selene_simple_runtime, helios_interface_builder};\n\n\
            let engine_builder = qis_engine()\n\
                .runtime(selene_simple_runtime()?)\n\
                .interface(helios_interface_builder())\n\
                .try_program(qis_program)?;\n\n\
            The Selene Helios interface is the reference implementation for QIS programs."
                .to_string(),
        ))
    }
}

/// Implement `IntoQisInterface` for HUGR bytes
///
/// Users must explicitly specify a runtime and interface.
impl IntoQisInterface for &[u8] {
    fn into_qis_interface(self) -> Result<OperationCollector, PecosError> {
        Err(PecosError::Processing(
            "No default interface implementation for HUGR bytes.\n\
            Please explicitly specify a runtime and interface when building the engine:\n\n\
            use pecos::qis_engine;\n\
            use pecos::{selene_simple_runtime, helios_interface_builder};\n\n\
            let engine_builder = qis_engine()\n\
                .runtime(selene_simple_runtime()?)\n\
                .interface(helios_interface_builder())\n\
                .try_program(hugr_program)?;"
                .to_string(),
        ))
    }
}

/// Implement `IntoQisInterface` for HUGR bytes (owned)
impl IntoQisInterface for Vec<u8> {
    fn into_qis_interface(self) -> Result<OperationCollector, PecosError> {
        Err(PecosError::Processing(
            "No default interface implementation for HUGR bytes.\n\
            Please explicitly specify a runtime and interface when building the engine:\n\n\
            use pecos::qis_engine;\n\
            use pecos::{selene_simple_runtime, helios_interface_builder};\n\n\
            let engine_builder = qis_engine()\n\
                .runtime(selene_simple_runtime()?)\n\
                .interface(helios_interface_builder())\n\
                .try_program(hugr_program)?;"
                .to_string(),
        ))
    }
}

/// Implement `IntoQisInterface` for `Hugr`
///
/// Users must explicitly specify a runtime and interface.
impl IntoQisInterface for Hugr {
    fn into_qis_interface(self) -> Result<OperationCollector, PecosError> {
        Err(PecosError::Processing(
            "No default interface implementation for HUGR programs.\n\
            Please explicitly specify a runtime and interface when building the engine:\n\n\
            use pecos::qis_engine;\n\
            use pecos::{selene_simple_runtime, helios_interface_builder};\n\n\
            let engine_builder = qis_engine()\n\
                .runtime(selene_simple_runtime()?)\n\
                .interface(helios_interface_builder())\n\
                .try_program(hugr_program)?;"
                .to_string(),
        ))
    }
}

/// Wrapper type to represent a QIS Control Engine Program
///
/// This is conceptually equivalent to `QisInterface`, but provides a
/// more semantically clear type name for the builder API.
#[derive(Debug, Clone)]
pub struct QisEngineProgram {
    interface: OperationCollector,
}

impl QisEngineProgram {
    /// Create a new program from a `QisInterface`
    #[must_use]
    pub fn new(interface: OperationCollector) -> Self {
        Self { interface }
    }

    /// Create a program from anything that can be converted to `QisInterface`
    ///
    /// # Errors
    /// Returns an error if the conversion fails
    pub fn from_program<P: IntoQisInterface>(program: P) -> Result<Self, PecosError> {
        let interface = program.into_qis_interface()?;
        Ok(Self::new(interface))
    }

    /// Get the underlying `QisInterface`
    #[must_use]
    pub fn into_interface(self) -> OperationCollector {
        self.interface
    }

    /// Get a reference to the underlying `QisInterface`
    #[must_use]
    pub fn interface(&self) -> &OperationCollector {
        &self.interface
    }
}

impl IntoQisInterface for QisEngineProgram {
    fn into_qis_interface(self) -> Result<OperationCollector, PecosError> {
        Ok(self.interface)
    }
}

impl From<OperationCollector> for QisEngineProgram {
    fn from(interface: OperationCollector) -> Self {
        Self::new(interface)
    }
}

// Tests for program conversion require actual interface implementations
// and are in the integration test files.
