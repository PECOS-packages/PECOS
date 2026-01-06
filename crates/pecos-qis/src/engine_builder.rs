//! Builder for `QisEngine` that integrates with PECOS `sim()` API

use crate::{IntoQisInterface, QisEngine};
use pecos_core::errors::PecosError;
use pecos_engines::ClassicalControlEngineBuilder;
use pecos_qis_ffi_types::OperationCollector;

/// Builder for creating `QisEngine` instances
pub struct QisEngineBuilder {
    runtime: Option<Box<dyn crate::runtime::QisRuntime>>,
    interface: Option<OperationCollector>,
    interface_builder: Option<Box<dyn crate::program::QisInterfaceBuilder>>,
    program_source: Option<String>, // Store original program source for loading
}

impl Clone for QisEngineBuilder {
    fn clone(&self) -> Self {
        Self {
            runtime: self.runtime.as_ref().map(|r| dyn_clone::clone_box(&**r)),
            interface: self.interface.clone(),
            // Clone the interface builder if present
            interface_builder: self
                .interface_builder
                .as_ref()
                .map(|b| dyn_clone::clone_box(&**b)),
            program_source: self.program_source.clone(),
        }
    }
}

impl QisEngineBuilder {
    /// Create a new builder without a runtime (user must call .`runtime()`)
    #[must_use]
    pub fn new() -> Self {
        Self {
            runtime: None,
            interface: None,
            interface_builder: None,
            program_source: None,
        }
    }

    /// Set a pre-built interface (for testing)
    #[must_use]
    pub fn with_interface(mut self, interface: OperationCollector) -> Self {
        self.interface = Some(interface);
        self
    }

    /// Set the program to use
    ///
    /// This is the preferred method for specifying the QIS program,
    /// consistent with other engines like `QASMEngine`.
    ///
    /// # Example
    /// ```rust
    /// use pecos_qis::qis_engine;
    /// use pecos_qis_ffi_types::{OperationCollector, QuantumOp};
    ///
    /// // Create an interface with quantum operations
    /// let mut interface = OperationCollector::new();
    /// let q0 = interface.allocate_qubit();
    /// let q1 = interface.allocate_qubit();
    /// interface.operations.push(QuantumOp::H(q0).into());
    /// interface.operations.push(QuantumOp::CX(q0, q1).into());
    ///
    /// // Use the fluent API with the program
    /// // (requires .runtime() to be added before calling .build())
    /// let builder = qis_engine().program(interface.clone());
    ///
    /// // Verify the interface has the correct structure
    /// assert_eq!(interface.allocated_qubits.len(), 2);
    /// assert_eq!(interface.operations.len(), 2);
    /// ```
    /// Set the program to use from any supported program type
    ///
    /// This method accepts any type that can be converted to `QisInterface`,
    /// including `Qis`, `Hugr`, etc. Panics on conversion errors.
    /// For error handling, use `try_program()` instead.
    ///
    /// # Example
    /// ```rust
    /// use pecos_qis::qis_engine;
    /// use pecos_qis_ffi_types::{OperationCollector, QuantumOp};
    ///
    /// // Create an interface with quantum operations
    /// let mut interface = OperationCollector::new();
    /// let q0 = interface.allocate_qubit();
    /// let q1 = interface.allocate_qubit();
    /// interface.operations.push(QuantumOp::H(q0).into());
    /// interface.operations.push(QuantumOp::CX(q0, q1).into());
    ///
    /// // Build with the program - program() will panic on invalid data
    /// // (requires .runtime() to be added before calling .build())
    /// let builder = qis_engine().program(interface.clone());
    ///
    /// // Verify the interface structure
    /// assert_eq!(interface.allocated_qubits.len(), 2);
    /// assert_eq!(interface.operations.len(), 2);
    /// ```
    ///
    /// # Panics
    /// Panics if the program cannot be converted to a QIS interface (e.g., compilation errors).
    #[must_use]
    pub fn program<P: IntoQisInterface + 'static>(self, program: P) -> Self {
        self.try_program(program).expect("Failed to set program")
    }

    /// Set the interface builder for the engine
    ///
    /// This allows you to explicitly specify which interface backend to use
    /// (JIT or Helios) when processing programs.
    ///
    /// # Example
    ///
    /// For examples of using custom interface builders, see the `pecos-qis` crate
    /// documentation which provides the `helios_interface_builder()` function.
    #[must_use]
    pub fn interface(
        mut self,
        builder: impl crate::program::QisInterfaceBuilder + 'static,
    ) -> Self {
        self.interface_builder = Some(Box::new(builder));
        self
    }

    /// Set the program to use from any supported program type (error handling version)
    ///
    /// This method accepts any type that can be converted to `QisInterface`,
    /// including `Qis`, `Hugr`, etc. Returns a Result because
    /// some conversions may fail (e.g., compilation errors).
    ///
    /// # Example
    /// ```rust
    /// use pecos_core::errors::PecosError;
    /// use pecos_qis::qis_engine;
    /// use pecos_qis_ffi_types::{OperationCollector, QuantumOp};
    ///
    /// // Create an interface with quantum operations
    /// let mut interface = OperationCollector::new();
    /// let q0 = interface.allocate_qubit();
    /// interface.operations.push(QuantumOp::H(q0).into());
    ///
    /// // Use try_program for error handling
    /// // (requires .runtime() to be added before calling .build())
    /// let builder = qis_engine().try_program(interface.clone())?;
    ///
    /// // Verify the interface structure
    /// assert_eq!(interface.allocated_qubits.len(), 1);
    /// assert_eq!(interface.operations.len(), 1);
    /// # Ok::<(), PecosError>(())
    /// ```
    ///
    /// # Errors
    /// Returns an error if the program cannot be converted to a QIS interface (e.g., compilation errors).
    pub fn try_program<P: IntoQisInterface + 'static>(
        mut self,
        program: P,
    ) -> Result<Self, PecosError> {
        // Check if the program is already an OperationCollector
        let any_program = &program as &dyn std::any::Any;

        if let Some(interface) = any_program.downcast_ref::<OperationCollector>() {
            // If an OperationCollector is directly provided and an interface builder was specified, error
            if self.interface_builder.is_some() {
                return Err(PecosError::with_context(
                    std::io::Error::new(std::io::ErrorKind::InvalidInput, "Invalid configuration"),
                    "Cannot use .interface() when providing a pre-built OperationCollector to .program()",
                ));
            }
            // Use the provided interface directly
            self.interface = Some(interface.clone());
        } else {
            // For other program types (Qis, Hugr), use the builder
            // Also store the original program source for loading into interface implementations
            if let Some(qis_prog) = any_program.downcast_ref::<pecos_programs::Qis>() {
                // Store the LLVM IR source for later loading
                match &qis_prog.content {
                    pecos_programs::QisContent::Ir(ir_string) => {
                        self.program_source = Some(ir_string.clone());
                    }
                    pecos_programs::QisContent::Bitcode(_bitcode) => {
                        // For bitcode, we'll need to convert or handle differently
                        // For now, skip storing source for bitcode programs
                        log::warn!("Bitcode programs not yet supported for interface loading");
                    }
                }
            }

            let interface = if let Some(builder) = &self.interface_builder {
                // Use the explicitly specified interface builder
                log::debug!("Using interface builder: {}", builder.name());
                if let Some(qis_prog) = any_program.downcast_ref::<pecos_programs::Qis>() {
                    log::debug!("Building interface from QIS program");
                    builder.build_from_qis_program(qis_prog.clone())?
                } else if let Some(hugr_prog) = any_program.downcast_ref::<pecos_programs::Hugr>() {
                    log::debug!("Building interface from HUGR program");
                    builder.build_from_hugr_program(hugr_prog.clone())?
                } else {
                    // Unknown type, use default conversion with the default backend (Helios)
                    log::debug!("Unknown program type, using into_qis_interface");
                    program.into_qis_interface()?
                }
            } else {
                // No interface builder specified, default to Helios
                log::debug!("No interface builder specified, using into_qis_interface");
                program.into_qis_interface()?
            };
            self.interface = Some(interface);
        }

        Ok(self)
    }

    /// Set the runtime to use
    ///
    /// This allows you to specify any runtime implementation.
    /// The runtime must implement the `QisRuntime` trait.
    ///
    /// The reference runtime is provided by the `pecos-qis` crate:
    /// - `pecos_qis::selene_simple_runtime()` - Selene-based implementation
    ///
    /// # Example
    ///
    /// For complete examples with runtime, see the `pecos-qis` crate documentation
    #[must_use]
    pub fn runtime(mut self, runtime: impl crate::runtime::QisRuntime + 'static) -> Self {
        self.runtime = Some(Box::new(runtime));
        self
    }
}

impl Default for QisEngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ClassicalControlEngineBuilder for QisEngineBuilder {
    type Engine = QisEngine;

    #[allow(clippy::too_many_lines)]
    fn build(self) -> Result<Self::Engine, PecosError> {
        log::debug!("QisEngineBuilder::build() called");

        // Check that a runtime was provided
        let runtime = self.runtime.ok_or_else(|| {
            PecosError::Processing(
                "No runtime specified. Please provide a runtime using .runtime().\n\
                Reference runtime:\n\
                - pecos_qis::selene_simple_runtime() - Selene-based implementation\n\
                Example: qis_engine().runtime(selene_simple_runtime()?).build()"
                    .to_string(),
            )
        })?;

        // Dynamic execution: when we have a program source and interface builder,
        // always use dynamic execution which properly handles measurement-dependent conditionals
        if let Some(program_source) = &self.program_source
            && let Some(builder) = &self.interface_builder
        {
            log::debug!(
                "Creating dynamic interface from program source ({} bytes)",
                program_source.len()
            );
            let qis_prog = pecos_programs::Qis::from_string(program_source);
            let dynamic_interface = builder.create_dynamic_interface_from_qis(qis_prog)?;
            log::debug!("Dynamic interface created successfully");

            let mut engine = QisEngine::new(dynamic_interface, runtime);

            // Store the builder and program source so clones can recreate their interfaces
            engine.set_dynamic_config(dyn_clone::clone_box(&**builder), program_source);

            log::debug!(
                "Dynamic engine created, interface present: {}",
                engine.has_interface()
            );
            return Ok(engine);
        }

        // QisEngine requires dynamic execution - OperationCollector alone is not sufficient
        if self.interface.is_some() && self.interface_builder.is_none() {
            return Err(PecosError::Processing(
                "QisEngine requires a dynamic-capable interface for LLVM execution.\n\
                OperationCollector alone is not supported.\n\n\
                Please use .interface() to specify an interface builder, e.g.:\n\
                  use pecos_qis::helios_interface_builder;\n\
                  qis_engine()\n\
                      .interface(helios_interface_builder()?)\n\
                      .program(qis_program)\n\
                      .runtime(selene_simple_runtime()?)\n\
                      .build()"
                    .to_string(),
            ));
        }

        if self.interface_builder.is_some() && self.program_source.is_none() {
            return Err(PecosError::Processing(
                "Interface builder specified but no program provided.\n\
                Please provide a program using .program() or .try_program()"
                    .to_string(),
            ));
        }

        // No interface builder or program - error
        Err(PecosError::Processing(
            "No interface implementation provided. Please specify an interface using:\n\
            - .interface() with a builder like helios_interface_builder()\n\
            - .program() to load a QIS/LLVM program\n\
            - .runtime() to specify the runtime\n\n\
            Example:\n\
              use pecos_qis::{helios_interface_builder, selene_simple_runtime};\n\
              qis_engine()\n\
                  .interface(helios_interface_builder()?)\n\
                  .program(qis_program)\n\
                  .runtime(selene_simple_runtime()?)\n\
                  .build()"
                .to_string(),
        ))
    }
}

/// Convenience function to create a `QisEngineBuilder`
///
/// Creates a builder that requires you to specify both a runtime and a program.
///
/// # Example
///
/// For complete examples with dynamic interface (LLVM execution), see the
/// `pecos-qis` crate documentation which provides `helios_interface_builder()`.
#[must_use]
pub fn qis_engine() -> QisEngineBuilder {
    QisEngineBuilder::new()
}

// Convenience From implementations for common program types
impl<P: IntoQisInterface + 'static> From<P> for QisEngineBuilder {
    fn from(program: P) -> Self {
        qis_engine().program(program)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creation() {
        // Basic builder creation - doesn't require a runtime
        let _builder = qis_engine();
    }

    // Note: Full builder tests with runtime and interface are in integration tests
    // since those require the actual runtime implementations.
}
