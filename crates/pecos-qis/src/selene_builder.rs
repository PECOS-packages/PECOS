//! Helios Interface Builder
//!
//! Builder pattern for creating Helios-based `QisInterfaces`.

use crate::QisHeliosInterface;
use crate::program::QisInterfaceBuilder;
use crate::qis_interface::{ProgramFormat, QisInterface};
use pecos_core::errors::PecosError;
use pecos_programs::{Qis, QisContent};
use pecos_qis_ffi_types::OperationCollector;

/// Helios-based interface builder
///
/// This builder creates `QisHeliosInterface` instances from various program formats.
#[derive(Debug, Clone)]
pub struct HeliosInterfaceBuilder;

impl HeliosInterfaceBuilder {
    /// Create a new Helios interface builder
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for HeliosInterfaceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl QisInterfaceBuilder for HeliosInterfaceBuilder {
    fn build_from_qis_program(&self, program: Qis) -> Result<OperationCollector, PecosError> {
        let mut interface = QisHeliosInterface::new();

        // Load the program into the interface
        match &program.content {
            QisContent::Ir(ir_text) => {
                interface
                    .load_program(ir_text.as_bytes(), ProgramFormat::LlvmIrText)
                    .map_err(|e| {
                        PecosError::Processing(format!(
                            "Failed to load QIS program into Helios interface: {e}"
                        ))
                    })?;
            }
            QisContent::Bitcode(bitcode) => {
                interface
                    .load_program(bitcode, ProgramFormat::QisBitcode)
                    .map_err(|e| {
                        PecosError::Processing(format!(
                            "Failed to load QIS bitcode into Helios interface: {e}"
                        ))
                    })?;
            }
        }

        // Collect operations using the interface trait method
        interface.collect_operations().map_err(|e| {
            PecosError::Processing(format!(
                "Failed to collect operations from Helios interface: {e}"
            ))
        })
    }

    fn build_from_interface(
        &self,
        interface: OperationCollector,
    ) -> Result<OperationCollector, PecosError> {
        // Already an OperationCollector, just return it
        Ok(interface)
    }

    fn name(&self) -> &'static str {
        "HeliosInterfaceBuilder"
    }

    fn create_dynamic_interface_from_qis(
        &self,
        program: Qis,
    ) -> Result<crate::qis_interface::BoxedInterface, PecosError> {
        let mut interface = QisHeliosInterface::new();

        // Load the program into the interface WITHOUT collecting operations
        match &program.content {
            QisContent::Ir(ir_text) => {
                interface
                    .load_program(ir_text.as_bytes(), ProgramFormat::LlvmIrText)
                    .map_err(|e| {
                        PecosError::Processing(format!(
                            "Failed to load QIS program into Helios interface: {e}"
                        ))
                    })?;
            }
            QisContent::Bitcode(bitcode) => {
                interface
                    .load_program(bitcode, ProgramFormat::QisBitcode)
                    .map_err(|e| {
                        PecosError::Processing(format!(
                            "Failed to load QIS bitcode into Helios interface: {e}"
                        ))
                    })?;
            }
        }

        // Return the interface without collecting operations - the engine will do that dynamically
        Ok(Box::new(interface))
    }
}

/// Convenience function to create a Helios interface builder
#[must_use]
pub fn helios_interface_builder() -> HeliosInterfaceBuilder {
    HeliosInterfaceBuilder::new()
}
