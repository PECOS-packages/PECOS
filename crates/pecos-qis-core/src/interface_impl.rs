//! Interface trait and implementations
//!
//! This module provides implementations of the `QisInterface` trait.

use crate::qis_interface::{InterfaceError, ProgramFormat, QisInterface};
use pecos_core::prelude::PecosError;
use pecos_qis_ffi_types::OperationCollector;
use std::collections::BTreeMap;

/// Simple wrapper for pre-built operation lists
///
/// This allows pre-built `OperationCollector` instances to be used directly
/// with the `QisEngine` without needing compilation.
pub struct SimpleQisInterface {
    operations: OperationCollector,
}

impl SimpleQisInterface {
    /// Create a new `SimpleQisInterface` from a pre-built operations list
    #[must_use]
    pub fn new(operations: OperationCollector) -> Self {
        Self { operations }
    }
}

impl QisInterface for SimpleQisInterface {
    fn load_program(
        &mut self,
        _program_bytes: &[u8],
        _format: ProgramFormat,
    ) -> Result<(), InterfaceError> {
        // Pre-built interface doesn't need to load programs
        Ok(())
    }

    fn collect_operations(&mut self) -> Result<OperationCollector, InterfaceError> {
        // Return the pre-built operations
        Ok(self.operations.clone())
    }

    fn execute_with_measurements(
        &mut self,
        _measurements: BTreeMap<usize, bool>,
    ) -> Result<OperationCollector, InterfaceError> {
        // For pre-built interfaces, just return the operations as-is
        // since there are no conditional paths
        Ok(self.operations.clone())
    }

    fn name(&self) -> &'static str {
        "Simple (Pre-built)"
    }

    fn reset(&mut self) -> Result<(), InterfaceError> {
        // Nothing to reset for pre-built interface
        Ok(())
    }
}

/// Convert `InterfaceError` to `PecosError`
#[must_use]
pub fn interface_error_to_pecos(err: InterfaceError) -> PecosError {
    match err {
        InterfaceError::LoadError(msg) => PecosError::Generic(format!("Load error: {msg}")),
        InterfaceError::ExecutionError(msg) => {
            PecosError::Generic(format!("Execution error: {msg}"))
        }
        InterfaceError::InvalidFormat(msg) => PecosError::Generic(format!("Invalid format: {msg}")),
        InterfaceError::Other(msg) => PecosError::Generic(msg),
    }
}
