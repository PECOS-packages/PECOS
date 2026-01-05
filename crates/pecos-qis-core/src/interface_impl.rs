//! Interface trait utilities
//!
//! This module provides utilities for working with `QisInterface` implementations.

use crate::qis_interface::InterfaceError;
use pecos_core::prelude::PecosError;

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
