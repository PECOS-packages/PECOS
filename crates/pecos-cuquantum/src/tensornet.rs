//! Safe wrapper for cuTensorNet tensor network contraction
//!
//! This module provides a safe Rust API for NVIDIA's cuTensorNet library,
//! which accelerates tensor network contractions on CUDA GPUs.
//!
//! Tensor network methods are used for simulating quantum circuits by
//! contracting tensor networks representing the circuit.

use crate::error::{Result, check_tensornet_status};
use pecos_cuquantum_sys::{CuQuantumBackend, cutensornetHandle_t};
use std::ptr;

/// Tensor network simulator using NVIDIA cuTensorNet
///
/// This struct manages a cuTensorNet handle, providing methods for
/// tensor network creation and contraction.
///
/// # Use Cases
///
/// - Simulating quantum circuits with many qubits but shallow depth
/// - Calculating expectation values
/// - Approximate simulation of larger circuits
///
/// # Example
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use pecos_cuquantum::CuTensorNet;
///
/// let sim = CuTensorNet::new()?;
/// # Ok(())
/// # }
/// ```
pub struct CuTensorNet {
    backend: &'static CuQuantumBackend,
    handle: cutensornetHandle_t,
}

impl CuTensorNet {
    /// Create a new tensor network handle
    ///
    /// # Errors
    /// Returns an error if:
    /// - cuTensorNet initialization fails
    /// - CUDA device is not available
    pub fn new() -> Result<Self> {
        let backend = pecos_cuquantum_sys::try_load().map_err(crate::CuQuantumError::from)?;

        let mut handle: cutensornetHandle_t = ptr::null_mut();

        // SAFETY: We pass a valid pointer to receive the handle
        unsafe {
            let status = (backend.cutensornetCreate)(&mut handle);
            check_tensornet_status(status)?;
        }

        Ok(Self { backend, handle })
    }

    /// Get the cuTensorNet version
    ///
    /// Returns the version as a single integer (e.g., 20000 for version 2.0.0),
    /// or 0 if the library is not available.
    #[must_use]
    pub fn version() -> usize {
        if let Ok(backend) = pecos_cuquantum_sys::try_load() {
            // SAFETY: This is a pure function with no side effects
            unsafe { (backend.cutensornetGetVersion)() }
        } else {
            0
        }
    }

    /// Get the raw handle for advanced usage
    ///
    /// # Safety
    /// The caller must not destroy the handle or use it after this
    /// `CuTensorNet` instance is dropped.
    #[must_use]
    pub fn raw_handle(&self) -> cutensornetHandle_t {
        self.handle
    }
}

impl Drop for CuTensorNet {
    fn drop(&mut self) {
        // SAFETY: We own the handle and it's valid
        unsafe {
            if !self.handle.is_null() {
                let _ = (self.backend.cutensornetDestroy)(self.handle);
            }
        }
    }
}

// CuTensorNet is not Send/Sync because CUDA handles are typically thread-local

#[cfg(test)]
mod tests {
    use crate::CuQuantumError;

    #[test]
    fn test_error_accessible() {
        // Test that CuQuantumError can be constructed with TensorNet variant
        let err = CuQuantumError::InvalidArgument("test".into());
        let msg = format!("{err}");
        assert!(msg.contains("test"));
    }
}
