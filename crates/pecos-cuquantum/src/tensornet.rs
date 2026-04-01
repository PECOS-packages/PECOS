//! Safe wrapper for cuTensorNet tensor network contraction
//!
//! This module provides a safe Rust API for NVIDIA's cuTensorNet library,
//! which accelerates tensor network contractions on CUDA GPUs.
//!
//! Tensor network methods are used for simulating quantum circuits by
//! contracting tensor networks representing the circuit.

use crate::error::{Result, check_tensornet_status};
use pecos_cuquantum_sys::cutensornetHandle_t;
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
/// ```no_run
/// use pecos_cuquantum::CuTensorNet;
///
/// let sim = CuTensorNet::new().unwrap();
/// // Use for tensor network contractions...
/// ```
pub struct CuTensorNet {
    handle: cutensornetHandle_t,
}

impl CuTensorNet {
    /// Create a new tensor network handle
    ///
    /// # Errors
    /// Returns an error if:
    /// - cuTensorNet initialization fails
    /// - CUDA device is not available
    #[allow(unreachable_code)]
    pub fn new() -> Result<Self> {
        #[cfg(cuquantum_stub)]
        return Err(crate::CuQuantumError::NotAvailable(
            "cuQuantum SDK is not installed. To use GPU-accelerated simulators, install the cuQuantum SDK:\n\
             1. Set CUQUANTUM_ROOT environment variable, or\n\
             2. Install via: pecos install cuquantum, or\n\
             3. Install system-wide to /usr/local/cuquantum/"
                .into(),
        ));

        let mut handle: cutensornetHandle_t = ptr::null_mut();

        // SAFETY: We pass a valid pointer to receive the handle
        unsafe {
            let status = pecos_cuquantum_sys::cutensornetCreate(&mut handle);
            check_tensornet_status(status)?;
        }

        Ok(Self { handle })
    }

    /// Get the cuTensorNet version
    ///
    /// Returns the version as a single integer (e.g., 20000 for version 2.0.0)
    #[must_use]
    pub fn version() -> usize {
        // SAFETY: This is a pure function with no side effects
        unsafe { pecos_cuquantum_sys::cutensornetGetVersion() }
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
                let _ = pecos_cuquantum_sys::cutensornetDestroy(self.handle);
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
