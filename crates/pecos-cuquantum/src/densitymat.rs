//! Safe wrapper for cuDensityMat density matrix simulation
//!
//! This module provides a safe Rust API for NVIDIA's cuDensityMat library,
//! which accelerates density matrix operations on CUDA GPUs.
//!
//! Density matrix simulation allows representing mixed quantum states,
//! which is essential for simulating open quantum systems with noise
//! and decoherence.

use crate::CuQuantumError;
use crate::error::{Result, check_densitymat_status};
use pecos_cuquantum_sys::{
    CuQuantumBackend, cudaDataType_t, cudensitymatHandle_t, cudensitymatState_t,
    cudensitymatStatePurity_t,
};
use std::ptr;

/// Density matrix simulator using NVIDIA cuDensityMat
///
/// This struct manages a cuDensityMat handle and state, providing methods for
/// density matrix operations including noisy quantum simulation.
///
/// # Advantages over State Vector
///
/// - Can represent mixed states (statistical mixtures)
/// - Natural representation for noise and decoherence
/// - Essential for open quantum system simulation
///
/// # Memory Requirements
///
/// Density matrices require O(4^n) memory vs O(2^n) for state vectors,
/// limiting practical simulation to fewer qubits.
///
/// # Example
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use pecos_cuquantum::CuDensityMat;
///
/// let sim = CuDensityMat::new(4)?;
/// # Ok(())
/// # }
/// ```
pub struct CuDensityMat {
    backend: &'static CuQuantumBackend,
    handle: cudensitymatHandle_t,
    state: cudensitymatState_t,
    num_qubits: usize,
}

impl CuDensityMat {
    /// Create a new density matrix simulator
    ///
    /// Initializes the state to the pure state |0...0><0...0|.
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits to simulate
    ///
    /// # Errors
    /// Returns an error if:
    /// - cuDensityMat initialization fails
    /// - CUDA device is not available
    /// - Memory allocation fails
    pub fn new(num_qubits: usize) -> Result<Self> {
        if num_qubits == 0 {
            return Err(CuQuantumError::InvalidArgument(
                "num_qubits must be at least 1".into(),
            ));
        }

        let backend = pecos_cuquantum_sys::try_load().map_err(CuQuantumError::from)?;

        let mut handle: cudensitymatHandle_t = ptr::null_mut();
        let mut state: cudensitymatState_t = ptr::null_mut();

        // SAFETY: We pass valid pointers to receive the handle and state
        unsafe {
            let status = (backend.cudensitymatCreate)(&mut handle);
            check_densitymat_status(status)?;

            // Create space mode extents - each qubit has dimension 2
            let space_mode_extents: Vec<i64> = vec![2i64; num_qubits];

            // Create a pure state (ket-bra representation)
            // New API requires numSpaceModes, spaceModeExtents, and batchSize
            let status = (backend.cudensitymatCreateState)(
                handle,
                cudensitymatStatePurity_t::CUDENSITYMAT_STATE_PURITY_PURE,
                num_qubits as i32,           // numSpaceModes
                space_mode_extents.as_ptr(), // spaceModeExtents
                1,                           // batchSize
                cudaDataType_t::CUDA_C_64F,  // Complex double precision
                &mut state,
            );
            if !pecos_cuquantum_sys::densitymat_is_success(status) {
                // Clean up handle if state creation failed
                let _ = (backend.cudensitymatDestroy)(handle);
                check_densitymat_status(status)?;
            }
        }

        Ok(Self {
            backend,
            handle,
            state,
            num_qubits,
        })
    }

    /// Get the number of qubits
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Get the cuDensityMat version
    ///
    /// Returns the version as a single integer, or 0 if the library is not available.
    #[must_use]
    pub fn version() -> usize {
        if let Ok(backend) = pecos_cuquantum_sys::try_load() {
            // SAFETY: This is a pure function with no side effects
            unsafe { (backend.cudensitymatGetVersion)() }
        } else {
            0
        }
    }

    /// Get the raw handle for advanced usage
    ///
    /// # Safety
    /// The caller must not destroy the handle or use it after this
    /// `CuDensityMat` instance is dropped.
    #[must_use]
    pub fn raw_handle(&self) -> cudensitymatHandle_t {
        self.handle
    }

    /// Get the raw state for advanced usage
    ///
    /// # Safety
    /// The caller must not destroy the state or use it after this
    /// `CuDensityMat` instance is dropped.
    #[must_use]
    pub fn raw_state(&self) -> cudensitymatState_t {
        self.state
    }
}

impl Drop for CuDensityMat {
    fn drop(&mut self) {
        // SAFETY: We own the handle and state, and they're valid
        unsafe {
            if !self.state.is_null() {
                let _ = (self.backend.cudensitymatDestroyState)(self.state);
            }
            if !self.handle.is_null() {
                let _ = (self.backend.cudensitymatDestroy)(self.handle);
            }
        }
    }
}

// CuDensityMat is not Send/Sync because CUDA handles are typically thread-local

// Note: Clone is not implemented for CuDensityMat because the cuDensityMat API
// does not provide a state copy function. The state is an opaque handle managed
// by the library, and there's no documented way to duplicate it without access
// to the underlying GPU memory layout.

#[cfg(test)]
mod tests {
    #[test]
    fn test_memory_requirements() {
        // Density matrix needs O(4^n) memory
        // For 10 qubits: 4^10 = 1M complex numbers = 16MB (complex64)
        // For 20 qubits: 4^20 = 1T complex numbers = 16TB (too large!)
        let n = 10usize;
        let density_matrix_size = 4usize.pow(n as u32);
        let state_vector_size = 2usize.pow(n as u32);
        assert_eq!(density_matrix_size, state_vector_size * state_vector_size);
    }

    // Note: Tests that create CuDensityMat instances are in integration tests
    // because they require cuDensityMat to be available at link time.
}
