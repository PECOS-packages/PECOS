//! Raw FFI bindings to NVIDIA cuQuantum SDK
//!
//! This crate provides low-level bindings to cuQuantum, NVIDIA's SDK for
//! accelerated quantum circuit simulation on GPUs.
//!
//! # Components
//!
//! cuQuantum includes several libraries:
//!
//! - **cuStateVec**: State vector simulation
//! - **cuStabilizer**: Stabilizer/Clifford circuit simulation
//! - **cuTensorNet**: Tensor network contraction (future)
//! - **cuDensityMat**: Density matrix simulation (future)
//!
//! # Usage
//!
//! These are raw FFI bindings. For a safe Rust API, use the `pecos-cuquantum` crate instead.
//!
//! ```text
//! use pecos_cuquantum_sys::*;
//! use std::ptr;
//!
//! unsafe {
//!     let mut handle: custatevecHandle_t = ptr::null_mut();
//!     let status = custatevecCreate(&mut handle);
//!     assert_eq!(status, custatevecStatus_t::CUSTATEVEC_STATUS_SUCCESS);
//!
//!     // ... use handle ...
//!
//!     custatevecDestroy(handle);
//! }
//! ```
//!
//! # Requirements
//!
//! - NVIDIA GPU with CUDA support
//! - CUDA Toolkit installed
//! - cuQuantum SDK installed (auto-detected from `~/.pecos/cuquantum/` or `CUQUANTUM_ROOT`)
//!
//! # Building
//!
//! If cuQuantum is not found at build time, stub bindings are generated that
//! allow compilation but will fail at link time if the functions are called.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(clippy::useless_transmute)]
#![allow(clippy::too_many_arguments)]

// Type definitions from bindgen (real cuQuantum headers) or stubs
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// Runtime library loader
pub mod loader;
pub use loader::{CuQuantumBackend, CuQuantumLoadError, is_available, try_load};

/// Check if a cuStateVec status indicates success
#[inline]
#[must_use]
pub fn is_success(status: custatevecStatus_t) -> bool {
    status == custatevecStatus_t::CUSTATEVEC_STATUS_SUCCESS
}

/// Convert a cuStateVec status to a Result
///
/// # Errors
/// Returns the status code if it indicates an error.
#[inline]
pub fn status_to_result(status: custatevecStatus_t) -> Result<(), custatevecStatus_t> {
    if is_success(status) {
        Ok(())
    } else {
        Err(status)
    }
}

// =============================================================================
// cuStabilizer helpers
// =============================================================================

/// Check if a cuStabilizer status indicates success
#[inline]
#[must_use]
pub fn stabilizer_is_success(status: custabilizerStatus_t) -> bool {
    status == custabilizerStatus_t::CUSTABILIZER_STATUS_SUCCESS
}

/// Convert a cuStabilizer status to a Result
///
/// # Errors
/// Returns the status code if it indicates an error.
#[inline]
pub fn stabilizer_status_to_result(
    status: custabilizerStatus_t,
) -> Result<(), custabilizerStatus_t> {
    if stabilizer_is_success(status) {
        Ok(())
    } else {
        Err(status)
    }
}

// =============================================================================
// cuTensorNet helpers
// =============================================================================

/// Check if a cuTensorNet status indicates success
#[inline]
#[must_use]
pub fn tensornet_is_success(status: cutensornetStatus_t) -> bool {
    status == cutensornetStatus_t::CUTENSORNET_STATUS_SUCCESS
}

/// Convert a cuTensorNet status to a Result
///
/// # Errors
/// Returns the status code if it indicates an error.
#[inline]
pub fn tensornet_status_to_result(status: cutensornetStatus_t) -> Result<(), cutensornetStatus_t> {
    if tensornet_is_success(status) {
        Ok(())
    } else {
        Err(status)
    }
}

// =============================================================================
// cuDensityMat helpers
// =============================================================================

/// Check if a cuDensityMat status indicates success
#[inline]
#[must_use]
pub fn densitymat_is_success(status: cudensitymatStatus_t) -> bool {
    status == cudensitymatStatus_t::CUDENSITYMAT_STATUS_SUCCESS
}

/// Convert a cuDensityMat status to a Result
///
/// # Errors
/// Returns the status code if it indicates an error.
#[inline]
pub fn densitymat_status_to_result(
    status: cudensitymatStatus_t,
) -> Result<(), cudensitymatStatus_t> {
    if densitymat_is_success(status) {
        Ok(())
    } else {
        Err(status)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_success() {
        assert!(is_success(custatevecStatus_t::CUSTATEVEC_STATUS_SUCCESS));
        assert!(!is_success(
            custatevecStatus_t::CUSTATEVEC_STATUS_INVALID_VALUE
        ));
    }

    #[test]
    fn test_status_to_result() {
        assert!(status_to_result(custatevecStatus_t::CUSTATEVEC_STATUS_SUCCESS).is_ok());
        assert!(status_to_result(custatevecStatus_t::CUSTATEVEC_STATUS_INVALID_VALUE).is_err());
    }

    #[test]
    fn test_complex_types() {
        // Verify complex types have expected layout
        let c32 = cuComplex { x: 1.0, y: 2.0 };
        assert_eq!(c32.x, 1.0);
        assert_eq!(c32.y, 2.0);

        let c64 = cuDoubleComplex { x: 1.0, y: 2.0 };
        assert_eq!(c64.x, 1.0);
        assert_eq!(c64.y, 2.0);
    }

    #[test]
    fn test_stabilizer_status_success() {
        assert!(stabilizer_is_success(
            custabilizerStatus_t::CUSTABILIZER_STATUS_SUCCESS
        ));
        assert!(!stabilizer_is_success(
            custabilizerStatus_t::CUSTABILIZER_STATUS_INVALID_VALUE
        ));
    }

    #[test]
    fn test_stabilizer_status_to_result() {
        assert!(
            stabilizer_status_to_result(custabilizerStatus_t::CUSTABILIZER_STATUS_SUCCESS).is_ok()
        );
        assert!(
            stabilizer_status_to_result(custabilizerStatus_t::CUSTABILIZER_STATUS_INVALID_VALUE)
                .is_err()
        );
    }

    // Note: custabilizerPauli_t no longer exists in cuQuantum 25.11+
    // The cuStabilizer API was redesigned to use a circuit-based "frame simulation" model

    #[test]
    fn test_tensornet_status_success() {
        assert!(tensornet_is_success(
            cutensornetStatus_t::CUTENSORNET_STATUS_SUCCESS
        ));
        assert!(!tensornet_is_success(
            cutensornetStatus_t::CUTENSORNET_STATUS_INVALID_VALUE
        ));
    }

    #[test]
    fn test_tensornet_status_to_result() {
        assert!(
            tensornet_status_to_result(cutensornetStatus_t::CUTENSORNET_STATUS_SUCCESS).is_ok()
        );
        assert!(
            tensornet_status_to_result(cutensornetStatus_t::CUTENSORNET_STATUS_INVALID_VALUE)
                .is_err()
        );
    }

    #[test]
    fn test_densitymat_status_success() {
        assert!(densitymat_is_success(
            cudensitymatStatus_t::CUDENSITYMAT_STATUS_SUCCESS
        ));
        assert!(!densitymat_is_success(
            cudensitymatStatus_t::CUDENSITYMAT_STATUS_INVALID_VALUE
        ));
    }

    #[test]
    fn test_densitymat_status_to_result() {
        assert!(
            densitymat_status_to_result(cudensitymatStatus_t::CUDENSITYMAT_STATUS_SUCCESS).is_ok()
        );
        assert!(
            densitymat_status_to_result(cudensitymatStatus_t::CUDENSITYMAT_STATUS_INVALID_VALUE)
                .is_err()
        );
    }
}
