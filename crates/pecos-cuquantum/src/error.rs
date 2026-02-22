//! Error types for cuQuantum operations

use pecos_cuquantum_sys::{
    cudensitymatStatus_t, custabilizerStatus_t, custatevecStatus_t, cutensornetStatus_t,
};
use thiserror::Error;

/// Error type for cuQuantum operations
#[derive(Error, Debug)]
pub enum CuQuantumError {
    /// cuStateVec operation failed
    #[error("cuStateVec error: {0}")]
    StateVec(StateVecError),

    /// cuStabilizer operation failed
    #[error("cuStabilizer error: {0}")]
    Stabilizer(StabilizerError),

    /// cuTensorNet operation failed
    #[error("cuTensorNet error: {0}")]
    TensorNet(TensorNetError),

    /// cuDensityMat operation failed
    #[error("cuDensityMat error: {0}")]
    DensityMat(DensityMatError),

    /// Invalid argument provided
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// Resource allocation failed
    #[error("Allocation failed: {0}")]
    AllocationFailed(String),

    /// Operation not supported
    #[error("Operation not supported: {0}")]
    NotSupported(String),

    /// CUDA error
    #[error("CUDA error: {0}")]
    Cuda(String),

    /// cuQuantum SDK not available
    #[error("cuQuantum SDK not available: {0}")]
    NotAvailable(String),
}

// =============================================================================
// cuStateVec errors
// =============================================================================

/// cuStateVec-specific error
#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateVecError {
    #[error("Not initialized")]
    NotInitialized,

    #[error("Allocation failed")]
    AllocFailed,

    #[error("Invalid value")]
    InvalidValue,

    #[error("Architecture mismatch")]
    ArchMismatch,

    #[error("Execution failed")]
    ExecutionFailed,

    #[error("Internal error")]
    InternalError,

    #[error("Not supported")]
    NotSupported,

    #[error("Insufficient workspace")]
    InsufficientWorkspace,

    #[error("Sampler not preprocessed")]
    SamplerNotPreprocessed,

    #[error("No device allocator")]
    NoDeviceAllocator,

    #[error("Device allocator error")]
    DeviceAllocatorError,

    #[error("Communicator error")]
    CommunicatorError,

    #[error("Loading library failed")]
    LoadingLibraryFailed,

    #[error("Unknown error code: {0}")]
    Unknown(i32),
}

impl From<custatevecStatus_t> for StateVecError {
    fn from(status: custatevecStatus_t) -> Self {
        match status {
            custatevecStatus_t::CUSTATEVEC_STATUS_SUCCESS => {
                // This shouldn't be converted to an error
                Self::Unknown(0)
            }
            custatevecStatus_t::CUSTATEVEC_STATUS_NOT_INITIALIZED => Self::NotInitialized,
            custatevecStatus_t::CUSTATEVEC_STATUS_ALLOC_FAILED => Self::AllocFailed,
            custatevecStatus_t::CUSTATEVEC_STATUS_INVALID_VALUE => Self::InvalidValue,
            custatevecStatus_t::CUSTATEVEC_STATUS_ARCH_MISMATCH => Self::ArchMismatch,
            custatevecStatus_t::CUSTATEVEC_STATUS_EXECUTION_FAILED => Self::ExecutionFailed,
            custatevecStatus_t::CUSTATEVEC_STATUS_INTERNAL_ERROR => Self::InternalError,
            custatevecStatus_t::CUSTATEVEC_STATUS_NOT_SUPPORTED => Self::NotSupported,
            custatevecStatus_t::CUSTATEVEC_STATUS_INSUFFICIENT_WORKSPACE => {
                Self::InsufficientWorkspace
            }
            custatevecStatus_t::CUSTATEVEC_STATUS_SAMPLER_NOT_PREPROCESSED => {
                Self::SamplerNotPreprocessed
            }
            custatevecStatus_t::CUSTATEVEC_STATUS_NO_DEVICE_ALLOCATOR => Self::NoDeviceAllocator,
            custatevecStatus_t::CUSTATEVEC_STATUS_DEVICE_ALLOCATOR_ERROR => {
                Self::DeviceAllocatorError
            }
            custatevecStatus_t::CUSTATEVEC_STATUS_COMMUNICATOR_ERROR => Self::CommunicatorError,
            custatevecStatus_t::CUSTATEVEC_STATUS_LOADING_LIBRARY_FAILED => {
                Self::LoadingLibraryFailed
            }
            // Handle new status codes added in later cuQuantum versions
            _ => Self::Unknown(status as i32),
        }
    }
}

impl From<custatevecStatus_t> for CuQuantumError {
    fn from(status: custatevecStatus_t) -> Self {
        Self::StateVec(StateVecError::from(status))
    }
}

// =============================================================================
// cuStabilizer errors
// =============================================================================

/// cuStabilizer-specific error
#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum StabilizerError {
    #[error("Not initialized")]
    NotInitialized,

    #[error("Allocation failed")]
    AllocFailed,

    #[error("Invalid value")]
    InvalidValue,

    #[error("Architecture mismatch")]
    ArchMismatch,

    #[error("Execution failed")]
    ExecutionFailed,

    #[error("Internal error")]
    InternalError,

    #[error("Not supported")]
    NotSupported,

    #[error("Insufficient workspace")]
    InsufficientWorkspace,

    #[error("Unknown error code: {0}")]
    Unknown(i32),
}

impl From<custabilizerStatus_t> for StabilizerError {
    fn from(status: custabilizerStatus_t) -> Self {
        match status {
            custabilizerStatus_t::CUSTABILIZER_STATUS_SUCCESS => {
                // This shouldn't be converted to an error
                Self::Unknown(0)
            }
            custabilizerStatus_t::CUSTABILIZER_STATUS_NOT_INITIALIZED => Self::NotInitialized,
            custabilizerStatus_t::CUSTABILIZER_STATUS_ALLOC_FAILED => Self::AllocFailed,
            custabilizerStatus_t::CUSTABILIZER_STATUS_INVALID_VALUE => Self::InvalidValue,
            custabilizerStatus_t::CUSTABILIZER_STATUS_INTERNAL_ERROR => Self::InternalError,
            custabilizerStatus_t::CUSTABILIZER_STATUS_NOT_SUPPORTED => Self::NotSupported,
            // Handle other status codes (including new ones added in later versions)
            _ => Self::Unknown(status as i32),
        }
    }
}

impl From<custabilizerStatus_t> for CuQuantumError {
    fn from(status: custabilizerStatus_t) -> Self {
        Self::Stabilizer(StabilizerError::from(status))
    }
}

// =============================================================================
// cuTensorNet errors
// =============================================================================

/// cuTensorNet-specific error
#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorNetError {
    #[error("Not initialized")]
    NotInitialized,

    #[error("Allocation failed")]
    AllocFailed,

    #[error("Invalid value")]
    InvalidValue,

    #[error("Architecture mismatch")]
    ArchMismatch,

    #[error("Mapping error")]
    MappingError,

    #[error("Execution failed")]
    ExecutionFailed,

    #[error("Internal error")]
    InternalError,

    #[error("Not supported")]
    NotSupported,

    #[error("License error")]
    LicenseError,

    #[error("cuBLAS error")]
    CublasError,

    #[error("CUDA error")]
    CudaError,

    #[error("Insufficient workspace")]
    InsufficientWorkspace,

    #[error("Insufficient driver")]
    InsufficientDriver,

    #[error("I/O error")]
    IoError,

    #[error("cuTensor error")]
    CutensorError,

    #[error("Unknown error code: {0}")]
    Unknown(i32),
}

impl From<cutensornetStatus_t> for TensorNetError {
    fn from(status: cutensornetStatus_t) -> Self {
        match status {
            cutensornetStatus_t::CUTENSORNET_STATUS_SUCCESS => Self::Unknown(0),
            cutensornetStatus_t::CUTENSORNET_STATUS_NOT_INITIALIZED => Self::NotInitialized,
            cutensornetStatus_t::CUTENSORNET_STATUS_ALLOC_FAILED => Self::AllocFailed,
            cutensornetStatus_t::CUTENSORNET_STATUS_INVALID_VALUE => Self::InvalidValue,
            cutensornetStatus_t::CUTENSORNET_STATUS_ARCH_MISMATCH => Self::ArchMismatch,
            cutensornetStatus_t::CUTENSORNET_STATUS_MAPPING_ERROR => Self::MappingError,
            cutensornetStatus_t::CUTENSORNET_STATUS_EXECUTION_FAILED => Self::ExecutionFailed,
            cutensornetStatus_t::CUTENSORNET_STATUS_INTERNAL_ERROR => Self::InternalError,
            cutensornetStatus_t::CUTENSORNET_STATUS_NOT_SUPPORTED => Self::NotSupported,
            cutensornetStatus_t::CUTENSORNET_STATUS_LICENSE_ERROR => Self::LicenseError,
            cutensornetStatus_t::CUTENSORNET_STATUS_CUBLAS_ERROR => Self::CublasError,
            cutensornetStatus_t::CUTENSORNET_STATUS_CUDA_ERROR => Self::CudaError,
            cutensornetStatus_t::CUTENSORNET_STATUS_INSUFFICIENT_WORKSPACE => {
                Self::InsufficientWorkspace
            }
            cutensornetStatus_t::CUTENSORNET_STATUS_INSUFFICIENT_DRIVER => Self::InsufficientDriver,
            cutensornetStatus_t::CUTENSORNET_STATUS_IO_ERROR => Self::IoError,
            // Handle other status codes (including new ones added in later versions)
            _ => Self::Unknown(status as i32),
        }
    }
}

impl From<cutensornetStatus_t> for CuQuantumError {
    fn from(status: cutensornetStatus_t) -> Self {
        Self::TensorNet(TensorNetError::from(status))
    }
}

// =============================================================================
// cuDensityMat errors
// =============================================================================

/// cuDensityMat-specific error
#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum DensityMatError {
    #[error("Not initialized")]
    NotInitialized,

    #[error("Allocation failed")]
    AllocFailed,

    #[error("Invalid value")]
    InvalidValue,

    #[error("Architecture mismatch")]
    ArchMismatch,

    #[error("Execution failed")]
    ExecutionFailed,

    #[error("Internal error")]
    InternalError,

    #[error("Not supported")]
    NotSupported,

    #[error("cuBLAS error")]
    CublasError,

    #[error("CUDA error")]
    CudaError,

    #[error("Insufficient workspace")]
    InsufficientWorkspace,

    #[error("Unknown error code: {0}")]
    Unknown(i32),
}

impl From<cudensitymatStatus_t> for DensityMatError {
    fn from(status: cudensitymatStatus_t) -> Self {
        match status {
            cudensitymatStatus_t::CUDENSITYMAT_STATUS_SUCCESS => Self::Unknown(0),
            cudensitymatStatus_t::CUDENSITYMAT_STATUS_NOT_INITIALIZED => Self::NotInitialized,
            cudensitymatStatus_t::CUDENSITYMAT_STATUS_ALLOC_FAILED => Self::AllocFailed,
            cudensitymatStatus_t::CUDENSITYMAT_STATUS_INVALID_VALUE => Self::InvalidValue,
            cudensitymatStatus_t::CUDENSITYMAT_STATUS_ARCH_MISMATCH => Self::ArchMismatch,
            cudensitymatStatus_t::CUDENSITYMAT_STATUS_EXECUTION_FAILED => Self::ExecutionFailed,
            cudensitymatStatus_t::CUDENSITYMAT_STATUS_INTERNAL_ERROR => Self::InternalError,
            cudensitymatStatus_t::CUDENSITYMAT_STATUS_NOT_SUPPORTED => Self::NotSupported,
            cudensitymatStatus_t::CUDENSITYMAT_STATUS_CUBLAS_ERROR => Self::CublasError,
            cudensitymatStatus_t::CUDENSITYMAT_STATUS_CUDA_ERROR => Self::CudaError,
            cudensitymatStatus_t::CUDENSITYMAT_STATUS_INSUFFICIENT_WORKSPACE => {
                Self::InsufficientWorkspace
            }
            // Handle new status codes added in later cuQuantum versions
            _ => Self::Unknown(status as i32),
        }
    }
}

impl From<cudensitymatStatus_t> for CuQuantumError {
    fn from(status: cudensitymatStatus_t) -> Self {
        Self::DensityMat(DensityMatError::from(status))
    }
}

// =============================================================================
// TryClone trait
// =============================================================================

/// A trait for types that can be cloned but may fail.
///
/// This is useful for GPU types where cloning requires allocating device memory
/// and performing memory copies, which can fail due to CUDA errors.
///
/// # Example
///
/// ```ignore
/// use pecos_cuquantum::{CuStateVec, TryClone};
///
/// let sim = CuStateVec::new(4)?;
/// let cloned = sim.try_clone()?;  // Returns Result, doesn't panic
/// ```
pub trait TryClone: Sized {
    /// Attempt to clone this value.
    ///
    /// # Errors
    /// Returns an error if the clone operation fails (e.g., CUDA memory allocation fails).
    fn try_clone(&self) -> Result<Self>;
}

// =============================================================================
// Result types and helpers
// =============================================================================

/// Result type for cuQuantum operations
pub type Result<T> = std::result::Result<T, CuQuantumError>;

/// Check a cuStateVec status and convert to Result
#[inline]
pub fn check_status(status: custatevecStatus_t) -> Result<()> {
    if pecos_cuquantum_sys::is_success(status) {
        Ok(())
    } else {
        Err(CuQuantumError::from(status))
    }
}

/// Check a cuStabilizer status and convert to Result
#[inline]
pub fn check_stabilizer_status(status: custabilizerStatus_t) -> Result<()> {
    if pecos_cuquantum_sys::stabilizer_is_success(status) {
        Ok(())
    } else {
        Err(CuQuantumError::from(status))
    }
}

/// Check a cuTensorNet status and convert to Result
#[inline]
pub fn check_tensornet_status(status: cutensornetStatus_t) -> Result<()> {
    if pecos_cuquantum_sys::tensornet_is_success(status) {
        Ok(())
    } else {
        Err(CuQuantumError::from(status))
    }
}

/// Check a cuDensityMat status and convert to Result
#[inline]
pub fn check_densitymat_status(status: cudensitymatStatus_t) -> Result<()> {
    if pecos_cuquantum_sys::densitymat_is_success(status) {
        Ok(())
    } else {
        Err(CuQuantumError::from(status))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_conversion() {
        let err = StateVecError::from(custatevecStatus_t::CUSTATEVEC_STATUS_INVALID_VALUE);
        assert_eq!(err, StateVecError::InvalidValue);
    }

    #[test]
    fn test_check_status_success() {
        let result = check_status(custatevecStatus_t::CUSTATEVEC_STATUS_SUCCESS);
        assert!(result.is_ok());
    }

    #[test]
    fn test_check_status_error() {
        let result = check_status(custatevecStatus_t::CUSTATEVEC_STATUS_INVALID_VALUE);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_display() {
        let err = CuQuantumError::StateVec(StateVecError::InvalidValue);
        let msg = format!("{err}");
        assert!(msg.contains("Invalid value"));
    }

    #[test]
    fn test_stabilizer_status_conversion() {
        let err = StabilizerError::from(custabilizerStatus_t::CUSTABILIZER_STATUS_INVALID_VALUE);
        assert_eq!(err, StabilizerError::InvalidValue);
    }

    #[test]
    fn test_check_stabilizer_status_success() {
        let result = check_stabilizer_status(custabilizerStatus_t::CUSTABILIZER_STATUS_SUCCESS);
        assert!(result.is_ok());
    }

    #[test]
    fn test_check_stabilizer_status_error() {
        let result =
            check_stabilizer_status(custabilizerStatus_t::CUSTABILIZER_STATUS_INVALID_VALUE);
        assert!(result.is_err());
    }

    #[test]
    fn test_stabilizer_error_display() {
        let err = CuQuantumError::Stabilizer(StabilizerError::InvalidValue);
        let msg = format!("{err}");
        assert!(msg.contains("Invalid value"));
    }

    #[test]
    fn test_tensornet_status_conversion() {
        let err = TensorNetError::from(cutensornetStatus_t::CUTENSORNET_STATUS_INVALID_VALUE);
        assert_eq!(err, TensorNetError::InvalidValue);
    }

    #[test]
    fn test_check_tensornet_status_success() {
        let result = check_tensornet_status(cutensornetStatus_t::CUTENSORNET_STATUS_SUCCESS);
        assert!(result.is_ok());
    }

    #[test]
    fn test_check_tensornet_status_error() {
        let result = check_tensornet_status(cutensornetStatus_t::CUTENSORNET_STATUS_INVALID_VALUE);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensornet_error_display() {
        let err = CuQuantumError::TensorNet(TensorNetError::InvalidValue);
        let msg = format!("{err}");
        assert!(msg.contains("Invalid value"));
    }

    #[test]
    fn test_densitymat_status_conversion() {
        let err = DensityMatError::from(cudensitymatStatus_t::CUDENSITYMAT_STATUS_INVALID_VALUE);
        assert_eq!(err, DensityMatError::InvalidValue);
    }

    #[test]
    fn test_check_densitymat_status_success() {
        let result = check_densitymat_status(cudensitymatStatus_t::CUDENSITYMAT_STATUS_SUCCESS);
        assert!(result.is_ok());
    }

    #[test]
    fn test_check_densitymat_status_error() {
        let result =
            check_densitymat_status(cudensitymatStatus_t::CUDENSITYMAT_STATUS_INVALID_VALUE);
        assert!(result.is_err());
    }

    #[test]
    fn test_densitymat_error_display() {
        let err = CuQuantumError::DensityMat(DensityMatError::InvalidValue);
        let msg = format!("{err}");
        assert!(msg.contains("Invalid value"));
    }
}
