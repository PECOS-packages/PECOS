//! Safe Rust wrapper for NVIDIA cuQuantum quantum simulation SDK
//!
//! This crate provides a safe, idiomatic Rust API for NVIDIA's cuQuantum SDK,
//! which accelerates quantum circuit simulation on CUDA GPUs.
//!
//! # Features
//!
//! - **State vector simulation** via [`CuStateVec`]
//! - **Stabilizer simulation** via [`CuStabilizer`]
//! - Implements PECOS traits ([`QuantumSimulator`], [`CliffordGateable`], [`ArbitraryRotationGateable`])
//! - Standard quantum gates (H, X, Y, Z, S, T, RX, RY, RZ, CX, CZ, SWAP)
//! - Measurement and sampling
//!
//! # Choosing a Simulator
//!
//! | Simulator | Qubits | Gates | Memory |
//! |-----------|--------|-------|--------|
//! | [`CuStateVec`] | ~30 | All | O(2^n) |
//! | [`CuStabilizer`] | 1000s | Clifford only | O(n^2) |
//!
//! # Requirements
//!
//! - NVIDIA GPU with CUDA support
//! - CUDA Toolkit installed
//! - cuQuantum SDK installed
//!
//! # Example
//!
//! ```
//! use pecos_cuquantum::CuStateVec;
//! use pecos_simulators::{QuantumSimulator, CliffordGateable};
//! use pecos_core::QubitId;
//!
//! fn main() -> pecos_cuquantum::Result<()> {
//!     let mut sim = CuStateVec::new(4)?;
//!
//!     // Create a Bell state
//!     sim.h(&[QubitId(0)]);
//!     sim.cx(&[(QubitId(0), QubitId(1))]);
//!
//!     // Measure
//!     let results = sim.mz(&[QubitId(0), QubitId(1)]);
//!
//!     // In a Bell state, q0 and q1 should always be correlated
//!     println!("q0={}, q1={}", results[0].outcome, results[1].outcome);
//!
//!     Ok(())
//! }
//! ```
//!
//! # Runtime Loading
//!
//! The underlying `pecos-cuquantum-sys` crate loads cuQuantum shared libraries
//! at runtime via `libloading`. Code will always compile and link, but constructors
//! (e.g., `CuStateVec::new()`) will return `Err(CuQuantumError::NotAvailable(...))`
//! if the libraries cannot be found. Use [`is_cuquantum_available()`] to check
//! at runtime whether the SDK is available.

pub mod densitymat;
pub mod error;
pub mod stabilizer;
pub mod statevec;
pub mod tensornet;

// Re-export main types
pub use densitymat::CuDensityMat;
pub use error::{
    CuQuantumError, DensityMatError, Result, StabilizerError, StateVecError, TensorNetError,
    TryClone,
};
pub use stabilizer::{CuFrameSimulator, CuStabilizer, FrameSimulationResults};
pub use statevec::CuStateVec;
pub use tensornet::CuTensorNet;

// Re-export PECOS traits for convenience
pub use pecos_core::QubitId;
pub use pecos_simulators::stabilizer_test_utils::{ForcedMeasurement, StabilizerSimulator};
pub use pecos_simulators::{
    ArbitraryRotationGateable, CliffordGateable, MeasurementResult, QuantumSimulator,
};

/// Check if cuQuantum is available at runtime
///
/// Returns `true` if the cuQuantum SDK libraries can be loaded.
/// When this returns `false`, constructors like `CuStateVec::new()` will return
/// `Err(CuQuantumError::NotAvailable(...))` with installation instructions.
#[must_use]
pub fn is_cuquantum_available() -> bool {
    pecos_cuquantum_sys::is_available()
}

/// Check if the cuStateVec backend can create a simulator on this machine.
///
/// This is stricter than [`is_cuquantum_available`]: it verifies not only that
/// cuQuantum libraries can be loaded, but also that a CUDA device/runtime can
/// initialize the cuStateVec handle and allocate a minimal state vector.
#[must_use]
pub fn is_custatevec_usable() -> bool {
    CuStateVec::new(1).is_ok()
}

/// Check if the cuStabilizer backend can create a frame simulator.
///
/// This is stricter than [`is_cuquantum_available`] and catches environments
/// where the libraries are present but the CUDA runtime cannot initialize.
#[must_use]
pub fn is_custabilizer_usable() -> bool {
    CuFrameSimulator::new(1, 1, 1).is_ok()
}

/// Check if the cuTensorNet backend can create a handle.
///
/// This is stricter than [`is_cuquantum_available`] and catches environments
/// where the libraries are present but the CUDA runtime cannot initialize.
#[must_use]
pub fn is_cutensornet_usable() -> bool {
    CuTensorNet::new().is_ok()
}

/// Check if the cuDensityMat backend can create a simulator on this machine.
///
/// This is stricter than [`is_cuquantum_available`] and catches environments
/// where the libraries are present but the CUDA runtime cannot initialize.
#[must_use]
pub fn is_cudensitymat_usable() -> bool {
    CuDensityMat::new(1).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_types() {
        // Test that error types are accessible
        let err = CuQuantumError::InvalidArgument("test".into());
        let msg = format!("{err}");
        assert!(msg.contains("test"));
    }

    #[test]
    fn test_statevec_error_conversion() {
        use pecos_cuquantum_sys::custatevecStatus_t;

        let err = StateVecError::from(custatevecStatus_t::CUSTATEVEC_STATUS_INVALID_VALUE);
        assert_eq!(err, StateVecError::InvalidValue);
    }

    #[test]
    fn test_trait_reexports() {
        // Test that traits are accessible
        fn _assert_quantum_simulator<T: QuantumSimulator>() {}
        fn _assert_clifford_gateable<T: CliffordGateable>() {}
        fn _assert_arbitrary_rotation<T: ArbitraryRotationGateable>() {}

        // CuStateVec should implement all traits
        _assert_quantum_simulator::<CuStateVec>();
        _assert_clifford_gateable::<CuStateVec>();
        _assert_arbitrary_rotation::<CuStateVec>();

        // CuStabilizer should implement Clifford traits only
        _assert_quantum_simulator::<CuStabilizer>();
        _assert_clifford_gateable::<CuStabilizer>();
        // Note: CuStabilizer does NOT implement ArbitraryRotationGateable
    }

    #[test]
    fn test_stabilizer_error_reexport() {
        // Test that StabilizerError is accessible
        let err = CuQuantumError::Stabilizer(StabilizerError::InvalidValue);
        let msg = format!("{err}");
        assert!(msg.contains("Invalid value"));
    }

    #[test]
    fn test_tensornet_error_reexport() {
        // Test that TensorNetError is accessible
        let err = CuQuantumError::TensorNet(TensorNetError::InvalidValue);
        let msg = format!("{err}");
        assert!(msg.contains("Invalid value"));
    }

    #[test]
    fn test_densitymat_error_reexport() {
        // Test that DensityMatError is accessible
        let err = CuQuantumError::DensityMat(DensityMatError::InvalidValue);
        let msg = format!("{err}");
        assert!(msg.contains("Invalid value"));
    }

    #[test]
    fn test_cutensornet_type_exists() {
        // Test that CuTensorNet type is accessible
        fn _assert_cutensornet(_: &CuTensorNet) {}
    }

    #[test]
    fn test_cudensitymat_type_exists() {
        // Test that CuDensityMat type is accessible
        fn _assert_cudensitymat(_: &CuDensityMat) {}
    }
}
