//! Cross-platform GPU quantum simulators
//!
//! GPU-accelerated quantum simulators using wgpu,
//! enabling simulation on multiple backends:
//!
//! - Metal (Apple Silicon)
//! - Vulkan (Linux, Windows, Android)
//! - DirectX 12 (Windows)
//! - WebGPU (browsers via WASM)
//!
//! # Simulators
//!
//! - [`GpuStateVec`] / [`GpuStateVec64`]: State vector simulator (f64 precision, default)
//! - [`GpuStateVec32`]: State vector simulator (f32 precision, faster)
//! - [`GpuStab`]: Stabilizer tableau simulator for Clifford circuits (experimental)
//!
//! # Example
//!
//! `GpuStateVec` aliases the f64 backend, which requires `SHADER_F64`. On
//! adapters without f64 support (e.g. Metal on Apple Silicon) `new()` returns
//! [`GpuError::UnsupportedFeature`]; the doctest skips in that case so it can
//! still exercise real GPU code where available. Use [`GpuStateVec32`] for a
//! universally portable f32 backend.
//!
//! ```
//! use pecos_gpu_sims::GpuStateVec;
//! use pecos_simulators::CliffordGateable;
//! use pecos_core::{qid, QubitId};
//!
//! // Skip cleanly on platforms without a GPU or without SHADER_F64.
//! let Ok(mut sim) = GpuStateVec::new(4) else { return };
//! sim.h(&qid(0));         // Hadamard on qubit 0
//! sim.cx(&[(QubitId(0), QubitId(1))]);    // CNOT with control=0, target=1
//! let _result = sim.mz(&[QubitId(0)]);  // Measure qubit 0
//! ```

pub mod circuit_compiler;
mod clifford_fusion;
mod gpu;
mod gpu64;
mod gpu_auto;
mod gpu_density_matrix;
mod gpu_influence_sampler;
mod gpu_noisy_sampler;
mod gpu_pauli_prop;
pub mod gpu_probe;
mod gpu_sampler;
mod gpu_stab;
mod gpu_stab_multi;
pub mod prelude;
pub mod state_access;

#[cfg(test)]
mod gpu_sampler_validation;

pub use circuit_compiler::{CompiledCircuit, Gate as CompiledGate, GateType};
pub use gpu::{GpuError, GpuStateVec32, RequiredFeature};
pub use gpu_auto::GpuStateVecAuto;
pub use gpu_density_matrix::{
    GpuDensityMatrix, GpuDensityMatrix32, GpuDensityMatrix64, GpuStateVecBackend,
};
pub use gpu64::GpuStateVec64;

/// Default GPU state vector simulator (f64 precision).
///
/// Use [`GpuStateVec32`] for f32 precision (faster but less accurate), or
/// [`GpuStateVecAuto`] to opt in to runtime precision selection (tries f64
/// first, falls back to f32 on adapters without `SHADER_F64`).
pub type GpuStateVec = GpuStateVec64;
pub use gpu_influence_sampler::{GpuInfluenceMapData, GpuInfluenceSampler, GpuSamplingResult};
pub use gpu_noisy_sampler::{
    BiasedDepolarizingNoiseSampler, CircuitBuilder, DepolarizingNoiseSampler, Gate,
    GpuNoisySampler, NoiseInjection, NoiseSampler, NoisyCircuitStep, Pauli, ShotResult,
};
pub use gpu_pauli_prop::GpuPauliProp;
pub use gpu_sampler::{GpuMeasurementSampler, GpuSampleResult};
pub use gpu_stab::GpuStab;
pub use gpu_stab_multi::GpuStabMulti;
pub use state_access::{GpuDensityMatrixHostAccess, GpuStateVectorHostAccess};

/// Default GPU stabilizer simulator using `PecosRng`
pub type DefaultGpuStab = GpuStab<pecos_random::PecosRng>;

/// Default multi-shot GPU stabilizer simulator using `PecosRng`
pub type DefaultGpuStabMulti = GpuStabMulti<pecos_random::PecosRng>;

use std::f64::consts::FRAC_1_SQRT_2;

/// Standard gate matrices as [`a_re`, `a_im`, `b_re`, `b_im`, `c_re`, `c_im`, `d_re`, `d_im`]
// GPU shaders work with f32 for performance. The precision loss from f64->f32
// conversion is acceptable for quantum simulation (errors are ~1e-7).
#[allow(clippy::cast_possible_truncation)]
pub mod gates {
    use super::FRAC_1_SQRT_2;

    /// Identity gate
    pub const I: [f32; 8] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];

    /// Pauli-X gate (NOT)
    pub const X: [f32; 8] = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0];

    /// Pauli-Y gate
    pub const Y: [f32; 8] = [0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0];

    /// Pauli-Z gate
    pub const Z: [f32; 8] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0];

    /// Hadamard gate
    pub const H: [f32; 8] = [
        FRAC_1_SQRT_2 as f32,
        0.0,
        FRAC_1_SQRT_2 as f32,
        0.0,
        FRAC_1_SQRT_2 as f32,
        0.0,
        -(FRAC_1_SQRT_2 as f32),
        0.0,
    ];

    /// S gate (sqrt(Z))
    pub const S: [f32; 8] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];

    /// S-dagger gate
    pub const SDG: [f32; 8] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0];

    // T gate = RZ(π/4) to match PECOS convention
    // RZ(θ) = [[e^(-iθ/2), 0], [0, e^(iθ/2)]]
    // RZ(π/4) = [[e^(-iπ/8), 0], [0, e^(iπ/8)]]
    // cos(π/8) ≈ 0.9238795, sin(π/8) ≈ 0.3826834
    const COS_PI_8: f32 = 0.923_879_5;
    const SIN_PI_8: f32 = 0.382_683_43;

    /// T gate (π/4 rotation around Z-axis, equivalent to `RZ(π/4)`)
    pub const T: [f32; 8] = [
        COS_PI_8, -SIN_PI_8, // e^(-iπ/8)
        0.0, 0.0, 0.0, 0.0, COS_PI_8, SIN_PI_8, // e^(iπ/8)
    ];

    /// T-dagger gate (-π/4 rotation around Z-axis, equivalent to `RZ(-π/4)`)
    pub const TDG: [f32; 8] = [
        COS_PI_8, SIN_PI_8, // e^(iπ/8)
        0.0, 0.0, 0.0, 0.0, COS_PI_8, -SIN_PI_8, // e^(-iπ/8)
    ];

    /// SX gate (sqrt(X))
    pub const SX: [f32; 8] = [0.5, 0.5, 0.5, -0.5, 0.5, -0.5, 0.5, 0.5];

    /// SX-dagger gate
    pub const SXDG: [f32; 8] = [0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5];

    /// SY gate (sqrt(Y))
    pub const SY: [f32; 8] = [0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5];

    /// SY-dagger gate
    pub const SYDG: [f32; 8] = [0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5];

    /// Create RX(theta) gate matrix
    #[must_use]
    pub fn rx(theta: f64) -> [f32; 8] {
        let c = (theta / 2.0).cos() as f32;
        let s = (theta / 2.0).sin() as f32;
        [c, 0.0, 0.0, -s, 0.0, -s, c, 0.0]
    }

    /// Create RY(theta) gate matrix
    #[must_use]
    pub fn ry(theta: f64) -> [f32; 8] {
        let c = (theta / 2.0).cos() as f32;
        let s = (theta / 2.0).sin() as f32;
        [c, 0.0, -s, 0.0, s, 0.0, c, 0.0]
    }

    /// Create RZ(theta) gate matrix
    #[must_use]
    pub fn rz(theta: f64) -> [f32; 8] {
        let c = (theta / 2.0).cos() as f32;
        let s = (theta / 2.0).sin() as f32;
        [c, -s, 0.0, 0.0, 0.0, 0.0, c, s]
    }
}
