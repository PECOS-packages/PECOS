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
mod gpu_bounded_enumeration;
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
pub use gpu_bounded_enumeration::{
    GpuBoundedEnumerationBackend, GpuBoundedEnumerationError,
    gpu_bounded_enumeration_code_distance, gpu_bounded_enumeration_stabilizer_distance,
    gpu_bounded_enumeration_x_distance, gpu_bounded_enumeration_z_distance,
};
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

/// Standard gate matrices as [`a_re`, `a_im`, `b_re`, `b_im`, `c_re`, `c_im`, `d_re`, `d_im`]
// GPU shaders work with f32 for performance. The precision loss from f64->f32
// conversion is acceptable for quantum simulation (errors are ~1e-7).
#[allow(clippy::cast_possible_truncation)]
pub mod gates {
    use pecos_core::Clifford;
    use pecos_core::gate_type::{GateType, single_qubit_matrix_to_f32};

    const fn canonical(gate: GateType) -> [f32; 8] {
        let Some(matrix) = gate.canonical_1q_matrix() else {
            panic!("gate has no canonical single-qubit matrix");
        };
        single_qubit_matrix_to_f32(matrix)
    }

    const fn canonical_clifford(gate: Clifford) -> [f32; 8] {
        let Some(matrix) = gate.canonical_1q_matrix() else {
            panic!("gate has no canonical single-qubit Clifford matrix");
        };
        single_qubit_matrix_to_f32(matrix)
    }

    /// Identity gate
    pub const I: [f32; 8] = canonical(GateType::I);

    /// Pauli-X gate (NOT)
    pub const X: [f32; 8] = canonical(GateType::X);

    /// Pauli-Y gate
    pub const Y: [f32; 8] = canonical(GateType::Y);

    /// Pauli-Z gate
    pub const Z: [f32; 8] = canonical(GateType::Z);

    /// Hadamard gate H/H1
    pub const H: [f32; 8] = canonical(GateType::H);

    /// H2 gate
    pub const H2: [f32; 8] = canonical_clifford(Clifford::H2);
    /// H3 gate
    pub const H3: [f32; 8] = canonical_clifford(Clifford::H3);
    /// H4 gate
    pub const H4: [f32; 8] = canonical_clifford(Clifford::H4);
    /// H5 gate
    pub const H5: [f32; 8] = canonical_clifford(Clifford::H5);
    /// H6 gate
    pub const H6: [f32; 8] = canonical_clifford(Clifford::H6);

    /// S gate (sqrt(Z))
    pub const S: [f32; 8] = canonical(GateType::SZ);

    /// S-dagger gate
    pub const SDG: [f32; 8] = canonical(GateType::SZdg);

    /// Conventional T gate, `diag(1, exp(i*pi/4))`.
    ///
    /// This is `exp(i*pi/8) RZ(pi/4)`.
    pub const T: [f32; 8] = canonical(GateType::T);

    /// Conventional T-dagger gate, `diag(1, exp(-i*pi/4))`.
    ///
    /// This is `exp(-i*pi/8) RZ(-pi/4)`.
    pub const TDG: [f32; 8] = canonical(GateType::Tdg);

    /// SX gate (sqrt(X))
    pub const SX: [f32; 8] = canonical(GateType::SX);

    /// SX-dagger gate
    pub const SXDG: [f32; 8] = canonical(GateType::SXdg);

    /// SY gate (sqrt(Y))
    pub const SY: [f32; 8] = canonical(GateType::SY);

    /// SY-dagger gate
    pub const SYDG: [f32; 8] = canonical(GateType::SYdg);

    /// Face gate F/F1
    pub const F: [f32; 8] = canonical(GateType::F);
    /// Adjoint Face gate F/F1
    pub const FDG: [f32; 8] = canonical(GateType::Fdg);
    /// F2 gate
    pub const F2: [f32; 8] = canonical_clifford(Clifford::F2);
    /// Adjoint F2 gate
    pub const F2DG: [f32; 8] = canonical_clifford(Clifford::F2dg);
    /// F3 gate
    pub const F3: [f32; 8] = canonical_clifford(Clifford::F3);
    /// Adjoint F3 gate
    pub const F3DG: [f32; 8] = canonical_clifford(Clifford::F3dg);
    /// F4 gate
    pub const F4: [f32; 8] = canonical_clifford(Clifford::F4);
    /// Adjoint F4 gate
    pub const F4DG: [f32; 8] = canonical_clifford(Clifford::F4dg);

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
