//! Cross-platform GPU quantum simulators
//!
//! This crate provides GPU-accelerated quantum simulators using wgpu,
//! enabling simulation on multiple backends:
//!
//! - Metal (Apple Silicon)
//! - Vulkan (Linux, Windows, Android)
//! - DirectX 12 (Windows)
//! - WebGPU (browsers via WASM)
//!
//! # Example
//!
//! ```no_run
//! use pecos_gpu_sims::WgpuStateVec;
//!
//! let mut sim = WgpuStateVec::new(4).unwrap(); // 4 qubits
//! sim.h(0);           // Hadamard on qubit 0
//! sim.cx(0, 1);       // CNOT with control=0, target=1
//! let result = sim.measure(0);  // Measure qubit 0
//! ```

mod gpu;

pub use gpu::WgpuStateVec;

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

    /// T gate (sqrt(S))
    pub const T: [f32; 8] = [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        FRAC_1_SQRT_2 as f32,
        FRAC_1_SQRT_2 as f32,
    ];

    /// T-dagger gate
    pub const TDG: [f32; 8] = [
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        FRAC_1_SQRT_2 as f32,
        -(FRAC_1_SQRT_2 as f32),
    ];

    /// SX gate (sqrt(X))
    pub const SX: [f32; 8] = [0.5, 0.5, 0.5, -0.5, 0.5, -0.5, 0.5, 0.5];

    /// SX-dagger gate
    pub const SXDG: [f32; 8] = [0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5];

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
