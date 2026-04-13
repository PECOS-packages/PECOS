// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! # pecos-lindblad
//!
//! Lindblad-to-Pauli-Lindblad noise synthesis for PECOS. Given a per-gate
//! Lindbladian `{H_ideal, noise, tau_g}`, produces the effective
//! Pauli-Lindblad rates `{lambda_k}` that feed Pauli-level QEC simulators.
//!
//! # Verified gate families
//!
//! | Gate | Paper eqs. (arXiv:2502.03462) | Constructor |
//! |---|---|---|
//! | 1Q identity + AD + PD (exact) | line 812 | [`Gate::identity`] |
//! | 1Q `X_theta` + AD + PD | eqs. 869-874 | [`Gate::x_theta`] |
//! | 2Q `CZ_theta` + AD + PD | eqs. 896-906 | [`Gate::cz_theta`] |
//! | 2Q `CX_theta` + AD + PD | eqs. 929-956 | [`Gate::cx_theta`] |
//!
//! All verified numerically to tolerance `1e-8` (1e-12 for the exact
//! identity closed form) in `tests/`.
//!
//! # Scope (current)
//!
//! - **Order**: leading-order Magnus (`Omega_1`) only. Sufficient for
//!   incoherent noise (amplitude damping, pure dephasing). Coherent noise
//!   (e.g. ZZ crosstalk) requires `Omega_2` + Pauli twirl logic -- not yet
//!   implemented.
//! - **Qubits**: 1 and 2, via diagonal and 2x2-block-diagonal `H_g` exp.
//!   General N>=3 or arbitrary non-block-diagonal 2Q gates need a proper
//!   Hermitian matrix-exponentiation path.
//! - **DemStabSim integration**: scaffolded via lossy scalar collapse
//!   (`PauliLindbladModel::total_rate`). The real bridge requires
//!   `pecos-qec::NoiseConfig` generalization -- see
//!   `design/lindblad_sim_skeleton.md`.
//!
//! # Example
//!
//! ```
//! use pecos_lindblad::{
//!     matrix::{self, Matrix},
//!     synthesize_identity_1q, Gate, Lindbladian, Pauli1, PauliString,
//! };
//!
//! // Build a 1-qubit amplitude-damping + pure-dephasing Lindbladian.
//! let beta_down = 1e-3; // per time unit
//! let beta_phi = 2e-3;
//! let d = 2;
//! let hamiltonian = matrix::zeros(d);
//! let collapse: Vec<(Matrix, f64)> = vec![
//!     (matrix::sigma_minus(), beta_down),
//!     (matrix::pauli_1q(Pauli1::Z), beta_phi / 2.0),
//! ];
//! let noise = Lindbladian::new(d, hamiltonian, collapse);
//!
//! // Construct an identity gate of duration 50.0 and synthesize.
//! let gate = Gate::identity(1, noise, 50.0);
//! let pl = synthesize_identity_1q(&gate);
//!
//! // Paper arXiv:2502.03462 line 812:
//! //   lambda_x = lambda_y = beta_down * tau_g / 4
//! //   lambda_z = beta_phi  * tau_g / 2
//! let lambda_x = pl.rate(&PauliString::single(Pauli1::X));
//! let lambda_z = pl.rate(&PauliString::single(Pauli1::Z));
//! assert!((lambda_x - beta_down * 50.0 / 4.0).abs() < 1e-12);
//! assert!((lambda_z - beta_phi  * 50.0 / 2.0).abs() < 1e-12);
//! ```
//!
//! See `design/lindblad_magnus_algorithm.md` for the math spec.

pub mod basis;
pub mod gate;
pub mod lindbladian;
pub mod matrix;
pub mod pauli_lindblad;
pub mod synthesis;

pub use basis::{Pauli1, PauliString};
pub use gate::Gate;
pub use lindbladian::Lindbladian;
pub use pauli_lindblad::PauliLindbladModel;
pub use synthesis::{
    synthesize_exact_unitary, synthesize_identity_1q, synthesize_numerical,
    synthesize_numerical_1q, DEFAULT_N_STEPS,
};
