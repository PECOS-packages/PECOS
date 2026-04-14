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
//! - **DemStabSim integration**: per-gate-type per-Pauli rates flow
//!   directly via `pecos-qec::PerGateTypeNoise` and
//!   `DemStabSim::per_gate_noise(...)` (no scalar collapse).
//! - **Non-Markovian**: time-convolutionless (TCL) time-local master
//!   equations supported via [`TimeDepLindbladian`] +
//!   [`synthesize_superop_time_dep`]. Covers 1/f dephasing, Gaussian
//!   decay, coloured coherent noise. Time-nonlocal (Nakajima-Zwanzig)
//!   memory-kernel equations remain a structural limit.
//!
//! # Example
//!
//! ```
//! use pecos_lindblad::{
//!     noise_models::ad_pd_1q, synthesize_identity_1q, Gate, Pauli1, PauliString,
//! };
//!
//! // Specify the device in physical (T_1, T_2) parameters.
//! let t1 = 100e-6;      // 100 us
//! let t2 = 80e-6;       // 80 us (requires T_2 <= 2 T_1)
//! let tau_g = 1e-6;     // 1 us gate duration
//!
//! let noise = ad_pd_1q(t1, t2);
//! let gate = Gate::identity(1, noise, tau_g);
//! let pl = synthesize_identity_1q(&gate);
//!
//! // Paper arXiv:2502.03462 line 812:
//! //   lambda_x = lambda_y = beta_down * tau_g / 4
//! //   lambda_z = beta_phi  * tau_g / 2
//! // with beta_down = 1/T_1, beta_phi = 1/T_2 - 1/(2 T_1).
//! let beta_down = 1.0 / t1;
//! let beta_phi = 1.0 / t2 - 1.0 / (2.0 * t1);
//! let lambda_x = pl.rate(&PauliString::single(Pauli1::X));
//! let lambda_z = pl.rate(&PauliString::single(Pauli1::Z));
//! assert!((lambda_x - beta_down * tau_g / 4.0).abs() < 1e-14);
//! assert!((lambda_z - beta_phi  * tau_g / 2.0).abs() < 1e-14);
//! ```
//!
//! See `design/lindblad_magnus_algorithm.md` for the math spec.

pub mod basis;
pub mod gate;
pub mod lindbladian;
pub mod matrix;
pub mod noise_models;
pub mod pauli_lindblad;
pub mod synthesis;
pub mod time_dep;

pub use basis::{Pauli1, PauliString};
pub use gate::Gate;
pub use lindbladian::Lindbladian;
pub use pauli_lindblad::PauliLindbladModel;
pub use synthesis::{
    synthesize_exact_unitary, synthesize_identity_1q, synthesize_numerical,
    synthesize_numerical_1q, synthesize_superop, synthesize_superop_identity,
    DEFAULT_N_SLICES, DEFAULT_N_STEPS,
};
pub use time_dep::{synthesize_superop_time_dep, HermitianFn, RateFn, TimeDepLindbladian};
