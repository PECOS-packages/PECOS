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
//! Pauli-Lindblad rates `{lambda_k}` that feed
//! [`pecos_qec::dem_stab::DemStabSim`] via
//! [`pecos_qec::fault_tolerance::dem_builder::PerGateTypeNoise`].
//!
//! # Golden path
//!
//! ```
//! use pecos_lindblad::{
//!     noise_models::ad_pd_1q, synthesize_identity_1q, Gate, Pauli1, PauliString,
//! };
//!
//! // Device parameters in physical (T_1, T_2) terms.
//! let t1 = 100e-6;
//! let t2 = 80e-6;
//! let tau_g = 1e-6;
//!
//! let noise = ad_pd_1q(t1, t2);
//! let gate = Gate::identity(1, noise, tau_g);
//! let pl = synthesize_identity_1q(&gate);
//!
//! let lambda_x = pl.rate(&PauliString::single(Pauli1::X));
//! assert!(lambda_x > 0.0);
//! ```
//!
//! # Picking a synthesis path
//!
//! - [`synthesize_identity_1q`] -- fastest for 1-qubit identity gates
//!   (closed-form, machine-precision).
//! - [`synthesize_numerical`] -- any gate with purely dissipative noise
//!   (AD, PD, depolarizing). Simpson's rule on the interaction-frame
//!   Lindbladian.
//! - [`synthesize_superop`] -- general: any gate, any mix of coherent
//!   and dissipative noise, all orders of Magnus. Slower but correct
//!   in all regimes.
//!
//! # Feature flags
//!
//! - `serde` -- (de)serialize [`PauliLindbladModel`] for caching.
//!
//! # Modules by audience
//!
//! - **Core forward synthesis**: [`Gate`], [`synthesize_identity_1q`],
//!   [`synthesize_numerical`], [`synthesize_superop`].
//! - **Noise-model verification** (diff helpers + analytic `(T_1, T_2)`
//!   recovery + Monte Carlo UQ): see [`noise_models`] and
//!   [`PauliLindbladModel::diff`], [`PauliLindbladModel::diagnose_gap`].
//! - **Non-Markovian (TCL)**: see [`time_dep`] for 1/f dephasing,
//!   Gaussian decay, coloured coherent noise.
//!
//! # Verified gate families (arXiv:2502.03462)
//!
//! | Gate | Paper eqs. | Constructor |
//! |---|---|---|
//! | 1Q identity + AD+PD (exact) | line 812 | [`Gate::identity`] |
//! | 1Q `X_theta` + AD+PD | 869-874 | [`Gate::x_theta`] |
//! | 2Q `CZ_theta` + AD+PD | 896-906 | [`Gate::cz_theta`] |
//! | 2Q `CX_theta` + AD+PD | 929-956 | [`Gate::cx_theta`] |
//! | 2Q coherent IZ/ZI/ZZ phase | 981, 986-990 | any `Gate` + `coherent_phase_2q` |
//! | 3Q `CX ⊗ I` + IZZ crosstalk | 1009-1011 | [`Gate::cx_theta_with_izz_crosstalk`] |
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

// Core API -- the golden path.
pub use basis::{Pauli1, PauliString};
pub use gate::Gate;
pub use lindbladian::Lindbladian;
pub use pauli_lindblad::PauliLindbladModel;
pub use synthesis::{
    DEFAULT_N_SLICES, DEFAULT_N_STEPS, synthesize_identity_1q, synthesize_numerical,
    synthesize_superop,
};

// Advanced synthesis paths -- specialized cases of `synthesize_superop`.
// Exposed for users who know they need the specific behavior.
pub use synthesis::{
    synthesize_exact_unitary, synthesize_numerical_1q, synthesize_superop_identity,
};

pub use time_dep::{HermitianFn, RateFn, TimeDepLindbladian, synthesize_superop_time_dep};
