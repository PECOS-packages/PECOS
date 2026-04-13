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
//! Lindblad-to-Pauli-Lindblad noise synthesis for PECOS.
//!
//! Phase 1 (current): 1-qubit identity-gate synthesis. Produces a
//! [`PauliLindbladModel`] from a [`Gate`] carrying a noise [`Lindbladian`]
//! and duration.
//!
//! Reference: Malekakhlagh et al., arXiv:2502.03462 (npj QI 2025).
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
pub use synthesis::{synthesize_identity_1q, synthesize_numerical_1q, DEFAULT_N_STEPS};
