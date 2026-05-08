// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
// express or implied. See the License for the specific language governing permissions and
// limitations under the License.

//! Hybrid stabilizer + tensor network simulation methods.
//!
//! This crate provides experimental implementations of methods that combine
//! Clifford/stabilizer tracking with tensor network (MPS) representations:
//!
//! - **MPS**: Matrix Product State engine (SVD truncation, gate application, contraction)
//! - **STN**: Stabilizer Tensor Networks (tableau + MPS coefficients)
//! - **MAST**: Magic state injection Augmented STN (deferred non-Clifford cost)
//!
//! # References
//!
//! - Masot-Llima, Garcia-Saez. "Stabilizer Tensor Networks: Universal Quantum Simulator
//!   on a Basis of Stabilizer States." PRL 133, 230601 (2024). arXiv:2403.08724.
//! - Nakhl, Harper, West, Dowling, Sevior, Quella, Usman. "Stabilizer Tensor Networks
//!   with Magic State Injection." PRL 134, 190602 (2025). arXiv:2411.12482.
//! - Reference implementation: <https://github.com/bsc-quantic/stabilizer-TN>

pub mod errors;
pub mod mps;
pub mod stab_mps;
