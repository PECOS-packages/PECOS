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
//! # Default configuration
//!
//! `StabMps` defaults to a maximum bond dimension of 128, a maximum relative
//! truncation error of `1e-8`, an SVD cutoff of `1e-12`, normalization after
//! non-Clifford gates, merged same-qubit RZ rotations, and exact singular
//! measurement. Pragmatic and lazy measurement modes, Pauli-frame tracking,
//! and numerical flag redetection remain opt-in. Singular `sample_bitstring`
//! follows the selected mode; plural `sample_bitstrings` is always exact by
//! construction. To
//! recover the former behavior explicitly, use `max_bond_dim(64)`,
//! `max_truncation_error(0.0)`, and `merge_rz(false)` on the builder.
//! `Mast` instead defaults `merge_rz` to false so every RZ immediately exposes
//! its injection and ancilla-capacity cost; enable it explicitly when batching
//! is desired.
//!
//! # Bitstring convention
//!
//! At the `StabMps`/`Mast` layer, public bitstrings use qubit-index order:
//! `bits[q]` is the bit for qubit `q`. Consequently, converting a `StabMps`
//! bitstring to the little-endian integer index used by
//! [`stab_mps::StabMps::state_vector`] gives
//! `index = sum(usize::from(bits[q]) << q)`. This convention applies equally
//! to bitstrings accepted by probability and amplitude reads and to rows
//! returned by the samplers. The lower-level [`mps::Mps::state_vector`] is an
//! internal tensor-chain read and orders site 0 as the most-significant bit.
//!
//! # References
//!
//! - Masot-Llima, Garcia-Saez. "Stabilizer Tensor Networks: Universal Quantum Simulator
//!   on a Basis of Stabilizer States." PRL 133, 230601 (2024). arXiv:2403.08724.
//! - Nakhl, Harper, West, Dowling, Sevior, Quella, Usman. "Stabilizer Tensor Networks
//!   with Magic State Injection." PRL 134, 190602 (2025). arXiv:2411.12482.
//! - Reference implementation: <https://github.com/bsc-quantic/stabilizer-TN>

/// Errors returned by matrix-product-state operations.
pub mod errors;
pub mod mps;
pub mod stab_mps;
