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

// The EEG crate is experimental physics/math code. Its core routines use
// numerical casts and dense index-based algebra, and the public API is still
// stabilizing. Keep this list narrow and fix ordinary style lints in code.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc
)]

//! Elementary Error Generator (EEG) analysis for coherent noise.
//!
//! Propagates error generators through Clifford circuits and produces
//! detector error probabilities at polynomial cost. Based on:
//! - Miller et al. arXiv:2504.15128 (simulation algorithm)
//! - Hines et al. arXiv:2603.18457 (DEM mapping)
//!
//! # Algorithm
//!
//! 1. Express noise as sparse EEG generators (H, S, C, A types)
//! 2. Propagate each generator forward through Clifford gates
//! 3. Combine via BCH formula (first order = sum)
//! 4. Classify by DEM event (which detectors each Pauli flips)
//! 5. Compute detection event probabilities

pub mod builder;
pub mod circuit;
pub mod coherent_dem;
pub mod correlation_table;
pub mod dem_generator;
pub mod dem_mapping;
pub mod dem_simulator;
pub mod eeg;
pub mod expand;
pub mod heisenberg;
pub mod noise;
pub mod noise_characterization;
pub mod noise_compression;
pub mod propagate;
pub mod stabilizer;
pub mod strong_sim;

/// Pauli bitmask type used throughout the EEG crate.
/// Pauli bitmask type used throughout the EEG crate. Uses SmallVec<[u64; 8]>
/// for 512 bits inline (zero allocation up to d=9 surface codes), with
/// automatic heap spillover for larger circuits.
pub type Bm = pecos_core::PauliBitmaskSmall;

// Re-export key types for convenience
pub use dem_mapping::{BchOrder, EegConfig, HFormula};
pub use noise::{NoiseInjection, NoiseSpec, UniformNoise};
