// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! # PECOS Experimental APIs
//!
//! **⚠️ WARNING: This crate contains experimental, unstable APIs.**
//!
//! All APIs in this crate:
//! - May change without notice between versions
//! - May be removed entirely
//! - May have incomplete documentation or testing
//! - Are not covered by semver guarantees
//!
//! Once APIs are stable and well-tested, they will be moved to the appropriate
//! stable crates (`pecos-engines`, `pecos-simulators`, etc.).
//!
//! ## Current Experimental Features
//!
//! - [`symbolic_executor`] - Symbolic circuit execution on simulators
//! - [`noisy_symbolic`] - Noisy symbolic measurement sampling with depolarizing noise
//!
//! ## Usage
//!
//! ```rust
//! use pecos_experimental::{execute_circuit_symbolic, SymbolicExecutionError};
//! use pecos_experimental::{
//!     NoisyMeasurementHistory,
//!     NoisyMeasurementHistoryBuilder,
//!     DepolarizingNoiseModel,
//! };
//! ```

pub mod noisy_symbolic;
pub mod symbolic_executor;

// Re-export main types at crate root for convenience
pub use noisy_symbolic::{
    DepolarizingNoiseModel, FaultEvent, NoisyMeasurementHistory, NoisyMeasurementHistoryBuilder,
    NoisyMeasurementResult, NoisyMeasurementSampler, Pauli,
};
pub use symbolic_executor::{SymbolicExecutionError, execute_circuit_symbolic};

#[cfg(test)]
mod symbolic_reference;
