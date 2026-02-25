// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! ZX calculus integration for PECOS.
//!
//! This crate provides ZX calculus capabilities for analyzing quantum circuits,
//! built on top of [QuiZX](https://github.com/zxcalc/quizx).
//!
//! # Modules
//!
//! - [`graph`] -- ZX graph helpers and metadata
//! - [`convert`] -- Circuit <-> ZX graph conversion
//! - [`pauli_web`] -- Pauli web computation and classification
//! - [`noise`] -- Noise model for annotating edges with error probabilities
//! - [`dem`] -- Detector Error Model extraction from Pauli webs
//! - [`viz`] -- SVG visualization of ZX diagrams
//! - [`graph_state`] -- Graph state representation and entanglement analysis
//! - [`symplectic`] -- Symplectic representation of Clifford unitaries
//! - [`stabilizer`] -- Stabilizer <-> ZX connections (feature-gated)

pub mod convert;
pub mod dem;
pub mod graph;
pub mod graph_state;
pub mod noise;
pub mod pauli_web;
pub mod symplectic;
pub mod tableau;
pub mod viz;

#[cfg(feature = "stabilizer")]
pub mod stabilizer;

// Re-export key QuiZX types for convenience
pub use quizx::graph::{EType, GraphLike, VType};
pub use quizx::vec_graph::Graph as ZxGraph;

// Re-export QuiZX modules users commonly need
pub use quizx::basic_rules;
pub use quizx::circuit as zx_circuit;
pub use quizx::simplify;
