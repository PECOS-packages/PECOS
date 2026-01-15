// Copyright 2024 The PECOS Developers
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

//! Quantum error correction utilities for PECOS.
//!
//! This crate provides tools for defining, verifying, and analyzing stabilizer
//! quantum error correcting codes.
//!
//! # Architecture
//!
//! The crate is organized into three levels:
//!
//! 1. **Abstract level** (`stabilizer_code`, `distance`): Stabilizer algebra, code verification,
//!    distance calculation. Works with mathematical structure of codes.
//!
//! 2. **Geometry level** (`geometry`, `surface`): Physical layout of codes - where qubits go,
//!    how stabilizers are arranged. Bridges abstract and circuit levels.
//!
//! 3. **Circuit level**: (future) Syndrome extraction circuits, fault tolerance testing.
//!    Will integrate with pecos-qsim for simulation.

pub mod distance;
pub mod geometry;
pub mod stabilizer_code;
pub mod surface;

pub use distance::{
    calculate_distance, find_min_weight_logicals, DistanceResult, DistanceSearchConfig,
    WeightedPauliIterator,
};
pub use geometry::{
    CheckSchedule, LogicalOperator, PauliOp, StabilizerCheck, StabilizerColor,
};
pub use stabilizer_code::{StabilizerCode, StabilizerCodeError};
pub use surface::{SurfaceCode, SurfaceCodeBuilder};
