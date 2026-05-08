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

//! Fast syndrome-graph Union-Find decoder.
//!
//! Purpose-built for surface codes and other QEC codes with matching-graph
//! structure. Works on the syndrome graph (not the Tanner graph), where nodes
//! are detectors and edges are error mechanisms.
//!
//! Design goals:
//! - Zero per-shot allocation (reusable flat arrays)
//! - No locks, no Arc, no hash sets (unlike MWPF's UF)
//! - Bounded worst-case latency
//! - Implements `ObservableDecoder` and `MatchingDecoder` for composability
//!   with `TwoPassDecoder` (correlated decoding)

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

pub mod astar;
pub mod bp_uf;
pub mod css_decoder;
pub mod decoder;
pub mod mini_bp;

pub mod windowed;

// Note: belief_matching (BP → PyMatching MWPM) lives in the Python bindings
// (fault_tolerance_bindings.rs) since it requires pecos-pymatching which is
// a C++ FFI crate. This crate stays pure Rust.

pub use astar::{AStarConfig, AStarDecoder};
pub use bp_uf::{BpSchedule, BpUfConfig, BpUfDecoder};
pub use css_decoder::{CssUfDecoder, QubitEdgeMapping};
pub use decoder::{UfDecoder, UfDecoderConfig};
pub use windowed::{
    BeamSearchConfig, BeamSearchWindowedDecoder, OverlappingWindowedDecoder,
    SandwichWindowedDecoder, StreamingWindowedDecoder, WindowedConfig, WindowedDecoder,
};
