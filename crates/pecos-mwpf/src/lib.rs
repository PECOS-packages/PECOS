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

//! MWPF hypergraph decoder module
//!
//! This module provides Rust bindings for the Minimum-Weight Parity Factor
//! decoder for quantum error correction. Unlike MWPM decoders, MWPF handles
//! hyperedges natively -- it can decode Y errors, depolarizing noise, color
//! codes, and small QLDPC codes with higher accuracy than graphlike decoders.
//!
//! Tradeoff: MWPF has a heavier worst-case latency tail than MWPM. Good for
//! offline benchmarks, correlated-noise studies, and accuracy-first decoding.

// Allow casts between float/int for weight conversions
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

pub mod core_traits;
pub mod decoder;
pub mod errors;

// Re-export main types
pub use decoder::{MwpfConfig, MwpfDecoder, MwpfDecodingResult, MwpfSolverType};
pub use errors::MwpfError;
