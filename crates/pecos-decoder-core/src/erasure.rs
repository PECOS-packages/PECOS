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

//! Erasure-aware observable decoder for neutral atom QEC.
//!
//! Neutral atoms have a dominant erasure channel: atom loss is detectable
//! via fluorescence, giving the decoder side-channel information about
//! which qubits were lost. This raises the surface code threshold from
//! ~1% to ~4%.
//!
//! The decoder receives:
//! - A syndrome (detection events)
//! - A list of erased qubit/edge indices (known error locations)
//!
//! Erased edges are set to zero weight (certain error) during matching,
//! guiding the decoder to incorporate them into the correction.

use crate::errors::DecoderError;

/// Trait for decoders that can handle erasure information alongside the syndrome.
///
/// For neutral atoms, erasures come from atom loss detection. The decoder
/// sets erased edge weights to zero (guaranteed error) and finds the
/// minimum-weight correction incorporating the known erasures.
pub trait ObservableErasureDecoder {
    /// Decode a syndrome with erasure side-channel information.
    ///
    /// `erasure_edges`: indices of edges (error mechanisms) known to have
    /// fired. These are set to zero weight during matching.
    ///
    /// # Errors
    ///
    /// Returns `DecoderError` if decoding fails.
    fn decode_with_erasures(
        &mut self,
        syndrome: &[u8],
        erasure_edges: &[usize],
    ) -> Result<u64, DecoderError>;
}
