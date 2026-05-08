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

//! Streaming decoder trait for real-time QEC.
//!
//! Accepts syndrome data incrementally (round by round) and emits
//! partial observable corrections as windows complete.

use crate::errors::DecoderError;

/// Streaming decoder that accepts syndrome data incrementally.
///
/// For real-time decoding where syndrome arrives round-by-round.
/// The decoder manages windows internally and emits committed
/// corrections as each window's core region becomes decodable.
pub trait StreamingDecoder {
    /// Feed one round of detection events.
    ///
    /// `round` is the time coordinate (0-indexed).
    /// `detectors` contains `(detector_index, value)` pairs for this round.
    ///
    /// Returns any newly committed observable corrections as a bitmask.
    /// Returns 0 if no window completed this round.
    ///
    /// # Errors
    ///
    /// Returns `DecoderError` if window decoding fails.
    fn feed_round(&mut self, round: usize, detectors: &[(u32, u8)]) -> Result<u64, DecoderError>;

    /// Signal that no more rounds will arrive.
    ///
    /// Forces decode of any remaining buffered windows and returns
    /// the final observable correction for uncommitted windows.
    ///
    /// # Errors
    ///
    /// Returns `DecoderError` if window decoding fails.
    fn flush(&mut self) -> Result<u64, DecoderError>;

    /// Total observable mask accumulated so far (XOR of all committed corrections).
    fn accumulated_obs(&self) -> u64;

    /// Reset for the next shot (clear syndrome buffer and accumulated state).
    fn reset(&mut self);
}
