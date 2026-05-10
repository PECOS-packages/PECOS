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

//! Decode budget and strategy framework for real-time QEC.
//!
//! Different hardware platforms have different time budgets for decoding:
//! superconducting (~1μs), neutral atoms (~1ms), ion traps (~10ms).
//! The framework selects the best decode strategy based on the budget.
//!
//! # Example
//!
//! ```
//! use std::time::Duration;
//!
//! use pecos_decoder_core::decode_budget::{DecodeBudget, DetectorRegion};
//!
//! let distance = 7;
//! let budget = DecodeBudget::from_reaction_time(Duration::from_millis(1), distance);
//! assert!(budget.is_windowed());
//! assert_eq!(budget.code_distance, distance);
//!
//! let first_round = DetectorRegion { start: 0, end: distance * distance };
//! assert!(first_round.contains(0));
//! assert!(!first_round.is_empty());
//! ```

use crate::errors::DecoderError;
use std::time::Duration;

/// Time and resource budget for decoding.
///
/// Two timing constraints govern QEC decoding:
///
/// - **Throughput**: decoder must keep up with syndrome generation to
///   avoid backlog. Measured in time per syndrome round.
/// - **Reaction time**: at feed-forward decision points (T gates, magic
///   state injection), the decoder must produce a correction within a
///   deadline. For Clifford-only circuits, this is unlimited (corrections
///   are metadata applied at the end).
///
/// The budget also controls the accuracy/latency trade-off via window
/// size and overlap parameters.
#[derive(Debug, Clone)]
pub struct DecodeBudget {
    /// Maximum wall-clock time per decode at a decision point.
    /// For Clifford circuits this is unlimited; for T gates it's
    /// the time between last syndrome and correction application.
    pub reaction_time: Duration,
    /// Maximum detectors to include in a single decode call.
    /// Controls memory usage and decode time.
    pub max_window_detectors: usize,
    /// Number of overlap rounds at window boundaries.
    /// More overlap = better accuracy, more compute.
    /// Set to 0 for non-overlapping (fastest, least accurate).
    pub overlap_rounds: usize,
    /// Code distance (used to scale window sizes).
    pub code_distance: usize,
}

impl DecodeBudget {
    /// Unlimited budget: full-circuit decode. Maximum accuracy.
    ///
    /// Use for Clifford-only circuits (no feed-forward decisions),
    /// offline simulation, or any situation where the decoder can
    /// take as long as needed.
    #[must_use]
    pub fn unlimited() -> Self {
        Self {
            reaction_time: Duration::from_hours(1),
            max_window_detectors: usize::MAX,
            overlap_rounds: usize::MAX,
            code_distance: 0,
        }
    }

    /// Create a budget from the reaction time at decision points.
    ///
    /// `reaction_time`: time available between last syndrome and when
    /// the correction must be applied (e.g., at T gate injection).
    ///
    /// Window size and overlap are scaled based on available time:
    /// - Very generous (>100ms): unlimited (full-circuit decode)
    /// - Generous (1ms - 100ms): large windows, d overlap
    /// - Medium (10μs - 1ms): d-round windows, d/2 overlap
    /// - Tight (<10μs): minimal windows, no overlap
    #[must_use]
    pub fn from_reaction_time(reaction_time: Duration, distance: usize) -> Self {
        let us = reaction_time.as_micros() as usize;

        let (max_dets, overlap) = if us >= 100_000 {
            (usize::MAX, usize::MAX)
        } else if us >= 1_000 {
            (distance * distance * 4 * distance, distance)
        } else if us >= 10 {
            (distance * distance * 2 * distance, distance / 2)
        } else {
            (distance * distance * 2, 0)
        };

        Self {
            reaction_time,
            max_window_detectors: max_dets,
            overlap_rounds: overlap,
            code_distance: distance,
        }
    }

    /// Create a budget with explicit parameters.
    #[must_use]
    pub fn with_params(
        reaction_time: Duration,
        max_window_detectors: usize,
        overlap_rounds: usize,
        code_distance: usize,
    ) -> Self {
        Self {
            reaction_time,
            max_window_detectors,
            overlap_rounds,
            code_distance,
        }
    }

    /// Whether the budget allows full-circuit decoding (unlimited window).
    #[must_use]
    pub fn is_unlimited(&self) -> bool {
        self.max_window_detectors == usize::MAX && self.overlap_rounds == usize::MAX
    }

    /// Whether windowed decoding is needed (non-unlimited budget).
    #[must_use]
    pub fn is_windowed(&self) -> bool {
        !self.is_unlimited()
    }
}

/// A region of detectors in the circuit.
///
/// Represents a contiguous block of detectors by their global indices.
/// Used by strategies to define decode/commit boundaries.
#[derive(Debug, Clone)]
pub struct DetectorRegion {
    /// First global detector index in this region.
    pub start: usize,
    /// One past the last global detector index.
    pub end: usize,
}

impl DetectorRegion {
    /// Number of detectors in this region.
    #[must_use]
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// Whether the region is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }

    /// Whether a detector index is in this region.
    #[must_use]
    pub fn contains(&self, det: usize) -> bool {
        det >= self.start && det < self.end
    }
}

/// Strategy for decoding a logical circuit.
///
/// Strategies implement different decode/commit patterns depending
/// on the time budget. All strategies produce the same type of output
/// (observable correction mask) but with different accuracy/latency
/// trade-offs.
pub trait DecodeStrategy: Send + Sync {
    /// Decode a syndrome and return the observable correction mask.
    ///
    /// The syndrome covers the full circuit. The strategy decides
    /// which portion to decode based on its internal state and budget.
    fn decode(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError>;

    /// Commit corrections for a detector region.
    ///
    /// After commitment, detectors in this region are excluded from
    /// future decode calls. Their corrections are accumulated into
    /// the committed observable mask.
    fn commit(&mut self, region: &DetectorRegion) -> Result<u64, DecoderError>;

    /// Total committed observable correction so far.
    fn committed_obs(&self) -> u64;

    /// Reset all state for the next shot.
    fn reset(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_budget_unlimited() {
        let b = DecodeBudget::unlimited();
        assert!(b.is_unlimited());
        assert!(!b.is_windowed());
    }

    #[test]
    fn test_budget_from_reaction_time() {
        // Very tight: no overlap
        let b = DecodeBudget::from_reaction_time(Duration::from_micros(1), 7);
        assert_eq!(b.overlap_rounds, 0);
        assert!(b.is_windowed());

        // Medium: d/2 overlap
        let b = DecodeBudget::from_reaction_time(Duration::from_micros(100), 7);
        assert_eq!(b.overlap_rounds, 3);

        // Generous: d overlap
        let b = DecodeBudget::from_reaction_time(Duration::from_millis(10), 7);
        assert_eq!(b.overlap_rounds, 7);

        // Very generous: unlimited
        let b = DecodeBudget::from_reaction_time(Duration::from_millis(200), 7);
        assert!(b.is_unlimited());
    }

    #[test]
    fn test_detector_region() {
        let r = DetectorRegion { start: 10, end: 20 };
        assert_eq!(r.len(), 10);
        assert!(r.contains(10));
        assert!(r.contains(19));
        assert!(!r.contains(20));
        assert!(!r.contains(9));
    }
}
