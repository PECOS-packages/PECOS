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

//! Two-pass correlated MWPM decoder (DGR-style).
//!
//! Wraps any `ObservableDecoder` that also supports a `decode_with_weights`
//! interface. After a training phase to build correlation statistics, each
//! shot is decoded twice:
//!
//! 1. First pass with alignment-corrected weights (from observed frequencies)
//! 2. Second pass with correlation-adjusted weights (from first-pass matching)
//!
//! This improves accuracy by exploiting pairwise edge correlations that
//! standard MWPM ignores.

use crate::correlated_reweighting::EdgeCorrelationTracker;
use crate::errors::DecoderError;

/// Configuration for the correlated decoder.
#[derive(Debug, Clone)]
pub struct CorrelatedDecoderConfig {
    /// Number of training shots before enabling correlation re-weighting.
    /// During training, shots are decoded normally and matchings are recorded.
    pub training_shots: usize,
    /// Whether to use alignment re-weighting (update base weights from
    /// observed edge frequencies).
    pub use_alignment: bool,
    /// Whether to use correlation re-weighting (per-shot two-pass decode).
    pub use_correlation: bool,
}

impl Default for CorrelatedDecoderConfig {
    fn default() -> Self {
        Self {
            training_shots: 1000,
            use_alignment: true,
            use_correlation: true,
        }
    }
}

/// Trait for decoders that can report which edges were matched.
///
/// This is needed for the correlation tracker to build statistics.
/// The decoder must return both the observable mask and the matched edge
/// indices from each decode.
pub trait MatchingDecoder {
    /// Decode a syndrome and return (`observable_mask`, `matched_edge_indices`).
    fn decode_with_matching(&mut self, syndrome: &[u8]) -> Result<(u64, Vec<usize>), DecoderError>;

    /// Decode with adjusted per-edge weights.
    /// The weights slice has one entry per edge in the matching graph.
    fn decode_with_weights(
        &mut self,
        syndrome: &[u8],
        weights: &[f64],
    ) -> Result<(u64, Vec<usize>), DecoderError>;

    /// Number of edges in the matching graph.
    fn num_edges(&self) -> usize;
}

/// Extension of `MatchingDecoder` that exposes per-edge metadata.
///
/// Used by overlapping/sandwich windowed decoders to classify matched edges
/// as core or buffer based on endpoint locations and weight thresholds.
pub trait EdgeTrackingDecoder: MatchingDecoder {
    /// First endpoint node index of the given edge.
    fn edge_node1(&self, edge_idx: usize) -> u32;

    /// Second endpoint node index. Boundary nodes have index >= `num_detectors()`.
    fn edge_node2(&self, edge_idx: usize) -> u32;

    /// Log-likelihood weight of the given edge.
    fn edge_weight(&self, edge_idx: usize) -> f64;

    /// Observable bitmask for the given edge.
    fn edge_obs_mask(&self, edge_idx: usize) -> u64;

    /// Number of detector nodes (not counting boundary/virtual nodes).
    fn num_detectors(&self) -> usize;
}

/// Two-pass correlated MWPM decoder.
///
/// Wraps a `MatchingDecoder` with DGR-style correlation tracking and
/// re-weighting. Transparent to the `ObservableDecoder` interface --
/// callers see the same API, just better accuracy.
pub struct CorrelatedDecoder<D: MatchingDecoder> {
    inner: D,
    tracker: EdgeCorrelationTracker,
    config: CorrelatedDecoderConfig,
    shots_decoded: usize,
    /// Base weights (from DEM, updated by alignment after training).
    base_weights: Vec<f64>,
    /// Buffer for matched-edge flags (avoids per-shot allocation).
    matched_flags: Vec<bool>,
}

impl<D: MatchingDecoder> CorrelatedDecoder<D> {
    /// Create a new correlated decoder wrapping an inner decoder.
    pub fn new(inner: D, base_weights: Vec<f64>, config: CorrelatedDecoderConfig) -> Self {
        let num_edges = inner.num_edges();
        Self {
            tracker: EdgeCorrelationTracker::new(num_edges),
            matched_flags: vec![false; num_edges],
            inner,
            config,
            shots_decoded: 0,
            base_weights,
        }
    }

    /// Whether the training phase is complete.
    #[must_use]
    pub fn is_trained(&self) -> bool {
        self.shots_decoded >= self.config.training_shots
    }

    /// Number of training shots remaining.
    #[must_use]
    pub fn training_remaining(&self) -> usize {
        self.config
            .training_shots
            .saturating_sub(self.shots_decoded)
    }
}

impl<D: MatchingDecoder> crate::ObservableDecoder for CorrelatedDecoder<D> {
    fn decode_to_observables(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        self.shots_decoded += 1;

        // During training: decode normally, record matchings
        if !self.is_trained() {
            let (mask, matched_edges) = self.inner.decode_with_matching(syndrome)?;
            self.tracker.record_matching(&matched_edges);

            // After training is complete, update base weights from alignment
            if self.is_trained() && self.config.use_alignment {
                self.base_weights = self.tracker.aligned_weights();
            }

            return Ok(mask);
        }

        // After training: two-pass decode with correlation adjustment

        // First pass: decode with (possibly alignment-corrected) base weights
        let (first_mask, first_matching) = self
            .inner
            .decode_with_weights(syndrome, &self.base_weights)?;

        // Record the matching for ongoing statistics
        self.tracker.record_matching(&first_matching);

        if !self.config.use_correlation {
            return Ok(first_mask);
        }

        // Build matched-edge flags for correlation adjustment
        self.matched_flags.fill(false);
        for &e in &first_matching {
            if e < self.matched_flags.len() {
                self.matched_flags[e] = true;
            }
        }

        // Compute correlation-adjusted weights
        let adjusted_weights = self
            .tracker
            .correlation_adjusted_weights(&self.base_weights, &self.matched_flags);

        // Second pass: re-decode with adjusted weights
        let (second_mask, _) = self
            .inner
            .decode_with_weights(syndrome, &adjusted_weights)?;

        Ok(second_mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ObservableDecoder;

    /// Simple mock decoder for testing.
    struct MockDecoder {
        num_edges: usize,
    }

    impl MatchingDecoder for MockDecoder {
        fn decode_with_matching(
            &mut self,
            _syndrome: &[u8],
        ) -> Result<(u64, Vec<usize>), DecoderError> {
            Ok((0, vec![0, 2]))
        }

        fn decode_with_weights(
            &mut self,
            _syndrome: &[u8],
            _weights: &[f64],
        ) -> Result<(u64, Vec<usize>), DecoderError> {
            Ok((0, vec![0, 2]))
        }

        fn num_edges(&self) -> usize {
            self.num_edges
        }
    }

    #[test]
    fn test_training_phase() {
        let mock = MockDecoder { num_edges: 5 };
        let weights = vec![1.0; 5];
        let config = CorrelatedDecoderConfig {
            training_shots: 3,
            ..Default::default()
        };
        let mut decoder = CorrelatedDecoder::new(mock, weights, config);

        assert!(!decoder.is_trained());
        assert_eq!(decoder.training_remaining(), 3);

        // Decode 3 training shots
        for _ in 0..3 {
            let _ = decoder.decode_to_observables(&[0, 0, 0]);
        }

        assert!(decoder.is_trained());
        assert_eq!(decoder.training_remaining(), 0);
    }
}
