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

//! Belief-matching: BP soft info → reweighted matching decoder.
//!
//! Wraps any `MatchingDecoder` with BP-computed edge weights.
//! The BP posteriors are computed by an external `BpWeightProvider`,
//! then fed to the matching decoder via `decode_with_weights`.
//!
//! This achieves belief-matching (Higgott 2022) with any matching backend:
//! - Fusion Blossom (MWPM, ~0.94% threshold)
//! - UF (already in BP+UF, ~0.5% threshold)
//! - `PyMatching` (if it had dynamic weights)

use crate::correlated_decoder::MatchingDecoder;
use crate::correlation_table::CorrelationTable;
use crate::errors::DecoderError;

/// Trait for providing BP-adjusted weights per syndrome.
pub trait BpWeightProvider {
    /// Compute BP-adjusted matching graph edge weights for a syndrome.
    /// Returns one weight per matching graph edge.
    fn compute_weights(&mut self, syndrome: &[u8]) -> Vec<f64>;

    /// Number of matching graph edges.
    fn num_edges(&self) -> usize;

    /// Check if this syndrome is trivial (predecoder can handle it).
    fn is_trivial(&self, syndrome: &[u8]) -> Option<u64>;
}

/// Belief-matching decoder: BP weights → matching decoder.
///
/// Optionally performs a second pass with correlation table adjustment
/// (correlated belief-matching) for exploiting X-Z cross-lattice correlations.
pub struct BpMatchingDecoder<M: MatchingDecoder, B: BpWeightProvider> {
    matching: M,
    bp: B,
    /// Optional correlation table for two-pass correlated decoding.
    correlation: Option<CorrelationTable>,
    /// Reusable buffer for adjusted weights in the second pass.
    adjusted_weights: Vec<f64>,
}

impl<M: MatchingDecoder, B: BpWeightProvider> BpMatchingDecoder<M, B> {
    /// Create a single-pass belief-matching decoder.
    pub fn new(matching: M, bp: B) -> Self {
        let n = bp.num_edges();
        Self {
            matching,
            bp,
            correlation: None,
            adjusted_weights: vec![0.0; n],
        }
    }

    /// Create a two-pass correlated belief-matching decoder.
    ///
    /// First pass: BP weights → MWPM → matched edges.
    /// Second pass: correlation table adjusts weights → MWPM again.
    pub fn with_correlations(matching: M, bp: B, correlation: CorrelationTable) -> Self {
        let n = bp.num_edges();
        Self {
            matching,
            bp,
            correlation: Some(correlation),
            adjusted_weights: vec![0.0; n],
        }
    }
}

impl<M: MatchingDecoder, B: BpWeightProvider> crate::ObservableDecoder for BpMatchingDecoder<M, B> {
    fn decode_to_observables(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        // Predecoder fast path: only for zero-defect syndromes.
        // At d>=5, always use full MWPM (predecoder can be suboptimal).
        // At d=3, BP + predecoder is actually better than MWPM for simple
        // syndromes, but we can't easily detect d here. So we skip the
        // predecoder and let MWPM handle everything for consistency.
        let num_defects = syndrome.iter().filter(|&&v| v != 0).count();
        if num_defects == 0 {
            return Ok(0);
        }

        // Compute BP-adjusted weights.
        let bp_weights = self.bp.compute_weights(syndrome);

        if let Some(corr) = &self.correlation
            && corr.has_correlations()
        {
            // Two-pass correlated belief-matching.

            // First pass: decode with BP weights to get matched edges.
            let (_, matched_edges) = self.matching.decode_with_weights(syndrome, &bp_weights)?;

            // Apply correlation adjustments to BP weights.
            self.adjusted_weights.copy_from_slice(&bp_weights);
            for &edge_idx in &matched_edges {
                if edge_idx < corr.implied_weights.len() {
                    for iw in &corr.implied_weights[edge_idx] {
                        if iw.conditional_weight < self.adjusted_weights[iw.target_edge_idx] {
                            self.adjusted_weights[iw.target_edge_idx] = iw.conditional_weight;
                        }
                    }
                }
            }

            // Second pass: decode with correlation-adjusted weights.
            let (obs, _) = self
                .matching
                .decode_with_weights(syndrome, &self.adjusted_weights)?;
            return Ok(obs);
        }

        // Single-pass belief-matching.
        let (obs, _) = self.matching.decode_with_weights(syndrome, &bp_weights)?;
        Ok(obs)
    }
}
