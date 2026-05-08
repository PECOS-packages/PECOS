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

//! Two-pass correlated decoder using pre-computed correlation table.
//!
//! Wraps any `MatchingDecoder` with a two-pass decode:
//! 1. First pass: standard decode to identify matched edges
//! 2. Apply conditional weight adjustments from `CorrelationTable`
//! 3. Second pass: re-decode with adjusted weights
//!
//! This is the decoder-agnostic equivalent of `PyMatching`'s correlated matching.

use crate::correlated_decoder::MatchingDecoder;
use crate::correlation_table::CorrelationTable;
use crate::errors::DecoderError;

/// Two-pass correlated decoder.
///
/// Uses a pre-computed `CorrelationTable` (from DEM decomposition) to
/// adjust edge weights between the first and second decode passes.
/// Works with any decoder implementing `MatchingDecoder`.
pub struct TwoPassDecoder<D: MatchingDecoder> {
    inner: D,
    correlation_table: CorrelationTable,
    base_weights: Vec<f64>,
    /// Reusable buffer for adjusted weights (avoids per-shot allocation).
    adjusted_weights: Vec<f64>,
}

impl<D: MatchingDecoder> TwoPassDecoder<D> {
    /// Create a new two-pass decoder.
    ///
    /// `base_weights` should have one entry per edge in the matching graph,
    /// in the same order as the decoder's edge indices.
    pub fn new(inner: D, base_weights: Vec<f64>, correlation_table: CorrelationTable) -> Self {
        let n = base_weights.len();
        Self {
            inner,
            correlation_table,
            adjusted_weights: vec![0.0; n],
            base_weights,
        }
    }

    /// Whether the correlation table has any correlations to exploit.
    #[must_use]
    pub fn has_correlations(&self) -> bool {
        self.correlation_table.has_correlations()
    }
}

impl<D: MatchingDecoder> crate::ObservableDecoder for TwoPassDecoder<D> {
    fn decode_to_observables(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        if !self.correlation_table.has_correlations() {
            // No correlations: single-pass decode (no overhead)
            let (mask, _) = self.inner.decode_with_matching(syndrome)?;
            return Ok(mask);
        }

        // First pass: decode to get matched edges
        let (_, matched_edges) = self.inner.decode_with_matching(syndrome)?;

        // Apply correlation adjustments: for each matched edge, look up
        // its implied weights and lower correlated edges' weights.
        self.adjusted_weights.copy_from_slice(&self.base_weights);
        for &edge_idx in &matched_edges {
            if edge_idx < self.correlation_table.implied_weights.len() {
                for iw in &self.correlation_table.implied_weights[edge_idx] {
                    // Only lower the weight (make the correlated edge more likely)
                    if iw.conditional_weight < self.adjusted_weights[iw.target_edge_idx] {
                        self.adjusted_weights[iw.target_edge_idx] = iw.conditional_weight;
                    }
                }
            }
        }

        // Second pass: decode with adjusted weights
        let (mask, _) = self
            .inner
            .decode_with_weights(syndrome, &self.adjusted_weights)?;
        Ok(mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ObservableDecoder;
    use crate::correlation_table::ImpliedWeight;

    struct MockDecoder {
        num_edges: usize,
        calls: std::cell::RefCell<usize>,
    }

    impl MatchingDecoder for MockDecoder {
        fn decode_with_matching(
            &mut self,
            _syndrome: &[u8],
        ) -> Result<(u64, Vec<usize>), DecoderError> {
            *self.calls.borrow_mut() += 1;
            Ok((0, vec![0])) // Always match edge 0
        }

        fn decode_with_weights(
            &mut self,
            _syndrome: &[u8],
            _weights: &[f64],
        ) -> Result<(u64, Vec<usize>), DecoderError> {
            *self.calls.borrow_mut() += 1;
            Ok((1, vec![0, 1])) // Different result with adjusted weights
        }

        fn num_edges(&self) -> usize {
            self.num_edges
        }
    }

    #[test]
    fn test_two_pass_with_correlations() {
        let mock = MockDecoder {
            num_edges: 3,
            calls: std::cell::RefCell::new(0),
        };
        let weights = vec![5.0, 5.0, 5.0];
        let table = CorrelationTable {
            implied_weights: vec![
                vec![ImpliedWeight {
                    target_edge_idx: 1,
                    conditional_weight: 2.0,
                }],
                vec![],
                vec![],
            ],
            num_edges: 3,
        };

        let mut decoder = TwoPassDecoder::new(mock, weights, table);
        let mask = decoder.decode_to_observables(&[1, 0, 0]).unwrap();

        // Second pass should be called (returns mask=1)
        assert_eq!(mask, 1);
        // Two calls: first pass + second pass
        assert_eq!(*decoder.inner.calls.borrow(), 2);
    }

    #[test]
    fn test_two_pass_no_correlations() {
        let mock = MockDecoder {
            num_edges: 3,
            calls: std::cell::RefCell::new(0),
        };
        let weights = vec![5.0, 5.0, 5.0];
        let table = CorrelationTable {
            implied_weights: vec![vec![], vec![], vec![]],
            num_edges: 3,
        };

        let mut decoder = TwoPassDecoder::new(mock, weights, table);
        let mask = decoder.decode_to_observables(&[1, 0, 0]).unwrap();

        // No correlations: single pass only (returns mask=0)
        assert_eq!(mask, 0);
        assert_eq!(*decoder.inner.calls.borrow(), 1);
    }
}
