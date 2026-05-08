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

//! Correlated edge re-weighting for MWPM decoders.
//!
//! Implements a simplified version of the DGR (Decoding Graph Re-weighting)
//! scheme from arXiv:2311.16214. The key idea: after an initial MWPM decode,
//! adjust edge weights based on pairwise correlations between edges, then
//! re-decode with the adjusted weights.
//!
//! Two phases:
//! 1. **Alignment**: track edge frequencies across shots, update weights to
//!    match observed probabilities (corrects for DEM inaccuracies).
//! 2. **Correlation**: for each shot, adjust weights based on which correlated
//!    edges were/weren't in the initial matching, then re-decode.

/// Tracks edge occurrence statistics for alignment and correlation re-weighting.
pub struct EdgeCorrelationTracker {
    /// Number of edges in the matching graph.
    num_edges: usize,
    /// Number of shots observed.
    num_shots: usize,
    /// Per-edge occurrence count (how many times edge appeared in a matching).
    edge_counts: Vec<u64>,
    /// Pairwise co-occurrence counts: `co_counts`[i * `num_edges` + j] = count of
    /// edges i and j both appearing in the same matching.
    /// Only stores upper triangle (i < j) to save memory.
    co_counts: Vec<u64>,
}

impl EdgeCorrelationTracker {
    /// Create a new tracker for the given number of edges.
    #[must_use]
    pub fn new(num_edges: usize) -> Self {
        // For large graphs, the co-occurrence matrix is O(E^2).
        // At d=5 with ~200 edges, this is ~40K entries (320KB) -- fine.
        // At d=9 with ~2000 edges, this is ~4M entries (32MB) -- manageable.
        let co_size = num_edges * (num_edges - 1) / 2;
        Self {
            num_edges,
            num_shots: 0,
            edge_counts: vec![0; num_edges],
            co_counts: vec![0; co_size],
        }
    }

    /// Index into the upper-triangle co-occurrence array.
    fn co_index(&self, i: usize, j: usize) -> usize {
        let (a, b) = if i < j { (i, j) } else { (j, i) };
        a * self.num_edges - a * (a + 1) / 2 + b - a - 1
    }

    /// Record a matching result: which edges were selected.
    pub fn record_matching(&mut self, matched_edges: &[usize]) {
        self.num_shots += 1;

        // Update single-edge counts
        for &e in matched_edges {
            if e < self.num_edges {
                self.edge_counts[e] += 1;
            }
        }

        // Update co-occurrence counts for all pairs in the matching
        for (idx_a, &e_a) in matched_edges.iter().enumerate() {
            for &e_b in &matched_edges[idx_a + 1..] {
                if e_a < self.num_edges && e_b < self.num_edges && e_a != e_b {
                    let co_idx = self.co_index(e_a, e_b);
                    if co_idx < self.co_counts.len() {
                        self.co_counts[co_idx] += 1;
                    }
                }
            }
        }
    }

    /// Get the empirical probability of edge e appearing in a matching.
    #[must_use]
    pub fn edge_probability(&self, e: usize) -> f64 {
        if self.num_shots == 0 || e >= self.num_edges {
            return 0.0;
        }
        self.edge_counts[e] as f64 / self.num_shots as f64
    }

    /// Get the empirical co-occurrence probability of edges i and j.
    #[must_use]
    pub fn co_occurrence_probability(&self, i: usize, j: usize) -> f64 {
        if self.num_shots == 0 || i >= self.num_edges || j >= self.num_edges || i == j {
            return 0.0;
        }
        let co_idx = self.co_index(i, j);
        if co_idx >= self.co_counts.len() {
            return 0.0;
        }
        self.co_counts[co_idx] as f64 / self.num_shots as f64
    }

    /// Number of shots recorded.
    #[must_use]
    pub fn num_shots(&self) -> usize {
        self.num_shots
    }

    /// Compute alignment-reweighted edge weights.
    ///
    /// Replaces original DEM weights with weights derived from observed
    /// edge frequencies: `w_e` = -`ln(p_e` / (1 - `p_e`)) where `p_e` is the
    /// empirical edge probability.
    #[must_use]
    pub fn aligned_weights(&self) -> Vec<f64> {
        (0..self.num_edges)
            .map(|e| {
                let p = self.edge_probability(e);
                if p <= 0.0 || p >= 1.0 {
                    // Edge never/always matched -- keep a default weight
                    if p <= 0.0 { 20.0 } else { 0.0 }
                } else {
                    ((1.0 - p) / p).ln()
                }
            })
            .collect()
    }

    /// Compute correlation-adjusted weights for a specific matching.
    ///
    /// Given an initial matching M, adjust each edge weight based on
    /// correlations with matched/unmatched edges (DGR Equation 1):
    ///
    ///   `w̃_j` = `w_j` - Σ_{`e_i` ∈ M} `p(e_i,e_j)/p(e_i)` + Σ_{`e_i` ∉ M} `p(e_i,e_j)/p(e_i)`
    ///
    /// Intuition: if edge j is correlated with matched edges, decrease its
    /// weight (make it more likely). If correlated with unmatched edges,
    /// increase its weight (make it less likely).
    /// Compute correlation-adjusted weights for a specific matching.
    ///
    /// Given an initial matching M, adjust each edge weight based on
    /// correlations with matched/unmatched edges (DGR Equation 1):
    ///
    ///   `w̃_j` = `w_j` - Σ_{`e_i` ∈ M} `p(e_i,e_j)/p(e_i)` + Σ_{`e_i` ∉ M} `p(e_i,e_j)/p(e_i)`
    ///
    /// Only edges with significant correlation (conditional probability >
    /// `min_conditional`) contribute to the sum. This avoids the bias from
    /// summing over hundreds of near-zero terms.
    #[must_use]
    pub fn correlation_adjusted_weights(
        &self,
        base_weights: &[f64],
        matched_edges: &[bool],
    ) -> Vec<f64> {
        self.correlation_adjusted_weights_filtered(base_weights, matched_edges, 0.05)
    }

    /// Correlation adjustment with configurable significance threshold.
    ///
    /// `min_conditional`: minimum p(i,j)/p(i) to include an edge pair.
    /// Higher values mean fewer pairs contribute (sparser adjustment).
    #[must_use]
    pub fn correlation_adjusted_weights_filtered(
        &self,
        base_weights: &[f64],
        matched_edges: &[bool],
        min_conditional: f64,
    ) -> Vec<f64> {
        let mut adjusted = base_weights.to_vec();

        for (j, adjusted_weight) in adjusted.iter_mut().enumerate().take(self.num_edges) {
            let mut adjustment = 0.0;
            for (i, matched) in matched_edges.iter().enumerate().take(self.num_edges) {
                if i == j {
                    continue;
                }
                let p_i = self.edge_probability(i);
                if p_i <= 0.0 {
                    continue;
                }
                let p_ij = self.co_occurrence_probability(i, j);
                let conditional = p_ij / p_i;

                // Only include significantly correlated pairs
                if conditional < min_conditional {
                    continue;
                }

                if *matched {
                    // Correlated edge is matched -> decrease weight (more likely).
                    // We only adjust for matched edges -- the unmatched term
                    // creates a large positive bias that dominates when most
                    // edges are unmatched.
                    adjustment -= conditional;
                }
            }
            *adjusted_weight += adjustment;
            // Clamp to prevent negative weights
            if *adjusted_weight < 0.0 {
                *adjusted_weight = 0.0;
            }
        }

        adjusted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracker_basic() {
        let mut tracker = EdgeCorrelationTracker::new(5);

        // Record some matchings
        tracker.record_matching(&[0, 2]);
        tracker.record_matching(&[0, 3]);
        tracker.record_matching(&[1, 2]);
        tracker.record_matching(&[0, 2]);

        assert_eq!(tracker.num_shots(), 4);
        assert!((tracker.edge_probability(0) - 0.75).abs() < 1e-10);
        assert!((tracker.edge_probability(1) - 0.25).abs() < 1e-10);
        assert!((tracker.edge_probability(2) - 0.75).abs() < 1e-10);
        assert!((tracker.edge_probability(4) - 0.0).abs() < 1e-10);

        // Co-occurrence: edges 0 and 2 both appear in 2 matchings
        assert!((tracker.co_occurrence_probability(0, 2) - 0.5).abs() < 1e-10);
        // Edges 0 and 3 both appear in 1 matching
        assert!((tracker.co_occurrence_probability(0, 3) - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_aligned_weights() {
        let mut tracker = EdgeCorrelationTracker::new(3);
        for _ in 0..100 {
            tracker.record_matching(&[0]);
        }
        for _ in 0..900 {
            tracker.record_matching(&[]);
        }
        // Edge 0 appears in 10% of matchings -> p=0.1 -> w = ln(0.9/0.1) = ln(9) ≈ 2.197
        let weights = tracker.aligned_weights();
        assert!((weights[0] - 9.0_f64.ln()).abs() < 0.01);
    }

    #[test]
    fn test_correlation_adjustment() {
        let mut tracker = EdgeCorrelationTracker::new(3);
        // Edges 0 and 1 are highly correlated (always appear together)
        for _ in 0..100 {
            tracker.record_matching(&[0, 1]);
        }
        for _ in 0..900 {
            tracker.record_matching(&[]);
        }

        let base_weights = vec![2.0, 2.0, 2.0];
        // If edge 0 is matched, edge 1's weight should decrease (correlated)
        let matched = vec![true, false, false];
        let adjusted = tracker.correlation_adjusted_weights(&base_weights, &matched);

        // Edge 1 should have lower weight (more likely given edge 0 is matched)
        assert!(adjusted[1] < base_weights[1]);
        // Edge 2 should be unchanged (no correlation with edge 0)
        assert!((adjusted[2] - base_weights[2]).abs() < 1e-10);
    }
}
