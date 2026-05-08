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

//! Pre-computed correlation table from DEM decomposition.
//!
//! Extracts pairwise edge correlations from the `^` decomposition in a DEM
//! and pre-computes conditional weights for two-pass correlated decoding.
//!
//! The algorithm (matching `PyMatching`'s implementation):
//!
//! 1. Parse decomposed error mechanisms (with `^` separators)
//! 2. For each pair of components (A, B) in a decomposed mechanism with
//!    probability p, accumulate joint probability: `p_joint = p_a*(1-p) + p*(1-p_a)`
//! 3. Also accumulate marginal probability for each edge
//! 4. Conditional probability: `P(B|A) = P_joint(A,B) / P_marginal(A)`
//! 5. Conditional weight: `w_cond = ln((1-P(B|A)) / P(B|A))`
//!
//! The conditional weight is only applied during decoding if it's LOWER than
//! the current weight (makes the correlated edge more likely).

use crate::errors::DecoderError;
use std::collections::BTreeMap;

/// An implied weight change: if edge (node1, node2) was matched,
/// change the weight of edge (`target_node1`, `target_node2`) to `conditional_weight`.
#[derive(Debug, Clone)]
pub struct ImpliedWeight {
    /// Target edge (by index in the matching graph).
    pub target_edge_idx: usize,
    /// Conditional weight: the weight edge should have given the source was matched.
    pub conditional_weight: f64,
}

/// Pre-computed correlation table from DEM decomposition.
///
/// For each edge in the matching graph, stores a list of implied weight
/// changes that should be applied when that edge is matched in the first
/// pass of a two-pass decode.
#[derive(Debug, Clone)]
pub struct CorrelationTable {
    /// For each edge index, the list of implied weights for correlated edges.
    pub implied_weights: Vec<Vec<ImpliedWeight>>,
    /// Number of edges.
    pub num_edges: usize,
}

/// Edge key for lookup: (`min_node`, `max_node`), where `max_node` = `u32::MAX` for boundary.
type EdgeKey = (u32, u32);

/// Independent probability combination (Bernoulli XOR):
/// `p_combined = p_a * (1 - p_b) + p_b * (1 - p_a)`
fn bernoulli_xor(p_a: f64, p_b: f64) -> f64 {
    p_a * (1.0 - p_b) + p_b * (1.0 - p_a)
}

/// Convert probability to weight for correlations: `ln((1-p) / p)`
/// Clamped to avoid infinite weights.
fn prob_to_weight(p: f64) -> f64 {
    let p_clamped = p.clamp(1e-15, 0.5);
    ((1.0 - p_clamped) / p_clamped).ln()
}

impl CorrelationTable {
    /// Build a correlation table from a DEM string.
    ///
    /// Parses the DEM, identifies decomposed error mechanisms (with `^`),
    /// computes joint probabilities between component pairs, and derives
    /// conditional weights.
    ///
    /// `edge_index_map` maps (node1, node2) pairs to edge indices in the
    /// matching graph (after merging). This must match the decoder's edge
    /// indexing.
    ///
    /// # Errors
    ///
    /// Returns `DecoderError` if the DEM is malformed.
    pub fn from_dem_str(
        dem: &str,
        edge_index_map: &BTreeMap<EdgeKey, usize>,
        num_edges: usize,
    ) -> Result<Self, DecoderError> {
        // Accumulate joint and marginal probabilities.
        // joint_probs[(edge_A, edge_B)] = P(A and B both fire from shared mechanisms)
        // marginal_probs[edge_A] = P(A fires from any mechanism)
        let mut joint_probs: BTreeMap<(EdgeKey, EdgeKey), f64> = BTreeMap::new();

        for line in dem.lines() {
            let line = line.trim();
            if !line.starts_with("error(") {
                continue;
            }

            let close_paren = line.find(')').ok_or_else(|| {
                DecoderError::InvalidConfiguration("Missing closing parenthesis".into())
            })?;
            let prob_str = &line[6..close_paren];
            let probability: f64 = prob_str.parse().map_err(|_| {
                DecoderError::InvalidConfiguration(format!("Invalid probability: {prob_str}"))
            })?;

            if probability <= 0.0 || probability > 0.5 {
                continue;
            }

            let tokens_str = &line[close_paren + 1..];
            let components: Vec<&str> = tokens_str.split('^').collect();

            if components.len() < 2 {
                // Non-decomposed mechanism: accumulate marginal only
                let key = parse_component_edge_key(components[0]);
                if let Some(key) = key {
                    let marginal = joint_probs.entry((key, key)).or_insert(0.0);
                    *marginal = bernoulli_xor(*marginal, probability);
                }
                continue;
            }

            // Decomposed mechanism: accumulate joint and marginal for all pairs
            let mut component_keys: Vec<EdgeKey> = Vec::new();
            for component in &components {
                if let Some(key) = parse_component_edge_key(component) {
                    component_keys.push(key);
                }
            }

            // Joint probabilities for all pairs
            for i in 0..component_keys.len() {
                for j in (i + 1)..component_keys.len() {
                    let k0 = component_keys[i];
                    let k1 = component_keys[j];
                    let p01 = joint_probs.entry((k0, k1)).or_insert(0.0);
                    *p01 = bernoulli_xor(*p01, probability);
                    let p10 = joint_probs.entry((k1, k0)).or_insert(0.0);
                    *p10 = bernoulli_xor(*p10, probability);
                }
            }

            // Marginal for each component
            for &key in &component_keys {
                let marginal = joint_probs.entry((key, key)).or_insert(0.0);
                *marginal = bernoulli_xor(*marginal, probability);
            }
        }

        // Build implied weight table
        let mut implied_weights: Vec<Vec<ImpliedWeight>> = vec![Vec::new(); num_edges];

        for (&(causal_key, affected_key), &joint_p) in &joint_probs {
            if causal_key == affected_key {
                continue; // Skip marginals
            }

            let marginal_p = joint_probs
                .get(&(causal_key, causal_key))
                .copied()
                .unwrap_or(0.0);
            if marginal_p <= 0.0 {
                continue;
            }

            let conditional_p = (joint_p / marginal_p).min(0.5);
            if conditional_p <= 0.0 {
                continue;
            }

            let conditional_weight = prob_to_weight(conditional_p);

            if let (Some(&causal_idx), Some(&affected_idx)) = (
                edge_index_map.get(&causal_key),
                edge_index_map.get(&affected_key),
            ) {
                implied_weights[causal_idx].push(ImpliedWeight {
                    target_edge_idx: affected_idx,
                    conditional_weight,
                });
            }
        }

        Ok(Self {
            implied_weights,
            num_edges,
        })
    }

    /// Check if the table has any correlations.
    #[must_use]
    pub fn has_correlations(&self) -> bool {
        self.implied_weights.iter().any(|v| !v.is_empty())
    }

    /// Number of correlated edge pairs.
    #[must_use]
    pub fn num_correlations(&self) -> usize {
        self.implied_weights.iter().map(Vec::len).sum()
    }
}

/// Parse detector indices from a DEM component string, return edge key.
fn parse_component_edge_key(component: &str) -> Option<EdgeKey> {
    let mut detectors: Vec<u32> = Vec::new();
    for token in component.split_whitespace() {
        if let Some(d_str) = token.strip_prefix('D')
            && let Ok(d) = d_str.parse::<u32>()
        {
            detectors.push(d);
        }
    }
    // Pure observables and hyperedges do not define graph edges.
    match detectors.len() {
        1 => Some((detectors[0], u32::MAX)), // Boundary edge
        2 => {
            let (a, b) = if detectors[0] <= detectors[1] {
                (detectors[0], detectors[1])
            } else {
                (detectors[1], detectors[0])
            };
            Some((a, b))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bernoulli_xor() {
        assert!((bernoulli_xor(0.1, 0.2) - 0.26).abs() < 1e-10);
        assert!((bernoulli_xor(0.0, 0.5) - 0.5).abs() < 1e-10);
        assert!((bernoulli_xor(0.5, 0.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_decomposed_dem() {
        // DEM with one decomposed mechanism: D0 D1 ^ D2 D3
        // When D0-D1 is matched, D2-D3 should get a lower weight
        let dem = "error(0.01) D0 D1 ^ D2 D3\nerror(0.02) D0 D1\nerror(0.02) D2 D3";

        let mut edge_map = BTreeMap::new();
        edge_map.insert((0, 1), 0usize); // D0-D1 = edge 0
        edge_map.insert((2, 3), 1usize); // D2-D3 = edge 1

        let table = CorrelationTable::from_dem_str(dem, &edge_map, 2).unwrap();
        assert!(table.has_correlations());

        // Edge 0 should have an implied weight for edge 1
        assert!(!table.implied_weights[0].is_empty());
        let iw = &table.implied_weights[0][0];
        assert_eq!(iw.target_edge_idx, 1);
        // The conditional weight should be lower than the unconditional
        let unconditional_weight = prob_to_weight(0.02 + 0.01); // approximate
        assert!(iw.conditional_weight < unconditional_weight);
    }

    #[test]
    fn test_no_decomposition() {
        let dem = "error(0.01) D0 D1\nerror(0.02) D2 D3";
        let mut edge_map = BTreeMap::new();
        edge_map.insert((0, 1), 0usize);
        edge_map.insert((2, 3), 1usize);

        let table = CorrelationTable::from_dem_str(dem, &edge_map, 2).unwrap();
        assert!(!table.has_correlations());
    }
}
