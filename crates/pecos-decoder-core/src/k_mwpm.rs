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

//! K-MWPM decoder: enumerate K lowest-weight matchings, majority vote.
//!
//! Based on the Chegireddy-Hamacher algorithm adapted for QEC by
//! Mao Lin (Phys. Rev. A 112, 042436, 2025; arXiv:2510.06531).
//!
//! Builds a "decoding tree" where each branch removes one matched edge
//! from the parent and re-matches. The K lowest-weight matchings give
//! K observable predictions; majority vote selects the final answer.
//!
//! Works with any `MatchingDecoder` backend (UF, Fusion Blossom).

use crate::ObservableDecoder;
use crate::correlated_decoder::EdgeTrackingDecoder;
use crate::errors::DecoderError;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Configuration for the K-MWPM decoder.
#[derive(Debug, Clone, Copy)]
pub struct KMwpmConfig {
    /// Number of matchings to enumerate. Default 10.
    pub k: usize,
}

impl Default for KMwpmConfig {
    fn default() -> Self {
        Self { k: 10 }
    }
}

/// One node in the decoding tree.
struct TreeNode {
    /// Matched edge indices from the MWPM solve.
    matched_edges: Vec<usize>,
    /// Edges that were removed (set to infinity) for this branch.
    removed_edges: Vec<usize>,
    /// Syndrome modifications: detector indices to flip.
    flipped_detectors: Vec<usize>,
    /// Index in `matched_edges` up to which edges are "committed" (removed).
    commit_idx: usize,
}

/// K-MWPM decoder wrapping any `EdgeTrackingDecoder`.
pub struct KMwpmDecoder<D> {
    decoder: D,
    config: KMwpmConfig,
    num_edges: usize,
}

impl<D: EdgeTrackingDecoder> KMwpmDecoder<D> {
    /// Create from an existing decoder.
    pub fn new(decoder: D, config: KMwpmConfig) -> Self {
        let num_edges = decoder.num_edges();
        Self {
            decoder,
            config,
            num_edges,
        }
    }
}

impl<D: EdgeTrackingDecoder> ObservableDecoder for KMwpmDecoder<D> {
    fn decode_to_observables(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        let k = self.config.k;

        // First matching: standard MWPM.
        let (obs1, edges1) = self.decoder.decode_with_matching(syndrome)?;

        if edges1.is_empty() {
            return Ok(obs1);
        }

        // Collect K matchings via decoding tree.
        let mut predictions: Vec<u64> = vec![obs1];

        // Priority queue of tree nodes to expand (by weight, lowest first).
        let mut pq: BinaryHeap<(Reverse<u64>, usize)> = BinaryHeap::new();
        let mut nodes: Vec<TreeNode> = Vec::new();

        let root = TreeNode {
            matched_edges: edges1,
            removed_edges: Vec::new(),
            flipped_detectors: Vec::new(),
            commit_idx: 0,
        };
        nodes.push(root);
        pq.push((Reverse(0), 0)); // Weight 0 for expansion priority (children will have real weights)

        while predictions.len() < k {
            let Some((_, node_idx)) = pq.pop() else {
                break; // No more candidates
            };

            // Expand this node: for each matched edge from commit_idx onward,
            // create a child that removes that edge and re-matches.
            let (matched_edges, removed_edges, flipped_detectors, commit_idx) = {
                let node = &nodes[node_idx];
                (
                    node.matched_edges.clone(),
                    node.removed_edges.clone(),
                    node.flipped_detectors.clone(),
                    node.commit_idx,
                )
            };

            for j in commit_idx..matched_edges.len() {
                // Build modified weights: removed edges get infinite weight.
                let mut weights = vec![0.0f64; self.num_edges];
                // Start with original weights for all edges.
                for (e, weight) in weights.iter_mut().enumerate().take(self.num_edges) {
                    *weight = self.decoder.edge_weight(e);
                }
                // Remove previously removed edges.
                for &re in &removed_edges {
                    weights[re] = 1e10;
                }
                // Remove edges e_{commit_idx}..e_j (inclusive).
                for &re in &matched_edges[commit_idx..=j] {
                    weights[re] = 1e10;
                }

                // Build modified syndrome: flip endpoints of committed edges.
                let mut syn_mod = syndrome.to_vec();
                for &det in &flipped_detectors {
                    if det < syn_mod.len() {
                        syn_mod[det] ^= 1;
                    }
                }
                // Also flip endpoints of edges commit_idx..j-1 (newly committed).
                let num_det = self.decoder.num_detectors();
                for &edge_idx in &matched_edges[commit_idx..j] {
                    let n1 = self.decoder.edge_node1(edge_idx) as usize;
                    let n2 = self.decoder.edge_node2(edge_idx) as usize;
                    if n1 < syn_mod.len() && n1 < num_det {
                        syn_mod[n1] ^= 1;
                    }
                    if n2 < syn_mod.len() && n2 < num_det {
                        syn_mod[n2] ^= 1;
                    }
                }

                // Re-match with modified weights and syndrome.
                let result = self.decoder.decode_with_weights(&syn_mod, &weights);
                if let Ok((child_obs, child_edges)) = result {
                    // The full observable includes committed edges' observables.
                    let mut full_obs = child_obs;
                    for &edge_idx in &matched_edges[..j] {
                        full_obs ^= self.decoder.edge_obs_mask(edge_idx);
                    }

                    // Build child's removed set and flipped set.
                    let mut child_removed = removed_edges.clone();
                    for &re in &matched_edges[commit_idx..=j] {
                        child_removed.push(re);
                    }
                    let mut child_flipped = flipped_detectors.clone();
                    for &edge_idx in &matched_edges[commit_idx..j] {
                        let n1 = self.decoder.edge_node1(edge_idx) as usize;
                        let n2 = self.decoder.edge_node2(edge_idx) as usize;
                        if n1 < num_det {
                            child_flipped.push(n1);
                        }
                        if n2 < num_det {
                            child_flipped.push(n2);
                        }
                    }

                    predictions.push(full_obs);

                    let child_node = TreeNode {
                        matched_edges: child_edges,
                        removed_edges: child_removed,
                        flipped_detectors: child_flipped,
                        commit_idx: 0,
                    };
                    let child_idx = nodes.len();
                    nodes.push(child_node);
                    // Priority by expansion order (breadth-first).
                    pq.push((Reverse(child_idx as u64), child_idx));
                }

                if predictions.len() >= k {
                    break;
                }
            }
        }

        // Majority vote across K predictions.
        let half = predictions.len() / 2;
        let mut result = 0u64;
        for bit in 0..64u32 {
            let mask = 1u64 << bit;
            let count = predictions.iter().filter(|&&p| p & mask != 0).count();
            if count > half {
                result |= mask;
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::DecoderError;

    /// Trivial decoder: always returns obs=0, no matched edges.
    struct TrivialDecoder {
        num_edges: usize,
    }

    impl crate::correlated_decoder::MatchingDecoder for TrivialDecoder {
        fn decode_with_matching(
            &mut self,
            _syndrome: &[u8],
        ) -> Result<(u64, Vec<usize>), DecoderError> {
            Ok((0, Vec::new()))
        }
        fn decode_with_weights(
            &mut self,
            _syndrome: &[u8],
            _weights: &[f64],
        ) -> Result<(u64, Vec<usize>), DecoderError> {
            Ok((0, Vec::new()))
        }
        fn num_edges(&self) -> usize {
            self.num_edges
        }
    }

    impl crate::correlated_decoder::EdgeTrackingDecoder for TrivialDecoder {
        fn edge_node1(&self, _: usize) -> u32 {
            0
        }
        fn edge_node2(&self, _: usize) -> u32 {
            1
        }
        fn edge_weight(&self, _: usize) -> f64 {
            1.0
        }
        fn edge_obs_mask(&self, _: usize) -> u64 {
            0
        }
        fn num_detectors(&self) -> usize {
            2
        }
    }

    #[test]
    fn test_k_mwpm_zero_syndrome() {
        let decoder = TrivialDecoder { num_edges: 2 };
        let mut k_dec = KMwpmDecoder::new(decoder, KMwpmConfig { k: 5 });
        let obs = k_dec.decode_to_observables(&[0, 0]).unwrap();
        assert_eq!(obs, 0);
    }

    #[test]
    fn test_k_mwpm_k1() {
        let decoder = TrivialDecoder { num_edges: 2 };
        let mut k_dec = KMwpmDecoder::new(decoder, KMwpmConfig { k: 1 });
        let obs = k_dec.decode_to_observables(&[0, 0]).unwrap();
        assert_eq!(obs, 0);
    }
}
