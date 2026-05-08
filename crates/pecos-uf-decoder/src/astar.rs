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

//! A* error-set decoder inspired by Tesseract (Google, arXiv:2503.10988).
//!
//! Searches over error mechanism subsets to find the minimum-weight
//! error set consistent with the syndrome. Uses `DetCost` heuristic
//! (per-detector minimum mechanism cost) for admissible pruning.
//!
//! Key pruning strategies from Tesseract:
//! - **Canonical expansion**: only expand mechanisms incident to the
//!   lowest-index unsatisfied detector
//! - **No-revisit-dets**: skip states with previously-seen residual syndromes
//! - **Beam**: skip states with too many residual defects
//! - **PQ limit**: terminate after a fixed number of expansions
//!
//! Uses u64 bitsets for compact state representation and fast operations.

use pecos_decoder_core::ObservableDecoder;
use pecos_decoder_core::dem::DemMatchingGraph;
use pecos_decoder_core::errors::DecoderError;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

/// A mechanism (error) in the DEM.
struct Mechanism {
    detectors: Vec<u32>,
    obs_mask: u64,
    weight: f64,
}

/// Configuration for the A* decoder.
#[derive(Debug, Clone, Copy)]
pub struct AStarConfig {
    /// Maximum priority queue pops before terminating.
    pub pq_limit: usize,
    /// Beam: skip states with > beam more residual defects than best seen.
    pub beam: usize,
}

impl Default for AStarConfig {
    fn default() -> Self {
        Self {
            pq_limit: 50_000,
            beam: 20,
        }
    }
}

/// Compact bitset for detector/mechanism membership.
#[derive(Clone, PartialEq, Eq, Hash)]
struct Bitset {
    words: Vec<u64>,
}

impl Bitset {
    fn new(n: usize) -> Self {
        Self {
            words: vec![0u64; n.div_ceil(64)],
        }
    }

    fn get(&self, i: usize) -> bool {
        let (word, bit) = (i / 64, i % 64);
        word < self.words.len() && (self.words[word] & (1u64 << bit)) != 0
    }

    fn set(&mut self, i: usize) {
        let (word, bit) = (i / 64, i % 64);
        if word < self.words.len() {
            self.words[word] |= 1u64 << bit;
        }
    }

    fn flip(&mut self, i: usize) {
        let (word, bit) = (i / 64, i % 64);
        if word < self.words.len() {
            self.words[word] ^= 1u64 << bit;
        }
    }

    fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Find the index of the lowest set bit, or None.
    fn lowest_set(&self) -> Option<usize> {
        for (wi, &w) in self.words.iter().enumerate() {
            if w != 0 {
                return Some(wi * 64 + w.trailing_zeros() as usize);
            }
        }
        None
    }
}

/// A* error-set search decoder.
pub struct AStarDecoder {
    mechanisms: Vec<Mechanism>,
    /// Per-detector: list of mechanism indices incident to this detector.
    det_to_mechs: Vec<Vec<usize>>,
    num_detectors: usize,
    num_mechanisms: usize,
    config: AStarConfig,
}

/// State in the A* search (compact representation).
struct SearchState {
    errors: Bitset,
    residual: Bitset,
    num_residual: usize,
    g_cost: f64,
    obs_mask: u64,
    /// Per-detector: how many included errors are incident to this detector.
    det_error_count: Vec<u8>,
    /// Mechanisms forbidden by `ByPrecedence`: were available at an earlier
    /// step but not chosen. Cannot be added in future steps.
    forbidden: Bitset,
}

impl AStarDecoder {
    /// Build from a DEM string (graphlike — 2-detector edges only).
    ///
    /// # Errors
    ///
    /// Returns `DecoderError` if the DEM is malformed.
    pub fn from_dem(dem: &str, config: AStarConfig) -> Result<Self, DecoderError> {
        let graph = DemMatchingGraph::from_dem_str(dem)?;
        let num_detectors = graph.num_detectors;

        let mut mechanisms = Vec::new();
        let mut det_to_mechs: Vec<Vec<usize>> = vec![Vec::new(); num_detectors];

        for edge in &graph.edges {
            let mut detectors = vec![edge.node1];
            if let Some(n2) = edge.node2 {
                detectors.push(n2);
            }

            let obs_mask: u64 = edge
                .observables
                .iter()
                .fold(0u64, |mask, &o| mask | (1 << o));

            let mech_idx = mechanisms.len();
            for &d in &detectors {
                if (d as usize) < num_detectors {
                    det_to_mechs[d as usize].push(mech_idx);
                }
            }

            mechanisms.push(Mechanism {
                detectors,
                obs_mask,
                weight: edge.weight,
            });
        }

        let num_mechanisms = mechanisms.len();

        Ok(Self {
            mechanisms,
            det_to_mechs,
            num_detectors,
            num_mechanisms,
            config,
        })
    }

    /// Build from a non-decomposed DEM (preserves hyperedges with 3+ detectors).
    ///
    /// This gives the A* search access to the full error structure including
    /// Y-error correlations that decomposition loses.
    ///
    /// # Errors
    ///
    /// Returns `DecoderError` if the DEM is malformed.
    pub fn from_dem_full(dem: &str, config: AStarConfig) -> Result<Self, DecoderError> {
        use pecos_decoder_core::dem::DemCheckMatrix;

        let dcm = DemCheckMatrix::from_dem_str(dem)
            .map_err(|e| DecoderError::InvalidGraph(e.to_string()))?;
        let num_detectors = dcm.num_detectors;

        let mut mechanisms = Vec::new();
        let mut det_to_mechs: Vec<Vec<usize>> = vec![Vec::new(); num_detectors];

        for m in 0..dcm.num_mechanisms {
            let p = dcm.error_priors[m];
            if p <= 0.0 {
                continue;
            }

            let detectors: Vec<u32> = (0..dcm.num_detectors)
                .filter(|&d| dcm.check_matrix[[d, m]] != 0)
                .map(|d| d as u32)
                .collect();

            if detectors.is_empty() {
                continue;
            }

            let weight = if p < 1.0 { ((1.0 - p) / p).ln() } else { 0.0 };

            let mut obs_mask = 0u64;
            for o in 0..dcm.num_observables {
                if dcm.observable_matrix[[o, m]] != 0 {
                    obs_mask |= 1 << o;
                }
            }

            let mech_idx = mechanisms.len();
            for &d in &detectors {
                if (d as usize) < num_detectors {
                    det_to_mechs[d as usize].push(mech_idx);
                }
            }

            mechanisms.push(Mechanism {
                detectors,
                obs_mask,
                weight,
            });
        }

        let num_mechanisms = mechanisms.len();

        for mechs in &mut det_to_mechs {
            mechs.sort_by(|&a, &b| {
                mechanisms[a]
                    .weight
                    .partial_cmp(&mechanisms[b].weight)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        Ok(Self {
            mechanisms,
            det_to_mechs,
            num_detectors,
            num_mechanisms,
            config,
        })
    }

    /// Compute `DetCost` heuristic: admissible lower bound on remaining cost.
    /// Optimized with early exit: mechanisms per detector are pre-sorted by weight,
    /// so once weight exceeds `current_min` * `max_possible_coverage`, we can stop.
    fn det_cost(&self, residual: &Bitset, errors: &Bitset) -> f64 {
        let mut cost = 0.0;
        for (wi, &w) in residual.words.iter().enumerate() {
            let mut bits = w;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                let d = wi * 64 + bit;
                bits &= bits - 1;

                if d >= self.num_detectors {
                    break;
                }

                let mut min_cost = f64::INFINITY;
                for &m in &self.det_to_mechs[d] {
                    if errors.get(m) {
                        continue;
                    }
                    let mech = &self.mechanisms[m];

                    // Early skip: if weight / max_coverage >= min_cost, skip.
                    // max_coverage = number of detectors in this mechanism.
                    let max_cov = mech.detectors.len() as f64;
                    if max_cov > 0.0 && mech.weight / max_cov >= min_cost {
                        continue;
                    }

                    let coverage = mech
                        .detectors
                        .iter()
                        .filter(|&&dd| {
                            (dd as usize) < self.num_detectors && residual.get(dd as usize)
                        })
                        .count();
                    if coverage > 0 {
                        let c = mech.weight / coverage as f64;
                        if c < min_cost {
                            min_cost = c;
                        }
                    }
                }
                if min_cost < f64::INFINITY {
                    cost += min_cost;
                }
            }
        }
        cost
    }
}

impl ObservableDecoder for AStarDecoder {
    fn decode_to_observables(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        let n = self.num_detectors;
        let m = self.num_mechanisms;

        // Build initial residual.
        let mut init_residual = Bitset::new(n);
        for (i, &v) in syndrome.iter().enumerate() {
            if v != 0 && i < n {
                init_residual.set(i);
            }
        }
        let num_defects = init_residual.count_ones();
        if num_defects == 0 {
            return Ok(0);
        }

        // A* priority queue and visited set.
        let mut pq: BinaryHeap<(Reverse<u64>, usize)> = BinaryHeap::new();
        let mut states: Vec<SearchState> = Vec::new();
        let mut visited: HashSet<Bitset> = HashSet::new();

        let init_errors = Bitset::new(m);
        let init_h = self.det_cost(&init_residual, &init_errors);

        states.push(SearchState {
            errors: init_errors,
            residual: init_residual.clone(),
            num_residual: num_defects,
            g_cost: 0.0,
            obs_mask: 0,
            det_error_count: vec![0u8; n],
            forbidden: Bitset::new(m),
        });
        pq.push((Reverse((0.0_f64 + init_h).to_bits()), 0));
        visited.insert(init_residual);

        let mut best_obs = 0u64;
        let best_cost = f64::INFINITY;
        let mut min_residual = num_defects;
        let mut pops = 0;

        while let Some((Reverse(_), state_idx)) = pq.pop() {
            pops += 1;
            if pops > self.config.pq_limit {
                break;
            }

            // Extract state data (avoids borrow conflict).
            let (s_num_res, s_g_cost, s_obs) = {
                let s = &states[state_idx];
                (s.num_residual, s.g_cost, s.obs_mask)
            };

            if s_num_res == 0 {
                best_obs = s_obs;
                break; // A* first solution is optimal (admissible heuristic).
            }

            if s_num_res > min_residual + self.config.beam {
                continue;
            }
            if s_num_res < min_residual {
                min_residual = s_num_res;
            }

            // Clone for expansion.
            let (s_errors, s_residual, s_det_counts, s_forbidden) = {
                let s = &states[state_idx];
                (
                    s.errors.clone(),
                    s.residual.clone(),
                    s.det_error_count.clone(),
                    s.forbidden.clone(),
                )
            };

            let lowest_det = match s_residual.lowest_set() {
                Some(d) if d < n => d,
                _ => continue,
            };

            // Collect candidate mechanisms incident to lowest_det.
            let candidates: Vec<usize> = self.det_to_mechs[lowest_det]
                .iter()
                .copied()
                .filter(|&mi| !s_errors.get(mi) && !s_forbidden.get(mi))
                .collect();

            for &mech_idx in &candidates {
                let mech = &self.mechanisms[mech_idx];

                // AtMostTwo: skip if adding this mechanism would place >2 errors
                // on any single detector.
                let at_most_two_ok = mech.detectors.iter().all(|&d| {
                    let di = d as usize;
                    di >= n || s_det_counts[di] < 2
                });
                if !at_most_two_ok {
                    continue;
                }

                let mut new_residual = s_residual.clone();
                let mut new_num = s_num_res;
                let mut new_det_counts = s_det_counts.clone();
                for &d in &mech.detectors {
                    let di = d as usize;
                    if di < n {
                        if new_residual.get(di) {
                            new_num -= 1;
                        } else {
                            new_num += 1;
                        }
                        new_residual.flip(di);
                        new_det_counts[di] += 1;
                    }
                }

                // ByPrecedence: all other candidate mechanisms at this step
                // become forbidden for this child. They were available but not chosen.
                let mut new_forbidden = s_forbidden.clone();
                for &other in &candidates {
                    if other != mech_idx {
                        new_forbidden.set(other);
                    }
                }

                // No-revisit-dets: skip if we've seen this residual before.
                if !visited.insert(new_residual.clone()) {
                    continue;
                }

                let mut new_errors = s_errors.clone();
                new_errors.set(mech_idx);

                let new_g = s_g_cost + mech.weight;
                let new_h = self.det_cost(&new_residual, &new_errors);
                let new_f = new_g + new_h;

                if new_f >= best_cost {
                    continue;
                }

                let idx = states.len();
                states.push(SearchState {
                    errors: new_errors,
                    residual: new_residual,
                    num_residual: new_num,
                    g_cost: new_g,
                    obs_mask: s_obs ^ mech.obs_mask,
                    det_error_count: new_det_counts,
                    forbidden: new_forbidden,
                });
                pq.push((Reverse(new_f.to_bits()), idx));
            }
        }

        Ok(best_obs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const D3_DEM: &str =
        include_str!("../../../examples/surface_code_circuits/surface_code_d3_z_stim.dem");

    #[test]
    fn test_astar_construction() {
        let dec = AStarDecoder::from_dem(D3_DEM, AStarConfig::default());
        assert!(dec.is_ok());
    }

    #[test]
    fn test_astar_no_errors() {
        let graph = DemMatchingGraph::from_dem_str(D3_DEM).unwrap();
        let mut dec = AStarDecoder::from_dem(D3_DEM, AStarConfig::default()).unwrap();
        let obs = dec
            .decode_to_observables(&vec![0u8; graph.num_detectors])
            .unwrap();
        assert_eq!(obs, 0);
    }

    #[test]
    fn test_astar_single_defect() {
        let graph = DemMatchingGraph::from_dem_str(D3_DEM).unwrap();
        let mut dec = AStarDecoder::from_dem(D3_DEM, AStarConfig::default()).unwrap();
        let mut syn = vec![0u8; graph.num_detectors];
        syn[0] = 1;
        // Should not panic — single defect resolves to boundary.
        let _obs = dec.decode_to_observables(&syn).unwrap();
    }
}
