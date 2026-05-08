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

//! BP+UF hybrid decoder.
//!
//! Runs truncated min-sum BP to get per-mechanism soft reliability scores,
//! then uses those scores to adjust UF edge weights. Mechanisms that BP
//! identifies as likely errors get lower weights in UF, improving the
//! quality of the UF clustering.
//!
//! This is a three-stage decoder:
//! 1. **BP stage**: 3-5 iterations of min-sum BP on the check matrix
//! 2. **Weight adjustment**: map BP posteriors to UF edge weights
//! 3. **UF stage**: standard weighted UF growth + peeling

use crate::decoder::{UfDecoder, UfDecoderConfig};
use crate::mini_bp::{self, BpGraph};
use pecos_decoder_core::correlated_decoder::MatchingDecoder;
use pecos_decoder_core::dem::{DemCheckMatrix, DemMatchingGraph};
use pecos_decoder_core::errors::DecoderError;

/// BP message schedule.
#[derive(Debug, Clone, Copy, Default)]
pub enum BpSchedule {
    /// Flooding: update all checks, then all variables. Fast, good for d<=7.
    #[default]
    Flooding,
    /// Serial: after each check update, immediately update connected variables.
    /// Better convergence on loopy graphs. Slower but maintains threshold at d>=9.
    Serial,
}

/// Which graph to run BP on.
#[derive(Debug, Clone, Copy, Default)]
pub enum BpGraphType {
    /// Auto: matching-graph BP at d<=4, Tanner-graph BP at d>=5.
    /// Gets the best of both worlds at each distance.
    #[default]
    Auto,
    /// Tanner graph from check matrix (decomposed DEM).
    TannerGraph,
    /// Matching graph (pairwise detector edges). Simpler topology,
    /// better convergence. Based on Hack et al. (2026).
    MatchingGraph,
}

/// Configuration for the BP+UF hybrid decoder.
#[derive(Debug, Clone, Copy)]
pub struct BpUfConfig {
    /// Number of BP iterations before UF.
    /// 0 = adaptive (scales with code distance). Default: 0.
    pub bp_iterations: usize,
    /// BP message schedule. Default: Flooding (fast, good for d<=7).
    pub bp_schedule: BpSchedule,
    /// Which graph to run BP on. Default: `TannerGraph`.
    pub bp_graph_type: BpGraphType,
    /// Min-sum scaling factor. Default: 0.625 (normalized min-sum).
    pub min_sum_scale: f64,
    /// How much BP posteriors influence UF weights.
    /// 0.0 = pure UF, 1.0 = fully trust BP. Default: 0.9.
    pub bp_weight_blend: f64,
    /// UF decoder config.
    pub uf_config: UfDecoderConfig,
}

impl Default for BpUfConfig {
    fn default() -> Self {
        Self::balanced()
    }
}

impl BpUfConfig {
    /// Balanced: flooding BP on Tanner graph, good for d=3-7. Fast.
    #[must_use]
    pub fn balanced() -> Self {
        Self {
            bp_iterations: 0,
            bp_schedule: BpSchedule::Flooding,
            bp_graph_type: BpGraphType::Auto,
            min_sum_scale: 0.625,
            bp_weight_blend: 0.9,
            uf_config: UfDecoderConfig::balanced(),
        }
    }

    /// Accurate: serial BP, maintains threshold at d=7-11+. Slower.
    #[must_use]
    pub fn accurate() -> Self {
        Self {
            bp_iterations: 0,
            bp_schedule: BpSchedule::Serial,
            bp_graph_type: BpGraphType::Auto,
            min_sum_scale: 0.625,
            bp_weight_blend: 0.9,
            uf_config: UfDecoderConfig::balanced(),
        }
    }

    /// Matching-graph BP: run BP on the simpler matching graph.
    #[must_use]
    pub fn matching_bp() -> Self {
        Self {
            bp_iterations: 0,
            bp_schedule: BpSchedule::Flooding,
            bp_graph_type: BpGraphType::MatchingGraph,
            min_sum_scale: 0.625,
            bp_weight_blend: 0.9,
            uf_config: UfDecoderConfig::balanced(),
        }
    }
}

/// BP+UF hybrid decoder.
pub struct BpUfDecoder {
    /// Inner UF decoder.
    uf: UfDecoder,
    /// Pre-computed sparse BP graph for Tanner graph BP.
    bp_graph: BpGraph,
    /// Matching graph (stored for matching-graph BP mode).
    matching_graph: Option<DemMatchingGraph>,
    /// Mapping from mechanism index to matching graph edge index.
    mechanism_to_edge: Vec<Option<usize>>,
    /// Base edge weights (from DEM, before BP adjustment).
    base_weights: Vec<f64>,
    /// BP-adjusted weights (reusable buffer).
    adjusted_weights: Vec<f64>,
    /// BP message buffers (reusable across shots).
    bp_c_to_v: Vec<f64>,
    bp_v_to_c: Vec<f64>,
    bp_posterior: Vec<f64>,
    /// Config.
    config: BpUfConfig,
}

impl BpUfDecoder {
    /// Create from a DEM string.
    ///
    /// # Errors
    ///
    /// Returns `DecoderError` if the DEM is malformed.
    pub fn from_dem(dem: &str, config: BpUfConfig) -> Result<Self, DecoderError> {
        let dcm = DemCheckMatrix::from_dem_str(dem)
            .map_err(|e| DecoderError::InvalidConfiguration(e.to_string()))?;
        let graph = DemMatchingGraph::from_dem_str(dem)?;
        let uf = UfDecoder::from_matching_graph(&graph, config.uf_config);

        // Build mechanism → edge mapping.
        // Each mechanism in the check matrix corresponds to a column.
        // The matching graph merges mechanisms by fault ID into edges.
        // We need to find which edge each mechanism ended up in.
        //
        // Approach: for each mechanism, find which detectors it touches,
        // then find the matching edge connecting those detectors.
        let mut mechanism_to_edge = vec![None; dcm.num_mechanisms];

        for (m, mechanism_edge) in mechanism_to_edge
            .iter_mut()
            .enumerate()
            .take(dcm.num_mechanisms)
        {
            let mut detectors: Vec<u32> = Vec::new();
            for d in 0..dcm.num_detectors {
                if dcm.check_matrix[[d, m]] != 0 {
                    detectors.push(d as u32);
                }
            }

            // Match to graph edge by detector pair.
            match detectors.len() {
                1 => {
                    // Boundary edge: one detector.
                    let d0 = detectors[0];
                    for (idx, edge) in graph.edges.iter().enumerate() {
                        if edge.node1 == d0 && edge.node2.is_none() {
                            *mechanism_edge = Some(idx);
                            break;
                        }
                    }
                }
                2 => {
                    // Internal edge: two detectors.
                    let (d0, d1) = (detectors[0], detectors[1]);
                    for (idx, edge) in graph.edges.iter().enumerate() {
                        if (edge.node1 == d0 && edge.node2 == Some(d1))
                            || (edge.node1 == d1 && edge.node2 == Some(d0))
                        {
                            *mechanism_edge = Some(idx);
                            break;
                        }
                    }
                }
                _ => {
                    // Hyperedge (3+ detectors): skip, no matching graph edge.
                }
            }
        }

        let base_weights: Vec<f64> = graph.edges.iter().map(|e| e.weight).collect();
        let adjusted_weights = base_weights.clone();

        let bp_graph_data = BpGraph::from_dcm(&dcm);
        let bp_c_to_v = vec![0.0; bp_graph_data.total_edges];
        let bp_v_to_c = vec![0.0; bp_graph_data.total_edges];
        let bp_posterior = Vec::with_capacity(bp_graph_data.num_vars);

        // Always store matching graph (needed for Auto and MatchingGraph modes).
        let matching_graph_stored = Some(graph);

        Ok(Self {
            uf,
            bp_graph: bp_graph_data,
            matching_graph: matching_graph_stored,
            mechanism_to_edge,
            base_weights,
            adjusted_weights,
            bp_c_to_v,
            bp_v_to_c,
            bp_posterior,
            config,
        })
    }
}

impl BpUfDecoder {
    /// Create from two DEMs: non-decomposed for BP, decomposed for matching graph.
    ///
    /// The non-decomposed DEM gives BP cleaner soft info (no decomposition
    /// artifacts). The decomposed DEM gives the matching graph edge structure
    /// needed for MWPM and correlation tables.
    ///
    /// # Errors
    ///
    /// Returns `DecoderError` if either DEM is malformed.
    pub fn from_dual_dem(
        bp_dem: &str,
        matching_dem: &str,
        config: BpUfConfig,
    ) -> Result<Self, DecoderError> {
        // BP graph from the non-decomposed DEM.
        let bp_dcm = DemCheckMatrix::from_dem_str(bp_dem)
            .map_err(|e| DecoderError::InvalidConfiguration(e.to_string()))?;
        let bp_graph = BpGraph::from_dcm(&bp_dcm);

        // Matching graph and UF from the decomposed DEM.
        let match_graph = DemMatchingGraph::from_dem_str(matching_dem)?;
        let uf = UfDecoder::from_matching_graph(&match_graph, config.uf_config);

        // Map BP mechanisms (non-decomposed) → matching graph edges (decomposed).
        let mut mechanism_to_edge = vec![None; bp_dcm.num_mechanisms];
        for (m, mechanism_edge) in mechanism_to_edge
            .iter_mut()
            .enumerate()
            .take(bp_dcm.num_mechanisms)
        {
            let mut detectors: Vec<u32> = Vec::new();
            for d in 0..bp_dcm.num_detectors {
                if bp_dcm.check_matrix[[d, m]] != 0 {
                    detectors.push(d as u32);
                }
            }
            match detectors.len() {
                1 => {
                    let d0 = detectors[0];
                    for (idx, edge) in match_graph.edges.iter().enumerate() {
                        if edge.node1 == d0 && edge.node2.is_none() {
                            *mechanism_edge = Some(idx);
                            break;
                        }
                    }
                }
                2 => {
                    let (d0, d1) = (detectors[0], detectors[1]);
                    for (idx, edge) in match_graph.edges.iter().enumerate() {
                        if (edge.node1 == d0 && edge.node2 == Some(d1))
                            || (edge.node1 == d1 && edge.node2 == Some(d0))
                        {
                            *mechanism_edge = Some(idx);
                            break;
                        }
                    }
                }
                _ => {} // Hyperedge
            }
        }

        let base_weights: Vec<f64> = match_graph.edges.iter().map(|e| e.weight).collect();
        let adjusted_weights = base_weights.clone();
        let total_edges = bp_graph.total_edges;

        Ok(Self {
            uf,
            bp_graph,
            matching_graph: None, // dual_dem mode uses Tanner graph BP
            mechanism_to_edge,
            base_weights,
            adjusted_weights,
            bp_c_to_v: vec![0.0; total_edges],
            bp_v_to_c: vec![0.0; total_edges],
            bp_posterior: Vec::with_capacity(bp_dcm.num_mechanisms),
            config,
        })
    }
}

impl pecos_decoder_core::bp_matching::BpWeightProvider for BpUfDecoder {
    fn compute_weights(&mut self, syndrome: &[u8]) -> Vec<f64> {
        let num_defects = syndrome.iter().filter(|&&v| v != 0).count();

        let iters = if self.config.bp_iterations > 0 {
            self.config.bp_iterations
        } else {
            let d_est = ((self
                .matching_graph
                .as_ref()
                .map_or(self.bp_graph.num_checks, |mg| mg.num_detectors)
                as f64)
                / 2.0)
                .sqrt();
            match num_defects {
                2..=3 => (d_est.ceil() as usize).min(3),
                4..=8 => (d_est.ceil() as usize).min(8),
                _ => (d_est.ceil() as usize).min(12),
            }
        };

        // Estimate code distance from matching graph detectors.
        // For surface codes: num_detectors ≈ num_stab * num_rounds ≈ d * 2d = 2d²
        // so d ≈ sqrt(num_detectors / 2).
        let num_det = self
            .matching_graph
            .as_ref()
            .map_or(self.bp_graph.num_checks, |mg| mg.num_detectors);
        let d_est = ((num_det as f64) / 2.0).sqrt();
        let use_matching_graph = match self.config.bp_graph_type {
            BpGraphType::MatchingGraph => true,
            BpGraphType::TannerGraph => false,
            BpGraphType::Auto => d_est < 5.5, // Matching graph at d<=4 (d_est≈4.9 at d=3)
        };

        if let (true, Some(mg)) = (use_matching_graph, &self.matching_graph) {
            // Matching-graph BP: simpler topology, better convergence.
            self.bp_posterior =
                mini_bp::matching_graph_bp(mg, syndrome, iters, self.config.min_sum_scale);
            // Matching-graph BP posteriors are already per-edge (no mechanism mapping needed).
            // Return them directly as weights.
            let mut weights = self.base_weights.clone();
            let d_est = ((self
                .matching_graph
                .as_ref()
                .map_or(self.bp_graph.num_checks, |mg| mg.num_detectors)
                as f64)
                / 2.0)
                .sqrt();
            let selectivity = 0.2 * d_est.max(1.0) / 3.0;
            for (edge_idx, &posterior) in self.bp_posterior.iter().enumerate() {
                if edge_idx < weights.len() {
                    let prior = self.base_weights[edge_idx];
                    let shift = (posterior - prior).abs();
                    if shift > selectivity * prior.abs().max(0.1) {
                        let bp_weight = if posterior > 10.0 {
                            posterior
                        } else if posterior < -10.0 {
                            0.01
                        } else {
                            (1.0 + posterior.exp()).ln()
                        };
                        let blend = 0.5;
                        weights[edge_idx] = (1.0 - blend) * prior + blend * bp_weight;
                    }
                }
            }
            return weights;
        }

        // At d>=5 with auto mode, selective BP rarely adjusts edges.
        // Skip BP entirely and return base weights (= FB_correlated behavior).
        // The correlation table second pass in BpMatchingDecoder handles accuracy.
        if matches!(self.config.bp_graph_type, BpGraphType::Auto) && d_est >= 5.5 {
            return self.base_weights.clone();
        }

        // Tanner-graph BP.
        self.bp_c_to_v.fill(0.0);
        self.bp_v_to_c.fill(0.0);
        let serial = matches!(self.config.bp_schedule, BpSchedule::Serial);
        mini_bp::min_sum_bp_into(
            &self.bp_graph,
            syndrome,
            iters,
            self.config.min_sum_scale,
            serial,
            &mut self.bp_c_to_v,
            &mut self.bp_v_to_c,
            &mut self.bp_posterior,
        );

        // Selective BP: only adjust edges where BP strongly disagrees with
        // the DEM prior. This avoids BP noise (which hurts at d>=5) while
        // capturing genuine syndrome-dependent information for edges where
        // BP has high confidence.
        //
        // An edge is adjusted if |posterior - prior| > threshold * |prior|.
        // This means BP must shift the LLR by a significant fraction of the
        // prior to be trusted.
        let mut weights = self.base_weights.clone();
        let d_est = ((self
            .matching_graph
            .as_ref()
            .map_or(self.bp_graph.num_checks, |mg| mg.num_detectors) as f64)
            / 2.0)
            .sqrt();
        // Selectivity threshold: higher at large d (BP less reliable).
        // At d=3: threshold=0.3 (accept moderate BP shifts).
        // At d=7: threshold=0.8 (only trust strong BP shifts).
        // At d=11: threshold=1.2 (very selective).
        let selectivity = 0.2 * d_est.max(1.0) / 3.0;

        for (m, &posterior) in self.bp_posterior.iter().enumerate() {
            if let Some(edge_idx) = self.mechanism_to_edge[m] {
                let prior = self.bp_graph.prior_llr[m];
                let shift = (posterior - prior).abs();

                // Only use BP weight if the shift is large relative to the prior.
                if shift > selectivity * prior.abs().max(0.1) {
                    let bp_weight = if posterior > 10.0 {
                        posterior
                    } else if posterior < -10.0 {
                        0.01
                    } else {
                        (1.0 + posterior.exp()).ln()
                    };
                    // Blend with a moderate factor for selected edges.
                    let blend = 0.5;
                    let blended = (1.0 - blend) * self.base_weights[edge_idx] + blend * bp_weight;
                    weights[edge_idx] = weights[edge_idx].min(blended);
                }
            }
        }
        weights
    }

    fn num_edges(&self) -> usize {
        self.base_weights.len()
    }

    fn is_trivial(&self, syndrome: &[u8]) -> Option<u64> {
        self.uf.predecode_clusters(syndrome)
    }
}

impl pecos_decoder_core::ObservableDecoder for BpUfDecoder {
    fn decode_to_observables(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        // Fast path: cluster predecoder handles isolated cases without BP.
        // This catches 0 defects, single defects, and isolated pairs.
        if let Some(obs) = self.uf.predecode_clusters(syndrome) {
            return Ok(obs);
        }

        let num_defects = syndrome.iter().filter(|&&v| v != 0).count();

        // Stage 1: Run truncated BP (reusing pre-allocated buffers).
        // Adaptive iterations: need enough for messages to propagate
        // across the graph, but not so many that BP oscillates.
        let iters = if self.config.bp_iterations > 0 {
            self.config.bp_iterations
        } else {
            // Estimate code distance from detector count.
            // Surface code: num_detectors ≈ d * num_rounds, num_rounds ≈ 2d
            // So num_detectors ≈ 2d^2, d ≈ sqrt(num_det / 2)
            let d_est = ((self
                .matching_graph
                .as_ref()
                .map_or(self.bp_graph.num_checks, |mg| mg.num_detectors)
                as f64)
                / 2.0)
                .sqrt();
            // Need ~d iterations for full propagation.
            // Scale down for few defects.
            let target = d_est.ceil() as usize;
            match num_defects {
                2..=3 => target.min(3), // few defects: local info sufficient
                4..=8 => target.min(8), // moderate: need more propagation
                _ => target.min(12),    // many defects: full propagation, capped
            }
        };

        self.bp_c_to_v.fill(0.0);
        self.bp_v_to_c.fill(0.0);
        let serial = matches!(self.config.bp_schedule, BpSchedule::Serial);
        mini_bp::min_sum_bp_into(
            &self.bp_graph,
            syndrome,
            iters,
            self.config.min_sum_scale,
            serial,
            &mut self.bp_c_to_v,
            &mut self.bp_v_to_c,
            &mut self.bp_posterior,
        );
        let posteriors = &self.bp_posterior;

        // Stage 2: Adjust UF edge weights using BP posteriors.
        //
        // BP posterior is an LLR: positive = likely no error, negative = likely error.
        // UF weight = ln((1-p)/p): positive, lower = more likely error.
        // Both are LLRs with the same sign convention.
        //
        // The posterior directly replaces the prior LLR as a better estimate.
        // We blend to avoid over-reliance on BP when it hasn't converged.
        self.adjusted_weights.copy_from_slice(&self.base_weights);

        let blend = self.config.bp_weight_blend;
        for (m, &posterior) in posteriors.iter().enumerate() {
            if let Some(edge_idx) = self.mechanism_to_edge[m] {
                // Map posterior LLR to positive UF weight.
                // Positive posterior = unlikely error = high weight.
                // Negative posterior = likely error = low weight.
                // Soft mapping: weight = log(1 + exp(posterior)) keeps
                // weights positive and smooth, approaching 0 for very
                // negative posteriors and linear for positive ones.
                let bp_weight = if posterior > 10.0 {
                    posterior // Avoid overflow in exp
                } else if posterior < -10.0 {
                    0.01 // Very likely error
                } else {
                    (1.0 + posterior.exp()).ln()
                };
                let blended = (1.0 - blend) * self.base_weights[edge_idx] + blend * bp_weight;
                // Take the minimum across mechanisms for this edge.
                self.adjusted_weights[edge_idx] = self.adjusted_weights[edge_idx].min(blended);
            }
        }

        // Stage 3: Use UF with BP-adjusted weights.
        let (mask, matched_edges) = self
            .uf
            .decode_with_weights(syndrome, &self.adjusted_weights)?;

        // Stage 4 (optional): Second pass -- use first-pass correction to
        // boost BP priors for matched edges, re-run BP, re-decode.
        // Only do this when the first pass found a substantial correction
        // and BP had enough iterations to produce meaningful posteriors.
        if matched_edges.len() >= 2 && iters >= 4 {
            let boost = 1.5;
            for &edge_idx in &matched_edges {
                if edge_idx < self.adjusted_weights.len() {
                    self.adjusted_weights[edge_idx] =
                        (self.adjusted_weights[edge_idx] - boost).max(0.01);
                }
            }
            let (mask2, _) = self
                .uf
                .decode_with_weights(syndrome, &self.adjusted_weights)?;
            return Ok(mask2);
        }

        Ok(mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_decoder_core::ObservableDecoder;

    const SIMPLE_DEM: &str = "\
error(0.1) D0 D1 L0
error(0.1) D1
";

    #[test]
    fn test_bp_uf_construction() {
        let dec = BpUfDecoder::from_dem(SIMPLE_DEM, BpUfConfig::default());
        assert!(dec.is_ok());
    }

    #[test]
    fn test_bp_uf_no_errors() {
        let mut dec = BpUfDecoder::from_dem(SIMPLE_DEM, BpUfConfig::default()).unwrap();
        let obs = dec.decode_to_observables(&[0, 0]).unwrap();
        assert_eq!(obs, 0);
    }

    #[test]
    fn test_bp_uf_with_errors() {
        let mut dec = BpUfDecoder::from_dem(SIMPLE_DEM, BpUfConfig::default()).unwrap();
        let obs = dec.decode_to_observables(&[1, 1]).unwrap();
        assert_eq!(obs, 1); // D0-D1 edge carries L0
    }

    const D3_DEM: &str =
        include_str!("../../../examples/surface_code_circuits/surface_code_d3_z_stim.dem");

    #[test]
    fn test_bp_uf_real_dem() {
        let mut dec = BpUfDecoder::from_dem(D3_DEM, BpUfConfig::default()).unwrap();
        // No errors
        let obs = dec.decode_to_observables(&[0u8; 24]).unwrap();
        assert_eq!(obs, 0);

        // Random syndromes shouldn't panic
        let mut rng = fastrand::Rng::with_seed(42);
        for _ in 0..100 {
            let syn: Vec<u8> = (0..24).map(|_| u8::from(rng.f64() < 0.05)).collect();
            let _ = dec.decode_to_observables(&syn).unwrap();
        }
    }
}
