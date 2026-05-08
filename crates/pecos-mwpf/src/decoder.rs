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

//! MWPF hypergraph decoder implementation
//!
//! Wraps the Minimum-Weight Parity Factor solver by Yue Wu (Yale).
//! Unlike MWPM decoders, MWPF handles hyperedges natively -- it does not
//! need graphlike decomposition, so it can decode Y errors, depolarizing
//! noise, color codes, and small QLDPC codes with higher accuracy.

use crate::errors::{MwpfError, Result};
use mwpf::mwpf_solver::{
    SolverBPWrapper, SolverBase, SolverSerialJointSingleHair, SolverSerialSingleHair,
    SolverSerialUnionFind, SolverTrait,
};
use mwpf::util::{HyperEdge, SolverInitializer, SyndromePattern};
use pecos_decoder_core::dem::DemCheckMatrix;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Which MWPF solver variant to use.
#[derive(Debug, Clone, Copy, Default)]
pub enum MwpfSolverType {
    /// Union-find only -- fastest, lowest accuracy.
    UnionFind,
    /// Single hair plugin pass -- moderate speed and accuracy.
    SingleHair,
    /// BP preprocessing + `JointSingleHair`. BP guides the solver to
    /// converge faster while maintaining accuracy.
    BpHybrid,
    /// Joint single hair with repeated optimization -- best accuracy, slowest.
    #[default]
    JointSingleHair,
}

/// Configuration for the MWPF decoder.
///
/// MWPF has three knobs at the solver level:
/// - `cluster_node_limit`: controls optimization depth (accuracy vs speed)
/// - `timeout`: wall-clock cap, falls back to union-find on expiry
/// - `only_solve_primal_once`: skip intermediate primal solutions
#[derive(Debug, Clone, Copy)]
pub struct MwpfConfig {
    /// Which solver variant to use. Default: `JointSingleHair` (best accuracy).
    pub solver_type: MwpfSolverType,

    /// Maximum number of nodes per cluster during optimization.
    /// Lower values are faster but less accurate.
    /// Default: 50 (paper's sweet spot for d=7 circuit-level).
    pub cluster_node_limit: usize,

    /// Timeout in seconds for the solver. When exceeded, the solver stops
    /// optimizing and returns the best solution found so far (union-find
    /// baseline). `None` means no timeout.
    ///
    /// Setting this is the main way to tame the p99 latency tail.
    pub timeout: Option<f64>,

    /// If true, solve the primal only once at the end instead of after each
    /// plugin iteration. Can be faster at the cost of missing local optima.
    pub only_solve_primal_once: bool,
}

impl Default for MwpfConfig {
    fn default() -> Self {
        Self {
            solver_type: MwpfSolverType::default(),
            cluster_node_limit: 50,
            timeout: None,
            only_solve_primal_once: false,
        }
    }
}

impl MwpfConfig {
    /// Build the `serde_json` config object for the MWPF solver.
    fn to_solver_config(self) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        map.insert(
            "cluster_node_limit".to_string(),
            serde_json::Number::from(self.cluster_node_limit).into(),
        );
        if let Some(t) = self.timeout
            && let Some(n) = serde_json::Number::from_f64(t)
        {
            map.insert("timeout".to_string(), serde_json::Value::Number(n));
        }
        if self.only_solve_primal_once {
            map.insert("only_solve_primal_once".to_string(), true.into());
        }
        serde_json::Value::Object(map)
    }
}

/// Decoding result from the MWPF decoder.
#[derive(Debug, Clone)]
pub struct MwpfDecodingResult {
    /// Observable prediction as a bitmask (bit i set = observable i flipped).
    pub observable_mask: u64,
    /// Edge indices in the solution subgraph.
    pub subgraph: Vec<usize>,
}

/// Internal solver enum holding any MWPF solver variant.
#[allow(clippy::large_enum_variant)] // Solver structs are owned to avoid extra solver indirection.
enum Solver {
    UnionFind(SolverSerialUnionFind),
    SingleHair(SolverSerialSingleHair),
    JointSingleHair(SolverSerialJointSingleHair),
    BpHybrid(SolverBPWrapper),
}

struct EdgeInfo {
    prob: f64,
    obs_mask: u64,
    best_prob: f64,
}

impl Solver {
    fn solve(&mut self, syndrome: SyndromePattern) {
        match self {
            Self::UnionFind(s) => s.solve(syndrome),
            Self::SingleHair(s) => s.solve(syndrome),
            Self::JointSingleHair(s) => s.solve(syndrome),
            Self::BpHybrid(s) => s.solve(syndrome),
        }
    }

    fn subgraph(&mut self) -> mwpf::util::OutputSubgraph {
        match self {
            Self::UnionFind(s) => s.subgraph(),
            Self::SingleHair(s) => s.subgraph(),
            Self::JointSingleHair(s) => s.subgraph(),
            Self::BpHybrid(s) => s.subgraph(),
        }
    }

    fn clear(&mut self) {
        match self {
            Self::UnionFind(s) => s.clear(),
            Self::SingleHair(s) => s.clear(),
            Self::JointSingleHair(s) => s.clear(),
            Self::BpHybrid(s) => s.clear(),
        }
    }
}

/// MWPF hypergraph decoder.
///
/// Constructed from a full (non-decomposed) DEM string. Each error mechanism
/// in the DEM becomes one hyperedge in the solver, preserving correlations
/// that MWPM decoders must decompose away.
pub struct MwpfDecoder {
    /// The MWPF solver instance.
    solver: Solver,
    /// Per-edge observable bitmask (indexed by deduped edge index).
    edge_obs: Vec<u64>,
    /// Number of detectors.
    num_detectors: usize,
    /// Reusable buffer for defect vertices (avoids per-shot allocation).
    defect_buf: Vec<usize>,
}

impl MwpfDecoder {
    /// Create a decoder from a DEM string and configuration.
    ///
    /// The DEM should be full (non-decomposed) to preserve hyperedges.
    ///
    /// # Errors
    ///
    /// Returns `MwpfError` if the DEM is malformed or the solver cannot be
    /// constructed.
    pub fn from_dem(dem_str: &str, config: MwpfConfig) -> Result<Self> {
        let dem = DemCheckMatrix::from_dem_str(dem_str)
            .map_err(|e| MwpfError::InvalidDem(e.to_string()))?;

        // Build hyperedges from the check matrix. Each mechanism (column) becomes
        // one HyperEdge with all its incident detectors.
        // Merge duplicate vertex sets and build per-edge observable masks.
        // Decomposed DEMs can have multiple mechanisms with the same detector set.
        // Merge by combining probabilities (independent union) and tracking the
        // observable from the highest-probability mechanism (first-observable-wins).
        let mut edge_map: BTreeMap<Vec<usize>, EdgeInfo> = BTreeMap::new();
        for m in 0..dem.num_mechanisms {
            let p = dem.error_priors[m];
            if p <= 0.0 {
                continue;
            }

            let vertices: Vec<usize> = (0..dem.num_detectors)
                .filter(|&d| dem.check_matrix[[d, m]] != 0)
                .collect();

            if vertices.is_empty() {
                continue;
            }

            // Compute this mechanism's observable mask.
            let mut obs: u64 = 0;
            for o in 0..dem.num_observables {
                if dem.observable_matrix[[o, m]] != 0 {
                    obs |= 1 << o;
                }
            }

            let entry = edge_map.entry(vertices).or_insert(EdgeInfo {
                prob: 0.0,
                obs_mask: obs,
                best_prob: p,
            });
            let old_p = entry.prob;
            entry.prob = old_p + p - old_p * p;
            if p > entry.best_prob {
                entry.obs_mask = obs;
                entry.best_prob = p;
            }
        }

        let mut hyperedges = Vec::with_capacity(edge_map.len());
        let mut edge_obs = Vec::with_capacity(edge_map.len());
        for (vertices, info) in edge_map {
            let weight = if info.prob < 1.0 {
                ((1.0 - info.prob) / info.prob).ln()
            } else {
                0.0
            };
            hyperedges.push(HyperEdge::new(vertices, weight.into()));
            edge_obs.push(info.obs_mask);
        }

        let initializer = Arc::new(SolverInitializer::new(dem.num_detectors, hyperedges));
        let solver_config = config.to_solver_config();
        let solver = match config.solver_type {
            MwpfSolverType::UnionFind => {
                Solver::UnionFind(SolverSerialUnionFind::new(&initializer, solver_config))
            }
            MwpfSolverType::SingleHair => {
                Solver::SingleHair(SolverSerialSingleHair::new(&initializer, solver_config))
            }
            MwpfSolverType::JointSingleHair => Solver::JointSingleHair(
                SolverSerialJointSingleHair::new(&initializer, solver_config),
            ),
            MwpfSolverType::BpHybrid => {
                let base = SolverBase {
                    inner: mwpf::mwpf_solver::SolverEnum::SolverSerialJointSingleHair(
                        SolverSerialJointSingleHair::new(&initializer, solver_config),
                    ),
                };
                // BP with 50 iterations and 0.5 application ratio (paper defaults).
                Solver::BpHybrid(SolverBPWrapper::new(base, 50, 0.5))
            }
        };

        Ok(Self {
            solver,
            edge_obs,
            num_detectors: dem.num_detectors,
            defect_buf: Vec::new(),
        })
    }

    /// Decode a syndrome and return the observable mask.
    ///
    /// The syndrome is a byte slice of length `num_detectors`, where
    /// non-zero entries indicate triggered detectors.
    ///
    /// # Errors
    ///
    /// Returns `MwpfError::DecodingFailed` if decoding fails.
    pub fn decode_syndrome(&mut self, syndrome: &[u8]) -> Result<MwpfDecodingResult> {
        // Reuse defect buffer across shots
        self.defect_buf.clear();
        for (i, &v) in syndrome.iter().enumerate() {
            if v != 0 {
                self.defect_buf.push(i);
            }
        }

        if self.defect_buf.is_empty() {
            return Ok(MwpfDecodingResult {
                observable_mask: 0,
                subgraph: Vec::new(),
            });
        }

        self.solver
            .solve(SyndromePattern::new_vertices(self.defect_buf.clone()));
        let output = self.solver.subgraph();
        self.solver.clear();

        // Compute observable mask from the correction subgraph.
        let mut observable_mask = 0u64;
        for &edge_idx in &output.subgraph {
            if edge_idx < self.edge_obs.len() {
                observable_mask ^= self.edge_obs[edge_idx];
            }
        }

        Ok(MwpfDecodingResult {
            observable_mask,
            subgraph: output.subgraph,
        })
    }

    /// Number of detectors in the model.
    #[must_use]
    pub fn num_detectors(&self) -> usize {
        self.num_detectors
    }

    /// Number of edges in the model (after deduplication).
    #[must_use]
    pub fn num_edges(&self) -> usize {
        self.edge_obs.len()
    }

    /// Number of observables in the model.
    #[must_use]
    pub fn num_observables(&self) -> usize {
        let mut max_bit = 0usize;
        for &obs in &self.edge_obs {
            if obs != 0 {
                max_bit = max_bit.max(64 - obs.leading_zeros() as usize);
            }
        }
        max_bit
    }
}
