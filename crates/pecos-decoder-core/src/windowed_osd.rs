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

//! Windowed observable subgraph decoder.
//!
//! Splits a DEM into time windows, runs per-observable subgraph decoding
//! within each window. This prevents the observing region from spanning
//! the full circuit at deep depths, maintaining decoding accuracy.
//!
//! Window types:
//! - **Non-overlapping**: each detector belongs to exactly one window
//! - **Overlapping**: buffer zones extend beyond the core for matching context
//!
//! The observable correction from each window is XOR'd together.

use std::collections::BTreeMap;

use crate::ObservableDecoder;
use crate::dem::{DemCheckMatrix, DemMatchingGraph, MatchingEdge, parse_detector_coords};
use crate::errors::DecoderError;
use crate::observable_subgraph::{ObservableSubgraphDecoder, StabCoords};

/// Configuration for windowed OSD.
#[derive(Debug, Clone)]
pub struct WindowedOsdConfig {
    /// Core window size in time steps.
    pub step: usize,
    /// Buffer size on each side (0 = non-overlapping).
    pub buffer: usize,
}

impl Default for WindowedOsdConfig {
    fn default() -> Self {
        Self { step: 8, buffer: 4 }
    }
}

/// A single time window with its own OSD.
pub struct OsdWindow {
    decoder: ObservableSubgraphDecoder,
    /// Maps local detector index → global detector index.
    local_to_global: Vec<usize>,
    num_local: usize,
    /// Which local detectors are in the core (vs buffer).
    _is_core: Vec<bool>,
}

/// Windowed observable subgraph decoder.
///
/// Splits the DEM into time windows, each decoded with its own OSD.
/// The observing region within each window is naturally bounded,
/// preventing the scaling degradation seen at deep circuits.
pub struct WindowedOsdDecoder {
    pub windows: Vec<OsdWindow>,
    _num_detectors: usize,
    /// Reusable window syndrome buffer
    window_syn: Vec<u8>,
}

impl WindowedOsdDecoder {
    /// Build from a DEM string with time-based windowing.
    ///
    /// # Errors
    ///
    /// Returns error if the DEM is malformed.
    pub fn from_dem<F>(
        dem: &str,
        stab_coords: &StabCoords,
        config: &WindowedOsdConfig,
        mut inner_factory: F,
    ) -> Result<Self, DecoderError>
    where
        F: FnMut(
            &DemMatchingGraph,
        ) -> Result<Box<dyn ObservableDecoder + Send + Sync>, DecoderError>,
    {
        // Parse detector coordinates to get time values
        let coords = parse_detector_coords(dem);
        let mut det_time: BTreeMap<usize, f64> = BTreeMap::new();
        for dc in &coords {
            if let Some(t) = dc.coords.last() {
                det_time.insert(dc.id as usize, *t);
            }
        }

        let dcm = DemCheckMatrix::from_dem_str(dem)
            .map_err(|e| DecoderError::InvalidGraph(e.to_string()))?;
        let num_detectors = dcm.num_detectors;

        // Find time range
        let min_t = det_time.values().copied().fold(f64::INFINITY, f64::min);
        let max_t = det_time.values().copied().fold(f64::NEG_INFINITY, f64::max);

        if max_t <= min_t {
            // Single time step or empty — just use full OSD
            let full_osd =
                ObservableSubgraphDecoder::from_dem(dem, stab_coords, &mut inner_factory)?;
            return Ok(Self {
                windows: vec![OsdWindow {
                    decoder: full_osd,
                    local_to_global: (0..num_detectors).collect(),
                    num_local: num_detectors,
                    _is_core: vec![true; num_detectors],
                }],
                _num_detectors: num_detectors,
                window_syn: vec![0u8; num_detectors],
            });
        }

        let step = config.step as f64;
        let buffer = config.buffer as f64;
        let mut windows = Vec::new();
        let mut t_start = min_t;
        let mut max_local = 0;

        while t_start <= max_t {
            let core_end = (t_start + step).min(max_t + 1.0);
            let win_start = (t_start - buffer).max(min_t);
            let win_end = (core_end + buffer).min(max_t + 1.0);

            // Detectors in this window
            let mut local_to_global = Vec::new();
            let mut is_core = Vec::new();

            for d in 0..num_detectors {
                if let Some(&t) = det_time.get(&d)
                    && t >= win_start
                    && t < win_end
                {
                    local_to_global.push(d);
                    is_core.push(t >= t_start && t < core_end);
                }
            }

            if local_to_global.is_empty() {
                t_start += step;
                continue;
            }

            let num_local = local_to_global.len();
            if num_local > max_local {
                max_local = num_local;
            }

            // Build sub-DEM for this window
            let mut inverse = vec![None; num_detectors];
            for (local, &global) in local_to_global.iter().enumerate() {
                inverse[global] = Some(local);
            }

            let mut edges = Vec::new();
            let mut skipped = 0;

            for m in 0..dcm.num_mechanisms {
                let p = dcm.error_priors[m];
                if p <= 0.0 {
                    continue;
                }

                let sub_dets: Vec<u32> = (0..dcm.num_detectors)
                    .filter(|&d| dcm.check_matrix[[d, m]] != 0)
                    .filter_map(|d| inverse[d].map(|s| s as u32))
                    .collect();

                if sub_dets.is_empty() {
                    continue;
                }

                let weight = if p < 1.0 { ((1.0 - p) / p).ln() } else { 0.0 };

                // Observable: include if ANY observable is flipped
                let mut observables = Vec::new();
                for o in 0..dcm.num_observables {
                    if dcm.observable_matrix[[o, m]] != 0 {
                        observables.push(o as u32);
                    }
                }

                match sub_dets.len() {
                    1 => edges.push(MatchingEdge {
                        node1: sub_dets[0],
                        node2: None,
                        weight,
                        observables,
                        probability: p,
                        fault_id: m,
                    }),
                    2 => edges.push(MatchingEdge {
                        node1: sub_dets[0],
                        node2: Some(sub_dets[1]),
                        weight,
                        observables,
                        probability: p,
                        fault_id: m,
                    }),
                    _ => skipped += 1,
                }
            }

            let edges = DemMatchingGraph::merge_parallel_edges(edges);
            let sub_graph = DemMatchingGraph {
                edges,
                num_detectors: num_local,
                num_observables: dcm.num_observables,
                skipped_hyperedges: skipped,
                detector_coords: Vec::new(),
            };

            // Build sub-DEM string with detector coordinate declarations.
            // The OSD needs these to classify detectors by (qubit, stab_type).
            let mut sub_dem_lines = Vec::new();
            for (local_id, &global_id) in local_to_global.iter().enumerate() {
                // Find this detector's coordinates from the parsed coords
                if let Some(dc) = coords.iter().find(|dc| dc.id as usize == global_id) {
                    let coord_str: Vec<String> = dc.coords.iter().map(|c| format!("{c}")).collect();
                    sub_dem_lines.push(format!("detector({}) D{local_id}", coord_str.join(", ")));
                }
            }
            sub_dem_lines.push(graph_to_dem_string(&sub_graph));
            let sub_dem = sub_dem_lines.join("\n");

            // Build OSD for this window using the sub-DEM
            let window_osd =
                ObservableSubgraphDecoder::from_dem(&sub_dem, stab_coords, &mut inner_factory)?;

            windows.push(OsdWindow {
                decoder: window_osd,
                local_to_global,
                num_local,
                _is_core: is_core,
            });

            t_start += step;
        }

        Ok(Self {
            windows,
            _num_detectors: num_detectors,
            window_syn: vec![0u8; max_local],
        })
    }
}

impl ObservableDecoder for WindowedOsdDecoder {
    fn decode_to_observables(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        let mut obs_mask = 0u64;

        for window in &mut self.windows {
            // Extract window syndrome
            let n = window.num_local;
            for (local, &global) in window.local_to_global.iter().enumerate() {
                self.window_syn[local] = if global < syndrome.len() {
                    syndrome[global]
                } else {
                    0
                };
            }

            // Decode this window
            let window_obs = window
                .decoder
                .decode_to_observables(&self.window_syn[..n])?;
            obs_mask ^= window_obs;
        }

        Ok(obs_mask)
    }
}

fn graph_to_dem_string(graph: &DemMatchingGraph) -> String {
    let mut lines = Vec::new();
    for edge in &graph.edges {
        let p = edge.probability;
        let mut targets = Vec::new();
        targets.push(format!("D{}", edge.node1));
        if let Some(n2) = edge.node2 {
            targets.push(format!("D{n2}"));
        }
        for &obs in &edge.observables {
            targets.push(format!("L{obs}"));
        }
        lines.push(format!("error({p}) {}", targets.join(" ")));
    }
    lines.join("\n")
}
