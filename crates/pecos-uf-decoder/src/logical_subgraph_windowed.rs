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

//! Windowed logical-subgraph decoder with correct sliding-window core-commit.
//!
//! The logical-subgraph decoder partitions a DEM per logical observable and
//! decodes each observable's subgraph independently (the coordinate observing
//! regions of Serra-Peralta et al., arXiv:2505.13599 / the `lomatching`
//! package). For deep circuits an observing region would span the whole circuit,
//! so we additionally window each subgraph in time.
//!
//! **Nesting: subgraph -> window.** Each per-observable subgraph is a clean
//! graphlike matching graph, so we wrap it in an
//! [`OverlappingWindowedDecoder`], which performs proper sliding-window
//! decoding: every window is decoded with a buffer for matching context, but
//! only correction edges whose BOTH endpoints lie in the window core are
//! committed (Tan et al., arXiv:2209.09219). The per-observable committed
//! observable flips are XORed.
//!
//! An earlier implementation windowed the full DEM first and then ran a subgraph
//! decoder per window, combining by a naive full-window observable XOR with no
//! core-commit. That double-counted error chains crossing a window boundary and
//! *anti-suppressed* (LER grew with code distance). The correct nesting here
//! reuses the tested core-commit machinery instead.

use pecos_decoder_core::ObservableDecoder;
use pecos_decoder_core::dem::DemMatchingGraph;
use pecos_decoder_core::errors::DecoderError;
use pecos_decoder_core::logical_subgraph::{
    LogicalSubgraph, MaxTimeRadius, StabCoords, partition_dem_by_logical_windowed,
};
use std::fmt::Write as _;

use crate::decoder::{UfDecoder, UfDecoderConfig};
use crate::windowed::{OverlappingWindowedDecoder, WindowedConfig};

/// One per-observable subgraph, windowed with sliding-window core-commit.
struct SubgraphWindowed {
    /// Which full-DEM observable this subgraph decodes (the global bit index).
    observable_idx: usize,
    /// Subgraph-local detector index -> full-DEM detector index.
    detector_map: Vec<usize>,
    /// Number of subgraph-local detectors.
    num_local: usize,
    /// The time-windowed decoder over this subgraph (returns local bit 0).
    decoder: OverlappingWindowedDecoder<UfDecoder>,
}

/// Windowed logical-subgraph decoder.
///
/// Partitions the DEM per observable, then windows each subgraph with an
/// [`OverlappingWindowedDecoder`] (sliding-window core-commit). Per-observable
/// committed observable flips are XORed into the final mask.
pub struct WindowedLogicalSubgraphDecoder {
    subgraphs: Vec<SubgraphWindowed>,
    /// Reusable subgraph-local syndrome buffer (sized to the largest subgraph).
    local_syn: Vec<u8>,
}

impl WindowedLogicalSubgraphDecoder {
    /// Build from a full DEM string and stabilizer coordinates.
    ///
    /// `max_time_radius` controls the per-observable observing region (see
    /// [`partition_dem_by_logical_windowed`]); pass `None` for the full region
    /// (the windowing then bounds the time extent instead).
    ///
    /// # Errors
    ///
    /// Returns `DecoderError` if the DEM is malformed or a subgraph decoder
    /// fails to build.
    pub fn from_dem(
        dem: &str,
        stab_coords: &StabCoords,
        max_time_radius: MaxTimeRadius,
        window_config: WindowedConfig,
    ) -> Result<Self, DecoderError> {
        let parts = partition_dem_by_logical_windowed(dem, stab_coords, max_time_radius)?;

        // Subgraph graphs do not carry detector coordinates, but the time-based
        // windowing needs them. Pull them from the full DEM and inject (mapped
        // to subgraph-local indices) when serializing each sub-DEM.
        let full_coords = DemMatchingGraph::from_dem_str(dem)?.detector_coords;

        let mut subgraphs = Vec::with_capacity(parts.len());
        let mut max_local = 0usize;
        for part in parts {
            if part.detector_map.is_empty() {
                // Observable with an empty region never flips: contributes 0.
                continue;
            }
            let sub_dem = subgraph_to_dem_string(&part, &full_coords);
            let decoder = OverlappingWindowedDecoder::from_dem(&sub_dem, window_config, |wdem| {
                UfDecoder::from_dem(wdem, UfDecoderConfig::windowed())
            })?;
            max_local = max_local.max(part.graph.num_detectors);
            subgraphs.push(SubgraphWindowed {
                observable_idx: part.observable_idx,
                detector_map: part.detector_map,
                num_local: part.graph.num_detectors,
                decoder,
            });
        }

        Ok(Self {
            subgraphs,
            local_syn: vec![0u8; max_local],
        })
    }

    /// Number of per-observable subgraphs that actually decode (non-empty).
    #[must_use]
    pub fn num_subgraphs(&self) -> usize {
        self.subgraphs.len()
    }

    /// Total number of windows across all subgraphs.
    #[must_use]
    pub fn num_windows(&self) -> usize {
        self.subgraphs.iter().map(|s| s.decoder.num_windows()).sum()
    }
}

impl ObservableDecoder for WindowedLogicalSubgraphDecoder {
    fn decode_to_observables(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        let mut obs_mask = 0u64;
        for sg in &mut self.subgraphs {
            let n = sg.num_local;
            for (local, &global) in sg.detector_map.iter().enumerate() {
                self.local_syn[local] = if global < syndrome.len() {
                    syndrome[global]
                } else {
                    0
                };
            }
            // The subgraph decodes a single observable as its local bit 0; map
            // that back to this observable's global bit.
            let sub_obs = sg.decoder.decode_to_observables(&self.local_syn[..n])?;
            if sub_obs & 1 != 0 {
                obs_mask |= 1u64 << sg.observable_idx;
            }
        }
        Ok(obs_mask)
    }
}

/// Serialize a subgraph back to a DEM string: detector coordinate lines (mapped
/// from the full DEM via `detector_map`) plus error lines. The coordinate lines
/// let [`OverlappingWindowedDecoder::from_dem`] derive per-detector times for
/// the windowing.
fn subgraph_to_dem_string(part: &LogicalSubgraph, full_coords: &[Option<Vec<f64>>]) -> String {
    let mut s = String::new();
    for (local, &global) in part.detector_map.iter().enumerate() {
        if let Some(Some(coords)) = full_coords.get(global) {
            let coord_str: Vec<String> = coords.iter().map(|c| format!("{c}")).collect();
            let _ = writeln!(s, "detector({}) D{local}", coord_str.join(", "));
        }
    }
    for edge in &part.graph.edges {
        let _ = write!(s, "error({})", edge.probability);
        let _ = write!(s, " D{}", edge.node1);
        if let Some(n2) = edge.node2 {
            let _ = write!(s, " D{n2}");
        }
        for &obs in &edge.observables {
            let _ = write!(s, " L{obs}");
        }
        let _ = writeln!(s);
    }
    s
}
