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

//! Logical-operator subgraph decoder for transversal gates.
//!
//! This decodes a *logical operator's subgraph of the DEM* — not the full DEM.
//! In the Heisenberg picture a detector error model records how errors meet
//! evolving operators: stabilizers evolve into **detectors**, logical operators
//! evolve into **observables**. This decoder groups detectors by their
//! stabilizer (X/Z, per qubit) and restricts the DEM to one logical operator at
//! a time — the errors that can flip it. Each logical operator is effectively
//! its own channel through the circuit (e.g. a transversal CX propagates
//! `XI → XX`, `IZ → ZZ`), so the per-operator problems decouple.
//!
//! The payoff (proved independently by Serra-Peralta et al. arXiv:2505.13599
//! and Cain et al. arXiv:2505.13587): restricted to one logical operator, the
//! subgraph is always graphlike — only 1-2 detector edges, matchable — even
//! when the full DEM contains weight-3+ hyperedges. So any MWPM-style decoder
//! handles each piece, and the results combine.
//!
//! Naming: abbreviate this "LS decoder" if needed — never "LSD", which collides
//! with BP-LSD (Localised Statistics Decoding) elsewhere in the workspace.
//!
//! # Algorithm
//!
//! 1. Classify each detector by (`logical_qubit`, `stabilizer_type`) using
//!    spatial coordinates
//! 2. For each observable, find its boundary edges (1-detector mechanisms)
//!    to identify which (qubit, `stab_type`) groups form its observing region
//! 3. Extract a sub-DEM restricted to those detectors
//! 4. Run any MWPM-compatible decoder on each subgraph independently
//! 5. Combine per-logical-operator corrections
//!
//! # Observing Region
//!
//! The observing region for observable k is NOT a transitive closure over
//! shared detectors. It is determined by the *physical structure*:
//! - Find boundary edges (1-detector + observable) for observable k
//! - Each boundary edge's detector belongs to a (qubit, `stab_type`) group
//! - ALL detectors in those groups form the observing region
//! - This preserves the graphlike property of each subgraph

pub mod committed;
pub mod window_plan;

use std::collections::{BTreeMap, BTreeSet};

use crate::ObservableDecoder;
use crate::dem::{DemMatchingGraph, MatchingEdge, SparseDem};
use crate::errors::DecoderError;
use crate::obs_mask::ObsMask;

// ============================================================================
// Stabilizer coordinate mapping
// ============================================================================

/// Identifies a group of detectors by logical qubit and stabilizer type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DetectorGroup {
    pub qubit_idx: usize,
    pub stab_type: StabType,
}

/// Stabilizer type (X or Z).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StabType {
    X,
    Z,
}

/// Stabilizer coordinate map for one logical qubit.
///
/// Maps stabilizer spatial positions to their type (X or Z).
/// Used to classify detectors by their coordinates.
#[derive(Debug, Clone)]
pub struct QubitStabCoords {
    /// X-stabilizer ancilla positions.
    pub x_positions: Vec<(f64, f64)>,
    /// Z-stabilizer ancilla positions.
    pub z_positions: Vec<(f64, f64)>,
}

/// Stabilizer coordinates for all logical qubits.
///
/// Entry `i` describes the stabilizers of logical qubit `i`.
pub type StabCoords = Vec<QubitStabCoords>;

/// Classify a detector's spatial coordinates into a `DetectorGroup`.
///
/// Finds the nearest stabilizer position across all qubits and returns
/// the matching (`qubit_idx`, `stab_type`). Uses exact floating-point
/// comparison with a small tolerance for rounding.
#[must_use]
pub fn classify_detector(x: f64, y: f64, stab_coords: &StabCoords) -> Option<DetectorGroup> {
    let eps = 0.01;
    for (qubit_idx, qsc) in stab_coords.iter().enumerate() {
        for &(sx, sy) in &qsc.x_positions {
            if (x - sx).abs() < eps && (y - sy).abs() < eps {
                return Some(DetectorGroup {
                    qubit_idx,
                    stab_type: StabType::X,
                });
            }
        }
        for &(sx, sy) in &qsc.z_positions {
            if (x - sx).abs() < eps && (y - sy).abs() < eps {
                return Some(DetectorGroup {
                    qubit_idx,
                    stab_type: StabType::Z,
                });
            }
        }
    }
    None
}

// ============================================================================
// Subgraph partitioning
// ============================================================================

/// A sub-DEM for one observable's observing region.
#[derive(Debug, Clone)]
pub struct LogicalSubgraph {
    /// Which observable this subgraph decodes.
    pub observable_idx: usize,
    /// Maps subgraph detector index → full DEM detector index.
    pub detector_map: Vec<usize>,
    /// Maps full DEM detector index → subgraph detector index (None if outside).
    pub inverse_map: Vec<Option<usize>>,
    /// The matching graph for this subgraph.
    pub graph: DemMatchingGraph,
}

/// Partition a DEM into per-logical-operator subgraphs using stabilizer coordinates.
///
/// This is the correct algorithm: uses the physical structure (which detectors
/// belong to which stabilizer type on which qubit) to determine observing
/// regions, rather than a topological transitive closure.
///
/// # Arguments
///
/// * `dem_str` — DEM string in Stim format. Must include `detector(...) D_i`
///   declarations with spatial coordinates.
/// * `stab_coords` — Per-qubit stabilizer coordinate map. Entry `i` gives
///   the X and Z ancilla positions for logical qubit `i`.
///
/// # Errors
///
/// Returns an error if the DEM is malformed or if a detector used in an error
/// mechanism has coordinates that match no stabilizer position in
/// `stab_coords`.
///
/// Extra time padding around each boundary edge.
/// `None` = exact boundary edge times only (default, matches lomatching).
/// `Some(r)` = include detectors at times `t ± r` around each boundary
/// edge time `t`, for additional matching context.
pub type MaxTimeRadius = Option<i64>;

pub fn partition_dem_by_logical(
    dem_str: &str,
    stab_coords: &StabCoords,
) -> Result<Vec<LogicalSubgraph>, DecoderError> {
    partition_dem_by_logical_windowed(dem_str, stab_coords, None)
}

pub fn partition_dem_by_logical_windowed(
    dem_str: &str,
    stab_coords: &StabCoords,
    max_time_radius: MaxTimeRadius,
) -> Result<Vec<LogicalSubgraph>, DecoderError> {
    // Region source (coordinate classification) -> shared subgraph extractor.
    let sdem = SparseDem::from_dem_str(dem_str)?;
    let membership = coordinate_membership_from_dem(&sdem, stab_coords, max_time_radius)?;
    subgraphs_from_membership(&sdem, &membership)
}

/// Per-observable detector membership: entry `k` is the sorted full-DEM detector
/// ids in logical observable `k`'s observing region.
///
/// This is the seam between a *region source* and the *subgraph extractor*
/// ([`subgraphs_from_membership`]). The coordinate region source lives here
/// ([`coordinate_membership_from_dem`]); a back-propagation source can live in a
/// higher crate (e.g. `pecos-qec`) — it only needs to produce this same
/// membership and hand it down, which preserves the `pecos-qec ->
/// pecos-decoder-core` dependency direction.
pub type ObservingRegions = Vec<Vec<usize>>;

/// Region source: derive each observable's detector membership from the DEM's
/// spatial stabilizer coordinates and boundary edges.
///
/// This is the algorithm `lomatching` ships for vertex selection
/// (`get_detector_indices_for_subgraphs`): find the 1-detector mechanisms that
/// flip each observable, bucket their detectors into `(qubit, stab_type)` groups
/// by coordinate, and include every detector of those groups at the boundary
/// times. `max_time_radius` widens the per-time inclusion (default `None` =
/// exact boundary times, matching lomatching).
///
/// # Errors
///
/// Returns an error if a detector used in an error mechanism has coordinates
/// that match no stabilizer position in `stab_coords`.
pub fn coordinate_membership_from_dem(
    sdem: &SparseDem,
    stab_coords: &StabCoords,
    max_time_radius: MaxTimeRadius,
) -> Result<ObservingRegions, DecoderError> {
    let coord_map = &sdem.detector_coords;

    // Detectors that appear in an error mechanism are the only ones that affect
    // decoding. Any such detector that fails to classify would be silently
    // dropped from every observing region and corrupt the decode, so treat that
    // as a configuration error rather than a silent fallback. Detectors that are
    // declared but never used (numbering gaps, unused ancillas) are ignored.
    let mut used = vec![false; sdem.num_detectors];
    for (_, dets, _) in &sdem.mechanisms {
        for &d in dets {
            if (d as usize) < sdem.num_detectors {
                used[d as usize] = true;
            }
        }
    }

    // Classify each detector into a (qubit, stab_type) group.
    let mut det_group: Vec<Option<DetectorGroup>> = vec![None; sdem.num_detectors];
    let mut group_detectors: BTreeMap<DetectorGroup, BTreeSet<usize>> = BTreeMap::new();
    let mut unclassified: Vec<usize> = Vec::new();

    for (d, group_slot) in det_group.iter_mut().enumerate().take(sdem.num_detectors) {
        let group = match coord_map.get(&d) {
            Some(coords) if coords.len() >= 2 => {
                classify_detector(coords[0], coords[1], stab_coords)
            }
            _ => None,
        };
        match group {
            Some(g) => {
                *group_slot = Some(g);
                group_detectors.entry(g).or_default().insert(d);
            }
            None if used[d] => unclassified.push(d),
            None => {}
        }
    }

    if !unclassified.is_empty() {
        let shown: Vec<String> = unclassified
            .iter()
            .take(5)
            .map(|&d| match coord_map.get(&d) {
                Some(c) => format!("D{d} at {c:?}"),
                None => format!("D{d} (no coordinates)"),
            })
            .collect();
        return Err(DecoderError::InvalidConfiguration(format!(
            "{} detector(s) used in error mechanisms could not be classified against \
             stab_coords (missing coordinates, or a coordinate scale / completeness \
             mismatch). Examples: {}",
            unclassified.len(),
            shown.join(", "),
        )));
    }

    // For each observable, collect its observing region.
    let mut membership: ObservingRegions = Vec::with_capacity(sdem.num_observables);

    for obs_idx in 0..sdem.num_observables {
        // Step 1: Find boundary edges — 1-detector mechanisms that flip this
        // observable. Collect (group, time) from each boundary detector.
        let mut group_times: BTreeMap<DetectorGroup, BTreeSet<i64>> = BTreeMap::new();

        for (_, dets, obs) in &sdem.mechanisms {
            if !obs.contains(&(obs_idx as u32)) {
                continue;
            }
            if dets.len() == 1 {
                let d = dets[0] as usize;
                if let Some(group) = det_group[d] {
                    let time = coord_map
                        .get(&d)
                        .and_then(|c| c.last().copied())
                        .map_or(0, |t| t as i64);
                    group_times.entry(group).or_default().insert(time);
                }
            }
        }

        // Step 2: For each (group, time) boundary edge, include ALL detectors of
        // that group at that time. This matches lomatching's per-time-step
        // approach: detectors are included only at times where boundary edges
        // exist, not across the full time range. With max_time_radius, extend
        // each boundary time by ±radius.
        let mut region_detectors = BTreeSet::new();
        for (group, times) in &group_times {
            if let Some(dets) = group_detectors.get(group) {
                for &d in dets {
                    let det_time = coord_map
                        .get(&d)
                        .and_then(|c| c.last().copied())
                        .map_or(0, |t| t as i64);
                    let in_region = if let Some(radius) = max_time_radius {
                        times.iter().any(|&t| (det_time - t).abs() <= radius)
                    } else {
                        times.contains(&det_time)
                    };
                    if in_region {
                        region_detectors.insert(d);
                    }
                }
            }
        }

        membership.push(region_detectors.into_iter().collect());
    }

    Ok(membership)
}

/// Subgraph extractor: turn per-observable detector membership into decodable
/// graphlike subgraphs. The membership may come from any region source — the
/// coordinate one ([`coordinate_membership_from_dem`]) or a back-propagation
/// builder in a higher crate.
///
/// # Errors
///
/// Supports arbitrarily many observables: flips are packed into a wide
/// [`ObsMask`] by [`ObservableDecoder::decode_obs`]. (The `u64`
/// [`ObservableDecoder::decode_to_observables`] convenience errors above 64; the
/// windowed paths still cap at 64 — see their constructors.)
pub fn subgraphs_from_membership(
    sdem: &SparseDem,
    membership: &[Vec<usize>],
) -> Result<Vec<LogicalSubgraph>, DecoderError> {
    let mut subgraphs = Vec::with_capacity(membership.len());

    for (obs_idx, detectors) in membership.iter().enumerate() {
        if detectors.is_empty() {
            subgraphs.push(LogicalSubgraph {
                observable_idx: obs_idx,
                detector_map: Vec::new(),
                inverse_map: vec![None; sdem.num_detectors],
                graph: DemMatchingGraph {
                    edges: Vec::new(),
                    num_detectors: 0,
                    num_observables: 1,
                    skipped_hyperedges: 0,
                    detector_coords: Vec::new(),
                },
            });
            continue;
        }

        // Build detector mapping (full DEM id -> subgraph-local index).
        // Validate caller-provided ids first: an out-of-range id would index
        // past `inverse_map` and panic, so return a decoder error instead.
        let detector_map: Vec<usize> = detectors.clone();
        if let Some(&bad) = detector_map.iter().find(|&&d| d >= sdem.num_detectors) {
            return Err(DecoderError::InvalidConfiguration(format!(
                "subgraphs_from_membership: observable {obs_idx} membership references \
                 detector {bad}, but the DEM has only {} detectors (D0..D{})",
                sdem.num_detectors,
                sdem.num_detectors.saturating_sub(1),
            )));
        }
        let mut inverse_map = vec![None; sdem.num_detectors];
        for (sub_idx, &full_idx) in detector_map.iter().enumerate() {
            inverse_map[full_idx] = Some(sub_idx);
        }

        // Extract edges for this subgraph by projecting mechanisms onto the
        // membership detectors.
        let mut edges = Vec::new();
        let mut skipped = 0;

        for (m, (p, dets, obs)) in sdem.mechanisms.iter().enumerate() {
            if *p <= 0.0 {
                continue;
            }

            let sub_dets: Vec<u32> = dets
                .iter()
                .filter_map(|&d| inverse_map[d as usize].map(|s| s as u32))
                .collect();

            if sub_dets.is_empty() {
                continue;
            }

            let weight = if *p < 1.0 { ((1.0 - p) / p).ln() } else { 0.0 };
            let flips_obs = obs.contains(&(obs_idx as u32));
            let observables = if flips_obs { vec![0u32] } else { vec![] };

            match sub_dets.len() {
                1 => edges.push(MatchingEdge {
                    node1: sub_dets[0],
                    node2: None,
                    weight,
                    observables,
                    probability: *p,
                    fault_id: m,
                }),
                2 => edges.push(MatchingEdge {
                    node1: sub_dets[0],
                    node2: Some(sub_dets[1]),
                    weight,
                    observables,
                    probability: *p,
                    fault_id: m,
                }),
                _ => skipped += 1,
            }
        }

        let num_sub = detector_map.len();
        let edges = DemMatchingGraph::merge_parallel_edges(edges);

        subgraphs.push(LogicalSubgraph {
            observable_idx: obs_idx,
            detector_map,
            inverse_map,
            graph: DemMatchingGraph {
                edges,
                num_detectors: num_sub,
                num_observables: 1,
                skipped_hyperedges: skipped,
                detector_coords: Vec::new(),
            },
        });
    }

    Ok(subgraphs)
}

// ============================================================================
// Decoder
// ============================================================================

/// Per-logical-operator subgraph decoder.
///
/// Wraps a factory function that creates per-subgraph inner decoders.
/// Any `ObservableDecoder` works as the inner decoder (UF, Fusion Blossom,
/// perturbed ensemble, etc.).
pub struct LogicalSubgraphDecoder {
    subgraphs: Vec<LogicalSubgraph>,
    decoders: Vec<Box<dyn ObservableDecoder + Send + Sync>>,
    num_observables: usize,
    sub_syndromes: Vec<Vec<u8>>,
}

impl LogicalSubgraphDecoder {
    /// Build from a DEM string, stabilizer coordinates, and inner decoder factory.
    ///
    /// # Errors
    ///
    /// Returns error if the DEM is malformed or the factory fails.
    pub fn from_dem<F>(
        dem: &str,
        stab_coords: &StabCoords,
        factory: F,
    ) -> Result<Self, DecoderError>
    where
        F: FnMut(
            &DemMatchingGraph,
        ) -> Result<Box<dyn ObservableDecoder + Send + Sync>, DecoderError>,
    {
        Self::from_dem_windowed(dem, stab_coords, None, factory)
    }

    pub fn from_dem_windowed<F>(
        dem: &str,
        stab_coords: &StabCoords,
        max_time_radius: MaxTimeRadius,
        mut factory: F,
    ) -> Result<Self, DecoderError>
    where
        F: FnMut(
            &DemMatchingGraph,
        ) -> Result<Box<dyn ObservableDecoder + Send + Sync>, DecoderError>,
    {
        let subgraphs = partition_dem_by_logical_windowed(dem, stab_coords, max_time_radius)?;
        let num_observables = subgraphs.len();

        let mut decoders = Vec::with_capacity(subgraphs.len());
        let mut sub_syndromes = Vec::with_capacity(subgraphs.len());
        for sg in &subgraphs {
            decoders.push(factory(&sg.graph)?);
            sub_syndromes.push(vec![0u8; sg.detector_map.len()]);
        }

        Ok(Self {
            subgraphs,
            decoders,
            num_observables,
            sub_syndromes,
        })
    }

    /// Build from a precomputed per-observable detector membership instead of
    /// from `stab_coords`.
    ///
    /// The membership may come from ANY region source — the coordinate path
    /// ([`coordinate_membership_from_dem`]) or a back-propagation / detecting-
    /// region source. This is the entry point for comparing alternative
    /// observing-region constructions (e.g. the paper's back-propagation region
    /// vs the coordinate group-fill) on the same DEM and decoders.
    ///
    /// # Errors
    ///
    /// Returns an error if the DEM is malformed or the factory fails.
    pub fn from_membership<F>(
        dem: &str,
        membership: &[Vec<usize>],
        mut factory: F,
    ) -> Result<Self, DecoderError>
    where
        F: FnMut(
            &DemMatchingGraph,
        ) -> Result<Box<dyn ObservableDecoder + Send + Sync>, DecoderError>,
    {
        let sdem = SparseDem::from_dem_str(dem)?;
        let subgraphs = subgraphs_from_membership(&sdem, membership)?;
        let num_observables = subgraphs.len();

        let mut decoders = Vec::with_capacity(subgraphs.len());
        let mut sub_syndromes = Vec::with_capacity(subgraphs.len());
        for sg in &subgraphs {
            decoders.push(factory(&sg.graph)?);
            sub_syndromes.push(vec![0u8; sg.detector_map.len()]);
        }

        Ok(Self {
            subgraphs,
            decoders,
            num_observables,
            sub_syndromes,
        })
    }

    /// Number of observables.
    #[must_use]
    pub fn num_observables(&self) -> usize {
        self.num_observables
    }

    /// Access a subgraph.
    #[must_use]
    pub fn subgraph(&self, obs_idx: usize) -> Option<&LogicalSubgraph> {
        self.subgraphs.get(obs_idx)
    }

    /// Build a coord-preserving per-observable window plan from these subgraphs
    /// and the full-DEM detector coordinates (indexed by global detector id).
    ///
    /// Subgraph matching graphs drop detector coordinates, so the windowed
    /// decoders need the full-DEM coords re-injected to time-window correctly.
    /// The plan also reports whether real windowing would happen or it
    /// degenerates to a single-window full decode (see
    /// [`window_plan::LogicalSubgraphWindowPlan`]).
    #[must_use]
    pub fn window_plan(
        &self,
        full_coords: &[Option<Vec<f64>>],
    ) -> window_plan::LogicalSubgraphWindowPlan {
        window_plan::LogicalSubgraphWindowPlan::new(&self.subgraphs, full_coords)
    }

    /// Per-observable observing regions: entry `k` is the sorted full-DEM
    /// detector ids in observable `k`'s subgraph. This is the membership the
    /// region source produced — exposed for differential testing against
    /// reference implementations (e.g. lomatching).
    #[must_use]
    pub fn observing_regions(&self) -> ObservingRegions {
        self.subgraphs
            .iter()
            .map(|sg| sg.detector_map.clone())
            .collect()
    }

    /// Batch decode multiple syndromes, returning error count.
    ///
    /// For each subgraph, extracts all sub-syndromes into a flat buffer
    /// and calls `decode_batch_to_observables` once — avoiding per-shot
    /// reset overhead in decoders like `PyMatching`.
    pub fn decode_count_batched(
        &mut self,
        syndromes: &[Vec<u8>],
        expected_masks: &[ObsMask],
    ) -> Result<usize, DecoderError> {
        let num_shots = syndromes.len();
        if num_shots == 0 {
            return Ok(0);
        }

        // Per-shot observable predictions, accumulated across subgraphs. Wide
        // (`ObsMask`) so >64 observables are not truncated.
        let mut shot_obs: Vec<ObsMask> = vec![ObsMask::new(); num_shots];

        for (sg, dec) in self.subgraphs.iter().zip(self.decoders.iter_mut()) {
            let n = sg.detector_map.len();
            if n == 0 {
                continue;
            }

            // Build flat sub-syndrome buffer: num_shots × n bytes.
            let mut flat = vec![0u8; num_shots * n];
            for (shot_idx, syn) in syndromes.iter().enumerate() {
                let row = &mut flat[shot_idx * n..(shot_idx + 1) * n];
                for (sub_idx, &full_idx) in sg.detector_map.iter().enumerate() {
                    row[sub_idx] = if full_idx < syn.len() {
                        syn[full_idx]
                    } else {
                        0
                    };
                }
            }

            // Batch decode this subgraph.
            let sub_masks = dec.decode_batch_to_observables(&flat, num_shots, n)?;

            for (shot_idx, sub_obs) in sub_masks.iter().enumerate() {
                if sub_obs.get(0) {
                    shot_obs[shot_idx].set(sg.observable_idx);
                }
            }
        }

        // Count errors.
        let errors = shot_obs
            .iter()
            .zip(expected_masks.iter())
            .filter(|(predicted, expected)| predicted != expected)
            .count();

        Ok(errors)
    }
}

impl ObservableDecoder for LogicalSubgraphDecoder {
    /// Narrowing wrapper over [`Self::decode_obs`]. Errors (rather than
    /// truncating) if the decoder has more than 64 observables; callers that may
    /// exceed 64 should use [`ObservableDecoder::decode_obs`].
    fn decode_to_observables(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        self.decode_obs(syndrome)?.to_u64().ok_or_else(|| {
            DecoderError::InvalidConfiguration(
                "decoder has more than 64 observables; use decode_obs() for the wide mask".into(),
            )
        })
    }

    /// Decode every per-observable subgraph and pack the flips into a wide
    /// [`ObsMask`], mapping each subgraph's local result to its GLOBAL observable
    /// index. Supports more than 64 observables with no truncation.
    fn decode_obs(&mut self, syndrome: &[u8]) -> Result<ObsMask, DecoderError> {
        let mut obs_mask = ObsMask::new();

        for (i, (sg, dec)) in self
            .subgraphs
            .iter()
            .zip(self.decoders.iter_mut())
            .enumerate()
        {
            let n = sg.detector_map.len();
            if n == 0 {
                continue;
            }

            let buf = &mut self.sub_syndromes[i];
            for (sub_idx, &full_idx) in sg.detector_map.iter().enumerate() {
                buf[sub_idx] = if full_idx < syndrome.len() {
                    syndrome[full_idx]
                } else {
                    0
                };
            }

            let sub_obs = dec.decode_to_observables(&buf[..n])?;

            if sub_obs & 1 != 0 {
                obs_mask.set(sg.observable_idx);
            }
        }

        Ok(obs_mask)
    }
}

/// Parallel per-logical-operator subgraph decoder using rayon.
pub struct ParallelLogicalSubgraphDecoder {
    subgraphs: Vec<LogicalSubgraph>,
    decoders: Vec<std::sync::Mutex<Box<dyn ObservableDecoder + Send>>>,
}

impl ParallelLogicalSubgraphDecoder {
    /// Build from a DEM string, stabilizer coordinates, and inner decoder factory.
    ///
    /// # Errors
    ///
    /// Returns error if the DEM is malformed or the factory fails.
    pub fn from_dem<F>(
        dem: &str,
        stab_coords: &StabCoords,
        mut factory: F,
    ) -> Result<Self, DecoderError>
    where
        F: FnMut(&DemMatchingGraph) -> Result<Box<dyn ObservableDecoder + Send>, DecoderError>,
    {
        let subgraphs = partition_dem_by_logical(dem, stab_coords)?;

        let mut decoders = Vec::with_capacity(subgraphs.len());
        for sg in &subgraphs {
            decoders.push(std::sync::Mutex::new(factory(&sg.graph)?));
        }

        Ok(Self {
            subgraphs,
            decoders,
        })
    }

    /// Decode using parallel subgraph decoding.
    ///
    /// # Errors
    ///
    /// Returns error if any subgraph decoder fails.
    pub fn decode_parallel(&self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        use rayon::prelude::*;

        let results: Vec<Result<bool, DecoderError>> = self
            .subgraphs
            .par_iter()
            .zip(self.decoders.par_iter())
            .map(|(sg, dec_mutex)| {
                let n = sg.detector_map.len();
                if n == 0 {
                    return Ok(false);
                }

                let mut sub_syn = vec![0u8; n];
                for (sub_idx, &full_idx) in sg.detector_map.iter().enumerate() {
                    sub_syn[sub_idx] = if full_idx < syndrome.len() {
                        syndrome[full_idx]
                    } else {
                        0
                    };
                }

                let mut dec = dec_mutex.lock().unwrap();
                let sub_obs = dec.decode_to_observables(&sub_syn)?;
                Ok(sub_obs & 1 != 0)
            })
            .collect();

        let mut obs_mask = ObsMask::new();
        for (sg, result) in self.subgraphs.iter().zip(results) {
            if result? {
                obs_mask.set(sg.observable_idx);
            }
        }
        // This convenience returns a u64; error (don't truncate) above 64.
        obs_mask.to_u64().ok_or_else(|| {
            DecoderError::InvalidConfiguration(
                "decoder has more than 64 observables; the parallel u64 path supports at most 64"
                    .into(),
            )
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    struct NullDecoder;
    impl ObservableDecoder for NullDecoder {
        fn decode_obs(&mut self, _: &[u8]) -> Result<crate::obs_mask::ObsMask, DecoderError> {
            Ok(crate::obs_mask::ObsMask::new())
        }
    }

    struct FixedDecoder(u64);
    impl ObservableDecoder for FixedDecoder {
        fn decode_obs(
            &mut self,
            syndrome: &[u8],
        ) -> Result<crate::obs_mask::ObsMask, DecoderError> {
            if syndrome.iter().any(|&v| v != 0) {
                Ok(crate::obs_mask::ObsMask::from_u64(self.0))
            } else {
                Ok(crate::obs_mask::ObsMask::new())
            }
        }
    }

    fn simple_stab_coords() -> StabCoords {
        // Two qubits with non-overlapping X/Z positions.
        vec![
            QubitStabCoords {
                x_positions: vec![(1.0, 0.0)],
                z_positions: vec![(0.0, 1.0)],
            },
            QubitStabCoords {
                x_positions: vec![(3.0, 0.0)],
                z_positions: vec![(2.0, 1.0)],
            },
        ]
    }

    #[test]
    fn test_classify_detector() {
        let sc = simple_stab_coords();
        assert_eq!(
            classify_detector(1.0, 0.0, &sc),
            Some(DetectorGroup {
                qubit_idx: 0,
                stab_type: StabType::X
            }),
        );
        assert_eq!(
            classify_detector(0.0, 1.0, &sc),
            Some(DetectorGroup {
                qubit_idx: 0,
                stab_type: StabType::Z
            }),
        );
        assert_eq!(
            classify_detector(3.0, 0.0, &sc),
            Some(DetectorGroup {
                qubit_idx: 1,
                stab_type: StabType::X
            }),
        );
        assert_eq!(classify_detector(99.0, 99.0, &sc), None);
    }

    #[test]
    fn test_partition_simple() {
        // Two detectors with coords, one observable.
        let dem = concat!(
            "detector(1, 0, 0) D0\n",
            "detector(0, 1, 0) D1\n",
            "error(0.01) D0 D1 L0\n",
            "error(0.01) D0 L0\n", // boundary edge → D0 is (qubit 0, X)
        );
        let sc = simple_stab_coords();
        let sgs = partition_dem_by_logical(dem, &sc).unwrap();
        assert_eq!(sgs.len(), 1);
        // Boundary edge D0 L0 → D0 is qubit 0 X-type.
        // Observing region = all qubit-0 X-type detectors = {D0}.
        // But D0-D1 is also an observable mechanism, and D1 is qubit 0 Z-type.
        // Since D1 is NOT in the same group as D0, it's excluded from the
        // observing region. The edge D0-D1 projects to D0-boundary within
        // the subgraph.
        assert_eq!(sgs[0].detector_map, vec![0]);
    }

    #[test]
    fn test_membership_seam_matches_partition() {
        // coordinate_membership_from_dem + subgraphs_from_membership must equal
        // the all-in-one partition (the seam is behaviour-preserving).
        let dem = concat!(
            "detector(1, 0, 0) D0\n",
            "detector(0, 1, 0) D1\n",
            "detector(3, 0, 0) D2\n",
            "detector(2, 1, 0) D3\n",
            "error(0.01) D0 L0\n",
            "error(0.01) D0 D1\n",
            "error(0.01) D2 L1\n",
            "error(0.01) D2 D3\n",
        );
        let sc = simple_stab_coords();

        let sdem = SparseDem::from_dem_str(dem).unwrap();
        let membership = coordinate_membership_from_dem(&sdem, &sc, None).unwrap();
        assert_eq!(membership, vec![vec![0usize], vec![2usize]]);

        let via_seam = subgraphs_from_membership(&sdem, &membership).unwrap();
        let direct = partition_dem_by_logical(dem, &sc).unwrap();
        assert_eq!(via_seam.len(), direct.len());
        for (a, b) in via_seam.iter().zip(direct.iter()) {
            assert_eq!(a.detector_map, b.detector_map);
        }
    }

    #[test]
    fn test_subgraphs_from_external_membership() {
        // A region source (e.g. a future back-propagation builder) can hand in
        // its own membership and the extractor builds the subgraphs from it.
        let dem = concat!(
            "detector(1, 0, 0) D0\n",
            "detector(1, 0, 1) D1\n",
            "error(0.01) D0 L0\n",
            "error(0.01) D0 D1\n",
        );
        let sdem = SparseDem::from_dem_str(dem).unwrap();
        // Membership the coordinate path would NOT produce on its own (both times).
        let membership = vec![vec![0usize, 1usize]];
        let sgs = subgraphs_from_membership(&sdem, &membership).unwrap();
        assert_eq!(sgs.len(), 1);
        assert_eq!(sgs[0].detector_map, vec![0, 1]);

        // >64 observable membership is now SUPPORTED (wide ObsMask), not rejected.
        let big: Vec<Vec<usize>> = (0..65).map(|_| Vec::new()).collect();
        assert_eq!(subgraphs_from_membership(&sdem, &big).unwrap().len(), 65);

        // An out-of-range membership detector id must error, not panic
        // (the DEM has 2 detectors D0,D1; detector 5 is past `inverse_map`).
        assert!(matches!(
            subgraphs_from_membership(&sdem, &[vec![5usize]]),
            Err(DecoderError::InvalidConfiguration(_))
        ));
    }

    #[test]
    fn test_membership_exact_time_vs_radius() {
        // D0 and D1 are both qubit-0 X at times 0 and 1. The boundary edge is at
        // time 0. Default (None) includes only the boundary time; a radius pulls
        // in the neighbouring time step.
        let dem = concat!(
            "detector(1, 0, 0) D0\n", // qubit 0 X, time 0
            "detector(1, 0, 1) D1\n", // qubit 0 X, time 1
            "error(0.01) D0 L0\n",    // boundary at time 0
            "error(0.01) D0 D1\n",
        );
        let sc = simple_stab_coords();
        let sdem = SparseDem::from_dem_str(dem).unwrap();

        let exact = coordinate_membership_from_dem(&sdem, &sc, None).unwrap();
        assert_eq!(exact, vec![vec![0usize]], "None = exact boundary time only");

        let widened = coordinate_membership_from_dem(&sdem, &sc, Some(1)).unwrap();
        assert_eq!(
            widened,
            vec![vec![0usize, 1usize]],
            "radius pulls in time 1"
        );
    }

    #[test]
    fn test_decomposed_mechanism_partitions() {
        // A `^`-decomposed mechanism is XOR-combined into one 2-detector
        // mechanism, so it is NOT a boundary edge and does not seed the region;
        // the single-detector `D0 L0` does. Partition must still succeed.
        let dem = concat!(
            "detector(1, 0, 0) D0\n",
            "detector(0, 1, 0) D1\n",
            "error(0.01) D0 L0\n",      // boundary → qubit 0 X
            "error(0.02) D0 ^ D1 L0\n", // decomposed; XOR -> {D0, D1}
        );
        let sc = simple_stab_coords();
        let sdem = SparseDem::from_dem_str(dem).unwrap();
        // The `^` line parsed as one mechanism with both detectors.
        assert_eq!(sdem.mechanisms.len(), 2);
        let membership = coordinate_membership_from_dem(&sdem, &sc, None).unwrap();
        assert_eq!(membership, vec![vec![0usize]]); // D1 (qubit-0 Z) excluded
    }

    #[test]
    fn test_partition_two_qubits() {
        let dem = concat!(
            "detector(1, 0, 0) D0\n",
            "detector(0, 1, 0) D1\n",
            "detector(3, 0, 0) D2\n",
            "detector(2, 1, 0) D3\n",
            "error(0.01) D0 L0\n", // boundary: D0 = qubit 0 X
            "error(0.01) D0 D1\n", // D0-D1 edge
            "error(0.01) D2 L1\n", // boundary: D2 = qubit 1 X
            "error(0.01) D2 D3\n", // D2-D3 edge
        );
        let sc = simple_stab_coords();
        let sgs = partition_dem_by_logical(dem, &sc).unwrap();
        assert_eq!(sgs.len(), 2);
        assert_eq!(sgs[0].detector_map, vec![0]); // qubit 0 X-type only
        assert_eq!(sgs[1].detector_map, vec![2]); // qubit 1 X-type only
    }

    #[test]
    fn test_decoder_routing() {
        let dem = concat!(
            "detector(1, 0, 0) D0\n",
            "detector(3, 0, 0) D1\n",
            "error(0.01) D0 L0\n",
            "error(0.01) D1 L1\n",
        );
        let sc = simple_stab_coords();
        let mut dec = LogicalSubgraphDecoder::from_dem(dem, &sc, |_| {
            Ok(Box::new(FixedDecoder(1)) as Box<dyn ObservableDecoder + Send + Sync>)
        })
        .unwrap();

        // Defect in obs 0's region only
        let obs = dec.decode_to_observables(&[1, 0]).unwrap();
        assert_eq!(obs, 0b01);

        // Defect in obs 1's region only
        let obs = dec.decode_to_observables(&[0, 1]).unwrap();
        assert_eq!(obs, 0b10);
    }

    #[test]
    fn test_unclassified_used_detector_errors() {
        // D1 is used in a mechanism but sits at coords matching no stabilizer.
        let dem = concat!(
            "detector(1, 0, 0) D0\n",
            "detector(99, 99, 0) D1\n",
            "error(0.01) D0 L0\n",
            "error(0.01) D0 D1\n", // D1 used here, but won't classify
        );
        let sc = simple_stab_coords();
        let err = partition_dem_by_logical(dem, &sc).unwrap_err();
        assert!(
            matches!(err, DecoderError::InvalidConfiguration(_)),
            "expected InvalidConfiguration, got {err:?}"
        );
    }

    #[test]
    fn test_unclassified_unused_detector_ignored() {
        // D1 has bad coords but is never used in a mechanism -> ignored, no error.
        let dem = concat!(
            "detector(1, 0, 0) D0\n",
            "detector(99, 99, 0) D1\n",
            "error(0.01) D0 L0\n",
        );
        let sc = simple_stab_coords();
        assert!(partition_dem_by_logical(dem, &sc).is_ok());
    }

    #[test]
    fn test_single_detector_d0() {
        // Regression: a DEM whose only detector is D0 must parse with
        // num_detectors == 1 (not 0). D0 used to never set has_any_detector
        // because `0 > max_detector` is false, yielding an empty det_group
        // and an out-of-bounds panic downstream.
        let dem = concat!(
            "detector(1, 0, 0) D0\n", // qubit 0 X
            "error(0.01) D0 L0\n",    // boundary edge on D0
        );
        let sc = simple_stab_coords();
        let sgs = partition_dem_by_logical(dem, &sc).unwrap();
        assert_eq!(sgs.len(), 1);
        assert_eq!(sgs[0].detector_map, vec![0]);
    }

    #[test]
    fn test_more_than_64_observables_decode_wide() {
        // 65 observables: construction now SUCCEEDS (no >64 reject), and the wide
        // `decode_obs` / `decode_count_batched` paths represent observable 64
        // without truncation. The `u64` convenience method errors instead.
        use std::fmt::Write;
        let mut dem = String::from("detector(1, 0, 0) D0\n");
        for l in 0..65 {
            writeln!(dem, "error(0.01) D0 L{l}").unwrap();
        }
        let sc = simple_stab_coords();

        // Every per-observable subgraph flips its observable when D0 fires.
        let mut dec = LogicalSubgraphDecoder::from_dem(&dem, &sc, |_| {
            Ok(Box::new(FixedDecoder(1)) as Box<dyn ObservableDecoder + Send + Sync>)
        })
        .unwrap();

        // Wide decode: all 65 observables flip when D0 is set; bit 64 is present.
        let wide = dec.decode_obs(&[1]).unwrap();
        assert_eq!(wide.count_ones(), 65);
        assert!(wide.get(64), "observable 64 must be representable");
        assert_eq!(wide.to_u64(), None, "65 bits does not fit a u64");

        // The u64 convenience errors rather than truncating.
        assert!(matches!(
            dec.decode_to_observables(&[1]),
            Err(DecoderError::InvalidConfiguration(_))
        ));

        // Batch decode-count compares wide masks: an all-flip expected mask
        // matches the all-flip prediction (zero errors); a narrower expected
        // mask (missing observable 64) is counted as an error.
        let mut all_flip = ObsMask::new();
        for l in 0..65 {
            all_flip.set(l);
        }
        let mut missing_64 = ObsMask::new();
        for l in 0..64 {
            missing_64.set(l);
        }
        assert_eq!(
            dec.decode_count_batched(&[vec![1]], std::slice::from_ref(&all_flip))
                .unwrap(),
            0
        );
        assert_eq!(
            dec.decode_count_batched(&[vec![1]], std::slice::from_ref(&missing_64))
                .unwrap(),
            1
        );
    }

    #[test]
    fn test_parallel_decoder() {
        let dem = concat!(
            "detector(1, 0, 0) D0\n",
            "detector(3, 0, 0) D1\n",
            "error(0.01) D0 L0\n",
            "error(0.01) D1 L1\n",
        );
        let sc = simple_stab_coords();
        let dec = ParallelLogicalSubgraphDecoder::from_dem(dem, &sc, |_| {
            Ok(Box::new(NullDecoder) as Box<dyn ObservableDecoder + Send>)
        })
        .unwrap();

        let obs = dec.decode_parallel(&[0, 0]).unwrap();
        assert_eq!(obs, 0);
    }
}
