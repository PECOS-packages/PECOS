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

//! Ghost protocol for modular per-qubit decoding across transversal gates.
//!
//! Based on Turner et al. (arXiv:2505.23567): decomposes order-3
//! hyperedges from transversal CNOT into per-qubit ghost edges.
//! Each qubit's decoder runs independently with sparse message passing,
//! enabling scalable decoding of many logical qubits.
//!
//! # Algorithm
//!
//! 1. Decompose cross-qubit hyperedges into ghost edge + ghost singleton
//! 2. Each qubit decoded independently with ghost edges in its graph
//! 3. If matching includes a ghost edge: refine syndrome, message partner
//! 4. Partner flips its ghost singleton defect, re-decodes
//! 5. Iterate until convergence (no new ghost edges detected)
//!
//! # References
//!
//! - Turner et al. "Scalable decoding protocols for fast transversal
//!   logic in the surface code" (arXiv:2505.23567, PRX Quantum 2026)
//! - Cain et al. "Fast correlated decoding of transversal logical
//!   algorithms" (arXiv:2505.13587)

/// A ghost edge: fragment of a cross-qubit hyperedge.
///
/// When a measurement error before a transversal CNOT creates a
/// 3-detector hyperedge spanning two qubits, it decomposes into:
/// - `ghost_edge`: time-like edge within one qubit (connects two
///   detectors on the same qubit across the gate boundary)
/// - `ghost_singleton`: boundary edge on the partner qubit (flips
///   one detector on the partner)
#[derive(Debug, Clone)]
pub struct GhostEdge {
    /// Qubit (patch) that owns this ghost edge.
    pub owner_qubit: usize,
    /// Local detector index for the first endpoint.
    pub det_a: u32,
    /// Local detector index for the second endpoint.
    pub det_b: u32,
    /// Partner qubit that owns the ghost singleton.
    pub partner_qubit: usize,
    /// Local detector index of the ghost singleton on the partner.
    pub partner_det: u32,
    /// Edge weight (log-likelihood ratio).
    pub weight: f64,
}

/// Ghost protocol state for iterative decoding.
///
/// Tracks ghost edges, messages between per-qubit decoders, and
/// syndrome refinement state. Created once per circuit structure,
/// reused across shots.
pub struct GhostProtocolState {
    /// Ghost edges grouped by owner qubit.
    pub ghost_edges: Vec<Vec<GhostEdge>>,
    /// Number of logical qubits (patches).
    pub num_qubits: usize,
    /// Maximum iterations before giving up.
    pub max_iterations: usize,
}

impl GhostProtocolState {
    /// Create ghost protocol state for a circuit.
    ///
    /// `ghost_edges`: all ghost edges, will be grouped by owner qubit.
    /// `num_qubits`: number of logical qubits.
    #[must_use]
    pub fn new(ghost_edges: Vec<GhostEdge>, num_qubits: usize) -> Self {
        let mut grouped = vec![Vec::new(); num_qubits];
        for ge in ghost_edges {
            if ge.owner_qubit < num_qubits {
                grouped[ge.owner_qubit].push(ge);
            }
        }
        Self {
            ghost_edges: grouped,
            num_qubits,
            max_iterations: 10,
        }
    }

    /// Number of ghost edges for a qubit.
    #[must_use]
    pub fn num_ghost_edges(&self, qubit: usize) -> usize {
        self.ghost_edges.get(qubit).map_or(0, std::vec::Vec::len)
    }

    /// Total ghost edges across all qubits.
    #[must_use]
    pub fn total_ghost_edges(&self) -> usize {
        self.ghost_edges.iter().map(std::vec::Vec::len).sum()
    }
}

/// Message from one qubit's decoder to another.
///
/// When a ghost edge is detected in the matching, the owner sends
/// this message to the partner: "flip your ghost singleton defect."
#[derive(Debug, Clone)]
pub struct GhostMessage {
    /// Target qubit.
    pub target_qubit: usize,
    /// Detector to flip on the target.
    pub flip_detector: u32,
}

/// Extract ghost edges from a DEM at transversal CNOT boundaries.
///
/// Identifies 3-detector mechanisms where:
/// - 2 detectors are on one qubit (the ghost edge endpoints)
/// - 1 detector is on another qubit (the ghost singleton)
///
/// The qubit assignment comes from the detector's spatial coordinates
/// and the stabilizer coordinate map.
///
/// Returns the ghost edges for the ghost protocol.
/// Extract ghost edges from a DEM by identifying 3-detector hyperedges
/// and decomposing them by qubit ownership.
///
/// For each 3-detector mechanism where 2 detectors are on one qubit
/// and 1 is on another:
/// - `ghost_edge` = the 2-detector pair (within one qubit)
/// - `ghost_singleton` = the lone detector (on the partner qubit)
#[must_use]
pub fn extract_ghost_edges_from_dem(
    dem_str: &str,
    stab_coords: &crate::observable_subgraph::StabCoords,
) -> Vec<GhostEdge> {
    use crate::observable_subgraph::classify_detector;
    use std::collections::BTreeMap;

    // Parse detector coordinates
    let det_coords = crate::dem::parse_detector_coords(dem_str);
    let mut coord_map: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    for dc in &det_coords {
        coord_map.insert(dc.id as usize, dc.coords.clone());
    }

    // Classify detectors by qubit
    let mut det_qubit: BTreeMap<usize, usize> = BTreeMap::new();
    for (&d, coords) in &coord_map {
        if coords.len() >= 2
            && let Some(group) = classify_detector(coords[0], coords[1], stab_coords)
        {
            det_qubit.insert(d, group.qubit_idx);
        }
    }

    let mut ghost_edges = Vec::new();

    for line in dem_str.lines() {
        let line = line.trim();
        if !line.starts_with("error(") {
            continue;
        }

        let Some(close) = line.find(')') else {
            continue;
        };

        let prob: f64 = match line[6..close].parse() {
            Ok(p) => p,
            Err(_) => continue,
        };

        let mut dets = Vec::new();
        for token in line[close + 1..].split_whitespace() {
            if let Some(d_str) = token.strip_prefix('D')
                && let Ok(d) = d_str.parse::<usize>()
            {
                dets.push(d);
            }
        }

        if dets.len() != 3 {
            continue;
        }

        let qs: Vec<Option<usize>> = dets.iter().map(|d| det_qubit.get(d).copied()).collect();

        if qs.iter().any(std::option::Option::is_none) {
            continue;
        }
        let qs: Vec<usize> = qs.into_iter().flatten().collect();

        let weight = if prob < 1.0 && prob > 0.0 {
            ((1.0 - prob) / prob).ln()
        } else {
            0.0
        };

        // Decompose: 2 on one qubit (ghost edge), 1 on another (singleton)
        let decompose =
            |owner_q: usize, a: usize, b: usize, partner_q: usize, c: usize| GhostEdge {
                owner_qubit: owner_q,
                det_a: a as u32,
                det_b: b as u32,
                partner_qubit: partner_q,
                partner_det: c as u32,
                weight,
            };

        if qs[0] == qs[1] && qs[0] != qs[2] {
            ghost_edges.push(decompose(qs[0], dets[0], dets[1], qs[2], dets[2]));
        } else if qs[0] == qs[2] && qs[0] != qs[1] {
            ghost_edges.push(decompose(qs[0], dets[0], dets[2], qs[1], dets[1]));
        } else if qs[1] == qs[2] && qs[1] != qs[0] {
            ghost_edges.push(decompose(qs[1], dets[1], dets[2], qs[0], dets[0]));
        }
    }

    ghost_edges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ghost_protocol_state() {
        let edges = vec![
            GhostEdge {
                owner_qubit: 0,
                det_a: 5,
                det_b: 10,
                partner_qubit: 1,
                partner_det: 7,
                weight: 3.0,
            },
            GhostEdge {
                owner_qubit: 1,
                det_a: 3,
                det_b: 8,
                partner_qubit: 0,
                partner_det: 12,
                weight: 3.0,
            },
        ];

        let state = GhostProtocolState::new(edges, 2);
        assert_eq!(state.num_qubits, 2);
        assert_eq!(state.num_ghost_edges(0), 1);
        assert_eq!(state.num_ghost_edges(1), 1);
        assert_eq!(state.total_ghost_edges(), 2);
    }

    #[test]
    fn test_empty_ghost_protocol() {
        let state = GhostProtocolState::new(Vec::new(), 4);
        assert_eq!(state.total_ghost_edges(), 0);
    }

    #[test]
    fn test_extract_ghost_edges_from_synthetic_dem() {
        use crate::observable_subgraph::QubitStabCoords;

        // Two qubits: qubit 0 has X-stab at (1,1) and Z-stab at (3,1),
        // qubit 1 has X-stab at (7,1) and Z-stab at (9,1).
        let stab_coords = vec![
            QubitStabCoords {
                x_positions: vec![(1.0, 1.0)],
                z_positions: vec![(3.0, 1.0)],
            },
            QubitStabCoords {
                x_positions: vec![(7.0, 1.0)],
                z_positions: vec![(9.0, 1.0)],
            },
        ];

        // DEM with:
        // - D0 at (1,1,0) -> qubit 0 (X-stab)
        // - D1 at (3,1,0) -> qubit 0 (Z-stab)
        // - D2 at (7,1,0) -> qubit 1 (X-stab)
        // - 3-body error: D0 D1 D2 (2 on qubit 0, 1 on qubit 1)
        // - 2-body error: D0 D1 (same qubit, no ghost edge)
        let dem = "\
            detector(1, 1, 0) D0\n\
            detector(3, 1, 0) D1\n\
            detector(7, 1, 0) D2\n\
            error(0.01) D0 D1 D2\n\
            error(0.02) D0 D1\n";

        let edges = extract_ghost_edges_from_dem(dem, &stab_coords);

        // Should extract exactly 1 ghost edge from the 3-body mechanism
        assert_eq!(edges.len(), 1);

        let e = &edges[0];
        assert_eq!(e.owner_qubit, 0); // D0 and D1 are on qubit 0
        assert_eq!(e.det_a, 0);
        assert_eq!(e.det_b, 1);
        assert_eq!(e.partner_qubit, 1); // D2 is on qubit 1
        assert_eq!(e.partner_det, 2);

        // Weight should be ln((1 - 0.01) / 0.01) ≈ 4.595
        assert!((e.weight - 4.595).abs() < 0.01);
    }

    #[test]
    fn test_extract_no_ghost_edges_graphlike_dem() {
        use crate::observable_subgraph::QubitStabCoords;

        let stab_coords = vec![QubitStabCoords {
            x_positions: vec![(1.0, 1.0)],
            z_positions: vec![(3.0, 1.0)],
        }];

        // Only 2-body errors -> no ghost edges
        let dem = "\
            detector(1, 1, 0) D0\n\
            detector(3, 1, 0) D1\n\
            error(0.01) D0 D1\n\
            error(0.005) D0\n";

        let edges = extract_ghost_edges_from_dem(dem, &stab_coords);
        assert_eq!(edges.len(), 0);
    }

    #[test]
    fn test_extract_three_same_qubit_no_ghost() {
        use crate::observable_subgraph::QubitStabCoords;

        let stab_coords = vec![QubitStabCoords {
            x_positions: vec![(1.0, 1.0), (1.0, 3.0)],
            z_positions: vec![(3.0, 1.0)],
        }];

        // All 3 detectors on same qubit -> no decomposition
        let dem = "\
            detector(1, 1, 0) D0\n\
            detector(1, 3, 0) D1\n\
            detector(3, 1, 0) D2\n\
            error(0.01) D0 D1 D2\n";

        let edges = extract_ghost_edges_from_dem(dem, &stab_coords);
        assert_eq!(edges.len(), 0);
    }
}
