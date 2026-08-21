// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific language governing permissions and
// limitations under the License.

//! Exact edge coloring of bipartite multigraphs.

/// An exact edge coloring of a bipartite multigraph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BipartiteEdgeColoring {
    colors: Vec<usize>,
    num_colors: usize,
}

impl BipartiteEdgeColoring {
    /// The color assigned to each input edge, in input order.
    #[must_use]
    pub fn colors(&self) -> &[usize] {
        &self.colors
    }

    /// The number of colors, equal to the graph's maximum degree.
    #[must_use]
    pub fn num_colors(&self) -> usize {
        self.num_colors
    }
}

/// Errors from [`bipartite_edge_coloring`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BipartiteEdgeColoringError {
    /// An edge names a left vertex outside `0..num_left`.
    LeftEndpointOutOfRange {
        /// Input edge index.
        edge: usize,
        /// Invalid vertex index.
        vertex: usize,
        /// Size of the left vertex set.
        num_left: usize,
    },
    /// An edge names a right vertex outside `0..num_right`.
    RightEndpointOutOfRange {
        /// Input edge index.
        edge: usize,
        /// Invalid vertex index.
        vertex: usize,
        /// Size of the right vertex set.
        num_right: usize,
    },
    /// An internal perfect matching was unexpectedly absent.
    MissingPerfectMatching {
        /// Color whose matching could not be found.
        color: usize,
    },
}

impl std::fmt::Display for BipartiteEdgeColoringError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LeftEndpointOutOfRange {
                edge,
                vertex,
                num_left,
            } => write!(
                formatter,
                "bipartite edge {edge} has left endpoint {vertex}, but there are {num_left} left vertices"
            ),
            Self::RightEndpointOutOfRange {
                edge,
                vertex,
                num_right,
            } => write!(
                formatter,
                "bipartite edge {edge} has right endpoint {vertex}, but there are {num_right} right vertices"
            ),
            Self::MissingPerfectMatching { color } => write!(
                formatter,
                "regularized bipartite graph has no perfect matching for color {color}"
            ),
        }
    }
}

impl std::error::Error for BipartiteEdgeColoringError {}

#[derive(Clone, Copy, Debug)]
struct Edge {
    left: usize,
    right: usize,
    original: Option<usize>,
}

/// Color every edge of a bipartite multigraph with exactly its maximum degree colors.
///
/// Parallel edges are distinct and are colored independently. The implementation pads both
/// vertex sets to the same size, deterministically adds dummy edges until the graph is
/// Delta-regular, and removes one deterministic perfect matching per color. A regular bipartite
/// multigraph has a perfect matching, and repeating the construction decomposes all edges into
/// Delta matchings. This is the constructive form of Konig's bipartite edge-coloring theorem.
/// Input edge order and ascending vertex order are the only tie breakers.
///
/// # Errors
///
/// Returns an error if an endpoint is out of range. A missing perfect matching reports an internal
/// invariant failure instead of returning a partial coloring.
pub fn bipartite_edge_coloring(
    num_left: usize,
    num_right: usize,
    input_edges: &[(usize, usize)],
) -> Result<BipartiteEdgeColoring, BipartiteEdgeColoringError> {
    let mut left_degrees = vec![0_usize; num_left];
    let mut right_degrees = vec![0_usize; num_right];
    let mut edges = Vec::with_capacity(input_edges.len());
    for (edge_index, &(left, right)) in input_edges.iter().enumerate() {
        if left >= num_left {
            return Err(BipartiteEdgeColoringError::LeftEndpointOutOfRange {
                edge: edge_index,
                vertex: left,
                num_left,
            });
        }
        if right >= num_right {
            return Err(BipartiteEdgeColoringError::RightEndpointOutOfRange {
                edge: edge_index,
                vertex: right,
                num_right,
            });
        }
        left_degrees[left] += 1;
        right_degrees[right] += 1;
        edges.push(Edge {
            left,
            right,
            original: Some(edge_index),
        });
    }

    let delta = left_degrees
        .iter()
        .chain(&right_degrees)
        .copied()
        .max()
        .unwrap_or(0);
    if delta == 0 {
        return Ok(BipartiteEdgeColoring {
            colors: Vec::new(),
            num_colors: 0,
        });
    }

    let side_size = num_left.max(num_right);
    left_degrees.resize(side_size, 0);
    right_degrees.resize(side_size, 0);

    // Pair the left and right deficits in stable vertex order. Multiple dummy edges are allowed,
    // just as multiple input edges are allowed.
    let mut left = 0;
    let mut right = 0;
    while left < side_size && right < side_size {
        while left < side_size && left_degrees[left] == delta {
            left += 1;
        }
        while right < side_size && right_degrees[right] == delta {
            right += 1;
        }
        if left == side_size || right == side_size {
            break;
        }
        edges.push(Edge {
            left,
            right,
            original: None,
        });
        left_degrees[left] += 1;
        right_degrees[right] += 1;
    }
    debug_assert!(left_degrees.iter().all(|&degree| degree == delta));
    debug_assert!(right_degrees.iter().all(|&degree| degree == delta));

    let mut adjacency = vec![Vec::new(); side_size];
    for (edge_index, edge) in edges.iter().enumerate() {
        adjacency[edge.left].push(edge_index);
    }
    let mut active = vec![true; edges.len()];
    let mut colors = vec![usize::MAX; input_edges.len()];

    for color in 0..delta {
        let matching = perfect_matching(side_size, &edges, &adjacency, &active)
            .ok_or(BipartiteEdgeColoringError::MissingPerfectMatching { color })?;
        for edge_index in matching {
            active[edge_index] = false;
            if let Some(original) = edges[edge_index].original {
                colors[original] = color;
            }
        }
    }
    debug_assert!(colors.iter().all(|&color| color < delta));

    Ok(BipartiteEdgeColoring {
        colors,
        num_colors: delta,
    })
}

fn perfect_matching(
    side_size: usize,
    edges: &[Edge],
    adjacency: &[Vec<usize>],
    active: &[bool],
) -> Option<Vec<usize>> {
    let mut matched_right = vec![None; side_size];
    for left in 0..side_size {
        let mut visited_right = vec![false; side_size];
        if !augment(
            left,
            edges,
            adjacency,
            active,
            &mut visited_right,
            &mut matched_right,
        ) {
            return None;
        }
    }
    matched_right.into_iter().collect()
}

fn augment(
    left: usize,
    edges: &[Edge],
    adjacency: &[Vec<usize>],
    active: &[bool],
    visited_right: &mut [bool],
    matched_right: &mut [Option<usize>],
) -> bool {
    for &edge_index in &adjacency[left] {
        if !active[edge_index] {
            continue;
        }
        let right = edges[edge_index].right;
        if visited_right[right] {
            continue;
        }
        visited_right[right] = true;
        let can_reassign = matched_right[right].is_none_or(|matched_edge| {
            augment(
                edges[matched_edge].left,
                edges,
                adjacency,
                active,
                visited_right,
                matched_right,
            )
        });
        if can_reassign {
            matched_right[right] = Some(edge_index);
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    use std::collections::BTreeSet;

    fn verify_coloring(
        num_left: usize,
        num_right: usize,
        edges: &[(usize, usize)],
        coloring: &BipartiteEdgeColoring,
    ) -> Result<(), String> {
        if coloring.colors().len() != edges.len() {
            return Err("not every edge has exactly one color".to_string());
        }
        let mut left_degrees = vec![0_usize; num_left];
        let mut right_degrees = vec![0_usize; num_right];
        for &(left, right) in edges {
            left_degrees[left] += 1;
            right_degrees[right] += 1;
        }
        let delta = left_degrees
            .iter()
            .chain(&right_degrees)
            .copied()
            .max()
            .unwrap_or(0);
        if coloring.num_colors() != delta {
            return Err(format!(
                "used {} colors for maximum degree {delta}",
                coloring.num_colors()
            ));
        }
        let mut seen = BTreeSet::new();
        for (edge, (&(left, right), &color)) in edges.iter().zip(coloring.colors()).enumerate() {
            if color >= delta {
                return Err(format!("edge {edge} has out-of-range color {color}"));
            }
            if !seen.insert((true, left, color)) {
                return Err(format!("two color-{color} edges share left vertex {left}"));
            }
            if !seen.insert((false, right, color)) {
                return Err(format!(
                    "two color-{color} edges share right vertex {right}"
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn colors_parallel_edges_and_unbalanced_vertex_sets() {
        let edges = vec![(0, 0), (0, 0), (0, 1), (1, 0), (2, 1)];
        let coloring = bipartite_edge_coloring(4, 2, &edges).unwrap();

        verify_coloring(4, 2, &edges, &coloring).unwrap();
        assert_eq!(coloring.num_colors(), 3);
        assert_eq!(
            coloring,
            bipartite_edge_coloring(4, 2, &edges).unwrap(),
            "stable input order must produce a deterministic coloring"
        );
    }

    #[test]
    fn seeded_random_multigraphs_have_exact_delta_colorings() {
        let mut rng = StdRng::seed_from_u64(0x25_ec01_0a71);
        for case in 0..256 {
            let num_left = rng.random_range(1..=12);
            let num_right = rng.random_range(1..=12);
            let num_edges = rng.random_range(0..=80);
            let edges = (0..num_edges)
                .map(|_| {
                    (
                        rng.random_range(0..num_left),
                        rng.random_range(0..num_right),
                    )
                })
                .collect::<Vec<_>>();

            let coloring = bipartite_edge_coloring(num_left, num_right, &edges)
                .unwrap_or_else(|error| panic!("random case {case}: {error}"));
            verify_coloring(num_left, num_right, &edges, &coloring)
                .unwrap_or_else(|error| panic!("random case {case}: {error}"));
        }
    }

    #[test]
    fn mutation_shared_vertex_same_color_is_caught() {
        let edges = vec![(0, 0), (0, 1), (1, 0)];
        let mut coloring = bipartite_edge_coloring(2, 2, &edges).unwrap();
        coloring.colors[1] = coloring.colors[0];

        let error = verify_coloring(2, 2, &edges, &coloring).unwrap_err();
        assert!(error.contains("share left vertex 0"), "{error}");
    }

    #[test]
    fn rejects_out_of_range_endpoints() {
        assert!(matches!(
            bipartite_edge_coloring(2, 1, &[(2, 0)]),
            Err(BipartiteEdgeColoringError::LeftEndpointOutOfRange { edge: 0, .. })
        ));
        assert!(matches!(
            bipartite_edge_coloring(2, 1, &[(0, 1)]),
            Err(BipartiteEdgeColoringError::RightEndpointOutOfRange { edge: 0, .. })
        ));
    }
}
