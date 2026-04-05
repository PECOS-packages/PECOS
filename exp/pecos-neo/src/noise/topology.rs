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

//! Topology helpers for spatial noise modeling.
//!
//! This module provides common qubit connectivity patterns used for defining
//! crosstalk neighborhoods and correlated noise models.
//!
//! # Overview
//!
//! Quantum devices have physical layouts that affect how errors propagate.
//! This module provides:
//!
//! - **Neighbor functions** - Define which qubits are adjacent for crosstalk
//! - **Distance functions** - Calculate qubit separation for correlation decay
//! - **Decay functions** - Model how correlations decrease with distance
//!
//! # Neighbor Functions
//!
//! Use these with `CompositeCrosstalkChannel::local()` to define which qubits
//! are affected by crosstalk:
//!
//! ```
//! use pecos_neo::noise::prelude::*;
//!
//! // 1D chain: qubit i has neighbors i-1 and i+1
//! let crosstalk = CompositeCrosstalkChannel::new("chain", prob(0.01, pauli()))
//!     .responds_to_measurement()
//!     .local(chain_neighbors);
//!
//! // 2D grid (5 columns): up/down/left/right neighbors
//! let crosstalk = CompositeCrosstalkChannel::new("grid", prob(0.01, pauli()))
//!     .responds_to_measurement()
//!     .local(grid_neighbors(5));
//! ```
//!
//! ## Grid Topology
//!
//! For a grid with `cols` columns, qubit positions are:
//!
//! ```text
//! cols = 4:
//!   0  1  2  3
//!   4  5  6  7
//!   8  9 10 11
//! ```
//!
//! Qubit 5 has neighbors: 4 (left), 6 (right), 1 (up), 9 (down).
//!
//! # Distance Functions
//!
//! Calculate separation between qubits for distance-weighted correlations:
//!
//! ```
//! use pecos_neo::noise::prelude::*;
//! use pecos_core::QubitId;
//!
//! // 1D chain distance: |i - j|
//! let d = chain_distance(QubitId(0), QubitId(5));  // d = 5.0
//!
//! // 2D grid Manhattan distance
//! let dist_fn = grid_distance(4);  // 4 columns
//! let d = dist_fn(QubitId(0), QubitId(5));  // d = 2.0 (1 right + 1 down)
//! ```
//!
//! # Decay Functions
//!
//! Model how correlations decrease with distance:
//!
//! ```
//! use pecos_neo::noise::prelude::*;
//!
//! // Exponential: corr * exp(-distance / decay_length)
//! let decay = exponential_decay(0.5, 2.0);
//! assert!((decay(0.0) - 0.5).abs() < 1e-10);   // At distance 0
//! assert!((decay(2.0) - 0.184).abs() < 0.01); // At distance 2
//!
//! // Gaussian: corr * exp(-(distance/width)^2)
//! let decay = gaussian_decay(1.0, 3.0);
//!
//! // Power law: corr / (1 + distance)^exponent
//! let decay = power_law_decay(1.0, 2.0);  // 1/r^2 decay
//! ```
//!
//! # Custom Connectivity
//!
//! For non-standard topologies (e.g., heavy-hex), use custom connectivity:
//!
//! ```
//! use pecos_neo::noise::topology::hex_neighbors_from_connectivity;
//!
//! // Define edges as (qubit_a, qubit_b) pairs
//! let connectivity = [
//!     (0, 1), (1, 2), (1, 3), (2, 4), (3, 4),
//! ];
//! let neighbors_fn = hex_neighbors_from_connectivity(&connectivity);
//! ```

use pecos_core::QubitId;

/// Neighbor function for a 1D chain topology.
///
/// Qubit `i` has neighbors `i-1` (if `i > 0`) and `i+1`.
/// Suitable for linear qubit arrays.
#[must_use]
pub fn chain_neighbors(gated: &[QubitId]) -> Vec<QubitId> {
    let mut neighbors = Vec::with_capacity(gated.len() * 2);
    for &QubitId(q) in gated {
        if q > 0 {
            neighbors.push(QubitId(q - 1));
        }
        neighbors.push(QubitId(q + 1));
    }
    neighbors
}

/// Create a neighbor function for a 2D grid topology with the given number of columns.
///
/// Qubit at position (row, col) has ID = row * cols + col.
/// Neighbors are: left (col-1), right (col+1), up (row-1), down (row+1).
///
/// # Arguments
///
/// * `cols` - Number of columns in the grid
///
/// # Example
///
/// ```
/// # use pecos_neo::noise::topology::grid_neighbors;
/// // 5x4 grid (5 columns, 4 rows)
/// let neighbors_fn = grid_neighbors(5);
/// ```
#[must_use]
pub fn grid_neighbors(cols: usize) -> fn(&[QubitId]) -> Vec<QubitId> {
    // Return a closure capturing cols
    // Since we need a fn pointer, we use a match on common grid sizes
    // For arbitrary sizes, users can create their own closure
    match cols {
        1 => grid_neighbors_1,
        2 => grid_neighbors_2,
        3 => grid_neighbors_3,
        4 => grid_neighbors_4,
        5 => grid_neighbors_5,
        6 => grid_neighbors_6,
        7 => grid_neighbors_7,
        8 => grid_neighbors_8,
        9 => grid_neighbors_9,
        10 => grid_neighbors_10,
        _ => chain_neighbors, // Fallback: treat as 1D chain
    }
}

// Generated grid neighbor functions for common sizes
fn grid_neighbors_impl(gated: &[QubitId], cols: usize) -> Vec<QubitId> {
    let mut neighbors = Vec::with_capacity(gated.len() * 4);
    for &QubitId(q) in gated {
        let col = q % cols;
        // Left neighbor
        if col > 0 {
            neighbors.push(QubitId(q - 1));
        }
        // Right neighbor
        if col + 1 < cols {
            neighbors.push(QubitId(q + 1));
        }
        // Up neighbor (previous row)
        if q >= cols {
            neighbors.push(QubitId(q - cols));
        }
        // Down neighbor (next row)
        neighbors.push(QubitId(q + cols));
    }
    neighbors
}

fn grid_neighbors_1(gated: &[QubitId]) -> Vec<QubitId> {
    grid_neighbors_impl(gated, 1)
}
fn grid_neighbors_2(gated: &[QubitId]) -> Vec<QubitId> {
    grid_neighbors_impl(gated, 2)
}
fn grid_neighbors_3(gated: &[QubitId]) -> Vec<QubitId> {
    grid_neighbors_impl(gated, 3)
}
fn grid_neighbors_4(gated: &[QubitId]) -> Vec<QubitId> {
    grid_neighbors_impl(gated, 4)
}
fn grid_neighbors_5(gated: &[QubitId]) -> Vec<QubitId> {
    grid_neighbors_impl(gated, 5)
}
fn grid_neighbors_6(gated: &[QubitId]) -> Vec<QubitId> {
    grid_neighbors_impl(gated, 6)
}
fn grid_neighbors_7(gated: &[QubitId]) -> Vec<QubitId> {
    grid_neighbors_impl(gated, 7)
}
fn grid_neighbors_8(gated: &[QubitId]) -> Vec<QubitId> {
    grid_neighbors_impl(gated, 8)
}
fn grid_neighbors_9(gated: &[QubitId]) -> Vec<QubitId> {
    grid_neighbors_impl(gated, 9)
}
fn grid_neighbors_10(gated: &[QubitId]) -> Vec<QubitId> {
    grid_neighbors_impl(gated, 10)
}

/// Create a neighbor function for a heavy-hex topology.
///
/// Heavy-hex is used by IBM quantum processors. Each data qubit has 2 or 3 neighbors.
/// This simplified version uses a connectivity list.
///
/// # Arguments
///
/// * `connectivity` - A list of (qubit, neighbor) pairs defining the connections
///
/// # Example
///
/// ```
/// # use pecos_neo::noise::topology::hex_neighbors_from_connectivity;
/// let hex_neighbors = hex_neighbors_from_connectivity(&[
///     (0, 1), (1, 2), (1, 3), // etc.
/// ]);
/// ```
pub fn hex_neighbors_from_connectivity(
    connectivity: &[(usize, usize)],
) -> impl Fn(&[QubitId]) -> Vec<QubitId> + Send + Sync + 'static {
    // Build adjacency list
    let max_qubit = connectivity
        .iter()
        .flat_map(|&(a, b)| [a, b])
        .max()
        .unwrap_or(0);

    let mut adjacency = vec![Vec::new(); max_qubit + 1];
    for &(a, b) in connectivity {
        adjacency[a].push(QubitId(b));
        adjacency[b].push(QubitId(a));
    }

    move |gated: &[QubitId]| {
        let mut neighbors = Vec::new();
        for &QubitId(q) in gated {
            if q < adjacency.len() {
                neighbors.extend_from_slice(&adjacency[q]);
            }
        }
        neighbors
    }
}

/// Distance function type for distance-weighted correlations.
pub type DistanceFn = fn(QubitId, QubitId) -> f64;

/// Manhattan distance on a 1D chain.
///
/// Distance = |i - j|
#[must_use]
#[allow(clippy::cast_precision_loss)] // distance value
pub fn chain_distance(a: QubitId, b: QubitId) -> f64 {
    #[allow(clippy::cast_possible_wrap)] // qubit indices fit in i64
    {
        (a.0 as i64 - b.0 as i64).unsigned_abs() as f64
    }
}

/// Create a Manhattan distance function for a 2D grid.
///
/// Distance = `|row_a - row_b| + |col_a - col_b|`
#[allow(clippy::cast_precision_loss)] // distance value
pub fn grid_distance(cols: usize) -> impl Fn(QubitId, QubitId) -> f64 + Send + Sync + 'static {
    move |a: QubitId, b: QubitId| {
        let row_a = a.0 / cols;
        let col_a = a.0 % cols;
        let row_b = b.0 / cols;
        let col_b = b.0 % cols;
        (row_a.abs_diff(row_b) + col_a.abs_diff(col_b)) as f64
    }
}

/// Exponential decay correlation based on distance.
///
/// Returns: `base_correlation * exp(-distance / decay_length)`
///
/// # Arguments
///
/// * `base_correlation` - Correlation factor for adjacent qubits (distance = 1)
/// * `decay_length` - Characteristic decay length
pub fn exponential_decay(base_correlation: f64, decay_length: f64) -> impl Fn(f64) -> f64 {
    move |distance: f64| base_correlation * (-distance / decay_length).exp()
}

/// Gaussian decay correlation based on distance.
///
/// Returns: `base_correlation * exp(-(distance/width)^2)`
///
/// # Arguments
///
/// * `base_correlation` - Correlation factor for same-site (distance = 0)
/// * `width` - Gaussian width parameter
pub fn gaussian_decay(base_correlation: f64, width: f64) -> impl Fn(f64) -> f64 {
    move |distance: f64| base_correlation * (-(distance / width).powi(2)).exp()
}

/// Power-law decay correlation based on distance.
///
/// Returns: `base_correlation / (1 + distance)^exponent`
///
/// # Arguments
///
/// * `base_correlation` - Correlation factor for same-site (distance = 0)
/// * `exponent` - Power-law exponent
pub fn power_law_decay(base_correlation: f64, exponent: f64) -> impl Fn(f64) -> f64 {
    move |distance: f64| base_correlation / (1.0 + distance).powf(exponent)
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_chain_neighbors() {
        // Qubit 0: neighbor is 1
        let n = chain_neighbors(&[QubitId(0)]);
        assert_eq!(n, vec![QubitId(1)]);

        // Qubit 5: neighbors are 4 and 6
        let n = chain_neighbors(&[QubitId(5)]);
        assert_eq!(n, vec![QubitId(4), QubitId(6)]);

        // Multiple qubits
        let n = chain_neighbors(&[QubitId(2), QubitId(4)]);
        assert_eq!(n, vec![QubitId(1), QubitId(3), QubitId(3), QubitId(5)]);
    }

    #[test]
    fn test_grid_neighbors() {
        // 3x3 grid (cols=3)
        // 0 1 2
        // 3 4 5
        // 6 7 8
        let neighbors_fn = grid_neighbors(3);

        // Corner (0): neighbors are 1 (right), 3 (down)
        let n = neighbors_fn(&[QubitId(0)]);
        assert!(n.contains(&QubitId(1)));
        assert!(n.contains(&QubitId(3)));
        assert_eq!(n.len(), 2);

        // Center (4): neighbors are 3, 5, 1, 7
        let n = neighbors_fn(&[QubitId(4)]);
        assert!(n.contains(&QubitId(3))); // left
        assert!(n.contains(&QubitId(5))); // right
        assert!(n.contains(&QubitId(1))); // up
        assert!(n.contains(&QubitId(7))); // down
        assert_eq!(n.len(), 4);

        // Edge (1): neighbors are 0, 2, 4
        let n = neighbors_fn(&[QubitId(1)]);
        assert!(n.contains(&QubitId(0))); // left
        assert!(n.contains(&QubitId(2))); // right
        assert!(n.contains(&QubitId(4))); // down
        assert_eq!(n.len(), 3);
    }

    #[test]
    fn test_chain_distance() {
        assert_eq!(chain_distance(QubitId(0), QubitId(0)), 0.0);
        assert_eq!(chain_distance(QubitId(0), QubitId(5)), 5.0);
        assert_eq!(chain_distance(QubitId(5), QubitId(0)), 5.0);
        assert_eq!(chain_distance(QubitId(3), QubitId(7)), 4.0);
    }

    #[test]
    fn test_grid_distance() {
        // 3x3 grid
        let dist = grid_distance(3);

        // Same position
        assert_eq!(dist(QubitId(4), QubitId(4)), 0.0);

        // Adjacent (right)
        assert_eq!(dist(QubitId(4), QubitId(5)), 1.0);

        // Adjacent (down)
        assert_eq!(dist(QubitId(4), QubitId(7)), 1.0);

        // Diagonal
        assert_eq!(dist(QubitId(0), QubitId(4)), 2.0);

        // Corner to corner
        assert_eq!(dist(QubitId(0), QubitId(8)), 4.0);
    }

    #[test]
    fn test_exponential_decay() {
        let decay = exponential_decay(0.5, 2.0);

        // Distance 0 -> 0.5
        let c0 = decay(0.0);
        assert!((c0 - 0.5).abs() < 1e-10);

        // Distance 2 -> 0.5 * e^(-1) ≈ 0.184
        let c2 = decay(2.0);
        assert!((c2 - 0.5 * (-1.0_f64).exp()).abs() < 1e-10);

        // Decays with distance
        assert!(decay(1.0) > decay(2.0));
        assert!(decay(2.0) > decay(5.0));
    }

    #[test]
    fn test_gaussian_decay() {
        let decay = gaussian_decay(1.0, 2.0);

        // Distance 0 -> 1.0
        let c0 = decay(0.0);
        assert!((c0 - 1.0).abs() < 1e-10);

        // Distance 2 -> e^(-1) ≈ 0.368
        let c2 = decay(2.0);
        assert!((c2 - (-1.0_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_power_law_decay() {
        let decay = power_law_decay(1.0, 2.0);

        // Distance 0 -> 1.0
        let c0 = decay(0.0);
        assert!((c0 - 1.0).abs() < 1e-10);

        // Distance 1 -> 1/4 = 0.25
        let c1 = decay(1.0);
        assert!((c1 - 0.25).abs() < 1e-10);

        // Distance 2 -> 1/9 ≈ 0.111
        let c2 = decay(2.0);
        assert!((c2 - 1.0 / 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_hex_neighbors() {
        // Simple hex connectivity
        let connectivity = [(0, 1), (1, 2), (1, 3), (2, 4), (3, 4)];
        let neighbors_fn = hex_neighbors_from_connectivity(&connectivity);

        // Qubit 1 has neighbors 0, 2, 3
        let n = neighbors_fn(&[QubitId(1)]);
        assert!(n.contains(&QubitId(0)));
        assert!(n.contains(&QubitId(2)));
        assert!(n.contains(&QubitId(3)));
        assert_eq!(n.len(), 3);

        // Qubit 0 has only neighbor 1
        let n = neighbors_fn(&[QubitId(0)]);
        assert_eq!(n, vec![QubitId(1)]);
    }
}
