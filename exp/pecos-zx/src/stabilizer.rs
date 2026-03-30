// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Stabilizer <-> ZX calculus connections.
//!
//! Provides conversions between stabilizer states (from `pecos-simulators`)
//! and ZX graph states.

use pecos_core::pauli::Pauli;
use pecos_core::{PauliString, QuarterPhase, QubitId};
use pecos_simulators::{Gens, SparseStab};

use crate::ZxGraph;

/// Create a ZX graph state diagram from an adjacency matrix.
///
/// This is a convenience wrapper around [`crate::graph::from_adjacency_matrix`]
/// for use in the stabilizer context.
#[must_use]
pub fn graph_state_from_adjacency(adj: &[bool], n: usize) -> ZxGraph {
    crate::graph::from_adjacency_matrix(adj, n)
}

/// Convert stabilizer generators to Pauli string representations.
///
/// Reads the `Gens` data structure (row_x, row_z, signs) and produces
/// human-readable `PauliString` representations for each generator.
#[must_use]
pub fn gens_to_pauli_strings(gens: &Gens, num_qubits: usize) -> Vec<PauliString> {
    let mut result = Vec::with_capacity(num_qubits);

    for gen_idx in 0..num_qubits {
        let mut paulis = Vec::new();

        for qubit in 0..num_qubits {
            let has_x = gens.row_x[gen_idx].contains(qubit);
            let has_z = gens.row_z[gen_idx].contains(qubit);

            let pauli = match (has_x, has_z) {
                (false, false) => continue,
                (true, false) => Pauli::X,
                (false, true) => Pauli::Z,
                (true, true) => Pauli::Y,
            };
            paulis.push((pauli, QubitId::new(qubit)));
        }

        let phase = if gens.signs_minus.contains(gen_idx) {
            QuarterPhase::MinusOne
        } else {
            QuarterPhase::PlusOne
        };

        result.push(PauliString::with_phase_and_paulis(phase, paulis));
    }

    result
}

/// Extract the graph state representation from a stabilizer state.
///
/// Returns the adjacency matrix and local Clifford corrections.
///
/// The algorithm:
/// 1. Row-reduce the X-block of the stabilizer tableau over GF(2)
/// 2. Read off the adjacency from the Z-block
///
/// The adjacency matrix is returned as a flat `n x n` `Vec<bool>` in row-major order.
/// Local Cliffords are returned as a description string per qubit.
#[must_use]
pub fn extract_graph_state(stab: &SparseStab) -> (Vec<bool>, Vec<String>) {
    let n = stab.num_qubits();
    let gens = stab.stabs();

    // Build dense X and Z matrices from the sparse representation
    let mut x_matrix = vec![vec![false; n]; n];
    let mut z_matrix = vec![vec![false; n]; n];

    for row in 0..n {
        for col in gens.row_x[row].iter() {
            if col < n {
                x_matrix[row][col] = true;
            }
        }
        for col in gens.row_z[row].iter() {
            if col < n {
                z_matrix[row][col] = true;
            }
        }
    }

    // Row-reduce X matrix over GF(2), applying same operations to Z matrix
    let mut pivot_cols = Vec::new();
    let mut pivot_row = 0;

    for col in 0..n {
        // Find pivot in this column
        let found = x_matrix[pivot_row..n]
            .iter()
            .position(|row_vec| row_vec[col])
            .map(|offset| offset + pivot_row);

        let Some(pivot) = found else { continue };

        // Swap rows
        x_matrix.swap(pivot, pivot_row);
        z_matrix.swap(pivot, pivot_row);

        // Eliminate other rows
        for row in 0..n {
            if row != pivot_row && x_matrix[row][col] {
                for c in 0..n {
                    x_matrix[row][c] ^= x_matrix[pivot_row][c];
                    z_matrix[row][c] ^= z_matrix[pivot_row][c];
                }
            }
        }

        pivot_cols.push(col);
        pivot_row += 1;
    }

    // After row reduction, if X block is identity, Z block gives adjacency
    // (up to local Clifford corrections)
    let mut adjacency = vec![false; n * n];
    let mut local_cliffords = vec![String::new(); n];

    for i in 0..n {
        for j in 0..n {
            if i != j {
                adjacency[i * n + j] = z_matrix[i][j];
            }
        }
        // Check if qubit i needed a local Clifford
        if i < pivot_cols.len() && pivot_cols[i] == i && !x_matrix[i][i] {
            local_cliffords[i] = "H".to_string();
        }
    }

    (adjacency, local_cliffords)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::PauliOperator;
    use pecos_simulators::CliffordGateable;
    use quizx::graph::GraphLike;

    fn qid(q: usize) -> QubitId {
        QubitId::new(q)
    }

    #[test]
    fn test_gens_to_pauli_strings_bell_state() {
        let mut stab = SparseStab::new(2);
        stab.h(&[qid(0)]);
        stab.cx(&[(qid(0), qid(1))]);

        let strings = gens_to_pauli_strings(stab.stabs(), 2);
        assert_eq!(strings.len(), 2);
        for s in &strings {
            assert_eq!(s.weight(), 2);
        }
    }

    #[test]
    fn test_graph_state_from_adjacency() {
        #[rustfmt::skip]
        let adj = vec![
            false, true,  false,
            true,  false, true,
            false, true,  false,
        ];
        let g = graph_state_from_adjacency(&adj, 3);
        assert_eq!(g.inputs().len(), 3);
        assert_eq!(g.outputs().len(), 3);
    }

    #[test]
    fn test_extract_graph_state_trivial() {
        let stab = SparseStab::new(3);
        let (adj, _cliffords) = extract_graph_state(&stab);
        assert_eq!(adj.len(), 9);
        assert!(adj.iter().all(|&x| !x));
    }
}
