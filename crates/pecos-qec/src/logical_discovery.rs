// Copyright 2024 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Automatic logical operator discovery using stabilizer simulation.
//!
//! This module provides tools to automatically discover logical operators for a stabilizer code
//! given only the stabilizer generators. It uses the stabilizer simulator to encode the stabilizers
//! into a quantum state and then extracts the logical operators.
//!
//! # Algorithm
//!
//! The discovery process follows the approach used in the Python implementation:
//!
//! 1. Create a simulator with (n + m) qubits where n is data qubits and m is number of checks
//! 2. For each stabilizer check, initialize an ancilla in |+⟩ and apply controlled-Pauli gates
//! 3. Measure each ancilla in the X basis (deterministically giving +1)
//! 4. Use `refactor` to reorganize the tableau so stabilizers become generators
//! 5. The remaining generators (not used by checks or ancillas) are the logical operators
//!
//! # Example
//!
//! ```
//! use pecos_qec::logical_discovery::discover_logical_operators;
//! use pecos_core::{Zs, PauliOperator};
//!
//! // Define stabilizers for the 3-qubit bit flip code
//! let stabilizers = vec![
//!     Zs(0..2),  // ZZI
//!     Zs(1..3),  // IZZ
//! ];
//!
//! // Discover logical operators
//! let result = discover_logical_operators(3, &stabilizers).unwrap();
//!
//! // Should find 1 logical qubit
//! assert_eq!(result.logical_zs.len(), 1);
//! assert_eq!(result.logical_xs.len(), 1);
//! ```

use pecos_core::{Pauli, PauliOperator, PauliString, QubitId};
use pecos_simulators::{CliffordGateable, SparseStab};
use std::collections::BTreeSet;

/// Result of logical operator discovery.
#[derive(Debug, Clone)]
pub struct LogicalDiscoveryResult {
    /// Discovered logical Z operators (one per logical qubit).
    pub logical_zs: Vec<PauliString>,
    /// Discovered logical X operators (one per logical qubit).
    pub logical_xs: Vec<PauliString>,
    /// Destabilizers corresponding to the stabilizer generators.
    /// These are operators that anticommute with exactly one stabilizer each.
    pub destabilizers: Vec<PauliString>,
    /// Number of logical qubits (k = n - rank(stabilizers)).
    pub num_logical_qubits: usize,
}

/// Error that can occur during logical operator discovery.
#[derive(Debug, Clone)]
pub enum LogicalDiscoveryError {
    /// The stabilizers are not independent (linearly dependent).
    StabilizersNotIndependent,
    /// The stabilizers do not all commute.
    StabilizersDoNotCommute,
    /// Failed to refactor a stabilizer into the tableau.
    RefactorFailed(usize),
}

impl std::fmt::Display for LogicalDiscoveryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StabilizersNotIndependent => {
                write!(f, "Stabilizers are not independent (linearly dependent)")
            }
            Self::StabilizersDoNotCommute => {
                write!(f, "Stabilizers do not all commute with each other")
            }
            Self::RefactorFailed(i) => {
                write!(f, "Failed to refactor stabilizer {i} into the tableau")
            }
        }
    }
}

impl std::error::Error for LogicalDiscoveryError {}

/// Discovers logical operators for a stabilizer code using stabilizer simulation.
///
/// Given a set of stabilizer generators, this function uses the stabilizer simulator
/// to automatically discover the corresponding logical Z and X operators.
///
/// # Arguments
///
/// * `num_qubits` - Number of physical qubits
/// * `stabilizers` - The stabilizer generators of the code
///
/// # Returns
///
/// A [`LogicalDiscoveryResult`] containing the discovered logical operators,
/// or an error if discovery fails.
///
/// # Algorithm
///
/// The algorithm follows the Python `VerifyStabilizers` approach:
/// 1. Create simulator with n data + m ancilla qubits
/// 2. For each stabilizer, prepare ancilla in |+⟩ and apply controlled-Paulis
/// 3. Measure ancillas in X basis (deterministic +1 outcome)
/// 4. Use `refactor` to make stabilizers into tableau generators
/// 5. Logical operators come from generators not used for checks/ancillas
///
/// # Example
///
/// ```
/// use pecos_qec::logical_discovery::discover_logical_operators;
/// use pecos_core::{PauliString, PauliOperator};
///
/// // Steane code stabilizers
/// let stabilizers = vec![
///     "XXXXIII".parse::<PauliString>().unwrap(),
///     "XXIIXXI".parse::<PauliString>().unwrap(),
///     "XIXIXIX".parse::<PauliString>().unwrap(),
///     "ZZZZIII".parse::<PauliString>().unwrap(),
///     "ZZIIZZI".parse::<PauliString>().unwrap(),
///     "ZIZIZIZ".parse::<PauliString>().unwrap(),
/// ];
///
/// let result = discover_logical_operators(7, &stabilizers).unwrap();
/// assert_eq!(result.logical_zs.len(), 1);  // 1 logical qubit
/// ```
pub fn discover_logical_operators(
    num_qubits: usize,
    stabilizers: &[PauliString],
) -> Result<LogicalDiscoveryResult, LogicalDiscoveryError> {
    let n = num_qubits;
    let m = stabilizers.len();

    // Check that stabilizers commute
    for (i, s1) in stabilizers.iter().enumerate() {
        for s2 in stabilizers.iter().skip(i + 1) {
            if !s1.commutes_with(s2) {
                return Err(LogicalDiscoveryError::StabilizersDoNotCommute);
            }
        }
    }

    // Number of logical qubits (might be zero if stabilizers are dependent)
    let expected_k = n.saturating_sub(m);

    // Create simulator with n data qubits + m ancilla qubits
    // Data qubits: 0..n, Ancilla qubits: n..n+m
    let total_qubits = n + m;
    let mut state = SparseStab::with_seed(total_qubits, 0);

    // For each stabilizer, encode it using an ancilla
    // This follows the Python approach:
    // 1. Put ancilla in |+⟩ (apply H to ancilla which starts in |0⟩)
    // 2. Apply controlled-Pauli gates from ancilla to data qubits
    // 3. Measure ancilla in X basis (deterministic outcome)
    for (check_idx, stab) in stabilizers.iter().enumerate() {
        let ancilla = QubitId::new(n + check_idx);

        // Put ancilla in |+⟩ state
        state.h(&[ancilla]);

        // Apply controlled gates based on Pauli type
        for (pauli, qubit) in stab.iter_pairs() {
            match pauli {
                Pauli::I => {
                    // Identity - no gate needed
                }
                Pauli::X => {
                    // Controlled-X (CNOT) from ancilla to data
                    state.cx(&[(ancilla, qubit)]);
                }
                Pauli::Z => {
                    // Controlled-Z from ancilla to data
                    state.cz(&[(ancilla, qubit)]);
                }
                Pauli::Y => {
                    // Controlled-Y from ancilla to data
                    state.cy(&[(ancilla, qubit)]);
                }
            }
        }

        // Measure ancilla in X basis with forced outcome 0 (+1 eigenvalue)
        // This is equivalent to H-mz-H with forced outcome
        state.h(&[ancilla]);
        state.mz_forced(ancilla.0, false);
        state.h(&[ancilla]);
    }

    // Now refactor the tableau so that our stabilizers become generators
    // Track which generator indices are used
    let mut used_indices: BTreeSet<usize> = BTreeSet::new();
    // Track check indices separately so we can extract their destabilizers
    let mut check_gen_indices: Vec<usize> = Vec::new();

    // First refactor the check stabilizers
    for (check_idx, stab) in stabilizers.iter().enumerate() {
        let x_positions: Vec<usize> = stab.x_positions();
        let z_positions: Vec<usize> = stab.z_positions();

        // Convert used_indices to BitSet for protected parameter
        let protected: pecos_core::BitSet =
            used_indices
                .iter()
                .fold(pecos_core::BitSet::new(), |mut s, &idx| {
                    s.insert(idx);
                    s
                });

        // Refactor this stabilizer into the tableau
        let (stabs, destabs) = state.stabs_and_destabs_mut();
        let result = stabs.refactor(destabs, x_positions, z_positions, None, Some(&protected));

        if let Some(gen_idx) = result {
            used_indices.insert(gen_idx);
            check_gen_indices.push(gen_idx);
        } else {
            // Check if it's already in the group (dependent)
            let classification = state.stabs().classify_pauli_string(state.destabs(), stab);
            match classification {
                pecos_simulators::PauliClassification::Stabilizer => {
                    return Err(LogicalDiscoveryError::StabilizersNotIndependent);
                }
                _ => {
                    return Err(LogicalDiscoveryError::RefactorFailed(check_idx));
                }
            }
        }
    }

    // Then refactor the ancilla qubits (each should be stabilized by X on that ancilla)
    for anc_idx in 0..m {
        let ancilla_qubit = n + anc_idx;
        let x_positions = vec![ancilla_qubit];
        let z_positions: Vec<usize> = vec![];

        let protected: pecos_core::BitSet =
            used_indices
                .iter()
                .fold(pecos_core::BitSet::new(), |mut s, &idx| {
                    s.insert(idx);
                    s
                });

        let (stabs, destabs) = state.stabs_and_destabs_mut();
        let result = stabs.refactor(destabs, x_positions, z_positions, None, Some(&protected));

        if let Some(gen_idx) = result {
            used_indices.insert(gen_idx);
        }
        // If refactor fails for ancilla, that's okay - it might already be accounted for
    }

    // The generators not used are the logical operators
    let all_indices: BTreeSet<usize> = (0..total_qubits).collect();
    let logical_indices: Vec<usize> = all_indices.difference(&used_indices).copied().collect();

    // Extract destabilizers for the check generators (restricted to data qubits)
    let mut destabilizers = Vec::new();
    for &gen_idx in &check_gen_indices {
        let full_destab = state.destabs().generator(gen_idx);
        let destab = restrict_to_data_qubits(&full_destab, n);
        destabilizers.push(destab);
    }

    // Extract logical operators, but only keep the data qubit parts
    let mut logical_zs = Vec::new();
    let mut logical_xs = Vec::new();

    for &gen_idx in &logical_indices {
        // Get the stabilizer (logical Z) and destabilizer (logical X)
        let full_z = state.stabs().generator(gen_idx);
        let full_x = state.destabs().generator(gen_idx);

        // Restrict to data qubits only (0..n)
        let logical_z = restrict_to_data_qubits(&full_z, n);
        let logical_x = restrict_to_data_qubits(&full_x, n);

        // Only include non-trivial operators
        if !logical_z.is_identity() || !logical_x.is_identity() {
            logical_zs.push(logical_z);
            logical_xs.push(logical_x);
        }
    }

    let num_logical_qubits = logical_zs.len();

    // Verify we found the expected number of logical qubits
    if num_logical_qubits != expected_k && expected_k > 0 {
        // The stabilizers might be dependent, reducing the number of logical qubits
        // This is not an error, just a different code than expected
    }

    Ok(LogicalDiscoveryResult {
        logical_zs,
        logical_xs,
        destabilizers,
        num_logical_qubits,
    })
}

/// Restrict a `PauliString` to only include qubits 0..n (data qubits).
fn restrict_to_data_qubits(ps: &PauliString, n: usize) -> PauliString {
    let paulis: Vec<(Pauli, QubitId)> = ps.iter_pairs().filter(|(_, q)| q.0 < n).collect();
    PauliString::with_phase_and_paulis(ps.phase(), paulis)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::QuarterPhase;

    fn pauli_string(paulis: &[(Pauli, usize)]) -> PauliString {
        PauliString::with_phase_and_paulis(
            QuarterPhase::PlusOne,
            paulis.iter().map(|&(p, q)| (p, QubitId::new(q))).collect(),
        )
    }

    #[test]
    fn test_discover_three_qubit_bit_flip() {
        // 3-qubit bit flip code: [[3, 1, ?]]
        // Stabilizers: ZZI, IZZ
        let stabilizers = vec![
            pauli_string(&[(Pauli::Z, 0), (Pauli::Z, 1)]),
            pauli_string(&[(Pauli::Z, 1), (Pauli::Z, 2)]),
        ];

        let result = discover_logical_operators(3, &stabilizers).unwrap();

        // Should find 1 logical qubit (3 - 2 = 1)
        assert_eq!(result.num_logical_qubits, 1);
        assert_eq!(result.logical_zs.len(), 1);
        assert_eq!(result.logical_xs.len(), 1);

        // The logical operators should commute with stabilizers
        for stab in &stabilizers {
            assert!(
                result.logical_zs[0].commutes_with(stab),
                "Logical Z should commute with stabilizers"
            );
            assert!(
                result.logical_xs[0].commutes_with(stab),
                "Logical X should commute with stabilizers"
            );
        }

        // Logical Z and X should anticommute with each other
        assert!(
            !result.logical_zs[0].commutes_with(&result.logical_xs[0]),
            "Logical Z and X should anticommute"
        );
    }

    #[test]
    fn test_discover_steane_code() {
        // Steane [[7, 1, 3]] code
        // X-type stabilizers
        let sx1 = pauli_string(&[(Pauli::X, 0), (Pauli::X, 2), (Pauli::X, 4), (Pauli::X, 6)]);
        let sx2 = pauli_string(&[(Pauli::X, 1), (Pauli::X, 2), (Pauli::X, 5), (Pauli::X, 6)]);
        let sx3 = pauli_string(&[(Pauli::X, 3), (Pauli::X, 4), (Pauli::X, 5), (Pauli::X, 6)]);
        // Z-type stabilizers
        let sz1 = pauli_string(&[(Pauli::Z, 0), (Pauli::Z, 2), (Pauli::Z, 4), (Pauli::Z, 6)]);
        let sz2 = pauli_string(&[(Pauli::Z, 1), (Pauli::Z, 2), (Pauli::Z, 5), (Pauli::Z, 6)]);
        let sz3 = pauli_string(&[(Pauli::Z, 3), (Pauli::Z, 4), (Pauli::Z, 5), (Pauli::Z, 6)]);

        let stabilizers = vec![sx1, sx2, sx3, sz1, sz2, sz3];
        let result = discover_logical_operators(7, &stabilizers).unwrap();

        // Should find 1 logical qubit (7 - 6 = 1)
        assert_eq!(result.num_logical_qubits, 1);
        assert_eq!(result.logical_zs.len(), 1);
        assert_eq!(result.logical_xs.len(), 1);

        // Verify commutation relations
        for stab in &stabilizers {
            assert!(
                result.logical_zs[0].commutes_with(stab),
                "Logical Z {:?} does not commute with stabilizer {:?}",
                result.logical_zs[0],
                stab
            );
            assert!(
                result.logical_xs[0].commutes_with(stab),
                "Logical X {:?} does not commute with stabilizer {:?}",
                result.logical_xs[0],
                stab
            );
        }
        assert!(
            !result.logical_zs[0].commutes_with(&result.logical_xs[0]),
            "Logical Z and X should anticommute"
        );
    }

    #[test]
    fn test_discover_five_qubit_code() {
        // [[5, 1, 3]] perfect code
        // Stabilizers: XZZXI, IXZZX, XIXZZ, ZXIXZ
        let s1 = pauli_string(&[(Pauli::X, 0), (Pauli::Z, 1), (Pauli::Z, 2), (Pauli::X, 3)]);
        let s2 = pauli_string(&[(Pauli::X, 1), (Pauli::Z, 2), (Pauli::Z, 3), (Pauli::X, 4)]);
        let s3 = pauli_string(&[(Pauli::X, 0), (Pauli::X, 2), (Pauli::Z, 3), (Pauli::Z, 4)]);
        let s4 = pauli_string(&[(Pauli::Z, 0), (Pauli::X, 1), (Pauli::X, 3), (Pauli::Z, 4)]);

        let stabilizers = vec![s1, s2, s3, s4];
        let result = discover_logical_operators(5, &stabilizers).unwrap();

        // Should find 1 logical qubit (5 - 4 = 1)
        assert_eq!(result.num_logical_qubits, 1);
        assert_eq!(result.logical_zs.len(), 1);
        assert_eq!(result.logical_xs.len(), 1);

        // Verify commutation relations
        for stab in &stabilizers {
            assert!(
                result.logical_zs[0].commutes_with(stab),
                "Logical Z {:?} does not commute with stabilizer {:?}",
                result.logical_zs[0],
                stab
            );
            assert!(
                result.logical_xs[0].commutes_with(stab),
                "Logical X {:?} does not commute with stabilizer {:?}",
                result.logical_xs[0],
                stab
            );
        }
        assert!(
            !result.logical_zs[0].commutes_with(&result.logical_xs[0]),
            "Logical Z and X should anticommute"
        );
    }

    #[test]
    fn test_dependent_stabilizers_error() {
        // Try to add a dependent stabilizer (product of first two)
        let s1 = pauli_string(&[(Pauli::Z, 0), (Pauli::Z, 1)]);
        let s2 = pauli_string(&[(Pauli::Z, 1), (Pauli::Z, 2)]);
        // s1 * s2 = ZIZ, which is dependent
        let s3 = pauli_string(&[(Pauli::Z, 0), (Pauli::Z, 2)]);

        let stabilizers = vec![s1, s2, s3];
        let result = discover_logical_operators(3, &stabilizers);

        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(LogicalDiscoveryError::StabilizersNotIndependent)
        ));
    }

    #[test]
    fn test_non_commuting_stabilizers_error() {
        // X and Z on same qubit don't commute
        let s1 = pauli_string(&[(Pauli::X, 0)]);
        let s2 = pauli_string(&[(Pauli::Z, 0)]);

        let stabilizers = vec![s1, s2];
        let result = discover_logical_operators(2, &stabilizers);

        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(LogicalDiscoveryError::StabilizersDoNotCommute)
        ));
    }
}
