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

//! Code distance calculation and minimum weight logical operator search.
//!
//! This module provides algorithms for computing the distance of a stabilizer code
//! by exhaustively searching for minimum weight logical operators.

use crate::StabilizerCode;
use pecos_core::{Pauli, PauliString, QubitId};

/// Result of a distance calculation, including the minimum weight logical operator found.
#[derive(Clone, Debug)]
pub struct DistanceResult {
    /// The code distance (minimum weight of any logical operator).
    pub distance: usize,
    /// A logical operator achieving the minimum weight.
    pub min_weight_operator: PauliString,
}

/// Configuration for distance search.
#[derive(Clone, Debug, Default)]
pub struct DistanceSearchConfig {
    /// Maximum weight to search up to (None for unlimited).
    pub max_weight: Option<usize>,
    /// Whether to search only CSS-type errors (X-only or Z-only).
    pub css_only: bool,
    /// Whether to be verbose during search.
    pub verbose: bool,
}

impl DistanceSearchConfig {
    /// Create a new config that searches up to the given weight.
    #[must_use]
    pub fn with_max_weight(max_weight: usize) -> Self {
        Self {
            max_weight: Some(max_weight),
            ..Default::default()
        }
    }

    /// Create a config for CSS-only search (faster for CSS codes).
    #[must_use]
    pub fn css() -> Self {
        Self {
            css_only: true,
            ..Default::default()
        }
    }
}

/// Generate all Pauli strings of a given weight on a set of qubits.
///
/// This is a helper iterator that generates all possible Pauli operators
/// of exactly the specified weight.
pub struct WeightedPauliIterator {
    num_qubits: usize,
    weight: usize,
    /// Current combination of qubit positions
    positions: Vec<usize>,
    /// Current Pauli assignment (0=X, 1=Y, 2=Z)
    paulis: Vec<usize>,
    /// Whether we've exhausted all combinations
    done: bool,
    /// Whether to use CSS mode (X and Z only)
    css_only: bool,
}

impl WeightedPauliIterator {
    /// Create a new iterator for Pauli operators of the given weight.
    #[must_use]
    pub fn new(num_qubits: usize, weight: usize, css_only: bool) -> Self {
        if weight == 0 || weight > num_qubits {
            return Self {
                num_qubits,
                weight,
                positions: vec![],
                paulis: vec![],
                done: true,
                css_only,
            };
        }

        // Initialize with first combination: 0, 1, 2, ..., weight-1
        let positions: Vec<usize> = (0..weight).collect();
        let paulis = vec![0; weight]; // All X initially

        Self {
            num_qubits,
            weight,
            positions,
            paulis,
            done: false,
            css_only,
        }
    }

    /// Advance to the next Pauli assignment.
    fn next_pauli(&mut self) -> bool {
        let max_pauli = if self.css_only { 1 } else { 2 }; // 0,1 for CSS, 0,1,2 for general

        // Try to increment the Pauli assignment
        for i in (0..self.weight).rev() {
            if self.paulis[i] < max_pauli {
                self.paulis[i] += 1;
                // Reset all following positions
                for j in (i + 1)..self.weight {
                    self.paulis[j] = 0;
                }
                return true;
            }
        }
        false
    }

    /// Advance to the next position combination.
    fn next_combination(&mut self) -> bool {
        // Reset Pauli assignments
        for p in &mut self.paulis {
            *p = 0;
        }

        // Find the rightmost position that can be incremented
        let mut i = self.weight;
        while i > 0 {
            i -= 1;
            if self.positions[i] < self.num_qubits - self.weight + i {
                self.positions[i] += 1;
                // Reset all following positions
                for j in (i + 1)..self.weight {
                    self.positions[j] = self.positions[j - 1] + 1;
                }
                return true;
            }
        }
        false
    }

    /// Convert current state to a `PauliString`.
    fn current_pauli_string(&self) -> PauliString {
        let paulis: Vec<(Pauli, QubitId)> = self
            .positions
            .iter()
            .zip(self.paulis.iter())
            .map(|(&pos, &p)| {
                let pauli = if self.css_only {
                    match p {
                        0 => Pauli::X,
                        _ => Pauli::Z,
                    }
                } else {
                    match p {
                        0 => Pauli::X,
                        1 => Pauli::Y,
                        _ => Pauli::Z,
                    }
                };
                (pauli, QubitId::new(pos))
            })
            .collect();

        PauliString::with_phase_and_paulis(pecos_core::QuarterPhase::PlusOne, paulis)
    }
}

impl Iterator for WeightedPauliIterator {
    type Item = PauliString;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let result = self.current_pauli_string();

        // Try to advance to next state
        if !self.next_pauli() && !self.next_combination() {
            self.done = true;
        }

        Some(result)
    }
}

/// Calculate the distance of a stabilizer code.
///
/// The distance is the minimum weight of any logical operator (an operator that
/// commutes with all stabilizers but is not in the stabilizer group).
///
/// # Warning
/// This is an exponential-time algorithm. For codes with many qubits, it may take
/// a very long time to complete. Use `config.max_weight` to limit the search.
#[must_use]
pub fn calculate_distance(
    code: &StabilizerCode,
    config: &DistanceSearchConfig,
) -> Option<DistanceResult> {
    let max_weight = config.max_weight.unwrap_or(code.num_qubits());

    // Build indices once for O(weight) lookups instead of O(num_stabilizers * weight)
    let stab_index = code.build_stabilizer_index();
    let log_index = code.build_logical_index();

    for weight in 1..=max_weight {
        if config.verbose {
            eprintln!("Checking weight {weight}...");
        }

        for pauli in WeightedPauliIterator::new(code.num_qubits(), weight, config.css_only) {
            if code.is_logical_error_indexed(&pauli, &stab_index, &log_index) {
                return Some(DistanceResult {
                    distance: weight,
                    min_weight_operator: pauli,
                });
            }
        }
    }

    None
}

/// Find all minimum weight logical operators.
///
/// Unlike `calculate_distance`, this returns all logical operators of the minimum weight,
/// not just one.
#[must_use]
pub fn find_min_weight_logicals(
    code: &StabilizerCode,
    config: &DistanceSearchConfig,
) -> Vec<PauliString> {
    let max_weight = config.max_weight.unwrap_or(code.num_qubits());
    let mut results = Vec::new();
    let mut found_distance = None;

    // Build indices once for O(weight) lookups instead of O(num_stabilizers * weight)
    let stab_index = code.build_stabilizer_index();
    let log_index = code.build_logical_index();

    for weight in 1..=max_weight {
        // If we've found logical operators and this weight is larger, stop
        if let Some(d) = found_distance
            && weight > d
        {
            break;
        }

        if config.verbose {
            eprintln!("Checking weight {weight}...");
        }

        for pauli in WeightedPauliIterator::new(code.num_qubits(), weight, config.css_only) {
            if code.is_logical_error_indexed(&pauli, &stab_index, &log_index) {
                if found_distance.is_none() {
                    found_distance = Some(weight);
                }
                results.push(pauli);
            }
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::{Pauli, PauliOperator};

    fn pauli_string(paulis: &[(Pauli, usize)]) -> PauliString {
        PauliString::with_phase_and_paulis(
            pecos_core::QuarterPhase::PlusOne,
            paulis.iter().map(|&(p, q)| (p, QubitId::new(q))).collect(),
        )
    }

    #[test]
    fn test_weighted_pauli_iterator_weight_1() {
        let iter = WeightedPauliIterator::new(3, 1, false);
        let paulis: Vec<_> = iter.collect();

        // Should have 3 qubits * 3 Paulis = 9 operators
        assert_eq!(paulis.len(), 9);

        // First few should be X0, Y0, Z0
        assert_eq!(paulis[0].weight(), 1);
    }

    #[test]
    fn test_weighted_pauli_iterator_css() {
        let iter = WeightedPauliIterator::new(3, 1, true);
        let paulis: Vec<_> = iter.collect();

        // Should have 3 qubits * 2 Paulis (X and Z only) = 6 operators
        assert_eq!(paulis.len(), 6);
    }

    #[test]
    fn test_weighted_pauli_iterator_weight_2() {
        let iter = WeightedPauliIterator::new(4, 2, false);
        let paulis: Vec<_> = iter.collect();

        // Should have C(4,2) * 3^2 = 6 * 9 = 54 operators
        assert_eq!(paulis.len(), 54);
    }

    #[test]
    fn test_three_qubit_bit_flip_distance() {
        // 3-qubit bit flip code should have distance 1 for X errors
        // (single X error is a logical error for this code when viewed as protecting against Z)

        let stab1 = pauli_string(&[(Pauli::Z, 0), (Pauli::Z, 1)]);
        let stab2 = pauli_string(&[(Pauli::Z, 1), (Pauli::Z, 2)]);
        let logical_z = pauli_string(&[(Pauli::Z, 0), (Pauli::Z, 1), (Pauli::Z, 2)]);
        let logical_x = pauli_string(&[(Pauli::X, 0), (Pauli::X, 1), (Pauli::X, 2)]);

        let code =
            StabilizerCode::new(3, vec![stab1, stab2], vec![logical_z], vec![logical_x]).unwrap();

        let config = DistanceSearchConfig::default();
        let result = calculate_distance(&code, &config);

        // The minimum weight logical operator for this code is a single Z
        // (Z on any qubit commutes with ZZ stabilizers and anticommutes with XXX)
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.distance, 1);
    }

    #[test]
    fn test_five_qubit_code_distance() {
        // The [[5,1,3]] perfect code
        // Stabilizers: XZZXI, IXZZX, XIXZZ, ZXIXZ
        let stab1 = pauli_string(&[(Pauli::X, 0), (Pauli::Z, 1), (Pauli::Z, 2), (Pauli::X, 3)]);
        let stab2 = pauli_string(&[(Pauli::X, 1), (Pauli::Z, 2), (Pauli::Z, 3), (Pauli::X, 4)]);
        let stab3 = pauli_string(&[(Pauli::X, 0), (Pauli::X, 2), (Pauli::Z, 3), (Pauli::Z, 4)]);
        let stab4 = pauli_string(&[(Pauli::Z, 0), (Pauli::X, 1), (Pauli::X, 3), (Pauli::Z, 4)]);

        // Logical operators for [[5,1,3]]: Z = ZZZZZ, X = XXXXX
        let logical_z = pauli_string(&[
            (Pauli::Z, 0),
            (Pauli::Z, 1),
            (Pauli::Z, 2),
            (Pauli::Z, 3),
            (Pauli::Z, 4),
        ]);
        let logical_x = pauli_string(&[
            (Pauli::X, 0),
            (Pauli::X, 1),
            (Pauli::X, 2),
            (Pauli::X, 3),
            (Pauli::X, 4),
        ]);

        let code = StabilizerCode::new(
            5,
            vec![stab1, stab2, stab3, stab4],
            vec![logical_z],
            vec![logical_x],
        )
        .unwrap();

        // Verify the code is valid
        assert!(code.verify().is_ok());

        let config = DistanceSearchConfig::with_max_weight(3);
        let result = calculate_distance(&code, &config);

        // The [[5,1,3]] code has distance 3
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.distance, 3);
    }
}
