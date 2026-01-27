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

/// A logical operator with information about which logical operations it implements.
#[derive(Clone, Debug)]
pub struct LogicalOperatorInfo {
    /// The Pauli operator.
    pub operator: PauliString,
    /// Weight of the operator.
    pub weight: usize,
    /// Which logical operators this is equivalent to.
    /// Each entry is a (type, index) pair where type is 'X' or 'Z' and index is the logical qubit.
    /// For example, `[('X', 0), ('Z', 1)]` means this operator is equivalent to `X_0` * `Z_1`.
    pub equivalent_logicals: Vec<(char, usize)>,
}

impl LogicalOperatorInfo {
    /// Returns a human-readable string describing the equivalent logical operators.
    ///
    /// For example: "X0", "Z1", "X0*Z1", etc.
    #[must_use]
    pub fn equivalence_string(&self) -> String {
        if self.equivalent_logicals.is_empty() {
            return "I".to_string();
        }
        self.equivalent_logicals
            .iter()
            .map(|(t, i)| format!("{t}{i}"))
            .collect::<Vec<_>>()
            .join("*")
    }
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
///
/// In CSS mode (`css_only=true`), only generates pure X errors (XXXX...) and
/// pure Z errors (ZZZZ...), not mixed XZ errors. This matches the Python
/// `gen_errors` behavior and is faster for CSS codes.
pub struct WeightedPauliIterator {
    num_qubits: usize,
    weight: usize,
    /// Current combination of qubit positions
    positions: Vec<usize>,
    /// Current Pauli assignment (0=X, 1=Y, 2=Z for general; 0=X, 1=Z for CSS)
    paulis: Vec<usize>,
    /// Whether we've exhausted all combinations
    done: bool,
    /// Whether to use CSS mode (pure X or pure Z only, no mixed)
    css_only: bool,
    /// In CSS mode: 0 = generating X errors, 1 = generating Z errors
    css_pauli_type: usize,
}

impl WeightedPauliIterator {
    /// Create a new iterator for Pauli operators of the given weight.
    ///
    /// If `css_only` is true, only pure X errors (XXXX...) and pure Z errors (ZZZZ...)
    /// are generated, not mixed XZ errors.
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
                css_pauli_type: 0,
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
            css_pauli_type: 0,
        }
    }

    /// Advance to the next Pauli assignment (only used in non-CSS mode).
    fn next_pauli(&mut self) -> bool {
        if self.css_only {
            // In CSS mode, we don't mix Paulis - handled by css_pauli_type
            return false;
        }

        // Try to increment the Pauli assignment
        for i in (0..self.weight).rev() {
            if self.paulis[i] < 2 {
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
                    // In CSS mode, css_pauli_type determines X or Z
                    if self.css_pauli_type == 0 {
                        Pauli::X
                    } else {
                        Pauli::Z
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
        if self.css_only {
            // In CSS mode: first iterate all X positions, then all Z positions
            if !self.next_combination() {
                if self.css_pauli_type == 0 {
                    // Switch from X to Z
                    self.css_pauli_type = 1;
                    // Reset positions to start
                    self.positions = (0..self.weight).collect();
                } else {
                    // Done with both X and Z
                    self.done = true;
                }
            }
        } else {
            // In general mode: iterate paulis first, then positions
            if !self.next_pauli() && !self.next_combination() {
                self.done = true;
            }
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
    find_min_weight_logicals_with_info(code, config)
        .into_iter()
        .map(|info| info.operator)
        .collect()
}

/// Find all minimum weight logical operators with equivalence information.
///
/// This returns detailed information about each found operator, including which
/// logical operators it's equivalent to (e.g., X0, Z1, X0*Z1, etc.).
///
/// # Example
///
/// ```
/// use pecos_qec::{StabilizerCode, DistanceSearchConfig, find_min_weight_logicals_with_info};
/// use pecos_core::{Pauli, PauliString, QubitId, QuarterPhase};
///
/// fn pauli_string(paulis: &[(Pauli, usize)]) -> PauliString {
///     PauliString::with_phase_and_paulis(
///         QuarterPhase::PlusOne,
///         paulis.iter().map(|&(p, q)| (p, QubitId::new(q))).collect(),
///     )
/// }
///
/// // 3-qubit bit flip code
/// let stab1 = pauli_string(&[(Pauli::Z, 0), (Pauli::Z, 1)]);
/// let stab2 = pauli_string(&[(Pauli::Z, 1), (Pauli::Z, 2)]);
/// let logical_z = pauli_string(&[(Pauli::Z, 0), (Pauli::Z, 1), (Pauli::Z, 2)]);
/// let logical_x = pauli_string(&[(Pauli::X, 0), (Pauli::X, 1), (Pauli::X, 2)]);
///
/// let code = StabilizerCode::new(3, vec![stab1, stab2], vec![logical_z], vec![logical_x]).unwrap();
///
/// let config = DistanceSearchConfig::with_max_weight(2);
/// let logicals = find_min_weight_logicals_with_info(&code, &config);
///
/// // Each found operator has equivalence info
/// for info in &logicals {
///     println!("Found {} with weight {}, equivalent to {}",
///              info.operator, info.weight, info.equivalence_string());
/// }
/// ```
#[must_use]
pub fn find_min_weight_logicals_with_info(
    code: &StabilizerCode,
    config: &DistanceSearchConfig,
) -> Vec<LogicalOperatorInfo> {
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

                // Determine which logical operators this is equivalent to
                let equivalent_logicals = classify_logical_equivalence_indexed(
                    &log_index,
                    code.num_logical_qubits(),
                    &pauli,
                );

                results.push(LogicalOperatorInfo {
                    operator: pauli,
                    weight,
                    equivalent_logicals,
                });
            }
        }
    }

    results
}

/// Classify which logical operators a given Pauli operator is equivalent to.
///
/// Uses precomputed column indices for O(weight) performance instead of
/// O(k * weight) where k is the number of logical qubits.
///
/// Returns a list of (type, index) pairs where type is 'X' or 'Z'.
/// - If the operator anticommutes with logical Z[i], it contains X[i]
/// - If the operator anticommutes with logical X[i], it contains Z[i]
fn classify_logical_equivalence_indexed(
    log_index: &crate::stabilizer_code::LogicalIndex,
    num_logical_qubits: usize,
    pauli: &PauliString,
) -> Vec<(char, usize)> {
    let mut result = Vec::new();

    // The logical index contains [Z_0, Z_1, ..., Z_{k-1}, X_0, X_1, ..., X_{k-1}]
    // So indices 0..k are logical Zs, and k..2k are logical Xs
    let anticommuting = log_index.find_anticommuting(pauli);

    for idx in anticommuting {
        if idx < num_logical_qubits {
            // Anticommutes with logical Z[idx] -> equivalent to X[idx]
            result.push(('X', idx));
        } else {
            // Anticommutes with logical X[idx - k] -> equivalent to Z[idx - k]
            result.push(('Z', idx - num_logical_qubits));
        }
    }

    // Sort for consistent output (X before Z, then by index)
    result.sort_unstable();

    result
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
        // CSS mode should only generate pure X and pure Z errors, not mixed
        let iter = WeightedPauliIterator::new(3, 1, true);
        let paulis: Vec<_> = iter.collect();

        // Weight 1: C(3,1) positions * 2 types (X, Z) = 6 operators
        assert_eq!(paulis.len(), 6);

        // First 3 should be pure X errors, last 3 should be pure Z errors
        for p in &paulis[0..3] {
            // Pure X: has X positions but no Z positions
            assert!(
                !p.x_positions().is_empty() && p.z_positions().is_empty(),
                "Expected pure X error, got {p:?}"
            );
        }
        for p in &paulis[3..6] {
            // Pure Z: has Z positions but no X positions
            assert!(
                p.x_positions().is_empty() && !p.z_positions().is_empty(),
                "Expected pure Z error, got {p:?}"
            );
        }
    }

    #[test]
    fn test_weighted_pauli_iterator_css_weight2() {
        // CSS mode at weight 2 should generate pure XX and pure ZZ, not XZ
        let iter = WeightedPauliIterator::new(4, 2, true);
        let paulis: Vec<_> = iter.collect();

        // Weight 2 on 4 qubits: C(4,2) = 6 positions * 2 types = 12 operators
        assert_eq!(paulis.len(), 12);

        // First 6 should be pure XX errors, last 6 should be pure ZZ errors
        for p in &paulis[0..6] {
            assert!(
                !p.x_positions().is_empty() && p.z_positions().is_empty(),
                "Expected pure X error, got {p:?}"
            );
        }
        for p in &paulis[6..12] {
            assert!(
                p.x_positions().is_empty() && !p.z_positions().is_empty(),
                "Expected pure Z error, got {p:?}"
            );
        }
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

    #[test]
    fn test_logical_equivalence_tracking() {
        // 3-qubit bit flip code
        let stab1 = pauli_string(&[(Pauli::Z, 0), (Pauli::Z, 1)]);
        let stab2 = pauli_string(&[(Pauli::Z, 1), (Pauli::Z, 2)]);
        let logical_z = pauli_string(&[(Pauli::Z, 0), (Pauli::Z, 1), (Pauli::Z, 2)]);
        let logical_x = pauli_string(&[(Pauli::X, 0), (Pauli::X, 1), (Pauli::X, 2)]);

        let code =
            StabilizerCode::new(3, vec![stab1, stab2], vec![logical_z], vec![logical_x]).unwrap();

        let config = DistanceSearchConfig::with_max_weight(2);
        let logicals = find_min_weight_logicals_with_info(&code, &config);

        // Should find single-qubit Z errors (equivalent to Z0)
        // and single-qubit X errors (equivalent to X0)
        assert!(!logicals.is_empty());

        // All weight-1 operators should have exactly one equivalent logical
        for info in &logicals {
            assert_eq!(info.weight, 1);
            assert!(!info.equivalent_logicals.is_empty());

            // Check that the equivalence string is sensible
            let equiv_str = info.equivalence_string();
            assert!(equiv_str == "X0" || equiv_str == "Z0", "Got: {equiv_str}");
        }
    }

    #[test]
    fn test_logical_equivalence_string() {
        let info = LogicalOperatorInfo {
            operator: pauli_string(&[(Pauli::X, 0)]),
            weight: 1,
            equivalent_logicals: vec![('X', 0), ('Z', 1)],
        };
        assert_eq!(info.equivalence_string(), "X0*Z1");

        let info2 = LogicalOperatorInfo {
            operator: pauli_string(&[(Pauli::Z, 0)]),
            weight: 1,
            equivalent_logicals: vec![],
        };
        assert_eq!(info2.equivalence_string(), "I");
    }
}
