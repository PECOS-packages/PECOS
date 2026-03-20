// Copyright 2026 The PECOS Developers
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

//! Fault tolerance checking based on stabilizer flipping.
//!
//! This module provides fault tolerance analysis that works at the stabilizer level
//! rather than measurement level. This is more fundamental and handles dynamic circuits
//! where stabilizers may not always be measured.
//!
//! # Key Insight
//!
//! An error E flips stabilizer S if and only if they anti-commute: {E, S} = -ES.
//! This is a purely algebraic property that doesn't depend on circuit structure.
//!
//! # Advantages over Measurement-Based Analysis
//!
//! 1. **Dynamic circuits**: Works even when stabilizers are conditionally measured
//! 2. **Fundamental**: Separates code properties from circuit structure
//! 3. **Efficient**: Anti-commutation check is O(weight) per stabilizer
//!
//! # Usage
//!
//! ```
//! use pecos_qec::{StabilizerCodeSpec, StabilizerFlipChecker, ErrorClass};
//! use pecos_core::{Xs, Zs, PauliString, QuarterPhase};
//!
//! let code = StabilizerCodeSpec::builder(3)
//!     .check(Zs([0, 1]))
//!     .check(Zs([1, 2]))
//!     .logical_z(Zs([0, 1, 2]))
//!     .logical_x(Xs([0]))
//!     .build()
//!     .unwrap();
//!
//! let checker = StabilizerFlipChecker::new(&code);
//!
//! // Check if a specific error is detectable
//! let x_error = PauliString::from_decomposed(QuarterPhase::PlusOne, [0], [], []);
//! let result = checker.classify_error(&x_error);
//! assert!(matches!(result, ErrorClass::DetectableLogical { .. }));
//! ```

use crate::StabilizerCodeSpec;
use pecos_core::{PauliOperator, PauliString, QuarterPhase};
use std::collections::{BTreeMap, BTreeSet};

/// Result of analyzing which stabilizers/logicals are flipped by an error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StabilizerFlips {
    /// Indices of stabilizers that are flipped (anti-commute with the error).
    pub stabilizers: BTreeSet<usize>,
    /// Indices of logical Z operators that are flipped.
    pub logical_zs: BTreeSet<usize>,
    /// Indices of logical X operators that are flipped.
    pub logical_xs: BTreeSet<usize>,
}

impl StabilizerFlips {
    /// Creates an empty flip result (identity error).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            stabilizers: BTreeSet::new(),
            logical_zs: BTreeSet::new(),
            logical_xs: BTreeSet::new(),
        }
    }

    /// Returns true if no stabilizers are flipped.
    #[must_use]
    pub fn is_undetectable(&self) -> bool {
        self.stabilizers.is_empty()
    }

    /// Returns true if any logical operator is flipped.
    #[must_use]
    pub fn has_logical_error(&self) -> bool {
        !self.logical_zs.is_empty() || !self.logical_xs.is_empty()
    }

    /// Returns the syndrome as a vector of bits.
    #[must_use]
    pub fn syndrome(&self, num_stabilizers: usize) -> Vec<bool> {
        (0..num_stabilizers)
            .map(|i| self.stabilizers.contains(&i))
            .collect()
    }

    /// Returns a compact representation of flipped stabilizers.
    #[must_use]
    pub fn syndrome_bits(&self) -> u64 {
        let mut bits = 0u64;
        for &idx in &self.stabilizers {
            if idx < 64 {
                bits |= 1 << idx;
            }
        }
        bits
    }
}

/// Classification of an error based on stabilizer flips.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorClass {
    /// Error is equivalent to a stabilizer (flips nothing).
    Stabilizer,

    /// Error is detectable (flips stabilizers) but causes no logical error.
    Detectable {
        /// Which stabilizers are flipped.
        syndrome: BTreeSet<usize>,
    },

    /// Error is undetectable but causes a logical error.
    /// This is a fatal error - no decoder can correct it.
    UndetectableLogical {
        /// Which logical Z operators are flipped.
        logical_zs: BTreeSet<usize>,
        /// Which logical X operators are flipped.
        logical_xs: BTreeSet<usize>,
    },

    /// Error is detectable and causes a logical error.
    /// Whether this is correctable depends on the decoder.
    DetectableLogical {
        /// Which stabilizers are flipped.
        syndrome: BTreeSet<usize>,
        /// Which logical Z operators are flipped.
        logical_zs: BTreeSet<usize>,
        /// Which logical X operators are flipped.
        logical_xs: BTreeSet<usize>,
    },
}

impl ErrorClass {
    /// Returns true if this error class is fatal (undetectable logical error).
    #[must_use]
    pub fn is_fatal(&self) -> bool {
        matches!(self, ErrorClass::UndetectableLogical { .. })
    }

    /// Returns true if this error is detectable.
    #[must_use]
    pub fn is_detectable(&self) -> bool {
        matches!(
            self,
            ErrorClass::Detectable { .. } | ErrorClass::DetectableLogical { .. }
        )
    }

    /// Returns true if this error causes a logical error.
    #[must_use]
    pub fn causes_logical_error(&self) -> bool {
        matches!(
            self,
            ErrorClass::UndetectableLogical { .. } | ErrorClass::DetectableLogical { .. }
        )
    }
}

/// Result of fault tolerance analysis based on stabilizer flips.
#[derive(Debug, Clone)]
pub struct StabilizerFlipAnalysis {
    /// Total number of errors analyzed.
    pub total_errors: usize,

    /// Number of errors equivalent to stabilizers (harmless).
    pub stabilizer_errors: usize,

    /// Number of detectable errors without logical effect.
    pub detectable_no_logical: usize,

    /// Number of undetectable logical errors (fatal).
    pub undetectable_logical: usize,

    /// Number of detectable errors with logical effect.
    pub detectable_with_logical: usize,

    /// Syndrome patterns that have ambiguous corrections.
    /// Maps syndrome -> (`correctable_count`, `uncorrectable_count`).
    pub ambiguous_syndromes: BTreeMap<u64, (usize, usize)>,

    /// Weight of errors analyzed.
    pub weight: usize,
}

impl StabilizerFlipAnalysis {
    /// Returns true if the code is fault-tolerant at this weight.
    ///
    /// A code is fault-tolerant if:
    /// 1. No undetectable logical errors exist
    /// 2. Each syndrome has a unique logical effect (no ambiguity)
    #[must_use]
    pub fn is_fault_tolerant(&self) -> bool {
        self.undetectable_logical == 0 && self.ambiguous_syndromes.is_empty()
    }

    /// Returns the failure modes.
    #[must_use]
    pub fn failures(&self) -> Vec<String> {
        let mut failures = Vec::new();

        if self.undetectable_logical > 0 {
            failures.push(format!(
                "{} undetectable logical errors",
                self.undetectable_logical
            ));
        }

        if !self.ambiguous_syndromes.is_empty() {
            failures.push(format!(
                "{} ambiguous syndromes",
                self.ambiguous_syndromes.len()
            ));
        }

        failures
    }
}

/// Column-based index for efficient anti-commutation checking.
///
/// For each qubit, tracks which operators have X or Z on that qubit.
#[derive(Debug, Clone)]
struct ColumnIndex {
    /// For each qubit, the set of operator indices that have X on that qubit.
    col_x: Vec<BTreeSet<usize>>,
    /// For each qubit, the set of operator indices that have Z on that qubit.
    col_z: Vec<BTreeSet<usize>>,
}

impl ColumnIndex {
    /// Build a column index from a list of Pauli operators.
    fn from_paulis(num_qubits: usize, operators: &[PauliString]) -> Self {
        let mut col_x: Vec<BTreeSet<usize>> = (0..num_qubits).map(|_| BTreeSet::new()).collect();
        let mut col_z: Vec<BTreeSet<usize>> = (0..num_qubits).map(|_| BTreeSet::new()).collect();

        for (op_idx, op) in operators.iter().enumerate() {
            for q in op.x_positions() {
                if q < num_qubits {
                    col_x[q].insert(op_idx);
                }
            }
            for q in op.z_positions() {
                if q < num_qubits {
                    col_z[q].insert(op_idx);
                }
            }
        }

        Self { col_x, col_z }
    }

    /// Find all operators that anti-commute with the given Pauli.
    fn find_anticommuting(&self, pauli: &PauliString) -> BTreeSet<usize> {
        let mut result = BTreeSet::new();

        // X on qubit q anti-commutes with Z on qubit q
        for q in pauli.x_positions() {
            if q < self.col_z.len() {
                for &idx in &self.col_z[q] {
                    if result.contains(&idx) {
                        result.remove(&idx);
                    } else {
                        result.insert(idx);
                    }
                }
            }
        }

        // Z on qubit q anti-commutes with X on qubit q
        for q in pauli.z_positions() {
            if q < self.col_x.len() {
                for &idx in &self.col_x[q] {
                    if result.contains(&idx) {
                        result.remove(&idx);
                    } else {
                        result.insert(idx);
                    }
                }
            }
        }

        result
    }
}

/// Fault tolerance checker based on stabilizer flipping.
///
/// This analyzer works at the stabilizer level rather than measurement level.
/// An error flips a stabilizer if and only if they anti-commute.
///
/// # Advantages
///
/// - **Dynamic circuits**: Works even when stabilizers are conditionally measured
/// - **Code-centric**: Analysis depends only on code properties, not circuit structure
/// - **Efficient**: O(weight) per stabilizer for anti-commutation check
pub struct StabilizerFlipChecker<'a> {
    /// The stabilizer code definition.
    code: &'a StabilizerCodeSpec,
    /// Column index for efficient stabilizer anti-commutation.
    stab_index: ColumnIndex,
    /// Column index for logical Z operators.
    logical_z_index: ColumnIndex,
    /// Column index for logical X operators.
    logical_x_index: ColumnIndex,
}

impl<'a> StabilizerFlipChecker<'a> {
    /// Creates a new stabilizer flip checker for the given code.
    #[must_use]
    pub fn new(code: &'a StabilizerCodeSpec) -> Self {
        let n = code.num_qubits();
        let stab_index = ColumnIndex::from_paulis(n, code.stabilizers());
        let logical_z_index = ColumnIndex::from_paulis(n, code.logical_zs());
        let logical_x_index = ColumnIndex::from_paulis(n, code.logical_xs());

        Self {
            code,
            stab_index,
            logical_z_index,
            logical_x_index,
        }
    }

    /// Returns the underlying stabilizer code.
    #[must_use]
    pub fn code(&self) -> &StabilizerCodeSpec {
        self.code
    }

    /// Compute which stabilizers and logicals are flipped by an error.
    #[must_use]
    pub fn compute_flips(&self, error: &PauliString) -> StabilizerFlips {
        StabilizerFlips {
            stabilizers: self.stab_index.find_anticommuting(error),
            logical_zs: self.logical_z_index.find_anticommuting(error),
            logical_xs: self.logical_x_index.find_anticommuting(error),
        }
    }

    /// Classify an error based on its stabilizer and logical flips.
    #[must_use]
    pub fn classify_error(&self, error: &PauliString) -> ErrorClass {
        let flips = self.compute_flips(error);

        match (flips.is_undetectable(), flips.has_logical_error()) {
            (true, false) => ErrorClass::Stabilizer,
            (false, false) => ErrorClass::Detectable {
                syndrome: flips.stabilizers,
            },
            (true, true) => ErrorClass::UndetectableLogical {
                logical_zs: flips.logical_zs,
                logical_xs: flips.logical_xs,
            },
            (false, true) => ErrorClass::DetectableLogical {
                syndrome: flips.stabilizers,
                logical_zs: flips.logical_zs,
                logical_xs: flips.logical_xs,
            },
        }
    }

    /// Analyze all weight-t errors for fault tolerance.
    ///
    /// This enumerates all Pauli errors of the given weight and classifies each.
    #[must_use]
    pub fn analyze_weight(&self, weight: usize) -> StabilizerFlipAnalysis {
        self.analyze_weight_with_types(weight, true, true, true)
    }

    /// Analyze weight-t errors with specific Pauli types.
    ///
    /// # Arguments
    /// * `weight` - Weight of errors to analyze
    /// * `include_x` - Include X errors
    /// * `include_y` - Include Y errors
    /// * `include_z` - Include Z errors
    #[must_use]
    pub fn analyze_weight_with_types(
        &self,
        weight: usize,
        include_x: bool,
        include_y: bool,
        include_z: bool,
    ) -> StabilizerFlipAnalysis {
        let mut analysis = StabilizerFlipAnalysis {
            total_errors: 0,
            stabilizer_errors: 0,
            detectable_no_logical: 0,
            undetectable_logical: 0,
            detectable_with_logical: 0,
            ambiguous_syndromes: BTreeMap::new(),
            weight,
        };

        // Build list of Pauli types to include
        let mut pauli_types = Vec::new();
        if include_x {
            pauli_types.push(1u8); // X
        }
        if include_y {
            pauli_types.push(2u8); // Y
        }
        if include_z {
            pauli_types.push(3u8); // Z
        }

        if pauli_types.is_empty() || weight == 0 {
            return analysis;
        }

        let n = self.code.num_qubits();

        // Track syndrome -> (no_logical_count, has_logical_count) for ambiguity detection
        let mut syndrome_outcomes: BTreeMap<u64, (usize, usize)> = BTreeMap::new();

        // Enumerate all weight-t errors
        for positions in combinations(n, weight) {
            // Enumerate all Pauli assignments on these positions
            for paulis in pauli_product(&pauli_types, weight) {
                let error = build_pauli_string(&positions, &paulis);
                let flips = self.compute_flips(&error);
                let syndrome = flips.syndrome_bits();

                analysis.total_errors += 1;

                let has_logical = flips.has_logical_error();

                if flips.is_undetectable() {
                    if has_logical {
                        analysis.undetectable_logical += 1;
                    } else {
                        analysis.stabilizer_errors += 1;
                    }
                } else {
                    // Detectable - track for ambiguity analysis
                    let entry = syndrome_outcomes.entry(syndrome).or_insert((0, 0));
                    if has_logical {
                        analysis.detectable_with_logical += 1;
                        entry.1 += 1;
                    } else {
                        analysis.detectable_no_logical += 1;
                        entry.0 += 1;
                    }
                }
            }
        }

        // Find ambiguous syndromes (both correctable and uncorrectable)
        for (syndrome, (no_logical, has_logical)) in syndrome_outcomes {
            if no_logical > 0 && has_logical > 0 {
                analysis
                    .ambiguous_syndromes
                    .insert(syndrome, (no_logical, has_logical));
            }
        }

        analysis
    }

    /// Quick check if any weight-t error causes an undetectable logical error.
    ///
    /// Returns early on first failure, more efficient than full analysis.
    #[must_use]
    pub fn has_undetectable_logical(&self, weight: usize) -> bool {
        let n = self.code.num_qubits();
        let pauli_types = [1u8, 2, 3]; // X, Y, Z

        for positions in combinations(n, weight) {
            for paulis in pauli_product(&pauli_types, weight) {
                let error = build_pauli_string(&positions, &paulis);
                let flips = self.compute_flips(&error);

                if flips.is_undetectable() && flips.has_logical_error() {
                    return true;
                }
            }
        }

        false
    }

    /// Compute the distance of the code.
    ///
    /// The distance is the minimum weight of an undetectable logical error.
    /// Returns None if no undetectable logical error is found up to `max_weight`.
    #[must_use]
    pub fn compute_distance(&self, max_weight: usize) -> Option<usize> {
        (1..=max_weight).find(|&w| self.has_undetectable_logical(w))
    }
}

// ============================================================================
// Helper functions for enumeration
// ============================================================================

/// Generates all k-combinations of indices from 0..n.
fn combinations(n: usize, k: usize) -> impl Iterator<Item = Vec<usize>> {
    CombinationIterator::new(n, k)
}

struct CombinationIterator {
    n: usize,
    k: usize,
    indices: Vec<usize>,
    done: bool,
}

impl CombinationIterator {
    fn new(n: usize, k: usize) -> Self {
        if k > n || k == 0 {
            return Self {
                n,
                k,
                indices: Vec::new(),
                done: true,
            };
        }

        let indices: Vec<usize> = (0..k).collect();
        Self {
            n,
            k,
            indices,
            done: false,
        }
    }
}

impl Iterator for CombinationIterator {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let result = self.indices.clone();

        // Advance to next combination
        let mut i = self.k;
        while i > 0 {
            i -= 1;
            if self.indices[i] < self.n - self.k + i {
                self.indices[i] += 1;
                for j in (i + 1)..self.k {
                    self.indices[j] = self.indices[j - 1] + 1;
                }
                return Some(result);
            }
        }

        self.done = true;
        Some(result)
    }
}

/// Generates all assignments of Pauli types to k positions.
fn pauli_product(types: &[u8], k: usize) -> impl Iterator<Item = Vec<u8>> + '_ {
    PauliProductIterator::new(types, k)
}

struct PauliProductIterator<'a> {
    types: &'a [u8],
    k: usize,
    indices: Vec<usize>,
    done: bool,
}

impl<'a> PauliProductIterator<'a> {
    fn new(types: &'a [u8], k: usize) -> Self {
        if types.is_empty() || k == 0 {
            return Self {
                types,
                k,
                indices: Vec::new(),
                done: true,
            };
        }

        Self {
            types,
            k,
            indices: vec![0; k],
            done: false,
        }
    }
}

impl Iterator for PauliProductIterator<'_> {
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let result: Vec<u8> = self.indices.iter().map(|&i| self.types[i]).collect();

        // Advance to next assignment
        for i in (0..self.k).rev() {
            self.indices[i] += 1;
            if self.indices[i] < self.types.len() {
                return Some(result);
            }
            self.indices[i] = 0;
        }

        self.done = true;
        Some(result)
    }
}

/// Build a `PauliString` from positions and Pauli types.
fn build_pauli_string(positions: &[usize], paulis: &[u8]) -> PauliString {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut zs = Vec::new();

    for (&pos, &pauli) in positions.iter().zip(paulis.iter()) {
        match pauli {
            1 => xs.push(pos), // X
            2 => ys.push(pos), // Y
            3 => zs.push(pos), // Z
            _ => {}
        }
    }

    PauliString::from_decomposed(QuarterPhase::PlusOne, xs, ys, zs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::{Xs, Zs};

    fn three_qubit_code() -> StabilizerCodeSpec {
        StabilizerCodeSpec::builder(3)
            .check(Zs([0, 1]))
            .check(Zs([1, 2]))
            .logical_z(Zs([0, 1, 2]))
            .logical_x(Xs([0]))
            .build()
            .unwrap()
    }

    /// Helper to create X-only `PauliString`
    fn pauli_x(qubits: &[usize]) -> PauliString {
        PauliString::from_decomposed(QuarterPhase::PlusOne, qubits.iter().copied(), [], [])
    }

    /// Helper to create Z-only `PauliString`
    fn pauli_z(qubits: &[usize]) -> PauliString {
        PauliString::from_decomposed(QuarterPhase::PlusOne, [], [], qubits.iter().copied())
    }

    #[test]
    fn test_stabilizer_flip_single_x() {
        let code = three_qubit_code();
        let checker = StabilizerFlipChecker::new(&code);

        // X on qubit 0 anti-commutes with Z0Z1 (stabilizer 0)
        let x0 = pauli_x(&[0]);
        let flips = checker.compute_flips(&x0);

        assert!(flips.stabilizers.contains(&0));
        assert!(!flips.stabilizers.contains(&1));
        // X0 commutes with Z logical (Z0Z1Z2) - odd overlap with Z
        assert!(flips.has_logical_error()); // Flips logical X
    }

    #[test]
    fn test_stabilizer_flip_z_error() {
        let code = three_qubit_code();
        let checker = StabilizerFlipChecker::new(&code);

        // Z error commutes with Z stabilizers
        let z0 = pauli_z(&[0]);
        let flips = checker.compute_flips(&z0);

        assert!(flips.stabilizers.is_empty());
        // Z0 anti-commutes with X logical (X0)
        assert!(flips.logical_xs.contains(&0));
    }

    #[test]
    fn test_classify_stabilizer_error() {
        let code = three_qubit_code();
        let checker = StabilizerFlipChecker::new(&code);

        // X0X1 is equivalent to stabilizer action on X0
        // Actually, let's check: X0X1 anticommutes with Z1Z2 (has Z on 1)
        // and commutes with Z0Z1 (X on both 0,1, Z on both 0,1 -> even)
        let x01 = pauli_x(&[0, 1]);
        let class = checker.classify_error(&x01);

        // X0X1 should flip stabilizer 1 (Z1Z2) but not stabilizer 0 (Z0Z1)
        match class {
            ErrorClass::Detectable { syndrome } => {
                assert!(syndrome.contains(&1));
            }
            _ => panic!("Expected Detectable, got {class:?}"),
        }
    }

    #[test]
    fn test_undetectable_logical_error() {
        let code = three_qubit_code();
        let checker = StabilizerFlipChecker::new(&code);

        // X0X1X2 commutes with all Z stabilizers but anti-commutes with Z logical
        let x012 = pauli_x(&[0, 1, 2]);
        let class = checker.classify_error(&x012);

        match class {
            ErrorClass::UndetectableLogical { logical_zs, .. } => {
                assert!(logical_zs.contains(&0));
            }
            _ => panic!("Expected UndetectableLogical, got {class:?}"),
        }
    }

    #[test]
    fn test_analyze_weight_1() {
        let code = three_qubit_code();
        let checker = StabilizerFlipChecker::new(&code);

        let analysis = checker.analyze_weight(1);

        println!("Weight-1 analysis (all Pauli types):");
        println!("  Total errors: {}", analysis.total_errors);
        println!("  Stabilizer errors: {}", analysis.stabilizer_errors);
        println!(
            "  Detectable (no logical): {}",
            analysis.detectable_no_logical
        );
        println!("  Undetectable logical: {}", analysis.undetectable_logical);
        println!(
            "  Detectable with logical: {}",
            analysis.detectable_with_logical
        );

        // 3 qubits * 3 Pauli types = 9 weight-1 errors
        assert_eq!(analysis.total_errors, 9);

        // The 3-qubit bit flip code only protects against X errors.
        // Z errors are undetectable and flip the logical X.
        // So there ARE undetectable logical errors at weight 1 (Z errors).
        assert!(analysis.undetectable_logical > 0);
    }

    #[test]
    fn test_analyze_weight_1_x_only() {
        let code = three_qubit_code();
        let checker = StabilizerFlipChecker::new(&code);

        // X-only analysis - the code protects against X errors
        let analysis = checker.analyze_weight_with_types(1, true, false, false);

        println!("Weight-1 X-only analysis:");
        println!("  Total errors: {}", analysis.total_errors);
        println!("  Undetectable logical: {}", analysis.undetectable_logical);

        // 3 qubits * 1 type = 3 errors
        assert_eq!(analysis.total_errors, 3);

        // No undetectable logical X errors at weight 1 for this distance-3 code
        assert_eq!(analysis.undetectable_logical, 0);
    }

    #[test]
    fn test_analyze_weight_3() {
        let code = three_qubit_code();
        let checker = StabilizerFlipChecker::new(&code);

        // X-only analysis at weight 3
        let analysis = checker.analyze_weight_with_types(3, true, false, false);

        println!("Weight-3 X-only analysis:");
        println!("  Total errors: {}", analysis.total_errors);
        println!("  Undetectable logical: {}", analysis.undetectable_logical);
        println!("  Is FT: {}", analysis.is_fault_tolerant());

        // At weight 3, X0X1X2 is an undetectable logical error
        assert!(analysis.undetectable_logical > 0);
        assert!(!analysis.is_fault_tolerant());
    }

    #[test]
    fn test_compute_distance() {
        let code = three_qubit_code();
        let checker = StabilizerFlipChecker::new(&code);

        // The overall distance is 1 (single Z error is undetectable logical)
        // because this code doesn't protect against Z errors.
        let distance = checker.compute_distance(5);
        assert_eq!(distance, Some(1));
    }

    #[test]
    fn test_x_distance() {
        let code = three_qubit_code();
        let checker = StabilizerFlipChecker::new(&code);

        // Check X-distance by looking for undetectable X-type logical errors
        let mut x_distance = None;
        for w in 1..=5 {
            let analysis = checker.analyze_weight_with_types(w, true, false, false);
            if analysis.undetectable_logical > 0 {
                x_distance = Some(w);
                break;
            }
        }
        // X-distance should be 3
        assert_eq!(x_distance, Some(3));
    }

    #[test]
    fn test_combinations_iterator() {
        let combs: Vec<_> = combinations(4, 2).collect();
        assert_eq!(combs.len(), 6); // C(4,2) = 6
        assert!(combs.contains(&vec![0, 1]));
        assert!(combs.contains(&vec![2, 3]));
    }

    #[test]
    fn test_pauli_product_iterator() {
        let types = [1u8, 3]; // X, Z
        let prods: Vec<_> = pauli_product(&types, 2).collect();
        assert_eq!(prods.len(), 4); // 2^2 = 4
        assert!(prods.contains(&vec![1, 1])); // XX
        assert!(prods.contains(&vec![1, 3])); // XZ
        assert!(prods.contains(&vec![3, 1])); // ZX
        assert!(prods.contains(&vec![3, 3])); // ZZ
    }

    #[test]
    fn test_css_mode_x_only() {
        let code = three_qubit_code();
        let checker = StabilizerFlipChecker::new(&code);

        // X-only analysis (for Z-distance)
        let analysis = checker.analyze_weight_with_types(1, true, false, false);

        // 3 qubits * 1 type = 3 errors
        assert_eq!(analysis.total_errors, 3);
    }

    // ========================================================================
    // Non-CSS code tests: 5-qubit [[5,1,3]] code
    // ========================================================================

    /// The [[5,1,3]] perfect code - smallest distance-3 code, non-CSS
    fn five_qubit_code() -> StabilizerCodeSpec {
        use pecos_core::{X, Z};
        // Stabilizers: XZZXI, IXZZX, XIXZZ, ZXIXZ (cyclic)
        // These have both X and Z components so it's non-CSS
        StabilizerCodeSpec::builder(5)
            .check(X(0) & Z(1) & Z(2) & X(3)) // XZZXI
            .check(X(1) & Z(2) & Z(3) & X(4)) // IXZZX
            .check(X(0) & X(2) & Z(3) & Z(4)) // XIXZZ
            .check(Z(0) & X(1) & X(3) & Z(4)) // ZXIXZ
            .logical_x(Xs([0, 1, 2, 3, 4])) // XXXXX
            .logical_z(Zs([0, 1, 2, 3, 4])) // ZZZZZ
            .build()
            .unwrap()
    }

    #[test]
    fn test_five_qubit_code_distance() {
        let code = five_qubit_code();
        let checker = StabilizerFlipChecker::new(&code);

        // The [[5,1,3]] code has distance 3
        let distance = checker.compute_distance(5);
        assert_eq!(distance, Some(3), "5-qubit code distance should be 3");
    }

    #[test]
    fn test_five_qubit_code_weight1_ft() {
        let code = five_qubit_code();
        let checker = StabilizerFlipChecker::new(&code);

        // Weight-1 errors should all be detectable (no undetectable logical)
        let analysis = checker.analyze_weight(1);
        assert_eq!(
            analysis.undetectable_logical, 0,
            "No undetectable logical errors at weight 1"
        );
        assert!(analysis.is_fault_tolerant(), "5-qubit code should be 1-FT");
    }

    #[test]
    fn test_five_qubit_code_weight2_ft() {
        let code = five_qubit_code();
        let checker = StabilizerFlipChecker::new(&code);

        // Weight-2 errors should all be detectable
        let analysis = checker.analyze_weight(2);
        assert_eq!(
            analysis.undetectable_logical, 0,
            "No undetectable logical errors at weight 2"
        );
    }

    #[test]
    fn test_five_qubit_code_weight3_not_ft() {
        let code = five_qubit_code();
        let checker = StabilizerFlipChecker::new(&code);

        // Weight-3 has undetectable logical errors (distance = 3)
        let analysis = checker.analyze_weight(3);
        assert!(
            analysis.undetectable_logical > 0,
            "Weight-3 should have undetectable logical errors"
        );
        assert!(
            !analysis.is_fault_tolerant(),
            "5-qubit code should not be 3-FT"
        );
    }

    // ========================================================================
    // Steane [[7,1,3]] code tests - verifies known theoretical results
    // ========================================================================

    fn steane_code() -> StabilizerCodeSpec {
        StabilizerCodeSpec::builder(7)
            .check(Xs([0, 2, 4, 6]))
            .check(Xs([1, 2, 5, 6]))
            .check(Xs([3, 4, 5, 6]))
            .check(Zs([0, 2, 4, 6]))
            .check(Zs([1, 2, 5, 6]))
            .check(Zs([3, 4, 5, 6]))
            .logical_x(Xs([0, 1, 2, 3, 4, 5, 6]))
            .logical_z(Zs([0, 1, 2, 3, 4, 5, 6]))
            .build()
            .unwrap()
    }

    #[test]
    fn test_steane_code_distance() {
        let code = steane_code();
        let checker = StabilizerFlipChecker::new(&code);

        let distance = checker.compute_distance(5);
        assert_eq!(distance, Some(3), "Steane code distance should be 3");
    }

    #[test]
    fn test_steane_code_parameters() {
        let code = steane_code();
        assert_eq!(code.num_qubits(), 7);
        assert_eq!(code.num_logical_qubits(), 1);
    }

    #[test]
    fn test_steane_code_weight1_ft() {
        let code = steane_code();
        let checker = StabilizerFlipChecker::new(&code);

        let analysis = checker.analyze_weight(1);
        assert!(analysis.is_fault_tolerant(), "Steane code should be 1-FT");
        assert_eq!(analysis.undetectable_logical, 0);
    }

    #[test]
    fn test_steane_code_weight2_ft() {
        let code = steane_code();
        let checker = StabilizerFlipChecker::new(&code);

        let analysis = checker.analyze_weight(2);
        assert!(analysis.is_fault_tolerant(), "Steane code should be 2-FT");
        assert_eq!(analysis.undetectable_logical, 0);
    }

    #[test]
    fn test_steane_code_weight3_not_ft() {
        let code = steane_code();
        let checker = StabilizerFlipChecker::new(&code);

        let analysis = checker.analyze_weight(3);
        assert!(
            !analysis.is_fault_tolerant(),
            "Steane code should NOT be 3-FT"
        );
        assert!(analysis.undetectable_logical > 0);
    }

    // ========================================================================
    // Y error specific tests
    // ========================================================================

    #[test]
    fn test_y_error_classification() {
        let code = three_qubit_code();
        let checker = StabilizerFlipChecker::new(&code);

        // Y = iXZ, so Y on qubit 0 has both X and Z components
        let y0 = PauliString::from_decomposed(QuarterPhase::PlusOne, [], [0], []);
        let flips = checker.compute_flips(&y0);

        // Y0 anti-commutes with Z0Z1 (due to X component)
        assert!(
            flips.stabilizers.contains(&0),
            "Y0 should flip stabilizer 0"
        );

        // Y0 also anti-commutes with X0 logical (due to Z component)
        assert!(flips.logical_xs.contains(&0), "Y0 should flip logical X");
    }

    #[test]
    fn test_y_only_analysis() {
        let code = steane_code();
        let checker = StabilizerFlipChecker::new(&code);

        // Y-only errors at weight 1
        let analysis = checker.analyze_weight_with_types(1, false, true, false);
        assert_eq!(analysis.total_errors, 7); // 7 qubits

        // Y errors are detectable in the Steane code
        assert_eq!(analysis.undetectable_logical, 0);
    }

    #[test]
    fn test_five_qubit_y_errors() {
        let code = five_qubit_code();
        let checker = StabilizerFlipChecker::new(&code);

        // In non-CSS codes, Y errors are handled same as X and Z
        let y_analysis = checker.analyze_weight_with_types(1, false, true, false);
        let x_analysis = checker.analyze_weight_with_types(1, true, false, false);

        // Both should detect all weight-1 errors
        assert_eq!(y_analysis.undetectable_logical, 0);
        assert_eq!(x_analysis.undetectable_logical, 0);
    }

    // ========================================================================
    // Syndrome pattern and ambiguity tests
    // ========================================================================

    #[test]
    fn test_unique_syndromes_steane() {
        let code = steane_code();
        let checker = StabilizerFlipChecker::new(&code);

        // All weight-1 errors should have unique syndromes (no ambiguity)
        let analysis = checker.analyze_weight(1);
        assert!(
            analysis.ambiguous_syndromes.is_empty(),
            "Weight-1 should have no syndrome ambiguity"
        );
    }

    #[test]
    fn test_syndrome_bits_computation() {
        let code = three_qubit_code();
        let checker = StabilizerFlipChecker::new(&code);

        // X0 flips stabilizer 0 only
        let x0 = pauli_x(&[0]);
        let flips = checker.compute_flips(&x0);
        assert_eq!(flips.syndrome_bits(), 0b01);

        // X1 flips both stabilizers
        let x1 = pauli_x(&[1]);
        let flips = checker.compute_flips(&x1);
        assert_eq!(flips.syndrome_bits(), 0b11);

        // X2 flips stabilizer 1 only
        let x2 = pauli_x(&[2]);
        let flips = checker.compute_flips(&x2);
        assert_eq!(flips.syndrome_bits(), 0b10);
    }

    #[test]
    fn test_detectable_logical_category() {
        let code = steane_code();
        let checker = StabilizerFlipChecker::new(&code);

        // Weight-3 errors at the distance boundary
        let analysis = checker.analyze_weight(3);

        // Should have some detectable errors that also cause logical errors
        // (these are caught but might overwhelm decoder)
        println!(
            "Weight-3: detectable_with_logical = {}",
            analysis.detectable_with_logical
        );
    }

    // ========================================================================
    // Exact distance boundary tests
    // ========================================================================

    #[test]
    fn test_boundary_below_distance() {
        let code = steane_code();
        let checker = StabilizerFlipChecker::new(&code);

        // Weight d-1 = 2 should be fully correctable
        let analysis = checker.analyze_weight(2);
        assert!(
            analysis.is_fault_tolerant(),
            "Weight d-1 should be fault tolerant"
        );
    }

    #[test]
    fn test_boundary_at_distance() {
        let code = steane_code();
        let checker = StabilizerFlipChecker::new(&code);

        // Weight d = 3 breaks fault tolerance
        let analysis = checker.analyze_weight(3);
        assert!(
            !analysis.is_fault_tolerant(),
            "Weight d should NOT be fault tolerant"
        );
    }

    #[test]
    fn test_three_qubit_x_distance_boundary() {
        let code = three_qubit_code();
        let checker = StabilizerFlipChecker::new(&code);

        // For X errors only: distance is 3
        let w2 = checker.analyze_weight_with_types(2, true, false, false);
        assert!(w2.is_fault_tolerant(), "X-only weight 2 should be FT");

        let w3 = checker.analyze_weight_with_types(3, true, false, false);
        assert!(!w3.is_fault_tolerant(), "X-only weight 3 should NOT be FT");
    }

    // ========================================================================
    // Error count verification tests
    // ========================================================================

    #[test]
    fn test_error_count_formulas() {
        // Verify our enumeration produces correct number of errors
        let code = steane_code();
        let checker = StabilizerFlipChecker::new(&code);
        let n = 7;

        // Weight-1: n * 3 (X, Y, Z on each qubit)
        let w1 = checker.analyze_weight(1);
        assert_eq!(w1.total_errors, n * 3);

        // Weight-2: C(n,2) * 3^2 = 21 * 9 = 189
        let w2 = checker.analyze_weight(2);
        assert_eq!(w2.total_errors, 21 * 9);
    }

    #[test]
    fn test_error_class_partition() {
        // Verify all errors are classified into exactly one category
        let code = steane_code();
        let checker = StabilizerFlipChecker::new(&code);

        let analysis = checker.analyze_weight(2);
        let sum = analysis.stabilizer_errors
            + analysis.detectable_no_logical
            + analysis.undetectable_logical
            + analysis.detectable_with_logical;

        assert_eq!(
            sum, analysis.total_errors,
            "Error classes should partition all errors"
        );
    }

    // ========================================================================
    // CSS code property verification
    // ========================================================================

    #[test]
    fn test_css_x_z_decoupling() {
        let code = steane_code();
        let checker = StabilizerFlipChecker::new(&code);

        // For CSS codes, X and Z distances should be equal (Steane is symmetric)
        let mut x_distance = None;
        let mut z_distance = None;

        for w in 1..=5 {
            if x_distance.is_none() {
                let analysis = checker.analyze_weight_with_types(w, true, false, false);
                if analysis.undetectable_logical > 0 {
                    x_distance = Some(w);
                }
            }
            if z_distance.is_none() {
                let analysis = checker.analyze_weight_with_types(w, false, false, true);
                if analysis.undetectable_logical > 0 {
                    z_distance = Some(w);
                }
            }
        }

        assert_eq!(x_distance, Some(3));
        assert_eq!(z_distance, Some(3));
        assert_eq!(
            x_distance, z_distance,
            "Steane code should have equal X and Z distances"
        );
    }

    #[test]
    fn test_asymmetric_css_code() {
        // 3-qubit repetition code has asymmetric X/Z distances
        let code = three_qubit_code();
        let checker = StabilizerFlipChecker::new(&code);

        // X-distance = 3 (protects against bit flips)
        let mut x_distance = None;
        for w in 1..=5 {
            let analysis = checker.analyze_weight_with_types(w, true, false, false);
            if analysis.undetectable_logical > 0 {
                x_distance = Some(w);
                break;
            }
        }

        // Z-distance = 1 (no protection against phase flips)
        let mut z_distance = None;
        for w in 1..=5 {
            let analysis = checker.analyze_weight_with_types(w, false, false, true);
            if analysis.undetectable_logical > 0 {
                z_distance = Some(w);
                break;
            }
        }

        assert_eq!(x_distance, Some(3), "X-distance should be 3");
        assert_eq!(z_distance, Some(1), "Z-distance should be 1");
    }

    // ========================================================================
    // Edge cases
    // ========================================================================

    #[test]
    fn test_weight_zero() {
        let code = steane_code();
        let checker = StabilizerFlipChecker::new(&code);

        let analysis = checker.analyze_weight(0);
        assert_eq!(analysis.total_errors, 0);
        assert!(analysis.is_fault_tolerant());
    }

    #[test]
    fn test_identity_is_stabilizer() {
        let code = steane_code();
        let checker = StabilizerFlipChecker::new(&code);

        // Identity error
        let identity = PauliString::identity();
        let class = checker.classify_error(&identity);

        assert!(
            matches!(class, ErrorClass::Stabilizer),
            "Identity should be classified as stabilizer"
        );
    }
}
