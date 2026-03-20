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

//! Stabilizer code representation and verification.
//!
//! This module provides tools for defining and verifying stabilizer quantum error correcting codes.

// Allow similar names for logical_xs/logical_zs - these are intentional and meaningful
#![allow(clippy::similar_names)]

use pecos_core::{PauliOperator, PauliString};
use std::collections::BTreeSet;
use thiserror::Error;

/// Errors that can occur during stabilizer code verification.
#[derive(Debug, Error)]
pub enum StabilizerCodeSpecError {
    /// Two stabilizer generators anticommute.
    #[error("Stabilizer generators {0} and {1} anticommute")]
    StabilizersAnticommute(usize, usize),

    /// Logical operators anticommute with each other (when they shouldn't).
    #[error("Logical operators of the same type anticommute: {0} and {1}")]
    LogicalOpsAnticommute(String, String),

    /// A logical operator anticommutes with a stabilizer.
    #[error("Logical operator {logical} anticommutes with stabilizer {stabilizer}")]
    LogicalAnticommutesWithStabilizer { logical: String, stabilizer: usize },

    /// Logical X and Z don't form proper pairs.
    #[error("Logical X{0} and Z{0} do not anticommute")]
    LogicalPairDoesNotAnticommute(usize),

    /// Logical X and Z from different pairs commute when they should be independent.
    #[error("Logical X{0} and Z{1} anticommute (should commute for different logical qubits)")]
    CrossLogicalAnticommute(usize, usize),

    /// Invalid code parameters.
    #[error("Invalid code: {0}")]
    InvalidCode(String),
}

/// Result type for stabilizer code operations.
pub type Result<T> = std::result::Result<T, StabilizerCodeSpecError>;

/// Represents a stabilizer quantum error correcting code.
///
/// A stabilizer code is defined by:
/// - A set of stabilizer generators (commuting Pauli operators that define the code space)
/// - Logical X and Z operators for each logical qubit
///
/// The code encodes `k` logical qubits into `n` physical qubits using `n-k` stabilizer generators.
#[derive(Clone, Debug)]
pub struct StabilizerCodeSpec {
    /// Number of physical (data) qubits.
    num_qubits: usize,
    /// Stabilizer generators.
    stabilizers: Vec<PauliString>,
    /// Destabilizers (operators that anticommute with exactly one stabilizer each).
    destabilizers: Vec<PauliString>,
    /// Logical Z operators (one per logical qubit).
    logical_zs: Vec<PauliString>,
    /// Logical X operators (one per logical qubit).
    logical_xs: Vec<PauliString>,
    /// Code distance (if computed).
    distance: Option<usize>,
}

/// Column-based index for efficient commutation checking.
///
/// For each qubit, tracks which operators have X or Z on that qubit.
/// This enables `O(weight)` commutation checks instead of `O(num_operators)`.
///
/// The key insight: operator A anticommutes with operator B if and only if
/// A's X positions overlap B's Z positions (or vice versa) an odd number of times.
/// Using column sets, we can find all anticommuting operators via XOR:
/// - For each X position q in A, XOR together `col_z[q]`
/// - For each Z position q in A, XOR together `col_x[q]`
/// - The result contains all operators that anticommute with A.
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

    /// Find all operators that anticommute with the given Pauli.
    ///
    /// Returns a set of indices into the original operator list.
    fn find_anticommuting(&self, pauli: &PauliString) -> BTreeSet<usize> {
        let mut result = BTreeSet::new();

        // X on qubit q anticommutes with Z on qubit q
        for q in pauli.x_positions() {
            if q < self.col_z.len() {
                // XOR: toggle membership
                for &idx in &self.col_z[q] {
                    if result.contains(&idx) {
                        result.remove(&idx);
                    } else {
                        result.insert(idx);
                    }
                }
            }
        }

        // Z on qubit q anticommutes with X on qubit q
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

    /// Check if the given Pauli commutes with all indexed operators.
    fn commutes_with_all(&self, pauli: &PauliString) -> bool {
        self.find_anticommuting(pauli).is_empty()
    }
}

impl StabilizerCodeSpec {
    /// Creates a new stabilizer code with the given parameters.
    ///
    /// # Parameters
    /// - `num_qubits`: Number of physical data qubits
    /// - `stabilizers`: The stabilizer generators
    /// - `logical_zs`: Logical Z operators (one per logical qubit)
    /// - `logical_xs`: Logical X operators (one per logical qubit)
    ///
    /// # Errors
    /// Returns an error if the logical X and Z vectors have different lengths.
    pub fn new(
        num_qubits: usize,
        stabilizers: Vec<PauliString>,
        logical_zs: Vec<PauliString>,
        logical_xs: Vec<PauliString>,
    ) -> Result<Self> {
        if logical_zs.len() != logical_xs.len() {
            return Err(StabilizerCodeSpecError::InvalidCode(
                "Number of logical X and Z operators must match".to_string(),
            ));
        }

        Ok(Self {
            num_qubits,
            stabilizers,
            destabilizers: Vec::new(),
            logical_zs,
            logical_xs,
            distance: None,
        })
    }

    /// Creates a new stabilizer code with destabilizers.
    ///
    /// # Parameters
    /// - `num_qubits`: Number of physical data qubits
    /// - `stabilizers`: The stabilizer generators
    /// - `destabilizers`: Destabilizers (one per stabilizer, anticommuting with exactly that stabilizer)
    /// - `logical_zs`: Logical Z operators (one per logical qubit)
    /// - `logical_xs`: Logical X operators (one per logical qubit)
    ///
    /// # Errors
    /// Returns an error if the logical X and Z vectors have different lengths.
    pub fn with_destabilizers(
        num_qubits: usize,
        stabilizers: Vec<PauliString>,
        destabilizers: Vec<PauliString>,
        logical_zs: Vec<PauliString>,
        logical_xs: Vec<PauliString>,
    ) -> Result<Self> {
        if logical_zs.len() != logical_xs.len() {
            return Err(StabilizerCodeSpecError::InvalidCode(
                "Number of logical X and Z operators must match".to_string(),
            ));
        }

        Ok(Self {
            num_qubits,
            stabilizers,
            destabilizers,
            logical_zs,
            logical_xs,
            distance: None,
        })
    }

    /// Creates a stabilizer code from just the stabilizers.
    ///
    /// The logical operators can be added later.
    #[must_use]
    pub fn from_stabilizers(num_qubits: usize, stabilizers: Vec<PauliString>) -> Self {
        Self {
            num_qubits,
            stabilizers,
            destabilizers: Vec::new(),
            logical_zs: Vec::new(),
            logical_xs: Vec::new(),
            distance: None,
        }
    }

    /// Creates a builder for constructing a stabilizer code.
    ///
    /// This provides a fluent API similar to Python's `VerifyStabilizers`.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_qec::StabilizerCodeSpec;
    /// use pecos_core::{Xs, Zs};
    ///
    /// // Build a 3-qubit bit flip code
    /// let code = StabilizerCodeSpec::builder(3)
    ///     .check(Zs([0, 1]))
    ///     .check(Zs([1, 2]))
    ///     .logical_z(Zs([0, 1, 2]))
    ///     .logical_x(Xs([0, 1, 2]))
    ///     .build()
    ///     .unwrap();
    ///
    /// assert_eq!(code.num_qubits(), 3);
    /// assert_eq!(code.num_logical_qubits(), 1);
    /// ```
    #[must_use]
    pub fn builder(num_qubits: usize) -> StabilizerCodeSpecBuilder {
        StabilizerCodeSpecBuilder::new(num_qubits)
    }

    /// Returns the number of physical qubits.
    #[inline]
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Returns the number of logical qubits encoded by this code.
    ///
    /// This is `n - s` where `n` is the number of physical qubits and `s` is
    /// the number of independent stabilizer generators.
    #[inline]
    #[must_use]
    pub fn num_logical_qubits(&self) -> usize {
        self.num_qubits.saturating_sub(self.stabilizers.len())
    }

    /// Returns the number of stabilizer generators.
    #[inline]
    #[must_use]
    pub fn num_stabilizers(&self) -> usize {
        self.stabilizers.len()
    }

    /// Returns a reference to the stabilizer generators.
    #[inline]
    #[must_use]
    pub fn stabilizers(&self) -> &[PauliString] {
        &self.stabilizers
    }

    /// Returns a reference to the destabilizers.
    ///
    /// Destabilizers are operators that anticommute with exactly one stabilizer each.
    /// The i-th destabilizer anticommutes with the i-th stabilizer and commutes with all others.
    #[inline]
    #[must_use]
    pub fn destabilizers(&self) -> &[PauliString] {
        &self.destabilizers
    }

    /// Returns a reference to the logical Z operators.
    #[inline]
    #[must_use]
    pub fn logical_zs(&self) -> &[PauliString] {
        &self.logical_zs
    }

    /// Returns a reference to the logical X operators.
    #[inline]
    #[must_use]
    pub fn logical_xs(&self) -> &[PauliString] {
        &self.logical_xs
    }

    /// Returns the code distance if it has been computed.
    #[inline]
    #[must_use]
    pub fn distance(&self) -> Option<usize> {
        self.distance
    }

    /// Sets the code distance.
    pub fn set_distance(&mut self, distance: usize) {
        self.distance = Some(distance);
    }

    /// Adds a logical Z operator.
    pub fn add_logical_z(&mut self, logical_z: PauliString) {
        self.logical_zs.push(logical_z);
    }

    /// Adds a logical X operator.
    pub fn add_logical_x(&mut self, logical_x: PauliString) {
        self.logical_xs.push(logical_x);
    }

    /// Returns the code parameters as a string in [[n, k, d]] notation.
    ///
    /// If distance is not computed, returns [[n, k, ?]].
    #[must_use]
    pub fn code_parameters(&self) -> String {
        let n = self.num_qubits;
        let k = self.num_logical_qubits();
        match self.distance {
            Some(d) => format!("[[{n}, {k}, {d}]]"),
            None => format!("[[{n}, {k}, ?]]"),
        }
    }

    // ========================================================================
    // Verification methods
    // ========================================================================

    /// Verifies that all stabilizer generators commute with each other.
    ///
    /// Returns `Ok(())` if all stabilizers commute.
    ///
    /// # Errors
    /// Returns [`StabilizerCodeSpecError::StabilizersAnticommute`] if any pair of
    /// stabilizers anticommute.
    pub fn verify_stabilizers_commute(&self) -> Result<()> {
        for i in 0..self.stabilizers.len() {
            for j in (i + 1)..self.stabilizers.len() {
                if !self.stabilizers[i].commutes_with(&self.stabilizers[j]) {
                    return Err(StabilizerCodeSpecError::StabilizersAnticommute(i, j));
                }
            }
        }
        Ok(())
    }

    /// Verifies that all logical Z operators commute with all stabilizers.
    ///
    /// # Errors
    /// Returns [`StabilizerCodeSpecError::LogicalAnticommutesWithStabilizer`] if any
    /// logical Z operator anticommutes with a stabilizer.
    pub fn verify_logical_zs_commute_with_stabilizers(&self) -> Result<()> {
        for (i, logical_z) in self.logical_zs.iter().enumerate() {
            for (j, stab) in self.stabilizers.iter().enumerate() {
                if !logical_z.commutes_with(stab) {
                    return Err(StabilizerCodeSpecError::LogicalAnticommutesWithStabilizer {
                        logical: format!("Z{i}"),
                        stabilizer: j,
                    });
                }
            }
        }
        Ok(())
    }

    /// Verifies that all logical X operators commute with all stabilizers.
    ///
    /// # Errors
    /// Returns [`StabilizerCodeSpecError::LogicalAnticommutesWithStabilizer`] if any
    /// logical X operator anticommutes with a stabilizer.
    pub fn verify_logical_xs_commute_with_stabilizers(&self) -> Result<()> {
        for (i, logical_x) in self.logical_xs.iter().enumerate() {
            for (j, stab) in self.stabilizers.iter().enumerate() {
                if !logical_x.commutes_with(stab) {
                    return Err(StabilizerCodeSpecError::LogicalAnticommutesWithStabilizer {
                        logical: format!("X{i}"),
                        stabilizer: j,
                    });
                }
            }
        }
        Ok(())
    }

    /// Verifies that all logical Z operators commute with each other.
    ///
    /// # Errors
    /// Returns [`StabilizerCodeSpecError::LogicalOpsAnticommute`] if any pair of
    /// logical Z operators anticommute.
    pub fn verify_logical_zs_commute(&self) -> Result<()> {
        for i in 0..self.logical_zs.len() {
            for j in (i + 1)..self.logical_zs.len() {
                if !self.logical_zs[i].commutes_with(&self.logical_zs[j]) {
                    return Err(StabilizerCodeSpecError::LogicalOpsAnticommute(
                        format!("Z{i}"),
                        format!("Z{j}"),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Verifies that all logical X operators commute with each other.
    ///
    /// # Errors
    /// Returns [`StabilizerCodeSpecError::LogicalOpsAnticommute`] if any pair of
    /// logical X operators anticommute.
    pub fn verify_logical_xs_commute(&self) -> Result<()> {
        for i in 0..self.logical_xs.len() {
            for j in (i + 1)..self.logical_xs.len() {
                if !self.logical_xs[i].commutes_with(&self.logical_xs[j]) {
                    return Err(StabilizerCodeSpecError::LogicalOpsAnticommute(
                        format!("X{i}"),
                        format!("X{j}"),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Verifies that logical `X_i` and `Z_i` anticommute (they form a proper pair).
    ///
    /// # Errors
    /// Returns [`StabilizerCodeSpecError::LogicalPairDoesNotAnticommute`] if any
    /// logical X and Z pair commute when they should anticommute.
    pub fn verify_logical_pairs_anticommute(&self) -> Result<()> {
        for i in 0..self.logical_xs.len().min(self.logical_zs.len()) {
            if self.logical_xs[i].commutes_with(&self.logical_zs[i]) {
                return Err(StabilizerCodeSpecError::LogicalPairDoesNotAnticommute(i));
            }
        }
        Ok(())
    }

    /// Verifies that logical operators from different logical qubits commute.
    ///
    /// `X_i` should commute with `Z_j` for i != j.
    ///
    /// # Errors
    /// Returns [`StabilizerCodeSpecError::CrossLogicalAnticommute`] if logical `X_i`
    /// anticommutes with `Z_j` for i != j.
    pub fn verify_cross_logical_commute(&self) -> Result<()> {
        for i in 0..self.logical_xs.len() {
            for j in 0..self.logical_zs.len() {
                if i != j && !self.logical_xs[i].commutes_with(&self.logical_zs[j]) {
                    return Err(StabilizerCodeSpecError::CrossLogicalAnticommute(i, j));
                }
            }
        }
        Ok(())
    }

    /// Performs full verification of the stabilizer code.
    ///
    /// This checks:
    /// 1. All stabilizers commute with each other
    /// 2. All logical operators commute with all stabilizers
    /// 3. Logical Z operators commute with each other
    /// 4. Logical X operators commute with each other
    /// 5. Logical `X_i` and `Z_i` anticommute (proper pairs)
    /// 6. Logical `X_i` and `Z_j` commute for i != j
    ///
    /// Returns `Ok(())` if all checks pass.
    ///
    /// # Errors
    /// Returns a [`StabilizerCodeSpecError`] if any verification check fails.
    pub fn verify(&self) -> Result<()> {
        self.verify_stabilizers_commute()?;
        self.verify_logical_zs_commute_with_stabilizers()?;
        self.verify_logical_xs_commute_with_stabilizers()?;
        self.verify_logical_zs_commute()?;
        self.verify_logical_xs_commute()?;
        self.verify_logical_pairs_anticommute()?;
        self.verify_cross_logical_commute()?;
        Ok(())
    }

    // ========================================================================
    // Pauli classification
    // ========================================================================

    /// Checks if a Pauli operator commutes with all stabilizers.
    ///
    /// This is a necessary (but not sufficient) condition for being in the
    /// stabilizer group or being a logical operator.
    #[must_use]
    pub fn commutes_with_all_stabilizers(&self, pauli: &PauliString) -> bool {
        self.stabilizers.iter().all(|s| pauli.commutes_with(s))
    }

    /// Checks if a Pauli operator anticommutes with any stabilizer.
    ///
    /// If true, the operator is an "error" that can be detected by syndrome measurement.
    #[must_use]
    pub fn is_detectable_error(&self, pauli: &PauliString) -> bool {
        !self.commutes_with_all_stabilizers(pauli)
    }

    /// Returns the indices of stabilizers that anticommute with the given Pauli operator.
    ///
    /// This is the "syndrome" that would be measured if this error occurred.
    #[must_use]
    pub fn syndrome(&self, pauli: &PauliString) -> Vec<usize> {
        self.stabilizers
            .iter()
            .enumerate()
            .filter(|(_, s)| !pauli.commutes_with(s))
            .map(|(i, _)| i)
            .collect()
    }

    /// Checks if a Pauli operator anticommutes with any logical operator.
    ///
    /// If a Pauli commutes with all stabilizers but anticommutes with a logical
    /// operator, it is a logical error.
    #[must_use]
    pub fn anticommutes_with_logical(&self, pauli: &PauliString) -> bool {
        self.logical_zs.iter().any(|z| !pauli.commutes_with(z))
            || self.logical_xs.iter().any(|x| !pauli.commutes_with(x))
    }

    /// Checks if a Pauli operator is a logical error.
    ///
    /// A logical error is an operator that:
    /// 1. Commutes with all stabilizers (undetectable)
    /// 2. Anticommutes with at least one logical operator (causes a logical error)
    #[must_use]
    pub fn is_logical_error(&self, pauli: &PauliString) -> bool {
        self.commutes_with_all_stabilizers(pauli) && self.anticommutes_with_logical(pauli)
    }

    // ========================================================================
    // Indexed (optimized) methods
    // ========================================================================

    /// Builds a column index for the stabilizers.
    ///
    /// The index enables `O(weight)` commutation checks instead of `O(num_stabilizers * weight)`.
    /// For repeated calls to [`Self::commutes_with_all_stabilizers`] or [`Self::is_logical_error`],
    /// building the index once and using the `_indexed` variants is much faster.
    #[must_use]
    pub fn build_stabilizer_index(&self) -> StabilizerIndex {
        StabilizerIndex(ColumnIndex::from_paulis(self.num_qubits, &self.stabilizers))
    }

    /// Builds a column index for the logical operators.
    ///
    /// The index enables `O(weight)` anticommutation checks instead of `O(num_logicals * weight)`.
    #[must_use]
    pub fn build_logical_index(&self) -> LogicalIndex {
        let mut all_logicals = self.logical_zs.clone();
        all_logicals.extend(self.logical_xs.iter().cloned());
        LogicalIndex(ColumnIndex::from_paulis(self.num_qubits, &all_logicals))
    }

    /// Checks if a Pauli commutes with all stabilizers using a precomputed index.
    ///
    /// This is `O(weight)` instead of `O(num_stabilizers * weight)`.
    #[must_use]
    pub fn commutes_with_all_stabilizers_indexed(
        &self,
        pauli: &PauliString,
        index: &StabilizerIndex,
    ) -> bool {
        index.0.commutes_with_all(pauli)
    }

    /// Checks if a Pauli anticommutes with any logical operator using a precomputed index.
    ///
    /// This is `O(weight)` instead of `O(num_logicals * weight)`.
    #[must_use]
    pub fn anticommutes_with_logical_indexed(
        &self,
        pauli: &PauliString,
        index: &LogicalIndex,
    ) -> bool {
        !index.0.commutes_with_all(pauli)
    }

    /// Checks if a Pauli is a logical error using precomputed indices.
    ///
    /// This is much faster for repeated checks, such as during distance calculation.
    #[must_use]
    pub fn is_logical_error_indexed(
        &self,
        pauli: &PauliString,
        stab_index: &StabilizerIndex,
        log_index: &LogicalIndex,
    ) -> bool {
        self.commutes_with_all_stabilizers_indexed(pauli, stab_index)
            && self.anticommutes_with_logical_indexed(pauli, log_index)
    }

    /// Returns the syndrome using a precomputed index.
    ///
    /// This is `O(weight)` instead of `O(num_stabilizers * weight)`.
    /// The result is a sorted vector of stabilizer indices that anticommute with the Pauli.
    #[must_use]
    pub fn syndrome_indexed(&self, pauli: &PauliString, index: &StabilizerIndex) -> Vec<usize> {
        index.0.find_anticommuting(pauli).into_iter().collect()
    }

    /// Builds both stabilizer and logical indices at once.
    ///
    /// This is a convenience method for when you need both indices.
    #[must_use]
    pub fn build_indices(&self) -> CodeIndices {
        CodeIndices {
            stabilizer: self.build_stabilizer_index(),
            logical: self.build_logical_index(),
        }
    }

    // ========================================================================
    // Distance calculation
    // ========================================================================

    /// Calculates the code distance by exhaustive search.
    ///
    /// The distance is the minimum weight of any logical operator.
    /// This method tries all Pauli errors starting from weight 1, returning
    /// the first weight at which a logical error is found.
    ///
    /// # Warning
    ///
    /// This is an exponential-time algorithm. For a code on `n` qubits,
    /// checking weight `w` requires examining `O(n^w * 3^w)` operators.
    /// Practical for codes with `n < 20` or so.
    ///
    /// # Returns
    ///
    /// A [`crate::DistanceResult`] containing the distance and the first logical error found.
    /// Returns `None` if no logical error exists (stabilizer state).
    #[must_use]
    pub fn calculate_distance(&mut self) -> Option<crate::DistanceResult> {
        self.calculate_distance_with_options(&crate::DistanceSearchConfig::default())
    }

    /// Calculates the code distance with configurable options.
    ///
    /// # Options
    ///
    /// - `css_only`: If true, only check X-only and Z-only errors (faster for CSS codes)
    /// - `max_weight`: Maximum weight to check (default: `num_qubits`)
    /// - `verbose`: If true, print progress messages
    ///
    /// # Returns
    ///
    /// A [`crate::DistanceResult`] containing the distance and the first logical error found.
    /// Returns `None` if no logical error exists up to `max_weight`.
    #[must_use]
    pub fn calculate_distance_with_options(
        &mut self,
        config: &crate::DistanceSearchConfig,
    ) -> Option<crate::DistanceResult> {
        let result = crate::calculate_distance(self, config);
        if let Some(ref r) = result {
            self.distance = Some(r.distance);
        }
        result
    }

    // ========================================================================
    // Logical operator discovery
    // ========================================================================

    /// Discovers logical operators using stabilizer simulation.
    ///
    /// This uses the stabilizer simulator to automatically find logical X and Z
    /// operators for the code based solely on the stabilizer generators.
    /// Any existing logical operators will be replaced.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_qec::StabilizerCodeSpec;
    /// use pecos_core::{Zs, PauliOperator};
    ///
    /// // Create code with just stabilizers (3-qubit bit flip code)
    /// let mut code = StabilizerCodeSpec::from_stabilizers(3, vec![
    ///     Zs([0, 1]),  // ZZI
    ///     Zs([1, 2]),  // IZZ
    /// ]);
    ///
    /// // Discover logical operators
    /// code.discover_logicals().unwrap();
    ///
    /// assert_eq!(code.logical_zs().len(), 1);
    /// assert_eq!(code.logical_xs().len(), 1);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The stabilizers don't all commute
    /// - The stabilizers are linearly dependent
    /// - Discovery fails for any other reason
    pub fn discover_logicals(&mut self) -> std::result::Result<(), crate::LogicalDiscoveryError> {
        let result = crate::discover_logical_operators(self.num_qubits, &self.stabilizers)?;
        self.logical_zs = result.logical_zs;
        self.logical_xs = result.logical_xs;
        self.destabilizers = result.destabilizers;
        Ok(())
    }

    /// Returns whether logical operators have been defined for this code.
    #[inline]
    #[must_use]
    pub fn has_logicals(&self) -> bool {
        !self.logical_zs.is_empty() && !self.logical_xs.is_empty()
    }

    // ========================================================================
    // Conversions
    // ========================================================================

    /// Converts the stabilizer generators into a [`PauliStabilizerGroup`].
    ///
    /// This enables access to the algebraic analysis methods available on
    /// [`PauliStabilizerGroup`] (rank, GF(2) membership, group enumeration,
    /// centralizer, Clifford conjugation, etc.).
    ///
    /// # Errors
    ///
    /// Returns an error if the stabilizers don't form a valid stabilizer group
    /// (e.g., non-commuting generators or non-real phases). This should not
    /// happen for properly constructed codes.
    pub fn to_stabilizer_group(
        &self,
    ) -> std::result::Result<
        pecos_quantum::PauliStabilizerGroup,
        pecos_quantum::stabilizer_group::PauliStabilizerGroupError,
    > {
        pecos_quantum::PauliStabilizerGroup::new(self.stabilizers.clone())
    }

    /// Creates a `StabilizerCodeSpec` from a [`PauliStabilizerGroup`].
    ///
    /// The resulting code has stabilizer generators but no logical operators
    /// or destabilizers. Use [`discover_logicals`](Self::discover_logicals)
    /// to compute them.
    #[must_use]
    pub fn from_stabilizer_group(group: &pecos_quantum::PauliStabilizerGroup) -> Self {
        Self {
            num_qubits: group.num_qubits(),
            stabilizers: group.stabilizers().to_vec(),
            destabilizers: Vec::new(),
            logical_zs: Vec::new(),
            logical_xs: Vec::new(),
            distance: None,
        }
    }

    /// Creates a `StabilizerCodeSpec` from a [`StabilizerCode`](crate::StabilizerCode),
    /// automatically discovering paired logical operators and destabilizers.
    ///
    /// This is the easiest way to get a full code specification from a
    /// `StabilizerCode`, which provides convenient constructors for standard codes.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_qec::{StabilizerCode, StabilizerCodeSpec};
    ///
    /// let spec = StabilizerCodeSpec::from_stabilizer_code(&StabilizerCode::steane()).unwrap();
    /// assert_eq!(spec.num_qubits(), 7);
    /// assert_eq!(spec.num_logical_qubits(), 1);
    /// assert_eq!(spec.logical_zs().len(), 1);
    /// assert_eq!(spec.logical_xs().len(), 1);
    /// assert!(spec.verify().is_ok());
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if logical operator discovery fails.
    pub fn from_stabilizer_code(
        code: &crate::StabilizerCode,
    ) -> std::result::Result<Self, crate::LogicalDiscoveryError> {
        let mut spec =
            Self::from_stabilizers(code.num_qubits(), code.group().stabilizers().to_vec());
        spec.discover_logicals()?;
        Ok(spec)
    }
}

/// Precomputed column index for stabilizer generators.
///
/// Use [`StabilizerCodeSpec::build_stabilizer_index`] to create one.
pub struct StabilizerIndex(ColumnIndex);

/// Precomputed column index for logical operators.
///
/// Use [`StabilizerCodeSpec::build_logical_index`] to create one.
pub struct LogicalIndex(ColumnIndex);

impl LogicalIndex {
    /// Find indices of logical operators that anticommute with the given Pauli.
    ///
    /// Returns a set of indices into the combined logical operators array,
    /// where indices 0..k are logical Zs and k..2k are logical Xs.
    #[must_use]
    pub fn find_anticommuting(&self, pauli: &PauliString) -> std::collections::BTreeSet<usize> {
        self.0.find_anticommuting(pauli)
    }
}

/// Both stabilizer and logical indices for a code.
///
/// Use [`StabilizerCodeSpec::build_indices`] to create one.
pub struct CodeIndices {
    /// Index for stabilizer generators.
    pub stabilizer: StabilizerIndex,
    /// Index for logical operators.
    pub logical: LogicalIndex,
}

// ============================================================================
// Builder
// ============================================================================

/// Builder for constructing stabilizer codes with a fluent API.
///
/// This provides an ergonomic way to define stabilizer codes, similar to
/// Python's `VerifyStabilizers` class.
///
/// # Example
///
/// ```
/// use pecos_qec::StabilizerCodeSpecBuilder;
/// use pecos_core::{Xs, Zs};
///
/// // Build the Steane [[7, 1, 3]] code
/// let code = StabilizerCodeSpecBuilder::new(7)
///     // X-type stabilizers
///     .check(Xs([0, 2, 4, 6]))
///     .check(Xs([1, 2, 5, 6]))
///     .check(Xs([3, 4, 5, 6]))
///     // Z-type stabilizers
///     .check(Zs([0, 2, 4, 6]))
///     .check(Zs([1, 2, 5, 6]))
///     .check(Zs([3, 4, 5, 6]))
///     // Logical operators
///     .logical_z(Zs(0..=6))
///     .logical_x(Xs(0..=6))
///     .build()
///     .unwrap();
///
/// assert!(code.verify().is_ok());
/// ```
#[derive(Clone, Debug, Default)]
pub struct StabilizerCodeSpecBuilder {
    num_qubits: usize,
    stabilizers: Vec<PauliString>,
    logical_zs: Vec<PauliString>,
    logical_xs: Vec<PauliString>,
}

impl StabilizerCodeSpecBuilder {
    /// Creates a new builder for a code with the specified number of qubits.
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            stabilizers: Vec::new(),
            logical_zs: Vec::new(),
            logical_xs: Vec::new(),
        }
    }

    /// Creates a builder pre-populated with stabilizers from a [`StabilizerCode`](crate::StabilizerCode).
    ///
    /// The builder has the stabilizers set but no logical operators.
    /// Use `.logical_z()` / `.logical_x()` to add them manually,
    /// or call `.build_with_discovered_logicals()` to auto-discover them.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_qec::{StabilizerCode, StabilizerCodeSpecBuilder};
    ///
    /// let spec = StabilizerCodeSpecBuilder::from_stabilizer_code(&StabilizerCode::steane())
    ///     .build_with_discovered_logicals()
    ///     .unwrap();
    ///
    /// assert_eq!(spec.num_qubits(), 7);
    /// assert!(spec.verify().is_ok());
    /// ```
    #[must_use]
    pub fn from_stabilizer_code(code: &crate::StabilizerCode) -> Self {
        Self {
            num_qubits: code.num_qubits(),
            stabilizers: code.group().stabilizers().to_vec(),
            logical_zs: Vec::new(),
            logical_xs: Vec::new(),
        }
    }

    /// Adds a stabilizer from a `PauliString` directly.
    #[must_use]
    pub fn stabilizer_pauli(mut self, pauli: PauliString) -> Self {
        self.stabilizers.push(pauli);
        self
    }

    /// Adds a stabilizer from an `UnitaryRep`.
    ///
    /// The operator must be convertible to a `PauliString` (i.e., a Pauli operator
    /// or tensor product of Pauli operators).
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_qec::StabilizerCodeSpecBuilder;
    /// use pecos_core::{Xs, Zs};
    ///
    /// let code = StabilizerCodeSpecBuilder::new(4)
    ///     .check(Zs(0..=1))           // ZZ on qubits 0,1
    ///     .check(Xs(0..=1) & Zs(2..=3)) // XXZZ
    ///     .build()
    ///     .unwrap();
    /// ```
    ///
    /// Accepts both `PauliString` and `UnitaryRep` (via `Into<UnitaryRep>`).
    ///
    /// # Panics
    ///
    /// Panics if the operator cannot be converted to a `PauliString`.
    #[must_use]
    pub fn check(mut self, op: impl Into<pecos_core::UnitaryRep>) -> Self {
        let ps = op
            .into()
            .try_to_pauli_string()
            .expect("UnitaryRep must be convertible to PauliString");
        self.stabilizers.push(ps);
        self
    }

    /// Adds a logical Z operator from a `PauliString` directly.
    #[must_use]
    pub fn logical_z_pauli(mut self, pauli: PauliString) -> Self {
        self.logical_zs.push(pauli);
        self
    }

    /// Adds a logical Z operator.
    ///
    /// Accepts both `PauliString` and `UnitaryRep` (via `Into<UnitaryRep>`).
    ///
    /// # Panics
    ///
    /// Panics if the operator cannot be converted to a `PauliString`.
    #[must_use]
    pub fn logical_z(mut self, op: impl Into<pecos_core::UnitaryRep>) -> Self {
        let ps = op
            .into()
            .try_to_pauli_string()
            .expect("UnitaryRep must be convertible to PauliString");
        self.logical_zs.push(ps);
        self
    }

    /// Adds a logical X operator from a `PauliString` directly.
    #[must_use]
    pub fn logical_x_pauli(mut self, pauli: PauliString) -> Self {
        self.logical_xs.push(pauli);
        self
    }

    /// Adds a logical X operator.
    ///
    /// Accepts both `PauliString` and `UnitaryRep` (via `Into<UnitaryRep>`).
    ///
    /// # Panics
    ///
    /// Panics if the operator cannot be converted to a `PauliString`.
    #[must_use]
    pub fn logical_x(mut self, op: impl Into<pecos_core::UnitaryRep>) -> Self {
        let ps = op
            .into()
            .try_to_pauli_string()
            .expect("UnitaryRep must be convertible to PauliString");
        self.logical_xs.push(ps);
        self
    }

    /// Builds the stabilizer code.
    ///
    /// # Errors
    ///
    /// Returns an error if the number of logical X and Z operators don't match.
    pub fn build(self) -> Result<StabilizerCodeSpec> {
        StabilizerCodeSpec::new(
            self.num_qubits,
            self.stabilizers,
            self.logical_zs,
            self.logical_xs,
        )
    }

    /// Builds the stabilizer code and verifies it.
    ///
    /// This is a convenience method that calls `build()` followed by `verify()`.
    ///
    /// # Errors
    ///
    /// Returns an error if the code fails to build or verification fails.
    pub fn build_verified(self) -> Result<StabilizerCodeSpec> {
        let code = self.build()?;
        code.verify()?;
        Ok(code)
    }

    /// Builds the stabilizer code and automatically discovers logical operators.
    ///
    /// This is useful when you only have the stabilizer generators and want
    /// the logical operators to be computed automatically using stabilizer
    /// simulation.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_qec::StabilizerCodeSpecBuilder;
    /// use pecos_core::Zs;
    ///
    /// // Build a 3-qubit bit flip code with auto-discovered logicals
    /// let code = StabilizerCodeSpecBuilder::new(3)
    ///     .check(Zs([0, 1]))  // ZZI
    ///     .check(Zs([1, 2]))  // IZZ
    ///     .build_with_discovered_logicals()
    ///     .unwrap();
    ///
    /// assert_eq!(code.num_logical_qubits(), 1);
    /// assert_eq!(code.logical_zs().len(), 1);
    /// assert_eq!(code.logical_xs().len(), 1);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The stabilizers don't all commute
    /// - The stabilizers are linearly dependent
    /// - Discovery fails for any other reason
    pub fn build_with_discovered_logicals(
        self,
    ) -> std::result::Result<StabilizerCodeSpec, crate::LogicalDiscoveryError> {
        let mut code = StabilizerCodeSpec::from_stabilizers(self.num_qubits, self.stabilizers);
        code.discover_logicals()?;
        Ok(code)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::Pauli;

    /// Helper to create a `PauliString` from a simple specification.
    fn pauli_string(paulis: &[(Pauli, usize)]) -> PauliString {
        use pecos_core::QubitId;
        PauliString::with_phase_and_paulis(
            pecos_core::QuarterPhase::PlusOne,
            paulis.iter().map(|&(p, q)| (p, QubitId::new(q))).collect(),
        )
    }

    #[test]
    fn test_three_qubit_bit_flip_code() {
        // 3-qubit bit flip code: [[3, 1, 1]]
        // Stabilizers: ZZI, IZZ
        // Logical Z: ZZZ
        // Logical X: XXX

        let stab1 = pauli_string(&[(Pauli::Z, 0), (Pauli::Z, 1)]);
        let stab2 = pauli_string(&[(Pauli::Z, 1), (Pauli::Z, 2)]);
        let logical_z = pauli_string(&[(Pauli::Z, 0), (Pauli::Z, 1), (Pauli::Z, 2)]);
        let logical_x = pauli_string(&[(Pauli::X, 0), (Pauli::X, 1), (Pauli::X, 2)]);

        let code = StabilizerCodeSpec::new(3, vec![stab1, stab2], vec![logical_z], vec![logical_x])
            .unwrap();

        assert_eq!(code.num_qubits(), 3);
        assert_eq!(code.num_logical_qubits(), 1);
        assert_eq!(code.num_stabilizers(), 2);

        // Verify the code
        assert!(code.verify().is_ok());
    }

    #[test]
    fn test_three_qubit_phase_flip_code() {
        // 3-qubit phase flip code: [[3, 1, 1]]
        // Stabilizers: XXI, IXX
        // Logical Z: ZZZ
        // Logical X: XXX

        let stab1 = pauli_string(&[(Pauli::X, 0), (Pauli::X, 1)]);
        let stab2 = pauli_string(&[(Pauli::X, 1), (Pauli::X, 2)]);
        let logical_z = pauli_string(&[(Pauli::Z, 0), (Pauli::Z, 1), (Pauli::Z, 2)]);
        let logical_x = pauli_string(&[(Pauli::X, 0), (Pauli::X, 1), (Pauli::X, 2)]);

        let code = StabilizerCodeSpec::new(3, vec![stab1, stab2], vec![logical_z], vec![logical_x])
            .unwrap();

        assert!(code.verify().is_ok());
    }

    #[test]
    fn test_anticommuting_stabilizers_error() {
        // Create two stabilizers that anticommute
        let stab1 = pauli_string(&[(Pauli::X, 0)]);
        let stab2 = pauli_string(&[(Pauli::Z, 0)]);

        let code = StabilizerCodeSpec::from_stabilizers(1, vec![stab1, stab2]);

        let result = code.verify_stabilizers_commute();
        assert!(matches!(
            result,
            Err(StabilizerCodeSpecError::StabilizersAnticommute(0, 1))
        ));
    }

    #[test]
    fn test_logical_pair_must_anticommute() {
        // Create a code where logical X and Z commute (invalid)
        let stab = pauli_string(&[(Pauli::Z, 0), (Pauli::Z, 1)]);
        let logical_z = pauli_string(&[(Pauli::Z, 0)]);
        let logical_x = pauli_string(&[(Pauli::Z, 1)]); // Should be X, not Z

        let code =
            StabilizerCodeSpec::new(2, vec![stab], vec![logical_z], vec![logical_x]).unwrap();

        let result = code.verify_logical_pairs_anticommute();
        assert!(matches!(
            result,
            Err(StabilizerCodeSpecError::LogicalPairDoesNotAnticommute(0))
        ));
    }

    #[test]
    fn test_syndrome_detection() {
        // 3-qubit bit flip code
        let stab1 = pauli_string(&[(Pauli::Z, 0), (Pauli::Z, 1)]);
        let stab2 = pauli_string(&[(Pauli::Z, 1), (Pauli::Z, 2)]);
        let code = StabilizerCodeSpec::from_stabilizers(3, vec![stab1, stab2]);

        // X error on qubit 0 should trigger stabilizer 0 only
        let x0 = pauli_string(&[(Pauli::X, 0)]);
        assert_eq!(code.syndrome(&x0), vec![0]);

        // X error on qubit 1 should trigger both stabilizers
        let x1 = pauli_string(&[(Pauli::X, 1)]);
        assert_eq!(code.syndrome(&x1), vec![0, 1]);

        // X error on qubit 2 should trigger stabilizer 1 only
        let x2 = pauli_string(&[(Pauli::X, 2)]);
        assert_eq!(code.syndrome(&x2), vec![1]);

        // Z errors should have no syndrome (commute with Z-type stabilizers)
        let z0 = pauli_string(&[(Pauli::Z, 0)]);
        assert!(code.syndrome(&z0).is_empty());
    }

    #[test]
    fn test_code_parameters_string() {
        let stab1 = pauli_string(&[(Pauli::Z, 0), (Pauli::Z, 1)]);
        let stab2 = pauli_string(&[(Pauli::Z, 1), (Pauli::Z, 2)]);
        let mut code = StabilizerCodeSpec::from_stabilizers(3, vec![stab1, stab2]);

        assert_eq!(code.code_parameters(), "[[3, 1, ?]]");

        code.set_distance(1);
        assert_eq!(code.code_parameters(), "[[3, 1, 1]]");
    }

    #[test]
    fn test_logical_error_detection() {
        // 3-qubit bit flip code
        let stab1 = pauli_string(&[(Pauli::Z, 0), (Pauli::Z, 1)]);
        let stab2 = pauli_string(&[(Pauli::Z, 1), (Pauli::Z, 2)]);
        let logical_z = pauli_string(&[(Pauli::Z, 0), (Pauli::Z, 1), (Pauli::Z, 2)]);
        let logical_x = pauli_string(&[(Pauli::X, 0), (Pauli::X, 1), (Pauli::X, 2)]);

        let code = StabilizerCodeSpec::new(3, vec![stab1, stab2], vec![logical_z], vec![logical_x])
            .unwrap();

        // Single X error is detectable (not a logical error)
        let x0 = pauli_string(&[(Pauli::X, 0)]);
        assert!(code.is_detectable_error(&x0));
        assert!(!code.is_logical_error(&x0));

        // XXX is a logical error (commutes with stabilizers, anticommutes with logical Z)
        let xxx = pauli_string(&[(Pauli::X, 0), (Pauli::X, 1), (Pauli::X, 2)]);
        assert!(!code.is_detectable_error(&xxx));
        assert!(code.is_logical_error(&xxx));
    }

    #[test]
    fn test_indexed_methods_match_non_indexed() {
        // 3-qubit bit flip code
        let stab1 = pauli_string(&[(Pauli::Z, 0), (Pauli::Z, 1)]);
        let stab2 = pauli_string(&[(Pauli::Z, 1), (Pauli::Z, 2)]);
        let logical_z = pauli_string(&[(Pauli::Z, 0), (Pauli::Z, 1), (Pauli::Z, 2)]);
        let logical_x = pauli_string(&[(Pauli::X, 0), (Pauli::X, 1), (Pauli::X, 2)]);

        let code = StabilizerCodeSpec::new(3, vec![stab1, stab2], vec![logical_z], vec![logical_x])
            .unwrap();

        let indices = code.build_indices();

        // Test various operators
        let test_cases = [
            pauli_string(&[(Pauli::X, 0)]), // Single X
            pauli_string(&[(Pauli::X, 1)]), // X on middle qubit
            pauli_string(&[(Pauli::Z, 0)]), // Single Z
            pauli_string(&[(Pauli::X, 0), (Pauli::X, 1), (Pauli::X, 2)]), // XXX
            pauli_string(&[(Pauli::Z, 0), (Pauli::Z, 1), (Pauli::Z, 2)]), // ZZZ
            pauli_string(&[(Pauli::Y, 0), (Pauli::Y, 1)]), // YY
        ];

        for pauli in &test_cases {
            // commutes_with_all_stabilizers should match
            assert_eq!(
                code.commutes_with_all_stabilizers(pauli),
                code.commutes_with_all_stabilizers_indexed(pauli, &indices.stabilizer),
                "commutes_with_all_stabilizers mismatch for {pauli:?}"
            );

            // anticommutes_with_logical should match
            assert_eq!(
                code.anticommutes_with_logical(pauli),
                code.anticommutes_with_logical_indexed(pauli, &indices.logical),
                "anticommutes_with_logical mismatch for {pauli:?}"
            );

            // is_logical_error should match
            assert_eq!(
                code.is_logical_error(pauli),
                code.is_logical_error_indexed(pauli, &indices.stabilizer, &indices.logical),
                "is_logical_error mismatch for {pauli:?}"
            );

            // syndrome should match (sorted since indexed returns BTreeSet order)
            let mut expected_syndrome = code.syndrome(pauli);
            expected_syndrome.sort_unstable();
            let mut indexed_syndrome = code.syndrome_indexed(pauli, &indices.stabilizer);
            indexed_syndrome.sort_unstable();
            assert_eq!(
                expected_syndrome, indexed_syndrome,
                "syndrome mismatch for {pauli:?}"
            );
        }
    }

    #[test]
    fn test_syndrome_indexed() {
        // 3-qubit bit flip code
        let stab1 = pauli_string(&[(Pauli::Z, 0), (Pauli::Z, 1)]);
        let stab2 = pauli_string(&[(Pauli::Z, 1), (Pauli::Z, 2)]);
        let code = StabilizerCodeSpec::from_stabilizers(3, vec![stab1, stab2]);
        let index = code.build_stabilizer_index();

        // X error on qubit 0 should trigger stabilizer 0 only
        let x0 = pauli_string(&[(Pauli::X, 0)]);
        assert_eq!(code.syndrome_indexed(&x0, &index), vec![0]);

        // X error on qubit 1 should trigger both stabilizers
        let x1 = pauli_string(&[(Pauli::X, 1)]);
        let mut syndrome = code.syndrome_indexed(&x1, &index);
        syndrome.sort_unstable();
        assert_eq!(syndrome, vec![0, 1]);

        // X error on qubit 2 should trigger stabilizer 1 only
        let x2 = pauli_string(&[(Pauli::X, 2)]);
        assert_eq!(code.syndrome_indexed(&x2, &index), vec![1]);

        // Z errors should have no syndrome
        let z0 = pauli_string(&[(Pauli::Z, 0)]);
        assert!(code.syndrome_indexed(&z0, &index).is_empty());
    }

    // ========================================================================
    // Distance calculation tests
    // ========================================================================

    #[test]
    fn test_distance_three_qubit_bit_flip() {
        // 3-qubit bit flip code: [[3, 1, 1]]
        // Distance should be 1 because single X errors commute with stabilizers
        // but the logical X (XXX) has distance 3... wait, single X errors are detectable.
        // Actually for bit flip code, Z errors are undetectable.
        // Single Z commutes with ZZ stabilizers and anticommutes with logical X (XXX).
        let stab1 = pauli_string(&[(Pauli::Z, 0), (Pauli::Z, 1)]);
        let stab2 = pauli_string(&[(Pauli::Z, 1), (Pauli::Z, 2)]);
        let logical_z = pauli_string(&[(Pauli::Z, 0), (Pauli::Z, 1), (Pauli::Z, 2)]);
        let logical_x = pauli_string(&[(Pauli::X, 0), (Pauli::X, 1), (Pauli::X, 2)]);

        let mut code =
            StabilizerCodeSpec::new(3, vec![stab1, stab2], vec![logical_z], vec![logical_x])
                .unwrap();

        let result = code.calculate_distance();
        assert!(result.is_some());
        let result = result.unwrap();

        // Single Z error is a logical error (commutes with stabilizers, anticommutes with logical X)
        assert_eq!(result.distance, 1);
        assert_eq!(code.distance(), Some(1));
    }

    #[test]
    fn test_distance_steane_code() {
        // Steane [[7, 1, 3]] code
        // X-type stabilizers
        let sx1 = pauli_string(&[(Pauli::X, 0), (Pauli::X, 2), (Pauli::X, 4), (Pauli::X, 6)]);
        let sx2 = pauli_string(&[(Pauli::X, 1), (Pauli::X, 2), (Pauli::X, 5), (Pauli::X, 6)]);
        let sx3 = pauli_string(&[(Pauli::X, 3), (Pauli::X, 4), (Pauli::X, 5), (Pauli::X, 6)]);
        // Z-type stabilizers
        let sz1 = pauli_string(&[(Pauli::Z, 0), (Pauli::Z, 2), (Pauli::Z, 4), (Pauli::Z, 6)]);
        let sz2 = pauli_string(&[(Pauli::Z, 1), (Pauli::Z, 2), (Pauli::Z, 5), (Pauli::Z, 6)]);
        let sz3 = pauli_string(&[(Pauli::Z, 3), (Pauli::Z, 4), (Pauli::Z, 5), (Pauli::Z, 6)]);
        // Logical operators
        let logical_z = pauli_string(&[
            (Pauli::Z, 0),
            (Pauli::Z, 1),
            (Pauli::Z, 2),
            (Pauli::Z, 3),
            (Pauli::Z, 4),
            (Pauli::Z, 5),
            (Pauli::Z, 6),
        ]);
        let logical_x = pauli_string(&[
            (Pauli::X, 0),
            (Pauli::X, 1),
            (Pauli::X, 2),
            (Pauli::X, 3),
            (Pauli::X, 4),
            (Pauli::X, 5),
            (Pauli::X, 6),
        ]);

        let mut code = StabilizerCodeSpec::new(
            7,
            vec![sx1, sx2, sx3, sz1, sz2, sz3],
            vec![logical_z],
            vec![logical_x],
        )
        .unwrap();

        let result = code.calculate_distance();
        assert!(result.is_some());
        let result = result.unwrap();

        assert_eq!(result.distance, 3);
        assert_eq!(code.code_parameters(), "[[7, 1, 3]]");
    }

    #[test]
    fn test_distance_css_mode() {
        // Test CSS mode optimization with Steane code
        let sx1 = pauli_string(&[(Pauli::X, 0), (Pauli::X, 2), (Pauli::X, 4), (Pauli::X, 6)]);
        let sx2 = pauli_string(&[(Pauli::X, 1), (Pauli::X, 2), (Pauli::X, 5), (Pauli::X, 6)]);
        let sx3 = pauli_string(&[(Pauli::X, 3), (Pauli::X, 4), (Pauli::X, 5), (Pauli::X, 6)]);
        let sz1 = pauli_string(&[(Pauli::Z, 0), (Pauli::Z, 2), (Pauli::Z, 4), (Pauli::Z, 6)]);
        let sz2 = pauli_string(&[(Pauli::Z, 1), (Pauli::Z, 2), (Pauli::Z, 5), (Pauli::Z, 6)]);
        let sz3 = pauli_string(&[(Pauli::Z, 3), (Pauli::Z, 4), (Pauli::Z, 5), (Pauli::Z, 6)]);
        let logical_z = pauli_string(&[
            (Pauli::Z, 0),
            (Pauli::Z, 1),
            (Pauli::Z, 2),
            (Pauli::Z, 3),
            (Pauli::Z, 4),
            (Pauli::Z, 5),
            (Pauli::Z, 6),
        ]);
        let logical_x = pauli_string(&[
            (Pauli::X, 0),
            (Pauli::X, 1),
            (Pauli::X, 2),
            (Pauli::X, 3),
            (Pauli::X, 4),
            (Pauli::X, 5),
            (Pauli::X, 6),
        ]);

        let mut code = StabilizerCodeSpec::new(
            7,
            vec![sx1, sx2, sx3, sz1, sz2, sz3],
            vec![logical_z],
            vec![logical_x],
        )
        .unwrap();

        // CSS mode should find the same distance for CSS codes
        let config = crate::DistanceSearchConfig::css();
        let result = code.calculate_distance_with_options(&config);
        assert!(result.is_some());
        assert_eq!(result.unwrap().distance, 3);
    }

    #[test]
    fn test_distance_five_qubit_code() {
        // [[5, 1, 3]] perfect code
        // Stabilizers: XZZXI, IXZZX, XIXZZ, ZXIXZ
        let s1 = pauli_string(&[(Pauli::X, 0), (Pauli::Z, 1), (Pauli::Z, 2), (Pauli::X, 3)]);
        let s2 = pauli_string(&[(Pauli::X, 1), (Pauli::Z, 2), (Pauli::Z, 3), (Pauli::X, 4)]);
        let s3 = pauli_string(&[(Pauli::X, 0), (Pauli::X, 2), (Pauli::Z, 3), (Pauli::Z, 4)]);
        let s4 = pauli_string(&[(Pauli::Z, 0), (Pauli::X, 1), (Pauli::X, 3), (Pauli::Z, 4)]);
        // Logical operators
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

        let mut code =
            StabilizerCodeSpec::new(5, vec![s1, s2, s3, s4], vec![logical_z], vec![logical_x])
                .unwrap();

        let result = code.calculate_distance();
        assert!(result.is_some());
        let result = result.unwrap();

        assert_eq!(result.distance, 3);
        assert_eq!(code.code_parameters(), "[[5, 1, 3]]");
    }

    // ========================================================================
    // Builder tests
    // ========================================================================

    #[test]
    fn test_builder_three_qubit_bit_flip() {
        use pecos_core::{Xs, Zs};

        // Build a 3-qubit bit flip code using the builder
        let code = StabilizerCodeSpec::builder(3)
            .check(Zs([0, 1]))
            .check(Zs([1, 2]))
            .logical_z(Zs([0, 1, 2]))
            .logical_x(Xs([0, 1, 2]))
            .build()
            .unwrap();

        assert_eq!(code.num_qubits(), 3);
        assert_eq!(code.num_logical_qubits(), 1);
        assert_eq!(code.num_stabilizers(), 2);
        assert!(code.verify().is_ok());
    }

    #[test]
    fn test_builder_steane_code() {
        use pecos_core::{Xs, Zs};

        // Build the Steane [[7, 1, 3]] code using the builder
        let code = StabilizerCodeSpecBuilder::new(7)
            // X-type stabilizers
            .check(Xs([0, 2, 4, 6]))
            .check(Xs([1, 2, 5, 6]))
            .check(Xs([3, 4, 5, 6]))
            // Z-type stabilizers
            .check(Zs([0, 2, 4, 6]))
            .check(Zs([1, 2, 5, 6]))
            .check(Zs([3, 4, 5, 6]))
            // Logical operators
            .logical_z(Zs(0..=6))
            .logical_x(Xs(0..=6))
            .build_verified()
            .unwrap();

        assert_eq!(code.num_qubits(), 7);
        assert_eq!(code.num_logical_qubits(), 1);
        assert_eq!(code.num_stabilizers(), 6);
    }

    #[test]
    fn test_builder_weight_two_stabilizer() {
        use pecos_core::Zs;

        // Test that weight-2 stabilizer is handled correctly
        let code = StabilizerCodeSpec::builder(3)
            .check(Zs([0, 2])) // Only Z on qubits 0 and 2
            .build()
            .unwrap();

        let stab = &code.stabilizers()[0];
        assert_eq!(stab.weight(), 2);
    }

    #[test]
    fn test_builder_with_operators() {
        use pecos_core::{Xs, Zs};

        // Build a 3-qubit bit flip code using operators
        let code = StabilizerCodeSpec::builder(3)
            .check(Zs(0..=1))
            .check(Zs(1..=2))
            .logical_z(Zs(0..=2))
            .logical_x(Xs(0..=2))
            .build()
            .unwrap();

        assert_eq!(code.num_qubits(), 3);
        assert_eq!(code.num_logical_qubits(), 1);
        assert_eq!(code.num_stabilizers(), 2);
        assert!(code.verify().is_ok());
    }

    #[test]
    fn test_builder_with_mixed_operators() {
        use pecos_core::{Xs, Zs};

        // Build using tensor product of Paulis
        let code = StabilizerCodeSpec::builder(4)
            .check(Xs(0..=1) & Zs(2..=3)) // XXZZ
            .build()
            .unwrap();

        let stab = &code.stabilizers()[0];
        assert_eq!(stab.weight(), 4);
    }

    #[test]
    fn test_builder_steane_with_operators() {
        use pecos_core::{Xs, Zs};

        // Build Steane code using operators
        let code = StabilizerCodeSpecBuilder::new(7)
            // X-type stabilizers (using specific qubit sets matching the Hamming code)
            .check(Xs([0, 2, 4, 6]))
            .check(Xs([1, 2, 5, 6]))
            .check(Xs([3, 4, 5, 6]))
            // Z-type stabilizers
            .check(Zs([0, 2, 4, 6]))
            .check(Zs([1, 2, 5, 6]))
            .check(Zs([3, 4, 5, 6]))
            // Logical operators
            .logical_z(Zs(0..=6))
            .logical_x(Xs(0..=6))
            .build_verified()
            .unwrap();

        assert_eq!(code.num_qubits(), 7);
        assert_eq!(code.num_logical_qubits(), 1);
        assert!(code.verify().is_ok());
    }

    #[test]
    fn test_discover_logicals() {
        use pecos_core::Zs;

        // Create a code with just stabilizers (3-qubit bit flip code)
        let mut code = StabilizerCodeSpec::from_stabilizers(
            3,
            vec![
                Zs([0, 1]), // ZZI
                Zs([1, 2]), // IZZ
            ],
        );

        assert!(!code.has_logicals());

        // Discover logical operators
        code.discover_logicals().unwrap();

        assert!(code.has_logicals());
        assert_eq!(code.logical_zs().len(), 1);
        assert_eq!(code.logical_xs().len(), 1);

        // Verify the discovered logicals are valid
        assert!(code.verify().is_ok());
    }

    #[test]
    fn test_build_with_discovered_logicals() {
        use pecos_core::{Xs, Zs};

        // Build Steane code with auto-discovered logicals
        let code = StabilizerCodeSpecBuilder::new(7)
            .check(Xs([0, 2, 4, 6]))
            .check(Xs([1, 2, 5, 6]))
            .check(Xs([3, 4, 5, 6]))
            .check(Zs([0, 2, 4, 6]))
            .check(Zs([1, 2, 5, 6]))
            .check(Zs([3, 4, 5, 6]))
            .build_with_discovered_logicals()
            .unwrap();

        assert_eq!(code.num_qubits(), 7);
        assert_eq!(code.num_logical_qubits(), 1);
        assert_eq!(code.logical_zs().len(), 1);
        assert_eq!(code.logical_xs().len(), 1);

        // Verify the discovered logicals are valid
        assert!(code.verify().is_ok());
    }

    // ========================================================================
    // Bridge conversion tests
    // ========================================================================

    #[test]
    fn test_to_stabilizer_group() {
        use pecos_core::Zs;

        let code = StabilizerCodeSpec::builder(3)
            .check(Zs([0, 1]))
            .check(Zs([1, 2]))
            .logical_z(Zs([0, 1, 2]))
            .logical_x(pecos_core::Xs([0]))
            .build()
            .unwrap();

        let group = code.to_stabilizer_group().unwrap();
        assert_eq!(group.rank(), 2);
        assert_eq!(group.num_qubits(), 3);
    }

    #[test]
    fn test_from_stabilizer_group() {
        let steane = crate::StabilizerCode::steane();
        let code = StabilizerCodeSpec::from_stabilizer_group(steane.group());

        assert_eq!(code.num_qubits(), 7);
        assert_eq!(code.num_stabilizers(), 6);
        assert!(code.verify_stabilizers_commute().is_ok());
        assert!(!code.has_logicals());
    }

    #[test]
    fn test_roundtrip_stabilizer_group() {
        use pecos_core::{Xs, Zs};

        let original = StabilizerCodeSpec::builder(7)
            .check(Xs([0, 2, 4, 6]))
            .check(Xs([1, 2, 5, 6]))
            .check(Xs([3, 4, 5, 6]))
            .check(Zs([0, 2, 4, 6]))
            .check(Zs([1, 2, 5, 6]))
            .check(Zs([3, 4, 5, 6]))
            .build()
            .unwrap();

        let group = original.to_stabilizer_group().unwrap();
        let roundtripped = StabilizerCodeSpec::from_stabilizer_group(&group);

        assert_eq!(roundtripped.num_qubits(), original.num_qubits());
        assert_eq!(roundtripped.num_stabilizers(), original.num_stabilizers());
        assert!(roundtripped.verify_stabilizers_commute().is_ok());
    }

    #[test]
    fn test_stabilizer_group_algebraic_analysis() {
        use pecos_core::pauli::constructors::*;

        // Use SurfaceCode and convert for algebraic analysis
        let surface = crate::SurfaceCode::rotated(3).unwrap();
        let stab_code = surface.to_stabilizer_code();
        let group = stab_code.to_stabilizer_group().unwrap();

        // Algebraic properties
        assert_eq!(group.num_qubits(), 9);
        assert!(group.is_independent());

        // Wrap in StabilizerCodeSpec for QEC analysis
        let code = crate::StabilizerCode::from_group(group);
        assert_eq!(code.num_logical_qubits(), 1);

        // Syndrome computation
        let syndrome = code.syndrome(&X(0));
        assert!(
            syndrome.iter().any(|&s| s),
            "X error should trigger at least one stabilizer"
        );
    }

    // ========================================================================
    // StabilizerCode -> StabilizerCodeSpec conversion tests
    // ========================================================================

    #[test]
    fn test_from_stabilizer_code_steane() {
        let code = crate::StabilizerCode::steane();
        let spec = StabilizerCodeSpec::from_stabilizer_code(&code).unwrap();

        assert_eq!(spec.num_qubits(), 7);
        assert_eq!(spec.num_logical_qubits(), 1);
        assert_eq!(spec.num_stabilizers(), 6);
        assert_eq!(spec.logical_zs().len(), 1);
        assert_eq!(spec.logical_xs().len(), 1);
        assert!(spec.verify().is_ok());
    }

    #[test]
    fn test_from_stabilizer_code_repetition() {
        let code = crate::StabilizerCode::repetition(5);
        let spec = StabilizerCodeSpec::from_stabilizer_code(&code).unwrap();

        assert_eq!(spec.num_qubits(), 5);
        assert_eq!(spec.num_logical_qubits(), 1);
        assert_eq!(spec.logical_zs().len(), 1);
        assert_eq!(spec.logical_xs().len(), 1);
        assert!(spec.verify().is_ok());
    }

    #[test]
    fn test_from_stabilizer_code_five_qubit() {
        let code = crate::StabilizerCode::five_qubit();
        let spec = StabilizerCodeSpec::from_stabilizer_code(&code).unwrap();

        assert_eq!(spec.num_qubits(), 5);
        assert_eq!(spec.num_logical_qubits(), 1);
        assert!(spec.verify().is_ok());
    }

    #[test]
    fn test_from_stabilizer_code_four_two_two() {
        let code = crate::StabilizerCode::four_two_two();
        let spec = StabilizerCodeSpec::from_stabilizer_code(&code).unwrap();

        assert_eq!(spec.num_qubits(), 4);
        assert_eq!(spec.num_logical_qubits(), 2);
        assert_eq!(spec.logical_zs().len(), 2);
        assert_eq!(spec.logical_xs().len(), 2);
        assert!(spec.verify().is_ok());
    }

    #[test]
    fn test_from_stabilizer_code_has_destabilizers() {
        let code = crate::StabilizerCode::steane();
        let spec = StabilizerCodeSpec::from_stabilizer_code(&code).unwrap();

        // discover_logicals also computes destabilizers
        assert_eq!(spec.destabilizers().len(), spec.num_stabilizers());
    }

    #[test]
    fn test_builder_from_stabilizer_code() {
        let code = crate::StabilizerCode::steane();
        let spec = StabilizerCodeSpecBuilder::from_stabilizer_code(&code)
            .build_with_discovered_logicals()
            .unwrap();

        assert_eq!(spec.num_qubits(), 7);
        assert_eq!(spec.num_logical_qubits(), 1);
        assert_eq!(spec.logical_zs().len(), 1);
        assert!(spec.verify().is_ok());
    }

    #[test]
    fn test_builder_from_stabilizer_code_with_manual_logicals() {
        use pecos_core::{Xs, Zs};

        let code = crate::StabilizerCode::repetition(3);
        let spec = StabilizerCodeSpecBuilder::from_stabilizer_code(&code)
            .logical_z(Zs([0, 1, 2]))
            .logical_x(Xs([0, 1, 2]))
            .build_verified()
            .unwrap();

        assert_eq!(spec.num_qubits(), 3);
        assert_eq!(spec.num_logical_qubits(), 1);
    }

    #[test]
    fn test_from_stabilizer_code_preserves_explicit_num_qubits() {
        use pecos_core::pauli::constructors::Z;
        use pecos_quantum::PauliStabilizerGroup;

        // StabilizerCode with explicit num_qubits > group touches
        let group = PauliStabilizerGroup::new(vec![Z(0)]).unwrap();
        let code = crate::StabilizerCode::new(group, 5);
        assert_eq!(code.num_qubits(), 5);
        assert_eq!(code.num_logical_qubits(), 4);

        let spec = StabilizerCodeSpec::from_stabilizer_code(&code).unwrap();
        assert_eq!(spec.num_qubits(), 5);
        assert_eq!(spec.num_logical_qubits(), 4);
        assert_eq!(spec.logical_zs().len(), 4);
        assert_eq!(spec.logical_xs().len(), 4);
        assert!(spec.verify().is_ok());
    }
}
