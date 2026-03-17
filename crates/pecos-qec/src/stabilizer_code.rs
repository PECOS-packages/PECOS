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

//! A stabilizer code: a [`PauliStabilizerGroup`] together with an explicit qubit count.
//!
//! [`StabilizerCode`] adds QEC-specific analysis (logical qubits, distance,
//! syndrome, logical operators) on top of the algebraic [`PauliStabilizerGroup`].
//! The explicit `num_qubits` is necessary because the stabilizer generators may
//! not touch all physical qubits, and code parameters depend on the full system size.
//!
//! # Examples
//!
//! ```
//! use pecos_qec::StabilizerCode;
//!
//! let code = StabilizerCode::repetition(3);
//! assert_eq!(code.num_qubits(), 3);
//! assert_eq!(code.num_logical_qubits(), 1);
//! assert_eq!(code.code_parameters(), "[[3, 1]]");
//! assert_eq!(code.distance(), Some(1));
//! ```

use pecos_core::{Pauli, PauliOperator, PauliString, QuarterPhase, QubitId};
use pecos_quantum::F2Matrix;
use pecos_quantum::PauliStabilizerGroup;

/// Converts a binary symplectic vector `(x_0..x_{n-1} | z_0..z_{n-1})` to a `PauliString`.
fn symplectic_vec_to_pauli(vec: &[u8], n: usize) -> PauliString {
    let mut paulis = Vec::new();
    for q in 0..n {
        let x = vec[q];
        let z = vec[n + q];
        let pauli = match (x, z) {
            (1, 0) => Pauli::X,
            (0, 1) => Pauli::Z,
            (1, 1) => Pauli::Y,
            _ => continue,
        };
        paulis.push((pauli, QubitId::new(q)));
    }
    PauliString::with_phase_and_paulis(QuarterPhase::PlusOne, paulis)
}

/// A stabilizer code: a [`PauliStabilizerGroup`] with an explicit qubit count.
///
/// This provides QEC-specific analysis methods that require knowing the total
/// number of physical qubits in the system.
#[derive(Debug, Clone)]
pub struct StabilizerCode {
    group: PauliStabilizerGroup,
    num_qubits: usize,
}

impl StabilizerCode {
    /// Creates a stabilizer code from a group and explicit qubit count.
    ///
    /// # Panics
    ///
    /// Panics if `num_qubits < group.num_qubits()`.
    #[must_use]
    pub fn new(group: PauliStabilizerGroup, num_qubits: usize) -> Self {
        assert!(
            num_qubits >= group.num_qubits(),
            "num_qubits ({num_qubits}) must be >= group.num_qubits() ({})",
            group.num_qubits()
        );
        Self { group, num_qubits }
    }

    /// Creates a stabilizer code from a group, inferring `num_qubits` from the generators.
    #[must_use]
    pub fn from_group(group: PauliStabilizerGroup) -> Self {
        let num_qubits = group.num_qubits();
        Self { group, num_qubits }
    }

    /// Returns a reference to the underlying stabilizer group.
    #[must_use]
    pub fn group(&self) -> &PauliStabilizerGroup {
        &self.group
    }

    /// Consumes this code and returns the underlying stabilizer group.
    #[must_use]
    pub fn into_group(self) -> PauliStabilizerGroup {
        self.group
    }

    /// Returns the number of physical qubits.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Returns the number of logical qubits: `n - rank`.
    #[must_use]
    pub fn num_logical_qubits(&self) -> usize {
        self.num_qubits.saturating_sub(self.group.rank())
    }

    /// Returns the code parameters as `[[n, k]]` where n is physical qubits and k is logical qubits.
    #[must_use]
    pub fn code_parameters(&self) -> String {
        let n = self.num_qubits;
        let k = self.num_logical_qubits();
        format!("[[{n}, {k}]]")
    }

    /// Returns a basis for the logical operators of the stabilizer code.
    ///
    /// These are Pauli strings that commute with all stabilizers but are not
    /// in the stabilizer group (i.e., they act non-trivially on the code space).
    /// The returned vectors are in binary symplectic form (length `2n`).
    ///
    /// For an `[[n, k]]` code, the logical subspace has dimension `2k`:
    /// `k` logical X operators and `k` logical Z operators.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_qec::StabilizerCode;
    /// use pecos_core::pauli::constructors::*;
    ///
    /// // Repetition code [[3,1]]: logicals are X_L = XXX, Z_L = Z on any qubit
    /// let code = StabilizerCode::repetition(3);
    /// let logicals = code.logical_operators();
    /// // 2k = 2 independent logical directions (X_L and Z_L)
    /// assert_eq!(logicals.len(), 2);
    /// ```
    #[must_use]
    pub fn logical_operators(&self) -> Vec<PauliString> {
        let n = self.num_qubits;
        let centralizer_basis = self.group.as_collection().centralizer_in(n);

        // Build the stabilizer symplectic matrix at the full num_qubits size,
        // since num_qubits may be larger than what the generators touch.
        let num_generators = self.group.stabilizers().len();
        let mut stab_mat = F2Matrix::zeros(num_generators, 2 * n);
        for (row_idx, stab) in self.group.stabilizers().iter().enumerate() {
            for q in stab.x_positions() {
                stab_mat.row_mut(row_idx)[q] = 1;
            }
            for q in stab.z_positions() {
                stab_mat.row_mut(row_idx)[n + q] = 1;
            }
        }
        let (stab_rref, stab_pivots) = stab_mat.row_reduce();

        // For each centralizer basis vector, reduce it modulo the stabilizer RREF.
        // If the residual is non-zero, it's a genuine logical operator.
        // Then reduce logicals among themselves to get an independent set.
        let mut logical_vecs: Vec<Vec<u8>> = Vec::new();

        for cvec in &centralizer_basis {
            let mut v = cvec.clone();

            // Reduce using stabilizer RREF
            for (row_idx, &pivot_col) in stab_pivots.iter().enumerate() {
                if v[pivot_col] == 1 {
                    for (col, vi) in v.iter_mut().enumerate() {
                        *vi ^= stab_rref.row(row_idx)[col];
                    }
                }
            }

            // If residual is non-zero, this is a logical direction
            if v.iter().any(|&b| b != 0) {
                logical_vecs.push(v);
            }
        }

        // Row-reduce the logical vectors to get an independent set
        if logical_vecs.len() > 1 {
            let mut log_mat = F2Matrix::zeros(logical_vecs.len(), 2 * n);
            for (i, v) in logical_vecs.iter().enumerate() {
                log_mat.row_mut(i).clone_from(v);
            }
            let (reduced, _) = log_mat.row_reduce();
            logical_vecs = (0..reduced.num_rows())
                .map(|i| reduced.row(i).to_vec())
                .filter(|r| r.iter().any(|&b| b != 0))
                .collect();
        }

        logical_vecs
            .iter()
            .map(|v| symplectic_vec_to_pauli(v, n))
            .collect()
    }

    /// Computes the code distance for small codes.
    ///
    /// The distance is the minimum weight of a non-trivial logical operator
    /// (a Pauli that commutes with all stabilizers but is not in the stabilizer group).
    ///
    /// Returns `None` if there are no logical qubits (k = 0).
    ///
    /// **Complexity**: O(2^k * 2^r) where k = number of logical operators and
    /// r = rank. Only suitable for small codes.
    ///
    /// # Panics
    ///
    /// Panics if `k + rank > 30` to prevent accidental exponential blowup.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_qec::StabilizerCode;
    /// use pecos_core::pauli::constructors::*;
    ///
    /// // Repetition code [[3,1,1]]: distance 1 (logical Z = Z on any single qubit)
    /// let code = StabilizerCode::repetition(3);
    /// assert_eq!(code.distance(), Some(1));
    /// ```
    #[must_use]
    pub fn distance(&self) -> Option<usize> {
        let logicals = self.logical_operators();
        if logicals.is_empty() {
            return None;
        }

        let n = self.num_qubits;
        let k = logicals.len();

        // Get the stabilizer generators in reduced form for coset optimization
        let reduced = self.group.row_reduce();
        let stab_paulis: Vec<&PauliString> = reduced.paulis().iter().collect();
        let r = stab_paulis.len();

        assert!(
            k + r <= 30,
            "distance() is O(2^(k+r)) and would enumerate 2^{} combinations; \
             use a different algorithm for large codes",
            k + r,
        );

        let mut min_weight = n + 1; // upper bound

        // For each non-zero combination of logical operators...
        for logical_mask in 1u64..(1u64 << k) {
            // Build the logical operator from combination of basis logicals
            let mut logical = PauliString::identity();
            for (i, log) in logicals.iter().enumerate() {
                if logical_mask & (1u64 << i) != 0 {
                    logical = logical * log.clone();
                }
            }

            // Try all combinations of stabilizers to minimize weight
            // (multiply by stabilizer elements to find minimum weight representative)
            for stab_mask in 0u64..(1u64 << r) {
                let mut candidate = logical.clone();
                for (i, stab) in stab_paulis.iter().enumerate() {
                    if stab_mask & (1u64 << i) != 0 {
                        candidate = candidate * (*stab).clone();
                    }
                }
                let w = candidate.weight();
                if w < min_weight {
                    min_weight = w;
                }
            }
        }

        Some(min_weight)
    }

    /// Computes the syndrome of an error Pauli against the stabilizer generators.
    ///
    /// Returns a binary vector of length `num_generators()` where entry `i` is `true`
    /// if the error anticommutes with generator `i` (i.e., would trigger that detector).
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_qec::StabilizerCode;
    /// use pecos_core::pauli::constructors::*;
    ///
    /// // Repetition code: ZZI, IZZ on 3 qubits
    /// let code = StabilizerCode::repetition(3);
    ///
    /// // X error on qubit 0 triggers first stabilizer only
    /// assert_eq!(code.syndrome(&X(0)), vec![true, false]);
    ///
    /// // X error on qubit 1 triggers both stabilizers
    /// assert_eq!(code.syndrome(&X(1)), vec![true, true]);
    ///
    /// // Z error commutes with all Z-stabilizers
    /// assert_eq!(code.syndrome(&Z(0)), vec![false, false]);
    /// ```
    #[must_use]
    pub fn syndrome(&self, error: &PauliString) -> Vec<bool> {
        self.group
            .stabilizers()
            .iter()
            .map(|stab| !stab.commutes_with(error))
            .collect()
    }

    /// Transforms all generators by a Clifford gate: each `g_i` -> `C g_i C†`.
    ///
    /// Returns a new `StabilizerCode` with the same `num_qubits` and transformed group.
    #[must_use]
    pub fn apply_clifford(
        &self,
        clifford: &pecos_core::clifford_rep::CliffordRep,
    ) -> StabilizerCode {
        StabilizerCode {
            group: self.group.apply_clifford(clifford),
            num_qubits: self.num_qubits,
        }
    }

    // ========================================================================
    // Standard code constructors
    // ========================================================================

    /// Creates the `[[n, 1, n]]` bit-flip repetition code on `n` qubits.
    ///
    /// Generators: `Z_i Z_{i+1}` for `i = 0..n-2`.
    ///
    /// This code detects (and corrects up to `(n-1)/2`) bit-flip (X) errors
    /// but provides no protection against phase (Z) errors.
    ///
    /// # Panics
    ///
    /// Panics if `n < 2`.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_qec::StabilizerCode;
    ///
    /// let code = StabilizerCode::repetition(3);
    /// assert_eq!(code.group().rank(), 2);
    /// assert_eq!(code.num_logical_qubits(), 1);
    /// assert_eq!(code.distance(), Some(1)); // Z-distance is 1
    /// ```
    #[must_use]
    pub fn repetition(n: usize) -> Self {
        assert!(
            n >= 2,
            "repetition code requires at least 2 qubits, got {n}"
        );
        use pecos_core::pauli::constructors::Zs;
        let generators: Vec<PauliString> = (0..n - 1).map(|i| Zs([i, i + 1])).collect();
        Self {
            group: PauliStabilizerGroup::from_generators_unchecked(generators),
            num_qubits: n,
        }
    }

    /// Creates the `[[7, 1, 3]]` Steane code.
    ///
    /// The Steane code is a CSS code based on the classical `[7,4,3]` Hamming code.
    /// It has 6 generators (3 X-type, 3 Z-type) and encodes 1 logical qubit
    /// into 7 physical qubits with distance 3.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_qec::StabilizerCode;
    ///
    /// let code = StabilizerCode::steane();
    /// assert_eq!(code.group().rank(), 6);
    /// assert_eq!(code.num_logical_qubits(), 1);
    /// assert_eq!(code.distance(), Some(3));
    /// ```
    #[must_use]
    pub fn steane() -> Self {
        use pecos_core::pauli::constructors::{Xs, Zs};
        let generators = vec![
            Xs([0, 2, 4, 6]),
            Xs([1, 2, 5, 6]),
            Xs([3, 4, 5, 6]),
            Zs([0, 2, 4, 6]),
            Zs([1, 2, 5, 6]),
            Zs([3, 4, 5, 6]),
        ];
        Self {
            group: PauliStabilizerGroup::from_generators_unchecked(generators),
            num_qubits: 7,
        }
    }

    /// Creates the `[[5, 1, 3]]` perfect code.
    ///
    /// The smallest code that can correct an arbitrary single-qubit error.
    /// It saturates the quantum Hamming bound and is not a CSS code.
    ///
    /// Generators: `XZZXI`, `IXZZX`, `XIXZZ`, `ZXIXZ`
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_qec::StabilizerCode;
    ///
    /// let code = StabilizerCode::five_qubit();
    /// assert_eq!(code.group().rank(), 4);
    /// assert_eq!(code.num_logical_qubits(), 1);
    /// assert_eq!(code.distance(), Some(3));
    /// ```
    #[must_use]
    pub fn five_qubit() -> Self {
        use pecos_core::pauli::constructors::{X, Z};
        let generators = vec![
            X(0) & Z(1) & Z(2) & X(3), // XZZXI
            X(1) & Z(2) & Z(3) & X(4), // IXZZX
            X(0) & X(2) & Z(3) & Z(4), // XIXZZ
            Z(0) & X(1) & X(3) & Z(4), // ZXIXZ
        ];
        Self {
            group: PauliStabilizerGroup::from_generators_unchecked(generators),
            num_qubits: 5,
        }
    }

    /// Creates the `[[9, 1, 3]]` Shor code.
    ///
    /// The first quantum error correcting code, using a concatenation of
    /// the 3-qubit bit-flip and phase-flip codes.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_qec::StabilizerCode;
    ///
    /// let code = StabilizerCode::shor();
    /// assert_eq!(code.group().rank(), 8);
    /// assert_eq!(code.num_logical_qubits(), 1);
    /// assert_eq!(code.distance(), Some(3));
    /// ```
    #[must_use]
    pub fn shor() -> Self {
        use pecos_core::pauli::constructors::{Xs, Zs};
        let generators = vec![
            Xs([0, 1]),
            Xs([1, 2]),
            Xs([3, 4]),
            Xs([4, 5]),
            Xs([6, 7]),
            Xs([7, 8]),
            Zs([0, 1, 2, 3, 4, 5]),
            Zs([3, 4, 5, 6, 7, 8]),
        ];
        Self {
            group: PauliStabilizerGroup::from_generators_unchecked(generators),
            num_qubits: 9,
        }
    }

    /// Creates the `[[4, 2, 2]]` detection code.
    ///
    /// The smallest code that can detect a single arbitrary error but cannot
    /// correct it. Encodes 2 logical qubits into 4 physical qubits.
    ///
    /// Generators: `XXXX`, `ZZZZ`
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_qec::StabilizerCode;
    ///
    /// let code = StabilizerCode::four_two_two();
    /// assert_eq!(code.group().rank(), 2);
    /// assert_eq!(code.num_logical_qubits(), 2);
    /// assert_eq!(code.distance(), Some(2));
    /// ```
    #[must_use]
    pub fn four_two_two() -> Self {
        use pecos_core::pauli::constructors::{Xs, Zs};
        let generators = vec![Xs([0, 1, 2, 3]), Zs([0, 1, 2, 3])];
        Self {
            group: PauliStabilizerGroup::from_generators_unchecked(generators),
            num_qubits: 4,
        }
    }

    /// Creates the toric code on an `L x L` torus with distance `L`.
    ///
    /// The toric code is a CSS code on a periodic square lattice with
    /// `2 * L^2` physical qubits encoding 2 logical qubits.
    ///
    /// # Panics
    ///
    /// Panics if `L < 2`.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_qec::StabilizerCode;
    ///
    /// let code = StabilizerCode::toric(3);
    /// assert_eq!(code.num_qubits(), 18);        // 2 * 3^2
    /// assert_eq!(code.num_logical_qubits(), 2);  // torus encodes 2 logicals
    /// assert_eq!(code.distance(), Some(3));
    /// ```
    #[must_use]
    pub fn toric(l: usize) -> Self {
        assert!(l >= 2, "toric code requires L >= 2, got {l}");
        use pecos_core::pauli::constructors::{Xs, Zs};

        let n = 2 * l * l;
        let horiz = |r: usize, c: usize| r * l + c;
        let vert = |r: usize, c: usize| l * l + r * l + c;

        let mut generators = Vec::new();

        // Vertex (star) stabilizers: X on the 4 edges touching vertex (r, c)
        for r in 0..l {
            for c in 0..l {
                if r == l - 1 && c == l - 1 {
                    continue; // skip last vertex (redundant)
                }
                let qubits = [
                    horiz(r, c),
                    horiz(r, (c + l - 1) % l),
                    vert(r, c),
                    vert((r + l - 1) % l, c),
                ];
                generators.push(Xs(qubits));
            }
        }

        // Plaquette (face) stabilizers: Z on the 4 edges around face (r, c)
        for r in 0..l {
            for c in 0..l {
                if r == l - 1 && c == l - 1 {
                    continue; // skip last plaquette (redundant)
                }
                let qubits = [
                    horiz(r, c),
                    horiz((r + 1) % l, c),
                    vert(r, c),
                    vert(r, (c + 1) % l),
                ];
                generators.push(Zs(qubits));
            }
        }

        Self {
            group: PauliStabilizerGroup::from_generators_unchecked(generators),
            num_qubits: n,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::pauli::constructors::*;

    // ========================================================================
    // Basic code parameter tests
    // ========================================================================

    #[test]
    fn test_repetition_code() {
        let code = StabilizerCode::from_group(
            PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])]).unwrap(),
        );
        assert_eq!(code.group().rank(), 2);
        assert_eq!(code.num_logical_qubits(), 1);
        assert_eq!(code.code_parameters(), "[[3, 1]]");
    }

    #[test]
    fn test_steane_code() {
        let code = StabilizerCode::from_group(
            PauliStabilizerGroup::new(vec![
                Xs([0, 2, 4, 6]),
                Xs([1, 2, 5, 6]),
                Xs([3, 4, 5, 6]),
                Zs([0, 2, 4, 6]),
                Zs([1, 2, 5, 6]),
                Zs([3, 4, 5, 6]),
            ])
            .unwrap(),
        );
        assert_eq!(code.group().rank(), 6);
        assert_eq!(code.num_logical_qubits(), 1);
        assert_eq!(code.code_parameters(), "[[7, 1]]");
    }

    #[test]
    fn test_five_qubit_code() {
        let code = StabilizerCode::from_group(
            PauliStabilizerGroup::new(vec![
                X(0) & Z(1) & Z(2) & X(3),
                X(1) & Z(2) & Z(3) & X(4),
                X(0) & X(2) & Z(3) & Z(4),
                Z(0) & X(1) & X(3) & Z(4),
            ])
            .unwrap(),
        );
        assert_eq!(code.group().rank(), 4);
        assert_eq!(code.num_logical_qubits(), 1);
        assert_eq!(code.code_parameters(), "[[5, 1]]");
    }

    // ========================================================================
    // Syndrome tests
    // ========================================================================

    #[test]
    fn test_syndrome_repetition_code() {
        let code = StabilizerCode::repetition(3);

        assert_eq!(code.syndrome(&X(0)), vec![true, false]);
        assert_eq!(code.syndrome(&X(1)), vec![true, true]);
        assert_eq!(code.syndrome(&X(2)), vec![false, true]);
        assert_eq!(code.syndrome(&Z(0)), vec![false, false]);
        assert_eq!(code.syndrome(&Z(1)), vec![false, false]);
    }

    #[test]
    fn test_syndrome_steane_code() {
        let code = StabilizerCode::steane();

        let syn = code.syndrome(&Z(0));
        assert!(syn[0]); // X on {0,2,4,6}
        assert!(!syn[1]); // X on {1,2,5,6}
        assert!(!syn[2]); // X on {3,4,5,6}
        assert!(!syn[3]);
        assert!(!syn[4]);
        assert!(!syn[5]);
    }

    #[test]
    fn test_syndrome_y_error() {
        let code = StabilizerCode::repetition(3);
        let syn = code.syndrome(&Y(1));
        assert_eq!(syn, vec![true, true]);
    }

    #[test]
    fn test_syndrome_multi_qubit_error() {
        let code = StabilizerCode::repetition(3);
        let error = X(0) & X(2);
        let syn = code.syndrome(&error);
        assert_eq!(syn, vec![true, true]);
    }

    #[test]
    fn test_syndrome_stabilizer_element() {
        let code = StabilizerCode::repetition(3);
        let syn = code.syndrome(&Zs([0, 1]));
        assert_eq!(syn, vec![false, false]);
    }

    #[test]
    fn test_syndrome_identity_error() {
        let code = StabilizerCode::repetition(3);
        let id = PauliString::identity();
        let s = code.syndrome(&id);
        assert!(s.iter().all(|&b| !b), "identity should have zero syndrome");
    }

    // ========================================================================
    // Logical operator tests
    // ========================================================================

    #[test]
    fn test_logical_operators_repetition_code() {
        let code = StabilizerCode::repetition(3);
        let logicals = code.logical_operators();
        assert_eq!(logicals.len(), 2);
    }

    #[test]
    fn test_logical_operators_steane_code() {
        let code = StabilizerCode::steane();
        let logicals = code.logical_operators();
        assert_eq!(logicals.len(), 2);
    }

    #[test]
    fn test_logical_operators_five_qubit_code() {
        let code = StabilizerCode::five_qubit();
        let logicals = code.logical_operators();
        assert_eq!(logicals.len(), 2);
        // Each logical should commute with all stabilizers
        for l in &logicals {
            for s in code.group().iter() {
                assert!(l.commutes_with(s));
            }
        }
        // Logicals should NOT be in the stabilizer group
        for l in &logicals {
            assert!(!code.group().contains(l));
        }
    }

    #[test]
    fn test_logical_operators_commute_with_stabilizers() {
        let code = StabilizerCode::steane();
        for l in code.logical_operators() {
            for s in code.group().iter() {
                assert!(
                    l.commutes_with(s),
                    "logical {} anticommutes with stabilizer {}",
                    l.to_sparse_str(),
                    s.to_sparse_str()
                );
            }
        }
    }

    #[test]
    fn test_logical_operators_anticommute_with_each_other() {
        let code = StabilizerCode::repetition(3);
        let logicals = code.logical_operators();
        assert_eq!(logicals.len(), 2);
        let mut found_anticommuting = false;
        for i in 0..logicals.len() {
            for j in (i + 1)..logicals.len() {
                if logicals[i].anticommutes_with(&logicals[j]) {
                    found_anticommuting = true;
                }
            }
        }
        assert!(found_anticommuting, "logical X and Z should anticommute");
    }

    #[test]
    fn logical_operators_full_rank() {
        let code = StabilizerCode::from_group(PauliStabilizerGroup::new(vec![Z(0), Z(1)]).unwrap());
        assert_eq!(code.num_logical_qubits(), 0);
        let logicals = code.logical_operators();
        assert!(logicals.is_empty());
    }

    // ========================================================================
    // Distance tests
    // ========================================================================

    #[test]
    fn test_distance_repetition_code() {
        let code = StabilizerCode::repetition(3);
        assert_eq!(code.distance(), Some(1));
    }

    #[test]
    fn test_distance_steane_code() {
        let code = StabilizerCode::steane();
        assert_eq!(code.distance(), Some(3));
    }

    #[test]
    fn test_distance_no_logicals() {
        let code = StabilizerCode::from_group(PauliStabilizerGroup::new(vec![Z(0), Z(1)]).unwrap());
        assert_eq!(code.distance(), None);
    }

    #[test]
    fn test_distance_five_qubit_code() {
        let code = StabilizerCode::five_qubit();
        assert_eq!(code.distance(), Some(3));
    }

    #[test]
    fn test_distance_with_redundant_generators() {
        let code = StabilizerCode::from_group(
            PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2]), Zs([0, 2])]).unwrap(),
        );
        assert!(!code.group().is_independent());
        assert_eq!(code.distance(), Some(1));
    }

    #[test]
    fn test_distance_full_rank() {
        let code = StabilizerCode::from_group(PauliStabilizerGroup::new(vec![Z(0), Z(1)]).unwrap());
        assert_eq!(code.num_logical_qubits(), 0);
        assert_eq!(code.distance(), None);
    }

    // ========================================================================
    // Standard code constructor tests
    // ========================================================================

    #[test]
    fn test_repetition_code_constructor() {
        let code = StabilizerCode::repetition(3);
        assert_eq!(code.group().rank(), 2);
        assert_eq!(code.num_logical_qubits(), 1);
        assert_eq!(code.num_qubits(), 3);
        assert!(code.group().contains(&Zs([0, 1])));
        assert!(code.group().contains(&Zs([1, 2])));
        assert!(code.group().contains(&Zs([0, 2])));
    }

    #[test]
    fn test_repetition_code_distance() {
        let code = StabilizerCode::repetition(3);
        assert_eq!(code.distance(), Some(1));

        let code5 = StabilizerCode::repetition(5);
        assert_eq!(code5.group().rank(), 4);
        assert_eq!(code5.num_logical_qubits(), 1);
        assert_eq!(code5.distance(), Some(1));
    }

    #[test]
    fn test_repetition_code_n2() {
        let code = StabilizerCode::repetition(2);
        assert_eq!(code.group().rank(), 1);
        assert_eq!(code.num_logical_qubits(), 1);
    }

    #[test]
    #[should_panic(expected = "at least 2 qubits")]
    fn test_repetition_code_n1_panics() {
        let _ = StabilizerCode::repetition(1);
    }

    #[test]
    fn test_steane_code_constructor() {
        let code = StabilizerCode::steane();
        assert_eq!(code.group().rank(), 6);
        assert_eq!(code.num_logical_qubits(), 1);
        assert_eq!(code.num_qubits(), 7);
        assert_eq!(code.distance(), Some(3));
    }

    #[test]
    fn test_five_qubit_code_constructor() {
        let code = StabilizerCode::five_qubit();
        assert_eq!(code.group().rank(), 4);
        assert_eq!(code.num_logical_qubits(), 1);
        assert_eq!(code.num_qubits(), 5);
        assert_eq!(code.distance(), Some(3));
    }

    #[test]
    fn test_shor_code_constructor() {
        let code = StabilizerCode::shor();
        assert_eq!(code.group().rank(), 8);
        assert_eq!(code.num_logical_qubits(), 1);
        assert_eq!(code.num_qubits(), 9);
        assert_eq!(code.distance(), Some(3));
    }

    #[test]
    fn test_four_two_two_code_constructor() {
        let code = StabilizerCode::four_two_two();
        assert_eq!(code.num_qubits(), 4);
        assert_eq!(code.group().rank(), 2);
        assert_eq!(code.num_logical_qubits(), 2);
        assert_eq!(code.distance(), Some(2));
    }

    #[test]
    fn test_toric_code_l2() {
        let code = StabilizerCode::toric(2);
        assert_eq!(code.num_qubits(), 8);
        assert_eq!(code.num_logical_qubits(), 2);
        assert_eq!(code.distance(), Some(2));
    }

    #[test]
    fn test_toric_code_l3() {
        let code = StabilizerCode::toric(3);
        assert_eq!(code.num_qubits(), 18);
        assert_eq!(code.num_logical_qubits(), 2);
        assert_eq!(code.distance(), Some(3));
    }

    #[test]
    #[should_panic(expected = "toric code requires L >= 2")]
    fn test_toric_code_l1_panics() {
        let _ = StabilizerCode::toric(1);
    }

    #[test]
    fn test_standard_codes_are_valid() {
        for code in [
            StabilizerCode::repetition(5),
            StabilizerCode::steane(),
            StabilizerCode::five_qubit(),
            StabilizerCode::shor(),
            StabilizerCode::four_two_two(),
            StabilizerCode::toric(2),
            StabilizerCode::toric(3),
        ] {
            assert!(code.group().is_independent());
            let result = PauliStabilizerGroup::new(code.group().stabilizers().to_vec());
            assert!(
                result.is_ok(),
                "code with {} qubits failed validation",
                code.num_qubits()
            );
        }
    }

    // ========================================================================
    // apply_clifford tests
    // ========================================================================

    #[test]
    fn test_apply_clifford_preserves_code_parameters() {
        use pecos_core::clifford_rep::CliffordRep;

        let code = StabilizerCode::steane();

        let h0 = CliffordRep::h(0).extended_to(7);
        let transformed = code.apply_clifford(&h0);

        assert_eq!(transformed.num_qubits(), 7);
        assert_eq!(transformed.num_logical_qubits(), 1);
        assert_eq!(transformed.group().rank(), 6);
    }

    #[test]
    fn test_apply_clifford_transforms_generators() {
        use pecos_core::clifford_rep::CliffordRep;

        // Repetition code: ZZ stabilizers
        let code = StabilizerCode::repetition(3);
        assert!(code.group().contains(&Zs([0, 1])));

        // H on qubit 0 maps Z->X on that qubit: ZZI -> XZI
        let h0 = CliffordRep::h(0).extended_to(3);
        let transformed = code.apply_clifford(&h0);

        // ZZI should have become XZI (no longer in original group)
        assert!(!transformed.group().contains(&Zs([0, 1])));
        // But code parameters are preserved
        assert_eq!(transformed.num_logical_qubits(), 1);
        assert_eq!(transformed.group().rank(), 2);
    }

    // ========================================================================
    // Constructor validation tests
    // ========================================================================

    #[test]
    fn test_new_with_explicit_num_qubits() {
        let group = PauliStabilizerGroup::new(vec![Z(0)]).unwrap();
        let code = StabilizerCode::new(group, 5);
        assert_eq!(code.num_qubits(), 5);
        assert_eq!(code.num_logical_qubits(), 4);
    }

    #[test]
    fn test_explicit_num_qubits_affects_logicals_and_distance() {
        // Z(0) stabilizer on 3 qubits: [[3, 2]] code
        let group = PauliStabilizerGroup::new(vec![Z(0)]).unwrap();
        let code = StabilizerCode::new(group, 3);

        assert_eq!(code.num_logical_qubits(), 2);

        let logicals = code.logical_operators();
        assert_eq!(logicals.len(), 4); // 2k = 4
        // All logicals should commute with the single stabilizer Z(0)
        for l in &logicals {
            assert!(l.commutes_with(&Z(0)));
        }

        // Distance should be 1 (single-qubit X on qubit 0 is a logical)
        assert_eq!(code.distance(), Some(1));

        // Compare: same stabilizer but inferred num_qubits = 1
        let group2 = PauliStabilizerGroup::new(vec![Z(0)]).unwrap();
        let code2 = StabilizerCode::from_group(group2);

        assert_eq!(code2.num_qubits(), 1);
        assert_eq!(code2.num_logical_qubits(), 0);
        assert!(code2.logical_operators().is_empty());
        assert_eq!(code2.distance(), None);
    }

    #[test]
    fn test_syndrome_with_extra_qubits() {
        // Repetition code stabilizers on qubits 0,1,2 but code has 5 qubits
        let group = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])]).unwrap();
        let code = StabilizerCode::new(group, 5);

        // X error on a stabilizer qubit still triggers syndrome
        assert_eq!(code.syndrome(&X(1)), vec![true, true]);

        // X error on an extra qubit triggers no syndrome
        assert_eq!(code.syndrome(&X(3)), vec![false, false]);
        assert_eq!(code.syndrome(&X(4)), vec![false, false]);
    }

    #[test]
    fn test_from_group_infers_num_qubits() {
        let group = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])]).unwrap();
        let code = StabilizerCode::from_group(group);
        assert_eq!(code.num_qubits(), 3);
    }

    #[test]
    #[should_panic(expected = "num_qubits (1) must be >= group.num_qubits() (3)")]
    fn test_new_rejects_too_small_num_qubits() {
        let group = PauliStabilizerGroup::new(vec![Zs([0, 1]), Zs([1, 2])]).unwrap();
        let _ = StabilizerCode::new(group, 1);
    }
}
