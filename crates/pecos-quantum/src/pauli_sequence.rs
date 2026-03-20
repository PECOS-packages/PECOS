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

//! An ordered sequence of Pauli strings with symplectic analysis tools.
//!
//! [`PauliSequence`] stores an ordered sequence of [`PauliString`]s and provides
//! analysis operations using the binary symplectic representation over GF(2),
//! such as rank, linear independence, membership testing, and commutativity checks.
//!
//! No constraints are enforced -- the Pauli strings can anticommute,
//! have any [`QuarterPhase`] (`{+1, -1, +i, -i}`), or be redundant.
//! For a constrained version that enforces commutativity and real phases
//! ([`Sign`]: `{+1, -1}`), see [`PauliStabilizerGroup`].
//!
//! [`PauliString`]: pecos_core::PauliString
//! [`QuarterPhase`]: pecos_core::QuarterPhase
//! [`Sign`]: pecos_core::Sign
//! [`PauliStabilizerGroup`]: crate::PauliStabilizerGroup
//!
//! # Examples
//!
//! ```
//! use pecos_quantum::PauliSequence;
//! use pecos_core::pauli::constructors::*;
//!
//! let paulis = PauliSequence::new(vec![
//!     Zs(&[0, 1]),
//!     Zs(&[1, 2]),
//! ]);
//!
//! assert_eq!(paulis.rank(), 2);
//! assert!(paulis.is_abelian());
//!
//! // ZIZ is in the span (GF(2) linear combination)
//! assert!(paulis.contains(&Zs(&[0, 2])));
//! // XII is not
//! assert!(!paulis.contains(&X(0)));
//! ```

use pecos_core::{ParsePauliStringError, PauliOperator, PauliString};
use std::fmt;
use std::str::FromStr;

/// A binary matrix over GF(2), represented row-major.
///
/// Each row is a 2n-bit vector representing a Pauli string in the binary
/// symplectic representation: `(x_0, ..., x_{n-1} | z_0, ..., z_{n-1})`
/// where `x_q = 1` if qubit q has X or Y, and `z_q = 1` if qubit q has Z or Y.
///
// TODO: Each bit is stored as a full `u8`. For large codes, consider packing
// into `u64` words or using a bitvec for 8x memory reduction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct F2Matrix {
    pub(crate) rows: Vec<Vec<u8>>,
    num_cols: usize,
}

impl F2Matrix {
    /// Creates a new F2 matrix with the given dimensions, initialized to zero.
    #[must_use]
    pub fn zeros(num_rows: usize, num_cols: usize) -> Self {
        Self {
            rows: vec![vec![0; num_cols]; num_rows],
            num_cols,
        }
    }

    /// Returns the number of rows.
    #[must_use]
    pub fn num_rows(&self) -> usize {
        self.rows.len()
    }

    /// Returns the number of columns.
    #[must_use]
    pub fn num_cols(&self) -> usize {
        self.num_cols
    }

    /// Returns a reference to the rows.
    #[must_use]
    pub fn rows(&self) -> &[Vec<u8>] {
        &self.rows
    }

    /// Returns a reference to a specific row.
    #[must_use]
    pub fn row(&self, i: usize) -> &[u8] {
        &self.rows[i]
    }

    /// Returns a mutable reference to a specific row.
    pub fn row_mut(&mut self, i: usize) -> &mut Vec<u8> {
        &mut self.rows[i]
    }

    /// Checks if a row is all zeros.
    #[must_use]
    pub fn is_zero_row(&self, i: usize) -> bool {
        self.rows[i].iter().all(|&b| b == 0)
    }

    /// XORs row `src` into row `dst` (row[dst] ^= row[src]).
    fn xor_row(&mut self, dst: usize, src: usize) {
        // Avoid borrow issues by doing index operations
        for col in 0..self.num_cols {
            self.rows[dst][col] ^= self.rows[src][col];
        }
    }

    /// Swaps two rows.
    fn swap_rows(&mut self, a: usize, b: usize) {
        self.rows.swap(a, b);
    }

    /// Performs Gaussian elimination over GF(2), returning the row echelon form
    /// and the pivot column positions.
    ///
    /// Returns `(reduced_matrix, pivot_columns)` where `pivot_columns[i]` is the
    /// column of the pivot in row `i`.
    #[must_use]
    pub fn row_reduce(&self) -> (Self, Vec<usize>) {
        let mut mat = self.clone();
        let mut pivots = Vec::new();
        let mut pivot_row = 0;

        for col in 0..mat.num_cols {
            // Find a row with a 1 in this column at or below pivot_row
            let mut found = None;
            for row in pivot_row..mat.num_rows() {
                if mat.rows[row][col] == 1 {
                    found = Some(row);
                    break;
                }
            }

            let Some(found_row) = found else {
                continue;
            };

            // Swap to pivot position
            mat.swap_rows(pivot_row, found_row);

            // Eliminate all other rows with a 1 in this column
            for row in 0..mat.num_rows() {
                if row != pivot_row && mat.rows[row][col] == 1 {
                    mat.xor_row(row, pivot_row);
                }
            }

            pivots.push(col);
            pivot_row += 1;
        }

        (mat, pivots)
    }

    /// Creates an identity matrix of the given size.
    #[must_use]
    pub fn identity(n: usize) -> Self {
        let mut mat = Self::zeros(n, n);
        for i in 0..n {
            mat.rows[i][i] = 1;
        }
        mat
    }

    /// Inverts a square matrix over GF(2), if it is invertible.
    ///
    /// Returns `None` if the matrix is not square or not invertible.
    #[must_use]
    pub fn invert(&self) -> Option<Self> {
        let n = self.num_rows();
        if n != self.num_cols {
            return None;
        }

        // Augment [A | I]
        let mut aug = Self::zeros(n, 2 * n);
        for i in 0..n {
            for j in 0..n {
                aug.rows[i][j] = self.rows[i][j];
            }
            aug.rows[i][n + i] = 1;
        }

        // Row-reduce the augmented matrix.
        // For an invertible matrix, every column has a pivot, so pivot_row == col.
        for col in 0..n {
            // Find pivot in this column at or below the diagonal
            let mut found = None;
            for row in col..n {
                if aug.rows[row][col] == 1 {
                    found = Some(row);
                    break;
                }
            }
            let Some(found_row) = found else {
                return None; // Not invertible
            };

            aug.swap_rows(col, found_row);

            // Eliminate all other rows
            for row in 0..n {
                if row != col && aug.rows[row][col] == 1 {
                    aug.xor_row(row, col);
                }
            }
        }

        // Extract the inverse from the right half
        let mut inv = Self::zeros(n, n);
        for i in 0..n {
            inv.rows[i] = aug.rows[i][n..2 * n].to_vec();
        }
        Some(inv)
    }

    /// Multiplies two matrices over GF(2).
    ///
    /// # Panics
    ///
    /// Panics if `self.num_cols() != other.num_rows()`.
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        assert_eq!(self.num_cols, other.num_rows());
        let m = self.num_rows();
        let p = other.num_cols;
        let mut result = Self::zeros(m, p);
        for i in 0..m {
            for j in 0..p {
                let mut sum = 0u8;
                for k in 0..self.num_cols {
                    sum ^= self.rows[i][k] & other.rows[k][j];
                }
                result.rows[i][j] = sum;
            }
        }
        result
    }

    /// Transposes the matrix.
    #[must_use]
    pub fn transpose(&self) -> Self {
        let m = self.num_rows();
        let n = self.num_cols;
        let mut result = Self::zeros(n, m);
        for i in 0..m {
            for j in 0..n {
                result.rows[j][i] = self.rows[i][j];
            }
        }
        result
    }

    /// Computes the (right) null space of this matrix over GF(2).
    ///
    /// Returns a set of column vectors `v` such that `self * v = 0` (mod 2).
    /// Each returned vector has length `num_cols`.
    #[must_use]
    pub fn kernel(&self) -> Vec<Vec<u8>> {
        // Augment with identity: [A | I_cols]
        let m = self.num_rows();
        let n = self.num_cols;
        let mut aug = Self::zeros(m, n + n);
        for i in 0..m {
            for j in 0..n {
                aug.rows[i][j] = self.rows[i][j];
            }
        }
        // We actually need to work with the transpose to find the right kernel.
        // kernel(A) = { v : A * v = 0 } = kernel of rows of A^T.
        // Equivalently, row-reduce A^T augmented with identity, and null rows
        // give the kernel vectors.

        // Build A^T (n x m)
        let mut at = Self::zeros(n, m + n);
        for i in 0..m {
            for j in 0..n {
                at.rows[j][i] = self.rows[i][j];
            }
        }
        // Augment with identity in the right block
        for j in 0..n {
            at.rows[j][m + j] = 1;
        }

        // Row-reduce A^T
        let (reduced, _pivots) = at.row_reduce();

        // Rows that are zero in the A^T part (columns 0..m) give kernel vectors
        // from the identity part (columns m..m+n).
        let mut basis = Vec::new();
        for i in 0..n {
            if reduced.rows[i][..m].iter().all(|&b| b == 0) {
                // This row's right block is a kernel vector
                basis.push(reduced.rows[i][m..m + n].to_vec());
            }
        }

        // Only include non-zero vectors
        basis.retain(|v| v.iter().any(|&b| b != 0));
        // Remove duplicates (shouldn't happen with RREF but be safe)
        basis.sort();
        basis.dedup();

        // Handle overcounting: only keep linearly independent vectors
        if basis.len() > 1 {
            let check = Self {
                rows: basis.clone(),
                num_cols: n,
            };
            let (_, _ind_pivots) = check.row_reduce();
            // The first ind_pivots.len() rows of the reduced form are independent
            // but we want the original basis vectors. Since we sorted, just take
            // the independent count. Actually, let's re-reduce properly.
            let (reduced_basis, _) = check.row_reduce();
            basis = reduced_basis
                .rows
                .into_iter()
                .filter(|r| r.iter().any(|&b| b != 0))
                .collect();
        }

        basis
    }
}

impl fmt::Display for F2Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, row) in self.rows.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            // Show the X block and Z block separated by |
            let n = self.num_cols / 2;
            for (j, &bit) in row.iter().enumerate() {
                if j == n {
                    write!(f, "|")?;
                }
                write!(f, "{bit}")?;
            }
        }
        Ok(())
    }
}

/// An ordered sequence of [`PauliString`]s with symplectic analysis tools.
///
/// Each entry carries a [`QuarterPhase`] (`{+1, -1, +i, -i}`). No constraints
/// are enforced -- the Pauli strings can anticommute, have any phase, or be
/// linearly dependent. Analysis operations use the binary symplectic
/// representation over GF(2).
///
/// For a constrained version that enforces commutativity and restricts phases
/// to [`Sign`] (`{+1, -1}`), see [`PauliStabilizerGroup`].
///
/// [`PauliString`]: pecos_core::PauliString
/// [`QuarterPhase`]: pecos_core::QuarterPhase
/// [`Sign`]: pecos_core::Sign
/// [`PauliStabilizerGroup`]: crate::PauliStabilizerGroup
///
/// # Examples
///
/// ```
/// use pecos_quantum::PauliSequence;
/// use pecos_core::pauli::constructors::*;
/// use pecos_core::PauliOperator;
///
/// let gens = PauliSequence::new(vec![
///     Zs(&[0, 1]),
///     Zs(&[1, 2]),
/// ]);
///
/// assert_eq!(gens.len(), 2);
/// assert_eq!(gens.num_qubits(), 3);
/// assert_eq!(gens.rank(), 2);
/// assert!(gens.is_abelian());
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PauliSequence {
    paulis: Vec<PauliString>,
}

impl PauliSequence {
    /// Creates a new `PauliSequence` from a sequence of Pauli strings.
    #[must_use]
    pub fn new(paulis: Vec<PauliString>) -> Self {
        Self { paulis }
    }

    /// Creates a `PauliSequence` from string representations.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::PauliSequence;
    ///
    /// let paulis = PauliSequence::from_strs(&["ZZI", "IZZ"]).unwrap();
    /// assert_eq!(paulis.len(), 2);
    /// assert_eq!(paulis.num_qubits(), 3);
    /// ```
    ///
    /// # Errors
    /// Returns an error if any string cannot be parsed as a `PauliString`.
    pub fn from_strs(strings: &[&str]) -> Result<Self, pecos_core::ParsePauliStringError> {
        let paulis: Vec<PauliString> = strings
            .iter()
            .map(|s| s.parse())
            .collect::<Result<_, _>>()?;

        Ok(Self { paulis })
    }

    /// Returns a reference to the Pauli strings.
    #[must_use]
    pub fn paulis(&self) -> &[PauliString] {
        &self.paulis
    }

    /// Returns the number of Pauli strings.
    #[must_use]
    pub fn len(&self) -> usize {
        self.paulis.len()
    }

    /// Returns `true` if the sequence is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.paulis.is_empty()
    }

    /// Returns the number of qubits (inferred as max qubit index + 1).
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.paulis
            .iter()
            .flat_map(PauliString::qubits)
            .max()
            .map_or(0, |m| m + 1)
    }

    /// Appends a Pauli string to the sequence.
    pub fn push(&mut self, pauli: PauliString) {
        self.paulis.push(pauli);
    }

    /// Removes and returns the Pauli string at the given index.
    ///
    /// # Panics
    ///
    /// Panics if `index >= len()`.
    pub fn remove(&mut self, index: usize) -> PauliString {
        self.paulis.remove(index)
    }

    /// Extends the sequence with an iterator of Pauli strings.
    pub fn extend(&mut self, paulis: impl IntoIterator<Item = PauliString>) {
        self.paulis.extend(paulis);
    }

    /// Iterates over the Pauli strings.
    pub fn iter(&self) -> impl Iterator<Item = &PauliString> {
        self.paulis.iter()
    }

    /// Converts the Pauli strings to a binary symplectic matrix over GF(2).
    ///
    /// Each Pauli string becomes a row of length 2n, where n = `num_qubits`.
    /// The first n columns are the X bits, the last n columns are the Z bits.
    /// (Y on qubit q sets both `x_q` and `z_q` to 1.)
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::PauliSequence;
    ///
    /// // ZZI and IZZ on 3 qubits
    /// let gens = PauliSequence::from_strs(&["ZZI", "IZZ"]).unwrap();
    /// let mat = gens.to_symplectic_matrix();
    ///
    /// // X block is all zeros (no X components)
    /// // Z block: [1,1,0] and [0,1,1]
    /// assert_eq!(mat.row(0), &[0, 0, 0, 1, 1, 0]);
    /// assert_eq!(mat.row(1), &[0, 0, 0, 0, 1, 1]);
    /// ```
    #[must_use]
    pub fn to_symplectic_matrix(&self) -> F2Matrix {
        let n = self.num_qubits();
        let mut mat = F2Matrix::zeros(self.paulis.len(), 2 * n);

        for (row_idx, generator) in self.paulis.iter().enumerate() {
            for q in generator.x_positions() {
                mat.rows[row_idx][q] = 1;
            }
            for q in generator.z_positions() {
                mat.rows[row_idx][n + q] = 1;
            }
        }

        mat
    }

    /// Computes the rank (number of linearly independent Pauli strings).
    ///
    /// This is the rank of the binary symplectic matrix over GF(2).
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::PauliSequence;
    /// use pecos_core::pauli::constructors::*;
    ///
    /// // Two independent generators
    /// let gens = PauliSequence::new(vec![Zs(&[0, 1]), Zs(&[1, 2])]);
    /// assert_eq!(gens.rank(), 2);
    ///
    /// // Adding a dependent generator (ZIZ = ZZI * IZZ in GF(2))
    /// let gens = PauliSequence::new(vec![Zs(&[0, 1]), Zs(&[1, 2]), Zs(&[0, 2])]);
    /// assert_eq!(gens.rank(), 2);
    /// ```
    #[must_use]
    pub fn rank(&self) -> usize {
        let mat = self.to_symplectic_matrix();
        let (_, pivots) = mat.row_reduce();
        pivots.len()
    }

    /// Checks if a Pauli string is in the GF(2) span of this sequence.
    ///
    /// This checks membership ignoring phase: whether the symplectic vector of `pauli`
    /// can be expressed as a GF(2) linear combination of the sequence's vectors.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::PauliSequence;
    /// use pecos_core::pauli::constructors::*;
    ///
    /// let gens = PauliSequence::new(vec![Zs(&[0, 1]), Zs(&[1, 2])]);
    ///
    /// // ZIZ = ZZI * IZZ (symplectic addition), so it's in the group
    /// assert!(gens.contains(&Zs(&[0, 2])));
    ///
    /// // X on qubit 0 is not in the group
    /// assert!(!gens.contains(&X(0)));
    ///
    /// // ZZI is in the group (it's a generator)
    /// assert!(gens.contains(&Zs(&[0, 1])));
    /// ```
    #[must_use]
    pub fn contains(&self, pauli: &PauliString) -> bool {
        let n = self.num_qubits();

        // If the target touches qubits beyond our generators, it can't be in the span
        let target_max = pauli.qubits().into_iter().max().map_or(0, |m| m + 1);
        if target_max > n {
            return false;
        }

        let mat = self.to_symplectic_matrix();
        let (reduced, pivots) = mat.row_reduce();

        // Build the target's symplectic vector
        let mut target = vec![0u8; 2 * n];
        for q in pauli.x_positions() {
            target[q] = 1;
        }
        for q in pauli.z_positions() {
            target[n + q] = 1;
        }

        // Eliminate the target using the reduced generators' pivots
        for (row_idx, &pivot_col) in pivots.iter().enumerate() {
            if target[pivot_col] == 1 {
                for (col, t) in target.iter_mut().enumerate() {
                    *t ^= reduced.rows[row_idx][col];
                }
            }
        }

        target.iter().all(|&b| b == 0)
    }

    /// Checks if a Pauli string is in the GF(2) span, including phase matching.
    ///
    /// This checks both that the symplectic vector is in the span of the sequence
    /// and that the phase matches (the product of the Pauli strings used to construct
    /// the target yields the same phase).
    #[must_use]
    pub fn contains_with_phase(&self, pauli: &PauliString) -> bool {
        let n = self.num_qubits();
        let k = self.paulis.len();

        // If the target touches qubits beyond our generators, it can't be in the span
        let target_max = pauli.qubits().into_iter().max().map_or(0, |m| m + 1);
        if target_max > n {
            return false;
        }

        // Build augmented matrix [symplectic | identity] to track which generators are used
        let aug_cols = 2 * n + k;
        let mut mat = F2Matrix::zeros(k, aug_cols);

        for (row_idx, generator) in self.paulis.iter().enumerate() {
            for q in generator.x_positions() {
                mat.rows[row_idx][q] = 1;
            }
            for q in generator.z_positions() {
                mat.rows[row_idx][n + q] = 1;
            }
            mat.rows[row_idx][2 * n + row_idx] = 1;
        }

        let (reduced, pivots) = mat.row_reduce();

        // Build the target vector
        let mut target = vec![0u8; aug_cols];
        for q in pauli.x_positions() {
            target[q] = 1;
        }
        for q in pauli.z_positions() {
            target[n + q] = 1;
        }

        // Eliminate the target using the reduced rows
        for (row_idx, &pivot_col) in pivots.iter().enumerate() {
            if target[pivot_col] == 1 {
                for (col, t) in target.iter_mut().enumerate() {
                    *t ^= reduced.rows[row_idx][col];
                }
            }
        }

        if !target[..2 * n].iter().all(|&b| b == 0) {
            return false;
        }

        // The tracking columns tell us which original generators were used
        let mut product = PauliString::identity();
        for (i, generator) in self.paulis.iter().enumerate() {
            if target[2 * n + i] == 1 {
                product = product * generator;
            }
        }

        product.phase() == pauli.phase()
    }

    /// Checks if all Pauli strings mutually commute.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::PauliSequence;
    /// use pecos_core::pauli::constructors::*;
    ///
    /// // Commuting generators
    /// let gens = PauliSequence::new(vec![Zs(&[0, 1]), Zs(&[1, 2])]);
    /// assert!(gens.is_abelian());
    ///
    /// // Non-commuting generators
    /// let gens = PauliSequence::new(vec![X(0), Z(0)]);
    /// assert!(!gens.is_abelian());
    /// ```
    #[must_use]
    pub fn is_abelian(&self) -> bool {
        for i in 0..self.paulis.len() {
            for j in (i + 1)..self.paulis.len() {
                if !self.paulis[i].commutes_with(&self.paulis[j]) {
                    return false;
                }
            }
        }
        true
    }

    /// Returns the commutation matrix.
    ///
    /// `result[i][j]` is `true` if entries `i` and `j` commute, `false` if they anticommute.
    /// The diagonal is always `true` (every operator commutes with itself).
    #[must_use]
    #[allow(clippy::needless_range_loop)] // symmetric update requires indexing both [i][j] and [j][i]
    pub fn commutation_matrix(&self) -> Vec<Vec<bool>> {
        let k = self.paulis.len();
        let mut matrix = vec![vec![true; k]; k];
        for i in 0..k {
            for j in (i + 1)..k {
                let commutes = self.paulis[i].commutes_with(&self.paulis[j]);
                matrix[i][j] = commutes;
                matrix[j][i] = commutes;
            }
        }
        matrix
    }

    /// Returns the sequence in row-reduced form.
    ///
    /// This returns a new `PauliSequence` where the Pauli strings are independent
    /// and in reduced row echelon form. Redundant entries are removed.
    ///
    /// Note: Phases are tracked by performing the corresponding Pauli multiplications
    /// alongside the GF(2) row operations.
    #[must_use]
    pub fn row_reduce(&self) -> Self {
        let k = self.paulis.len();
        let mut mat = self.to_symplectic_matrix();
        let mut paulis: Vec<PauliString> = self.paulis.clone();

        let mut pivot_row = 0;

        for col in 0..mat.num_cols {
            let mut found = None;
            for row in pivot_row..k {
                if mat.rows[row][col] == 1 {
                    found = Some(row);
                    break;
                }
            }

            let Some(found_row) = found else {
                continue;
            };

            mat.swap_rows(pivot_row, found_row);
            paulis.swap(pivot_row, found_row);

            for row in 0..k {
                if row != pivot_row && mat.rows[row][col] == 1 {
                    mat.xor_row(row, pivot_row);
                    let pivot_ps = paulis[pivot_row].clone();
                    paulis[row] = paulis[row].clone() * pivot_ps;
                }
            }

            pivot_row += 1;
        }

        let reduced: Vec<PauliString> = paulis.into_iter().take(pivot_row).collect();
        Self { paulis: reduced }
    }

    /// Computes the centralizer: all `n`-qubit Pauli strings (ignoring phase) that
    /// commute with every element in this sequence.
    ///
    /// Returns a basis for the centralizer as symplectic vectors (each of length `2n`).
    /// Uses the inferred qubit count (max qubit index + 1).
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::PauliSequence;
    /// use pecos_core::pauli::constructors::*;
    ///
    /// // Repetition code: ZZI, IZZ on 3 qubits
    /// // Centralizer dimension = 2n - rank = 6 - 2 = 4
    /// let gens = PauliSequence::new(vec![Zs(&[0, 1]), Zs(&[1, 2])]);
    /// let cent = gens.centralizer();
    /// assert_eq!(cent.len(), 4);
    /// ```
    #[must_use]
    pub fn centralizer(&self) -> Vec<Vec<u8>> {
        self.centralizer_in(self.num_qubits())
    }

    /// Computes the centralizer with an explicit qubit count.
    ///
    /// Use this when the system has more qubits than the generators touch
    /// (e.g., a stabilizer code embedded in a larger system).
    #[must_use]
    pub fn centralizer_in(&self, num_qubits: usize) -> Vec<Vec<u8>> {
        let n = num_qubits;
        let mut mat = F2Matrix::zeros(self.paulis.len(), 2 * n);

        for (row_idx, generator) in self.paulis.iter().enumerate() {
            for q in generator.x_positions() {
                if q < n {
                    mat.rows[row_idx][q] = 1;
                }
            }
            for q in generator.z_positions() {
                if q < n {
                    mat.rows[row_idx][n + q] = 1;
                }
            }
        }

        // Build S * Omega where Omega swaps X and Z blocks
        let mut s_omega = F2Matrix::zeros(mat.num_rows(), 2 * n);
        for i in 0..mat.num_rows() {
            for j in 0..n {
                s_omega.rows[i][j] = mat.rows[i][n + j]; // Z block -> first half
                s_omega.rows[i][n + j] = mat.rows[i][j]; // X block -> second half
            }
        }

        s_omega.kernel()
    }
}

impl PauliSequence {
    /// Returns the dense string representation, one Pauli per line.
    ///
    /// Each line uses `pauli_str` padded to `num_qubits`, without phase prefix.
    /// This is the standard tableau format.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::PauliSequence;
    ///
    /// let seq: PauliSequence = "ZZI\nIZZ".parse().unwrap();
    /// assert_eq!(seq.to_dense_str(), "ZZI\nIZZ");
    /// ```
    #[must_use]
    pub fn to_dense_str(&self) -> String {
        let n = self.num_qubits();
        self.paulis
            .iter()
            .map(|p| p.pauli_str(Some(n)))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Returns the sparse string representation, one Pauli per line.
    ///
    /// Each line uses the sparse format (`"X0 Z2"`) with phase prefix.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::PauliSequence;
    /// use pecos_core::pauli::constructors::*;
    ///
    /// let seq = PauliSequence::new(vec![X(0) & Z(2), Z(1)]);
    /// assert_eq!(seq.to_sparse_str(), "+X0 Z2\n+Z1");
    /// ```
    #[must_use]
    pub fn to_sparse_str(&self) -> String {
        self.paulis
            .iter()
            .map(PauliString::to_sparse_str)
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Transforms all Pauli strings by a Clifford gate: each `P_i` -> `C P_i C†`.
    ///
    /// Returns a new `PauliSequence` with the transformed Pauli strings.
    #[must_use]
    pub fn apply_clifford(
        &self,
        clifford: &pecos_core::clifford_rep::CliffordRep,
    ) -> PauliSequence {
        let transformed: Vec<PauliString> = self.paulis.iter().map(|p| clifford.apply(p)).collect();
        PauliSequence::new(transformed)
    }
}

impl FromStr for PauliSequence {
    type Err = ParsePauliStringError;

    /// Parses a `PauliSequence` from newline-delimited Pauli strings.
    ///
    /// Each line is parsed via [`PauliString::from_str`] (auto-detecting
    /// sparse vs dense format). Blank lines are skipped.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::PauliSequence;
    /// use std::str::FromStr;
    ///
    /// // Dense format
    /// let seq: PauliSequence = "ZZI\nIZZ".parse().unwrap();
    /// assert_eq!(seq.len(), 2);
    /// assert_eq!(seq.num_qubits(), 3);
    ///
    /// // Sparse format
    /// let seq: PauliSequence = "X0 Z2\nZ1".parse().unwrap();
    /// assert_eq!(seq.len(), 2);
    /// ```
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let paulis: Vec<PauliString> = s
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(str::parse)
            .collect::<Result<_, _>>()?;

        Ok(Self { paulis })
    }
}

impl fmt::Display for PauliSequence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_dense_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::pauli::constructors::*;

    #[test]
    fn test_new() {
        let gens = PauliSequence::new(vec![Zs([0, 1]), Zs([1, 2])]);
        assert_eq!(gens.len(), 2);
        assert_eq!(gens.num_qubits(), 3);
    }

    #[test]
    fn test_from_strs() {
        let gens = PauliSequence::from_strs(&["ZZI", "IZZ"]).unwrap();
        assert_eq!(gens.len(), 2);
        assert_eq!(gens.num_qubits(), 3);
    }

    #[test]
    fn test_symplectic_matrix() {
        let gens = PauliSequence::from_strs(&["ZZI", "IZZ"]).unwrap();
        let mat = gens.to_symplectic_matrix();
        assert_eq!(mat.num_rows(), 2);
        assert_eq!(mat.num_cols(), 6);
        // ZZI: x=[0,0,0] z=[1,1,0]
        assert_eq!(mat.row(0), &[0, 0, 0, 1, 1, 0]);
        // IZZ: x=[0,0,0] z=[0,1,1]
        assert_eq!(mat.row(1), &[0, 0, 0, 0, 1, 1]);
    }

    #[test]
    fn test_symplectic_matrix_y() {
        // Y has both X and Z bits set
        let gens = PauliSequence::new(vec![Y(0)]);
        let mat = gens.to_symplectic_matrix();
        assert_eq!(mat.row(0), &[1, 1]); // x=1, z=1
    }

    #[test]
    fn test_rank_independent() {
        let gens = PauliSequence::new(vec![Zs([0, 1]), Zs([1, 2])]);
        assert_eq!(gens.rank(), 2);
    }

    #[test]
    fn test_rank_dependent() {
        // ZIZ = ZZI * IZZ (symplectic: 110 XOR 011 = 101), so rank should still be 2
        let gens = PauliSequence::new(vec![Zs([0, 1]), Zs([1, 2]), Zs([0, 2])]);
        assert_eq!(gens.rank(), 2);
    }

    #[test]
    fn test_rank_single() {
        let gens = PauliSequence::new(vec![X(0)]);
        assert_eq!(gens.rank(), 1);
    }

    #[test]
    fn test_rank_empty() {
        let gens = PauliSequence::new(vec![]);
        assert_eq!(gens.rank(), 0);
    }

    #[test]
    fn test_contains_generator() {
        let gens = PauliSequence::new(vec![Zs([0, 1]), Zs([1, 2])]);
        assert!(gens.contains(&Zs([0, 1])));
        assert!(gens.contains(&Zs([1, 2])));
    }

    #[test]
    fn test_contains_product() {
        let gens = PauliSequence::new(vec![Zs([0, 1]), Zs([1, 2])]);
        // ZIZ = ZZI * IZZ (symplectic: 110 XOR 011 = 101)
        assert!(gens.contains(&Zs([0, 2])));
    }

    #[test]
    fn test_not_contains() {
        let gens = PauliSequence::new(vec![Zs([0, 1]), Zs([1, 2])]);
        assert!(!gens.contains(&X(0)));
        assert!(!gens.contains(&Z(0)));
    }

    #[test]
    fn test_contains_identity() {
        let gens = PauliSequence::new(vec![Zs([0, 1])]);
        // Identity is always in the group (product of zero generators)
        assert!(gens.contains(&I()));
    }

    #[test]
    fn test_is_abelian_commuting() {
        let gens = PauliSequence::new(vec![Zs([0, 1]), Zs([1, 2])]);
        assert!(gens.is_abelian());
    }

    #[test]
    fn test_is_abelian_anticommuting() {
        let gens = PauliSequence::new(vec![X(0), Z(0)]);
        assert!(!gens.is_abelian());
    }

    #[test]
    fn test_commutation_matrix() {
        let gens = PauliSequence::new(vec![X(0), Z(0), Y(0)]);
        let cm = gens.commutation_matrix();
        // X,Z anticommute
        assert!(!cm[0][1]);
        assert!(!cm[1][0]);
        // X,Y anticommute
        assert!(!cm[0][2]);
        // Z,Y anticommute
        assert!(!cm[1][2]);
        // Self-commutation
        assert!(cm[0][0]);
        assert!(cm[1][1]);
        assert!(cm[2][2]);
    }

    #[test]
    fn test_row_reduce() {
        // ZIZ = ZZI * IZZ, so one generator is redundant
        let gens = PauliSequence::new(vec![Zs([0, 1]), Zs([1, 2]), Zs([0, 2])]);
        let reduced = gens.row_reduce();
        assert_eq!(reduced.len(), 2);
        assert_eq!(reduced.rank(), 2);
    }

    #[test]
    fn test_display() {
        let gens = PauliSequence::from_strs(&["ZZI", "IZZ"]).unwrap();
        let s = format!("{gens}");
        assert_eq!(s, "ZZI\nIZZ");
    }

    #[test]
    fn test_steane_code() {
        // [[7,1,3]] Steane code
        let gens = PauliSequence::new(vec![
            Xs([0, 2, 4, 6]),
            Xs([1, 2, 5, 6]),
            Xs([3, 4, 5, 6]),
            Zs([0, 2, 4, 6]),
            Zs([1, 2, 5, 6]),
            Zs([3, 4, 5, 6]),
        ]);
        assert_eq!(gens.rank(), 6);
        assert!(gens.is_abelian());

        // Logical operators should not be in the stabilizer group
        assert!(!gens.contains(&Xs([0, 1, 2, 3, 4, 5, 6])));
        assert!(!gens.contains(&Zs([0, 1, 2, 3, 4, 5, 6])));
    }

    #[test]
    fn test_contains_with_phase() {
        let gens = PauliSequence::new(vec![Zs([0, 1]), Zs([1, 2])]);

        // +ZZI is in the group with correct phase
        assert!(gens.contains_with_phase(&Zs([0, 1])));

        // -ZZI should not be in the group (wrong phase)
        assert!(!gens.contains_with_phase(&(-Zs([0, 1]))));
    }

    #[test]
    fn test_f2_matrix_display() {
        let gens = PauliSequence::from_strs(&["XZ", "ZX"]).unwrap();
        let mat = gens.to_symplectic_matrix();
        let s = format!("{mat}");
        assert_eq!(s, "10|01\n01|10");
    }

    // ========================================================================
    // FromStr / to_dense_str / to_sparse_str tests
    // ========================================================================

    #[test]
    fn test_from_str_dense() {
        let seq: PauliSequence = "ZZI\nIZZ".parse().unwrap();
        assert_eq!(seq.len(), 2);
        assert_eq!(seq.num_qubits(), 3);
    }

    #[test]
    fn test_from_str_sparse() {
        let seq: PauliSequence = "X0 Z2\nZ1".parse().unwrap();
        assert_eq!(seq.len(), 2);
        assert_eq!(seq.num_qubits(), 3);
    }

    #[test]
    fn test_from_str_blank_lines() {
        let seq: PauliSequence = "\nZZI\n\nIZZ\n".parse().unwrap();
        assert_eq!(seq.len(), 2);
    }

    #[test]
    fn test_from_str_empty() {
        let seq: PauliSequence = "".parse().unwrap();
        assert_eq!(seq.len(), 0);
    }

    #[test]
    fn test_to_dense_str() {
        let seq = PauliSequence::from_strs(&["ZZI", "IZZ"]).unwrap();
        assert_eq!(seq.to_dense_str(), "ZZI\nIZZ");
    }

    #[test]
    fn test_to_sparse_str() {
        let seq = PauliSequence::new(vec![Zs([0, 1]), Zs([1, 2])]);
        assert_eq!(seq.to_sparse_str(), "+Z0 Z1\n+Z1 Z2");
    }

    #[test]
    fn test_roundtrip_dense() {
        let original = PauliSequence::from_strs(&["XZI", "IYZ"]).unwrap();
        let s = original.to_dense_str();
        let roundtripped: PauliSequence = s.parse().unwrap();
        assert_eq!(roundtripped.len(), original.len());
        assert_eq!(roundtripped.num_qubits(), original.num_qubits());
    }

    #[test]
    fn test_roundtrip_sparse() {
        let original = PauliSequence::new(vec![X(0) & Z(2), Z(1)]);
        let s = original.to_sparse_str();
        let roundtripped: PauliSequence = s.parse().unwrap();
        assert_eq!(roundtripped.len(), original.len());
    }

    // ========================================================================
    // push / extend / centralizer tests
    // ========================================================================

    #[test]
    fn test_push() {
        let mut seq = PauliSequence::new(vec![X(0)]);
        assert_eq!(seq.len(), 1);
        assert_eq!(seq.num_qubits(), 1);

        seq.push(Z(2));
        assert_eq!(seq.len(), 2);
        assert_eq!(seq.num_qubits(), 3);
    }

    #[test]
    fn test_extend() {
        let mut seq = PauliSequence::new(Vec::new());
        seq.extend(vec![Zs([0, 1]), Zs([1, 2])]);
        assert_eq!(seq.len(), 2);
        assert_eq!(seq.num_qubits(), 3);
    }

    #[test]
    fn test_centralizer_repetition_code() {
        // ZZI, IZZ on 3 qubits: centralizer dimension = 2*3 - 2 = 4
        let gens = PauliSequence::new(vec![Zs([0, 1]), Zs([1, 2])]);
        let cent = gens.centralizer();
        assert_eq!(cent.len(), 4);
    }

    #[test]
    fn test_centralizer_single_qubit() {
        // Z on 1 qubit: centralizer = {Z, I} = dimension 1
        let gens = PauliSequence::new(vec![Z(0)]);
        let cent = gens.centralizer();
        assert_eq!(cent.len(), 1);
    }

    #[test]
    fn test_centralizer_empty() {
        // No generators, inferred n=0: centralizer is trivially empty
        let gens = PauliSequence::new(Vec::new());
        let cent = gens.centralizer();
        assert_eq!(cent.len(), 0);
    }

    #[test]
    fn test_centralizer_in_explicit_qubits() {
        // No generators on 2 qubits: everything commutes, dimension = 2*2 = 4
        let gens = PauliSequence::new(Vec::new());
        let cent = gens.centralizer_in(2);
        assert_eq!(cent.len(), 4);
    }

    #[test]
    fn test_f2_kernel() {
        // Identity matrix: kernel is empty
        let mut mat = F2Matrix::zeros(2, 2);
        mat.rows[0][0] = 1;
        mat.rows[1][1] = 1;
        assert!(mat.kernel().is_empty());

        // Zero matrix 2x3: kernel dimension = 3
        let mat = F2Matrix::zeros(2, 3);
        assert_eq!(mat.kernel().len(), 3);
    }

    #[test]
    fn test_f2_kernel_rank_deficient() {
        // [[1,0,0],[1,0,0]]: rank 1, kernel dimension = 3 - 1 = 2
        let mut mat = F2Matrix::zeros(2, 3);
        mat.rows[0][0] = 1;
        mat.rows[1][0] = 1;
        let kern = mat.kernel();
        assert_eq!(kern.len(), 2);
        // Each kernel vector should satisfy A * v = 0
        for v in &kern {
            for row in mat.rows() {
                let dot: u8 = row.iter().zip(v.iter()).map(|(a, b)| a & b).sum::<u8>() % 2;
                assert_eq!(dot, 0);
            }
        }
    }

    #[test]
    fn test_f2_kernel_rectangular() {
        // 1x4 matrix [1,1,0,0]: kernel dim = 3
        let mut mat = F2Matrix::zeros(1, 4);
        mat.rows[0][0] = 1;
        mat.rows[0][1] = 1;
        let kern = mat.kernel();
        assert_eq!(kern.len(), 3);
    }

    #[test]
    fn test_centralizer_steane_code() {
        // [[7,1]] Steane code: 6 generators, centralizer dimension = 14 - 6 = 8
        let gens = PauliSequence::new(vec![
            Xs([0, 2, 4, 6]),
            Xs([1, 2, 5, 6]),
            Xs([3, 4, 5, 6]),
            Zs([0, 2, 4, 6]),
            Zs([1, 2, 5, 6]),
            Zs([3, 4, 5, 6]),
        ]);
        let cent = gens.centralizer();
        assert_eq!(cent.len(), 8); // 6 stabilizer directions + 2 logical
    }

    #[test]
    fn test_centralizer_five_qubit_code() {
        // [[5,1,3]]: 4 generators, centralizer dimension = 10 - 4 = 6
        let gens = PauliSequence::new(vec![
            X(0) & Z(1) & Z(2) & X(3),
            X(1) & Z(2) & Z(3) & X(4),
            X(0) & X(2) & Z(3) & Z(4),
            Z(0) & X(1) & X(3) & Z(4),
        ]);
        let cent = gens.centralizer();
        assert_eq!(cent.len(), 6);
    }

    #[test]
    fn test_push_identity_doesnt_change_num_qubits() {
        let mut seq = PauliSequence::new(vec![X(5)]);
        seq.push(PauliString::identity());
        assert_eq!(seq.num_qubits(), 6);
        assert_eq!(seq.len(), 2);
    }

    #[test]
    fn test_push_smaller_qubit() {
        let mut seq = PauliSequence::new(vec![X(5)]);
        seq.push(Z(0));
        assert_eq!(seq.num_qubits(), 6);
        assert_eq!(seq.len(), 2);
    }

    #[test]
    fn test_push_with_phase() {
        let mut seq = PauliSequence::new(vec![]);
        seq.push(-X(0));
        assert_eq!(seq.len(), 1);
        let ps = &seq.paulis()[0];
        assert_eq!(ps.phase(), pecos_core::QuarterPhase::MinusOne);
    }

    #[test]
    fn test_extend_empty() {
        let mut seq = PauliSequence::new(vec![X(0)]);
        let before = seq.len();
        seq.extend(Vec::<PauliString>::new());
        assert_eq!(seq.len(), before, "extend with empty should be a no-op");
    }

    #[test]
    fn test_symplectic_matrix_y_operator_both_bits() {
        // Y has both x and z bits set
        let seq = PauliSequence::new(vec![Y(0)]);
        let mat = seq.to_symplectic_matrix();
        // For 1 qubit, symplectic vector is [x0, z0]
        assert_eq!(mat.rows[0], vec![1, 1], "Y should set both x and z bits");
    }

    #[test]
    fn test_f2_matrix_row_reduce_empty() {
        let mat = F2Matrix::zeros(0, 3);
        let (reduced, pivots) = mat.row_reduce();
        assert!(reduced.rows.is_empty());
        assert!(pivots.is_empty());
    }

    #[test]
    fn test_f2_matrix_kernel_tall_matrix() {
        // More rows than columns: 3x2 matrix
        let mut mat = F2Matrix::zeros(3, 2);
        mat.rows[0] = vec![1, 0];
        mat.rows[1] = vec![0, 1];
        mat.rows[2] = vec![1, 1]; // row 2 = row 0 + row 1 (redundant)
        // Full column rank => kernel is empty
        let kern = mat.kernel();
        assert!(
            kern.is_empty(),
            "full column rank matrix should have trivial kernel"
        );
    }

    #[test]
    fn test_f2_matrix_kernel_identity() {
        // Identity matrix: full rank, trivial kernel
        let mut mat = F2Matrix::zeros(3, 3);
        mat.rows[0] = vec![1, 0, 0];
        mat.rows[1] = vec![0, 1, 0];
        mat.rows[2] = vec![0, 0, 1];
        let kern = mat.kernel();
        assert!(kern.is_empty());
    }

    #[test]
    fn test_centralizer_with_y_generators() {
        // Single Y generator on 1 qubit: Y commutes with Y and I
        // Centralizer of Y in 1-qubit Paulis should be dimension 1 (just Y itself plus I)
        let seq = PauliSequence::new(vec![Y(0)]);
        let cent = seq.centralizer();
        // For 1 qubit, the symplectic space is 2D.
        // Y has vector [1,1]. The centralizer kernel should have dimension 2 - 1 = 1.
        assert_eq!(
            cent.len(),
            1,
            "centralizer of single Y on 1 qubit should have dim 1"
        );
    }

    // ========================================================================
    // F2Matrix tests
    // ========================================================================

    #[test]
    fn test_f2_identity() {
        let id = F2Matrix::identity(3);
        assert_eq!(id.num_rows(), 3);
        assert_eq!(id.num_cols(), 3);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(id.row(i)[j], u8::from(i == j),);
            }
        }
    }

    #[test]
    fn test_f2_invert_identity() {
        let id = F2Matrix::identity(4);
        let inv = id.invert().unwrap();
        assert_eq!(inv, id);
    }

    #[test]
    fn test_f2_invert_swap_matrix() {
        // Swap rows 0 and 1: [[0,1],[1,0]]
        let mut m = F2Matrix::zeros(2, 2);
        m.rows[0][1] = 1;
        m.rows[1][0] = 1;
        let inv = m.invert().unwrap();
        // Swap is self-inverse
        assert_eq!(inv, m);
    }

    #[test]
    fn test_f2_invert_upper_triangular() {
        // [[1,1],[0,1]] over GF(2) is self-inverse
        let mut m = F2Matrix::zeros(2, 2);
        m.rows[0][0] = 1;
        m.rows[0][1] = 1;
        m.rows[1][1] = 1;
        let inv = m.invert().unwrap();
        assert_eq!(inv, m);
    }

    #[test]
    fn test_f2_invert_singular() {
        // [[1,1],[1,1]] is singular
        let mut m = F2Matrix::zeros(2, 2);
        m.rows[0] = vec![1, 1];
        m.rows[1] = vec![1, 1];
        assert!(m.invert().is_none());
    }

    #[test]
    fn test_f2_invert_nonsquare() {
        let m = F2Matrix::zeros(2, 3);
        assert!(m.invert().is_none());
    }

    #[test]
    fn test_f2_mul() {
        // [[1,1],[0,1]] * [[1,0],[1,1]] = [[0,1],[1,1]] over GF(2)
        let mut a = F2Matrix::zeros(2, 2);
        a.rows[0] = vec![1, 1];
        a.rows[1] = vec![0, 1];

        let mut b = F2Matrix::zeros(2, 2);
        b.rows[0] = vec![1, 0];
        b.rows[1] = vec![1, 1];

        let c = a.mul(&b);
        assert_eq!(c.rows[0], vec![0, 1]);
        assert_eq!(c.rows[1], vec![1, 1]);
    }

    #[test]
    fn test_f2_mul_inverse_gives_identity() {
        // Invertible 3x3 matrix over GF(2)
        let mut m = F2Matrix::zeros(3, 3);
        m.rows[0] = vec![1, 1, 0];
        m.rows[1] = vec![0, 1, 1];
        m.rows[2] = vec![1, 1, 1];

        let inv = m.invert().unwrap();
        let product = m.mul(&inv);
        assert_eq!(product, F2Matrix::identity(3));

        // Also check the other direction
        let product2 = inv.mul(&m);
        assert_eq!(product2, F2Matrix::identity(3));
    }

    #[test]
    fn test_f2_transpose() {
        let mut m = F2Matrix::zeros(2, 3);
        m.rows[0] = vec![1, 0, 1];
        m.rows[1] = vec![0, 1, 0];
        let t = m.transpose();
        assert_eq!(t.num_rows(), 3);
        assert_eq!(t.num_cols(), 2);
        assert_eq!(t.rows[0], vec![1, 0]);
        assert_eq!(t.rows[1], vec![0, 1]);
        assert_eq!(t.rows[2], vec![1, 0]);
    }

    // ========================================================================
    // apply_clifford test
    // ========================================================================

    #[test]
    fn test_apply_clifford() {
        use pecos_core::clifford_rep::CliffordRep;

        let seq = PauliSequence::new(vec![Z(0), Z(1)]);
        let h_all = CliffordRep::h(0).compose(&CliffordRep::h(1));
        let transformed = seq.apply_clifford(&h_all);

        // H: Z -> X on both qubits
        assert!(transformed.contains(&X(0)));
        assert!(transformed.contains(&X(1)));
    }

    #[test]
    fn test_apply_clifford_identity() {
        use pecos_core::clifford_rep::CliffordRep;

        let seq = PauliSequence::new(vec![X(0) & Z(1), Y(0)]);
        let id = CliffordRep::identity(2);
        let transformed = seq.apply_clifford(&id);

        assert!(transformed.contains(&(X(0) & Z(1))));
        assert!(transformed.contains(&Y(0)));
    }

    #[test]
    fn test_apply_clifford_phase_preservation() {
        use pecos_core::QuarterPhase;
        use pecos_core::clifford_rep::CliffordRep;

        // Z gate: X -> -X
        let seq = PauliSequence::new(vec![X(0)]);
        let z_gate = CliffordRep::z(0);
        let transformed = seq.apply_clifford(&z_gate);

        let p = &transformed.paulis()[0];
        assert_eq!(p.get(0), pecos_core::Pauli::X);
        assert_eq!(p.phase(), QuarterPhase::MinusOne);
    }

    #[test]
    fn test_apply_clifford_multi_qubit_pauli() {
        use pecos_core::clifford_rep::CliffordRep;

        // CX on ZZ -> Z_0 * (Z_0 Z_1) = Z_1
        let seq = PauliSequence::new(vec![Zs([0, 1])]);
        let cx = CliffordRep::cx(0, 1);
        let transformed = seq.apply_clifford(&cx);

        assert!(transformed.contains(&Z(1)));
    }

    #[test]
    fn test_apply_clifford_empty_sequence() {
        use pecos_core::clifford_rep::CliffordRep;

        let seq = PauliSequence::new(vec![]);
        let h = CliffordRep::h(0).extended_to(2);
        let transformed = seq.apply_clifford(&h);

        assert!(transformed.is_empty());
    }

    // ========================================================================
    // Additional F2Matrix tests
    // ========================================================================

    #[test]
    fn test_f2_identity_1x1() {
        let id = F2Matrix::identity(1);
        assert_eq!(id.rows[0], vec![1]);
    }

    #[test]
    fn test_f2_invert_1x1() {
        // [[1]] is invertible
        let mut m = F2Matrix::zeros(1, 1);
        m.rows[0] = vec![1];
        let inv = m.invert().unwrap();
        assert_eq!(inv.rows[0], vec![1]);

        // [[0]] is not invertible
        let z = F2Matrix::zeros(1, 1);
        assert!(z.invert().is_none());
    }

    #[test]
    fn test_f2_mul_identity() {
        let id = F2Matrix::identity(3);
        let mut m = F2Matrix::zeros(3, 3);
        m.rows[0] = vec![1, 1, 0];
        m.rows[1] = vec![0, 1, 1];
        m.rows[2] = vec![1, 1, 1];

        // I * A = A
        assert_eq!(id.mul(&m), m);
        // A * I = A
        assert_eq!(m.mul(&id), m);
    }

    #[test]
    fn test_f2_transpose_square() {
        let mut m = F2Matrix::zeros(2, 2);
        m.rows[0] = vec![1, 1];
        m.rows[1] = vec![0, 1];
        let t = m.transpose();
        assert_eq!(t.rows[0], vec![1, 0]);
        assert_eq!(t.rows[1], vec![1, 1]);
    }

    #[test]
    fn test_f2_double_transpose() {
        let mut m = F2Matrix::zeros(2, 3);
        m.rows[0] = vec![1, 0, 1];
        m.rows[1] = vec![0, 1, 0];
        let tt = m.transpose().transpose();
        assert_eq!(tt, m);
    }

    #[test]
    fn test_f2_invert_4x4() {
        // A larger invertible matrix over GF(2)
        let mut m = F2Matrix::zeros(4, 4);
        m.rows[0] = vec![1, 0, 0, 1];
        m.rows[1] = vec![0, 1, 0, 1];
        m.rows[2] = vec![0, 0, 1, 1];
        m.rows[3] = vec![1, 1, 1, 0];

        let inv = m.invert().unwrap();
        assert_eq!(m.mul(&inv), F2Matrix::identity(4));
        assert_eq!(inv.mul(&m), F2Matrix::identity(4));
    }
}
