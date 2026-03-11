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

use crate::{CliffordGateable, GensGeneric, MeasurementResult, QuantumSimulator};
use core::fmt::Debug;
use core::mem;
use pecos_core::{BitSet, IndexSet, QubitId, RngManageable, SortedVecSet, VecSet};
use pecos_rng::rng_ext::RngProbabilityExt;
use pecos_rng::{PecosRng, Rng, SeedableRng};

/// A sparse representation of a stabilizer state using the stabilizer/destabilizer formalism.
///
/// This implementation is based on the work found in the thesis "Quantum Algorithms, Architecture,
/// and Error Correction" by Ciarán Ryan-Anderson (<https://arxiv.org/abs/1812.04735>).
///
/// # State Representation
/// The quantum state is represented by:
/// - A set of n stabilizer generators that mutually commute
/// - A set of n destabilizer generators where destab\[i\] anti-commutes with stab\[i\] and
///   commutes with all other stabilizers
///
/// The implementation uses a sparse matrix representation for efficiency and speed, storing:
/// - Row-wise X and Z Pauli operators
/// - Column-wise X and Z Pauli operators
/// - Signs (± and ±i) for each generator
///
/// # Type Parameters
/// - R: A random number generator type, defaults to `PecosRng`
///
/// # Examples
/// ```rust
/// use pecos_core::{qid, qid2};
/// use pecos_qsim::{QuantumSimulator, CliffordGateable, SparseStab};
///
/// // Create a new 2-qubit stabilizer state
/// let mut sim = SparseStab::new(2);
///
/// // Create Bell state |Φ+> = (|00> + |11>)/√2
/// sim.h(&qid(0))
///    .cx(&qid2(0, 1));
///
/// // Measure the two qubits in the Z basis
/// let r0 = sim.mz(&qid(0)).into_iter().next().unwrap();
/// let r1 = sim.mz(&qid(1)).into_iter().next().unwrap();
///
/// // Both measurements should equal each other
/// assert_eq!(r0.outcome, r1.outcome);
/// // But should be random
/// assert!(!r0.is_deterministic);
/// ```
///
/// # Measurement Behavior
/// Measurements can be either:
/// - Deterministic: The outcome is predetermined by the current stabilizer state
/// - Non-deterministic: The outcome is random with 50-50 probability
///
/// The measurement functions return both the outcome and whether it was deterministic.
///
/// # Gate Operations
/// The simulator supports common Clifford gates:
/// - Pauli gates (X, Y, Z)
/// - Hadamard (H)
/// - Phase gates (S = SZ = √Z)
/// - CX and other 2-qubit Clifford gates
///
/// Each gate operation updates the stabilizer and destabilizer generators according to
/// the appropriate Heisenberg representation transformations.
///
/// # Memory Efficiency
/// The sparse representation is memory efficient for:
/// - States with local correlations
/// - Circuit intermediates with limited entanglement
/// - Error correction scenarios where most stabilizers are low-weight
///
/// # Performance Considerations
/// - Row/column access patterns are optimized for common operations
/// - Signs are stored separately from Pauli operators
/// - Non-deterministic measurements require tableau updates
///
/// # Limitations
/// - Only supports Clifford operations
/// - Cannot represent arbitrary quantum states
/// - Measurement outcomes are truly random (not pseudo-random)
///
/// # References
/// 1. Aaronson & Gottesman, "Improved Simulation of Stabilizer Circuits"
///    <https://arxiv.org/abs/quant-ph/0406196>
/// 2. Ryan-Anderson, "Quantum Algorithms, Architecture, and Error Correction"
///    <https://arxiv.org/abs/1812.04735>
///
/// Generic sparse stabilizer simulator over set type S.
#[derive(Clone, Debug)]
pub struct SparseStabGeneric<S: IndexSet = BitSet, R: SeedableRng + Rng + Debug = PecosRng> {
    pub(crate) num_qubits: usize,
    pub(crate) stabs: GensGeneric<S>,
    pub(crate) destabs: GensGeneric<S>,
    pub(crate) rng: R,
}

/// Default sparse stabilizer simulator using `BitSet` for O(1) toggle operations.
pub type SparseStab<R = PecosRng> = SparseStabGeneric<BitSet, R>;

/// Sparse stabilizer simulator using `BitSet` (same as `SparseStab`).
pub type SparseStabBitSet<R = PecosRng> = SparseStabGeneric<BitSet, R>;

/// Sparse stabilizer simulator using `SortedVecSet` for O(n+m) XOR operations.
///
/// This is the recommended Vec-based simulator. It keeps elements sorted,
/// enabling merge-based XOR operations that are O(n+m) instead of O(n*m).
///
/// Performance characteristics:
/// - O(n) toggle operations (maintains sorted order)
/// - O(n+m) XOR operations (merge algorithm)
/// - Best Vec-based option for d >= 5
///
/// For best overall performance, use `SparseStab` (BitSet-based) instead.
pub type SparseStabVecSet<R = PecosRng> = SparseStabGeneric<SortedVecSet, R>;

/// Alias for `SparseStabVecSet`.
pub type SparseStabSortedVecSet<R = PecosRng> = SparseStabVecSet<R>;

/// Sparse stabilizer simulator using unsorted `VecSet`.
///
/// This variant has O(1) toggle but O(n*m) XOR. Faster than `SparseStabVecSet`
/// only for very small circuits (distance < 5).
pub type SparseStabUnsortedVecSet<R = PecosRng> = SparseStabGeneric<VecSet<usize>, R>;

/// Constructors for `SparseStab` with the default set and RNG types.
///
/// These methods provide ergonomic construction without needing to specify types.
impl SparseStabGeneric<BitSet, PecosRng> {
    /// Create a new stabilizer simulator with the default RNG.
    ///
    /// This is the most common constructor - it uses the default `PecosRng` seeded
    /// from the operating system's random number generator.
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits in the system
    ///
    /// # Examples
    /// ```rust
    /// use pecos_qsim::SparseStab;
    ///
    /// // Create a new 2-qubit stabilizer state
    /// let mut sim = SparseStab::new(2);
    /// ```
    #[inline]
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        let rng = rand::make_rng();
        Self::with_rng(num_qubits, rng)
    }

    /// Create a new stabilizer simulator with a specific seed.
    ///
    /// This method allows for deterministic behavior by setting a specific seed for the
    /// random number generator.
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits in the system
    /// * `seed` - Seed value for the random number generator
    ///
    /// # Examples
    /// ```rust
    /// use pecos_qsim::SparseStab;
    ///
    /// // Create a simulator with a specific seed for reproducibility
    /// let state = SparseStab::with_seed(2, 42);
    /// ```
    #[inline]
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        let rng = PecosRng::seed_from_u64(seed);
        Self::with_rng(num_qubits, rng)
    }
}

/// Constructors for `SparseStabSortedVecSet` with the default RNG type.
impl SparseStabGeneric<SortedVecSet, PecosRng> {
    /// Create a new SortedVecSet-based stabilizer simulator with the default RNG.
    #[inline]
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        let rng = rand::make_rng();
        Self::with_rng(num_qubits, rng)
    }

    /// Create a new SortedVecSet-based stabilizer simulator with a specific seed.
    #[inline]
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        let rng = PecosRng::seed_from_u64(seed);
        Self::with_rng(num_qubits, rng)
    }
}

/// Constructors for `SparseStabUnsortedVecSet` with the default RNG type.
impl SparseStabGeneric<VecSet<usize>, PecosRng> {
    /// Create a new unsorted VecSet-based stabilizer simulator with the default RNG.
    #[inline]
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        let rng = rand::make_rng();
        Self::with_rng(num_qubits, rng)
    }

    /// Create a new unsorted VecSet-based stabilizer simulator with a specific seed.
    #[inline]
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        let rng = PecosRng::seed_from_u64(seed);
        Self::with_rng(num_qubits, rng)
    }
}

/// Methods available on `SparseStabGeneric` with any set and RNG types.
impl<S, R> SparseStabGeneric<S, R>
where
    S: IndexSet,
    R: SeedableRng + Rng + Debug,
{
    /// Returns the number of qubits in the system
    ///
    /// # Returns
    /// * `usize` - The total number of qubits this simulator is configured to handle
    ///
    /// # Examples
    /// ```rust
    /// use pecos_qsim::{QuantumSimulator, SparseStab};
    /// let state = SparseStab::new(2);
    /// let num = state.num_qubits();
    /// assert_eq!(num, 2);
    /// ```
    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Create a stabilizer simulator with a custom RNG.
    ///
    /// Use this when you need a specific RNG type or have an existing RNG instance.
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits in the system
    /// * `rng` - The random number generator to use
    ///
    /// # Examples
    /// ```rust
    /// use pecos_qsim::SparseStab;
    /// use rand::SeedableRng;
    /// use rand::rngs::SmallRng;
    ///
    /// let rng = SmallRng::seed_from_u64(42);
    /// let sim = SparseStab::with_rng(2, rng);
    /// ```
    #[inline]
    pub fn with_rng(num_qubits: usize, rng: R) -> Self {
        let mut stab = Self {
            num_qubits,
            stabs: GensGeneric::<S>::new(num_qubits),
            destabs: GensGeneric::<S>::new(num_qubits),
            rng,
        };
        stab.reset();
        stab
    }

    #[inline]
    pub fn reset(&mut self) -> &mut Self {
        self.stabs.init_all_z();
        self.destabs.init_all_x();
        self
    }

    /// Returns generator data as sparse index vectors.
    ///
    /// Returns `(col_x, col_z, row_x, row_z)` where each is a `Vec<Vec<usize>>`.
    pub fn gens_data(&self, is_stab: bool) -> crate::GensData {
        let gens = if is_stab { &self.stabs } else { &self.destabs };

        let col_x: Vec<Vec<usize>> = gens.col_x.iter().map(|s| s.iter().collect()).collect();
        let col_z: Vec<Vec<usize>> = gens.col_z.iter().map(|s| s.iter().collect()).collect();
        let row_x: Vec<Vec<usize>> = gens.row_x.iter().map(|s| s.iter().collect()).collect();
        let row_z: Vec<Vec<usize>> = gens.row_z.iter().map(|s| s.iter().collect()).collect();

        (col_x, col_z, row_x, row_z)
    }

    #[inline]
    pub fn verify_matrix(&self) {
        Self::check_row_eq_col(&self.stabs);
        Self::check_row_eq_col(&self.destabs);

        // TODO: Check that stabilizers commute.
        // TODO: Check destabilizers commute.
        // TODO: Check that only stab[i] anti-commutes with destab[j] only iff i == j;
        todo!()
    }

    #[inline]
    fn check_row_eq_col(gens: &GensGeneric<S>) {
        // TODO: Verify that this is doing what is intended...
        for (i, row) in gens.row_x.iter().enumerate() {
            for j in row.iter() {
                assert!(
                    gens.col_x[j].contains(i),
                    "Column-wise sparse matrix doesn't match row-wise spare matrix"
                );
            }
        }
    }

    /// Utility that creates a string for the Pauli generates of a `Gens`.
    #[inline]
    fn tableau_string(num_qubits: usize, gens: &GensGeneric<S>) -> String {
        // TODO: calculate signs so we are really doing Y and not W
        let mut result =
            String::with_capacity(num_qubits * gens.row_x.len() + gens.row_x.len() + 2);
        for i in 0..gens.row_x.len() {
            if gens.signs_minus.contains(i) {
                result.push('-');
            } else {
                result.push('+');
            }
            if gens.signs_i.contains(i) {
                result.push('i');
            }

            for qubit in 0..num_qubits {
                let in_row_x = gens.row_x[i].contains(qubit);
                let in_row_z = gens.row_z[i].contains(qubit);

                let char = match (in_row_x, in_row_z) {
                    (false, false) => 'I',
                    (true, false) => 'X',
                    (false, true) => 'Z',
                    (true, true) => 'Y',
                };
                result.push(char);
            }
            result.push('\n');
        }

        result
    }

    /// Produces a textual representation of the stabilizer in tableau form.
    #[inline]
    pub fn stab_tableau(&self) -> String {
        Self::tableau_string(self.num_qubits, &self.stabs)
    }

    /// Produces a textual representation of the destabilizer in tableau form.
    #[inline]
    pub fn destab_tableau(&self) -> String {
        Self::tableau_string(self.num_qubits, &self.destabs)
    }

    /// Negate the sign of a stabilizer generator.
    #[inline]
    pub fn neg(&mut self, s: usize) {
        self.stabs.signs_minus.toggle(s);
    }

    #[inline]
    pub fn signs_minus(&self) -> &S {
        &self.stabs.signs_minus
    }

    /// Returns an immutable reference to the stabilizer generators.
    ///
    /// This is useful for operations like classifying Pauli strings or
    /// extracting generator information.
    #[inline]
    pub fn stabs(&self) -> &GensGeneric<S> {
        &self.stabs
    }

    /// Returns a mutable reference to the stabilizer generators.
    ///
    /// Use with caution - modifying stabilizers directly can break
    /// the stabilizer/destabilizer relationship invariants.
    #[inline]
    pub fn stabs_mut(&mut self) -> &mut GensGeneric<S> {
        &mut self.stabs
    }

    /// Returns an immutable reference to the destabilizer generators.
    #[inline]
    pub fn destabs(&self) -> &GensGeneric<S> {
        &self.destabs
    }

    /// Returns a mutable reference to the destabilizer generators.
    ///
    /// Use with caution - modifying destabilizers directly can break
    /// the stabilizer/destabilizer relationship invariants.
    #[inline]
    pub fn destabs_mut(&mut self) -> &mut GensGeneric<S> {
        &mut self.destabs
    }

    /// Returns mutable references to both stabilizer and destabilizer generators.
    ///
    /// This is useful for operations like `refactor` that need mutable access
    /// to both generators simultaneously.
    ///
    /// Use with caution - modifying generators directly can break
    /// the stabilizer/destabilizer relationship invariants.
    #[inline]
    pub fn stabs_and_destabs_mut(&mut self) -> (&mut GensGeneric<S>, &mut GensGeneric<S>) {
        (&mut self.stabs, &mut self.destabs)
    }

    #[inline]
    fn deterministic_meas(&mut self, q: usize) -> MeasurementResult {
        // Use optimized intersection_count to avoid iterator creation overhead
        let mut num_minuses = self.destabs.col_x[q].intersection_count(&self.stabs.signs_minus);

        let num_is = self.destabs.col_x[q].intersection_count(&self.stabs.signs_i);

        let mut cumulative_x = S::new();
        for row in self.destabs.col_x[q].iter() {
            num_minuses += self.stabs.row_z[row].intersection_count(&cumulative_x);
            cumulative_x.xor_assign(&self.stabs.row_x[row]);
        }
        if num_is & 3 != 0 {
            // num_is % 4 != 0
            num_minuses += 1;
        }
        let outcome = num_minuses & 1 != 0; // num_minuses % 2 != 0 (is odd)
        MeasurementResult {
            outcome,
            is_deterministic: true,
        }
    }

    #[allow(clippy::too_many_lines)]
    #[inline]
    fn nondeterministic_meas(&mut self, q: usize, result: bool) -> MeasurementResult {
        // Clone only stabs.col_x[q] initially - defer destabs clone until needed
        let mut anticom_stabs_col = self.stabs.col_x[q].clone();

        let mut smallest_wt = 2 * self.num_qubits + 2;
        let mut removed_id: Option<usize> = None;

        for stab_id in anticom_stabs_col.iter() {
            let weight = self.stabs.row_x[stab_id].len() + self.stabs.row_z[stab_id].len();

            if weight < smallest_wt {
                smallest_wt = weight;
                removed_id = Some(stab_id);
                // Early termination: weight 1 is optimal (single-qubit Pauli)
                if weight == 1 {
                    break;
                }
            }
        }

        let id = removed_id.expect("Critical error: removed_id was None");

        anticom_stabs_col.remove(id);
        // Use take_clearing: takes the row contents but preserves capacity for reuse.
        // This enables toggle_unchecked in CX gate since rows will have capacity.
        let removed_row_x = self.stabs.row_x[id].take_clearing();
        let removed_row_z = self.stabs.row_z[id].take_clearing();

        if self.stabs.signs_minus.contains(id) {
            self.stabs.signs_minus.xor_assign(&anticom_stabs_col);
        }

        if self.stabs.signs_i.contains(id) {
            self.stabs.signs_i.remove(id);

            // Fused: XOR intersection into signs_minus, then XOR signs_i with anticom_stabs_col
            // This replaces the SmallVec allocations and separate loops
            self.stabs
                .signs_i
                .xor_intersection_into(&anticom_stabs_col, &mut self.stabs.signs_minus);
            self.stabs.signs_i.xor_assign(&anticom_stabs_col);
        }

        for g in anticom_stabs_col.iter() {
            let num_minuses = removed_row_z.intersection_count(&self.stabs.row_x[g]);

            if num_minuses & 1 != 0 {
                // num_minuses % 2 != 0 (is odd)
                self.stabs.signs_minus.toggle(g);
            }

            self.stabs.row_x[g].xor_assign(&removed_row_x);
            self.stabs.row_z[g].xor_assign(&removed_row_z);
        }

        // Fused loops: XOR and remove in single pass
        for i in removed_row_x.iter() {
            self.stabs.col_x[i].xor_assign(&anticom_stabs_col);
            self.stabs.col_x[i].remove(id);
        }

        for i in removed_row_z.iter() {
            self.stabs.col_z[i].xor_assign(&anticom_stabs_col);
            self.stabs.col_z[i].remove(id);
        }

        // Remove replaced stabilizer with the measured stabilizer
        self.stabs.col_z[q].insert(id);

        // Row update - no need to clear since we used take() above
        self.stabs.row_z[id].insert(q);

        for i in self.destabs.row_x[id].iter() {
            self.destabs.col_x[i].remove(id);
        }

        for i in self.destabs.row_z[id].iter() {
            self.destabs.col_z[i].remove(id);
        }

        // Clone destabs.col_x[q] only when needed (deferred from start of function)
        let mut anticom_destabs_col = self.destabs.col_x[q].clone();
        anticom_destabs_col.remove(id);

        for i in removed_row_x.iter() {
            self.destabs.col_x[i].insert(id);
            self.destabs.col_x[i].xor_assign(&anticom_destabs_col);
        }

        for i in removed_row_z.iter() {
            self.destabs.col_z[i].insert(id);
            self.destabs.col_z[i].xor_assign(&anticom_destabs_col);
        }

        for row in anticom_destabs_col.iter() {
            self.destabs.row_x[row].xor_assign(&removed_row_x);
            self.destabs.row_z[row].xor_assign(&removed_row_z);
        }

        self.destabs.row_x[id] = removed_row_x;
        self.destabs.row_z[id] = removed_row_z;

        let outcome = self.apply_outcome(id, result);
        MeasurementResult {
            outcome,
            is_deterministic: false,
        }
    }

    /// Measurement of the +`Z_q` operator where random outcomes are forced to a particular value.
    #[inline]
    pub fn mz_forced(&mut self, q: usize, forced_outcome: bool) -> MeasurementResult {
        if self.stabs.col_x[q].is_empty() {
            // There are no stabilizers that anti-commute with Z_q
            self.deterministic_meas(q)
        } else {
            self.nondeterministic_meas(q, forced_outcome)
        }
    }

    /// Preparation of the +`Z_q` operator where random outcomes are forced to a particular value.
    #[inline]
    pub fn pz_forced(&mut self, q: usize, forced_outcome: bool) -> &mut Self {
        let result = self.mz_forced(q, forced_outcome);
        if result.outcome {
            // Inline X gate: X -> X, Z -> -Z
            self.stabs.signs_minus.xor_assign(&self.stabs.col_z[q]);
        }
        self
    }

    /// Apply measurement outcome
    #[inline]
    fn apply_outcome(&mut self, id: usize, meas_outcome: bool) -> bool {
        if meas_outcome {
            self.stabs.signs_minus.insert(id);
        } else {
            self.stabs.signs_minus.remove(id);
        }
        meas_outcome
    }
}

impl<S, R> QuantumSimulator for SparseStabGeneric<S, R>
where
    S: IndexSet,
    R: SeedableRng + Rng + Debug,
{
    #[inline]
    fn reset(&mut self) -> &mut Self {
        Self::reset(self)
    }
}

impl<S, R> CliffordGateable for SparseStabGeneric<S, R>
where
    S: IndexSet,
    R: SeedableRng + Rng + Debug,
{
    // TODO: pub fun p(&mut self, pauli: &pauli, q: U) { todo!() }
    // TODO: pub fun m(&mut self, pauli: &pauli, q: U) -> bool { todo!() }

    /// Pauli X gate. X -> X, Z -> -Z
    #[inline]
    fn x(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qu = q.index();
            self.stabs.signs_minus.xor_assign(&self.stabs.col_z[qu]);
        }
        self
    }

    /// Pauli Y gate. X -> -X, Z -> -Z
    #[inline]
    fn y(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qu = q.index();
            // Fused: XOR elements in (col_x[qu] ⊕ col_z[qu]) into signs_minus
            self.stabs.col_x[qu]
                .xor_symmetric_difference_into(&self.stabs.col_z[qu], &mut self.stabs.signs_minus);
        }
        self
    }

    /// Pauli Z gate. X -> -X, Z -> Z
    #[inline]
    fn z(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.stabs
                .signs_minus
                .xor_assign(&self.stabs.col_x[q.index()]);
        }
        self
    }

    /// Sqrt of Z gate.
    ///     X -> iW = Y
    ///     Z -> Z
    ///     W -> iX
    ///     Y -> -X
    #[inline]
    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qu = q.index();

            // X -> i
            // ---------------------
            // i * i = -1
            // stabs.signs_minus ^= stabs.signs_i & stabs.col_x[qubit]
            // For each X add an i unless there is already an i there then delete it.
            // stabs.signs_i ^= stabs.col_x[qubit]
            // Fused: XOR elements in (signs_i ∩ col_x[qu]) into signs_minus
            self.stabs
                .signs_i
                .xor_intersection_into(&self.stabs.col_x[qu], &mut self.stabs.signs_minus);
            self.stabs.signs_i.xor_assign(&self.stabs.col_x[qu]);

            for g in [&mut self.stabs, &mut self.destabs] {
                g.col_z[qu].xor_assign(&g.col_x[qu]);

                for i in g.col_x[qu].iter() {
                    g.row_z[i].toggle(qu);
                }
            }
        }
        self
    }

    /// Hadamard gate. X -> Z, Z -> X
    #[inline]
    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qu = q.index();

            // Fused: XOR elements in (col_x[qu] ∩ col_z[qu]) into signs_minus
            self.stabs.col_x[qu]
                .xor_intersection_into(&self.stabs.col_z[qu], &mut self.stabs.signs_minus);

            for g in [&mut self.stabs, &mut self.destabs] {
                // Elements in col_x but not in col_z: X -> Z
                for i in g.col_x[qu].iter() {
                    if !g.col_z[qu].contains(i) {
                        g.row_x[i].remove(qu);
                        g.row_z[i].insert(qu);
                    }
                }

                // Elements in col_z but not in col_x: Z -> X
                for i in g.col_z[qu].iter() {
                    if !g.col_x[qu].contains(i) {
                        g.row_z[i].remove(qu);
                        g.row_x[i].insert(qu);
                    }
                }

                mem::swap(&mut g.col_x[qu], &mut g.col_z[qu]);
            }
        }
        self
    }

    /// Applies a CX or CNOT (Controlled-X) gate between two qubits.
    ///
    /// The CX performs the transformation:
    /// - |0>|b> -> |0>|b>
    /// - |1>|b> -> |1>|b XOR 1>
    ///
    /// In the Heisenberg picture, it transforms the Pauli operators as:
    /// - IX -> IX
    /// - XI -> XX
    /// - IZ -> ZZ
    /// - ZI -> ZI
    ///
    /// CX: +IX -> +IX; +IZ -> +ZZ; +XI -> +XX; +ZI -> +ZI
    #[inline]
    fn cx(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "CX requires pairs of qubits"
        );

        for pair in qubits.chunks_exact(2) {
            let q1 = pair[0].index();
            let q2 = pair[1].index();

            for g in &mut [&mut self.stabs, &mut self.destabs] {
                // SAFETY: q1 != q2 is guaranteed by the debug_assert at the start of cx.
                // We need mutable access to two different column entries simultaneously.
                // Using unsafe to avoid the split_at_mut overhead.
                unsafe {
                    // Handle col_x: toggle q2 in row_x[i] for each i in col_x[q1], then XOR columns
                    let col_x_q1 = g.col_x.get_unchecked(q1);
                    for i in col_x_q1.iter() {
                        g.row_x.get_unchecked_mut(i).toggle(q2);
                    }
                    let col_x_q1 = std::ptr::from_ref::<S>(g.col_x.get_unchecked(q1));
                    let col_x_q2 = g.col_x.get_unchecked_mut(q2);
                    col_x_q2.xor_assign(&*col_x_q1);

                    // Handle col_z: toggle q1 in row_z[i] for each i in col_z[q2], then XOR columns
                    let col_z_q2 = g.col_z.get_unchecked(q2);
                    for i in col_z_q2.iter() {
                        g.row_z.get_unchecked_mut(i).toggle(q1);
                    }
                    let col_z_q2 = std::ptr::from_ref::<S>(g.col_z.get_unchecked(q2));
                    let col_z_q1 = g.col_z.get_unchecked_mut(q1);
                    col_z_q1.xor_assign(&*col_z_q2);
                }
            }
        }
        self
    }

    /// Measures qubits in the Z basis.
    ///
    /// Returns a vector containing:
    /// - The measurement outcome (true = |1>, false = |0>)
    /// - Whether the measurement was deterministic
    ///
    /// The measurement can be:
    /// - Deterministic: The outcome is fixed by the current stabilizer state
    /// - Non-deterministic: The outcome is random with 50% probability for each result
    #[inline]
    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        let mut results = Vec::with_capacity(qubits.len());

        for &q in qubits {
            let qu = q.index();
            let deterministic = self.stabs.col_x[qu].is_empty();

            let result = if deterministic {
                // There are no stabilizers that anti-commute with Z_q
                self.deterministic_meas(qu)
            } else {
                let outcome = self.rng.coin_flip();
                self.nondeterministic_meas(qu, outcome)
            };
            results.push(result);
        }

        results
    }
}

impl<S, R> RngManageable for SparseStabGeneric<S, R>
where
    S: IndexSet,
    R: SeedableRng + Rng + Debug,
{
    type Rng = R;

    fn set_rng(&mut self, rng: Self::Rng) {
        self.rng = rng;
    }

    /// Get a read-only reference to the internal random number generator
    ///
    /// This method provides access to the RNG for inspection or to retrieve
    /// information from it (such as recorded values from a `RecordingRng`).
    ///
    /// # Returns
    /// A reference to the internal RNG
    #[inline]
    fn rng(&self) -> &Self::Rng {
        &self.rng
    }

    /// Get a mutable reference to the internal random number generator
    ///
    /// This method provides mutable access to the RNG for direct manipulation.
    /// This is an advanced feature that should be used with care.
    ///
    /// # Returns
    /// A mutable reference to the internal RNG
    #[inline]
    fn rng_mut(&mut self) -> &mut Self::Rng {
        &mut self.rng
    }
}

// Implement StabilizerTableauSimulator trait for SparseStabGeneric
use crate::stabilizer_tableau::StabilizerTableauSimulator;

impl<S, R> StabilizerTableauSimulator for SparseStabGeneric<S, R>
where
    S: IndexSet,
    R: SeedableRng + Rng + Debug,
{
    fn stab_tableau(&self) -> String {
        Self::tableau_string(self.num_qubits, &self.stabs)
    }

    fn destab_tableau(&self) -> String {
        Self::tableau_string(self.num_qubits, &self.destabs)
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}

// ============================================================================
// SparseStabHybrid - Uses VecSet for Paulis, BitSet for signs
// ============================================================================

use crate::GensHybrid;

/// Hybrid sparse stabilizer simulator using `VecSet` for Pauli data and `BitSet` for signs.
///
/// This combines the benefits of both set types:
/// - `VecSet` is faster for gate operations on small sets (typical stabilizer weights 2-4)
/// - `BitSet` is faster for sign membership checks during measurements (O(1) vs O(n))
///
/// The hybrid approach is particularly beneficial for multi-round simulations like
/// surface code syndrome extraction, where sign sets grow over time.
#[derive(Clone, Debug)]
pub struct SparseStabHybrid<R: SeedableRng + Rng + Debug = PecosRng> {
    pub(crate) num_qubits: usize,
    pub(crate) stabs: GensHybrid,
    pub(crate) destabs: GensHybrid,
    rng: R,
    // Scratch buffers for measurement to avoid repeated allocations
    scratch_stabs_col: VecSet<usize>,
    scratch_destabs_col: VecSet<usize>,
}

impl SparseStabHybrid<PecosRng> {
    /// Create a new hybrid stabilizer simulator with the default RNG.
    #[inline]
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        let rng = rand::make_rng();
        Self::with_rng(num_qubits, rng)
    }

    /// Create a new hybrid stabilizer simulator with a specific seed.
    #[inline]
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        let rng = PecosRng::seed_from_u64(seed);
        Self::with_rng(num_qubits, rng)
    }
}

impl<R> SparseStabHybrid<R>
where
    R: SeedableRng + Rng + Debug,
{
    /// Create a hybrid stabilizer simulator with a custom RNG.
    #[inline]
    pub fn with_rng(num_qubits: usize, rng: R) -> Self {
        let mut stab = Self {
            num_qubits,
            stabs: GensHybrid::new(num_qubits),
            destabs: GensHybrid::new(num_qubits),
            rng,
            scratch_stabs_col: VecSet::new(),
            scratch_destabs_col: VecSet::new(),
        };
        stab.reset();
        stab
    }

    /// Returns the number of qubits in the system.
    #[inline]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Reset to the |0...0> state.
    #[inline]
    pub fn reset(&mut self) -> &mut Self {
        self.stabs.init_all_z();
        self.destabs.init_all_x();
        self
    }

    /// Negate the sign of a stabilizer generator.
    #[inline]
    pub fn neg(&mut self, s: usize) {
        self.stabs.signs_minus.toggle(s);
    }

    /// Get the `signs_minus` `BitSet`.
    #[inline]
    pub fn signs_minus(&self) -> &BitSet {
        &self.stabs.signs_minus
    }

    /// Helper to produce a string representation of a generator set in tableau form.
    #[inline]
    fn tableau_string(num_qubits: usize, gens: &GensHybrid) -> String {
        let mut result =
            String::with_capacity(num_qubits * gens.row_x.len() + gens.row_x.len() + 2);
        for i in 0..gens.row_x.len() {
            if gens.signs_minus.contains(i) {
                result.push('-');
            } else {
                result.push('+');
            }
            if gens.signs_i.contains(i) {
                result.push('i');
            }

            for qubit in 0..num_qubits {
                let in_row_x = gens.row_x[i].contains(qubit);
                let in_row_z = gens.row_z[i].contains(qubit);

                let char = match (in_row_x, in_row_z) {
                    (false, false) => 'I',
                    (true, false) => 'X',
                    (false, true) => 'Z',
                    (true, true) => 'Y',
                };
                result.push(char);
            }
            result.push('\n');
        }

        result
    }

    /// Produces a textual representation of the stabilizer in tableau form.
    #[inline]
    pub fn stab_tableau(&self) -> String {
        Self::tableau_string(self.num_qubits, &self.stabs)
    }

    /// Produces a textual representation of the destabilizer in tableau form.
    #[inline]
    pub fn destab_tableau(&self) -> String {
        Self::tableau_string(self.num_qubits, &self.destabs)
    }

    #[inline]
    fn deterministic_meas(&mut self, q: usize) -> MeasurementResult {
        // Use BitSet's optimized slice-based intersection count
        let mut num_minuses = self
            .stabs
            .signs_minus
            .intersection_count_slice(self.destabs.col_x[q].as_slice());

        let num_is = self
            .stabs
            .signs_i
            .intersection_count_slice(self.destabs.col_x[q].as_slice());

        let mut cumulative_x: VecSet<usize> = VecSet::new();
        for row in self.destabs.col_x[q].iter().copied() {
            num_minuses += self.stabs.row_z[row].intersection_count(&cumulative_x);
            cumulative_x.xor_assign(&self.stabs.row_x[row]);
        }
        if num_is & 3 != 0 {
            num_minuses += 1;
        }
        let outcome = num_minuses & 1 != 0;
        MeasurementResult {
            outcome,
            is_deterministic: true,
        }
    }

    #[allow(clippy::too_many_lines)]
    #[inline]
    fn nondeterministic_meas(&mut self, q: usize, result: bool) -> MeasurementResult {
        // Find the stabilizer with smallest weight to remove
        let mut smallest_wt = 2 * self.num_qubits + 2;
        let mut removed_id: Option<usize> = None;

        for stab_id in self.stabs.col_x[q].iter().copied() {
            let weight = self.stabs.row_x[stab_id].len() + self.stabs.row_z[stab_id].len();

            if weight < smallest_wt {
                smallest_wt = weight;
                removed_id = Some(stab_id);
                // Early termination: weight 1 is optimal (single-qubit Pauli)
                if weight == 1 {
                    break;
                }
            }
        }

        let id = removed_id.expect("Critical error: removed_id was None");

        // Reuse scratch buffer to avoid allocation - take it, use it, put it back
        let mut anticom_stabs_col = std::mem::take(&mut self.scratch_stabs_col);
        anticom_stabs_col.clone_from(&self.stabs.col_x[q]);
        anticom_stabs_col.remove(id);

        let removed_row_x = std::mem::take(&mut self.stabs.row_x[id]);
        let removed_row_z = std::mem::take(&mut self.stabs.row_z[id]);

        // Cross-type: BitSet signs XOR with VecSet column (use pre-computed clone)
        if self.stabs.signs_minus.contains(id) {
            self.stabs
                .signs_minus
                .xor_assign_slice(anticom_stabs_col.as_slice());
        }

        if self.stabs.signs_i.contains(id) {
            self.stabs.signs_i.remove(id);

            // Cross-type: XOR (BitSet signs_i ∩ VecSet anticom_stabs_col) into BitSet signs_minus
            self.stabs
                .signs_minus
                .xor_intersection_slice(anticom_stabs_col.as_slice(), &self.stabs.signs_i);
            self.stabs
                .signs_i
                .xor_assign_slice(anticom_stabs_col.as_slice());
        }

        // Process all anticommuting stabilizers (already excludes id)
        for g in anticom_stabs_col.iter().copied() {
            let num_minuses = removed_row_z.intersection_count(&self.stabs.row_x[g]);

            if num_minuses & 1 != 0 {
                self.stabs.signs_minus.toggle(g);
            }

            self.stabs.row_x[g].xor_assign(&removed_row_x);
            self.stabs.row_z[g].xor_assign(&removed_row_z);
        }

        // Fused loops: XOR and remove in single pass
        for i in removed_row_x.iter().copied() {
            self.stabs.col_x[i].xor_assign(&anticom_stabs_col);
            self.stabs.col_x[i].remove(id);
        }

        for i in removed_row_z.iter().copied() {
            self.stabs.col_z[i].xor_assign(&anticom_stabs_col);
            self.stabs.col_z[i].remove(id);
        }

        self.stabs.col_z[q].insert(id);
        self.stabs.row_z[id].insert(q);

        for i in self.destabs.row_x[id].iter().copied() {
            self.destabs.col_x[i].remove(id);
        }

        for i in self.destabs.row_z[id].iter().copied() {
            self.destabs.col_z[i].remove(id);
        }

        // Reuse scratch buffer for destabs col
        let mut anticom_destabs_col = std::mem::take(&mut self.scratch_destabs_col);
        anticom_destabs_col.clone_from(&self.destabs.col_x[q]);
        anticom_destabs_col.remove(id);

        for i in removed_row_x.iter().copied() {
            self.destabs.col_x[i].insert(id);
            self.destabs.col_x[i].xor_assign(&anticom_destabs_col);
        }

        for i in removed_row_z.iter().copied() {
            self.destabs.col_z[i].insert(id);
            self.destabs.col_z[i].xor_assign(&anticom_destabs_col);
        }

        // Use anticom_destabs_col (already has id removed) to avoid per-iteration check
        for row in anticom_destabs_col.iter().copied() {
            self.destabs.row_x[row].xor_assign(&removed_row_x);
            self.destabs.row_z[row].xor_assign(&removed_row_z);
        }

        self.destabs.row_x[id] = removed_row_x;
        self.destabs.row_z[id] = removed_row_z;

        // Put scratch buffers back for reuse
        self.scratch_stabs_col = anticom_stabs_col;
        self.scratch_destabs_col = anticom_destabs_col;

        let outcome = self.apply_outcome(id, result);
        MeasurementResult {
            outcome,
            is_deterministic: false,
        }
    }

    /// Measurement of the +`Z_q` operator where random outcomes are forced to a particular value.
    #[inline]
    pub fn mz_forced(&mut self, q: usize, forced_outcome: bool) -> MeasurementResult {
        if self.stabs.col_x[q].is_empty() {
            self.deterministic_meas(q)
        } else {
            self.nondeterministic_meas(q, forced_outcome)
        }
    }

    /// Preparation of the +`Z_q` operator where random outcomes are forced to a particular value.
    #[inline]
    pub fn pz_forced(&mut self, q: usize, forced_outcome: bool) -> &mut Self {
        let result = self.mz_forced(q, forced_outcome);
        if result.outcome {
            // Cross-type: BitSet signs_minus XOR with VecSet col_z (optimized slice)
            self.stabs
                .signs_minus
                .xor_assign_slice(self.stabs.col_z[q].as_slice());
        }
        self
    }

    #[inline]
    fn apply_outcome(&mut self, id: usize, meas_outcome: bool) -> bool {
        if meas_outcome {
            self.stabs.signs_minus.insert(id);
        } else {
            self.stabs.signs_minus.remove(id);
        }
        meas_outcome
    }

    /// Convert this hybrid simulator to a pure BitSet-based simulator.
    ///
    /// This is useful when the tableau has become dense (many elements per row)
    /// and `BitSet`'s O(1) operations would be faster than `VecSet`'s O(n) operations.
    ///
    /// The conversion iterates over all `VecSet` elements to populate the `BitSets`,
    /// which is `O(total_elements)` where `total_elements` is the sum of all set sizes.
    #[must_use]
    pub fn to_bitset(self) -> SparseStabGeneric<BitSet, R> {
        // Helper to convert a slice of VecSets to a Vec of BitSets
        fn convert_sets(sets: &[VecSet<usize>], num_qubits: usize) -> Vec<BitSet> {
            sets.iter()
                .map(|vs| {
                    let mut bs = BitSet::with_capacity(num_qubits);
                    for &elem in vs {
                        bs.insert(elem);
                    }
                    bs
                })
                .collect()
        }

        let n = self.num_qubits;

        // Convert Gens (stabs and destabs)
        let stabs = GensGeneric::from_parts(
            n,
            convert_sets(&self.stabs.col_x, n),
            convert_sets(&self.stabs.col_z, n),
            convert_sets(&self.stabs.row_x, n),
            convert_sets(&self.stabs.row_z, n),
            self.stabs.signs_minus,
            self.stabs.signs_i,
        );

        let destabs = GensGeneric::from_parts(
            n,
            convert_sets(&self.destabs.col_x, n),
            convert_sets(&self.destabs.col_z, n),
            convert_sets(&self.destabs.row_x, n),
            convert_sets(&self.destabs.row_z, n),
            self.destabs.signs_minus,
            self.destabs.signs_i,
        );

        SparseStabGeneric {
            num_qubits: n,
            stabs,
            destabs,
            rng: self.rng,
        }
    }
}

impl<R> QuantumSimulator for SparseStabHybrid<R>
where
    R: SeedableRng + Rng + Debug,
{
    #[inline]
    fn reset(&mut self) -> &mut Self {
        Self::reset(self)
    }
}

impl<R> CliffordGateable for SparseStabHybrid<R>
where
    R: SeedableRng + Rng + Debug,
{
    /// Pauli X gate. X -> X, Z -> -Z
    #[inline]
    fn x(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qu = q.index();
            // Cross-type: BitSet signs_minus XOR with VecSet col_z (optimized slice)
            self.stabs
                .signs_minus
                .xor_assign_slice(self.stabs.col_z[qu].as_slice());
        }
        self
    }

    /// Pauli Y gate. X -> -X, Z -> -Z
    #[inline]
    fn y(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qu = q.index();
            // Cross-type: VecSet symmetric difference into BitSet
            self.stabs.col_x[qu].xor_symmetric_difference_into_bitset(
                &self.stabs.col_z[qu],
                &mut self.stabs.signs_minus,
            );
        }
        self
    }

    /// Pauli Z gate. X -> -X, Z -> Z
    #[inline]
    fn z(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            // Cross-type: BitSet signs_minus XOR with VecSet col_x (optimized slice)
            self.stabs
                .signs_minus
                .xor_assign_slice(self.stabs.col_x[q.index()].as_slice());
        }
        self
    }

    /// Sqrt of Z gate.
    #[inline]
    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qu = q.index();

            // Cross-type: XOR (BitSet signs_i ∩ VecSet col_x) into BitSet signs_minus (optimized slice)
            self.stabs
                .signs_minus
                .xor_intersection_slice(self.stabs.col_x[qu].as_slice(), &self.stabs.signs_i);
            self.stabs
                .signs_i
                .xor_assign_slice(self.stabs.col_x[qu].as_slice());

            for g in [&mut self.stabs, &mut self.destabs] {
                g.col_z[qu].xor_assign(&g.col_x[qu]);

                for i in g.col_x[qu].iter().copied() {
                    g.row_z[i].toggle(qu);
                }
            }
        }
        self
    }

    /// Hadamard gate. X -> Z, Z -> X
    #[inline]
    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qu = q.index();

            // Cross-type: VecSet intersection into BitSet
            self.stabs.col_x[qu]
                .xor_intersection_into_bitset(&self.stabs.col_z[qu], &mut self.stabs.signs_minus);

            for g in [&mut self.stabs, &mut self.destabs] {
                // Elements in col_x but not in col_z: X -> Z
                for i in g.col_x[qu].iter().copied() {
                    if !g.col_z[qu].contains(i) {
                        g.row_x[i].remove(qu);
                        g.row_z[i].insert(qu);
                    }
                }

                // Elements in col_z but not in col_x: Z -> X
                for i in g.col_z[qu].iter().copied() {
                    if !g.col_x[qu].contains(i) {
                        g.row_z[i].remove(qu);
                        g.row_x[i].insert(qu);
                    }
                }

                mem::swap(&mut g.col_x[qu], &mut g.col_z[qu]);
            }
        }
        self
    }

    /// Controlled-X (CNOT) gate.
    #[inline]
    fn cx(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "CX requires pairs of qubits"
        );

        for pair in qubits.chunks_exact(2) {
            let q1 = pair[0].index();
            let q2 = pair[1].index();

            for g in [&mut self.stabs, &mut self.destabs] {
                let (qu_min, qu_max) = if q1 < q2 { (q1, q2) } else { (q2, q1) };

                // Handle col_x
                {
                    let (_left, right) = g.col_x.split_at_mut(qu_min);
                    let (mid, right) = right.split_at_mut(qu_max - qu_min);
                    let col_x_min = &mut mid[0];
                    let col_x_max = &mut right[0];

                    let (col_x_qu1, col_x_qu2) = if q1 < q2 {
                        (col_x_min, col_x_max)
                    } else {
                        (col_x_max, col_x_min)
                    };

                    for i in col_x_qu1.iter().copied() {
                        g.row_x[i].toggle(q2);
                    }
                    col_x_qu2.xor_assign(col_x_qu1);
                }

                // Handle col_z
                {
                    let (_left, right) = g.col_z.split_at_mut(qu_min);
                    let (mid, right) = right.split_at_mut(qu_max - qu_min);
                    let col_z_min = &mut mid[0];
                    let col_z_max = &mut right[0];

                    let (col_z_qu1, col_z_qu2) = if q1 < q2 {
                        (col_z_min, col_z_max)
                    } else {
                        (col_z_max, col_z_min)
                    };

                    for i in col_z_qu2.iter().copied() {
                        g.row_z[i].toggle(q1);
                    }
                    col_z_qu1.xor_assign(col_z_qu2);
                }
            }
        }
        self
    }

    /// Measures qubits in the Z basis.
    #[inline]
    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        let mut results = Vec::with_capacity(qubits.len());

        for &q in qubits {
            let qu = q.index();
            let deterministic = self.stabs.col_x[qu].is_empty();

            let result = if deterministic {
                self.deterministic_meas(qu)
            } else {
                let outcome = self.rng.coin_flip();
                self.nondeterministic_meas(qu, outcome)
            };
            results.push(result);
        }

        results
    }
}

impl<R> RngManageable for SparseStabHybrid<R>
where
    R: SeedableRng + Rng + Debug,
{
    type Rng = R;

    fn set_rng(&mut self, rng: Self::Rng) {
        self.rng = rng;
    }

    #[inline]
    fn rng(&self) -> &Self::Rng {
        &self.rng
    }

    #[inline]
    fn rng_mut(&mut self) -> &mut Self::Rng {
        &mut self.rng
    }
}

impl<R> StabilizerTableauSimulator for SparseStabHybrid<R>
where
    R: SeedableRng + Rng + Debug,
{
    fn stab_tableau(&self) -> String {
        Self::tableau_string(self.num_qubits, &self.stabs)
    }

    fn destab_tableau(&self) -> String {
        Self::tableau_string(self.num_qubits, &self.destabs)
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}

// ============================================================================
// ForcedMeasurement trait implementations for probability comparison tests
// ============================================================================

use crate::stabilizer_test_utils::{ForcedMeasurement, StabilizerSimulator};

impl<S, R> ForcedMeasurement for SparseStabGeneric<S, R>
where
    S: IndexSet,
    R: SeedableRng + Rng + Debug,
{
    fn mz_forced(&mut self, qubit: usize, forced_outcome: bool) -> MeasurementResult {
        SparseStabGeneric::mz_forced(self, qubit, forced_outcome)
    }
}

impl<R> ForcedMeasurement for SparseStabHybrid<R>
where
    R: SeedableRng + Rng + Debug,
{
    fn mz_forced(&mut self, qubit: usize, forced_outcome: bool) -> MeasurementResult {
        SparseStabHybrid::mz_forced(self, qubit, forced_outcome)
    }
}

// ============================================================================
// StabilizerSimulator implementations
// ============================================================================

impl StabilizerSimulator for SparseStabGeneric<BitSet, PecosRng> {
    fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self::with_seed(num_qubits, seed)
    }
}

impl StabilizerSimulator for SparseStabGeneric<SortedVecSet, PecosRng> {
    fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self::with_seed(num_qubits, seed)
    }
}

impl StabilizerSimulator for SparseStabGeneric<VecSet<usize>, PecosRng> {
    fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self::with_seed(num_qubits, seed)
    }
}

impl StabilizerSimulator for SparseStabHybrid<PecosRng> {
    fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self::with_seed(num_qubits, seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CliffordGateable, Gens};
    use pecos_core::QubitId;

    // Helper to create qubit slice for single qubit
    fn q(n: usize) -> [QubitId; 1] {
        [QubitId(n)]
    }

    // Helper to create qubit slice for two qubits
    fn q2(a: usize, b: usize) -> [QubitId; 2] {
        [QubitId(a), QubitId(b)]
    }

    fn check_matrix(m: &[&str], gens: &Gens) {
        for (r, v) in m.iter().enumerate() {
            let (_, phase, v) = split_pauli(v);

            // TODO: Allow +Y in place of +iW
            // TODO: Return bools instead of doing the asserts here...

            match phase {
                "+" => {
                    assert!(!gens.signs_minus.contains(r));
                    assert!(!gens.signs_i.contains(r));
                }
                "-" => {
                    assert!(gens.signs_minus.contains(r));
                    assert!(!gens.signs_i.contains(r));
                }
                "+i" => {
                    assert!(!gens.signs_minus.contains(r));
                    assert!(gens.signs_i.contains(r));
                }
                "-i" => {
                    assert!(gens.signs_minus.contains(r));
                    assert!(gens.signs_i.contains(r));
                }
                _ => unreachable!(),
            }

            for (c, val) in v.chars().enumerate() {
                match val {
                    'I' => {
                        assert!(!gens.col_x[c].contains(r));
                        assert!(!gens.col_z[c].contains(r));
                        assert!(!gens.row_x[r].contains(c));
                        assert!(!gens.row_z[r].contains(c));
                    }
                    'X' => {
                        assert!(gens.col_x[c].contains(r));
                        assert!(!gens.col_z[c].contains(r));
                        assert!(gens.row_x[r].contains(c));
                        assert!(!gens.row_z[r].contains(c));
                    }
                    'Z' => {
                        assert!(!gens.col_x[c].contains(r));
                        assert!(gens.col_z[c].contains(r));
                        assert!(!gens.row_x[r].contains(c));
                        assert!(gens.row_z[r].contains(c));
                    }
                    'W' => {
                        assert!(gens.col_x[c].contains(r));
                        assert!(gens.col_z[c].contains(r));
                        assert!(gens.row_x[r].contains(c));
                        assert!(gens.row_z[r].contains(c));
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    #[inline]
    fn check_state(state: &SparseStab, stabs: &[&str], destabs: &[&str]) {
        check_matrix(stabs, &state.stabs);
        check_matrix(destabs, &state.destabs);
        // SparseStab::verify_matrix(&state);
        // TODO: Add matrix verification func
    }

    #[inline]
    fn split_pauli(pauli_str: &str) -> (usize, &str, &str) {
        let (phase, pauli_str) = if pauli_str.contains("+i") || pauli_str.contains("-i") {
            pauli_str.split_at(2)
        } else if pauli_str.contains('+') || pauli_str.contains('-') || pauli_str.contains('i') {
            pauli_str.split_at(1)
        } else {
            ("+", pauli_str)
        };
        let n = pauli_str.chars().count();

        let phase = if phase == "i" { "+i" } else { phase };

        (n, phase, pauli_str)
    }

    fn prep_pauli_gens(pauli_vec: &[&str], gens: &mut Gens) {
        // TODO: Think about how to automatically determine the destabilizers you need so you can optionally only provide stabilizers...

        gens.signs_i.clear();
        gens.signs_minus.clear();

        let (n, _, _) = split_pauli(pauli_vec[0]);

        for u in 0..n {
            gens.col_x[u].clear();
            gens.col_z[u].clear();
            gens.row_x[u].clear();
            gens.row_z[u].clear();
        }

        for (ru, pauli_str) in pauli_vec.iter().enumerate() {
            let (n_, phase, pauli_str) = split_pauli(pauli_str);

            assert_eq!(
                n, n_,
                "The number of qubits differs between the first generator and another!"
            );

            match phase {
                "+" => {}
                "-" => {
                    gens.signs_minus.insert(ru);
                }
                "+i" => {
                    gens.signs_i.insert(ru);
                }
                "-i" => {
                    gens.signs_minus.insert(ru);
                    gens.signs_i.insert(ru);
                }
                _ => unreachable!(),
            }

            for (cu, p) in pauli_str.chars().enumerate() {
                match p {
                    'I' => {}
                    'X' => {
                        gens.col_x[cu].insert(ru);
                        gens.row_x[ru].insert(cu);
                    }
                    'W' => {
                        gens.col_x[cu].insert(ru);
                        gens.col_z[cu].insert(ru);
                        gens.row_x[ru].insert(cu);
                        gens.row_z[ru].insert(cu);
                    }
                    'Z' => {
                        gens.col_z[cu].insert(ru);
                        gens.row_z[ru].insert(cu);
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    fn prep_state(stabs: &[&str], destabs: &[&str]) -> SparseStab {
        let mut state = SparseStab::new(3);
        prep_pauli_gens(stabs, &mut state.stabs);
        prep_pauli_gens(destabs, &mut state.destabs);

        state
    }

    #[test]
    fn test_setting_up_stab_state() {
        let tab_stab = vec!["XII", "iIWI", "IIZ"];
        let tab_destab = vec!["ZII", "IXI", "IIX"];

        let state = prep_state(&tab_stab, &tab_destab);
        check_state(&state, &tab_stab, &tab_destab);
    }

    #[test]
    fn test_setting_up_neg_stab_state() {
        let tab_stab = vec!["-XII", "-iIWI", "-IIZ"];
        let tab_destab = vec!["ZII", "IXI", "IIX"];

        let state = prep_state(&tab_stab, &tab_destab);
        check_state(&state, &tab_stab, &tab_destab);
    }

    #[test]
    fn test_nondeterministic_px() {
        for _ in 1_u32..=100 {
            let mut state = prep_state(&["Z"], &["X"]);
            let r0 = state.mpx(&q(0)).into_iter().next().unwrap();
            let meas = state.mx(&q(0)).into_iter().next().unwrap();
            let m1 = meas.outcome;
            let d1 = meas.is_deterministic;
            let m1_int = u8::from(m1);

            assert_eq!(m1_int, 0); // |+X>
            assert!(!r0.is_deterministic); // Not deterministic
            assert!(d1); // Deterministic
        }
    }

    #[test]
    fn test_deterministic_px() {
        let mut state = prep_state(&["X"], &["Z"]);
        let r0 = state.mpx(&q(0)).into_iter().next().unwrap();
        let m0_int = u8::from(r0.outcome);

        assert!(r0.is_deterministic); // Deterministic
        assert_eq!(m0_int, 0); // |+X>
    }

    #[test]
    fn test_nondeterministic_pnx() {
        for _ in 1_u32..=100 {
            let mut state = prep_state(&["Z"], &["X"]);
            let r0 = state.mpnx(&q(0)).into_iter().next().unwrap();
            let result = state.mx(&q(0)).into_iter().next().unwrap();
            let m1_int = u8::from(result.outcome);

            assert_eq!(m1_int, 1); // |-X>
            assert!(!r0.is_deterministic); // Not deterministic
            assert!(result.is_deterministic); // Deterministic
        }
    }

    #[test]
    fn test_deterministic_pnx() {
        let mut state = prep_state(&["-X"], &["Z"]);
        let r0 = state.mpnx(&q(0)).into_iter().next().unwrap();
        let m0_int = u8::from(r0.outcome);

        assert!(r0.is_deterministic); // Deterministic
        assert_eq!(m0_int, 0); // |-X>
    }

    #[test]
    fn test_nondeterministic_py() {
        for _ in 1_u32..=100 {
            let mut state = prep_state(&["Z"], &["X"]);
            let r0 = state.mpy(&q(0)).into_iter().next().unwrap();
            let r1 = state.my(&q(0)).into_iter().next().unwrap();
            let m1_int = u8::from(r1.outcome);

            assert_eq!(m1_int, 0); // |+Y>
            assert!(!r0.is_deterministic); // Not deterministic
            assert!(r1.is_deterministic); // Deterministic
        }
    }

    #[test]
    fn test_deterministic_py() {
        let mut state = prep_state(&["iW"], &["Z"]);
        let r0 = state.mpy(&q(0)).into_iter().next().unwrap();
        let m0_int = u8::from(r0.outcome);

        assert!(r0.is_deterministic); // Deterministic
        assert_eq!(m0_int, 0); // |+Y>
    }

    #[test]
    fn test_nondeterministic_pny() {
        for _ in 1_u32..=100 {
            let mut state = prep_state(&["Z"], &["X"]);
            let r0 = state.mpny(&q(0)).into_iter().next().unwrap();
            let r1 = state.my(&q(0)).into_iter().next().unwrap();
            let m1_int = u8::from(r1.outcome);

            assert_eq!(m1_int, 1); // |-Y>
            assert!(!r0.is_deterministic); // Not deterministic
            assert!(r1.is_deterministic); // Deterministic
        }
    }

    #[test]
    fn test_deterministic_pny() {
        let mut state = prep_state(&["-iW"], &["Z"]);
        let r0 = state.mpny(&q(0)).into_iter().next().unwrap();
        let m0_int = u8::from(r0.outcome);

        assert!(r0.is_deterministic); // Deterministic
        assert_eq!(m0_int, 0); // |-Y>
    }

    #[test]
    fn test_nondeterministic_pz() {
        for _ in 1_u32..=100 {
            let mut state = prep_state(&["X"], &["Z"]);
            let r0 = state.mpz(&q(0)).into_iter().next().unwrap();
            let r1 = state.mz(&q(0)).into_iter().next().unwrap();
            let m1_int = u8::from(r1.outcome);

            assert_eq!(m1_int, 0); // |0>
            assert!(!r0.is_deterministic); // Not deterministic
            assert!(r1.is_deterministic); // Deterministic
        }
    }

    #[test]
    fn test_deterministic_pz() {
        let mut state = prep_state(&["Z"], &["X"]);
        let r0 = state.mpz(&q(0)).into_iter().next().unwrap();
        let m0_int = u8::from(r0.outcome);

        assert!(r0.is_deterministic); // Deterministic
        assert_eq!(m0_int, 0); // |+Z>
    }

    #[test]
    fn test_nondeterministic_pnz() {
        for _ in 1_u32..=100 {
            let mut state = prep_state(&["X"], &["Z"]);
            let r0 = state.mpnz(&q(0)).into_iter().next().unwrap();
            let r1 = state.mz(&q(0)).into_iter().next().unwrap();
            let m1_int = u8::from(r1.outcome);

            assert_eq!(m1_int, 1); // |1>
            assert!(!r0.is_deterministic); // Not deterministic
            assert!(r1.is_deterministic); // Deterministic
        }
    }

    #[test]
    fn test_deterministic_pnz() {
        let mut state = prep_state(&["-Z"], &["X"]);
        let r0 = state.mpnz(&q(0)).into_iter().next().unwrap();
        let m0_int = u8::from(r0.outcome);

        assert!(r0.is_deterministic); // Deterministic
        assert_eq!(m0_int, 0); // |-Z>
    }

    #[test]
    fn test_nondeterministic_mx() {
        let mut state = prep_state(&["Z"], &["X"]);
        let r = state.mx(&q(0)).into_iter().next().unwrap();
        assert!(!r.is_deterministic);
    }

    #[test]
    fn test_deterministic_mx() {
        let mut state0 = prep_state(&["X"], &["Z"]);
        let r0 = state0.mx(&q(0)).into_iter().next().unwrap();
        assert!(r0.is_deterministic);
        assert!(!r0.outcome);

        let mut state1 = prep_state(&["-X"], &["Z"]);
        let r1 = state1.mx(&q(0)).into_iter().next().unwrap();
        assert!(r1.is_deterministic);
        assert!(r1.outcome);
    }

    #[test]
    fn test_nondeterministic_mnx() {
        let mut state = prep_state(&["Z"], &["X"]);
        let r = state.mnx(&q(0)).into_iter().next().unwrap();
        assert!(!r.is_deterministic);
    }

    #[test]
    fn test_deterministic_mnx() {
        let mut state0 = prep_state(&["-X"], &["Z"]);
        let r0 = state0.mnx(&q(0)).into_iter().next().unwrap();
        assert!(r0.is_deterministic);
        assert!(!r0.outcome);

        let mut state1 = prep_state(&["X"], &["Z"]);
        let r1 = state1.mnx(&q(0)).into_iter().next().unwrap();
        assert!(r1.is_deterministic);
        assert!(r1.outcome);
    }

    #[test]
    fn test_nondeterministic_my() {
        let mut state = prep_state(&["Z"], &["X"]);
        let r = state.my(&q(0)).into_iter().next().unwrap();
        assert!(!r.is_deterministic);
    }

    #[test]
    fn test_deterministic_my() {
        let mut state0 = prep_state(&["iW"], &["Z"]);
        let r0 = state0.my(&q(0)).into_iter().next().unwrap();
        assert!(r0.is_deterministic);
        assert!(!r0.outcome);

        let mut state1 = prep_state(&["-iW"], &["Z"]);
        let r1 = state1.my(&q(0)).into_iter().next().unwrap();
        assert!(r1.is_deterministic);
        assert!(r1.outcome);
    }

    #[test]
    fn test_nondeterministic_mny() {
        let mut state = prep_state(&["Z"], &["X"]);
        let r = state.mny(&q(0)).into_iter().next().unwrap();
        assert!(!r.is_deterministic);
    }

    #[test]
    fn test_deterministic_mny() {
        let mut state0 = prep_state(&["-iW"], &["Z"]);
        let r0 = state0.mny(&q(0)).into_iter().next().unwrap();
        assert!(r0.is_deterministic);
        assert!(!r0.outcome);

        let mut state1 = prep_state(&["iW"], &["Z"]);
        let r1 = state1.mny(&q(0)).into_iter().next().unwrap();
        assert!(r1.is_deterministic);
        assert!(r1.outcome);
    }

    #[test]
    fn test_nondeterministic_mz() {
        let mut state = prep_state(&["X"], &["Z"]);
        let r = state.mz(&q(0)).into_iter().next().unwrap();
        assert!(!r.is_deterministic);
    }

    #[test]
    fn test_deterministic_mz() {
        let mut state0 = prep_state(&["Z"], &["X"]);
        let r0 = state0.mz(&q(0)).into_iter().next().unwrap();
        assert!(r0.is_deterministic);
        assert!(!r0.outcome);

        let mut state1 = prep_state(&["-Z"], &["X"]);
        let r1 = state1.mz(&q(0)).into_iter().next().unwrap();
        assert!(r1.is_deterministic);
        assert!(r1.outcome);
    }

    #[test]
    fn test_nondeterministic_mnz() {
        let mut state = prep_state(&["X"], &["Z"]);
        let r = state.mnz(&q(0)).into_iter().next().unwrap();
        assert!(!r.is_deterministic);
    }

    #[test]
    fn test_deterministic_mnz() {
        let mut state0 = prep_state(&["Z"], &["X"]);
        let r0 = state0.mnz(&q(0)).into_iter().next().unwrap();
        assert!(r0.is_deterministic);
        assert!(r0.outcome);

        let mut state1 = prep_state(&["-Z"], &["X"]);
        let r1 = state1.mnz(&q(0)).into_iter().next().unwrap();
        assert!(r1.is_deterministic);
        assert!(!r1.outcome);
    }

    #[test]
    fn test_identity() {
        // I: +X -> +X; +Z -> +Z; +Y -> +Y;

        // +X -> +X
        let mut state = prep_state(&["X"], &["Z"]);
        state.identity(&q(0));
        check_state(&state, &["X"], &["Z"]);

        // +Y -> -Y
        let mut state = prep_state(&["iW"], &["X"]);
        state.identity(&q(0));
        check_state(&state, &["iW"], &["X"]);

        // +Z -> -Z
        let mut state = prep_state(&["Z"], &["X"]);
        state.identity(&q(0));
        check_state(&state, &["Z"], &["X"]);

        // -IYI -> +IYI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.identity(&q(1));
        check_state(&state, &["-iIWI"], &["IXI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_x() {
        // X: +X -> +X; +Z -> -Z; +Y -> -Y;

        // +X -> +X
        let mut state = prep_state(&["X"], &["Z"]);
        state.x(&q(0));
        check_state(&state, &["X"], &["Z"]);

        // +Y -> -Y
        let mut state = prep_state(&["iW"], &["X"]);
        state.x(&q(0));
        check_state(&state, &["-iW"], &["X"]);

        // +Z -> -Z
        let mut state = prep_state(&["Z"], &["X"]);
        state.x(&q(0));
        check_state(&state, &["-Z"], &["X"]);

        // -IYI -> +IYI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.x(&q(1));
        check_state(&state, &["iIWI"], &["IXI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_y() {
        // Y: +X -> -X; +Z -> -Z; +Y -> +Y;

        // +X -> -X
        let mut state = prep_state(&["X"], &["Z"]);
        state.y(&q(0));
        check_state(&state, &["-X"], &["Z"]);

        // +Y -> +Y
        let mut state = prep_state(&["iW"], &["X"]);
        state.y(&q(0));
        check_state(&state, &["iW"], &["X"]);

        // +Z -> -Z
        let mut state = prep_state(&["Z"], &["X"]);
        state.y(&q(0));
        check_state(&state, &["-Z"], &["X"]);

        // -IXI -> +IXI
        let mut state = prep_state(&["-IXI"], &["IZI"]);
        state.y(&q(1));
        check_state(&state, &["IXI"], &["IZI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_z() {
        // Z: +X -> -X; +Z -> +Z; +Y -> -Y;

        // +X -> -X
        let mut state = prep_state(&["X"], &["Z"]);
        state.z(&q(0));
        check_state(&state, &["-X"], &["Z"]);

        // +Y -> -Y
        let mut state = prep_state(&["iW"], &["X"]);
        state.z(&q(0));
        check_state(&state, &["-iW"], &["X"]);

        // +Z -> +Z
        let mut state = prep_state(&["Z"], &["X"]);
        state.z(&q(0));
        check_state(&state, &["Z"], &["X"]);

        // -IXI -> +IXI
        let mut state = prep_state(&["-IXI"], &["IZI"]);
        state.z(&q(1));
        check_state(&state, &["IXI"], &["IZI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_sx() {
        // SX: +X -> +X; +Z -> -Y; +Y -> +Z;

        // +X -> +X
        let mut state = prep_state(&["X"], &["Z"]);
        state.sx(&q(0));
        check_state(&state, &["X"], &["W"]);

        // +Y -> +Z
        let mut state = prep_state(&["iW"], &["X"]);
        state.sx(&q(0));
        check_state(&state, &["Z"], &["X"]);

        // +Z -> -Y
        let mut state = prep_state(&["Z"], &["X"]);
        state.sx(&q(0));
        check_state(&state, &["-iW"], &["X"]);

        // -IYI -> -IZI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.sx(&q(1));
        check_state(&state, &["-IZI"], &["IXI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_sxdg() {
        // SXdg: +X -> +X; +Z -> +Y; +Y -> -Z;

        // +X -> +X
        let mut state = prep_state(&["X"], &["Z"]);
        state.sxdg(&q(0));
        check_state(&state, &["X"], &["W"]);

        // +Y -> -Z
        let mut state = prep_state(&["iW"], &["X"]);
        state.sxdg(&q(0));
        check_state(&state, &["-Z"], &["X"]);

        // +Z -> +Y
        let mut state = prep_state(&["Z"], &["X"]);
        state.sxdg(&q(0));
        check_state(&state, &["iW"], &["X"]);

        // -IYI -> +IZI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.sxdg(&q(1));
        check_state(&state, &["IZI"], &["IXI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_sy() {
        // SY: +X -> -Z; +Z -> +X; +Y -> +Y;

        // +X -> -Z
        let mut state = prep_state(&["X"], &["Z"]);
        state.sy(&q(0));
        check_state(&state, &["-Z"], &["X"]);

        // +Y -> +Y
        let mut state = prep_state(&["iW"], &["X"]);
        state.sy(&q(0));
        check_state(&state, &["iW"], &["Z"]);

        // +Z -> +X
        let mut state = prep_state(&["Z"], &["X"]);
        state.sy(&q(0));
        check_state(&state, &["X"], &["Z"]);

        // -IYI -> -IYI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.sy(&q(1));
        check_state(&state, &["-iIWI"], &["IZI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_sydg() {
        // SYdg: +X -> +Z; +Z -> -X; +Y -> +Y;

        // +X -> +Z
        let mut state = prep_state(&["X"], &["Z"]);
        state.sydg(&q(0));
        check_state(&state, &["Z"], &["X"]);

        // +Y -> +Y
        let mut state = prep_state(&["iW"], &["X"]);
        state.sydg(&q(0));
        check_state(&state, &["iW"], &["Z"]);

        // +Z -> -X
        let mut state = prep_state(&["Z"], &["X"]);
        state.sydg(&q(0));
        check_state(&state, &["-X"], &["Z"]);

        // -IYI -> -IYI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.sydg(&q(1));
        check_state(&state, &["-iIWI"], &["IZI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_sz() {
        // SZ: +X -> +Y; +Z -> +Z; +Y -> -X;

        // +X -> +Y
        let mut state = prep_state(&["X"], &["Z"]);
        state.sz(&q(0));
        check_state(&state, &["iW"], &["Z"]);

        // +Y -> -X
        let mut state = prep_state(&["iW"], &["X"]);
        state.sz(&q(0));
        check_state(&state, &["-X"], &["W"]);

        // +Z -> +Z
        let mut state = prep_state(&["Z"], &["X"]);
        state.sz(&q(0));
        check_state(&state, &["Z"], &["W"]);

        // -IYI -> +IXI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.sz(&q(1));
        check_state(&state, &["IXI"], &["IWI"]);
    }

    #[test]
    fn test_szdg() {
        // SZdg: +X -> -Y; +Z -> +Z; +Y -> +X;

        // +X -> -Y
        let mut state = prep_state(&["X"], &["Z"]);
        state.szdg(&q(0));
        check_state(&state, &["-iW"], &["Z"]);

        // +Y -> +X
        let mut state = prep_state(&["iW"], &["X"]);
        state.szdg(&q(0));
        check_state(&state, &["X"], &["W"]);

        // +Z -> +Z
        let mut state = prep_state(&["Z"], &["X"]);
        state.szdg(&q(0));
        check_state(&state, &["Z"], &["W"]);

        // -IYI -> -IXI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.szdg(&q(1));
        check_state(&state, &["-IXI"], &["IWI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_h() {
        // H: X -> Z; Z -> X; Y -> -Y;

        // +X -> +Z
        let mut state = prep_state(&["X"], &["Z"]);
        state.h(&q(0));
        check_state(&state, &["Z"], &["X"]);

        // +Y -> -Y
        let mut state = prep_state(&["iW"], &["X"]);
        state.h(&q(0));
        check_state(&state, &["-iW"], &["Z"]);

        // +Z -> +X
        let mut state = prep_state(&["Z"], &["X"]);
        state.h(&q(0));
        check_state(&state, &["X"], &["Z"]);

        // -IYI -> +IYI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.h(&q(1));
        check_state(&state, &["iIWI"], &["IZI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_h2() {
        // H2: X -> -Z, Z -> -X, Y -> -Y

        // +X -> -Z
        let mut state = prep_state(&["X"], &["Z"]);
        state.h2(&q(0));
        check_state(&state, &["-Z"], &["X"]);

        // +Y -> -Y
        let mut state = prep_state(&["iW"], &["X"]);
        state.h2(&q(0));
        check_state(&state, &["-iW"], &["Z"]);

        // +Z -> -X
        let mut state = prep_state(&["Z"], &["X"]);
        state.h2(&q(0));
        check_state(&state, &["-X"], &["Z"]);

        // -IYI -> +IYI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.h2(&q(1));
        check_state(&state, &["iIWI"], &["IZI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_h3() {
        // H3: X -> Y, Z -> -Z, Y -> X

        // +X -> Y
        let mut state = prep_state(&["X"], &["Z"]);
        state.h3(&q(0));
        check_state(&state, &["iW"], &["Z"]);

        // +Y -> +X
        let mut state = prep_state(&["iW"], &["X"]);
        state.h3(&q(0));
        check_state(&state, &["X"], &["W"]);

        // +Z -> -Z
        let mut state = prep_state(&["Z"], &["X"]);
        state.h3(&q(0));
        check_state(&state, &["-Z"], &["W"]);

        // -IYI -> -IXI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.h3(&q(1));
        check_state(&state, &["-IXI"], &["IWI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_h4() {
        // H4: X -> -Y, Z -> -Z, Y -> -X

        // +X -> -Y
        let mut state = prep_state(&["X"], &["Z"]);
        state.h4(&q(0));
        check_state(&state, &["-iW"], &["Z"]);

        // +Y -> -X
        let mut state = prep_state(&["iW"], &["X"]);
        state.h4(&q(0));
        check_state(&state, &["-X"], &["W"]);

        // +Z -> -Z
        let mut state = prep_state(&["Z"], &["X"]);
        state.h4(&q(0));
        check_state(&state, &["-Z"], &["W"]);

        // -IYI -> IXI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.h4(&q(1));
        check_state(&state, &["IXI"], &["IWI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_h5() {
        // H5: X -> -X, Z -> Y, Y -> Z

        // +X -> -X
        let mut state = prep_state(&["X"], &["Z"]);
        state.h5(&q(0));
        check_state(&state, &["-X"], &["W"]);

        // +Y -> +Z
        let mut state = prep_state(&["iW"], &["X"]);
        state.h5(&q(0));
        check_state(&state, &["Z"], &["X"]);

        // +Z -> +Y
        let mut state = prep_state(&["Z"], &["X"]);
        state.h5(&q(0));
        check_state(&state, &["iW"], &["X"]);

        // -IYI -> -IZI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.h5(&q(1));
        check_state(&state, &["-IZI"], &["IXI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_h6() {
        // H6: X -> -X, Z -> -Y, Y -> -Z

        // +X -> -X
        let mut state = prep_state(&["X"], &["Z"]);
        state.h6(&q(0));
        check_state(&state, &["-X"], &["W"]);

        // +Y -> -Z
        let mut state = prep_state(&["iW"], &["X"]);
        state.h6(&q(0));
        check_state(&state, &["-Z"], &["X"]);

        // +Z -> -Y
        let mut state = prep_state(&["Z"], &["X"]);
        state.h6(&q(0));
        check_state(&state, &["-iW"], &["X"]);

        // -IYI -> IZI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.h6(&q(1));
        check_state(&state, &["IZI"], &["IXI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_f() {
        // F: X -> Y, Z -> X, Y -> Z

        // +X -> +Y
        let mut state = prep_state(&["X"], &["Z"]);
        state.f(&q(0));
        check_state(&state, &["iW"], &["X"]);

        // +Y -> +Z
        let mut state = prep_state(&["iW"], &["X"]);
        state.f(&q(0));
        check_state(&state, &["Z"], &["W"]);

        // +Z -> +X
        let mut state = prep_state(&["Z"], &["X"]);
        state.f(&q(0));
        check_state(&state, &["X"], &["W"]);

        // -IYI -> -IZI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.f(&q(1));
        check_state(&state, &["-IZI"], &["IWI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_fdg() {
        // Fdg: X -> Z, Z -> Y, Y -> X

        // +X -> +Z
        let mut state = prep_state(&["X"], &["Z"]);
        state.fdg(&q(0));
        check_state(&state, &["Z"], &["W"]);

        // +Y -> +X
        let mut state = prep_state(&["iW"], &["X"]);
        state.fdg(&q(0));
        check_state(&state, &["X"], &["Z"]);

        // +Z -> +Y
        let mut state = prep_state(&["Z"], &["X"]);
        state.fdg(&q(0));
        check_state(&state, &["iW"], &["Z"]);

        // -IYI -> -IXI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.fdg(&q(1));
        check_state(&state, &["-IXI"], &["IZI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_f2() {
        // F2: X -> -Z, Z -> Y, Y -> -X

        // +X -> -Z
        let mut state = prep_state(&["X"], &["Z"]);
        state.f2(&q(0));
        check_state(&state, &["-Z"], &["W"]);

        // +Y -> -X
        let mut state = prep_state(&["iW"], &["X"]);
        state.f2(&q(0));
        check_state(&state, &["-X"], &["Z"]);

        // +Z -> +Y
        let mut state = prep_state(&["Z"], &["X"]);
        state.f2(&q(0));
        check_state(&state, &["iW"], &["Z"]);

        // -IYI -> IXI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.f2(&q(1));
        check_state(&state, &["IXI"], &["IZI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_f2dg() {
        // F2dg: X -> -Y, Z -> -X, Y -> Z

        // +X -> -Y
        let mut state = prep_state(&["X"], &["Z"]);
        state.f2dg(&q(0));
        check_state(&state, &["-iW"], &["X"]);

        // +Y -> +Z
        let mut state = prep_state(&["iW"], &["X"]);
        state.f2dg(&q(0));
        check_state(&state, &["Z"], &["W"]);

        // +Z -> -X
        let mut state = prep_state(&["Z"], &["X"]);
        state.f2dg(&q(0));
        check_state(&state, &["-X"], &["W"]);

        // -IYI -> -IZI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.f2dg(&q(1));
        check_state(&state, &["-IZI"], &["IWI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_f3() {
        // F3: X -> Y, Z -> -X, Y -> -Z

        // +X -> +Y
        let mut state = prep_state(&["X"], &["Z"]);
        state.f3(&q(0));
        check_state(&state, &["iW"], &["X"]);

        // +Y -> -Z
        let mut state = prep_state(&["iW"], &["X"]);
        state.f3(&q(0));
        check_state(&state, &["-Z"], &["W"]);

        // +Z -> -X
        let mut state = prep_state(&["Z"], &["X"]);
        state.f3(&q(0));
        check_state(&state, &["-X"], &["W"]);

        // -IYI -> IZI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.f3(&q(1));
        check_state(&state, &["IZI"], &["IWI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_f3dg() {
        // F3dg: X -> -Z, Z -> -Y, Y -> X

        // +X -> -Z
        let mut state = prep_state(&["X"], &["Z"]);
        state.f3dg(&q(0));
        check_state(&state, &["-Z"], &["W"]);

        // +Y -> +X
        let mut state = prep_state(&["iW"], &["X"]);
        state.f3dg(&q(0));
        check_state(&state, &["X"], &["Z"]);

        // +Z -> -Y
        let mut state = prep_state(&["Z"], &["X"]);
        state.f3dg(&q(0));
        check_state(&state, &["-iW"], &["Z"]);

        // -IYI -> -IXI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.f3dg(&q(1));
        check_state(&state, &["-IXI"], &["IZI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_f4() {
        // F4: X -> Z, Z -> -Y, Y -> -X

        // +X -> +Z
        let mut state = prep_state(&["X"], &["Z"]);
        state.f4(&q(0));
        check_state(&state, &["Z"], &["W"]);

        // +Y -> -X
        let mut state = prep_state(&["iW"], &["X"]);
        state.f4(&q(0));
        check_state(&state, &["-X"], &["Z"]);

        // +Z -> -Y
        let mut state = prep_state(&["Z"], &["X"]);
        state.f4(&q(0));
        check_state(&state, &["-iW"], &["Z"]);

        // -IYI -> IXI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.f4(&q(1));
        check_state(&state, &["IXI"], &["IZI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_f4dg() {
        // F4dg: X -> -Y, Z -> X, Y -> -Z

        // +X -> -Y
        let mut state = prep_state(&["X"], &["Z"]);
        state.f4dg(&q(0));
        check_state(&state, &["-iW"], &["X"]);

        // +Y -> -Z
        let mut state = prep_state(&["iW"], &["X"]);
        state.f4dg(&q(0));
        check_state(&state, &["-Z"], &["W"]);

        // +Z -> +X
        let mut state = prep_state(&["Z"], &["X"]);
        state.f4dg(&q(0));
        check_state(&state, &["X"], &["W"]);

        // -IYI -> +IZI
        let mut state = prep_state(&["-iIWI"], &["IXI"]);
        state.f4dg(&q(1));
        check_state(&state, &["IZI"], &["IWI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_cx() {
        // CX: +IX -> +IX; +IZ -> +ZZ; +XI -> +XX; +ZI -> +ZI;

        // TODO: Expand the set of stabilizer transformations evaluated.

        // +IX -> +IX
        let mut state = prep_state(&["IX"], &["IZ"]);
        state.cx(&q2(0, 1));
        check_state(&state, &["IX"], &["ZZ"]);

        // +IZ -> +ZZ
        let mut state = prep_state(&["IZ"], &["IX"]);
        state.cx(&q2(0, 1));
        check_state(&state, &["ZZ"], &["IX"]);

        // +XI -> +XX
        let mut state = prep_state(&["XI"], &["ZI"]);
        state.cx(&q2(0, 1));
        check_state(&state, &["XX"], &["ZI"]);

        // +ZI -> +ZI
        let mut state = prep_state(&["ZI"], &["XI"]);
        state.cx(&q2(0, 1));
        check_state(&state, &["ZI"], &["XX"]);
    }

    #[test]
    fn test_cy() {
        // CY: +IX -> +ZX; +IZ -> +ZZ; +XI -> +XY; +ZI -> +ZI;
        // Note: CY = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ Y (standard convention)

        // TODO: Expand the set of stabilizer transformations evaluated.

        // +IX -> +ZX
        let mut state = prep_state(&["IX"], &["IZ"]);
        state.cy(&q2(0, 1));
        check_state(&state, &["ZX"], &["ZZ"]);

        // +IZ -> +ZZ
        let mut state = prep_state(&["IZ"], &["IX"]);
        state.cy(&q2(0, 1));
        check_state(&state, &["ZZ"], &["ZX"]);

        // +XI -> +XY = +iXW (Y = iXZ = iW)
        let mut state = prep_state(&["XI"], &["ZI"]);
        state.cy(&q2(0, 1));
        check_state(&state, &["+iXW"], &["ZI"]);

        // +ZI -> +ZI
        let mut state = prep_state(&["ZI"], &["XI"]);
        state.cy(&q2(0, 1));
        check_state(&state, &["ZI"], &["XW"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_cz() {
        // CZ: +IX -> +ZX; +IZ -> +IZ; +XI -> +XZ; +ZI -> +ZI;

        // TODO: Expand the set of stabilizer transformations evaluated.

        // +IX -> +ZX
        let mut state = prep_state(&["IX"], &["IZ"]);
        state.cz(&q2(0, 1));
        check_state(&state, &["ZX"], &["IZ"]);

        // +IZ -> +IZ
        let mut state = prep_state(&["IZ"], &["IX"]);
        state.cz(&q2(0, 1));
        check_state(&state, &["IZ"], &["ZX"]);

        // +XI -> +XZ
        let mut state = prep_state(&["XI"], &["ZI"]);
        state.cz(&q2(0, 1));
        check_state(&state, &["XZ"], &["ZI"]);

        // +ZI -> +ZI
        let mut state = prep_state(&["ZI"], &["XI"]);
        state.cz(&q2(0, 1));
        check_state(&state, &["ZI"], &["XZ"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_sxx() {
        // SXX: XI -> XI
        //      IX -> IX
        //      ZI -> -YX
        //      IZ -> -XY

        // TODO: Expand the set of stabilizer transformations evaluated.

        // +IX -> +XI
        let mut state = prep_state(&["IX"], &["IZ"]);
        state.sxx(&q2(0, 1));
        check_state(&state, &["IX"], &["XW"]);

        // +IZ -> -XY
        let mut state = prep_state(&["IZ"], &["IX"]);
        state.sxx(&q2(0, 1));
        check_state(&state, &["-iXW"], &["IX"]);

        // +XI -> +XI
        let mut state = prep_state(&["XI"], &["ZI"]);
        state.sxx(&q2(0, 1));
        check_state(&state, &["XI"], &["WX"]);

        // +ZI -> -YX
        let mut state = prep_state(&["ZI"], &["XI"]);
        state.sxx(&q2(0, 1));
        check_state(&state, &["-iWX"], &["XI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_sxxdg() {
        // SXXdg: XI -> XI
        //        IX -> IX
        //        ZI -> YX
        //        IZ -> XY

        // TODO: Expand the set of stabilizer transformations evaluated.

        // +IX -> +XI
        let mut state = prep_state(&["IX"], &["IZ"]);
        state.sxxdg(&q2(0, 1));
        check_state(&state, &["IX"], &["XW"]);

        // +IZ -> +XY
        let mut state = prep_state(&["IZ"], &["IX"]);
        state.sxxdg(&q2(0, 1));
        check_state(&state, &["iXW"], &["IX"]);

        // +XI -> +XI
        let mut state = prep_state(&["XI"], &["ZI"]);
        state.sxxdg(&q2(0, 1));
        check_state(&state, &["XI"], &["WX"]);

        // +ZI -> +YX
        let mut state = prep_state(&["ZI"], &["XI"]);
        state.sxxdg(&q2(0, 1));
        check_state(&state, &["iWX"], &["XI"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_syy() {
        // SYY: XI -> -ZY
        //      IX -> -YZ
        //      ZI -> XY
        //      IZ -> YX

        // TODO: Expand the set of stabilizer transformations evaluated.

        // +IX -> -YZ
        let mut state = prep_state(&["IX"], &["IZ"]);
        state.syy(&q2(0, 1));
        check_state(&state, &["-iWZ"], &["WX"]);

        // +IZ -> +YX
        let mut state = prep_state(&["IZ"], &["IX"]);
        state.syy(&q2(0, 1));
        check_state(&state, &["iWX"], &["WZ"]);

        // +XI -> -ZY
        let mut state = prep_state(&["XI"], &["ZI"]);
        state.syy(&q2(0, 1));
        check_state(&state, &["-iZW"], &["XW"]);

        // +ZI -> +XY
        let mut state = prep_state(&["ZI"], &["XI"]);
        state.syy(&q2(0, 1));
        check_state(&state, &["iXW"], &["ZW"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_syydg() {
        // SYYdg: XI -> ZY
        //        IX -> YZ
        //        ZI -> -XY
        //        IZ -> -YX

        // TODO: Expand the set of stabilizer transformations evaluated.

        // +IX -> YZ
        let mut state = prep_state(&["IX"], &["IZ"]);
        state.syydg(&q2(0, 1));
        check_state(&state, &["iWZ"], &["WX"]);

        // +IZ -> -YX
        let mut state = prep_state(&["IZ"], &["IX"]);
        state.syydg(&q2(0, 1));
        check_state(&state, &["-iWX"], &["WZ"]);

        // +XI -> ZY
        let mut state = prep_state(&["XI"], &["ZI"]);
        state.syydg(&q2(0, 1));
        check_state(&state, &["iZW"], &["XW"]);
        // +ZI -> +XY
        let mut state = prep_state(&["ZI"], &["XI"]);
        state.syydg(&q2(0, 1));
        check_state(&state, &["-iXW"], &["ZW"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_szz() {
        // SZZ: +IX -> +ZY;
        //      +IZ -> +IZ;
        //      +XI -> +ZY;
        //      +ZI -> +ZI;

        // TODO: Expand the set of stabilizer transformations evaluated.

        // +IX -> ZY
        let mut state = prep_state(&["IX"], &["IZ"]);
        state.szz(&q2(0, 1));
        check_state(&state, &["iZW"], &["IZ"]);

        // +IZ -> IZ
        let mut state = prep_state(&["IZ"], &["IX"]);
        state.szz(&q2(0, 1));
        check_state(&state, &["IZ"], &["ZW"]);

        // +XI -> YZ
        let mut state = prep_state(&["XI"], &["ZI"]);
        state.szz(&q2(0, 1));
        check_state(&state, &["iWZ"], &["ZI"]);

        // +ZI -> ZI
        let mut state = prep_state(&["ZI"], &["XI"]);
        state.szz(&q2(0, 1));
        check_state(&state, &["ZI"], &["WZ"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_szzdg() {
        // SZZ: +IX -> -ZY;
        //      +IZ -> +IZ;
        //      +XI -> -ZY;
        //      +ZI -> +ZI;

        // TODO: Expand the set of stabilizer transformations evaluated.

        // +IX -> -ZY
        let mut state = prep_state(&["IX"], &["IZ"]);
        state.szzdg(&q2(0, 1));
        check_state(&state, &["-iZW"], &["IZ"]);

        // +IZ -> IZ
        let mut state = prep_state(&["IZ"], &["IX"]);
        state.szzdg(&q2(0, 1));
        check_state(&state, &["IZ"], &["ZW"]);

        // +XI -> -YZ
        let mut state = prep_state(&["XI"], &["ZI"]);
        state.szzdg(&q2(0, 1));
        check_state(&state, &["-iWZ"], &["ZI"]);

        // +ZI -> ZI
        let mut state = prep_state(&["ZI"], &["XI"]);
        state.szzdg(&q2(0, 1));
        check_state(&state, &["ZI"], &["WZ"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_swap() {
        // SWAP: +IX -> +XI;
        //       +IZ -> +ZI;
        //       +XI -> +IX;
        //       +ZI -> +IZ;

        // TODO: Expand the set of stabilizer transformations evaluated.

        // +IX -> +XI
        let mut state = prep_state(&["IX"], &["IZ"]);
        state.swap(&q2(0, 1));
        check_state(&state, &["XI"], &["ZI"]);

        // +IZ -> +ZI
        let mut state = prep_state(&["IZ"], &["IX"]);
        state.swap(&q2(0, 1));
        check_state(&state, &["ZI"], &["XI"]);

        // +XI -> +IX
        let mut state = prep_state(&["XI"], &["ZI"]);
        state.swap(&q2(0, 1));
        check_state(&state, &["IX"], &["IZ"]);

        // +ZI -> +IZ
        let mut state = prep_state(&["ZI"], &["XI"]);
        state.swap(&q2(0, 1));
        check_state(&state, &["IZ"], &["IX"]);
    }

    #[test]
    #[expect(clippy::shadow_unrelated)]
    fn test_g2() {
        // G2: +XI -> +IX
        //     +IX -> +XI
        //     +ZI -> +XZ
        //     +IZ -> +ZX

        // TODO: Expand the set of stabilizer transformations evaluated.

        // +IX -> +XI
        let mut state = prep_state(&["IX"], &["IZ"]);
        state.g(&q2(0, 1));
        check_state(&state, &["XI"], &["ZX"]);

        // +IZ -> +ZX
        let mut state = prep_state(&["IZ"], &["IX"]);
        state.g(&q2(0, 1));
        check_state(&state, &["ZX"], &["XI"]);

        // +XI -> +IX
        let mut state = prep_state(&["XI"], &["ZI"]);
        state.g(&q2(0, 1));
        check_state(&state, &["IX"], &["XZ"]);

        // +ZI -> +XZ
        let mut state = prep_state(&["ZI"], &["XI"]);
        state.g(&q2(0, 1));
        check_state(&state, &["XZ"], &["IX"]);
    }

    fn one_bit_z_teleport(mut state: SparseStab) -> (SparseStab, bool) {
        state.cx(&q2(1, 0)).h(&q(1));
        let r1 = state.mz(&q(1)).into_iter().next().unwrap();
        if r1.outcome {
            state.z(&q(0));
        }
        (state, r1.is_deterministic)
    }

    /// Test one-bit Z teleportation of |+X>
    #[test]
    fn test_nondeterministic_mz_one_bit_z_teleportation_of_x() {
        // See: arXiv:quant-ph/0002039

        for _ in 1_u32..=100 {
            let d1;
            let mut state: SparseStab = SparseStab::new(2);
            state.h(&q(1)); // Set input to |+>
            (state, d1) = one_bit_z_teleport(state);
            // X basis meas
            state.h(&q(0));
            let r0 = state.mz(&q(0)).into_iter().next().unwrap();
            let m0_int = u8::from(r0.outcome);
            assert_eq!(m0_int, 0); // |+> -> 0 == false
            assert!(!d1); // Not deterministic
            assert!(r0.is_deterministic); // Deterministic
        }
    }

    /// Test one-bit Z teleportation of |-X>
    #[test]
    fn test_nondeterministic_mz_one_bit_z_teleportation_of_nx() {
        // See: arXiv:quant-ph/0002039

        for _ in 1_u32..=100 {
            let d1;
            let mut state: SparseStab = SparseStab::new(2);
            state.x(&q(1));
            state.h(&q(1)); // Set input to |->
            (state, d1) = one_bit_z_teleport(state);
            // X basis meas
            state.h(&q(0));
            let r0 = state.mz(&q(0)).into_iter().next().unwrap();
            let m0_int = u8::from(r0.outcome);
            assert_eq!(m0_int, 1); // |-> -> 1 == true
            assert!(!d1); // Not deterministic
            assert!(r0.is_deterministic); // Deterministic
        }
    }

    /// Test one-bit Z teleportation of |+Y>
    #[test]
    fn test_nondeterministic_mz_one_bit_z_teleportation_of_y() {
        // See: arXiv:quant-ph/0002039

        for _ in 1_u32..=100 {
            let d1;
            let mut state: SparseStab = SparseStab::new(2);
            state.sxdg(&q(1)); // Set input to |+i>
            (state, d1) = one_bit_z_teleport(state);
            // Y basis meas
            state.sx(&q(0)); // Y -> Z
            let r0 = state.mz(&q(0)).into_iter().next().unwrap();
            let m0_int = u8::from(r0.outcome);
            assert_eq!(m0_int, 0); // |+X> -> 0 == false
            assert!(!d1); // Not deterministic
            assert!(r0.is_deterministic); // Deterministic
        }
    }

    /// Test one-bit Z teleportation of |-Y>
    #[test]
    fn test_nondeterministic_mz_one_bit_z_teleportation_of_ny() {
        // See: arXiv:quant-ph/0002039

        for _ in 1_u32..=100 {
            let d1;
            let mut state: SparseStab = SparseStab::new(2);
            state.x(&q(1));
            state.sxdg(&q(1)); // Set input to |-i>
            (state, d1) = one_bit_z_teleport(state);
            // Y basis meas
            state.sx(&q(0)); // Y -> Z
            let r0 = state.mz(&q(0)).into_iter().next().unwrap();
            let m0_int = u8::from(r0.outcome);
            assert_eq!(m0_int, 1); // |-Y> -> 1 == true
            assert!(!d1); // Not deterministic
            assert!(r0.is_deterministic); // Deterministic
        }
    }

    /// Test one-bit Z teleportation of |+Z>
    #[test]
    fn test_nondeterministic_mz_one_bit_z_teleportation_of_z() {
        // See: arXiv:quant-ph/0002039

        for _ in 1_u32..=100 {
            let d1;
            let mut state: SparseStab = SparseStab::new(2);
            // Set input to |0>
            (state, d1) = one_bit_z_teleport(state);
            let r0 = state.mz(&q(0)).into_iter().next().unwrap();
            let m0_int = u8::from(r0.outcome);
            assert_eq!(m0_int, 0); // |0>
            assert!(!d1); // Not deterministic
            assert!(r0.is_deterministic); // Deterministic
        }
    }

    /// Test one-bit Z teleportation of |-Z>
    #[test]
    fn test_nondeterministic_mz_one_bit_z_teleportation_of_nz() {
        // See: arXiv:quant-ph/0002039

        for _ in 1_u32..=100 {
            let d1;
            let mut state: SparseStab = SparseStab::new(2);
            state.x(&q(1)); // Set input to |1>
            (state, d1) = one_bit_z_teleport(state);
            let r0 = state.mz(&q(0)).into_iter().next().unwrap();
            let m0_int = u8::from(r0.outcome);
            assert_eq!(m0_int, 1); // |1> -> 1 == true
            assert!(!d1); // Not deterministic
            assert!(r0.is_deterministic); // Deterministic
        }
    }

    fn teleport(mut state: SparseStab) -> (SparseStab, bool, bool) {
        // |psi> -----.-H-MZ=m0
        //            |
        // |0>   -H-.-X---MZ=m1
        //          |
        // |0>   ---X------------X^m1-Z^m0-MZ=m2

        state.h(&q(1));
        state.cx(&q2(1, 2));
        state.cx(&q2(0, 1));
        state.h(&q(0));
        let r0 = state.mz(&q(0)).into_iter().next().unwrap();
        let r1 = state.mz(&q(1)).into_iter().next().unwrap();
        if r1.outcome {
            state.x(&q(2));
        }
        if r0.outcome {
            state.z(&q(2));
        }
        (state, r0.is_deterministic, r1.is_deterministic)
    }

    #[test]
    fn test_nondeterministic_mz_via_teleportation_x() {
        for _ in 1_u32..=100 {
            let d0;
            let d1;
            let mut state: SparseStab = SparseStab::new(3);
            state.h(&q(0));
            (state, d0, d1) = teleport(state);
            state.h(&q(2));
            let r2 = state.mz(&q(2)).into_iter().next().unwrap();
            let m2_int = u8::from(r2.outcome);
            assert_eq!(m2_int, 0);
            assert!(!d0);
            assert!(!d1);
            assert!(r2.is_deterministic);
        }
    }

    #[test]
    fn test_nondeterministic_mz_via_teleportation_nx() {
        for _ in 1_u32..=100 {
            let d0;
            let d1;
            let mut state: SparseStab = SparseStab::new(3);
            state.x(&q(0));
            state.h(&q(0));
            (state, d0, d1) = teleport(state);
            state.h(&q(2));
            let r2 = state.mz(&q(2)).into_iter().next().unwrap();
            let m2_int = u8::from(r2.outcome);

            assert_eq!(m2_int, 1);
            assert!(!d0);
            assert!(!d1);
            assert!(r2.is_deterministic);
        }
    }

    #[test]
    fn test_nondeterministic_mz_via_teleportation_y() {
        for _ in 1_u32..=100 {
            let d0;
            let d1;
            let mut state: SparseStab = SparseStab::new(3);
            state.sxdg(&q(0));
            (state, d0, d1) = teleport(state);
            state.sx(&q(2));
            let r2 = state.mz(&q(2)).into_iter().next().unwrap();
            let m2_int = u8::from(r2.outcome);
            assert_eq!(m2_int, 0);
            assert!(!d0);
            assert!(!d1);
            assert!(r2.is_deterministic);
        }
    }

    #[test]
    fn test_nondeterministic_mz_via_teleportation_ny() {
        for _ in 1_u32..=100 {
            let d0;
            let d1;
            let mut state: SparseStab = SparseStab::new(3);
            state.x(&q(0));
            state.sxdg(&q(0));
            (state, d0, d1) = teleport(state);
            state.sx(&q(2));
            let r2 = state.mz(&q(2)).into_iter().next().unwrap();
            let m2_int = u8::from(r2.outcome);
            assert_eq!(m2_int, 1);
            assert!(!d0);
            assert!(!d1);
            assert!(r2.is_deterministic);
        }
    }

    #[test]
    fn test_nondeterministic_mz_via_teleportation_z() {
        for _ in 1_u32..=100 {
            let d0;
            let d1;
            let mut state: SparseStab = SparseStab::new(3);
            (state, d0, d1) = teleport(state);
            let r2 = state.mz(&q(2)).into_iter().next().unwrap();
            let m2_int = u8::from(r2.outcome);

            assert_eq!(m2_int, 0);
            assert!(!d0);
            assert!(!d1);
            assert!(r2.is_deterministic);
        }
    }

    #[test]
    fn test_nondeterministic_mz_via_teleportation_nz() {
        for _ in 1_u32..=100 {
            let d0;
            let d1;
            let mut state: SparseStab = SparseStab::new(3);
            state.x(&q(0)); // input state |-Z>
            (state, d0, d1) = teleport(state);
            let r2 = state.mz(&q(2)).into_iter().next().unwrap();
            let m2_int = u8::from(r2.outcome);

            assert_eq!(m2_int, 1);
            assert!(!d0);
            assert!(!d1);
            assert!(r2.is_deterministic);
        }
    }

    // TODO: Consider "forcing" the random number for cleaner testing.
    // TODO: Consider a seed to still have random numbers but make them predictable

    // ========================================================================
    // Generic Test Suite (using stabilizer_test_utils)
    // ========================================================================

    use crate::stabilizer_test_utils;

    // ========================================================================
    // SparseStab (BitSet) Tests
    // ========================================================================

    #[test]
    fn test_bitset_basic_stabilizer_suite() {
        let mut sim = SparseStab::new(3);
        stabilizer_test_utils::run_basic_stabilizer_test_suite(&mut sim, 3);
    }

    #[test]
    fn test_bitset_full_stabilizer_suite() {
        let mut sim = SparseStab::new(3);
        stabilizer_test_utils::run_full_stabilizer_test_suite(&mut sim, 3);
    }

    // ========================================================================
    // SparseStabVecSet Tests
    // ========================================================================

    #[test]
    fn test_vecset_basic_stabilizer_suite() {
        let mut sim = SparseStabVecSet::new(3);
        stabilizer_test_utils::run_basic_stabilizer_test_suite(&mut sim, 3);
    }

    #[test]
    fn test_vecset_full_stabilizer_suite() {
        let mut sim = SparseStabVecSet::new(3);
        stabilizer_test_utils::run_full_stabilizer_test_suite(&mut sim, 3);
    }

    // ========================================================================
    // SparseStabHybrid Tests
    // ========================================================================

    #[test]
    fn test_hybrid_basic_stabilizer_suite() {
        let mut sim = SparseStabHybrid::new(3);
        stabilizer_test_utils::run_basic_stabilizer_test_suite(&mut sim, 3);
    }

    #[test]
    fn test_hybrid_full_stabilizer_suite() {
        let mut sim = SparseStabHybrid::new(3);
        stabilizer_test_utils::run_full_stabilizer_test_suite(&mut sim, 3);
    }
}
