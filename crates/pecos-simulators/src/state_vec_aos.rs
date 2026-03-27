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

use super::arbitrary_rotation_gateable::ArbitraryRotationGateable;
use super::clifford_gateable::{CliffordGateable, MeasurementResult};
use super::quantum_simulator::QuantumSimulator;
use pecos_core::{Angle64, QubitId, RngManageable};
use pecos_random::{PecosRng, Rng, RngProbabilityExt, SeedableRng};

use core::fmt::Debug;
use num_complex::Complex64;

/// A quantum state simulator using the state vector representation
///
/// `StateVecAoS` maintains the full quantum state as a complex vector with 2ⁿ amplitudes
/// for n qubits. This allows exact simulation of quantum operations but requires
/// memory that scales exponentially with the number of qubits.
///
/// # Type Parameters
/// * `R` - Random number generator type implementing `Rng + SeedableRng` traits
///
/// # Examples
/// ```rust
/// use pecos_simulators::StateVecAoS;
///
/// // Create a new 2-qubit system
/// let mut state = StateVecAoS::new(2);
///
/// // Prepare a superposition state
/// state.prepare_plus_state();
/// ```
#[derive(Clone, Debug)]
pub struct StateVecAoS<R = PecosRng>
where
    R: Rng + SeedableRng + Debug,
{
    num_qubits: usize,
    state: Vec<Complex64>,
    rng: R,
}

impl StateVecAoS {
    /// Create a new state initialized to |0...0⟩
    ///
    /// # Examples
    /// ```rust
    /// use pecos_simulators::StateVecAoS;
    ///
    /// // Initialize a 3-qubit state vector in the |000⟩ state
    /// let state_vec = StateVecAoS::new(3);
    ///
    /// // Confirm the state is |000⟩
    /// let prob = state_vec.probability(0);
    /// assert!((prob - 1.0).abs() < 1e-10);
    /// ```
    #[inline]
    #[must_use]
    pub fn new(num_qubits: usize) -> StateVecAoS<PecosRng> {
        let rng = rand::make_rng();
        StateVecAoS::with_rng(num_qubits, rng)
    }

    /// Create a new state vector simulator with a specific seed for the random number generator
    ///
    /// This method allows for deterministic behavior by setting a specific seed for the
    /// random number generator, while still using the default RNG type (`PecosRng`).
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits in the system
    /// * `seed` - Seed value for the random number generator
    ///
    /// # Examples
    /// ```rust
    /// use pecos_simulators::StateVecAoS;
    ///
    /// // Create a simulator with a specific seed
    /// let state = StateVecAoS::with_seed(2, 42);
    /// ```
    #[inline]
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> StateVecAoS<PecosRng> {
        let rng = PecosRng::seed_from_u64(seed);
        StateVecAoS::with_rng(num_qubits, rng)
    }
}

impl<R> StateVecAoS<R>
where
    R: Rng + SeedableRng + Debug,
{
    /// Returns the number of qubits in the system
    ///
    /// # Returns
    /// * `usize` - The total number of qubits this simulator is configured to handle
    ///
    /// # Examples
    /// ```rust
    /// use pecos_simulators::{QuantumSimulator, StateVecAoS};
    /// let state = StateVecAoS::new(2);
    /// let num = state.num_qubits();
    /// assert_eq!(num, 2);
    /// ```
    #[inline]
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Create a new state vector with a custom random number generator. By doing so, one may set a
    /// seed or utilize a different base random number generator.
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits in the system
    /// * `rng` - Random number generator implementing `Rng + SeedableRng` traits
    ///
    /// # Examples
    /// ```rust
    /// use pecos_simulators::StateVecAoS;
    /// use pecos_random::{PecosRng, SeedableRng};
    ///
    /// let rng = PecosRng::seed_from_u64(42);
    /// let state = StateVecAoS::with_rng(2, rng);
    /// ```
    #[inline]
    #[must_use]
    pub fn with_rng(num_qubits: usize, rng: R) -> Self {
        let size = 1 << num_qubits; // 2^n
        let mut state = vec![Complex64::new(0.0, 0.0); size];
        state[0] = Complex64::new(1.0, 0.0); // Prep |0...0>
        StateVecAoS {
            num_qubits,
            state,
            rng,
        }
    }

    /// Initialize from a custom state vector
    ///
    /// # Examples
    /// ```rust
    /// use num_complex::Complex64;
    /// use pecos_simulators::StateVecAoS;
    /// use pecos_random::{PecosRng, SeedableRng};
    ///
    /// let custom_state = vec![
    ///     Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
    ///     Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
    ///     Complex64::new(0.0, 0.0),
    ///     Complex64::new(0.0, 0.0),
    /// ];
    ///
    /// let rng: PecosRng = rand::make_rng();
    /// let state_vec = StateVecAoS::from_state(custom_state, rng);
    /// ```
    ///
    /// # Panics
    /// Code will panic if the input state requires more qubits then `StateVecAoS` has.
    #[inline]
    #[must_use]
    pub fn from_state(state: Vec<Complex64>, rng: R) -> Self {
        let num_qubits = state.len().trailing_zeros() as usize;
        assert_eq!(1 << num_qubits, state.len(), "Invalid state vector size");
        StateVecAoS {
            num_qubits,
            state,
            rng,
        }
    }

    /// Prepare a specific computational basis state
    ///
    /// # Convention
    /// Note: The binary representation of the basis state uses a different ordering than
    /// standard quantum notation. For example:
    /// - |01⟩ corresponds to binary `0b10` (decimal 2)
    /// - |10⟩ corresponds to binary `0b01` (decimal 1)
    ///
    /// This is because in quantum notation the leftmost qubit is the most significant,
    /// while in binary representation the rightmost bit is the most significant.
    ///
    /// # Examples
    /// ```rust
    /// use pecos_simulators::StateVecAoS;
    ///
    /// let mut state_vec = StateVecAoS::new(3);
    /// state_vec.prepare_computational_basis(0b110);  // Prepares |011⟩
    /// ```
    ///
    /// # Panics
    /// Code will panic if `basis_state` >= `2^num_qubits` (i.e., if the basis state index is too large for the number of qubits)
    #[inline]
    pub fn prepare_computational_basis(&mut self, basis_state: usize) -> &mut Self {
        assert!(basis_state < 1 << self.num_qubits);
        self.state.fill(Complex64::new(0.0, 0.0));
        self.state[basis_state] = Complex64::new(1.0, 0.0);
        self
    }

    /// Prepare all qubits in the |+⟩ state, creating an equal superposition of all basis states
    ///
    /// This operation prepares the state (1/√2ⁿ)(|0...0⟩ + |0...1⟩ + ... + |1...1⟩)
    /// where n is the number of qubits.
    ///
    /// # Examples
    /// ```rust
    /// use pecos_simulators::StateVecAoS;
    /// let mut state = StateVecAoS::new(2);
    /// state.prepare_plus_state();
    /// ```
    #[inline]
    pub fn prepare_plus_state(&mut self) -> &mut Self {
        let factor = Complex64::new(1.0 / f64::from(1 << self.num_qubits).sqrt(), 0.0);
        self.state.fill(factor);
        self
    }

    /// Returns reference to the state vector
    ///
    /// The state vector is guaranteed to be normalized such that the sum of
    /// probability amplitudes squared equals 1.
    ///
    /// # Examples
    /// ```rust
    /// use pecos_simulators::StateVecAoS;
    ///
    /// // Initialize a 2-qubit state vector
    /// let state_vec = StateVecAoS::new(2);
    ///
    /// // Access the state vector
    /// let state = state_vec.state();
    ///
    /// // Verify the state is initialized to |00⟩
    /// assert_eq!(state[0].re, 1.0);
    /// assert_eq!(state[0].im, 0.0);
    /// for amp in &state[1..] {
    ///     assert_eq!(amp.re, 0.0);
    ///     assert_eq!(amp.im, 0.0);
    /// }
    /// ```
    ///
    /// # Returns
    /// A slice containing the complex amplitudes of the quantum state
    #[inline]
    #[must_use]
    pub fn state(&self) -> &[Complex64] {
        &self.state
    }

    /// Returns the probability of measuring a specific basis state
    ///
    /// # Convention
    /// Note: The binary representation of the basis state uses a different ordering than
    /// standard quantum notation. For example:
    /// - |01⟩ corresponds to binary `0b10` (decimal 2)
    /// - |10⟩ corresponds to binary `0b01` (decimal 1)
    ///
    /// This is because in quantum notation the leftmost qubit is the most significant,
    /// while in binary representation the rightmost bit is the most significant.
    ///
    /// # Examples
    /// ```rust
    /// use pecos_simulators::StateVecAoS;
    ///
    /// // Initialize a 2-qubit state vector
    /// let mut state_vec = StateVecAoS::new(2);
    ///
    /// // Prepare the |01⟩ state (corresponds to binary 10)
    /// state_vec.prepare_computational_basis(0b10);
    ///
    /// // Get the probability of measuring |01⟩
    /// let prob = state_vec.probability(0b10);  // Use binary 10 for |01⟩
    /// assert!((prob - 1.0).abs() < 1e-10);
    ///
    /// // Probability of measuring |00⟩ should be 0
    /// let prob_zero = state_vec.probability(0);
    /// assert!(prob_zero.abs() < 1e-10);
    /// ```
    ///
    /// # Panics
    /// Code will panic if `basis_state` >= `2^num_qubits` (i.e., if the basis state index is too large for the number of qubits)
    #[inline]
    #[must_use]
    pub fn probability(&self, basis_state: usize) -> f64 {
        assert!(basis_state < 1 << self.num_qubits);
        self.state[basis_state].norm_sqr()
    }

    /// Apply a general single-qubit unitary gate
    ///
    /// # Examples
    /// ```
    /// use pecos_simulators::StateVecAoS;
    /// use std::f64::consts::FRAC_1_SQRT_2;
    /// use num_complex::Complex64;
    /// let mut q = StateVecAoS::new(1);
    /// // Apply Hadamard gate
    /// q.single_qubit_rotation(0,
    ///     Complex64::new(FRAC_1_SQRT_2, 0.0),  // u00
    ///     Complex64::new(FRAC_1_SQRT_2, 0.0),  // u01
    ///     Complex64::new(FRAC_1_SQRT_2, 0.0),  // u10
    ///     Complex64::new(-FRAC_1_SQRT_2, 0.0), // u11
    /// );
    /// ```
    ///
    /// # Safety
    /// This function assumes that:
    /// - `qubit` is a valid qubit index (i.e., `< number of qubits`).
    /// - These conditions must be ensured by the caller or a higher-level component.
    #[inline]
    pub fn single_qubit_rotation(
        &mut self,
        qubit: usize,
        u00: Complex64,
        u01: Complex64,
        u10: Complex64,
        u11: Complex64,
    ) -> &mut Self {
        let step = 1 << qubit;
        for i in (0..self.state.len()).step_by(2 * step) {
            for offset in 0..step {
                let j = i + offset;
                let k = j ^ step;

                let a = self.state[j];
                let b = self.state[k];

                self.state[j] = u00 * a + u01 * b;
                self.state[k] = u10 * a + u11 * b;
            }
        }
        self
    }

    /// Apply a general two-qubit unitary given by a 4x4 complex matrix
    /// U = [[u00, u01, u02, u03],
    ///      [u10, u11, u12, u13],
    ///      [u20, u21, u22, u23],
    ///      [u30, u31, u32, u33]]
    ///
    /// # Examples
    /// ```rust
    /// use num_complex::Complex64;
    /// use pecos_simulators::StateVecAoS;
    ///
    /// let mut state_vec = StateVecAoS::new(2);
    ///
    /// let cnot_gate = [
    ///     [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
    ///     [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
    ///     [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
    ///     [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    /// ];
    ///
    /// state_vec.prepare_computational_basis(2);  // |01⟩
    /// println!("|01⟩: {:?}", state_vec.state());
    /// state_vec.two_qubit_unitary(1, 0, cnot_gate);  // Control: qubit 1, Target: qubit 0
    ///
    /// println!("|11⟩: {:?}", state_vec.state());
    ///
    /// let prob = state_vec.probability(3);  // Expect |11⟩
    /// println!("prob: {:?}", prob);
    /// assert!((prob - 1.0).abs() < 1e-10);
    /// ```
    ///
    /// ```rust
    /// use num_complex::Complex64;
    /// use pecos_simulators::StateVecAoS;
    ///
    /// // Initialize a 2-qubit state vector
    /// let mut state_vec = StateVecAoS::new(2);
    ///
    /// // Define a SWAP gate as a 4x4 unitary matrix
    /// let swap_gate = [
    ///     [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
    ///     [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
    ///     [Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
    ///     [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
    /// ];
    ///
    /// // Prepare the |01⟩ state
    /// state_vec.prepare_computational_basis(2);
    ///
    /// // Apply the SWAP gate to qubits 0 and 1
    /// state_vec.two_qubit_unitary(0, 1, swap_gate);
    /// println!("{:?}", state_vec.state());
    ///
    /// // Verify the state is now |10⟩
    /// let prob = state_vec.probability(1);
    /// assert!((prob - 1.0).abs() < 1e-10);
    /// ```
    ///
    /// # Safety
    /// This function assumes that:
    /// - `qubit1` and `qubit2` are valid qubit indices (i.e., `< number of qubits`).
    /// - `qubit1 != qubit2`.
    /// - These conditions must be ensured by the caller or a higher-level component.
    #[inline]
    pub fn two_qubit_unitary(
        &mut self,
        qubit1: usize,
        qubit2: usize,
        matrix: [[Complex64; 4]; 4],
    ) -> &mut Self {
        let n = self.num_qubits;
        let size = 1 << n;

        // Use a temporary buffer to avoid overwriting data during updates
        let mut new_state = vec![Complex64::new(0.0, 0.0); size];

        for i in 0..size {
            // Extract qubit1 and qubit2 bits
            let control_bit = (i >> qubit1) & 1;
            let target_bit = (i >> qubit2) & 1;

            // Map (control_bit, target_bit) to basis index (00, 01, 10, 11)
            let basis_idx = (control_bit << 1) | target_bit;

            for (j, &row) in matrix.iter().enumerate() {
                // Calculate the index after flipping qubit1 and qubit2 qubits
                let flipped_i = (i & !(1 << qubit1) & !(1 << qubit2))
                    | (((j >> 1) & 1) << qubit1)
                    | ((j & 1) << qubit2);

                // Apply the matrix to the relevant amplitudes
                new_state[flipped_i] += row[basis_idx] * self.state[i];
            }
        }

        self.state = new_state;
        self
    }
}

impl<R> QuantumSimulator for StateVecAoS<R>
where
    R: Rng + SeedableRng + Debug,
{
    /// # Examples
    /// ```rust
    /// use pecos_simulators::{QuantumSimulator, StateVecAoS};
    ///
    /// // Initialize a 2-qubit state vector
    /// let mut state_vec = StateVecAoS::new(2);
    ///
    /// // Prepare a different state
    /// state_vec.prepare_computational_basis(3); // |11⟩
    ///
    /// // Reset the state back to |00⟩
    /// state_vec.reset();
    ///
    /// // Verify the state is |00⟩
    /// let prob_zero = state_vec.probability(0);
    /// assert!((prob_zero - 1.0).abs() < 1e-10);
    /// ```
    #[inline]
    fn reset(&mut self) -> &mut Self {
        self.prepare_computational_basis(0)
    }
}

impl<R> CliffordGateable for StateVecAoS<R>
where
    R: Rng + SeedableRng + Debug,
{
    /// Implementation of Pauli-X gate for state vectors.
    ///
    /// See [`CliffordGateable::x`] for mathematical details and gate properties.
    /// This implementation uses direct state vector manipulation for performance.
    #[inline]
    fn x(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qubit = q.index();
            let step = 1 << qubit;

            for i in (0..self.state.len()).step_by(2 * step) {
                for offset in 0..step {
                    self.state.swap(i + offset, i + offset + step);
                }
            }
        }
        self
    }

    /// Implementation of Pauli-Y gate for state vectors.
    ///
    /// See [`CliffordGateable::y`] for mathematical details and gate properties.
    /// This implementation directly updates complex amplitudes including phases.
    #[inline]
    fn y(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qubit = q.index();
            for i in 0..self.state.len() {
                if (i >> qubit) & 1 == 0 {
                    let flipped_i = i ^ (1 << qubit);
                    let temp = self.state[i];
                    self.state[i] = -Complex64::i() * self.state[flipped_i];
                    self.state[flipped_i] = Complex64::i() * temp;
                }
            }
        }
        self
    }

    /// Implementation of Pauli-Z gate for state vectors.
    ///
    /// See [`CliffordGateable::z`] for mathematical details and gate properties.
    /// Takes advantage of diagonal structure for optimal performance.
    #[inline]
    fn z(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qubit = q.index();
            for i in 0..self.state.len() {
                if (i >> qubit) & 1 == 1 {
                    self.state[i] = -self.state[i];
                }
            }
        }
        self
    }

    /// Implementation of square root of Z (S) gate for state vectors.
    ///
    /// See [`CliffordGateable::sz`] for mathematical details and gate properties.
    /// Implemented using optimized single-qubit rotation.
    #[inline]
    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.single_qubit_rotation(
                q.index(),
                Complex64::new(1.0, 0.0), // u00
                Complex64::new(0.0, 0.0), // u01
                Complex64::new(0.0, 0.0), // u10
                Complex64::new(0.0, 1.0), // u11
            );
        }
        self
    }

    /// Implementation of Hadamard gate for state vectors.
    ///
    /// See [`CliffordGateable::h`] for mathematical details and gate properties.
    /// This implementation directly computes the superposition amplitudes.
    #[inline]
    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        let factor = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
        for &q in qubits {
            let qubit = q.index();
            let step = 1 << qubit;

            for i in (0..self.state.len()).step_by(2 * step) {
                for offset in 0..step {
                    let j = i + offset;
                    let paired_j = j ^ step;

                    let a = self.state[j];
                    let b = self.state[paired_j];

                    self.state[j] = factor * (a + b);
                    self.state[paired_j] = factor * (a - b);
                }
            }
        }
        self
    }

    /// Implementation of controlled-X (CNOT) gate for state vectors.
    ///
    /// See [`CliffordGateable::cx`] for mathematical details and gate properties.
    /// Uses bit manipulation for fast controlled operations.
    #[inline]
    fn cx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(qa, qb) in pairs {
            let q1 = qa.index();
            let q2 = qb.index();
            for i in 0..self.state.len() {
                let control_val = (i >> q1) & 1;
                let target_val = (i >> q2) & 1;
                if control_val == 1 && target_val == 0 {
                    let flipped_i = i ^ (1 << q2);
                    self.state.swap(i, flipped_i);
                }
            }
        }
        self
    }

    /// Implementation of controlled-Y gate for state vectors.
    ///
    /// See [`CliffordGateable::cy`] for mathematical details and gate properties.
    /// Combines bit manipulation with phase updates for controlled-Y operation.
    #[inline]
    fn cy(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(qa, qb) in pairs {
            let q1 = qa.index();
            let q2 = qb.index();
            // Only process when control bit is 1 and target bit is 0
            for i in 0..self.state.len() {
                let control_val = (i >> q1) & 1;
                let target_val = (i >> q2) & 1;

                if control_val == 1 && target_val == 0 {
                    let flipped_i = i ^ (1 << q2);

                    // Y gate has different phases than X
                    // Y = [[0, -i], [i, 0]]
                    let temp = self.state[i];
                    self.state[i] = -Complex64::i() * self.state[flipped_i];
                    self.state[flipped_i] = Complex64::i() * temp;
                }
            }
        }
        self
    }

    /// Implementation of controlled-Z gate for state vectors.
    ///
    /// See [`CliffordGateable::cz`] for mathematical details and gate properties.
    /// Takes advantage of diagonal structure for optimal performance.
    #[inline]
    fn cz(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(qa, qb) in pairs {
            let q1 = qa.index();
            let q2 = qb.index();
            // CZ is simpler - just add phase when both control and target are 1
            for i in 0..self.state.len() {
                let control_val = (i >> q1) & 1;
                let target_val = (i >> q2) & 1;

                if control_val == 1 && target_val == 1 {
                    self.state[i] = -self.state[i];
                }
            }
        }
        self
    }

    /// Implementation of SWAP gate for state vectors.
    ///
    /// See [`CliffordGateable::swap`] for mathematical details and gate properties.
    /// Uses bit manipulation for efficient state vector updates.
    #[inline]
    fn swap(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(qa, qb) in pairs {
            let q1 = qa.index();
            let q2 = qb.index();
            let step1 = 1 << q1;
            let step2 = 1 << q2;

            for i in 0..self.state.len() {
                let bit1 = (i >> q1) & 1;
                let bit2 = (i >> q2) & 1;

                if bit1 != bit2 {
                    let swapped_index = i ^ step1 ^ step2;
                    if i < swapped_index {
                        self.state.swap(i, swapped_index);
                    }
                }
            }
        }
        self
    }

    /// Implementation of Z-basis measurement for state vectors.
    ///
    /// See [`CliffordGateable::mz`] for mathematical details and measurement properties.
    /// Computes measurement probabilities and performs state collapse.
    ///
    /// # Examples
    /// ```rust
    /// use pecos_simulators::{QuantumSimulator, StateVecAoS, CliffordGateable};
    /// use pecos_core::QubitId;
    ///
    /// let mut state = StateVecAoS::new(2);
    ///
    /// // Create Bell state and measure first qubit
    /// state.h(&[QubitId(0)]).cx(&[(QubitId(0), QubitId(1))]);
    /// let results = state.mz(&[QubitId(0), QubitId(1)]);
    /// // Both qubits will have the same outcome due to entanglement
    /// assert_eq!(results[0].outcome, results[1].outcome);
    /// ```
    ///
    /// # Safety
    /// This function assumes that:
    /// - All qubit indices are valid (i.e., `< number of qubits`).
    /// - These conditions must be ensured by the caller or a higher-level component.
    #[inline]
    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        let mut results = Vec::with_capacity(qubits.len());

        for &q in qubits {
            let qubit = q.index();
            let step = 1 << qubit;
            let mut prob_one = 0.0;

            // Calculate probability of measuring |1>
            for i in (0..self.state.len()).step_by(2 * step) {
                for offset in 0..step {
                    let idx = i + offset + step; // Target bit = 1 positions
                    prob_one += self.state[idx].norm_sqr();
                }
            }

            // Decide measurement outcome
            let result = usize::from(self.rng.bernoulli(prob_one));

            // Collapse and normalize state
            let mut norm = 0.0;
            for i in 0..self.state.len() {
                let bit = (i >> qubit) & 1;
                if bit == result {
                    norm += self.state[i].norm_sqr();
                } else {
                    self.state[i] = Complex64::new(0.0, 0.0);
                }
            }

            let norm_inv = 1.0 / norm.sqrt();
            for amp in &mut self.state {
                *amp *= norm_inv;
            }

            let is_deterministic = !(1e-10..=1.0 - 1e-10).contains(&prob_one);
            results.push(MeasurementResult {
                outcome: result != 0,
                is_deterministic,
            });
        }

        results
    }
}

impl<R> ArbitraryRotationGateable for StateVecAoS<R>
where
    R: Rng + SeedableRng + Debug,
{
    /// Implementation of rotation around the X-axis.
    ///
    /// See [`ArbitraryRotationGateable::rx`] for mathematical details and gate properties.
    /// This implementation directly updates amplitudes in the state vector for optimal performance.
    ///
    /// # Examples
    /// ```rust
    /// use pecos_simulators::{QuantumSimulator, StateVecAoS, ArbitraryRotationGateable};
    /// use pecos_core::{QubitId, Angle64};
    /// use std::f64::consts::FRAC_PI_2;
    ///
    /// let mut state = StateVecAoS::new(1);
    ///
    /// // Create superposition with phase
    /// state.rx(Angle64::from_radians(FRAC_PI_2), &[QubitId(0)]);  // Creates (|0> - i|1>)/sqrt(2)
    /// ```
    ///
    /// # Safety
    /// This function assumes that:
    /// - All qubit indices are valid (i.e., `< number of qubits`).
    /// - These conditions must be ensured by the caller or a higher-level component.
    #[inline]
    fn rx(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let theta = theta.to_radians_signed();
        let cos = (theta / 2.0).cos();
        let sin = (theta / 2.0).sin();
        for &q in qubits {
            self.single_qubit_rotation(
                q.index(),
                Complex64::new(cos, 0.0),
                Complex64::new(0.0, -sin),
                Complex64::new(0.0, -sin),
                Complex64::new(cos, 0.0),
            );
        }
        self
    }

    /// Implementation of rotation around the Y-axis.
    ///
    /// See [`ArbitraryRotationGateable::ry`] for mathematical details and gate properties.
    /// This implementation directly updates amplitudes in the state vector for optimal performance.
    ///
    /// # Examples
    /// ```rust
    /// use pecos_simulators::{QuantumSimulator, StateVecAoS, ArbitraryRotationGateable};
    /// use pecos_core::{QubitId, Angle64};
    /// use std::f64::consts::PI;
    ///
    /// let mut state = StateVecAoS::new(1);
    ///
    /// // Create equal superposition
    /// state.ry(Angle64::from_radians(PI/2.0), &[QubitId(0)]);  // Creates (|0> + |1>)/sqrt(2)
    /// ```
    ///
    /// # Safety
    /// This function assumes that:
    /// - All qubit indices are valid (i.e., `< number of qubits`).
    /// - These conditions must be ensured by the caller or a higher-level component.
    #[inline]
    fn ry(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let theta = theta.to_radians_signed();
        let cos = (theta / 2.0).cos();
        let sin = (theta / 2.0).sin();
        for &q in qubits {
            self.single_qubit_rotation(
                q.index(),
                Complex64::new(cos, 0.0),
                Complex64::new(-sin, 0.0),
                Complex64::new(sin, 0.0),
                Complex64::new(cos, 0.0),
            );
        }
        self
    }

    /// Implementation of rotation around the Z-axis.
    ///
    /// See [`ArbitraryRotationGateable::rz`] for mathematical details and gate properties.
    /// Takes advantage of the diagonal structure in computational basis for optimal performance.
    ///
    /// # Examples
    /// ```rust
    /// use pecos_simulators::{QuantumSimulator, StateVecAoS, CliffordGateable, ArbitraryRotationGateable};
    /// use pecos_core::{QubitId, Angle64};
    /// use std::f64::consts::FRAC_PI_4;
    ///
    /// let mut state = StateVecAoS::new(1);
    ///
    /// // Create superposition and add phase
    /// state.h(&[QubitId(0)]).rz(Angle64::from_radians(FRAC_PI_4), &[QubitId(0)]);  // T gate equivalent
    /// ```
    ///
    /// # Safety
    /// This function assumes that:
    /// - All qubit indices are valid (i.e., `< number of qubits`).
    /// - These conditions must be ensured by the caller or a higher-level component.
    fn rz(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let theta = theta.to_radians_signed();
        let e_pos = Complex64::from_polar(1.0, -theta / 2.0);
        let e_neg = Complex64::from_polar(1.0, theta / 2.0);

        for &q in qubits {
            self.single_qubit_rotation(
                q.index(),
                e_pos,
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                e_neg,
            );
        }
        self
    }

    /// Implementation of general single-qubit unitary rotation.
    ///
    /// See [`ArbitraryRotationGateable::u`] for mathematical details and gate properties.
    /// This implementation directly updates amplitudes in the state vector.
    ///
    /// # Examples
    /// ```rust
    /// use pecos_simulators::{QuantumSimulator, StateVecAoS, ArbitraryRotationGateable};
    /// use pecos_core::{QubitId, Angle64};
    /// use std::f64::consts::{PI, FRAC_PI_2};
    ///
    /// let mut state = StateVecAoS::new(1);
    ///
    /// // Create arbitrary rotation (equivalent to TH up to global phase)
    /// state.u(Angle64::from_radians(FRAC_PI_2), Angle64::from_radians(0.0), Angle64::from_radians(FRAC_PI_2), &[QubitId(0)]);
    /// ```
    ///
    /// # Safety
    /// This function assumes that:
    /// - All qubit indices are valid (i.e., `< number of qubits`).
    /// - These conditions must be ensured by the caller or a higher-level component.
    #[inline]
    fn u(
        &mut self,
        theta: Angle64,
        phi: Angle64,
        lambda: Angle64,
        qubits: &[QubitId],
    ) -> &mut Self {
        let theta = theta.to_radians_signed();
        let phi = phi.to_radians_signed();
        let lambda = lambda.to_radians_signed();
        let cos = (theta / 2.0).cos();
        let sin = (theta / 2.0).sin();

        // Calculate matrix elements
        let u00 = Complex64::new(cos, 0.0);
        let u01 = -Complex64::from_polar(sin, lambda);
        let u10 = Complex64::from_polar(sin, phi);
        let u11 = Complex64::from_polar(cos, phi + lambda);

        for &q in qubits {
            self.single_qubit_rotation(q.index(), u00, u01, u10, u11);
        }
        self
    }

    /// Implementation of single-qubit rotation in XY plane.
    ///
    /// See [`ArbitraryRotationGateable::r1xy`] for mathematical details and gate properties.
    /// Optimized for rotations in the XY plane of the Bloch sphere.
    ///
    /// # Examples
    /// ```rust
    /// use pecos_simulators::{QuantumSimulator, StateVecAoS, ArbitraryRotationGateable};
    /// use pecos_core::{QubitId, Angle64};
    /// use std::f64::consts::FRAC_PI_2;
    ///
    /// let mut state = StateVecAoS::new(1);
    ///
    /// // 90-degree rotation around X+Y axis
    /// state.r1xy(Angle64::from_radians(FRAC_PI_2), Angle64::from_radians(FRAC_PI_2), &[QubitId(0)]);
    /// ```
    ///
    /// # Safety
    /// This function assumes that:
    /// - All qubit indices are valid (i.e., `< number of qubits`).
    /// - These conditions must be ensured by the caller or a higher-level component.
    #[inline]
    fn r1xy(&mut self, theta: Angle64, phi: Angle64, qubits: &[QubitId]) -> &mut Self {
        let theta = theta.to_radians_signed();
        let phi = phi.to_radians_signed();
        let cos = (theta / 2.0).cos();
        let sin = (theta / 2.0).sin();

        // Calculate the matrix elements
        let r00 = Complex64::new(cos, 0.0); // cos(theta/2)
        let r01 = -Complex64::new(0.0, sin) * Complex64::from_polar(1.0, -phi); // -i sin(theta/2) e^(-i*phi)
        let r10 = -Complex64::new(0.0, sin) * Complex64::from_polar(1.0, phi); // -i sin(theta/2) e^(i*phi)
        let r11 = Complex64::new(cos, 0.0); // cos(theta/2)

        for &q in qubits {
            // Apply the single-qubit rotation using the matrix elements
            self.single_qubit_rotation(q.index(), r00, r01, r10, r11);
        }
        self
    }

    /// Implementation of two-qubit XX rotation.
    ///
    /// See [`ArbitraryRotationGateable::rxx`] for mathematical details and gate properties.
    /// This implementation directly updates amplitudes in the state vector for optimal performance.
    ///
    /// # Examples
    /// ```rust
    /// use pecos_simulators::{QuantumSimulator, StateVecAoS, ArbitraryRotationGateable};
    /// use pecos_core::{QubitId, Angle64};
    /// use std::f64::consts::FRAC_PI_2;
    ///
    /// let mut state = StateVecAoS::new(2);
    ///
    /// // Create maximally entangled state
    /// state.rxx(Angle64::from_radians(FRAC_PI_2), &[(QubitId(0), QubitId(1))]);  // Creates (|00> - i|11>)/sqrt(2)
    /// ```
    ///
    /// # Safety
    /// This function assumes that:
    /// - `qubits.len() % 2 == 0` (pairs of qubits).
    /// - All qubit indices are valid and distinct within each pair.
    /// - These conditions must be ensured by the caller or a higher-level component.
    #[inline]
    fn rxx(&mut self, theta: Angle64, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let theta = theta.to_radians_signed();
        let cos = (theta / 2.0).cos();
        let sin = (theta / 2.0).sin();
        let neg_i_sin = Complex64::new(0.0, -sin); // -i*sin

        for &(q0, q1) in pairs {
            let first = q0.index();
            let second = q1.index();

            // Make sure q1 < q2 for consistent ordering
            let (q1, q2) = if first < second {
                (first, second)
            } else {
                (second, first)
            };

            for i in 0..self.state.len() {
                let bit1 = (i >> q1) & 1;
                let bit2 = (i >> q2) & 1;

                if bit1 == 0 && bit2 == 0 {
                    let i01 = i ^ (1 << q2);
                    let i10 = i ^ (1 << q1);
                    let i11 = i ^ (1 << q1) ^ (1 << q2);

                    let a00 = self.state[i];
                    let a01 = self.state[i01];
                    let a10 = self.state[i10];
                    let a11 = self.state[i11];

                    // Apply the correct RXX matrix
                    self.state[i] = cos * a00 + neg_i_sin * a11;
                    self.state[i01] = cos * a01 + neg_i_sin * a10;
                    self.state[i10] = cos * a10 + neg_i_sin * a01;
                    self.state[i11] = cos * a11 + neg_i_sin * a00;
                }
            }
        }
        self
    }

    /// Implementation of the RYY(theta) gate for state vectors.
    ///
    /// See [`ArbitraryRotationGateable::ryy`] for mathematical details and gate properties.
    /// This implementation directly updates amplitudes in the state vector for optimal performance.
    ///
    /// # Examples
    /// ```rust
    /// use pecos_simulators::{QuantumSimulator, StateVecAoS, CliffordGateable, ArbitraryRotationGateable};
    /// use pecos_core::{QubitId, Angle64};
    /// use std::f64::consts::FRAC_PI_2;
    ///
    /// let mut state = StateVecAoS::new(2);
    ///
    /// // Create entangled state
    /// state.h(&[QubitId(0)])
    ///      .cx(&[(QubitId(0), QubitId(1))]);
    ///
    /// // Apply RYY rotation
    /// state.ryy(Angle64::from_radians(FRAC_PI_2), &[(QubitId(0), QubitId(1))]);
    /// ```
    ///
    /// # Safety
    /// This function assumes that:
    /// - `qubits.len() % 2 == 0` (pairs of qubits).
    /// - All qubit indices are valid and distinct within each pair.
    /// - These conditions must be ensured by the caller or a higher-level component.
    #[inline]
    fn ryy(&mut self, theta: Angle64, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let theta = theta.to_radians_signed();
        let cos = (theta / 2.0).cos();
        let i_sin = Complex64::new(0.0, 1.0) * (theta / 2.0).sin();

        for &(qa, qb) in pairs {
            let q1 = qa.index();
            let q2 = qb.index();

            // No need to reorder q1 and q2 since we're using explicit masks
            let mask1 = 1 << q1;
            let mask2 = 1 << q2;

            for i in 0..self.state.len() {
                // Only process each set of 4 states once
                if (i & (mask1 | mask2)) == 0 {
                    let i00 = i;
                    let i01 = i | mask2;
                    let i10 = i | mask1;
                    let i11 = i | mask1 | mask2;

                    let a00 = self.state[i00];
                    let a01 = self.state[i01];
                    let a10 = self.state[i10];
                    let a11 = self.state[i11];

                    self.state[i00] = cos * a00 + i_sin * a11;
                    self.state[i01] = cos * a01 - i_sin * a10;
                    self.state[i10] = cos * a10 - i_sin * a01;
                    self.state[i11] = cos * a11 + i_sin * a00;
                }
            }
        }
        self
    }

    /// Implementation of the RZZ(theta) gate for state vectors.
    ///
    /// See [`ArbitraryRotationGateable::rzz`] for mathematical details and gate properties.
    /// Takes advantage of the diagonal structure in the computational basis for optimal performance.
    ///
    /// # Examples
    /// ```rust
    /// use pecos_simulators::{QuantumSimulator, StateVecAoS, CliffordGateable, ArbitraryRotationGateable};
    /// use pecos_core::{QubitId, Angle64};
    /// use std::f64::consts::PI;
    ///
    /// let mut state = StateVecAoS::new(3);
    ///
    /// // Create GHZ state
    /// state.h(&[QubitId(0)])
    ///      .cx(&[(QubitId(0), QubitId(1))])
    ///      .cx(&[(QubitId(1), QubitId(2))]);
    ///
    /// // Apply phase rotation between first and last qubit
    /// state.rzz(Angle64::from_radians(PI/4.0), &[(QubitId(0), QubitId(2))]);
    /// ```
    ///
    /// # Safety
    /// This function assumes that:
    /// - `qubits.len() % 2 == 0` (pairs of qubits).
    /// - All qubit indices are valid and distinct within each pair.
    /// - These conditions must be ensured by the caller or a higher-level component.
    fn rzz(&mut self, theta: Angle64, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let theta = theta.to_radians_signed();

        for &(qa, qb) in pairs {
            let q1 = qa.index();
            let q2 = qb.index();

            // RZZ is diagonal in computational basis - just add phases
            for i in 0..self.state.len() {
                let bit1 = (i >> q1) & 1;
                let bit2 = (i >> q2) & 1;

                // Phase depends on parity of bits
                let phase = if bit1 ^ bit2 == 0 {
                    // Same bits (00 or 11) -> e^(-i*theta/2)
                    Complex64::from_polar(1.0, -theta / 2.0)
                } else {
                    // Different bits (01 or 10) -> e^(i*theta/2)
                    Complex64::from_polar(1.0, theta / 2.0)
                };

                self.state[i] *= phase;
            }
        }
        self
    }
}

impl<R> RngManageable for StateVecAoS<R>
where
    R: Rng + SeedableRng + Debug,
{
    type Rng = R;

    /// Replace the random number generator with a new one
    ///
    /// This method allows replacing the RNG without recreating the entire simulator,
    /// which is useful for setting seeds after initialization.
    ///
    /// # Arguments
    /// * `rng` - A new random number generator implementing the `Rng + SeedableRng` traits
    #[inline]
    fn set_rng(&mut self, rng: R) {
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

/// Test suite for state vector quantum simulation.
///
/// # Organization
/// The tests are organized into several categories:
/// - Basic state manipulation and access (new state, preparation, etc.)
/// - Single-qubit gate operations (X, Y, Z, H, etc.)
/// - Two-qubit gate operations (CX, CY, CZ, SWAP)
/// - Rotation gates (RX, RY, RZ, RXX, RYY, RZZ)
/// - Measurement operations
/// - Gate relationships and decompositions
/// - Edge cases and numerical stability
/// - System scaling and locality
///
/// # Testing Strategy
/// Tests verify:
/// 1. Basic correctness: Each operation produces expected output states
/// 2. Mathematical properties: Unitarity, phase relationships, commutation
/// 3. Physical requirements: State normalization, measurement statistics
/// 4. Implementation properties: Numerical stability, locality of operations
/// 5. Gate relationships: Standard decompositions and equivalent implementations
///
/// # Helper Functions
/// - `assert_states_equal`: Compares quantum states up to global phase
/// - `assert_state_vectors_match`: Detailed comparison with tolerance checking
///
/// # Notes
/// - Tests use standard tolerances of 1e-10 for floating point comparisons
/// - Random tests use fixed seeds for reproducibility
/// - Large system tests verify scaling behavior up to 20 qubits
#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::QubitId;
    use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, FRAC_PI_3, FRAC_PI_4, FRAC_PI_6, PI, TAU};

    use num_complex::Complex64;

    // Helper to create qubit slice for single qubit
    fn qid(n: usize) -> [QubitId; 1] {
        [QubitId(n)]
    }

    /// Compare two quantum states up to global phase.
    ///
    /// This function ensures that two state vectors represent the same quantum state,
    /// accounting for potential differences in global phase.
    ///
    /// # Arguments
    /// - `state1`: A reference to the first state vector.
    /// - `state2`: A reference to the second state vector.
    ///
    /// # Panics
    /// The function will panic if the states differ in norm or relative phase beyond a small numerical tolerance.
    fn assert_states_equal(state1: &[Complex64], state2: &[Complex64]) {
        const TOLERANCE: f64 = 1e-10;

        if state1[0].norm() < TOLERANCE && state2[0].norm() < TOLERANCE {
            // Both first components near zero, compare other components directly
            for (index, (a, b)) in state1.iter().zip(state2.iter()).enumerate() {
                assert!(
                    (a.norm() - b.norm()).abs() < TOLERANCE,
                    "States differ in magnitude at index {index}: {a} vs {b}"
                );
            }
        } else {
            // Get phase from the first pair of non-zero components
            let ratio = match state1
                .iter()
                .zip(state2.iter())
                .find(|(a, b)| a.norm() > TOLERANCE && b.norm() > TOLERANCE)
            {
                Some((a, b)) => b / a,
                None => panic!("States have no corresponding non-zero components"),
            };
            println!("Phase ratio between states: {ratio:?}");

            for (index, (a, b)) in state1.iter().zip(state2.iter()).enumerate() {
                assert!(
                    (a * ratio - b).norm() < TOLERANCE,
                    "States differ at index {index}: {a} vs {b} (adjusted with ratio {ratio:?}), diff = {}",
                    (a * ratio - b).norm()
                );
            }
        }
    }

    // Core functionality tests
    // ========================
    #[test]
    fn test_new_state() {
        // Verify the initial state is correctly set to |0>
        let q = StateVecAoS::new(2);
        assert_eq!(q.state[0], Complex64::new(1.0, 0.0));
        for i in 1..4 {
            assert_eq!(q.state[i], Complex64::new(0.0, 0.0));
        }
    }

    #[test]
    fn test_reset() {
        let mut state_vec = StateVecAoS::new(2);

        state_vec.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]); // Create Bell state
        state_vec.reset(); // Reset to |00⟩

        assert!((state_vec.probability(0) - 1.0).abs() < 1e-10);
        for i in 1..state_vec.state.len() {
            assert!(state_vec.state[i].norm() < 1e-10);
        }
    }

    #[test]
    fn test_probability() {
        let mut state_vec = StateVecAoS::new(1);

        // Prepare |+⟩ state
        state_vec.h(&qid(0));

        let prob_zero = state_vec.probability(0);
        let prob_one = state_vec.probability(1);

        assert!((prob_zero - 0.5).abs() < 1e-10);
        assert!((prob_one - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_prepare_computational_basis_all_states() {
        let num_qubits = 3;
        let mut state_vec = StateVecAoS::new(num_qubits);

        for basis_state in 0..(1 << num_qubits) {
            state_vec.prepare_computational_basis(basis_state);
            for i in 0..state_vec.state.len() {
                if i == basis_state {
                    assert!((state_vec.state[i].norm() - 1.0).abs() < 1e-10);
                } else {
                    assert!(state_vec.state[i].norm() < 1e-10);
                }
            }
        }
    }

    // Single qubit gate fundamentals
    // ==============================
    #[test]
    fn test_x() {
        let mut q = StateVecAoS::new(1);

        // Check initial state is |0>
        assert!((q.state[0].re - 1.0).abs() < 1e-10);
        assert!(q.state[1].norm() < 1e-10);

        // Test X on |0> -> |1>
        q.x(&qid(0));
        assert!(q.state[0].norm() < 1e-10);
        assert!((q.state[1].re - 1.0).abs() < 1e-10);

        // Test X on |1> -> |0>
        q.x(&qid(0));
        assert!((q.state[0].re - 1.0).abs() < 1e-10);
        assert!(q.state[1].norm() < 1e-10);

        // Test X on superposition
        q.h(&qid(0));
        let initial_state = q.state.clone();
        q.x(&qid(0)); // X|+> = |+>
        for (state, initial) in q.state.iter().zip(initial_state.iter()) {
            assert!((state - initial).norm() < 1e-10);
        }

        // Test X on second qubit of two-qubit system
        let mut q = StateVecAoS::new(2);
        q.x(&qid(1));
        assert!(q.state[0].norm() < 1e-10);
        assert!((q.state[2].re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_y() {
        let mut q = StateVecAoS::new(1);

        // Test Y on |0⟩ -> i|1⟩
        q.y(&qid(0));
        assert!(q.state[0].norm() < 1e-10);
        assert!((q.state[1] - Complex64::i()).norm() < 1e-10);

        // Test Y on i|1⟩ -> |0⟩
        q.y(&qid(0));
        assert!((q.state[0].re - 1.0).abs() < 1e-10);
        assert!(q.state[1].norm() < 1e-10);

        // Test Y on |+⟩
        let mut q = StateVecAoS::new(1);
        q.h(&qid(0)); // Create |+⟩
        q.y(&qid(0)); // Should give i|-⟩
        let expected = FRAC_1_SQRT_2;
        assert!((q.state[0].im + expected).abs() < 1e-10);
        assert!((q.state[1].im - expected).abs() < 1e-10);
    }

    #[test]
    fn test_z() {
        let mut q = StateVecAoS::new(1);

        // Test Z on |0⟩ -> |0⟩
        q.z(&qid(0));
        assert!((q.state[0].re - 1.0).abs() < 1e-10);
        assert!(q.state[1].norm() < 1e-10);

        // Test Z on |1⟩ -> -|1⟩
        q.x(&qid(0)); // Prepare |1⟩
        q.z(&qid(0));
        assert!(q.state[0].norm() < 1e-10);
        assert!((q.state[1].re + 1.0).abs() < 1e-10);

        // Test Z on |+⟩ -> |-⟩
        let mut q = StateVecAoS::new(1);
        q.h(&qid(0)); // Create |+⟩
        q.z(&qid(0)); // Should give |-⟩
        let expected = FRAC_1_SQRT_2;
        assert!((q.state[0].re - expected).abs() < 1e-10);
        assert!((q.state[1].re + expected).abs() < 1e-10);
    }

    #[test]
    fn test_h() {
        let mut q = StateVecAoS::new(1);
        q.h(&qid(0));

        assert!((q.state[0].re - FRAC_1_SQRT_2).abs() < 1e-10);
        assert!((q.state[1].re - FRAC_1_SQRT_2).abs() < 1e-10);
    }

    // TODO: add...
    #[test]
    fn test_sz() {}

    // Two qubit gate fundamentals
    // ===========================
    #[test]
    fn test_cx() {
        let mut q = StateVecAoS::new(2);
        // Prep |+>
        q.h(&qid(0));
        q.cx(&[(QubitId(0), QubitId(1))]);

        // Should be in Bell state (|00> + |11>)/sqrt(2)
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!((q.state[0].re - expected).abs() < 1e-10);
        assert!((q.state[3].re - expected).abs() < 1e-10);
        assert!(q.state[1].norm() < 1e-10);
        assert!(q.state[2].norm() < 1e-10);
    }

    #[test]
    fn test_cy() {
        let mut q = StateVecAoS::new(2);

        // Create |+0⟩ state
        q.h(&qid(0));

        // Apply CY to get entangled state
        q.cy(&[(QubitId(0), QubitId(1))]);

        // Should be (|00⟩ + i|11⟩)/√2
        let expected = FRAC_1_SQRT_2;
        assert!((q.state[0].re - expected).abs() < 1e-10); // |00⟩ amplitude
        assert!(q.state[1].norm() < 1e-10); // |01⟩ amplitude
        assert!(q.state[2].norm() < 1e-10); // |10⟩ amplitude
        assert!((q.state[3].im - expected).abs() < 1e-10); // |11⟩ amplitude
    }

    #[test]
    fn test_cz() {
        let mut q = StateVecAoS::new(2);

        // Create |++⟩ state
        q.h(&qid(0));
        q.h(&qid(1));

        // Apply CZ
        q.cz(&[(QubitId(0), QubitId(1))]);

        // Should be (|00⟩ + |01⟩ + |10⟩ - |11⟩)/2
        let expected = 0.5;
        assert!((q.state[0].re - expected).abs() < 1e-10); // |00⟩ amplitude
        assert!((q.state[1].re - expected).abs() < 1e-10); // |01⟩ amplitude
        assert!((q.state[2].re - expected).abs() < 1e-10); // |10⟩ amplitude
        assert!((q.state[3].re + expected).abs() < 1e-10); // |11⟩ amplitude
    }

    #[test]
    fn test_swap() {
        let mut q = StateVecAoS::new(2);
        q.x(&qid(0));
        q.swap(&[(QubitId(0), QubitId(1))]);

        assert!(q.state[0].norm() < 1e-10);
        assert!((q.state[2].re - 1.0).abs() < 1e-10);
    }

    // Basic measurement tests
    // =======================

    #[test]
    fn test_mz() {
        // Test 1: Measuring |0> state
        let mut q = StateVecAoS::new(1);
        let result = q.mz(&qid(0)).into_iter().next().unwrap();
        assert!(!result.outcome);
        assert!((q.state[0].re - 1.0).abs() < 1e-10);
        assert!(q.state[1].norm() < 1e-10);

        // Test 2: Measuring |1> state
        let mut q = StateVecAoS::new(1);
        q.x(&qid(0));
        let result = q.mz(&qid(0)).into_iter().next().unwrap();
        assert!(result.outcome);
        assert!(q.state[0].norm() < 1e-10);
        assert!((q.state[1].re - 1.0).abs() < 1e-10);

        // Test 3: Measuring superposition state multiple times
        let mut zeros = 0;
        let trials = 1000;

        for _ in 0..trials {
            let mut q = StateVecAoS::new(1);
            q.h(&qid(0));
            let result = q.mz(&qid(0)).into_iter().next().unwrap();
            if !result.outcome {
                zeros += 1;
            }
        }

        // Check if measurements are roughly equally distributed
        let ratio = f64::from(zeros) / f64::from(trials);
        assert!((ratio - 0.5).abs() < 0.1); // Should be close to 0.5...

        // Test 4: Measuring one qubit of a Bell state
        let mut q = StateVecAoS::new(2);
        q.h(&qid(0));
        q.cx(&[(QubitId(0), QubitId(1))]);

        // Measure first qubit
        let result1 = q.mz(&qid(0)).into_iter().next().unwrap();
        // Measure second qubit - should match first
        let result2 = q.mz(&qid(1)).into_iter().next().unwrap();
        assert_eq!(result1.outcome, result2.outcome);
    }

    #[test]
    fn test_measurement_consistency() {
        let mut q = StateVecAoS::new(1);

        // Put qubit in |1⟩ state
        q.x(&qid(0));

        // Measure twice - result should be the same
        let result1 = q.mz(&qid(0)).into_iter().next().unwrap();
        let result2 = q.mz(&qid(0)).into_iter().next().unwrap();

        assert!(result1.outcome);
        assert!(result2.outcome);
    }

    #[test]
    fn test_measurement_collapse() {
        let mut state_vec = StateVecAoS::new(1);

        // Prepare |+⟩ = (|0⟩ + |1⟩) / √2
        state_vec.h(&qid(0));

        // Simulate a measurement
        let result = state_vec.mz(&qid(0)).into_iter().next().unwrap();

        // State should collapse to |0⟩ or |1⟩
        if result.outcome {
            assert!((state_vec.probability(1) - 1.0).abs() < 1e-10);
        } else {
            assert!((state_vec.probability(0) - 1.0).abs() < 1e-10);
        }
    }

    // test Pauli-basis prep
    // =====================
    #[test]
    fn test_pz() {
        let mut q = StateVecAoS::new(1);

        q.h(&qid(0));
        assert!((q.state[0].re - FRAC_1_SQRT_2).abs() < 1e-10);
        assert!((q.state[1].re - FRAC_1_SQRT_2).abs() < 1e-10);

        q.pz(&qid(0));

        assert!((q.state[0].re - 1.0).abs() < 1e-10);
        assert!(q.state[1].norm() < 1e-10);
    }

    #[test]
    fn test_pz_multiple_qubits() {
        let mut q = StateVecAoS::new(2);

        q.h(&qid(0));
        q.cx(&[(QubitId(0), QubitId(1))]);

        q.pz(&qid(0));

        let prob_0 = q.state[0].norm_sqr() + q.state[2].norm_sqr();
        let prob_1 = q.state[1].norm_sqr() + q.state[3].norm_sqr();

        assert!((prob_0 - 1.0).abs() < 1e-10);
        assert!(prob_1 < 1e-10);
    }

    // Basic single-qubit rotation gate tests
    // ======================================
    #[test]
    fn test_rx() {
        // Test RX gate functionality
        let mut q = StateVecAoS::new(1);

        // RX(π) should flip |0⟩ to -i|1⟩
        q.rx(Angle64::from_radians(PI), &qid(0));
        assert!(q.state[0].norm() < 1e-10);
        assert!((q.state[1].norm() - 1.0).abs() < 1e-10);

        // RX(2π) should return to the initial state up to global phase
        let mut q = StateVecAoS::new(1);
        q.rx(Angle64::from_radians(2.0 * PI), &qid(0));
        assert!((q.state[0].norm() - 1.0).abs() < 1e-10);
        assert!(q.state[1].norm() < 1e-10);
    }

    #[test]
    fn test_ry() {
        let mut q = StateVecAoS::new(1);

        // RY(π) should flip |0⟩ to |1⟩
        q.ry(Angle64::from_radians(PI), &qid(0));
        assert!(q.state[0].norm() < 1e-10); // Close to zero
        assert!((q.state[1].norm() - 1.0).abs() < 1e-10); // Magnitude 1 for |1⟩

        // Two RY(π) rotations should return to the initial state
        q.ry(Angle64::from_radians(PI), &qid(0));
        assert!((q.state[0].norm() - 1.0).abs() < 1e-10); // Magnitude 1 for |0⟩
        assert!(q.state[1].norm() < 1e-10); // Close to zero
    }

    #[test]
    fn test_rz() {
        let mut q = StateVecAoS::new(1);

        // RZ should only add phases, not change probabilities
        q.h(&qid(0)); // Put qubit in superposition
        let probs_before: Vec<f64> = q.state.iter().map(num_complex::Complex::norm_sqr).collect();

        q.rz(Angle64::from_radians(FRAC_PI_2), &qid(0)); // Rotate Z by π/2
        let probs_after: Vec<f64> = q.state.iter().map(num_complex::Complex::norm_sqr).collect();

        for (p1, p2) in probs_before.iter().zip(probs_after.iter()) {
            assert!((p1 - p2).abs() < 1e-10); // Probabilities unchanged
        }
    }

    #[test]
    fn test_u() {
        let mut q = StateVecAoS::new(1);

        // Apply some arbitrary rotation
        let theta = PI / 5.0;
        let phi = PI / 7.0;
        let lambda = PI / 3.0;
        q.u(
            Angle64::from_radians(theta),
            Angle64::from_radians(phi),
            Angle64::from_radians(lambda),
            &qid(0),
        );

        // Verify normalization is preserved
        let norm: f64 = q.state.iter().map(num_complex::Complex::norm_sqr).sum();
        assert!((norm - 1.0).abs() < 1e-10);

        // Verify expected amplitudes
        let expected_0 = (theta / 2.0).cos();
        assert!((q.state[0].re - expected_0).abs() < 1e-10);

        let expected_1_mag = (theta / 2.0).sin();
        assert!((q.state[1].norm() - expected_1_mag).abs() < 1e-10);
    }
    #[test]
    fn test_r1xy() {
        // Initialize state vectors with one qubit in the |0⟩ state.
        let mut state_vec_r1xy = StateVecAoS::new(1);
        let mut trait_r1xy = StateVecAoS::new(1);

        // Define angles for the test.
        let theta = FRAC_PI_3;
        let phi = FRAC_PI_4;

        // Apply the manual `r1xy` implementation.
        state_vec_r1xy.r1xy(
            Angle64::from_radians(theta),
            Angle64::from_radians(phi),
            &qid(0),
        );

        // Apply the `r1xy` implementation from the `ArbitraryRotationGateable` trait.
        ArbitraryRotationGateable::r1xy(
            &mut trait_r1xy,
            Angle64::from_radians(theta),
            Angle64::from_radians(phi),
            &qid(0),
        );

        // Use the `assert_states_equal` function to compare the states up to a global phase.
        assert_states_equal(&state_vec_r1xy.state, &trait_r1xy.state);
    }

    // Basic two-qubit rotation gate tests
    // ===================================
    #[test]
    fn test_rxx() {
        // Test 1: RXX(π/2) on |00⟩ should give (|00⟩ - i|11⟩)/√2
        let mut q = StateVecAoS::new(2);
        q.rxx(
            Angle64::from_radians(FRAC_PI_2),
            &[(QubitId(0), QubitId(1))],
        );

        let expected = FRAC_1_SQRT_2;
        assert!((q.state[0].re - expected).abs() < 1e-10);
        assert!(q.state[1].norm() < 1e-10);
        assert!(q.state[2].norm() < 1e-10);
        assert!((q.state[3].im + expected).abs() < 1e-10);

        // Test 2: RXX(2π) should return to original state up to global phase
        let mut q = StateVecAoS::new(2);
        q.h(&qid(0)); // Create some initial state
        let initial = q.state.clone();
        q.rxx(Angle64::from_radians(TAU), &[(QubitId(0), QubitId(1))]);

        // Compare up to global phase
        if q.state[0].norm() > 1e-10 {
            let phase = q.state[0] / initial[0];
            for (a, b) in q.state.iter().zip(initial.iter()) {
                assert!((a - b * phase).norm() < 1e-10);
            }
        }

        // Test 3: RXX(π) should flip |00⟩ to |11⟩ up to phase
        let mut q = StateVecAoS::new(2);
        q.rxx(Angle64::from_radians(PI), &[(QubitId(0), QubitId(1))]);

        // Should get -i|11⟩
        assert!(q.state[0].norm() < 1e-10);
        assert!(q.state[1].norm() < 1e-10);
        assert!(q.state[2].norm() < 1e-10);
        assert!((q.state[3] - Complex64::new(0.0, -1.0)).norm() < 1e-10);
    }

    #[test]
    fn test_ryy() {
        let expected = FRAC_1_SQRT_2;

        // Test all basis states for RYY(π/2)
        // |00⟩ -> (1/√2)|00⟩ - i(1/√2)|11⟩
        let mut q = StateVecAoS::new(2);
        q.ryy(
            Angle64::from_radians(FRAC_PI_2),
            &[(QubitId(0), QubitId(1))],
        );
        assert!((q.state[0].re - expected).abs() < 1e-10);
        assert!(q.state[1].norm() < 1e-10);
        assert!(q.state[2].norm() < 1e-10);
        assert!((q.state[3].im - expected).abs() < 1e-10);

        // |11⟩ -> i(1/√2)|00⟩ + (1/√2)|11⟩
        let mut q = StateVecAoS::new(2);
        q.x(&qid(0)).x(&qid(1)); // Prepare |11⟩
        q.ryy(
            Angle64::from_radians(FRAC_PI_2),
            &[(QubitId(0), QubitId(1))],
        );
        assert!((q.state[0].im - expected).abs() < 1e-10);
        assert!(q.state[1].norm() < 1e-10);
        assert!(q.state[2].norm() < 1e-10);
        assert!((q.state[3].re - expected).abs() < 1e-10);

        // |01⟩ -> (1/√2)|01⟩ + i(1/√2)|10⟩
        let mut q = StateVecAoS::new(2);
        q.x(&qid(1)); // Prepare |01⟩
        q.ryy(
            Angle64::from_radians(FRAC_PI_2),
            &[(QubitId(0), QubitId(1))],
        );
        assert!(q.state[0].norm() < 1e-10);
        assert!(q.state[1].re.abs() < 1e-10);
        assert!((q.state[1].im + expected).abs() < 1e-10);
        assert!((q.state[2].re - expected).abs() < 1e-10);
        assert!(q.state[2].im.abs() < 1e-10);
        assert!(q.state[3].norm() < 1e-10);

        // |10⟩ -> (1/√2)|10⟩ + i(1/√2)|01⟩
        let mut q = StateVecAoS::new(2);
        q.x(&qid(0)); // Prepare |10⟩
        q.ryy(
            Angle64::from_radians(FRAC_PI_2),
            &[(QubitId(0), QubitId(1))],
        );
        assert!(q.state[0].norm() < 1e-10);
        assert!((q.state[1].re - expected).abs() < 1e-10);
        assert!(q.state[1].im.abs() < 1e-10);
        assert!(q.state[2].re.abs() < 1e-10);
        assert!((q.state[2].im + expected).abs() < 1e-10);
        assert!(q.state[3].norm() < 1e-10);

        // Test properties

        // 1. Periodicity: RYY(2π) = I
        let mut q = StateVecAoS::new(2);
        q.h(&qid(0)); // Create non-trivial initial state
        let initial = q.state.clone();
        q.ryy(Angle64::from_radians(TAU), &[(QubitId(0), QubitId(1))]);
        // Need to account for potential global phase
        if q.state[0].norm() > 1e-10 {
            let phase = q.state[0] / initial[0];
            for (a, b) in q.state.iter().zip(initial.iter()) {
                assert!(
                    (a - b * phase).norm() < 1e-10,
                    "Periodicity test failed: a={a}, b={b}"
                );
            }
        }

        // 2. Composition: RYY(θ₁)RYY(θ₂) = RYY(θ₁ + θ₂)
        let mut q1 = StateVecAoS::new(2);
        let mut q2 = StateVecAoS::new(2);
        q1.h(&qid(0)); // Create non-trivial initial state
        q2.h(&qid(0)); // Same initial state
        q1.ryy(
            Angle64::from_radians(FRAC_PI_3),
            &[(QubitId(0), QubitId(1))],
        )
        .ryy(
            Angle64::from_radians(FRAC_PI_6),
            &[(QubitId(0), QubitId(1))],
        );
        q2.ryy(
            Angle64::from_radians(FRAC_PI_2),
            &[(QubitId(0), QubitId(1))],
        );
        // Compare up to global phase
        if q1.state[0].norm() > 1e-10 {
            let phase = q1.state[0] / q2.state[0];
            for (a, b) in q1.state.iter().zip(q2.state.iter()) {
                assert!(
                    (a - b * phase).norm() < 1e-10,
                    "Composition test failed: a={a}, b={b}"
                );
            }
        }

        // 3. Symmetry: RYY(θ,0,1) = RYY(θ,1,0)
        let mut q1 = StateVecAoS::new(2);
        let mut q2 = StateVecAoS::new(2);
        q1.h(&qid(0)).h(&qid(1)); // Create non-trivial initial state
        q2.h(&qid(0)).h(&qid(1)); // Same initial state
        q1.ryy(
            Angle64::from_radians(FRAC_PI_3),
            &[(QubitId(0), QubitId(1))],
        );
        q2.ryy(
            Angle64::from_radians(FRAC_PI_3),
            &[(QubitId(1), QubitId(0))],
        );
        // States should be exactly equal (no phase difference)
        for (a, b) in q1.state.iter().zip(q2.state.iter()) {
            assert!((a - b).norm() < 1e-10, "Symmetry test failed: a={a}, b={b}");
        }
    }

    #[test]
    fn test_rzz() {
        // Test 1: RZZ(π) on (|00⟩ + |11⟩)/√2 should give itself
        let mut q = StateVecAoS::new(2);
        // Create Bell state
        q.h(&qid(0));
        q.cx(&[(QubitId(0), QubitId(1))]);
        let initial = q.state.clone();

        q.rzz(Angle64::from_radians(PI), &[(QubitId(0), QubitId(1))]);

        // Compare up to global phase
        if q.state[0].norm() > 1e-10 {
            let phase = q.state[0] / initial[0];
            for (a, b) in q.state.iter().zip(initial.iter()) {
                assert!((a - b * phase).norm() < 1e-10);
            }
        }

        // Test 2: RZZ(π/2) on |++⟩
        let mut q = StateVecAoS::new(2);
        q.h(&qid(0));
        q.h(&qid(1));
        q.rzz(
            Angle64::from_radians(FRAC_PI_2),
            &[(QubitId(0), QubitId(1))],
        );

        // e^(-iπ/4) = (1-i)/√2
        // e^(iπ/4) = (1+i)/√2
        let factor = 0.5; // 1/2 for the |++⟩ normalization
        let exp_minus_i_pi_4 = Complex64::new(1.0, -1.0) / (2.0_f64.sqrt());
        let exp_plus_i_pi_4 = Complex64::new(1.0, 1.0) / (2.0_f64.sqrt());

        assert!((q.state[0] - factor * exp_minus_i_pi_4).norm() < 1e-10); // |00⟩
        assert!((q.state[1] - factor * exp_plus_i_pi_4).norm() < 1e-10); // |01⟩
        assert!((q.state[2] - factor * exp_plus_i_pi_4).norm() < 1e-10); // |10⟩
        assert!((q.state[3] - factor * exp_minus_i_pi_4).norm() < 1e-10); // |11⟩
    }

    // Core mathematical properties
    // ============================
    #[test]
    fn test_normalization() {
        let mut q = StateVecAoS::new(1);
        q.h(&qid(0)).sz(&qid(0));
        let norm: f64 = q.state.iter().map(num_complex::Complex::norm_sqr).sum();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_unitarity() {
        let mut q = StateVecAoS::new(1);
        q.h(&qid(0));
        let initial = q.state.clone();
        q.h(&qid(0)).h(&qid(0));
        assert_states_equal(&q.state, &initial);
    }

    #[test]
    fn test_pauli_relations() {
        let mut q1 = StateVecAoS::new(1);
        let mut q2 = StateVecAoS::new(1);

        // Store initial state
        let initial_state = q1.state.clone();

        // Test XYZ sequence
        q1.x(&qid(0));
        q1.y(&qid(0));
        q1.z(&qid(0));

        // XYZ = -iI, so state should be -i times initial state
        if initial_state[0].norm() > 1e-10 {
            let phase = q1.state[0] / initial_state[0];
            assert!((phase + Complex64::i()).norm() < 1e-10); // Changed to +Complex64::i()
        }

        // Test YZX sequence - should give same result
        q2.y(&qid(0));
        q2.z(&qid(0));
        q2.x(&qid(0));

        // Compare q1 and q2 up to global phase
        if q1.state[0].norm() > 1e-10 {
            let phase = q2.state[0] / q1.state[0];
            let phase_norm = phase.norm();
            assert!((phase_norm - 1.0).abs() < 1e-10);

            for (a, b) in q1.state.iter().zip(q2.state.iter()) {
                assert!((a * phase - b).norm() < 1e-10);
            }
        }
    }

    // test core general gates
    // =======================
    #[test]
    fn test_single_qubit_rotation() {
        let mut q = StateVecAoS::new(1);

        // Test 1: Hadamard gate
        let h00 = Complex64::new(FRAC_1_SQRT_2, 0.0);
        let h01 = Complex64::new(FRAC_1_SQRT_2, 0.0);
        let h10 = Complex64::new(FRAC_1_SQRT_2, 0.0);
        let h11 = Complex64::new(-FRAC_1_SQRT_2, 0.0);

        q.single_qubit_rotation(0, h00, h01, h10, h11);
        assert!((q.state[0].re - FRAC_1_SQRT_2).abs() < 1e-10);
        assert!((q.state[1].re - FRAC_1_SQRT_2).abs() < 1e-10);

        // Test 2: X gate
        let mut q = StateVecAoS::new(1);
        let x00 = Complex64::new(0.0, 0.0);
        let x01 = Complex64::new(1.0, 0.0);
        let x10 = Complex64::new(1.0, 0.0);
        let x11 = Complex64::new(0.0, 0.0);

        q.single_qubit_rotation(0, x00, x01, x10, x11);
        assert!(q.state[0].norm() < 1e-10);
        assert!((q.state[1].re - 1.0).abs() < 1e-10);

        // Test 3: Phase gate
        let mut q = StateVecAoS::new(1);
        let p00 = Complex64::new(1.0, 0.0);
        let p01 = Complex64::new(0.0, 0.0);
        let p10 = Complex64::new(0.0, 0.0);
        let p11 = Complex64::new(0.0, 1.0);

        q.single_qubit_rotation(0, p00, p01, p10, p11);
        assert!((q.state[0].re - 1.0).abs() < 1e-10);
        assert!(q.state[1].norm() < 1e-10);

        // Test 4: Y gate using unitary
        let mut q = StateVecAoS::new(1);
        let y00 = Complex64::new(0.0, 0.0);
        let y01 = Complex64::new(0.0, -1.0);
        let y10 = Complex64::new(0.0, 1.0);
        let y11 = Complex64::new(0.0, 0.0);

        q.single_qubit_rotation(0, y00, y01, y10, y11);
        assert!(q.state[0].norm() < 1e-10);
        assert!((q.state[1].im - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_unitary_properties() {
        let mut q = StateVecAoS::new(1);

        // Create random state with Hadamard
        q.h(&qid(0));

        // Apply Z gate as unitary
        let z00 = Complex64::new(1.0, 0.0);
        let z01 = Complex64::new(0.0, 0.0);
        let z10 = Complex64::new(0.0, 0.0);
        let z11 = Complex64::new(-1.0, 0.0);

        let initial = q.state.clone();
        q.single_qubit_rotation(0, z00, z01, z10, z11);

        // Check normalization is preserved
        let norm: f64 = q.state.iter().map(num_complex::Complex::norm_sqr).sum();
        assert!((norm - 1.0).abs() < 1e-10);

        // Apply Z again - should get back original state
        q.single_qubit_rotation(0, z00, z01, z10, z11);

        for (a, b) in q.state.iter().zip(initial.iter()) {
            assert!((a - b).norm() < 1e-10);
        }
    }

    #[test]
    fn test_two_qubit_unitary_cnot() {
        // Test that we can implement CNOT using the general unitary
        let mut q1 = StateVecAoS::new(2);
        let mut q2 = StateVecAoS::new(2);

        // CNOT matrix
        let cnot = [
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        ];

        // Create Bell state using both methods
        q1.h(&qid(0));
        q1.cx(&[(QubitId(0), QubitId(1))]);

        q2.h(&qid(0));
        q2.two_qubit_unitary(0, 1, cnot);

        // Compare results
        for (a, b) in q1.state.iter().zip(q2.state.iter()) {
            assert!((a - b).norm() < 1e-10);
        }
    }

    #[test]
    fn test_two_qubit_unitary_swap() {
        // Test SWAP gate
        let mut q = StateVecAoS::new(2);

        // Prepare |10⟩ state
        q.x(&qid(0));

        // SWAP matrix
        let swap = [
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        ];

        q.two_qubit_unitary(0, 1, swap);

        // Should be in |01⟩ state
        assert!(q.state[0].norm() < 1e-10);
        assert!((q.state[2].re - 1.0).abs() < 1e-10);
    }
}
