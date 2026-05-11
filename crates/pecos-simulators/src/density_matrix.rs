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
use super::state_vec::StateVec;
use super::state_vec_soa::StateVecSoA;
use nalgebra::DMatrix;
use pecos_core::{Angle64, ChannelExpr, QubitId, RngManageable};
use pecos_quantum::{ChannelError, KrausOps};
use pecos_random::{PecosRng, Rng, RngExt, SeedableRng};

use core::fmt::{Debug, Display, Formatter, Write};
use num_complex::Complex64;
use std::error::Error;

const PURE_STATE_TOLERANCE: f64 = 1e-10;

/// Error returned when converting between simulator state representations.
#[derive(Clone, Debug, PartialEq)]
pub enum StateConversionError {
    /// The density matrix is not rank-1 within the conversion tolerance.
    MixedDensityMatrix { residual: f64 },
}

impl Display for StateConversionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MixedDensityMatrix { residual } => write!(
                f,
                "density matrix is not a pure state; reconstruction residual is {residual}"
            ),
        }
    }
}

impl Error for StateConversionError {}

/// A quantum state simulator using the density matrix representation via the Choi-Jamiolkowski isomorphism
///
/// `DensityMatrix` represents an N-qubit density matrix as a 2N-qubit state vector,
/// which allows reusing the state vector operations for density matrix simulation.
/// This enables the simulation of both pure and mixed quantum states, including the effects of noise.
///
/// # Type Parameters
/// * `R` - Random number generator type implementing `Rng + SeedableRng` traits
///
/// # Examples
/// ```rust
/// use pecos_simulators::DensityMatrix;
///
/// // Create a new 2-qubit system
/// let mut state = DensityMatrix::new(2);
///
/// // Prepare a superposition state
/// state.prepare_plus_state();
/// ```
#[derive(Clone, Debug)]
pub struct DensityMatrix<R = PecosRng>
where
    R: Rng + SeedableRng + Debug + Clone,
{
    /// Number of qubits in the physical system
    num_physical_qubits: usize,

    /// The underlying state vector (representing a 2N-qubit system)
    state_vector: StateVec<R>,
}

impl DensityMatrix {
    /// Create a new density matrix initialized to |0...0⟩⟨0...0|
    ///
    /// # Examples
    /// ```rust
    /// use pecos_simulators::DensityMatrix;
    ///
    /// // Initialize a 3-qubit density matrix in the |000⟩⟨000| state
    /// let mut density_matrix = DensityMatrix::new(3);
    ///
    /// // Confirm the state is |000⟩⟨000|
    /// let prob = density_matrix.probability(0);
    /// assert!((prob - 1.0).abs() < 1e-10);
    /// ```
    #[inline]
    #[must_use]
    pub fn new(num_physical_qubits: usize) -> DensityMatrix<PecosRng> {
        let rng = rand::make_rng();
        DensityMatrix::with_rng(num_physical_qubits, rng)
    }

    /// Create a new density matrix simulator with a specific seed for the random number generator
    ///
    /// This method allows for deterministic behavior by setting a specific seed for the
    /// random number generator, while still using the default RNG type (`PecosRng`).
    ///
    /// # Arguments
    /// * `num_physical_qubits` - Number of qubits in the physical system
    /// * `seed` - Seed value for the random number generator
    ///
    /// # Examples
    /// ```rust
    /// use pecos_simulators::DensityMatrix;
    ///
    /// // Create a simulator with a specific seed
    /// let state = DensityMatrix::with_seed(2, 42);
    /// ```
    #[inline]
    #[must_use]
    pub fn with_seed(num_physical_qubits: usize, seed: u64) -> DensityMatrix<PecosRng> {
        let rng = PecosRng::seed_from_u64(seed);
        DensityMatrix::with_rng(num_physical_qubits, rng)
    }
}

impl<R> DensityMatrix<R>
where
    R: Rng + SeedableRng + Debug + Clone,
{
    /// Returns the number of qubits in the physical system
    ///
    /// # Returns
    /// * `usize` - The total number of physical qubits this simulator is configured to handle
    ///
    /// # Examples
    /// ```rust
    /// use pecos_simulators::{QuantumSimulator, DensityMatrix, qid};
    /// let state = DensityMatrix::new(2);
    /// let num = state.num_qubits();
    /// assert_eq!(num, 2);
    /// ```
    #[inline]
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_physical_qubits
    }

    /// Create a new density matrix with a custom random number generator
    ///
    /// # Arguments
    /// * `num_physical_qubits` - Number of qubits in the physical system
    /// * `rng` - Random number generator implementing `Rng + SeedableRng` traits
    ///
    /// # Examples
    /// ```rust
    /// use pecos_simulators::DensityMatrix;
    /// use pecos_random::{PecosRng, SeedableRng};
    ///
    /// let rng = PecosRng::seed_from_u64(42);
    /// let state = DensityMatrix::with_rng(2, rng);
    /// ```
    #[inline]
    #[must_use]
    pub fn with_rng(num_physical_qubits: usize, rng: R) -> Self {
        // Create a state vector with twice the number of qubits
        let state_vector = StateVec::with_rng(2 * num_physical_qubits, rng);

        DensityMatrix {
            num_physical_qubits,
            state_vector,
        }
    }

    /// Returns the underlying state vector representation
    ///
    /// This provides access to the 2N-qubit state vector that represents the density matrix
    /// via the Choi-Jamiolkowski isomorphism.
    ///
    /// # Returns
    /// * `&StateVec<R>` - A reference to the underlying state vector
    #[inline]
    #[must_use]
    pub fn state_vector(&self) -> &StateVec<R> {
        &self.state_vector
    }

    /// Returns a mutable reference to the underlying state vector
    ///
    /// # Returns
    /// * `&mut StateVec<R>` - A mutable reference to the underlying state vector
    #[inline]
    #[must_use]
    pub fn state_vector_mut(&mut self) -> &mut StateVec<R> {
        &mut self.state_vector
    }

    /// Returns the density matrix as a 2D Vector of Complex64 values
    ///
    /// This function extracts the actual density matrix from the Choi representation,
    /// making it easier to inspect the quantum state.
    ///
    /// # Returns
    /// * `Vec<Vec<Complex64>>` - A 2D vector representing the density matrix
    ///
    /// # Examples
    /// ```rust
    /// use pecos_core::QubitId;
    /// use pecos_simulators::{DensityMatrix, CliffordGateable, qid};
    ///
    /// // Create a Bell state
    /// let mut state = DensityMatrix::new(2);
    /// state.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]);
    ///
    /// // Get the density matrix representation
    /// let rho = state.get_density_matrix();
    ///
    /// // A Bell state should have non-zero elements at [0,0], [0,3], [3,0], and [3,3]
    /// assert!(rho[0][0].re.abs() > 0.0);
    /// assert!(rho[0][3].re.abs() > 0.0);
    /// assert!(rho[3][0].re.abs() > 0.0);
    /// assert!(rho[3][3].re.abs() > 0.0);
    /// ```
    #[inline]
    #[must_use]
    pub fn get_density_matrix(&mut self) -> Vec<Vec<Complex64>> {
        let sv = self.state_vector.state();
        Self::density_matrix_from_state_vec(&sv, self.num_physical_qubits)
    }

    /// Get the density matrix without flushing pending gates.
    ///
    /// This method is used for Display since Display requires &self.
    /// WARNING: May return stale data if gates are pending.
    fn get_density_matrix_no_flush(&self) -> Vec<Vec<Complex64>> {
        let sv = self.state_vector.state_no_flush();
        Self::density_matrix_from_state_vec(&sv, self.num_physical_qubits)
    }

    /// Helper to compute density matrix from a state vector.
    fn density_matrix_from_state_vec(
        sv: &[Complex64],
        num_physical_qubits: usize,
    ) -> Vec<Vec<Complex64>> {
        let n = num_physical_qubits;
        let dim = 1 << n;

        // Initialize density matrix with zeros
        let mut rho = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];

        // Extract density matrix elements from the Choi representation
        for (row, rho_row) in rho.iter_mut().enumerate() {
            for (col, rho_element) in rho_row.iter_mut().enumerate() {
                // Calculate the corresponding elements in the state vector
                // For density matrix element ρ_{row,col}
                let mut element = Complex64::new(0.0, 0.0);

                for i in 0..dim {
                    // Map row/col to the corresponding indices in the state vector
                    let idx1 = (row << n) | i;
                    let idx2 = (col << n) | i;

                    // Sum over the corresponding pairs of amplitudes
                    element += sv[idx1] * sv[idx2].conj();
                }

                *rho_element = element;
            }
        }

        rho
    }

    /// Returns a formatted string representation of the density matrix
    ///
    /// This function generates a human-readable string representation of the
    /// density matrix, with options to control the formatting.
    ///
    /// # Arguments
    /// * `precision` - Number of decimal places to show (default: 4)
    /// * `threshold` - Minimum absolute value to display (smaller values shown as 0, default: 1e-10)
    ///
    /// # Returns
    /// * `String` - A formatted string representation of the density matrix
    ///
    /// # Examples
    /// ```rust
    /// use pecos_core::QubitId;
    /// use pecos_simulators::{DensityMatrix, CliffordGateable, qid};
    ///
    /// // Create a Bell state
    /// let mut state = DensityMatrix::new(2);
    /// state.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]);
    ///
    /// // Print the density matrix with 6 decimal places
    /// let matrix_str = state.density_matrix_to_string(6, 1e-10);
    /// println!("{}", matrix_str);
    /// ```
    #[inline]
    #[must_use]
    pub fn density_matrix_to_string(&mut self, precision: usize, threshold: f64) -> String {
        let rho = self.get_density_matrix();
        Self::format_density_matrix(&rho, precision, threshold)
    }

    /// Format density matrix to string without flushing pending gates.
    /// Used by Display since Display requires &self.
    fn density_matrix_to_string_no_flush(&self, precision: usize, threshold: f64) -> String {
        let rho = self.get_density_matrix_no_flush();
        Self::format_density_matrix(&rho, precision, threshold)
    }

    /// Helper to format a density matrix to string.
    fn format_density_matrix(rho: &[Vec<Complex64>], precision: usize, threshold: f64) -> String {
        let dim = rho.len();

        let mut result = String::with_capacity(dim * dim * (precision + 8));
        result.push_str("Density matrix (ρ):\n");

        for rho_row in rho {
            result.push('[');
            for (col, val) in rho_row.iter().enumerate() {
                // Apply threshold to small values
                let re = if val.re.abs() < threshold {
                    0.0
                } else {
                    val.re
                };
                let im = if val.im.abs() < threshold {
                    0.0
                } else {
                    val.im
                };

                // Format the complex number
                if im.abs() < threshold {
                    // Real number
                    write!(result, "{re:.precision$}").unwrap();
                } else if re.abs() < threshold {
                    // Imaginary number
                    write!(result, "{im:.precision$}i").unwrap();
                } else {
                    // Full complex number
                    let sign = if im >= 0.0 { "+" } else { "-" };
                    write!(
                        result,
                        "{:.*}{}{:.*}i",
                        precision,
                        re,
                        sign,
                        precision,
                        im.abs()
                    )
                    .unwrap();
                }

                if col < dim - 1 {
                    result.push_str(", ");
                }
            }
            result.push_str("]\n");
        }

        result
    }

    /// Returns the density matrix as a flattened complex vector in row-major order
    ///
    /// This function extracts the actual density matrix from the Choi representation
    /// and returns it as a 1D vector in row-major order, which is compatible with
    /// many numerical computing libraries.
    ///
    /// # Returns
    /// * `Vec<Complex64>` - A flattened row-major vector representation of the density matrix
    ///
    /// # Examples
    /// ```rust
    /// use pecos_core::QubitId;
    /// use pecos_simulators::{DensityMatrix, CliffordGateable, qid};
    ///
    /// // Create a Bell state
    /// let mut state = DensityMatrix::new(2);
    /// state.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]);
    ///
    /// // Get the flattened density matrix
    /// let flat_rho = state.get_flattened_density_matrix();
    /// ```
    #[inline]
    #[must_use]
    pub fn get_flattened_density_matrix(&mut self) -> Vec<Complex64> {
        self.get_density_matrix().into_iter().flatten().collect()
    }

    /// Returns the probability of measuring a specific basis state
    ///
    /// # Arguments
    /// * `basis_state` - The computational basis state to measure
    ///
    /// # Returns
    /// * `f64` - The probability of measuring the given basis state
    ///
    /// # Panics
    /// Code will panic if `basis_state` >= `2^num_qubits` (i.e., if the basis state index is too large for the number of qubits)
    #[inline]
    #[must_use]
    pub fn probability(&mut self, basis_state: usize) -> f64 {
        assert!(basis_state < 1 << self.num_physical_qubits);

        // In the Choi representation, the diagonal elements of the density matrix
        // correspond to specific elements in the state vector
        let n = self.num_physical_qubits;
        let basis_mask = (1 << n) - 1;

        // Get state once and cache it
        let sv = self.state_vector.state();

        // Calculate probability by summing appropriate elements
        let mut prob = 0.0;
        for i in 0..(1 << n) {
            // Map to the corresponding index in the state vector
            // For the element ρ_{basis_state,basis_state} in density matrix
            let state_idx = ((basis_state & basis_mask) << n) | (i & basis_mask);
            prob += sv[state_idx].norm_sqr();
        }

        prob
    }

    /// Calculate the purity of the quantum state, defined as Tr(rho^2)
    ///
    /// For pure states, purity = 1
    /// For mixed states, purity < 1 (with minimum value 1/2^n for maximally mixed state)
    ///
    /// # Returns
    /// * `f64` - The purity of the quantum state
    #[inline]
    #[must_use]
    pub fn purity(&mut self) -> f64 {
        // Purity = Tr(rho^2) = sum_{i,j} |rho[i][j]|^2
        self.get_density_matrix()
            .iter()
            .flatten()
            .map(Complex64::norm_sqr)
            .sum()
    }

    /// Check if the quantum state is pure
    ///
    /// A pure state has purity = 1
    ///
    /// # Returns
    /// * `bool` - True if the state is pure, false otherwise
    #[inline]
    #[must_use]
    pub fn is_pure(&mut self) -> bool {
        const TOLERANCE: f64 = 1e-10;
        (self.purity() - 1.0).abs() < TOLERANCE
    }

    /// Prepare a specific computational basis state
    ///
    /// # Arguments
    /// * `basis_state` - The computational basis state to prepare
    ///
    /// # Returns
    /// * `&mut Self` - Returns self for method chaining
    ///
    /// # Panics
    /// Code will panic if `basis_state` >= `2^num_qubits` (i.e., if the basis state index is too large for the number of qubits)
    #[inline]
    pub fn prepare_computational_basis(&mut self, basis_state: usize) -> &mut Self {
        assert!(basis_state < 1 << self.num_physical_qubits);

        // Reset the state vector
        let n = self.num_physical_qubits;
        let sv_size = 1 << (2 * n);
        let mut new_state = vec![Complex64::new(0.0, 0.0); sv_size];

        // In Choi representation, a pure state |ψ⟩⟨ψ| has a specific pattern
        // For computational basis state |basis_state⟩, we set the corresponding element
        let idx = (basis_state << n) | basis_state;
        new_state[idx] = Complex64::new(1.0, 0.0);

        // Update the state vector
        let new_sv = StateVec::from_state(&new_state, self.state_vector.rng().clone());
        *self.state_vector_mut() = new_sv;

        self
    }

    /// Prepare all qubits in the |+⟩ state, creating a pure state of tensor product of |+⟩ states
    ///
    /// # Returns
    /// * `&mut Self` - Returns self for method chaining
    #[inline]
    pub fn prepare_plus_state(&mut self) -> &mut Self {
        let n = self.num_physical_qubits;

        // First prepare |0...0⟩ state
        self.prepare_computational_basis(0);

        // Apply Hadamard gates to all qubits
        for q in 0..n {
            self.h(&[QubitId(q)]);
        }

        self
    }

    /// Prepare the maximally mixed state I/2ⁿ
    ///
    /// # Returns
    /// * `&mut Self` - Returns self for method chaining
    #[inline]
    pub fn prepare_maximally_mixed(&mut self) -> &mut Self {
        let n = self.num_physical_qubits;
        let sv_size = 1 << (2 * n);
        let dim = 1 << n;
        let mut new_state = vec![Complex64::new(0.0, 0.0); sv_size];

        // In Choi representation, for density matrix element rho_{i,j}:
        // rho_{i,j} = sum_k psi[(i<<n)|k] * psi*[(j<<n)|k]
        //
        // For I/dim (maximally mixed state), we need rho_{i,i} = 1/dim.
        // Setting psi[(i<<n)|i] = 1/sqrt(dim) gives:
        // rho_{i,i} = |1/sqrt(dim)|^2 = 1/dim
        // Note: dim is always 2^n for n qubits, which is safe to cast for realistic quantum systems
        #[allow(clippy::cast_precision_loss)]
        let factor = 1.0 / (dim as f64).sqrt();

        // Set diagonal elements for the maximally mixed state
        for i in 0..dim {
            let idx = (i << n) | i;
            new_state[idx] = Complex64::new(factor, 0.0);
        }

        // Update the state vector
        let new_sv = StateVec::from_state(&new_state, self.state_vector.rng().clone());
        *self.state_vector_mut() = new_sv;

        self
    }

    /// Apply a depolarizing noise channel to a qubit
    ///
    /// The depolarizing channel applies a random Pauli error (X, Y, or Z) with probability p/3 each,
    /// or leaves the state unchanged with probability 1-p.
    ///
    /// # Arguments
    /// * `qubit` - Target qubit
    /// * `probability` - Probability of applying a Pauli error (0.0 to 1.0)
    ///
    /// # Returns
    /// * `&mut Self` - Returns self for method chaining
    #[inline]
    pub fn apply_depolarizing_noise(&mut self, qubit: usize, probability: f64) -> &mut Self {
        // Ensure probability is in valid range
        let p = probability.clamp(0.0, 1.0);

        if p < f64::EPSILON {
            // No noise, return unchanged
            return self;
        }

        // Depolarizing channel: rho -> (1-p) rho + (p/3)(X rho X + Y rho Y + Z rho Z)
        //
        // For density matrix elements:
        // - Off-diagonal (qubit q differs): scale by (1 - 4p/3)
        // - Diagonal in qubit q: rho_{i,j} -> (1-2p/3) rho_{i,j} + (2p/3) rho_{i^q, j^q}
        //   where i^q means i with bit q flipped

        let n = self.num_physical_qubits;
        let dim = 1 << n;
        let qubit_mask = 1 << qubit;

        // Get current density matrix
        let rho = self.get_density_matrix();

        // Apply depolarizing transformation
        let mut new_rho = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];

        for i in 0..dim {
            for j in 0..dim {
                let i_bit = (i & qubit_mask) != 0;
                let j_bit = (j & qubit_mask) != 0;

                if i_bit == j_bit {
                    // Diagonal in qubit q: mix with flipped qubit
                    let i_flipped = i ^ qubit_mask;
                    let j_flipped = j ^ qubit_mask;
                    new_rho[i][j] = (1.0 - 2.0 * p / 3.0) * rho[i][j]
                        + (2.0 * p / 3.0) * rho[i_flipped][j_flipped];
                } else {
                    // Off-diagonal in qubit q: scale by (1 - 4p/3)
                    new_rho[i][j] = (1.0 - 4.0 * p / 3.0) * rho[i][j];
                }
            }
        }

        // Convert new density matrix back to Choi representation using Cholesky decomposition
        // rho = L L^dagger, then set psi[(i<<n)|j] = L[i][j]
        self.set_from_density_matrix(&new_rho);

        self
    }

    /// Set the Choi state from a density matrix using Cholesky decomposition
    ///
    /// This finds a purification of the given density matrix.
    fn set_from_density_matrix(&mut self, rho: &[Vec<Complex64>]) {
        let n = self.num_physical_qubits;
        let dim = rho.len();

        // Compute Cholesky decomposition: rho = L L^dagger
        // For numerical stability, add a small epsilon to diagonal if needed
        let mut l = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];

        for i in 0..dim {
            for j in 0..=i {
                let mut sum = rho[i][j];

                for (li_k, lj_k) in l[i].iter().take(j).zip(l[j].iter().take(j)) {
                    sum -= li_k * lj_k.conj();
                }

                if i == j {
                    // Diagonal element
                    let diag_val = sum.re.max(0.0); // Ensure non-negative
                    l[i][j] = Complex64::new(diag_val.sqrt(), 0.0);
                } else if l[j][j].norm() > 1e-15 {
                    // Off-diagonal element
                    l[i][j] = sum / l[j][j];
                }
            }
        }

        // Create new Choi state: psi[(i<<n)|j] = L[i][j]
        let sv_size = 1 << (2 * n);
        let mut new_state = vec![Complex64::new(0.0, 0.0); sv_size];

        for (i, l_row) in l.iter().enumerate() {
            for (j, l_ij) in l_row.iter().enumerate() {
                let idx = (i << n) | j;
                new_state[idx] = *l_ij;
            }
        }

        let new_sv = StateVec::from_state(&new_state, self.state_vector.rng().clone());
        *self.state_vector_mut() = new_sv;
    }

    /// Apply a amplitude damping noise channel to a qubit
    ///
    /// The amplitude damping channel models energy dissipation in a quantum system,
    /// representing the process where a qubit in state |1⟩ decays to state |0⟩ with probability gamma.
    ///
    /// # Arguments
    /// * `qubit` - Target qubit
    /// * `gamma` - Damping parameter (0.0 to 1.0)
    ///
    /// # Returns
    /// * `&mut Self` - Returns self for method chaining
    #[inline]
    pub fn apply_amplitude_damping(&mut self, qubit: usize, gamma: f64) -> &mut Self {
        let gamma = gamma.clamp(0.0, 1.0);
        if gamma < f64::EPSILON {
            return self;
        }

        // Amplitude damping via Kraus ops
        //   E_0 = |0><0| + sqrt(1-g)|1><1|,   E_1 = sqrt(g)|0><1|
        // gives the density-matrix transformation
        //   rho_{a,b} -> E(rho)_{a,b} =
        //     (a,b both bit_q=0): rho_{a,b} + g * rho_{a|q, b|q}
        //     (one of a,b bit_q=1): sqrt(1-g) * rho_{a,b}
        //     (both bit_q=1):     (1-g) * rho_{a,b}
        //
        // Apply on the density matrix, then Cholesky-re-purify the Choi state.
        // This preserves the invariant that `probability()` reads rho_{k,k} as
        // sum_i |psi[(k<<n)|i]|^2 -- the direct-Choi shortcut used previously
        // broke that identity for partial damping.
        let n = self.num_physical_qubits;
        let dim = 1usize << n;
        let qubit_mask = 1usize << qubit;

        let rho = self.get_density_matrix();
        let mut new_rho = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];
        let sqrt_1mg = (1.0 - gamma).sqrt();
        for i in 0..dim {
            let i1 = (i & qubit_mask) != 0;
            for j in 0..dim {
                let j1 = (j & qubit_mask) != 0;
                new_rho[i][j] = match (i1, j1) {
                    (false, false) => {
                        let ii = i | qubit_mask;
                        let jj = j | qubit_mask;
                        rho[i][j] + gamma * rho[ii][jj]
                    }
                    (true, true) => (1.0 - gamma) * rho[i][j],
                    _ => sqrt_1mg * rho[i][j],
                };
            }
        }
        self.set_from_density_matrix(&new_rho);
        self
    }

    /// Apply a phase damping noise channel to a qubit
    ///
    /// The phase damping channel models pure decoherence without energy dissipation,
    /// causing the loss of quantum information without changing the probabilities of
    /// measuring the system in the computational basis.
    ///
    /// # Arguments
    /// * `qubit` - Target qubit
    /// * `lambda` - Damping parameter (0.0 to 1.0)
    ///
    /// # Returns
    /// * `&mut Self` - Returns self for method chaining
    #[inline]
    pub fn apply_phase_damping(&mut self, qubit: usize, lambda: f64) -> &mut Self {
        let lambda = lambda.clamp(0.0, 1.0);
        if lambda < f64::EPSILON {
            return self;
        }

        // Phase damping via Kraus ops
        //   E_0 = |0><0| + sqrt(1-l)|1><1|,   E_1 = sqrt(l)|1><1|
        // gives
        //   rho_{a,b} unchanged when bit_q(a) == bit_q(b)
        //   rho_{a,b} -> sqrt(1-l) * rho_{a,b} when they differ
        // (the two Kraus contributions sum so the diag is preserved).
        //
        // Apply on the density matrix, then Cholesky-re-purify so
        // `probability()` / `purity()` stay consistent with the Choi state.
        let n = self.num_physical_qubits;
        let dim = 1usize << n;
        let qubit_mask = 1usize << qubit;

        let rho = self.get_density_matrix();
        let mut new_rho = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];
        let sqrt_1ml = (1.0 - lambda).sqrt();
        for i in 0..dim {
            let i1 = (i & qubit_mask) != 0;
            for j in 0..dim {
                let j1 = (j & qubit_mask) != 0;
                new_rho[i][j] = if i1 == j1 {
                    rho[i][j]
                } else {
                    sqrt_1ml * rho[i][j]
                };
            }
        }
        self.set_from_density_matrix(&new_rho);

        self
    }

    /// Apply a bit flip noise channel to a qubit
    ///
    /// The bit flip channel flips the qubit from |0⟩ to |1⟩ or from |1⟩ to |0⟩
    /// with probability p.
    ///
    /// # Arguments
    /// * `qubit` - Target qubit
    /// * `probability` - Probability of a bit flip (0.0 to 1.0)
    ///
    /// # Returns
    /// * `&mut Self` - Returns self for method chaining
    #[inline]
    pub fn apply_bit_flip(&mut self, qubit: usize, probability: f64) -> &mut Self {
        // Ensure probability is in valid range
        let p = probability.clamp(0.0, 1.0);

        if p < f64::EPSILON {
            // No noise, return unchanged
            return self;
        }

        // Bit flip channel: rho -> (1-p) rho + p X rho X
        //
        // For density matrix elements:
        // rho_{i,j} -> (1-p) rho_{i,j} + p rho_{i^q, j^q}
        // where i^q means i with bit q flipped

        let n = self.num_physical_qubits;
        let dim = 1 << n;
        let qubit_mask = 1 << qubit;

        // Get current density matrix
        let rho = self.get_density_matrix();

        // Apply bit flip transformation
        let mut new_rho = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];

        for i in 0..dim {
            for j in 0..dim {
                let i_flipped = i ^ qubit_mask;
                let j_flipped = j ^ qubit_mask;
                new_rho[i][j] = (1.0 - p) * rho[i][j] + p * rho[i_flipped][j_flipped];
            }
        }

        // Convert back to Choi representation
        self.set_from_density_matrix(&new_rho);

        self
    }

    /// Apply a phase flip noise channel to a qubit
    ///
    /// The phase flip channel applies a Z operation on the qubit
    /// with probability p, introducing a relative phase of -1 between |0⟩ and |1⟩.
    ///
    /// # Arguments
    /// * `qubit` - Target qubit
    /// * `probability` - Probability of a phase flip (0.0 to 1.0)
    ///
    /// # Returns
    /// * `&mut Self` - Returns self for method chaining
    #[inline]
    pub fn apply_phase_flip(&mut self, qubit: usize, probability: f64) -> &mut Self {
        // Ensure probability is in valid range
        let p = probability.clamp(0.0, 1.0);

        if p < f64::EPSILON {
            // No noise, return unchanged
            return self;
        }

        // Phase flip channel: rho -> (1-p) rho + p Z rho Z
        //
        // For density matrix elements:
        // - If bit q of i = bit q of j: rho_{i,j} unchanged
        // - If bit q of i != bit q of j: rho_{i,j} -> (1-2p) rho_{i,j}

        let n = self.num_physical_qubits;
        let dim = 1 << n;
        let qubit_mask = 1 << qubit;

        // Get current density matrix
        let rho = self.get_density_matrix();

        // Apply phase flip transformation
        let mut new_rho = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];

        for i in 0..dim {
            for j in 0..dim {
                let i_bit = (i & qubit_mask) != 0;
                let j_bit = (j & qubit_mask) != 0;

                if i_bit == j_bit {
                    // Diagonal in qubit q: unchanged
                    new_rho[i][j] = rho[i][j];
                } else {
                    // Off-diagonal in qubit q: scale by (1-2p)
                    new_rho[i][j] = (1.0 - 2.0 * p) * rho[i][j];
                }
            }
        }

        // Convert back to Choi representation
        self.set_from_density_matrix(&new_rho);

        self
    }

    /// Apply a symbolic channel expression to this density matrix.
    ///
    /// Supported expressions are those convertible to same-Hilbert-space Kraus
    /// operators: unitary, mixed-unitary, amplitude damping, phase damping,
    /// tensor, and composition. Erasure, leakage, and gate instruments are
    /// intentionally rejected because they need extra flag/outcome semantics.
    ///
    /// # Errors
    ///
    /// Returns an error if the channel expression is unsupported or invalid for
    /// this simulator's qubit count.
    pub fn apply_channel_expr(&mut self, channel: &ChannelExpr) -> Result<&mut Self, ChannelError> {
        let kraus = KrausOps::from_channel_expr_with_num_qubits(channel, self.num_physical_qubits)?;
        self.apply_kraus_ops(&kraus)
    }

    /// Apply Kraus operators to this density matrix.
    ///
    /// # Errors
    ///
    /// Returns an error if the Kraus operators are not defined on the same
    /// number of qubits as this density matrix.
    pub fn apply_kraus_ops(&mut self, kraus: &KrausOps) -> Result<&mut Self, ChannelError> {
        let num_qubits_u32 = u32::try_from(self.num_physical_qubits).map_err(|_| {
            ChannelError::DimensionOverflow {
                num_qubits: self.num_physical_qubits,
            }
        })?;
        let dim = 1usize
            .checked_shl(num_qubits_u32)
            .ok_or(ChannelError::DimensionOverflow {
                num_qubits: self.num_physical_qubits,
            })?;
        if kraus.num_qubits() != self.num_physical_qubits {
            let actual_qubits_u32 =
                u32::try_from(kraus.num_qubits()).map_err(|_| ChannelError::DimensionOverflow {
                    num_qubits: kraus.num_qubits(),
                })?;
            let actual_dim =
                1usize
                    .checked_shl(actual_qubits_u32)
                    .ok_or(ChannelError::DimensionOverflow {
                        num_qubits: kraus.num_qubits(),
                    })?;
            return Err(ChannelError::InvalidMatrixShape {
                expected_rows: dim,
                expected_cols: dim,
                rows: actual_dim,
                cols: actual_dim,
            });
        }

        let rho = self.get_density_matrix();
        let flat: Vec<Complex64> = rho.iter().flat_map(|row| row.iter().copied()).collect();
        let rho_matrix = DMatrix::from_row_slice(dim, dim, &flat);
        let mut evolved = DMatrix::zeros(dim, dim);
        for operator in kraus.operators() {
            evolved += operator * &rho_matrix * operator.adjoint();
        }

        let new_rho: Vec<Vec<Complex64>> = (0..dim)
            .map(|row| (0..dim).map(|col| evolved[(row, col)]).collect())
            .collect();
        self.set_from_density_matrix(&new_rho);
        Ok(self)
    }
}

impl<R> From<&StateVecSoA<R>> for DensityMatrix<R>
where
    R: Rng + SeedableRng + Debug + Clone,
{
    fn from(state: &StateVecSoA<R>) -> Self {
        let mut state = state.clone();
        let amplitudes = state.state();
        let dim = amplitudes.len();
        let num_physical_qubits = dim.trailing_zeros() as usize;
        let mut purification = vec![Complex64::new(0.0, 0.0); dim * dim];

        for (row, amplitude) in amplitudes.iter().enumerate() {
            purification[row << num_physical_qubits] = *amplitude;
        }

        Self {
            num_physical_qubits,
            state_vector: StateVec::from_state(&purification, state.rng().clone()),
        }
    }
}

impl<R> TryFrom<&DensityMatrix<R>> for Vec<Complex64>
where
    R: Rng + SeedableRng + Debug + Clone,
{
    type Error = StateConversionError;

    fn try_from(density_matrix: &DensityMatrix<R>) -> Result<Self, Self::Error> {
        let mut density_matrix = density_matrix.clone();
        let rho = density_matrix.get_density_matrix();
        pure_state_from_density_matrix(&rho)
    }
}

fn pure_state_from_density_matrix(
    rho: &[Vec<Complex64>],
) -> Result<Vec<Complex64>, StateConversionError> {
    let dim = rho.len();
    let (pivot, pivot_probability) = rho
        .iter()
        .enumerate()
        .map(|(i, row)| (i, row[i].re.max(0.0)))
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .unwrap_or((0, 0.0));

    if pivot_probability <= PURE_STATE_TOLERANCE {
        return Err(StateConversionError::MixedDensityMatrix { residual: 1.0 });
    }

    let pivot_amplitude = pivot_probability.sqrt();
    let state: Vec<Complex64> = (0..dim)
        .map(|row| rho[row][pivot] / pivot_amplitude)
        .collect();

    let mut residual = 0.0_f64;
    for row in 0..dim {
        for col in 0..dim {
            let reconstructed = state[row] * state[col].conj();
            residual = residual.max((rho[row][col] - reconstructed).norm());
        }
    }

    if residual > PURE_STATE_TOLERANCE {
        return Err(StateConversionError::MixedDensityMatrix { residual });
    }

    Ok(state)
}

impl<R> Display for DensityMatrix<R>
where
    R: Rng + SeedableRng + Debug + Clone,
{
    /// Formats the density matrix using default formatting parameters.
    ///
    /// This implementation uses 4 decimal places and a threshold of 1e-10.
    ///
    /// # Examples
    /// ```rust
    /// use pecos_core::QubitId;
    /// use pecos_simulators::{DensityMatrix, CliffordGateable, qid};
    ///
    /// // Create a Bell state
    /// let mut state = DensityMatrix::new(2);
    /// state.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]);
    ///
    /// // Print the density matrix with default formatting
    /// println!("{}", state);
    /// ```
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.density_matrix_to_string_no_flush(4, 1e-10))
    }
}

impl<R> QuantumSimulator for DensityMatrix<R>
where
    R: Rng + SeedableRng + Debug + Clone,
{
    fn num_qubits(&self) -> usize {
        self.num_physical_qubits
    }

    /// Reset the quantum state to |0...0⟩⟨0...0|
    ///
    /// # Returns
    /// * `&mut Self` - Returns self for method chaining
    #[inline]
    fn reset(&mut self) -> &mut Self {
        self.prepare_computational_basis(0)
    }
}

impl<R> RngManageable for DensityMatrix<R>
where
    R: Rng + SeedableRng + Debug + Clone,
{
    type Rng = R;

    /// Replace the random number generator with a new one
    ///
    /// # Arguments
    /// * `rng` - New random number generator
    #[inline]
    fn set_rng(&mut self, rng: R) {
        self.state_vector.set_rng(rng);
    }

    /// Get a reference to the random number generator
    ///
    /// # Returns
    /// * `&Self::Rng` - Reference to the RNG
    #[inline]
    fn rng(&self) -> &Self::Rng {
        self.state_vector.rng()
    }

    /// Get a mutable reference to the random number generator
    ///
    /// # Returns
    /// * `&mut Self::Rng` - Mutable reference to the RNG
    #[inline]
    fn rng_mut(&mut self) -> &mut Self::Rng {
        self.state_vector.rng_mut()
    }
}

impl<R> CliffordGateable for DensityMatrix<R>
where
    R: Rng + SeedableRng + Debug + Clone,
{
    /// Apply the Hadamard gate to the given qubits
    ///
    /// # Arguments
    /// * `qubits` - Target qubits
    ///
    /// # Returns
    /// * `&mut Self` - Returns self for method chaining
    #[inline]
    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        let n = self.num_physical_qubits;

        for &q in qubits {
            let qubit = q.index();
            // Apply H to the system qubit
            self.state_vector_mut().h(&[QubitId(qubit)]);

            // Apply H* (= H since H is Hermitian) to the environment qubit
            self.state_vector_mut().h(&[QubitId(qubit + n)]);
        }

        self
    }

    /// Apply the S gate to the given qubits
    ///
    /// # Arguments
    /// * `qubits` - Target qubits
    ///
    /// # Returns
    /// * `&mut Self` - Returns self for method chaining
    #[inline]
    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        let n = self.num_physical_qubits;

        for &q in qubits {
            let qubit = q.index();
            // Apply S to the system qubit
            self.state_vector_mut().sz(&[QubitId(qubit)]);

            // For the environment qubit, we need S* which is S dagger
            // S dagger is the inverse of S, which is implemented as szdg in the state vector
            self.state_vector_mut().szdg(&[QubitId(qubit + n)]);
        }

        self
    }

    /// Apply the controlled-X (CNOT) gate
    ///
    /// # Arguments
    /// * `qubits` - Pairs of (control, target) qubits
    ///
    /// # Returns
    /// * `&mut Self` - Returns self for method chaining
    #[inline]
    fn cx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let n = self.num_physical_qubits;

        for &(control, target) in pairs {
            let control = control.index();
            let target = target.index();

            // Apply CX to the system qubits
            self.state_vector_mut()
                .cx(&[(QubitId(control), QubitId(target))]);

            // Apply CX* to the environment qubits
            // CX is real so CX* = CX
            self.state_vector_mut()
                .cx(&[(QubitId(control + n), QubitId(target + n))]);
        }

        self
    }

    /// Measure qubits in the Z basis and collapse the state
    ///
    /// # Arguments
    /// * `qubits` - The qubits to measure
    ///
    /// # Returns
    /// * `Vec<MeasurementResult>` - Contains the outcome and whether it was deterministic for each qubit
    #[inline]
    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        let mut results = Vec::with_capacity(qubits.len());

        for &q in qubits {
            let qubit = q.index();
            // First calculate the probabilities of measuring 0 and 1
            let n = self.num_physical_qubits;
            let mut prob_one = 0.0;

            // Calculate probability of measuring 1
            for i in 0..(1 << n) {
                if (i & (1 << qubit)) != 0 {
                    // This is a state where qubit is 1
                    prob_one += self.probability(i);
                }
            }

            // Determine if measurement is deterministic
            let is_deterministic = !(1e-10..=1.0 - 1e-10).contains(&prob_one);

            // Determine outcome
            let outcome = if is_deterministic {
                prob_one > 0.5
            } else {
                self.state_vector.rng_mut().random_range(0.0..1.0) < prob_one
            };

            // Apply the measurement projection: rho -> P_m rho P_m / Tr(P_m rho P_m)
            // In the Choi representation, index (row << n) | col corresponds to rho_{row,col}
            // The projector P_m zeros out rows/cols where the measured qubit doesn't match outcome
            let qubit_mask = 1 << qubit;
            let target_bit = if outcome { qubit_mask } else { 0 };

            let sv = self.state_vector.state();
            let sv_size = 1 << (2 * n);

            // Create new state with projected amplitudes
            let mut new_state = vec![Complex64::new(0.0, 0.0); sv_size];
            let mut norm_sq = 0.0;

            for idx in 0..sv_size {
                let row = idx >> n;
                let col = idx & ((1 << n) - 1);

                // Check if both row and column have the correct qubit value
                let row_matches = (row & qubit_mask) == target_bit;
                let col_matches = (col & qubit_mask) == target_bit;

                if row_matches && col_matches {
                    new_state[idx] = sv[idx];
                    norm_sq += sv[idx].norm_sqr();
                }
            }

            // Renormalize the state
            if norm_sq > 1e-15 {
                let norm = norm_sq.sqrt();
                for amplitude in &mut new_state {
                    *amplitude /= norm;
                }
            }

            // Update the state vector
            let new_sv = StateVec::from_state(&new_state, self.state_vector.rng().clone());
            *self.state_vector_mut() = new_sv;

            results.push(MeasurementResult {
                outcome,
                is_deterministic,
            });
        }

        results
    }
}

impl<R> ArbitraryRotationGateable for DensityMatrix<R>
where
    R: Rng + SeedableRng + Debug + Clone,
{
    /// Apply a rotation around the X-axis
    ///
    /// # Arguments
    /// * `theta` - Rotation angle
    /// * `qubits` - Target qubits
    ///
    /// # Returns
    /// * `&mut Self` - Returns self for method chaining
    #[inline]
    fn rx(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let n = self.num_physical_qubits;

        for &q in qubits {
            let qubit = q.index();
            let sys_qubits = [QubitId(qubit)];
            let env_qubits = [QubitId(qubit + n)];

            // Apply RX to the system qubit
            self.state_vector_mut().rx(theta, &sys_qubits);

            // Apply RX* to the environment qubit
            // RX(-theta) = Z * RX(theta) * Z
            self.state_vector_mut().z(&env_qubits);
            self.state_vector_mut().rx(theta, &env_qubits);
            self.state_vector_mut().z(&env_qubits);
        }

        self
    }

    /// Apply a rotation around the Y-axis
    ///
    /// # Arguments
    /// * `theta` - Rotation angle
    /// * `qubits` - Target qubits
    ///
    /// # Returns
    /// * `&mut Self` - Returns self for method chaining
    #[inline]
    fn ry(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let n = self.num_physical_qubits;

        for &q in qubits {
            let qubit = q.index();
            let sys_qubits = [QubitId(qubit)];
            let env_qubits = [QubitId(qubit + n)];

            // Apply RY to the system qubit
            self.state_vector_mut().ry(theta, &sys_qubits);

            // Apply RY* to the environment qubit
            // RY is a real matrix, so RY* = RY
            self.state_vector_mut().ry(theta, &env_qubits);
        }

        self
    }

    /// Apply a rotation around the Z-axis
    ///
    /// # Arguments
    /// * `theta` - Rotation angle
    /// * `qubits` - Target qubits
    ///
    /// # Returns
    /// * `&mut Self` - Returns self for method chaining
    #[inline]
    fn rz(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        let n = self.num_physical_qubits;

        for &q in qubits {
            let qubit = q.index();
            let sys_qubits = [QubitId(qubit)];
            let env_qubits = [QubitId(qubit + n)];

            // Apply RZ to the system qubit
            self.state_vector_mut().rz(theta, &sys_qubits);

            // Apply RZ* to the environment qubit
            // RZ(-theta) = X * RZ(theta) * X
            self.state_vector_mut().x(&env_qubits);
            self.state_vector_mut().rz(theta, &env_qubits);
            self.state_vector_mut().x(&env_qubits);
        }

        self
    }

    /// Apply a two-qubit ZZ rotation
    ///
    /// # Arguments
    /// * `theta` - Rotation angle
    /// * `qubits` - Pairs of qubits
    ///
    /// # Returns
    /// * `&mut Self` - Returns self for method chaining
    #[inline]
    fn rzz(&mut self, theta: Angle64, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let n = self.num_physical_qubits;

        for &(q1, q2) in pairs {
            let q1 = q1.index();
            let q2 = q2.index();
            let sys_pairs = [(QubitId(q1), QubitId(q2))];
            let env_pairs = [(QubitId(q1 + n), QubitId(q2 + n))];

            // Apply RZZ to the system qubits
            self.state_vector_mut().rzz(theta, &sys_pairs);

            // Apply RZZ* to the environment qubits
            // RZZ(-theta) = (X tensor I) * RZZ(theta) * (X tensor I)
            self.state_vector_mut().x(&[env_pairs[0].0]);
            self.state_vector_mut().rzz(theta, &env_pairs);
            self.state_vector_mut().x(&[env_pairs[0].0]);
        }

        self
    }
}

impl crate::density_matrix_test_utils::DensityMatrixSimulator for DensityMatrix {
    fn with_seed(num_qubits: usize, seed: u64) -> Self {
        DensityMatrix::with_seed(num_qubits, seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::{QubitId, qid};

    #[test]
    fn test_new_density_matrix() {
        // Create a new 1-qubit density matrix
        let mut dm = DensityMatrix::new(1);

        // Check that it represents |0⟩⟨0|
        assert!((dm.probability(0) - 1.0).abs() < 1e-10);
        assert!(dm.probability(1) < 1e-10);

        // Check that it's a pure state
        assert!(dm.is_pure());
    }

    #[test]
    fn test_prepare_computational_basis() {
        // Test preparing different computational basis states
        let mut dm = DensityMatrix::new(2);

        // Prepare |01⟩⟨01|
        dm.prepare_computational_basis(1);
        assert!((dm.probability(1) - 1.0).abs() < 1e-10);
        assert!(dm.probability(0) < 1e-10);
        assert!(dm.probability(2) < 1e-10);
        assert!(dm.probability(3) < 1e-10);

        // Prepare |10⟩⟨10|
        dm.prepare_computational_basis(2);
        assert!((dm.probability(2) - 1.0).abs() < 1e-10);
        assert!(dm.probability(0) < 1e-10);
        assert!(dm.probability(1) < 1e-10);
        assert!(dm.probability(3) < 1e-10);
    }

    #[test]
    fn test_reset() {
        // Test that reset returns to |0...0⟩⟨0...0|
        let mut dm = DensityMatrix::new(2);

        // Prepare a different state
        dm.prepare_computational_basis(3);

        // Reset
        dm.reset();

        // Check state is |00⟩⟨00|
        assert!((dm.probability(0) - 1.0).abs() < 1e-10);
        assert!(dm.probability(1) < 1e-10);
        assert!(dm.probability(2) < 1e-10);
        assert!(dm.probability(3) < 1e-10);
    }

    #[test]
    fn test_x_gate() {
        // Test X gate on computational basis state
        let mut dm = DensityMatrix::new(1);

        // Apply X to |0⟩⟨0|
        dm.x(&qid(0));

        // Check state is |1⟩⟨1|
        assert!(dm.probability(0) < 1e-10);
        assert!((dm.probability(1) - 1.0).abs() < 1e-10);

        // Apply X again to return to |0⟩⟨0|
        dm.x(&qid(0));

        // Check state is |0⟩⟨0|
        assert!((dm.probability(0) - 1.0).abs() < 1e-10);
        assert!(dm.probability(1) < 1e-10);
    }

    #[test]
    fn test_h_gate() {
        // Test H gate creating superposition
        let mut dm = DensityMatrix::new(1);

        // Apply H to |0⟩⟨0|
        dm.h(&qid(0));

        // Check probabilities are 0.5 for both outcomes
        assert!((dm.probability(0) - 0.5).abs() < 1e-10);
        assert!((dm.probability(1) - 0.5).abs() < 1e-10);

        // Apply H again to return to |0⟩⟨0|
        dm.h(&qid(0));

        // Check state is |0⟩⟨0|
        assert!((dm.probability(0) - 1.0).abs() < 1e-10);
        assert!(dm.probability(1) < 1e-10);
    }

    #[test]
    fn test_bell_state() {
        // Test creating a Bell state
        let mut dm = DensityMatrix::new(2);

        // Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        dm.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]);

        // Check probabilities
        assert!((dm.probability(0) - 0.5).abs() < 1e-10);
        assert!(dm.probability(1) < 1e-10);
        assert!(dm.probability(2) < 1e-10);
        assert!((dm.probability(3) - 0.5).abs() < 1e-10);

        // State should be pure
        assert!(dm.is_pure());
    }

    #[test]
    fn channel_expr_bit_flip_applies_to_density_matrix() {
        let mut dm = DensityMatrix::new(1);
        dm.apply_channel_expr(&pecos_core::channel::BitFlip(1.0, 0))
            .unwrap();

        assert!(dm.probability(0) < 1e-10);
        assert!((dm.probability(1) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn channel_expr_embeds_local_channel_in_larger_density_matrix() {
        let mut dm = DensityMatrix::new(2);
        dm.apply_channel_expr(&pecos_core::channel::BitFlip(1.0, 1))
            .unwrap();

        assert!(dm.probability(0) < 1e-10);
        assert!((dm.probability(2) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn tensor_channel_expr_applies_to_noncontiguous_density_matrix_qubits() {
        let mut dm = DensityMatrix::new(3);
        let channel = ChannelExpr::Tensor(vec![
            pecos_core::channel::BitFlip(1.0, 0),
            pecos_core::channel::BitFlip(1.0, 2),
        ]);

        dm.apply_channel_expr(&channel).unwrap();

        assert!(dm.probability(0) < 1e-10);
        assert!((dm.probability(5) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn out_of_range_channel_expr_is_rejected_without_mutating_state() {
        let mut dm = DensityMatrix::new(2);
        dm.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]);
        let before = dm.get_flattened_density_matrix();

        let err = dm
            .apply_channel_expr(&pecos_core::channel::BitFlip(0.5, 2))
            .expect_err("channel should not apply outside the simulator range");

        assert!(matches!(
            err,
            ChannelError::QubitOutOfRange {
                num_qubits: 2,
                qubit: 2
            }
        ));
        let after = dm.get_flattened_density_matrix();
        assert_eq!(after, before);
    }

    #[test]
    fn state_vector_converts_to_density_matrix_and_back() {
        let mut state = StateVecSoA::new(2);
        state.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]);

        let mut density_matrix = DensityMatrix::from(&state);
        assert!((density_matrix.probability(0) - 0.5).abs() < 1e-10);
        assert!(density_matrix.probability(1) < 1e-10);
        assert!(density_matrix.probability(2) < 1e-10);
        assert!((density_matrix.probability(3) - 0.5).abs() < 1e-10);

        let recovered = Vec::<Complex64>::try_from(&density_matrix).unwrap();
        let expected = state.state();

        for (actual, expected) in recovered.iter().zip(expected.iter()) {
            assert!((*actual - *expected).norm() < 1e-10);
        }
    }

    #[test]
    fn mixed_density_matrix_rejects_state_vector_conversion() {
        let mut density_matrix = DensityMatrix::new(1);
        density_matrix.prepare_maximally_mixed();

        let err = Vec::<Complex64>::try_from(&density_matrix).unwrap_err();
        assert!(matches!(
            err,
            StateConversionError::MixedDensityMatrix { .. }
        ));
    }

    // Additional tests for other gates and operations would be added here
}
