use num_complex::Complex64;
use rand::Rng;

#[derive(Clone, Debug)]
pub struct StateVec {
    num_qubits: usize,
    state: Vec<Complex64>,
}

impl StateVec {
    /// Create a new state initialized to |0...0⟩
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        let size = 1 << num_qubits; // 2^n
        let mut state = vec![Complex64::new(0.0, 0.0); size];
        state[0] = Complex64::new(1.0, 0.0); // Prep |0...0>
        StateVec { num_qubits, state }
    }

    /// Initialize from a custom state vector
    ///
    /// # Panics
    ///
    /// Panics if the input state requires more qubits then `StateVec` has.
    #[must_use]
    pub fn from_state(state: Vec<Complex64>) -> Self {
        let num_qubits = state.len().trailing_zeros() as usize;
        assert_eq!(1 << num_qubits, state.len(), "Invalid state vector size");
        StateVec { num_qubits, state }
    }

    /// Prepare a specific computational basis state
    ///
    /// # Panics
    ///
    /// Panics if `basis_state` >= `2^num_qubits` (i.e., if the basis state index is too large for the number of qubits)
    pub fn prepare_computational_basis(&mut self, basis_state: usize) {
        assert!(basis_state < 1 << self.num_qubits);
        self.state.fill(Complex64::new(0.0, 0.0));
        self.state[basis_state] = Complex64::new(1.0, 0.0);
    }

    /// Prepare all qubits in |+⟩ state
    pub fn prepare_plus_state(&mut self) {
        let factor = Complex64::new(1.0 / f64::from(1 << self.num_qubits), 0.0);
        self.state.fill(factor);
    }

    /// Returns the number of qubits in the system
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Returns reference to the state vector
    #[must_use]
    pub fn state(&self) -> &[Complex64] {
        &self.state
    }

    /// Returns the probability of measuring a specific basis state
    ///
    /// # Panics
    ///
    /// Panics if `basis_state` >= `2^num_qubits` (i.e., if the basis state index is too large for the number of qubits)
    #[must_use]
    pub fn probability(&self, basis_state: usize) -> f64 {
        assert!(basis_state < 1 << self.num_qubits);
        self.state[basis_state].norm_sqr()
    }

    /// Apply Hadamard gate to the target qubit
    ///
    /// # Panics
    ///
    /// Panics if target qubit index is >= number of qubits
    pub fn hadamard(&mut self, target: usize) {
        assert!(target < self.num_qubits);
        let factor = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
        let step = 1 << target;

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

    /// Apply Pauli-X gate
    ///
    /// # Panics
    ///
    /// Panics if target qubit index is >= number of qubits
    pub fn x(&mut self, target: usize) {
        assert!(target < self.num_qubits);
        let step = 1 << target;

        for i in (0..self.state.len()).step_by(2 * step) {
            for offset in 0..step {
                self.state.swap(i + offset, i + offset + step);
            }
        }
    }

    /// Apply Y = [[0, -i], [i, 0]] gate to target qubit
    ///
    /// # Panics
    ///
    /// Panics if target qubit index is >= number of qubits
    pub fn y(&mut self, target: usize) {
        assert!(target < self.num_qubits);

        for i in 0..self.state.len() {
            if (i >> target) & 1 == 0 {
                let flipped_i = i ^ (1 << target);
                let temp = self.state[i];
                self.state[i] = -Complex64::i() * self.state[flipped_i];
                self.state[flipped_i] = Complex64::i() * temp;
            }
        }
    }

    /// Apply Z = [[1, 0], [0, -1]] gate to target qubit
    ///
    /// # Panics
    ///
    /// Panics if target qubit index is >= number of qubits
    pub fn z(&mut self, target: usize) {
        assert!(target < self.num_qubits);

        for i in 0..self.state.len() {
            if (i >> target) & 1 == 1 {
                self.state[i] = -self.state[i];
            }
        }
    }

    /// Gate RX(θ) = exp(-i θ X/2) = cos(θ/2) I - i*sin(θ/2) X
    /// RX(θ) = [[cos(θ/2), -i*sin(θ/2)],
    ///          [-i*sin(θ/2), cos(θ/2)]]
    ///
    /// # Panics
    ///
    /// Panics if target qubit index is >= number of qubits
    pub fn rx(&mut self, theta: f64, target: usize) {
        let cos = (theta / 2.0).cos();
        let sin = (theta / 2.0).sin();
        let neg_i_sin = Complex64::new(0.0, -sin);

        self.single_qubit_rotation(
            target,
            Complex64::new(cos, 0.0), // u00
            neg_i_sin,                // u01
            neg_i_sin,                // u10
            Complex64::new(cos, 0.0), // u11
        );
    }

    /// Gate RY(θ) = exp(-i θ Y/2) = cos(θ/2) I - i*sin(θ/2) Y
    /// RY(θ) = [[cos(θ/2), -sin(θ/2)],
    ///          [-sin(θ/2), cos(θ/2)]]
    ///
    /// # Panics
    ///
    /// Panics if target qubit index is >= number of qubits
    pub fn ry(&mut self, theta: f64, target: usize) {
        let cos = (theta / 2.0).cos();
        let sin = (theta / 2.0).sin();

        self.single_qubit_rotation(
            target,
            Complex64::new(cos, 0.0),  // u00
            Complex64::new(-sin, 0.0), // u01
            Complex64::new(sin, 0.0),  // u10
            Complex64::new(cos, 0.0),  // u11
        );
    }

    /// Gate RZ(θ) = exp(-i θ Z/2) = cos(θ/2) I - i*sin(θ/2) Z
    /// RZ(θ) = [[cos(θ/2)-i*sin(θ/2), 0],
    ///          [0, cos(θ/2)+i*sin(θ/2)]]
    ///
    /// # Panics
    ///
    /// Panics if target qubit index is >= number of qubits
    pub fn rz(&mut self, theta: f64, target: usize) {
        let exp_minus_i_theta_2 = Complex64::from_polar(1.0, -theta / 2.0);
        let exp_plus_i_theta_2 = Complex64::from_polar(1.0, theta / 2.0);

        self.single_qubit_rotation(
            target,
            exp_minus_i_theta_2,      // u00
            Complex64::new(0.0, 0.0), // u01
            Complex64::new(0.0, 0.0), // u10
            exp_plus_i_theta_2,       // u11
        );
    }

    /// Apply a general single-qubit unitary gate
    ///
    /// # Examples
    /// ```
    /// use pecos_qsim::state_vec::StateVec;
    /// use std::f64::consts::FRAC_1_SQRT_2;
    /// use num_complex::Complex64;
    /// let mut q = StateVec::new(1);
    /// // Apply Hadamard gate
    /// q.single_qubit_rotation(0,
    ///     Complex64::new(FRAC_1_SQRT_2, 0.0),  // u00
    ///     Complex64::new(FRAC_1_SQRT_2, 0.0),  // u01
    ///     Complex64::new(FRAC_1_SQRT_2, 0.0),  // u10
    ///     Complex64::new(-FRAC_1_SQRT_2, 0.0), // u11
    /// );
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if target qubit index is >= number of qubits
    pub fn single_qubit_rotation(
        &mut self,
        target: usize,
        u00: Complex64,
        u01: Complex64,
        u10: Complex64,
        u11: Complex64,
    ) {
        assert!(target < self.num_qubits);

        let step = 1 << target;
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
    }

    /// Apply a SWAP gate between two qubits
    ///
    /// # Panics
    ///
    /// Panics if target qubit1 or qubit2 index is >= number of qubits
    pub fn swap(&mut self, qubit1: usize, qubit2: usize) {
        assert!(qubit1 < self.num_qubits && qubit2 < self.num_qubits);
        if qubit1 == qubit2 {
            return; // No-op if qubits are the same
        }

        let step1 = 1 << qubit1;
        let step2 = 1 << qubit2;

        for i in 0..self.state.len() {
            let bit1 = (i >> qubit1) & 1;
            let bit2 = (i >> qubit2) & 1;

            if bit1 != bit2 {
                let swapped_index = i ^ step1 ^ step2;
                if i < swapped_index {
                    self.state.swap(i, swapped_index);
                }
            }
        }
    }

    /// Apply U3(theta, phi, lambda) gate
    /// U3 = [[cos(θ/2), -e^(iλ)sin(θ/2)],
    ///       [e^(iφ)sin(θ/2), e^(i(λ+φ))cos(θ/2)]]
    ///
    /// # Panics
    ///
    /// Panics if target qubit index is >= number of qubits
    pub fn u3(&mut self, target: usize, theta: f64, phi: f64, lambda: f64) {
        assert!(target < self.num_qubits);

        let cos = (theta / 2.0).cos();
        let sin = (theta / 2.0).sin();

        // Calculate matrix elements
        let u00 = Complex64::new(cos, 0.0);
        let u01 = -Complex64::from_polar(sin, lambda);
        let u10 = Complex64::from_polar(sin, phi);
        let u11 = Complex64::from_polar(cos, phi + lambda);

        // Apply the unitary
        for i in 0..self.state.len() {
            if (i >> target) & 1 == 0 {
                let i1 = i ^ (1 << target);
                let a0 = self.state[i];
                let a1 = self.state[i1];

                self.state[i] = u00 * a0 + u01 * a1;
                self.state[i1] = u10 * a0 + u11 * a1;
            }
        }
    }

    /// Apply controlled-X gate
    /// CX = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ X
    ///
    /// # Panics
    ///
    /// Panics if target qubit1 or qubit2 index is >= number of qubits or qubit1 == qubit2
    pub fn cx(&mut self, control: usize, target: usize) {
        assert!(control < self.num_qubits);
        assert!(target < self.num_qubits);
        assert_ne!(control, target);

        for i in 0..self.state.len() {
            let control_val = (i >> control) & 1;
            let target_val = (i >> target) & 1;
            if control_val == 1 && target_val == 0 {
                let flipped_i = i ^ (1 << target);
                self.state.swap(i, flipped_i);
            }
        }
    }

    /// Apply controlled-Y gate
    /// CY = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ Y
    ///
    /// # Panics
    ///
    /// Panics if target qubit1 or qubit2 index is >= number of qubits or qubit1 == qubit2
    pub fn cy(&mut self, control: usize, target: usize) {
        assert!(control < self.num_qubits);
        assert!(target < self.num_qubits);
        assert_ne!(control, target);

        // Only process when control bit is 1 and target bit is 0
        for i in 0..self.state.len() {
            let control_val = (i >> control) & 1;
            let target_val = (i >> target) & 1;

            if control_val == 1 && target_val == 0 {
                let flipped_i = i ^ (1 << target);

                // Y gate has different phases than X
                // Y = [[0, -i], [i, 0]]
                let temp = self.state[i];
                self.state[i] = -Complex64::i() * self.state[flipped_i];
                self.state[flipped_i] = Complex64::i() * temp;
            }
        }
    }

    /// Apply controlled-Z gate
    /// CZ = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ Z
    ///
    /// # Panics
    ///
    /// Panics if target qubit1 or qubit2 index is >= number of qubits or qubit1 == qubit2
    pub fn cz(&mut self, control: usize, target: usize) {
        assert!(control < self.num_qubits);
        assert!(target < self.num_qubits);
        assert_ne!(control, target);

        // CZ is simpler - just add phase when both control and target are 1
        for i in 0..self.state.len() {
            let control_val = (i >> control) & 1;
            let target_val = (i >> target) & 1;

            if control_val == 1 && target_val == 1 {
                self.state[i] = -self.state[i];
            }
        }
    }

    /// Apply RXX(θ) = exp(-i θ XX/2) gate
    /// This implements evolution under the XX coupling between two qubits
    ///
    /// # Panics
    ///
    /// Panics if target qubit1 or qubit2 index is >= number of qubits or qubit1 == qubit2
    pub fn rxx(&mut self, theta: f64, qubit1: usize, qubit2: usize) {
        assert!(qubit1 < self.num_qubits);
        assert!(qubit2 < self.num_qubits);
        assert_ne!(qubit1, qubit2);

        let cos = (theta / 2.0).cos();
        let sin = (theta / 2.0).sin();
        let neg_i_sin = Complex64::new(0.0, -sin); // -i*sin

        // Make sure qubit1 < qubit2 for consistent ordering
        let (q1, q2) = if qubit1 < qubit2 {
            (qubit1, qubit2)
        } else {
            (qubit2, qubit1)
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

    /// Apply RYY(θ) = exp(-i θ YY/2) gate
    ///
    /// # Panics
    ///
    /// Panics if target qubit1 or qubit2 index is >= number of qubits or qubit1 == qubit2
    pub fn ryy(&mut self, theta: f64, qubit1: usize, qubit2: usize) {
        assert!(qubit1 < self.num_qubits);
        assert!(qubit2 < self.num_qubits);
        assert_ne!(qubit1, qubit2);

        let cos = (theta / 2.0).cos();
        let sin = (theta / 2.0).sin();
        let neg_i_sin = Complex64::new(0.0, -sin);

        let (q1, q2) = if qubit1 < qubit2 {
            (qubit1, qubit2)
        } else {
            (qubit2, qubit1)
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

                // YY has an extra minus sign compared to XX when acting on |01⟩ and |10⟩
                self.state[i] = cos * a00 + neg_i_sin * a11;
                self.state[i01] = cos * a01 - neg_i_sin * a10;
                self.state[i10] = cos * a10 - neg_i_sin * a01;
                self.state[i11] = cos * a11 + neg_i_sin * a00;
            }
        }
    }

    /// Apply RZZ(θ) = exp(-i θ ZZ/2) gate
    ///
    /// # Panics
    ///
    /// Panics if target qubit1 or qubit2 index is >= number of qubits or qubit1 == qubit2
    pub fn rzz(&mut self, theta: f64, qubit1: usize, qubit2: usize) {
        assert!(qubit1 < self.num_qubits);
        assert!(qubit2 < self.num_qubits);
        assert_ne!(qubit1, qubit2);

        // RZZ is diagonal in computational basis - just add phases
        for i in 0..self.state.len() {
            let bit1 = (i >> qubit1) & 1;
            let bit2 = (i >> qubit2) & 1;

            // Phase depends on parity of bits
            let phase = if bit1 ^ bit2 == 0 {
                // Same bits (00 or 11) -> e^(-iθ/2)
                Complex64::from_polar(1.0, -theta / 2.0)
            } else {
                // Different bits (01 or 10) -> e^(iθ/2)
                Complex64::from_polar(1.0, theta / 2.0)
            };

            self.state[i] *= phase;
        }
    }

    /// Apply a general two-qubit unitary given by a 4x4 complex matrix
    /// U = [[u00, u01, u02, u03],
    ///      [u10, u11, u12, u13],
    ///      [u20, u21, u22, u23],
    ///      [u30, u31, u32, u33]]
    ///
    /// # Panics
    ///
    /// Panics if target qubit1 or qubit2 index is >= number of qubits or qubit1 == qubit2
    pub fn two_qubit_unitary(&mut self, qubit1: usize, qubit2: usize, matrix: [[Complex64; 4]; 4]) {
        assert!(qubit1 < self.num_qubits);
        assert!(qubit2 < self.num_qubits);
        assert_ne!(qubit1, qubit2);

        // Make sure qubit1 < qubit2 for consistent ordering
        let (q1, q2) = if qubit1 < qubit2 {
            (qubit1, qubit2)
        } else {
            (qubit2, qubit1)
        };

        // Process state vector in groups of 4 amplitudes
        for i in 0..self.state.len() {
            let bit1 = (i >> q1) & 1;
            let bit2 = (i >> q2) & 1;

            // Only process each set of 4 states once
            if bit1 == 0 && bit2 == 0 {
                // Calculate indices for all four basis states
                let i00 = i;
                let i01 = i ^ (1 << q2);
                let i10 = i ^ (1 << q1);
                let i11 = i ^ (1 << q1) ^ (1 << q2);

                // Store original amplitudes
                let a00 = self.state[i00];
                let a01 = self.state[i01];
                let a10 = self.state[i10];
                let a11 = self.state[i11];

                // Apply the 4x4 unitary transformation
                self.state[i00] = matrix[0][0] * a00
                    + matrix[0][1] * a01
                    + matrix[0][2] * a10
                    + matrix[0][3] * a11;
                self.state[i01] = matrix[1][0] * a00
                    + matrix[1][1] * a01
                    + matrix[1][2] * a10
                    + matrix[1][3] * a11;
                self.state[i10] = matrix[2][0] * a00
                    + matrix[2][1] * a01
                    + matrix[2][2] * a10
                    + matrix[2][3] * a11;
                self.state[i11] = matrix[3][0] * a00
                    + matrix[3][1] * a01
                    + matrix[3][2] * a10
                    + matrix[3][3] * a11;
            }
        }
    }

    /// Measure a single qubit in the Z basis and collapse the state
    ///
    /// # Panics
    ///
    /// Panics if target qubit index is >= number of qubits
    pub fn measure(&mut self, target: usize) -> usize {
        assert!(target < self.num_qubits);
        let mut rng = rand::thread_rng();

        let step = 1 << target;
        let mut prob_one = 0.0;

        // Calculate probability of measuring |1⟩
        for i in (0..self.state.len()).step_by(2 * step) {
            for offset in 0..step {
                let idx = i + offset + step; // Target bit = 1 positions
                prob_one += self.state[idx].norm_sqr();
            }
        }

        // Decide measurement outcome
        let result = usize::from(rng.gen::<f64>() < prob_one);

        // Collapse and normalize state
        let mut norm = 0.0;
        for i in 0..self.state.len() {
            let bit = (i >> target) & 1;
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

        result
    }

    /// Reset a qubit to the |0⟩ state
    ///
    /// # Panics
    ///
    /// Panics if target qubit index is >= number of qubits
    pub fn reset(&mut self, target: usize) {
        assert!(target < self.num_qubits);

        // Measure the qubit
        let result = self.measure(target);

        // If we got |1⟩, apply X to flip it to |0⟩
        if result == 1 {
            self.x(target);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, FRAC_PI_3, PI, TAU};

    #[test]
    fn test_new_state() {
        let q = StateVec::new(2);
        assert_eq!(q.state[0], Complex64::new(1.0, 0.0));
        for i in 1..4 {
            assert_eq!(q.state[i], Complex64::new(0.0, 0.0));
        }
    }

    #[test]
    fn test_rx() {
        let mut q = StateVec::new(1);

        // RX(π) should be equivalent to X up to global phase
        q.rx(PI, 0);
        assert!(q.state[0].norm() < 1e-10);
        assert!((q.state[1].norm() - 1.0).abs() < 1e-10);

        // RX(2π) should return to initial state up to global phase
        let mut q = StateVec::new(1);
        q.rx(2.0 * PI, 0);
        assert!((q.state[0].norm() - 1.0).abs() < 1e-10);
        assert!(q.state[1].norm() < 1e-10);
    }

    #[test]
    fn test_ry() {
        let mut q = StateVec::new(1);

        // RY(π) should be equivalent to X up to global phase
        q.ry(PI, 0);
        assert!(q.state[0].norm() < 1e-10);
        assert!((q.state[1].norm() - 1.0).abs() < 1e-10);

        // RY(2π) should return to initial state up to global phase
        let mut q = StateVec::new(1);
        q.ry(2.0 * PI, 0);
        assert!((q.state[0].norm() - 1.0).abs() < 1e-10);
        assert!(q.state[1].norm() < 1e-10);
    }

    #[test]
    fn test_rz() {
        let mut q = StateVec::new(1);

        // RZ should only add phases, not change probabilities
        q.hadamard(0); // Put in superposition first
        let probs_before: Vec<f64> = q.state.iter().map(num_complex::Complex::norm_sqr).collect();

        q.rz(FRAC_PI_2, 0);
        let probs_after: Vec<f64> = q.state.iter().map(num_complex::Complex::norm_sqr).collect();

        // Probabilities should remain unchanged
        for (p1, p2) in probs_before.iter().zip(probs_after.iter()) {
            assert!((p1 - p2).abs() < 1e-10);
        }

        // RZ(2π) should return to initial state up to global phase
        let mut q = StateVec::new(1);
        q.hadamard(0);
        let state_before = q.state.clone();
        q.rz(TAU, 0);

        // States should be identical up to global phase
        let phase = q.state[0] / state_before[0];
        for (a, b) in q.state.iter().zip(state_before.iter()) {
            assert!((a - b * phase).norm() < 1e-10);
        }
    }

    #[test]
    fn test_hadamard() {
        let mut q = StateVec::new(1);
        q.hadamard(0);

        assert!((q.state[0].re - FRAC_1_SQRT_2).abs() < 1e-10);
        assert!((q.state[1].re - FRAC_1_SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_swap() {
        let mut q = StateVec::new(2);
        q.x(0);
        q.swap(0, 1);

        assert!(q.state[0].norm() < 1e-10);
        assert!((q.state[2].re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cx() {
        let mut q = StateVec::new(2);
        // Prep |+>
        q.hadamard(0);
        q.cx(0, 1);

        // Should be in Bell state (|00> + |11>)/sqrt(2)
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!((q.state[0].re - expected).abs() < 1e-10);
        assert!((q.state[3].re - expected).abs() < 1e-10);
        assert!(q.state[1].norm() < 1e-10);
        assert!(q.state[2].norm() < 1e-10);
    }

    #[test]
    fn test_cy() {
        let mut q = StateVec::new(2);

        // Create |+0⟩ state
        q.hadamard(0);

        // Apply CY to get entangled state
        q.cy(0, 1);

        // Should be (|00⟩ + i|11⟩)/√2
        let expected = FRAC_1_SQRT_2;
        assert!((q.state[0].re - expected).abs() < 1e-10); // |00⟩ amplitude
        assert!(q.state[1].norm() < 1e-10); // |01⟩ amplitude
        assert!(q.state[2].norm() < 1e-10); // |10⟩ amplitude
        assert!((q.state[3].im - expected).abs() < 1e-10); // |11⟩ amplitude
    }

    #[test]
    fn test_cz() {
        let mut q = StateVec::new(2);

        // Create |++⟩ state
        q.hadamard(0);
        q.hadamard(1);

        // Apply CZ
        q.cz(0, 1);

        // Should be (|00⟩ + |01⟩ + |10⟩ - |11⟩)/2
        let expected = 0.5;
        assert!((q.state[0].re - expected).abs() < 1e-10); // |00⟩ amplitude
        assert!((q.state[1].re - expected).abs() < 1e-10); // |01⟩ amplitude
        assert!((q.state[2].re - expected).abs() < 1e-10); // |10⟩ amplitude
        assert!((q.state[3].re + expected).abs() < 1e-10); // |11⟩ amplitude
    }

    #[test]
    fn test_control_target_independence() {
        // Test that CY and CZ work regardless of which qubit is control/target
        let mut q1 = StateVec::new(2);
        let mut q2 = StateVec::new(2);

        // Prepare same initial state
        q1.hadamard(0);
        q1.hadamard(1);
        q2.hadamard(0);
        q2.hadamard(1);

        // Apply gates with different control/target
        q1.cz(0, 1);
        q2.cz(1, 0);

        // Results should be identical
        for (a, b) in q1.state.iter().zip(q2.state.iter()) {
            assert!((a - b).norm() < 1e-10);
        }
    }

    #[test]
    fn test_x() {
        let mut q = StateVec::new(1);

        // Check initial state is |0>
        assert!((q.state[0].re - 1.0).abs() < 1e-10);
        assert!(q.state[1].norm() < 1e-10);

        // Test X on |0> -> |1>
        q.x(0);
        assert!(q.state[0].norm() < 1e-10);
        assert!((q.state[1].re - 1.0).abs() < 1e-10);

        // Test X on |1> -> |0>
        q.x(0);
        assert!((q.state[0].re - 1.0).abs() < 1e-10);
        assert!(q.state[1].norm() < 1e-10);

        // Test X on superposition
        q.hadamard(0);
        let initial_state = q.state.clone();
        q.x(0); // X|+> = |+>
        for (state, initial) in q.state.iter().zip(initial_state.iter()) {
            assert!((state - initial).norm() < 1e-10);
        }

        // Test X on second qubit of two-qubit system
        let mut q = StateVec::new(2);
        q.x(1);
        assert!(q.state[0].norm() < 1e-10);
        assert!((q.state[2].re - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_y() {
        let mut q = StateVec::new(1);

        // Test Y on |0⟩ -> i|1⟩
        q.y(0);
        assert!(q.state[0].norm() < 1e-10);
        assert!((q.state[1] - Complex64::i()).norm() < 1e-10);

        // Test Y on i|1⟩ -> |0⟩
        q.y(0);
        assert!((q.state[0].re - 1.0).abs() < 1e-10);
        assert!(q.state[1].norm() < 1e-10);

        // Test Y on |+⟩
        let mut q = StateVec::new(1);
        q.hadamard(0); // Create |+⟩
        q.y(0); // Should give i|-⟩
        let expected = FRAC_1_SQRT_2;
        assert!((q.state[0].im + expected).abs() < 1e-10);
        assert!((q.state[1].im - expected).abs() < 1e-10);
    }

    #[test]
    fn test_z() {
        let mut q = StateVec::new(1);

        // Test Z on |0⟩ -> |0⟩
        q.z(0);
        assert!((q.state[0].re - 1.0).abs() < 1e-10);
        assert!(q.state[1].norm() < 1e-10);

        // Test Z on |1⟩ -> -|1⟩
        q.x(0); // Prepare |1⟩
        q.z(0);
        assert!(q.state[0].norm() < 1e-10);
        assert!((q.state[1].re + 1.0).abs() < 1e-10);

        // Test Z on |+⟩ -> |-⟩
        let mut q = StateVec::new(1);
        q.hadamard(0); // Create |+⟩
        q.z(0); // Should give |-⟩
        let expected = FRAC_1_SQRT_2;
        assert!((q.state[0].re - expected).abs() < 1e-10);
        assert!((q.state[1].re + expected).abs() < 1e-10);
    }

    #[test]
    fn test_pauli_relations() {
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);

        // Store initial state
        let initial_state = q1.state.clone();

        // Test XYZ sequence
        q1.x(0);
        q1.y(0);
        q1.z(0);

        // XYZ = -iI, so state should be -i times initial state
        if initial_state[0].norm() > 1e-10 {
            let phase = q1.state[0] / initial_state[0];
            assert!((phase + Complex64::i()).norm() < 1e-10); // Changed to +Complex64::i()
        }

        // Test YZX sequence - should give same result
        q2.y(0);
        q2.z(0);
        q2.x(0);

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

    #[test]
    fn test_rxx() {
        // Test 1: RXX(π/2) on |00⟩ should give (|00⟩ - i|11⟩)/√2
        let mut q = StateVec::new(2);
        q.rxx(FRAC_PI_2, 0, 1);

        let expected = FRAC_1_SQRT_2;
        assert!((q.state[0].re - expected).abs() < 1e-10);
        assert!(q.state[1].norm() < 1e-10);
        assert!(q.state[2].norm() < 1e-10);
        assert!((q.state[3].im + expected).abs() < 1e-10);

        // Test 2: RXX(2π) should return to original state up to global phase
        let mut q = StateVec::new(2);
        q.hadamard(0); // Create some initial state
        let initial = q.state.clone();
        q.rxx(TAU, 0, 1);

        // Compare up to global phase
        if q.state[0].norm() > 1e-10 {
            let phase = q.state[0] / initial[0];
            for (a, b) in q.state.iter().zip(initial.iter()) {
                assert!((a - b * phase).norm() < 1e-10);
            }
        }

        // Test 3: RXX(π) should flip |00⟩ to |11⟩ up to phase
        let mut q = StateVec::new(2);
        q.rxx(PI, 0, 1);

        // Should get -i|11⟩
        assert!(q.state[0].norm() < 1e-10);
        assert!(q.state[1].norm() < 1e-10);
        assert!(q.state[2].norm() < 1e-10);
        assert!((q.state[3] - Complex64::new(0.0, -1.0)).norm() < 1e-10);
    }

    #[test]
    fn test_rxx_symmetry() {
        // Test that RXX is symmetric under exchange of qubits
        let mut q1 = StateVec::new(2);
        let mut q2 = StateVec::new(2);

        // Prepare same non-trivial initial state
        q1.hadamard(0);
        q1.hadamard(1);
        q2.hadamard(0);
        q2.hadamard(1);

        // Apply RXX with different qubit orders
        q1.rxx(FRAC_PI_3, 0, 1);
        q2.rxx(FRAC_PI_3, 1, 0);

        // Results should be identical
        for (a, b) in q1.state.iter().zip(q2.state.iter()) {
            assert!((a - b).norm() < 1e-10);
        }
    }

    #[test]
    fn test_ryy() {
        // Test 1: RYY(π/2) on |00⟩ should give (|00⟩ - i|11⟩)/√2
        let mut q = StateVec::new(2);
        q.ryy(FRAC_PI_2, 0, 1);

        let expected = FRAC_1_SQRT_2;
        assert!((q.state[0].re - expected).abs() < 1e-10);
        assert!(q.state[1].norm() < 1e-10);
        assert!(q.state[2].norm() < 1e-10);
        assert!((q.state[3].im + expected).abs() < 1e-10);

        // Test 2: RYY(2π) should return to original state up to global phase
        let mut q = StateVec::new(2);
        q.hadamard(0); // Create some initial state
        let initial = q.state.clone();
        q.ryy(TAU, 0, 1);

        if q.state[0].norm() > 1e-10 {
            let phase = q.state[0] / initial[0];
            for (a, b) in q.state.iter().zip(initial.iter()) {
                assert!((a - b * phase).norm() < 1e-10);
            }
        }
    }

    #[test]
    fn test_rzz() {
        // Test 1: RZZ(π) on (|00⟩ + |11⟩)/√2 should give itself
        let mut q = StateVec::new(2);
        // Create Bell state
        q.hadamard(0);
        q.cx(0, 1);
        let initial = q.state.clone();

        q.rzz(PI, 0, 1);

        // Compare up to global phase
        if q.state[0].norm() > 1e-10 {
            let phase = q.state[0] / initial[0];
            for (a, b) in q.state.iter().zip(initial.iter()) {
                assert!((a - b * phase).norm() < 1e-10);
            }
        }

        // Test 2: RZZ(π/2) on |++⟩
        let mut q = StateVec::new(2);
        q.hadamard(0);
        q.hadamard(1);
        q.rzz(FRAC_PI_2, 0, 1);

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

    #[test]
    fn test_rotation_symmetries() {
        // Test that all rotations are symmetric under exchange of qubits
        let mut q1 = StateVec::new(2);
        let mut q2 = StateVec::new(2);

        // Prepare same non-trivial initial state
        q1.hadamard(0);
        q1.hadamard(1);
        q2.hadamard(0);
        q2.hadamard(1);

        let theta = PI / 3.0;

        // Test RYY symmetry
        q1.ryy(theta, 0, 1);
        q2.ryy(theta, 1, 0);

        for (a, b) in q1.state.iter().zip(q2.state.iter()) {
            assert!((a - b).norm() < 1e-10);
        }

        // Test RZZ symmetry
        let mut q1 = StateVec::new(2);
        let mut q2 = StateVec::new(2);
        q1.hadamard(0);
        q1.hadamard(1);
        q2.hadamard(0);
        q2.hadamard(1);

        q1.rzz(theta, 0, 1);
        q2.rzz(theta, 1, 0);

        for (a, b) in q1.state.iter().zip(q2.state.iter()) {
            assert!((a - b).norm() < 1e-10);
        }
    }

    #[test]
    fn test_measure2() {
        let mut q = StateVec::new(1);
        q.hadamard(0);
        let result = q.measure(0);

        // Check collapse to |0⟩ or |1⟩
        assert!(result == 0 || result == 1);
        let norm: f64 = q.state.iter().map(num_complex::Complex::norm_sqr).sum();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_measure() {
        // Test 1: Meauring |0> state
        let mut q = StateVec::new(1);
        let result = q.measure(0);
        assert_eq!(result, 0);
        assert!((q.state[0].re - 1.0).abs() < 1e-10);
        assert!(q.state[1].norm() < 1e-10);

        // Test 2: Measuring |1> state
        let mut q = StateVec::new(1);
        q.x(0);
        let result = q.measure(0);
        assert_eq!(result, 1);
        assert!(q.state[0].norm() < 1e-10);
        assert!((q.state[1].re - 1.0).abs() < 1e-10);

        // Test 3: Measuring superposition state multiple times
        let mut zeros = 0;
        let trials = 1000;

        for _ in 0..trials {
            let mut q = StateVec::new(1);
            q.hadamard(0);
            let result = q.measure(0);
            if result == 0 {
                zeros += 1;
            }
        }

        // Check if measurements are roughly equally distributed
        let ratio = f64::from(zeros) / f64::from(trials);
        assert!((ratio - 0.5).abs() < 0.1); // Should be close to 0.5...

        // Test 4: Measuring one qubit of a Bell state
        let mut q = StateVec::new(2);
        q.hadamard(0);
        q.cx(0, 1);

        // Measure first qubit
        let result1 = q.measure(0);
        // Measure second qubit - should match first
        let result2 = q.measure(1);
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_reset() {
        let mut q = StateVec::new(1);

        q.hadamard(0);
        assert!((q.state[0].re - FRAC_1_SQRT_2).abs() < 1e-10);
        assert!((q.state[1].re - FRAC_1_SQRT_2).abs() < 1e-10);

        q.reset(0);

        assert!((q.state[0].re - 1.0).abs() < 1e-10);
        assert!(q.state[1].norm() < 1e-10);
    }

    #[test]
    fn test_reset_multiple_qubits() {
        let mut q = StateVec::new(2);

        q.hadamard(0);
        q.cx(0, 1);

        q.reset(0);

        let prob_0 = q.state[0].norm_sqr() + q.state[2].norm_sqr();
        let prob_1 = q.state[1].norm_sqr() + q.state[3].norm_sqr();

        assert!((prob_0 - 1.0).abs() < 1e-10);
        assert!(prob_1 < 1e-10);
    }

    #[test]
    fn test_single_qubit_rotation() {
        let mut q = StateVec::new(1);

        // Test 1: Hadamard gate
        let h00 = Complex64::new(FRAC_1_SQRT_2, 0.0);
        let h01 = Complex64::new(FRAC_1_SQRT_2, 0.0);
        let h10 = Complex64::new(FRAC_1_SQRT_2, 0.0);
        let h11 = Complex64::new(-FRAC_1_SQRT_2, 0.0);

        q.single_qubit_rotation(0, h00, h01, h10, h11);
        assert!((q.state[0].re - FRAC_1_SQRT_2).abs() < 1e-10);
        assert!((q.state[1].re - FRAC_1_SQRT_2).abs() < 1e-10);

        // Test 2: X gate
        let mut q = StateVec::new(1);
        let x00 = Complex64::new(0.0, 0.0);
        let x01 = Complex64::new(1.0, 0.0);
        let x10 = Complex64::new(1.0, 0.0);
        let x11 = Complex64::new(0.0, 0.0);

        q.single_qubit_rotation(0, x00, x01, x10, x11);
        assert!(q.state[0].norm() < 1e-10);
        assert!((q.state[1].re - 1.0).abs() < 1e-10);

        // Test 3: Phase gate
        let mut q = StateVec::new(1);
        let p00 = Complex64::new(1.0, 0.0);
        let p01 = Complex64::new(0.0, 0.0);
        let p10 = Complex64::new(0.0, 0.0);
        let p11 = Complex64::new(0.0, 1.0);

        q.single_qubit_rotation(0, p00, p01, p10, p11);
        assert!((q.state[0].re - 1.0).abs() < 1e-10);
        assert!(q.state[1].norm() < 1e-10);

        // Test 4: Y gate using unitary
        let mut q = StateVec::new(1);
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
        let mut q = StateVec::new(1);

        // Create random state with Hadamard
        q.hadamard(0);

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
    fn test_u3_special_cases() {
        // Test 1: U3(π, 0, π) should be X gate
        let mut q = StateVec::new(1);
        q.u3(0, PI, 0.0, PI);
        assert!(q.state[0].norm() < 1e-10);
        assert!((q.state[1].re - 1.0).abs() < 1e-10);

        // Test 2: Hadamard gate
        // H = U3(π/2, 0, π)
        let mut q = StateVec::new(1);
        q.u3(0, PI / 2.0, 0.0, PI);
        assert!((q.state[0].re - FRAC_1_SQRT_2).abs() < 1e-10);
        assert!((q.state[1].re - FRAC_1_SQRT_2).abs() < 1e-10);

        // Test 3: U3(0, 0, π) should be Z gate
        let mut q = StateVec::new(1);
        q.hadamard(0); // First put in superposition
        let initial = q.state.clone();
        q.u3(0, 0.0, 0.0, PI);
        assert!((q.state[0] - initial[0]).norm() < 1e-10);
        assert!((q.state[1] + initial[1]).norm() < 1e-10);

        // Additional test: U3(π/2, π/2, -π/2) should be S†H
        let mut q = StateVec::new(1);
        q.u3(0, PI / 2.0, PI / 2.0, -PI / 2.0);
        // This creates the state (|0⟩ + i|1⟩)/√2
        assert!((q.state[0].re - FRAC_1_SQRT_2).abs() < 1e-10);
        assert!((q.state[1].im - FRAC_1_SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_u3_composition() {
        let mut q1 = StateVec::new(1);
        let q2 = StateVec::new(1);

        // Two U3 gates that should multiply to identity
        q1.u3(0, PI / 3.0, PI / 4.0, PI / 6.0);
        q1.u3(0, -PI / 3.0, -PI / 6.0, -PI / 4.0);

        // Compare with initial state
        for (a, b) in q1.state.iter().zip(q2.state.iter()) {
            assert!((a - b).norm() < 1e-10);
        }
    }

    #[test]
    fn test_u3_arbitrary() {
        let mut q = StateVec::new(1);

        // Apply some arbitrary rotation
        let theta = PI / 5.0;
        let phi = PI / 7.0;
        let lambda = PI / 3.0;
        q.u3(0, theta, phi, lambda);

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
    fn test_two_qubit_unitary_cnot() {
        // Test that we can implement CNOT using the general unitary
        let mut q1 = StateVec::new(2);
        let mut q2 = StateVec::new(2);

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
        q1.hadamard(0);
        q1.cx(0, 1);

        q2.hadamard(0);
        q2.two_qubit_unitary(0, 1, cnot);

        // Compare results
        for (a, b) in q1.state.iter().zip(q2.state.iter()) {
            assert!((a - b).norm() < 1e-10);
        }
    }

    #[test]
    fn test_two_qubit_unitary_swap() {
        // Test SWAP gate
        let mut q = StateVec::new(2);

        // Prepare |10⟩ state
        q.x(0);

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

    #[test]
    fn test_two_qubit_unitary_properties() {
        let mut q = StateVec::new(2);

        // Create a non-trivial state
        q.hadamard(0);
        q.hadamard(1);

        // iSWAP matrix
        let iswap = [
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 1.0),
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

        q.two_qubit_unitary(0, 1, iswap);

        // Verify normalization is preserved
        let norm: f64 = q.state.iter().map(num_complex::Complex::norm_sqr).sum();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_single_qubit_locality() {
        // Test on 3 qubit system that gates only affect their target
        let mut q = StateVec::new(3);

        // Prepare state |+⟩|0⟩|0⟩
        q.hadamard(0); // Affects least significant bit

        // Apply X to qubit 2 (most significant bit)
        q.x(2);

        // Check that qubit 0 is still in |+⟩ state
        // When qubit 2 is |1⟩, check LSB still shows |+⟩
        assert!((q.state[4].re - FRAC_1_SQRT_2).abs() < 1e-10); // |100⟩
        assert!((q.state[5].re - FRAC_1_SQRT_2).abs() < 1e-10); // |101⟩
    }

    #[test]
    fn test_bit_indexing() {
        let mut q = StateVec::new(3);

        println!("Initial state (|000⟩):");
        for i in 0..8 {
            println!("  {:03b}: {:.3}", i, q.state[i]);
        }

        // Put |+⟩ on qubit 0 (LSB)
        q.hadamard(0);

        println!("\nAfter H on qubit 0:");
        for i in 0..8 {
            println!("  {:03b}: {:.3}", i, q.state[i]);
        }

        // Check state is |+⟩|0⟩|0⟩
        // Only indices that differ in LSB (qubit 0) should be FRAC_1_SQRT_2
        for i in 0..8 {
            let qubit0 = i & 1;
            let qubit1 = (i >> 1) & 1;
            let qubit2 = (i >> 2) & 1;

            let expected = if qubit1 == 0 && qubit2 == 0 {
                FRAC_1_SQRT_2
            } else {
                0.0
            };

            if (q.state[i].re - expected).abs() >= 1e-10 {
                println!("\nMismatch at index {i}: {i:03b}");
                println!("Qubit values: q2={qubit2}, q1={qubit1}, q0={qubit0}");
                println!("Expected {}, got {}", expected, q.state[i].re);
            }
            assert!((q.state[i].re - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_two_qubit_locality() {
        let mut q = StateVec::new(4);

        println!("Initial state:");
        for i in 0..16 {
            println!("  {:04b}: {:.3}", i, q.state[i]);
        }

        // Prepare |+⟩ on qubit 0 (LSB)
        q.hadamard(0);

        println!("\nAfter H on qubit 0:");
        for i in 0..16 {
            println!("  {:04b}: {:.3}", i, q.state[i]);
        }

        // Apply CX between qubits 2,3
        q.cx(2, 3);

        println!("\nAfter CX on qubits 2,3:");
        for i in 0..16 {
            println!("  {:04b}: {:.3}", i, q.state[i]);

            // Extract qubit values
            // let _q0 = i & 1;
            let q1 = (i >> 1) & 1;
            let q2 = (i >> 2) & 1;
            let q3 = (i >> 3) & 1;

            // Only states with q0=0 or q0=1 and q1=q2=q3=0 should have amplitude
            let expected = if q1 == 0 && q2 == 0 && q3 == 0 {
                FRAC_1_SQRT_2
            } else {
                0.0
            };

            if (q.state[i].re - expected).abs() >= 1e-10 {
                println!("Mismatch at {i:04b}");
                println!("Expected {}, got {}", expected, q.state[i].re);
            }
            assert!((q.state[i].re - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_two_qubit_gate_locality() {
        let mut q = StateVec::new(3);

        // Prepare state |+⟩|0⟩|0⟩
        q.hadamard(0);

        // Apply CX on qubits 1 and 2 (no effect on qubit 0)
        q.cx(1, 2);

        // Qubit 0 should remain in superposition
        let expected_amp = 1.0 / 2.0_f64.sqrt();
        assert!((q.state[0].re - expected_amp).abs() < 1e-10);
        assert!((q.state[1].re - expected_amp).abs() < 1e-10);
    }

    #[test]
    fn test_rotation_locality() {
        let mut q = StateVec::new(3);

        println!("Initial state:");
        for i in 0..8 {
            println!("  {:03b}: {:.3}", i, q.state[i]);
        }

        // Prepare |+⟩ on qubit 0 (LSB)
        q.hadamard(0);

        println!("\nAfter H on qubit 0:");
        for i in 0..8 {
            println!("  {:03b}: {:.3}", i, q.state[i]);
        }

        // Apply rotation to qubit 1
        q.rx(PI / 2.0, 1);

        println!("\nAfter RX on qubit 1:");
        for i in 0..8 {
            println!("  {:03b}: {:.3}", i, q.state[i]);
        }

        // Check each basis state contribution
        for i in 0..8 {
            let expected = FRAC_1_SQRT_2;
            if (q.state[i].norm() - expected).abs() >= 1e-10 {
                println!("\nMismatch at index {i}: {i:03b}");
                println!("Expected norm {}, got {}", expected, q.state[i].norm());
            }
        }
    }

    #[test]
    fn test_large_system_hadamard() {
        let num_qubits = 10; // 10 qubits => state vector size = 1024
        let mut q = StateVec::new(num_qubits);

        // Apply Hadamard gate to the 0th qubit
        q.hadamard(0);

        // Check that |0...0> and |1...0> have equal amplitude
        let expected_amp = 1.0 / 2.0_f64.sqrt();
        assert!((q.state[0].re - expected_amp).abs() < 1e-10);
        assert!((q.state[1].re - expected_amp).abs() < 1e-10);

        // Ensure all other amplitudes remain zero
        for i in 2..q.state.len() {
            assert!(q.state[i].norm() < 1e-10);
        }
    }

    #[test]
    #[should_panic(expected = "assertion failed: target < self.num_qubits")]
    fn test_invalid_qubit_index_single() {
        let mut q = StateVec::new(2);
        q.x(3); // Invalid qubit index
    }

    #[test]
    #[should_panic(expected = "assertion failed: target < self.num_qubits")]
    fn test_invalid_qubit_index_controlled() {
        let mut q = StateVec::new(2);
        q.cx(1, 2); // Invalid target qubit
    }

    #[test]
    #[should_panic(expected = "assertion `left != right` failed\n  left: 0\n right: 0")]
    fn test_control_equals_target() {
        let mut q = StateVec::new(2);
        q.cx(0, 0); // Control and target are the same
    }

    #[test]
    fn test_measurement_consistency() {
        let mut q = StateVec::new(1);

        // Put qubit in |1⟩ state
        q.x(0);

        // Measure twice - result should be the same
        let result1 = q.measure(0);
        let result2 = q.measure(0);

        assert_eq!(result1, 1);
        assert_eq!(result2, 1);
    }

    #[test]
    fn test_measurement_on_entangled_state() {
        let mut q = StateVec::new(2);

        // Create Bell state (|00⟩ + |11⟩) / sqrt(2)
        q.hadamard(0);
        q.cx(0, 1);

        // Measure the first qubit
        let result1 = q.measure(0);

        // Measure the second qubit - should match the first
        let result2 = q.measure(1);

        assert_eq!(result1, result2);
    }
}
