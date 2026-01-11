// Copyright 2025 The PECOS Developers
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

//! Qulacs quantum simulator bindings for PECOS.
//!
//! This crate provides Rust bindings to the Qulacs quantum simulator C++ library,
//! enabling high-performance quantum circuit simulation.

mod bridge;

use bridge::ffi;
use num_complex::Complex64;
use pecos_core::{IndexableElement, RngManageable};
use pecos_qsim::{
    ArbitraryRotationGateable, CliffordGateable, MeasurementResult, QuantumSimulator,
};
use pecos_rng::PecosRng;
use rand::{RngCore, SeedableRng};
use std::fmt::Debug;

/// A quantum state simulator using Qulacs C++ backend.
///
/// `QulacsStateVec` maintains the full quantum state as a complex vector with 2ⁿ amplitudes
/// for n qubits using the high-performance Qulacs C++ library.
///
/// # Type Parameters
/// * `R` - Random number generator type implementing `RngCore + SeedableRng` traits
pub struct QulacsStateVec<R = PecosRng>
where
    R: RngCore + SeedableRng + Debug,
{
    state: cxx::UniquePtr<ffi::QulacsState>,
    num_qubits: usize,
    rng: R,
}

// Implement Clone for QulacsStateVec
impl<R> Clone for QulacsStateVec<R>
where
    R: RngCore + SeedableRng + Debug + Clone,
{
    fn clone(&self) -> Self {
        let mut new_rng = self.rng.clone();
        let mut new_state = ffi::clone_quantum_state(&self.state);
        // Seed the cloned state's C++ RNG with a new value
        let seed = new_rng.next_u32();
        ffi::set_seed(new_state.pin_mut(), seed);
        Self {
            state: new_state,
            num_qubits: self.num_qubits,
            rng: new_rng,
        }
    }
}

impl QulacsStateVec {
    /// Create a new state initialized to |0...0⟩
    #[inline]
    #[must_use]
    pub fn new(num_qubits: usize) -> QulacsStateVec<PecosRng> {
        let rng = PecosRng::from_os_rng();
        QulacsStateVec::with_rng(num_qubits, rng)
    }

    /// Create a new state vector simulator with a specific seed for the random number generator
    #[inline]
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> QulacsStateVec<PecosRng> {
        let rng = PecosRng::seed_from_u64(seed);
        QulacsStateVec::with_rng(num_qubits, rng)
    }
}

impl<R> QulacsStateVec<R>
where
    R: RngCore + SeedableRng + Debug,
{
    /// Create a new state vector with a custom random number generator.
    #[inline]
    #[must_use]
    pub fn with_rng(num_qubits: usize, mut rng: R) -> Self {
        let mut state = ffi::create_quantum_state(num_qubits);
        // Seed the C++ RNG with a value from our Rust RNG
        let seed = rng.next_u32();
        ffi::set_seed(state.pin_mut(), seed);
        Self {
            state,
            num_qubits,
            rng,
        }
    }

    /// Returns the number of qubits in the system
    #[inline]
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Convert PECOS qubit index to Qulacs qubit index
    /// PECOS uses MSB-first ordering (q0 is leftmost/most significant)
    /// Qulacs uses LSB-first ordering (q0 is rightmost/least significant)
    #[inline]
    fn convert_qubit_index(&self, pecos_qubit: usize) -> usize {
        if pecos_qubit >= self.num_qubits {
            // Return the same index to let Qulacs handle the error
            // This prevents panic in Rust and allows proper error propagation
            return pecos_qubit;
        }
        self.num_qubits
            .saturating_sub(1)
            .saturating_sub(pecos_qubit)
    }

    /// Convert PECOS basis state to Qulacs basis state by reversing bit order
    #[inline]
    fn convert_basis_state(&self, pecos_basis: usize) -> usize {
        let mut qulacs_basis = 0;
        for i in 0..self.num_qubits {
            if (pecos_basis >> i) & 1 == 1 {
                // Bit i in PECOS maps to bit (n-1-i) in Qulacs
                qulacs_basis |= 1 << (self.num_qubits - 1 - i);
            }
        }
        qulacs_basis
    }

    /// Prepare the state as a specific computational basis state
    ///
    /// # Panics
    /// Panics if `basis_state` is greater than or equal to 2^n where n is the number of qubits.
    #[inline]
    pub fn prepare_computational_basis(&mut self, basis_state: usize) -> &mut Self {
        assert!(basis_state < 1 << self.num_qubits);
        let qulacs_basis = self.convert_basis_state(basis_state);
        ffi::set_computational_basis(self.state.pin_mut(), qulacs_basis as u64);
        self
    }

    /// Prepare all qubits in the |+⟩ state, creating an equal superposition of all basis states
    #[inline]
    pub fn prepare_plus_state(&mut self) -> &mut Self {
        ffi::reset(self.state.pin_mut());
        for i in 0..self.num_qubits {
            self.h(i);
        }
        self
    }

    /// Returns the state vector
    #[inline]
    #[must_use]
    pub fn state(&self) -> Vec<Complex64> {
        let size = ffi::get_vector_size(&self.state);
        let mut vector = Vec::with_capacity(size);

        // Since we convert qubit indices when applying gates,
        // the state vector is already in the correct ordering for PECOS
        // We just need to retrieve it directly
        for idx in 0..size {
            let amp = ffi::get_amplitude(&self.state, idx as u64);
            vector.push(Complex64::new(amp[0], amp[1]));
        }

        vector
    }

    /// Returns the probability of measuring a specific basis state
    ///
    /// # Panics
    /// Panics if `basis_state` is greater than or equal to 2^n where n is the number of qubits.
    #[inline]
    #[must_use]
    pub fn probability(&self, basis_state: usize) -> f64 {
        assert!(basis_state < 1 << self.num_qubits);
        let qulacs_basis = self.convert_basis_state(basis_state);
        let amp = ffi::get_amplitude(&self.state, qulacs_basis as u64);
        amp[0] * amp[0] + amp[1] * amp[1]
    }

    /// Apply a general single-qubit unitary gate
    #[inline]
    pub fn single_qubit_rotation(
        &mut self,
        _qubit: usize,
        _u00: Complex64,
        _u01: Complex64,
        _u10: Complex64,
        _u11: Complex64,
    ) -> &mut Self {
        // This would need to be implemented in C++ side
        // For now, we can use the basic gates to approximate
        // TODO: Add proper single_qubit_unitary to C++ wrapper
        self
    }

    /// Apply a general two-qubit unitary given by a 4x4 complex matrix
    pub fn two_qubit_unitary(
        &mut self,
        _qubit1: usize,
        _qubit2: usize,
        _matrix: [[Complex64; 4]; 4],
    ) -> &mut Self {
        // This would need to be implemented in C++ side
        // TODO: Add proper two_qubit_unitary to C++ wrapper
        self
    }
}

// Implement QuantumSimulator trait
impl<R> QuantumSimulator for QulacsStateVec<R>
where
    R: RngCore + SeedableRng + Debug,
{
    fn reset(&mut self) -> &mut Self {
        ffi::reset(self.state.pin_mut());
        self
    }
}

// Implement CliffordGateable trait
impl<R, I> CliffordGateable<I> for QulacsStateVec<R>
where
    R: RngCore + SeedableRng + Debug,
    I: IndexableElement,
{
    fn x(&mut self, q: I) -> &mut Self {
        let qulacs_qubit = self.convert_qubit_index(q.to_index());
        ffi::apply_x(self.state.pin_mut(), qulacs_qubit);
        self
    }

    fn y(&mut self, q: I) -> &mut Self {
        let qulacs_qubit = self.convert_qubit_index(q.to_index());
        ffi::apply_y(self.state.pin_mut(), qulacs_qubit);
        self
    }

    fn z(&mut self, q: I) -> &mut Self {
        let qulacs_qubit = self.convert_qubit_index(q.to_index());
        ffi::apply_z(self.state.pin_mut(), qulacs_qubit);
        self
    }

    fn h(&mut self, q: I) -> &mut Self {
        let qulacs_qubit = self.convert_qubit_index(q.to_index());
        ffi::apply_h(self.state.pin_mut(), qulacs_qubit);
        self
    }

    fn sz(&mut self, q: I) -> &mut Self {
        let qulacs_qubit = self.convert_qubit_index(q.to_index());
        ffi::apply_s(self.state.pin_mut(), qulacs_qubit);
        self
    }

    fn szdg(&mut self, q: I) -> &mut Self {
        let qulacs_qubit = self.convert_qubit_index(q.to_index());
        ffi::apply_sdag(self.state.pin_mut(), qulacs_qubit);
        self
    }

    // NOTE: sx, sxdg, sy, sydg use the default trait implementations to ensure
    // consistency with StateVec's decomposition. The native Qulacs sqrt gates
    // have different conventions that cause state vector mismatches.

    fn cx(&mut self, q1: I, q2: I) -> &mut Self {
        let qulacs_q1 = self.convert_qubit_index(q1.to_index());
        let qulacs_q2 = self.convert_qubit_index(q2.to_index());
        ffi::apply_cnot(self.state.pin_mut(), qulacs_q1, qulacs_q2);
        self
    }

    fn cy(&mut self, q1: I, q2: I) -> &mut Self {
        // CY can be implemented using CX and single-qubit gates
        // CY = (I ⊗ Sdg) CX (I ⊗ S)
        self.szdg(q2);
        self.cx(q1, q2);
        self.sz(q2);
        self
    }

    fn cz(&mut self, q1: I, q2: I) -> &mut Self {
        let qulacs_q1 = self.convert_qubit_index(q1.to_index());
        let qulacs_q2 = self.convert_qubit_index(q2.to_index());
        ffi::apply_cz(self.state.pin_mut(), qulacs_q1, qulacs_q2);
        self
    }

    fn swap(&mut self, q1: I, q2: I) -> &mut Self {
        let qulacs_q1 = self.convert_qubit_index(q1.to_index());
        let qulacs_q2 = self.convert_qubit_index(q2.to_index());
        ffi::apply_swap(self.state.pin_mut(), qulacs_q1, qulacs_q2);
        self
    }

    fn mz(&mut self, q: I) -> MeasurementResult {
        let pecos_qubit = q.to_index();
        let qulacs_qubit = self.convert_qubit_index(pecos_qubit);
        let prob_zero = ffi::get_marginal_probability(&self.state, qulacs_qubit);
        let is_deterministic = prob_zero.abs() < 1e-10 || (prob_zero - 1.0).abs() < 1e-10;

        // The C++ measure_z function uses its own RNG (which we've seeded)
        // and properly collapses the state
        let outcome_bit = ffi::measure_z(self.state.pin_mut(), qulacs_qubit);
        let outcome = outcome_bit != 0;

        MeasurementResult {
            outcome,
            is_deterministic,
        }
    }

    // Override the f() gate - the default implementation in the trait has the wrong order
    // The F gate matrix is [[1+i, 1-i], [1+i, -1+i]]/2 which equals SZ @ SX as a matrix
    // But when applying gates sequentially, we need SX first then SZ
    fn f(&mut self, q: I) -> &mut Self {
        // Apply SX then SZ to get F = SZ @ SX matrix
        // This is because applying gates sequentially means the rightmost gate is applied first
        self.sx(q);
        self.sz(q);
        self
    }

    // Similarly for fdg - F† = (SZ @ SX)† = SX† @ SZ†
    // But when applying gates sequentially, we apply SZ† first then SX†
    fn fdg(&mut self, q: I) -> &mut Self {
        self.szdg(q);
        self.sxdg(q);
        self
    }
}

// Implement ArbitraryRotationGateable trait
impl<R, I> ArbitraryRotationGateable<I> for QulacsStateVec<R>
where
    R: RngCore + SeedableRng + Debug,
    I: IndexableElement,
{
    fn rx(&mut self, angle: f64, q: I) -> &mut Self {
        let qulacs_qubit = self.convert_qubit_index(q.to_index());
        ffi::apply_rx(self.state.pin_mut(), qulacs_qubit, angle);
        self
    }

    fn ry(&mut self, angle: f64, q: I) -> &mut Self {
        let qulacs_qubit = self.convert_qubit_index(q.to_index());
        ffi::apply_ry(self.state.pin_mut(), qulacs_qubit, angle);
        self
    }

    fn rz(&mut self, angle: f64, q: I) -> &mut Self {
        let qulacs_qubit = self.convert_qubit_index(q.to_index());
        // Both Qulacs and PECOS StateVec use the same convention: diag(e^(-iθ/2), e^(iθ/2))
        // No phase correction needed
        ffi::apply_rz(self.state.pin_mut(), qulacs_qubit, angle);
        self
    }

    fn t(&mut self, q: I) -> &mut Self {
        let qulacs_qubit = self.convert_qubit_index(q.to_index());
        ffi::apply_t(self.state.pin_mut(), qulacs_qubit);
        self
    }

    fn tdg(&mut self, q: I) -> &mut Self {
        let qulacs_qubit = self.convert_qubit_index(q.to_index());
        ffi::apply_tdag(self.state.pin_mut(), qulacs_qubit);
        self
    }

    fn rzz(&mut self, angle: f64, q1: I, q2: I) -> &mut Self {
        // RZZ(θ) = exp(-i θ/2 Z⊗Z)
        // Decomposition: CNOT(q1,q2), RZ(θ, q2), CNOT(q1,q2)
        // Actually gives: diag(e^(-iθ/2), e^(iθ/2), e^(iθ/2), e^(-iθ/2))
        let q1_raw = q1.to_index();
        let q2_raw = q2.to_index();
        let q1_conv = self.convert_qubit_index(q1_raw);
        let q2_conv = self.convert_qubit_index(q2_raw);
        ffi::apply_cnot(self.state.pin_mut(), q1_conv, q2_conv);
        ffi::apply_rz(self.state.pin_mut(), q2_conv, angle);
        ffi::apply_cnot(self.state.pin_mut(), q1_conv, q2_conv);
        self
    }

    // Override the rzzryyrxx method to fix the order of operations
    // The default trait implementation has a reversed order
    // We want RXX @ RYY @ RZZ as the final matrix, but when applying
    // gates sequentially, we apply them in the opposite order
    fn rzzryyrxx(&mut self, theta: f64, phi: f64, lambda: f64, q1: I, q2: I) -> &mut Self {
        // Apply RZZ first, then RYY, then RXX to get RXX @ RYY @ RZZ matrix
        self.rzz(lambda, q1, q2).ryy(phi, q1, q2).rxx(theta, q1, q2)
    }
}

// Implement RngManageable trait
impl<R> RngManageable for QulacsStateVec<R>
where
    R: RngCore + SeedableRng + Debug,
{
    type Rng = R;

    fn rng(&self) -> &Self::Rng {
        &self.rng
    }

    fn rng_mut(&mut self) -> &mut Self::Rng {
        &mut self.rng
    }

    fn set_rng(&mut self, mut rng: Self::Rng) {
        // Re-seed the C++ RNG when setting a new Rust RNG
        let seed = rng.next_u32();
        ffi::set_seed(self.state.pin_mut(), seed);
        self.rng = rng;
    }
}

// SAFETY: QulacsStateVec is Send + Sync because:
// 1. Each QulacsState instance in C++ is completely independent (no shared global state)
// 2. UniquePtr provides exclusive ownership
// 3. The RNG is required to be Send + Sync
// 4. All operations on QulacsState are self-contained
unsafe impl<R> Send for QulacsStateVec<R> where R: RngCore + SeedableRng + Debug + Send {}

unsafe impl<R> Sync for QulacsStateVec<R> where R: RngCore + SeedableRng + Debug + Sync {}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod thread_test;
