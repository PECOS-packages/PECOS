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

use pecos_core::QubitId;
use pecos_qsim::{CliffordGateable, MeasurementResult, QuantumSimulator};

// Include the cxx-generated bindings
#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("pecos-cppsparsesim/src/cxx_shim.h");

        type StateWrapper;

        // Factory function
        fn create_state_wrapper(num_qubits: u64, reserve_buckets: i32) -> UniquePtr<StateWrapper>;

        // Member functions of StateWrapper - no prefix needed!
        fn set_seed(self: Pin<&mut StateWrapper>, seed: u32);
        fn clear(self: Pin<&mut StateWrapper>);
        fn hadamard(self: Pin<&mut StateWrapper>, qubit: u64);
        fn bitflip(self: Pin<&mut StateWrapper>, qubit: u64);
        fn phaseflip(self: Pin<&mut StateWrapper>, qubit: u64);
        fn Y(self: Pin<&mut StateWrapper>, qubit: u64);
        fn phaserot(self: Pin<&mut StateWrapper>, qubit: u64);
        fn SZdg(self: Pin<&mut StateWrapper>, qubit: u64);
        fn SY(self: Pin<&mut StateWrapper>, qubit: u64);
        fn SYdg(self: Pin<&mut StateWrapper>, qubit: u64);
        fn SX(self: Pin<&mut StateWrapper>, qubit: u64);
        fn SXdg(self: Pin<&mut StateWrapper>, qubit: u64);
        fn H2(self: Pin<&mut StateWrapper>, qubit: u64);
        fn H3(self: Pin<&mut StateWrapper>, qubit: u64);
        fn H4(self: Pin<&mut StateWrapper>, qubit: u64);
        fn H5(self: Pin<&mut StateWrapper>, qubit: u64);
        fn H6(self: Pin<&mut StateWrapper>, qubit: u64);
        fn F(self: Pin<&mut StateWrapper>, qubit: u64);
        fn F2(self: Pin<&mut StateWrapper>, qubit: u64);
        fn F3(self: Pin<&mut StateWrapper>, qubit: u64);
        fn F4(self: Pin<&mut StateWrapper>, qubit: u64);
        fn Fdg(self: Pin<&mut StateWrapper>, qubit: u64);
        fn F2dg(self: Pin<&mut StateWrapper>, qubit: u64);
        fn F3dg(self: Pin<&mut StateWrapper>, qubit: u64);
        fn F4dg(self: Pin<&mut StateWrapper>, qubit: u64);
        fn cx(self: Pin<&mut StateWrapper>, control: u64, target: u64);
        fn cy(self: Pin<&mut StateWrapper>, control: u64, target: u64);
        fn cz(self: Pin<&mut StateWrapper>, qubit1: u64, qubit2: u64);
        fn swap(self: Pin<&mut StateWrapper>, qubit1: u64, qubit2: u64);
        fn g2(self: Pin<&mut StateWrapper>, qubit1: u64, qubit2: u64);
        fn sxx(self: Pin<&mut StateWrapper>, qubit1: u64, qubit2: u64);
        fn sxxdg(self: Pin<&mut StateWrapper>, qubit1: u64, qubit2: u64);
        fn measure(
            self: Pin<&mut StateWrapper>,
            qubit: u64,
            forced_outcome: i32,
            collapse: bool,
        ) -> u32;

        // Tableau access methods
        fn get_num_qubits(self: &StateWrapper) -> u64;
        fn has_stab_x(self: &StateWrapper, gen_id: u64, qubit: u64) -> bool;
        fn has_stab_z(self: &StateWrapper, gen_id: u64, qubit: u64) -> bool;
        fn has_destab_x(self: &StateWrapper, gen_id: u64, qubit: u64) -> bool;
        fn has_destab_z(self: &StateWrapper, gen_id: u64, qubit: u64) -> bool;
        fn get_sign_minus(self: &StateWrapper, gen_id: u64) -> bool;
        fn get_sign_i(self: &StateWrapper, gen_id: u64) -> bool;
    }
}

/// A C++ sparse stabilizer state simulator wrapped for Rust
///
/// This is a wrapper around the C++ sparse simulator implementation,
/// providing the same interface as `SparseStab` but using the C++ backend.
pub struct CppSparseStab {
    state: cxx::UniquePtr<ffi::StateWrapper>,
    num_qubits: usize,
}

// SAFETY: CppSparseStab can be safely sent between threads because:
// 1. The C++ StateWrapper manages its own memory properly
// 2. Each instance is used by only one thread at a time
// 3. The underlying C++ code has no shared state between instances
// 4. cxx::UniquePtr provides exclusive ownership
unsafe impl Send for CppSparseStab {}

// SAFETY: CppSparseStab can be safely shared between threads because:
// 1. The underlying C++ StateWrapper is thread-safe for concurrent read access
// 2. Each instance maintains its own independent state
// 3. No global/shared mutable state is accessed
// 4. cxx::UniquePtr ensures exclusive ownership semantics
unsafe impl Sync for CppSparseStab {}

impl CppSparseStab {
    /// Create a new C++ sparse stabilizer simulator
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        let state = ffi::create_state_wrapper(num_qubits as u64, 0);
        // C++ constructor already initializes with random_device seed
        Self { state, num_qubits }
    }

    /// Create a new simulator with a specific seed
    ///
    /// # Panics
    ///
    /// Panics if the C++ state wrapper creation fails (should never happen in normal usage)
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        let mut state = ffi::create_state_wrapper(num_qubits as u64, 0);
        // Use the provided seed for C++ RNG, truncating to 32-bit as C++ expects
        #[allow(clippy::cast_possible_truncation)]
        let seed_u32 = seed as u32;
        state.as_mut().unwrap().set_seed(seed_u32);
        Self { state, num_qubits }
    }

    /// Create a new simulator with a specific seed (alias for `with_seed`)
    #[must_use]
    pub fn new_with_seed(num_qubits: usize, seed: u32) -> Self {
        Self::with_seed(num_qubits, u64::from(seed))
    }

    /// Get the number of qubits
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Set the RNG seed for this simulator instance
    ///
    /// # Panics
    ///
    /// Panics if the C++ state wrapper is not initialized (should never happen in normal usage)
    pub fn set_seed(&mut self, seed: u64) {
        // Truncate to 32-bit as C++ expects
        #[allow(clippy::cast_possible_truncation)]
        let seed_u32 = seed as u32;
        self.state.as_mut().unwrap().set_seed(seed_u32);
    }

    /// Internal helper for measurement
    fn internal_measure(
        &mut self,
        qubit: usize,
        forced_outcome: Option<bool>,
        collapse: bool,
    ) -> MeasurementResult {
        let forced = match forced_outcome {
            None => -1,
            Some(false) => 0,
            Some(true) => 1,
        };

        let outcome_raw = self
            .state
            .as_mut()
            .unwrap()
            .measure(qubit as u64, forced, collapse);
        let outcome = outcome_raw != 0;

        // Wrapper doesn't care about determinism - always return false
        MeasurementResult {
            outcome,
            is_deterministic: false,
        }
    }
}

impl QuantumSimulator for CppSparseStab {
    fn reset(&mut self) -> &mut Self {
        self.state.as_mut().unwrap().clear();
        // Don't reset the RNG - just reset the quantum state
        // This matches the behavior of the pure Rust simulator
        self
    }
}

impl CliffordGateable for CppSparseStab {
    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.state.as_mut().unwrap().hadamard(q.index() as u64);
        }
        self
    }

    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.state.as_mut().unwrap().phaserot(q.index() as u64);
        }
        self
    }

    fn cx(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "CX requires pairs of qubits"
        );
        for pair in qubits.chunks_exact(2) {
            let control = pair[0].index() as u64;
            let target = pair[1].index() as u64;
            self.state.as_mut().unwrap().cx(control, target);
        }
        self
    }

    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        qubits
            .iter()
            .map(|&q| self.internal_measure(q.index(), None, true))
            .collect()
    }

    // Override with native C++ implementations for better performance

    fn x(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.state.as_mut().unwrap().bitflip(q.index() as u64);
        }
        self
    }

    fn y(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.state.as_mut().unwrap().Y(q.index() as u64);
        }
        self
    }

    fn z(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.state.as_mut().unwrap().phaseflip(q.index() as u64);
        }
        self
    }

    fn szdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.state.as_mut().unwrap().SZdg(q.index() as u64);
        }
        self
    }

    fn sy(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.state.as_mut().unwrap().SY(q.index() as u64);
        }
        self
    }

    fn sydg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.state.as_mut().unwrap().SYdg(q.index() as u64);
        }
        self
    }

    fn sx(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.state.as_mut().unwrap().SX(q.index() as u64);
        }
        self
    }

    fn sxdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.state.as_mut().unwrap().SXdg(q.index() as u64);
        }
        self
    }

    fn h2(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.state.as_mut().unwrap().H2(q.index() as u64);
        }
        self
    }

    fn h3(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.state.as_mut().unwrap().H3(q.index() as u64);
        }
        self
    }

    fn h4(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.state.as_mut().unwrap().H4(q.index() as u64);
        }
        self
    }

    fn h5(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.state.as_mut().unwrap().H5(q.index() as u64);
        }
        self
    }

    fn h6(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.state.as_mut().unwrap().H6(q.index() as u64);
        }
        self
    }

    fn f(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.state.as_mut().unwrap().F(q.index() as u64);
        }
        self
    }

    fn fdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.state.as_mut().unwrap().Fdg(q.index() as u64);
        }
        self
    }

    fn f2(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.state.as_mut().unwrap().F2(q.index() as u64);
        }
        self
    }

    fn f2dg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.state.as_mut().unwrap().F2dg(q.index() as u64);
        }
        self
    }

    fn f3(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.state.as_mut().unwrap().F3(q.index() as u64);
        }
        self
    }

    fn f3dg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.state.as_mut().unwrap().F3dg(q.index() as u64);
        }
        self
    }

    fn f4(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.state.as_mut().unwrap().F4(q.index() as u64);
        }
        self
    }

    fn f4dg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.state.as_mut().unwrap().F4dg(q.index() as u64);
        }
        self
    }

    fn cy(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "CY requires pairs of qubits"
        );
        for pair in qubits.chunks_exact(2) {
            let control = pair[0].index() as u64;
            let target = pair[1].index() as u64;
            self.state.as_mut().unwrap().cy(control, target);
        }
        self
    }

    fn cz(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "CZ requires pairs of qubits"
        );
        for pair in qubits.chunks_exact(2) {
            let q1 = pair[0].index() as u64;
            let q2 = pair[1].index() as u64;
            self.state.as_mut().unwrap().cz(q1, q2);
        }
        self
    }

    fn swap(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "SWAP requires pairs of qubits"
        );
        for pair in qubits.chunks_exact(2) {
            let q1 = pair[0].index() as u64;
            let q2 = pair[1].index() as u64;
            self.state.as_mut().unwrap().swap(q1, q2);
        }
        self
    }
}

// Additional convenience methods
impl CppSparseStab {
    /// Force a specific measurement outcome on multiple qubits
    pub fn force_measure(&mut self, qubits: &[QubitId], outcome: bool) -> Vec<MeasurementResult> {
        qubits
            .iter()
            .map(|q| self.internal_measure(q.index(), Some(outcome), true))
            .collect()
    }

    /// Measure without collapsing the state
    pub fn peek_measure(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        qubits
            .iter()
            .map(|q| self.internal_measure(q.index(), None, false))
            .collect()
    }

    /// Apply G2 gate (CZ.H(1).H(2).CZ)
    ///
    /// # Panics
    ///
    /// Panics if the C++ state wrapper is not initialized (should never happen in normal usage)
    pub fn g2(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "G2 requires pairs of qubits"
        );
        for pair in qubits.chunks_exact(2) {
            let q1 = pair[0].index() as u64;
            let q2 = pair[1].index() as u64;
            self.state.as_mut().unwrap().g2(q1, q2);
        }
        self
    }

    /// Apply SXX gate (sqrt(XX))
    ///
    /// # Panics
    ///
    /// Panics if the C++ state wrapper is not initialized (should never happen in normal usage)
    pub fn sxx(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "SXX requires pairs of qubits"
        );
        for pair in qubits.chunks_exact(2) {
            let q1 = pair[0].index() as u64;
            let q2 = pair[1].index() as u64;
            self.state.as_mut().unwrap().sxx(q1, q2);
        }
        self
    }

    /// Apply `SXXdg` gate (sqrt(XX)†)
    ///
    /// # Panics
    ///
    /// Panics if the C++ state wrapper is not initialized (should never happen in normal usage)
    pub fn sxxdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        debug_assert!(
            qubits.len().is_multiple_of(2),
            "SXXdg requires pairs of qubits"
        );
        for pair in qubits.chunks_exact(2) {
            let q1 = pair[0].index() as u64;
            let q2 = pair[1].index() as u64;
            self.state.as_mut().unwrap().sxxdg(q1, q2);
        }
        self
    }

    /// Get the stabilizer tableau as a string
    #[must_use]
    pub fn stab_tableau(&self) -> String {
        self.format_generators(true)
    }

    /// Get the destabilizer tableau as a string
    #[must_use]
    pub fn destab_tableau(&self) -> String {
        self.format_generators(false)
    }

    /// Format generators into tableau string
    fn format_generators(&self, is_stab: bool) -> String {
        let mut result = String::new();
        let state_ref = self.state.as_ref().unwrap();
        // Safe to cast as num_qubits should never exceed usize::MAX
        #[allow(clippy::cast_possible_truncation)]
        let num_qubits = state_ref.get_num_qubits() as usize;

        for gen_id in 0..num_qubits {
            // Determine the sign of this generator
            let has_minus = state_ref.get_sign_minus(gen_id as u64);
            let has_i = state_ref.get_sign_i(gen_id as u64);

            let sign = match (has_minus, has_i) {
                (false, false) => "+", // +1
                (true, false) => "-",  // -1
                (false, true) => "i",  // +i
                (true, true) => "-i",  // -i
            };

            result.push_str(sign);

            // Build the Pauli string for this generator
            for qubit in 0..num_qubits {
                let (has_x, has_z) = if is_stab {
                    (
                        state_ref.has_stab_x(gen_id as u64, qubit as u64),
                        state_ref.has_stab_z(gen_id as u64, qubit as u64),
                    )
                } else {
                    (
                        state_ref.has_destab_x(gen_id as u64, qubit as u64),
                        state_ref.has_destab_z(gen_id as u64, qubit as u64),
                    )
                };

                let pauli = match (has_x, has_z) {
                    (true, true) => 'Y',   // Both X and Z -> Y
                    (true, false) => 'X',  // Only X -> X
                    (false, true) => 'Z',  // Only Z -> Z
                    (false, false) => 'I', // Neither -> I
                };

                result.push(pauli);
            }

            result.push('\n');
        }

        result
    }
}

// Implement StabilizerTableauSimulator trait
use pecos_qsim::StabilizerTableauSimulator;

impl StabilizerTableauSimulator for CppSparseStab {
    fn stab_tableau(&self) -> String {
        self.format_generators(true)
    }

    fn destab_tableau(&self) -> String {
        self.format_generators(false)
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}
