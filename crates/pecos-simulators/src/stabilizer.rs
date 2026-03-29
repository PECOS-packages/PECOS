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

//! Default stabilizer simulator with automatic implementation selection.
//!
//! [`Stabilizer`] is the recommended stabilizer simulator for most use cases. It automatically
//! selects the best underlying implementation based on the number of qubits and workload
//! characteristics.
//!
//! # Example
//!
//! ```rust
//! use pecos_core::{QubitId, qid};
//! use pecos_simulators::{Stabilizer, CliffordGateable, QuantumSimulator};
//!
//! // Create a stabilizer simulator
//! let mut sim = Stabilizer::new(2);
//!
//! // Create a Bell state
//! sim.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]);
//!
//! // Measure
//! let results = sim.mz(&[QubitId(0), QubitId(1)]);
//! assert_eq!(results[0].outcome, results[1].outcome);
//! ```
//!
//! # Implementation Selection
//!
//! Currently uses [`SparseStab`] (BitSet-based sparse row+column representation),
//! which benchmarks show is fastest for QEC workloads at realistic distances
//! (d >= 11). The selection logic may be refined in future versions based on
//! qubit count and workload patterns.

use crate::{
    CliffordGateable, MeasurementResult, QuantumSimulator, SparseStab, StabilizerTableauSimulator,
};
use pecos_core::{QubitId, RngManageable};
use pecos_random::PecosRng;

/// Default stabilizer simulator with automatic implementation selection.
///
/// This is the recommended stabilizer simulator for most use cases. It provides
/// efficient Clifford circuit simulation by automatically selecting the best
/// underlying implementation.
///
/// See the [module documentation](self) for more details.
#[derive(Debug, Clone)]
pub struct Stabilizer {
    inner: SparseStab,
}

impl Stabilizer {
    /// Create a new stabilizer simulator with the given number of qubits.
    ///
    /// All qubits are initialized in the |0⟩ state.
    ///
    /// # Example
    ///
    /// ```rust
    /// use pecos_simulators::Stabilizer;
    ///
    /// let sim = Stabilizer::new(10);
    /// assert_eq!(sim.num_qubits(), 10);
    /// ```
    #[inline]
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self {
            inner: SparseStab::new(num_qubits),
        }
    }

    /// Create a new stabilizer simulator with a specific RNG seed.
    ///
    /// Using the same seed guarantees reproducible measurement outcomes.
    ///
    /// # Example
    ///
    /// ```rust
    /// use pecos_simulators::Stabilizer;
    ///
    /// let sim = Stabilizer::with_seed(10, 42);
    /// ```
    #[inline]
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self {
            inner: SparseStab::with_seed(num_qubits, seed),
        }
    }

    /// Returns the number of qubits in this simulator.
    #[inline]
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Returns generator data as sparse index vectors.
    ///
    /// Returns `(col_x, col_z, row_x, row_z)` where each is a `Vec<Vec<usize>>`.
    #[must_use]
    pub fn gens_data(&self, is_stab: bool) -> crate::GensData {
        self.inner.gens_data(is_stab)
    }
}

impl QuantumSimulator for Stabilizer {
    #[inline]
    fn reset(&mut self) -> &mut Self {
        self.inner.reset();
        self
    }
}

impl CliffordGateable for Stabilizer {
    #[inline]
    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.inner.h(qubits);
        self
    }

    #[inline]
    fn x(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.inner.x(qubits);
        self
    }

    #[inline]
    fn y(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.inner.y(qubits);
        self
    }

    #[inline]
    fn z(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.inner.z(qubits);
        self
    }

    #[inline]
    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.inner.sz(qubits);
        self
    }

    #[inline]
    fn szdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.inner.szdg(qubits);
        self
    }

    #[inline]
    fn cx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        self.inner.cx(pairs);
        self
    }

    #[inline]
    fn cz(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        self.inner.cz(pairs);
        self
    }

    #[inline]
    fn swap(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        self.inner.swap(pairs);
        self
    }

    #[inline]
    fn mx(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        self.inner.mx(qubits)
    }

    #[inline]
    fn my(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        self.inner.my(qubits)
    }

    #[inline]
    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        self.inner.mz(qubits)
    }
}

impl RngManageable for Stabilizer {
    type Rng = PecosRng;

    #[inline]
    fn set_rng(&mut self, rng: Self::Rng) {
        self.inner.set_rng(rng);
    }

    #[inline]
    fn rng(&self) -> &Self::Rng {
        self.inner.rng()
    }

    #[inline]
    fn rng_mut(&mut self) -> &mut Self::Rng {
        self.inner.rng_mut()
    }
}

impl StabilizerTableauSimulator for Stabilizer {
    fn stab_tableau(&self) -> String {
        self.inner.stab_tableau()
    }

    fn destab_tableau(&self) -> String {
        self.inner.destab_tableau()
    }

    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }
}

// ============================================================================
// ForcedMeasurement and StabilizerSimulator implementations
// ============================================================================

use crate::stabilizer_test_utils::{ForcedMeasurement, StabilizerSimulator};

impl ForcedMeasurement for Stabilizer {
    fn mz_forced(&mut self, qubit: usize, forced_outcome: bool) -> MeasurementResult {
        self.inner.mz_forced(qubit, forced_outcome)
    }
}

impl StabilizerSimulator for Stabilizer {
    fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self::with_seed(num_qubits, seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stabilizer_test_utils::run_full_stabilizer_test_suite;

    #[test]
    fn test_stab_basic() {
        let mut sim = Stabilizer::new(2);
        sim.h(&[QubitId(0)]);
        sim.cx(&[(QubitId(0), QubitId(1))]);
        let results = sim.mz(&[QubitId(0), QubitId(1)]);
        assert_eq!(results[0].outcome, results[1].outcome);
    }

    #[test]
    fn test_stab_full_suite() {
        let mut sim = Stabilizer::with_seed(8, 42);
        run_full_stabilizer_test_suite(&mut sim, 8);
    }
}
