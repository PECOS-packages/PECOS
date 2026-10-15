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

use super::arbitrary_rotation_gateable::ArbitraryRotationGateable;
use super::clifford_gateable::{CliffordGateable, MeasurementResult};
use super::quantum_simulator::QuantumSimulator;
use pecos_core::{Angle64, QubitId, RngManageable};
use pecos_random::{PecosRng, Rng, RngProbabilityExt, SeedableRng, resolve_seed};

use core::fmt::Debug;

/// A quantum simulator that ignores all quantum operations and uses coin tosses for measurements
///
/// `CoinToss` is a minimal simulator that treats all quantum gates as no-ops and returns
/// random measurement results with a configurable probability. This is useful for:
/// - Debugging classical logic paths in quantum algorithms
/// - Testing error correction protocols with random noise
/// - Rapid prototyping where quantum coherence isn't important
///
/// # Type Parameters
/// * `R` - Random number generator type implementing `Rng + SeedableRng` traits
///
/// # Examples
/// ```rust
/// use pecos_simulators::CoinToss;
///
/// // Create a new 4-qubit coin toss simulator with 50% probability of measuring |1⟩
/// let mut sim = CoinToss::new(4);
///
/// // Create with custom probability and seed
/// let mut biased_sim = CoinToss::with_prob_and_seed(4, 0.8, Some(42));
/// ```
#[derive(Clone, Debug)]
pub struct CoinToss<R = PecosRng>
where
    R: Rng + SeedableRng + Debug,
{
    num_qubits: usize,
    prob: f64,
    rng: R,
}

impl CoinToss<PecosRng> {
    /// Create a new `CoinToss` simulator with default 50% measurement probability
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits in the system
    ///
    /// # Examples
    /// ```rust
    /// use pecos_simulators::CoinToss;
    /// let mut sim = CoinToss::new(4);
    /// ```
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self::with_prob_and_seed(num_qubits, 0.5, None)
    }

    /// Create a new `CoinToss` simulator with custom probability
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits in the system
    /// * `prob` - Probability of measuring |1⟩ (must be between 0.0 and 1.0)
    ///
    /// # Panics
    /// Panics if `prob` is not in the range [0.0, 1.0]
    ///
    /// # Examples
    /// ```rust
    /// use pecos_simulators::CoinToss;
    /// let mut sim = CoinToss::with_prob(4, 0.8); // 80% chance of measuring |1⟩
    /// ```
    #[must_use]
    pub fn with_prob(num_qubits: usize, prob: f64) -> Self {
        Self::with_prob_and_seed(num_qubits, prob, None)
    }

    /// Create a new `CoinToss` simulator with a specific seed
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits in the system
    /// * `seed` - Optional seed for reproducible randomness
    ///
    /// # Examples
    /// ```rust
    /// use pecos_simulators::CoinToss;
    /// let mut sim = CoinToss::with_seed(4, Some(42));
    /// ```
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: Option<u64>) -> Self {
        Self::with_prob_and_seed(num_qubits, 0.5, seed)
    }

    /// Create a new `CoinToss` simulator with custom probability and seed
    ///
    /// # Arguments
    /// * `num_qubits` - Number of qubits in the system
    /// * `prob` - Probability of measuring |1⟩ (must be between 0.0 and 1.0)
    /// * `seed` - Optional seed for reproducible randomness
    ///
    /// # Panics
    /// Panics if `prob` is not in the range [0.0, 1.0]
    ///
    /// # Examples
    /// ```rust
    /// use pecos_simulators::CoinToss;
    /// let mut sim = CoinToss::with_prob_and_seed(4, 0.3, Some(123));
    /// ```
    #[must_use]
    pub fn with_prob_and_seed(num_qubits: usize, prob: f64, seed: Option<u64>) -> Self {
        assert!(
            (0.0..=1.0).contains(&prob),
            "Probability must be between 0.0 and 1.0, got {prob}"
        );

        let rng = PecosRng::seed_from_u64(resolve_seed(seed));

        Self {
            num_qubits,
            prob,
            rng,
        }
    }
}

impl<R> CoinToss<R>
where
    R: Rng + SeedableRng + Debug,
{
    /// Returns the number of qubits in the system
    ///
    /// # Returns
    /// The number of qubits being simulated
    ///
    /// # Examples
    /// ```rust
    /// use pecos_simulators::CoinToss;
    /// let sim = CoinToss::new(5);
    /// assert_eq!(sim.num_qubits(), 5);
    /// ```
    #[inline]
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Get the current measurement probability
    ///
    /// # Returns
    /// The probability of measuring |1⟩ (between 0.0 and 1.0)
    ///
    /// # Examples
    /// ```rust
    /// use pecos_simulators::CoinToss;
    /// let sim = CoinToss::with_prob(3, 0.8);
    /// assert_eq!(sim.prob(), 0.8);
    /// ```
    #[inline]
    #[must_use]
    pub fn prob(&self) -> f64 {
        self.prob
    }

    /// Set the measurement probability
    ///
    /// # Arguments
    /// * `prob` - New probability (must be between 0.0 and 1.0)
    ///
    /// # Panics
    /// Panics if `prob` is not in the range [0.0, 1.0]
    ///
    /// # Examples
    /// ```rust
    /// use pecos_simulators::CoinToss;
    /// let mut sim = CoinToss::new(2);
    /// sim.set_prob(0.3);
    /// assert_eq!(sim.prob(), 0.3);
    /// ```
    pub fn set_prob(&mut self, prob: f64) {
        assert!(
            (0.0..=1.0).contains(&prob),
            "Probability must be between 0.0 and 1.0, got {prob}"
        );
        self.prob = prob;
    }

    /// Set seed for reproducible randomness
    ///
    /// This is similar to the Python `CoinToss` interface and `StateVec`'s seed functionality.
    ///
    /// # Arguments
    /// * `seed` - Seed value for the random number generator
    ///
    /// # Errors
    ///
    /// Returns an error if the seed cannot be set (currently never fails).
    ///
    /// # Examples
    /// ```rust
    /// use pecos_simulators::CoinToss;
    /// let mut sim = CoinToss::new(2);
    /// sim.set_seed(42);
    /// ```
    pub fn set_seed(&mut self, seed: u64) {
        let new_rng = R::seed_from_u64(seed);
        self.set_rng(new_rng);
    }
}

impl<R> QuantumSimulator for CoinToss<R>
where
    R: Rng + SeedableRng + Debug,
{
    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn reset(&mut self) -> &mut Self {
        // CoinToss is stateless, so reset is a no-op
        self
    }
}

impl<R> RngManageable for CoinToss<R>
where
    R: Rng + SeedableRng + Debug,
{
    type Rng = R;

    fn set_rng(&mut self, rng: Self::Rng) {
        self.rng = rng;
    }

    fn rng(&self) -> &Self::Rng {
        &self.rng
    }

    fn rng_mut(&mut self) -> &mut Self::Rng {
        &mut self.rng
    }
}

impl<R> CliffordGateable for CoinToss<R>
where
    R: Rng + SeedableRng + Debug,
{
    // All quantum gates are no-ops in CoinToss - they all return self for chaining
    fn h(&mut self, _qubits: &[QubitId]) -> &mut Self {
        self
    }
    fn sz(&mut self, _qubits: &[QubitId]) -> &mut Self {
        self
    }
    fn cx(&mut self, _pairs: &[(QubitId, QubitId)]) -> &mut Self {
        self
    }

    // Measurement returns random results based on the configured probability
    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        qubits
            .iter()
            .map(|_| MeasurementResult {
                outcome: self.rng.bernoulli(self.prob),
                is_deterministic: false,
            })
            .collect()
    }
}

impl<R> ArbitraryRotationGateable for CoinToss<R>
where
    R: Rng + SeedableRng + Debug,
{
    // All rotation gates are no-ops in CoinToss - they all return self for chaining
    fn rx(&mut self, _theta: Angle64, _qubits: &[QubitId]) -> &mut Self {
        self
    }
    fn rz(&mut self, _theta: Angle64, _qubits: &[QubitId]) -> &mut Self {
        self
    }
    fn rzz(&mut self, _theta: Angle64, _pairs: &[(QubitId, QubitId)]) -> &mut Self {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_coin_toss() {
        let sim = CoinToss::new(4);
        assert_eq!(sim.num_qubits(), 4);
        assert!((sim.prob() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_with_prob() {
        let sim = CoinToss::with_prob(3, 0.8);
        assert_eq!(sim.num_qubits(), 3);
        assert!((sim.prob() - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    #[should_panic(expected = "Probability must be between 0.0 and 1.0")]
    fn test_invalid_prob_high() {
        let _ = CoinToss::with_prob(2, 1.5);
    }

    #[test]
    #[should_panic(expected = "Probability must be between 0.0 and 1.0")]
    fn test_invalid_prob_low() {
        let _ = CoinToss::with_prob(2, -0.1);
    }

    #[test]
    fn test_with_seed_reproducible() {
        let mut sim1 = CoinToss::with_seed(2, Some(42));
        let mut sim2 = CoinToss::with_seed(2, Some(42));

        // Should produce identical sequences with same seed
        for _ in 0..10 {
            let result1 = sim1.mz(&[QubitId(0)]);
            let result2 = sim2.mz(&[QubitId(0)]);
            assert_eq!(result1[0].outcome, result2[0].outcome);
        }
    }

    #[test]
    fn test_prob_setter() {
        let mut sim = CoinToss::new(2);
        assert!((sim.prob() - 0.5).abs() < f64::EPSILON);

        sim.set_prob(0.9);
        assert!((sim.prob() - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_gates_are_noop() {
        let mut sim = CoinToss::new(2);

        // All gates should succeed and return self for chaining
        sim.h(&[QubitId(0)])
            .sz(&[QubitId(0)])
            .cx(&[(QubitId(0), QubitId(1))]);
        // If we get here without panic, gates work as expected
    }

    #[test]
    fn test_measurements_distribution() {
        let mut sim = CoinToss::with_prob_and_seed(1, 0.0, Some(42));

        // With prob=0.0, should always measure |0>
        for _ in 0..100 {
            assert!(!sim.mz(&[QubitId(0)])[0].outcome);
        }

        sim.set_prob(1.0);
        // With prob=1.0, should always measure |1>
        for _ in 0..100 {
            assert!(sim.mz(&[QubitId(0)])[0].outcome);
        }
    }

    #[test]
    fn test_reset_is_noop() {
        let mut sim = CoinToss::new(3);
        let prob_before = sim.prob();
        sim.reset();
        assert!((sim.prob() - prob_before).abs() < f64::EPSILON);
    }

    #[test]
    fn test_rotation_gates_are_noop() {
        let mut sim = CoinToss::new(2);

        // All rotation gates should succeed and return self for chaining
        sim.rx(Angle64::from_radians(1.5), &[QubitId(0)])
            .ry(Angle64::from_radians(0.5), &[QubitId(1)])
            .rz(Angle64::from_radians(2.1), &[QubitId(0)])
            .rzz(Angle64::from_radians(0.8), &[(QubitId(0), QubitId(1))]);
        // If we get here without panic, rotation gates work as expected
    }

    #[test]
    fn test_num_qubits() {
        let sim = CoinToss::new(5);
        assert_eq!(sim.num_qubits(), 5);
    }

    #[test]
    fn test_set_seed() {
        let mut sim1 = CoinToss::new(2);
        let mut sim2 = CoinToss::new(2);

        sim1.set_seed(123);
        sim2.set_seed(123);

        // Should produce identical sequences with same seed
        for _ in 0..10 {
            let result1 = sim1.mz(&[QubitId(0)]);
            let result2 = sim2.mz(&[QubitId(0)]);
            assert_eq!(result1[0].outcome, result2[0].outcome);
        }
    }
}
