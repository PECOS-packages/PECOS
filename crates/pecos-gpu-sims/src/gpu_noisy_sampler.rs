//! GPU Noisy Circuit Sampler
//!
//! Provides a way to run multiple shots of noisy circuits using the single-shot
//! GPU stabilizer simulator. This allows using sophisticated noise models while
//! leveraging GPU acceleration.
//!
//! # Performance
//!
//! GPU resources are reused across shots - only the tableau state is reset between
//! shots. For simple depolarizing noise with even better performance through true
//! GPU-side batching, use `GpuStabMulti` with `enable_noise()` instead.
//!
//! # Usage
//!
//! ```
//! use pecos_gpu_sims::{GpuNoisySampler, DepolarizingNoiseSampler, CircuitBuilder};
//!
//! let noise = DepolarizingNoiseSampler::new(0.001, 0.01, 0.005);
//! let mut sampler = GpuNoisySampler::new(5, noise);
//!
//! let results = sampler.sample(1000, |c: &mut CircuitBuilder| {
//!     c.h(0);
//!     c.noise_1q(0);  // Inject noise on qubit 0
//!     c.cx(0, 1);
//!     c.noise_2q(0, 1);  // Inject noise on both qubits
//!     c.mz(0);
//!     c.mz(1);
//! });
//! ```

use crate::GpuStab;
use pecos_core::QubitId;
use pecos_qsim::{CliffordGateable, QuantumSimulator};
use pecos_rng::{PecosRng, SeedableRng, time_seed};
use std::fmt::Debug;

/// Represents a Pauli operator for noise injection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pauli {
    I, // Identity (no error)
    X,
    Y,
    Z,
}

/// Simplified noise sampling trait for GPU circuit execution.
///
/// This trait provides a simple interface for sampling noise without the
/// complexity of the full `ControlEngine` interface.
pub trait NoiseSampler: Send {
    /// Sample a single-qubit error. Returns the Pauli to apply (I = no error).
    fn sample_1q(&mut self, qubit: usize) -> Pauli;

    /// Sample a two-qubit error. Returns Paulis to apply to each qubit.
    fn sample_2q(&mut self, qubit_a: usize, qubit_b: usize) -> (Pauli, Pauli);

    /// Sample a measurement error. Returns true if the outcome should be flipped.
    ///
    /// For biased noise models, call `set_measurement_outcome` before this method
    /// to enable outcome-dependent error rates.
    fn sample_meas(&mut self, qubit: usize) -> bool;

    /// Set the actual measurement outcome for biased error sampling.
    ///
    /// This should be called before `sample_meas` when using noise models with
    /// asymmetric measurement errors (e.g., `BiasedDepolarizingNoiseSampler`).
    /// The default implementation is a no-op for symmetric noise models.
    fn set_measurement_outcome(&mut self, _qubit: usize, _outcome: bool) {
        // Default no-op for symmetric noise models
    }

    /// Reset/reseed the noise sampler for a new shot.
    fn reseed(&mut self, seed: u64);

    /// Clone the sampler (for parallel execution).
    fn clone_box(&self) -> Box<dyn NoiseSampler>;
}

/// A simple depolarizing noise sampler.
///
/// Applies random Pauli errors with specified probabilities:
/// - `p1`: Single-qubit gate error probability (applies X, Y, or Z with equal probability)
/// - `p2`: Two-qubit gate error probability (applies one of 15 non-identity Paulis)
/// - `p_meas`: Measurement bit-flip probability
pub struct DepolarizingNoiseSampler {
    p1: f64,
    p2: f64,
    p_meas: f64,
    rng: PecosRng,
}

impl Clone for DepolarizingNoiseSampler {
    fn clone(&self) -> Self {
        Self {
            p1: self.p1,
            p2: self.p2,
            p_meas: self.p_meas,
            rng: PecosRng::from_rng(&mut rand::rng()),
        }
    }
}

impl DepolarizingNoiseSampler {
    /// Create a new depolarizing noise sampler.
    #[must_use]
    pub fn new(p1: f64, p2: f64, p_meas: f64) -> Self {
        Self {
            p1,
            p2,
            p_meas,
            rng: PecosRng::seed_from_u64(time_seed()),
        }
    }

    /// Create with a specific seed for reproducibility.
    #[must_use]
    pub fn with_seed(p1: f64, p2: f64, p_meas: f64, seed: u64) -> Self {
        Self {
            p1,
            p2,
            p_meas,
            rng: PecosRng::seed_from_u64(seed),
        }
    }

    fn random_pauli(&mut self) -> Pauli {
        match self.rng.next_u32() % 3 {
            0 => Pauli::X,
            1 => Pauli::Y,
            _ => Pauli::Z,
        }
    }

    fn occurs(&mut self, probability: f64) -> bool {
        let threshold = (probability * u64::MAX as f64) as u64;
        self.rng.next_u64() < threshold
    }
}

impl NoiseSampler for DepolarizingNoiseSampler {
    fn sample_1q(&mut self, _qubit: usize) -> Pauli {
        if self.occurs(self.p1) {
            self.random_pauli()
        } else {
            Pauli::I
        }
    }

    fn sample_2q(&mut self, _qubit_a: usize, _qubit_b: usize) -> (Pauli, Pauli) {
        if self.occurs(self.p2) {
            // Sample one of 15 non-identity two-qubit Paulis
            // Each qubit independently gets I, X, Y, or Z, excluding II
            let selection = self.rng.next_u32() % 15;
            let pauli_a = match selection / 4 {
                0 => Pauli::I,
                1 => Pauli::X,
                2 => Pauli::Y,
                _ => Pauli::Z,
            };
            let pauli_b = match selection % 4 {
                0 if pauli_a == Pauli::I => Pauli::X, // Avoid II
                0 => Pauli::I,
                1 => Pauli::X,
                2 => Pauli::Y,
                _ => Pauli::Z,
            };
            (pauli_a, pauli_b)
        } else {
            (Pauli::I, Pauli::I)
        }
    }

    fn sample_meas(&mut self, _qubit: usize) -> bool {
        self.occurs(self.p_meas)
    }

    fn reseed(&mut self, seed: u64) {
        self.rng = PecosRng::seed_from_u64(seed);
    }

    fn clone_box(&self) -> Box<dyn NoiseSampler> {
        Box::new(self.clone())
    }
}

/// A biased depolarizing noise sampler with asymmetric measurement errors.
pub struct BiasedDepolarizingNoiseSampler {
    p1: f64,
    p2: f64,
    p_meas_0: f64, // Probability of flipping 0 -> 1
    p_meas_1: f64, // Probability of flipping 1 -> 0
    rng: PecosRng,
    /// Cached measurement outcomes for biased flip (set during measurement)
    pending_meas_outcomes: Vec<bool>,
}

impl Clone for BiasedDepolarizingNoiseSampler {
    fn clone(&self) -> Self {
        Self {
            p1: self.p1,
            p2: self.p2,
            p_meas_0: self.p_meas_0,
            p_meas_1: self.p_meas_1,
            rng: PecosRng::from_rng(&mut rand::rng()),
            pending_meas_outcomes: self.pending_meas_outcomes.clone(),
        }
    }
}

impl BiasedDepolarizingNoiseSampler {
    /// Create a new biased depolarizing noise sampler.
    #[must_use]
    pub fn new(p1: f64, p2: f64, p_meas_0: f64, p_meas_1: f64) -> Self {
        Self {
            p1,
            p2,
            p_meas_0,
            p_meas_1,
            rng: PecosRng::seed_from_u64(time_seed()),
            pending_meas_outcomes: Vec::new(),
        }
    }

    /// Create with a specific seed for reproducibility.
    #[must_use]
    pub fn with_seed(p1: f64, p2: f64, p_meas_0: f64, p_meas_1: f64, seed: u64) -> Self {
        Self {
            p1,
            p2,
            p_meas_0,
            p_meas_1,
            rng: PecosRng::seed_from_u64(seed),
            pending_meas_outcomes: Vec::new(),
        }
    }

    /// Set the actual measurement outcome for biased error sampling.
    /// This should be called after the actual measurement to determine if bias applies.
    pub fn set_measurement_outcome(&mut self, outcome: bool) {
        self.pending_meas_outcomes.push(outcome);
    }

    fn random_pauli(&mut self) -> Pauli {
        match self.rng.next_u32() % 3 {
            0 => Pauli::X,
            1 => Pauli::Y,
            _ => Pauli::Z,
        }
    }

    fn occurs(&mut self, probability: f64) -> bool {
        let threshold = (probability * u64::MAX as f64) as u64;
        self.rng.next_u64() < threshold
    }
}

impl NoiseSampler for BiasedDepolarizingNoiseSampler {
    fn sample_1q(&mut self, _qubit: usize) -> Pauli {
        if self.occurs(self.p1) {
            self.random_pauli()
        } else {
            Pauli::I
        }
    }

    fn sample_2q(&mut self, _qubit_a: usize, _qubit_b: usize) -> (Pauli, Pauli) {
        if self.occurs(self.p2) {
            let selection = self.rng.next_u32() % 15;
            let pauli_a = match selection / 4 {
                0 => Pauli::I,
                1 => Pauli::X,
                2 => Pauli::Y,
                _ => Pauli::Z,
            };
            let pauli_b = match selection % 4 {
                0 if pauli_a == Pauli::I => Pauli::X,
                0 => Pauli::I,
                1 => Pauli::X,
                2 => Pauli::Y,
                _ => Pauli::Z,
            };
            (pauli_a, pauli_b)
        } else {
            (Pauli::I, Pauli::I)
        }
    }

    fn sample_meas(&mut self, _qubit: usize) -> bool {
        // For biased noise, we need to know the actual outcome
        // Use the pending outcome if available, otherwise assume uniform error
        if let Some(outcome) = self.pending_meas_outcomes.pop() {
            let p = if outcome {
                self.p_meas_1
            } else {
                self.p_meas_0
            };
            self.occurs(p)
        } else {
            // Fallback: use average of the two probabilities
            self.occurs(f64::midpoint(self.p_meas_0, self.p_meas_1))
        }
    }

    fn set_measurement_outcome(&mut self, _qubit: usize, outcome: bool) {
        self.pending_meas_outcomes.push(outcome);
    }

    fn reseed(&mut self, seed: u64) {
        self.rng = PecosRng::seed_from_u64(seed);
        self.pending_meas_outcomes.clear();
    }

    fn clone_box(&self) -> Box<dyn NoiseSampler> {
        Box::new(self.clone())
    }
}

/// Operations that can be added to a circuit.
#[derive(Debug, Clone)]
pub enum CircuitOp {
    // Single-qubit gates
    H(usize),
    S(usize),
    Sdg(usize),
    X(usize),
    Y(usize),
    Z(usize),

    // Two-qubit gates
    Cx(usize, usize),
    Cz(usize, usize),
    Swap(usize, usize),

    // Measurements
    Mz(usize),

    // Noise injection points
    Noise1Q(usize),
    Noise2Q(usize, usize),
}

/// Builder for constructing circuits with noise injection points.
#[derive(Default)]
pub struct CircuitBuilder {
    ops: Vec<CircuitOp>,
}

impl CircuitBuilder {
    /// Create a new empty circuit builder.
    #[must_use]
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    /// Hadamard gate.
    pub fn h(&mut self, qubit: usize) -> &mut Self {
        self.ops.push(CircuitOp::H(qubit));
        self
    }

    /// S gate (sqrt Z).
    pub fn s(&mut self, qubit: usize) -> &mut Self {
        self.ops.push(CircuitOp::S(qubit));
        self
    }

    /// S-dagger gate.
    pub fn sdg(&mut self, qubit: usize) -> &mut Self {
        self.ops.push(CircuitOp::Sdg(qubit));
        self
    }

    /// Pauli X gate.
    pub fn x(&mut self, qubit: usize) -> &mut Self {
        self.ops.push(CircuitOp::X(qubit));
        self
    }

    /// Pauli Y gate.
    pub fn y(&mut self, qubit: usize) -> &mut Self {
        self.ops.push(CircuitOp::Y(qubit));
        self
    }

    /// Pauli Z gate.
    pub fn z(&mut self, qubit: usize) -> &mut Self {
        self.ops.push(CircuitOp::Z(qubit));
        self
    }

    /// CNOT gate.
    pub fn cx(&mut self, control: usize, target: usize) -> &mut Self {
        self.ops.push(CircuitOp::Cx(control, target));
        self
    }

    /// CZ gate.
    pub fn cz(&mut self, qubit_a: usize, qubit_b: usize) -> &mut Self {
        self.ops.push(CircuitOp::Cz(qubit_a, qubit_b));
        self
    }

    /// SWAP gate.
    pub fn swap(&mut self, qubit_a: usize, qubit_b: usize) -> &mut Self {
        self.ops.push(CircuitOp::Swap(qubit_a, qubit_b));
        self
    }

    /// Measure qubit in Z basis.
    pub fn mz(&mut self, qubit: usize) -> &mut Self {
        self.ops.push(CircuitOp::Mz(qubit));
        self
    }

    /// Mark a noise injection point for single-qubit noise.
    pub fn noise_1q(&mut self, qubit: usize) -> &mut Self {
        self.ops.push(CircuitOp::Noise1Q(qubit));
        self
    }

    /// Mark a noise injection point for two-qubit noise.
    pub fn noise_2q(&mut self, qubit_a: usize, qubit_b: usize) -> &mut Self {
        self.ops.push(CircuitOp::Noise2Q(qubit_a, qubit_b));
        self
    }

    /// Get the operations in this circuit.
    #[must_use]
    pub fn ops(&self) -> &[CircuitOp] {
        &self.ops
    }

    /// Clear the circuit for reuse.
    pub fn clear(&mut self) {
        self.ops.clear();
    }
}

/// Result from a single shot of a noisy circuit.
#[derive(Debug, Clone)]
pub struct ShotResult {
    /// Measurement outcomes in order of measurement operations.
    pub outcomes: Vec<bool>,
}

/// GPU-accelerated noisy circuit sampler.
///
/// Runs multiple shots of a circuit with noise injection using the single-shot
/// GPU stabilizer simulator. Each shot samples fresh noise from the provided
/// noise model.
pub struct GpuNoisySampler<N: NoiseSampler> {
    num_qubits: usize,
    noise_sampler: N,
    master_rng: PecosRng,
}

impl<N: NoiseSampler> GpuNoisySampler<N> {
    /// Create a new GPU noisy sampler.
    pub fn new(num_qubits: usize, noise_sampler: N) -> Self {
        Self {
            num_qubits,
            noise_sampler,
            master_rng: PecosRng::seed_from_u64(time_seed()),
        }
    }

    /// Create with a specific seed for reproducibility.
    pub fn with_seed(num_qubits: usize, noise_sampler: N, seed: u64) -> Self {
        Self {
            num_qubits,
            noise_sampler,
            master_rng: PecosRng::seed_from_u64(seed),
        }
    }

    /// Sample multiple shots of a circuit with noise.
    ///
    /// The `circuit_fn` is called once to build the circuit, which is then
    /// executed for each shot with fresh noise samples.
    pub fn sample<F>(&mut self, shots: usize, circuit_fn: F) -> Result<Vec<ShotResult>, String>
    where
        F: Fn(&mut CircuitBuilder),
    {
        // Build the circuit once
        let mut builder = CircuitBuilder::new();
        circuit_fn(&mut builder);
        let ops = builder.ops().to_vec();

        let mut results = Vec::with_capacity(shots);

        // Create GPU simulator once and reuse - avoids expensive GPU resource creation per shot
        let initial_seed = self.master_rng.next_u64();
        let mut sim = GpuStab::<PecosRng>::with_seed(self.num_qubits, initial_seed)
            .map_err(|e| format!("Failed to create GPU simulator: {e}"))?;

        for _ in 0..shots {
            // Reseed noise sampler for this shot
            let shot_seed = self.master_rng.next_u64();
            self.noise_sampler.reseed(shot_seed);

            // Reset tableau state for new shot (GPU resources are reused)
            sim.reset();

            let mut outcomes = Vec::new();

            // Execute the circuit with noise injection
            for op in &ops {
                match op {
                    CircuitOp::H(q) => {
                        sim.h(&[QubitId(*q)]);
                    }
                    CircuitOp::S(q) => {
                        sim.sz(&[QubitId(*q)]);
                    }
                    CircuitOp::Sdg(q) => {
                        sim.szdg(&[QubitId(*q)]);
                    }
                    CircuitOp::X(q) => {
                        sim.x(&[QubitId(*q)]);
                    }
                    CircuitOp::Y(q) => {
                        sim.y(&[QubitId(*q)]);
                    }
                    CircuitOp::Z(q) => {
                        sim.z(&[QubitId(*q)]);
                    }
                    CircuitOp::Cx(ctrl, tgt) => {
                        sim.cx(&[QubitId(*ctrl), QubitId(*tgt)]);
                    }
                    CircuitOp::Cz(a, b) => {
                        sim.cz(&[QubitId(*a), QubitId(*b)]);
                    }
                    CircuitOp::Swap(a, b) => {
                        sim.swap(&[QubitId(*a), QubitId(*b)]);
                    }
                    CircuitOp::Mz(q) => {
                        let results = sim.mz(&[QubitId(*q)]);
                        let mut outcome = results[0].outcome;

                        // Set outcome for biased noise models before sampling
                        self.noise_sampler.set_measurement_outcome(*q, outcome);

                        // Apply measurement noise
                        if self.noise_sampler.sample_meas(*q) {
                            outcome = !outcome;
                        }

                        outcomes.push(outcome);
                    }
                    CircuitOp::Noise1Q(q) => {
                        // Sample and apply single-qubit noise
                        match self.noise_sampler.sample_1q(*q) {
                            Pauli::I => {}
                            Pauli::X => {
                                sim.x(&[QubitId(*q)]);
                            }
                            Pauli::Y => {
                                sim.y(&[QubitId(*q)]);
                            }
                            Pauli::Z => {
                                sim.z(&[QubitId(*q)]);
                            }
                        }
                    }
                    CircuitOp::Noise2Q(a, b) => {
                        // Sample and apply two-qubit noise
                        let (pa, pb) = self.noise_sampler.sample_2q(*a, *b);
                        match pa {
                            Pauli::I => {}
                            Pauli::X => {
                                sim.x(&[QubitId(*a)]);
                            }
                            Pauli::Y => {
                                sim.y(&[QubitId(*a)]);
                            }
                            Pauli::Z => {
                                sim.z(&[QubitId(*a)]);
                            }
                        }
                        match pb {
                            Pauli::I => {}
                            Pauli::X => {
                                sim.x(&[QubitId(*b)]);
                            }
                            Pauli::Y => {
                                sim.y(&[QubitId(*b)]);
                            }
                            Pauli::Z => {
                                sim.z(&[QubitId(*b)]);
                            }
                        }
                    }
                }
            }

            results.push(ShotResult { outcomes });
        }

        Ok(results)
    }

    /// Get the number of qubits.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_depolarizing_sampler() {
        let mut sampler = DepolarizingNoiseSampler::with_seed(0.5, 0.5, 0.5, 42);

        // With 50% probability, we should see some errors
        let mut errors_1q = 0;
        let mut errors_2q = 0;
        let mut errors_meas = 0;

        for _ in 0..100 {
            if sampler.sample_1q(0) != Pauli::I {
                errors_1q += 1;
            }
            let (pa, pb) = sampler.sample_2q(0, 1);
            if pa != Pauli::I || pb != Pauli::I {
                errors_2q += 1;
            }
            if sampler.sample_meas(0) {
                errors_meas += 1;
            }
        }

        // Should have roughly 50% errors for each
        assert!(errors_1q > 30 && errors_1q < 70, "1Q errors: {errors_1q}");
        assert!(errors_2q > 30 && errors_2q < 70, "2Q errors: {errors_2q}");
        assert!(
            errors_meas > 30 && errors_meas < 70,
            "Meas errors: {errors_meas}"
        );
    }

    #[test]
    fn test_circuit_builder() {
        let mut builder = CircuitBuilder::new();
        builder.h(0).noise_1q(0).cx(0, 1).noise_2q(0, 1).mz(0).mz(1);

        assert_eq!(builder.ops().len(), 6);
        assert!(matches!(builder.ops()[0], CircuitOp::H(0)));
        assert!(matches!(builder.ops()[1], CircuitOp::Noise1Q(0)));
        assert!(matches!(builder.ops()[2], CircuitOp::Cx(0, 1)));
        assert!(matches!(builder.ops()[3], CircuitOp::Noise2Q(0, 1)));
        assert!(matches!(builder.ops()[4], CircuitOp::Mz(0)));
        assert!(matches!(builder.ops()[5], CircuitOp::Mz(1)));
    }

    #[test]
    fn test_circuit_builder_all_gates() {
        // Test all gate types in CircuitBuilder
        let mut builder = CircuitBuilder::new();
        builder
            .h(0)
            .s(0)
            .sdg(0)
            .x(0)
            .y(0)
            .z(0)
            .cx(0, 1)
            .cz(0, 1)
            .swap(0, 1)
            .noise_1q(0)
            .noise_2q(0, 1)
            .mz(0);

        assert_eq!(builder.ops().len(), 12);
        assert!(matches!(builder.ops()[0], CircuitOp::H(0)));
        assert!(matches!(builder.ops()[1], CircuitOp::S(0)));
        assert!(matches!(builder.ops()[2], CircuitOp::Sdg(0)));
        assert!(matches!(builder.ops()[3], CircuitOp::X(0)));
        assert!(matches!(builder.ops()[4], CircuitOp::Y(0)));
        assert!(matches!(builder.ops()[5], CircuitOp::Z(0)));
        assert!(matches!(builder.ops()[6], CircuitOp::Cx(0, 1)));
        assert!(matches!(builder.ops()[7], CircuitOp::Cz(0, 1)));
        assert!(matches!(builder.ops()[8], CircuitOp::Swap(0, 1)));
        assert!(matches!(builder.ops()[9], CircuitOp::Noise1Q(0)));
        assert!(matches!(builder.ops()[10], CircuitOp::Noise2Q(0, 1)));
        assert!(matches!(builder.ops()[11], CircuitOp::Mz(0)));

        // Test clear
        builder.clear();
        assert_eq!(builder.ops().len(), 0);
    }

    #[test]
    fn test_gpu_noisy_sampler_all_gates() {
        // Test that all gates execute correctly in GpuNoisySampler
        let noise = DepolarizingNoiseSampler::with_seed(0.0, 0.0, 0.0, 42);
        let mut sampler = GpuNoisySampler::with_seed(3, noise, 123);

        // Test swap gate: put qubit 0 in |1>, swap with qubit 1, measure both
        let results = sampler
            .sample(10, |c| {
                c.x(0); // qubit 0 = |1>
                c.swap(0, 1); // now qubit 0 = |0>, qubit 1 = |1>
                c.mz(0);
                c.mz(1);
            })
            .unwrap();

        for result in &results {
            assert!(!result.outcomes[0], "Qubit 0 should be |0> after swap");
            assert!(result.outcomes[1], "Qubit 1 should be |1> after swap");
        }

        // Test cz gate: creates phase, no bit flip effect on computational basis
        let results = sampler
            .sample(10, |c| {
                c.x(0); // qubit 0 = |1>
                c.x(1); // qubit 1 = |1>
                c.cz(0, 1); // applies phase, no change to |11>
                c.mz(0);
                c.mz(1);
            })
            .unwrap();

        for result in &results {
            assert!(result.outcomes[0], "Qubit 0 should still be |1>");
            assert!(result.outcomes[1], "Qubit 1 should still be |1>");
        }

        // Test S gate: S*S = Z, which on |+> gives |->
        // H puts qubit in |+>, S*S*H = H*Z puts in |->
        // Measuring |-> in Z basis gives random results, but in X basis (H then Z) gives 1
        let results = sampler
            .sample(10, |c| {
                c.h(0); // |+>
                c.s(0);
                c.s(0); // S*S = Z, so now |->
                c.h(0); // H|-> = |1>
                c.mz(0);
            })
            .unwrap();

        for result in &results {
            assert!(result.outcomes[0], "H*S*S*H|0> = H*Z*H|0> = X|0> = |1>");
        }

        // Test Sdg gate: Sdg*Sdg = Z
        let results = sampler
            .sample(10, |c| {
                c.h(0);
                c.sdg(0);
                c.sdg(0);
                c.h(0);
                c.mz(0);
            })
            .unwrap();

        for result in &results {
            assert!(result.outcomes[0], "H*Sdg*Sdg*H|0> = H*Z*H|0> = X|0> = |1>");
        }

        // Test Y gate: Y|0> = i|1>, measuring gives 1
        let results = sampler
            .sample(10, |c| {
                c.y(0);
                c.mz(0);
            })
            .unwrap();

        for result in &results {
            assert!(result.outcomes[0], "Y|0> should measure as 1");
        }

        // Test Z gate: Z|0> = |0>
        let results = sampler
            .sample(10, |c| {
                c.z(0);
                c.mz(0);
            })
            .unwrap();

        for result in &results {
            assert!(!result.outcomes[0], "Z|0> should measure as 0");
        }
    }

    #[test]
    fn test_gpu_noisy_sampler_noiseless() {
        // With zero noise, deterministic measurement should always give 0
        let noise = DepolarizingNoiseSampler::with_seed(0.0, 0.0, 0.0, 42);
        let mut sampler = GpuNoisySampler::with_seed(2, noise, 123);

        let results = sampler
            .sample(100, |c| {
                // Qubit 0 starts in |0>, qubit 1 starts in |0>
                // No gates, just measure - should always give 0
                c.mz(0);
                c.mz(1);
            })
            .unwrap();

        // All measurements should be 0 (deterministic)
        for result in &results {
            assert_eq!(result.outcomes.len(), 2);
            assert!(!result.outcomes[0], "Qubit 0 in |0> should measure 0");
            assert!(!result.outcomes[1], "Qubit 1 in |0> should measure 0");
        }
    }

    #[test]
    fn test_gpu_noisy_sampler_bell_state() {
        // Test that GpuNoisySampler correctly handles Bell state correlations
        // This relies on GpuStab's non-deterministic measurement properly updating the tableau
        let noise = DepolarizingNoiseSampler::with_seed(0.0, 0.0, 0.0, 42); // No noise
        let mut sampler = GpuNoisySampler::with_seed(2, noise, 123);

        let results = sampler
            .sample(100, |c| {
                // Create Bell state: |00> + |11>
                c.h(0);
                c.cx(0, 1);
                c.mz(0);
                c.mz(1);
            })
            .unwrap();

        // Verify that outcomes are perfectly correlated: both 0 or both 1
        let mut correlated_count = 0;
        for result in &results {
            if result.outcomes[0] == result.outcomes[1] {
                correlated_count += 1;
            }
        }

        assert_eq!(
            correlated_count,
            results.len(),
            "Bell state measurements should be 100% correlated, got {}%",
            correlated_count * 100 / results.len()
        );

        // Also verify we see a roughly 50/50 split of |00> vs |11>
        let ones_count = results.iter().filter(|r| r.outcomes[0]).count();
        assert!(
            ones_count > 20 && ones_count < 80,
            "Expected roughly 50/50 split for Bell state, got {} ones out of {}",
            ones_count,
            results.len()
        );
    }

    #[test]
    fn test_gpu_noisy_sampler_with_noise() {
        // With high noise, some shots should have errors
        let noise = DepolarizingNoiseSampler::with_seed(0.0, 0.0, 0.5, 42);
        let mut sampler = GpuNoisySampler::with_seed(1, noise, 123);

        let results = sampler
            .sample(100, |c| {
                // Qubit in |0> state
                c.mz(0);
            })
            .unwrap();

        // With 50% measurement error, about half should be flipped to 1
        let ones = results.iter().filter(|r| r.outcomes[0]).count();
        assert!(ones > 30 && ones < 70, "Expected ~50% ones, got {ones}");
    }

    #[test]
    fn test_gpu_noisy_sampler_gate_noise() {
        // With gate noise but no measurement noise
        let noise = DepolarizingNoiseSampler::with_seed(0.5, 0.0, 0.0, 42);
        let mut sampler = GpuNoisySampler::with_seed(1, noise, 123);

        let results = sampler
            .sample(100, |c| {
                // Apply noise point after identity
                c.noise_1q(0);
                c.mz(0);
            })
            .unwrap();

        // With 50% gate noise, some shots should show errors
        let ones = results.iter().filter(|r| r.outcomes[0]).count();
        // X and Y errors will flip the outcome, Z won't
        // So expect ~33% errors (2/3 of 50% error rate)
        assert!(
            ones > 10 && ones < 50,
            "Expected some errors from gate noise, got {ones} ones"
        );
    }

    #[test]
    fn test_deterministic_with_seed() {
        let noise1 = DepolarizingNoiseSampler::with_seed(0.1, 0.1, 0.1, 42);
        let noise2 = DepolarizingNoiseSampler::with_seed(0.1, 0.1, 0.1, 42);

        let mut sampler1 = GpuNoisySampler::with_seed(2, noise1, 123);
        let mut sampler2 = GpuNoisySampler::with_seed(2, noise2, 123);

        let results1 = sampler1
            .sample(50, |c| {
                c.h(0);
                c.noise_1q(0);
                c.cx(0, 1);
                c.noise_2q(0, 1);
                c.mz(0);
                c.mz(1);
            })
            .unwrap();

        let results2 = sampler2
            .sample(50, |c| {
                c.h(0);
                c.noise_1q(0);
                c.cx(0, 1);
                c.noise_2q(0, 1);
                c.mz(0);
                c.mz(1);
            })
            .unwrap();

        // Same seed should give identical results
        for (r1, r2) in results1.iter().zip(results2.iter()) {
            assert_eq!(
                r1.outcomes, r2.outcomes,
                "Same seed should give same results"
            );
        }
    }

    #[test]
    fn test_biased_depolarizing_sampler() {
        // Test the biased depolarizing noise sampler with asymmetric measurement errors
        let mut sampler = BiasedDepolarizingNoiseSampler::with_seed(0.0, 0.0, 0.8, 0.2, 42);

        // p_meas_0 = 0.8: high probability of flipping 0 -> 1
        // p_meas_1 = 0.2: low probability of flipping 1 -> 0

        // Test measurement errors - set outcome to 0 (will flip with 80% prob)
        let mut flips_from_0 = 0;
        for _ in 0..100 {
            // Use the NoiseSampler trait method
            NoiseSampler::set_measurement_outcome(&mut sampler, 0, false);
            if sampler.sample_meas(0) {
                flips_from_0 += 1;
            }
        }

        // Should flip ~80% of the time when outcome is 0
        assert!(
            flips_from_0 > 60 && flips_from_0 < 95,
            "Should flip ~80% when outcome is 0, got {flips_from_0}%"
        );

        // Test measurement errors - set outcome to 1 (will flip with 20% prob)
        let mut flips_from_1 = 0;
        for _ in 0..100 {
            NoiseSampler::set_measurement_outcome(&mut sampler, 0, true);
            if sampler.sample_meas(0) {
                flips_from_1 += 1;
            }
        }

        // Should flip ~20% of the time when outcome is 1
        assert!(
            flips_from_1 > 5 && flips_from_1 < 40,
            "Should flip ~20% when outcome is 1, got {flips_from_1}%"
        );
    }

    #[test]
    fn test_biased_noise_with_gpu_sampler() {
        // Test BiasedDepolarizingNoiseSampler with GpuNoisySampler
        // High measurement error rate for 0->1 flip, low for 1->0

        // Put qubit in |0> state with high 0->1 error rate
        let noise = BiasedDepolarizingNoiseSampler::with_seed(0.0, 0.0, 0.7, 0.0, 42);
        let mut sampler = GpuNoisySampler::with_seed(1, noise, 123);

        let results = sampler
            .sample(100, |c| {
                // Qubit starts in |0>
                c.mz(0);
            })
            .unwrap();

        // With 70% 0->1 flip rate, most should measure as 1
        let ones = results.iter().filter(|r| r.outcomes[0]).count();
        assert!(
            ones > 50 && ones < 90,
            "Expected ~70% ones (0->1 flips), got {ones}%"
        );

        // Put qubit in |1> state with low 1->0 error rate
        let noise = BiasedDepolarizingNoiseSampler::with_seed(0.0, 0.0, 0.0, 0.2, 42);
        let mut sampler = GpuNoisySampler::with_seed(1, noise, 456);

        let results = sampler
            .sample(100, |c| {
                c.x(0); // Put in |1> state
                c.mz(0);
            })
            .unwrap();

        // With 20% 1->0 flip rate, most should still measure as 1
        let ones = results.iter().filter(|r| r.outcomes[0]).count();
        assert!(
            ones > 60 && ones < 95,
            "Expected ~80% ones (some 1->0 flips), got {ones}%"
        );
    }
}
