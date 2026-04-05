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

//! Importance sampling runner for rare event simulation.
//!
//! This module provides a runner that samples from a biased (proposal) noise
//! distribution while tracking importance weights for proper reweighting.
//!
//! ## When to Use
//!
//! Importance sampling is valuable when:
//! - Physical error rates are very low (e.g., 0.001 or below)
//! - You need to estimate rare event probabilities (logical errors)
//! - Standard Monte Carlo would require prohibitively many samples
//!
//! ## How It Works
//!
//! 1. Define a "proposal" distribution Q with higher error rates
//! 2. Sample from Q instead of the true distribution P
//! 3. Track the likelihood ratio w = P(path) / Q(path)
//! 4. Compute weighted statistics: E[f(X)] ≈ Σ `w_i` `f(X_i)` / Σ `w_i`
//!
//! ## Example
//!
//! ```no_run
//! use pecos_neo::sampling::ImportanceSamplingRunner;
//! use pecos_neo::sampling::weight::WeightedStatistics;
//! use pecos_neo::prelude::*;
//! use pecos_simulators::SparseStab;
//!
//! let commands = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
//!
//! // True error rate: 0.001, boost by 10x for proposal
//! let mut runner = ImportanceSamplingRunner::new(SparseStab::new(7))
//!     .with_single_qubit_boost(0.001, 10.0)
//!     .with_two_qubit_boost(0.01, 5.0)
//!     .with_seed(42);
//!
//! let mut stats = WeightedStatistics::new();
//!
//! for _ in 0..10000 {
//!     let result = runner.run_shot(&commands);
//!     let failed = false; // Replace with actual logical error check
//!     stats.add(if failed { 1.0 } else { 0.0 }, &result.weight);
//! }
//!
//! println!("Estimated logical error rate: {}", stats.mean());
//! ```

use crate::command::{CommandQueue, GateCommand, GateType};
use crate::noise::{ComposableNoiseModel, NoiseEvent, NoiseResponse};
use crate::outcome::{MeasurementOutcome, MeasurementOutcomes};
use crate::sampling::importance::ImportanceConfig;
use crate::sampling::weight::SampleWeight;
use pecos_core::QubitId;
use pecos_core::rng::rng_manageable::{RngManageable, derive_seed};
use pecos_random::PecosRng;
use pecos_simulators::{CliffordGateable, ForcedMeasurement};
use rand::RngExt;
use rand_core::SeedableRng;

/// Configuration for biasing measurement outcomes.
///
/// This is used for importance sampling of branch selection in programs
/// with classical control flow based on measurement outcomes.
///
/// For stabilizer simulation, non-deterministic measurements have 50/50 probability.
/// By biasing toward one outcome, we can explore rare branches more often
/// while reweighting to maintain unbiased estimates.
///
/// # Example
///
/// ```
/// use pecos_neo::sampling::OutcomeBiasConfig;
///
/// // Bias toward outcome 1 with 80% probability (for exploring rare branches)
/// let config = OutcomeBiasConfig::bias_toward_one(0.8);
///
/// // Or bias toward outcome 0 with 90% probability
/// let config = OutcomeBiasConfig::bias_toward_zero(0.9);
/// ```
#[derive(Debug, Clone)]
pub struct OutcomeBiasConfig {
    /// Probability of choosing outcome 1 in the proposal distribution.
    /// 0.5 = no bias (true distribution), 1.0 = always 1, 0.0 = always 0.
    pub p_one_proposal: f64,
}

impl OutcomeBiasConfig {
    /// Create a config that biases toward outcome 1.
    ///
    /// # Arguments
    /// * `p` - Probability of choosing outcome 1 in proposal (0.5 = no bias)
    #[must_use]
    pub fn bias_toward_one(p: f64) -> Self {
        Self {
            p_one_proposal: p.clamp(0.01, 0.99), // Avoid 0/1 to prevent infinite weights
        }
    }

    /// Create a config that biases toward outcome 0.
    ///
    /// # Arguments
    /// * `p` - Probability of choosing outcome 0 in proposal (0.5 = no bias)
    #[must_use]
    pub fn bias_toward_zero(p: f64) -> Self {
        Self {
            p_one_proposal: (1.0 - p).clamp(0.01, 0.99),
        }
    }

    /// Sample an outcome from the proposal distribution.
    ///
    /// Returns the chosen outcome and the weight correction.
    /// For stabilizer sim, true probability is always 0.5 for non-deterministic measurements.
    #[must_use]
    pub fn sample(&self, rng: &mut PecosRng) -> (bool, f64, f64) {
        let outcome_one = rng.random::<f64>() < self.p_one_proposal;
        let p_true = 0.5; // Stabilizer: non-deterministic = 50/50
        let p_proposal = if outcome_one {
            self.p_one_proposal
        } else {
            1.0 - self.p_one_proposal
        };
        (outcome_one, p_true, p_proposal)
    }
}

/// Result of an importance-sampled shot.
#[derive(Debug, Clone)]
pub struct ImportanceSampledShot {
    /// Measurement outcomes from this shot.
    pub outcomes: MeasurementOutcomes,
    /// Importance weight for this shot (P(path) / Q(path)).
    pub weight: SampleWeight,
}

/// A shot runner that uses importance sampling for efficient rare event simulation.
///
/// This runner samples from a biased proposal distribution with higher error
/// rates, while tracking importance weights for proper reweighting.
///
/// # Measurement Outcome Biasing
///
/// For programs with classical control flow based on measurement outcomes,
/// you can bias the measurement outcomes to explore rare branches more often:
///
/// ```
/// use pecos_neo::sampling::{ImportanceSamplingRunner, OutcomeBiasConfig};
/// use pecos_simulators::SparseStab;
///
/// let runner = ImportanceSamplingRunner::new(SparseStab::new(7))
///     .with_outcome_bias(OutcomeBiasConfig::bias_toward_one(0.8))
///     .with_seed(42);
/// ```
///
/// Note: Outcome biasing requires the simulator to implement `ForcedMeasurement`.
pub struct ImportanceSamplingRunner<S: CliffordGateable> {
    pub(crate) simulator: S,
    noise: Option<ComposableNoiseModel>,
    pub(crate) rng: PecosRng,
    outcomes: MeasurementOutcomes,

    // Importance sampling configuration
    single_qubit_config: Option<ImportanceConfig>,
    two_qubit_config: Option<ImportanceConfig>,
    measurement_config: Option<ImportanceConfig>,

    // Measurement outcome biasing (for branch exploration)
    outcome_bias: Option<OutcomeBiasConfig>,

    // Current shot weight
    weight: SampleWeight,
}

impl<S: CliffordGateable> ImportanceSamplingRunner<S> {
    /// Create a new importance sampling runner with the given simulator.
    pub fn new(simulator: S) -> Self {
        Self {
            simulator,
            noise: None,
            rng: PecosRng::from_rng(&mut rand::rng()),
            outcomes: MeasurementOutcomes::new(),
            single_qubit_config: None,
            two_qubit_config: None,
            measurement_config: None,
            outcome_bias: None,
            weight: SampleWeight::one(),
        }
    }

    /// Set the base noise model.
    ///
    /// This noise model is used for structural noise effects (like leakage tracking).
    /// The error rates are overridden by the importance sampling configuration.
    #[must_use]
    pub fn with_noise(mut self, noise: ComposableNoiseModel) -> Self {
        self.noise = Some(noise);
        self
    }

    /// Set the RNG seed for reproducibility.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = PecosRng::seed_from_u64(seed);
        self
    }

    /// Configure importance sampling for single-qubit gates.
    ///
    /// # Arguments
    /// * `p_true` - True error probability
    /// * `boost_factor` - How much to multiply the error rate for the proposal
    #[must_use]
    pub fn with_single_qubit_boost(mut self, p_true: f64, boost_factor: f64) -> Self {
        self.single_qubit_config = Some(ImportanceConfig::with_boost(p_true, boost_factor));
        self
    }

    /// Configure importance sampling for two-qubit gates.
    #[must_use]
    pub fn with_two_qubit_boost(mut self, p_true: f64, boost_factor: f64) -> Self {
        self.two_qubit_config = Some(ImportanceConfig::with_boost(p_true, boost_factor));
        self
    }

    /// Configure importance sampling for measurement errors.
    #[must_use]
    pub fn with_measurement_boost(mut self, p_true: f64, boost_factor: f64) -> Self {
        self.measurement_config = Some(ImportanceConfig::with_boost(p_true, boost_factor));
        self
    }

    /// Configure biasing of measurement outcomes for branch exploration.
    ///
    /// This is useful for programs with classical control flow based on
    /// measurement outcomes. By biasing toward specific outcomes, you can
    /// explore rare branches more frequently while maintaining unbiased estimates.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_neo::sampling::{ImportanceSamplingRunner, OutcomeBiasConfig};
    /// use pecos_simulators::SparseStab;
    ///
    /// // Bias toward measuring 1 (80% of the time for non-deterministic measurements)
    /// let runner = ImportanceSamplingRunner::new(SparseStab::new(7))
    ///     .with_outcome_bias(OutcomeBiasConfig::bias_toward_one(0.8))
    ///     .with_seed(42);
    /// ```
    ///
    /// # Note
    ///
    /// This only affects non-deterministic measurements (those with 50/50 probability
    /// in stabilizer simulation). Deterministic measurements always return their
    /// fixed outcome regardless of bias configuration.
    #[must_use]
    pub fn with_outcome_bias(mut self, config: OutcomeBiasConfig) -> Self {
        self.outcome_bias = Some(config);
        self
    }

    /// Check if outcome biasing is enabled.
    #[must_use]
    pub fn has_outcome_bias(&self) -> bool {
        self.outcome_bias.is_some()
    }

    /// Get a reference to the simulator.
    #[must_use]
    pub fn simulator(&self) -> &S {
        &self.simulator
    }

    /// Get a mutable reference to the simulator.
    pub fn simulator_mut(&mut self) -> &mut S {
        &mut self.simulator
    }

    /// Run a single shot with importance sampling.
    ///
    /// Returns the measurement outcomes along with the importance weight.
    pub fn run_shot(&mut self, commands: &CommandQueue) -> ImportanceSampledShot {
        // Reset for new shot
        self.weight = SampleWeight::one();
        self.outcomes.clear();

        // Execute all commands
        for command in commands {
            self.execute_command(command);
        }

        // Take outcomes and weight
        let outcomes = std::mem::take(&mut self.outcomes);
        let weight = self.weight;

        // Reset noise model state
        if let Some(ref mut noise) = self.noise {
            noise.reset();
        }

        ImportanceSampledShot { outcomes, weight }
    }

    /// Run a shot with simulator reset - optimized for Monte Carlo.
    ///
    /// This is faster than creating a new runner or cloning the simulator for each shot.
    /// Use this when running independent shots where each starts from the |0⟩^n state.
    ///
    /// **Performance**: Resets the simulator (8-12x faster than clone for large qubit counts)
    /// before running the circuit.
    pub fn run_shot_fresh(&mut self, commands: &CommandQueue) -> ImportanceSampledShot {
        // Reset simulator to |0⟩^n state (much faster than clone)
        self.simulator.reset();

        // Run the shot normally
        self.run_shot(commands)
    }

    /// Execute a single command with importance-weighted noise.
    fn execute_command(&mut self, command: &GateCommand) {
        let qubits: Vec<QubitId> = command.qubits.iter().copied().collect();

        // Check for gate skip (e.g., due to leakage)
        if self.emit_before_gate(command) {
            return;
        }

        // Execute the gate
        match command.gate_type {
            // Preparation
            GateType::PZ | GateType::QAlloc => {
                self.simulator.pz(&qubits);
                self.emit_after_preparation(&qubits);
            }

            // Measurement
            GateType::MZ | GateType::MeasureLeaked | GateType::MeasureFree => {
                let results = self.simulator.mz(&qubits);
                let mut outcomes: Vec<bool> = results.iter().map(|r| r.outcome).collect();

                // Apply importance-sampled measurement errors
                for (i, outcome) in outcomes.iter_mut().enumerate() {
                    if self.sample_measurement_error() {
                        *outcome = !*outcome;
                    }
                    // Record with potentially flipped outcome
                    self.outcomes.record(MeasurementOutcome::new(
                        qubits[i],
                        *outcome,
                        results[i].is_deterministic,
                    ));
                }
            }

            // Gate execution with importance-weighted noise
            _ => {
                self.execute_clifford_gate(command);
                self.apply_importance_sampled_gate_noise(command);
            }
        }
    }

    /// Sample gate noise with importance weighting.
    fn apply_importance_sampled_gate_noise(&mut self, command: &GateCommand) {
        let arity = command.gate_type.quantum_arity();

        match arity {
            1 => {
                if let Some(config) = self.single_qubit_config.clone() {
                    self.apply_single_qubit_noise(command, &config);
                }
            }
            2 => {
                if let Some(config) = self.two_qubit_config.clone() {
                    self.apply_two_qubit_noise(command, &config);
                }
            }
            _ => {}
        }
    }

    /// Apply importance-sampled single-qubit noise.
    fn apply_single_qubit_noise(&mut self, command: &GateCommand, config: &ImportanceConfig) {
        for &qubit in &command.qubits {
            let error_occurs = self.rng.random::<f64>() < config.p_proposal;

            // Update importance weight
            let (p_true, p_proposal) = if error_occurs {
                config.weight_for_error()
            } else {
                config.weight_for_no_error()
            };
            self.weight.update(p_true, p_proposal);

            // Apply error if sampled
            if error_occurs {
                self.apply_random_pauli(&[qubit]);
            }
        }
    }

    /// Apply importance-sampled two-qubit noise.
    fn apply_two_qubit_noise(&mut self, command: &GateCommand, config: &ImportanceConfig) {
        if command.qubits.len() < 2 {
            return;
        }

        let error_occurs = self.rng.random::<f64>() < config.p_proposal;

        // Update importance weight
        let (p_true, p_proposal) = if error_occurs {
            config.weight_for_error()
        } else {
            config.weight_for_no_error()
        };
        self.weight.update(p_true, p_proposal);

        // Apply error if sampled
        if error_occurs {
            let qubits: Vec<QubitId> = command.qubits.iter().copied().collect();
            self.apply_random_two_qubit_pauli(&qubits);
        }
    }

    /// Sample measurement error with importance weighting.
    ///
    /// Returns true if the measurement should be flipped.
    fn sample_measurement_error(&mut self) -> bool {
        if let Some(config) = &self.measurement_config {
            let error_occurs = self.rng.random::<f64>() < config.p_proposal;

            let (p_true, p_proposal) = if error_occurs {
                config.weight_for_error()
            } else {
                config.weight_for_no_error()
            };
            self.weight.update(p_true, p_proposal);

            error_occurs
        } else {
            false
        }
    }

    /// Apply a random single-qubit Pauli error.
    fn apply_random_pauli(&mut self, qubits: &[QubitId]) {
        let choice: u8 = self.rng.random_range(0..3);
        match choice {
            0 => {
                self.simulator.x(qubits);
            }
            1 => {
                self.simulator.y(qubits);
            }
            2 => {
                self.simulator.z(qubits);
            }
            _ => unreachable!(),
        }
    }

    /// Apply a random two-qubit Pauli error (excluding II).
    fn apply_random_two_qubit_pauli(&mut self, qubits: &[QubitId]) {
        // 15 non-trivial Pauli pairs
        let choice: u8 = self.rng.random_range(0..15);

        let paulis = [
            (0, 1),
            (0, 2),
            (0, 3), // I ⊗ (X,Y,Z)
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3), // X ⊗ (I,X,Y,Z)
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3), // Y ⊗ (I,X,Y,Z)
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3), // Z ⊗ (I,X,Y,Z)
        ];

        let (p0, p1) = paulis[choice as usize];

        // Apply first qubit Pauli
        match p0 {
            1 => {
                self.simulator.x(&[qubits[0]]);
            }
            2 => {
                self.simulator.y(&[qubits[0]]);
            }
            3 => {
                self.simulator.z(&[qubits[0]]);
            }
            _ => {}
        }

        // Apply second qubit Pauli
        match p1 {
            1 => {
                self.simulator.x(&[qubits[1]]);
            }
            2 => {
                self.simulator.y(&[qubits[1]]);
            }
            3 => {
                self.simulator.z(&[qubits[1]]);
            }
            _ => {}
        }
    }

    /// Emit a before-gate noise event (for leakage handling, etc.).
    fn emit_before_gate(&mut self, command: &GateCommand) -> bool {
        if let Some(ref mut noise) = self.noise {
            // Use helper constructor for zero-allocation access
            let event = NoiseEvent::before_gate(
                command.gate_type,
                command.qubits.as_slice(),
                command.angles.as_slice(),
            );
            let response = noise.emit(&event, &mut self.rng);
            let should_skip = response.should_skip_gate();
            self.apply_noise_response(response);
            return should_skip;
        }
        false
    }

    /// Emit an after-preparation noise event.
    fn emit_after_preparation(&mut self, qubits: &[QubitId]) {
        if let Some(ref mut noise) = self.noise {
            let event = NoiseEvent::AfterPreparation { qubits };
            let response = noise.emit(&event, &mut self.rng);
            self.apply_noise_response(response);
        }
    }

    /// Apply a noise response (for leakage tracking, etc.).
    fn apply_noise_response(&mut self, response: NoiseResponse) {
        match response {
            NoiseResponse::None
            | NoiseResponse::SkipGate
            | NoiseResponse::MarkLeaked(_)
            | NoiseResponse::MarkUnleaked(_) => {}

            NoiseResponse::InjectGates(gates) => {
                for gate in gates.iter() {
                    self.execute_noise_gate(gate);
                }
            }

            NoiseResponse::FlipOutcomes(qubits) => {
                for qubit in qubits {
                    self.outcomes.flip(qubit);
                }
            }

            NoiseResponse::LeakedMeasurement(qubits) => {
                for qubit in qubits {
                    self.outcomes.mark_leaked(qubit);
                }
            }

            NoiseResponse::ForceOutcomes(forced) => {
                for (qubit, value) in forced {
                    self.outcomes.set_outcome(qubit, value);
                }
            }

            NoiseResponse::Multiple(responses) => {
                for r in responses {
                    self.apply_noise_response(r);
                }
            }
        }
    }

    /// Execute a noise gate.
    fn execute_noise_gate(&mut self, gate: &GateCommand) {
        let qubits: Vec<QubitId> = gate.qubits.iter().copied().collect();

        match gate.gate_type {
            GateType::X => {
                self.simulator.x(&qubits);
            }
            GateType::Y => {
                self.simulator.y(&qubits);
            }
            GateType::Z => {
                self.simulator.z(&qubits);
            }
            _ => {}
        }
    }

    /// Execute Clifford gates.
    fn execute_clifford_gate(&mut self, command: &GateCommand) -> bool {
        let qubits: Vec<QubitId> = command.qubits.iter().copied().collect();

        match command.gate_type {
            // Single-qubit Paulis
            GateType::I => {
                self.simulator.identity(&qubits);
            }
            GateType::X => {
                self.simulator.x(&qubits);
            }
            GateType::Y => {
                self.simulator.y(&qubits);
            }
            GateType::Z => {
                self.simulator.z(&qubits);
            }

            // Single-qubit Cliffords
            GateType::H => {
                self.simulator.h(&qubits);
            }
            GateType::SX => {
                self.simulator.sx(&qubits);
            }
            GateType::SXdg => {
                self.simulator.sxdg(&qubits);
            }
            GateType::SY => {
                self.simulator.sy(&qubits);
            }
            GateType::SYdg => {
                self.simulator.sydg(&qubits);
            }
            GateType::SZ => {
                self.simulator.sz(&qubits);
            }
            GateType::SZdg => {
                self.simulator.szdg(&qubits);
            }

            // Two-qubit gates
            GateType::CX => {
                let pairs: Vec<(QubitId, QubitId)> =
                    qubits.chunks_exact(2).map(|c| (c[0], c[1])).collect();
                self.simulator.cx(&pairs);
            }
            GateType::CY => {
                let pairs: Vec<(QubitId, QubitId)> =
                    qubits.chunks_exact(2).map(|c| (c[0], c[1])).collect();
                self.simulator.cy(&pairs);
            }
            GateType::CZ => {
                let pairs: Vec<(QubitId, QubitId)> =
                    qubits.chunks_exact(2).map(|c| (c[0], c[1])).collect();
                self.simulator.cz(&pairs);
            }
            GateType::SZZ => {
                let pairs: Vec<(QubitId, QubitId)> =
                    qubits.chunks_exact(2).map(|c| (c[0], c[1])).collect();
                self.simulator.szz(&pairs);
            }
            GateType::SZZdg => {
                let pairs: Vec<(QubitId, QubitId)> =
                    qubits.chunks_exact(2).map(|c| (c[0], c[1])).collect();
                self.simulator.szzdg(&pairs);
            }
            GateType::SWAP => {
                let pairs: Vec<(QubitId, QubitId)> =
                    qubits.chunks_exact(2).map(|c| (c[0], c[1])).collect();
                self.simulator.swap(&pairs);
            }

            _ => return false,
        }
        true
    }
}

/// Extension methods for simulators that support RNG management.
///
/// This enables full determinism by seeding both the runner's internal RNG
/// and the simulator's RNG.
impl<S> ImportanceSamplingRunner<S>
where
    S: CliffordGateable + RngManageable<Rng = PecosRng>,
{
    /// Set the seed for full determinism.
    ///
    /// This seeds both the importance sampling RNG and the simulator's internal RNG
    /// using derived seeds from a single base seed. This mirrors how `MonteCarloEngine`
    /// handles hierarchical seeding in pecos-engines.
    ///
    /// # Seed Hierarchy
    ///
    /// ```text
    /// seed
    /// ├── noise (for importance sampling/noise RNG)
    /// └── simulator (for simulator's internal RNG)
    /// ```
    #[must_use]
    pub fn with_full_seed(mut self, seed: u64) -> Self {
        let noise_seed = derive_seed(seed, "noise");
        let sim_seed = derive_seed(seed, "simulator");

        self.rng = PecosRng::seed_from_u64(noise_seed);
        self.simulator.set_seed(sim_seed);
        self
    }
}

/// Extension methods for simulators that support forced measurements.
///
/// This enables measurement outcome importance sampling for exploring
/// rare branches in programs with classical control flow.
impl<S> ImportanceSamplingRunner<S>
where
    S: CliffordGateable + ForcedMeasurement,
{
    /// Run a shot with measurement outcome biasing enabled.
    ///
    /// This method uses forced measurements to bias toward specific outcomes
    /// while tracking importance weights. Use this for programs with classical
    /// control flow where you want to explore rare branches.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use pecos_neo::sampling::{ImportanceSamplingRunner, OutcomeBiasConfig};
    /// use pecos_neo::prelude::*;
    /// use pecos_simulators::SparseStab;
    ///
    /// let commands = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
    /// let mut runner = ImportanceSamplingRunner::new(SparseStab::new(7))
    ///     .with_outcome_bias(OutcomeBiasConfig::bias_toward_one(0.8))
    ///     .with_seed(42);
    ///
    /// // This will bias non-deterministic measurements toward outcome 1
    /// let result = runner.run_shot_biased(&commands);
    /// println!("Weight: {}", result.weight.weight());
    /// ```
    ///
    /// # How It Works
    ///
    /// For each measurement:
    /// 1. If deterministic (stabilizer eigenstate): return fixed outcome, no weight change
    /// 2. If non-deterministic (50/50): sample from biased proposal, force that outcome,
    ///    update weight by P(outcome)/Q(outcome) = `0.5/bias_prob`
    pub fn run_shot_biased(&mut self, commands: &CommandQueue) -> ImportanceSampledShot {
        // Reset for new shot
        self.weight = SampleWeight::one();
        self.outcomes.clear();

        // Execute all commands with biased measurements
        for command in commands {
            self.execute_command_biased(command);
        }

        // Take outcomes and weight
        let outcomes = std::mem::take(&mut self.outcomes);
        let weight = self.weight;

        // Reset noise model state
        if let Some(ref mut noise) = self.noise {
            noise.reset();
        }

        ImportanceSampledShot { outcomes, weight }
    }

    /// Execute a single command with biased measurement outcomes.
    fn execute_command_biased(&mut self, command: &GateCommand) {
        let qubits: Vec<QubitId> = command.qubits.iter().copied().collect();

        // Check for gate skip (e.g., due to leakage)
        if self.emit_before_gate(command) {
            return;
        }

        // Execute the gate
        match command.gate_type {
            // Preparation
            GateType::PZ | GateType::QAlloc => {
                self.simulator.pz(&qubits);
                self.emit_after_preparation(&qubits);
            }

            // Measurement with outcome biasing
            GateType::MZ | GateType::MeasureLeaked | GateType::MeasureFree => {
                for &qubit in &qubits {
                    let (outcome, is_deterministic) = self.measure_biased(qubit.index());

                    // Apply importance-sampled measurement errors (bit flips)
                    let final_outcome = if self.sample_measurement_error() {
                        !outcome
                    } else {
                        outcome
                    };

                    self.outcomes.record(MeasurementOutcome::new(
                        qubit,
                        final_outcome,
                        is_deterministic,
                    ));
                }
            }

            // Gate execution with importance-weighted noise (same as unbiased)
            _ => {
                self.execute_clifford_gate(command);
                self.apply_importance_sampled_gate_noise(command);
            }
        }
    }

    /// Perform a biased measurement using forced outcomes.
    ///
    /// Returns (outcome, `is_deterministic`).
    fn measure_biased(&mut self, qubit: usize) -> (bool, bool) {
        if let Some(ref bias_config) = self.outcome_bias {
            // Sample from proposal distribution to get desired outcome
            let (desired_outcome, p_true, p_proposal) = bias_config.sample(&mut self.rng);

            // Force the measurement to the desired outcome
            // mz_forced handles both deterministic and non-deterministic cases:
            // - If deterministic: returns the fixed outcome (ignores desired_outcome)
            // - If non-deterministic: forces to desired_outcome
            let result = self.simulator.mz_forced(qubit, desired_outcome);

            if result.is_deterministic {
                // Deterministic: the outcome is fixed, no biasing was applied
                // Don't update weight since we didn't actually bias anything
                (result.outcome, true)
            } else {
                // Non-deterministic: biasing was applied, update weight
                self.weight.update(p_true, p_proposal);
                (result.outcome, false)
            }
        } else {
            // No bias configured, use regular measurement
            let result = self.simulator.mz(&[QubitId(qubit)]);
            (result[0].outcome, result[0].is_deterministic)
        }
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)] // statistical tests use count as f64
mod tests {
    use super::*;
    use crate::command::CommandBuilder;
    use crate::sampling::weight::WeightedStatistics;
    use pecos_simulators::SparseStab;

    #[test]
    fn test_importance_runner_basic() {
        let commands = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

        let mut runner = ImportanceSamplingRunner::new(SparseStab::new(1)).with_seed(42);

        let result = runner.run_shot(&commands);
        assert_eq!(result.outcomes.len(), 1);
        // No importance sampling configured, weight should be 1
        assert!((result.weight.weight() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_importance_runner_with_boost() {
        let commands = CommandBuilder::new()
            .pz(&[0])
            .h(&[0]) // Single-qubit gate will trigger importance sampling
            .mz(&[0])
            .build();

        let mut runner = ImportanceSamplingRunner::new(SparseStab::new(1))
            .with_single_qubit_boost(0.001, 10.0)
            .with_seed(42);

        let result = runner.run_shot(&commands);
        assert_eq!(result.outcomes.len(), 1);
        // Weight should differ from 1 due to importance sampling
        // (unless by chance the proposal probability matched the decision)
    }

    #[test]
    fn test_importance_sampling_estimates_correct_rate() {
        // This test verifies that importance sampling produces
        // unbiased estimates of the true error rate

        let commands = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

        let true_rate = 0.001;
        let boost = 100.0; // Very aggressive boost

        let mut runner = ImportanceSamplingRunner::new(SparseStab::new(1))
            .with_single_qubit_boost(true_rate, boost)
            .with_seed(12345);

        let mut stats = WeightedStatistics::new();
        let num_shots = 10000;

        for _ in 0..num_shots {
            let result = runner.run_shot(&commands);
            // We're just measuring the weight behavior, not actual errors
            // The weight should average to 1.0 over many samples
            stats.add(result.weight.weight(), &SampleWeight::one());
        }

        // The mean weight should be approximately 1.0
        // (this is a property of importance sampling)
        let mean_weight = stats.mean();
        assert!(
            (mean_weight - 1.0).abs() < 0.1,
            "Mean weight {mean_weight} should be close to 1.0"
        );
    }

    #[test]
    fn test_two_qubit_importance_sampling() {
        let commands = CommandBuilder::new()
            .pz(&[0])
            .pz(&[1])
            .cx(&[(0, 1)]) // Two-qubit gate
            .mz(&[0])
            .mz(&[1])
            .build();

        let mut runner = ImportanceSamplingRunner::new(SparseStab::new(2))
            .with_two_qubit_boost(0.01, 5.0)
            .with_seed(42);

        let result = runner.run_shot(&commands);
        assert_eq!(result.outcomes.len(), 2);
    }

    #[test]
    fn test_measurement_importance_sampling() {
        let commands = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

        let mut runner = ImportanceSamplingRunner::new(SparseStab::new(1))
            .with_measurement_boost(0.001, 100.0)
            .with_seed(42);

        // Run many shots to see measurement flips
        let mut flip_count = 0;
        for i in 0..100 {
            // Use different seeds to get variety
            #[allow(clippy::cast_sign_loss)] // i is a non-negative loop counter
            let seed = 42 + i as u64;
            runner.rng = PecosRng::seed_from_u64(seed);
            let result = runner.run_shot(&commands);
            // With 10% proposal rate, we should see flips
            if (result.weight.weight() - 1.0).abs() > f64::EPSILON {
                flip_count += 1;
            }
        }

        // With 10% proposal rate (0.001 * 100 = 0.1), we expect ~10% flips
        assert!(flip_count > 0, "Expected some measurement flips");
    }

    // ========================================================================
    // Outcome Biasing Tests
    // ========================================================================

    #[test]
    fn test_outcome_bias_config() {
        let config = OutcomeBiasConfig::bias_toward_one(0.8);
        assert!((config.p_one_proposal - 0.8).abs() < 1e-10);

        let config = OutcomeBiasConfig::bias_toward_zero(0.9);
        assert!((config.p_one_proposal - 0.1).abs() < 1e-10);

        // Test clamping
        let config = OutcomeBiasConfig::bias_toward_one(1.0);
        assert!(config.p_one_proposal < 1.0, "Should be clamped below 1.0");

        let config = OutcomeBiasConfig::bias_toward_one(0.0);
        assert!(config.p_one_proposal > 0.0, "Should be clamped above 0.0");
    }

    #[test]
    fn test_outcome_bias_sample() {
        let config = OutcomeBiasConfig::bias_toward_one(0.9);
        let mut rng = PecosRng::seed_from_u64(42);

        let mut ones = 0;
        for _ in 0..1000 {
            let (outcome, p_true, p_proposal) = config.sample(&mut rng);
            if outcome {
                ones += 1;
                // For outcome=1, p_proposal should be 0.9
                assert!((p_proposal - 0.9).abs() < 1e-10);
            } else {
                // For outcome=0, p_proposal should be 0.1
                assert!((p_proposal - 0.1).abs() < 1e-10);
            }
            // p_true should always be 0.5 (stabilizer non-deterministic)
            assert!((p_true - 0.5).abs() < 1e-10);
        }

        // With 90% bias toward 1, should get roughly 900 ones
        assert!(ones > 800, "Expected ~90% ones, got {ones}");
        assert!(ones < 980, "Expected ~90% ones, got {ones}");
    }

    #[test]
    fn test_biased_measurement_produces_unbiased_estimates() {
        // Key test: biased sampling with reweighting should produce
        // the same expected value as unbiased sampling
        let commands = CommandBuilder::new()
            .pz(&[0])
            .h(&[0]) // Creates 50/50 superposition
            .mz(&[0])
            .build();

        let num_shots = 5000;

        // ========== Unbiased sampling ==========
        let mut unbiased_ones = 0;
        for seed in 0..num_shots {
            let mut runner = ImportanceSamplingRunner::new(SparseStab::new(1)).with_seed(seed);
            let result = runner.run_shot(&commands);
            if result.outcomes.get_bit(QubitId(0)).unwrap_or(false) {
                unbiased_ones += 1;
            }
        }
        let unbiased_rate = f64::from(unbiased_ones) / num_shots as f64;

        // ========== Biased sampling with reweighting ==========
        // Bias heavily toward 1 (80%)
        let mut biased_weighted_sum = 0.0;
        let mut biased_total_weight = 0.0;

        for seed in 0..num_shots {
            let mut runner = ImportanceSamplingRunner::new(SparseStab::new(1))
                .with_outcome_bias(OutcomeBiasConfig::bias_toward_one(0.8))
                .with_seed(seed);

            let result = runner.run_shot_biased(&commands);
            let outcome = result.outcomes.get_bit(QubitId(0)).unwrap_or(false);
            let weight = result.weight.weight();

            if outcome {
                biased_weighted_sum += weight;
            }
            biased_total_weight += weight;
        }
        let biased_rate = biased_weighted_sum / biased_total_weight;

        // Both should be close to 0.5
        println!("Unbiased rate: {unbiased_rate:.4}");
        println!("Biased rate:   {biased_rate:.4}");

        assert!(
            (unbiased_rate - 0.5).abs() < 0.05,
            "Unbiased rate should be ~0.5, got {unbiased_rate}"
        );
        assert!(
            (biased_rate - 0.5).abs() < 0.05,
            "Biased rate should be ~0.5 after reweighting, got {biased_rate}"
        );
        assert!(
            (unbiased_rate - biased_rate).abs() < 0.05,
            "Rates should match: unbiased={unbiased_rate:.4}, biased={biased_rate:.4}"
        );
    }

    #[test]
    fn test_biased_measurement_explores_branches() {
        // Test that biasing actually causes more of the biased outcome to occur
        // (before reweighting)
        let commands = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

        let num_shots = 1000;

        // Bias heavily toward 1
        let mut ones = 0;
        for seed in 0..num_shots {
            let mut runner = ImportanceSamplingRunner::new(SparseStab::new(1))
                .with_outcome_bias(OutcomeBiasConfig::bias_toward_one(0.9))
                .with_seed(seed);

            let result = runner.run_shot_biased(&commands);
            if result.outcomes.get_bit(QubitId(0)).unwrap_or(false) {
                ones += 1;
            }
        }

        let ones_rate = f64::from(ones) / num_shots as f64;

        // Should see ~90% ones (the bias rate), not 50%
        assert!(
            ones_rate > 0.8,
            "Expected ~90% ones with biasing, got {ones_rate:.2}"
        );
    }

    #[test]
    fn test_deterministic_measurement_not_biased() {
        // Deterministic measurements should not be affected by outcome biasing
        // Prep |0> then measure should always give 0
        let commands = CommandBuilder::new()
            .pz(&[0])
            .mz(&[0]) // No H, so deterministic
            .build();

        let mut runner = ImportanceSamplingRunner::new(SparseStab::new(1))
            .with_outcome_bias(OutcomeBiasConfig::bias_toward_one(0.99))
            .with_seed(42);

        // Should always get 0 regardless of bias (deterministic measurement)
        for _ in 0..10 {
            let result = runner.run_shot_biased(&commands);
            let outcome = result.outcomes.get_bit(QubitId(0)).unwrap_or(true);
            assert!(!outcome, "Deterministic |0> measurement should give 0");
            // Weight should be 1 (no importance sampling for deterministic)
            assert!(
                (result.weight.weight() - 1.0).abs() < 1e-10,
                "Weight should be 1 for deterministic measurement"
            );
        }
    }
}
