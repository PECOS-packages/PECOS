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

//! Simple shot runner for quantum simulation.
//!
//! Executes gate commands on a simulator, applying noise as configured.

use crate::command::{CommandQueue, GateCommand, GateType};
use crate::noise::{ComposableNoiseModel, NoiseEvent, NoiseResponse};
use crate::outcome::{MeasurementOutcome, MeasurementOutcomes};
use pecos_core::rng::rng_manageable::{derive_seed, RngManageable};
use pecos_core::{Angle64, QubitId};
use smallvec::SmallVec;
use pecos_qsim::{ArbitraryRotationGateable, CliffordGateable};
use pecos_rng::PecosRng;
use rand_core::SeedableRng;

/// Simple shot runner that executes commands on a simulator.
///
/// The runner handles:
/// - Gate execution on the simulator
/// - Noise application via the composable noise model
/// - Measurement outcome collection
///
/// # Example
///
/// ```ignore
/// use pecos_neo::prelude::*;
/// use pecos_qsim::SparseStab;
///
/// let commands = CommandBuilder::new()
///     .prep(0)
///     .h(0)
///     .measure(0)
///     .build();
///
/// let mut runner = ShotRunner::new(SparseStab::new(1));
/// let outcomes = runner.execute(&commands);
/// ```
pub struct ShotRunner<S: CliffordGateable> {
    simulator: S,
    noise: Option<ComposableNoiseModel>,
    rng: PecosRng,
    outcomes: MeasurementOutcomes,
}

impl<S: CliffordGateable> ShotRunner<S> {
    /// Create a new shot runner with the given simulator.
    pub fn new(simulator: S) -> Self {
        Self {
            simulator,
            noise: None,
            rng: PecosRng::from_rng(&mut rand::rng()),
            outcomes: MeasurementOutcomes::new(),
        }
    }

    /// Set the noise model.
    #[must_use]
    pub fn with_noise(mut self, noise: ComposableNoiseModel) -> Self {
        self.noise = Some(noise);
        self
    }

    /// Set the RNG seed for noise operations.
    ///
    /// Note: This only seeds the noise RNG. For full determinism with simulators
    /// that have their own RNG (e.g., for measurement outcomes), use `with_full_seed()`.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = PecosRng::seed_from_u64(seed);
        self
    }

    /// Set the RNG directly for noise operations.
    #[must_use]
    pub fn with_rng(mut self, rng: PecosRng) -> Self {
        self.rng = rng;
        self
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

    /// Execute a command queue and return measurement outcomes.
    pub fn execute(&mut self, commands: &CommandQueue) -> &MeasurementOutcomes {
        self.outcomes.clear();

        for command in commands {
            self.execute_command(command);
        }

        &self.outcomes
    }

    /// Execute a single shot and return outcomes, then reset for next shot.
    pub fn run_shot(&mut self, commands: &CommandQueue) -> MeasurementOutcomes {
        self.execute(commands);
        let outcomes = std::mem::take(&mut self.outcomes);

        // Reset noise model state for next shot
        if let Some(ref mut noise) = self.noise {
            noise.reset();
        }

        outcomes
    }

    /// Execute a shot with simulator reset - optimized for Monte Carlo.
    ///
    /// This is faster than creating a new runner or cloning the simulator for each shot.
    /// Use this when running independent shots where each starts from the |0⟩^n state.
    ///
    /// **Performance**: Resets the simulator (8-12x faster than clone for large qubit counts)
    /// before running the circuit. This is ideal for Monte Carlo simulations where circuits
    /// typically begin with Prep gates anyway.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut runner = ShotRunner::new(SparseStab::new(10));
    /// for _ in 0..1000 {
    ///     let outcomes = runner.run_shot_fresh(&commands);
    ///     // Process outcomes...
    /// }
    /// ```
    pub fn run_shot_fresh(&mut self, commands: &CommandQueue) -> MeasurementOutcomes {
        // Reset simulator to |0⟩^n state (much faster than clone)
        self.simulator.reset();

        // Run the shot normally
        self.run_shot(commands)
    }

    /// Execute a single command (Clifford gates only).
    ///
    /// Non-Clifford gates (rotations) are skipped. Use `execute_all()` with
    /// an `ArbitraryRotationGateable` simulator to include rotation gates.
    fn execute_command(&mut self, command: &GateCommand) {
        // Use as_slice() instead of collecting into Vec - zero allocation
        let qubits = command.qubits.as_slice();

        // Emit before-gate event for noise - may skip the gate (e.g., for leaked qubits)
        if self.emit_before_gate(command) {
            // Gate was skipped (e.g., due to leakage)
            // Still emit after-gate for channels that want to inject errors
            if !command.gate_type.is_measurement() && !command.gate_type.is_preparation() {
                self.emit_after_gate(command);
            }
            return;
        }

        // Execute the gate
        match command.gate_type {
            // Preparation
            GateType::Prep | GateType::QAlloc => {
                self.simulator.pz(qubits);
                self.emit_after_preparation(qubits);
            }

            // Measurement
            GateType::Measure | GateType::MeasureLeaked | GateType::MeasureFree => {
                self.emit_before_measurement(qubits);
                let results = self.simulator.mz(qubits);
                // Pre-size outcomes array based on results length (stack allocation for small cases)
                let outcomes: SmallVec<[bool; 4]> = results.iter().map(|r| r.outcome).collect();
                self.record_measurements(command.gate_type, qubits, &results);
                self.emit_after_measurement(qubits, outcomes.as_slice());
            }

            // Idle - emit idle time event for noise
            GateType::Idle => {
                if let Some(duration) = command.get_idle_duration() {
                    self.emit_idle_time(qubits, duration);
                }
            }

            // Try Clifford gates, skip non-Clifford
            _ => {
                // execute_clifford_gate returns false for non-Clifford gates (skipped)
                self.execute_clifford_gate(command);
            }
        }

        // Emit after-gate event (except for measurement/prep/idle which have their own)
        if !command.gate_type.is_measurement()
            && !command.gate_type.is_preparation()
            && command.gate_type != GateType::Idle
        {
            self.emit_after_gate(command);
        }
    }

    /// Emit a before-gate noise event and apply any responses.
    ///
    /// Returns `true` if the gate should be skipped (e.g., for leaked qubits).
    fn emit_before_gate(&mut self, command: &GateCommand) -> bool {
        if let Some(ref mut noise) = self.noise {
            // Use as_slice() for zero-allocation access
            let event = NoiseEvent::BeforeGate {
                gate_type: command.gate_type,
                qubits: command.qubits.as_slice(),
                angles: command.angles.as_slice(),
            };
            let response = noise.emit(event, &mut self.rng);
            let should_skip = response.should_skip_gate();
            self.apply_noise_response(response);
            return should_skip;
        }
        false
    }

    /// Emit an after-gate noise event and apply any responses.
    fn emit_after_gate(&mut self, command: &GateCommand) {
        if let Some(ref mut noise) = self.noise {
            // Use as_slice() for zero-allocation access
            let event = NoiseEvent::AfterGate {
                gate_type: command.gate_type,
                qubits: command.qubits.as_slice(),
                angles: command.angles.as_slice(),
            };
            let response = noise.emit(event, &mut self.rng);
            self.apply_noise_response(response);
        }
    }

    /// Emit an after-preparation noise event.
    fn emit_after_preparation(&mut self, qubits: &[QubitId]) {
        if let Some(ref mut noise) = self.noise {
            let event = NoiseEvent::AfterPreparation { qubits };
            let response = noise.emit(event, &mut self.rng);
            self.apply_noise_response(response);
        }
    }

    /// Emit an idle time noise event.
    fn emit_idle_time(&mut self, qubits: &[QubitId], duration: pecos_core::TimeUnits) {
        if let Some(ref mut noise) = self.noise {
            let event = NoiseEvent::IdleTime { qubits, duration };
            let response = noise.emit(event, &mut self.rng);
            self.apply_noise_response(response);
        }
    }

    /// Emit a before-measurement noise event.
    fn emit_before_measurement(&mut self, qubits: &[QubitId]) {
        if let Some(ref mut noise) = self.noise {
            let event = NoiseEvent::BeforeMeasurement { qubits };
            let response = noise.emit(event, &mut self.rng);
            self.apply_noise_response(response);
        }
    }

    /// Emit an after-measurement noise event.
    fn emit_after_measurement(&mut self, qubits: &[QubitId], outcomes: &[bool]) {
        if let Some(ref mut noise) = self.noise {
            let event = NoiseEvent::AfterMeasurement { qubits, outcomes };
            let response = noise.emit(event, &mut self.rng);
            self.apply_noise_response(response);
        }
    }

    /// Record measurement results.
    ///
    /// Handles leakage-aware measurement:
    /// - For `MeasureLeaked`: returns outcome with `is_leaked = true` for leaked qubits
    /// - For regular `Measure`: forces outcome to 1 for leaked qubits (as in hardware)
    fn record_measurements(
        &mut self,
        gate_type: GateType,
        qubits: &[QubitId],
        results: &[pecos_qsim::MeasurementResult],
    ) {
        for (&qubit, result) in qubits.iter().zip(results.iter()) {
            // Check if qubit is leaked (from noise context)
            let is_leaked = self
                .noise
                .as_ref()
                .is_some_and(|n| n.context().is_leaked(qubit));

            let outcome = if is_leaked {
                if gate_type == GateType::MeasureLeaked {
                    // MeasureLeaked: record with is_leaked flag (outcome value doesn't matter)
                    MeasurementOutcome::leaked(qubit)
                } else {
                    // Regular Measure on leaked qubit: force outcome to 1
                    MeasurementOutcome {
                        qubit,
                        outcome: true, // Leaked qubits return 1
                        is_deterministic: true,
                        is_leaked: true,
                    }
                }
            } else {
                // Normal measurement
                MeasurementOutcome::new(qubit, result.outcome, result.is_deterministic)
            };

            self.outcomes.record(outcome);
        }
    }

    /// Apply a noise response (inject gates, flip outcomes, etc.).
    fn apply_noise_response(&mut self, response: NoiseResponse) {
        match response {
            // No-ops: None, SkipGate (handled by caller), leakage tracking (handled by composer)
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
                // Mark these measurements as coming from leaked qubits (outcome = 2)
                for qubit in qubits {
                    self.outcomes.mark_leaked(qubit);
                }
            }

            NoiseResponse::ForceOutcomes(forced) => {
                // Force outcomes to specific values
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

    /// Execute a noise gate (injected Pauli error).
    fn execute_noise_gate(&mut self, gate: &GateCommand) {
        // Use as_slice() for zero-allocation access
        let qubits = gate.qubits.as_slice();

        match gate.gate_type {
            GateType::X => {
                self.simulator.x(qubits);
            }
            GateType::Y => {
                self.simulator.y(qubits);
            }
            GateType::Z => {
                self.simulator.z(qubits);
            }
            _ => {
                // Other gates shouldn't appear as noise, but handle gracefully
            }
        }
    }

    /// Execute Clifford gates on the simulator.
    ///
    /// Returns `true` if the gate was handled, `false` if it's a non-Clifford gate.
    fn execute_clifford_gate(&mut self, command: &GateCommand) -> bool {
        // Use as_slice() for zero-allocation access
        let qubits = command.qubits.as_slice();

        match command.gate_type {
            // Single-qubit Paulis
            GateType::I => {
                self.simulator.identity(qubits);
            }
            GateType::X => {
                self.simulator.x(qubits);
            }
            GateType::Y => {
                self.simulator.y(qubits);
            }
            GateType::Z => {
                self.simulator.z(qubits);
            }

            // Single-qubit Cliffords
            GateType::H => {
                self.simulator.h(qubits);
            }
            GateType::SX => {
                self.simulator.sx(qubits);
            }
            GateType::SXdg => {
                self.simulator.sxdg(qubits);
            }
            GateType::SY => {
                self.simulator.sy(qubits);
            }
            GateType::SYdg => {
                self.simulator.sydg(qubits);
            }
            GateType::SZ => {
                self.simulator.sz(qubits);
            }
            GateType::SZdg => {
                self.simulator.szdg(qubits);
            }

            // Two-qubit gates
            GateType::CX => {
                self.simulator.cx(qubits);
            }
            GateType::CY => {
                self.simulator.cy(qubits);
            }
            GateType::CZ => {
                self.simulator.cz(qubits);
            }
            GateType::SZZ => {
                self.simulator.szz(qubits);
            }
            GateType::SZZdg => {
                self.simulator.szzdg(qubits);
            }
            GateType::SWAP => {
                self.simulator.swap(qubits);
            }

            // Non-Clifford gates - not handled here
            _ => return false,
        }
        true
    }
}

/// Extension methods for simulators that support RNG management.
///
/// This impl block is only available when the simulator implements
/// `RngManageable`, enabling full determinism by seeding both the
/// noise RNG and the simulator's internal RNG.
impl<S> ShotRunner<S>
where
    S: CliffordGateable + RngManageable<Rng = PecosRng>,
{
    /// Set the seed for full determinism.
    ///
    /// This seeds both the noise RNG and the simulator's internal RNG
    /// using derived seeds from a single base seed. This ensures that
    /// both noise operations and measurement outcomes are deterministic.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_neo::prelude::*;
    /// use pecos_qsim::SparseStab;
    ///
    /// let commands = CommandBuilder::new()
    ///     .prep(0)
    ///     .h(0)
    ///     .measure(0)
    ///     .build();
    ///
    /// // Both runs will produce identical results
    /// let mut runner1 = ShotRunner::new(SparseStab::new(1)).with_full_seed(42);
    /// let mut runner2 = ShotRunner::new(SparseStab::new(1)).with_full_seed(42);
    ///
    /// let outcomes1 = runner1.run_shot(&commands);
    /// let outcomes2 = runner2.run_shot(&commands);
    ///
    /// assert_eq!(outcomes1.get_bit(QubitId(0)), outcomes2.get_bit(QubitId(0)));
    /// ```
    #[must_use]
    pub fn with_full_seed(mut self, seed: u64) -> Self {
        // Derive separate seeds for noise and simulator RNGs
        let noise_seed = derive_seed(seed, "noise");
        let sim_seed = derive_seed(seed, "simulator");

        self.rng = PecosRng::seed_from_u64(noise_seed);
        self.simulator.set_seed(sim_seed);
        self
    }

    /// Seed the simulator's internal RNG directly.
    ///
    /// This only seeds the simulator's RNG, not the noise RNG.
    /// For full determinism, use `with_full_seed()`.
    pub fn seed_simulator(&mut self, seed: u64) {
        self.simulator.set_seed(seed);
    }

    /// Set full seed (mutable version of `with_full_seed`).
    ///
    /// Seeds both the noise RNG and simulator RNG for full determinism.
    pub fn set_full_seed(&mut self, seed: u64) {
        let noise_seed = derive_seed(seed, "noise");
        let sim_seed = derive_seed(seed, "simulator");

        self.rng = PecosRng::seed_from_u64(noise_seed);
        self.simulator.set_seed(sim_seed);
    }
}

/// Extension methods for simulators that support arbitrary rotation gates.
///
/// This impl block is only available when the simulator implements
/// `ArbitraryRotationGateable`, enabling execution of non-Clifford gates
/// like RX, RY, RZ, RXX, RYY, RZZ, T, Tdg, etc.
impl<S: ArbitraryRotationGateable> ShotRunner<S> {
    /// Execute a command queue including arbitrary rotation gates.
    ///
    /// This method handles all gates including non-Clifford rotations.
    /// Use this when your simulator supports `ArbitraryRotationGateable`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pecos_neo::prelude::*;
    /// use some_statevec_sim::StateVec; // A simulator with ArbitraryRotationGateable
    ///
    /// let commands = CommandBuilder::new()
    ///     .prep(0)
    ///     .custom_gate(GateType::RX, &[0], &[Angle64::QUARTER_TURN])
    ///     .measure(0)
    ///     .build();
    ///
    /// let mut runner = ShotRunner::new(StateVec::new(1));
    /// let outcomes = runner.execute_all(&commands);
    /// ```
    pub fn execute_all(&mut self, commands: &CommandQueue) -> &MeasurementOutcomes {
        self.outcomes.clear();

        for command in commands {
            self.execute_command_universal(command);
        }

        &self.outcomes
    }

    /// Execute a single shot with rotation gates, then reset for next shot.
    pub fn run_shot_all(&mut self, commands: &CommandQueue) -> MeasurementOutcomes {
        self.execute_all(commands);
        let outcomes = std::mem::take(&mut self.outcomes);

        // Reset noise model state for next shot
        if let Some(ref mut noise) = self.noise {
            noise.reset();
        }

        outcomes
    }

    /// Execute a single command including rotation gates.
    fn execute_command_universal(&mut self, command: &GateCommand) {
        // Use as_slice() for zero-allocation access
        let qubits = command.qubits.as_slice();

        // Emit before-gate event for noise - may skip the gate (e.g., for leaked qubits)
        if self.emit_before_gate(command) {
            // Gate was skipped (e.g., due to leakage)
            // Still emit after-gate for channels that want to inject errors
            if !command.gate_type.is_measurement() && !command.gate_type.is_preparation() {
                self.emit_after_gate(command);
            }
            return;
        }

        // Execute the gate
        match command.gate_type {
            // Preparation
            GateType::Prep | GateType::QAlloc => {
                self.simulator.pz(qubits);
                self.emit_after_preparation(qubits);
            }

            // Measurement
            GateType::Measure | GateType::MeasureLeaked | GateType::MeasureFree => {
                self.emit_before_measurement(qubits);
                let results = self.simulator.mz(qubits);
                // Pre-size outcomes array based on results length (stack allocation for small cases)
                let outcomes: SmallVec<[bool; 4]> = results.iter().map(|r| r.outcome).collect();
                self.record_measurements(command.gate_type, qubits, &results);
                self.emit_after_measurement(qubits, outcomes.as_slice());
            }

            // Idle - emit idle time event for noise
            GateType::Idle => {
                if let Some(duration) = command.get_idle_duration() {
                    self.emit_idle_time(qubits, duration);
                }
            }

            // Try Clifford gates first, then rotation gates
            _ => {
                if !self.execute_clifford_gate(command) {
                    // Handle rotation and other non-Clifford gates
                    self.execute_rotation_gate(command, qubits);
                }
            }
        }

        // Emit after-gate event (except for measurement/prep/idle which have their own)
        if !command.gate_type.is_measurement()
            && !command.gate_type.is_preparation()
            && command.gate_type != GateType::Idle
        {
            self.emit_after_gate(command);
        }
    }

    /// Execute rotation and other non-Clifford gates.
    fn execute_rotation_gate(&mut self, command: &GateCommand, qubits: &[QubitId]) {
        let angle = command.angles.first().copied().unwrap_or(Angle64::ZERO);
        let angle2 = command.angles.get(1).copied().unwrap_or(Angle64::ZERO);
        let angle3 = command.angles.get(2).copied().unwrap_or(Angle64::ZERO);

        match command.gate_type {
            // Single-qubit rotations
            GateType::RX => {
                self.simulator.rx(angle, qubits);
            }
            GateType::RY => {
                self.simulator.ry(angle, qubits);
            }
            GateType::RZ => {
                self.simulator.rz(angle, qubits);
            }
            GateType::T => {
                self.simulator.t(qubits);
            }
            GateType::Tdg => {
                self.simulator.tdg(qubits);
            }
            GateType::U => {
                self.simulator.u(angle, angle2, angle3, qubits);
            }
            GateType::R1XY => {
                self.simulator.r1xy(angle, angle2, qubits);
            }

            // Two-qubit rotations
            GateType::RXX => {
                self.simulator.rxx(angle, qubits);
            }
            GateType::RYY => {
                self.simulator.ryy(angle, qubits);
            }
            GateType::RZZ => {
                self.simulator.rzz(angle, qubits);
            }

            // CRZ decomposition: RZ(theta/2), CX, RZ(-theta/2), CX
            GateType::CRZ => {
                if qubits.len() >= 2 {
                    let control = qubits[0];
                    let target = qubits[1];
                    let half_angle = angle / 2u64;
                    self.simulator.rz(half_angle, &[target]);
                    self.simulator.cx(&[control, target]);
                    self.simulator.rz(-half_angle, &[target]);
                    self.simulator.cx(&[control, target]);
                }
            }

            // CCX (Toffoli) decomposition using T, Tdg, H, CX
            GateType::CCX => {
                if qubits.len() >= 3 {
                    let c1 = qubits[0];
                    let c2 = qubits[1];
                    let target = qubits[2];
                    // Standard decomposition of Toffoli gate
                    self.simulator.h(&[target]);
                    self.simulator.cx(&[c2, target]);
                    self.simulator.tdg(&[target]);
                    self.simulator.cx(&[c1, target]);
                    self.simulator.t(&[target]);
                    self.simulator.cx(&[c2, target]);
                    self.simulator.tdg(&[target]);
                    self.simulator.cx(&[c1, target]);
                    self.simulator.t(&[c2]);
                    self.simulator.t(&[target]);
                    self.simulator.h(&[target]);
                    self.simulator.cx(&[c1, c2]);
                    self.simulator.t(&[c1]);
                    self.simulator.tdg(&[c2]);
                    self.simulator.cx(&[c1, c2]);
                }
            }

            // No-ops or handled elsewhere
            _ => {}
        }
    }
}

// ============================================================================
// Circuit Integration Methods
// ============================================================================

impl<S: CliffordGateable> ShotRunner<S> {
    /// Execute a [`TickCircuit`] and return measurement outcomes.
    ///
    /// This is a convenience method that converts the `TickCircuit` to a
    /// [`CommandQueue`] and executes it.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pecos_neo::prelude::*;
    /// use pecos_quantum::TickCircuit;
    /// use pecos_qsim::SparseStab;
    ///
    /// let mut circuit = TickCircuit::new();
    /// circuit.tick().pz(&[0, 1]);
    /// circuit.tick().h(&[0]);
    /// circuit.tick().cx(&[(0, 1)]);
    /// circuit.tick().mz(&[0, 1]);
    ///
    /// let mut runner = ShotRunner::new(SparseStab::new(2)).with_seed(42);
    /// let outcomes = runner.execute_tick(&circuit);
    /// ```
    pub fn execute_tick(
        &mut self,
        circuit: &pecos_quantum::TickCircuit,
    ) -> &MeasurementOutcomes {
        let commands = crate::command::CommandQueue::from(circuit);
        self.execute(&commands)
    }

    /// Execute a [`TickCircuit`] for a single shot, then reset for next shot.
    pub fn run_shot_tick(
        &mut self,
        circuit: &pecos_quantum::TickCircuit,
    ) -> MeasurementOutcomes {
        let commands = crate::command::CommandQueue::from(circuit);
        self.run_shot(&commands)
    }

    /// Execute a [`DagCircuit`] and return measurement outcomes.
    ///
    /// Gates are executed in topological order, respecting dependencies.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pecos_neo::prelude::*;
    /// use pecos_quantum::DagCircuit;
    /// use pecos_qsim::SparseStab;
    ///
    /// let mut dag = DagCircuit::new();
    /// dag.pz(0);
    /// dag.pz(1);
    /// dag.h(0);
    /// dag.cx(0, 1);
    /// dag.mz(0);
    /// dag.mz(1);
    ///
    /// let mut runner = ShotRunner::new(SparseStab::new(2)).with_seed(42);
    /// let outcomes = runner.execute_dag(&dag);
    /// ```
    pub fn execute_dag(
        &mut self,
        circuit: &pecos_quantum::DagCircuit,
    ) -> &MeasurementOutcomes {
        let commands = crate::command::CommandQueue::from(circuit);
        self.execute(&commands)
    }

    /// Execute a [`DagCircuit`] for a single shot, then reset for next shot.
    pub fn run_shot_dag(
        &mut self,
        circuit: &pecos_quantum::DagCircuit,
    ) -> MeasurementOutcomes {
        let commands = crate::command::CommandQueue::from(circuit);
        self.run_shot(&commands)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::CommandBuilder;
    use crate::noise::single_qubit::SingleQubitChannel;
    use pecos_qsim::{SparseStab, StateVec};

    #[test]
    fn test_basic_execution() {
        let commands = CommandBuilder::new().prep(0).h(0).measure(0).build();

        let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(42);
        let outcomes = runner.execute(&commands);

        assert_eq!(outcomes.len(), 1);
    }

    #[test]
    fn test_bell_state() {
        let commands = CommandBuilder::new()
            .prep(0)
            .prep(1)
            .h(0)
            .cx(0, 1)
            .measure(0)
            .measure(1)
            .build();

        let mut runner = ShotRunner::new(SparseStab::new(2)).with_seed(42);
        let outcomes = runner.execute(&commands);

        assert_eq!(outcomes.len(), 2);

        // Bell state: outcomes should be correlated
        let o0 = outcomes.get_bit(QubitId(0)).unwrap();
        let o1 = outcomes.get_bit(QubitId(1)).unwrap();
        assert_eq!(o0, o1, "Bell state outcomes should be equal");
    }

    #[test]
    fn test_with_noise() {
        let commands = CommandBuilder::new()
            .prep(0)
            .h(0) // Single-qubit gate will trigger noise
            .measure(0)
            .build();

        let noise = ComposableNoiseModel::new().add_channel(SingleQubitChannel::depolarizing(0.0)); // No actual noise

        let mut runner = ShotRunner::new(SparseStab::new(1))
            .with_noise(noise)
            .with_seed(42);

        let outcomes = runner.execute(&commands);
        assert_eq!(outcomes.len(), 1);
    }

    #[test]
    fn test_run_shot_resets() {
        let commands = CommandBuilder::new().prep(0).measure(0).build();

        let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(42);

        let outcomes1 = runner.run_shot(&commands);
        let outcomes2 = runner.run_shot(&commands);

        assert_eq!(outcomes1.len(), 1);
        assert_eq!(outcomes2.len(), 1);
    }

    #[test]
    fn test_measure_leaked_on_leaked_qubit() {
        use crate::command::GateCommand;
        use crate::noise::leakage::LeakageChannel;

        // Build a command queue with MeasureLeaked
        let mut commands = CommandBuilder::new().prep(0).build();
        commands.push(GateCommand::new(
            GateType::MeasureLeaked,
            smallvec::smallvec![QubitId(0)],
        ));

        // Create noise model and manually mark qubit as leaked
        let mut noise = ComposableNoiseModel::new().add_channel(LeakageChannel::new());
        noise.context_mut().mark_leaked(QubitId(0));

        let mut runner = ShotRunner::new(SparseStab::new(1))
            .with_noise(noise)
            .with_seed(42);
        let outcomes = runner.execute(&commands);

        // Should have an outcome with is_leaked = true
        let outcome = outcomes.get(QubitId(0)).unwrap();
        assert!(outcome.is_leaked, "MeasureLeaked on leaked qubit should set is_leaked");
        assert_eq!(
            outcome.as_int_leaked(),
            2,
            "MeasureLeaked on leaked qubit should return 2"
        );
    }

    #[test]
    fn test_regular_measure_on_leaked_qubit() {
        use crate::noise::leakage::LeakageChannel;

        // Build a command queue with regular Measure
        let commands = CommandBuilder::new().prep(0).measure(0).build();

        // Create noise model and manually mark qubit as leaked
        let mut noise = ComposableNoiseModel::new().add_channel(LeakageChannel::new());
        noise.context_mut().mark_leaked(QubitId(0));

        let mut runner = ShotRunner::new(SparseStab::new(1))
            .with_noise(noise)
            .with_seed(42);
        let outcomes = runner.execute(&commands);

        // Should have outcome = 1 (forced for leaked qubits)
        let outcome = outcomes.get(QubitId(0)).unwrap();
        assert!(outcome.outcome, "Regular Measure on leaked qubit should force outcome to 1");
        assert!(outcome.is_leaked, "Regular Measure on leaked qubit should mark is_leaked");
    }

    #[test]
    fn test_measure_leaked_on_non_leaked_qubit() {
        use crate::command::GateCommand;
        use crate::noise::leakage::LeakageChannel;

        // Build a command queue with MeasureLeaked on a non-leaked qubit
        let mut commands = CommandBuilder::new().prep(0).build();
        commands.push(GateCommand::new(
            GateType::MeasureLeaked,
            smallvec::smallvec![QubitId(0)],
        ));

        let noise = ComposableNoiseModel::new().add_channel(LeakageChannel::new());

        let mut runner = ShotRunner::new(SparseStab::new(1))
            .with_noise(noise)
            .with_seed(42);
        let outcomes = runner.execute(&commands);

        // Should have normal outcome (0 after Prep)
        let outcome = outcomes.get(QubitId(0)).unwrap();
        assert!(!outcome.is_leaked, "MeasureLeaked on non-leaked qubit should not set is_leaked");
        assert!(!outcome.outcome, "Prep followed by MeasureLeaked should give 0");
    }

    #[test]
    fn test_execute_all_with_rotation_gates() {
        use crate::command::GateCommand;
        use pecos_core::Angle64;

        // Build a command queue with rotation gates
        let mut commands = CommandBuilder::new().prep(0).build();

        // RX(pi) flips |0> to |1>
        commands.push(GateCommand::with_angles(
            GateType::RX,
            smallvec::smallvec![QubitId(0)],
            smallvec::smallvec![Angle64::HALF_TURN],
        ));

        commands.push(GateCommand::new(
            GateType::Measure,
            smallvec::smallvec![QubitId(0)],
        ));

        // Use StateVec which supports ArbitraryRotationGateable
        let mut runner = ShotRunner::new(StateVec::new(1)).with_seed(42);
        let outcomes = runner.execute_all(&commands);

        // RX(pi)|0> = i|1>, measurement gives 1
        let outcome = outcomes.get(QubitId(0)).unwrap();
        assert!(outcome.outcome, "RX(pi) on |0> should give |1>");
    }

    #[test]
    fn test_execute_all_rxx_gate() {
        use crate::command::GateCommand;
        use pecos_core::Angle64;

        // Prepare Bell-like state with RXX
        let mut commands = CommandBuilder::new().prep(0).prep(1).build();

        // H on both qubits
        commands.push(GateCommand::new(
            GateType::H,
            smallvec::smallvec![QubitId(0)],
        ));
        commands.push(GateCommand::new(
            GateType::H,
            smallvec::smallvec![QubitId(1)],
        ));

        // RXX(pi/2) entangles the qubits
        commands.push(GateCommand::with_angles(
            GateType::RXX,
            smallvec::smallvec![QubitId(0), QubitId(1)],
            smallvec::smallvec![Angle64::QUARTER_TURN],
        ));

        // Measure in X basis (H then measure)
        commands.push(GateCommand::new(
            GateType::H,
            smallvec::smallvec![QubitId(0)],
        ));
        commands.push(GateCommand::new(
            GateType::H,
            smallvec::smallvec![QubitId(1)],
        ));

        commands.push(GateCommand::new(
            GateType::Measure,
            smallvec::smallvec![QubitId(0)],
        ));
        commands.push(GateCommand::new(
            GateType::Measure,
            smallvec::smallvec![QubitId(1)],
        ));

        let mut runner = ShotRunner::new(StateVec::new(2)).with_seed(42);
        let outcomes = runner.execute_all(&commands);

        assert_eq!(outcomes.len(), 2);
    }

    #[test]
    fn test_run_shot_all() {
        use crate::command::GateCommand;
        use pecos_core::Angle64;

        let mut commands = CommandBuilder::new().prep(0).build();
        commands.push(GateCommand::with_angles(
            GateType::RX,
            smallvec::smallvec![QubitId(0)],
            smallvec::smallvec![Angle64::HALF_TURN], // RX(pi) = X gate, flips |0> to |1>
        ));
        commands.push(GateCommand::new(
            GateType::Measure,
            smallvec::smallvec![QubitId(0)],
        ));

        let mut runner = ShotRunner::new(StateVec::new(1)).with_seed(42);

        // Run multiple shots
        for _ in 0..10 {
            let outcomes = runner.run_shot_all(&commands);
            let outcome = outcomes.get(QubitId(0)).unwrap();
            // RX(pi) on |0> gives |1>
            assert!(outcome.outcome, "RX(pi) on |0> should give |1>");
        }
    }

    #[test]
    fn test_idle_time_emission() {
        use crate::noise::idle::IdleChannel;

        // Create a circuit with idle time
        let commands = CommandBuilder::new()
            .prep(0)
            .idle(0, 100u64) // 100 time units
            .measure(0)
            .build();

        // Create idle channel with 100% error rate per ns
        // With 100ns, we should get 100% error probability (capped at 1.0)
        let noise = ComposableNoiseModel::new().add_channel(IdleChannel::linear(1.0));

        let mut runner = ShotRunner::new(SparseStab::new(1))
            .with_noise(noise)
            .with_seed(42);

        let outcomes = runner.execute(&commands);
        assert_eq!(outcomes.len(), 1);
        // The idle channel should have triggered (Z errors on the qubit)
    }

    #[test]
    fn test_noise_on_rzz_gate() {
        use crate::noise::two_qubit::TwoQubitChannel;
        use pecos_core::Angle64;

        // Build a circuit with RZZ gate
        let commands = CommandBuilder::new()
            .prep(0)
            .prep(1)
            .rzz(0, 1, Angle64::QUARTER_TURN)
            .measure(0)
            .measure(1)
            .build();

        // Create a two-qubit channel that responds to RZZ
        let noise = ComposableNoiseModel::new().add_channel(TwoQubitChannel::depolarizing(0.0)); // No noise for determinism

        let mut runner = ShotRunner::new(StateVec::new(2))
            .with_noise(noise)
            .with_seed(42);

        let outcomes = runner.execute_all(&commands);
        assert_eq!(outcomes.len(), 2);
    }

    #[test]
    fn test_crz_decomposition() {
        use pecos_core::Angle64;

        // CRZ(theta) should apply a controlled-RZ
        // Test: |+1> -> CRZ(pi) -> |+1> (control=1, so RZ(pi) is applied to target)
        let commands = CommandBuilder::new()
            .prep(0)
            .prep(1)
            .x(0) // Control qubit = |1>
            .h(1) // Target qubit = |+>
            .gate(GateCommand::with_angles(
                GateType::CRZ,
                smallvec::smallvec![QubitId(0), QubitId(1)],
                smallvec::smallvec![Angle64::HALF_TURN], // CRZ(pi)
            ))
            .h(1) // Convert phase to amplitude
            .measure(1)
            .build();

        let mut runner = ShotRunner::new(StateVec::new(2)).with_seed(42);
        let outcomes = runner.execute_all(&commands);

        // With control=1 and CRZ(pi), target |+> becomes |->
        // After H, |-> becomes |1>
        let outcome = outcomes.get(QubitId(1)).unwrap();
        assert!(outcome.outcome, "CRZ(pi) with control=1 should flip target phase");
    }

    #[test]
    fn test_ccx_decomposition() {
        // CCX (Toffoli) should flip target when both controls are 1
        let commands = CommandBuilder::new()
            .prep(0)
            .prep(1)
            .prep(2)
            .x(0) // Control 1 = |1>
            .x(1) // Control 2 = |1>
            // Target = |0>, should flip to |1>
            .ccx(0, 1, 2)
            .measure(2)
            .build();

        let mut runner = ShotRunner::new(StateVec::new(3)).with_seed(42);
        let outcomes = runner.execute_all(&commands);

        // With both controls = 1, target should flip from 0 to 1
        let outcome = outcomes.get(QubitId(2)).unwrap();
        assert!(outcome.outcome, "CCX with both controls=1 should flip target");
    }

    #[test]
    fn test_ccx_no_flip_when_control_zero() {
        // CCX should NOT flip target when one control is 0
        let commands = CommandBuilder::new()
            .prep(0)
            .prep(1)
            .prep(2)
            .x(0) // Control 1 = |1>
            // Control 2 = |0> (not set)
            // Target = |0>, should stay |0>
            .ccx(0, 1, 2)
            .measure(2)
            .build();

        let mut runner = ShotRunner::new(StateVec::new(3)).with_seed(42);
        let outcomes = runner.execute_all(&commands);

        // With one control = 0, target should stay 0
        let outcome = outcomes.get(QubitId(2)).unwrap();
        assert!(!outcome.outcome, "CCX with one control=0 should not flip target");
    }
}
