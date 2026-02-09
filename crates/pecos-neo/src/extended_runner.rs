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

//! Extended runner for GateId-based execution with decomposition.
//!
//! This runner executes `AdaptedSequence` (which uses `GateId`) and supports:
//! - Trait-based native gate execution (compile-time checked)
//! - Custom gate overrides (including for core gates)
//! - Automatic decomposition fallback from `GateDefinitions`
//! - Noise integration with full gate metadata
//!
//! # Design Philosophy
//!
//! Gate support is determined by trait bounds at compile time:
//! - `CliffordGateable` → H, X, Y, Z, SX, SY, SZ, CX, CY, CZ, SWAP, etc.
//! - `ArbitraryRotationGateable` → T, Tdg, RX, RY, RZ, RXX, RYY, RZZ
//!
//! Custom gates or temporary implementation swaps use `GateOverrides`.
//!
//! # Unified `run()` Method
//!
//! There is a single `run()` method that handles all gates. The constructor
//! determines which gates are executed natively:
//!
//! - `ExtendedRunner::new()` - Clifford gates only, rotation gates decompose
//! - `ExtendedRunner::rotations()` - Clifford + rotation gates (requires
//!   `ArbitraryRotationGateable` simulator)
//!
//! # Example
//!
//! ```ignore
//! use pecos_neo::extended_runner::{ExtendedRunner, GateOverrides};
//! use pecos_neo::extensible::{GateDefinitions, OpBuilder, gates};
//! use pecos_qsim::{SparseStab, StateVec};
//!
//! let definitions = GateDefinitions::new();
//!
//! // Clifford-only circuit with Clifford simulator
//! let clifford_circuit = OpBuilder::new()
//!     .prep_z(QubitId(0))
//!     .h(QubitId(0))
//!     .meas_z(QubitId(0), ResultId(0))
//!     .build();
//!
//! let mut runner = ExtendedRunner::new(SparseStab::new(1), definitions.clone());
//! let outcomes = runner.run(&clifford_circuit)?;
//!
//! // Circuit with rotations using rotation-capable simulator
//! let rotation_circuit = OpBuilder::new()
//!     .prep_z(QubitId(0))
//!     .rx(QubitId(0), Angle64::QUARTER_TURN)
//!     .meas_z(QubitId(0), ResultId(0))
//!     .build();
//!
//! let mut runner = ExtendedRunner::rotations(StateVec::new(1), definitions.clone());
//! let outcomes = runner.run(&rotation_circuit)?;  // Same run() method!
//!
//! // With custom gate overrides
//! let overrides = GateOverrides::new()
//!     .register(my_custom_gate, |sim, qubits, _angles| {
//!         sim.h(qubits);  // Implement as H
//!         true
//!     });
//! let mut runner = ExtendedRunner::new(sim, definitions)
//!     .with_overrides(overrides);
//! ```

use crate::command::GateType;
use crate::extensible::{
    AdaptedOp, AdaptedSequence, GateDefinitions, GateId,
    MeasBasis, PrepBasis, ResultId,
};
use std::collections::HashMap;
use crate::noise::{ComposableNoiseModel, NoiseEvent, NoiseResponse};
use crate::outcome::{MeasurementOutcome, MeasurementOutcomes};
use pecos_core::{Angle64, QubitId};
use pecos_qsim::{ArbitraryRotationGateable, CliffordGateable};
use pecos_rng::PecosRng;
use rand_core::SeedableRng;

/// Function signature for custom gate executors.
///
/// Takes a mutable reference to the simulator, qubit operands, and angles.
/// Returns `true` if the gate was executed successfully, `false` otherwise.
pub type GateExecutorFn<S> = fn(&mut S, &[QubitId], &[Angle64]) -> bool;

/// Function signature for rotation gate execution.
///
/// Used internally to enable `run()` to execute rotation gates when the
/// simulator supports `ArbitraryRotationGateable`. Set by `rotations()` constructor.
type RotationExecutorFn<S> = fn(&mut S, GateId, &[QubitId], &[Angle64]) -> bool;

/// Registry of custom gate implementations.
///
/// Allows registering custom executors for any `GateId`, including core gates.
/// When a gate has an override registered, the override takes precedence over
/// both trait-based execution and decomposition.
///
/// # Example
///
/// ```ignore
/// use pecos_neo::extended_runner::GateOverrides;
/// use pecos_neo::extensible::gates;
///
/// // Register a custom implementation for a gate
/// let overrides: GateOverrides<SparseStab> = GateOverrides::new()
///     .register(my_custom_gate, |sim, qubits, _angles| {
///         sim.h(qubits);
///         true
///     });
///
/// // Override a core gate with different behavior
/// let overrides = overrides
///     .register(gates::H, |sim, qubits, _angles| {
///         // Custom H implementation (e.g., for debugging)
///         sim.sz(qubits);
///         sim.h(qubits);
///         sim.szdg(qubits);
///         true
///     });
/// ```
pub struct GateOverrides<S> {
    overrides: HashMap<GateId, GateExecutorFn<S>>,
}

impl<S> Default for GateOverrides<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S> GateOverrides<S> {
    /// Create an empty override registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            overrides: HashMap::new(),
        }
    }

    /// Register a custom executor for a gate.
    ///
    /// The executor function takes the simulator, qubits, and angles,
    /// and returns `true` if execution succeeded.
    #[must_use]
    pub fn register(mut self, gate_id: GateId, executor: GateExecutorFn<S>) -> Self {
        self.overrides.insert(gate_id, executor);
        self
    }

    /// Register a custom executor (mutable version).
    pub fn insert(&mut self, gate_id: GateId, executor: GateExecutorFn<S>) {
        self.overrides.insert(gate_id, executor);
    }

    /// Remove an override.
    pub fn remove(&mut self, gate_id: GateId) -> Option<GateExecutorFn<S>> {
        self.overrides.remove(&gate_id)
    }

    /// Check if a gate has an override.
    #[must_use]
    pub fn contains(&self, gate_id: GateId) -> bool {
        self.overrides.contains_key(&gate_id)
    }

    /// Get the executor for a gate, if registered.
    #[must_use]
    pub fn get(&self, gate_id: GateId) -> Option<&GateExecutorFn<S>> {
        self.overrides.get(&gate_id)
    }

    /// Number of registered overrides.
    #[must_use]
    pub fn len(&self) -> usize {
        self.overrides.len()
    }

    /// Check if no overrides are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.overrides.is_empty()
    }
}

/// Extended runner that executes GateId-based circuits with decomposition support.
///
/// Unlike `ShotRunner` which uses `GateType`, this runner works with `GateId`
/// enabling uniform treatment of core and custom gates.
///
/// # Gate Execution Order
///
/// Gates are executed in this order of precedence:
/// 1. **Overrides**: Custom executors registered via `GateOverrides`
/// 2. **Trait methods**: Core gates via `CliffordGateable` / `ArbitraryRotationGateable`
/// 3. **Decomposition**: Expand using `GateDefinitions`
/// 4. **Error**: If none of the above apply
///
/// # Usage
///
/// For Clifford-only simulators (e.g., `SparseStab`):
/// ```ignore
/// let mut runner = ExtendedRunner::new(SparseStab::new(2), definitions);
/// let outcomes = runner.run(&circuit)?;
/// ```
///
/// For rotation-capable simulators (e.g., `StateVec`):
/// ```ignore
/// let mut runner = ExtendedRunner::rotations(StateVec::new(2), definitions);
/// let outcomes = runner.run(&circuit)?;  // Rotation gates work automatically
/// ```
///
/// With custom gate overrides:
/// ```ignore
/// let overrides = GateOverrides::new()
///     .register(my_gate, |sim, qubits, _| { sim.h(qubits); true });
/// let mut runner = ExtendedRunner::new(sim, definitions)
///     .with_overrides(overrides);
/// ```
pub struct ExtendedRunner<S: CliffordGateable> {
    /// The quantum simulator.
    simulator: S,
    /// Gate definitions for decomposition and metadata.
    definitions: GateDefinitions,
    /// Custom gate implementations (overrides trait methods and decomposition).
    overrides: Option<GateOverrides<S>>,
    /// Rotation gate executor - set by `rotations()` constructor for
    /// simulators implementing `ArbitraryRotationGateable`.
    rotation_executor: Option<RotationExecutorFn<S>>,
    /// Optional noise model.
    noise: Option<ComposableNoiseModel>,
    /// RNG for noise operations.
    rng: PecosRng,
    /// Measurement outcomes.
    outcomes: MeasurementOutcomes,
    /// Internal result storage for conditional operations.
    results: Vec<bool>,
    /// Maximum decomposition depth to prevent infinite recursion.
    max_decomp_depth: usize,
}

impl<S: CliffordGateable> ExtendedRunner<S> {
    /// Create a new runner with Clifford gate support.
    ///
    /// The simulator can natively execute standard Clifford gates (H, X, Y, Z,
    /// SX, SY, SZ, CX, CY, CZ, SWAP, etc.). Other gates require decomposition
    /// or custom overrides.
    pub fn new(simulator: S, definitions: GateDefinitions) -> Self {
        Self {
            simulator,
            definitions,
            overrides: None,
            rotation_executor: None,
            noise: None,
            rng: PecosRng::from_rng(&mut rand::rng()),
            outcomes: MeasurementOutcomes::new(),
            results: Vec::new(),
            max_decomp_depth: 10,
        }
    }

    /// Check if rotation gates are enabled.
    ///
    /// Returns `true` if this runner was created with `rotations()` constructor.
    #[must_use]
    pub fn has_rotation_support(&self) -> bool {
        self.rotation_executor.is_some()
    }

    /// Check if a gate has an override registered.
    #[must_use]
    pub fn has_override(&self, gate_id: GateId) -> bool {
        self.overrides.as_ref().is_some_and(|o| o.contains(gate_id))
    }

    /// Get the overrides registry (for inspection).
    #[must_use]
    pub fn overrides(&self) -> Option<&GateOverrides<S>> {
        self.overrides.as_ref()
    }

    /// Set custom gate overrides.
    #[must_use]
    pub fn with_overrides(mut self, overrides: GateOverrides<S>) -> Self {
        self.overrides = Some(overrides);
        self
    }

    /// Set the noise model.
    #[must_use]
    pub fn with_noise(mut self, mut noise: ComposableNoiseModel) -> Self {
        // Propagate gate definitions to noise model
        noise = noise.with_gate_definitions(self.definitions.clone());
        self.noise = Some(noise);
        self
    }

    /// Set the RNG seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = PecosRng::seed_from_u64(seed);
        self
    }

    /// Set maximum decomposition depth.
    #[must_use]
    pub fn with_max_decomp_depth(mut self, depth: usize) -> Self {
        self.max_decomp_depth = depth;
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

    /// Get gate definitions.
    #[must_use]
    pub fn definitions(&self) -> &GateDefinitions {
        &self.definitions
    }

    /// Execute a circuit and return measurement outcomes.
    pub fn run(&mut self, circuit: &AdaptedSequence) -> Result<&MeasurementOutcomes, ExecutionError> {
        self.outcomes.clear();
        self.results.clear();
        self.results.resize(circuit.result_count, false);

        self.execute_ops(&circuit.ops, 0)?;

        Ok(&self.outcomes)
    }

    /// Execute a single shot and return outcomes, then reset.
    pub fn run_shot(&mut self, circuit: &AdaptedSequence) -> Result<MeasurementOutcomes, ExecutionError> {
        self.run(circuit)?;
        let outcomes = std::mem::take(&mut self.outcomes);

        // Reset for next shot
        self.simulator.reset();
        if let Some(ref mut noise) = self.noise {
            noise.reset();
        }

        Ok(outcomes)
    }

    /// Execute a list of operations.
    fn execute_ops(&mut self, ops: &[AdaptedOp], depth: usize) -> Result<(), ExecutionError> {
        if depth > self.max_decomp_depth {
            return Err(ExecutionError::MaxDecompositionDepthExceeded);
        }

        for op in ops {
            self.execute_op(op, depth)?;
        }
        Ok(())
    }

    /// Execute a single operation.
    fn execute_op(&mut self, op: &AdaptedOp, depth: usize) -> Result<(), ExecutionError> {
        match op {
            AdaptedOp::Gate { gate_id, qubits, angles } => {
                self.execute_gate(*gate_id, qubits, angles, depth)?;
            }
            AdaptedOp::Prep { qubit, basis } => {
                self.execute_prep(*qubit, *basis);
            }
            AdaptedOp::Measure { qubit, basis, result } => {
                self.execute_measure(*qubit, *basis, *result);
            }
            AdaptedOp::Conditional { condition, if_one, if_zero } => {
                let result_val = self.results.get(condition.0 as usize).copied().unwrap_or(false);
                if result_val {
                    self.execute_ops(if_one, depth)?;
                } else {
                    self.execute_ops(if_zero, depth)?;
                }
            }
            AdaptedOp::XorResult { target, source } => {
                let src_val = self.results.get(source.0 as usize).copied().unwrap_or(false);
                if let Some(tgt) = self.results.get_mut(target.0 as usize) {
                    *tgt ^= src_val;
                }
            }
            AdaptedOp::OutputResult { .. } => {
                // Output marking - handled by caller
            }
        }
        Ok(())
    }

    /// Execute a gate using overrides, trait methods, or decomposition.
    ///
    /// Execution order:
    /// 1. Check overrides - custom implementations take precedence
    /// 2. Try Clifford trait methods (for core Clifford gates)
    /// 3. Try rotation trait methods (if `rotations()` constructor was used)
    /// 4. Fall back to decomposition from `GateDefinitions`
    /// 5. Error if none of the above succeed
    fn execute_gate(
        &mut self,
        gate_id: GateId,
        qubits: &[QubitId],
        angles: &[Angle64],
        depth: usize,
    ) -> Result<(), ExecutionError> {
        // Emit before-gate noise event
        let skip = self.emit_before_gate(gate_id, qubits, angles);
        if skip {
            return Ok(());
        }

        // Try execution in order of precedence:
        // 1. Overrides
        // 2. Clifford gates
        // 3. Rotation gates (if enabled via rotations() constructor)
        let executed = self.try_execute_override(gate_id, qubits, angles)
            || self.try_execute_clifford(gate_id, qubits)
            || self.rotation_executor
                .is_some_and(|executor| executor(&mut self.simulator, gate_id, qubits, angles));

        if !executed {
            // Fall back to decomposition
            self.execute_via_decomposition(gate_id, qubits, angles, depth)?;
        }

        // Emit after-gate noise event
        self.emit_after_gate(gate_id, qubits, angles);

        Ok(())
    }

    /// Try to execute a gate via registered overrides.
    ///
    /// Returns `true` if an override was found and executed successfully.
    fn try_execute_override(&mut self, gate_id: GateId, qubits: &[QubitId], angles: &[Angle64]) -> bool {
        self.overrides
            .as_ref()
            .and_then(|o| o.get(gate_id))
            .is_some_and(|executor| executor(&mut self.simulator, qubits, angles))
    }

    /// Try to execute a Clifford gate natively via trait methods.
    ///
    /// Returns `true` if the gate was executed, `false` if it's not a Clifford gate.
    /// This method only handles gates that `CliffordGateable` supports.
    fn try_execute_clifford(&mut self, gate_id: GateId, qubits: &[QubitId]) -> bool {
        // Only core gates can be dispatched via GateType
        let Some(gate_type) = gate_id.try_to_gate_type() else {
            return false;
        };

        // Execute Clifford gates via CliffordGateable trait
        match gate_type {
            // Identity / Idle - no action needed
            GateType::I | GateType::Idle => true,

            // Single-qubit Paulis
            GateType::X => { self.simulator.x(qubits); true }
            GateType::Y => { self.simulator.y(qubits); true }
            GateType::Z => { self.simulator.z(qubits); true }

            // Single-qubit Cliffords
            GateType::H => { self.simulator.h(qubits); true }
            GateType::SX => { self.simulator.sx(qubits); true }
            GateType::SXdg => { self.simulator.sxdg(qubits); true }
            GateType::SY => { self.simulator.sy(qubits); true }
            GateType::SYdg => { self.simulator.sydg(qubits); true }
            GateType::SZ => { self.simulator.sz(qubits); true }
            GateType::SZdg => { self.simulator.szdg(qubits); true }

            // Two-qubit Cliffords
            GateType::CX => { self.simulator.cx(qubits); true }
            GateType::CY => { self.simulator.cy(qubits); true }
            GateType::CZ => { self.simulator.cz(qubits); true }
            GateType::SWAP => { self.simulator.swap(qubits); true }
            GateType::SZZ => { self.simulator.szz(qubits); true }
            GateType::SZZdg => { self.simulator.szzdg(qubits); true }

            // Everything else: rotation gates, prep/measure, multi-qubit gates
            // These either need ArbitraryRotationGateable or decomposition
            _ => false,
        }
    }

    /// Execute a gate via decomposition.
    fn execute_via_decomposition(
        &mut self,
        gate_id: GateId,
        qubits: &[QubitId],
        angles: &[Angle64],
        depth: usize,
    ) -> Result<(), ExecutionError> {
        // Look up decomposition and collect ops to avoid borrowing self across execute_gate calls
        let Some(decomp_entry) = self.definitions.decomposition(gate_id) else {
            return Err(ExecutionError::NoDecomposition { gate_id });
        };

        // Collect and instantiate decomposed operations first to release the borrow
        let instantiated_ops: Vec<_> = decomp_entry
            .decomposition
            .expand()
            .map(|op| op.instantiate(qubits, angles))
            .collect();

        // Now execute each operation
        for inst in instantiated_ops {
            self.execute_gate(inst.gate, &inst.qubits, &inst.angles, depth + 1)?;
        }

        Ok(())
    }

    /// Execute preparation.
    fn execute_prep(&mut self, qubit: QubitId, basis: PrepBasis) {
        // Standard prep is in Z basis
        self.simulator.pz(&[qubit]);

        // Rotate to target basis
        match basis {
            PrepBasis::Z => {} // Already in Z
            PrepBasis::X => { self.simulator.h(&[qubit]); }
            PrepBasis::Y => {
                self.simulator.h(&[qubit]);
                self.simulator.sz(&[qubit]);
            }
        }
    }

    /// Execute measurement.
    fn execute_measure(&mut self, qubit: QubitId, basis: MeasBasis, result_id: ResultId) {
        // Rotate to Z basis for measurement
        match basis {
            MeasBasis::Z => {}
            MeasBasis::X => { self.simulator.h(&[qubit]); }
            MeasBasis::Y => {
                self.simulator.szdg(&[qubit]);
                self.simulator.h(&[qubit]);
            }
        }

        // Perform measurement - mz returns Vec<MeasurementResult>
        let results = self.simulator.mz(&[qubit]);
        let meas_result = results.first();
        let outcome = meas_result.is_some_and(|r| r.outcome);
        let is_deterministic = meas_result.is_none_or(|r| r.is_deterministic);

        // Store result
        if let Some(slot) = self.results.get_mut(result_id.0 as usize) {
            *slot = outcome;
        }

        // Record in outcomes
        self.outcomes.record(MeasurementOutcome::new(qubit, outcome, is_deterministic));

        // Rotate back (for non-destructive measurement semantics)
        match basis {
            MeasBasis::Z => {}
            MeasBasis::X => { self.simulator.h(&[qubit]); }
            MeasBasis::Y => {
                self.simulator.h(&[qubit]);
                self.simulator.sz(&[qubit]);
            }
        }
    }

    /// Emit before-gate noise event, returns true if gate should be skipped.
    fn emit_before_gate(&mut self, gate_id: GateId, qubits: &[QubitId], angles: &[Angle64]) -> bool {
        let Some(ref mut noise) = self.noise else {
            return false;
        };

        // Get GateType if available (for noiseless gate check)
        let gate_type = gate_id.try_to_gate_type().unwrap_or(GateType::I);

        let event = NoiseEvent::BeforeGate {
            gate_type,
            qubits,
            angles,
            gate_id: Some(gate_id),
        };

        let response = noise.emit(event, &mut self.rng);
        let should_skip = response.should_skip_gate();
        self.apply_noise_response(response);
        should_skip
    }

    /// Emit after-gate noise event.
    fn emit_after_gate(&mut self, gate_id: GateId, qubits: &[QubitId], angles: &[Angle64]) {
        let Some(ref mut noise) = self.noise else {
            return;
        };

        let gate_type = gate_id.try_to_gate_type().unwrap_or(GateType::I);

        let event = NoiseEvent::AfterGate {
            gate_type,
            qubits,
            angles,
            gate_id: Some(gate_id),
        };

        let response = noise.emit(event, &mut self.rng);
        self.apply_noise_response(response);
    }

    /// Apply a noise response.
    ///
    /// # Panics
    /// Panics if a noise channel injects a gate type that isn't a Pauli (I, X, Y, Z).
    fn apply_noise_response(&mut self, response: NoiseResponse) {
        match response {
            // No action needed:
            // - None/SkipGate: SkipGate is handled at call site via should_skip_gate
            // - Leakage tracking: Handled by the noise context, not the runner
            NoiseResponse::None
            | NoiseResponse::SkipGate
            | NoiseResponse::MarkLeaked(_)
            | NoiseResponse::MarkUnleaked(_)
            | NoiseResponse::LeakedMeasurement(_) => {}

            // Execute injected gates (must be Paulis)
            NoiseResponse::InjectGates(gates) => {
                for gate_cmd in gates.iter() {
                    let qubits = gate_cmd.qubits.as_slice();
                    match gate_cmd.gate_type {
                        GateType::I => {} // Identity - no action
                        GateType::X => { self.simulator.x(qubits); }
                        GateType::Y => { self.simulator.y(qubits); }
                        GateType::Z => { self.simulator.z(qubits); }
                        other => {
                            panic!(
                                "ExtendedRunner: noise channel injected unsupported gate type {other:?}. \
                                 Only Pauli gates (I, X, Y, Z) are supported for noise injection."
                            );
                        }
                    }
                }
            }

            // Outcome manipulation - these should be handled in the measurement path
            // by the noise model before we record outcomes. If we get here, it means
            // the noise model returned these after the fact, which we can't handle.
            NoiseResponse::FlipOutcomes(qubits) => {
                if !qubits.is_empty() {
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "Warning: ExtendedRunner received FlipOutcomes for {} qubits but \
                         cannot retroactively flip outcomes. This may indicate a noise model issue.",
                        qubits.len()
                    );
                }
            }
            NoiseResponse::ForceOutcomes(outcomes) => {
                if !outcomes.is_empty() {
                    #[cfg(debug_assertions)]
                    eprintln!(
                        "Warning: ExtendedRunner received ForceOutcomes for {} qubits but \
                         cannot retroactively force outcomes. This may indicate a noise model issue.",
                        outcomes.len()
                    );
                }
            }

            // Recursive case for combined responses
            NoiseResponse::Multiple(responses) => {
                for r in responses {
                    self.apply_noise_response(r);
                }
            }
        }
    }
}

/// Extension for simulators that support arbitrary rotations.
///
/// When using a simulator that implements `ArbitraryRotationGateable`, use the
/// `rotations()` constructor to enable native execution of rotation gates.
/// The same `run()` method works - rotation gates are handled automatically.
impl<S> ExtendedRunner<S>
where
    S: CliffordGateable + ArbitraryRotationGateable,
{
    /// Create a runner with rotation gate support.
    ///
    /// For simulators implementing `ArbitraryRotationGateable`, this constructor
    /// enables native execution of rotation gates (T, Tdg, RX, RY, RZ, RXX, RYY, RZZ).
    ///
    /// Use the same `run()` method - rotation gates are handled automatically.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use pecos_neo::prelude::*;
    /// use pecos_qsim::StateVec;
    ///
    /// let circuit = OpBuilder::new()
    ///     .prep_z(QubitId(0))
    ///     .gate1_angle(gates::RX, QubitId(0), Angle64::QUARTER_TURN)
    ///     .meas_z(QubitId(0), ResultId(0))
    ///     .build();
    ///
    /// // RX is executed natively via ArbitraryRotationGateable
    /// let mut runner = ExtendedRunner::rotations(StateVec::new(1), definitions);
    /// let outcomes = runner.run(&circuit)?;
    /// ```
    pub fn rotations(simulator: S, definitions: GateDefinitions) -> Self {
        let mut runner = Self::new(simulator, definitions);
        runner.rotation_executor = Some(Self::execute_rotation_gate);
        runner
    }

    /// Execute a rotation gate natively.
    ///
    /// This is a static function used as a callback when rotation support is enabled.
    fn execute_rotation_gate(sim: &mut S, gate_id: GateId, qubits: &[QubitId], angles: &[Angle64]) -> bool {
        let Some(gate_type) = gate_id.try_to_gate_type() else {
            return false;
        };

        match gate_type {
            GateType::T => { sim.t(qubits); true }
            GateType::Tdg => { sim.tdg(qubits); true }
            GateType::RX => {
                if let Some(&angle) = angles.first() {
                    sim.rx(angle, qubits);
                    true
                } else {
                    false
                }
            }
            GateType::RY => {
                if let Some(&angle) = angles.first() {
                    sim.ry(angle, qubits);
                    true
                } else {
                    false
                }
            }
            GateType::RZ => {
                if let Some(&angle) = angles.first() {
                    sim.rz(angle, qubits);
                    true
                } else {
                    false
                }
            }
            GateType::RXX => {
                if let Some(&angle) = angles.first() {
                    sim.rxx(angle, qubits);
                    true
                } else {
                    false
                }
            }
            GateType::RYY => {
                if let Some(&angle) = angles.first() {
                    sim.ryy(angle, qubits);
                    true
                } else {
                    false
                }
            }
            GateType::RZZ => {
                if let Some(&angle) = angles.first() {
                    sim.rzz(angle, qubits);
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }

}

/// Errors during execution.
#[derive(Debug, Clone)]
pub enum ExecutionError {
    /// No decomposition found for a gate.
    NoDecomposition { gate_id: GateId },
    /// Maximum decomposition depth exceeded (possible infinite recursion).
    MaxDecompositionDepthExceeded,
}

impl std::fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoDecomposition { gate_id } => {
                write!(f, "No decomposition found for gate ID {}", gate_id.0)
            }
            Self::MaxDecompositionDepthExceeded => {
                write!(f, "Maximum decomposition depth exceeded")
            }
        }
    }
}

impl std::error::Error for ExecutionError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extensible::{GateSpec, GateCategory, OpBuilder, gates};
    use pecos_qsim::SparseStab;

    #[test]
    fn test_basic_execution() {
        let gates_def = GateDefinitions::new();
        let circuit = OpBuilder::new()
            .prep_z(QubitId(0))
            .h(QubitId(0))
            .meas_z(QubitId(0), ResultId(0))
            .build();

        let mut runner = ExtendedRunner::new(SparseStab::new(1), gates_def)
            .with_seed(42);

        let outcomes = runner.run(&circuit).unwrap();
        assert_eq!(outcomes.len(), 1);
    }

    #[test]
    fn test_bell_state() {
        let gates_def = GateDefinitions::new();
        let circuit = OpBuilder::new()
            .prep_z(QubitId(0))
            .prep_z(QubitId(1))
            .h(QubitId(0))
            .cx(QubitId(0), QubitId(1))
            .meas_z(QubitId(0), ResultId(0))
            .meas_z(QubitId(1), ResultId(1))
            .build();

        let mut runner = ExtendedRunner::new(SparseStab::new(2), gates_def)
            .with_seed(42);

        let outcomes = runner.run(&circuit).unwrap();
        assert_eq!(outcomes.len(), 2);

        // Bell state: outcomes should be correlated
        let o0 = outcomes.get_bit(QubitId(0)).unwrap();
        let o1 = outcomes.get_bit(QubitId(1)).unwrap();
        assert_eq!(o0, o1, "Bell state outcomes should be equal");
    }

    #[test]
    fn test_conditional_operation() {
        let gates_def = GateDefinitions::new();

        // Prepare |1⟩, measure, conditionally apply X
        let circuit = OpBuilder::new()
            .prep_z(QubitId(0))
            .prep_z(QubitId(1))
            .x(QubitId(0))  // |1⟩
            .meas_z(QubitId(0), ResultId(0))
            .if_one(ResultId(0), |b| b.x(QubitId(1)))  // Flip qubit 1 if qubit 0 is 1
            .meas_z(QubitId(1), ResultId(1))
            .build();

        let mut runner = ExtendedRunner::new(SparseStab::new(2), gates_def)
            .with_seed(42);

        let outcomes = runner.run(&circuit).unwrap();

        // Qubit 0 measured 1, so qubit 1 should be flipped to 1
        assert!(outcomes.get_bit(QubitId(0)).unwrap());
        assert!(outcomes.get_bit(QubitId(1)).unwrap());
    }

    #[test]
    fn test_custom_gate_needs_decomposition() {
        let mut gates_def = GateDefinitions::new();

        // Register a custom gate without decomposition
        let custom_id = gates_def.register(
            GateSpec::new("CustomNoDecomp")
                .with_quantum_arity(1)
                .with_category(GateCategory::SingleQubitUnitary)
        );

        let circuit = OpBuilder::new()
            .prep_z(QubitId(0))
            .gate1(custom_id, QubitId(0))
            .meas_z(QubitId(0), ResultId(0))
            .build();

        let mut runner = ExtendedRunner::new(SparseStab::new(1), gates_def);

        // Should fail because no decomposition
        let result = runner.run(&circuit);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ExecutionError::NoDecomposition { .. }));
    }

    #[test]
    fn test_run_shot_resets() {
        let gates_def = GateDefinitions::new();
        let circuit = OpBuilder::new()
            .prep_z(QubitId(0))
            .meas_z(QubitId(0), ResultId(0))
            .build();

        let mut runner = ExtendedRunner::new(SparseStab::new(1), gates_def)
            .with_seed(42);

        let outcomes1 = runner.run_shot(&circuit).unwrap();
        let outcomes2 = runner.run_shot(&circuit).unwrap();

        assert_eq!(outcomes1.len(), 1);
        assert_eq!(outcomes2.len(), 1);
    }

    #[test]
    fn test_x_basis_measurement() {
        let gates_def = GateDefinitions::new();

        // Prepare |+⟩ and measure in X basis - should always give 0
        let circuit = OpBuilder::new()
            .prep_x(QubitId(0))
            .meas_x(QubitId(0), ResultId(0))
            .build();

        let mut runner = ExtendedRunner::new(SparseStab::new(1), gates_def)
            .with_seed(42);

        let outcomes = runner.run(&circuit).unwrap();
        assert!(!outcomes.get_bit(QubitId(0)).unwrap(), "|+⟩ measured in X should give 0");
    }

    #[test]
    fn test_gate_override_custom_gate() {
        let mut gates_def = GateDefinitions::new();

        // Register a custom gate without decomposition
        let custom_id = gates_def.register(
            GateSpec::new("CustomGate")
                .with_quantum_arity(1)
                .with_category(GateCategory::SingleQubitUnitary)
        );

        // Provide an override that implements it as X
        let overrides: GateOverrides<SparseStab> = GateOverrides::new()
            .register(custom_id, |sim, qubits, _angles| {
                sim.x(qubits);
                true
            });

        let circuit = OpBuilder::new()
            .prep_z(QubitId(0))
            .gate1(custom_id, QubitId(0))  // Should apply X via override
            .meas_z(QubitId(0), ResultId(0))
            .build();

        let mut runner = ExtendedRunner::new(SparseStab::new(1), gates_def)
            .with_overrides(overrides)
            .with_seed(42);

        let outcomes = runner.run(&circuit).unwrap();
        // Prep |0⟩, apply X, measure -> should get 1
        assert!(outcomes.get_bit(QubitId(0)).unwrap(), "X gate should flip |0⟩ to |1⟩");
    }

    #[test]
    fn test_gate_override_core_gate() {
        let gates_def = GateDefinitions::new();

        // Override the H gate to do nothing (identity)
        let overrides: GateOverrides<SparseStab> = GateOverrides::new()
            .register(gates::H, |_sim, _qubits, _angles| {
                // Do nothing - override H with identity
                true
            });

        let circuit = OpBuilder::new()
            .prep_z(QubitId(0))
            .h(QubitId(0))  // Should do nothing due to override
            .meas_z(QubitId(0), ResultId(0))
            .build();

        let mut runner = ExtendedRunner::new(SparseStab::new(1), gates_def)
            .with_overrides(overrides)
            .with_seed(42);

        let outcomes = runner.run(&circuit).unwrap();
        // Prep |0⟩, H overridden to I, measure -> should get 0
        assert!(!outcomes.get_bit(QubitId(0)).unwrap(), "Overridden H should leave |0⟩ unchanged");
    }

    #[test]
    fn test_overrides_api() {
        let mut overrides: GateOverrides<SparseStab> = GateOverrides::new();
        assert!(overrides.is_empty());
        assert_eq!(overrides.len(), 0);

        overrides.insert(gates::H, |sim, qubits, _| { sim.h(qubits); true });
        assert!(!overrides.is_empty());
        assert_eq!(overrides.len(), 1);
        assert!(overrides.contains(gates::H));
        assert!(!overrides.contains(gates::X));

        overrides.remove(gates::H);
        assert!(overrides.is_empty());
    }

    #[test]
    fn test_rotations_unified_run() {
        use pecos_core::Angle64;
        use pecos_qsim::StateVec;

        let gates_def = GateDefinitions::new();

        // Circuit with RX(pi) which flips |0⟩ to |1⟩
        let circuit = OpBuilder::new()
            .prep_z(QubitId(0))
            .rx(QubitId(0), Angle64::HALF_TURN) // RX(pi)
            .meas_z(QubitId(0), ResultId(0))
            .build();

        // Use rotations() constructor - run() handles rotations automatically
        let mut runner = ExtendedRunner::rotations(StateVec::new(1), gates_def)
            .with_seed(42);

        assert!(runner.has_rotation_support(), "rotations() should enable rotation support");

        let outcomes = runner.run(&circuit).unwrap();
        // RX(pi) on |0⟩ gives |1⟩
        assert!(outcomes.get_bit(QubitId(0)).unwrap(), "RX(pi) should flip |0⟩ to |1⟩");
    }

    #[test]
    fn test_rotation_without_support_decomposes() {
        // Use SparseStab (Clifford only) with a circuit containing T gate
        // T gate should be decomposed (or error if no decomposition)
        let gates_def = GateDefinitions::new();

        let circuit = OpBuilder::new()
            .prep_z(QubitId(0))
            .gate1(gates::T, QubitId(0))  // T gate - not Clifford
            .meas_z(QubitId(0), ResultId(0))
            .build();

        // Use new() - no rotation support, T will try to decompose
        let mut runner = ExtendedRunner::new(SparseStab::new(1), gates_def);

        assert!(!runner.has_rotation_support(), "new() should not enable rotation support");

        // Should fail because T gate has no decomposition in default GateDefinitions
        let result = runner.run(&circuit);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ExecutionError::NoDecomposition { .. }));
    }
}
