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

use crate::command::{CommandQueue, GateCommand, GateType, SignalStore};
use crate::extensible::{GateDefinitions, GateId};
use crate::noise::context::NoiseContext;
use crate::noise::{ComposableNoiseModel, NoiseEvent, NoiseResponse};
use crate::outcome::{MeasurementOutcome, MeasurementOutcomes};
use pecos_core::rng::rng_manageable::{RngManageable, derive_seed};
use pecos_core::{Angle64, QubitId, Signal, TimeUnits};
use pecos_qsim::{ArbitraryRotationGateable, CliffordGateable};
use pecos_rng::PecosRng;
use rand_core::SeedableRng;
use smallvec::SmallVec;
use std::any::{Any, TypeId};

/// Type-erased observe-only signal handler.
type ErasedHandler = Box<dyn Fn(&dyn Any) + Send + Sync>;

/// Type-erased response-producing signal handler.
type ErasedResponseHandler = Box<dyn Fn(&dyn Any, &DispatchContext<'_>) -> NoiseResponse + Send + Sync>;

/// Registry of signal handlers, keyed by signal `TypeId`.
///
/// Uses a flat `Vec` rather than a `HashMap` -- with typically 1-3 signal
/// types registered, linear scan on contiguous memory is faster than hashing.
struct SignalHandlerRegistry {
    handlers: Vec<(TypeId, Vec<ErasedHandler>)>,
    response_handlers: Vec<(TypeId, Vec<ErasedResponseHandler>)>,
}

impl SignalHandlerRegistry {
    fn new() -> Self {
        Self {
            handlers: Vec::new(),
            response_handlers: Vec::new(),
        }
    }

    fn add(&mut self, type_id: TypeId, handler: ErasedHandler) {
        if let Some((_, handlers)) = self.handlers.iter_mut().find(|(id, _)| *id == type_id) {
            handlers.push(handler);
        } else {
            self.handlers.push((type_id, vec![handler]));
        }
    }

    fn add_response(&mut self, type_id: TypeId, handler: ErasedResponseHandler) {
        if let Some((_, handlers)) = self.response_handlers.iter_mut().find(|(id, _)| *id == type_id) {
            handlers.push(handler);
        } else {
            self.response_handlers.push((type_id, vec![handler]));
        }
    }

    fn call(&self, type_id: TypeId, data: &dyn Any) {
        if let Some((_, handlers)) = self.handlers.iter().find(|(id, _)| *id == type_id) {
            for handler in handlers {
                handler(data);
            }
        }
    }

    fn call_response(&self, type_id: TypeId, data: &dyn Any, ctx: &DispatchContext<'_>) -> NoiseResponse {
        let mut combined = NoiseResponse::None;
        if let Some((_, handlers)) = self.response_handlers.iter().find(|(id, _)| *id == type_id) {
            for handler in handlers {
                let response = handler(data, ctx);
                if !response.is_none() {
                    combined = combined.combine(response);
                }
            }
        }
        combined
    }

    fn has_response_handlers(&self) -> bool {
        !self.response_handlers.is_empty()
    }
}

/// Context provided to gate event dispatch handlers.
///
/// Contains the event data relevant to the current dispatch point,
/// plus optional read-only access to the noise model's context.
///
/// Fields not relevant to the current event are `None`/empty. Since
/// handlers are registered per-event-type, they know which fields
/// are populated.
pub struct DispatchContext<'a> {
    /// Gate type (for gate events).
    pub gate_type: GateType,
    /// Qubits involved.
    pub qubits: &'a [QubitId],
    /// Angle parameters (for parameterized gates).
    pub angles: &'a [Angle64],
    /// Gate ID (for extensible gate identification).
    pub gate_id: Option<GateId>,
    /// Measurement outcomes (populated for `AfterMeasurement` events).
    pub outcomes: Option<&'a [bool]>,
    /// Idle duration (populated for `IdleTime` events).
    pub duration: Option<TimeUnits>,
    /// Read-only access to the noise context (if a noise model is present).
    pub noise_context: Option<&'a NoiseContext>,
}

/// Type-erased gate event handler.
type ErasedGateHandler = Box<dyn Fn(&DispatchContext<'_>) -> NoiseResponse + Send + Sync>;

/// A handler with an associated priority for ordering.
struct PrioritizedHandler {
    handler: ErasedGateHandler,
    priority: i32,
}

/// Per-event-type handler storage.
///
/// Each Vec is kept sorted by priority (higher runs first).
/// Using flat Vecs avoids runtime event-type filtering.
struct GateEventHandlers {
    before_gate: Vec<PrioritizedHandler>,
    after_gate: Vec<PrioritizedHandler>,
    before_measurement: Vec<PrioritizedHandler>,
    after_measurement: Vec<PrioritizedHandler>,
    after_preparation: Vec<PrioritizedHandler>,
    idle: Vec<PrioritizedHandler>,
}

impl GateEventHandlers {
    fn new() -> Self {
        Self {
            before_gate: Vec::new(),
            after_gate: Vec::new(),
            before_measurement: Vec::new(),
            after_measurement: Vec::new(),
            after_preparation: Vec::new(),
            idle: Vec::new(),
        }
    }

    /// Insert a handler into a Vec and re-sort by priority (higher first).
    fn insert(vec: &mut Vec<PrioritizedHandler>, handler: ErasedGateHandler, priority: i32) {
        vec.push(PrioritizedHandler { handler, priority });
        // Stable sort so same-priority handlers keep registration order
        vec.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Dispatch all handlers in a Vec and combine their responses.
    fn dispatch(handlers: &[PrioritizedHandler], ctx: &DispatchContext<'_>) -> NoiseResponse {
        let mut combined = NoiseResponse::None;
        for h in handlers {
            let response = (h.handler)(ctx);
            if !response.is_none() {
                combined = combined.combine(response);
            }
        }
        combined
    }
}

/// Cursor for tracking position within a signal channel during dispatch.
///
/// Stores a channel index (resolved once at cursor creation) to avoid
/// repeated `TypeId` lookups in the inner loop.
struct SignalCursor {
    type_id: TypeId,
    /// Index into `SignalStore::channels` -- resolved once, not per-dispatch.
    channel_idx: usize,
    /// Current position within the channel's entries.
    entry_idx: usize,
    /// Cached length of the channel (avoids vtable call in hot loop).
    len: usize,
}

/// Simple shot runner that executes commands on a simulator.
///
/// The runner handles:
/// - Gate execution on the simulator
/// - Noise application via the composable noise model
/// - Measurement outcome collection
/// - Signal dispatch to registered handlers and the noise model
///
/// # Example
///
/// ```
/// use pecos_neo::prelude::*;
/// use pecos_qsim::SparseStab;
///
/// let commands = CommandBuilder::new()
///     .pz(0)
///     .h(0)
///     .mz(0)
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
    /// Optional gate definitions for noise model integration.
    gate_definitions: Option<GateDefinitions>,
    /// Signal handler registry for user-defined signal callbacks.
    signal_handlers: SignalHandlerRegistry,
    /// Gate event handlers for user-defined dispatch callbacks.
    gate_handlers: GateEventHandlers,
}

impl<S: CliffordGateable> ShotRunner<S> {
    /// Create a new shot runner with the given simulator.
    pub fn new(simulator: S) -> Self {
        Self {
            simulator,
            noise: None,
            rng: PecosRng::from_rng(&mut rand::rng()),
            outcomes: MeasurementOutcomes::new(),
            gate_definitions: None,
            signal_handlers: SignalHandlerRegistry::new(),
            gate_handlers: GateEventHandlers::new(),
        }
    }

    /// Register a handler that will be called when a signal of type `Sig` is dispatched.
    ///
    /// Handlers are called in registration order. Multiple handlers can be registered
    /// for the same signal type.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_neo::prelude::*;
    /// use pecos_qsim::SparseStab;
    /// use pecos_core::impl_signal;
    /// use std::sync::atomic::{AtomicU64, Ordering};
    /// use std::sync::Arc;
    ///
    /// #[derive(Copy, Clone, Debug)]
    /// struct RoundBoundary(pub i64);
    /// impl_signal!(RoundBoundary);
    ///
    /// let counter = Arc::new(AtomicU64::new(0));
    /// let counter_clone = counter.clone();
    ///
    /// let mut runner = ShotRunner::new(SparseStab::new(1));
    /// runner.on_signal(move |_signal: &RoundBoundary| {
    ///     counter_clone.fetch_add(1, Ordering::Relaxed);
    /// });
    /// ```
    pub fn on_signal<Sig: Signal>(&mut self, handler: impl Fn(&Sig) + Send + Sync + 'static) -> &mut Self {
        let erased: ErasedHandler = Box::new(move |data: &dyn Any| {
            if let Some(signal) = data.downcast_ref::<Sig>() {
                handler(signal);
            }
        });
        self.signal_handlers.add(TypeId::of::<Sig>(), erased);
        self
    }

    /// Register a response-producing signal handler.
    ///
    /// Unlike [`on_signal`](Self::on_signal), this handler receives a
    /// [`DispatchContext`] (providing read-only noise context) and returns
    /// a [`NoiseResponse`] that is applied to the simulation.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_neo::prelude::*;
    /// use pecos_neo::noise::NoiseResponse;
    /// use pecos_qsim::SparseStab;
    /// use pecos_core::impl_signal;
    ///
    /// #[derive(Copy, Clone, Debug)]
    /// struct InjectError;
    /// impl_signal!(InjectError);
    ///
    /// let mut runner = ShotRunner::new(SparseStab::new(2));
    /// runner.on_signal_with_response(|_signal: &InjectError, _ctx| {
    ///     NoiseResponse::inject_gate(
    ///         pecos_neo::command::GateCommand::x(0.into()),
    ///     )
    /// });
    /// ```
    pub fn on_signal_with_response<Sig: Signal>(
        &mut self,
        handler: impl Fn(&Sig, &DispatchContext<'_>) -> NoiseResponse + Send + Sync + 'static,
    ) -> &mut Self {
        let erased: ErasedResponseHandler = Box::new(move |data: &dyn Any, ctx: &DispatchContext<'_>| {
            if let Some(signal) = data.downcast_ref::<Sig>() {
                handler(signal, ctx)
            } else {
                NoiseResponse::None
            }
        });
        self.signal_handlers.add_response(TypeId::of::<Sig>(), erased);
        self
    }

    // ================================================================
    // Gate event handler registration
    // ================================================================

    /// Register a handler called before each gate is applied.
    ///
    /// Handlers run in priority order (higher priority first, default 0).
    /// Before-gate handlers run before the noise model's `BeforeGate` emission.
    /// Return `NoiseResponse::SkipGate` to prevent the gate from executing.
    pub fn on_before_gate(
        &mut self,
        handler: impl Fn(&DispatchContext<'_>) -> NoiseResponse + Send + Sync + 'static,
    ) -> &mut Self {
        GateEventHandlers::insert(&mut self.gate_handlers.before_gate, Box::new(handler), 0);
        self
    }

    /// Register a before-gate handler with explicit priority.
    pub fn on_before_gate_with_priority(
        &mut self,
        priority: i32,
        handler: impl Fn(&DispatchContext<'_>) -> NoiseResponse + Send + Sync + 'static,
    ) -> &mut Self {
        GateEventHandlers::insert(
            &mut self.gate_handlers.before_gate,
            Box::new(handler),
            priority,
        );
        self
    }

    /// Register a handler called after each gate is applied.
    ///
    /// After-gate handlers run after the noise model's `AfterGate` emission.
    pub fn on_after_gate(
        &mut self,
        handler: impl Fn(&DispatchContext<'_>) -> NoiseResponse + Send + Sync + 'static,
    ) -> &mut Self {
        GateEventHandlers::insert(&mut self.gate_handlers.after_gate, Box::new(handler), 0);
        self
    }

    /// Register an after-gate handler with explicit priority.
    pub fn on_after_gate_with_priority(
        &mut self,
        priority: i32,
        handler: impl Fn(&DispatchContext<'_>) -> NoiseResponse + Send + Sync + 'static,
    ) -> &mut Self {
        GateEventHandlers::insert(
            &mut self.gate_handlers.after_gate,
            Box::new(handler),
            priority,
        );
        self
    }

    /// Register a handler called before each measurement.
    ///
    /// Before-measurement handlers run before the noise model's `BeforeMeasurement` emission.
    pub fn on_before_measurement(
        &mut self,
        handler: impl Fn(&DispatchContext<'_>) -> NoiseResponse + Send + Sync + 'static,
    ) -> &mut Self {
        GateEventHandlers::insert(
            &mut self.gate_handlers.before_measurement,
            Box::new(handler),
            0,
        );
        self
    }

    /// Register a before-measurement handler with explicit priority.
    pub fn on_before_measurement_with_priority(
        &mut self,
        priority: i32,
        handler: impl Fn(&DispatchContext<'_>) -> NoiseResponse + Send + Sync + 'static,
    ) -> &mut Self {
        GateEventHandlers::insert(
            &mut self.gate_handlers.before_measurement,
            Box::new(handler),
            priority,
        );
        self
    }

    /// Register a handler called after each measurement.
    ///
    /// After-measurement handlers run after the noise model's `AfterMeasurement` emission.
    /// The `DispatchContext::outcomes` field is populated with measurement results.
    pub fn on_after_measurement(
        &mut self,
        handler: impl Fn(&DispatchContext<'_>) -> NoiseResponse + Send + Sync + 'static,
    ) -> &mut Self {
        GateEventHandlers::insert(
            &mut self.gate_handlers.after_measurement,
            Box::new(handler),
            0,
        );
        self
    }

    /// Register an after-measurement handler with explicit priority.
    pub fn on_after_measurement_with_priority(
        &mut self,
        priority: i32,
        handler: impl Fn(&DispatchContext<'_>) -> NoiseResponse + Send + Sync + 'static,
    ) -> &mut Self {
        GateEventHandlers::insert(
            &mut self.gate_handlers.after_measurement,
            Box::new(handler),
            priority,
        );
        self
    }

    /// Register a handler called after each preparation.
    ///
    /// After-preparation handlers run after the noise model's `AfterPreparation` emission.
    pub fn on_after_preparation(
        &mut self,
        handler: impl Fn(&DispatchContext<'_>) -> NoiseResponse + Send + Sync + 'static,
    ) -> &mut Self {
        GateEventHandlers::insert(
            &mut self.gate_handlers.after_preparation,
            Box::new(handler),
            0,
        );
        self
    }

    /// Register an after-preparation handler with explicit priority.
    pub fn on_after_preparation_with_priority(
        &mut self,
        priority: i32,
        handler: impl Fn(&DispatchContext<'_>) -> NoiseResponse + Send + Sync + 'static,
    ) -> &mut Self {
        GateEventHandlers::insert(
            &mut self.gate_handlers.after_preparation,
            Box::new(handler),
            priority,
        );
        self
    }

    /// Register a handler called during idle time.
    ///
    /// The `DispatchContext::duration` field is populated with the idle duration.
    pub fn on_idle(
        &mut self,
        handler: impl Fn(&DispatchContext<'_>) -> NoiseResponse + Send + Sync + 'static,
    ) -> &mut Self {
        GateEventHandlers::insert(&mut self.gate_handlers.idle, Box::new(handler), 0);
        self
    }

    /// Register an idle handler with explicit priority.
    pub fn on_idle_with_priority(
        &mut self,
        priority: i32,
        handler: impl Fn(&DispatchContext<'_>) -> NoiseResponse + Send + Sync + 'static,
    ) -> &mut Self {
        GateEventHandlers::insert(&mut self.gate_handlers.idle, Box::new(handler), priority);
        self
    }

    /// Set the noise model.
    ///
    /// If gate definitions have been set on this runner, they will be
    /// automatically propagated to the noise model's context.
    #[must_use]
    pub fn with_noise(mut self, mut noise: ComposableNoiseModel) -> Self {
        // Propagate gate definitions to noise model if we have them
        if let Some(ref defs) = self.gate_definitions {
            noise = noise.with_gate_definitions(defs.clone());
        }
        self.noise = Some(noise);
        self
    }

    /// Set gate definitions for this runner.
    ///
    /// Gate definitions provide metadata about gates (category, arity, etc.)
    /// and are automatically propagated to the noise model if one is set.
    ///
    /// # Example
    ///
    /// ```
    /// # use pecos_neo::prelude::*;
    /// # use pecos_neo::noise::prelude::*;
    /// # use pecos_qsim::SparseStab;
    /// use pecos_neo::extensible::GateDefinitions;
    ///
    /// # let sim = SparseStab::new(1);
    /// # let noise = ComposableNoiseModel::new();
    /// let gates = GateDefinitions::builder()
    ///     .with_category_noise(GateCategory::SingleQubitUnitary, 0.001)
    ///     .build_or_panic();
    ///
    /// let runner = ShotRunner::new(sim)
    ///     .with_gate_definitions(gates)
    ///     .with_noise(noise);
    /// ```
    #[must_use]
    pub fn with_gate_definitions(mut self, defs: GateDefinitions) -> Self {
        // If noise model already exists, propagate definitions to it
        if let Some(ref mut noise) = self.noise {
            // We need to update the noise model's context
            // Since with_gate_definitions consumes, we need to swap
            let old_noise = std::mem::take(noise);
            *noise = old_noise.with_gate_definitions(defs.clone());
        }
        self.gate_definitions = Some(defs);
        self
    }

    /// Get gate definitions if set.
    #[must_use]
    pub fn gate_definitions(&self) -> Option<&GateDefinitions> {
        self.gate_definitions.as_ref()
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
    ///
    /// When signals are present in the command queue, they are dispatched
    /// to registered handlers and the noise model at their recorded positions.
    /// The no-signal path has zero overhead.
    pub fn execute(&mut self, commands: &CommandQueue) -> &MeasurementOutcomes {
        self.outcomes.clear();

        if commands.has_signals() {
            self.execute_with_signals(commands, Self::execute_command);
        } else {
            for command in commands {
                self.execute_command(command);
            }
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
    /// ```
    /// # use pecos_neo::prelude::*;
    /// # use pecos_qsim::SparseStab;
    /// # let commands = CommandBuilder::new().pz(0).h(0).mz(0).build();
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
        let qubits = command.qubits.as_slice();

        // Dispatch before-gate event - may skip the gate (e.g., for leaked qubits)
        if self.dispatch_before_gate(command) {
            // Gate was skipped (e.g., due to leakage)
            // Still emit after-gate for channels that want to inject errors
            if !command.gate_type.is_measurement() && !command.gate_type.is_preparation() {
                self.dispatch_after_gate(command);
            }
            return;
        }

        // Execute the gate
        match command.gate_type {
            // Preparation
            GateType::PZ | GateType::QAlloc => {
                self.simulator.pz(qubits);
                self.dispatch_after_preparation(command);
            }

            // Measurement
            GateType::MZ | GateType::MeasureLeaked | GateType::MeasureFree => {
                self.dispatch_before_measurement(command);
                let results = self.simulator.mz(qubits);
                let outcomes: SmallVec<[bool; 4]> = results.iter().map(|r| r.outcome).collect();
                self.record_measurements(command.gate_type, qubits, &results);
                self.dispatch_after_measurement(command, outcomes.as_slice());
            }

            // Idle
            GateType::Idle => {
                if let Some(duration) = command.get_idle_duration() {
                    self.dispatch_idle(command, duration);
                }
            }

            // Try Clifford gates, skip non-Clifford
            _ => {
                self.execute_clifford_gate(command);
            }
        }

        // Dispatch after-gate event (except for measurement/prep/idle which have their own)
        if !command.gate_type.is_measurement()
            && !command.gate_type.is_preparation()
            && command.gate_type != GateType::Idle
        {
            self.dispatch_after_gate(command);
        }
    }

    // ================================================================
    // Dispatch coordination methods
    // ================================================================
    //
    // These methods coordinate user handlers and the noise model.
    // Execution order:
    //   Before-events: user handlers first, then noise model
    //   After-events:  noise model first, then user handlers

    /// Build a `DispatchContext` for signal dispatch (no gate info).
    fn signal_context(&self) -> DispatchContext<'_> {
        DispatchContext {
            gate_type: GateType::I,
            qubits: &[],
            angles: &[],
            gate_id: None,
            outcomes: None,
            duration: None,
            noise_context: self.noise.as_ref().map(ComposableNoiseModel::context),
        }
    }

    /// Build a `DispatchContext` for a gate command.
    fn gate_context<'a>(&'a self, command: &'a GateCommand) -> DispatchContext<'a> {
        DispatchContext {
            gate_type: command.gate_type,
            qubits: command.qubits.as_slice(),
            angles: command.angles.as_slice(),
            gate_id: Some(command.gate_type.to_gate_id()),
            outcomes: None,
            duration: None,
            noise_context: self.noise.as_ref().map(ComposableNoiseModel::context),
        }
    }

    /// Dispatch before-gate event. Returns `true` if the gate should be skipped.
    fn dispatch_before_gate(&mut self, command: &GateCommand) -> bool {
        // Fast path: no handlers registered, go directly to noise model
        if self.gate_handlers.before_gate.is_empty() {
            return self.emit_before_gate_noise(command);
        }

        // 1. User before-gate handlers
        let ctx = self.gate_context(command);
        let user_response =
            GateEventHandlers::dispatch(&self.gate_handlers.before_gate, &ctx);

        // 2. Noise model BeforeGate
        let noise_response = self.emit_before_gate_noise_raw(command);

        // 3. Combine
        let combined = user_response.combine(noise_response);
        let should_skip = combined.should_skip_gate();
        self.apply_noise_response(combined);
        should_skip
    }

    /// Dispatch after-gate event.
    fn dispatch_after_gate(&mut self, command: &GateCommand) {
        // Fast path
        if self.gate_handlers.after_gate.is_empty() {
            self.emit_after_gate_noise(command);
            return;
        }

        // 1. Noise model AfterGate
        let noise_response = self.emit_after_gate_noise_raw(command);

        // 2. User after-gate handlers
        let ctx = self.gate_context(command);
        let user_response =
            GateEventHandlers::dispatch(&self.gate_handlers.after_gate, &ctx);

        // 3. Combine and apply
        let combined = noise_response.combine(user_response);
        self.apply_noise_response(combined);
    }

    /// Dispatch before-measurement event.
    fn dispatch_before_measurement(&mut self, command: &GateCommand) {
        let qubits = command.qubits.as_slice();

        // Fast path
        if self.gate_handlers.before_measurement.is_empty() {
            self.emit_before_measurement_noise(qubits);
            return;
        }

        // 1. User before-measurement handlers
        let ctx = self.gate_context(command);
        let user_response =
            GateEventHandlers::dispatch(&self.gate_handlers.before_measurement, &ctx);

        // 2. Noise model BeforeMeasurement
        let noise_response = self.emit_before_measurement_noise_raw(qubits);

        // 3. Combine and apply
        let combined = user_response.combine(noise_response);
        self.apply_noise_response(combined);
    }

    /// Dispatch after-measurement event.
    fn dispatch_after_measurement(&mut self, command: &GateCommand, outcomes: &[bool]) {
        let qubits = command.qubits.as_slice();

        // Fast path
        if self.gate_handlers.after_measurement.is_empty() {
            self.emit_after_measurement_noise(qubits, outcomes);
            return;
        }

        // 1. Noise model AfterMeasurement
        let noise_response = self.emit_after_measurement_noise_raw(qubits, outcomes);

        // 2. User after-measurement handlers
        let ctx = DispatchContext {
            gate_type: command.gate_type,
            qubits,
            angles: command.angles.as_slice(),
            gate_id: Some(command.gate_type.to_gate_id()),
            outcomes: Some(outcomes),
            duration: None,
            noise_context: self.noise.as_ref().map(ComposableNoiseModel::context),
        };
        let user_response =
            GateEventHandlers::dispatch(&self.gate_handlers.after_measurement, &ctx);

        // 3. Combine and apply
        let combined = noise_response.combine(user_response);
        self.apply_noise_response(combined);
    }

    /// Dispatch after-preparation event.
    fn dispatch_after_preparation(&mut self, command: &GateCommand) {
        let qubits = command.qubits.as_slice();

        // Fast path
        if self.gate_handlers.after_preparation.is_empty() {
            self.emit_after_preparation_noise(qubits);
            return;
        }

        // 1. Noise model AfterPreparation
        let noise_response = self.emit_after_preparation_noise_raw(qubits);

        // 2. User after-preparation handlers
        let ctx = self.gate_context(command);
        let user_response =
            GateEventHandlers::dispatch(&self.gate_handlers.after_preparation, &ctx);

        // 3. Combine and apply
        let combined = noise_response.combine(user_response);
        self.apply_noise_response(combined);
    }

    /// Dispatch idle event.
    fn dispatch_idle(&mut self, command: &GateCommand, duration: TimeUnits) {
        let qubits = command.qubits.as_slice();

        // Fast path
        if self.gate_handlers.idle.is_empty() {
            self.emit_idle_noise(qubits, duration);
            return;
        }

        // 1. Noise model IdleTime
        let noise_response = self.emit_idle_noise_raw(qubits, duration);

        // 2. User idle handlers (idle is an "after" style event)
        let ctx = DispatchContext {
            gate_type: command.gate_type,
            qubits,
            angles: command.angles.as_slice(),
            gate_id: None,
            outcomes: None,
            duration: Some(duration),
            noise_context: self.noise.as_ref().map(ComposableNoiseModel::context),
        };
        let user_response =
            GateEventHandlers::dispatch(&self.gate_handlers.idle, &ctx);

        // 3. Combine and apply
        let combined = noise_response.combine(user_response);
        self.apply_noise_response(combined);
    }

    // ================================================================
    // Noise-only emission methods (used by dispatch and fast paths)
    // ================================================================

    /// Emit before-gate to noise model only. Returns `true` if gate should be skipped.
    fn emit_before_gate_noise(&mut self, command: &GateCommand) -> bool {
        let response = self.emit_before_gate_noise_raw(command);
        let should_skip = response.should_skip_gate();
        self.apply_noise_response(response);
        should_skip
    }

    /// Emit before-gate to noise model, returning the raw response.
    fn emit_before_gate_noise_raw(&mut self, command: &GateCommand) -> NoiseResponse {
        if let Some(ref mut noise) = self.noise {
            let event = NoiseEvent::before_gate(
                command.gate_type,
                command.qubits.as_slice(),
                command.angles.as_slice(),
            );
            return noise.emit(event, &mut self.rng);
        }
        NoiseResponse::None
    }

    /// Emit after-gate to noise model only.
    fn emit_after_gate_noise(&mut self, command: &GateCommand) {
        let response = self.emit_after_gate_noise_raw(command);
        self.apply_noise_response(response);
    }

    /// Emit after-gate to noise model, returning the raw response.
    fn emit_after_gate_noise_raw(&mut self, command: &GateCommand) -> NoiseResponse {
        if let Some(ref mut noise) = self.noise {
            let event = NoiseEvent::after_gate(
                command.gate_type,
                command.qubits.as_slice(),
                command.angles.as_slice(),
            );
            return noise.emit(event, &mut self.rng);
        }
        NoiseResponse::None
    }

    /// Emit before-measurement to noise model only.
    fn emit_before_measurement_noise(&mut self, qubits: &[QubitId]) {
        let response = self.emit_before_measurement_noise_raw(qubits);
        self.apply_noise_response(response);
    }

    /// Emit before-measurement to noise model, returning the raw response.
    fn emit_before_measurement_noise_raw(&mut self, qubits: &[QubitId]) -> NoiseResponse {
        if let Some(ref mut noise) = self.noise {
            let event = NoiseEvent::BeforeMeasurement { qubits };
            return noise.emit(event, &mut self.rng);
        }
        NoiseResponse::None
    }

    /// Emit after-measurement to noise model only.
    fn emit_after_measurement_noise(&mut self, qubits: &[QubitId], outcomes: &[bool]) {
        let response = self.emit_after_measurement_noise_raw(qubits, outcomes);
        self.apply_noise_response(response);
    }

    /// Emit after-measurement to noise model, returning the raw response.
    fn emit_after_measurement_noise_raw(
        &mut self,
        qubits: &[QubitId],
        outcomes: &[bool],
    ) -> NoiseResponse {
        if let Some(ref mut noise) = self.noise {
            let event = NoiseEvent::AfterMeasurement { qubits, outcomes };
            return noise.emit(event, &mut self.rng);
        }
        NoiseResponse::None
    }

    /// Emit after-preparation to noise model only.
    fn emit_after_preparation_noise(&mut self, qubits: &[QubitId]) {
        let response = self.emit_after_preparation_noise_raw(qubits);
        self.apply_noise_response(response);
    }

    /// Emit after-preparation to noise model, returning the raw response.
    fn emit_after_preparation_noise_raw(&mut self, qubits: &[QubitId]) -> NoiseResponse {
        if let Some(ref mut noise) = self.noise {
            let event = NoiseEvent::AfterPreparation { qubits };
            return noise.emit(event, &mut self.rng);
        }
        NoiseResponse::None
    }

    /// Emit idle to noise model only.
    fn emit_idle_noise(&mut self, qubits: &[QubitId], duration: TimeUnits) {
        let response = self.emit_idle_noise_raw(qubits, duration);
        self.apply_noise_response(response);
    }

    /// Emit idle to noise model, returning the raw response.
    fn emit_idle_noise_raw(&mut self, qubits: &[QubitId], duration: TimeUnits) -> NoiseResponse {
        if let Some(ref mut noise) = self.noise {
            let event = NoiseEvent::IdleTime { qubits, duration };
            return noise.emit(event, &mut self.rng);
        }
        NoiseResponse::None
    }

    /// Execute commands with interleaved signal dispatch.
    ///
    /// Uses cursor-based scanning: one cursor per signal type tracks its
    /// position in the sorted entries. At each gate index, all signals
    /// at that position are dispatched before the gate executes.
    fn execute_with_signals(
        &mut self,
        commands: &CommandQueue,
        execute_fn: fn(&mut Self, &GateCommand),
    ) {
        let store = commands.signals();
        let mut cursors: SmallVec<[SignalCursor; 4]> = SmallVec::with_capacity(store.channel_count());
        for ch_idx in 0..store.channel_count() {
            let (type_id, channel) = store.channel_at(ch_idx).unwrap();
            cursors.push(SignalCursor {
                type_id,
                channel_idx: ch_idx,
                entry_idx: 0,
                len: channel.len(),
            });
        }

        for (gate_idx, command) in commands.iter().enumerate() {
            self.dispatch_signals_at(gate_idx as u32, store, &mut cursors);
            execute_fn(self, command);
        }
        // Dispatch trailing signals (positioned after the last gate)
        self.dispatch_signals_at(commands.len() as u32, store, &mut cursors);
    }

    /// Dispatch all signals at a given command position.
    ///
    /// Uses the positions slice directly (contiguous `u32`s, no vtable call
    /// per element) for the hot-path position check. Signal data is only
    /// accessed on match via a single vtable call.
    fn dispatch_signals_at(
        &mut self,
        pos: u32,
        store: &SignalStore,
        cursors: &mut [SignalCursor],
    ) {
        let has_response_handlers = self.signal_handlers.has_response_handlers();

        for cursor in cursors.iter_mut() {
            // Resolve channel once per cursor (index was set at creation time)
            let Some((_, channel)) = store.channel_at(cursor.channel_idx) else {
                continue;
            };
            // Get the positions slice -- contiguous u32s, no vtable per element
            let positions = channel.positions();

            while cursor.entry_idx < cursor.len {
                if positions[cursor.entry_idx] != pos {
                    break;
                }
                let data = channel
                    .entry_data(cursor.entry_idx)
                    .expect("entry_data should exist at valid index");

                // 1. Call registered observe-only signal handlers
                self.signal_handlers.call(cursor.type_id, data);

                // 2. Call response-producing signal handlers
                if has_response_handlers {
                    let ctx = self.signal_context();
                    let response = self.signal_handlers.call_response(cursor.type_id, data, &ctx);
                    if !response.is_none() {
                        self.apply_noise_response(response);
                    }
                }

                // 3. Emit to noise model
                self.emit_signal_to_noise(cursor.type_id, data);

                cursor.entry_idx += 1;
            }
        }
    }

    /// Emit a signal event to the noise model.
    fn emit_signal_to_noise(&mut self, type_id: TypeId, data: &dyn Any) {
        if let Some(ref mut noise) = self.noise {
            let event = NoiseEvent::Signal { type_id, data };
            let response = noise.emit(event, &mut self.rng);
            self.apply_noise_response(response);
        }
    }

    /// Record measurement results.
    ///
    /// Handles leakage-aware measurement:
    /// - For `MeasureLeaked`: returns outcome with `is_leaked = true` for leaked qubits
    /// - For regular `MZ`: forces outcome to 1 for leaked qubits (as in hardware)
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
                    // Regular MZ on leaked qubit: force outcome to 1
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
    ///     .pz(0)
    ///     .h(0)
    ///     .mz(0)
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
    /// ```
    /// use pecos_neo::prelude::*;
    /// use pecos_qsim::StateVec;
    ///
    /// let commands = CommandBuilder::new()
    ///     .pz(0)
    ///     .rx(0, Angle64::QUARTER_TURN)
    ///     .mz(0)
    ///     .build();
    ///
    /// let mut runner = ShotRunner::new(StateVec::new(1));
    /// let outcomes = runner.execute_all(&commands);
    /// ```
    pub fn execute_all(&mut self, commands: &CommandQueue) -> &MeasurementOutcomes {
        self.outcomes.clear();

        if commands.has_signals() {
            self.execute_with_signals(commands, Self::execute_command_universal);
        } else {
            for command in commands {
                self.execute_command_universal(command);
            }
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
        let qubits = command.qubits.as_slice();

        // Dispatch before-gate event - may skip the gate (e.g., for leaked qubits)
        if self.dispatch_before_gate(command) {
            // Gate was skipped (e.g., due to leakage)
            // Still emit after-gate for channels that want to inject errors
            if !command.gate_type.is_measurement() && !command.gate_type.is_preparation() {
                self.dispatch_after_gate(command);
            }
            return;
        }

        // Execute the gate
        match command.gate_type {
            // Preparation
            GateType::PZ | GateType::QAlloc => {
                self.simulator.pz(qubits);
                self.dispatch_after_preparation(command);
            }

            // Measurement
            GateType::MZ | GateType::MeasureLeaked | GateType::MeasureFree => {
                self.dispatch_before_measurement(command);
                let results = self.simulator.mz(qubits);
                let outcomes: SmallVec<[bool; 4]> = results.iter().map(|r| r.outcome).collect();
                self.record_measurements(command.gate_type, qubits, &results);
                self.dispatch_after_measurement(command, outcomes.as_slice());
            }

            // Idle
            GateType::Idle => {
                if let Some(duration) = command.get_idle_duration() {
                    self.dispatch_idle(command, duration);
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

        // Dispatch after-gate event (except for measurement/prep/idle which have their own)
        if !command.gate_type.is_measurement()
            && !command.gate_type.is_preparation()
            && command.gate_type != GateType::Idle
        {
            self.dispatch_after_gate(command);
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
    /// Execute a [`pecos_quantum::TickCircuit`] and return measurement outcomes.
    ///
    /// This is a convenience method that converts the `TickCircuit` to a
    /// [`CommandQueue`] and executes it.
    ///
    /// # Example
    ///
    /// ```
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
    pub fn execute_tick(&mut self, circuit: &pecos_quantum::TickCircuit) -> &MeasurementOutcomes {
        let commands = crate::command::CommandQueue::from(circuit);
        self.execute(&commands)
    }

    /// Execute a [`pecos_quantum::TickCircuit`] for a single shot, then reset for next shot.
    pub fn run_shot_tick(&mut self, circuit: &pecos_quantum::TickCircuit) -> MeasurementOutcomes {
        let commands = crate::command::CommandQueue::from(circuit);
        self.run_shot(&commands)
    }

    /// Execute a [`pecos_quantum::DagCircuit`] and return measurement outcomes.
    ///
    /// Gates are executed in topological order, respecting dependencies.
    ///
    /// # Example
    ///
    /// ```
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
    pub fn execute_dag(&mut self, circuit: &pecos_quantum::DagCircuit) -> &MeasurementOutcomes {
        let commands = crate::command::CommandQueue::from(circuit);
        self.execute(&commands)
    }

    /// Execute a [`pecos_quantum::DagCircuit`] for a single shot, then reset for next shot.
    pub fn run_shot_dag(&mut self, circuit: &pecos_quantum::DagCircuit) -> MeasurementOutcomes {
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
        let commands = CommandBuilder::new().pz(0).h(0).mz(0).build();

        let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(42);
        let outcomes = runner.execute(&commands);

        assert_eq!(outcomes.len(), 1);
    }

    #[test]
    fn test_bell_state() {
        let commands = CommandBuilder::new()
            .pz(0)
            .pz(1)
            .h(0)
            .cx(0, 1)
            .mz(0)
            .mz(1)
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
            .pz(0)
            .h(0) // Single-qubit gate will trigger noise
            .mz(0)
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
        let commands = CommandBuilder::new().pz(0).mz(0).build();

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
        let mut commands = CommandBuilder::new().pz(0).build();
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
        assert!(
            outcome.is_leaked,
            "MeasureLeaked on leaked qubit should set is_leaked"
        );
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
        let commands = CommandBuilder::new().pz(0).mz(0).build();

        // Create noise model and manually mark qubit as leaked
        let mut noise = ComposableNoiseModel::new().add_channel(LeakageChannel::new());
        noise.context_mut().mark_leaked(QubitId(0));

        let mut runner = ShotRunner::new(SparseStab::new(1))
            .with_noise(noise)
            .with_seed(42);
        let outcomes = runner.execute(&commands);

        // Should have outcome = 1 (forced for leaked qubits)
        let outcome = outcomes.get(QubitId(0)).unwrap();
        assert!(
            outcome.outcome,
            "Regular MZ on leaked qubit should force outcome to 1"
        );
        assert!(
            outcome.is_leaked,
            "Regular MZ on leaked qubit should mark is_leaked"
        );
    }

    #[test]
    fn test_measure_leaked_on_non_leaked_qubit() {
        use crate::command::GateCommand;
        use crate::noise::leakage::LeakageChannel;

        // Build a command queue with MeasureLeaked on a non-leaked qubit
        let mut commands = CommandBuilder::new().pz(0).build();
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
        assert!(
            !outcome.is_leaked,
            "MeasureLeaked on non-leaked qubit should not set is_leaked"
        );
        assert!(
            !outcome.outcome,
            "Prep followed by MeasureLeaked should give 0"
        );
    }

    #[test]
    fn test_execute_all_with_rotation_gates() {
        use crate::command::GateCommand;
        use pecos_core::Angle64;

        // Build a command queue with rotation gates
        let mut commands = CommandBuilder::new().pz(0).build();

        // RX(pi) flips |0> to |1>
        commands.push(GateCommand::with_angles(
            GateType::RX,
            smallvec::smallvec![QubitId(0)],
            smallvec::smallvec![Angle64::HALF_TURN],
        ));

        commands.push(GateCommand::new(
            GateType::MZ,
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
        let mut commands = CommandBuilder::new().pz(0).pz(1).build();

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
            GateType::MZ,
            smallvec::smallvec![QubitId(0)],
        ));
        commands.push(GateCommand::new(
            GateType::MZ,
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

        let mut commands = CommandBuilder::new().pz(0).build();
        commands.push(GateCommand::with_angles(
            GateType::RX,
            smallvec::smallvec![QubitId(0)],
            smallvec::smallvec![Angle64::HALF_TURN], // RX(pi) = X gate, flips |0> to |1>
        ));
        commands.push(GateCommand::new(
            GateType::MZ,
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
            .pz(0)
            .idle(0, 100u64) // 100 time units
            .mz(0)
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
            .pz(0)
            .pz(1)
            .rzz(0, 1, Angle64::QUARTER_TURN)
            .mz(0)
            .mz(1)
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
            .pz(0)
            .pz(1)
            .x(0) // Control qubit = |1>
            .h(1) // Target qubit = |+>
            .gate(GateCommand::with_angles(
                GateType::CRZ,
                smallvec::smallvec![QubitId(0), QubitId(1)],
                smallvec::smallvec![Angle64::HALF_TURN], // CRZ(pi)
            ))
            .h(1) // Convert phase to amplitude
            .mz(1)
            .build();

        let mut runner = ShotRunner::new(StateVec::new(2)).with_seed(42);
        let outcomes = runner.execute_all(&commands);

        // With control=1 and CRZ(pi), target |+> becomes |->
        // After H, |-> becomes |1>
        let outcome = outcomes.get(QubitId(1)).unwrap();
        assert!(
            outcome.outcome,
            "CRZ(pi) with control=1 should flip target phase"
        );
    }

    #[test]
    fn test_ccx_decomposition() {
        // CCX (Toffoli) should flip target when both controls are 1
        let commands = CommandBuilder::new()
            .pz(0)
            .pz(1)
            .pz(2)
            .x(0) // Control 1 = |1>
            .x(1) // Control 2 = |1>
            // Target = |0>, should flip to |1>
            .ccx(0, 1, 2)
            .mz(2)
            .build();

        let mut runner = ShotRunner::new(StateVec::new(3)).with_seed(42);
        let outcomes = runner.execute_all(&commands);

        // With both controls = 1, target should flip from 0 to 1
        let outcome = outcomes.get(QubitId(2)).unwrap();
        assert!(
            outcome.outcome,
            "CCX with both controls=1 should flip target"
        );
    }

    #[test]
    fn test_ccx_no_flip_when_control_zero() {
        // CCX should NOT flip target when one control is 0
        let commands = CommandBuilder::new()
            .pz(0)
            .pz(1)
            .pz(2)
            .x(0) // Control 1 = |1>
            // Control 2 = |0> (not set)
            // Target = |0>, should stay |0>
            .ccx(0, 1, 2)
            .mz(2)
            .build();

        let mut runner = ShotRunner::new(StateVec::new(3)).with_seed(42);
        let outcomes = runner.execute_all(&commands);

        // With one control = 0, target should stay 0
        let outcome = outcomes.get(QubitId(2)).unwrap();
        assert!(
            !outcome.outcome,
            "CCX with one control=0 should not flip target"
        );
    }

    #[test]
    fn test_with_gate_definitions() {
        use crate::extensible::{GateCategory, GateDefinitions};
        use crate::noise::CategoryBasedChannel;

        // Create gate definitions with category-based noise
        let gates = GateDefinitions::builder()
            .with_category_noise(GateCategory::SingleQubitUnitary, 0.0) // No noise for testing
            .with_category_noise(GateCategory::TwoQubitUnitary, 0.0)
            .build_or_panic();

        // Create a category-based noise channel
        let noise = ComposableNoiseModel::new().add_channel(
            CategoryBasedChannel::new().with_category(GateCategory::SingleQubitUnitary, 0.0),
        );

        // Create runner with gate definitions
        let commands = CommandBuilder::new().pz(0).h(0).mz(0).build();

        let mut runner = ShotRunner::new(SparseStab::new(1))
            .with_gate_definitions(gates)
            .with_noise(noise)
            .with_seed(42);

        let outcomes = runner.execute(&commands);
        assert_eq!(outcomes.len(), 1);
    }

    #[test]
    fn test_gate_definitions_propagated_to_noise() {
        use crate::extensible::{GateCategory, GateDefinitions};
        use crate::noise::CategoryBasedChannel;

        // Create definitions with a custom gate
        let mut gates = GateDefinitions::new();
        let custom_id = gates.register(
            crate::extensible::GateSpec::new("CustomGate")
                .with_quantum_arity(1)
                .with_category(GateCategory::SingleQubitUnitary),
        );

        // Noise channel that uses category filtering
        let noise = ComposableNoiseModel::new().add_channel(
            CategoryBasedChannel::new().with_category(GateCategory::SingleQubitUnitary, 0.0),
        );

        // Runner should propagate definitions to noise model
        let runner = ShotRunner::new(SparseStab::new(1))
            .with_gate_definitions(gates)
            .with_noise(noise);

        // Verify definitions are accessible
        assert!(runner.gate_definitions().is_some());
        let defs = runner.gate_definitions().unwrap();
        assert_eq!(defs.name(custom_id), Some("CustomGate"));
    }

    // ================================================================
    // Signal dispatch tests
    // ================================================================

    mod signal_tests {
        use super::*;
        use pecos_core::impl_signal;
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::sync::Arc;

        #[derive(Copy, Clone, Debug)]
        struct RoundBoundary(pub i64);
        impl_signal!(RoundBoundary);

        #[derive(Copy, Clone, Debug)]
        struct Temperature(pub f64);
        impl_signal!(Temperature);

        #[test]
        fn signal_handler_called() {
            let counter = Arc::new(AtomicU64::new(0));
            let counter_clone = counter.clone();

            let mut commands = CommandBuilder::new().pz(0).h(0).mz(0).build();
            commands.signal(RoundBoundary(1));

            let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(42);
            runner.on_signal(move |_: &RoundBoundary| {
                counter_clone.fetch_add(1, Ordering::Relaxed);
            });

            runner.execute(&commands);
            assert_eq!(counter.load(Ordering::Relaxed), 1);
        }

        #[test]
        fn signal_at_various_positions() {
            let positions = Arc::new(std::sync::Mutex::new(Vec::new()));
            let positions_clone = positions.clone();

            let mut commands = CommandBuilder::new().build();
            // Signal at position 0 (before any gates)
            commands.signal(RoundBoundary(0));
            commands.push(GateCommand::pz(QubitId(0)));
            // Signal at position 1 (after prep)
            commands.signal(RoundBoundary(1));
            commands.push(GateCommand::h(QubitId(0)));
            commands.push(GateCommand::mz(QubitId(0)));
            // Signal at position 3 (after last gate)
            commands.signal(RoundBoundary(2));

            let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(42);
            runner.on_signal(move |sig: &RoundBoundary| {
                positions_clone.lock().unwrap().push(sig.0);
            });

            runner.execute(&commands);
            let received = positions.lock().unwrap().clone();
            assert_eq!(received, vec![0, 1, 2]);
        }

        #[test]
        fn multiple_signal_types_interleaved() {
            let round_count = Arc::new(AtomicU64::new(0));
            let temp_count = Arc::new(AtomicU64::new(0));
            let rc = round_count.clone();
            let tc = temp_count.clone();

            let mut commands = CommandBuilder::new().pz(0).build();
            commands.signal(RoundBoundary(1));
            commands.signal(Temperature(300.0));
            commands.push(GateCommand::h(QubitId(0)));
            commands.signal(RoundBoundary(2));
            commands.push(GateCommand::mz(QubitId(0)));

            let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(42);
            runner.on_signal(move |_: &RoundBoundary| {
                rc.fetch_add(1, Ordering::Relaxed);
            });
            runner.on_signal(move |_: &Temperature| {
                tc.fetch_add(1, Ordering::Relaxed);
            });

            runner.execute(&commands);
            assert_eq!(round_count.load(Ordering::Relaxed), 2);
            assert_eq!(temp_count.load(Ordering::Relaxed), 1);
        }

        #[test]
        fn multiple_handlers_same_type() {
            let counter1 = Arc::new(AtomicU64::new(0));
            let counter2 = Arc::new(AtomicU64::new(0));
            let c1 = counter1.clone();
            let c2 = counter2.clone();

            let mut commands = CommandBuilder::new().pz(0).mz(0).build();
            commands.signal(RoundBoundary(1));

            let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(42);
            runner.on_signal(move |_: &RoundBoundary| {
                c1.fetch_add(1, Ordering::Relaxed);
            });
            runner.on_signal(move |_: &RoundBoundary| {
                c2.fetch_add(10, Ordering::Relaxed);
            });

            runner.execute(&commands);
            assert_eq!(counter1.load(Ordering::Relaxed), 1);
            assert_eq!(counter2.load(Ordering::Relaxed), 10);
        }

        #[test]
        fn no_signal_fast_path() {
            // No signals: should not trigger any dispatch overhead
            let commands = CommandBuilder::new().pz(0).h(0).mz(0).build();
            assert!(!commands.has_signals());

            let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(42);
            let outcomes = runner.execute(&commands);
            assert_eq!(outcomes.len(), 1);
        }

        #[test]
        fn trailing_signals_dispatched() {
            let counter = Arc::new(AtomicU64::new(0));
            let cc = counter.clone();

            let mut commands = CommandBuilder::new().pz(0).mz(0).build();
            // Signal after the last gate
            commands.signal(RoundBoundary(99));

            let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(42);
            runner.on_signal(move |_: &RoundBoundary| {
                cc.fetch_add(1, Ordering::Relaxed);
            });

            runner.execute(&commands);
            assert_eq!(counter.load(Ordering::Relaxed), 1);
        }

        #[test]
        fn leading_signals_dispatched() {
            let counter = Arc::new(AtomicU64::new(0));
            let cc = counter.clone();

            let mut commands = CommandQueue::new();
            // Signal before any gates (position 0)
            commands.signal(RoundBoundary(0));
            commands.push(GateCommand::pz(QubitId(0)));
            commands.push(GateCommand::mz(QubitId(0)));

            let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(42);
            runner.on_signal(move |_: &RoundBoundary| {
                cc.fetch_add(1, Ordering::Relaxed);
            });

            runner.execute(&commands);
            assert_eq!(counter.load(Ordering::Relaxed), 1);
        }

        #[test]
        fn signal_handler_receives_correct_value() {
            let received = Arc::new(std::sync::Mutex::new(Vec::new()));
            let received_clone = received.clone();

            let mut commands = CommandBuilder::new().pz(0).mz(0).build();
            commands.signal(Temperature(42.5));
            commands.signal(Temperature(99.9));

            let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(42);
            runner.on_signal(move |sig: &Temperature| {
                received_clone.lock().unwrap().push(sig.0);
            });

            runner.execute(&commands);
            let values = received.lock().unwrap().clone();
            assert_eq!(values, vec![42.5, 99.9]);
        }

        #[test]
        fn signal_dispatch_with_execute_all() {
            let counter = Arc::new(AtomicU64::new(0));
            let cc = counter.clone();

            let mut commands = CommandBuilder::new().pz(0).mz(0).build();
            commands.signal(RoundBoundary(1));

            let mut runner = ShotRunner::new(StateVec::new(1)).with_seed(42);
            runner.on_signal(move |_: &RoundBoundary| {
                cc.fetch_add(1, Ordering::Relaxed);
            });

            runner.execute_all(&commands);
            assert_eq!(counter.load(Ordering::Relaxed), 1);
        }

        #[test]
        fn signal_emitted_to_noise_model() {
            use crate::noise::NoiseChannel;
            use pecos_rng::PecosRng;
            use crate::noise::context::NoiseContext;
            use std::sync::atomic::AtomicBool;

            // Create a custom noise channel that records if it received a signal
            #[derive(Clone)]
            struct SignalRecorder {
                received: Arc<AtomicBool>,
            }

            impl NoiseChannel for SignalRecorder {
                fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
                    matches!(event, NoiseEvent::Signal { .. })
                }

                fn apply(
                    &self,
                    event: &NoiseEvent<'_>,
                    _context: &mut NoiseContext,
                    _rng: &mut PecosRng,
                ) -> NoiseResponse {
                    if let NoiseEvent::Signal { data, .. } = event {
                        if data.downcast_ref::<RoundBoundary>().is_some() {
                            self.received.store(true, Ordering::Relaxed);
                        }
                    }
                    NoiseResponse::None
                }

                fn name(&self) -> &'static str {
                    "SignalRecorder"
                }

                fn clone_box(&self) -> Box<dyn NoiseChannel> {
                    Box::new(self.clone())
                }
            }

            let received = Arc::new(AtomicBool::new(false));
            let noise = ComposableNoiseModel::new().add_channel(SignalRecorder {
                received: received.clone(),
            });

            let mut commands = CommandBuilder::new().pz(0).mz(0).build();
            commands.signal(RoundBoundary(1));

            let mut runner = ShotRunner::new(SparseStab::new(1))
                .with_noise(noise)
                .with_seed(42);
            runner.execute(&commands);

            assert!(
                received.load(Ordering::Relaxed),
                "Noise channel should have received the signal event"
            );
        }

        #[test]
        fn signals_dont_affect_gate_execution() {
            // Verify that signals interleaved with gates don't change outcomes.
            // Uses a deterministic circuit (no superposition) to avoid
            // dependence on simulator RNG.
            let commands_no_signals = CommandBuilder::new()
                .pz(0)
                .pz(1)
                .x(0) // deterministic: |0> -> |1>
                .cx(0, 1) // deterministic: |10> -> |11>
                .mz(0)
                .mz(1)
                .build();

            let mut commands_with_signals = CommandBuilder::new()
                .pz(0)
                .pz(1)
                .x(0)
                .cx(0, 1)
                .mz(0)
                .mz(1)
                .build();
            commands_with_signals.signal(RoundBoundary(1));
            commands_with_signals.signal(Temperature(300.0));

            let mut runner1 = ShotRunner::new(SparseStab::new(2)).with_seed(42);
            let mut runner2 = ShotRunner::new(SparseStab::new(2)).with_seed(42);

            let o1 = runner1.execute(&commands_no_signals);
            let o2 = runner2.execute(&commands_with_signals);

            assert_eq!(
                o1.get_bit(QubitId(0)),
                o2.get_bit(QubitId(0)),
            );
            assert_eq!(
                o1.get_bit(QubitId(1)),
                o2.get_bit(QubitId(1)),
            );
            // Both should be 1
            assert_eq!(o2.get_bit(QubitId(0)), Some(true));
            assert_eq!(o2.get_bit(QubitId(1)), Some(true));
        }

        #[test]
        fn signal_with_response_injects_gate() {
            use pecos_qsim::SparseStab;

            let mut commands = CommandBuilder::new().pz(0).build();
            commands.signal(RoundBoundary(1));
            commands.push(GateCommand::mz(QubitId(0)));

            let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(42);
            runner.on_signal_with_response(|_: &RoundBoundary, _ctx| {
                NoiseResponse::inject_gate(GateCommand::x(QubitId(0)))
            });

            let outcomes = runner.execute(&commands);
            // Signal injected X before MZ, so qubit flips from |0> to |1>
            assert_eq!(outcomes.get_bit(QubitId(0)), Some(true));
        }

        #[test]
        fn signal_with_response_receives_noise_context() {
            use pecos_qsim::SparseStab;

            let saw_context = Arc::new(AtomicU64::new(0));
            let sc = saw_context.clone();

            let mut commands = CommandBuilder::new().pz(0).mz(0).build();
            commands.signal(RoundBoundary(1));

            let noise = ComposableNoiseModel::new();
            let mut runner = ShotRunner::new(SparseStab::new(1))
                .with_seed(42)
                .with_noise(noise);
            runner.on_signal_with_response(move |_: &RoundBoundary, ctx| {
                if ctx.noise_context.is_some() {
                    sc.fetch_add(1, Ordering::Relaxed);
                }
                NoiseResponse::None
            });

            runner.execute(&commands);
            assert_eq!(saw_context.load(Ordering::Relaxed), 1);
        }

        #[test]
        fn signal_with_response_no_context_without_noise() {
            use pecos_qsim::SparseStab;

            let saw_none = Arc::new(AtomicU64::new(0));
            let sn = saw_none.clone();

            let mut commands = CommandBuilder::new().pz(0).mz(0).build();
            commands.signal(RoundBoundary(1));

            let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(42);
            runner.on_signal_with_response(move |_: &RoundBoundary, ctx| {
                if ctx.noise_context.is_none() {
                    sn.fetch_add(1, Ordering::Relaxed);
                }
                NoiseResponse::None
            });

            runner.execute(&commands);
            assert_eq!(saw_none.load(Ordering::Relaxed), 1);
        }

        #[test]
        fn signal_with_response_coexists_with_observe_only() {
            use pecos_qsim::SparseStab;

            let observe_count = Arc::new(AtomicU64::new(0));
            let oc = observe_count.clone();

            let mut commands = CommandBuilder::new().pz(0).build();
            commands.signal(RoundBoundary(1));
            commands.push(GateCommand::mz(QubitId(0)));

            let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(42);

            // Observe-only handler
            runner.on_signal(move |_: &RoundBoundary| {
                oc.fetch_add(1, Ordering::Relaxed);
            });

            // Response handler injects X
            runner.on_signal_with_response(|_: &RoundBoundary, _ctx| {
                NoiseResponse::inject_gate(GateCommand::x(QubitId(0)))
            });

            let outcomes = runner.execute(&commands);
            // Observe-only handler was called
            assert_eq!(observe_count.load(Ordering::Relaxed), 1);
            // Response handler injected X, so qubit is |1>
            assert_eq!(outcomes.get_bit(QubitId(0)), Some(true));
        }

        #[test]
        fn signal_with_response_multiple_handlers_combine() {
            use pecos_qsim::SparseStab;

            let mut commands = CommandBuilder::new().pz(0).pz(1).build();
            commands.signal(RoundBoundary(1));
            commands.push(GateCommand::mz(QubitId(0)));
            commands.push(GateCommand::mz(QubitId(1)));

            let mut runner = ShotRunner::new(SparseStab::new(2)).with_seed(42);

            // First handler injects X on qubit 0
            runner.on_signal_with_response(|_: &RoundBoundary, _ctx| {
                NoiseResponse::inject_gate(GateCommand::x(QubitId(0)))
            });

            // Second handler injects X on qubit 1
            runner.on_signal_with_response(|_: &RoundBoundary, _ctx| {
                NoiseResponse::inject_gate(GateCommand::x(QubitId(1)))
            });

            let outcomes = runner.execute(&commands);
            // Both qubits flipped
            assert_eq!(outcomes.get_bit(QubitId(0)), Some(true));
            assert_eq!(outcomes.get_bit(QubitId(1)), Some(true));
        }

        #[test]
        fn signal_with_response_via_execute_all() {
            use pecos_qsim::StateVec;

            let mut commands = CommandBuilder::new().pz(0).build();
            commands.signal(RoundBoundary(1));
            commands.push(GateCommand::mz(QubitId(0)));

            let mut runner = ShotRunner::new(StateVec::new(1)).with_seed(42);
            runner.on_signal_with_response(|_: &RoundBoundary, _ctx| {
                NoiseResponse::inject_gate(GateCommand::x(QubitId(0)))
            });

            runner.execute_all(&commands);
            assert_eq!(runner.outcomes.get_bit(QubitId(0)), Some(true));
        }

        #[test]
        fn signal_dispatch_ordering_observe_before_response() {
            use pecos_qsim::SparseStab;

            let order = Arc::new(std::sync::Mutex::new(Vec::new()));
            let o1 = order.clone();
            let o2 = order.clone();

            let mut commands = CommandBuilder::new().pz(0).mz(0).build();
            commands.signal(RoundBoundary(1));

            let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(42);

            // Observe-only registered first
            runner.on_signal(move |_: &RoundBoundary| {
                o1.lock().unwrap().push("observe");
            });

            // Response handler registered second
            runner.on_signal_with_response(move |_: &RoundBoundary, _ctx| {
                o2.lock().unwrap().push("response");
                NoiseResponse::None
            });

            runner.execute(&commands);
            let calls = order.lock().unwrap();
            assert_eq!(calls.as_slice(), &["observe", "response"]);
        }

        #[test]
        fn empty_queue_with_signals_dispatches() {
            use pecos_qsim::SparseStab;

            let counter = Arc::new(AtomicU64::new(0));
            let cc = counter.clone();

            let mut commands = CommandQueue::new();
            commands.signal(RoundBoundary(1));
            commands.signal(RoundBoundary(2));

            let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(42);
            runner.on_signal(move |_: &RoundBoundary| {
                cc.fetch_add(1, Ordering::Relaxed);
            });

            runner.execute(&commands);
            assert_eq!(counter.load(Ordering::Relaxed), 2);
        }

        #[test]
        fn multiple_signal_types_at_same_position() {
            use pecos_qsim::SparseStab;

            let round_count = Arc::new(AtomicU64::new(0));
            let temp_count = Arc::new(AtomicU64::new(0));
            let rc = round_count.clone();
            let tc = temp_count.clone();

            let mut commands = CommandBuilder::new().pz(0).mz(0).build();
            // Both signals at position 2 (after all gates)
            commands.signal(RoundBoundary(1));
            commands.signal(Temperature(300.0));

            let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(42);
            runner.on_signal(move |_: &RoundBoundary| {
                rc.fetch_add(1, Ordering::Relaxed);
            });
            runner.on_signal(move |_: &Temperature| {
                tc.fetch_add(1, Ordering::Relaxed);
            });

            runner.execute(&commands);
            assert_eq!(round_count.load(Ordering::Relaxed), 1);
            assert_eq!(temp_count.load(Ordering::Relaxed), 1);
        }

        #[test]
        fn run_shot_dispatches_signals_and_resets() {
            use pecos_qsim::SparseStab;

            let counter = Arc::new(AtomicU64::new(0));
            let cc = counter.clone();

            let mut commands = CommandBuilder::new().pz(0).mz(0).build();
            commands.signal(RoundBoundary(1));

            let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(42);
            runner.on_signal(move |_: &RoundBoundary| {
                cc.fetch_add(1, Ordering::Relaxed);
            });

            // Run two shots -- signals dispatch each time
            let _o1 = runner.run_shot(&commands);
            let _o2 = runner.run_shot(&commands);
            assert_eq!(counter.load(Ordering::Relaxed), 2);
        }
    }

    // ================================================================
    // Gate event handler tests
    // ================================================================

    mod gate_handler_tests {
        use super::*;
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::sync::Arc;

        #[test]
        fn before_gate_handler_called_with_correct_context() {
            let seen_gate = Arc::new(std::sync::Mutex::new(Vec::new()));
            let seen_clone = seen_gate.clone();

            let commands = CommandBuilder::new().pz(0).h(0).mz(0).build();

            let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(42);
            runner.on_before_gate(move |ctx: &DispatchContext<'_>| {
                seen_clone.lock().unwrap().push((
                    ctx.gate_type,
                    ctx.qubits.to_vec(),
                ));
                NoiseResponse::None
            });

            runner.execute(&commands);
            let seen = seen_gate.lock().unwrap();
            // Should have been called for Prep, H, and Measure
            assert_eq!(seen.len(), 3);
            // H on qubit 0
            assert_eq!(seen[1].0, GateType::H);
            assert_eq!(seen[1].1, vec![QubitId(0)]);
        }

        #[test]
        fn after_gate_handler_called() {
            let counter = Arc::new(AtomicU64::new(0));
            let cc = counter.clone();

            let commands = CommandBuilder::new().pz(0).h(0).mz(0).build();

            let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(42);
            runner.on_after_gate(move |_ctx: &DispatchContext<'_>| {
                cc.fetch_add(1, Ordering::Relaxed);
                NoiseResponse::None
            });

            runner.execute(&commands);
            // After-gate is called for H only (not Prep, Measure, or Idle)
            assert_eq!(counter.load(Ordering::Relaxed), 1);
        }

        #[test]
        fn handler_priority_ordering() {
            let order = Arc::new(std::sync::Mutex::new(Vec::new()));
            let o1 = order.clone();
            let o2 = order.clone();
            let o3 = order.clone();

            let commands = CommandBuilder::new().pz(0).h(0).mz(0).build();

            let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(42);

            // Register in non-priority order
            runner.on_before_gate_with_priority(0, move |_| {
                o2.lock().unwrap().push("mid");
                NoiseResponse::None
            });
            runner.on_before_gate_with_priority(-10, move |_| {
                o3.lock().unwrap().push("low");
                NoiseResponse::None
            });
            runner.on_before_gate_with_priority(10, move |_| {
                o1.lock().unwrap().push("high");
                NoiseResponse::None
            });

            runner.execute(&commands);
            let order = order.lock().unwrap();
            // For each of the 3 commands, handlers run in priority order
            // First command (Prep): high, mid, low
            assert_eq!(order[0], "high");
            assert_eq!(order[1], "mid");
            assert_eq!(order[2], "low");
        }

        #[test]
        fn handler_skip_gate_response() {
            // Register a handler that skips H gates
            let commands = CommandBuilder::new()
                .pz(0)
                .h(0) // This should be skipped
                .mz(0)
                .build();

            let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(42);
            runner.on_before_gate(move |ctx: &DispatchContext<'_>| {
                if ctx.gate_type == GateType::H {
                    NoiseResponse::SkipGate
                } else {
                    NoiseResponse::None
                }
            });

            let outcomes = runner.execute(&commands);
            // If H is skipped, qubit stays in |0> state -> measure 0
            let outcome = outcomes.get_bit(QubitId(0)).unwrap();
            assert!(!outcome, "H gate should have been skipped, so result should be 0");
        }

        #[test]
        fn handler_inject_gates_response() {
            // Register an after-gate handler that injects Z after first H
            let commands = CommandBuilder::new()
                .pz(0)
                .h(0)
                .h(0) // H twice = identity, but we inject Z in between
                .mz(0)
                .build();

            let injected = Arc::new(AtomicU64::new(0));
            let ic = injected.clone();

            let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(42);
            runner.on_after_gate(move |ctx: &DispatchContext<'_>| {
                if ctx.gate_type == GateType::H {
                    let prev = ic.fetch_add(1, Ordering::Relaxed);
                    if prev == 0 {
                        // Only inject after first H
                        return NoiseResponse::inject_gate(GateCommand::z(QubitId(0)));
                    }
                }
                NoiseResponse::None
            });

            let outcomes = runner.execute(&commands);
            // Prep |0> -> H -> |+> -> inject Z -> Z|+> = |-> -> H -> |1>
            // So we should get 1
            let outcome = outcomes.get_bit(QubitId(0)).unwrap();
            assert!(outcome, "Injected Z should cause final measurement to be 1");
        }

        #[test]
        fn multiple_handlers_combine_responses() {
            let commands = CommandBuilder::new()
                .pz(0)
                .pz(1)
                .h(0)
                .h(1)
                .mz(0)
                .mz(1)
                .build();

            // Two after-gate handlers each inject X on their respective qubits
            let mut runner = ShotRunner::new(SparseStab::new(2)).with_seed(42);
            runner.on_after_gate(move |ctx: &DispatchContext<'_>| {
                if ctx.gate_type == GateType::H && ctx.qubits[0] == QubitId(0) {
                    NoiseResponse::inject_gate(GateCommand::z(QubitId(0)))
                } else {
                    NoiseResponse::None
                }
            });
            runner.on_after_gate(move |ctx: &DispatchContext<'_>| {
                if ctx.gate_type == GateType::H && ctx.qubits[0] == QubitId(1) {
                    NoiseResponse::inject_gate(GateCommand::z(QubitId(1)))
                } else {
                    NoiseResponse::None
                }
            });

            // Both handlers fired, both injected Z
            runner.execute(&commands);
        }

        #[test]
        fn handlers_coexist_with_noise_model() {
            let handler_called = Arc::new(AtomicU64::new(0));
            let hc = handler_called.clone();

            let commands = CommandBuilder::new().pz(0).h(0).mz(0).build();

            let noise = ComposableNoiseModel::new()
                .add_channel(SingleQubitChannel::depolarizing(0.0)); // No actual noise

            let mut runner = ShotRunner::new(SparseStab::new(1))
                .with_noise(noise)
                .with_seed(42);

            runner.on_before_gate(move |_: &DispatchContext<'_>| {
                hc.fetch_add(1, Ordering::Relaxed);
                NoiseResponse::None
            });

            let outcomes = runner.execute(&commands);
            assert_eq!(outcomes.len(), 1);
            // Handler should have been called for all 3 commands
            assert_eq!(handler_called.load(Ordering::Relaxed), 3);
        }

        #[test]
        fn no_handler_fast_path() {
            // Without handlers, deterministic circuit gives correct results
            let commands = CommandBuilder::new()
                .pz(0)
                .pz(1)
                .x(0) // deterministic: |0> -> |1>
                .cx(0, 1) // deterministic: |10> -> |11>
                .mz(0)
                .mz(1)
                .build();

            let mut runner = ShotRunner::new(SparseStab::new(2)).with_seed(42);
            let outcomes = runner.execute(&commands);

            assert_eq!(outcomes.get_bit(QubitId(0)), Some(true));
            assert_eq!(outcomes.get_bit(QubitId(1)), Some(true));
        }

        #[test]
        fn before_measurement_handler() {
            let called = Arc::new(AtomicU64::new(0));
            let cc = called.clone();

            let commands = CommandBuilder::new().pz(0).h(0).mz(0).build();

            let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(42);
            runner.on_before_measurement(move |ctx: &DispatchContext<'_>| {
                cc.fetch_add(1, Ordering::Relaxed);
                assert!(ctx.gate_type.is_measurement());
                NoiseResponse::None
            });

            runner.execute(&commands);
            assert_eq!(called.load(Ordering::Relaxed), 1);
        }

        #[test]
        fn after_measurement_handler_with_outcomes() {
            let outcomes_seen = Arc::new(std::sync::Mutex::new(Vec::new()));
            let oc = outcomes_seen.clone();

            // Prep(0) then measure -> should give 0
            let commands = CommandBuilder::new().pz(0).mz(0).build();

            let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(42);
            runner.on_after_measurement(move |ctx: &DispatchContext<'_>| {
                if let Some(outcomes) = ctx.outcomes {
                    oc.lock().unwrap().extend_from_slice(outcomes);
                }
                NoiseResponse::None
            });

            runner.execute(&commands);
            let seen = outcomes_seen.lock().unwrap();
            assert_eq!(seen.len(), 1);
            assert!(!seen[0], "Prep then measure should give 0");
        }

        #[test]
        fn idle_handler_with_duration() {
            let duration_seen = Arc::new(std::sync::Mutex::new(Vec::new()));
            let dc = duration_seen.clone();

            let commands = CommandBuilder::new()
                .pz(0)
                .idle(0, 100u64)
                .mz(0)
                .build();

            let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(42);
            runner.on_idle(move |ctx: &DispatchContext<'_>| {
                if let Some(dur) = ctx.duration {
                    dc.lock().unwrap().push(dur);
                }
                NoiseResponse::None
            });

            runner.execute(&commands);
            let seen = duration_seen.lock().unwrap();
            assert_eq!(seen.len(), 1);
            assert_eq!(seen[0], TimeUnits::new(100));
        }

        #[test]
        fn after_preparation_handler() {
            let called = Arc::new(AtomicU64::new(0));
            let cc = called.clone();

            let commands = CommandBuilder::new().pz(0).pz(1).mz(0).build();

            let mut runner = ShotRunner::new(SparseStab::new(2)).with_seed(42);
            runner.on_after_preparation(move |ctx: &DispatchContext<'_>| {
                assert!(ctx.gate_type.is_preparation());
                cc.fetch_add(1, Ordering::Relaxed);
                NoiseResponse::None
            });

            runner.execute(&commands);
            assert_eq!(called.load(Ordering::Relaxed), 2);
        }

        #[test]
        fn dispatch_context_has_noise_context_when_model_present() {
            let has_noise_ctx = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let hc = has_noise_ctx.clone();

            let commands = CommandBuilder::new().pz(0).h(0).mz(0).build();

            let noise = ComposableNoiseModel::new()
                .add_channel(SingleQubitChannel::depolarizing(0.0));

            let mut runner = ShotRunner::new(SparseStab::new(1))
                .with_noise(noise)
                .with_seed(42);

            runner.on_before_gate(move |ctx: &DispatchContext<'_>| {
                if ctx.noise_context.is_some() {
                    hc.store(true, Ordering::Relaxed);
                }
                NoiseResponse::None
            });

            runner.execute(&commands);
            assert!(
                has_noise_ctx.load(Ordering::Relaxed),
                "DispatchContext should have noise_context when noise model is present"
            );
        }

        #[test]
        fn dispatch_context_noise_context_none_without_model() {
            let has_noise_ctx = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let hc = has_noise_ctx.clone();

            let commands = CommandBuilder::new().pz(0).h(0).mz(0).build();

            let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(42);
            runner.on_before_gate(move |ctx: &DispatchContext<'_>| {
                if ctx.noise_context.is_some() {
                    hc.store(true, Ordering::Relaxed);
                }
                NoiseResponse::None
            });

            runner.execute(&commands);
            assert!(
                !has_noise_ctx.load(Ordering::Relaxed),
                "DispatchContext should not have noise_context when no noise model"
            );
        }

        #[test]
        fn existing_noise_behavior_unchanged() {
            // Ensure noise model still works with handler infrastructure
            // Uses deterministic circuit (no superposition)
            let commands = CommandBuilder::new()
                .pz(0)
                .pz(1)
                .x(0) // deterministic: |0> -> |1>
                .cx(0, 1) // deterministic: |10> -> |11>
                .mz(0)
                .mz(1)
                .build();

            let noise = ComposableNoiseModel::new()
                .add_channel(SingleQubitChannel::depolarizing(0.0));

            let mut runner = ShotRunner::new(SparseStab::new(2))
                .with_noise(noise)
                .with_seed(42);

            let outcomes = runner.execute(&commands);

            // Deterministic result: both qubits should be 1
            assert_eq!(outcomes.get_bit(QubitId(0)), Some(true));
            assert_eq!(outcomes.get_bit(QubitId(1)), Some(true));
        }

        #[test]
        fn handler_with_execute_all() {
            let counter = Arc::new(AtomicU64::new(0));
            let cc = counter.clone();

            let commands = CommandBuilder::new()
                .pz(0)
                .rx(0, Angle64::HALF_TURN)
                .mz(0)
                .build();

            let mut runner = ShotRunner::new(StateVec::new(1)).with_seed(42);
            runner.on_before_gate(move |_: &DispatchContext<'_>| {
                cc.fetch_add(1, Ordering::Relaxed);
                NoiseResponse::None
            });

            let outcomes = runner.execute_all(&commands);
            assert!(outcomes.get_bit(QubitId(0)).unwrap());
            // 3 commands: Prep, RX, Measure
            assert_eq!(counter.load(Ordering::Relaxed), 3);
        }

        #[test]
        fn gate_context_has_angles() {
            let angles_seen = Arc::new(std::sync::Mutex::new(Vec::new()));
            let ac = angles_seen.clone();

            let commands = CommandBuilder::new()
                .pz(0)
                .rx(0, Angle64::QUARTER_TURN)
                .mz(0)
                .build();

            let mut runner = ShotRunner::new(StateVec::new(1)).with_seed(42);
            runner.on_before_gate(move |ctx: &DispatchContext<'_>| {
                if ctx.gate_type == GateType::RX {
                    ac.lock().unwrap().extend_from_slice(ctx.angles);
                }
                NoiseResponse::None
            });

            runner.execute_all(&commands);
            let seen = angles_seen.lock().unwrap();
            assert_eq!(seen.len(), 1);
            assert_eq!(seen[0], Angle64::QUARTER_TURN);
        }

        #[test]
        fn gate_context_has_gate_id() {
            let gate_ids = Arc::new(std::sync::Mutex::new(Vec::new()));
            let gc = gate_ids.clone();

            let commands = CommandBuilder::new().pz(0).h(0).mz(0).build();

            let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(42);
            runner.on_before_gate(move |ctx: &DispatchContext<'_>| {
                gc.lock().unwrap().push(ctx.gate_id);
                NoiseResponse::None
            });

            runner.execute(&commands);
            let ids = gate_ids.lock().unwrap();
            // All should have gate IDs (derived from gate_type)
            for id in ids.iter() {
                assert!(id.is_some(), "gate_id should always be populated");
            }
        }

        #[test]
        fn gate_id_maps_correctly_for_each_gate_type() {
            use crate::extensible::gates;

            let gate_data = Arc::new(std::sync::Mutex::new(Vec::new()));
            let gd = gate_data.clone();

            let commands = CommandBuilder::new()
                .pz(0)
                .pz(1)
                .h(0)
                .cx(0, 1)
                .rz(0, Angle64::QUARTER_TURN)
                .mz(0)
                .mz(1)
                .build();

            let mut runner = ShotRunner::new(StateVec::new(2)).with_seed(42);
            runner.on_before_gate(move |ctx: &DispatchContext<'_>| {
                gd.lock().unwrap().push((ctx.gate_type, ctx.gate_id.unwrap()));
                NoiseResponse::None
            });

            runner.execute_all(&commands);
            let data = gate_data.lock().unwrap();
            assert_eq!(data[0], (GateType::PZ, gates::PZ));
            assert_eq!(data[1], (GateType::PZ, gates::PZ));
            assert_eq!(data[2], (GateType::H, gates::H));
            assert_eq!(data[3], (GateType::CX, gates::CX));
            assert_eq!(data[4], (GateType::RZ, gates::RZ));
            assert_eq!(data[5], (GateType::MZ, gates::MZ));
            assert_eq!(data[6], (GateType::MZ, gates::MZ));
        }

        #[test]
        fn after_measurement_outcomes_match_actual_results() {
            let outcome_data = Arc::new(std::sync::Mutex::new(Vec::new()));
            let od = outcome_data.clone();

            // Prep |0>, X -> |1>, measure -> should get 1
            let commands = CommandBuilder::new().pz(0).x(0).mz(0).build();

            let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(42);
            runner.on_after_measurement(move |ctx: &DispatchContext<'_>| {
                od.lock().unwrap().push(ctx.outcomes.unwrap().to_vec());
                NoiseResponse::None
            });

            let outcomes = runner.execute(&commands);
            let recorded = outcome_data.lock().unwrap();
            assert_eq!(recorded.len(), 1);
            assert_eq!(recorded[0], vec![true]); // |1> measured
            assert_eq!(outcomes.get_bit(QubitId(0)), Some(true));
        }

        #[test]
        fn after_measurement_outcomes_multi_qubit() {
            let outcome_data = Arc::new(std::sync::Mutex::new(Vec::new()));
            let od = outcome_data.clone();

            // Prep both, X on q0 only -> q0=1, q1=0
            let commands = CommandBuilder::new()
                .pz(0)
                .pz(1)
                .x(0)
                .mz(0)
                .mz(1)
                .build();

            let mut runner = ShotRunner::new(SparseStab::new(2)).with_seed(42);
            runner.on_after_measurement(move |ctx: &DispatchContext<'_>| {
                od.lock().unwrap().push((
                    ctx.qubits.to_vec(),
                    ctx.outcomes.unwrap().to_vec(),
                ));
                NoiseResponse::None
            });

            runner.execute(&commands);
            let recorded = outcome_data.lock().unwrap();
            assert_eq!(recorded.len(), 2);
            // First measurement: q0 = 1
            assert_eq!(recorded[0].0, vec![QubitId(0)]);
            assert_eq!(recorded[0].1, vec![true]);
            // Second measurement: q1 = 0
            assert_eq!(recorded[1].0, vec![QubitId(1)]);
            assert_eq!(recorded[1].1, vec![false]);
        }

        #[test]
        fn idle_handler_gate_id_is_none() {
            let gate_ids = Arc::new(std::sync::Mutex::new(Vec::new()));
            let gc = gate_ids.clone();

            let commands = CommandBuilder::new()
                .pz(0)
                .idle(0, TimeUnits::new(50))
                .mz(0)
                .build();

            let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(42);
            runner.on_idle(move |ctx: &DispatchContext<'_>| {
                gc.lock().unwrap().push(ctx.gate_id);
                NoiseResponse::None
            });

            runner.execute(&commands);
            let ids = gate_ids.lock().unwrap();
            assert_eq!(ids.len(), 1);
            assert_eq!(ids[0], Option::None, "idle gate_id should be None");
        }

        #[test]
        fn same_priority_preserves_registration_order() {
            let order = Arc::new(std::sync::Mutex::new(Vec::new()));
            let o1 = order.clone();
            let o2 = order.clone();
            let o3 = order.clone();

            let commands = CommandBuilder::new().pz(0).h(0).mz(0).build();

            let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(42);

            // All same priority (0), should run in registration order
            runner.on_before_gate(move |_| {
                o1.lock().unwrap().push("first");
                NoiseResponse::None
            });
            runner.on_before_gate(move |_| {
                o2.lock().unwrap().push("second");
                NoiseResponse::None
            });
            runner.on_before_gate(move |_| {
                o3.lock().unwrap().push("third");
                NoiseResponse::None
            });

            runner.execute(&commands);
            let calls = order.lock().unwrap();
            // Should see pattern repeated for each gate: first, second, third
            assert_eq!(calls[0], "first");
            assert_eq!(calls[1], "second");
            assert_eq!(calls[2], "third");
        }

        #[test]
        fn skip_gate_plus_inject_gates_skips_but_still_injects() {
            // Handler 1 says skip, handler 2 says inject X.
            // Expected: original gate skipped, but X still injected.
            let commands = CommandBuilder::new()
                .pz(0)
                .z(0) // this gate should be skipped
                .mz(0)
                .build();

            let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(42);

            // High-priority handler skips the Z gate
            runner.on_before_gate_with_priority(10, |ctx: &DispatchContext<'_>| {
                if ctx.gate_type == GateType::Z {
                    NoiseResponse::SkipGate
                } else {
                    NoiseResponse::None
                }
            });

            // Low-priority handler injects X when Z is attempted
            runner.on_before_gate_with_priority(-10, |ctx: &DispatchContext<'_>| {
                if ctx.gate_type == GateType::Z {
                    NoiseResponse::inject_gate(GateCommand::x(QubitId(0)))
                } else {
                    NoiseResponse::None
                }
            });

            let outcomes = runner.execute(&commands);
            // Z was skipped, X was injected: |0> -> X -> |1>
            assert_eq!(outcomes.get_bit(QubitId(0)), Some(true));
        }

        #[test]
        fn double_flip_outcomes_cancels_out() {
            let commands = CommandBuilder::new().pz(0).x(0).mz(0).build();

            let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(42);

            // Two handlers both flip q0 outcome -> should cancel out
            runner.on_after_measurement(|ctx: &DispatchContext<'_>| {
                NoiseResponse::FlipOutcomes(smallvec::smallvec![ctx.qubits[0]])
            });
            runner.on_after_measurement(|ctx: &DispatchContext<'_>| {
                NoiseResponse::FlipOutcomes(smallvec::smallvec![ctx.qubits[0]])
            });

            let outcomes = runner.execute(&commands);
            // X puts qubit in |1>, double flip cancels: still 1
            assert_eq!(outcomes.get_bit(QubitId(0)), Some(true));
        }

        #[test]
        fn gate_handlers_plus_signal_handlers_coexist() {
            let gate_count = Arc::new(AtomicU64::new(0));
            let signal_count = Arc::new(AtomicU64::new(0));
            let gc = gate_count.clone();
            let sc = signal_count.clone();

            use pecos_core::impl_signal;
            #[derive(Copy, Clone, Debug)]
            struct Marker;
            impl_signal!(Marker);

            let mut commands = CommandBuilder::new().pz(0).x(0).mz(0).build();
            commands.signal(Marker);

            let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(42);

            runner.on_before_gate(move |_| {
                gc.fetch_add(1, Ordering::Relaxed);
                NoiseResponse::None
            });

            runner.on_signal(move |_: &Marker| {
                sc.fetch_add(1, Ordering::Relaxed);
            });

            let outcomes = runner.execute(&commands);
            // Gate handler called for PZ, X, MZ = 3
            assert_eq!(gate_count.load(Ordering::Relaxed), 3);
            // Signal handler called once
            assert_eq!(signal_count.load(Ordering::Relaxed), 1);
            // Circuit still works: X on |0> -> measure 1
            assert_eq!(outcomes.get_bit(QubitId(0)), Some(true));
        }

        #[test]
        fn signal_response_plus_gate_handler_response_both_applied() {
            use pecos_core::impl_signal;
            #[derive(Copy, Clone, Debug)]
            struct FlipQ0;
            impl_signal!(FlipQ0);

            // Circuit: prep both, signal (injects X on q0), then X on q1 via gate handler, then measure
            let mut commands = CommandBuilder::new().pz(0).pz(1).build();
            commands.signal(FlipQ0);
            commands.push(GateCommand::h(QubitId(0))); // dummy gate to trigger handler
            commands.push(GateCommand::h(QubitId(0))); // H*H = I
            commands.push(GateCommand::mz(QubitId(0)));
            commands.push(GateCommand::mz(QubitId(1)));

            let mut runner = ShotRunner::new(SparseStab::new(2)).with_seed(42);

            // Signal injects X on q0
            runner.on_signal_with_response(|_: &FlipQ0, _ctx| {
                NoiseResponse::inject_gate(GateCommand::x(QubitId(0)))
            });

            // After-gate handler injects X on q1 after the first H gate
            let injected = Arc::new(AtomicU64::new(0));
            let ic = injected.clone();
            runner.on_after_gate(move |ctx: &DispatchContext<'_>| {
                if ctx.gate_type == GateType::H {
                    let prev = ic.fetch_add(1, Ordering::Relaxed);
                    if prev == 0 {
                        return NoiseResponse::inject_gate(GateCommand::x(QubitId(1)));
                    }
                }
                NoiseResponse::None
            });

            let outcomes = runner.execute(&commands);
            // q0: |0> -> X (signal) -> H -> H -> |1>
            assert_eq!(outcomes.get_bit(QubitId(0)), Some(true));
            // q1: |0> -> X (gate handler after first H) -> |1>
            assert_eq!(outcomes.get_bit(QubitId(1)), Some(true));
        }

        #[test]
        fn execute_all_dispatches_all_handler_types() {
            let after_gate_count = Arc::new(AtomicU64::new(0));
            let before_meas_count = Arc::new(AtomicU64::new(0));
            let after_meas_count = Arc::new(AtomicU64::new(0));
            let after_prep_count = Arc::new(AtomicU64::new(0));
            let idle_count = Arc::new(AtomicU64::new(0));

            let ag = after_gate_count.clone();
            let bm = before_meas_count.clone();
            let am = after_meas_count.clone();
            let ap = after_prep_count.clone();
            let ic = idle_count.clone();

            let commands = CommandBuilder::new()
                .pz(0)              // triggers after_preparation
                .h(0)               // triggers after_gate
                .idle(0, TimeUnits::new(10)) // triggers idle
                .h(0)               // triggers after_gate
                .mz(0)              // triggers before/after_measurement
                .build();

            let mut runner = ShotRunner::new(StateVec::new(1)).with_seed(42);

            runner.on_after_gate(move |_| { ag.fetch_add(1, Ordering::Relaxed); NoiseResponse::None });
            runner.on_before_measurement(move |_| { bm.fetch_add(1, Ordering::Relaxed); NoiseResponse::None });
            runner.on_after_measurement(move |_| { am.fetch_add(1, Ordering::Relaxed); NoiseResponse::None });
            runner.on_after_preparation(move |_| { ap.fetch_add(1, Ordering::Relaxed); NoiseResponse::None });
            runner.on_idle(move |_| { ic.fetch_add(1, Ordering::Relaxed); NoiseResponse::None });

            runner.execute_all(&commands);

            assert_eq!(after_gate_count.load(Ordering::Relaxed), 2, "after_gate for 2 H gates");
            assert_eq!(before_meas_count.load(Ordering::Relaxed), 1, "before_measurement");
            assert_eq!(after_meas_count.load(Ordering::Relaxed), 1, "after_measurement");
            assert_eq!(after_prep_count.load(Ordering::Relaxed), 1, "after_preparation");
            assert_eq!(idle_count.load(Ordering::Relaxed), 1, "idle");
        }

        #[test]
        fn run_shot_dispatches_gate_handlers() {
            let counter = Arc::new(AtomicU64::new(0));
            let cc = counter.clone();

            let commands = CommandBuilder::new().pz(0).x(0).mz(0).build();

            let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(42);
            runner.on_before_gate(move |_| {
                cc.fetch_add(1, Ordering::Relaxed);
                NoiseResponse::None
            });

            let o1 = runner.run_shot(&commands);
            assert_eq!(o1.get_bit(QubitId(0)), Some(true));
            assert_eq!(counter.load(Ordering::Relaxed), 3); // PZ, X, MZ

            // Second shot also dispatches
            let _o2 = runner.run_shot(&commands);
            assert_eq!(counter.load(Ordering::Relaxed), 6);
        }

        #[test]
        fn run_shot_all_dispatches_gate_handlers() {
            let counter = Arc::new(AtomicU64::new(0));
            let cc = counter.clone();

            let commands = CommandBuilder::new()
                .pz(0)
                .rx(0, Angle64::HALF_TURN)
                .mz(0)
                .build();

            let mut runner = ShotRunner::new(StateVec::new(1)).with_seed(42);
            runner.on_before_gate(move |_| {
                cc.fetch_add(1, Ordering::Relaxed);
                NoiseResponse::None
            });

            let outcomes = runner.run_shot_all(&commands);
            assert_eq!(outcomes.get_bit(QubitId(0)), Some(true));
            assert_eq!(counter.load(Ordering::Relaxed), 3);
        }
    }
}
