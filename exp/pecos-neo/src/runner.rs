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

//! Unified quantum simulation runner.
//!
//! Executes both `CommandQueue` (GateType-based) and `AdaptedSequence` (GateId-based)
//! circuits on a simulator, with noise, signal dispatch, gate overrides, and automatic
//! decomposition.
//!
//! # Gate Execution Order
//!
//! For each gate, the runner follows this precedence chain:
//! 1. **Overrides**: Custom executors registered via `GateOverrides`
//! 2. **Clifford trait methods**: Core Clifford gates via `CliffordGateable`
//! 3. **Rotation trait methods**: If `rotations()` constructor was used
//! 4. **Decomposition**: Expand using `GateDefinitions`
//! 5. **Error**: If none of the above apply
//!
//! Before and after each gate, noise events and user handlers are dispatched.
//!
//! # Example
//!
//! ```
//! use pecos_neo::prelude::*;
//! use pecos_simulators::SparseStab;
//!
//! let commands = CommandBuilder::new()
//!     .pz(&[0])
//!     .h(&[0])
//!     .mz(&[0])
//!     .build();
//!
//! let mut state = SparseStab::new(1);
//! let mut runner = CircuitRunner::<SparseStab>::new().with_seed(42);
//! let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
//! ```

use crate::command::{CommandQueue, GateCommand, GateType, SignalStore};
use crate::extensible::{
    AdaptedOp, AdaptedSequence, GateDefinitions, GateId, MeasBasis, PrepBasis, ResultId,
};
use crate::noise::context::NoiseContext;
use crate::noise::{ComposableNoiseModel, NoiseEvent, NoiseResponse};
use crate::outcome::{MeasurementOutcome, MeasurementOutcomes};
use pecos_core::rng::rng_manageable::{RngManageable, derive_seed};
use pecos_core::{Angle64, QubitId, Signal, TimeUnits};
use pecos_random::PecosRng;
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable};
use rand_core::SeedableRng;
use smallvec::SmallVec;
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::Arc;

/// Convert a flat qubit slice to a vector of pairs.
fn flat_to_pairs(qubits: &[QubitId]) -> SmallVec<[(QubitId, QubitId); 4]> {
    qubits.chunks_exact(2).map(|c| (c[0], c[1])).collect()
}

// ============================================================================
// Signal Handler Infrastructure
// ============================================================================

/// Type-erased observe-only signal handler.
type ErasedHandler = Arc<dyn Fn(&dyn Any) + Send + Sync>;

/// Type-erased response-producing signal handler.
type ErasedResponseHandler =
    Arc<dyn Fn(&dyn Any, &DispatchContext<'_>) -> NoiseResponse + Send + Sync>;

/// Registry of signal handlers, keyed by signal `TypeId`.
///
/// Uses a flat `Vec` rather than a `HashMap` -- with typically 1-3 signal
/// types registered, linear scan on contiguous memory is faster than hashing.
#[derive(Clone, Default)]
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
        if let Some((_, handlers)) = self
            .response_handlers
            .iter_mut()
            .find(|(id, _)| *id == type_id)
        {
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

    fn call_response(
        &self,
        type_id: TypeId,
        data: &dyn Any,
        ctx: &DispatchContext<'_>,
    ) -> NoiseResponse {
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

// ============================================================================
// Dispatch Context
// ============================================================================

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

// ============================================================================
// Gate Event Handlers
// ============================================================================

/// Type-erased gate event handler.
type ErasedGateHandler = Arc<dyn Fn(&DispatchContext<'_>) -> NoiseResponse + Send + Sync>;

/// A handler with an associated priority for ordering.
#[derive(Clone)]
struct PrioritizedHandler {
    handler: ErasedGateHandler,
    priority: i32,
}

/// Per-event-type handler storage.
///
/// Each Vec is kept sorted by priority (higher runs first).
/// Using flat Vecs avoids runtime event-type filtering.
#[derive(Clone, Default)]
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

// ============================================================================
// EventHandlers (public, cloneable handler collection for sim_neo)
// ============================================================================

/// A cloneable collection of gate event and signal handlers.
///
/// `EventHandlers` lets you register handlers once and pass them through
/// `sim_neo().event_handlers(...)` so that each parallel worker receives
/// a clone. Handlers are stored as `Arc<dyn Fn>`, so cloning is cheap.
///
/// # Example
///
/// ```
/// use pecos_neo::prelude::*;
/// use std::sync::atomic::{AtomicUsize, Ordering};
/// use std::sync::Arc;
///
/// let counter = Arc::new(AtomicUsize::new(0));
/// let c = counter.clone();
///
/// let handlers = EventHandlers::new()
///     .on_before_gate(move |_ctx| {
///         c.fetch_add(1, Ordering::Relaxed);
///         NoiseResponse::None
///     });
/// ```
#[derive(Clone, Default)]
pub struct EventHandlers {
    gate_handlers: GateEventHandlers,
    signal_handlers: SignalHandlerRegistry,
}

impl EventHandlers {
    /// Create an empty handler collection.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if no handlers have been registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.gate_handlers.before_gate.is_empty()
            && self.gate_handlers.after_gate.is_empty()
            && self.gate_handlers.before_measurement.is_empty()
            && self.gate_handlers.after_measurement.is_empty()
            && self.gate_handlers.after_preparation.is_empty()
            && self.gate_handlers.idle.is_empty()
            && self.signal_handlers.handlers.is_empty()
            && self.signal_handlers.response_handlers.is_empty()
    }

    // ================================================================
    // Gate event handler registration (builder pattern)
    // ================================================================

    /// Register a handler called before each gate is applied.
    #[must_use]
    pub fn on_before_gate(
        mut self,
        handler: impl Fn(&DispatchContext<'_>) -> NoiseResponse + Send + Sync + 'static,
    ) -> Self {
        GateEventHandlers::insert(&mut self.gate_handlers.before_gate, Arc::new(handler), 0);
        self
    }

    /// Register a before-gate handler with explicit priority.
    #[must_use]
    pub fn on_before_gate_with_priority(
        mut self,
        priority: i32,
        handler: impl Fn(&DispatchContext<'_>) -> NoiseResponse + Send + Sync + 'static,
    ) -> Self {
        GateEventHandlers::insert(
            &mut self.gate_handlers.before_gate,
            Arc::new(handler),
            priority,
        );
        self
    }

    /// Register a handler called after each gate is applied.
    #[must_use]
    pub fn on_after_gate(
        mut self,
        handler: impl Fn(&DispatchContext<'_>) -> NoiseResponse + Send + Sync + 'static,
    ) -> Self {
        GateEventHandlers::insert(&mut self.gate_handlers.after_gate, Arc::new(handler), 0);
        self
    }

    /// Register an after-gate handler with explicit priority.
    #[must_use]
    pub fn on_after_gate_with_priority(
        mut self,
        priority: i32,
        handler: impl Fn(&DispatchContext<'_>) -> NoiseResponse + Send + Sync + 'static,
    ) -> Self {
        GateEventHandlers::insert(
            &mut self.gate_handlers.after_gate,
            Arc::new(handler),
            priority,
        );
        self
    }

    /// Register a handler called before each measurement.
    #[must_use]
    pub fn on_before_measurement(
        mut self,
        handler: impl Fn(&DispatchContext<'_>) -> NoiseResponse + Send + Sync + 'static,
    ) -> Self {
        GateEventHandlers::insert(
            &mut self.gate_handlers.before_measurement,
            Arc::new(handler),
            0,
        );
        self
    }

    /// Register a before-measurement handler with explicit priority.
    #[must_use]
    pub fn on_before_measurement_with_priority(
        mut self,
        priority: i32,
        handler: impl Fn(&DispatchContext<'_>) -> NoiseResponse + Send + Sync + 'static,
    ) -> Self {
        GateEventHandlers::insert(
            &mut self.gate_handlers.before_measurement,
            Arc::new(handler),
            priority,
        );
        self
    }

    /// Register a handler called after each measurement.
    #[must_use]
    pub fn on_after_measurement(
        mut self,
        handler: impl Fn(&DispatchContext<'_>) -> NoiseResponse + Send + Sync + 'static,
    ) -> Self {
        GateEventHandlers::insert(
            &mut self.gate_handlers.after_measurement,
            Arc::new(handler),
            0,
        );
        self
    }

    /// Register an after-measurement handler with explicit priority.
    #[must_use]
    pub fn on_after_measurement_with_priority(
        mut self,
        priority: i32,
        handler: impl Fn(&DispatchContext<'_>) -> NoiseResponse + Send + Sync + 'static,
    ) -> Self {
        GateEventHandlers::insert(
            &mut self.gate_handlers.after_measurement,
            Arc::new(handler),
            priority,
        );
        self
    }

    /// Register a handler called after each preparation.
    #[must_use]
    pub fn on_after_preparation(
        mut self,
        handler: impl Fn(&DispatchContext<'_>) -> NoiseResponse + Send + Sync + 'static,
    ) -> Self {
        GateEventHandlers::insert(
            &mut self.gate_handlers.after_preparation,
            Arc::new(handler),
            0,
        );
        self
    }

    /// Register an after-preparation handler with explicit priority.
    #[must_use]
    pub fn on_after_preparation_with_priority(
        mut self,
        priority: i32,
        handler: impl Fn(&DispatchContext<'_>) -> NoiseResponse + Send + Sync + 'static,
    ) -> Self {
        GateEventHandlers::insert(
            &mut self.gate_handlers.after_preparation,
            Arc::new(handler),
            priority,
        );
        self
    }

    /// Register a handler called during idle time.
    #[must_use]
    pub fn on_idle(
        mut self,
        handler: impl Fn(&DispatchContext<'_>) -> NoiseResponse + Send + Sync + 'static,
    ) -> Self {
        GateEventHandlers::insert(&mut self.gate_handlers.idle, Arc::new(handler), 0);
        self
    }

    /// Register an idle handler with explicit priority.
    #[must_use]
    pub fn on_idle_with_priority(
        mut self,
        priority: i32,
        handler: impl Fn(&DispatchContext<'_>) -> NoiseResponse + Send + Sync + 'static,
    ) -> Self {
        GateEventHandlers::insert(&mut self.gate_handlers.idle, Arc::new(handler), priority);
        self
    }

    // ================================================================
    // Signal handler registration (builder pattern)
    // ================================================================

    /// Register a handler that will be called when a signal of type `Sig` is dispatched.
    #[must_use]
    pub fn on_signal<Sig: Signal>(
        mut self,
        handler: impl Fn(&Sig) + Send + Sync + 'static,
    ) -> Self {
        let erased: ErasedHandler = Arc::new(move |data: &dyn Any| {
            if let Some(signal) = data.downcast_ref::<Sig>() {
                handler(signal);
            }
        });
        self.signal_handlers.add(TypeId::of::<Sig>(), erased);
        self
    }

    /// Register a response-producing signal handler.
    #[must_use]
    pub fn on_signal_with_response<Sig: Signal>(
        mut self,
        handler: impl Fn(&Sig, &DispatchContext<'_>) -> NoiseResponse + Send + Sync + 'static,
    ) -> Self {
        let erased: ErasedResponseHandler =
            Arc::new(move |data: &dyn Any, ctx: &DispatchContext<'_>| {
                if let Some(signal) = data.downcast_ref::<Sig>() {
                    handler(signal, ctx)
                } else {
                    NoiseResponse::None
                }
            });
        self.signal_handlers
            .add_response(TypeId::of::<Sig>(), erased);
        self
    }
}

/// Cursor for tracking position within a signal channel during dispatch.
struct SignalCursor {
    type_id: TypeId,
    /// Index into `SignalStore::channels` -- resolved once, not per-dispatch.
    channel_idx: usize,
    /// Current position within the channel's entries.
    entry_idx: usize,
    /// Cached length of the channel.
    len: usize,
}

// ============================================================================
// Gate Overrides
// ============================================================================

/// Function signature for custom gate executors.
///
/// Takes a mutable reference to the simulator, angles, and qubit operands.
/// Returns `true` if the gate was executed successfully, `false` otherwise.
pub type GateExecutorFn<S> = fn(&mut S, &[Angle64], &[QubitId]) -> bool;

/// Function signature for rotation gate execution.
///
/// Used internally to enable `execute_gate()` to execute rotation gates when the
/// simulator supports `ArbitraryRotationGateable`. Set by `rotations()` constructor.
type RotationExecutorFn<S> = fn(&mut S, GateId, &[Angle64], &[QubitId]) -> bool;

/// Registry of custom gate implementations.
///
/// Allows registering custom executors for any `GateId`, including core gates.
/// When a gate has an override registered, the override takes precedence over
/// both trait-based execution and decomposition.
///
/// # Example
///
/// ```
/// # use pecos_simulators::SparseStab;
/// use pecos_simulators::CliffordGateable;
/// use pecos_neo::runner::GateOverrides;
/// use pecos_neo::extensible::gates;
///
/// let overrides: GateOverrides<SparseStab> = GateOverrides::new()
///     .register(gates::X, |sim, _angles, qubits| {
///         sim.h(qubits);
///         true
///     });
/// ```
pub struct GateOverrides<S> {
    overrides: HashMap<GateId, GateExecutorFn<S>>,
}

impl<S> Clone for GateOverrides<S> {
    fn clone(&self) -> Self {
        Self {
            overrides: self.overrides.clone(),
        }
    }
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

// ============================================================================
// Execution Error
// ============================================================================

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

// ============================================================================
// CircuitRunner
// ============================================================================

/// Stateless quantum simulation runner.
///
/// Applies noise and circuits to a simulator state, producing measurement outcomes.
/// The runner does not own the simulator -- it borrows it during execution.
///
/// Three granularity levels:
/// - [`apply_circuit`](CircuitRunner::apply_circuit) -- full circuit execution (common case)
/// - [`apply_gate`](CircuitRunner::apply_gate) -- single gate with noise/handlers (interpreter mode)
/// - [`apply_noise`](CircuitRunner::apply_noise) -- apply a noise event directly to state
///
/// # Constructors
///
/// - [`CircuitRunner::new()`](CircuitRunner::new) -- Clifford gates with default definitions
/// - [`CircuitRunner::with_definitions(defs)`](CircuitRunner::with_definitions) -- Explicit definitions
/// - [`CircuitRunner::rotations()`](CircuitRunner::rotations) -- Clifford + rotation gates (requires
///   `ArbitraryRotationGateable`)
/// - [`CircuitRunner::rotations_with_definitions(defs)`](CircuitRunner::rotations_with_definitions)
///   -- Rotation gates with explicit definitions
///
/// # Example
///
/// ```
/// use pecos_neo::prelude::*;
/// use pecos_simulators::SparseStab;
///
/// let commands = CommandBuilder::new()
///     .pz(&[0])
///     .h(&[0])
///     .cx(&[(0, 1)])
///     .mz(&[0])
///     .mz(&[1])
///     .build();
///
/// let mut state = SparseStab::new(2);
/// let mut runner = CircuitRunner::<SparseStab>::new().with_seed(42);
/// let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
///
/// let o0 = outcomes.get_bit(QubitId(0)).unwrap();
/// let o1 = outcomes.get_bit(QubitId(1)).unwrap();
/// assert_eq!(o0, o1);
/// ```
pub struct CircuitRunner<S: CliffordGateable> {
    // Configuration (set at construction, immutable during execution)
    definitions: GateDefinitions,
    overrides: Option<GateOverrides<S>>,
    rotation_executor: Option<RotationExecutorFn<S>>,
    max_decomp_depth: usize,

    // Execution state (owned by runner, mutated during execution)
    rng: PecosRng,
    noise: Option<ComposableNoiseModel>,
    signal_handlers: SignalHandlerRegistry,
    gate_handlers: GateEventHandlers,

    // Scratch buffers (cleared at start of apply_circuit)
    outcomes: MeasurementOutcomes,
    results: Vec<bool>,
}

// ============================================================================
// Constructors and Configuration
// ============================================================================

impl<S: CliffordGateable> Default for CircuitRunner<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: CliffordGateable> CircuitRunner<S> {
    /// Create a new runner with Clifford gate support and default definitions.
    ///
    /// The type parameter `S` is inferred from usage (e.g., `apply_circuit(&mut sparse_stab, ...)`)
    /// or via turbofish (`CircuitRunner::<SparseStab>::new()`).
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_neo::prelude::*;
    /// use pecos_simulators::SparseStab;
    ///
    /// let mut runner = CircuitRunner::<SparseStab>::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self::with_definitions(GateDefinitions::new())
    }

    /// Create a new runner with Clifford gate support and explicit definitions.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_neo::prelude::*;
    /// use pecos_simulators::SparseStab;
    ///
    /// let defs = GateDefinitions::new();
    /// let mut runner = CircuitRunner::<SparseStab>::with_definitions(defs);
    /// ```
    #[must_use]
    pub fn with_definitions(definitions: GateDefinitions) -> Self {
        Self {
            definitions,
            rng: PecosRng::from_rng(&mut rand::rng()),
            outcomes: MeasurementOutcomes::new(),
            overrides: None,
            rotation_executor: None,
            results: Vec::new(),
            max_decomp_depth: 10,
            noise: None,
            signal_handlers: SignalHandlerRegistry::new(),
            gate_handlers: GateEventHandlers::new(),
        }
    }

    /// Check if rotation gates are enabled.
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
    ///
    /// Gate definitions are automatically propagated to the noise model's context.
    #[must_use]
    pub fn with_noise(mut self, mut noise: ComposableNoiseModel) -> Self {
        noise = noise.with_gate_definitions(self.definitions.clone());
        self.noise = Some(noise);
        self
    }

    /// Set the RNG seed for noise operations.
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

    /// Set maximum decomposition depth.
    #[must_use]
    pub fn with_max_decomp_depth(mut self, depth: usize) -> Self {
        self.max_decomp_depth = depth;
        self
    }

    /// Merge an [`EventHandlers`] collection into this runner.
    ///
    /// Extends the existing gate and signal handler registries. Gate handler
    /// Vecs are re-sorted by priority after merging.
    #[must_use]
    pub fn with_event_handlers(mut self, handlers: EventHandlers) -> Self {
        // Merge gate handlers
        fn merge_vec(dst: &mut Vec<PrioritizedHandler>, src: Vec<PrioritizedHandler>) {
            if !src.is_empty() {
                dst.extend(src);
                dst.sort_by(|a, b| b.priority.cmp(&a.priority));
            }
        }
        merge_vec(
            &mut self.gate_handlers.before_gate,
            handlers.gate_handlers.before_gate,
        );
        merge_vec(
            &mut self.gate_handlers.after_gate,
            handlers.gate_handlers.after_gate,
        );
        merge_vec(
            &mut self.gate_handlers.before_measurement,
            handlers.gate_handlers.before_measurement,
        );
        merge_vec(
            &mut self.gate_handlers.after_measurement,
            handlers.gate_handlers.after_measurement,
        );
        merge_vec(
            &mut self.gate_handlers.after_preparation,
            handlers.gate_handlers.after_preparation,
        );
        merge_vec(&mut self.gate_handlers.idle, handlers.gate_handlers.idle);

        // Merge signal handlers by TypeId
        for (type_id, src_handlers) in handlers.signal_handlers.handlers {
            if let Some((_, dst_handlers)) = self
                .signal_handlers
                .handlers
                .iter_mut()
                .find(|(id, _)| *id == type_id)
            {
                dst_handlers.extend(src_handlers);
            } else {
                self.signal_handlers.handlers.push((type_id, src_handlers));
            }
        }
        for (type_id, src_handlers) in handlers.signal_handlers.response_handlers {
            if let Some((_, dst_handlers)) = self
                .signal_handlers
                .response_handlers
                .iter_mut()
                .find(|(id, _)| *id == type_id)
            {
                dst_handlers.extend(src_handlers);
            } else {
                self.signal_handlers
                    .response_handlers
                    .push((type_id, src_handlers));
            }
        }

        self
    }

    /// Get gate definitions.
    #[must_use]
    pub fn definitions(&self) -> &GateDefinitions {
        &self.definitions
    }

    // ================================================================
    // Signal handler registration
    // ================================================================

    /// Register a handler that will be called when a signal of type `Sig` is dispatched.
    ///
    /// Handlers are called in registration order. Multiple handlers can be registered
    /// for the same signal type.
    pub fn on_signal<Sig: Signal>(
        &mut self,
        handler: impl Fn(&Sig) + Send + Sync + 'static,
    ) -> &mut Self {
        let erased: ErasedHandler = Arc::new(move |data: &dyn Any| {
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
    /// [`DispatchContext`] and returns a [`NoiseResponse`] that is applied to the simulation.
    pub fn on_signal_with_response<Sig: Signal>(
        &mut self,
        handler: impl Fn(&Sig, &DispatchContext<'_>) -> NoiseResponse + Send + Sync + 'static,
    ) -> &mut Self {
        let erased: ErasedResponseHandler =
            Arc::new(move |data: &dyn Any, ctx: &DispatchContext<'_>| {
                if let Some(signal) = data.downcast_ref::<Sig>() {
                    handler(signal, ctx)
                } else {
                    NoiseResponse::None
                }
            });
        self.signal_handlers
            .add_response(TypeId::of::<Sig>(), erased);
        self
    }

    // ================================================================
    // Gate event handler registration
    // ================================================================

    /// Register a handler called before each gate is applied.
    ///
    /// Handlers run in priority order (higher priority first, default 0).
    /// Return `NoiseResponse::SkipGate` to prevent the gate from executing.
    pub fn on_before_gate(
        &mut self,
        handler: impl Fn(&DispatchContext<'_>) -> NoiseResponse + Send + Sync + 'static,
    ) -> &mut Self {
        GateEventHandlers::insert(&mut self.gate_handlers.before_gate, Arc::new(handler), 0);
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
            Arc::new(handler),
            priority,
        );
        self
    }

    /// Register a handler called after each gate is applied.
    pub fn on_after_gate(
        &mut self,
        handler: impl Fn(&DispatchContext<'_>) -> NoiseResponse + Send + Sync + 'static,
    ) -> &mut Self {
        GateEventHandlers::insert(&mut self.gate_handlers.after_gate, Arc::new(handler), 0);
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
            Arc::new(handler),
            priority,
        );
        self
    }

    /// Register a handler called before each measurement.
    pub fn on_before_measurement(
        &mut self,
        handler: impl Fn(&DispatchContext<'_>) -> NoiseResponse + Send + Sync + 'static,
    ) -> &mut Self {
        GateEventHandlers::insert(
            &mut self.gate_handlers.before_measurement,
            Arc::new(handler),
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
            Arc::new(handler),
            priority,
        );
        self
    }

    /// Register a handler called after each measurement.
    ///
    /// The `DispatchContext::outcomes` field is populated with measurement results.
    pub fn on_after_measurement(
        &mut self,
        handler: impl Fn(&DispatchContext<'_>) -> NoiseResponse + Send + Sync + 'static,
    ) -> &mut Self {
        GateEventHandlers::insert(
            &mut self.gate_handlers.after_measurement,
            Arc::new(handler),
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
            Arc::new(handler),
            priority,
        );
        self
    }

    /// Register a handler called after each preparation.
    pub fn on_after_preparation(
        &mut self,
        handler: impl Fn(&DispatchContext<'_>) -> NoiseResponse + Send + Sync + 'static,
    ) -> &mut Self {
        GateEventHandlers::insert(
            &mut self.gate_handlers.after_preparation,
            Arc::new(handler),
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
            Arc::new(handler),
            priority,
        );
        self
    }

    /// Register a handler called during idle time.
    pub fn on_idle(
        &mut self,
        handler: impl Fn(&DispatchContext<'_>) -> NoiseResponse + Send + Sync + 'static,
    ) -> &mut Self {
        GateEventHandlers::insert(&mut self.gate_handlers.idle, Arc::new(handler), 0);
        self
    }

    /// Register an idle handler with explicit priority.
    pub fn on_idle_with_priority(
        &mut self,
        priority: i32,
        handler: impl Fn(&DispatchContext<'_>) -> NoiseResponse + Send + Sync + 'static,
    ) -> &mut Self {
        GateEventHandlers::insert(&mut self.gate_handlers.idle, Arc::new(handler), priority);
        self
    }

    // ================================================================
    // CommandQueue execution
    // ================================================================

    /// Apply a circuit to the given simulator state, returning measurement outcomes.
    ///
    /// Clears the internal outcomes buffer, executes all commands (with noise/handlers/signals),
    /// resets the noise model, and returns the accumulated outcomes.
    ///
    /// The caller is responsible for resetting the simulator state if needed
    /// (e.g., `state.reset()` before calling this method).
    pub fn apply_circuit(
        &mut self,
        state: &mut S,
        commands: &CommandQueue,
    ) -> Result<MeasurementOutcomes, ExecutionError> {
        self.outcomes.clear();

        if commands.has_signals() {
            self.execute_queue_with_signals(state, commands)?;
        } else {
            for command in commands {
                self.execute_queue_command(state, command)?;
            }
        }

        let outcomes = std::mem::take(&mut self.outcomes);

        // Reset noise model state for next shot
        if let Some(ref mut noise) = self.noise {
            noise.reset();
        }

        Ok(outcomes)
    }

    // ================================================================
    // Outcome access
    // ================================================================

    /// Take the accumulated measurement outcomes, leaving the buffer empty.
    ///
    /// Useful after a sequence of `apply_gate` calls to retrieve results.
    pub fn take_outcomes(&mut self) -> MeasurementOutcomes {
        std::mem::take(&mut self.outcomes)
    }

    /// Clear accumulated outcomes without returning them.
    pub fn clear_outcomes(&mut self) {
        self.outcomes.clear();
    }

    /// Reset the runner's execution state for a new shot.
    ///
    /// Clears accumulated outcomes and resets the noise model's internal context
    /// (leakage tracking, qubit history, etc.). Call this between shots when using
    /// `apply_gate` in interpreter mode.
    ///
    /// Note: This does **not** reset the simulator state or the RNG.
    /// The caller should reset the simulator separately if needed.
    pub fn reset(&mut self) {
        self.outcomes.clear();
        if let Some(ref mut noise) = self.noise {
            noise.reset();
        }
    }

    // ================================================================
    // Single gate execution (interpreter mode)
    // ================================================================

    /// Execute a single gate through the full pipeline.
    ///
    /// Runs before handlers -> noise -> gate execution -> noise -> after handlers.
    /// Measurement outcomes accumulate in the internal buffer; retrieve them
    /// via [`take_outcomes`](CircuitRunner::take_outcomes).
    pub fn apply_gate(
        &mut self,
        state: &mut S,
        gate_type: GateType,
        qubits: &[QubitId],
        angles: &[Angle64],
    ) -> Result<(), ExecutionError> {
        let command = GateCommand::with_angles(gate_type, qubits.to_vec(), angles.to_vec());
        self.execute_queue_command(state, &command)
    }

    // ================================================================
    // Direct noise injection
    // ================================================================

    /// Apply a noise event directly to the simulator state.
    ///
    /// Emits the event to the noise model, applies the response to state,
    /// and returns the response. Useful for idle noise between manually-applied
    /// gates, testing noise models, or custom execution loops.
    pub fn apply_noise(&mut self, state: &mut S, event: NoiseEvent<'_>) -> NoiseResponse {
        let Some(ref mut noise) = self.noise else {
            return NoiseResponse::None;
        };

        let response = noise.emit(event, &mut self.rng);
        self.apply_noise_response(state, response.clone());
        response
    }

    /// Execute a single command from a `CommandQueue`.
    fn execute_queue_command(
        &mut self,
        sim: &mut S,
        command: &GateCommand,
    ) -> Result<(), ExecutionError> {
        let qubits = command.qubits.as_slice();

        match command.gate_type {
            // Preparation
            GateType::PZ | GateType::QAlloc => {
                sim.pz(qubits);
                self.dispatch_after_preparation(sim, command);
            }

            // Measurement
            GateType::MZ | GateType::MeasureLeaked | GateType::MeasureFree => {
                self.dispatch_before_measurement(sim, command);
                let results = sim.mz(qubits);
                let outcomes: SmallVec<[bool; 4]> = results.iter().map(|r| r.outcome).collect();
                self.record_measurements(command.gate_type, qubits, &results);
                self.dispatch_after_measurement(sim, command, outcomes.as_slice());
            }

            // Idle
            GateType::Idle => {
                if let Some(duration) = command.get_idle_duration() {
                    self.dispatch_idle(sim, command, duration);
                }
            }

            // All other gates: convert to GateId and use unified chain
            _ => {
                let gate_id = command.gate_type.to_gate_id();
                let skip = self.dispatch_before_gate_for_id(
                    sim,
                    gate_id,
                    command.gate_type,
                    qubits,
                    command.angles.as_slice(),
                );
                if skip {
                    // Still emit after-gate for channels that want to inject errors
                    self.dispatch_after_gate_for_id(
                        sim,
                        gate_id,
                        command.gate_type,
                        qubits,
                        command.angles.as_slice(),
                    );
                    return Ok(());
                }

                // Execute through unified precedence chain
                let executed =
                    self.try_execute_override(sim, gate_id, qubits, command.angles.as_slice())
                        || self.try_execute_clifford(sim, gate_id, qubits)
                        || self.rotation_executor.is_some_and(|executor| {
                            executor(sim, gate_id, command.angles.as_slice(), qubits)
                        });

                if !executed {
                    self.execute_via_decomposition(
                        sim,
                        gate_id,
                        qubits,
                        command.angles.as_slice(),
                        0,
                    )?;
                }

                self.dispatch_after_gate_for_id(
                    sim,
                    gate_id,
                    command.gate_type,
                    qubits,
                    command.angles.as_slice(),
                );
            }
        }

        Ok(())
    }

    /// Execute commands with interleaved signal dispatch.
    fn execute_queue_with_signals(
        &mut self,
        sim: &mut S,
        commands: &CommandQueue,
    ) -> Result<(), ExecutionError> {
        let store = commands.signals();
        let mut cursors: SmallVec<[SignalCursor; 4]> =
            SmallVec::with_capacity(store.channel_count());
        for ch_idx in 0..store.channel_count() {
            let (type_id, channel) = store
                .channel_at(ch_idx)
                .expect("ch_idx is within 0..channel_count()");
            cursors.push(SignalCursor {
                type_id,
                channel_idx: ch_idx,
                entry_idx: 0,
                len: channel.len(),
            });
        }

        for (gate_idx, command) in commands.iter().enumerate() {
            self.dispatch_signals_at(sim, gate_idx as u32, store, &mut cursors);
            self.execute_queue_command(sim, command)?;
        }
        // Dispatch trailing signals (positioned after the last gate)
        self.dispatch_signals_at(sim, commands.len() as u32, store, &mut cursors);
        Ok(())
    }

    // ================================================================
    // AdaptedSequence execution
    // ================================================================

    /// Apply an `AdaptedSequence` circuit to the given simulator state.
    ///
    /// Clears the internal outcomes buffer, executes all operations (with noise/handlers),
    /// resets the noise model, and returns the accumulated outcomes.
    ///
    /// This path uses `GateId` natively and supports conditional execution,
    /// multi-basis prep/measure, and result tracking.
    pub fn apply_adapted_circuit(
        &mut self,
        state: &mut S,
        circuit: &AdaptedSequence,
    ) -> Result<MeasurementOutcomes, ExecutionError> {
        self.outcomes.clear();
        self.results.clear();
        self.results.resize(circuit.result_count, false);

        self.execute_ops(state, &circuit.ops, 0)?;

        let outcomes = std::mem::take(&mut self.outcomes);

        // Reset noise model state for next shot
        if let Some(ref mut noise) = self.noise {
            noise.reset();
        }

        Ok(outcomes)
    }

    /// Execute a list of operations.
    fn execute_ops(
        &mut self,
        sim: &mut S,
        ops: &[AdaptedOp],
        depth: usize,
    ) -> Result<(), ExecutionError> {
        if depth > self.max_decomp_depth {
            return Err(ExecutionError::MaxDecompositionDepthExceeded);
        }

        for op in ops {
            self.execute_op(sim, op, depth)?;
        }
        Ok(())
    }

    /// Execute a single operation from an `AdaptedSequence`.
    fn execute_op(
        &mut self,
        sim: &mut S,
        op: &AdaptedOp,
        depth: usize,
    ) -> Result<(), ExecutionError> {
        match op {
            AdaptedOp::Gate {
                gate_id,
                qubits,
                angles,
            } => {
                self.execute_gate(sim, *gate_id, qubits, angles, depth)?;
            }
            AdaptedOp::Prep { qubit, basis } => {
                self.execute_prep(sim, *qubit, *basis);
            }
            AdaptedOp::Measure {
                qubit,
                basis,
                result,
            } => {
                self.execute_measure(sim, *qubit, *basis, *result);
            }
            AdaptedOp::Conditional {
                condition,
                if_one,
                if_zero,
            } => {
                let result_val = self
                    .results
                    .get(condition.0 as usize)
                    .copied()
                    .unwrap_or(false);
                if result_val {
                    self.execute_ops(sim, if_one, depth)?;
                } else {
                    self.execute_ops(sim, if_zero, depth)?;
                }
            }
            AdaptedOp::XorResult { target, source } => {
                let src_val = self
                    .results
                    .get(source.0 as usize)
                    .copied()
                    .unwrap_or(false);
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

    // ================================================================
    // Core gate execution (unified precedence chain)
    // ================================================================

    /// Execute a gate using the unified precedence chain:
    /// overrides -> Clifford -> rotation -> decomposition.
    ///
    /// Before and after the gate, noise events and user handlers are dispatched.
    fn execute_gate(
        &mut self,
        sim: &mut S,
        gate_id: GateId,
        qubits: &[QubitId],
        angles: &[Angle64],
        depth: usize,
    ) -> Result<(), ExecutionError> {
        // Emit before-gate noise event
        let skip = self.emit_before_gate(sim, gate_id, qubits, angles);
        if skip {
            return Ok(());
        }

        // Try execution in order of precedence
        let executed = self.try_execute_override(sim, gate_id, qubits, angles)
            || self.try_execute_clifford(sim, gate_id, qubits)
            || self
                .rotation_executor
                .is_some_and(|executor| executor(sim, gate_id, angles, qubits));

        if !executed {
            self.execute_via_decomposition(sim, gate_id, qubits, angles, depth)?;
        }

        // Emit after-gate noise event
        self.emit_after_gate(sim, gate_id, qubits, angles);

        Ok(())
    }

    /// Try to execute a gate via registered overrides.
    fn try_execute_override(
        &mut self,
        sim: &mut S,
        gate_id: GateId,
        qubits: &[QubitId],
        angles: &[Angle64],
    ) -> bool {
        self.overrides
            .as_ref()
            .and_then(|o| o.get(gate_id))
            .is_some_and(|executor| executor(sim, angles, qubits))
    }

    /// Try to execute a Clifford gate natively via trait methods.
    fn try_execute_clifford(&mut self, sim: &mut S, gate_id: GateId, qubits: &[QubitId]) -> bool {
        let Some(gate_type) = gate_id.try_to_gate_type() else {
            return false;
        };

        match gate_type {
            GateType::I | GateType::Idle => true,
            GateType::X => {
                sim.x(qubits);
                true
            }
            GateType::Y => {
                sim.y(qubits);
                true
            }
            GateType::Z => {
                sim.z(qubits);
                true
            }
            GateType::H => {
                sim.h(qubits);
                true
            }
            GateType::SX => {
                sim.sx(qubits);
                true
            }
            GateType::SXdg => {
                sim.sxdg(qubits);
                true
            }
            GateType::SY => {
                sim.sy(qubits);
                true
            }
            GateType::SYdg => {
                sim.sydg(qubits);
                true
            }
            GateType::SZ => {
                sim.sz(qubits);
                true
            }
            GateType::SZdg => {
                sim.szdg(qubits);
                true
            }
            GateType::CX => {
                let pairs = flat_to_pairs(qubits);
                sim.cx(&pairs);
                true
            }
            GateType::CY => {
                let pairs = flat_to_pairs(qubits);
                sim.cy(&pairs);
                true
            }
            GateType::CZ => {
                let pairs = flat_to_pairs(qubits);
                sim.cz(&pairs);
                true
            }
            GateType::SZZ => {
                let pairs = flat_to_pairs(qubits);
                sim.szz(&pairs);
                true
            }
            GateType::SZZdg => {
                let pairs = flat_to_pairs(qubits);
                sim.szzdg(&pairs);
                true
            }
            GateType::SWAP => {
                let pairs = flat_to_pairs(qubits);
                sim.swap(&pairs);
                true
            }
            _ => false,
        }
    }

    /// Execute a gate via decomposition from `GateDefinitions`.
    fn execute_via_decomposition(
        &mut self,
        sim: &mut S,
        gate_id: GateId,
        qubits: &[QubitId],
        angles: &[Angle64],
        depth: usize,
    ) -> Result<(), ExecutionError> {
        let Some(decomp_entry) = self.definitions.decomposition(gate_id) else {
            return Err(ExecutionError::NoDecomposition { gate_id });
        };

        let instantiated_ops: Vec<_> = decomp_entry
            .decomposition
            .expand()
            .map(|op| op.instantiate(qubits, angles))
            .collect();

        for inst in instantiated_ops {
            self.execute_gate(sim, inst.gate, &inst.qubits, &inst.angles, depth + 1)?;
        }

        Ok(())
    }

    // ================================================================
    // Multi-basis prep/measure (for AdaptedSequence)
    // ================================================================

    /// Execute preparation in a given basis.
    fn execute_prep(&mut self, sim: &mut S, qubit: QubitId, basis: PrepBasis) {
        sim.pz(&[qubit]);
        match basis {
            PrepBasis::Z => {}
            PrepBasis::X => {
                sim.h(&[qubit]);
            }
            PrepBasis::Y => {
                sim.h(&[qubit]);
                sim.sz(&[qubit]);
            }
        }
    }

    /// Execute measurement in a given basis with result tracking.
    fn execute_measure(
        &mut self,
        sim: &mut S,
        qubit: QubitId,
        basis: MeasBasis,
        result_id: ResultId,
    ) {
        // Rotate to Z basis
        match basis {
            MeasBasis::Z => {}
            MeasBasis::X => {
                sim.h(&[qubit]);
            }
            MeasBasis::Y => {
                sim.szdg(&[qubit]);
                sim.h(&[qubit]);
            }
        }

        // Perform measurement
        let results = sim.mz(&[qubit]);
        let meas_result = results.first();
        let outcome = meas_result.is_some_and(|r| r.outcome);
        let is_deterministic = meas_result.is_none_or(|r| r.is_deterministic);

        // Store result for conditionals
        if let Some(slot) = self.results.get_mut(result_id.0 as usize) {
            *slot = outcome;
        }

        // Record in outcomes
        self.outcomes
            .record(MeasurementOutcome::new(qubit, outcome, is_deterministic));

        // Rotate back (non-destructive measurement semantics)
        match basis {
            MeasBasis::Z => {}
            MeasBasis::X => {
                sim.h(&[qubit]);
            }
            MeasBasis::Y => {
                sim.h(&[qubit]);
                sim.sz(&[qubit]);
            }
        }
    }

    // ================================================================
    // Dispatch coordination (user handlers + noise model)
    // ================================================================

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

    /// Dispatch before-gate event for a gate identified by `GateId` (used by both paths).
    /// Returns `true` if the gate should be skipped.
    fn dispatch_before_gate_for_id(
        &mut self,
        sim: &mut S,
        gate_id: GateId,
        gate_type: GateType,
        qubits: &[QubitId],
        angles: &[Angle64],
    ) -> bool {
        // Fast path: no handlers registered, go directly to noise model
        if self.gate_handlers.before_gate.is_empty() {
            return self.emit_before_gate_noise_for_id(sim, gate_id, gate_type, qubits, angles);
        }

        // 1. User before-gate handlers
        let ctx = DispatchContext {
            gate_type,
            qubits,
            angles,
            gate_id: Some(gate_id),
            outcomes: None,
            duration: None,
            noise_context: self.noise.as_ref().map(ComposableNoiseModel::context),
        };
        let user_response = GateEventHandlers::dispatch(&self.gate_handlers.before_gate, &ctx);

        // 2. Noise model BeforeGate
        let noise_response =
            self.emit_before_gate_noise_raw_for_id(gate_id, gate_type, qubits, angles);

        // 3. Combine
        let combined = user_response.combine(noise_response);
        let should_skip = combined.should_skip_gate();
        self.apply_noise_response(sim, combined);
        should_skip
    }

    /// Dispatch after-gate event for a gate identified by `GateId`.
    fn dispatch_after_gate_for_id(
        &mut self,
        sim: &mut S,
        gate_id: GateId,
        gate_type: GateType,
        qubits: &[QubitId],
        angles: &[Angle64],
    ) {
        // Fast path
        if self.gate_handlers.after_gate.is_empty() {
            self.emit_after_gate_noise_for_id(sim, gate_id, gate_type, qubits, angles);
            return;
        }

        // 1. Noise model AfterGate
        let noise_response =
            self.emit_after_gate_noise_raw_for_id(gate_id, gate_type, qubits, angles);

        // 2. User after-gate handlers
        let ctx = DispatchContext {
            gate_type,
            qubits,
            angles,
            gate_id: Some(gate_id),
            outcomes: None,
            duration: None,
            noise_context: self.noise.as_ref().map(ComposableNoiseModel::context),
        };
        let user_response = GateEventHandlers::dispatch(&self.gate_handlers.after_gate, &ctx);

        // 3. Combine and apply
        let combined = noise_response.combine(user_response);
        self.apply_noise_response(sim, combined);
    }

    /// Dispatch before-measurement event.
    fn dispatch_before_measurement(&mut self, sim: &mut S, command: &GateCommand) {
        let qubits = command.qubits.as_slice();

        // Fast path
        if self.gate_handlers.before_measurement.is_empty() {
            self.emit_before_measurement_noise(sim, qubits);
            return;
        }

        let ctx = self.gate_context(command);
        let user_response =
            GateEventHandlers::dispatch(&self.gate_handlers.before_measurement, &ctx);
        let noise_response = self.emit_before_measurement_noise_raw(qubits);
        let combined = user_response.combine(noise_response);
        self.apply_noise_response(sim, combined);
    }

    /// Dispatch after-measurement event.
    fn dispatch_after_measurement(
        &mut self,
        sim: &mut S,
        command: &GateCommand,
        outcomes: &[bool],
    ) {
        let qubits = command.qubits.as_slice();

        // Fast path
        if self.gate_handlers.after_measurement.is_empty() {
            self.emit_after_measurement_noise(sim, qubits, outcomes);
            return;
        }

        let noise_response = self.emit_after_measurement_noise_raw(qubits, outcomes);
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
        let combined = noise_response.combine(user_response);
        self.apply_noise_response(sim, combined);
    }

    /// Dispatch after-preparation event.
    fn dispatch_after_preparation(&mut self, sim: &mut S, command: &GateCommand) {
        let qubits = command.qubits.as_slice();

        // Fast path
        if self.gate_handlers.after_preparation.is_empty() {
            self.emit_after_preparation_noise(sim, qubits);
            return;
        }

        let noise_response = self.emit_after_preparation_noise_raw(qubits);
        let ctx = self.gate_context(command);
        let user_response =
            GateEventHandlers::dispatch(&self.gate_handlers.after_preparation, &ctx);
        let combined = noise_response.combine(user_response);
        self.apply_noise_response(sim, combined);
    }

    /// Dispatch idle event.
    fn dispatch_idle(&mut self, sim: &mut S, command: &GateCommand, duration: TimeUnits) {
        let qubits = command.qubits.as_slice();

        // Fast path
        if self.gate_handlers.idle.is_empty() {
            self.emit_idle_noise(sim, qubits, duration);
            return;
        }

        let noise_response = self.emit_idle_noise_raw(qubits, duration);
        let ctx = DispatchContext {
            gate_type: command.gate_type,
            qubits,
            angles: command.angles.as_slice(),
            gate_id: None,
            outcomes: None,
            duration: Some(duration),
            noise_context: self.noise.as_ref().map(ComposableNoiseModel::context),
        };
        let user_response = GateEventHandlers::dispatch(&self.gate_handlers.idle, &ctx);
        let combined = noise_response.combine(user_response);
        self.apply_noise_response(sim, combined);
    }

    // ================================================================
    // Noise emission methods (for AdaptedSequence path via execute_gate)
    // ================================================================

    /// Emit before-gate to noise model (`AdaptedSequence` path). Returns `true` if gate should be skipped.
    fn emit_before_gate(
        &mut self,
        sim: &mut S,
        gate_id: GateId,
        qubits: &[QubitId],
        angles: &[Angle64],
    ) -> bool {
        let Some(ref mut noise) = self.noise else {
            return false;
        };

        let gate_type = gate_id.try_to_gate_type().unwrap_or(GateType::I);
        let event = NoiseEvent::BeforeGate {
            gate_type,
            qubits,
            angles,
            gate_id: Some(gate_id),
        };

        let response = noise.emit(event, &mut self.rng);
        let should_skip = response.should_skip_gate();
        self.apply_noise_response(sim, response);
        should_skip
    }

    /// Emit after-gate to noise model (`AdaptedSequence` path).
    fn emit_after_gate(
        &mut self,
        sim: &mut S,
        gate_id: GateId,
        qubits: &[QubitId],
        angles: &[Angle64],
    ) {
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
        self.apply_noise_response(sim, response);
    }

    // ================================================================
    // Noise emission methods (for CommandQueue path)
    // ================================================================

    /// Emit before-gate noise for a gate identified by `GateId`. Returns `true` if gate should be skipped.
    fn emit_before_gate_noise_for_id(
        &mut self,
        sim: &mut S,
        gate_id: GateId,
        gate_type: GateType,
        qubits: &[QubitId],
        angles: &[Angle64],
    ) -> bool {
        let response = self.emit_before_gate_noise_raw_for_id(gate_id, gate_type, qubits, angles);
        let should_skip = response.should_skip_gate();
        self.apply_noise_response(sim, response);
        should_skip
    }

    fn emit_before_gate_noise_raw_for_id(
        &mut self,
        gate_id: GateId,
        gate_type: GateType,
        qubits: &[QubitId],
        angles: &[Angle64],
    ) -> NoiseResponse {
        if let Some(ref mut noise) = self.noise {
            let event = NoiseEvent::BeforeGate {
                gate_type,
                qubits,
                angles,
                gate_id: Some(gate_id),
            };
            return noise.emit(event, &mut self.rng);
        }
        NoiseResponse::None
    }

    fn emit_after_gate_noise_for_id(
        &mut self,
        sim: &mut S,
        gate_id: GateId,
        gate_type: GateType,
        qubits: &[QubitId],
        angles: &[Angle64],
    ) {
        let response = self.emit_after_gate_noise_raw_for_id(gate_id, gate_type, qubits, angles);
        self.apply_noise_response(sim, response);
    }

    fn emit_after_gate_noise_raw_for_id(
        &mut self,
        gate_id: GateId,
        gate_type: GateType,
        qubits: &[QubitId],
        angles: &[Angle64],
    ) -> NoiseResponse {
        if let Some(ref mut noise) = self.noise {
            let event = NoiseEvent::AfterGate {
                gate_type,
                qubits,
                angles,
                gate_id: Some(gate_id),
            };
            return noise.emit(event, &mut self.rng);
        }
        NoiseResponse::None
    }

    fn emit_before_measurement_noise(&mut self, sim: &mut S, qubits: &[QubitId]) {
        let response = self.emit_before_measurement_noise_raw(qubits);
        self.apply_noise_response(sim, response);
    }

    fn emit_before_measurement_noise_raw(&mut self, qubits: &[QubitId]) -> NoiseResponse {
        if let Some(ref mut noise) = self.noise {
            let event = NoiseEvent::BeforeMeasurement { qubits };
            return noise.emit(event, &mut self.rng);
        }
        NoiseResponse::None
    }

    fn emit_after_measurement_noise(&mut self, sim: &mut S, qubits: &[QubitId], outcomes: &[bool]) {
        let response = self.emit_after_measurement_noise_raw(qubits, outcomes);
        self.apply_noise_response(sim, response);
    }

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

    fn emit_after_preparation_noise(&mut self, sim: &mut S, qubits: &[QubitId]) {
        let response = self.emit_after_preparation_noise_raw(qubits);
        self.apply_noise_response(sim, response);
    }

    fn emit_after_preparation_noise_raw(&mut self, qubits: &[QubitId]) -> NoiseResponse {
        if let Some(ref mut noise) = self.noise {
            let event = NoiseEvent::AfterPreparation { qubits };
            return noise.emit(event, &mut self.rng);
        }
        NoiseResponse::None
    }

    fn emit_idle_noise(&mut self, sim: &mut S, qubits: &[QubitId], duration: TimeUnits) {
        let response = self.emit_idle_noise_raw(qubits, duration);
        self.apply_noise_response(sim, response);
    }

    fn emit_idle_noise_raw(&mut self, qubits: &[QubitId], duration: TimeUnits) -> NoiseResponse {
        if let Some(ref mut noise) = self.noise {
            let event = NoiseEvent::IdleTime { qubits, duration };
            return noise.emit(event, &mut self.rng);
        }
        NoiseResponse::None
    }

    // ================================================================
    // Signal dispatch
    // ================================================================

    /// Dispatch all signals at a given command position.
    fn dispatch_signals_at(
        &mut self,
        sim: &mut S,
        pos: u32,
        store: &SignalStore,
        cursors: &mut [SignalCursor],
    ) {
        let has_response_handlers = self.signal_handlers.has_response_handlers();

        for cursor in cursors.iter_mut() {
            let Some((_, channel)) = store.channel_at(cursor.channel_idx) else {
                continue;
            };
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
                    let response = self
                        .signal_handlers
                        .call_response(cursor.type_id, data, &ctx);
                    if !response.is_none() {
                        self.apply_noise_response(sim, response);
                    }
                }

                // 3. Emit to noise model
                self.emit_signal_to_noise(sim, cursor.type_id, data);

                cursor.entry_idx += 1;
            }
        }
    }

    /// Emit a signal event to the noise model.
    fn emit_signal_to_noise(&mut self, sim: &mut S, type_id: TypeId, data: &dyn Any) {
        if let Some(ref mut noise) = self.noise {
            let event = NoiseEvent::Signal { type_id, data };
            let response = noise.emit(event, &mut self.rng);
            self.apply_noise_response(sim, response);
        }
    }

    // ================================================================
    // Measurement recording and noise response application
    // ================================================================

    /// Record measurement results with leakage awareness.
    fn record_measurements(
        &mut self,
        gate_type: GateType,
        qubits: &[QubitId],
        results: &[pecos_simulators::MeasurementResult],
    ) {
        for (&qubit, result) in qubits.iter().zip(results.iter()) {
            let is_leaked = self
                .noise
                .as_ref()
                .is_some_and(|n| n.context().is_leaked(qubit));

            let outcome = if is_leaked {
                if gate_type == GateType::MeasureLeaked {
                    MeasurementOutcome::leaked(qubit)
                } else {
                    MeasurementOutcome {
                        qubit,
                        outcome: true,
                        is_deterministic: true,
                        is_leaked: true,
                    }
                }
            } else {
                MeasurementOutcome::new(qubit, result.outcome, result.is_deterministic)
            };

            self.outcomes.record(outcome);
        }
    }

    /// Apply a noise response (inject gates, flip outcomes, etc.).
    fn apply_noise_response(&mut self, sim: &mut S, response: NoiseResponse) {
        match response {
            NoiseResponse::None
            | NoiseResponse::SkipGate
            | NoiseResponse::MarkLeaked(_)
            | NoiseResponse::MarkUnleaked(_) => {}

            NoiseResponse::InjectGates(gates) => {
                for gate in gates.iter() {
                    Self::execute_noise_gate(sim, gate);
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
                    self.apply_noise_response(sim, r);
                }
            }
        }
    }

    /// Execute a noise gate (injected Pauli error).
    fn execute_noise_gate(sim: &mut S, gate: &GateCommand) {
        let qubits = gate.qubits.as_slice();
        match gate.gate_type {
            GateType::X => {
                sim.x(qubits);
            }
            GateType::Y => {
                sim.y(qubits);
            }
            GateType::Z => {
                sim.z(qubits);
            }
            _ => {}
        }
    }
}

// ============================================================================
// RNG management (for simulators with RngManageable)
// ============================================================================

impl<S> CircuitRunner<S>
where
    S: CliffordGateable + RngManageable<Rng = PecosRng>,
{
    /// Set the seed for full determinism.
    ///
    /// Seeds both the noise RNG and the simulator's internal RNG using
    /// derived seeds from a single base seed.
    #[must_use]
    pub fn with_full_seed(mut self, state: &mut S, seed: u64) -> Self {
        let noise_seed = derive_seed(seed, "noise");
        let sim_seed = derive_seed(seed, "simulator");
        self.rng = PecosRng::seed_from_u64(noise_seed);
        state.set_seed(sim_seed);
        self
    }

    /// Set full seed (mutable version).
    ///
    /// Seeds both the noise RNG and the simulator's internal RNG using
    /// derived seeds from a single base seed.
    pub fn set_full_seed(&mut self, state: &mut S, seed: u64) {
        let noise_seed = derive_seed(seed, "noise");
        let sim_seed = derive_seed(seed, "simulator");
        self.rng = PecosRng::seed_from_u64(noise_seed);
        state.set_seed(sim_seed);
    }
}

// ============================================================================
// Rotation gate support (for ArbitraryRotationGateable simulators)
// ============================================================================

impl<S> CircuitRunner<S>
where
    S: CliffordGateable + ArbitraryRotationGateable,
{
    /// Create a runner with rotation gate support and default definitions.
    ///
    /// For simulators implementing `ArbitraryRotationGateable`, this constructor
    /// enables native execution of rotation gates (T, Tdg, RX, RY, RZ, etc.).
    #[must_use]
    pub fn rotations() -> Self {
        Self::rotations_with_definitions(GateDefinitions::new())
    }

    /// Create a runner with rotation gate support and explicit definitions.
    pub fn rotations_with_definitions(definitions: GateDefinitions) -> Self {
        let mut runner = Self::with_definitions(definitions);
        runner.rotation_executor = Some(Self::execute_rotation_gate);
        runner
    }

    /// Execute a rotation gate natively.
    fn execute_rotation_gate(
        sim: &mut S,
        gate_id: GateId,
        angles: &[Angle64],
        qubits: &[QubitId],
    ) -> bool {
        let Some(gate_type) = gate_id.try_to_gate_type() else {
            return false;
        };

        match gate_type {
            GateType::T => {
                sim.t(qubits);
                true
            }
            GateType::Tdg => {
                sim.tdg(qubits);
                true
            }
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
            GateType::U => {
                let angle = angles.first().copied().unwrap_or(Angle64::ZERO);
                let angle2 = angles.get(1).copied().unwrap_or(Angle64::ZERO);
                let angle3 = angles.get(2).copied().unwrap_or(Angle64::ZERO);
                sim.u(angle, angle2, angle3, qubits);
                true
            }
            GateType::R1XY => {
                let angle = angles.first().copied().unwrap_or(Angle64::ZERO);
                let angle2 = angles.get(1).copied().unwrap_or(Angle64::ZERO);
                sim.r1xy(angle, angle2, qubits);
                true
            }
            GateType::RXX => {
                if let Some(&angle) = angles.first() {
                    let pairs = flat_to_pairs(qubits);
                    sim.rxx(angle, &pairs);
                    true
                } else {
                    false
                }
            }
            GateType::RYY => {
                if let Some(&angle) = angles.first() {
                    let pairs = flat_to_pairs(qubits);
                    sim.ryy(angle, &pairs);
                    true
                } else {
                    false
                }
            }
            GateType::RZZ => {
                if let Some(&angle) = angles.first() {
                    let pairs = flat_to_pairs(qubits);
                    sim.rzz(angle, &pairs);
                    true
                } else {
                    false
                }
            }
            // CRZ decomposition: RZ(theta/2), CX, RZ(-theta/2), CX
            GateType::CRZ => {
                if qubits.len() >= 2 {
                    let control = qubits[0];
                    let target = qubits[1];
                    let angle = angles.first().copied().unwrap_or(Angle64::ZERO);
                    let half_angle = angle / 2u64;
                    sim.rz(half_angle, &[target]);
                    sim.cx(&[(control, target)]);
                    sim.rz(-half_angle, &[target]);
                    sim.cx(&[(control, target)]);
                    true
                } else {
                    false
                }
            }
            // CCX (Toffoli) decomposition
            GateType::CCX => {
                if qubits.len() >= 3 {
                    let c1 = qubits[0];
                    let c2 = qubits[1];
                    let target = qubits[2];
                    sim.h(&[target]);
                    sim.cx(&[(c2, target)]);
                    sim.tdg(&[target]);
                    sim.cx(&[(c1, target)]);
                    sim.t(&[target]);
                    sim.cx(&[(c2, target)]);
                    sim.tdg(&[target]);
                    sim.cx(&[(c1, target)]);
                    sim.t(&[c2]);
                    sim.t(&[target]);
                    sim.h(&[target]);
                    sim.cx(&[(c1, c2)]);
                    sim.t(&[c1]);
                    sim.tdg(&[c2]);
                    sim.cx(&[(c1, c2)]);
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::CommandBuilder;
    use crate::extensible::{GateCategory, GateSpec, OpBuilder, gates};
    use crate::noise::single_qubit::SingleQubitChannel;
    use pecos_simulators::{SparseStab, StateVec};

    // ================================================================
    // Basic execution tests
    // ================================================================

    #[test]
    fn test_basic_execution() {
        let commands = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

        let mut state = SparseStab::new(1);
        let mut runner = CircuitRunner::<SparseStab>::new().with_seed(42);
        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();

        assert_eq!(outcomes.len(), 1);
    }

    #[test]
    fn test_bell_state() {
        let commands = CommandBuilder::new()
            .pz(&[0])
            .pz(&[1])
            .h(&[0])
            .cx(&[(0, 1)])
            .mz(&[0])
            .mz(&[1])
            .build();

        let mut state = SparseStab::new(2);
        let mut runner = CircuitRunner::<SparseStab>::new().with_seed(42);
        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();

        assert_eq!(outcomes.len(), 2);
        let o0 = outcomes.get_bit(QubitId(0)).unwrap();
        let o1 = outcomes.get_bit(QubitId(1)).unwrap();
        assert_eq!(o0, o1, "Bell state outcomes should be equal");
    }

    #[test]
    fn test_with_noise() {
        let commands = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

        let noise = ComposableNoiseModel::new().add_channel(SingleQubitChannel::depolarizing(0.0));

        let mut state = SparseStab::new(1);
        let mut runner = CircuitRunner::<SparseStab>::new()
            .with_noise(noise)
            .with_seed(42);

        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
        assert_eq!(outcomes.len(), 1);
    }

    #[test]
    fn test_apply_circuit_resets_noise() {
        let commands = CommandBuilder::new().pz(&[0]).mz(&[0]).build();

        let mut state = SparseStab::new(1);
        let mut runner = CircuitRunner::<SparseStab>::new().with_seed(42);

        let outcomes1 = runner.apply_circuit(&mut state, &commands).unwrap();
        let outcomes2 = runner.apply_circuit(&mut state, &commands).unwrap();

        assert_eq!(outcomes1.len(), 1);
        assert_eq!(outcomes2.len(), 1);
    }

    #[test]
    fn test_measure_leaked_on_leaked_qubit() {
        use crate::command::GateCommand;
        use crate::noise::leakage::LeakageChannel;

        let mut commands = CommandBuilder::new().pz(&[0]).build();
        commands.push(GateCommand::new(
            GateType::MeasureLeaked,
            smallvec::smallvec![QubitId(0)],
        ));

        let mut noise = ComposableNoiseModel::new().add_channel(LeakageChannel::new());
        noise.context_mut().mark_leaked(QubitId(0));

        let mut state = SparseStab::new(1);
        let mut runner = CircuitRunner::<SparseStab>::new()
            .with_noise(noise)
            .with_seed(42);
        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();

        let outcome = outcomes.get(QubitId(0)).unwrap();
        assert!(outcome.is_leaked);
        assert_eq!(outcome.as_int_leaked(), 2);
    }

    #[test]
    fn test_regular_measure_on_leaked_qubit() {
        use crate::noise::leakage::LeakageChannel;

        let commands = CommandBuilder::new().pz(&[0]).mz(&[0]).build();

        let mut noise = ComposableNoiseModel::new().add_channel(LeakageChannel::new());
        noise.context_mut().mark_leaked(QubitId(0));

        let mut state = SparseStab::new(1);
        let mut runner = CircuitRunner::<SparseStab>::new()
            .with_noise(noise)
            .with_seed(42);
        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();

        let outcome = outcomes.get(QubitId(0)).unwrap();
        assert!(outcome.outcome);
        assert!(outcome.is_leaked);
    }

    #[test]
    fn test_measure_leaked_on_non_leaked_qubit() {
        use crate::command::GateCommand;
        use crate::noise::leakage::LeakageChannel;

        let mut commands = CommandBuilder::new().pz(&[0]).build();
        commands.push(GateCommand::new(
            GateType::MeasureLeaked,
            smallvec::smallvec![QubitId(0)],
        ));

        let noise = ComposableNoiseModel::new().add_channel(LeakageChannel::new());

        let mut state = SparseStab::new(1);
        let mut runner = CircuitRunner::<SparseStab>::new()
            .with_noise(noise)
            .with_seed(42);
        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();

        let outcome = outcomes.get(QubitId(0)).unwrap();
        assert!(!outcome.is_leaked);
        assert!(!outcome.outcome);
    }

    // ================================================================
    // Rotation gate tests
    // ================================================================

    #[test]
    fn test_rotation_gates() {
        use crate::command::GateCommand;
        use pecos_core::Angle64;

        let mut commands = CommandBuilder::new().pz(&[0]).build();
        commands.push(GateCommand::with_angles(
            GateType::RX,
            smallvec::smallvec![QubitId(0)],
            smallvec::smallvec![Angle64::HALF_TURN],
        ));
        commands.push(GateCommand::new(
            GateType::MZ,
            smallvec::smallvec![QubitId(0)],
        ));

        let mut state = StateVec::new(1);
        let mut runner = CircuitRunner::<StateVec>::rotations().with_seed(42);
        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();

        let outcome = outcomes.get(QubitId(0)).unwrap();
        assert!(outcome.outcome, "RX(pi) on |0> should give |1>");
    }

    #[test]
    fn test_crz_decomposition() {
        use pecos_core::Angle64;

        let commands = CommandBuilder::new()
            .pz(&[0])
            .pz(&[1])
            .x(&[0])
            .h(&[1])
            .gate(GateCommand::with_angles(
                GateType::CRZ,
                smallvec::smallvec![QubitId(0), QubitId(1)],
                smallvec::smallvec![Angle64::HALF_TURN],
            ))
            .h(&[1])
            .mz(&[1])
            .build();

        let mut state = StateVec::new(2);
        let mut runner = CircuitRunner::<StateVec>::rotations().with_seed(42);
        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();

        let outcome = outcomes.get(QubitId(1)).unwrap();
        assert!(
            outcome.outcome,
            "CRZ(pi) with control=1 should flip target phase"
        );
    }

    #[test]
    fn test_ccx_decomposition() {
        let commands = CommandBuilder::new()
            .pz(&[0])
            .pz(&[1])
            .pz(&[2])
            .x(&[0])
            .x(&[1])
            .ccx(&[(0, 1, 2)])
            .mz(&[2])
            .build();

        let mut state = StateVec::new(3);
        let mut runner = CircuitRunner::<StateVec>::rotations().with_seed(42);
        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();

        let outcome = outcomes.get(QubitId(2)).unwrap();
        assert!(
            outcome.outcome,
            "CCX with both controls=1 should flip target"
        );
    }

    #[test]
    fn test_ccx_no_flip_when_control_zero() {
        let commands = CommandBuilder::new()
            .pz(&[0])
            .pz(&[1])
            .pz(&[2])
            .x(&[0])
            .ccx(&[(0, 1, 2)])
            .mz(&[2])
            .build();

        let mut state = StateVec::new(3);
        let mut runner = CircuitRunner::<StateVec>::rotations().with_seed(42);
        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();

        let outcome = outcomes.get(QubitId(2)).unwrap();
        assert!(
            !outcome.outcome,
            "CCX with one control=0 should not flip target"
        );
    }

    #[test]
    fn test_idle_time_emission() {
        use crate::noise::idle::IdleChannel;

        let commands = CommandBuilder::new()
            .pz(&[0])
            .idle(&[0], 100u64)
            .mz(&[0])
            .build();

        let noise = ComposableNoiseModel::new().add_channel(IdleChannel::linear(1.0));

        let mut state = SparseStab::new(1);
        let mut runner = CircuitRunner::<SparseStab>::new()
            .with_noise(noise)
            .with_seed(42);

        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
        assert_eq!(outcomes.len(), 1);
    }

    #[test]
    fn test_with_gate_definitions() {
        use crate::extensible::{GateCategory, GateDefinitions};
        use crate::noise::CategoryBasedChannel;

        let gates = GateDefinitions::builder()
            .with_category_noise(GateCategory::SingleQubitUnitary, 0.0)
            .with_category_noise(GateCategory::TwoQubitUnitary, 0.0)
            .build_or_panic();

        let noise = ComposableNoiseModel::new().add_channel(
            CategoryBasedChannel::new().with_category(GateCategory::SingleQubitUnitary, 0.0),
        );

        let commands = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

        let mut state = SparseStab::new(1);
        let mut runner = CircuitRunner::<SparseStab>::with_definitions(gates)
            .with_noise(noise)
            .with_seed(42);

        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
        assert_eq!(outcomes.len(), 1);
    }

    #[test]
    fn test_gate_definitions_propagated_to_noise() {
        use crate::extensible::{GateCategory, GateDefinitions};
        use crate::noise::CategoryBasedChannel;

        let mut gates = GateDefinitions::new();
        let custom_id = gates.register(
            GateSpec::new("CustomGate")
                .with_quantum_arity(1)
                .with_category(GateCategory::SingleQubitUnitary),
        );

        let noise = ComposableNoiseModel::new().add_channel(
            CategoryBasedChannel::new().with_category(GateCategory::SingleQubitUnitary, 0.0),
        );

        let runner = CircuitRunner::<SparseStab>::with_definitions(gates).with_noise(noise);

        assert_eq!(runner.definitions().name(custom_id), Some("CustomGate"));
    }

    // ================================================================
    // Signal dispatch tests
    // ================================================================

    #[allow(dead_code)]
    mod signal_tests {
        use super::*;
        use pecos_core::impl_signal;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU64, Ordering};

        #[derive(Copy, Clone, Debug)]
        struct RoundBoundary(pub i64);
        impl_signal!(RoundBoundary);

        #[derive(Copy, Clone, Debug)]
        struct Temperature(f64);
        impl_signal!(Temperature);

        #[test]
        fn test_signal_handler_called() {
            let counter = Arc::new(AtomicU64::new(0));
            let counter_clone = counter.clone();

            let commands = CommandBuilder::new()
                .pz(&[0])
                .signal(RoundBoundary(1))
                .h(&[0])
                .signal(RoundBoundary(2))
                .mz(&[0])
                .build();

            let mut state = SparseStab::new(1);
            let mut runner = CircuitRunner::<SparseStab>::new().with_seed(42);
            runner.on_signal(move |_: &RoundBoundary| {
                counter_clone.fetch_add(1, Ordering::Relaxed);
            });

            let _outcomes = runner.apply_circuit(&mut state, &commands).unwrap();

            assert_eq!(
                counter.load(Ordering::Relaxed),
                2,
                "Signal should be dispatched twice"
            );
        }

        #[test]
        fn test_multiple_signal_types() {
            let round_count = Arc::new(AtomicU64::new(0));
            let temp_count = Arc::new(AtomicU64::new(0));
            let rc = round_count.clone();
            let tc = temp_count.clone();

            let commands = CommandBuilder::new()
                .pz(&[0])
                .signal(RoundBoundary(1))
                .signal(Temperature(300.0))
                .h(&[0])
                .mz(&[0])
                .build();

            let mut state = SparseStab::new(1);
            let mut runner = CircuitRunner::<SparseStab>::new().with_seed(42);
            runner.on_signal(move |_: &RoundBoundary| {
                rc.fetch_add(1, Ordering::Relaxed);
            });
            runner.on_signal(move |_: &Temperature| {
                tc.fetch_add(1, Ordering::Relaxed);
            });

            let _outcomes = runner.apply_circuit(&mut state, &commands).unwrap();

            assert_eq!(round_count.load(Ordering::Relaxed), 1);
            assert_eq!(temp_count.load(Ordering::Relaxed), 1);
        }
    }

    // ================================================================
    // AdaptedSequence tests
    // ================================================================

    #[test]
    fn test_adapted_basic_execution() {
        let gates_def = GateDefinitions::new();
        let circuit = OpBuilder::new()
            .pz(QubitId(0))
            .h(QubitId(0))
            .mz(QubitId(0), ResultId(0))
            .build();

        let mut state = SparseStab::new(1);
        let mut runner = CircuitRunner::<SparseStab>::with_definitions(gates_def).with_seed(42);

        let outcomes = runner.apply_adapted_circuit(&mut state, &circuit).unwrap();
        assert_eq!(outcomes.len(), 1);
    }

    #[test]
    fn test_adapted_bell_state() {
        let gates_def = GateDefinitions::new();
        let circuit = OpBuilder::new()
            .pz(QubitId(0))
            .pz(QubitId(1))
            .h(QubitId(0))
            .cx(QubitId(0), QubitId(1))
            .mz(QubitId(0), ResultId(0))
            .mz(QubitId(1), ResultId(1))
            .build();

        let mut state = SparseStab::new(2);
        let mut runner = CircuitRunner::<SparseStab>::with_definitions(gates_def).with_seed(42);

        let outcomes = runner.apply_adapted_circuit(&mut state, &circuit).unwrap();
        assert_eq!(outcomes.len(), 2);
        let o0 = outcomes.get_bit(QubitId(0)).unwrap();
        let o1 = outcomes.get_bit(QubitId(1)).unwrap();
        assert_eq!(o0, o1, "Bell state outcomes should be equal");
    }

    #[test]
    fn test_conditional_operation() {
        let gates_def = GateDefinitions::new();

        let circuit = OpBuilder::new()
            .pz(QubitId(0))
            .pz(QubitId(1))
            .x(QubitId(0))
            .mz(QubitId(0), ResultId(0))
            .if_one(ResultId(0), |b| b.x(QubitId(1)))
            .mz(QubitId(1), ResultId(1))
            .build();

        let mut state = SparseStab::new(2);
        let mut runner = CircuitRunner::<SparseStab>::with_definitions(gates_def).with_seed(42);

        let outcomes = runner.apply_adapted_circuit(&mut state, &circuit).unwrap();
        assert!(outcomes.get_bit(QubitId(0)).unwrap());
        assert!(outcomes.get_bit(QubitId(1)).unwrap());
    }

    #[test]
    fn test_custom_gate_needs_decomposition() {
        let mut gates_def = GateDefinitions::new();

        let custom_id = gates_def.register(
            GateSpec::new("CustomNoDecomp")
                .with_quantum_arity(1)
                .with_category(GateCategory::SingleQubitUnitary),
        );

        let circuit = OpBuilder::new()
            .pz(QubitId(0))
            .gate1(custom_id, QubitId(0))
            .mz(QubitId(0), ResultId(0))
            .build();

        let mut state = SparseStab::new(1);
        let mut runner = CircuitRunner::<SparseStab>::with_definitions(gates_def);

        let result = runner.apply_adapted_circuit(&mut state, &circuit);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ExecutionError::NoDecomposition { .. }
        ));
    }

    #[test]
    fn test_adapted_circuit_resets_noise() {
        let gates_def = GateDefinitions::new();
        let circuit = OpBuilder::new()
            .pz(QubitId(0))
            .mz(QubitId(0), ResultId(0))
            .build();

        let mut state = SparseStab::new(1);
        let mut runner = CircuitRunner::<SparseStab>::with_definitions(gates_def).with_seed(42);

        let outcomes1 = runner.apply_adapted_circuit(&mut state, &circuit).unwrap();
        state.reset();
        let outcomes2 = runner.apply_adapted_circuit(&mut state, &circuit).unwrap();

        assert_eq!(outcomes1.len(), 1);
        assert_eq!(outcomes2.len(), 1);
    }

    #[test]
    fn test_x_basis_measurement() {
        let gates_def = GateDefinitions::new();

        let circuit = OpBuilder::new()
            .px(QubitId(0))
            .mx(QubitId(0), ResultId(0))
            .build();

        let mut state = SparseStab::new(1);
        let mut runner = CircuitRunner::<SparseStab>::with_definitions(gates_def).with_seed(42);

        let outcomes = runner.apply_adapted_circuit(&mut state, &circuit).unwrap();
        assert!(
            !outcomes.get_bit(QubitId(0)).unwrap(),
            "|+> measured in X should give 0"
        );
    }

    #[test]
    fn test_gate_override_custom_gate() {
        let mut gates_def = GateDefinitions::new();

        let custom_id = gates_def.register(
            GateSpec::new("CustomGate")
                .with_quantum_arity(1)
                .with_category(GateCategory::SingleQubitUnitary),
        );

        let overrides: GateOverrides<SparseStab> =
            GateOverrides::new().register(custom_id, |sim, _angles, qubits| {
                sim.x(qubits);
                true
            });

        let circuit = OpBuilder::new()
            .pz(QubitId(0))
            .gate1(custom_id, QubitId(0))
            .mz(QubitId(0), ResultId(0))
            .build();

        let mut state = SparseStab::new(1);
        let mut runner = CircuitRunner::<SparseStab>::with_definitions(gates_def)
            .with_overrides(overrides)
            .with_seed(42);

        let outcomes = runner.apply_adapted_circuit(&mut state, &circuit).unwrap();
        assert!(outcomes.get_bit(QubitId(0)).unwrap());
    }

    #[test]
    fn test_gate_override_core_gate() {
        let gates_def = GateDefinitions::new();

        let overrides: GateOverrides<SparseStab> =
            GateOverrides::new().register(gates::H, |_sim, _angles, _qubits| {
                true // Do nothing
            });

        let circuit = OpBuilder::new()
            .pz(QubitId(0))
            .h(QubitId(0))
            .mz(QubitId(0), ResultId(0))
            .build();

        let mut state = SparseStab::new(1);
        let mut runner = CircuitRunner::<SparseStab>::with_definitions(gates_def)
            .with_overrides(overrides)
            .with_seed(42);

        let outcomes = runner.apply_adapted_circuit(&mut state, &circuit).unwrap();
        assert!(!outcomes.get_bit(QubitId(0)).unwrap());
    }

    #[test]
    fn test_overrides_api() {
        let mut overrides: GateOverrides<SparseStab> = GateOverrides::new();
        assert!(overrides.is_empty());
        assert_eq!(overrides.len(), 0);

        overrides.insert(gates::H, |sim, _, qubits| {
            sim.h(qubits);
            true
        });
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

        let gates_def = GateDefinitions::new();

        let circuit = OpBuilder::new()
            .pz(QubitId(0))
            .rx(QubitId(0), Angle64::HALF_TURN)
            .mz(QubitId(0), ResultId(0))
            .build();

        let mut state = StateVec::new(1);
        let mut runner =
            CircuitRunner::<StateVec>::rotations_with_definitions(gates_def).with_seed(42);

        assert!(runner.has_rotation_support());

        let outcomes = runner.apply_adapted_circuit(&mut state, &circuit).unwrap();
        assert!(outcomes.get_bit(QubitId(0)).unwrap());
    }

    #[test]
    fn test_rotation_without_support_decomposes() {
        let gates_def = GateDefinitions::new();

        let circuit = OpBuilder::new()
            .pz(QubitId(0))
            .gate1(gates::T, QubitId(0))
            .mz(QubitId(0), ResultId(0))
            .build();

        let mut state = SparseStab::new(1);
        let mut runner = CircuitRunner::<SparseStab>::with_definitions(gates_def);

        assert!(!runner.has_rotation_support());

        let result = runner.apply_adapted_circuit(&mut state, &circuit);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ExecutionError::NoDecomposition { .. }
        ));
    }

    // ================================================================
    // apply_gate (interpreter mode) tests
    // ================================================================

    #[test]
    fn test_apply_gate_basic() {
        let mut state = SparseStab::new(1);
        let mut runner = CircuitRunner::<SparseStab>::new().with_seed(42);

        runner
            .apply_gate(&mut state, GateType::PZ, &[QubitId(0)], &[])
            .unwrap();
        runner
            .apply_gate(&mut state, GateType::H, &[QubitId(0)], &[])
            .unwrap();
        runner
            .apply_gate(&mut state, GateType::MZ, &[QubitId(0)], &[])
            .unwrap();

        let outcomes = runner.take_outcomes();
        assert_eq!(outcomes.len(), 1);
    }
}
