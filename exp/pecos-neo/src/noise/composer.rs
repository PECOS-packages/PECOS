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

//! Composable noise model that combines multiple channels.
//!
//! The noise model uses a plugin-based architecture:
//!
//! - **Plugins** bundle related functionality
//! - **Event handlers** update context state
//! - **Noise channels** produce noise responses
//! - **Context observers** react to state changes

use super::context::NoiseContext;
use super::idle::IdleChannel;
use super::plugin::{ContextObserver, EventHandler, NoiseModelConfig, NoisePlugin};
use super::{
    EventKinds, NoiseChannel, NoiseEvent, NoiseEventKind, NoiseGateRequirement, NoiseResponse,
};
use crate::command::GateType;
use pecos_core::{QubitId, TimeScale};
use pecos_random::PecosRng;

/// A composable noise model that combines multiple noise channels.
///
/// Uses a plugin-based architecture where functionality is registered
/// through plugins rather than hardcoded.
///
/// # Plugin-Based Usage (Recommended)
///
/// ```
/// use pecos_neo::noise::*;
/// use pecos_neo::noise::plugins::*;
///
/// let noise = ComposableNoiseModel::new()
///     .add_plugin(&CorePlugin)           // State tracking
///     .add_plugin(&LeakagePlugin::new()) // Leakage handling
///     .add_plugin(&DepolarizingPlugin::new(0.01, 0.02));
/// ```
///
/// # Direct Channel Usage (Legacy)
///
/// ```
/// use pecos_neo::noise::*;
///
/// let noise = ComposableNoiseModel::new()
///     .add_channel(SingleQubitChannel::depolarizing(0.01))
///     .add_channel(TwoQubitChannel::depolarizing(0.02));
/// ```
pub struct ComposableNoiseModel {
    /// Event handlers that update context state (run before channels).
    event_handlers: Vec<Box<dyn EventHandler>>,

    /// Noise channels that produce noise responses.
    channels: Vec<Box<dyn NoiseChannel>>,

    /// Indices into the source vectors, preserving their dispatch order.
    channel_buckets: [Vec<usize>; EventKinds::COUNT],
    handler_buckets: [Vec<usize>; EventKinds::COUNT],

    /// Runner capabilities required by configured gate-injection mechanisms.
    gate_requirements: Vec<NoiseGateRequirement>,

    /// Observers that react to context state changes.
    observers: Vec<Box<dyn ContextObserver>>,

    /// Shared noise context.
    context: NoiseContext,

    /// Time scale for interpreting `TimeUnits` as physical time.
    ///
    /// When set, this defines what 1 `TimeUnit` represents (e.g., nanoseconds).
    /// Used by convenience methods that accept physical time parameters.
    time_scale: Option<TimeScale>,
}

impl std::fmt::Debug for ComposableNoiseModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Buckets are derived indices, not part of the model's diagnostic view.
        let Self {
            event_handlers,
            channels,
            gate_requirements,
            observers,
            context,
            time_scale,
            channel_buckets: _,
            handler_buckets: _,
        } = self;
        f.debug_struct("ComposableNoiseModel")
            .field("time_scale", &time_scale)
            .field("gate_requirements", &gate_requirements)
            .field("event_handler_count", &event_handlers.len())
            .field(
                "event_handler_names",
                &event_handlers.iter().map(|h| h.name()).collect::<Vec<_>>(),
            )
            .field("channel_count", &channels.len())
            .field(
                "channel_names",
                &channels.iter().map(|c| c.name()).collect::<Vec<_>>(),
            )
            .field("observer_count", &observers.len())
            .field("context", &context)
            .finish()
    }
}

impl Default for ComposableNoiseModel {
    fn default() -> Self {
        Self::new()
    }
}

impl ComposableNoiseModel {
    /// Create a new empty composable noise model.
    #[must_use]
    pub fn new() -> Self {
        Self {
            event_handlers: Vec::new(),
            channels: Vec::new(),
            channel_buckets: std::array::from_fn(|_| Vec::new()),
            handler_buckets: std::array::from_fn(|_| Vec::new()),
            gate_requirements: Vec::new(),
            observers: Vec::new(),
            context: NoiseContext::new(),
            time_scale: None,
        }
    }

    /// Set the time scale for this noise model.
    ///
    /// This defines what 1 `TimeUnit` represents in physical time.
    /// When set, convenience methods that accept physical time parameters
    /// will use this scale for conversion.
    ///
    /// # Example
    /// ```
    /// use pecos_neo::noise::ComposableNoiseModel;
    /// use pecos_core::TimeScale;
    ///
    /// let noise = ComposableNoiseModel::new()
    ///     .with_time_scale(TimeScale::NANOSECONDS);  // 1 TimeUnit = 1 ns
    /// ```
    #[must_use]
    pub fn with_time_scale(mut self, scale: TimeScale) -> Self {
        self.time_scale = Some(scale);
        self
    }

    /// Get the time scale for this noise model, if set.
    #[must_use]
    pub fn time_scale(&self) -> Option<TimeScale> {
        self.time_scale
    }

    /// Set gate definitions for this noise model.
    ///
    /// When set, noise channels can query gate metadata (category, arity, etc.)
    /// via the `NoiseContext`. This enables category-based noise filtering and
    /// uniform treatment of core and custom gates.
    ///
    /// # Example
    /// ```no_run
    /// use pecos_neo::noise::ComposableNoiseModel;
    /// use pecos_neo::extensible::{GateDefinitions, GateSpec, GateCategory};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let gates = GateDefinitions::builder()
    ///     .define_gate("MyGate", GateSpec::new("MyGate").with_quantum_arity(2))
    ///     .with_category_noise(GateCategory::TwoQubitUnitary, 0.02)
    ///     .build()?;
    ///
    /// let noise = ComposableNoiseModel::new()
    ///     .with_gate_definitions(gates);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn with_gate_definitions(mut self, defs: crate::extensible::GateDefinitions) -> Self {
        self.context.set_gate_definitions(defs);
        self
    }

    /// Get gate definitions if set.
    #[must_use]
    pub fn gate_definitions(&self) -> Option<&crate::extensible::GateDefinitions> {
        self.context.gate_definitions()
    }

    /// Add a plugin to the model.
    ///
    /// Plugins can register event handlers, channels, and observers.
    /// This is the recommended way to configure the noise model.
    #[must_use]
    pub fn add_plugin(mut self, plugin: &(impl NoisePlugin + 'static)) -> Self {
        let mut config = NoiseModelConfig::new();
        plugin.build(&mut config);

        // Transfer registered components from config to model
        self.event_handlers.extend(config.event_handlers);
        self.gate_requirements.extend(
            config
                .channels
                .iter()
                .flat_map(|channel| channel.gate_requirements()),
        );
        for channel in config.channels {
            self.push_channel(channel);
        }
        self.observers.extend(config.observers);

        // Keep handlers sorted by priority (high to low) for efficient iteration
        self.event_handlers
            .sort_by_key(|h| std::cmp::Reverse(h.priority()));
        self.rebuild_handler_buckets();

        self
    }

    /// Add a noise channel directly to the model.
    ///
    /// For plugin-based configuration, use `add_plugin()` instead.
    #[must_use]
    pub fn add_channel(mut self, channel: impl NoiseChannel + 'static) -> Self {
        self.gate_requirements.extend(channel.gate_requirements());
        self.push_channel(Box::new(channel));
        self
    }

    /// Add a channel while recording the builder setter that configured it.
    pub(crate) fn add_channel_configured_by(
        mut self,
        channel: impl NoiseChannel + 'static,
        configured_by: &'static str,
        fix: &'static str,
    ) -> Self {
        self.gate_requirements
            .extend(channel.gate_requirements().into_iter().map(|requirement| {
                NoiseGateRequirement {
                    configured_by,
                    fix,
                    ..requirement
                }
            }));
        self.push_channel(Box::new(channel));
        self
    }

    /// Add a pre-boxed noise channel to the model.
    ///
    /// This is useful when you have a `Box<dyn NoiseChannel>` from a builder
    /// or other source. For most cases, use [`Self::add_channel`] instead.
    #[must_use]
    pub fn add_boxed_channel(mut self, channel: Box<dyn NoiseChannel>) -> Self {
        self.gate_requirements.extend(channel.gate_requirements());
        self.push_channel(channel);
        self
    }

    /// Validate gate-injection requirements against a runner configuration.
    ///
    /// # Errors
    ///
    /// Returns a diagnostic naming the configuring setter, incompatible runner,
    /// and concrete fix when an injected gate is unsupported.
    pub(crate) fn validate_runner_gate_support(
        &self,
        runner: &str,
        has_rotation_support: bool,
    ) -> Result<(), String> {
        for requirement in &self.gate_requirements {
            let gate_type = requirement.gate_type;
            if supports_clifford_noise_gate(gate_type)
                || (has_rotation_support && supports_rotation_noise_gate(gate_type))
            {
                continue;
            }

            let needs_rotation = supports_rotation_noise_gate(gate_type);
            let fix = if runner == "ImportanceSamplingRunner" && needs_rotation {
                "switch to a stochastic noise mechanism (for coherent idle configuration, use \
                 the stochastic idle family with with_p_idle_coherent(false)); \
                 ImportanceSamplingRunner does not provide a rotation executor"
            } else {
                requirement.fix
            };
            let limitation = if needs_rotation {
                "has no rotation executor and cannot represent that noise gate"
            } else {
                "cannot execute that injected gate with any supported executor"
            };
            return Err(format!(
                "{} configures a noise mechanism that can inject {gate_type:?}, but {runner} \
                 {limitation}; {fix}.",
                requirement.configured_by
            ));
        }

        Ok(())
    }

    /// Add an event handler directly to the model.
    ///
    /// For plugin-based configuration, use `add_plugin()` instead.
    #[must_use]
    pub fn add_event_handler(mut self, handler: impl EventHandler + 'static) -> Self {
        index_kinds(
            &mut self.handler_buckets,
            self.event_handlers.len(),
            handler.event_kinds(),
        );
        self.event_handlers.push(Box::new(handler));
        self
    }

    fn push_channel(&mut self, channel: Box<dyn NoiseChannel>) {
        index_kinds(
            &mut self.channel_buckets,
            self.channels.len(),
            channel.event_kinds(),
        );
        self.channels.push(channel);
    }

    fn rebuild_handler_buckets(&mut self) {
        for bucket in &mut self.handler_buckets {
            bucket.clear();
        }
        for (index, handler) in self.event_handlers.iter().enumerate() {
            index_kinds(&mut self.handler_buckets, index, handler.event_kinds());
        }
    }

    /// Add a context observer directly to the model.
    ///
    /// For plugin-based configuration, use `add_plugin()` instead.
    #[must_use]
    pub fn add_observer(mut self, observer: impl ContextObserver + 'static) -> Self {
        self.observers.push(Box::new(observer));
        self
    }

    /// Add an idle channel with T1/T2 times in physical units.
    ///
    /// Requires `with_time_scale()` to be called first. T2 is total transverse coherence time,
    /// not pure-dephasing Tphi. The first-order Pauli-twirl mapping, physical bound, validity
    /// domain, and numerical compatibility note are documented by [`IdleChannel::from_t1_t2`].
    ///
    /// # Arguments
    /// * `t1_seconds` - T1 relaxation time in seconds
    /// * `t2_seconds` - Total T2 transverse coherence time in seconds
    ///
    /// # Panics
    /// Panics if `with_time_scale()` has not been called, if either time is non-finite or not
    /// greater than zero, or if `t2_seconds > 2 * t1_seconds`.
    ///
    /// # Example
    /// ```
    /// use pecos_neo::noise::ComposableNoiseModel;
    /// use pecos_core::TimeScale;
    ///
    /// let noise = ComposableNoiseModel::new()
    ///     .with_time_scale(TimeScale::NANOSECONDS)
    ///     .with_idle_t1_t2(50e-6, 30e-6);  // T1=50us, T2=30us
    /// ```
    #[must_use]
    pub fn with_idle_t1_t2(self, t1_seconds: f64, t2_seconds: f64) -> Self {
        let scale = self
            .time_scale
            .expect("with_time_scale() must be called before with_idle_t1_t2()");
        let channel = IdleChannel::from_t1_t2_seconds(t1_seconds, t2_seconds, scale);
        self.add_channel(channel)
    }

    /// Get the number of channels in the model.
    #[must_use]
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    /// Get the number of event handlers in the model.
    #[must_use]
    pub fn event_handler_count(&self) -> usize {
        self.event_handlers.len()
    }

    /// Get the names of all channels in the model.
    #[must_use]
    pub fn channel_names(&self) -> Vec<&str> {
        self.channels.iter().map(|c| c.name()).collect()
    }

    /// Get a summary description of the noise model.
    ///
    /// Returns a human-readable string describing all channels.
    #[must_use]
    pub fn describe(&self) -> String {
        use std::fmt::Write;
        let mut result = String::new();
        writeln!(result, "ComposableNoiseModel").unwrap();
        writeln!(result, "====================").unwrap();
        writeln!(result, "Channels: {}", self.channels.len()).unwrap();
        writeln!(result, "Event handlers: {}", self.event_handlers.len()).unwrap();
        writeln!(result).unwrap();

        if !self.channels.is_empty() {
            writeln!(result, "Channel list:").unwrap();
            for (i, channel) in self.channels.iter().enumerate() {
                writeln!(result, "  {}. {}", i + 1, channel.name()).unwrap();
            }
        }

        result
    }

    /// Get a reference to the noise context.
    #[must_use]
    pub fn context(&self) -> &NoiseContext {
        &self.context
    }

    /// Get a mutable reference to the noise context.
    pub fn context_mut(&mut self) -> &mut NoiseContext {
        &mut self.context
    }

    /// Mark a gate type as noiseless (no noise applied).
    ///
    /// Useful for software-implemented gates that don't correspond to
    /// physical operations.
    #[must_use]
    pub fn with_noiseless_gate(mut self, gate_type: crate::command::GateType) -> Self {
        self.context.add_noiseless_gate(gate_type);
        self
    }

    /// Mark multiple gate types as noiseless.
    #[must_use]
    pub fn with_noiseless_gates(mut self, gate_types: &[crate::command::GateType]) -> Self {
        for &gate_type in gate_types {
            self.context.add_noiseless_gate(gate_type);
        }
        self
    }

    /// Emit an event and collect responses from all relevant channels.
    ///
    /// Processing order:
    /// 1. Run event handlers (state updates)
    /// 2. Run noise channels (produce responses)
    /// 3. Apply state changes from responses (leakage)
    /// 4. Notify observers of state changes
    pub fn emit(&mut self, event: &NoiseEvent<'_>, rng: &mut PecosRng) -> NoiseResponse {
        // 1. Run event handlers for state updates
        self.run_event_handlers(event);

        // 2. Collect responses from noise channels using try_apply for efficiency
        let mut combined = NoiseResponse::None;
        for &index in &self.channel_buckets[event.kind().index()] {
            let channel = &self.channels[index];
            // try_apply combines responds_to + apply in one call
            // filter out NoiseResponse::None to avoid unnecessary combine calls
            if let Some(response) = channel
                .try_apply(event, &mut self.context, rng)
                .filter(|r| !r.is_none())
            {
                combined = combined.combine(response);
            }
        }

        // 3. Apply state changes and notify observers (skip if no response)
        if !combined.is_none() {
            let observer_response = self.apply_state_changes_with_observers(&combined, rng);
            if !observer_response.is_none() {
                combined = combined.combine(observer_response);
            }
        }

        combined
    }

    /// Run all event handlers that respond to this event.
    ///
    /// Handlers are pre-sorted by priority (high to low) in `add_plugin()`,
    /// so we only need to filter, not sort.
    fn run_event_handlers(&mut self, event: &NoiseEvent<'_>) {
        // Collect indices of handlers that respond to this event.
        // This allows us to release the borrow on self.event_handlers
        // before mutably borrowing self.context.
        let bucket = &self.handler_buckets[event.kind().index()];
        let handler_indices: smallvec::SmallVec<[usize; 4]> = bucket
            .iter()
            .copied()
            .filter(|&i| self.event_handlers[i].handles(event))
            .collect();

        for i in handler_indices {
            self.event_handlers[i].handle(event, &mut self.context);
        }
    }

    /// Apply state changes from a noise response and notify observers.
    fn apply_state_changes_with_observers(
        &mut self,
        response: &NoiseResponse,
        rng: &mut PecosRng,
    ) -> NoiseResponse {
        let mut observer_responses = NoiseResponse::None;

        match response {
            NoiseResponse::MarkLeaked(qubits) => {
                for &q in qubits {
                    self.context.mark_leaked(q);
                    // Notify observers
                    let obs_response = self.notify_leaked(q, rng);
                    if !obs_response.is_none() {
                        observer_responses = observer_responses.combine(obs_response);
                    }
                }
            }
            NoiseResponse::MarkUnleaked(qubits) => {
                for &q in qubits {
                    self.context.mark_unleaked(q);
                    // Notify observers
                    let obs_response = self.notify_unleaked(q, rng);
                    if !obs_response.is_none() {
                        observer_responses = observer_responses.combine(obs_response);
                    }
                }
            }
            NoiseResponse::Multiple(responses) => {
                for r in responses {
                    let obs_response = self.apply_state_changes_with_observers(r, rng);
                    if !obs_response.is_none() {
                        observer_responses = observer_responses.combine(obs_response);
                    }
                }
            }
            NoiseResponse::None
            | NoiseResponse::InjectGates(_)
            | NoiseResponse::FlipOutcomes(_)
            | NoiseResponse::ForceOutcomes(_)
            | NoiseResponse::LeakedMeasurement(_)
            | NoiseResponse::SkipGate => {}
        }

        observer_responses
    }

    /// Notify observers that a qubit was leaked.
    fn notify_leaked(&self, qubit: QubitId, rng: &mut PecosRng) -> NoiseResponse {
        let mut combined = NoiseResponse::None;
        for observer in &self.observers {
            let response = observer.on_leaked(qubit, &self.context, rng);
            if !response.is_none() {
                combined = combined.combine(response);
            }
        }
        combined
    }

    /// Notify observers that a qubit was unleaked.
    fn notify_unleaked(&self, qubit: QubitId, rng: &mut PecosRng) -> NoiseResponse {
        let mut combined = NoiseResponse::None;
        for observer in &self.observers {
            let response = observer.on_unleaked(qubit, &self.context, rng);
            if !response.is_none() {
                combined = combined.combine(response);
            }
        }
        combined
    }

    /// Reset the noise model state for a new shot.
    pub fn reset(&mut self) {
        self.context.reset();
    }

    /// Check if the model has any channels.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.channels.is_empty() && self.event_handlers.is_empty()
    }
}

fn index_kinds(buckets: &mut [Vec<usize>; EventKinds::COUNT], index: usize, kinds: EventKinds) {
    for kind in NoiseEventKind::ALL {
        if kinds.contains(kind) {
            buckets[kind.index()].push(index);
        }
    }
}

fn supports_clifford_noise_gate(gate_type: GateType) -> bool {
    matches!(
        gate_type,
        GateType::I
            | GateType::X
            | GateType::Y
            | GateType::Z
            | GateType::H
            | GateType::F
            | GateType::Fdg
            | GateType::SX
            | GateType::SXdg
            | GateType::SY
            | GateType::SYdg
            | GateType::SZ
            | GateType::SZdg
            | GateType::CX
            | GateType::CY
            | GateType::CZ
            | GateType::SZZ
            | GateType::SZZdg
            | GateType::SXX
            | GateType::SXXdg
            | GateType::SYY
            | GateType::SYYdg
            | GateType::SWAP
    )
}

fn supports_rotation_noise_gate(gate_type: GateType) -> bool {
    matches!(
        gate_type,
        GateType::T
            | GateType::Tdg
            | GateType::RX
            | GateType::RY
            | GateType::RZ
            | GateType::U
            | GateType::RXY1Q
            | GateType::RXX
            | GateType::RYY
            | GateType::RZZ
            | GateType::CCX
    )
}

// ============================================================================
// From implementations for ergonomic noise model construction
// ============================================================================

impl Clone for ComposableNoiseModel {
    fn clone(&self) -> Self {
        Self {
            event_handlers: self.event_handlers.iter().map(|h| h.clone_box()).collect(),
            channels: self.channels.iter().map(|c| c.clone_box()).collect(),
            channel_buckets: self.channel_buckets.clone(),
            handler_buckets: self.handler_buckets.clone(),
            gate_requirements: self.gate_requirements.clone(),
            observers: self.observers.iter().map(|o| o.clone_box()).collect(),
            context: self.context.clone(),
            time_scale: self.time_scale,
        }
    }
}

impl<C: NoiseChannel + 'static> From<C> for ComposableNoiseModel {
    fn from(channel: C) -> Self {
        Self::new().add_channel(channel)
    }
}

impl From<super::GeneralNoiseModelBuilder> for ComposableNoiseModel {
    /// Convert a `GeneralNoiseModelBuilder` directly to a `ComposableNoiseModel`.
    ///
    /// This allows passing the builder without calling `.build()`:
    ///
    /// ```no_run
    /// use pecos_neo::tool::sim_neo;
    /// use pecos_neo::noise::GeneralNoiseModelBuilder;
    /// use pecos_neo::command::CommandQueue;
    ///
    /// let circuit = CommandQueue::new();
    ///
    /// // Both of these work:
    /// sim_neo(circuit.clone()).auto().noise(GeneralNoiseModelBuilder::new().with_p1(0.01).build());
    /// sim_neo(circuit).noise(GeneralNoiseModelBuilder::new().with_p1(0.01));  // No .build()!
    /// ```
    fn from(builder: super::GeneralNoiseModelBuilder) -> Self {
        builder.build()
    }
}

#[cfg(test)]
mod tests {
    use crate::noise::event_kind_tests::{representative, witnesses};
    use std::sync::{Arc, Mutex};

    type DispatchLog = Arc<Mutex<Vec<(&'static str, &'static str, NoiseEventKind)>>>;

    #[derive(Clone)]
    struct Recorder {
        label: &'static str,
        priority: i32,
        selected: bool,
        log: DispatchLog,
    }

    impl NoiseChannel for Recorder {
        fn responds_to(&self, _event: &NoiseEvent<'_>) -> bool {
            // Deliberately differs from try_apply to pin optimized dispatch.
            false
        }

        fn apply(
            &self,
            _event: &NoiseEvent<'_>,
            _ctx: &mut NoiseContext,
            _rng: &mut PecosRng,
        ) -> NoiseResponse {
            panic!("the composer must call try_apply");
        }

        fn try_apply(
            &self,
            event: &NoiseEvent<'_>,
            _ctx: &mut NoiseContext,
            _rng: &mut PecosRng,
        ) -> Option<NoiseResponse> {
            self.log
                .lock()
                .unwrap()
                .push(("try_apply", self.label, event.kind()));
            Some(NoiseResponse::None)
        }

        fn name(&self) -> &'static str {
            self.label
        }
        fn clone_box(&self) -> Box<dyn NoiseChannel> {
            Box::new(self.clone())
        }
    }

    impl EventHandler for Recorder {
        fn handles(&self, event: &NoiseEvent<'_>) -> bool {
            self.log
                .lock()
                .unwrap()
                .push(("handles", self.label, event.kind()));
            self.selected
        }

        fn handle(&self, event: &NoiseEvent<'_>, _ctx: &mut NoiseContext) {
            self.log
                .lock()
                .unwrap()
                .push(("handle", self.label, event.kind()));
        }

        fn name(&self) -> &'static str {
            self.label
        }
        fn priority(&self) -> i32 {
            self.priority
        }
        fn clone_box(&self) -> Box<dyn EventHandler> {
            Box::new(self.clone())
        }
    }

    struct RecordingPlugin {
        handlers: Vec<Recorder>,
        channels: Vec<Recorder>,
    }

    #[derive(Clone)]
    struct KindRecorder {
        inner: Recorder,
        kinds: EventKinds,
    }

    impl NoiseChannel for KindRecorder {
        fn event_kinds(&self) -> EventKinds {
            self.kinds
        }
        fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
            self.kinds.contains(event.kind())
        }
        fn apply(
            &self,
            _event: &NoiseEvent<'_>,
            _ctx: &mut NoiseContext,
            _rng: &mut PecosRng,
        ) -> NoiseResponse {
            NoiseResponse::None
        }
        fn try_apply(
            &self,
            event: &NoiseEvent<'_>,
            ctx: &mut NoiseContext,
            rng: &mut PecosRng,
        ) -> Option<NoiseResponse> {
            let response = self.inner.try_apply(event, ctx, rng);
            if self.responds_to(event) {
                response
            } else {
                None
            }
        }
        fn name(&self) -> &'static str {
            self.inner.label
        }
        fn clone_box(&self) -> Box<dyn NoiseChannel> {
            Box::new(self.clone())
        }
    }

    impl EventHandler for KindRecorder {
        fn event_kinds(&self) -> EventKinds {
            self.kinds
        }
        fn handles(&self, event: &NoiseEvent<'_>) -> bool {
            self.inner.handles(event) && self.kinds.contains(event.kind())
        }
        fn handle(&self, event: &NoiseEvent<'_>, ctx: &mut NoiseContext) {
            self.inner.handle(event, ctx);
        }
        fn clone_box(&self) -> Box<dyn EventHandler> {
            Box::new(self.clone())
        }
    }

    #[test]
    fn excluded_kinds_do_not_visit_channels_or_handlers() {
        let log: DispatchLog = Arc::default();
        let recorder = |label, kinds| KindRecorder {
            inner: Recorder {
                label,
                priority: 0,
                selected: true,
                log: log.clone(),
            },
            kinds,
        };
        let prep = recorder("prep", EventKinds::of(NoiseEventKind::AfterPreparation));
        let never = recorder("never", EventKinds::NONE);
        let mut model = ComposableNoiseModel::new()
            .add_channel(never.clone())
            .add_channel(prep.clone())
            .add_event_handler(never)
            .add_event_handler(prep);
        let mut rng = PecosRng::seed_from_u64(42);
        for kind in NoiseEventKind::ALL {
            log.lock().unwrap().clear();
            model.emit(&representative(kind), &mut rng);
            if kind == NoiseEventKind::AfterPreparation {
                assert_eq!(
                    *log.lock().unwrap(),
                    [
                        ("handles", "prep", kind),
                        ("handle", "prep", kind),
                        ("try_apply", "prep", kind),
                    ]
                );
            } else {
                assert!(log.lock().unwrap().is_empty());
            }
        }
    }

    impl NoisePlugin for RecordingPlugin {
        fn build(&self, config: &mut NoiseModelConfig) {
            for handler in &self.handlers {
                config.add_event_handler(handler.clone());
            }
            for channel in &self.channels {
                config.add_channel(channel.clone());
            }
        }
    }

    #[test]
    fn default_declarations_dispatch_every_kind_at_every_insertion_point() {
        let log: DispatchLog = Arc::default();
        let recorder = |label| Recorder {
            label,
            priority: 0,
            selected: true,
            log: log.clone(),
        };
        let mut model = ComposableNoiseModel::new()
            .add_channel(recorder("direct"))
            .add_channel_configured_by(recorder("configured"), "test", "test")
            .add_boxed_channel(Box::new(recorder("boxed")))
            .add_plugin(&RecordingPlugin {
                handlers: vec![recorder("plugin_handler")],
                channels: vec![recorder("plugin_channel")],
            })
            .add_event_handler(recorder("direct_handler"));
        assert_eq!(
            model.channel_names(),
            ["direct", "configured", "boxed", "plugin_channel"]
        );
        let mut rng = PecosRng::seed_from_u64(42);
        for kind in NoiseEventKind::ALL {
            log.lock().unwrap().clear();
            model.emit(&representative(kind), &mut rng);
            assert_eq!(
                *log.lock().unwrap(),
                [
                    ("handles", "plugin_handler", kind),
                    ("handles", "direct_handler", kind),
                    ("handle", "plugin_handler", kind),
                    ("handle", "direct_handler", kind),
                    ("try_apply", "direct", kind),
                    ("try_apply", "configured", kind),
                    ("try_apply", "boxed", kind),
                    ("try_apply", "plugin_channel", kind),
                ]
            );
        }
    }

    #[test]
    fn handler_buckets_preserve_stable_sort_append_and_two_phase_dispatch() {
        let log: DispatchLog = Arc::default();
        let recorder = |label, priority, selected| Recorder {
            label,
            priority,
            selected,
            log: log.clone(),
        };
        let mut model = ComposableNoiseModel::new()
            .add_event_handler(recorder("direct_first", 10, true))
            .add_plugin(&RecordingPlugin {
                handlers: vec![
                    recorder("low", 1, false),
                    recorder("high_a", 20, true),
                    recorder("high_b", 20, true),
                ],
                channels: vec![],
            })
            .add_event_handler(recorder("direct_last", 100, true));
        let kind = NoiseEventKind::AfterGate;
        let event = representative(kind);
        let mut rng = PecosRng::seed_from_u64(42);
        model.emit(&event, &mut rng);
        assert_eq!(
            *log.lock().unwrap(),
            [
                ("handles", "high_a", kind),
                ("handles", "high_b", kind),
                ("handles", "direct_first", kind),
                ("handles", "low", kind),
                ("handles", "direct_last", kind),
                ("handle", "high_a", kind),
                ("handle", "high_b", kind),
                ("handle", "direct_first", kind),
                ("handle", "direct_last", kind),
            ]
        );

        // Another plugin sorts the entire vector, including previous direct appends.
        model = model.add_plugin(&RecordingPlugin {
            handlers: vec![recorder("high_c", 20, true)],
            channels: vec![],
        });
        log.lock().unwrap().clear();
        model.emit(&event, &mut rng);
        assert_eq!(
            *log.lock().unwrap(),
            [
                ("handles", "direct_last", kind),
                ("handles", "high_a", kind),
                ("handles", "high_b", kind),
                ("handles", "high_c", kind),
                ("handles", "direct_first", kind),
                ("handles", "low", kind),
                ("handle", "direct_last", kind),
                ("handle", "high_a", kind),
                ("handle", "high_b", kind),
                ("handle", "high_c", kind),
                ("handle", "direct_first", kind),
            ]
        );
    }

    // ALL adapters recreate flat-vector dispatch without changing any predicate,
    // optimized try_apply, gate requirements, priority or response.
    struct AllChannel(Box<dyn NoiseChannel>);
    impl NoiseChannel for AllChannel {
        fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
            self.0.responds_to(event)
        }
        fn apply(
            &self,
            event: &NoiseEvent<'_>,
            ctx: &mut NoiseContext,
            rng: &mut PecosRng,
        ) -> NoiseResponse {
            self.0.apply(event, ctx, rng)
        }
        fn try_apply(
            &self,
            event: &NoiseEvent<'_>,
            ctx: &mut NoiseContext,
            rng: &mut PecosRng,
        ) -> Option<NoiseResponse> {
            self.0.try_apply(event, ctx, rng)
        }
        fn name(&self) -> &'static str {
            self.0.name()
        }
        fn priority(&self) -> i32 {
            self.0.priority()
        }
        fn gate_requirements(&self) -> smallvec::SmallVec<[NoiseGateRequirement; 2]> {
            self.0.gate_requirements()
        }
        fn clone_box(&self) -> Box<dyn NoiseChannel> {
            Box::new(Self(self.0.clone_box()))
        }
    }

    struct AllHandler(Box<dyn EventHandler>);
    impl EventHandler for AllHandler {
        fn handles(&self, event: &NoiseEvent<'_>) -> bool {
            self.0.handles(event)
        }
        fn handle(&self, event: &NoiseEvent<'_>, ctx: &mut NoiseContext) {
            self.0.handle(event, ctx);
        }
        fn name(&self) -> &'static str {
            self.0.name()
        }
        fn priority(&self) -> i32 {
            self.0.priority()
        }
        fn clone_box(&self) -> Box<dyn EventHandler> {
            Box::new(Self(self.0.clone_box()))
        }
    }

    fn assert_same_response(actual: &NoiseResponse, expected: &NoiseResponse) {
        use NoiseResponse::{
            FlipOutcomes, ForceOutcomes, InjectGates, LeakedMeasurement, MarkLeaked, MarkUnleaked,
            Multiple, None, SkipGate,
        };
        match (actual, expected) {
            (None, None) | (SkipGate, SkipGate) => {}
            (InjectGates(a), InjectGates(b)) => assert_eq!(a, b),
            (FlipOutcomes(a), FlipOutcomes(b))
            | (MarkLeaked(a), MarkLeaked(b))
            | (MarkUnleaked(a), MarkUnleaked(b))
            | (LeakedMeasurement(a), LeakedMeasurement(b)) => assert_eq!(a, b),
            (ForceOutcomes(a), ForceOutcomes(b)) => assert_eq!(a, b),
            (Multiple(a), Multiple(b)) => {
                assert_eq!(a.len(), b.len());
                for (a, b) in a.iter().zip(b) {
                    assert_same_response(a, b);
                }
            }
            _ => panic!("response variants differ: {actual:?} vs {expected:?}"),
        }
    }

    #[derive(Clone)]
    struct OptimizedEmptyChannel;

    impl NoiseChannel for OptimizedEmptyChannel {
        fn event_kinds(&self) -> EventKinds {
            EventKinds::NONE
        }

        fn responds_to(&self, _event: &NoiseEvent<'_>) -> bool {
            true
        }

        fn apply(
            &self,
            _event: &NoiseEvent<'_>,
            _ctx: &mut NoiseContext,
            _rng: &mut PecosRng,
        ) -> NoiseResponse {
            NoiseResponse::None
        }

        fn try_apply(
            &self,
            _event: &NoiseEvent<'_>,
            _ctx: &mut NoiseContext,
            _rng: &mut PecosRng,
        ) -> Option<NoiseResponse> {
            None
        }

        fn name(&self) -> &'static str {
            "OptimizedEmptyChannel"
        }

        fn clone_box(&self) -> Box<dyn NoiseChannel> {
            Box::new(self.clone())
        }
    }

    #[derive(Clone)]
    struct RandomProbeChannel;

    impl NoiseChannel for RandomProbeChannel {
        fn responds_to(&self, _event: &NoiseEvent<'_>) -> bool {
            true
        }

        fn apply(
            &self,
            _event: &NoiseEvent<'_>,
            _ctx: &mut NoiseContext,
            rng: &mut PecosRng,
        ) -> NoiseResponse {
            if rng.next_u64() > u64::MAX / 2 {
                NoiseResponse::SkipGate
            } else {
                NoiseResponse::None
            }
        }

        fn name(&self) -> &'static str {
            "RandomProbeChannel"
        }

        fn clone_box(&self) -> Box<dyn NoiseChannel> {
            Box::new(self.clone())
        }
    }

    #[test]
    fn importance_wrapper_preserves_rng_when_inner_optimized_dispatch_is_empty() {
        use crate::sampling::importance::{ImportanceConfig, ImportanceSamplingChannel};

        // The inner declaration covers its optimized try_apply, not responds_to.
        // The wrapper uses responds_to and consumes a proposal draw regardless
        // of whether the inner channel produces a response.
        let wrapper = ImportanceSamplingChannel::new(
            OptimizedEmptyChannel,
            ImportanceConfig::with_boost(0.02, 10.0),
        );
        let mut bucketed = ComposableNoiseModel::new()
            .add_channel(wrapper.clone())
            .add_channel(RandomProbeChannel);
        let mut flat = ComposableNoiseModel::new()
            .add_channel(AllChannel(Box::new(wrapper)))
            .add_channel(AllChannel(Box::new(RandomProbeChannel)));
        let mut rng = PecosRng::seed_from_u64(42);
        let mut flat_rng = PecosRng::seed_from_u64(42);
        let mut events = vec![NoiseEvent::BeforeCircuit { num_qubits: 1 }];
        events.extend(witnesses());

        for (index, event) in events.iter().enumerate() {
            let expected = flat.emit(event, &mut flat_rng);
            if index == 0 {
                // Seed 42 pins the reproducer: omitting the proposal draw makes
                // the probe emit SkipGate instead of None on the first event.
                assert_same_response(&expected, &NoiseResponse::None);
            }
            assert_same_response(&bucketed.emit(event, &mut rng), &expected);
        }
        assert_eq!(rng.next_u64(), flat_rng.next_u64());
    }

    fn equivalence_model(idle_family: usize) -> ComposableNoiseModel {
        use crate::noise::composite::channel::{
            BatchCompositeChannel, CompositeChannel, CompositeCrosstalkChannel,
            CompositeEventFilter,
        };
        use crate::noise::composite::prelude::pauli;
        use crate::noise::*;
        use crate::sampling::importance::{ImportanceConfig, ImportanceSamplingChannel};
        let axes = std::collections::BTreeMap::from([("Z".into(), 1.0)]);
        let builder = GeneralNoiseModelBuilder::new()
            .with_p1(0.2)
            .with_p2(0.3)
            .with_p1_emission_ratio(0.2)
            .with_p2_emission_ratio(0.2)
            .with_p1_seepage(0.2)
            .with_p2_seepage(0.2)
            .with_p_prep(0.2)
            .with_p_prep_leak_ratio(0.2)
            .with_p_meas(0.2, 0.3)
            .with_p_meas_state_flip(0.2)
            .with_p_prep_crosstalk(0.2)
            .with_p_meas_crosstalk(0.3, 0.2)
            .with_p_idle_linear(0.1, &axes)
            .with_idle_after_2q(1.0);
        let builder = match idle_family {
            0 => builder.with_p_idle_sin_squared(0.2, &axes),
            1 => builder.with_p_idle_quadratic(0.2),
            2 => builder
                .with_p_idle_quadratic(0.2)
                .with_p_idle_coherent(true),
            _ => unreachable!("test defines three idle configurations"),
        };
        builder
            .build()
            .add_channel(CategoryBasedChannel::new().with_default(0.2))
            .add_channel(GateDependentChannel::new().with_gate_error(GateType::H, 0.2))
            .add_channel(GateIdDependentChannel::new().with_gate_type_error(GateType::CX, 0.2))
            .add_channel(CorrelatedNoiseChannel::new(0.2, 0.5))
            .add_channel(
                PerGatePauliChannel::new()
                    .with_base(0.2, 0.2)
                    .with_meas_init(0.2, 0.2),
            )
            .add_channel(ImportanceSamplingChannel::new(
                SingleQubitChannel::depolarizing(0.2),
                ImportanceConfig::with_boost(0.02, 10.0),
            ))
            .add_channel(
                CompositeChannel::new("composite", pauli())
                    .with_filter(CompositeEventFilter::BeforeGate)
                    .with_filter(CompositeEventFilter::AfterReset)
                    .with_filter(CompositeEventFilter::BetweenLayers),
            )
            .add_channel(
                BatchCompositeChannel::new("batch", 0.2, pauli())
                    .with_filter(CompositeEventFilter::AnyGate),
            )
            .add_channel(
                CompositeCrosstalkChannel::new("crosstalk", pauli())
                    .responds_to_gates()
                    .responds_to_measurement()
                    .responds_to_preparation(),
            )
    }

    #[test]
    fn bucketed_and_all_kind_dispatch_and_clone_emit_identically() {
        for idle_family in 0..3 {
            let mut bucketed = equivalence_model(idle_family);
            let mut flat = equivalence_model(idle_family);
            flat.channels = flat
                .channels
                .into_iter()
                .map(|channel| Box::new(AllChannel(channel)) as Box<dyn NoiseChannel>)
                .collect();
            flat.event_handlers = flat
                .event_handlers
                .into_iter()
                .map(|handler| Box::new(AllHandler(handler)) as Box<dyn EventHandler>)
                .collect();
            flat.channel_buckets = std::array::from_fn(|_| Vec::new());
            for (index, channel) in flat.channels.iter().enumerate() {
                index_kinds(&mut flat.channel_buckets, index, channel.event_kinds());
            }
            flat.rebuild_handler_buckets();
            assert_eq!(bucketed.channel_names(), flat.channel_names());
            assert_eq!(bucketed.gate_requirements, flat.gate_requirements);
            assert_eq!(bucketed.describe(), flat.describe());
            let mut cloned = bucketed.clone();
            assert_eq!(bucketed.channel_buckets, cloned.channel_buckets);
            assert_eq!(bucketed.handler_buckets, cloned.handler_buckets);
            let mut rng = PecosRng::seed_from_u64(12345);
            let mut flat_rng = PecosRng::seed_from_u64(12345);
            let mut clone_rng = PecosRng::seed_from_u64(12345);
            let mut events = vec![NoiseEvent::AfterPreparation {
                qubits: &[QubitId(0), QubitId(1), QubitId(2), QubitId(3)],
            }];
            events.extend(witnesses());
            for _ in 0..4 {
                for event in &events {
                    let response = bucketed.emit(event, &mut rng);
                    assert_same_response(&response, &flat.emit(event, &mut flat_rng));
                    assert_same_response(&response, &cloned.emit(event, &mut clone_rng));
                }
            }
            let next = rng.random::<u64>();
            assert_eq!(next, flat_rng.random::<u64>());
            assert_eq!(next, clone_rng.random::<u64>());
        }
    }

    use super::*;
    use crate::command::{GateCommand, GateType};
    use crate::noise::plugins::CorePlugin;
    use pecos_core::QubitId;
    use rand::RngExt;

    // Simple test channel that always responds with an X gate
    #[derive(Clone)]
    struct TestChannel {
        probability: f64,
    }

    impl NoiseChannel for TestChannel {
        fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
            matches!(event, NoiseEvent::AfterGate { .. })
        }

        fn apply(
            &self,
            event: &NoiseEvent<'_>,
            _ctx: &mut NoiseContext,
            rng: &mut PecosRng,
        ) -> NoiseResponse {
            if let NoiseEvent::AfterGate { qubits, .. } = event
                && rng.random::<f64>() < self.probability
            {
                return NoiseResponse::inject_gate(GateCommand::x(qubits[0]));
            }
            NoiseResponse::None
        }

        fn name(&self) -> &'static str {
            "TestChannel"
        }

        fn clone_box(&self) -> Box<dyn NoiseChannel> {
            Box::new(self.clone())
        }
    }

    #[test]
    fn test_composable_noise_model() {
        let model = ComposableNoiseModel::new().add_channel(TestChannel { probability: 1.0 });

        assert_eq!(model.channel_count(), 1);
    }

    #[test]
    fn test_emit_event() {
        let mut model = ComposableNoiseModel::new().add_channel(TestChannel { probability: 1.0 });

        let qubits = [QubitId(0)];
        let angles = [];
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::H,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
        };

        let mut rng = PecosRng::seed_from_u64(42);
        let response = model.emit(&event, &mut rng);

        assert!(matches!(response, NoiseResponse::InjectGates(_)));
    }

    #[test]
    fn test_plugin_based_model() {
        let model = ComposableNoiseModel::new()
            .add_plugin(&CorePlugin)
            .add_channel(TestChannel { probability: 1.0 });

        assert_eq!(model.event_handler_count(), 2); // Prep + Meas handlers
        assert_eq!(model.channel_count(), 1);
    }

    #[test]
    fn test_core_plugin_state_tracking() {
        let mut model = ComposableNoiseModel::new().add_plugin(&CorePlugin);

        // Emit preparation event
        let qubits = [QubitId(0)];
        let prep_event = NoiseEvent::AfterPreparation { qubits: &qubits };

        let mut rng = PecosRng::seed_from_u64(42);
        model.emit(&prep_event, &mut rng);

        // Qubit should now be tracked as active
        assert!(model.context().is_active(QubitId(0)));
        assert!(model.context().exists(QubitId(0)));

        // Emit measurement event
        let outcomes = [false];
        let meas_event = NoiseEvent::AfterMeasurement {
            qubits: &qubits,
            outcomes: &outcomes,
        };
        model.emit(&meas_event, &mut rng);

        // Qubit should now be inactive
        assert!(!model.context().is_active(QubitId(0)));
        assert!(model.context().exists(QubitId(0))); // Still exists
    }
}
