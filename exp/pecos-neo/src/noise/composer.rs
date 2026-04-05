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
use super::{NoiseChannel, NoiseEvent, NoiseResponse};
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
        f.debug_struct("ComposableNoiseModel")
            .field("time_scale", &self.time_scale)
            .field("event_handler_count", &self.event_handlers.len())
            .field(
                "event_handler_names",
                &self
                    .event_handlers
                    .iter()
                    .map(|h| h.name())
                    .collect::<Vec<_>>(),
            )
            .field("channel_count", &self.channels.len())
            .field(
                "channel_names",
                &self.channels.iter().map(|c| c.name()).collect::<Vec<_>>(),
            )
            .field("observer_count", &self.observers.len())
            .field("context", &self.context)
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
        self.channels.extend(config.channels);
        self.observers.extend(config.observers);

        // Keep handlers sorted by priority (high to low) for efficient iteration
        self.event_handlers
            .sort_by_key(|h| std::cmp::Reverse(h.priority()));

        self
    }

    /// Add a noise channel directly to the model.
    ///
    /// For plugin-based configuration, use `add_plugin()` instead.
    #[must_use]
    pub fn add_channel(mut self, channel: impl NoiseChannel + 'static) -> Self {
        self.channels.push(Box::new(channel));
        self
    }

    /// Add a pre-boxed noise channel to the model.
    ///
    /// This is useful when you have a `Box<dyn NoiseChannel>` from a builder
    /// or other source. For most cases, use [`Self::add_channel`] instead.
    #[must_use]
    pub fn add_boxed_channel(mut self, channel: Box<dyn NoiseChannel>) -> Self {
        self.channels.push(channel);
        self
    }

    /// Add an event handler directly to the model.
    ///
    /// For plugin-based configuration, use `add_plugin()` instead.
    #[must_use]
    pub fn add_event_handler(mut self, handler: impl EventHandler + 'static) -> Self {
        self.event_handlers.push(Box::new(handler));
        self
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
    /// Requires `with_time_scale()` to be called first.
    ///
    /// # Arguments
    /// * `t1_seconds` - T1 relaxation time in seconds
    /// * `t2_seconds` - T2 dephasing time in seconds
    ///
    /// # Panics
    /// Panics if `with_time_scale()` has not been called.
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
        // Convert physical times to time units
        let t1_units = scale.from_seconds(t1_seconds).as_f64();
        let t2_units = scale.from_seconds(t2_seconds).as_f64();
        let channel = IdleChannel::from_t1_t2(t1_units, t2_units);
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
        for channel in &self.channels {
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
        let handler_indices: smallvec::SmallVec<[usize; 4]> = self
            .event_handlers
            .iter()
            .enumerate()
            .filter(|(_, h)| h.handles(event))
            .map(|(i, _)| i)
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

// ============================================================================
// From implementations for ergonomic noise model construction
// ============================================================================

impl Clone for ComposableNoiseModel {
    fn clone(&self) -> Self {
        Self {
            event_handlers: self.event_handlers.iter().map(|h| h.clone_box()).collect(),
            channels: self.channels.iter().map(|c| c.clone_box()).collect(),
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
    /// sim_neo(circuit.clone()).noise(GeneralNoiseModelBuilder::new().with_p1(0.01).build());
    /// sim_neo(circuit).noise(GeneralNoiseModelBuilder::new().with_p1(0.01));  // No .build()!
    /// ```
    fn from(builder: super::GeneralNoiseModelBuilder) -> Self {
        builder.build()
    }
}

#[cfg(test)]
mod tests {
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
