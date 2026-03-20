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

//! Bridge between composite primitives and the noise channel system.
//!
//! This module provides `CompositeChannel`, an adapter that wraps a composite `Primitive`
//! and implements the `NoiseChannel` trait, allowing composite-based noise models
//! to be used with the existing `ComposableNoiseModel`.

use super::Primitive;
use super::batch::GeometricSampler;
use super::response::CompositeResponse;
use crate::noise::{NoiseChannel, NoiseContext, NoiseEvent, NoiseResponse};
use pecos_core::QubitId;
use pecos_rng::PecosRng;
use smallvec::smallvec;

/// Events that a composite channel can respond to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositeEventFilter {
    /// Respond to single-qubit gates (after gate).
    SingleQubitGate,
    /// Respond to two-qubit gates (after gate).
    TwoQubitGate,
    /// Respond to all gates (after gate).
    AnyGate,
    /// Respond to preparation events.
    Preparation,
    /// Respond to measurement events (before measurement).
    BeforeMeasurement,
    /// Respond to measurement events (after measurement).
    AfterMeasurement,
    /// Respond to idle time events.
    IdleTime,
    /// Respond to before-gate events (for skip logic).
    BeforeGate,
    /// Respond to reset events (mid-circuit reset).
    AfterReset,
    /// Respond to circuit start events.
    BeforeCircuit,
    /// Respond to circuit end events.
    AfterCircuit,
    /// Respond to between-layer events.
    BetweenLayers,
}

impl CompositeEventFilter {
    /// Check if this filter matches the given event.
    #[allow(clippy::match_same_arms)] // Arms intentionally separate for clarity
    #[must_use]
    pub fn matches(self, event: &NoiseEvent<'_>) -> bool {
        match (self, event) {
            (Self::SingleQubitGate, NoiseEvent::AfterGate { qubits, .. }) => qubits.len() == 1,
            (Self::TwoQubitGate, NoiseEvent::AfterGate { qubits, .. }) => qubits.len() == 2,
            (Self::AnyGate, NoiseEvent::AfterGate { .. }) => true,
            (Self::Preparation, NoiseEvent::AfterPreparation { .. }) => true,
            (Self::BeforeMeasurement, NoiseEvent::BeforeMeasurement { .. }) => true,
            (Self::AfterMeasurement, NoiseEvent::AfterMeasurement { .. }) => true,
            (Self::IdleTime, NoiseEvent::IdleTime { .. }) => true,
            (Self::BeforeGate, NoiseEvent::BeforeGate { .. }) => true,
            (Self::AfterReset, NoiseEvent::AfterReset { .. }) => true,
            (Self::BeforeCircuit, NoiseEvent::BeforeCircuit { .. }) => true,
            (Self::AfterCircuit, NoiseEvent::AfterCircuit { .. }) => true,
            (Self::BetweenLayers, NoiseEvent::BetweenLayers { .. }) => true,
            _ => false,
        }
    }
}

/// A noise channel backed by a composite primitive.
///
/// This adapter allows composite-based primitives to be used within the
/// `ComposableNoiseModel` system. The primitive is applied to each
/// qubit involved in matching events.
///
/// # Performance
///
/// When a probability is set via `.with_probability()`, the channel uses
/// geometric sampling for O(n*p) complexity instead of O(n). This enables
/// efficient processing of millions of qubits:
///
/// - 1M qubits at p=1e-4: ~7 µs (vs ~16 seconds with linear)
/// - Automatically falls back to linear for high probability or small counts
///
/// # Example
///
/// ```
/// use pecos_neo::noise::composite::prelude::*;
/// use pecos_neo::noise::composite::channel::{CompositeChannel, CompositeEventFilter};
/// use pecos_neo::noise::ComposableNoiseModel;
///
/// // Fast path: probability at channel level (geometric sampling)
/// let channel = CompositeChannel::new("sq_depolarizing", pauli())
///     .with_probability(0.01)
///     .with_filter(CompositeEventFilter::SingleQubitGate);
///
/// // Or with complex primitives (no probability = linear, applies to all)
/// let skip_channel = CompositeChannel::new("skip_leaked", skip_if_leaked())
///     .with_filter(CompositeEventFilter::BeforeGate);
///
/// let model = ComposableNoiseModel::new()
///     .add_channel(channel)
///     .add_channel(skip_channel);
/// ```
pub struct CompositeChannel<P: Primitive> {
    name: &'static str,
    primitive: P,
    filters: Vec<CompositeEventFilter>,
    priority: i32,
    /// Optional geometric sampler for fast batch processing.
    sampler: Option<GeometricSampler>,
    /// Threshold below which geometric sampling is used (default: 0.01).
    geometric_threshold: f64,
    /// Minimum qubit count for geometric sampling (default: 100).
    min_qubits_for_geometric: usize,
}

impl<P: Primitive + Clone> Clone for CompositeChannel<P> {
    fn clone(&self) -> Self {
        Self {
            name: self.name,
            primitive: self.primitive.clone(),
            filters: self.filters.clone(),
            priority: self.priority,
            sampler: self.sampler,
            geometric_threshold: self.geometric_threshold,
            min_qubits_for_geometric: self.min_qubits_for_geometric,
        }
    }
}

impl<P: Primitive> CompositeChannel<P> {
    /// Create a new composite channel with the given name and primitive.
    #[must_use]
    pub fn new(name: &'static str, primitive: P) -> Self {
        Self {
            name,
            primitive,
            filters: Vec::new(),
            priority: 0,
            sampler: None,
            geometric_threshold: 0.01,
            min_qubits_for_geometric: 100,
        }
    }

    /// Set the probability for this channel, enabling geometric sampling.
    ///
    /// When set, the channel uses O(n*p) geometric sampling instead of O(n)
    /// linear iteration. This is dramatically faster for low probabilities
    /// and high qubit counts.
    ///
    /// **Important**: When using `.with_probability()`, the primitive should
    /// NOT include a `prob()` wrapper - the channel handles probability filtering.
    #[must_use]
    pub fn with_probability(mut self, probability: f64) -> Self {
        if probability > 0.0 && probability < 1.0 {
            self.sampler = Some(GeometricSampler::new(probability));
        }
        self
    }

    /// Add an event filter that this channel responds to.
    #[must_use]
    pub fn with_filter(mut self, filter: CompositeEventFilter) -> Self {
        self.filters.push(filter);
        self
    }

    /// Set the priority of this channel (higher = earlier).
    #[must_use]
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Set the probability threshold below which geometric sampling is used.
    #[must_use]
    pub fn with_geometric_threshold(mut self, threshold: f64) -> Self {
        self.geometric_threshold = threshold;
        self
    }

    /// Set the minimum qubit count for geometric sampling.
    #[must_use]
    pub fn with_min_qubits(mut self, min: usize) -> Self {
        self.min_qubits_for_geometric = min;
        self
    }

    /// Check if geometric sampling should be used.
    fn use_geometric(&self, num_qubits: usize) -> bool {
        if let Some(sampler) = &self.sampler {
            sampler.probability() < self.geometric_threshold
                && num_qubits >= self.min_qubits_for_geometric
        } else {
            false
        }
    }

    /// Apply using geometric sampling (O(n*p) - fast for low probability).
    #[inline]
    fn apply_geometric(
        &self,
        qubits: &[QubitId],
        event: &NoiseEvent<'_>,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> NoiseResponse {
        let sampler = self
            .sampler
            .as_ref()
            .expect("sampler must be set for geometric");
        let indices = sampler.sample_range(0, qubits.len(), rng);

        if indices.is_empty() {
            return NoiseResponse::None;
        }

        // For measurement events, get outcomes (rare in geometric path)
        let outcomes = match event {
            NoiseEvent::AfterMeasurement { outcomes, .. } => Some(*outcomes),
            _ => None,
        };

        // Fast path: process candidates without expensive context updates
        // Note: We skip set_current_qubit_index to avoid O(n) copies per event
        let mut combined = NoiseResponse::None;
        for idx in indices {
            let qubit = qubits[idx];

            // Only set outcome context for measurement events
            if let Some(outcomes) = outcomes
                && idx < outcomes.len()
            {
                ctx.set_current_outcome(outcomes[idx]);
            }

            let composite_response = self.primitive.apply(qubit, ctx, rng);
            if !composite_response.is_none() {
                let noise_response = composite_to_noise_response(composite_response, qubit);
                combined = combined.combine(noise_response);
            }

            if outcomes.is_some() {
                ctx.clear_current_outcome();
            }
        }

        combined
    }

    /// Apply using linear iteration (O(n) - for high probability or small counts).
    #[inline]
    fn apply_linear(
        &self,
        qubits: &[QubitId],
        event: &NoiseEvent<'_>,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> NoiseResponse {
        let outcomes = match event {
            NoiseEvent::AfterMeasurement { outcomes, .. } => Some(*outcomes),
            _ => None,
        };

        // If we have a sampler but conditions don't favor geometric, use threshold check
        let threshold = self.sampler.as_ref().map(GeometricSampler::threshold);

        // Clear fired flags before processing (for two-stage primitives)
        ctx.clear_fired_flags();

        // Check if this primitive needs two-pass processing
        if self.primitive.needs_two_pass() {
            return self.apply_two_pass(qubits, outcomes, threshold, ctx, rng);
        }

        // Single-pass processing (standard)
        let mut combined = NoiseResponse::None;
        for (i, &qubit) in qubits.iter().enumerate() {
            // If probability is set, do threshold check
            if let Some(thresh) = threshold
                && rng.next_u64() >= thresh
            {
                continue;
            }

            // Set qubit index for correlated actions
            ctx.set_current_qubit_index(i, qubits);

            // Only set outcome context for measurement events
            if let Some(outcomes) = outcomes
                && i < outcomes.len()
            {
                ctx.set_current_outcome(outcomes[i]);
            }

            let composite_response = self.primitive.apply(qubit, ctx, rng);
            let noise_response = composite_to_noise_response(composite_response, qubit);
            combined = combined.combine(noise_response);

            if outcomes.is_some() {
                ctx.clear_current_outcome();
            }
        }

        combined
    }

    /// Apply using two-pass processing for `TwoStage` primitives.
    ///
    /// Pass 1: Run `apply_stage1` on all qubits (sampling phase)
    /// Pass 2: Run `apply_stage2` on all qubits (effect phase)
    #[inline]
    fn apply_two_pass(
        &self,
        qubits: &[QubitId],
        outcomes: Option<&[bool]>,
        threshold: Option<u64>,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> NoiseResponse {
        let mut combined = NoiseResponse::None;

        // === Pass 1: Stage 1 on all qubits ===
        for (i, &qubit) in qubits.iter().enumerate() {
            // If probability is set, do threshold check
            // Note: For two-stage, the probability is typically in stage1
            if let Some(thresh) = threshold
                && rng.next_u64() >= thresh
            {
                continue;
            }

            ctx.set_current_qubit_index(i, qubits);

            if let Some(outcomes) = outcomes
                && i < outcomes.len()
            {
                ctx.set_current_outcome(outcomes[i]);
            }

            let composite_response = self.primitive.apply_stage1(qubit, ctx, rng);
            let noise_response = composite_to_noise_response(composite_response, qubit);
            combined = combined.combine(noise_response);

            if outcomes.is_some() {
                ctx.clear_current_outcome();
            }
        }

        // === Pass 2: Stage 2 on all qubits ===
        for (i, &qubit) in qubits.iter().enumerate() {
            ctx.set_current_qubit_index(i, qubits);

            if let Some(outcomes) = outcomes
                && i < outcomes.len()
            {
                ctx.set_current_outcome(outcomes[i]);
            }

            let composite_response = self.primitive.apply_stage2(qubit, ctx, rng);
            let noise_response = composite_to_noise_response(composite_response, qubit);
            combined = combined.combine(noise_response);

            if outcomes.is_some() {
                ctx.clear_current_outcome();
            }
        }

        combined
    }
}

impl<P: Primitive + Clone + 'static> NoiseChannel for CompositeChannel<P> {
    fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
        self.filters.iter().any(|f| f.matches(event))
    }

    fn apply(
        &self,
        event: &NoiseEvent<'_>,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> NoiseResponse {
        let qubits = event.qubits();
        if qubits.is_empty() {
            return NoiseResponse::None;
        }

        // For gate events, set gate context for dynamic probability primitives
        match event {
            NoiseEvent::BeforeGate {
                gate_type, angles, ..
            }
            | NoiseEvent::AfterGate {
                gate_type, angles, ..
            } => {
                ctx.set_current_gate(*gate_type, angles, qubits.len());
            }
            NoiseEvent::IdleTime { duration, .. } => {
                ctx.set_current_idle(*duration);
            }
            _ => {}
        }

        // Choose processing strategy based on probability and qubit count
        let result = if self.use_geometric(qubits.len()) {
            self.apply_geometric(qubits, event, ctx, rng)
        } else {
            self.apply_linear(qubits, event, ctx, rng)
        };

        // Clear context after processing
        ctx.clear_current_gate();
        ctx.clear_current_idle();
        ctx.clear_correlation();

        result
    }

    fn name(&self) -> &'static str {
        self.name
    }

    fn priority(&self) -> i32 {
        self.priority
    }

    fn clone_box(&self) -> Box<dyn NoiseChannel> {
        Box::new(self.clone())
    }
}

/// Convert a `CompositeResponse` to a `NoiseResponse`.
///
/// The qubit parameter is used for leak/unleak/flip responses since
/// `CompositeResponse` doesn't track which qubit is affected (it's applied per-qubit).
fn composite_to_noise_response(response: CompositeResponse, qubit: QubitId) -> NoiseResponse {
    match response {
        CompositeResponse::None => NoiseResponse::None,
        CompositeResponse::SkipGate => NoiseResponse::SkipGate,
        CompositeResponse::Leak => NoiseResponse::MarkLeaked(smallvec![qubit]),
        CompositeResponse::Unleak => NoiseResponse::MarkUnleaked(smallvec![qubit]),
        CompositeResponse::FlipOutcome => NoiseResponse::FlipOutcomes(smallvec![qubit]),
        CompositeResponse::ForceOutcome(value) => {
            // Force the measurement outcome to a specific value
            NoiseResponse::ForceOutcomes(smallvec![(qubit, value)])
        }
        CompositeResponse::LeakedMeasurement => NoiseResponse::LeakedMeasurement(smallvec![qubit]),
        CompositeResponse::InjectGates(gates) => {
            if gates.is_empty() {
                NoiseResponse::None
            } else {
                NoiseResponse::InjectGates(Box::new(gates.into_iter().collect()))
            }
        }
        CompositeResponse::Multiple(responses) => {
            let converted: Vec<NoiseResponse> = responses
                .into_iter()
                .map(|r| composite_to_noise_response(r, qubit))
                .filter(|r| !r.is_none())
                .collect();

            match converted.len() {
                0 => NoiseResponse::None,
                1 => converted
                    .into_iter()
                    .next()
                    .expect("len is 1, so next() returns Some"),
                _ => NoiseResponse::Multiple(converted),
            }
        }
    }
}

/// Builder for creating composite-based noise channels with common patterns.
pub struct CompositeChannelBuilder;

impl CompositeChannelBuilder {
    /// Create a single-qubit gate noise channel.
    ///
    /// The primitive is applied after each single-qubit gate.
    #[must_use]
    pub fn single_qubit<P: Primitive>(name: &'static str, primitive: P) -> CompositeChannel<P> {
        CompositeChannel::new(name, primitive).with_filter(CompositeEventFilter::SingleQubitGate)
    }

    /// Create a two-qubit gate noise channel.
    ///
    /// The primitive is applied after each two-qubit gate.
    #[must_use]
    pub fn two_qubit<P: Primitive>(name: &'static str, primitive: P) -> CompositeChannel<P> {
        CompositeChannel::new(name, primitive).with_filter(CompositeEventFilter::TwoQubitGate)
    }

    /// Create a gate noise channel that responds to all gates.
    ///
    /// The primitive is applied after each gate.
    #[must_use]
    pub fn any_gate<P: Primitive>(name: &'static str, primitive: P) -> CompositeChannel<P> {
        CompositeChannel::new(name, primitive).with_filter(CompositeEventFilter::AnyGate)
    }

    /// Create a preparation noise channel.
    ///
    /// The primitive is applied after each preparation.
    #[must_use]
    pub fn preparation<P: Primitive>(name: &'static str, primitive: P) -> CompositeChannel<P> {
        CompositeChannel::new(name, primitive).with_filter(CompositeEventFilter::Preparation)
    }

    /// Create a before-gate channel (for skip logic).
    ///
    /// The primitive is applied before each gate, typically used for
    /// leakage-based gate skipping.
    #[must_use]
    pub fn before_gate<P: Primitive>(name: &'static str, primitive: P) -> CompositeChannel<P> {
        CompositeChannel::new(name, primitive)
            .with_filter(CompositeEventFilter::BeforeGate)
            .with_priority(100) // High priority to run before other channels
    }

    /// Create an idle noise channel.
    ///
    /// The primitive is applied during idle periods. Use `prob_linear` and
    /// `prob_quadratic` primitives for time-dependent error rates.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_neo::noise::composite::prelude::*;
    ///
    /// // T1-like relaxation with linear time dependence
    /// let t1_channel = CompositeChannelBuilder::idle("t1_decay",
    ///     prob_linear(0.001, inject_z())
    /// );
    /// ```
    #[must_use]
    pub fn idle<P: Primitive>(name: &'static str, primitive: P) -> CompositeChannel<P> {
        CompositeChannel::new(name, primitive).with_filter(CompositeEventFilter::IdleTime)
    }

    /// Create a measurement noise channel (before measurement).
    ///
    /// The primitive is applied before each measurement.
    #[must_use]
    pub fn before_measurement<P: Primitive>(
        name: &'static str,
        primitive: P,
    ) -> CompositeChannel<P> {
        CompositeChannel::new(name, primitive).with_filter(CompositeEventFilter::BeforeMeasurement)
    }

    /// Create a measurement noise channel (after measurement).
    ///
    /// The primitive is applied after each measurement.
    #[must_use]
    pub fn after_measurement<P: Primitive>(
        name: &'static str,
        primitive: P,
    ) -> CompositeChannel<P> {
        CompositeChannel::new(name, primitive).with_filter(CompositeEventFilter::AfterMeasurement)
    }

    /// Create a reset noise channel (after mid-circuit reset).
    ///
    /// The primitive is applied after each reset operation.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_neo::noise::composite::prelude::*;
    ///
    /// // Apply preparation-like noise after reset
    /// let reset_channel = CompositeChannelBuilder::after_reset("reset_noise",
    ///     prob(0.001, pauli())
    /// );
    /// ```
    #[must_use]
    pub fn after_reset<P: Primitive>(name: &'static str, primitive: P) -> CompositeChannel<P> {
        CompositeChannel::new(name, primitive).with_filter(CompositeEventFilter::AfterReset)
    }

    /// Create a before-circuit channel.
    ///
    /// The primitive is applied once at the start of circuit execution.
    /// Useful for initializing noise model state or applying initial errors.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_neo::noise::composite::prelude::*;
    ///
    /// // Initialize all qubits with small preparation error
    /// let init_channel = CompositeChannelBuilder::before_circuit("initial_noise",
    ///     prob(0.001, pauli())
    /// );
    /// ```
    #[must_use]
    pub fn before_circuit<P: Primitive>(name: &'static str, primitive: P) -> CompositeChannel<P> {
        CompositeChannel::new(name, primitive).with_filter(CompositeEventFilter::BeforeCircuit)
    }

    /// Create an after-circuit channel.
    ///
    /// The primitive is applied once at the end of circuit execution.
    /// Useful for final measurements or cleanup operations.
    #[must_use]
    pub fn after_circuit<P: Primitive>(name: &'static str, primitive: P) -> CompositeChannel<P> {
        CompositeChannel::new(name, primitive).with_filter(CompositeEventFilter::AfterCircuit)
    }

    /// Create a between-layers channel.
    ///
    /// The primitive is applied between circuit layers.
    /// Useful for idle noise that accumulates between synchronization points.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_neo::noise::composite::prelude::*;
    ///
    /// // Apply dephasing between circuit layers
    /// let layer_channel = CompositeChannelBuilder::between_layers("layer_dephasing",
    ///     prob(0.0005, inject_z())
    /// );
    /// ```
    #[must_use]
    pub fn between_layers<P: Primitive>(name: &'static str, primitive: P) -> CompositeChannel<P> {
        CompositeChannel::new(name, primitive).with_filter(CompositeEventFilter::BetweenLayers)
    }
}

/// Convenience type alias for boxed composite channels.
pub type BoxedCompositeChannel = CompositeChannel<Box<dyn Primitive>>;

// ============================================================================
// Batch Flow Channel - Geometric Sampling Integration (Legacy)
// ============================================================================

/// A composite channel optimized for high qubit counts with geometric sampling.
///
/// **Note**: This is now a legacy alias. Use `CompositeChannel::with_probability()`
/// instead for the same functionality with a cleaner API.
///
/// This channel uses the geometric sampling optimization for low-probability
/// noise events. Instead of iterating over all qubits and checking probability
/// for each one, it uses geometric sampling to find only the affected qubits
/// in O(n*p) time instead of O(n).
///
/// # Performance
///
/// For 1M qubits at p=1e-5: ~200ns (vs ~700µs for linear approach)
///
/// # Usage
///
/// Use `BatchCompositeChannel` when:
/// - Processing many qubits (10K+)
/// - Probability is low (p < 0.01)
/// - The inner primitive doesn't need to see non-affected qubits
///
/// Use regular `CompositeChannel` when:
/// - Small qubit counts
/// - High probability events
/// - The primitive needs to see all qubits (e.g., for correlation tracking)
///
/// # Example
///
/// ```
/// use pecos_neo::noise::composite::prelude::*;
/// use pecos_neo::noise::composite::channel::BatchCompositeChannel;
///
/// // Create a batch channel for single-qubit depolarizing
/// let noise = pauli();  // Just the Pauli selection, no prob wrapper
/// let channel = BatchCompositeChannel::new("batch_depol", 1e-4, noise)
///     .with_filter(CompositeEventFilter::SingleQubitGate);
///
/// // With 1M qubits, this runs in ~1µs instead of ~700µs
/// ```
pub struct BatchCompositeChannel<P: Primitive> {
    name: &'static str,
    sampler: GeometricSampler,
    /// The inner primitive to apply to affected qubits.
    /// NOTE: This should NOT include a probability wrapper - the batch
    /// channel handles probability filtering via geometric sampling.
    primitive: P,
    filters: Vec<CompositeEventFilter>,
    priority: i32,
    /// Threshold below which geometric sampling is used (default: 0.01)
    geometric_threshold: f64,
    /// Minimum qubit count for geometric sampling (default: 100)
    min_qubits_for_geometric: usize,
}

impl<P: Primitive + Clone> Clone for BatchCompositeChannel<P> {
    fn clone(&self) -> Self {
        Self {
            name: self.name,
            sampler: self.sampler,
            primitive: self.primitive.clone(),
            filters: self.filters.clone(),
            priority: self.priority,
            geometric_threshold: self.geometric_threshold,
            min_qubits_for_geometric: self.min_qubits_for_geometric,
        }
    }
}

impl<P: Primitive> BatchCompositeChannel<P> {
    /// Create a new batch composite channel.
    ///
    /// # Arguments
    /// * `name` - Channel name for debugging
    /// * `probability` - Error probability (used for geometric sampling)
    /// * `primitive` - The inner primitive to apply (should NOT include prob wrapper)
    ///
    /// # Panics
    /// Panics if probability is not in (0, 1).
    #[must_use]
    pub fn new(name: &'static str, probability: f64, primitive: P) -> Self {
        Self {
            name,
            sampler: GeometricSampler::new(probability.clamp(1e-15, 1.0 - 1e-15)),
            primitive,
            filters: Vec::new(),
            priority: 0,
            geometric_threshold: 0.01,
            min_qubits_for_geometric: 100,
        }
    }

    /// Add an event filter.
    #[must_use]
    pub fn with_filter(mut self, filter: CompositeEventFilter) -> Self {
        self.filters.push(filter);
        self
    }

    /// Set the priority.
    #[must_use]
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Set the probability threshold below which geometric sampling is used.
    #[must_use]
    pub fn with_geometric_threshold(mut self, threshold: f64) -> Self {
        self.geometric_threshold = threshold;
        self
    }

    /// Set the minimum qubit count for geometric sampling.
    #[must_use]
    pub fn with_min_qubits(mut self, min: usize) -> Self {
        self.min_qubits_for_geometric = min;
        self
    }

    /// Get the probability.
    #[must_use]
    pub fn probability(&self) -> f64 {
        self.sampler.probability()
    }

    /// Determine whether to use geometric sampling for this event.
    fn use_geometric(&self, num_qubits: usize) -> bool {
        self.sampler.probability() < self.geometric_threshold
            && num_qubits >= self.min_qubits_for_geometric
    }

    /// Process qubits using geometric sampling, inline without Vec allocation.
    ///
    /// This is the fast path for low-probability, high-qubit-count scenarios.
    /// Samples indices directly and processes candidates inline.
    #[inline]
    fn process_geometric_inline(
        &self,
        qubits: &[QubitId],
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> NoiseResponse {
        // Sample indices using geometric distribution (O(n*p))
        let indices = self.sampler.sample_range(0, qubits.len(), rng);

        if indices.is_empty() {
            return NoiseResponse::None;
        }

        // Process candidates inline, combining responses
        let mut combined = NoiseResponse::None;
        for idx in indices {
            let qubit = qubits[idx];
            let composite_response = self.primitive.apply(qubit, ctx, rng);
            if !composite_response.is_none() {
                let noise_response = composite_to_noise_response(composite_response, qubit);
                combined = combined.combine(noise_response);
            }
        }

        combined
    }

    /// Process qubits using linear scanning (fallback for high probability or small counts).
    #[inline]
    fn process_linear_inline(
        &self,
        qubits: &[QubitId],
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> NoiseResponse {
        let prob = self.sampler.probability();
        let threshold = self.sampler.threshold();

        let mut combined = NoiseResponse::None;
        for &qubit in qubits {
            // Use u64 threshold comparison for speed
            if rng.next_u64() < threshold {
                let composite_response = self.primitive.apply(qubit, ctx, rng);
                if !composite_response.is_none() {
                    let noise_response = composite_to_noise_response(composite_response, qubit);
                    combined = combined.combine(noise_response);
                }
            }
        }

        // Handle edge case: if prob >= 1.0, threshold wraps and we miss everything
        // This shouldn't happen with proper clamping, but be safe
        if prob >= 1.0 {
            for &qubit in qubits {
                let composite_response = self.primitive.apply(qubit, ctx, rng);
                if !composite_response.is_none() {
                    let noise_response = composite_to_noise_response(composite_response, qubit);
                    combined = combined.combine(noise_response);
                }
            }
        }

        combined
    }
}

impl<P: Primitive + Clone + 'static> NoiseChannel for BatchCompositeChannel<P> {
    fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
        self.filters.iter().any(|f| f.matches(event))
    }

    fn apply(
        &self,
        event: &NoiseEvent<'_>,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> NoiseResponse {
        let qubits = event.qubits();
        if qubits.is_empty() {
            return NoiseResponse::None;
        }

        // For gate events, set gate context
        match event {
            NoiseEvent::BeforeGate {
                gate_type, angles, ..
            }
            | NoiseEvent::AfterGate {
                gate_type, angles, ..
            } => {
                ctx.set_current_gate(*gate_type, angles, qubits.len());
            }
            NoiseEvent::IdleTime { duration, .. } => {
                ctx.set_current_idle(*duration);
            }
            _ => {}
        }

        // Optimized inline processing: sample indices and process without collecting
        let result = if self.use_geometric(qubits.len()) {
            // Geometric sampling: generate candidate indices directly (O(n*p))
            // Process inline without intermediate Vec allocation
            self.process_geometric_inline(qubits, ctx, rng)
        } else {
            // Linear fallback for small counts or high probability
            self.process_linear_inline(qubits, ctx, rng)
        };

        ctx.clear_current_gate();
        ctx.clear_current_idle();
        ctx.clear_correlation();

        result
    }

    fn name(&self) -> &'static str {
        self.name
    }

    fn priority(&self) -> i32 {
        self.priority
    }

    fn clone_box(&self) -> Box<dyn NoiseChannel> {
        Box::new(self.clone())
    }
}

/// Builder methods for batch channels.
impl CompositeChannelBuilder {
    /// Create a batch single-qubit gate noise channel with geometric sampling.
    ///
    /// The primitive should NOT include a probability wrapper - the batch
    /// channel handles probability filtering.
    #[must_use]
    pub fn batch_single_qubit<P: Primitive>(
        name: &'static str,
        probability: f64,
        primitive: P,
    ) -> BatchCompositeChannel<P> {
        BatchCompositeChannel::new(name, probability, primitive)
            .with_filter(CompositeEventFilter::SingleQubitGate)
    }

    /// Create a batch two-qubit gate noise channel with geometric sampling.
    #[must_use]
    pub fn batch_two_qubit<P: Primitive>(
        name: &'static str,
        probability: f64,
        primitive: P,
    ) -> BatchCompositeChannel<P> {
        BatchCompositeChannel::new(name, probability, primitive)
            .with_filter(CompositeEventFilter::TwoQubitGate)
    }

    /// Create a batch any-gate noise channel with geometric sampling.
    #[must_use]
    pub fn batch_any_gate<P: Primitive>(
        name: &'static str,
        probability: f64,
        primitive: P,
    ) -> BatchCompositeChannel<P> {
        BatchCompositeChannel::new(name, probability, primitive)
            .with_filter(CompositeEventFilter::AnyGate)
    }

    /// Create a batch idle noise channel with geometric sampling.
    #[must_use]
    pub fn batch_idle<P: Primitive>(
        name: &'static str,
        probability: f64,
        primitive: P,
    ) -> BatchCompositeChannel<P> {
        BatchCompositeChannel::new(name, probability, primitive)
            .with_filter(CompositeEventFilter::IdleTime)
    }

    /// Create a batch reset noise channel with geometric sampling.
    #[must_use]
    pub fn batch_after_reset<P: Primitive>(
        name: &'static str,
        probability: f64,
        primitive: P,
    ) -> BatchCompositeChannel<P> {
        BatchCompositeChannel::new(name, probability, primitive)
            .with_filter(CompositeEventFilter::AfterReset)
    }

    /// Create a batch before-circuit noise channel with geometric sampling.
    #[must_use]
    pub fn batch_before_circuit<P: Primitive>(
        name: &'static str,
        probability: f64,
        primitive: P,
    ) -> BatchCompositeChannel<P> {
        BatchCompositeChannel::new(name, probability, primitive)
            .with_filter(CompositeEventFilter::BeforeCircuit)
    }

    /// Create a batch after-circuit noise channel with geometric sampling.
    #[must_use]
    pub fn batch_after_circuit<P: Primitive>(
        name: &'static str,
        probability: f64,
        primitive: P,
    ) -> BatchCompositeChannel<P> {
        BatchCompositeChannel::new(name, probability, primitive)
            .with_filter(CompositeEventFilter::AfterCircuit)
    }

    /// Create a batch between-layers noise channel with geometric sampling.
    #[must_use]
    pub fn batch_between_layers<P: Primitive>(
        name: &'static str,
        probability: f64,
        primitive: P,
    ) -> BatchCompositeChannel<P> {
        BatchCompositeChannel::new(name, probability, primitive)
            .with_filter(CompositeEventFilter::BetweenLayers)
    }
}

// ============================================================================
// Crosstalk Channel
// ============================================================================

/// Type alias for neighbor functions used in crosstalk channels.
pub type NeighborFn = fn(&[QubitId]) -> Vec<QubitId>;

/// A crosstalk channel that applies noise to qubits other than the operated ones.
///
/// Crosstalk occurs when operations on some qubits affect other qubits in the system.
/// This channel uses a composite primitive to define what noise to apply to each affected qubit.
///
/// # Example
///
/// ```
/// use pecos_neo::noise::composite::prelude::*;
///
/// // Create crosstalk that applies Pauli errors to other qubits during measurement
/// let crosstalk = CompositeCrosstalkChannel::new("meas_crosstalk", prob(0.01, pauli()))
///     .responds_to_measurement()
///     .global();  // Affects all other active qubits
/// ```
pub struct CompositeCrosstalkChannel<P: Primitive> {
    name: &'static str,
    primitive: P,
    events: Vec<CompositeEventFilter>,
    /// If Some, only affect these qubits as neighbors. If None, affect all active qubits.
    neighbor_fn: Option<NeighborFn>,
    priority: i32,
}

impl<P: Primitive + Clone> Clone for CompositeCrosstalkChannel<P> {
    fn clone(&self) -> Self {
        Self {
            name: self.name,
            primitive: self.primitive.clone(),
            events: self.events.clone(),
            neighbor_fn: self.neighbor_fn,
            priority: self.priority,
        }
    }
}

impl<P: Primitive> CompositeCrosstalkChannel<P> {
    /// Create a new crosstalk channel.
    #[must_use]
    pub fn new(name: &'static str, primitive: P) -> Self {
        Self {
            name,
            primitive,
            events: Vec::new(),
            neighbor_fn: None,
            priority: -10, // Lower priority, run after main gate noise
        }
    }

    /// Respond to measurement events.
    #[must_use]
    pub fn responds_to_measurement(mut self) -> Self {
        self.events.push(CompositeEventFilter::AfterMeasurement);
        self
    }

    /// Respond to preparation events.
    #[must_use]
    pub fn responds_to_preparation(mut self) -> Self {
        self.events.push(CompositeEventFilter::Preparation);
        self
    }

    /// Respond to gate events.
    #[must_use]
    pub fn responds_to_gates(mut self) -> Self {
        self.events.push(CompositeEventFilter::AnyGate);
        self
    }

    /// Set as global crosstalk (affects all other active qubits).
    #[must_use]
    pub fn global(mut self) -> Self {
        self.neighbor_fn = None;
        self
    }

    /// Set as local crosstalk with a neighbor function.
    ///
    /// The function takes the gated qubits and returns their neighbors.
    #[must_use]
    pub fn local(mut self, neighbor_fn: NeighborFn) -> Self {
        self.neighbor_fn = Some(neighbor_fn);
        self
    }

    /// Set the priority of this channel.
    #[must_use]
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Get the target qubits for crosstalk.
    fn get_targets(&self, gated_qubits: &[QubitId], ctx: &NoiseContext) -> Vec<QubitId> {
        match self.neighbor_fn {
            Some(neighbor_fn) => {
                let neighbors = neighbor_fn(gated_qubits);
                ctx.local_crosstalk_targets(gated_qubits, &neighbors)
            }
            None => ctx.global_crosstalk_targets(gated_qubits),
        }
    }
}

impl<P: Primitive + Clone + 'static> NoiseChannel for CompositeCrosstalkChannel<P> {
    fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
        self.events.iter().any(|f| f.matches(event))
    }

    fn apply(
        &self,
        event: &NoiseEvent<'_>,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> NoiseResponse {
        let gated_qubits = event.qubits();
        let targets = self.get_targets(gated_qubits, ctx);

        if targets.is_empty() {
            return NoiseResponse::None;
        }

        // Apply the primitive to each target qubit and combine responses
        let mut combined = NoiseResponse::None;
        for target in targets {
            let composite_response = self.primitive.apply(target, ctx, rng);
            let noise_response = composite_to_noise_response(composite_response, target);
            combined = combined.combine(noise_response);
        }

        combined
    }

    fn name(&self) -> &'static str {
        self.name
    }

    fn priority(&self) -> i32 {
        self.priority
    }

    fn clone_box(&self) -> Box<dyn NoiseChannel> {
        Box::new(self.clone())
    }
}

impl CompositeChannelBuilder {
    /// Create a global crosstalk channel that affects all other active qubits.
    ///
    /// The primitive is applied to each other active qubit when the specified
    /// events occur.
    #[must_use]
    pub fn crosstalk<P: Primitive>(
        name: &'static str,
        primitive: P,
    ) -> CompositeCrosstalkChannel<P> {
        CompositeCrosstalkChannel::new(name, primitive).global()
    }

    /// Create a measurement crosstalk channel.
    ///
    /// Applies noise to other qubits during measurement operations.
    #[must_use]
    pub fn measurement_crosstalk<P: Primitive>(
        name: &'static str,
        primitive: P,
    ) -> CompositeCrosstalkChannel<P> {
        CompositeCrosstalkChannel::new(name, primitive)
            .responds_to_measurement()
            .global()
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::command::GateType;
    use crate::noise::composite::prelude::*;
    use crate::seq;

    #[test]
    fn test_flow_channel_single_qubit() {
        let noise = prob(1.0, pauli());
        let channel = CompositeChannelBuilder::single_qubit("test", noise);

        let qubits = [QubitId(0)];
        let angles = [];
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::H,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
        };

        assert!(NoiseChannel::responds_to(&channel, &event));

        // Two-qubit event should not match
        let qubits2 = [QubitId(0), QubitId(1)];
        let event2 = NoiseEvent::AfterGate {
            gate_type: GateType::CX,
            qubits: &qubits2,
            angles: &angles,
            gate_id: None,
        };
        assert!(!NoiseChannel::responds_to(&channel, &event2));
    }

    #[test]
    fn test_flow_channel_applies_noise() {
        // Always inject X gate
        let noise = inject_x();
        let channel = CompositeChannelBuilder::single_qubit("test", noise);

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        let qubits = [QubitId(0)];
        let angles = [];
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::H,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
        };

        let response = channel.apply(&event, &mut ctx, &mut rng);

        match response {
            NoiseResponse::InjectGates(gates) => {
                assert_eq!(gates.len(), 1);
                assert_eq!(gates[0].gate_type, GateType::X);
            }
            _ => panic!("Expected InjectGates response"),
        }
    }

    #[test]
    fn test_flow_channel_skip_gate() {
        // Skip if leaked
        let noise = skip_if_leaked();
        let channel = CompositeChannelBuilder::before_gate("test", noise);

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        // Mark qubit as leaked
        ctx.mark_leaked(QubitId(0));

        let qubits = [QubitId(0)];
        let angles = [];
        let event = NoiseEvent::BeforeGate {
            gate_type: GateType::H,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
        };

        let response = channel.apply(&event, &mut ctx, &mut rng);
        assert!(response.should_skip_gate());
    }

    #[test]
    fn test_flow_channel_leak_response() {
        let noise = prob(1.0, leak());
        let channel = CompositeChannelBuilder::single_qubit("test", noise);

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        let qubits = [QubitId(0)];
        let angles = [];
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::H,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
        };

        let response = channel.apply(&event, &mut ctx, &mut rng);

        match response {
            NoiseResponse::MarkLeaked(qubits) => {
                assert_eq!(qubits.len(), 1);
                assert_eq!(qubits[0], QubitId(0));
            }
            _ => panic!("Expected MarkLeaked response"),
        }
    }

    #[test]
    fn test_flow_channel_two_qubit() {
        let noise = pauli();
        let channel = CompositeChannelBuilder::two_qubit("test", noise);

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        let qubits = [QubitId(0), QubitId(1)];
        let angles = [];
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::CX,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
        };

        assert!(NoiseChannel::responds_to(&channel, &event));

        // Apply noise - should get response for both qubits
        let response = channel.apply(&event, &mut ctx, &mut rng);

        // Should have gates injected (Pauli on each qubit)
        match response {
            NoiseResponse::Multiple(responses) => {
                assert_eq!(responses.len(), 2);
            }
            NoiseResponse::InjectGates(_) => {
                // Also valid if combined into single InjectGates
            }
            _ => panic!("Expected Multiple or InjectGates response"),
        }
    }

    #[test]
    fn test_flow_channel_complex_tree() {
        // Build realistic SQ noise
        let noise = seq![skip_if_leaked(), prob(0.5, when_leaked(seep(), pauli())),];

        let channel = CompositeChannelBuilder::single_qubit("sq_noise", noise);

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        let qubits = [QubitId(0)];
        let angles = [];
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::H,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
        };

        // Run multiple times to get statistical coverage
        let mut pauli_count = 0;
        for _ in 0..1000 {
            let response = channel.apply(&event, &mut ctx, &mut rng);
            match response {
                NoiseResponse::InjectGates(_) => pauli_count += 1,
                NoiseResponse::Multiple(ref rs) => {
                    if rs
                        .iter()
                        .any(|r| matches!(r, NoiseResponse::InjectGates(_)))
                    {
                        pauli_count += 1;
                    }
                }
                _ => {}
            }
        }

        // Should be roughly 50%
        let rate = f64::from(pauli_count) / 1000.0;
        assert!(
            (rate - 0.5).abs() < 0.1,
            "Expected ~50% pauli rate, got {rate}"
        );
    }

    // ========================================================================
    // Crosstalk Channel Tests
    // ========================================================================

    #[test]
    fn test_crosstalk_channel_global() {
        // Create crosstalk that always applies X to other qubits
        let crosstalk = CompositeCrosstalkChannel::new("test_crosstalk", inject_x())
            .responds_to_measurement()
            .global();

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        // Prepare multiple qubits
        ctx.mark_prepared(QubitId(0));
        ctx.mark_prepared(QubitId(1));
        ctx.mark_prepared(QubitId(2));

        // Measurement on qubit 0 should affect qubits 1 and 2
        let qubits = [QubitId(0)];
        let outcomes = [false];
        let event = NoiseEvent::AfterMeasurement {
            qubits: &qubits,
            outcomes: &outcomes,
        };

        assert!(NoiseChannel::responds_to(&crosstalk, &event));

        let response = crosstalk.apply(&event, &mut ctx, &mut rng);

        // Should have X gates injected for qubits 1 and 2
        match response {
            NoiseResponse::InjectGates(gates) => {
                assert_eq!(gates.len(), 2);
                let affected: Vec<_> = gates.iter().map(|g| g.qubits[0]).collect();
                assert!(affected.contains(&QubitId(1)));
                assert!(affected.contains(&QubitId(2)));
                assert!(!affected.contains(&QubitId(0))); // Not the measured qubit
            }
            NoiseResponse::Multiple(responses) => {
                let mut affected = Vec::new();
                for r in responses {
                    if let NoiseResponse::InjectGates(gates) = r {
                        affected.extend(gates.iter().map(|g| g.qubits[0]));
                    }
                }
                assert!(affected.contains(&QubitId(1)));
                assert!(affected.contains(&QubitId(2)));
            }
            _ => panic!("Expected InjectGates or Multiple response"),
        }
    }

    #[test]
    fn test_crosstalk_channel_local() {
        // Define neighbor function: qubit i has neighbors i-1 and i+1
        fn neighbors(gated: &[QubitId]) -> Vec<QubitId> {
            let mut result = Vec::new();
            for &QubitId(q) in gated {
                if q > 0 {
                    result.push(QubitId(q - 1));
                }
                result.push(QubitId(q + 1));
            }
            result
        }

        let crosstalk = CompositeCrosstalkChannel::new("local_crosstalk", inject_z())
            .responds_to_gates()
            .local(neighbors);

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        // Prepare qubits 0-10
        for i in 0..=10 {
            ctx.mark_prepared(QubitId(i));
        }

        // Gate on qubit 5 should only affect neighbors (4 and 6)
        let qubits = [QubitId(5)];
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::H,
            qubits: &qubits,
            angles: &[],
            gate_id: None,
        };

        assert!(NoiseChannel::responds_to(&crosstalk, &event));

        let response = crosstalk.apply(&event, &mut ctx, &mut rng);

        // Should only affect neighbors (4 and 6)
        match response {
            NoiseResponse::InjectGates(gates) => {
                assert_eq!(gates.len(), 2);
                for gate in gates.iter() {
                    let qubit = gate.qubits[0];
                    assert!(
                        qubit == QubitId(4) || qubit == QubitId(6),
                        "Only neighbors should be affected, got {qubit:?}"
                    );
                }
            }
            NoiseResponse::Multiple(responses) => {
                for r in responses {
                    if let NoiseResponse::InjectGates(gates) = r {
                        for gate in gates.iter() {
                            let qubit = gate.qubits[0];
                            assert!(qubit == QubitId(4) || qubit == QubitId(6));
                        }
                    }
                }
            }
            _ => panic!("Expected InjectGates or Multiple response"),
        }
    }

    #[test]
    fn test_crosstalk_channel_probabilistic() {
        // Create crosstalk with 50% probability
        let crosstalk = CompositeCrosstalkChannel::new("prob_crosstalk", prob(0.5, inject_x()))
            .responds_to_preparation()
            .global();

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        // Prepare qubits
        ctx.mark_prepared(QubitId(0));
        ctx.mark_prepared(QubitId(1));
        ctx.mark_prepared(QubitId(2));

        let qubits = [QubitId(0)];
        let event = NoiseEvent::AfterPreparation { qubits: &qubits };

        // Run many times and count how often qubit 1 is affected
        let mut affected_count = 0;
        for _ in 0..1000 {
            let response = crosstalk.apply(&event, &mut ctx, &mut rng);
            let has_q1 = match response {
                NoiseResponse::InjectGates(ref gates) => {
                    gates.iter().any(|g| g.qubits[0] == QubitId(1))
                }
                NoiseResponse::Multiple(ref responses) => responses.iter().any(|r| {
                    if let NoiseResponse::InjectGates(gates) = r {
                        gates.iter().any(|g| g.qubits[0] == QubitId(1))
                    } else {
                        false
                    }
                }),
                _ => false,
            };
            if has_q1 {
                affected_count += 1;
            }
        }

        // Should be roughly 50%
        let rate = f64::from(affected_count) / 1000.0;
        assert!(
            (rate - 0.5).abs() < 0.1,
            "Expected ~50% crosstalk rate, got {rate}"
        );
    }

    #[test]
    fn test_crosstalk_excludes_leaked_qubits() {
        // Crosstalk should not affect leaked qubits
        let crosstalk = CompositeCrosstalkChannel::new("test", inject_x())
            .responds_to_measurement()
            .global();

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        // Prepare qubits
        ctx.mark_prepared(QubitId(0));
        ctx.mark_prepared(QubitId(1));
        ctx.mark_prepared(QubitId(2));

        // Leak qubit 1
        ctx.mark_leaked(QubitId(1));

        let qubits = [QubitId(0)];
        let outcomes = [false];
        let event = NoiseEvent::AfterMeasurement {
            qubits: &qubits,
            outcomes: &outcomes,
        };

        let response = crosstalk.apply(&event, &mut ctx, &mut rng);

        // Should only affect qubit 2 (qubit 1 is leaked)
        match response {
            NoiseResponse::InjectGates(gates) => {
                assert_eq!(gates.len(), 1);
                assert_eq!(gates[0].qubits[0], QubitId(2));
            }
            _ => panic!("Expected single InjectGates response"),
        }
    }

    #[test]
    fn test_crosstalk_no_targets() {
        let crosstalk = CompositeCrosstalkChannel::new("test", inject_x())
            .responds_to_measurement()
            .global();

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        // Only prepare the measured qubit - no crosstalk targets
        ctx.mark_prepared(QubitId(0));

        let qubits = [QubitId(0)];
        let outcomes = [false];
        let event = NoiseEvent::AfterMeasurement {
            qubits: &qubits,
            outcomes: &outcomes,
        };

        let response = crosstalk.apply(&event, &mut ctx, &mut rng);

        // No targets, so no response
        assert!(response.is_none());
    }

    // ========================================================================
    // BatchCompositeChannel Tests
    // ========================================================================

    #[test]
    fn test_batch_composite_channel_basic() {
        // Create a batch channel that always applies X (100% probability)
        let channel = BatchCompositeChannel::new("test", 1.0 - 1e-10, inject_x())
            .with_filter(CompositeEventFilter::SingleQubitGate);

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        let qubits = [QubitId(0)];
        let angles = [];
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::H,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
        };

        assert!(NoiseChannel::responds_to(&channel, &event));

        let response = channel.apply(&event, &mut ctx, &mut rng);

        match response {
            NoiseResponse::InjectGates(gates) => {
                assert_eq!(gates.len(), 1);
                assert_eq!(gates[0].gate_type, GateType::X);
            }
            _ => panic!("Expected InjectGates response, got {response:?}"),
        }
    }

    #[test]
    fn test_batch_composite_channel_zero_probability() {
        // Zero probability should produce no events
        let channel = BatchCompositeChannel::new("test", 1e-15, inject_x())
            .with_filter(CompositeEventFilter::SingleQubitGate);

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        // Create many qubits
        let qubits: Vec<_> = (0..1000).map(QubitId).collect();
        let angles = [];
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::H,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
        };

        // Very low probability - should almost always be empty
        let response = channel.apply(&event, &mut ctx, &mut rng);
        assert!(response.is_none(), "Expected no events at p=1e-15");
    }

    #[test]
    fn test_batch_composite_channel_statistical() {
        // Test that batch channel produces correct statistical distribution
        let channel = BatchCompositeChannel::new("test", 0.1, pauli())
            .with_filter(CompositeEventFilter::SingleQubitGate)
            .with_min_qubits(10); // Lower threshold for testing

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        let qubits: Vec<_> = (0..1000).map(QubitId).collect();
        let angles = [];
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::H,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
        };

        // Run once and check we get roughly 10% affected
        let response = channel.apply(&event, &mut ctx, &mut rng);
        let count = match response {
            NoiseResponse::InjectGates(gates) => gates.len(),
            NoiseResponse::Multiple(responses) => responses
                .iter()
                .filter_map(|r| {
                    if let NoiseResponse::InjectGates(g) = r {
                        Some(g.len())
                    } else {
                        None
                    }
                })
                .sum(),
            _ => 0,
        };

        // Should be roughly 100 (10% of 1000), allow wide margin
        assert!(
            count > 50 && count < 200,
            "Expected ~100 events, got {count}"
        );
    }

    #[test]
    fn test_batch_composite_channel_low_probability() {
        // Test geometric sampling with low probability
        let channel = BatchCompositeChannel::new("test", 0.001, inject_x())
            .with_filter(CompositeEventFilter::SingleQubitGate)
            .with_min_qubits(100);

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        let qubits: Vec<_> = (0..10_000).map(QubitId).collect();
        let angles = [];
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::H,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
        };

        // Run and verify roughly 0.1% affected
        let response = channel.apply(&event, &mut ctx, &mut rng);
        let count = match response {
            NoiseResponse::InjectGates(gates) => gates.len(),
            NoiseResponse::Multiple(responses) => responses
                .iter()
                .filter_map(|r| {
                    if let NoiseResponse::InjectGates(g) = r {
                        Some(g.len())
                    } else {
                        None
                    }
                })
                .sum(),
            _ => 0,
        };

        // Should be roughly 10 (0.1% of 10000)
        assert!(count < 50, "Expected ~10 events at p=0.001, got {count}");
    }

    #[test]
    fn test_batch_composite_channel_builder_methods() {
        // Test the builder helper methods
        let sq_channel = CompositeChannelBuilder::batch_single_qubit("sq_test", 0.01, pauli());
        assert_eq!(sq_channel.probability(), 0.01);

        let tq_channel = CompositeChannelBuilder::batch_two_qubit("tq_test", 0.001, pauli());
        assert_eq!(tq_channel.probability(), 0.001);

        let any_channel = CompositeChannelBuilder::batch_any_gate("any_test", 0.005, pauli());
        assert_eq!(any_channel.probability(), 0.005);
    }

    #[test]
    fn test_batch_composite_channel_uses_geometric_for_low_p() {
        // Verify that geometric sampling is used when conditions are met
        let channel = BatchCompositeChannel::new("test", 0.001, inject_x())
            .with_filter(CompositeEventFilter::SingleQubitGate)
            .with_geometric_threshold(0.01)
            .with_min_qubits(100);

        // Should use geometric: p=0.001 < 0.01 threshold and would have many qubits
        assert!(channel.use_geometric(1000));

        // Should NOT use geometric: too few qubits
        assert!(!channel.use_geometric(50));

        // Should NOT use geometric: probability too high
        let high_p_channel = BatchCompositeChannel::new("test", 0.1, inject_x())
            .with_filter(CompositeEventFilter::SingleQubitGate);
        assert!(!high_p_channel.use_geometric(1000));
    }

    // ========================================================================
    // Two-Stage Flow Processing Tests
    // ========================================================================

    #[test]
    fn test_two_stage_fired_flags_integration() {
        use crate::noise::composite::action::SampleEmissionWithProb;
        use crate::noise::composite::condition::{
            Condition, IFired, PartnerFired, PartnerOnlyFired,
        };

        // Simulate two-stage processing manually
        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        let qubits = [QubitId(0), QubitId(1)];

        // Stage 1: Sample emission for each qubit (100% probability for deterministic test)
        ctx.clear_fired_flags();

        // Qubit 0: fires
        ctx.set_current_qubit_index(0, &qubits);
        let action0 = SampleEmissionWithProb::new(1.0);
        let _ = crate::noise::composite::action::GateAction::apply(
            &action0,
            QubitId(0),
            &mut ctx,
            &mut rng,
        );

        // Qubit 1: does not fire (0% probability)
        ctx.set_current_qubit_index(1, &qubits);
        let action1 = SampleEmissionWithProb::new(0.0);
        let _ = crate::noise::composite::action::GateAction::apply(
            &action1,
            QubitId(1),
            &mut ctx,
            &mut rng,
        );

        // Stage 2: Check conditions
        // From qubit 0's perspective
        ctx.set_current_qubit_index(0, &qubits);
        assert!(
            IFired.evaluate(QubitId(0), &ctx),
            "Qubit 0 should have fired"
        );
        assert!(
            !PartnerFired.evaluate(QubitId(0), &ctx),
            "Partner (qubit 1) should NOT have fired"
        );
        assert!(
            !PartnerOnlyFired.evaluate(QubitId(0), &ctx),
            "PartnerOnlyFired should be false (I fired)"
        );

        // From qubit 1's perspective
        ctx.set_current_qubit_index(1, &qubits);
        assert!(
            !IFired.evaluate(QubitId(1), &ctx),
            "Qubit 1 should NOT have fired"
        );
        assert!(
            PartnerFired.evaluate(QubitId(1), &ctx),
            "Partner (qubit 0) should have fired"
        );
        assert!(
            PartnerOnlyFired.evaluate(QubitId(1), &ctx),
            "PartnerOnlyFired should be true"
        );
    }

    #[test]
    fn test_two_stage_partner_depolarize_scenario() {
        use crate::noise::composite::CompositeResponse;
        use crate::noise::composite::action::{
            GateAction, IndependentEmissionWithPartnerDepolarize,
        };

        // Test the full emission-with-partner-depolarize action
        let emission_prob = 0.5;
        let shots = 2000;

        let mut _partner_depolarize_count = 0;
        let mut both_leaked_count = 0;
        let mut neither_leaked_count = 0;

        for seed in 0..shots {
            let mut ctx = NoiseContext::new();
            let mut rng = PecosRng::seed_from_u64(seed);
            let qubits = [QubitId(0), QubitId(1)];

            let action = IndependentEmissionWithPartnerDepolarize::new(emission_prob);

            // Process qubit 0
            ctx.set_current_qubit_index(0, &qubits);
            let _ = GateAction::apply(&action, QubitId(0), &mut ctx, &mut rng);

            // Process qubit 1
            ctx.set_current_qubit_index(1, &qubits);
            let response = GateAction::apply(&action, QubitId(1), &mut ctx, &mut rng);

            let q0_leaked = ctx.is_leaked(QubitId(0));
            let q1_leaked = ctx.is_leaked(QubitId(1));

            if q0_leaked && q1_leaked {
                both_leaked_count += 1;
            } else if !q0_leaked && !q1_leaked {
                neither_leaked_count += 1;
            } else {
                // One leaked, one didn't - should have partner depolarize
                // Check if response contains InjectGates
                match response {
                    CompositeResponse::InjectGates(_) => {
                        _partner_depolarize_count += 1;
                    }
                    CompositeResponse::Multiple(ref parts) => {
                        if parts
                            .iter()
                            .any(|p| matches!(p, CompositeResponse::InjectGates(_)))
                        {
                            _partner_depolarize_count += 1;
                        }
                    }
                    _ => {}
                }
            }
        }

        // With 50% emission probability:
        // - P(neither leaks) = 0.5 * 0.5 = 25%
        // - P(both leak) = 0.5 * 0.5 = 25%
        // - P(exactly one leaks) = 50%
        let expected_one_leaked = shots / 2;
        let tolerance = (0.15 * expected_one_leaked as f64) as i32;

        // Verify we got reasonable distribution
        assert!(
            (both_leaked_count - (shots / 4) as i32).abs() < tolerance,
            "Both leaked count off: expected ~{}, got {}",
            shots / 4,
            both_leaked_count
        );
        assert!(
            (neither_leaked_count - (shots / 4) as i32).abs() < tolerance,
            "Neither leaked count off: expected ~{}, got {}",
            shots / 4,
            neither_leaked_count
        );
    }

    #[test]
    fn test_two_stage_with_when_condition() {
        use crate::noise::composite::condition::{Condition, PartnerOnlyFired};

        // Test building a primitive that uses PartnerOnlyFired condition
        let noise = seq![
            // Stage 1 would sample emission (simulated by setting fired flags)
            // Stage 2 uses condition
            when(partner_only_fired(), pauli(), nothing())
        ];

        let channel = CompositeChannelBuilder::two_qubit("test", noise);

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        // Simulate: qubit 0 fired, qubit 1 didn't
        ctx.set_fired(0, true);
        ctx.set_fired(1, false);

        let qubits = [QubitId(0), QubitId(1)];
        let angles = [];
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::CX,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
        };

        // Qubit 0 perspective: partner (qubit 1) didn't fire, I did -> PartnerOnlyFired = false
        ctx.set_current_qubit_index(0, &qubits);
        assert!(!PartnerOnlyFired.evaluate(QubitId(0), &ctx));

        // Qubit 1 perspective: partner (qubit 0) fired, I didn't -> PartnerOnlyFired = true
        ctx.set_current_qubit_index(1, &qubits);
        assert!(PartnerOnlyFired.evaluate(QubitId(1), &ctx));

        // Clear fired flags and apply channel
        ctx.clear_fired_flags();
        let response = channel.apply(&event, &mut ctx, &mut rng);

        // Without fired flags set, neither should trigger PartnerOnlyFired
        // So we should get None responses (nothing() action)
        if let NoiseResponse::Multiple(responses) = response {
            // All should be None
            assert!(responses.iter().all(|r| matches!(r, NoiseResponse::None)));
        }
    }

    #[test]
    fn test_clear_fired_flags_between_gates() {
        let mut ctx = NoiseContext::new();
        let qubits = [QubitId(0), QubitId(1)];

        // Set fired flags
        ctx.set_current_qubit_index(0, &qubits);
        ctx.set_fired(0, true);
        ctx.set_fired(1, true);

        assert!(ctx.is_fired(0));
        assert!(ctx.is_fired(1));

        // Clear flags (as would happen between gates)
        ctx.clear_fired_flags();

        assert!(!ctx.is_fired(0));
        assert!(!ctx.is_fired(1));
    }

    // ========================================================================
    // New Event Type Builder Tests
    // ========================================================================

    #[test]
    fn test_after_reset_channel() {
        let channel = CompositeChannelBuilder::after_reset("reset_noise", inject_x());

        let qubits = [QubitId(0)];
        let reset_event = NoiseEvent::AfterReset { qubits: &qubits };

        assert!(NoiseChannel::responds_to(&channel, &reset_event));

        // Should NOT respond to other events
        let gate_event = NoiseEvent::AfterGate {
            gate_type: GateType::H,
            qubits: &qubits,
            angles: &[],
            gate_id: None,
        };
        assert!(!NoiseChannel::responds_to(&channel, &gate_event));

        // Apply and verify
        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);
        let response = channel.apply(&reset_event, &mut ctx, &mut rng);

        match response {
            NoiseResponse::InjectGates(gates) => {
                assert_eq!(gates.len(), 1);
                assert_eq!(gates[0].gate_type, GateType::X);
            }
            _ => panic!("Expected InjectGates response"),
        }
    }

    #[test]
    fn test_before_circuit_channel() {
        let channel = CompositeChannelBuilder::before_circuit("init_noise", inject_z());

        let qubits = [QubitId(0), QubitId(1), QubitId(2)];
        let circuit_event = NoiseEvent::BeforeCircuit { num_qubits: 3 };

        assert!(NoiseChannel::responds_to(&channel, &circuit_event));

        // Should NOT respond to other events
        let gate_event = NoiseEvent::AfterGate {
            gate_type: GateType::H,
            qubits: &qubits,
            angles: &[],
            gate_id: None,
        };
        assert!(!NoiseChannel::responds_to(&channel, &gate_event));
    }

    #[test]
    fn test_after_circuit_channel() {
        let channel = CompositeChannelBuilder::after_circuit("final_noise", pauli());

        let circuit_event = NoiseEvent::AfterCircuit { num_qubits: 3 };

        assert!(NoiseChannel::responds_to(&channel, &circuit_event));

        // Should NOT respond to preparation
        let prep_event = NoiseEvent::AfterPreparation {
            qubits: &[QubitId(0)],
        };
        assert!(!NoiseChannel::responds_to(&channel, &prep_event));
    }

    #[test]
    fn test_between_layers_channel() {
        let channel =
            CompositeChannelBuilder::between_layers("layer_dephasing", prob(0.5, inject_z()));

        let qubits = [QubitId(0), QubitId(1)];
        let layer_event = NoiseEvent::BetweenLayers {
            qubits: &qubits,
            layer_index: 1,
        };

        assert!(NoiseChannel::responds_to(&channel, &layer_event));

        // Should NOT respond to idle time
        let idle_event = NoiseEvent::IdleTime {
            qubits: &qubits,
            duration: pecos_core::TimeUnits::new(1),
        };
        assert!(!NoiseChannel::responds_to(&channel, &idle_event));

        // Apply multiple times to verify probabilistic behavior
        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);
        let mut z_count = 0;

        for _ in 0..1000 {
            let response = channel.apply(&layer_event, &mut ctx, &mut rng);
            match response {
                NoiseResponse::InjectGates(_) => z_count += 1,
                NoiseResponse::Multiple(ref rs) => {
                    if rs
                        .iter()
                        .any(|r| matches!(r, NoiseResponse::InjectGates(_)))
                    {
                        z_count += 1;
                    }
                }
                _ => {}
            }
        }

        // Should be roughly 50% per qubit (2 qubits), so high overall
        let rate = f64::from(z_count) / 1000.0;
        assert!(
            rate > 0.3 && rate < 0.95,
            "Expected moderate injection rate, got {rate}"
        );
    }

    #[test]
    fn test_batch_new_event_builders() {
        // Test that batch versions compile and work
        let reset_ch = CompositeChannelBuilder::batch_after_reset("reset", 0.01, pauli());
        assert_eq!(reset_ch.probability(), 0.01);

        let before_ch = CompositeChannelBuilder::batch_before_circuit("before", 0.005, inject_x());
        assert_eq!(before_ch.probability(), 0.005);

        let after_ch = CompositeChannelBuilder::batch_after_circuit("after", 0.001, inject_z());
        assert_eq!(after_ch.probability(), 0.001);

        let layers_ch = CompositeChannelBuilder::batch_between_layers("layers", 0.0001, pauli());
        assert_eq!(layers_ch.probability(), 0.0001);
    }

    #[test]
    fn test_flow_event_filter_new_types() {
        // Test that the new filter types match correctly
        let qubits = [QubitId(0)];

        // AfterReset
        let reset_event = NoiseEvent::AfterReset { qubits: &qubits };
        assert!(CompositeEventFilter::AfterReset.matches(&reset_event));
        assert!(!CompositeEventFilter::AnyGate.matches(&reset_event));
        assert!(!CompositeEventFilter::Preparation.matches(&reset_event));

        // BeforeCircuit
        let before_event = NoiseEvent::BeforeCircuit { num_qubits: 1 };
        assert!(CompositeEventFilter::BeforeCircuit.matches(&before_event));
        assert!(!CompositeEventFilter::AfterCircuit.matches(&before_event));

        // AfterCircuit
        let after_event = NoiseEvent::AfterCircuit { num_qubits: 1 };
        assert!(CompositeEventFilter::AfterCircuit.matches(&after_event));
        assert!(!CompositeEventFilter::BeforeCircuit.matches(&after_event));

        // BetweenLayers
        let layers_event = NoiseEvent::BetweenLayers {
            qubits: &qubits,
            layer_index: 0,
        };
        assert!(CompositeEventFilter::BetweenLayers.matches(&layers_event));
        assert!(!CompositeEventFilter::IdleTime.matches(&layers_event));
    }
}
