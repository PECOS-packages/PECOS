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

//! Plugin system for composable noise models.
//!
//! This module provides a Bevy-inspired plugin architecture for building noise models.
//! Instead of hardcoding behavior, functionality is registered through plugins.
//!
//! ## Core Concepts
//!
//! - **Plugins**: Bundle related functionality (state handlers, noise channels, observers)
//! - **Event Handlers**: React to events and update context state (no noise)
//! - **Noise Channels**: React to events and produce noise (existing trait)
//! - **Context Observers**: React to context state changes
//!
//! ## Example
//!
//! ```
//! use pecos_neo::noise::ComposableNoiseModel;
//! use pecos_neo::noise::plugins::{CorePlugin, DepolarizingPlugin, LeakagePlugin};
//!
//! let model = ComposableNoiseModel::new()
//!     .add_plugin(&CorePlugin)           // Fundamental state tracking
//!     .add_plugin(&LeakagePlugin::new()) // Leakage handling
//!     .add_plugin(&DepolarizingPlugin::new(0.01, 0.02));
//! ```

use super::{NoiseChannel, NoiseContext, NoiseEvent, NoiseResponse};
use pecos_core::QubitId;
use pecos_random::PecosRng;

/// A plugin that bundles related noise model functionality.
///
/// Plugins can register:
/// - Event handlers for state tracking
/// - Noise channels for error injection
/// - Context observers for reacting to state changes
pub trait NoisePlugin: Send + Sync {
    /// Build the plugin by registering handlers, channels, and observers.
    fn build(&self, config: &mut NoiseModelConfig);

    /// Optional name for debugging.
    fn name(&self) -> &'static str {
        "UnnamedPlugin"
    }
}

/// Configuration for building a noise model.
///
/// Collects event handlers, channels, and observers from plugins.
#[derive(Default)]
pub struct NoiseModelConfig {
    /// Event handlers that update context state (run before channels).
    pub(crate) event_handlers: Vec<Box<dyn EventHandler>>,

    /// Noise channels that produce noise responses.
    pub(crate) channels: Vec<Box<dyn NoiseChannel>>,

    /// Observers that react to context state changes.
    pub(crate) observers: Vec<Box<dyn ContextObserver>>,
}

impl NoiseModelConfig {
    /// Create a new empty configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an event handler.
    pub fn add_event_handler(&mut self, handler: impl EventHandler + 'static) -> &mut Self {
        self.event_handlers.push(Box::new(handler));
        self
    }

    /// Add a noise channel.
    pub fn add_channel(&mut self, channel: impl NoiseChannel + 'static) -> &mut Self {
        self.channels.push(Box::new(channel));
        self
    }

    /// Add a context observer.
    pub fn add_observer(&mut self, observer: impl ContextObserver + 'static) -> &mut Self {
        self.observers.push(Box::new(observer));
        self
    }
}

/// Handles events by updating context state.
///
/// Event handlers run before noise channels and don't produce noise.
/// They're used for fundamental state tracking like:
/// - Marking qubits as prepared/active
/// - Marking qubits as measured/inactive
pub trait EventHandler: Send + Sync {
    /// Check if this handler should process the event.
    fn handles(&self, event: &NoiseEvent<'_>) -> bool;

    /// Process the event and update context state.
    fn handle(&self, event: &NoiseEvent<'_>, ctx: &mut NoiseContext);

    /// Optional name for debugging.
    fn name(&self) -> &'static str {
        "UnnamedHandler"
    }

    /// Priority for ordering (higher = runs first).
    fn priority(&self) -> i32 {
        0
    }

    /// Clone this handler into a boxed trait object.
    fn clone_box(&self) -> Box<dyn EventHandler>;
}

/// Observes changes to the noise context and can produce responses.
///
/// Observers are notified when context state changes, allowing
/// reactive behavior like:
/// - Triggering effects when a qubit becomes leaked
/// - Applying noise when a qubit is prepared
pub trait ContextObserver: Send + Sync {
    /// Called when a qubit is marked as leaked.
    fn on_leaked(
        &self,
        _qubit: QubitId,
        _ctx: &NoiseContext,
        _rng: &mut PecosRng,
    ) -> NoiseResponse {
        NoiseResponse::None
    }

    /// Called when a qubit is marked as unleaked (seepage).
    fn on_unleaked(
        &self,
        _qubit: QubitId,
        _ctx: &NoiseContext,
        _rng: &mut PecosRng,
    ) -> NoiseResponse {
        NoiseResponse::None
    }

    /// Called when a qubit is prepared.
    fn on_prepared(
        &self,
        _qubit: QubitId,
        _ctx: &NoiseContext,
        _rng: &mut PecosRng,
    ) -> NoiseResponse {
        NoiseResponse::None
    }

    /// Called when a qubit is measured.
    fn on_measured(
        &self,
        _qubit: QubitId,
        _ctx: &NoiseContext,
        _rng: &mut PecosRng,
    ) -> NoiseResponse {
        NoiseResponse::None
    }

    /// Optional name for debugging.
    fn name(&self) -> &'static str {
        "UnnamedObserver"
    }

    /// Clone this observer into a boxed trait object.
    fn clone_box(&self) -> Box<dyn ContextObserver>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct TestHandler;

    impl EventHandler for TestHandler {
        fn handles(&self, event: &NoiseEvent<'_>) -> bool {
            matches!(event, NoiseEvent::AfterPreparation { .. })
        }

        fn handle(&self, event: &NoiseEvent<'_>, ctx: &mut NoiseContext) {
            if let NoiseEvent::AfterPreparation { qubits } = event {
                for &qubit in *qubits {
                    ctx.mark_prepared(qubit);
                }
            }
        }

        fn name(&self) -> &'static str {
            "TestHandler"
        }

        fn clone_box(&self) -> Box<dyn EventHandler> {
            Box::new(*self)
        }
    }

    struct TestPlugin;

    impl NoisePlugin for TestPlugin {
        fn build(&self, config: &mut NoiseModelConfig) {
            config.add_event_handler(TestHandler);
        }

        fn name(&self) -> &'static str {
            "TestPlugin"
        }
    }

    #[test]
    fn test_plugin_builds_config() {
        let mut config = NoiseModelConfig::new();
        TestPlugin.build(&mut config);

        assert_eq!(config.event_handlers.len(), 1);
        assert_eq!(config.event_handlers[0].name(), "TestHandler");
    }

    #[test]
    fn test_event_handler() {
        let handler = TestHandler;
        let qubits = [QubitId(0)];
        let event = NoiseEvent::AfterPreparation { qubits: &qubits };

        assert!(handler.handles(&event));

        let mut ctx = NoiseContext::new();
        handler.handle(&event, &mut ctx);

        assert!(ctx.is_active(QubitId(0)));
    }
}
