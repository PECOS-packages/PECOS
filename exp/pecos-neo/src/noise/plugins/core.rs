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

//! Core plugin providing fundamental state tracking.
//!
//! This plugin handles the basic qubit lifecycle:
//! - Preparation marks qubits as active
//! - Measurement marks qubits as inactive
//!
//! This plugin should almost always be included as it provides
//! the foundation for other plugins like crosstalk.

use crate::noise::plugin::{EventHandler, NoiseModelConfig, NoisePlugin};
use crate::noise::{NoiseContext, NoiseEvent};

/// Core plugin that handles fundamental state tracking.
///
/// Registers event handlers for:
/// - `AfterPreparation`: Marks qubits as prepared and active
/// - `AfterMeasurement`: Marks qubits as inactive
///
/// This plugin should be added first to ensure state is properly
/// tracked before other plugins process events.
///
/// # Example
///
/// ```
/// # use pecos_neo::noise::ComposableNoiseModel;
/// use pecos_neo::noise::plugins::CorePlugin;
///
/// let model = ComposableNoiseModel::new()
///     .add_plugin(&CorePlugin);  // Always add first
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct CorePlugin;

impl CorePlugin {
    /// Create a new core plugin.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl NoisePlugin for CorePlugin {
    fn build(&self, config: &mut NoiseModelConfig) {
        config.add_event_handler(PreparationStateHandler);
        config.add_event_handler(MeasurementStateHandler);
    }

    fn name(&self) -> &'static str {
        "CorePlugin"
    }
}

/// Handles preparation events by marking qubits as active.
#[derive(Debug, Clone, Copy)]
struct PreparationStateHandler;

impl EventHandler for PreparationStateHandler {
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
        "PreparationStateHandler"
    }

    fn priority(&self) -> i32 {
        // High priority - state tracking should happen first
        1000
    }

    fn clone_box(&self) -> Box<dyn EventHandler> {
        Box::new(*self)
    }
}

/// Handles measurement events by marking qubits as inactive.
#[derive(Debug, Clone, Copy)]
struct MeasurementStateHandler;

impl EventHandler for MeasurementStateHandler {
    fn handles(&self, event: &NoiseEvent<'_>) -> bool {
        matches!(event, NoiseEvent::AfterMeasurement { .. })
    }

    fn handle(&self, event: &NoiseEvent<'_>, ctx: &mut NoiseContext) {
        if let NoiseEvent::AfterMeasurement { qubits, .. } = event {
            for &qubit in *qubits {
                ctx.mark_measured(qubit);
            }
        }
    }

    fn name(&self) -> &'static str {
        "MeasurementStateHandler"
    }

    fn priority(&self) -> i32 {
        // High priority - state tracking should happen first
        1000
    }

    fn clone_box(&self) -> Box<dyn EventHandler> {
        Box::new(*self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::QubitId;

    #[test]
    fn test_core_plugin_registers_handlers() {
        let plugin = CorePlugin::new();
        let mut config = NoiseModelConfig::new();
        plugin.build(&mut config);

        assert_eq!(config.event_handlers.len(), 2);
    }

    #[test]
    fn test_preparation_handler() {
        let handler = PreparationStateHandler;
        let qubits = [QubitId(0), QubitId(1)];
        let event = NoiseEvent::AfterPreparation { qubits: &qubits };

        assert!(handler.handles(&event));

        let mut ctx = NoiseContext::new();
        handler.handle(&event, &mut ctx);

        assert!(ctx.is_active(QubitId(0)));
        assert!(ctx.is_active(QubitId(1)));
        assert!(ctx.exists(QubitId(0)));
        assert!(ctx.exists(QubitId(1)));
    }

    #[test]
    fn test_measurement_handler() {
        let handler = MeasurementStateHandler;
        let qubits = [QubitId(0)];
        let outcomes = [false];
        let event = NoiseEvent::AfterMeasurement {
            qubits: &qubits,
            outcomes: &outcomes,
        };

        assert!(handler.handles(&event));

        let mut ctx = NoiseContext::new();
        ctx.mark_prepared(QubitId(0)); // Must be prepared first
        assert!(ctx.is_active(QubitId(0)));

        handler.handle(&event, &mut ctx);

        assert!(!ctx.is_active(QubitId(0)));
        assert!(ctx.exists(QubitId(0))); // Still exists, just inactive
    }

    #[test]
    fn test_handler_priority() {
        let prep = PreparationStateHandler;
        let meas = MeasurementStateHandler;

        // Both should have high priority
        assert!(prep.priority() > 0);
        assert!(meas.priority() > 0);
    }
}
