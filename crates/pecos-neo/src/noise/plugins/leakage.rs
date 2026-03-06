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

//! Leakage plugin providing all leakage-related effects.
//!
//! This plugin handles:
//! - Gate skipping when qubits are leaked
//! - Depolarizing non-leaked partners in two-qubit gates
//! - Random measurement outcomes for leaked qubits
//! - Conversion of leakage to depolarizing based on scale factor

use crate::noise::leakage::LeakageChannel;
use crate::noise::plugin::{NoiseModelConfig, NoisePlugin};

/// Plugin that handles all leakage-related effects.
///
/// Leakage occurs when qubits leave the computational subspace.
/// This plugin ensures correct handling of leaked qubits:
///
/// - **Gate skipping**: Gates on leaked qubits are not applied
/// - **Partner depolarizing**: For two-qubit gates, non-leaked partners
///   receive random Pauli errors
/// - **Measurement handling**: Leaked qubits produce random results
///
/// ## Leakage Scale
///
/// The `leakage_scale` parameter controls how leakage events are handled:
/// - `1.0`: All leakage events remain as leakage (default)
/// - `0.0`: All leakage events become depolarizing noise instead
/// - Between: Probabilistic conversion
///
/// # Example
///
/// ```
/// use pecos_neo::noise::plugins::LeakagePlugin;
///
/// // Full leakage (default)
/// let plugin = LeakagePlugin::new();
///
/// // No leakage - convert all to depolarizing
/// let plugin = LeakagePlugin::no_leakage();
///
/// // Partial leakage
/// let plugin = LeakagePlugin::with_scale(0.5);
/// ```
#[derive(Debug, Clone)]
pub struct LeakagePlugin {
    /// Scale factor for leakage (0.0 = all depolarizing, 1.0 = all leakage).
    pub leakage_scale: f64,
}

impl Default for LeakagePlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl LeakagePlugin {
    /// Create a leakage plugin with full leakage (scale = 1.0).
    #[must_use]
    pub fn new() -> Self {
        Self { leakage_scale: 1.0 }
    }

    /// Create a leakage plugin that converts all leakage to depolarizing.
    #[must_use]
    pub fn no_leakage() -> Self {
        Self { leakage_scale: 0.0 }
    }

    /// Create a leakage plugin with a specific scale factor.
    #[must_use]
    pub fn with_scale(scale: f64) -> Self {
        Self {
            leakage_scale: scale,
        }
    }
}

impl NoisePlugin for LeakagePlugin {
    fn build(&self, config: &mut NoiseModelConfig) {
        // Register the leakage channel which handles all leakage effects
        config.add_channel(LeakageChannel {
            leakage_scale: self.leakage_scale,
        });
    }

    fn name(&self) -> &'static str {
        "LeakagePlugin"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leakage_plugin_default() {
        let plugin = LeakagePlugin::new();
        assert!((plugin.leakage_scale - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_leakage_plugin_no_leakage() {
        let plugin = LeakagePlugin::no_leakage();
        assert!((plugin.leakage_scale - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_leakage_plugin_with_scale() {
        let plugin = LeakagePlugin::with_scale(0.5);
        assert!((plugin.leakage_scale - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_leakage_plugin_builds_config() {
        let plugin = LeakagePlugin::new();
        let mut config = NoiseModelConfig::new();
        plugin.build(&mut config);

        assert_eq!(config.channels.len(), 1);
        assert_eq!(config.channels[0].name(), "LeakageChannel");
    }
}
