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

//! Depolarizing noise plugin for single and two-qubit gates.

use crate::noise::plugin::{NoiseModelConfig, NoisePlugin};
use crate::noise::single_qubit::SingleQubitChannel;
use crate::noise::two_qubit::TwoQubitChannel;

/// Plugin that adds depolarizing noise to gates.
///
/// Registers `SingleQubitChannel` and `TwoQubitChannel` with the specified
/// error probabilities.
///
/// # Example
///
/// ```
/// use pecos_neo::noise::plugins::DepolarizingPlugin;
///
/// // 1% single-qubit errors, 2% two-qubit errors
/// let plugin = DepolarizingPlugin::new(0.01, 0.02);
/// ```
#[derive(Debug, Clone)]
pub struct DepolarizingPlugin {
    /// Single-qubit gate error probability.
    pub p1: f64,
    /// Two-qubit gate error probability.
    pub p2: f64,
}

impl DepolarizingPlugin {
    /// Create a depolarizing noise plugin.
    #[must_use]
    pub fn new(p1: f64, p2: f64) -> Self {
        Self { p1, p2 }
    }

    /// Create with only single-qubit noise.
    #[must_use]
    pub fn single_qubit_only(p1: f64) -> Self {
        Self { p1, p2: 0.0 }
    }

    /// Create with only two-qubit noise.
    #[must_use]
    pub fn two_qubit_only(p2: f64) -> Self {
        Self { p1: 0.0, p2 }
    }
}

impl NoisePlugin for DepolarizingPlugin {
    fn build(&self, config: &mut NoiseModelConfig) {
        if self.p1 > 0.0 {
            config.add_channel(SingleQubitChannel::depolarizing(self.p1));
        }
        if self.p2 > 0.0 {
            config.add_channel(TwoQubitChannel::depolarizing(self.p2));
        }
    }

    fn name(&self) -> &'static str {
        "DepolarizingPlugin"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_depolarizing_plugin() {
        let plugin = DepolarizingPlugin::new(0.01, 0.02);
        let mut config = NoiseModelConfig::new();
        plugin.build(&mut config);

        assert_eq!(config.channels.len(), 2);
    }

    #[test]
    fn test_single_qubit_only() {
        let plugin = DepolarizingPlugin::single_qubit_only(0.01);
        let mut config = NoiseModelConfig::new();
        plugin.build(&mut config);

        assert_eq!(config.channels.len(), 1);
    }

    #[test]
    fn test_zero_probabilities() {
        let plugin = DepolarizingPlugin::new(0.0, 0.0);
        let mut config = NoiseModelConfig::new();
        plugin.build(&mut config);

        assert_eq!(config.channels.len(), 0);
    }
}
