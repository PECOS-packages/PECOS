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
use crate::noise::validation::ValidationResult;

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
/// let plugin = DepolarizingPlugin::new(0.01, 0.02)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct DepolarizingPlugin {
    single_qubit: SingleQubitChannel,
    two_qubit: TwoQubitChannel,
}

impl DepolarizingPlugin {
    /// Create depolarizing noise for both gate arities.
    ///
    /// # Errors
    /// Rejects probabilities outside [0, 1] or NaN.
    pub fn new(p1: f64, p2: f64) -> ValidationResult<Self> {
        Ok(Self {
            single_qubit: SingleQubitChannel::depolarizing(p1)?,
            two_qubit: TwoQubitChannel::depolarizing(p2)?,
        })
    }
    /// Create single-qubit depolarizing noise.
    ///
    /// # Errors
    /// Rejects probabilities outside [0, 1] or NaN.
    pub fn single_qubit_only(p1: f64) -> ValidationResult<Self> {
        Self::new(p1, 0.0)
    }
    /// Create two-qubit depolarizing noise.
    ///
    /// # Errors
    /// Rejects probabilities outside [0, 1] or NaN.
    pub fn two_qubit_only(p2: f64) -> ValidationResult<Self> {
        Self::new(0.0, p2)
    }
}

impl NoisePlugin for DepolarizingPlugin {
    fn build(&self, config: &mut NoiseModelConfig) {
        if self.single_qubit.error_probability() > 0.0 {
            config.add_channel(self.single_qubit.clone());
        }
        if self.two_qubit.error_probability() > 0.0 {
            config.add_channel(self.two_qubit.clone());
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
        let plugin = DepolarizingPlugin::new(0.01, 0.02).unwrap();
        let mut config = NoiseModelConfig::new();
        plugin.build(&mut config);

        assert_eq!(config.channels.len(), 2);
    }

    #[test]
    fn test_single_qubit_only() {
        let plugin = DepolarizingPlugin::single_qubit_only(0.01).unwrap();
        let mut config = NoiseModelConfig::new();
        plugin.build(&mut config);

        assert_eq!(config.channels.len(), 1);
    }

    #[test]
    fn test_zero_probabilities() {
        let plugin = DepolarizingPlugin::new(0.0, 0.0).unwrap();
        let mut config = NoiseModelConfig::new();
        plugin.build(&mut config);

        assert_eq!(config.channels.len(), 0);
    }
}
