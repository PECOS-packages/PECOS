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

//! Measurement noise plugin for readout errors.

use crate::noise::measurement::MeasurementChannel;
use crate::noise::plugin::{NoiseModelConfig, NoisePlugin};
use crate::noise::validation::ValidationResult;

/// Plugin that adds measurement (readout) errors.
///
/// Supports asymmetric measurement errors where the probability of
/// misreading 0 as 1 differs from misreading 1 as 0.
///
/// # Example
///
/// ```
/// use pecos_neo::noise::plugins::MeasurementNoisePlugin;
///
/// // Symmetric 1% measurement error
/// let plugin = MeasurementNoisePlugin::symmetric(0.01)?;
///
/// // Asymmetric: 2% chance of 0->1, 1% chance of 1->0
/// let plugin = MeasurementNoisePlugin::asymmetric(0.02, 0.01)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct MeasurementNoisePlugin {
    channel: MeasurementChannel,
}

impl MeasurementNoisePlugin {
    /// Create symmetric readout noise.
    ///
    /// # Errors
    /// Rejects probabilities outside [0, 1] or NaN.
    pub fn symmetric(p: f64) -> ValidationResult<Self> {
        Ok(Self {
            channel: MeasurementChannel::symmetric(p)?,
        })
    }
    /// Create asymmetric readout noise.
    ///
    /// # Errors
    /// Rejects probabilities outside [0, 1] or NaN.
    pub fn asymmetric(p_0_to_1: f64, p_1_to_0: f64) -> ValidationResult<Self> {
        Ok(Self {
            channel: MeasurementChannel::asymmetric(p_0_to_1, p_1_to_0)?,
        })
    }
}

impl NoisePlugin for MeasurementNoisePlugin {
    fn build(&self, config: &mut NoiseModelConfig) {
        if self.channel.is_active() {
            config.add_channel(self.channel.clone());
        }
    }

    fn name(&self) -> &'static str {
        "MeasurementNoisePlugin"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetric_measurement() {
        let plugin = MeasurementNoisePlugin::symmetric(0.01).unwrap();
        let mut config = NoiseModelConfig::new();
        plugin.build(&mut config);

        assert_eq!(config.channels.len(), 1);
    }

    #[test]
    fn test_asymmetric_measurement() {
        let plugin = MeasurementNoisePlugin::asymmetric(0.02, 0.01).unwrap();
        let mut config = NoiseModelConfig::new();
        plugin.build(&mut config);

        assert_eq!(config.channels.len(), 1);
    }

    #[test]
    fn test_zero_probabilities() {
        let plugin = MeasurementNoisePlugin::symmetric(0.0).unwrap();
        let mut config = NoiseModelConfig::new();
        plugin.build(&mut config);

        assert_eq!(config.channels.len(), 0);
    }
}
