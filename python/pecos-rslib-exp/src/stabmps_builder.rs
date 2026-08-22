// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
// express or implied. See the License for the specific language governing permissions and
// limitations under the License.

//! `StabMps` backend for `sim_neo`.
//!
//! Provides a `SimulatorFactory` implementation that creates `StabMps` simulators
//! with configurable parameters (`measurement`, `max_bond_dim`, etc.).

use pecos_neo::noise::ComposableNoiseModel;
use pecos_neo::program::{DynProgramRunner, ProgramRunner};
use pecos_neo::tool::SimulatorFactory;
use pecos_stab_tn::stab_mps::{MeasurementMode, StabMps};

/// Configuration for the `StabMps` backend.
///
/// Carries simulator parameters through the builder-of-builders pattern.
/// Implements `SimulatorFactory` so it can be used with `custom_backend()`.
#[derive(Debug, Clone)]
pub struct StabMpsBuilder {
    /// Singular-measurement policy.
    pub measurement: MeasurementMode,
    /// Maximum MPS bond dimension.
    pub max_bond_dim: usize,
    /// Maximum truncation error for MPS compression.
    /// Zero disables adaptive truncation while preserving cutoff and cap truncation.
    pub max_truncation_error: Option<f64>,
    /// Merge consecutive RZ on same qubit before decomposition.
    pub merge_rz: bool,
}

impl Default for StabMpsBuilder {
    fn default() -> Self {
        Self {
            measurement: MeasurementMode::default(),
            max_bond_dim: 128,
            max_truncation_error: Some(1e-8),
            merge_rz: true,
        }
    }
}

impl StabMpsBuilder {
    /// Create with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Select the singular-measurement policy.
    #[must_use]
    pub fn with_measurement(mut self, measurement: MeasurementMode) -> Self {
        self.measurement = measurement;
        self
    }

    /// Set maximum bond dimension.
    #[must_use]
    pub fn with_max_bond_dim(mut self, bd: usize) -> Self {
        self.max_bond_dim = bd;
        self
    }

    /// Set maximum truncation error.
    ///
    /// # Panics
    ///
    /// Panics if `err` is negative, NaN, or infinite.
    #[must_use]
    pub fn with_max_truncation_error(mut self, err: f64) -> Self {
        assert!(
            err.is_finite() && err >= 0.0,
            "max_truncation_error must be finite and non-negative"
        );
        self.max_truncation_error = Some(err);
        self
    }

    /// Enable RZ merging.
    #[must_use]
    pub fn with_merge_rz(mut self, merge: bool) -> Self {
        self.merge_rz = merge;
        self
    }
}

impl SimulatorFactory for StabMpsBuilder {
    fn create_runner(
        &self,
        num_qubits: usize,
        noise: Option<ComposableNoiseModel>,
        seed: Option<u64>,
    ) -> Box<dyn DynProgramRunner> {
        let mut builder = StabMps::builder(num_qubits);
        builder = builder.measurement(self.measurement);
        builder = builder.max_bond_dim(self.max_bond_dim);
        if let Some(err) = self.max_truncation_error {
            builder = builder.max_truncation_error(err);
        }
        builder = builder.merge_rz(self.merge_rz);
        if let Some(s) = seed {
            builder = builder.seed(s);
        }
        let sim = builder.build();

        let mut runner = ProgramRunner::rotations(sim);
        if let Some(n) = noise {
            runner = runner.with_noise(n);
        }
        if let Some(s) = seed {
            runner = runner.with_seed(s);
        }
        Box::new(runner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sim_neo_measurement_mode_defaults_and_selection() {
        let default = StabMpsBuilder::default();
        assert_eq!(default.measurement, MeasurementMode::Exact);
        let pragmatic = default.with_measurement(MeasurementMode::Pragmatic);
        assert_eq!(pragmatic.measurement, MeasurementMode::Pragmatic);
        let lazy = pragmatic.with_measurement(MeasurementMode::Lazy);
        assert_eq!(lazy.measurement, MeasurementMode::Lazy);
    }
}
