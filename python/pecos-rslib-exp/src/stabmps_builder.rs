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
//! with configurable parameters (`lazy_measure`, `max_bond_dim`, etc.).

use pecos_neo::noise::ComposableNoiseModel;
use pecos_neo::program::{DynProgramRunner, ProgramRunner};
use pecos_neo::tool::SimulatorFactory;
use pecos_stab_tn::stab_mps::StabMps;

/// Configuration for the `StabMps` backend.
///
/// Carries simulator parameters through the builder-of-builders pattern.
/// Implements `SimulatorFactory` so it can be used with `custom_backend()`.
#[derive(Debug, Clone)]
pub struct StabMpsBuilder {
    /// Use lazy measurement (correct for non-Clifford, slower).
    pub lazy_measure: bool,
    /// Maximum MPS bond dimension.
    pub max_bond_dim: usize,
    /// Maximum truncation error for MPS compression.
    /// None = disabled (library default, use fixed bond dim cap only).
    pub max_truncation_error: Option<f64>,
    /// Merge consecutive RZ on same qubit before decomposition.
    pub merge_rz: bool,
}

impl Default for StabMpsBuilder {
    fn default() -> Self {
        Self {
            lazy_measure: false,
            max_bond_dim: 64,
            max_truncation_error: None,
            merge_rz: false,
        }
    }
}

impl StabMpsBuilder {
    /// Create with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable lazy measurement (correct for non-Clifford states).
    #[must_use]
    pub fn with_lazy_measure(mut self, lazy: bool) -> Self {
        self.lazy_measure = lazy;
        self
    }

    /// Set maximum bond dimension.
    #[must_use]
    pub fn with_max_bond_dim(mut self, bd: usize) -> Self {
        self.max_bond_dim = bd;
        self
    }

    /// Set maximum truncation error.
    #[must_use]
    pub fn with_max_truncation_error(mut self, err: f64) -> Self {
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
        if self.lazy_measure {
            builder = builder.lazy_measure(true);
        }
        builder = builder.max_bond_dim(self.max_bond_dim);
        if let Some(err) = self.max_truncation_error {
            builder = builder.max_truncation_error(err);
        }
        if self.merge_rz {
            builder = builder.merge_rz(true);
        }
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
