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

//! `MemStabSim` -- Clifford + depolarizing-family noise simulator that samples
//! **raw measurement outcomes** via a Measurement Noise Model (MNM).
//!
//! Sibling to [`crate::dem_stab::DemStabSim`]. Same underlying fault-influence
//! machinery, different aggregation level:
//!
//! | | `DemStabSim` | `MemStabSim` |
//! |---|---|---|
//! | Output | Detector + observable flips | Raw measurement outcomes |
//! | Use case | Research batch / decoder input | Classical-engine-facing backend |
//! | Backing primitive | `DemSampler` (DEM mechanisms) | `MeasurementNoiseModel` (MNM mechanisms) |
//!
//! Use `MemStabSim` when a classical control engine needs per-shot raw measurement
//! records (and will compute its own detectors/observables from them). Use
//! `DemStabSim` when you just want detector events for a decoder.
//!
//! # Scope
//!
//! Clifford circuits only, no classical feed-forward. For adaptive circuits use
//! `sparse_stab` + `pecos-neo` instead. For non-Clifford circuits use `CliffordRz`,
//! `STN`, or `MAST`.
//!
//! # Example
//!
//! ```
//! use pecos_qec::mem_stab::MemStabSim;
//! use pecos_qec::fault_tolerance::dem_builder::NoiseConfig;
//! use pecos_quantum::DagCircuit;
//! use rand::SeedableRng;
//! use rand::rngs::SmallRng;
//!
//! let mut dag = DagCircuit::new();
//! dag.pz(&[2]);
//! dag.cx(&[(0, 2)]);
//! dag.cx(&[(1, 2)]);
//! dag.mz(&[2]);
//!
//! let sim = MemStabSim::builder()
//!     .circuit(dag)
//!     .noise(NoiseConfig::uniform(0.01))
//!     .build()
//!     .unwrap();
//!
//! let mut rng = SmallRng::seed_from_u64(42);
//! let outcomes = sim.sample(&mut rng);
//! assert_eq!(outcomes.len(), sim.num_measurements());
//! ```

use crate::fault_tolerance::dem_builder::{MeasurementNoiseModel, MemBuilder, NoiseConfig};
use crate::fault_tolerance::propagator::DagFaultAnalyzer;
use pecos_quantum::DagCircuit;
use rand::Rng;
use thiserror::Error;

/// Errors that can occur when building a [`MemStabSim`].
#[derive(Debug, Error)]
pub enum MemStabError {
    /// Builder called without a circuit.
    #[error("MemStabSim requires a circuit; call .circuit(dag) before .build()")]
    MissingCircuit,
}

/// Clifford + depolarizing-family noise simulator that samples raw measurement outcomes.
///
/// Built once via [`MemStabSim::builder`], sampled many times via [`Self::sample`] or
/// [`Self::sample_batch`]. The underlying [`MeasurementNoiseModel`] is constructed
/// eagerly at build time.
#[derive(Debug, Clone)]
pub struct MemStabSim {
    mnm: MeasurementNoiseModel,
}

impl MemStabSim {
    /// Start building a [`MemStabSim`].
    #[must_use]
    pub fn builder() -> MemStabSimBuilder {
        MemStabSimBuilder::default()
    }

    /// Number of measurements in the compiled circuit.
    #[must_use]
    pub fn num_measurements(&self) -> usize {
        self.mnm.num_measurements
    }

    /// Number of error mechanisms in the compiled MNM.
    #[must_use]
    pub fn num_mechanisms(&self) -> usize {
        self.mnm.mechanisms.len()
    }

    /// Access the underlying [`MeasurementNoiseModel`].
    #[must_use]
    pub fn mnm(&self) -> &MeasurementNoiseModel {
        &self.mnm
    }

    /// Sample one shot of raw measurement outcomes.
    ///
    /// Length of the returned vector equals [`Self::num_measurements`].
    pub fn sample<R: Rng>(&self, rng: &mut R) -> Vec<bool> {
        self.mnm.sample(rng)
    }

    /// Sample one shot into a preallocated buffer.
    ///
    /// `outcomes` must have length equal to [`Self::num_measurements`]; the buffer
    /// is cleared before sampling.
    pub fn sample_into<R: Rng>(&self, outcomes: &mut [bool], rng: &mut R) {
        self.mnm.sample_into(outcomes, rng);
    }

    /// Sample `num_shots` independent shots.
    ///
    /// Returns a `Vec` of length `num_shots`; each inner vector has length
    /// [`Self::num_measurements`].
    #[must_use]
    pub fn sample_batch<R: Rng>(&self, num_shots: usize, rng: &mut R) -> Vec<Vec<bool>> {
        let mut out = Vec::with_capacity(num_shots);
        let mut buf = vec![false; self.num_measurements()];
        for _ in 0..num_shots {
            self.mnm.sample_into(&mut buf, rng);
            out.push(buf.clone());
        }
        out
    }
}

/// Builder for [`MemStabSim`].
#[derive(Debug, Default)]
pub struct MemStabSimBuilder {
    circuit: Option<DagCircuit>,
    noise: NoiseConfig,
    measurement_order: Option<Vec<usize>>,
}

impl MemStabSimBuilder {
    /// Set the circuit. Required.
    #[must_use]
    pub fn circuit(mut self, dag: DagCircuit) -> Self {
        self.circuit = Some(dag);
        self
    }

    /// Set the noise configuration.
    #[must_use]
    pub fn noise(mut self, config: NoiseConfig) -> Self {
        self.noise = config;
        self
    }

    /// Set the measurement order mapping from a `TickCircuit` (advanced).
    #[must_use]
    pub fn measurement_order(mut self, order: Vec<usize>) -> Self {
        self.measurement_order = Some(order);
        self
    }

    /// Build the [`MemStabSim`], consuming the builder.
    ///
    /// # Errors
    ///
    /// Returns [`MemStabError::MissingCircuit`] if no circuit was set.
    pub fn build(self) -> Result<MemStabSim, MemStabError> {
        let dag = self.circuit.ok_or(MemStabError::MissingCircuit)?;

        let analyzer = DagFaultAnalyzer::new(&dag);
        let influence_map = analyzer.build_influence_map();

        let mut builder = MemBuilder::new(&influence_map).with_noise(
            self.noise.p1,
            self.noise.p2,
            self.noise.p_meas,
            self.noise.p_prep,
        );

        if let Some(order) = self.measurement_order {
            builder = builder.with_measurement_order(order);
        }

        Ok(MemStabSim {
            mnm: builder.build(),
        })
    }
}
