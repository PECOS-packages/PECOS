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

//! `DemStabSim` -- Clifford + depolarizing-family noise simulator backed by DEM sampling.
//!
//! Wraps the existing DAG -> fault-influence -> DEM-sampler pipeline as a single
//! simulator type that consumes a static [`DagCircuit`] plus detector / observable
//! definitions plus a [`NoiseConfig`] and produces shot batches of detector and
//! observable flips.
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
//! use pecos_qec::dem_stab::DemStabSim;
//! use pecos_qec::fault_tolerance::dem_builder::{DemOutput, DetectorDef, NoiseConfig};
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
//! let sim = DemStabSim::builder()
//!     .circuit(dag)
//!     .noise(NoiseConfig::uniform(0.01))
//!     .detectors(vec![DetectorDef::new(0).with_records([-1])])
//!     .build()
//!     .unwrap();
//!
//! let mut rng = SmallRng::seed_from_u64(42);
//! let batch = sim.sample_batch(100, &mut rng);
//! assert_eq!(batch.detector_flips.len(), 100);
//! ```

use crate::fault_tolerance::dem_builder::{
    DemOutput, DemSampler, DemSamplerBuilder, DetectorDef, DetectorErrorModel,
    DetectorValidationError, NoiseConfig, PerGateTypeNoise,
};
use crate::fault_tolerance::propagator::DagFaultAnalyzer;
use pecos_quantum::DagCircuit;
use rand_core::Rng;
use thiserror::Error;

/// Errors that can occur when building a [`DemStabSim`].
#[derive(Debug, Error)]
pub enum DemStabError {
    /// Builder called without a circuit.
    #[error("DemStabSim requires a circuit; call .circuit(dag) before .build()")]
    MissingCircuit,
    /// Detector definitions are invalid for the circuit.
    #[error(transparent)]
    DetectorValidation(#[from] DetectorValidationError),
}

/// Shot-batch output from [`DemStabSim::sample_batch`].
///
/// `detector_flips[i]` is the bit-vector of detector outcomes for shot `i` (length
/// equals the number of registered detectors). `observable_flips[i]` is the
/// corresponding observable outcomes.
#[derive(Debug, Clone)]
pub struct DemStabShotBatch {
    /// Per-shot detector flip vectors. Outer length = `num_shots`, inner length = `num_detectors`.
    pub detector_flips: Vec<Vec<bool>>,
    /// Per-shot observable flip vectors. Outer length = `num_shots`, inner length = `num_observables`.
    pub observable_flips: Vec<Vec<bool>>,
}

/// Clifford + depolarizing-family noise simulator backed by DEM sampling.
///
/// Built once via [`DemStabSim::builder`], sampled many times via [`Self::sample_batch`].
/// The underlying [`DemSampler`] is constructed eagerly at build time; subsequent
/// shots reuse the cached mechanism table.
#[derive(Debug, Clone)]
pub struct DemStabSim {
    sampler: DemSampler,
    /// Detector definitions preserved from the builder, used to produce
    /// a text-serializable [`DetectorErrorModel`] with full metadata.
    detectors: Vec<DetectorDef>,
    observables: Vec<DemOutput>,
}

impl DemStabSim {
    /// Start building a [`DemStabSim`].
    #[must_use]
    pub fn builder() -> DemStabSimBuilder {
        DemStabSimBuilder::default()
    }

    /// Number of registered detectors.
    #[must_use]
    pub fn num_detectors(&self) -> usize {
        self.sampler.num_detectors()
    }

    /// Number of registered observables.
    #[must_use]
    pub fn num_observables(&self) -> usize {
        self.sampler.num_observables()
    }

    /// Number of error mechanisms in the compiled DEM.
    #[must_use]
    pub fn num_mechanisms(&self) -> usize {
        self.sampler.num_mechanisms()
    }

    /// Access the underlying [`DemSampler`] for advanced use (e.g. statistics-only APIs).
    #[must_use]
    pub fn sampler(&self) -> &DemSampler {
        &self.sampler
    }

    /// Produce a [`DetectorErrorModel`] reflecting the compiled mechanism
    /// set and the detector / observable definitions the builder was
    /// given. Use [`DetectorErrorModel::to_string`] for Stim-compatible
    /// text output.
    ///
    /// Note: the probabilities are recovered from the sampler's stored
    /// `u64` thresholds, which round-trips to ~machine precision.
    #[must_use]
    pub fn detector_error_model(&self) -> DetectorErrorModel {
        let mut dem = self.sampler.to_detector_error_model();
        for det in &self.detectors {
            dem.add_detector(det.clone());
        }
        for obs in &self.observables {
            dem.add_observable(obs.clone());
        }
        dem
    }

    /// Sample `num_shots` independent shots from the compiled DEM.
    #[must_use]
    pub fn sample_batch<R: Rng>(&self, num_shots: usize, rng: &mut R) -> DemStabShotBatch {
        let (detector_flips, observable_flips) = self.sampler.sample_batch(num_shots, rng);
        DemStabShotBatch {
            detector_flips,
            observable_flips,
        }
    }
}

/// Builder for [`DemStabSim`].
#[derive(Debug, Default)]
pub struct DemStabSimBuilder {
    circuit: Option<DagCircuit>,
    noise: NoiseConfig,
    per_gate_noise: Option<PerGateTypeNoise>,
    detectors: Vec<DetectorDef>,
    observables: Vec<DemOutput>,
    measurement_order: Option<Vec<usize>>,
}

impl DemStabSimBuilder {
    /// Set the circuit. Required.
    #[must_use]
    pub fn circuit(mut self, dag: DagCircuit) -> Self {
        self.circuit = Some(dag);
        self
    }

    /// Set the uniform-depolarizing noise configuration. When both this
    /// and [`Self::per_gate_noise`] are set, the per-gate spec takes
    /// precedence.
    #[must_use]
    pub fn noise(mut self, config: NoiseConfig) -> Self {
        self.noise = config;
        self
    }

    /// Set a per-gate-type per-Pauli noise specification. Overrides
    /// [`Self::noise`] scalars for any gate type present in the spec.
    /// Intended consumer for `pecos-lindblad::PauliLindbladModel`
    /// adapter output.
    #[must_use]
    pub fn per_gate_noise(mut self, cfg: PerGateTypeNoise) -> Self {
        self.per_gate_noise = Some(cfg);
        self
    }

    /// Register detectors by [`DetectorDef`].
    #[must_use]
    pub fn detectors(mut self, detectors: Vec<DetectorDef>) -> Self {
        self.detectors = detectors;
        self
    }

    /// Register measurement-record observables.
    #[must_use]
    pub fn observables(mut self, observables: Vec<DemOutput>) -> Self {
        self.observables = observables;
        self
    }

    /// Set the measurement order mapping from a `TickCircuit` (advanced).
    #[must_use]
    pub fn measurement_order(mut self, order: Vec<usize>) -> Self {
        self.measurement_order = Some(order);
        self
    }

    /// Build the [`DemStabSim`], consuming the builder.
    ///
    /// # Errors
    ///
    /// Returns [`DemStabError::MissingCircuit`] if no circuit was set.
    pub fn build(self) -> Result<DemStabSim, DemStabError> {
        let dag = self.circuit.ok_or(DemStabError::MissingCircuit)?;

        let analyzer = DagFaultAnalyzer::new(&dag);
        let influence_map = analyzer.build_influence_map();

        let detector_records: Vec<Vec<i32>> =
            self.detectors.iter().map(|d| d.records.to_vec()).collect();
        let observable_records: Vec<Vec<i32>> = self
            .observables
            .iter()
            .map(|o| o.records.to_vec())
            .collect();

        let mut builder = DemSamplerBuilder::new(&influence_map);
        builder = if let Some(cfg) = self.per_gate_noise {
            builder.with_per_gate_noise(cfg)
        } else {
            builder.with_noise(
                self.noise.p1,
                self.noise.p2,
                self.noise.p_meas,
                self.noise.p_prep,
            )
        };
        builder = builder
            .with_detector_records(detector_records)
            .with_observable_records(observable_records);

        if let Some(order) = self.measurement_order {
            builder = builder.with_measurement_order(order);
        }

        let detectors = self.detectors.clone();
        let observables = self.observables.clone();

        Ok(DemStabSim {
            sampler: builder.build()?,
            detectors,
            observables,
        })
    }
}
