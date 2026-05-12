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

//! Unified sampler for noisy QEC measurement outcomes.
//!
//! This sampler unifies the DEM (detector-level) and MNM (measurement-level)
//! sampling paths into a single type. Internally it uses [`DemSampler`]'s
//! efficient geometric-skip engine for fault mechanism sampling, then applies
//! an optional detector basis change and non-deterministic coin flips depending
//! on the requested output mode.
//!
//! # Coordinate systems
//!
//! Deterministic measurements form a basis in `Z_2`. User-defined detectors are
//! linear combinations (XOR chains) of these measurements — a change of basis.
//! The sampler always builds its mechanism table in raw measurement coordinates,
//! then applies the basis change at build time if detector definitions are
//! provided.
//!
//! # Construction modes
//!
//! - **Raw measurements**: each deterministic measurement is its own "detector."
//!   Output includes coin flips for non-deterministic measurements.
//! - **Auto-detected detectors**: uses the influence builder's detector
//!   definitions (round-to-round XOR of stabilizer measurements).
//! - **User-defined detectors**: arbitrary XOR combinations of measurements,
//!   validated at build time.

use super::dem_sampler::SamplingEngine;
use super::types::{DemOutput, NoiseConfig, PerGateTypeNoise};
use crate::fault_tolerance::propagator::{DagFaultInfluenceMap, DemOutputKind};
use pecos_core::prelude::GateType;
use pecos_num::z2_linalg::z2_rank_from_records;
use pecos_random::RngProbabilityExt;
use rand_core::Rng;

/// Errors from detector definition validation.
#[derive(Debug, Clone)]
pub enum DetectorValidationError {
    /// A detector definition references a non-deterministic measurement.
    NonDeterministicReference {
        detector_id: usize,
        measurement_idx: usize,
    },
    /// Detector definitions are not linearly independent over `Z_2`.
    LinearlyDependent { rank: usize, num_detectors: usize },
    /// Circuit contains gates not supported by the symbolic determinism analysis.
    /// Raw measurement mode requires all gates to be in the supported Clifford
    /// subset (`H`, `X`, `Y`, `Z`, `SZ`, `SZdg`, `CX`, `CZ`, `SWAP`, `MZ`, `PZ`, `I`).
    UnsupportedGateForDeterminismAnalysis { gate_type: String },
}

impl std::fmt::Display for DetectorValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonDeterministicReference {
                detector_id,
                measurement_idx,
            } => {
                write!(
                    f,
                    "Detector {detector_id} references non-deterministic measurement {measurement_idx}. \
                     Detectors should only XOR deterministic measurements."
                )
            }
            Self::LinearlyDependent {
                rank,
                num_detectors,
            } => {
                write!(
                    f,
                    "Detector definitions are not linearly independent: \
                     rank {rank} < {num_detectors} detectors. \
                     Some detectors are redundant (XOR of other detectors)."
                )
            }
            Self::UnsupportedGateForDeterminismAnalysis { gate_type } => {
                write!(
                    f,
                    "Circuit contains gate type '{gate_type}' which is not supported by \
                     raw measurement determinism analysis. Supported Clifford gates: \
                     H, X, Y, Z, SZ, SZdg, CX, CZ, SWAP, MZ, PZ/QAlloc, I/Idle."
                )
            }
        }
    }
}

impl std::error::Error for DetectorValidationError {}

/// Error returned when a sampler backend is asked to directly evaluate tracked
/// Paulis it only preserves as metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackedPauliSamplingError {
    backend: &'static str,
    num_tracked_paulis: usize,
}

impl TrackedPauliSamplingError {
    fn new(backend: &'static str, num_tracked_paulis: usize) -> Self {
        Self {
            backend,
            num_tracked_paulis,
        }
    }

    /// Backend that rejected direct tracked-Pauli sampling.
    #[must_use]
    pub fn backend(&self) -> &'static str {
        self.backend
    }

    /// Number of tracked Paulis carried as metadata by that backend.
    #[must_use]
    pub fn num_tracked_paulis(&self) -> usize {
        self.num_tracked_paulis
    }
}

impl std::fmt::Display for TrackedPauliSamplingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} cannot directly sample tracked Pauli flips for {} tracked Pauli(s). \
             This backend samples decoder-facing detectors and observables only; tracked \
             Paulis are preserved as PECOS metadata and fault effects.",
            self.backend, self.num_tracked_paulis
        )
    }
}

impl std::error::Error for TrackedPauliSamplingError {}

/// Output mode for the unified sampler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputMode {
    /// Output raw measurement values (deterministic flips + non-det coin flips).
    RawMeasurements,
    /// Output detector events (XOR of measurement groups) + observable flips.
    DetectorEvents,
}

/// Unified sampler that handles both measurement-level and detector-level output.
///
/// Uses [`DemSampler`]'s geometric-skip engine internally. The mechanism table
/// is always in the output coordinate system (raw measurements or user detectors),
/// determined at build time.
/// Result of dual-mode sampling: both raw measurements and detector events.
#[derive(Debug, Clone)]
pub struct DualSampleResult {
    /// Raw measurement values (deterministic flips + non-det coin flips).
    pub raw_measurements: Vec<bool>,
    /// Detector events (XOR of measurement groups).
    pub detector_events: Vec<bool>,
    /// Standard DEM `L<n>` observable output flips.
    pub dem_output_flips: Vec<bool>,
}

/// Labels for sampler output channels.
#[derive(Debug, Clone, Default)]
pub struct SamplerLabels {
    /// Labels for output channels (raw measurements or detectors, depending on mode).
    pub outputs: Vec<Option<String>>,
    /// Labels for standard DEM `L<n>` observable outputs.
    /// Indices match `per_dem_output` in `SamplingStatistics`.
    pub dem_output_labels: Vec<Option<String>>,
    /// Full PECOS metadata for standard DEM `L<n>` observables.
    pub dem_outputs: Vec<Option<DemOutput>>,
    /// Labels for PECOS tracked Paulis.
    pub tracked_pauli_labels: Vec<Option<String>>,
    /// Full PECOS metadata for tracked Paulis in their own ID space.
    pub tracked_paulis: Vec<Option<DemOutput>>,
    /// Labels for dual-output detector channels.
    pub dual_detectors: Vec<Option<String>>,
}

fn dem_outputs_by_id(targets: &[DemOutput], num_dem_outputs: usize) -> Vec<Option<DemOutput>> {
    let mut by_id = vec![None; num_dem_outputs];
    for target in targets {
        let idx = target.id as usize;
        if idx < by_id.len() {
            by_id[idx] = Some(target.clone());
        }
    }
    by_id
}

fn labels_from_dem_outputs(targets: &[Option<DemOutput>]) -> Vec<Option<String>> {
    targets
        .iter()
        .map(|target| target.as_ref().and_then(|target| target.label.clone()))
        .collect()
}

fn dem_outputs_from_influence_map(
    influence_map: &DagFaultInfluenceMap,
    num_dem_outputs: usize,
) -> Vec<Option<DemOutput>> {
    let mut targets = vec![None; num_dem_outputs];
    for (internal_id, metadata) in influence_map.dem_output_metadata.iter().enumerate() {
        if metadata.kind == DemOutputKind::Observable {
            #[allow(clippy::cast_possible_truncation)] // DEM output count fits in u32
            if let Some(dem_output_id) =
                influence_map.observable_id_for_internal_dem_output(internal_id as u32)
            {
                let idx = dem_output_id as usize;
                if idx < targets.len() {
                    targets[idx] = Some(DemOutput::from_metadata(dem_output_id, metadata));
                }
            }
        }
    }
    targets
}

fn tracked_paulis_from_influence_map(
    influence_map: &DagFaultInfluenceMap,
) -> Vec<Option<DemOutput>> {
    let mut tracked_paulis = Vec::new();
    for metadata in &influence_map.dem_output_metadata {
        if metadata.kind == DemOutputKind::TrackedPauli {
            #[allow(clippy::cast_possible_truncation)] // tracked-Pauli count fits in u32
            let id = tracked_paulis.len() as u32;
            tracked_paulis.push(Some(DemOutput::from_metadata(id, metadata)));
        }
    }
    tracked_paulis
}

fn dem_outputs_from_records(
    influence_map: &DagFaultInfluenceMap,
    observable_records: &[Vec<i32>],
    num_dem_outputs: usize,
) -> Vec<Option<DemOutput>> {
    let mut targets = dem_outputs_from_influence_map(influence_map, num_dem_outputs);

    for (record_id, records) in observable_records.iter().enumerate() {
        let dem_output_id = record_id;
        if dem_output_id < targets.len() {
            if let Some(target) = &mut targets[dem_output_id] {
                if target.records.is_empty() {
                    target.records = DemOutput::new(target.id)
                        .with_records(records.iter().copied())
                        .records;
                }
                target.kind.get_or_insert(DemOutputKind::Observable);
            } else {
                #[allow(clippy::cast_possible_truncation)] // DEM output count fits in u32
                {
                    targets[dem_output_id] = Some(
                        DemOutput::new(dem_output_id as u32).with_records(records.iter().copied()),
                    );
                }
            }
        }
    }

    targets
}

fn merge_dem_output_metadata(
    mut labels: SamplerLabels,
    targets: Vec<Option<DemOutput>>,
    tracked_paulis: Vec<Option<DemOutput>>,
) -> SamplerLabels {
    if labels.dem_outputs.len() < targets.len() {
        labels.dem_outputs.resize(targets.len(), None);
    }
    for (idx, target) in targets.into_iter().enumerate() {
        if labels.dem_outputs[idx].is_none() {
            labels.dem_outputs[idx] = target;
        }
    }

    let target_labels = labels_from_dem_outputs(&labels.dem_outputs);
    if labels.dem_output_labels.len() < target_labels.len() {
        labels.dem_output_labels.resize(target_labels.len(), None);
    }
    for (idx, label) in target_labels.into_iter().enumerate() {
        if labels.dem_output_labels[idx].is_none() {
            labels.dem_output_labels[idx] = label;
        }
    }

    if labels.tracked_paulis.len() < tracked_paulis.len() {
        labels.tracked_paulis.resize(tracked_paulis.len(), None);
    }
    for (idx, tracked_pauli) in tracked_paulis.into_iter().enumerate() {
        if labels.tracked_paulis[idx].is_none() {
            labels.tracked_paulis[idx] = tracked_pauli;
        }
    }

    let tracked_pauli_labels = labels_from_dem_outputs(&labels.tracked_paulis);
    if labels.tracked_pauli_labels.len() < tracked_pauli_labels.len() {
        labels
            .tracked_pauli_labels
            .resize(tracked_pauli_labels.len(), None);
    }
    for (idx, label) in tracked_pauli_labels.into_iter().enumerate() {
        if labels.tracked_pauli_labels[idx].is_none() {
            labels.tracked_pauli_labels[idx] = label;
        }
    }

    labels
}

#[derive(Debug, Clone)]
pub struct DemSampler {
    /// The efficient sampling engine (mechanism table in raw measurement coords).
    inner: SamplingEngine,

    /// Which output indices are non-deterministic (true = coin flip, not from mechanisms).
    /// Length = `num_outputs` (full measurement space in raw mode).
    non_det_mask: Vec<bool>,

    /// Deterministic measurement dependencies for raw mode.
    /// `measurement_deps[i] = Some((deps, flip))` means measurement i is determined by
    /// XOR(measurements[j] for j in deps) XOR flip. None = non-det (coin flip) or fault-only.
    /// Used to propagate non-det coin flips through the dependency chain.
    measurement_deps: Vec<Option<(Vec<usize>, bool)>>,

    /// Detector definitions for dual-output mode.
    /// Each entry is a list of absolute measurement indices to XOR.
    detector_records_abs: Vec<Vec<usize>>,

    /// Output mode this sampler was built for.
    mode: OutputMode,

    /// Total number of output channels (measurements or detectors).
    num_outputs: usize,

    /// Total number of outputs in the DEM `L<n>` namespace.
    num_dem_outputs: usize,

    /// Optional labels for output channels.
    labels: SamplerLabels,

    /// Remap table for raw mode: engine index → absolute measurement index.
    /// When set, the engine operates in compressed coordinates (only fault-reachable
    /// measurements) and the output is expanded to the full measurement space.
    /// None when engine coordinates == output coordinates (no expansion needed).
    raw_remap: Option<Vec<usize>>,
}

impl DemSampler {
    /// Build a `DemSampler` directly from an annotated circuit and noise config.
    ///
    /// This is the simplest way to go from circuit to sampler. It:
    /// 1. Builds a raw-measurement influence map via `DagFaultAnalyzer`
    /// 2. Extracts detector, observable, and Pauli check annotations from the circuit
    /// 3. Applies the noise configuration
    /// 4. Returns a ready-to-sample `DemSampler`
    ///
    /// For circuits with Pauli check annotations, this also builds
    /// the influence map with those checks via `InfluenceBuilder`.
    ///
    /// # Errors
    ///
    /// Returns [`DetectorValidationError`] if any detector references a
    /// non-deterministic measurement or the detectors are linearly dependent.
    ///
    /// # Example
    ///
    /// ```
    /// use rand::SeedableRng;
    /// use rand::rngs::StdRng;
    ///
    /// use pecos_qec::fault_tolerance::dem_builder::{DemSampler, NoiseConfig};
    /// use pecos_quantum::DagCircuit;
    ///
    /// let dag = DagCircuit::new();
    /// let noise = NoiseConfig::uniform(0.01);
    /// let sampler = DemSampler::from_circuit(&dag, &noise).unwrap();
    ///
    /// let mut rng = StdRng::seed_from_u64(123);
    /// let (det, obs) = sampler.sample(&mut rng);
    /// assert!(det.is_empty());
    /// assert!(obs.is_empty());
    /// ```
    /// Build a sampler from a `TickCircuit` and noise parameters.
    ///
    /// Converts to `DagCircuit` internally. Returns detector-mode sampler.
    pub fn from_tick_circuit(
        circuit: &pecos_quantum::TickCircuit,
        noise: &super::types::NoiseConfig,
    ) -> Result<Self, DetectorValidationError> {
        let dag = pecos_quantum::DagCircuit::from(circuit);
        Self::from_circuit(&dag, noise)
    }

    /// Build a sampler from a `DagCircuit` and noise parameters.
    ///
    /// # Errors
    ///
    /// Returns [`DetectorValidationError`] when detector metadata is invalid
    /// for the circuit's measurement record.
    pub fn from_circuit(
        circuit: &pecos_quantum::DagCircuit,
        noise: &super::types::NoiseConfig,
    ) -> Result<Self, DetectorValidationError> {
        // Build the DetectorErrorModel via DemBuilder (single code path for
        // DEM computation), then convert to sampler.
        use super::builder::DemBuilder;
        use crate::fault_tolerance::influence_builder::InfluenceBuilder;
        use crate::fault_tolerance::propagator::DagFaultAnalyzer;

        let mut influence_map = DagFaultAnalyzer::new(circuit).build_influence_map();
        let annotation_map = InfluenceBuilder::new(circuit)
            .with_circuit_annotations(circuit)
            .build();
        influence_map.merge_dem_outputs_from(&annotation_map);

        // Extract metadata before building (avoids ownership issues with builder methods)
        let det_json = {
            use pecos_num::graph::Attribute;
            circuit.get_attr("detectors").and_then(|a| {
                if let Attribute::String(s) = a {
                    Some(s.clone())
                } else {
                    None
                }
            })
        };
        let observables_json = {
            use pecos_num::graph::Attribute;
            circuit.get_attr("observables").and_then(|a| {
                if let Attribute::String(s) = a {
                    Some(s.clone())
                } else {
                    None
                }
            })
        };
        let num_meas = {
            use pecos_num::graph::Attribute;
            circuit.get_attr("num_measurements").and_then(|a| {
                if let Attribute::String(s) = a {
                    s.parse::<usize>().ok()
                } else {
                    None
                }
            })
        };

        // Build DemBuilder, applying detector/DEM-output JSON if available.
        // with_detectors_json/with_observables_json consume self, so we
        // chain them carefully.
        let builder = DemBuilder::new(&influence_map).with_noise_config(noise.clone());

        let builder = if let Some(ref dj) = det_json {
            builder.with_detectors_json(dj).unwrap_or_else(|_| {
                DemBuilder::new(&influence_map).with_noise_config(noise.clone())
            })
        } else {
            builder
        };

        let builder = if let Some(ref oj) = observables_json {
            builder.with_observables_json(oj).unwrap_or_else(|_| {
                DemBuilder::new(&influence_map).with_noise_config(noise.clone())
            })
        } else {
            builder
        };

        let builder = if let Some(n) = num_meas {
            builder.with_num_measurements(n)
        } else {
            builder
        };

        let dem = builder.build();
        Ok(Self::from_detector_error_model(&dem))
    }

    /// Wrap a raw [`SamplingEngine`] as a detector-mode `DemSampler`.
    ///
    /// Used when the engine was constructed externally (e.g., from
    /// [`ParsedDem::to_dem_sampler`]).
    #[must_use]
    /// Create a `DemSampler` from a pre-built `SamplingEngine`.
    pub fn from_engine(engine: SamplingEngine) -> Self {
        let num_outputs = engine.num_detectors();
        let num_dem_outputs = engine.num_dem_outputs();
        Self {
            inner: engine,
            non_det_mask: Vec::new(),
            detector_records_abs: Vec::new(),
            mode: OutputMode::DetectorEvents,
            num_outputs,
            num_dem_outputs,
            labels: SamplerLabels::default(),
            raw_remap: None,
            measurement_deps: Vec::new(),
        }
    }

    /// Build a detector-event sampler from a [`DetectorErrorModel`], preserving
    /// PECOS metadata for observables and tracked Paulis.
    #[must_use]
    pub fn from_detector_error_model(dem: &super::types::DetectorErrorModel) -> Self {
        let (mechanisms, _coords) = dem.to_mechanisms();
        let engine =
            SamplingEngine::from_mechanisms(mechanisms, dem.num_detectors(), dem.num_dem_outputs());
        let mut sampler = Self::from_engine(engine);
        sampler.labels.dem_outputs = dem_outputs_by_id(dem.dem_outputs(), dem.num_dem_outputs());
        sampler.labels.dem_output_labels = labels_from_dem_outputs(&sampler.labels.dem_outputs);
        sampler.labels.tracked_paulis =
            dem_outputs_by_id(dem.tracked_paulis(), dem.num_tracked_paulis());
        sampler.labels.tracked_pauli_labels =
            labels_from_dem_outputs(&sampler.labels.tracked_paulis);
        sampler
    }

    /// Attach observable and tracked-Pauli metadata to an existing sampler.
    ///
    /// This is useful for parser paths where the sampling engine projects to
    /// detector/observable columns but the original PECOS DEM still declared
    /// tracked Paulis in a separate ID space.
    #[must_use]
    pub fn with_dem_output_metadata(
        mut self,
        dem_outputs: Vec<Option<DemOutput>>,
        tracked_paulis: Vec<Option<DemOutput>>,
    ) -> Self {
        self.labels.dem_outputs = dem_outputs;
        self.labels.dem_output_labels = labels_from_dem_outputs(&self.labels.dem_outputs);
        self.labels.tracked_paulis = tracked_paulis;
        self.labels.tracked_pauli_labels = labels_from_dem_outputs(&self.labels.tracked_paulis);
        self
    }

    /// Reconstruct a detector error model from the compiled mechanism table.
    ///
    /// The returned model contains mechanism probabilities and effects. Higher
    /// level wrappers that own detector / observable definitions should add
    /// those declarations to preserve metadata in serialized text.
    #[must_use]
    pub fn to_detector_error_model(&self) -> super::types::DetectorErrorModel {
        self.inner.to_detector_error_model()
    }

    /// Create a `DemSampler` directly from an influence map with per-location
    /// probabilities (raw measurement mode).
    #[must_use]
    pub fn from_influence_map(
        influence_map: &DagFaultInfluenceMap,
        per_location_probs: &[f64],
    ) -> Self {
        let default_noise = super::NoiseConfig::default();
        let inner =
            SamplingEngine::from_influence_map(influence_map, per_location_probs, &default_noise);
        let num_outputs = inner.num_detectors();
        let num_dem_outputs = inner.num_dem_outputs();
        let mut labels = SamplerLabels::default();
        labels.dem_outputs = dem_outputs_from_influence_map(influence_map, num_dem_outputs);
        labels.dem_output_labels = labels_from_dem_outputs(&labels.dem_outputs);
        labels.tracked_paulis = tracked_paulis_from_influence_map(influence_map);
        labels.tracked_pauli_labels = labels_from_dem_outputs(&labels.tracked_paulis);
        Self {
            inner,
            non_det_mask: Vec::new(),
            detector_records_abs: Vec::new(),
            mode: OutputMode::RawMeasurements,
            num_outputs,
            num_dem_outputs,
            labels,
            raw_remap: None,
            measurement_deps: Vec::new(),
        }
    }

    /// Number of output channels (measurements in raw mode, detectors in detector mode).
    #[must_use]
    pub fn num_outputs(&self) -> usize {
        self.num_outputs
    }

    /// Number of detectors (alias for [`num_outputs`] in detector mode).
    #[must_use]
    pub fn num_detectors(&self) -> usize {
        self.num_outputs
    }

    /// Number of observables.
    #[must_use]
    pub fn num_observables(&self) -> usize {
        self.num_dem_outputs
    }

    /// Number of DEM `L<n>` output channels.
    #[must_use]
    pub fn num_dem_outputs(&self) -> usize {
        self.num_dem_outputs
    }

    /// Number of tracked Paulis.
    #[must_use]
    pub fn num_tracked_paulis(&self) -> usize {
        self.labels.tracked_paulis.len()
    }

    /// Standard observable `L<n>` IDs selected from this sampler.
    #[must_use]
    pub fn observable_ids(&self) -> Vec<usize> {
        (0..self.num_dem_outputs).collect()
    }

    /// PECOS tracked-Pauli IDs selected from this sampler.
    ///
    /// Decoder-facing DEM samplers do not directly evaluate tracked Paulis:
    /// tracked Paulis are preserved in metadata and in PECOS DEM fault
    /// effects, but the sampled bit columns are detectors plus standard
    /// observable `L<n>` outputs only.
    ///
    /// # Errors
    ///
    /// Returns [`TrackedPauliSamplingError`] when tracked Paulis are
    /// present and the caller is asking for a direct sampled tracked-Pauli
    /// output space.
    pub fn tracked_pauli_ids(&self) -> Result<Vec<usize>, TrackedPauliSamplingError> {
        self.ensure_tracked_pauli_sampling_supported()?;
        Ok(Vec::new())
    }

    /// Sample direct tracked-Pauli flips.
    ///
    /// This returns an empty vector when the sampler carries no tracked
    /// Paulis. If tracked Paulis are present, this backend fails
    /// explicitly instead of returning silently empty data.
    ///
    /// # Errors
    ///
    /// Returns [`TrackedPauliSamplingError`] when tracked Paulis are
    /// present because [`DemSampler`] samples detector and observable columns,
    /// not tracked-Pauli columns.
    pub fn sample_tracked_pauli_flips<R: Rng>(
        &self,
        _rng: &mut R,
    ) -> Result<Vec<bool>, TrackedPauliSamplingError> {
        self.ensure_tracked_pauli_sampling_supported()?;
        Ok(Vec::new())
    }

    /// Sample direct tracked-Pauli flips for multiple shots.
    ///
    /// # Errors
    ///
    /// Returns [`TrackedPauliSamplingError`] when tracked Paulis are
    /// present for the same reason as [`Self::sample_tracked_pauli_flips`].
    pub fn sample_tracked_pauli_batch<R: Rng>(
        &self,
        num_shots: usize,
        _rng: &mut R,
    ) -> Result<Vec<Vec<bool>>, TrackedPauliSamplingError> {
        self.ensure_tracked_pauli_sampling_supported()?;
        Ok(vec![Vec::new(); num_shots])
    }

    fn ensure_tracked_pauli_sampling_supported(&self) -> Result<(), TrackedPauliSamplingError> {
        let num_tracked_paulis = self.num_tracked_paulis();
        if num_tracked_paulis == 0 {
            Ok(())
        } else {
            Err(TrackedPauliSamplingError::new(
                "DemSampler",
                num_tracked_paulis,
            ))
        }
    }

    /// Bit mask selecting observable outputs.
    ///
    /// Existing decoder APIs use `u64` observable masks, so outputs with index
    /// \>= 64 are not representable here and are ignored consistently with the
    /// existing mask-based paths.
    #[must_use]
    pub fn observable_dem_output_mask(&self) -> u64 {
        self.observable_ids()
            .into_iter()
            .filter(|&idx| idx < u64::BITS as usize)
            .fold(0u64, |acc, idx| acc | (1u64 << idx))
    }

    /// Converts a sampled DEM-output flip vector into an observable-only mask.
    #[must_use]
    pub fn observable_mask_from_dem_output_flips(&self, flips: &[bool]) -> u64 {
        let observable_mask = self.observable_dem_output_mask();
        flips
            .iter()
            .enumerate()
            .filter(|(idx, flipped)| {
                **flipped && *idx < u64::BITS as usize && (observable_mask & (1u64 << *idx)) != 0
            })
            .fold(0u64, |acc, (idx, _)| acc | (1u64 << idx))
    }

    /// Number of mechanisms in the sampler.
    #[must_use]
    pub fn num_mechanisms(&self) -> usize {
        self.inner.num_mechanisms()
    }

    /// Average mechanism firing probability.
    #[must_use]
    pub fn average_error_probability(&self) -> f64 {
        self.inner.average_error_probability()
    }

    /// Maximum mechanism firing probability.
    #[must_use]
    pub fn max_error_probability(&self) -> f64 {
        self.inner.max_error_probability()
    }

    /// Get the labels for this sampler's output channels.
    #[must_use]
    pub fn labels(&self) -> &SamplerLabels {
        &self.labels
    }

    /// Output mode this sampler was built for.
    #[must_use]
    pub fn mode(&self) -> OutputMode {
        self.mode
    }

    /// Finalize raw measurement outputs: expand coordinates, apply non-det coin
    /// flips, and propagate deterministic dependencies.
    ///
    /// This is the single post-processing path for all raw-mode sampling methods.
    fn finalize_raw_output<R: Rng>(&self, engine_outputs: Vec<bool>, rng: &mut R) -> Vec<bool> {
        // Step 1: Expand engine output to full measurement space if remapping
        let mut outputs = if let Some(ref remap) = self.raw_remap {
            let mut full = vec![false; self.num_outputs];
            for (engine_idx, &abs_idx) in remap.iter().enumerate() {
                if engine_idx < engine_outputs.len() && abs_idx < full.len() {
                    full[abs_idx] = engine_outputs[engine_idx];
                }
            }
            full
        } else {
            engine_outputs
        };

        // Step 2: Add coin flips for non-deterministic measurements
        for (i, &is_non_det) in self.non_det_mask.iter().enumerate() {
            if is_non_det && i < outputs.len() {
                outputs[i] ^= rng.coin_flip();
            }
        }

        // Step 3: Propagate deterministic measurement dependencies.
        // For m_i with deps {j, k, ...}: m_i XOR= XOR(m_j, m_k, ...) XOR flip
        // Dependencies are always to earlier measurements (processed in order).
        for i in 0..outputs.len().min(self.measurement_deps.len()) {
            if let Some((ref deps, flip)) = self.measurement_deps[i] {
                let dep_xor = deps
                    .iter()
                    .filter(|&&j| j < outputs.len())
                    .fold(flip, |acc, &j| acc ^ outputs[j]);
                outputs[i] ^= dep_xor;
            }
        }

        outputs
    }

    /// Sample a single shot.
    ///
    /// Returns `(outputs, dem_output_flips)` where outputs are either raw
    /// measurement values or detector events depending on the mode.
    #[must_use]
    pub fn sample<R: Rng>(&self, rng: &mut R) -> (Vec<bool>, Vec<bool>) {
        let (engine_outputs, dem_outputs) = self.inner.sample(rng);

        let outputs = if self.mode == OutputMode::RawMeasurements {
            self.finalize_raw_output(engine_outputs, rng)
        } else {
            engine_outputs
        };

        (outputs, dem_outputs)
    }

    /// Sample multiple shots.
    #[must_use]
    pub fn sample_batch<R: Rng>(
        &self,
        num_shots: usize,
        rng: &mut R,
    ) -> (Vec<Vec<bool>>, Vec<Vec<bool>>) {
        let (engine_batches, all_dem_outputs) = self.inner.sample_batch(num_shots, rng);

        let all_outputs: Vec<Vec<bool>> = if self.mode == OutputMode::RawMeasurements {
            engine_batches
                .into_iter()
                .map(|engine_out| self.finalize_raw_output(engine_out, rng))
                .collect()
        } else {
            engine_batches
        };

        (all_outputs, all_dem_outputs)
    }

    /// Batch sample using geometric skip — O(fired) instead of O(all mechanisms).
    ///
    /// Returns columnar bit-packed data:
    /// - detector columns: `[num_detectors][ceil(num_shots/64)]` u64 words
    /// - `L<n>` target columns: `[num_dem_outputs][ceil(num_shots/64)]` u64 words
    ///
    /// Much faster than `sample_batch` at low error rates where few mechanisms fire.
    /// Only available in detector-event mode (not raw measurement mode).
    ///
    /// # Panics
    ///
    /// Panics if the sampler is in raw measurement mode.
    #[must_use]
    pub fn sample_batch_geometric<R: Rng>(
        &self,
        num_shots: usize,
        rng: &mut R,
    ) -> (Vec<Vec<u64>>, Vec<Vec<u64>>) {
        assert!(
            self.mode != OutputMode::RawMeasurements,
            "sample_batch_geometric() does not support raw measurement mode \
             (requires non-det coin flips + dependency propagation per shot). \
             Use sample_batch() instead."
        );
        self.inner.sample_batch_columnar_geometric(num_shots, rng)
    }

    /// Sample a single shot and return both raw measurements and detector events.
    ///
    /// This uses a single RNG sequence to produce both outputs consistently.
    /// Requires the sampler to have been built in raw measurement mode with
    /// detector definitions stored via the builder.
    ///
    /// Returns `None` if no detector definitions are available.
    #[must_use]
    pub fn sample_dual<R: Rng>(&self, rng: &mut R) -> Option<DualSampleResult> {
        if self.detector_records_abs.is_empty() {
            return None;
        }

        // Sample mechanism flips in raw measurement coordinates
        let (raw_flips, dem_output_flips) = self.inner.sample(rng);

        // Finalize raw measurements (expand, coin flips, dependency propagation)
        let raw_measurements = self.finalize_raw_output(raw_flips, rng);

        // Compute detector events from FINALIZED raw measurements
        // (includes non-det coin flips and dependency propagation)
        let detector_events: Vec<bool> = self
            .detector_records_abs
            .iter()
            .map(|record| {
                record.iter().fold(false, |acc, &idx| {
                    acc ^ raw_measurements.get(idx).copied().unwrap_or(false)
                })
            })
            .collect();

        Some(DualSampleResult {
            raw_measurements,
            detector_events,
            dem_output_flips,
        })
    }

    /// Compute statistics with a user-provided RNG.
    #[must_use]
    pub fn sample_statistics_with_rng<R: Rng>(
        &self,
        num_shots: usize,
        rng: &mut R,
    ) -> super::dem_sampler::SamplingStatistics {
        let observable_indices = self.observable_ids();
        self.inner
            .sample_statistics_with_rng_for_observable_indices(num_shots, rng, &observable_indices)
    }

    /// Compute statistics without storing individual shots.
    ///
    /// Delegates to [`DemSampler::sample_statistics`] which auto-selects
    /// the fastest algorithm. Non-deterministic coin flips do NOT affect
    /// statistics since they are independent of faults and cancel in
    /// expectation for any well-formed detector.
    #[must_use]
    pub fn sample_statistics(
        &self,
        num_shots: usize,
        seed: u64,
    ) -> super::dem_sampler::SamplingStatistics {
        let observable_indices = self.observable_ids();
        self.inner
            .sample_statistics_for_observable_indices(num_shots, seed, &observable_indices)
    }
}

// ============================================================================
// Builder
// ============================================================================

/// Builder for [`DemSampler`].
///
/// Constructs a sampler from a fault influence map and noise parameters.
/// The output mode (raw measurements vs detector events) is determined by
/// how the builder is configured.
pub struct DemSamplerBuilder<'a> {
    influence_map: &'a DagFaultInfluenceMap,
    noise: NoiseConfig,
    per_gate: Option<PerGateTypeNoise>,
    output_mode: OutputMode,
    detector_records: Option<Vec<Vec<i32>>>,
    observable_records: Option<Vec<Vec<i32>>>,
    measurement_order: Option<Vec<usize>>,
    detector_records_abs: Option<Vec<Vec<usize>>>,
    labels: SamplerLabels,
}

impl<'a> DemSamplerBuilder<'a> {
    /// Create a new builder. Default mode is raw measurements.
    #[must_use]
    pub fn new(influence_map: &'a DagFaultInfluenceMap) -> Self {
        Self {
            influence_map,
            noise: NoiseConfig::default(),
            per_gate: None,
            output_mode: OutputMode::RawMeasurements,
            detector_records: None,
            observable_records: None,
            measurement_order: None,
            detector_records_abs: None,
            labels: SamplerLabels::default(),
        }
    }

    /// Set noise parameters.
    #[must_use]
    pub fn with_noise(mut self, p1: f64, p2: f64, p_meas: f64, p_prep: f64) -> Self {
        self.noise = NoiseConfig::new(p1, p2, p_meas, p_prep);
        self
    }

    /// Set noise from a `NoiseConfig` (includes `p_idle` if set).
    #[must_use]
    pub fn with_noise_config(mut self, config: NoiseConfig) -> Self {
        self.noise = config;
        self
    }

    /// Set per-gate-type and optional per-qubit Pauli rates.
    #[must_use]
    pub fn with_per_gate_noise(mut self, config: PerGateTypeNoise) -> Self {
        self.noise = config.base.clone();
        self.per_gate = Some(config);
        self
    }

    /// Set uniform noise (same probability for all gate types, including idle).
    #[must_use]
    pub fn with_uniform_noise(self, p: f64) -> Self {
        let mut s = self.with_noise(p, p, p, p);
        s.noise.p_idle = p;
        s
    }

    /// Set idle gate noise rate.
    #[must_use]
    pub fn with_idle_noise(mut self, p_idle: f64) -> Self {
        self.noise.p_idle = p_idle;
        self
    }

    /// Request raw measurement output (default).
    ///
    /// Each deterministic measurement is its own output channel. Non-deterministic
    /// measurements get independent coin flips.
    #[must_use]
    pub fn raw_measurements(mut self) -> Self {
        self.output_mode = OutputMode::RawMeasurements;
        self.detector_records = None;
        self.observable_records = None;
        self
    }

    /// Request detector-event output with the given detector/DEM output definitions.
    ///
    /// Detector records use DEM-style negative offsets: `[-1]` means "the last
    /// measurement", `[-3, -1]` means "XOR of the last and third-to-last."
    #[must_use]
    pub fn with_detectors(
        mut self,
        detector_records: Vec<Vec<i32>>,
        observable_records: Vec<Vec<i32>>,
    ) -> Self {
        self.output_mode = OutputMode::DetectorEvents;
        self.detector_records = Some(detector_records);
        self.observable_records = Some(observable_records);
        self
    }

    /// Set detector records directly (without observables).
    #[must_use]
    pub fn with_detector_records(mut self, records: Vec<Vec<i32>>) -> Self {
        self.output_mode = OutputMode::DetectorEvents;
        self.detector_records = Some(records);
        if self.observable_records.is_none() {
            self.observable_records = Some(Vec::new());
        }
        self
    }

    /// Set observable definitions directly.
    #[must_use]
    pub fn with_observable_records(mut self, records: Vec<Vec<i32>>) -> Self {
        self.observable_records = Some(records);
        self
    }

    /// Set detector definitions from JSON.
    ///
    /// Format: `[{"id": 0, "records": [-1, -5]}, ...]`
    ///
    /// # Errors
    /// Returns an error if the JSON is malformed.
    pub fn with_detectors_json(self, json: &str) -> Result<Self, String> {
        let records = parse_records_json(json);
        Ok(self.with_detector_records(records))
    }

    /// Set observable definitions from JSON.
    ///
    /// Format: `[{"id": 0, "records": [-1, -3, -5]}, ...]`
    ///
    /// # Errors
    /// Returns an error if the JSON is malformed.
    pub fn with_observables_json(self, json: &str) -> Result<Self, String> {
        let records = parse_records_json(json);
        Ok(self.with_observable_records(records))
    }

    /// Enable dual output (raw measurements + detector events from same sample).
    ///
    /// When building in raw measurement mode, stores the detector definitions
    /// so that [`DemSampler::sample_dual`] can compute both outputs.
    /// The records use absolute measurement indices (not negative offsets).
    #[must_use]
    pub fn with_dual_output(mut self, detector_records_abs: Vec<Vec<usize>>) -> Self {
        self.detector_records_abs = Some(detector_records_abs);
        self
    }

    /// Extract detector, observable, and tracked-Pauli definitions from a [`DagCircuit`]'s
    /// in-circuit annotations.
    ///
    /// Extract annotations from a [`DagCircuit`] and configure the sampler.
    ///
    /// Detector annotations are mapped to auto-detected detector indices.
    /// Observables are converted to measurement-record outputs. Tracked
    /// Paulis remain unmeasured Pauli annotations and are carried
    /// through PECOS metadata only.
    #[must_use]
    pub fn with_circuit_annotations(mut self, circuit: &pecos_quantum::DagCircuit) -> Self {
        use pecos_quantum::AnnotationKind;

        let mut node_to_meas_idx: std::collections::BTreeMap<usize, usize> =
            std::collections::BTreeMap::new();
        for (meas_idx, &(node, _qubit, _basis)) in
            self.influence_map.measurements.iter().enumerate()
        {
            node_to_meas_idx.entry(node).or_insert(meas_idx);
        }

        let detectors: Vec<&pecos_quantum::PauliAnnotation> = circuit.detectors().collect();
        let observables: Vec<&pecos_quantum::PauliAnnotation> = circuit.observables().collect();

        // Map user-defined detector annotations to auto-detected detector indices
        if !detectors.is_empty() {
            // For each IM measurement index, find which auto-detector contains it
            let mut meas_idx_to_auto_det: Vec<Option<usize>> =
                vec![None; self.influence_map.measurements.len()];
            for (det_idx, det) in self.influence_map.detectors.iter().enumerate() {
                for meas_id in &det.measurements {
                    for (im_idx, &(_node, qubit, basis)) in
                        self.influence_map.measurements.iter().enumerate()
                    {
                        if qubit == meas_id.qubit
                            && basis == meas_id.basis
                            && meas_idx_to_auto_det[im_idx].is_none()
                        {
                            meas_idx_to_auto_det[im_idx] = Some(det_idx);
                            break;
                        }
                    }
                }
            }

            // Map each user detector: measurement_nodes → IM meas index → auto-detector index
            let det_records_abs: Vec<Vec<usize>> = detectors
                .iter()
                .map(|ann| {
                    if let AnnotationKind::Detector {
                        measurement_nodes, ..
                    } = &ann.kind
                    {
                        measurement_nodes
                            .iter()
                            .filter_map(|&node| {
                                let im_idx = node_to_meas_idx.get(&node)?;
                                meas_idx_to_auto_det[*im_idx]
                            })
                            .collect()
                    } else {
                        Vec::new()
                    }
                })
                .collect();

            self.labels.dual_detectors = detectors.iter().map(|a| a.label.clone()).collect();
            self.detector_records_abs = Some(det_records_abs);
        }

        if !observables.is_empty() && self.observable_records.is_none() {
            let records = if let Ok(num_measurements) =
                i32::try_from(self.influence_map.measurements.len())
            {
                observables
                    .iter()
                    .map(|ann| {
                        if let AnnotationKind::Observable { measurement_nodes } = &ann.kind {
                            measurement_nodes
                                .iter()
                                .filter_map(|node| node_to_meas_idx.get(node).copied())
                                .filter_map(|meas_idx| {
                                    i32::try_from(meas_idx)
                                        .ok()
                                        .map(|meas_idx| meas_idx - num_measurements)
                                })
                                .collect()
                        } else {
                            Vec::new()
                        }
                    })
                    .collect()
            } else {
                vec![Vec::new(); observables.len()]
            };
            self.observable_records = Some(records);
        }

        let observable_labels: Vec<Option<String>> =
            observables.iter().map(|a| a.label.clone()).collect();
        if !observable_labels.is_empty() {
            self.labels.dem_output_labels = observable_labels;
        }

        let tracked_pauli_labels: Vec<Option<String>> = circuit
            .annotations()
            .iter()
            .filter(|a| matches!(a.kind, AnnotationKind::TrackedPauli))
            .map(|a| a.label.clone())
            .collect();
        if !tracked_pauli_labels.is_empty() {
            self.labels.tracked_pauli_labels = tracked_pauli_labels;
        }

        self
    }

    /// Set the measurement order for legacy circuits without `MeasId` on gates.
    ///
    /// **Not needed for circuits built with `TickCircuit.mz()`** — the `MeasId`
    /// values on gates ensure correct ordering automatically.
    #[must_use]
    pub fn with_measurement_order(mut self, order: Vec<usize>) -> Self {
        self.measurement_order = Some(order);
        self
    }

    /// Build the sampler.
    ///
    /// # Errors
    ///
    /// Returns an error if detector definitions reference non-deterministic
    /// measurements or are not linearly independent over `Z_2`.
    pub fn build(self) -> Result<DemSampler, DetectorValidationError> {
        match self.output_mode {
            OutputMode::RawMeasurements => Ok(self.build_raw()),
            OutputMode::DetectorEvents => self.build_detector(),
        }
    }

    /// Build in raw measurement mode.
    ///
    /// Mechanism table is in measurement coordinates. Non-deterministic
    /// measurements are identified and marked for coin-flip output.
    fn build_raw(self) -> DemSampler {
        let num_measurements = self.influence_map.measurements.len();

        // Build per-location probabilities from gate-type noise
        let per_location_probs = self.compute_per_location_probs();

        // Build mechanism table in raw measurement coordinates
        let inner = SamplingEngine::from_influence_map(
            self.influence_map,
            &per_location_probs,
            &self.noise,
        );

        // Identify non-deterministic measurements.
        // A measurement is deterministic if the influence builder found it
        // as part of a detector definition. If it's NOT in any detector,
        // it might be non-deterministic (first-round stabilizer, data readout).
        //
        // Conservative approach: mark a measurement as non-deterministic if
        // it doesn't appear in any detector definition. This isn't perfect
        // (some deterministic measurements might not be in detectors) but
        // is safe — extra coin flips on deterministic measurements that
        // happen to not be in detectors just add noise.
        let mut in_detector = vec![false; num_measurements];
        for det in &self.influence_map.detectors {
            for m in &det.measurements {
                // Find measurement index by matching qubit + tick
                for (idx, &(_node, qubit, _basis)) in
                    self.influence_map.measurements.iter().enumerate()
                {
                    if qubit == m.qubit {
                        in_detector[idx] = true;
                    }
                }
            }
        }
        let non_det_mask: Vec<bool> = in_detector.iter().map(|&in_det| !in_det).collect();

        let num_dem_outputs = inner.num_dem_outputs();
        let dem_outputs = dem_outputs_from_influence_map(self.influence_map, num_dem_outputs);
        let tracked_paulis = tracked_paulis_from_influence_map(self.influence_map);

        DemSampler {
            inner,
            non_det_mask,
            detector_records_abs: self.detector_records_abs.unwrap_or_default(),
            mode: OutputMode::RawMeasurements,
            num_outputs: num_measurements,
            num_dem_outputs,
            labels: merge_dem_output_metadata(self.labels, dem_outputs, tracked_paulis),
            raw_remap: None,
            measurement_deps: Vec::new(), // No expansion needed (engine covers all measurements)
        }
    }

    /// Build in detector-event mode.
    ///
    /// Validates detector definitions, then uses `DemSamplerBuilder` to build
    /// the mechanism table in detector coordinates.
    fn build_detector(self) -> Result<DemSampler, DetectorValidationError> {
        use super::dem_sampler::SamplingEngineBuilder;

        let num_measurements = self.influence_map.measurements.len();

        // Validate: check which measurements are deterministic (before partial move)
        let deterministic = self.compute_deterministic_mask();

        let detector_records = self.detector_records.unwrap_or_default();
        let observable_records = self.observable_records.unwrap_or_default();
        let num_detectors = detector_records.len();

        // Check that all detector records reference deterministic measurements
        for (det_id, records) in detector_records.iter().enumerate() {
            for &offset in records {
                // Resolve offset to an absolute index: negative offsets count
                // backward from the end of the measurement list.
                #[allow(clippy::cast_sign_loss)] // offset is non-negative in else branch
                let abs_idx = if offset < 0 {
                    let neg = offset.unsigned_abs() as usize;
                    if neg > num_measurements {
                        continue;
                    }
                    num_measurements - neg
                } else {
                    offset as usize
                };

                if abs_idx < num_measurements && !deterministic[abs_idx] {
                    return Err(DetectorValidationError::NonDeterministicReference {
                        detector_id: det_id,
                        measurement_idx: abs_idx,
                    });
                }
            }
        }

        // Check linear independence via Gaussian elimination over Z_2
        if num_detectors > 0 {
            let rank = z2_rank_from_records(&detector_records, num_measurements);
            if rank < num_detectors {
                return Err(DetectorValidationError::LinearlyDependent {
                    rank,
                    num_detectors,
                });
            }
        }

        let mut builder = SamplingEngineBuilder::new(self.influence_map)
            .with_noise(
                self.noise.p1,
                self.noise.p2,
                self.noise.p_meas,
                self.noise.p_prep,
            )
            .with_detector_records(detector_records)
            .with_observable_records(observable_records.clone());

        if let Some(per_gate) = self.per_gate {
            builder = builder.with_per_gate_noise(per_gate);
        } else if self.noise.uses_dedicated_idle_noise() {
            builder = builder.with_idle_noise_config(self.noise.clone());
        }

        if let Some(order) = self.measurement_order {
            builder = builder.with_measurement_order(order);
        }

        let inner = builder.build();
        let num_dem_outputs = inner.num_dem_outputs();
        let dem_outputs =
            dem_outputs_from_records(self.influence_map, &observable_records, num_dem_outputs);
        let tracked_paulis = tracked_paulis_from_influence_map(self.influence_map);

        Ok(DemSampler {
            inner,
            non_det_mask: Vec::new(),
            detector_records_abs: Vec::new(),
            mode: OutputMode::DetectorEvents,
            num_outputs: num_detectors,
            num_dem_outputs,
            labels: merge_dem_output_metadata(self.labels, dem_outputs, tracked_paulis),
            raw_remap: None,
            measurement_deps: Vec::new(),
        })
    }

    /// Compute which measurements are deterministic.
    ///
    /// A measurement is considered deterministic if it appears in at least
    /// one detector definition in the influence map.
    fn compute_deterministic_mask(&self) -> Vec<bool> {
        let num_measurements = self.influence_map.measurements.len();
        let mut deterministic = vec![false; num_measurements];

        for det in &self.influence_map.detectors {
            for m in &det.measurements {
                for (idx, &(_node, qubit, _basis)) in
                    self.influence_map.measurements.iter().enumerate()
                {
                    if qubit == m.qubit {
                        deterministic[idx] = true;
                    }
                }
            }
        }

        deterministic
    }

    /// Compute per-location error probabilities from gate-type noise config.
    ///
    /// Returns the total error probability per location. For T1/T2 idle noise,
    /// this is the sum of the biased Pauli probabilities.
    fn compute_per_location_probs(&self) -> Vec<f64> {
        compute_location_probs_from_noise(&self.influence_map.locations, &self.noise)
    }
}

/// Compute per-location total error probabilities from noise config.
///
/// For T1/T2 idle noise, returns the sum of biased Pauli probabilities.
/// For all other gates, returns the gate-type probability.
pub(crate) fn compute_location_probs_from_noise(
    locations: &[super::super::propagator::dag::DagSpacetimeLocation],
    noise: &NoiseConfig,
) -> Vec<f64> {
    locations
        .iter()
        .map(|loc| {
            #[allow(clippy::match_same_arms)]
            match loc.gate_type {
                GateType::PZ | GateType::QAlloc => noise.p_prep,
                GateType::MZ | GateType::MeasureFree => noise.p_meas,
                GateType::CX
                | GateType::CZ
                | GateType::CY
                | GateType::SZZ
                | GateType::SZZdg
                | GateType::SXX
                | GateType::SXXdg
                | GateType::SYY
                | GateType::SYYdg
                | GateType::SWAP
                | GateType::RXX
                | GateType::RYY
                | GateType::RZZ => noise.p2,
                GateType::Idle => {
                    if noise.uses_dedicated_idle_noise() {
                        // Duration values are small integers; precision loss is not a concern.
                        #[allow(clippy::cast_precision_loss)]
                        let duration = loc.idle_duration.max(1) as f64;
                        noise.idle_pauli_probs(duration).total()
                    } else {
                        0.0
                    }
                }
                _ => noise.p1,
            }
        })
        .collect()
}

/// Get the per-qubit error probability for a gate fault location.
pub(crate) fn gate_location_prob_from_locations(
    loc: &super::super::propagator::dag::GateFaultLocation<'_>,
    loc_probs: &[f64],
    all_locations: &[super::super::propagator::dag::DagSpacetimeLocation],
) -> f64 {
    for (i, l) in all_locations.iter().enumerate() {
        if l.node == loc.node && l.before == loc.before {
            return loc_probs[i];
        }
    }
    0.0
}

/// Parse detector or observable definitions from JSON.
///
/// Run noiseless symbolic simulation on a `TickCircuit` to identify non-deterministic measurements.
///
/// Returns a Vec<bool> where true = non-deterministic (needs coin flip).
/// Uses `SymbolicSparseStab` which tracks measurement determinism symbolically.
/// Run noiseless symbolic simulation to identify non-deterministic measurements
/// and their dependency structure.
///
/// Returns:
/// - `Vec<bool>`: non-det mask (true = needs coin flip)
/// - `Vec<Option<(Vec<usize>, bool)>>`: per-measurement dependencies
///   (Some((deps, flip)) for deterministic measurements, None for non-det)
///
/// Only supports the Clifford gate subset. Returns error for unsupported gates.
fn parse_records_json(json: &str) -> Vec<Vec<i32>> {
    let json = json.trim();
    if json.is_empty() || json == "[]" {
        return Vec::new();
    }

    let mut results = Vec::new();
    let mut depth = 0;
    let mut start = None;

    for (i, c) in json.char_indices() {
        match c {
            '{' => {
                if depth == 1 {
                    start = Some(i);
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 1 {
                    if let Some(s) = start {
                        let obj_str = &json[s..i + c.len_utf8()];
                        results.push(extract_records_array(obj_str));
                    }
                    start = None;
                }
            }
            '[' if depth == 0 => depth = 1,
            ']' if depth == 1 => depth = 0,
            _ => {}
        }
    }

    results
}

/// Extract measurement record indices from a JSON object string.
///
/// Prefers `"meas_ids"` (absolute `MeasId` IDs) when available.
/// Also accepts `"records"` for DEM-style negative offsets.
fn extract_records_array(json: &str) -> Vec<i32> {
    // Prefer meas_ids (absolute, stable IDs from MeasId)
    if let Some(pos) = json.find("\"meas_ids\"") {
        let rest = &json[pos..];
        if let (Some(arr_start), Some(arr_end)) = (rest.find('['), rest.find(']'))
            && arr_start < arr_end
        {
            let arr_str = &rest[arr_start + 1..arr_end];
            let ids: Vec<i32> = arr_str
                .split(',')
                .filter_map(|s| s.trim().parse::<i32>().ok())
                .collect();
            if !ids.is_empty() {
                // Convert absolute MeasId IDs to negative offsets:
                // not needed — the DemBuilder resolves negative offsets against
                // num_measurements. With absolute IDs, we store them as positive
                // values and handle them in the DemBuilder's build_measurement_mappings.
                //
                // For now, keep the negative-offset convention internally but
                // convert: absolute ID i becomes offset -(num_measurements - i).
                // We don't know num_measurements here, so return the absolute IDs
                // as positive i32. The DemBuilder recognizes positive values as
                // absolute MeasId indices.
                return ids;
            }
        }
    }

    // Fallback: "records" with negative offsets
    if let Some(pos) = json.find("\"records\"") {
        let rest = &json[pos..];
        if let (Some(arr_start), Some(arr_end)) = (rest.find('['), rest.find(']'))
            && arr_start < arr_end
        {
            let arr_str = &rest[arr_start + 1..arr_end];
            return arr_str
                .split(',')
                .filter_map(|s| s.trim().parse::<i32>().ok())
                .collect();
        }
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fault_tolerance::InfluenceBuilder;
    use pecos_quantum::DagCircuit;
    use pecos_random::PecosRng;

    fn repetition_code(rounds: usize) -> DagCircuit {
        let mut dag = DagCircuit::new();
        for _ in 0..rounds {
            dag.pz(&[3]);
            dag.pz(&[4]);
            dag.cx(&[(0, 3)]);
            dag.cx(&[(1, 3)]);
            dag.cx(&[(1, 4)]);
            dag.cx(&[(2, 4)]);
            dag.mz(&[3]);
            dag.mz(&[4]);
        }
        dag
    }

    #[test]
    fn raw_mode_output_length_matches_measurements() {
        let circuit = repetition_code(2);
        let im = InfluenceBuilder::new(&circuit).with_z(&[0, 1, 2]).build();

        let sampler = DemSamplerBuilder::new(&im)
            .with_uniform_noise(0.01)
            .raw_measurements()
            .build()
            .unwrap();

        let mut rng = PecosRng::seed_from_u64(42);
        let (outputs, _obs) = sampler.sample(&mut rng);

        assert_eq!(outputs.len(), im.measurements.len());
        assert_eq!(sampler.mode(), OutputMode::RawMeasurements);
    }

    #[test]
    fn zero_noise_raw_mode_deterministic_measurements_are_zero() {
        let circuit = repetition_code(3);
        let im = InfluenceBuilder::new(&circuit).with_z(&[0, 1, 2]).build();

        let sampler = DemSamplerBuilder::new(&im)
            .with_uniform_noise(0.0)
            .raw_measurements()
            .build()
            .unwrap();

        // With zero noise, deterministic measurement flips should all be false.
        // Non-deterministic ones get coin flips so we can't assert on those.
        // But the mechanism-driven part should be all-zero.
        let stats = sampler.sample_statistics(1000, 42);
        assert_eq!(stats.syndrome_count, 0);
        assert_eq!(stats.logical_error_count, 0);
    }

    #[test]
    fn raw_mode_matches_dem_sampler_from_influence_map() {
        let circuit = repetition_code(3);
        let im = InfluenceBuilder::new(&circuit).with_z(&[0, 1, 2]).build();

        let p = 0.01;
        let num_shots = 20_000;

        // DemSampler raw mode
        let sampler = DemSamplerBuilder::new(&im)
            .with_uniform_noise(p)
            .raw_measurements()
            .build()
            .unwrap();

        let unified_stats = sampler.sample_statistics(num_shots, 42);

        // DemSampler::from_influence_map (same mechanism construction)
        let probs = vec![p; im.locations.len()];
        let dem = DemSampler::from_influence_map(&im, &probs);
        let dem_stats = dem.sample_statistics(num_shots, 42);

        // Same seed, same mechanism construction → identical results
        assert_eq!(unified_stats.syndrome_count, dem_stats.syndrome_count);
        assert_eq!(
            unified_stats.logical_error_count,
            dem_stats.logical_error_count
        );
    }

    #[test]
    fn detector_mode_output_length_matches_definitions() {
        let circuit = repetition_code(3);
        let im = InfluenceBuilder::new(&circuit).build();

        // Define 2 simple detectors (last two measurements)
        let detector_records = vec![vec![-1i32], vec![-2]];
        let observable_records = vec![vec![-1i32]]; // 1 observable

        let sampler = DemSamplerBuilder::new(&im)
            .with_noise(0.001, 0.01, 0.005, 0.001)
            .with_detectors(detector_records, observable_records)
            .build()
            .unwrap();

        let mut rng = PecosRng::seed_from_u64(42);
        let (det_events, obs_flips) = sampler.sample(&mut rng);

        assert_eq!(det_events.len(), 2);
        assert_eq!(obs_flips.len(), 1);
        assert_eq!(sampler.mode(), OutputMode::DetectorEvents);
    }

    #[test]
    fn detector_mode_accepts_observable_aliases() {
        let circuit = repetition_code(3);
        let im = InfluenceBuilder::new(&circuit).build();

        let records_sampler = DemSamplerBuilder::new(&im)
            .with_detector_records(vec![vec![-1]])
            .with_observable_records(vec![vec![-1]])
            .build()
            .unwrap();

        assert_eq!(records_sampler.num_detectors(), 1);
        assert_eq!(records_sampler.num_dem_outputs(), 1);
        assert_eq!(records_sampler.num_observables(), 1);
        assert_eq!(records_sampler.num_tracked_paulis(), 0);
        assert_eq!(records_sampler.mode(), OutputMode::DetectorEvents);

        let json_sampler = DemSamplerBuilder::new(&im)
            .with_detectors_json(r#"[{"id":0,"records":[-1]}]"#)
            .unwrap()
            .with_observables_json(r#"[{"id":0,"records":[-1]}]"#)
            .unwrap()
            .build()
            .unwrap();

        assert_eq!(json_sampler.num_detectors(), 1);
        assert_eq!(json_sampler.num_dem_outputs(), 1);
        assert_eq!(json_sampler.num_observables(), 1);
        assert_eq!(json_sampler.num_tracked_paulis(), 0);
        assert_eq!(json_sampler.mode(), OutputMode::DetectorEvents);
    }

    #[test]
    fn from_circuit_preserves_tracked_paulis() {
        use crate::fault_tolerance::dem_builder::NoiseConfig;
        use pecos_core::pauli::X;

        let mut circuit = DagCircuit::new();
        circuit.pz(&[0]);
        circuit.h(&[0]);
        circuit.tracked_pauli_labeled("x_check", X(0));

        let noise = NoiseConfig::new(0.03, 0.0, 0.0, 0.0);
        let sampler = DemSampler::from_circuit(&circuit, &noise).unwrap();

        assert_eq!(sampler.num_tracked_paulis(), 1);
        assert_eq!(sampler.num_observables(), 0);
        assert_eq!(
            sampler.labels().tracked_pauli_labels[0].as_deref(),
            Some("x_check")
        );
        let op = sampler.labels().tracked_paulis[0].as_ref().unwrap();
        assert_eq!(op.label.as_deref(), Some("x_check"));
        assert_eq!(
            op.kind,
            Some(crate::fault_tolerance::DemOutputKind::TrackedPauli)
        );
        assert_eq!(op.pauli.as_ref().unwrap().to_sparse_str(), "+X0");
    }

    #[test]
    fn detector_mode_keeps_observables_unshifted_with_tracked_paulis() {
        use pecos_core::pauli::X;

        let mut circuit = DagCircuit::new();
        circuit.pz(&[0]);
        circuit.h(&[0]);
        circuit.tracked_pauli_labeled("x_check", X(0));
        circuit.mz(&[0]);

        let im = InfluenceBuilder::new(&circuit)
            .with_circuit_annotations(&circuit)
            .build();

        let sampler = DemSamplerBuilder::new(&im)
            .with_noise(0.03, 0.0, 0.02, 0.0)
            .with_detectors(Vec::new(), vec![vec![-1]])
            .build()
            .unwrap();

        assert_eq!(sampler.num_dem_outputs(), 1);
        assert_eq!(sampler.num_observables(), 1);
        assert_eq!(sampler.num_tracked_paulis(), 1);
        assert_eq!(sampler.labels().dem_outputs.len(), 1);
        assert_eq!(
            sampler.labels().dem_outputs[0].as_ref().unwrap().kind,
            Some(crate::fault_tolerance::DemOutputKind::Observable)
        );
        assert_eq!(
            sampler.labels().tracked_paulis[0].as_ref().unwrap().kind,
            Some(crate::fault_tolerance::DemOutputKind::TrackedPauli)
        );
    }

    #[test]
    fn detector_mode_does_not_double_apply_annotation_observable_records() {
        let mut circuit = DagCircuit::new();
        circuit.pz(&[0]);
        let meas = circuit.mz(&[0]);
        circuit.observable_labeled("obs0", &[meas[0]]);

        let im = InfluenceBuilder::new(&circuit)
            .with_circuit_annotations(&circuit)
            .build();

        let sampler = DemSamplerBuilder::new(&im)
            .with_noise(0.0, 0.0, 1.0, 0.0)
            .with_detectors(Vec::new(), vec![vec![-1]])
            .build()
            .unwrap();

        assert_eq!(sampler.num_dem_outputs(), 1);
        assert_eq!(sampler.num_observables(), 1);
        assert_eq!(sampler.num_tracked_paulis(), 0);
        assert_eq!(
            sampler.labels().dem_outputs[0]
                .as_ref()
                .unwrap()
                .label
                .as_deref(),
            Some("obs0")
        );
        assert_eq!(
            sampler.labels().dem_outputs[0]
                .as_ref()
                .unwrap()
                .records
                .as_slice(),
            &[-1]
        );

        let mut rng = PecosRng::seed_from_u64(42);
        let (_detectors, observables) = sampler.sample(&mut rng);
        assert_eq!(observables, vec![true]);
    }

    #[test]
    fn from_detector_error_model_preserves_observable_and_tracked_pauli_split() {
        use super::super::builder::DemBuilder;
        use pecos_core::pauli::X;
        use pecos_quantum::Attribute;

        let mut circuit = DagCircuit::new();
        circuit.pz(&[0]);
        circuit.h(&[0]);
        circuit.tracked_pauli_labeled("x_check", X(0));
        circuit.mz(&[0]);
        circuit.set_attr("num_measurements", Attribute::String("1".to_string()));
        circuit.set_attr(
            "observables",
            Attribute::String(r#"[{"id":0,"records":[-1]}]"#.to_string()),
        );

        let dem = DemBuilder::from_circuit(&circuit, 0.03, 0.0, 0.02, 0.0);

        let sampler = DemSampler::from_detector_error_model(&dem);

        assert_eq!(sampler.num_dem_outputs(), 1);
        assert_eq!(sampler.num_observables(), 1);
        assert_eq!(sampler.num_tracked_paulis(), 1);
        assert_eq!(
            sampler.labels().dem_outputs[0].as_ref().unwrap().kind,
            Some(crate::fault_tolerance::DemOutputKind::Observable)
        );
        assert_eq!(
            sampler.labels().tracked_paulis[0].as_ref().unwrap().kind,
            Some(crate::fault_tolerance::DemOutputKind::TrackedPauli)
        );
    }

    #[test]
    fn sampler_paths_preserve_output_split_for_noiseless_and_forced_faults() {
        use super::super::builder::DemBuilder;
        use super::super::types::NoiseConfig;
        use pecos_core::pauli::X;
        use pecos_quantum::Attribute;

        fn assert_metadata(sampler: &DemSampler) {
            assert_eq!(sampler.num_detectors(), 1);
            assert_eq!(sampler.num_dem_outputs(), 1);
            assert_eq!(sampler.num_observables(), 1);
            assert_eq!(sampler.num_tracked_paulis(), 1);
            assert_eq!(sampler.observable_ids(), vec![0]);
            let err = sampler.tracked_pauli_ids().unwrap_err();
            assert_eq!(err.backend(), "DemSampler");
            assert_eq!(err.num_tracked_paulis(), 1);
            assert!(
                err.to_string()
                    .contains("cannot directly sample tracked Pauli flips")
            );
            assert_eq!(
                sampler.labels().dem_outputs[0]
                    .as_ref()
                    .unwrap()
                    .label
                    .as_deref(),
                Some("obs0")
            );
            assert_eq!(
                sampler.labels().tracked_paulis[0]
                    .as_ref()
                    .unwrap()
                    .label
                    .as_deref(),
                Some("tracked_x0")
            );
        }

        fn sample_once(sampler: &DemSampler) -> (Vec<bool>, Vec<bool>) {
            let mut rng = PecosRng::seed_from_u64(123);
            sampler.sample(&mut rng)
        }

        let mut circuit = DagCircuit::new();
        circuit.pz(&[0]);
        let meas = circuit.mz(&[0]);
        circuit.detector_labeled("det0", &[meas[0]]);
        circuit.observable_labeled("obs0", &[meas[0]]);
        circuit.tracked_pauli_labeled("tracked_x0", X(0));
        circuit.set_attr("num_measurements", Attribute::String("1".to_string()));
        circuit.set_attr(
            "detectors",
            Attribute::String(r#"[{"id":0,"records":[-1],"label":"det0"}]"#.to_string()),
        );
        circuit.set_attr(
            "observables",
            Attribute::String(r#"[{"id":0,"records":[-1],"label":"obs0"}]"#.to_string()),
        );

        let noiseless = DemSampler::from_circuit(&circuit, &NoiseConfig::default()).unwrap();
        assert_metadata(&noiseless);
        assert_eq!(sample_once(&noiseless), (vec![false], vec![false]));

        let forced_noise = NoiseConfig::new(0.0, 0.0, 1.0, 0.0);
        let from_circuit = DemSampler::from_circuit(&circuit, &forced_noise).unwrap();
        assert_metadata(&from_circuit);
        assert_eq!(sample_once(&from_circuit), (vec![true], vec![true]));

        let dem = DemBuilder::from_circuit(&circuit, 0.0, 0.0, 1.0, 0.0);
        let from_dem = DemSampler::from_detector_error_model(&dem);
        assert_metadata(&from_dem);
        assert_eq!(sample_once(&from_dem), (vec![true], vec![true]));

        let influence_map = InfluenceBuilder::new(&circuit)
            .with_circuit_annotations(&circuit)
            .build();
        let from_builder = DemSamplerBuilder::new(&influence_map)
            .with_noise(0.0, 0.0, 1.0, 0.0)
            .with_detector_records(vec![vec![-1]])
            .with_observable_records(vec![vec![-1]])
            .build()
            .unwrap();
        assert_metadata(&from_builder);
        assert_eq!(sample_once(&from_builder), (vec![true], vec![true]));
    }

    #[test]
    fn sampler_xors_detectors_and_observables_while_tracked_paulis_stay_metadata() {
        use super::super::types::{DetectorDef, DetectorErrorModel, FaultMechanism};
        use pecos_core::pauli::Z;

        let mut dem = DetectorErrorModel::new();
        dem.add_detector(DetectorDef::new(0));
        dem.add_observable(DemOutput::new(0).with_records([-1]).with_label("L0"));
        dem.add_tracked_pauli(DemOutput::new(0).with_pauli(Z(3)).with_label("tracked_z3"));
        dem.add_direct_contribution(FaultMechanism::from_unsorted([0], [0]), 1.0);
        dem.add_direct_contribution(FaultMechanism::from_unsorted([0], []), 1.0);

        let sampler = DemSampler::from_detector_error_model(&dem);
        let mut rng = PecosRng::seed_from_u64(99);

        assert_eq!(sampler.num_detectors(), 1);
        assert_eq!(sampler.num_observables(), 1);
        assert_eq!(sampler.num_tracked_paulis(), 1);
        assert_eq!(
            sampler.labels().tracked_paulis[0]
                .as_ref()
                .unwrap()
                .label
                .as_deref(),
            Some("tracked_z3")
        );
        assert_eq!(sampler.sample(&mut rng), (vec![false], vec![true]));
    }

    #[test]
    fn raw_mode_without_dem_outputs_reports_zero_dem_outputs() {
        let mut circuit = DagCircuit::new();
        circuit.pz(&[0]);
        circuit.h(&[0]);
        circuit.mz(&[0]);
        let im = InfluenceBuilder::new(&circuit).build();

        let sampler = DemSamplerBuilder::new(&im)
            .with_uniform_noise(0.01)
            .raw_measurements()
            .build()
            .unwrap();

        assert_eq!(sampler.num_dem_outputs(), 0);
        assert_eq!(sampler.num_observables(), 0);
        assert_eq!(sampler.num_tracked_paulis(), 0);
    }

    #[test]
    fn observable_mask_ignores_tracked_pauli_outputs() {
        use super::super::builder::DemBuilder;
        use pecos_core::pauli::X;
        use pecos_quantum::Attribute;

        let mut circuit = DagCircuit::new();
        circuit.pz(&[0]);
        circuit.h(&[0]);
        circuit.tracked_pauli_labeled("x_check", X(0));
        circuit.mz(&[0]);
        circuit.set_attr("num_measurements", Attribute::String("1".to_string()));
        circuit.set_attr(
            "observables",
            Attribute::String(r#"[{"id":0,"records":[-1]}]"#.to_string()),
        );

        let dem = DemBuilder::from_circuit(&circuit, 0.03, 0.0, 0.02, 0.0);
        let sampler = DemSampler::from_detector_error_model(&dem);

        assert_eq!(sampler.observable_ids(), vec![0]);
        assert_eq!(
            sampler
                .tracked_pauli_ids()
                .unwrap_err()
                .num_tracked_paulis(),
            1
        );
        assert_eq!(sampler.observable_dem_output_mask(), 1);
        assert_eq!(sampler.observable_mask_from_dem_output_flips(&[false]), 0);
        assert_eq!(sampler.observable_mask_from_dem_output_flips(&[true]), 1);
    }

    #[test]
    fn tracked_pauli_direct_sampling_fails_explicitly_when_unsupported() {
        use super::super::types::{DetectorErrorModel, FaultMechanism};
        use pecos_core::pauli::X;

        let mut dem = DetectorErrorModel::new();
        dem.add_tracked_pauli(DemOutput::new(0).with_pauli(X(0)).with_label("tracked_x0"));
        dem.add_direct_contribution(
            FaultMechanism::from_unsorted_with_tracked_paulis([], [], [0]),
            0.25,
        );

        let sampler = DemSampler::from_detector_error_model(&dem);
        let mut rng = PecosRng::seed_from_u64(17);

        let err = sampler
            .sample_tracked_pauli_flips(&mut rng)
            .expect_err("DemSampler should reject direct tracked-Pauli sampling");
        assert_eq!(err.backend(), "DemSampler");
        assert_eq!(err.num_tracked_paulis(), 1);
        assert!(
            err.to_string()
                .contains("samples decoder-facing detectors and observables only")
        );

        let err = sampler
            .sample_tracked_pauli_batch(4, &mut rng)
            .expect_err("DemSampler should reject direct tracked-Pauli batch sampling");
        assert_eq!(err.num_tracked_paulis(), 1);

        let empty = DemSampler::from_detector_error_model(&DetectorErrorModel::new());
        assert_eq!(
            empty.sample_tracked_pauli_flips(&mut rng).unwrap(),
            Vec::<bool>::new()
        );
        assert_eq!(
            empty.sample_tracked_pauli_batch(3, &mut rng).unwrap(),
            vec![Vec::<bool>::new(), Vec::new(), Vec::new()]
        );
    }

    #[test]
    fn high_noise_produces_nonzero_rates_both_modes() {
        let circuit = repetition_code(2);
        let im = InfluenceBuilder::new(&circuit).with_z(&[0, 1, 2]).build();

        let p = 0.1;
        let num_shots = 5_000;

        // Raw mode
        let raw_sampler = DemSamplerBuilder::new(&im)
            .with_uniform_noise(p)
            .raw_measurements()
            .build()
            .unwrap();
        let raw_stats = raw_sampler.sample_statistics(num_shots, 42);
        assert!(
            raw_stats.syndrome_rate() > 0.05,
            "Raw mode should detect syndromes at p=0.1"
        );

        // Detector mode with simple detectors
        let detector_records = vec![vec![-1i32], vec![-2]];
        let observable_records: Vec<Vec<i32>> = vec![];
        let det_sampler = DemSamplerBuilder::new(&im)
            .with_uniform_noise(p)
            .with_detectors(detector_records, observable_records)
            .build()
            .unwrap();
        let det_stats = det_sampler.sample_statistics(num_shots, 42);
        assert!(
            det_stats.syndrome_rate() > 0.05,
            "Detector mode should detect syndromes at p=0.1"
        );
    }

    #[test]
    fn dual_output_returns_none_without_definitions() {
        let circuit = repetition_code(2);
        let im = InfluenceBuilder::new(&circuit).with_z(&[0, 1, 2]).build();

        let sampler = DemSamplerBuilder::new(&im)
            .with_uniform_noise(0.01)
            .raw_measurements()
            .build()
            .unwrap();

        let mut rng = PecosRng::seed_from_u64(42);
        assert!(sampler.sample_dual(&mut rng).is_none());
    }

    #[test]
    fn dual_output_produces_both_views() {
        let circuit = repetition_code(3);
        let im = InfluenceBuilder::new(&circuit).with_z(&[0, 1, 2]).build();

        // Define detectors: first and second measurements
        let det_defs = vec![vec![0usize], vec![1]];

        let sampler = DemSamplerBuilder::new(&im)
            .with_uniform_noise(0.05)
            .raw_measurements()
            .with_dual_output(det_defs)
            .build()
            .unwrap();

        let mut rng = PecosRng::seed_from_u64(42);
        let result = sampler.sample_dual(&mut rng).unwrap();

        // Raw measurements should have length = num measurements
        assert_eq!(result.raw_measurements.len(), im.measurements.len());
        // Detector events should have length = 2 (our 2 detector defs)
        assert_eq!(result.detector_events.len(), 2);
    }

    #[test]
    fn dual_output_detector_events_consistent_with_raw() {
        let circuit = repetition_code(3);
        let im = InfluenceBuilder::new(&circuit).with_z(&[0, 1, 2]).build();

        // Detector = XOR of measurements 0 and 1
        let det_defs = vec![vec![0usize, 1]];

        let sampler = DemSamplerBuilder::new(&im)
            .with_uniform_noise(0.1)
            .raw_measurements()
            .with_dual_output(det_defs)
            .build()
            .unwrap();

        // Run many shots and verify detector = raw[0] XOR raw[1]
        let mut rng = PecosRng::seed_from_u64(42);
        for _ in 0..100 {
            let result = sampler.sample_dual(&mut rng).unwrap();
            let expected_det = result.raw_measurements[0] ^ result.raw_measurements[1];
            assert_eq!(
                result.detector_events[0], expected_det,
                "Detector event should equal XOR of raw measurements 0 and 1"
            );
        }
    }
}
