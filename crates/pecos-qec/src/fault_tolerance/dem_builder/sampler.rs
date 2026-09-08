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
use super::types::{DemOutput, NoiseConfig, PerGateTypeNoise, is_two_qubit_noise_gate};
use crate::fault_tolerance::propagator::{
    DagFaultInfluenceMap, DemOutputKind, is_supported_prep_gate,
};
use pecos_core::prelude::GateType;
use pecos_decoder_core::obs_mask::ObsMask;
use pecos_num::z2_linalg::z2_rank_from_records;
use pecos_random::RngProbabilityExt;
use rand_core::Rng;

/// Errors from detector definition validation.
#[derive(Debug, Clone)]
pub enum DetectorValidationError {
    /// Circuit gate whose action Pauli propagation cannot represent.
    UnsupportedGate(crate::fault_tolerance::propagator::UnsupportedGateError),
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
    /// Circuit detector/observable metadata is malformed.
    InvalidMetadata { message: String },
    /// A detector or observable annotation references a node that cannot be
    /// resolved to a measurement in the influence map.
    UnresolvableAnnotationRef {
        /// "detector" or "observable".
        output_kind: &'static str,
        /// Index among that kind's annotations.
        annotation_index: usize,
        /// The unresolvable measurement id.
        meas_id: pecos_core::MeasId,
    },
}

impl std::fmt::Display for DetectorValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedGate(error) => write!(f, "DEM sampler {error}"),
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
            Self::InvalidMetadata { message } => {
                write!(f, "Invalid detector/observable metadata: {message}")
            }
            Self::UnresolvableAnnotationRef {
                output_kind,
                annotation_index,
                meas_id,
            } => write!(
                f,
                "{output_kind} annotation {annotation_index} references MeasId({}), which \
                 does not resolve to a measurement in the influence map",
                meas_id.index()
            ),
        }
    }
}

impl std::error::Error for DetectorValidationError {}

impl From<crate::fault_tolerance::propagator::UnsupportedGateError> for DetectorValidationError {
    fn from(error: crate::fault_tolerance::propagator::UnsupportedGateError) -> Self {
        Self::UnsupportedGate(error)
    }
}

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
    /// Also returned when the `TickCircuit` cannot be converted to a
    /// `DagCircuit` (two measurements sharing a `MeasId`).
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
        if let Some(error) =
            crate::fault_tolerance::propagator::first_unsupported_tick_gate(circuit)
        {
            return Err(DetectorValidationError::UnsupportedGate(error));
        }
        let dag = pecos_quantum::DagCircuit::try_from(circuit).map_err(|err| {
            DetectorValidationError::InvalidMetadata {
                message: err.to_string(),
            }
        })?;
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
        if let Some(error) = influence_map.unsupported_gate() {
            return Err(DetectorValidationError::UnsupportedGate(error.clone()));
        }
        let annotation_map = InfluenceBuilder::new(circuit)
            .with_circuit_annotations()
            .map_err(|err| DetectorValidationError::InvalidMetadata {
                message: err.to_string(),
            })?
            .build()
            .map_err(|err| DetectorValidationError::InvalidMetadata {
                message: err.to_string(),
            })?;
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
            builder.with_detectors_json(dj).map_err(|err| {
                DetectorValidationError::InvalidMetadata {
                    message: err.to_string(),
                }
            })?
        } else {
            builder
        };

        let builder = if let Some(ref oj) = observables_json {
            builder.with_observables_json(oj).map_err(|err| {
                DetectorValidationError::InvalidMetadata {
                    message: err.to_string(),
                }
            })?
        } else {
            builder
        };

        // `try_build` enforces num_measurements == influence-map count, so a
        // metadata override that disagrees with the circuit is rejected there.
        let builder = if let Some(n) = num_meas {
            builder.with_num_measurements(n)
        } else {
            builder
        };

        let dem = builder
            .try_build()
            .map_err(|err| DetectorValidationError::InvalidMetadata {
                message: err.to_string(),
            })?;
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
    ///
    /// # Errors
    ///
    /// Returns [`DetectorValidationError::UnsupportedGate`] if the influence
    /// map came from a circuit Pauli propagation cannot faithfully represent.
    pub fn from_influence_map(
        influence_map: &DagFaultInfluenceMap,
        per_location_probs: &[f64],
    ) -> Result<Self, DetectorValidationError> {
        let default_noise = super::NoiseConfig::default();
        let inner =
            SamplingEngine::from_influence_map(influence_map, per_location_probs, &default_noise)?;
        let num_outputs = inner.num_detectors();
        let num_dem_outputs = inner.num_dem_outputs();
        let mut labels = SamplerLabels::default();
        labels.dem_outputs = dem_outputs_from_influence_map(influence_map, num_dem_outputs);
        labels.dem_output_labels = labels_from_dem_outputs(&labels.dem_outputs);
        labels.tracked_paulis = tracked_paulis_from_influence_map(influence_map);
        labels.tracked_pauli_labels = labels_from_dem_outputs(&labels.tracked_paulis);
        Ok(Self {
            inner,
            non_det_mask: Vec::new(),
            detector_records_abs: Vec::new(),
            mode: OutputMode::RawMeasurements,
            num_outputs,
            num_dem_outputs,
            labels,
            raw_remap: None,
            measurement_deps: Vec::new(),
        })
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
    /// Wide bitmask of which DEM outputs are decoder-facing observables.
    ///
    /// Returns an [`ObsMask`], so more than 64 observables are represented with
    /// no truncation. Compute this once, up front, and pass it to
    /// [`Self::observable_mask_from_dem_output_flips`] per shot.
    #[must_use]
    pub fn observable_dem_output_mask(&self) -> ObsMask {
        let mut mask = ObsMask::new();
        for idx in self.observable_ids() {
            mask.set(idx);
        }
        mask
    }

    /// Converts a sampled DEM-output flip vector into an observable-only wide mask.
    ///
    /// `observable_mask` is the value returned by
    /// [`Self::observable_dem_output_mask`]; pass it in so the per-shot path does
    /// not recompute it.
    #[must_use]
    pub fn observable_mask_from_dem_output_flips(
        &self,
        flips: &[bool],
        observable_mask: &ObsMask,
    ) -> ObsMask {
        let mut mask = ObsMask::new();
        for (idx, flipped) in flips.iter().enumerate() {
            if *flipped && observable_mask.get(idx) {
                mask.set(idx);
            }
        }
        mask
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
        let records = super::builder::parse_detector_record_vectors(json, self.influence_map)
            .map_err(|err| err.to_string())?;
        Ok(self.with_detector_records(records))
    }

    /// Set observable definitions from JSON.
    ///
    /// Format: `[{"id": 0, "records": [-1, -3, -5]}, ...]`
    ///
    /// # Errors
    /// Returns an error if the JSON is malformed, fails schema validation, or
    /// references measurements out of range for the circuit.
    pub fn with_observables_json(self, json: &str) -> Result<Self, String> {
        let records = super::builder::parse_observable_record_vectors(json, self.influence_map)
            .map_err(|err| err.to_string())?;
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
    /// # Errors
    ///
    /// Returns [`DetectorValidationError::UnresolvableAnnotationRef`] when a
    /// detector or observable annotation references a node that does not
    /// resolve to a measurement in the influence map. Such references used to
    /// be dropped silently, producing outputs with missing terms.
    ///
    /// # Panics
    ///
    /// Panics if a resolved measurement index exceeds `i32` while the
    /// measurement count fits -- impossible, since every index is below the
    /// count, which is checked first.
    pub fn with_circuit_annotations(
        mut self,
        circuit: &pecos_quantum::DagCircuit,
    ) -> Result<Self, DetectorValidationError> {
        use pecos_quantum::AnnotationKind;

        // A duplicate stamped id would make `meas_index_of` bind to the first
        // holder -- an ambiguous, silently-wrong bind. Mirrors the guard on
        // the JSON path (`reject_duplicate_stamped_meas_ids`).
        let mut seen = std::collections::BTreeSet::new();
        for mid in &self.influence_map.meas_ids {
            if !seen.insert(mid.index()) {
                return Err(DetectorValidationError::InvalidMetadata {
                    message: format!(
                        "duplicate stable MeasId {} in the circuit; each \
                         measurement must have a unique stamped id",
                        mid.index()
                    ),
                });
            }
        }

        let detectors: Vec<&pecos_quantum::PauliAnnotation> = circuit.detectors().collect();
        let observables: Vec<&pecos_quantum::PauliAnnotation> = circuit.observables().collect();

        // Map each user detector directly to absolute raw-measurement indices --
        // the coordinate space `detector_records_abs` documents and `sample_dual`
        // XORs. An earlier version routed through a first-match-by-qubit/basis
        // auto-detector lookup, which both stored the wrong coordinate space and
        // rejected legitimate detectors on circuits that measure one qubit more
        // than once.
        if !detectors.is_empty() {
            let mut det_records_abs: Vec<Vec<usize>> = Vec::with_capacity(detectors.len());
            for (annotation_index, ann) in detectors.iter().enumerate() {
                let AnnotationKind::Detector {
                    measurement_ids, ..
                } = &ann.kind
                else {
                    det_records_abs.push(Vec::new());
                    continue;
                };
                let mut resolved = Vec::with_capacity(measurement_ids.len());
                for &meas_id in measurement_ids {
                    let Some(im_idx) = self.influence_map.meas_index_of(meas_id) else {
                        return Err(DetectorValidationError::UnresolvableAnnotationRef {
                            output_kind: "detector",
                            annotation_index,
                            meas_id,
                        });
                    };
                    resolved.push(im_idx);
                }
                det_records_abs.push(resolved);
            }

            self.labels.dual_detectors = detectors.iter().map(|a| a.label.clone()).collect();
            self.detector_records_abs = Some(det_records_abs);
        }

        if !observables.is_empty() && self.observable_records.is_none() {
            let num_measurements =
                i32::try_from(self.influence_map.measurements.len()).map_err(|_| {
                    DetectorValidationError::InvalidMetadata {
                        message: format!(
                            "{} measurements exceed the i32 record-offset range",
                            self.influence_map.measurements.len()
                        ),
                    }
                })?;
            let mut records: Vec<Vec<i32>> = Vec::with_capacity(observables.len());
            for (annotation_index, ann) in observables.iter().enumerate() {
                let AnnotationKind::Observable { measurement_ids } = &ann.kind else {
                    records.push(Vec::new());
                    continue;
                };
                let mut resolved = Vec::with_capacity(measurement_ids.len());
                for &meas_id in measurement_ids {
                    let Some(meas_idx) = self.influence_map.meas_index_of(meas_id) else {
                        return Err(DetectorValidationError::UnresolvableAnnotationRef {
                            output_kind: "observable",
                            annotation_index,
                            meas_id,
                        });
                    };
                    let meas_idx = i32::try_from(meas_idx)
                        .expect("meas_idx < measurements.len(), which fits in i32");
                    resolved.push(meas_idx - num_measurements);
                }
                records.push(resolved);
            }
            let records = records;
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

        Ok(self)
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
        // A supplied measurement order must cover every measurement, otherwise
        // detector/observable record offsets validated against the circuit's
        // measurement count would resolve in a different (shorter/longer) frame
        // at sample time and silently misbind. (See sampler-JSON validation.)
        if let Some(ref order) = self.measurement_order {
            // A supplied order feeds the qubit-occurrence heuristic, which
            // needs per-qubit chronology on both sides. Minted (positional)
            // ids keep it; external non-positional ids can reorder a qubit's
            // measurements in the map, and the caller's record order is then
            // not recoverable.
            let n = self.influence_map.meas_ids.len();
            let mut seen = vec![false; n];
            let positional = self
                .influence_map
                .meas_ids
                .iter()
                .all(|mid| seen.get_mut(mid.index()).map(|slot| *slot = true).is_some())
                && seen.into_iter().all(|s| s);
            if !positional {
                return Err(DetectorValidationError::InvalidMetadata {
                    message: "measurement_order cannot be combined with a circuit \
                              whose stable MeasIds are non-positional; the \
                              caller's record order is not recoverable"
                        .to_string(),
                });
            }
            let expected = self.influence_map.measurements.len();
            if order.len() != expected {
                return Err(DetectorValidationError::InvalidMetadata {
                    message: format!(
                        "measurement_order has {} entries but the circuit performs \
                         {expected} measurement(s); a measurement order must cover \
                         every measurement so record offsets resolve in the same frame",
                        order.len()
                    ),
                });
            }
        }
        match self.output_mode {
            OutputMode::RawMeasurements => self.build_raw(),
            OutputMode::DetectorEvents => self.build_detector(),
        }
    }

    /// Build in raw measurement mode.
    ///
    /// Mechanism table is in measurement coordinates. Non-deterministic
    /// measurements are identified and marked for coin-flip output.
    fn build_raw(self) -> Result<DemSampler, DetectorValidationError> {
        let num_measurements = self.influence_map.measurements.len();

        // Build per-location probabilities from gate-type noise
        let per_location_probs = self.compute_per_location_probs();

        // Build mechanism table in raw measurement coordinates
        let inner = SamplingEngine::from_influence_map(
            self.influence_map,
            &per_location_probs,
            &self.noise,
        )?;

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

        Ok(DemSampler {
            inner,
            non_det_mask,
            detector_records_abs: self.detector_records_abs.unwrap_or_default(),
            mode: OutputMode::RawMeasurements,
            num_outputs: num_measurements,
            num_dem_outputs,
            labels: merge_dem_output_metadata(self.labels, dem_outputs, tracked_paulis),
            raw_remap: None,
            measurement_deps: Vec::new(), // No expansion needed (engine covers all measurements)
        })
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
            .with_noise_config(self.noise.clone())
            .with_detector_records(detector_records)
            .with_observable_records(observable_records.clone());

        if let Some(per_gate) = self.per_gate {
            builder = builder.with_per_gate_noise(per_gate);
        }

        if let Some(order) = self.measurement_order {
            builder = builder.with_measurement_order(order);
        }

        let inner = builder.build()?;
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
                gate_type if is_supported_prep_gate(gate_type) => noise.p_prep,
                GateType::MX | GateType::MZ | GateType::MeasureFree | GateType::MPZ => noise.p_meas,
                gate_type if is_two_qubit_noise_gate(gate_type) => {
                    noise.p2_rate_for_gate(loc.gate_type)
                }
                GateType::Idle => {
                    if noise.uses_dedicated_idle_noise() {
                        let duration = loc.idle_duration;
                        noise.idle_pauli_probs(duration).total()
                    } else {
                        0.0
                    }
                }
                _ => noise.p1_rate_for_gate(loc.gate_type),
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

#[cfg(test)]
mod tests {
    /// The reviewer's happy-path witness for the rewritten detector mapping:
    /// `detector_records_abs` must hold absolute raw-measurement indices, so a
    /// single-measurement detector's event equals that measurement's flip on
    /// every shot. The old auto-detector hop stored detector ids in the field,
    /// which reads the wrong measurement whenever the numberings differ.
    ///
    /// `raw_measurements` is FLIP space, so the discriminator is statistical:
    /// noise makes the two measurements' flips independent, and the test
    /// asserts its own non-vacuity by requiring measurement 1's flip to vary
    /// AND to disagree with measurement 0's on at least one shot. A fixture
    /// where the compared streams never differ proves nothing -- the first
    /// version of this test had exactly that hole, twice.
    #[test]
    fn dual_output_detector_events_xor_the_named_raw_measurements() {
        use pecos_quantum::DagCircuit;
        let mut dag = DagCircuit::new();
        dag.pz(&[0]);
        dag.mz(&[0]); // raw measurement 0
        dag.pz(&[1]);
        let named = dag.mz(&[1]); // raw measurement 1 -- the detector's target
        dag.detector(&named).expect("refs are from this circuit");

        let im =
            crate::fault_tolerance::propagator::DagFaultAnalyzer::new(&dag).build_influence_map();
        let sampler = DemSamplerBuilder::new(&im)
            .with_uniform_noise(0.3)
            .raw_measurements()
            .with_circuit_annotations(&dag)
            .expect("the annotation resolves")
            .build()
            .expect("the detector is valid");

        let mut rng = PecosRng::seed_from_u64(42);
        let (mut saw_one_flip, mut saw_one_clear, mut saw_disagreement) = (false, false, false);
        for _ in 0..64 {
            let shot = sampler
                .sample_dual(&mut rng)
                .expect("annotations enable dual output");
            assert_eq!(
                shot.detector_events[0], shot.raw_measurements[1],
                "the detector names raw measurement 1; reading any other \
                 numbering lands on a different flip stream"
            );
            saw_one_flip |= shot.raw_measurements[1];
            saw_one_clear |= !shot.raw_measurements[1];
            saw_disagreement |= shot.raw_measurements[0] != shot.raw_measurements[1];
        }
        assert!(
            saw_one_flip && saw_one_clear && saw_disagreement,
            "vacuity check: measurement 1's flip must vary and must disagree \
             with measurement 0 at least once, or the equality above proves nothing"
        );
    }

    /// An observable annotation referencing an id absent from the influence
    /// map used to be silently dropped, thinning the observable. It is now an
    /// error naming the annotation and the id.
    #[test]
    fn an_unresolvable_observable_ref_is_an_error() {
        use pecos_quantum::DagCircuit;
        let mut dag = DagCircuit::new();
        dag.pz(&[0]);
        dag.mz(&[0]);
        dag.add_annotation(pecos_quantum::PauliAnnotation {
            pauli: pecos_core::PauliString::zs(&[0usize]),
            kind: pecos_quantum::AnnotationKind::Observable {
                measurement_ids: vec![pecos_core::MeasId::from_raw(99)],
            },
            label: None,
        });
        let im =
            crate::fault_tolerance::propagator::DagFaultAnalyzer::new(&dag).build_influence_map();

        let err = DemSamplerBuilder::new(&im)
            .with_uniform_noise(0.01)
            .with_circuit_annotations(&dag)
            .map(|_| ())
            .expect_err("node 99 resolves to no measurement");
        assert!(matches!(
            err,
            DetectorValidationError::UnresolvableAnnotationRef {
                output_kind: "observable",
                meas_id,
                ..
            } if meas_id == pecos_core::MeasId::from_raw(99)
        ));
    }

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
        let im = InfluenceBuilder::new(&circuit)
            .with_z(&[0, 1, 2])
            .build()
            .expect("circuit is replayable");

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
        let im = InfluenceBuilder::new(&circuit)
            .with_z(&[0, 1, 2])
            .build()
            .expect("circuit is replayable");

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
        let im = InfluenceBuilder::new(&circuit)
            .with_z(&[0, 1, 2])
            .build()
            .expect("circuit is replayable");

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
        let dem = DemSampler::from_influence_map(&im, &probs).unwrap();
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
        let im = InfluenceBuilder::new(&circuit)
            .build()
            .expect("circuit is replayable");

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
        let im = InfluenceBuilder::new(&circuit)
            .build()
            .expect("circuit is replayable");

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
            .with_circuit_annotations()
            .expect("annotations resolve against the circuit")
            .build()
            .expect("circuit is replayable");

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
        circuit
            .observable_labeled("obs0", &[meas[0]])
            .expect("refs are from this circuit");

        let im = InfluenceBuilder::new(&circuit)
            .with_circuit_annotations()
            .expect("annotations resolve against the circuit")
            .build()
            .expect("circuit is replayable");

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

        let dem = DemBuilder::from_circuit(&circuit, 0.03, 0.0, 0.02, 0.0).unwrap();

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
        circuit
            .detector_labeled("det0", &[meas[0]])
            .expect("refs are from this circuit");
        circuit
            .observable_labeled("obs0", &[meas[0]])
            .expect("refs are from this circuit");
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

        let dem = DemBuilder::from_circuit(&circuit, 0.0, 0.0, 1.0, 0.0).unwrap();
        let from_dem = DemSampler::from_detector_error_model(&dem);
        assert_metadata(&from_dem);
        assert_eq!(sample_once(&from_dem), (vec![true], vec![true]));

        let influence_map = InfluenceBuilder::new(&circuit)
            .with_circuit_annotations()
            .expect("annotations resolve against the circuit")
            .build()
            .expect("circuit is replayable");
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
    fn observable_dem_output_mask_supports_above_64_observables() {
        // >64 observables are now represented in a wide ObsMask with no
        // truncation (the old u64 cap is lifted; observable 64 is present).
        use super::super::builder::DemBuilder;
        use pecos_quantum::Attribute;

        let n: usize = 65;
        let mut circuit = DagCircuit::new();
        for i in 0..n {
            circuit.pz(&[i]);
        }
        for i in 0..n {
            circuit.mz(&[i]);
        }
        circuit.set_attr("num_measurements", Attribute::String(n.to_string()));
        let n_i64 = i64::try_from(n).unwrap();
        let obs: Vec<String> = (0..n)
            .map(|i| {
                let rec = i64::try_from(i).unwrap() - n_i64;
                format!(r#"{{"id":{i},"records":[{rec}]}}"#)
            })
            .collect();
        circuit.set_attr(
            "observables",
            Attribute::String(format!("[{}]", obs.join(","))),
        );

        let dem = DemBuilder::from_circuit(&circuit, 0.03, 0.0, 0.02, 0.0).unwrap();
        let sampler = DemSampler::from_detector_error_model(&dem);
        assert_eq!(sampler.num_dem_outputs(), n);

        let mask = sampler.observable_dem_output_mask();
        assert_eq!(mask.count_ones(), u32::try_from(n).unwrap());
        assert!(mask.get(64), "observable 64 must be representable");
        assert_eq!(mask.to_u64(), None, "65 observables do not fit a u64");
    }

    #[test]
    fn raw_mode_without_dem_outputs_reports_zero_dem_outputs() {
        let mut circuit = DagCircuit::new();
        circuit.pz(&[0]);
        circuit.h(&[0]);
        circuit.mz(&[0]);
        let im = InfluenceBuilder::new(&circuit)
            .build()
            .expect("circuit is replayable");

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

        let dem = DemBuilder::from_circuit(&circuit, 0.03, 0.0, 0.02, 0.0).unwrap();
        let sampler = DemSampler::from_detector_error_model(&dem);

        assert_eq!(sampler.observable_ids(), vec![0]);
        assert_eq!(
            sampler
                .tracked_pauli_ids()
                .unwrap_err()
                .num_tracked_paulis(),
            1
        );
        let obs_mask = sampler.observable_dem_output_mask();
        assert_eq!(obs_mask, ObsMask::from_u64(1));
        assert!(
            sampler
                .observable_mask_from_dem_output_flips(&[false], &obs_mask)
                .is_zero()
        );
        assert_eq!(
            sampler.observable_mask_from_dem_output_flips(&[true], &obs_mask),
            ObsMask::from_u64(1)
        );
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
        let im = InfluenceBuilder::new(&circuit)
            .with_z(&[0, 1, 2])
            .build()
            .expect("circuit is replayable");

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
        let im = InfluenceBuilder::new(&circuit)
            .with_z(&[0, 1, 2])
            .build()
            .expect("circuit is replayable");

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
        let im = InfluenceBuilder::new(&circuit)
            .with_z(&[0, 1, 2])
            .build()
            .expect("circuit is replayable");

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
        let im = InfluenceBuilder::new(&circuit)
            .with_z(&[0, 1, 2])
            .build()
            .expect("circuit is replayable");

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
