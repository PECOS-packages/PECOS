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

//! Reusable detector-error-model slices and just-in-time window assembly.
//!
//! A slice owns the fault mechanisms introduced by one bounded circuit block. Detector targets
//! are relative to the slice's round, so a contribution can expose dependencies on earlier or
//! later slices without depending on absolute detector numbers. At assembly time, a
//! [`DemSliceInstance`] maps the slice-local detector and observable identities into the current
//! logical schedule, and [`DemStitcher`] produces the structured [`DetectorErrorModel`] for one
//! commit-plus-buffer decoding window.
//!
//! This module deliberately does not compile physical circuits into slices. It is the stable
//! representation and stitching layer between an operation-template compiler and a windowed
//! decoder. Keeping those responsibilities separate lets circuit frontends choose how they
//! describe initialization, idle, logical-gate, and terminal templates while sharing the same
//! boundary and aggregation semantics.

use super::types::{DemOutput, DetectorDef, DetectorErrorModel, FaultContribution, FaultMechanism};
use smallvec::{Array, SmallVec};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::sync::Arc;

/// A detector target relative to the round at which its owning slice is instantiated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RelativeDetectorTarget {
    /// Slice-local detector identity.
    pub detector: u32,
    /// Signed syndrome-round offset from the owning slice.
    pub round_offset: i32,
}

impl RelativeDetectorTarget {
    /// Create a relative detector target.
    #[must_use]
    pub const fn new(detector: u32, round_offset: i32) -> Self {
        Self {
            detector,
            round_offset,
        }
    }
}

/// A detector/observable effect expressed in slice-local identities.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SliceFaultMechanism {
    /// Relative detector targets toggled by the mechanism.
    pub detectors: SmallVec<[RelativeDetectorTarget; 4]>,
    /// Slice-local standard DEM output (`L<n>`) identities toggled by the mechanism.
    pub dem_outputs: SmallVec<[u32; 2]>,
    /// Slice-local PECOS tracked-Pauli (`TP<n>`) identities toggled by the mechanism.
    pub tracked_paulis: SmallVec<[u32; 2]>,
}

impl SliceFaultMechanism {
    /// Construct a canonical mechanism from possibly unsorted targets.
    ///
    /// Repeated targets cancel by parity, matching detector-error-model XOR semantics.
    #[must_use]
    pub fn from_unsorted(
        detectors: impl IntoIterator<Item = RelativeDetectorTarget>,
        dem_outputs: impl IntoIterator<Item = u32>,
    ) -> Self {
        Self::from_unsorted_with_tracked_paulis(detectors, dem_outputs, std::iter::empty())
    }

    /// Construct a canonical mechanism including PECOS tracked-Pauli targets.
    #[must_use]
    pub fn from_unsorted_with_tracked_paulis(
        detectors: impl IntoIterator<Item = RelativeDetectorTarget>,
        dem_outputs: impl IntoIterator<Item = u32>,
        tracked_paulis: impl IntoIterator<Item = u32>,
    ) -> Self {
        Self {
            detectors: parity_sorted(detectors),
            dem_outputs: parity_sorted(dem_outputs),
            tracked_paulis: parity_sorted(tracked_paulis),
        }
    }

    /// Return the XOR of two local mechanisms.
    #[must_use]
    pub fn xor(&self, other: &Self) -> Self {
        Self::from_unsorted_with_tracked_paulis(
            self.detectors
                .iter()
                .copied()
                .chain(other.detectors.iter().copied()),
            self.dem_outputs
                .iter()
                .copied()
                .chain(other.dem_outputs.iter().copied()),
            self.tracked_paulis
                .iter()
                .copied()
                .chain(other.tracked_paulis.iter().copied()),
        )
    }

    /// Whether this mechanism has no detector, standard-output, or tracked-Pauli effect.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.detectors.is_empty() && self.dem_outputs.is_empty() && self.tracked_paulis.is_empty()
    }
}

fn parity_sorted<T, A>(values: impl IntoIterator<Item = T>) -> SmallVec<A>
where
    T: Copy + Ord,
    A: Array<Item = T>,
{
    let mut toggled = BTreeSet::new();
    for value in values {
        if !toggled.remove(&value) {
            toggled.insert(value);
        }
    }
    toggled.into_iter().collect()
}

/// One independent fault contribution owned by a DEM slice.
#[derive(Debug, Clone)]
pub enum DemSliceContribution {
    /// A direct contribution with no source decomposition.
    Direct {
        /// Complete detector/output effect.
        effect: SliceFaultMechanism,
        /// Independent occurrence probability.
        probability: f64,
    },
    /// A contribution whose effect is the XOR of X- and Z-like source components.
    YDecomposed {
        /// X-like component effect.
        x_effect: SliceFaultMechanism,
        /// Z-like component effect.
        z_effect: SliceFaultMechanism,
        /// Independent occurrence probability.
        probability: f64,
    },
    /// A contribution with an arbitrary source-frame decomposition.
    SourceDecomposed {
        /// Source components whose XOR is the complete effect.
        components: Vec<SliceFaultMechanism>,
        /// Independent occurrence probability.
        probability: f64,
    },
}

impl DemSliceContribution {
    /// Create a direct contribution.
    #[must_use]
    pub const fn direct(effect: SliceFaultMechanism, probability: f64) -> Self {
        Self::Direct {
            effect,
            probability,
        }
    }

    /// Create a source-decomposed contribution.
    #[must_use]
    pub const fn y_decomposed(
        x_effect: SliceFaultMechanism,
        z_effect: SliceFaultMechanism,
        probability: f64,
    ) -> Self {
        Self::YDecomposed {
            x_effect,
            z_effect,
            probability,
        }
    }

    /// Create a contribution with an arbitrary source-frame decomposition.
    #[must_use]
    pub fn source_decomposed(
        components: impl IntoIterator<Item = SliceFaultMechanism>,
        probability: f64,
    ) -> Self {
        Self::SourceDecomposed {
            components: components.into_iter().collect(),
            probability,
        }
    }

    fn probability(&self) -> f64 {
        match self {
            Self::Direct { probability, .. }
            | Self::YDecomposed { probability, .. }
            | Self::SourceDecomposed { probability, .. } => *probability,
        }
    }

    fn components(&self) -> SmallVec<[&SliceFaultMechanism; 4]> {
        match self {
            Self::Direct { effect, .. } => smallvec::smallvec![effect],
            Self::YDecomposed {
                x_effect, z_effect, ..
            } => smallvec::smallvec![x_effect, z_effect],
            Self::SourceDecomposed { components, .. } => components.iter().collect(),
        }
    }

    fn combined_effect(&self) -> SliceFaultMechanism {
        self.components()
            .into_iter()
            .fold(SliceFaultMechanism::default(), |effect, component| {
                effect.xor(component)
            })
    }
}

/// Declares one detector identity produced by each instance of a slice.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DemSliceDetector {
    /// Slice-local detector identity used by relative targets.
    pub id: u32,
    /// Optional template-local spatial coordinates. Time is supplied by the instance.
    pub coords: Option<[f64; 2]>,
    /// Whether an instance emits this detector at its owning round.
    ///
    /// A non-emitting entry is a temporal port identity only. Contributions may target it in an
    /// adjacent round, where another slice instance is responsible for the declaration.
    pub emitted: bool,
}

impl DemSliceDetector {
    /// Create a detector without spatial coordinates.
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self {
            id,
            coords: None,
            emitted: true,
        }
    }

    /// Create a temporal port identity that is not emitted by this slice.
    #[must_use]
    pub const fn port(id: u32) -> Self {
        Self {
            id,
            coords: None,
            emitted: false,
        }
    }

    /// Attach template-local spatial coordinates.
    #[must_use]
    pub const fn with_coords(mut self, coords: [f64; 2]) -> Self {
        self.coords = Some(coords);
        self
    }
}

/// Declared bound on how far a slice's detector effects may extend in time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DemTemporalHorizon {
    /// Maximum number of rounds before the owning slice.
    pub past_rounds: u32,
    /// Maximum number of rounds after the owning slice.
    pub future_rounds: u32,
}

impl DemTemporalHorizon {
    /// Create a temporal horizon.
    #[must_use]
    pub const fn new(past_rounds: u32, future_rounds: u32) -> Self {
        Self {
            past_rounds,
            future_rounds,
        }
    }
}

/// A reusable, absolute-index-free detector-error-model fragment.
#[derive(Debug, Clone)]
pub struct DemSlice {
    name: String,
    detectors: Vec<DemSliceDetector>,
    contributions: Vec<DemSliceContribution>,
    horizon: DemTemporalHorizon,
    local_dem_outputs: BTreeSet<u32>,
    local_tracked_paulis: BTreeSet<u32>,
}

/// Explicit identity map used when extracting a reusable slice from an existing DEM.
///
/// Detector keys are source-model `D<n>` identities and values are slice-local,
/// relative-round targets. Standard outputs and tracked Paulis are mapped into
/// their corresponding slice-local identity spaces.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DemSliceModelMap {
    /// Source detector to relative slice target.
    pub detectors: BTreeMap<u32, RelativeDetectorTarget>,
    /// Source `L<n>` output to slice-local `L<n>` identity.
    pub dem_outputs: BTreeMap<u32, u32>,
    /// Source `TP<n>` output to slice-local `TP<n>` identity.
    pub tracked_paulis: BTreeMap<u32, u32>,
}

impl DemSliceModelMap {
    /// Create an empty model map.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add or replace a source detector mapping.
    #[must_use]
    pub fn with_detector(mut self, source_detector: u32, target: RelativeDetectorTarget) -> Self {
        self.detectors.insert(source_detector, target);
        self
    }

    /// Add or replace a source standard-output mapping.
    #[must_use]
    pub fn with_dem_output(mut self, source_output: u32, local_output: u32) -> Self {
        self.dem_outputs.insert(source_output, local_output);
        self
    }

    /// Add or replace a source tracked-Pauli mapping.
    #[must_use]
    pub fn with_tracked_pauli(mut self, source_output: u32, local_output: u32) -> Self {
        self.tracked_paulis.insert(source_output, local_output);
        self
    }
}

/// Deterministic cache for reusable DEM slices.
///
/// The cache deliberately leaves key policy to the operation-template compiler. A frontend can
/// include physical circuit identity, code geometry, temporal horizon, and noise topology while
/// excluding absolute round numbers and logical relabeling state.
#[derive(Debug, Clone)]
pub struct DemSliceCache<K> {
    entries: BTreeMap<K, Arc<DemSlice>>,
}

impl<K> Default for DemSliceCache<K> {
    fn default() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }
}

impl<K: Ord> DemSliceCache<K> {
    /// Create an empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of cached physical templates.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache contains no templates.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Look up a cached slice.
    #[must_use]
    pub fn get(&self, key: &K) -> Option<Arc<DemSlice>> {
        self.entries.get(key).cloned()
    }

    /// Insert or replace a cached slice.
    pub fn insert(&mut self, key: K, slice: Arc<DemSlice>) -> Option<Arc<DemSlice>> {
        self.entries.insert(key, slice)
    }

    /// Return a cached slice, compiling and inserting it on a miss.
    ///
    /// A failed compilation leaves the cache unchanged.
    ///
    /// # Errors
    ///
    /// Returns the error produced by `compile` on a cache miss.
    pub fn get_or_try_insert_with<E>(
        &mut self,
        key: K,
        compile: impl FnOnce() -> Result<DemSlice, E>,
    ) -> Result<Arc<DemSlice>, E> {
        if let Some(slice) = self.entries.get(&key) {
            return Ok(Arc::clone(slice));
        }
        let slice = Arc::new(compile()?);
        self.entries.insert(key, Arc::clone(&slice));
        Ok(slice)
    }
}

impl DemSlice {
    /// Validate and create a reusable slice.
    ///
    /// # Errors
    ///
    /// Returns an error for duplicate or undeclared detector identities, invalid probabilities,
    /// or detector targets outside the declared temporal horizon.
    pub fn new(
        name: impl Into<String>,
        detectors: Vec<DemSliceDetector>,
        contributions: Vec<DemSliceContribution>,
        horizon: DemTemporalHorizon,
    ) -> Result<Self, DemSliceStitchError> {
        let name = name.into();
        let mut declared = BTreeSet::new();
        for detector in &detectors {
            if !declared.insert(detector.id) {
                return Err(DemSliceStitchError::DuplicateLocalDetector {
                    slice: name,
                    detector: detector.id,
                });
            }
        }

        let mut local_dem_outputs = BTreeSet::new();
        let mut local_tracked_paulis = BTreeSet::new();
        for contribution in &contributions {
            let probability = contribution.probability();
            if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
                return Err(DemSliceStitchError::InvalidProbability {
                    slice: name,
                    probability,
                });
            }
            for effect in contribution.components() {
                for target in &effect.detectors {
                    if !declared.contains(&target.detector) {
                        return Err(DemSliceStitchError::UndeclaredLocalDetector {
                            slice: name,
                            detector: target.detector,
                        });
                    }
                    let within_past = target.round_offset >= 0
                        || target.round_offset.unsigned_abs() <= horizon.past_rounds;
                    let within_future = target.round_offset <= 0
                        || target.round_offset.unsigned_abs() <= horizon.future_rounds;
                    if !within_past || !within_future {
                        return Err(DemSliceStitchError::TemporalHorizonExceeded {
                            slice: name,
                            detector: target.detector,
                            round_offset: target.round_offset,
                            horizon,
                        });
                    }
                }
                local_dem_outputs.extend(effect.dem_outputs.iter().copied());
                local_tracked_paulis.extend(effect.tracked_paulis.iter().copied());
            }
        }

        Ok(Self {
            name,
            detectors,
            contributions,
            horizon,
            local_dem_outputs,
            local_tracked_paulis,
        })
    }

    /// Extract a reusable relative-address slice from a bounded structured DEM.
    ///
    /// This adapter preserves each [`super::types::FaultContribution`] as one
    /// independent source, including Y-specific and arbitrary source-frame
    /// decomposition components. Every source detector declaration must have
    /// an explicit relative mapping; this prevents a halo or boundary detector
    /// from disappearing silently.
    ///
    /// Source detectors mapped to offset zero become emitted local detectors.
    /// Identities seen only at non-zero offsets become non-emitting temporal
    /// ports. Spatial coordinates are inherited from the source declarations;
    /// the source time coordinate is replaced by the instance round.
    ///
    /// # Errors
    ///
    /// Returns an error for incomplete or inconsistent identity maps, source
    /// components whose XOR does not match their complete effect, or any normal
    /// slice validation failure.
    pub fn from_detector_error_model(
        name: impl Into<String>,
        model: &DetectorErrorModel,
        source_map: &DemSliceModelMap,
        horizon: DemTemporalHorizon,
    ) -> Result<Self, DemSliceStitchError> {
        Self::from_detector_error_model_selected(name.into(), model, source_map, horizon, |_| {
            Ok(true)
        })
    }

    /// Extract only contributions owned by a set of influence-map locations.
    ///
    /// A source contribution is included when all of its location indices are
    /// in `owned_locations`, and omitted when none are. Partial ownership is an
    /// error because splitting one correlated physical source between slices
    /// would double-count or decorrelate it. Unattributed contributions also
    /// fail loudly because their ownership cannot be established.
    ///
    /// # Errors
    ///
    /// Returns an error for unattributed or partially owned contributions in
    /// addition to the errors from [`Self::from_detector_error_model`].
    pub fn from_detector_error_model_for_locations(
        name: impl Into<String>,
        model: &DetectorErrorModel,
        source_map: &DemSliceModelMap,
        horizon: DemTemporalHorizon,
        owned_locations: &BTreeSet<u32>,
    ) -> Result<Self, DemSliceStitchError> {
        let name = name.into();
        Self::from_detector_error_model_selected(
            name.clone(),
            model,
            source_map,
            horizon,
            |contribution| {
                if contribution.location_indices.is_empty() {
                    return Err(DemSliceStitchError::UnattributedSourceContribution {
                        slice: name.clone(),
                    });
                }
                let owned = contribution
                    .location_indices
                    .iter()
                    .filter(|location| owned_locations.contains(location))
                    .count();
                if owned == 0 {
                    Ok(false)
                } else if owned == contribution.location_indices.len() {
                    Ok(true)
                } else {
                    Err(DemSliceStitchError::PartialSourceOwnership {
                        slice: name.clone(),
                        location_indices: contribution.location_indices.to_vec(),
                    })
                }
            },
        )
    }

    fn from_detector_error_model_selected(
        name: String,
        model: &DetectorErrorModel,
        source_map: &DemSliceModelMap,
        horizon: DemTemporalHorizon,
        include: impl Fn(&FaultContribution) -> Result<bool, DemSliceStitchError>,
    ) -> Result<Self, DemSliceStitchError> {
        let source_detectors: BTreeMap<_, _> = model
            .detectors
            .iter()
            .map(|detector| (detector.id, detector))
            .collect();

        for source_detector in source_detectors.keys() {
            if !source_map.detectors.contains_key(source_detector) {
                return Err(DemSliceStitchError::MissingSourceDetectorMapping {
                    slice: name,
                    source_detector: *source_detector,
                });
            }
        }

        let mut local_detectors: BTreeMap<u32, (bool, Option<[f64; 2]>)> = BTreeMap::new();
        for (&source_detector, target) in &source_map.detectors {
            let declaration = source_detectors.get(&source_detector).ok_or_else(|| {
                DemSliceStitchError::UnknownSourceDetector {
                    slice: name.clone(),
                    source_detector,
                }
            })?;
            let coords = declaration.coords.map(|[x, y, _]| [x, y]);
            let emitted = target.round_offset == 0;
            match local_detectors.entry(target.detector) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert((emitted, coords));
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    let (already_emitted, existing_coords) = entry.get_mut();
                    if emitted && *already_emitted {
                        return Err(DemSliceStitchError::DuplicateEmittedSourceDetector {
                            slice: name,
                            local_detector: target.detector,
                        });
                    }
                    if let (Some(existing), Some(incoming)) = (*existing_coords, coords)
                        && !spatial_coords_equal(existing, incoming)
                    {
                        return Err(DemSliceStitchError::ConflictingLocalDetectorCoordinates {
                            slice: name,
                            local_detector: target.detector,
                        });
                    }
                    *already_emitted |= emitted;
                    if existing_coords.is_none() {
                        *existing_coords = coords;
                    }
                }
            }
        }

        let detectors = local_detectors
            .into_iter()
            .map(|(id, (emitted, coords))| DemSliceDetector {
                id,
                coords,
                emitted,
            })
            .collect();

        let mut contributions = Vec::new();
        for contribution in model.contributions() {
            if include(contribution)? {
                contributions.push(map_source_contribution(&name, contribution, source_map)?);
            }
        }

        Self::new(name, detectors, contributions, horizon)
    }

    /// Human-readable template name, used in diagnostics and cache inspection.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Detectors declared at the slice's owning round.
    #[must_use]
    pub fn detectors(&self) -> &[DemSliceDetector] {
        &self.detectors
    }

    /// Independent fault contributions owned by the slice.
    #[must_use]
    pub fn contributions(&self) -> &[DemSliceContribution] {
        &self.contributions
    }

    /// Validated temporal reach of the slice.
    #[must_use]
    pub const fn horizon(&self) -> DemTemporalHorizon {
        self.horizon
    }
}

fn map_source_contribution(
    slice: &str,
    contribution: &FaultContribution,
    source_map: &DemSliceModelMap,
) -> Result<DemSliceContribution, DemSliceStitchError> {
    let complete = map_source_mechanism(slice, &contribution.effect, source_map)?;
    if let Some((x_effect, z_effect)) = contribution.decomposition_components() {
        let x_effect = map_source_mechanism(slice, &x_effect, source_map)?;
        let z_effect = map_source_mechanism(slice, &z_effect, source_map)?;
        if x_effect.xor(&z_effect) != complete {
            return Err(DemSliceStitchError::SourceComponentEffectMismatch {
                slice: slice.to_owned(),
            });
        }
        return Ok(DemSliceContribution::y_decomposed(
            x_effect,
            z_effect,
            contribution.probability,
        ));
    }

    let source_components = contribution.source_component_effects().or_else(|| {
        contribution
            .direct_component_effects()
            .map(|(first, second)| smallvec::smallvec![first, second])
    });
    if let Some(source_components) = source_components {
        let components: Vec<_> = source_components
            .iter()
            .map(|component| map_source_mechanism(slice, component, source_map))
            .collect::<Result<_, _>>()?;
        let combined = components
            .iter()
            .fold(SliceFaultMechanism::default(), |effect, component| {
                effect.xor(component)
            });
        if combined != complete {
            return Err(DemSliceStitchError::SourceComponentEffectMismatch {
                slice: slice.to_owned(),
            });
        }
        Ok(DemSliceContribution::source_decomposed(
            components,
            contribution.probability,
        ))
    } else {
        Ok(DemSliceContribution::direct(
            complete,
            contribution.probability,
        ))
    }
}

fn map_source_mechanism(
    slice: &str,
    source: &FaultMechanism,
    source_map: &DemSliceModelMap,
) -> Result<SliceFaultMechanism, DemSliceStitchError> {
    let detectors = source
        .detectors
        .iter()
        .map(|detector| {
            source_map.detectors.get(detector).copied().ok_or_else(|| {
                DemSliceStitchError::MissingSourceDetectorMapping {
                    slice: slice.to_owned(),
                    source_detector: *detector,
                }
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let dem_outputs = source
        .dem_outputs
        .iter()
        .map(|output| {
            source_map.dem_outputs.get(output).copied().ok_or_else(|| {
                DemSliceStitchError::MissingSourceDemOutputMapping {
                    slice: slice.to_owned(),
                    source_output: *output,
                }
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let tracked_paulis = source
        .tracked_paulis
        .iter()
        .map(|output| {
            source_map
                .tracked_paulis
                .get(output)
                .copied()
                .ok_or_else(|| DemSliceStitchError::MissingSourceTrackedPauliMapping {
                    slice: slice.to_owned(),
                    source_output: *output,
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(SliceFaultMechanism::from_unsorted_with_tracked_paulis(
        detectors,
        dem_outputs,
        tracked_paulis,
    ))
}

fn spatial_coords_equal(left: [f64; 2], right: [f64; 2]) -> bool {
    left.into_iter()
        .zip(right)
        .all(|(left, right)| left.partial_cmp(&right) == Some(std::cmp::Ordering::Equal))
}

/// Placement of one slice-local detector in an algorithm-wide detector stream.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DemDetectorPlacement {
    /// Stable stream identity, such as a code-block/check pair.
    pub stream_id: u32,
    /// Optional absolute spatial coordinates. When absent, template coordinates are retained.
    pub coords: Option<[f64; 2]>,
}

impl DemDetectorPlacement {
    /// Create a placement that retains the template's spatial coordinates.
    #[must_use]
    pub const fn new(stream_id: u32) -> Self {
        Self {
            stream_id,
            coords: None,
        }
    }

    /// Override the detector's absolute spatial coordinates.
    #[must_use]
    pub const fn with_coords(mut self, coords: [f64; 2]) -> Self {
        self.coords = Some(coords);
        self
    }
}

/// One scheduled use of a cached DEM slice.
#[derive(Debug, Clone)]
pub struct DemSliceInstance {
    slice: Arc<DemSlice>,
    round: i64,
    detector_map: BTreeMap<u32, DemDetectorPlacement>,
    dem_output_map: BTreeMap<u32, u32>,
    tracked_pauli_map: BTreeMap<u32, u32>,
}

impl DemSliceInstance {
    /// Instantiate a slice with identity detector and output mappings.
    #[must_use]
    pub fn identity(slice: Arc<DemSlice>, round: i64) -> Self {
        let detector_map = slice
            .detectors
            .iter()
            .map(|detector| (detector.id, DemDetectorPlacement::new(detector.id)))
            .collect();
        let dem_output_map = slice
            .local_dem_outputs
            .iter()
            .map(|&output| (output, output))
            .collect();
        let tracked_pauli_map = slice
            .local_tracked_paulis
            .iter()
            .map(|&output| (output, output))
            .collect();
        Self {
            slice,
            round,
            detector_map,
            dem_output_map,
            tracked_pauli_map,
        }
    }

    /// Replace a local detector's stream placement.
    #[must_use]
    pub fn with_detector_placement(
        mut self,
        local_detector: u32,
        placement: DemDetectorPlacement,
    ) -> Self {
        self.detector_map.insert(local_detector, placement);
        self
    }

    /// Replace a local standard DEM-output mapping.
    #[must_use]
    pub fn with_dem_output(mut self, local_output: u32, global_output: u32) -> Self {
        self.dem_output_map.insert(local_output, global_output);
        self
    }

    /// Replace a local PECOS tracked-Pauli mapping.
    #[must_use]
    pub fn with_tracked_pauli(mut self, local_output: u32, global_output: u32) -> Self {
        self.tracked_pauli_map.insert(local_output, global_output);
        self
    }

    /// The round at which this instance owns its contributions and detector declarations.
    #[must_use]
    pub const fn round(&self) -> i64 {
        self.round
    }

    /// The cached slice used by this instance.
    #[must_use]
    pub fn slice(&self) -> &Arc<DemSlice> {
        &self.slice
    }
}

/// Whether unresolved forward temporal ports are legal at the end of a window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DemBoundaryKind {
    /// Terminal boundary: every future target must be resolved inside the window.
    Hard,
    /// Sliding boundary: future targets may project to the decoder boundary.
    Soft,
}

/// Half-open commit-plus-buffer window specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DemWindowSpec {
    /// First syndrome round included in the window.
    pub start_round: i64,
    /// Number of rounds whose corrections may be committed.
    pub commit_rounds: u32,
    /// Number of look-ahead rounds after the commit region.
    pub buffer_rounds: u32,
    /// Boundary behavior after the final buffer round.
    pub forward_boundary: DemBoundaryKind,
}

impl DemWindowSpec {
    /// Create a window with a hard backward boundary and the requested forward boundary.
    #[must_use]
    pub const fn new(
        start_round: i64,
        commit_rounds: u32,
        buffer_rounds: u32,
        forward_boundary: DemBoundaryKind,
    ) -> Self {
        Self {
            start_round,
            commit_rounds,
            buffer_rounds,
            forward_boundary,
        }
    }

    fn commit_end(self) -> Result<i64, DemSliceStitchError> {
        if self.commit_rounds == 0 {
            return Err(DemSliceStitchError::EmptyCommitRegion);
        }
        self.start_round
            .checked_add(i64::from(self.commit_rounds))
            .ok_or(DemSliceStitchError::RoundOverflow)
    }

    fn end(self) -> Result<i64, DemSliceStitchError> {
        self.commit_end()?
            .checked_add(i64::from(self.buffer_rounds))
            .ok_or(DemSliceStitchError::RoundOverflow)
    }
}

/// Algorithm-wide address assigned to one detector in a stitched model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct StitchedDetectorAddress {
    /// Syndrome round.
    pub round: i64,
    /// Stable detector-stream identity supplied by the slice instance.
    pub stream_id: u32,
}

/// Diagnostics describing boundary projection performed during assembly.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DemStitchDiagnostics {
    /// Contributions with one or more targets projected through the hard backward boundary.
    pub projected_past_contributions: usize,
    /// Contributions with one or more targets projected through the soft forward boundary.
    pub projected_future_contributions: usize,
}

/// A structured window DEM and its local-to-algorithm detector map.
#[derive(Debug, Clone)]
pub struct StitchedDem {
    /// Assembled decoder-facing model.
    pub model: DetectorErrorModel,
    /// `D<n>` index to algorithm-wide detector address.
    pub detector_addresses: Vec<StitchedDetectorAddress>,
    /// Boundary-projection diagnostics.
    pub diagnostics: DemStitchDiagnostics,
}

/// Just-in-time assembler for reusable DEM slice instances.
#[derive(Debug, Clone)]
pub struct DemStitcher {
    spec: DemWindowSpec,
    observables: Vec<DemOutput>,
    tracked_paulis: Vec<DemOutput>,
}

impl DemStitcher {
    /// Create a stitcher for one commit-plus-buffer window.
    #[must_use]
    pub const fn new(spec: DemWindowSpec) -> Self {
        Self {
            spec,
            observables: Vec::new(),
            tracked_paulis: Vec::new(),
        }
    }

    /// Supply algorithm-wide standard DEM-output definitions.
    #[must_use]
    pub fn with_observables(mut self, observables: Vec<DemOutput>) -> Self {
        self.observables = observables;
        self
    }

    /// Supply algorithm-wide PECOS tracked-Pauli definitions.
    #[must_use]
    pub fn with_tracked_paulis(mut self, tracked_paulis: Vec<DemOutput>) -> Self {
        self.tracked_paulis = tracked_paulis;
        self
    }

    /// Assemble a structured detector error model from scheduled slice instances.
    ///
    /// Instances before `start_round` may be supplied as a bounded source halo. Their detector
    /// declarations are omitted, but contributions reaching into the window are retained and
    /// projected through the hard backward boundary. Instances after the window are not needed:
    /// future dependencies are already present as relative targets on slices inside the window.
    ///
    /// # Errors
    ///
    /// Returns an error when mappings are incomplete, an in-window target has no detector
    /// declaration, a hard terminal boundary has an unresolved future target, or a contribution
    /// reaches from the commit region beyond the supplied buffer.
    pub fn stitch(
        &self,
        instances: &[DemSliceInstance],
    ) -> Result<StitchedDem, DemSliceStitchError> {
        let commit_end = self.spec.commit_end()?;
        let end_round = self.spec.end()?;

        let mut declarations: BTreeMap<StitchedDetectorAddress, Option<[f64; 2]>> = BTreeMap::new();
        for instance in instances {
            if instance.round < self.spec.start_round || instance.round >= end_round {
                continue;
            }
            for detector in &instance.slice.detectors {
                if !detector.emitted {
                    continue;
                }
                let placement = instance.detector_map.get(&detector.id).ok_or_else(|| {
                    DemSliceStitchError::MissingDetectorMapping {
                        slice: instance.slice.name.clone(),
                        detector: detector.id,
                    }
                })?;
                let address = StitchedDetectorAddress {
                    round: instance.round,
                    stream_id: placement.stream_id,
                };
                let coords = placement.coords.or(detector.coords);
                if let Some(existing) = declarations.insert(address, coords)
                    && existing != coords
                {
                    return Err(DemSliceStitchError::ConflictingDetectorDeclaration { address });
                }
            }
        }

        let mut address_to_id = BTreeMap::new();
        let mut detector_addresses = Vec::with_capacity(declarations.len());
        let mut model = DetectorErrorModel::with_capacity(declarations.len(), 0);
        for (index, (address, coords)) in declarations.into_iter().enumerate() {
            let id = u32::try_from(index).map_err(|_| DemSliceStitchError::TooManyDetectors)?;
            address_to_id.insert(address, id);
            detector_addresses.push(address);
            let mut detector = DetectorDef::new(id);
            if let Some([x, y]) = coords {
                detector.coords = Some([x, y, round_coordinate(address.round)?]);
            }
            model.add_detector(detector);
        }

        for observable in &self.observables {
            model.add_observable(observable.clone());
        }
        for tracked_pauli in &self.tracked_paulis {
            model.add_tracked_pauli(tracked_pauli.clone());
        }

        let mut diagnostics = DemStitchDiagnostics::default();
        let mut referenced_outputs = BTreeSet::new();
        let mut referenced_tracked_paulis = BTreeSet::new();
        let context = StitchContext {
            spec: self.spec,
            commit_end,
            end_round,
            address_to_id: &address_to_id,
        };
        for instance in instances {
            for contribution in &instance.slice.contributions {
                if !contribution_is_relevant(contribution, instance, self.spec, end_round)? {
                    continue;
                }
                let mut projection = Projection::default();
                let complete = instantiate_effect(
                    &contribution.combined_effect(),
                    instance,
                    &context,
                    &mut projection,
                    true,
                )?;
                match contribution {
                    DemSliceContribution::Direct { probability, .. } => {
                        referenced_outputs.extend(complete.dem_outputs.iter().copied());
                        referenced_tracked_paulis.extend(complete.tracked_paulis.iter().copied());
                        model.add_direct_contribution(complete, *probability);
                    }
                    DemSliceContribution::YDecomposed {
                        x_effect,
                        z_effect,
                        probability,
                    } => {
                        let mut component_projection = Projection::default();
                        let x_effect = instantiate_effect(
                            x_effect,
                            instance,
                            &context,
                            &mut component_projection,
                            false,
                        )?;
                        let z_effect = instantiate_effect(
                            z_effect,
                            instance,
                            &context,
                            &mut component_projection,
                            false,
                        )?;
                        debug_assert_eq!(x_effect.xor(&z_effect), complete);
                        referenced_outputs.extend(x_effect.dem_outputs.iter().copied());
                        referenced_outputs.extend(z_effect.dem_outputs.iter().copied());
                        referenced_tracked_paulis.extend(x_effect.tracked_paulis.iter().copied());
                        referenced_tracked_paulis.extend(z_effect.tracked_paulis.iter().copied());
                        model.add_y_decomposed_contribution(&x_effect, &z_effect, *probability);
                    }
                    DemSliceContribution::SourceDecomposed {
                        components,
                        probability,
                    } => {
                        let mut component_projection = Projection::default();
                        let components: Vec<_> = components
                            .iter()
                            .map(|component| {
                                instantiate_effect(
                                    component,
                                    instance,
                                    &context,
                                    &mut component_projection,
                                    false,
                                )
                            })
                            .collect::<Result<_, _>>()?;
                        debug_assert_eq!(
                            components
                                .iter()
                                .fold(FaultMechanism::new(), |effect, component| {
                                    effect.xor(component)
                                }),
                            complete
                        );
                        for component in &components {
                            referenced_outputs.extend(component.dem_outputs.iter().copied());
                            referenced_tracked_paulis
                                .extend(component.tracked_paulis.iter().copied());
                        }
                        model.add_source_decomposed_contribution(components, *probability);
                    }
                }

                if projection.past {
                    diagnostics.projected_past_contributions += 1;
                }
                if projection.future {
                    diagnostics.projected_future_contributions += 1;
                }
            }
        }

        let declared_outputs: BTreeSet<u32> =
            self.observables.iter().map(|output| output.id).collect();
        for output in referenced_outputs.difference(&declared_outputs) {
            model.add_observable(DemOutput::new(*output));
        }
        let declared_tracked_paulis: BTreeSet<u32> =
            self.tracked_paulis.iter().map(|output| output.id).collect();
        for output in referenced_tracked_paulis.difference(&declared_tracked_paulis) {
            model.add_tracked_pauli(DemOutput::new(*output));
        }

        Ok(StitchedDem {
            model,
            detector_addresses,
            diagnostics,
        })
    }
}

fn round_coordinate(round: i64) -> Result<f64, DemSliceStitchError> {
    const MAX_EXACT_F64_INTEGER: u64 = 1_u64 << f64::MANTISSA_DIGITS;
    if round.unsigned_abs() > MAX_EXACT_F64_INTEGER {
        return Err(DemSliceStitchError::InexactRoundCoordinate { round });
    }
    // The bound above proves that the integer is exactly representable as an f64.
    #[allow(clippy::cast_precision_loss)]
    Ok(round as f64)
}

fn contribution_is_relevant(
    contribution: &DemSliceContribution,
    instance: &DemSliceInstance,
    spec: DemWindowSpec,
    end_round: i64,
) -> Result<bool, DemSliceStitchError> {
    if instance.round >= spec.start_round && instance.round < end_round {
        return Ok(true);
    }

    for target in &contribution.combined_effect().detectors {
        let round = instance
            .round
            .checked_add(i64::from(target.round_offset))
            .ok_or(DemSliceStitchError::RoundOverflow)?;
        if round >= spec.start_round && round < end_round {
            return Ok(true);
        }
    }
    Ok(false)
}

#[derive(Debug, Default)]
struct Projection {
    past: bool,
    future: bool,
    touches_commit: bool,
}

struct StitchContext<'a> {
    spec: DemWindowSpec,
    commit_end: i64,
    end_round: i64,
    address_to_id: &'a BTreeMap<StitchedDetectorAddress, u32>,
}

fn instantiate_effect(
    local: &SliceFaultMechanism,
    instance: &DemSliceInstance,
    context: &StitchContext<'_>,
    projection: &mut Projection,
    validate_boundary: bool,
) -> Result<FaultMechanism, DemSliceStitchError> {
    let mut detectors = Vec::with_capacity(local.detectors.len());
    for target in &local.detectors {
        let placement = instance.detector_map.get(&target.detector).ok_or_else(|| {
            DemSliceStitchError::MissingDetectorMapping {
                slice: instance.slice.name.clone(),
                detector: target.detector,
            }
        })?;
        let round = instance
            .round
            .checked_add(i64::from(target.round_offset))
            .ok_or(DemSliceStitchError::RoundOverflow)?;
        if round < context.spec.start_round {
            projection.past = true;
            continue;
        }
        if round >= context.end_round {
            projection.future = true;
            if validate_boundary && context.spec.forward_boundary == DemBoundaryKind::Hard {
                return Err(DemSliceStitchError::UnresolvedHardForwardPort {
                    slice: instance.slice.name.clone(),
                    source_round: instance.round,
                    target_round: round,
                });
            }
            continue;
        }
        if round < context.commit_end {
            projection.touches_commit = true;
        }
        let address = StitchedDetectorAddress {
            round,
            stream_id: placement.stream_id,
        };
        let id = context.address_to_id.get(&address).ok_or(
            DemSliceStitchError::MissingDetectorDeclaration {
                slice: instance.slice.name.clone(),
                address,
            },
        )?;
        detectors.push(*id);
    }

    if validate_boundary && projection.future && projection.touches_commit {
        return Err(DemSliceStitchError::BufferTooSmall {
            slice: instance.slice.name.clone(),
            source_round: instance.round,
            commit_end: context.commit_end,
            window_end: context.end_round,
        });
    }

    let mut outputs = Vec::with_capacity(local.dem_outputs.len());
    for output in &local.dem_outputs {
        let mapped = instance.dem_output_map.get(output).ok_or_else(|| {
            DemSliceStitchError::MissingDemOutputMapping {
                slice: instance.slice.name.clone(),
                output: *output,
            }
        })?;
        outputs.push(*mapped);
    }

    let mut tracked_paulis = Vec::with_capacity(local.tracked_paulis.len());
    for output in &local.tracked_paulis {
        let mapped = instance.tracked_pauli_map.get(output).ok_or_else(|| {
            DemSliceStitchError::MissingTrackedPauliMapping {
                slice: instance.slice.name.clone(),
                output: *output,
            }
        })?;
        tracked_paulis.push(*mapped);
    }

    Ok(FaultMechanism::from_unsorted_with_tracked_paulis(
        detectors,
        outputs,
        tracked_paulis,
    ))
}

/// Error returned while validating or stitching reusable DEM slices.
#[derive(Debug, Clone, PartialEq)]
pub enum DemSliceStitchError {
    /// A slice declares the same local detector more than once.
    DuplicateLocalDetector { slice: String, detector: u32 },
    /// A contribution references a detector the slice does not declare.
    UndeclaredLocalDetector { slice: String, detector: u32 },
    /// A source DEM detector declaration has no relative slice mapping.
    MissingSourceDetectorMapping { slice: String, source_detector: u32 },
    /// A source detector mapping names no declaration in the source DEM.
    UnknownSourceDetector { slice: String, source_detector: u32 },
    /// Multiple source detectors map to one emitted local detector at offset zero.
    DuplicateEmittedSourceDetector { slice: String, local_detector: u32 },
    /// Source declarations mapped to one local detector disagree on spatial coordinates.
    ConflictingLocalDetectorCoordinates { slice: String, local_detector: u32 },
    /// A source DEM output has no slice-local mapping.
    MissingSourceDemOutputMapping { slice: String, source_output: u32 },
    /// A source tracked Pauli has no slice-local mapping.
    MissingSourceTrackedPauliMapping { slice: String, source_output: u32 },
    /// Recorded source components do not XOR to their complete contribution effect.
    SourceComponentEffectMismatch { slice: String },
    /// A contribution has no influence-map location indices from which to determine ownership.
    UnattributedSourceContribution { slice: String },
    /// Only part of one correlated source was assigned to this slice.
    PartialSourceOwnership {
        slice: String,
        location_indices: Vec<u32>,
    },
    /// A contribution probability is NaN, infinite, negative, or greater than one.
    InvalidProbability { slice: String, probability: f64 },
    /// A target exceeds the slice's declared temporal reach.
    TemporalHorizonExceeded {
        slice: String,
        detector: u32,
        round_offset: i32,
        horizon: DemTemporalHorizon,
    },
    /// An instance does not map one of its slice's local detectors.
    MissingDetectorMapping { slice: String, detector: u32 },
    /// An instance does not map one of its slice's standard DEM outputs.
    MissingDemOutputMapping { slice: String, output: u32 },
    /// An instance does not map one of its slice's PECOS tracked Paulis.
    MissingTrackedPauliMapping { slice: String, output: u32 },
    /// Two instances declare different coordinates for the same algorithm-wide detector.
    ConflictingDetectorDeclaration { address: StitchedDetectorAddress },
    /// A contribution targets an in-window detector no instance declared.
    MissingDetectorDeclaration {
        slice: String,
        address: StitchedDetectorAddress,
    },
    /// A hard terminal boundary has an unresolved future target.
    UnresolvedHardForwardPort {
        slice: String,
        source_round: i64,
        target_round: i64,
    },
    /// A contribution reaches from the commit region through the entire buffer.
    BufferTooSmall {
        slice: String,
        source_round: i64,
        commit_end: i64,
        window_end: i64,
    },
    /// The commit region must contain at least one round.
    EmptyCommitRegion,
    /// Round arithmetic overflowed `i64`.
    RoundOverflow,
    /// A round does not fit exactly in a DEM's floating-point time coordinate.
    InexactRoundCoordinate { round: i64 },
    /// The stitched model contains more detectors than fit in a DEM `u32` identity.
    TooManyDetectors,
}

impl fmt::Display for DemSliceStitchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateLocalDetector { slice, detector } => {
                write!(
                    f,
                    "DEM slice {slice:?} declares local detector {detector} more than once"
                )
            }
            Self::UndeclaredLocalDetector { slice, detector } => write!(
                f,
                "DEM slice {slice:?} references undeclared local detector {detector}"
            ),
            Self::MissingSourceDetectorMapping {
                slice,
                source_detector,
            } => write!(
                f,
                "source DEM detector D{source_detector} has no relative mapping for slice {slice:?}"
            ),
            Self::UnknownSourceDetector {
                slice,
                source_detector,
            } => write!(
                f,
                "slice {slice:?} maps source detector D{source_detector}, but the source DEM does not declare it"
            ),
            Self::DuplicateEmittedSourceDetector {
                slice,
                local_detector,
            } => write!(
                f,
                "slice {slice:?} maps multiple offset-zero source detectors to emitted local detector {local_detector}"
            ),
            Self::ConflictingLocalDetectorCoordinates {
                slice,
                local_detector,
            } => write!(
                f,
                "source declarations mapped to slice {slice:?} local detector {local_detector} have conflicting spatial coordinates"
            ),
            Self::MissingSourceDemOutputMapping {
                slice,
                source_output,
            } => write!(
                f,
                "source DEM output L{source_output} has no local mapping for slice {slice:?}"
            ),
            Self::MissingSourceTrackedPauliMapping {
                slice,
                source_output,
            } => write!(
                f,
                "source tracked Pauli TP{source_output} has no local mapping for slice {slice:?}"
            ),
            Self::SourceComponentEffectMismatch { slice } => write!(
                f,
                "source decomposition components do not XOR to their complete effect in slice {slice:?}"
            ),
            Self::UnattributedSourceContribution { slice } => write!(
                f,
                "source contribution in slice {slice:?} has no influence-map location identity"
            ),
            Self::PartialSourceOwnership {
                slice,
                location_indices,
            } => write!(
                f,
                "slice {slice:?} owns only part of correlated source locations {location_indices:?}"
            ),
            Self::InvalidProbability { slice, probability } => write!(
                f,
                "DEM slice {slice:?} has invalid probability {probability}; expected a finite value in [0, 1]"
            ),
            Self::TemporalHorizonExceeded {
                slice,
                detector,
                round_offset,
                horizon,
            } => write!(
                f,
                "DEM slice {slice:?} target ({detector}, {round_offset:+}) exceeds its declared horizon (-{}, +{})",
                horizon.past_rounds, horizon.future_rounds
            ),
            Self::MissingDetectorMapping { slice, detector } => write!(
                f,
                "DEM slice instance {slice:?} has no mapping for local detector {detector}"
            ),
            Self::MissingDemOutputMapping { slice, output } => write!(
                f,
                "DEM slice instance {slice:?} has no mapping for local output L{output}"
            ),
            Self::MissingTrackedPauliMapping { slice, output } => write!(
                f,
                "DEM slice instance {slice:?} has no mapping for local tracked Pauli TP{output}"
            ),
            Self::ConflictingDetectorDeclaration { address } => write!(
                f,
                "conflicting declarations for detector stream {} at round {}",
                address.stream_id, address.round
            ),
            Self::MissingDetectorDeclaration { slice, address } => write!(
                f,
                "DEM slice {slice:?} targets detector stream {} at round {}, but no instance declares it",
                address.stream_id, address.round
            ),
            Self::UnresolvedHardForwardPort {
                slice,
                source_round,
                target_round,
            } => write!(
                f,
                "DEM slice {slice:?} at round {source_round} has an unresolved target at round {target_round} across a hard forward boundary"
            ),
            Self::BufferTooSmall {
                slice,
                source_round,
                commit_end,
                window_end,
            } => write!(
                f,
                "DEM slice {slice:?} at round {source_round} affects the commit region ending at {commit_end} and protrudes beyond window end {window_end}; increase the buffer"
            ),
            Self::EmptyCommitRegion => write!(f, "a DEM window commit region cannot be empty"),
            Self::RoundOverflow => write!(f, "DEM slice round arithmetic overflowed i64"),
            Self::InexactRoundCoordinate { round } => write!(
                f,
                "DEM slice round {round} cannot be represented exactly as an f64 detector coordinate"
            ),
            Self::TooManyDetectors => write!(f, "stitched DEM has more than u32::MAX detectors"),
        }
    }
}

impl Error for DemSliceStitchError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn target(detector: u32, round_offset: i32) -> RelativeDetectorTarget {
        RelativeDetectorTarget::new(detector, round_offset)
    }

    fn direct(
        probability: f64,
        detectors: impl IntoIterator<Item = RelativeDetectorTarget>,
    ) -> DemSliceContribution {
        DemSliceContribution::direct(
            SliceFaultMechanism::from_unsorted(detectors, std::iter::empty()),
            probability,
        )
    }

    fn bulk_slice() -> Arc<DemSlice> {
        Arc::new(
            DemSlice::new(
                "bulk",
                vec![DemSliceDetector::new(0).with_coords([2.0, 3.0])],
                vec![direct(0.01, [target(0, 0), target(0, 1)])],
                DemTemporalHorizon::new(0, 1),
            )
            .unwrap(),
        )
    }

    #[test]
    fn repeated_bulk_slice_stitches_a_soft_window() {
        let slice = bulk_slice();
        let instances: Vec<_> = (0..3)
            .map(|round| DemSliceInstance::identity(Arc::clone(&slice), round))
            .collect();
        let stitched = DemStitcher::new(DemWindowSpec::new(0, 2, 1, DemBoundaryKind::Soft))
            .stitch(&instances)
            .unwrap();

        assert_eq!(stitched.detector_addresses.len(), 3);
        assert_eq!(stitched.model.detectors[0].coords, Some([2.0, 3.0, 0.0]));
        assert_eq!(stitched.diagnostics.projected_future_contributions, 1);
        assert_eq!(
            stitched.model.to_mechanisms().0,
            vec![
                (0.01, vec![0, 1], vec![]),
                (0.01, vec![1, 2], vec![]),
                (0.01, vec![2], vec![]),
            ]
        );
    }

    #[test]
    fn buffer_must_cover_every_commit_touching_contribution() {
        let slice = bulk_slice();
        let instances: Vec<_> = (0..3)
            .map(|round| DemSliceInstance::identity(Arc::clone(&slice), round))
            .collect();
        let error = DemStitcher::new(DemWindowSpec::new(0, 3, 0, DemBoundaryKind::Soft))
            .stitch(&instances)
            .unwrap_err();

        assert!(matches!(error, DemSliceStitchError::BufferTooSmall { .. }));
    }

    #[test]
    fn hard_forward_boundary_rejects_an_unresolved_port() {
        let slice = bulk_slice();
        let instance = DemSliceInstance::identity(slice, 0);
        let error = DemStitcher::new(DemWindowSpec::new(0, 1, 0, DemBoundaryKind::Hard))
            .stitch(&[instance])
            .unwrap_err();

        assert!(matches!(
            error,
            DemSliceStitchError::UnresolvedHardForwardPort { .. }
        ));
    }

    #[test]
    fn hard_boundary_validates_the_combined_source_effect() {
        let future_component = SliceFaultMechanism::from_unsorted(
            [RelativeDetectorTarget::new(0, 1)],
            std::iter::empty(),
        );
        let slice = Arc::new(
            DemSlice::new(
                "cancelled source components",
                vec![DemSliceDetector::port(0)],
                vec![DemSliceContribution::source_decomposed(
                    [future_component.clone(), future_component],
                    0.125,
                )],
                DemTemporalHorizon::new(0, 1),
            )
            .unwrap(),
        );
        let stitched = DemStitcher::new(DemWindowSpec::new(0, 1, 0, DemBoundaryKind::Hard))
            .stitch(&[DemSliceInstance::identity(slice, 0)])
            .unwrap();

        assert!(stitched.model.contributions().is_empty());
        assert_eq!(stitched.diagnostics, DemStitchDiagnostics::default());
    }

    #[test]
    fn relabeling_routes_local_detectors_and_observables() {
        let slice = Arc::new(
            DemSlice::new(
                "mapped",
                vec![DemSliceDetector::new(0)],
                vec![DemSliceContribution::direct(
                    SliceFaultMechanism::from_unsorted([target(0, 0)], [0]),
                    0.125,
                )],
                DemTemporalHorizon::new(0, 0),
            )
            .unwrap(),
        );
        let instance = DemSliceInstance::identity(slice, 7)
            .with_detector_placement(0, DemDetectorPlacement::new(42).with_coords([5.0, 6.0]))
            .with_dem_output(0, 3);
        let stitched = DemStitcher::new(DemWindowSpec::new(7, 1, 0, DemBoundaryKind::Hard))
            .stitch(&[instance])
            .unwrap();

        assert_eq!(
            stitched.detector_addresses,
            vec![StitchedDetectorAddress {
                round: 7,
                stream_id: 42,
            }]
        );
        assert_eq!(
            stitched.model.to_mechanisms().0,
            vec![(0.125, vec![0], vec![3])]
        );
        assert_eq!(stitched.model.num_observables(), 4);
    }

    #[test]
    fn equivalent_soft_boundary_columns_combine_by_xor_probability() {
        let slice = Arc::new(
            DemSlice::new(
                "merge",
                vec![DemSliceDetector::new(0), DemSliceDetector::new(1)],
                vec![
                    direct(0.1, [target(0, 0), target(0, 1)]),
                    direct(0.2, [target(0, 0), target(1, 1)]),
                ],
                DemTemporalHorizon::new(0, 1),
            )
            .unwrap(),
        );
        let instance = DemSliceInstance::identity(slice, 1);
        let stitched = DemStitcher::new(DemWindowSpec::new(0, 1, 1, DemBoundaryKind::Soft))
            .stitch(&[instance])
            .unwrap();

        let mechanisms = stitched.model.to_mechanisms().0;
        assert_eq!(mechanisms.len(), 1);
        assert_eq!(mechanisms[0].1, vec![0]);
        assert!((mechanisms[0].0 - 0.26).abs() < 1e-12);
        assert_eq!(stitched.diagnostics.projected_future_contributions, 2);
    }

    #[test]
    fn cache_compiles_once_and_relabeling_stays_on_instances() {
        let mut cache = DemSliceCache::new();
        let mut compile_count = 0;
        let first = cache
            .get_or_try_insert_with("idle", || {
                compile_count += 1;
                Ok::<_, DemSliceStitchError>((*bulk_slice()).clone())
            })
            .unwrap();
        let second = cache
            .get_or_try_insert_with("idle", || {
                compile_count += 1;
                Ok::<_, DemSliceStitchError>((*bulk_slice()).clone())
            })
            .unwrap();

        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(compile_count, 1);
        assert_eq!(cache.len(), 1);

        let left = DemSliceInstance::identity(Arc::clone(&first), 0)
            .with_detector_placement(0, DemDetectorPlacement::new(10));
        let right = DemSliceInstance::identity(first, 0)
            .with_detector_placement(0, DemDetectorPlacement::new(20));
        assert_eq!(left.slice().name(), right.slice().name());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn slice_construction_rejects_a_false_temporal_bound() {
        let error = DemSlice::new(
            "bad horizon",
            vec![DemSliceDetector::new(0)],
            vec![direct(0.01, [target(0, 2)])],
            DemTemporalHorizon::new(0, 1),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            DemSliceStitchError::TemporalHorizonExceeded { .. }
        ));
    }

    #[test]
    fn missing_in_window_port_destination_fails_loudly() {
        let slice = bulk_slice();
        let instance = DemSliceInstance::identity(slice, 0);
        let error = DemStitcher::new(DemWindowSpec::new(0, 2, 0, DemBoundaryKind::Hard))
            .stitch(&[instance])
            .unwrap_err();

        assert!(matches!(
            error,
            DemSliceStitchError::MissingDetectorDeclaration { .. }
        ));
    }

    #[test]
    fn non_emitting_port_resolves_to_an_adjacent_slice_declaration() {
        let init = Arc::new(
            DemSlice::new(
                "init",
                vec![DemSliceDetector::port(0).with_coords([1.0, 2.0])],
                vec![direct(0.02, [target(0, 1)])],
                DemTemporalHorizon::new(0, 1),
            )
            .unwrap(),
        );
        let next = Arc::new(
            DemSlice::new(
                "next",
                vec![DemSliceDetector::new(0).with_coords([1.0, 2.0])],
                vec![],
                DemTemporalHorizon::new(0, 0),
            )
            .unwrap(),
        );
        let stitched = DemStitcher::new(DemWindowSpec::new(0, 2, 0, DemBoundaryKind::Hard))
            .stitch(&[
                DemSliceInstance::identity(init, 0),
                DemSliceInstance::identity(next, 1),
            ])
            .unwrap();

        assert_eq!(stitched.detector_addresses.len(), 1);
        assert_eq!(stitched.model.detectors[0].coords, Some([1.0, 2.0, 1.0]));
        assert_eq!(
            stitched.model.to_mechanisms().0,
            vec![(0.02, vec![0], vec![])]
        );
    }

    #[test]
    fn source_halo_projects_through_the_hard_backward_boundary() {
        let source_slice = bulk_slice();
        let declaration_slice = Arc::new(
            DemSlice::new(
                "declaration",
                vec![DemSliceDetector::new(0)],
                vec![],
                DemTemporalHorizon::new(0, 0),
            )
            .unwrap(),
        );
        let instances = [
            DemSliceInstance::identity(source_slice, -1),
            DemSliceInstance::identity(declaration_slice, 0),
        ];
        let stitched = DemStitcher::new(DemWindowSpec::new(0, 1, 0, DemBoundaryKind::Hard))
            .stitch(&instances)
            .unwrap();

        assert_eq!(
            stitched.model.to_mechanisms().0,
            vec![(0.01, vec![0], vec![])]
        );
        assert_eq!(stitched.diagnostics.projected_past_contributions, 1);
    }

    #[test]
    fn structured_dem_adapter_preserves_source_components_and_pecos_outputs() {
        let mut source = DetectorErrorModel::new();
        for id in 0..4 {
            source.add_detector(DetectorDef::new(id).with_coords([f64::from(id), 2.0, 11.0]));
        }
        source.add_observable(DemOutput::new(2));
        source.add_tracked_pauli(DemOutput::new(4));
        source.add_source_decomposed_contribution(
            [
                FaultMechanism::from_unsorted_with_tracked_paulis([0, 1], [2], [4]),
                FaultMechanism::from_unsorted([1, 2], []),
                FaultMechanism::from_unsorted([3], []),
            ],
            0.125,
        );

        let source_map = (0..4).fold(
            DemSliceModelMap::new()
                .with_dem_output(2, 0)
                .with_tracked_pauli(4, 1),
            |source_map, detector| {
                source_map.with_detector(detector, RelativeDetectorTarget::new(detector, 0))
            },
        );
        let slice = Arc::new(
            DemSlice::from_detector_error_model(
                "adapted",
                &source,
                &source_map,
                DemTemporalHorizon::new(0, 0),
            )
            .unwrap(),
        );
        let instance = DemSliceInstance::identity(slice, 7)
            .with_dem_output(0, 6)
            .with_tracked_pauli(1, 8);
        let stitched = DemStitcher::new(DemWindowSpec::new(7, 1, 0, DemBoundaryKind::Hard))
            .stitch(&[instance])
            .unwrap();

        assert_eq!(
            stitched.model.to_mechanisms().0,
            vec![(0.125, vec![0, 2, 3], vec![6])]
        );
        let contribution = &stitched.model.contributions()[0];
        assert_eq!(contribution.effect.tracked_paulis.as_slice(), &[8]);
        let components = contribution.source_component_effects().unwrap();
        assert_eq!(components.len(), 3);
        assert_eq!(components[0].detectors.as_slice(), &[0, 1]);
        assert_eq!(components[0].dem_outputs.as_slice(), &[6]);
        assert_eq!(components[0].tracked_paulis.as_slice(), &[8]);
    }

    #[test]
    fn structured_dem_adapter_infers_forward_port_from_relative_map() {
        let mut source = DetectorErrorModel::new();
        source.add_detector(DetectorDef::new(10).with_coords([1.0, 2.0, 0.0]));
        source.add_detector(DetectorDef::new(11).with_coords([1.0, 2.0, 1.0]));
        source.add_direct_contribution(FaultMechanism::from_unsorted([10, 11], []), 0.01);
        let source_map = DemSliceModelMap::new()
            .with_detector(10, target(0, 0))
            .with_detector(11, target(0, 1));

        let slice = Arc::new(
            DemSlice::from_detector_error_model(
                "bulk from DEM",
                &source,
                &source_map,
                DemTemporalHorizon::new(0, 1),
            )
            .unwrap(),
        );
        assert_eq!(
            slice.detectors(),
            &[DemSliceDetector::new(0).with_coords([1.0, 2.0])]
        );

        let instances: Vec<_> = (0..3)
            .map(|round| DemSliceInstance::identity(Arc::clone(&slice), round))
            .collect();
        let stitched = DemStitcher::new(DemWindowSpec::new(0, 2, 1, DemBoundaryKind::Soft))
            .stitch(&instances)
            .unwrap();
        assert_eq!(
            stitched.model.to_mechanisms().0,
            vec![
                (0.01, vec![0, 1], vec![]),
                (0.01, vec![1, 2], vec![]),
                (0.01, vec![2], vec![]),
            ]
        );
    }

    #[test]
    fn dem_builder_output_round_trips_through_a_slice_exactly() {
        use crate::fault_tolerance::dem_builder::{DemBuilder, ParsedDem, compare_dems_exact};
        use pecos_quantum::{Attribute, TickCircuit};

        let mut circuit = TickCircuit::new();
        circuit.tick().px(&[0]);
        let measurement = circuit.tick().mx(&[0]);
        circuit
            .detector(&measurement)
            .expect("the MX reference belongs to the circuit");
        circuit.set_meta("num_measurements", Attribute::String("1".to_string()));
        circuit.set_meta(
            "detectors",
            Attribute::String(r#"[{"id":0,"coords":[2.0,3.0,0.0],"meas_ids":[0]}]"#.to_string()),
        );
        circuit.set_meta("observables", Attribute::String("[]".to_string()));

        let reference = DemBuilder::try_from_tick_circuit(&circuit, 0.01, 0.0, 0.02, 0.03)
            .expect("the physical template builds");
        let owned_locations: BTreeSet<_> = reference
            .contributions()
            .iter()
            .flat_map(|contribution| contribution.location_indices.iter().copied())
            .collect();
        assert!(!owned_locations.is_empty());
        let slice = Arc::new(
            DemSlice::from_detector_error_model_for_locations(
                "physical MX template",
                &reference,
                &DemSliceModelMap::new().with_detector(0, target(0, 0)),
                DemTemporalHorizon::new(0, 0),
                &owned_locations,
            )
            .expect("the structured DEM adapts to a slice"),
        );
        let stitched = DemStitcher::new(DemWindowSpec::new(0, 1, 0, DemBoundaryKind::Hard))
            .stitch(&[DemSliceInstance::identity(slice, 0)])
            .expect("the one-round slice stitches");

        let reference: ParsedDem = reference.to_string().parse().unwrap();
        let stitched: ParsedDem = stitched.model.to_string().parse().unwrap();
        let equivalence = compare_dems_exact(&reference, &stitched, 1e-12);
        assert!(equivalence.equivalent, "{:#?}", equivalence.details);
    }

    #[test]
    fn owned_physical_round_slices_equal_the_full_repeated_measurement_dem() {
        use crate::fault_tolerance::dem_builder::{DemBuilder, ParsedDem, compare_dems_exact};
        use crate::fault_tolerance::propagator::DagFaultAnalyzer;
        use pecos_quantum::DagCircuit;

        let mut circuit = DagCircuit::new();
        circuit.pz(&[0]);
        let mut owned_nodes = Vec::new();
        for _ in 0..3 {
            circuit.pz(&[1]);
            circuit.cx(&[(0, 1)]);
            owned_nodes.push(circuit.last_added_node().unwrap());
            circuit.mz(&[1]);
        }
        let influence_map = DagFaultAnalyzer::new(&circuit).build_influence_map();
        let reference = DemBuilder::new(&influence_map)
            .with_noise(0.0, 0.05, 0.0, 0.0)
            .with_detectors_json(
                r#"[
                    {"id":0,"coords":[2.0,3.0,0.0],"meas_ids":[0]},
                    {"id":1,"coords":[2.0,3.0,1.0],"meas_ids":[0,1]},
                    {"id":2,"coords":[2.0,3.0,2.0],"meas_ids":[1,2]}
                ]"#,
            )
            .unwrap()
            .with_num_measurements(3)
            .try_build()
            .expect("the repeated syndrome-measurement circuit builds");
        let owned_by_round: Vec<BTreeSet<u32>> = owned_nodes
            .iter()
            .map(|owned_node| {
                influence_map
                    .locations
                    .iter()
                    .enumerate()
                    .filter(|(_, location)| location.node == *owned_node)
                    .map(|(index, _)| u32::try_from(index).unwrap())
                    .collect()
            })
            .collect();
        assert!(owned_by_round.iter().all(|locations| !locations.is_empty()));

        let partial_ownership = BTreeSet::from([*reference
            .contributions()
            .iter()
            .find(|contribution| contribution.location_indices.len() > 1)
            .expect("two-qubit CX sources have multiple location indices")
            .location_indices
            .first()
            .unwrap()]);
        let all_at_anchor = (0..3_u32).fold(DemSliceModelMap::new(), |source_map, detector| {
            source_map.with_detector(detector, target(0, i32::try_from(detector).unwrap()))
        });
        let error = DemSlice::from_detector_error_model_for_locations(
            "partially owned CX",
            &reference,
            &all_at_anchor,
            DemTemporalHorizon::new(0, 2),
            &partial_ownership,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            DemSliceStitchError::PartialSourceOwnership { .. }
        ));

        let mut slices = Vec::new();
        for round in 0..3_i32 {
            let source_map = (0..3_u32).fold(DemSliceModelMap::new(), |source_map, detector| {
                source_map.with_detector(
                    detector,
                    target(0, i32::try_from(detector).unwrap() - round),
                )
            });
            slices.push(Arc::new(
                DemSlice::from_detector_error_model_for_locations(
                    format!("measurement round {round}"),
                    &reference,
                    &source_map,
                    DemTemporalHorizon::new(2, 2),
                    &owned_by_round[usize::try_from(round).unwrap()],
                )
                .expect("the owned physical round extracts"),
            ));
        }
        assert!(slices.iter().any(|slice| {
            slice.contributions().iter().any(|contribution| {
                contribution.components().iter().any(|component| {
                    component
                        .detectors
                        .iter()
                        .any(|target| target.round_offset != 0)
                })
            })
        }));

        let instances: Vec<_> = slices
            .into_iter()
            .enumerate()
            .map(|(round, slice)| DemSliceInstance::identity(slice, i64::try_from(round).unwrap()))
            .collect();
        let stitched = DemStitcher::new(DemWindowSpec::new(0, 3, 0, DemBoundaryKind::Hard))
            .stitch(&instances)
            .expect("the physical slices stitch");

        let reference: ParsedDem = reference.to_string().parse().unwrap();
        let stitched: ParsedDem = stitched.model.to_string().parse().unwrap();
        let equivalence = compare_dems_exact(&reference, &stitched, 1e-12);
        assert!(equivalence.equivalent, "{:#?}", equivalence.details);
    }
}
