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

//! Types for Detector Error Model (DEM) generation.
//!
//! This module provides data structures for representing fault mechanisms,
//! detectors, observables, and PECOS tracked Paulis.
//!
//! # Terminology
//!
//! - **Detectors** are syndrome bits defined by measurement-record parity.
//! - **Observables** are values observed through measurements. In a DEM they are
//!   defined by measurement records and rendered as standard `L<n>` observable
//!   outputs.
//! - **Tracked Paulis** are not measured values and are not applied to the
//!   simulated computation. They are Pauli operators annotated at a circuit point
//!   (for example a logical operator, stabilizer, or other Pauli of interest);
//!   PECOS reports whether each fault event anticommutes with, and therefore
//!   would flip, that operator under propagation.
//!
//! PECOS keeps the standard `L<n>` namespace reserved for measurement-record
//! observables only. Tracked Paulis are PECOS metadata with their own
//! ID space, so decoders can ignore them while PECOS tools can still inspect
//! them.
//!
//! # Output Formats
//!
//! The DEM supports two output formats:
//!
//! - [`DetectorErrorModel::to_string()`] - Non-decomposed format. Each
//!   mechanism is output once with its combined probability.
//!
//! - [`DetectorErrorModel::to_string_decomposed()`] - Decomposed format.
//!   Hyperedge errors (3+ detectors) are decomposed into graphlike components,
//!   and 2-detector mechanisms may have multiple representations for decoder
//!   compatibility.
//!
//! Decomposed errors use the `^` separator to indicate XOR composition:
//!
//! ```text
//! error(0.01) D0 D1 ^ D2 D3
//! ```
//!
//! This indicates an error decomposed into two parts whose XOR equals the
//! original mechanism.

use pecos_core::PauliString;
use pecos_core::gate_type::GateType;
use rand::RngExt;
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::hash::{Hash, Hasher};

use crate::fault_tolerance::propagator::{DemOutputKind, DemOutputMetadata, Pauli};

// ============================================================================
// Error Source Tracking
// ============================================================================

/// Classification of error sources for decomposition decisions.
///
/// This tracks how an error contribution was generated, which determines
/// how it should be output in the decomposed DEM format:
/// - Direct errors (X, Z channels) -> output as direct form
/// - Direct one-sided component errors -> output as direct form for now, but
///   keep their source family distinct for later decomposition policy work
/// - Y-decomposed errors -> output as decomposed form (X ^ Z)
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FaultSourceType {
    /// Direct X or Z error channel - outputs as direct form only.
    /// These represent single Pauli errors that cannot be further decomposed.
    Direct,

    /// Direct two-location source where exactly one per-location component equals
    /// the full effect and the other component is empty.
    ///
    /// These rows currently render the same as `Direct`, but the subtype is kept
    /// so decomposition policy can distinguish them later without reconstructing
    /// the family from builder-time component metadata.
    DirectOneSidedComponent,

    /// Y error decomposed as X^Z - outputs as decomposed form.
    /// The X and Z component effects are stored for decomposition output.
    YDecomposed {
        /// Detector effect of the X component (sorted detector IDs).
        x_detectors: SmallVec<[u32; 4]>,
        /// DEM-output effect of the X component (sorted `L<n>` IDs).
        x_dem_outputs: SmallVec<[u32; 2]>,
        /// Detector effect of the Z component (sorted detector IDs).
        z_detectors: SmallVec<[u32; 4]>,
        /// DEM-output effect of the Z component (sorted `L<n>` IDs).
        z_dem_outputs: SmallVec<[u32; 2]>,
    },
}

/// Coarse source-family classification for direct contributions.
///
/// This is intentionally descriptive instead of prescriptive: it keeps the main
/// direct source families separate for downstream analysis without changing
/// rendered DEM behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DirectSourceFamily {
    /// Single-location direct source without a Y Pauli label.
    SingleLocation,

    /// Single-location direct source with a Y Pauli label.
    SingleLocationY,

    /// Two-location direct source routed from a Y-containing channel.
    TwoLocationPlainY,

    /// Two-location direct source with recorded per-location components.
    TwoLocationComponent,

    /// Two-location direct source where exactly one component is non-empty.
    TwoLocationOneSidedComponent,

    /// Fallback for other direct-source shapes.
    Other,
}

/// An error contribution with source tracking.
///
/// This represents a single error source's contribution to the DEM,
/// tracking both its effect and how it was generated. Multiple contributions
/// with the same effect are grouped at output time, with their source types
/// determining how they are output (direct vs decomposed forms).
#[derive(Debug, Clone)]
pub struct FaultContribution {
    /// The detector/DEM-output effect of this error.
    pub effect: FaultMechanism,

    /// Probability of this error.
    pub probability: f64,

    /// Source classification for decomposition decisions.
    pub source_type: FaultSourceType,

    /// Fault location indices in the influence map that produced this contribution.
    pub location_indices: SmallVec<[u32; 2]>,

    /// Original Pauli channel at each tracked location.
    pub paulis: SmallVec<[Pauli; 2]>,

    /// Gate type at each tracked source location.
    pub source_gate_types: SmallVec<[GateType; 2]>,

    /// Whether each tracked source location is before (`true`) or after (`false`) its gate.
    pub source_before_flags: SmallVec<[bool; 2]>,

    /// Coarse direct-source family for read-only analysis.
    pub direct_source_family: Option<DirectSourceFamily>,

    /// Optional per-location component effects for direct multi-location sources.
    ///
    /// These are builder-time component effects whose XOR equals `effect`. They are
    /// currently recorded for direct two-qubit channel sources to aid decomposition
    /// analysis without changing emitted DEM behavior.
    pub direct_component_effects: Option<(FaultMechanism, FaultMechanism)>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct SourceMetadata<'a, Index> {
    location_indices: &'a [Index],
    paulis: &'a [Pauli],
    gate_types: &'a [GateType],
    before_flags: &'a [bool],
}

impl<'a, Index> SourceMetadata<'a, Index> {
    pub(crate) const fn new(
        location_indices: &'a [Index],
        paulis: &'a [Pauli],
        gate_types: &'a [GateType],
        before_flags: &'a [bool],
    ) -> Self {
        Self {
            location_indices,
            paulis,
            gate_types,
            before_flags,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DirectSourceComponents<'a> {
    first: &'a FaultMechanism,
    second: &'a FaultMechanism,
}

impl<'a> DirectSourceComponents<'a> {
    pub(crate) const fn new(first: &'a FaultMechanism, second: &'a FaultMechanism) -> Self {
        Self { first, second }
    }
}

impl FaultContribution {
    fn classify_direct_source_family(
        location_indices: &[u32],
        paulis: &[Pauli],
        direct_component_effects: Option<(&FaultMechanism, &FaultMechanism)>,
    ) -> Option<DirectSourceFamily> {
        if location_indices.is_empty() {
            return None;
        }

        let has_y = paulis.contains(&Pauli::Y);

        match location_indices.len() {
            1 => Some(if has_y {
                DirectSourceFamily::SingleLocationY
            } else {
                DirectSourceFamily::SingleLocation
            }),
            2 => {
                if let Some((first, second)) = direct_component_effects {
                    if first.is_empty() ^ second.is_empty() {
                        Some(DirectSourceFamily::TwoLocationOneSidedComponent)
                    } else {
                        Some(DirectSourceFamily::TwoLocationComponent)
                    }
                } else if has_y {
                    Some(DirectSourceFamily::TwoLocationPlainY)
                } else {
                    Some(DirectSourceFamily::Other)
                }
            }
            _ => Some(DirectSourceFamily::Other),
        }
    }

    /// Creates a new direct error contribution (X or Z channel).
    #[must_use]
    pub fn direct(effect: FaultMechanism, probability: f64) -> Self {
        Self {
            effect,
            probability,
            source_type: FaultSourceType::Direct,
            location_indices: SmallVec::new(),
            paulis: SmallVec::new(),
            source_gate_types: SmallVec::new(),
            source_before_flags: SmallVec::new(),
            direct_source_family: None,
            direct_component_effects: None,
        }
    }

    /// Creates a new direct error contribution with source metadata.
    #[must_use]
    fn direct_with_source(
        effect: FaultMechanism,
        probability: f64,
        source: SourceMetadata<'_, u32>,
    ) -> Self {
        debug_assert_eq!(source.location_indices.len(), source.paulis.len());
        debug_assert_eq!(source.location_indices.len(), source.gate_types.len());
        debug_assert_eq!(source.location_indices.len(), source.before_flags.len());
        Self {
            effect,
            probability,
            source_type: FaultSourceType::Direct,
            location_indices: source.location_indices.iter().copied().collect(),
            paulis: source.paulis.iter().copied().collect(),
            source_gate_types: source.gate_types.iter().copied().collect(),
            source_before_flags: source.before_flags.iter().copied().collect(),
            direct_source_family: Self::classify_direct_source_family(
                source.location_indices,
                source.paulis,
                None,
            ),
            direct_component_effects: None,
        }
    }

    /// Creates a new direct error contribution with source metadata and
    /// per-location component effects.
    #[must_use]
    fn direct_with_source_components(
        effect: FaultMechanism,
        probability: f64,
        source: SourceMetadata<'_, u32>,
        components: DirectSourceComponents<'_>,
    ) -> Self {
        debug_assert_eq!(source.location_indices.len(), source.paulis.len());
        debug_assert_eq!(source.location_indices.len(), source.gate_types.len());
        debug_assert_eq!(source.location_indices.len(), source.before_flags.len());
        let source_type = if (components.first == &effect && components.second.is_empty())
            || (components.second == &effect && components.first.is_empty())
        {
            FaultSourceType::DirectOneSidedComponent
        } else {
            FaultSourceType::Direct
        };
        Self {
            effect,
            probability,
            source_type,
            location_indices: source.location_indices.iter().copied().collect(),
            paulis: source.paulis.iter().copied().collect(),
            source_gate_types: source.gate_types.iter().copied().collect(),
            source_before_flags: source.before_flags.iter().copied().collect(),
            direct_source_family: Self::classify_direct_source_family(
                source.location_indices,
                source.paulis,
                Some((components.first, components.second)),
            ),
            direct_component_effects: Some((components.first.clone(), components.second.clone())),
        }
    }

    /// Creates a new Y-decomposed error contribution.
    ///
    /// The combined effect is stored along with the X and Z component effects,
    /// allowing the decomposed form (X ^ Z) to be output.
    #[must_use]
    pub fn y_decomposed(
        combined_effect: FaultMechanism,
        x_effect: &FaultMechanism,
        z_effect: &FaultMechanism,
        probability: f64,
    ) -> Self {
        Self {
            effect: combined_effect,
            probability,
            source_type: FaultSourceType::YDecomposed {
                x_detectors: x_effect.detectors.clone(),
                x_dem_outputs: x_effect.dem_outputs.clone(),
                z_detectors: z_effect.detectors.clone(),
                z_dem_outputs: z_effect.dem_outputs.clone(),
            },
            location_indices: SmallVec::new(),
            paulis: SmallVec::new(),
            source_gate_types: SmallVec::new(),
            source_before_flags: SmallVec::new(),
            direct_source_family: None,
            direct_component_effects: None,
        }
    }

    /// Creates a new Y-decomposed error contribution with source metadata.
    #[must_use]
    fn y_decomposed_with_source(
        combined_effect: FaultMechanism,
        x_effect: &FaultMechanism,
        z_effect: &FaultMechanism,
        probability: f64,
        source: SourceMetadata<'_, u32>,
    ) -> Self {
        debug_assert_eq!(source.location_indices.len(), source.paulis.len());
        debug_assert_eq!(source.location_indices.len(), source.gate_types.len());
        debug_assert_eq!(source.location_indices.len(), source.before_flags.len());
        Self {
            effect: combined_effect,
            probability,
            source_type: FaultSourceType::YDecomposed {
                x_detectors: x_effect.detectors.clone(),
                x_dem_outputs: x_effect.dem_outputs.clone(),
                z_detectors: z_effect.detectors.clone(),
                z_dem_outputs: z_effect.dem_outputs.clone(),
            },
            location_indices: source.location_indices.iter().copied().collect(),
            paulis: source.paulis.iter().copied().collect(),
            source_gate_types: source.gate_types.iter().copied().collect(),
            source_before_flags: source.before_flags.iter().copied().collect(),
            direct_source_family: None,
            direct_component_effects: None,
        }
    }

    /// Returns true if this is a direct (non-decomposable) source.
    #[must_use]
    pub fn is_direct(&self) -> bool {
        matches!(
            self.source_type,
            FaultSourceType::Direct | FaultSourceType::DirectOneSidedComponent
        )
    }

    /// Returns the X and Z components if this is a Y-decomposed source.
    #[must_use]
    pub fn decomposition_components(&self) -> Option<(FaultMechanism, FaultMechanism)> {
        match &self.source_type {
            FaultSourceType::YDecomposed {
                x_detectors,
                x_dem_outputs,
                z_detectors,
                z_dem_outputs,
            } => {
                let x = FaultMechanism::from_sorted(x_detectors.clone(), x_dem_outputs.clone());
                let z = FaultMechanism::from_sorted(z_detectors.clone(), z_dem_outputs.clone());
                Some((x, z))
            }
            FaultSourceType::Direct | FaultSourceType::DirectOneSidedComponent => None,
        }
    }

    /// Returns the per-location component effects for a direct multi-location source.
    #[must_use]
    pub fn direct_component_effects(&self) -> Option<(FaultMechanism, FaultMechanism)> {
        self.direct_component_effects.clone()
    }
}

/// Aggregated source-tracked information for one unique effect.
#[derive(Debug, Clone)]
pub struct ContributionEffectSummary {
    /// The detector/DEM-output effect being summarized.
    pub effect: FaultMechanism,
    /// Total number of contributing sources for this effect.
    pub num_contributions: usize,
    /// Total probability summed over contributing sources.
    pub total_probability: f64,
    /// Number of direct contributions.
    pub direct_count: usize,
    /// Total probability from direct contributions.
    pub direct_probability: f64,
    /// Number of Y-decomposed contributions.
    pub y_decomposed_count: usize,
    /// Total probability from Y-decomposed contributions.
    pub y_decomposed_probability: f64,
    /// Number of builder-marked graphlike-decomposable two-qubit sources for this effect.
    ///
    /// This is only non-zero for 2-detector, 0-DEM-output effects. It reflects the
    /// dormant representation-diversity bookkeeping recorded by the DEM builder.
    pub graphlike_decomposable_count: u32,
}

/// Structured summary of how tracked contributions render before final regrouping.
#[derive(Debug, Clone)]
pub struct ContributionRenderSummary {
    /// Original full detector/DEM-output effect before rendering.
    pub effect: FaultMechanism,
    /// Rendered targets string that this contribution group maps to.
    pub rendered_targets: String,
    /// Number of tracked contributions in this pre-regroup bucket.
    pub num_contributions: usize,
    /// Total probability in this pre-regroup bucket.
    pub total_probability: f64,
    /// Probability after combining same-target contributions within this bucket.
    pub combined_probability: f64,
    /// Counts of source types in this bucket.
    pub source_type_counts: BTreeMap<String, usize>,
    /// Probability totals of source types in this bucket.
    pub source_type_probabilities: BTreeMap<String, f64>,
    /// Counts of direct source families in this bucket.
    pub direct_source_family_counts: BTreeMap<String, usize>,
    /// Probability totals of direct source families in this bucket.
    pub direct_source_family_probabilities: BTreeMap<String, f64>,
}

/// Per-contribution render record before final regrouping.
///
/// This keeps the exact rendered target string attached to one tracked
/// contribution, without aggregating inside the pre-regroup bucket. It is a
/// lower-level view than [`ContributionRenderSummary`] and is useful for
/// inspecting within-effect render policies.
#[derive(Debug, Clone)]
pub struct ContributionRenderRecord {
    /// Rendered targets string that this contribution maps to.
    pub rendered_targets: String,
    /// Coarse render strategy used for this contribution.
    pub render_strategy: ContributionRenderStrategy,
    /// Optional targets implied by recorded direct component effects.
    ///
    /// This is descriptive only: it does not imply the current render pass uses
    /// these targets.
    pub recorded_component_targets: Option<String>,
    /// Original tracked contribution before regrouping.
    pub contribution: FaultContribution,
}

/// Coarse render strategy used for one contribution in the decomposed DEM pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContributionRenderStrategy {
    /// Used source-specific decomposition components.
    SourceComponents,
    /// Used recorded direct-source component targets instead of the direct edge.
    RecordedComponents,
    /// Kept a 2-detector, 0-DEM-output effect graphlike as-is.
    TwoDetectorDirect,
    /// Decomposed a hyperedge using graphlike effect decomposition.
    HyperedgeGraphlike,
    /// Rendered directly from the full effect.
    EffectDirect,
}

/// Policy for rendering direct 2-detector effects in decomposed DEM output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TwoDetectorDirectRenderPolicy {
    /// Preserve the current direct-edge rendering.
    KeepDirect,
    /// Prefer recorded builder-time component targets when they differ from the
    /// direct edge. This is intended for source-aware render experiments.
    PreferRecordedComponents,
}

// ============================================================================
// Error Mechanism
// ============================================================================

/// A fault mechanism: a set of detectors and `L<n>` targets that flip together.
///
/// When an error occurs, it flips a specific set of detectors and may flip
/// `L<n>` targets. Mechanisms with the same effect are aggregated together.
///
/// Detector and `L<n>` target indices are stored in sorted order for canonical representation.
#[derive(Clone, Default)]
pub struct FaultMechanism {
    /// Detector indices that flip together (sorted).
    pub detectors: SmallVec<[u32; 4]>,
    /// DEM `L<n>` target indices that flip together (sorted).
    ///
    /// New code should treat these as standard observable `L<n>` output channels.
    pub dem_outputs: SmallVec<[u32; 2]>,
    /// PECOS tracked-Pauli indices that flip together (sorted).
    ///
    /// These are rendered as `TP<n>` only in PECOS DEM text. Standard DEM text
    /// and decoder-facing mechanism tables intentionally ignore them.
    pub tracked_paulis: SmallVec<[u32; 2]>,
}

impl FaultMechanism {
    /// Creates a new empty fault mechanism.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a mechanism from unsorted detector and DEM-output indices.
    #[must_use]
    pub fn from_unsorted(
        detectors: impl IntoIterator<Item = u32>,
        dem_outputs: impl IntoIterator<Item = u32>,
    ) -> Self {
        Self::from_unsorted_with_tracked_paulis(detectors, dem_outputs, std::iter::empty())
    }

    /// Creates a mechanism from unsorted detector, DEM-output, and tracked-Pauli indices.
    #[must_use]
    pub fn from_unsorted_with_tracked_paulis(
        detectors: impl IntoIterator<Item = u32>,
        dem_outputs: impl IntoIterator<Item = u32>,
        tracked_paulis: impl IntoIterator<Item = u32>,
    ) -> Self {
        let mut dets: SmallVec<[u32; 4]> = detectors.into_iter().collect();
        let mut dem_outputs: SmallVec<[u32; 2]> = dem_outputs.into_iter().collect();
        let mut tracked_paulis: SmallVec<[u32; 2]> = tracked_paulis.into_iter().collect();
        dets.sort_unstable();
        dem_outputs.sort_unstable();
        tracked_paulis.sort_unstable();
        Self {
            detectors: dets,
            dem_outputs,
            tracked_paulis,
        }
    }

    /// Creates a mechanism from pre-sorted detector and DEM-output indices.
    #[must_use]
    pub fn from_sorted(detectors: SmallVec<[u32; 4]>, dem_outputs: SmallVec<[u32; 2]>) -> Self {
        Self::from_sorted_with_tracked_paulis(detectors, dem_outputs, SmallVec::new())
    }

    /// Creates a mechanism from pre-sorted detector, DEM-output, and tracked-Pauli indices.
    #[must_use]
    pub fn from_sorted_with_tracked_paulis(
        detectors: SmallVec<[u32; 4]>,
        dem_outputs: SmallVec<[u32; 2]>,
        tracked_paulis: SmallVec<[u32; 2]>,
    ) -> Self {
        debug_assert!(
            detectors.windows(2).all(|w| w[0] <= w[1]),
            "detectors must be sorted"
        );
        debug_assert!(
            dem_outputs.windows(2).all(|w| w[0] <= w[1]),
            "dem_outputs must be sorted"
        );
        debug_assert!(
            tracked_paulis.windows(2).all(|w| w[0] <= w[1]),
            "tracked_paulis must be sorted"
        );
        Self {
            detectors,
            dem_outputs,
            tracked_paulis,
        }
    }

    /// Returns true if this mechanism has no effect (empty).
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.detectors.is_empty() && self.dem_outputs.is_empty() && self.tracked_paulis.is_empty()
    }

    /// Returns true if this mechanism has no decoder-facing effect.
    ///
    /// This ignores PECOS tracked-Pauli effects, matching standard DEM and
    /// decoder-facing sampler behavior.
    #[inline]
    #[must_use]
    pub fn is_standard_empty(&self) -> bool {
        self.detectors.is_empty() && self.dem_outputs.is_empty()
    }

    /// Returns the decoder-facing projection of this mechanism.
    #[must_use]
    pub fn standard_effect(&self) -> Self {
        Self {
            detectors: self.detectors.clone(),
            dem_outputs: self.dem_outputs.clone(),
            tracked_paulis: SmallVec::new(),
        }
    }

    /// Returns the number of detectors in this mechanism.
    #[inline]
    #[must_use]
    pub fn num_detectors(&self) -> usize {
        self.detectors.len()
    }

    /// Returns the number of outputs in the DEM `L<n>` namespace.
    #[inline]
    #[must_use]
    pub fn num_dem_outputs(&self) -> usize {
        self.dem_outputs.len()
    }

    /// Returns the number of tracked Pauli outputs in this mechanism.
    #[inline]
    #[must_use]
    pub fn num_tracked_paulis(&self) -> usize {
        self.tracked_paulis.len()
    }

    /// XOR this mechanism with another, returning the combined effect.
    ///
    /// Used when combining correlated errors (e.g., two-qubit gate errors).
    #[must_use]
    pub fn xor(&self, other: &Self) -> Self {
        Self {
            detectors: symmetric_difference_4(&self.detectors, &other.detectors),
            dem_outputs: symmetric_difference_2(&self.dem_outputs, &other.dem_outputs),
            tracked_paulis: symmetric_difference_2(&self.tracked_paulis, &other.tracked_paulis),
        }
    }

    /// Returns true if this mechanism is graphlike.
    ///
    /// A graphlike mechanism has at most 2 detectors.
    /// DEM outputs do not affect graph-likeness; MWPM decoders attach them as
    /// frame-change masks on graph edges.
    #[inline]
    #[must_use]
    pub fn is_graphlike(&self) -> bool {
        self.detectors.len() <= 2
    }

    /// Returns true if this mechanism is a hyperedge (not graphlike).
    ///
    /// Hyperedge mechanisms have 3+ detectors and need to be decomposed into
    /// graphlike components for MWPM decoders.
    #[inline]
    #[must_use]
    pub fn is_hyperedge(&self) -> bool {
        self.detectors.len() > 2
    }
}

/// Computes symmetric difference of two sorted slices (4-element variant).
fn symmetric_difference_4(a: &SmallVec<[u32; 4]>, b: &SmallVec<[u32; 4]>) -> SmallVec<[u32; 4]> {
    let mut result = SmallVec::new();
    let mut i = 0;
    let mut j = 0;

    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => {
                result.push(a[i]);
                i += 1;
            }
            Ordering::Greater => {
                result.push(b[j]);
                j += 1;
            }
            Ordering::Equal => {
                // Same element in both - XOR cancels
                i += 1;
                j += 1;
            }
        }
    }

    result.extend_from_slice(&a[i..]);
    result.extend_from_slice(&b[j..]);
    result
}

/// Computes symmetric difference of two sorted slices (2-element variant).
fn symmetric_difference_2(a: &SmallVec<[u32; 2]>, b: &SmallVec<[u32; 2]>) -> SmallVec<[u32; 2]> {
    let mut result = SmallVec::new();
    let mut i = 0;
    let mut j = 0;

    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => {
                result.push(a[i]);
                i += 1;
            }
            Ordering::Greater => {
                result.push(b[j]);
                j += 1;
            }
            Ordering::Equal => {
                // Same element in both - XOR cancels
                i += 1;
                j += 1;
            }
        }
    }

    result.extend_from_slice(&a[i..]);
    result.extend_from_slice(&b[j..]);
    result
}

impl PartialEq for FaultMechanism {
    fn eq(&self, other: &Self) -> bool {
        self.detectors == other.detectors
            && self.dem_outputs == other.dem_outputs
            && self.tracked_paulis == other.tracked_paulis
    }
}

impl Eq for FaultMechanism {}

impl Hash for FaultMechanism {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.detectors.hash(state);
        self.dem_outputs.hash(state);
        self.tracked_paulis.hash(state);
    }
}

impl PartialOrd for FaultMechanism {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FaultMechanism {
    fn cmp(&self, other: &Self) -> Ordering {
        self.detectors
            .cmp(&other.detectors)
            .then_with(|| self.dem_outputs.cmp(&other.dem_outputs))
            .then_with(|| self.tracked_paulis.cmp(&other.tracked_paulis))
    }
}

impl fmt::Debug for FaultMechanism {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FaultMechanism(dets={:?}, dem_outputs={:?}, tracked_paulis={:?})",
            self.detectors.as_slice(),
            self.dem_outputs.as_slice(),
            self.tracked_paulis.as_slice()
        )
    }
}

// ============================================================================
// Decomposed Error
// ============================================================================

/// A decomposed fault mechanism with optional decomposition into graphlike parts.
///
/// When an error affects 3+ detectors (a hyperedge), it can be decomposed into
/// a combination of graphlike errors (affecting 1-2 detectors each) connected
/// by `^` separators indicating XOR composition.
#[derive(Clone, Debug)]
pub struct DecomposedFault {
    /// The component fault mechanisms (separated by `^` in DEM format).
    /// For graphlike errors, this has a single element.
    /// For decomposed hyperedges, this has multiple elements.
    pub components: SmallVec<[FaultMechanism; 2]>,
}

impl DecomposedFault {
    /// Creates a new decomposed error from a single mechanism.
    #[must_use]
    pub fn single(mechanism: FaultMechanism) -> Self {
        let mut components = SmallVec::new();
        components.push(mechanism);
        Self { components }
    }

    /// Creates a decomposed error from multiple components.
    #[must_use]
    pub fn decomposed(components: impl IntoIterator<Item = FaultMechanism>) -> Self {
        Self {
            components: components.into_iter().collect(),
        }
    }

    /// Returns the full effect of this error (XOR of all components).
    #[must_use]
    pub fn full_effect(&self) -> FaultMechanism {
        let mut result = FaultMechanism::new();
        for component in &self.components {
            result = result.xor(component);
        }
        result
    }

    /// Returns true if this error is graphlike (affects at most 2 detectors per component).
    #[must_use]
    pub fn is_graphlike(&self) -> bool {
        self.components.iter().all(|c| c.num_detectors() <= 2)
    }

    /// Formats this error for DEM output.
    #[must_use]
    pub fn to_stim_targets(&self) -> String {
        self.components
            .iter()
            .map(format_mechanism_targets)
            .collect::<Vec<_>>()
            .join(" ^ ")
    }

    /// Formats this error for PECOS DEM output, including tracked Pauli `TP<n>` targets.
    #[must_use]
    pub fn to_pecos_targets(&self) -> String {
        self.components
            .iter()
            .map(format_pecos_mechanism_targets)
            .collect::<Vec<_>>()
            .join(" ^ ")
    }
}

// ============================================================================
// Hyperedge Decomposition
// ============================================================================

/// Finds all valid graphlike decompositions of a hyperedge mechanism.
///
/// A hyperedge is an fault mechanism with 3+ detectors.
/// For MWPM decoders, hyperedges must be decomposed into XOR combinations
/// of graphlike components (≤2 detectors, ≤1 logical each).
///
/// # Arguments
///
/// * `hyperedge` - The mechanism to decompose (must be a hyperedge)
/// * `graphlike_set` - Set of available graphlike mechanisms to use as components
///
/// # Returns
///
/// A graphlike decomposition whose XOR equals the original hyperedge.
/// Returns `None` if no valid decomposition exists.
///
/// # Algorithm
///
/// Uses a detector-driven recursive search over graphlike components whose
/// detector sets are subsets of the hyperedge. This is more general than the
/// older fixed-width 2-part/3-part search, and it allows decompositions into 4+
/// graphlike pieces when needed.
///
/// Decompositions are filtered to only include components whose detectors are
/// subsets of the original hyperedge's detectors, so decomposition does not
/// introduce extra detector symptoms.
///
/// # Selection
///
/// The search returns the first valid decomposition found using a deterministic
/// ordering that prefers detector pairs before singlets.
#[cfg(test)]
fn find_hyperedge_decomposition(
    hyperedge: &FaultMechanism,
    graphlike_set: &BTreeSet<FaultMechanism>,
) -> Option<Vec<FaultMechanism>> {
    GraphlikeDecompositionIndex::new(graphlike_set).find_hyperedge_decomposition(hyperedge)
}

struct GraphlikeDecompositionIndex {
    graphlike_set: BTreeSet<FaultMechanism>,
    /// Indexed by detector ID; see `SingletonDecompositionIndex` for the same
    /// pattern and rationale. Detector IDs are dense `0..num_detectors`.
    candidates_by_detector: Vec<Vec<FaultMechanism>>,
}

impl GraphlikeDecompositionIndex {
    fn new(graphlike_set: &BTreeSet<FaultMechanism>) -> Self {
        let max_det = graphlike_set
            .iter()
            .flat_map(|c| c.detectors.iter().copied())
            .max();

        let mut candidates_by_detector: Vec<Vec<FaultMechanism>> =
            max_det.map_or_else(Vec::new, |m| vec![Vec::new(); m as usize + 1]);

        for candidate in graphlike_set {
            for &det in &candidate.detectors {
                candidates_by_detector[det as usize].push(candidate.clone());
            }
        }
        for values in &mut candidates_by_detector {
            values.sort_by(|a, b| {
                b.detectors
                    .len()
                    .cmp(&a.detectors.len())
                    .then_with(|| a.cmp(b))
            });
        }
        Self {
            graphlike_set: graphlike_set.clone(),
            candidates_by_detector,
        }
    }

    fn find_hyperedge_decomposition(
        &self,
        hyperedge: &FaultMechanism,
    ) -> Option<Vec<FaultMechanism>> {
        // If already graphlike, no decomposition needed
        if hyperedge.is_graphlike() {
            return Some(vec![hyperedge.clone()]);
        }

        // Collect the set of detectors in the hyperedge
        let hyperedge_dets: BTreeSet<u32> = hyperedge.detectors.iter().copied().collect();

        let decomp_dets_valid = |decomp: &[FaultMechanism]| -> bool {
            decomp
                .iter()
                .flat_map(|m| m.detectors.iter())
                .all(|d| hyperedge_dets.contains(d))
        };

        let mut memo = BTreeMap::new();
        let result = self.search_decomposition(hyperedge, &mut memo);
        result.filter(|decomp| decomp_dets_valid(decomp))
    }

    fn search_decomposition(
        &self,
        remaining: &FaultMechanism,
        memo: &mut BTreeMap<FaultMechanism, Option<Vec<FaultMechanism>>>,
    ) -> Option<Vec<FaultMechanism>> {
        if let Some(cached) = memo.get(remaining) {
            return cached.clone();
        }

        if remaining.is_empty() {
            let result = Some(Vec::new());
            memo.insert(remaining.clone(), result.clone());
            return result;
        }

        if remaining.is_graphlike() && self.graphlike_set.contains(remaining) {
            let result = Some(vec![remaining.clone()]);
            memo.insert(remaining.clone(), result.clone());
            return result;
        }

        if let Some(&pivot) = remaining.detectors.first()
            && let Some(candidates) = self.candidates_by_detector.get(pivot as usize)
        {
            for candidate in candidates {
                if !candidate
                    .detectors
                    .iter()
                    .all(|d| remaining.detectors.contains(d))
                {
                    continue;
                }
                if !shares_element(candidate, remaining) {
                    continue;
                }

                let next = remaining.xor(candidate);

                // Require strict detector-count progress to prevent cycles.
                if next.detectors.len() >= remaining.detectors.len() {
                    continue;
                }

                if let Some(suffix) = self.search_decomposition(&next, memo) {
                    let mut combined = Vec::with_capacity(suffix.len() + 1);
                    combined.push(candidate.clone());
                    combined.extend(suffix);
                    combined.sort();
                    let result = Some(combined);
                    memo.insert(remaining.clone(), result.clone());
                    return result;
                }
            }
        }

        memo.insert(remaining.clone(), None);
        None
    }
}

/// Finds a decomposition of a graphlike effect into singleton detector components.
///
/// This is used for "maximal" decomposition modes that prefer singleton
/// detector symptoms whenever the required singleton effects already exist as
/// standalone mechanisms in the DEM.
fn find_singleton_decomposition(
    effect: &FaultMechanism,
    index: &SingletonDecompositionIndex,
) -> Option<Vec<FaultMechanism>> {
    if effect.is_empty() {
        return Some(Vec::new());
    }
    if effect.num_detectors() <= 1 {
        return Some(vec![effect.clone()]);
    }
    if index.is_empty() {
        return None;
    }

    let mut memo: BTreeMap<FaultMechanism, Option<Vec<FaultMechanism>>> = BTreeMap::new();
    search_singleton_decomposition(effect, &index.candidates_by_detector, &mut memo)
}

/// Pre-computed bucket of singleton (1-detector) mechanisms indexed by detector ID.
///
/// Built once per render pass; detector IDs are dense `0..num_detectors`, so a
/// `Vec<Vec<_>>` indexed by detector ID beats a `BTreeMap<u32, Vec<_>>` both on
/// lookup (O(1) vs O(log n)) and on per-call rebuild cost. Profiling flagged the
/// rebuild-per-call pattern as ~28% of `to_string_decomposed_maximally` time
/// before this was lifted out.
struct SingletonDecompositionIndex {
    /// `candidates_by_detector[det]` lists every singleton mechanism whose sole
    /// detector is `det`, sorted by `(dem_outputs.len, dem_outputs, detectors)` so the
    /// decomposition search prefers simpler candidates deterministically.
    candidates_by_detector: Vec<Vec<FaultMechanism>>,
}

impl SingletonDecompositionIndex {
    fn new() -> Self {
        Self {
            candidates_by_detector: Vec::new(),
        }
    }

    fn from_contributions(contributions: &[FaultContribution]) -> Self {
        let mut singletons: BTreeSet<FaultMechanism> = BTreeSet::new();
        for contrib in contributions {
            if contrib.effect.num_detectors() == 1 {
                singletons.insert(contrib.effect.clone());
            }
        }

        let Some(max_det) = singletons.iter().map(|c| c.detectors[0]).max() else {
            return Self::new();
        };

        let mut candidates_by_detector: Vec<Vec<FaultMechanism>> =
            vec![Vec::new(); max_det as usize + 1];
        for candidate in singletons {
            let det = candidate.detectors[0] as usize;
            candidates_by_detector[det].push(candidate);
        }
        for candidates in &mut candidates_by_detector {
            candidates.sort_by(|a, b| {
                a.dem_outputs
                    .len()
                    .cmp(&b.dem_outputs.len())
                    .then_with(|| a.dem_outputs.cmp(&b.dem_outputs))
                    .then_with(|| a.detectors.cmp(&b.detectors))
            });
        }
        Self {
            candidates_by_detector,
        }
    }

    fn is_empty(&self) -> bool {
        self.candidates_by_detector.is_empty()
    }
}

fn search_singleton_decomposition(
    remaining: &FaultMechanism,
    candidates_by_detector: &[Vec<FaultMechanism>],
    memo: &mut BTreeMap<FaultMechanism, Option<Vec<FaultMechanism>>>,
) -> Option<Vec<FaultMechanism>> {
    if let Some(cached) = memo.get(remaining) {
        return cached.clone();
    }
    if remaining.is_empty() {
        return Some(Vec::new());
    }

    let Some(&first_det) = remaining.detectors.first() else {
        memo.insert(remaining.clone(), None);
        return None;
    };

    let result = candidates_by_detector
        .get(first_det as usize)
        .and_then(|candidates| {
            for candidate in candidates {
                let next = remaining.xor(candidate);
                if next.num_detectors() >= remaining.num_detectors() {
                    continue;
                }
                if let Some(mut tail) =
                    search_singleton_decomposition(&next, candidates_by_detector, memo)
                {
                    let mut parts = Vec::with_capacity(tail.len() + 1);
                    parts.push(candidate.clone());
                    parts.append(&mut tail);
                    return Some(parts);
                }
            }
            None
        });

    memo.insert(remaining.clone(), result.clone());
    result
}

/// Checks if two mechanisms share at least one detector or logical.
fn shares_element(a: &FaultMechanism, b: &FaultMechanism) -> bool {
    // Check detectors
    for d in &a.detectors {
        if b.detectors.contains(d) {
            return true;
        }
    }
    // Check dem_outputs
    for l in &a.dem_outputs {
        if b.dem_outputs.contains(l) {
            return true;
        }
    }
    false
}

fn convert_location_indices(location_indices: &[usize]) -> SmallVec<[u32; 2]> {
    location_indices
        .iter()
        .map(|&idx| u32::try_from(idx).expect("fault location index must fit into u32"))
        .collect()
}

/// Converts a DEM measurement-record offset to an absolute measurement index.
///
/// Negative offsets count backward from the end of the measurement record
/// (`-1` is the last measurement). Positive offsets are treated as absolute
/// indices.
///
/// Returns `None` whenever the resulting index would land outside
/// `0..num_measurements`. Callers should treat a `None` as a malformed input
/// (parser/user-supplied offset was too large or too negative); it is never
/// produced by internally-generated offsets, so silently dropping such a
/// contribution rather than panicking is the intended behavior.
#[must_use]
pub fn record_offset_to_absolute_index(num_measurements: usize, offset: i32) -> Option<usize> {
    if offset < 0 {
        num_measurements.checked_add_signed(isize::try_from(offset).ok()?)
    } else {
        usize::try_from(offset).ok()
    }
}

// ============================================================================
// Detector Definition
// ============================================================================

/// A detector definition with coordinates and measurement records.
///
/// Detectors are defined as the XOR of one or more measurement outcomes.
/// When all measurements are correct, the detector value is deterministic.
#[derive(Debug, Clone)]
pub struct DetectorDef {
    /// Unique detector ID.
    pub id: u32,
    /// Optional 3D coordinates (x, y, t) for visualization.
    pub coords: Option<[f64; 3]>,
    /// Measurement record offsets (negative indices from end of record).
    pub records: SmallVec<[i32; 2]>,
}

impl DetectorDef {
    /// Creates a new detector definition.
    #[must_use]
    pub fn new(id: u32) -> Self {
        Self {
            id,
            coords: None,
            records: SmallVec::new(),
        }
    }

    /// Sets the detector coordinates.
    #[must_use]
    pub fn with_coords(mut self, coords: [f64; 3]) -> Self {
        self.coords = Some(coords);
        self
    }

    /// Adds a measurement record offset.
    pub fn add_record(&mut self, record: i32) {
        self.records.push(record);
    }

    /// Sets the measurement records.
    #[must_use]
    pub fn with_records(mut self, records: impl IntoIterator<Item = i32>) -> Self {
        self.records = records.into_iter().collect();
        self
    }
}

// ============================================================================
// DEM Outputs
// ============================================================================

/// Metadata for a non-detector output definition.
///
/// Observables are rendered as standard `L<n>` targets. Tracked Paulis
/// use the same metadata shape but live in a separate PECOS-only ID space and
/// are never rendered as `L<n>` because they are unmeasured Pauli-operator
/// annotations, not measurement-record observables.
#[derive(Debug, Clone)]
pub struct DemOutput {
    /// Unique ID within this output's ID space.
    pub id: u32,
    /// Measurement record offsets (negative indices from end of record), when
    /// this output is tied to measurement records.
    pub records: SmallVec<[i32; 4]>,
    /// PECOS DEM output kind, when known.
    pub kind: Option<DemOutputKind>,
    /// Pauli string whose flip is tracked, when this came from a Pauli
    /// annotation or logical operator builder.
    pub pauli: Option<PauliString>,
    /// Optional user label.
    pub label: Option<String>,
}

impl DemOutput {
    /// Creates a new unclassified DEM output.
    #[must_use]
    pub fn new(id: u32) -> Self {
        Self {
            id,
            records: SmallVec::new(),
            kind: None,
            pauli: None,
            label: None,
        }
    }

    /// Creates a DEM output from PECOS propagation metadata.
    #[must_use]
    pub fn from_metadata(id: u32, metadata: &DemOutputMetadata) -> Self {
        Self {
            id,
            records: SmallVec::new(),
            kind: Some(metadata.kind),
            pauli: Some(metadata.pauli.clone()),
            label: metadata.label.clone(),
        }
    }

    /// Sets the measurement records.
    #[must_use]
    pub fn with_records(mut self, records: impl IntoIterator<Item = i32>) -> Self {
        self.records.clear();
        for record in records {
            toggle_dem_output_record(&mut self.records, record);
        }
        self.kind.get_or_insert(DemOutputKind::Observable);
        self
    }

    /// Sets the DEM output kind.
    #[must_use]
    pub fn with_kind(mut self, kind: DemOutputKind) -> Self {
        self.kind = Some(kind);
        self
    }

    /// Sets the tracked Pauli string.
    #[must_use]
    pub fn with_pauli(mut self, mut pauli: PauliString) -> Self {
        // A DEM output flip is an anticommutation property; global phase
        // has no meaning for DEM/sampler output.
        pauli.set_phase(pecos_core::QuarterPhase::PlusOne);
        self.pauli = Some(pauli);
        self
    }

    /// Sets a user-facing label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Returns true when this DEM output is an observable.
    #[must_use]
    pub fn is_observable(&self) -> bool {
        match self.kind {
            Some(DemOutputKind::Observable) => true,
            Some(DemOutputKind::TrackedPauli) => false,
            None => !self.records.is_empty(),
        }
    }

    /// Returns true when this DEM output is a tracked Pauli.
    #[must_use]
    pub fn is_tracked_pauli(&self) -> bool {
        match self.kind {
            Some(DemOutputKind::TrackedPauli) => true,
            Some(DemOutputKind::Observable) => false,
            None => self.pauli.is_some() && self.records.is_empty(),
        }
    }
}

fn merge_record_parity(existing: &mut SmallVec<[i32; 4]>, incoming: SmallVec<[i32; 4]>) {
    for record in incoming {
        toggle_dem_output_record(existing, record);
    }
}

fn toggle_dem_output_record(records: &mut SmallVec<[i32; 4]>, record: i32) {
    if let Some(pos) = records
        .iter()
        .position(|&existing_record| existing_record == record)
    {
        records.remove(pos);
    } else {
        records.push(record);
    }
}

/// Error returned when parsing or applying PECOS DEM metadata JSON.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PecosDemMetadataError {
    message: String,
}

impl PecosDemMetadataError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// Human-readable parse/apply error.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for PecosDemMetadataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for PecosDemMetadataError {}

// ============================================================================
// Noise Configuration
// ============================================================================

/// Per-Pauli fault probability weights.
///
/// Maps `PauliString` to relative probability. Entries must sum to ~1.0.
/// Used to customize the fault distribution for single-qubit or two-qubit gates.
///
/// # Examples
///
/// ```
/// use pecos_core::pauli::{X, Y, Z};
/// use pecos_qec::fault_tolerance::dem_builder::PauliWeights;
///
/// // Single-qubit: biased toward dephasing
/// let w = PauliWeights::from([(Z(0), 0.8), (X(0), 0.1), (Y(0), 0.1)]);
/// assert_eq!(w.entries().len(), 3);
///
/// // Two-qubit: uniform (convenience)
/// let w = PauliWeights::uniform_2q();
/// assert_eq!(w.entries().len(), 15);
/// ```
#[derive(Debug, Clone)]
pub struct PauliWeights {
    /// (`PauliString`, weight) pairs. Weights must sum to ~1.0.
    entries: Vec<(pecos_core::PauliString, f64)>,
}

impl PauliWeights {
    /// Create from an iterator of `(PauliString, weight)` pairs.
    ///
    /// Validates that weights sum to approximately 1.0 (within 1e-6).
    ///
    /// # Panics
    ///
    /// Panics if weights don't sum to ~1.0 or if any weight is negative.
    pub fn new(entries: impl IntoIterator<Item = (pecos_core::PauliString, f64)>) -> Self {
        let entries: Vec<_> = entries.into_iter().collect();
        let sum: f64 = entries.iter().map(|(_, w)| w).sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "PauliWeights must sum to 1.0, got {sum}"
        );
        for (ps, w) in &entries {
            assert!(*w >= 0.0, "Weight for {ps} must be non-negative, got {w}");
        }
        Self { entries }
    }

    /// Uniform weights for single-qubit gates: X, Y, Z each with 1/3.
    #[must_use]
    pub fn uniform_1q() -> Self {
        use pecos_core::pauli::{X, Y, Z};
        Self {
            entries: vec![(X(0), 1.0 / 3.0), (Y(0), 1.0 / 3.0), (Z(0), 1.0 / 3.0)],
        }
    }

    /// Uniform weights for two-qubit gates: all 15 non-identity Paulis at 1/15.
    #[must_use]
    pub fn uniform_2q() -> Self {
        use pecos_core::pauli::{X, Y, Z};
        let w = 1.0 / 15.0;
        Self {
            entries: vec![
                (X(1), w),
                (Y(1), w),
                (Z(1), w),
                (X(0), w),
                (X(0) & X(1), w),
                (X(0) & Y(1), w),
                (X(0) & Z(1), w),
                (Y(0), w),
                (Y(0) & X(1), w),
                (Y(0) & Y(1), w),
                (Y(0) & Z(1), w),
                (Z(0), w),
                (Z(0) & X(1), w),
                (Z(0) & Y(1), w),
                (Z(0) & Z(1), w),
            ],
        }
    }

    /// Look up the weight for a specific `PauliString`.
    ///
    /// Matches by Pauli type pattern only, ignoring qubit IDs.
    /// E.g., `X(3) & Z(7)` matches a weight entry `X(0) & Z(1)` because
    /// both have the pattern [X, Z] (sorted by qubit position).
    #[must_use]
    pub fn weight_for(&self, pauli: &pecos_core::PauliString) -> f64 {
        let query_pattern = pauli_pattern(pauli);
        self.entries
            .iter()
            .find(|(ps, _)| pauli_pattern(ps) == query_pattern)
            .map_or(0.0, |(_, w)| *w)
    }

    /// Get all entries as `(PauliString, weight)` pairs.
    #[must_use]
    pub fn entries(&self) -> &[(pecos_core::PauliString, f64)] {
        &self.entries
    }
}

impl<const N: usize> From<[(pecos_core::PauliString, f64); N]> for PauliWeights {
    fn from(entries: [(pecos_core::PauliString, f64); N]) -> Self {
        Self::new(entries)
    }
}

/// Noise model configuration for circuit-level fault analysis.
#[derive(Debug, Clone)]
pub struct NoiseConfig {
    /// Single-qubit gate error rate.
    pub p1: f64,
    /// Two-qubit gate error rate.
    pub p2: f64,
    /// Measurement error rate.
    pub p_meas: f64,
    /// Initialization (prep) error rate.
    pub p_prep: f64,
    /// Idle gate error rate per time unit.
    ///
    /// The actual error probability for an idle gate is `p_idle * duration`
    /// (clamped to [0, 1]), where `duration` is the gate's `TimeUnits` value.
    /// Default is 0.0 (no idle noise).
    pub p_idle: f64,
    /// Optional T1 relaxation time (in the same time units as idle duration).
    ///
    /// When set (along with T2), idle noise uses the Pauli-twirled
    /// amplitude damping + dephasing model instead of uniform depolarizing.
    /// This gives biased noise: P(Z) > P(X) = P(Y).
    pub t1: Option<f64>,
    /// Optional T2 dephasing time (must satisfy T2 <= 2*T1).
    pub t2: Option<f64>,
    /// Optional per-Pauli weights for single-qubit gates.
    ///
    /// Maps each Pauli fault to its relative probability. Must sum to ~1.0.
    /// Default (None) = uniform depolarizing.
    pub p1_weights: Option<PauliWeights>,
    /// Optional per-Pauli weights for two-qubit gates.
    ///
    /// Maps each two-qubit Pauli fault to its relative probability. Must sum to ~1.0.
    /// Default (None) = uniform depolarizing.
    pub p2_weights: Option<PauliWeights>,
    /// Coherent idle RZ rotation angle per time unit.
    ///
    /// When set (> 0), idle gates contribute a coherent Z rotation in addition
    /// to any stochastic idle noise. Idle fault locations with the same
    /// detector set have their angles accumulated coherently (angles add),
    /// giving probability `sin²(total_angle/2)` instead of independent combination.
    ///
    /// This is the EEG H-type noise model for idle gates. Default is 0.0.
    pub idle_rz: f64,
}

/// Per-Pauli error probabilities for a single qubit.
#[derive(Debug, Clone, Copy)]
pub struct PauliProbs {
    /// Probability of X error.
    pub px: f64,
    /// Probability of Y error.
    pub py: f64,
    /// Probability of Z error.
    pub pz: f64,
}

impl PauliProbs {
    /// Total error probability (px + py + pz).
    #[must_use]
    pub fn total(&self) -> f64 {
        self.px + self.py + self.pz
    }

    /// Uniform depolarizing: each Pauli with probability p/3.
    #[must_use]
    pub fn depolarizing(p: f64) -> Self {
        Self {
            px: p / 3.0,
            py: p / 3.0,
            pz: p / 3.0,
        }
    }

    /// Pauli-twirled T1/T2 noise for idle duration t.
    ///
    /// Approximates the combined amplitude damping (T1) and pure
    /// dephasing (T2) channel as a Pauli channel via Pauli twirling:
    ///
    ///   P(X) = P(Y) = (1 - e^{-t/T1}) / 4
    ///   P(Z) = (1 - e^{-t/T2}) / 2 - (1 - e^{-t/T1}) / 4
    ///
    /// Requires T2 <= 2*T1 (physical constraint).
    #[must_use]
    pub fn from_t1_t2(t: f64, t1: f64, t2: f64) -> Self {
        let gamma = 1.0 - (-t / t1).exp(); // amplitude damping parameter
        let lambda_t2 = 1.0 - (-t / t2).exp(); // total dephasing parameter

        let px = gamma / 4.0;
        let py = gamma / 4.0;
        let pz = (lambda_t2 / 2.0 - gamma / 4.0).max(0.0);

        Self { px, py, pz }
    }
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            p1: 0.01,
            p2: 0.01,
            p_meas: 0.01,
            p_prep: 0.01,
            p_idle: 0.0,
            t1: None,
            t2: None,
            p1_weights: None,
            p2_weights: None,
            idle_rz: 0.0,
        }
    }
}

impl NoiseConfig {
    /// Creates a new noise configuration (idle defaults to `None`).
    #[must_use]
    pub fn new(p1: f64, p2: f64, p_meas: f64, p_prep: f64) -> Self {
        Self {
            p1,
            p2,
            p_meas,
            p_prep,
            p_idle: 0.0,
            t1: None,
            t2: None,
            p1_weights: None,
            p2_weights: None,
            idle_rz: 0.0,
        }
    }

    /// Creates a new noise configuration with uniform depolarizing idle noise.
    #[must_use]
    pub fn with_idle(p1: f64, p2: f64, p_meas: f64, p_prep: f64, p_idle: f64) -> Self {
        Self {
            p1,
            p2,
            p_meas,
            p_prep,
            p_idle,
            t1: None,
            t2: None,
            p1_weights: None,
            p2_weights: None,
            idle_rz: 0.0,
        }
    }

    /// Creates a uniform noise configuration (including depolarizing idle).
    #[must_use]
    pub fn uniform(p: f64) -> Self {
        Self {
            p1: p,
            p2: p,
            p_meas: p,
            p_prep: p,
            p_idle: p,
            t1: None,
            t2: None,
            p1_weights: None,
            p2_weights: None,
            idle_rz: 0.0,
        }
    }

    /// Sets the idle noise rate on an existing config (uniform depolarizing).
    #[must_use]
    pub fn set_idle(mut self, p_idle: f64) -> Self {
        self.p_idle = p_idle;
        self
    }

    /// Sets T1/T2 relaxation times for idle noise.
    ///
    /// When set, idle gates use the Pauli-twirled T1/T2 model instead of
    /// uniform depolarizing. This produces biased noise where Z errors
    /// (dephasing) are more likely than X/Y errors (relaxation).
    ///
    /// T1 and T2 are in the same time units as idle gate durations.
    /// Must satisfy T2 <= 2*T1 (physical constraint).
    ///
    /// # Panics
    ///
    /// Panics if `t2 > 2 * t1`, which violates the physical constraint
    /// that the dephasing time cannot exceed twice the relaxation time.
    #[must_use]
    pub fn set_t1_t2(mut self, t1: f64, t2: f64) -> Self {
        assert!(
            t2 <= 2.0 * t1,
            "T2 ({t2}) must be <= 2*T1 ({}) (physical constraint)",
            2.0 * t1
        );
        self.t1 = Some(t1);
        self.t2 = Some(t2);
        self
    }

    /// Sets custom per-Pauli weights for single-qubit gates.
    ///
    /// ```
    /// use pecos_core::pauli::{X, Y, Z};
    /// use pecos_qec::fault_tolerance::dem_builder::{NoiseConfig, PauliWeights};
    ///
    /// let noise = NoiseConfig::uniform(0.001).set_p1_weights(PauliWeights::from([
    ///     (X(0), 0.1), (Y(0), 0.1), (Z(0), 0.8),
    /// ]));
    /// assert_eq!(noise.p1_weights.as_ref().unwrap().weight_for(&Z(7)), 0.8);
    /// ```
    #[must_use]
    pub fn set_p1_weights(mut self, weights: PauliWeights) -> Self {
        self.p1_weights = Some(weights);
        self
    }

    /// Sets custom per-Pauli weights for two-qubit gates.
    #[must_use]
    pub fn set_p2_weights(mut self, weights: PauliWeights) -> Self {
        self.p2_weights = Some(weights);
        self
    }

    /// Sets idle noise from a coherent RZ rotation angle per time unit.
    ///
    /// Converts `idle_rz` (the angle theta of an RZ(theta) rotation applied
    /// per idle time unit) to an equivalent stochastic Z-biased noise:
    ///
    ///   P(Z) = sin^2(theta/2)   per idle time unit
    ///   P(X) = P(Y) = 0
    ///
    /// This is the exact Pauli twirling of a pure dephasing channel and
    /// gives the non-EEG DEM builder correct idle noise behavior including
    /// proper correlations through the fault influence map.
    #[must_use]
    pub fn set_idle_rz(mut self, idle_rz: f64) -> Self {
        self.idle_rz = idle_rz;
        let pz = (idle_rz / 2.0).sin().powi(2);
        // Use T1/T2 model: T1=infinity (no amplitude damping), T2 chosen to match pz.
        // From the T1/T2 model: pz = (1 - exp(-t/T2))/2 for T1=inf, t=1.
        // Solve: T2 = -1/ln(1 - 2*pz)
        // This provides a stochastic representation for non-coherent code paths.
        if pz > 0.0 && pz < 0.5 {
            let t2 = -1.0 / (1.0 - 2.0 * pz).ln();
            let t1 = 1e15; // effectively infinity
            self.t1 = Some(t1);
            self.t2 = Some(t2);
        }
        self.p_idle = 0.0;
        self
    }

    /// Compute per-Pauli idle noise probabilities for a given duration.
    ///
    /// If T1/T2 are set, uses the Pauli-twirled model (biased noise).
    /// Otherwise, uses uniform depolarizing with `p_idle * duration`.
    #[must_use]
    pub fn idle_pauli_probs(&self, duration: f64) -> PauliProbs {
        if let (Some(t1), Some(t2)) = (self.t1, self.t2) {
            PauliProbs::from_t1_t2(duration, t1, t2)
        } else {
            PauliProbs::depolarizing((self.p_idle * duration).min(1.0))
        }
    }

    /// Returns true when idle locations use the dedicated idle-noise model.
    ///
    /// Otherwise `Idle` is a no-op for noise.
    #[must_use]
    pub fn uses_dedicated_idle_noise(&self) -> bool {
        self.p_idle > 0.0 || matches!((self.t1, self.t2), (Some(_), Some(_)))
    }
}

/// Extract the Pauli type pattern from a `PauliString`, ignoring qubit IDs.
///
/// Returns the sequence of Pauli types sorted by qubit position.
/// E.g., X(3) & Z(7) -> [X, Z], same as X(0) & Z(1) -> [X, Z].
fn pauli_pattern(ps: &pecos_core::PauliString) -> Vec<pecos_core::Pauli> {
    ps.paulis().iter().map(|&(p, _)| p).collect()
}

fn pecos_metadata_dem_output_value(target: &DemOutput) -> serde_json::Value {
    serde_json::json!({
        "id": target.id,
        "kind": target.kind.map_or("dem_output", DemOutputKind::as_str),
        "label": target.label,
        "pauli": target.pauli.as_ref().map(PauliString::to_sparse_str),
        "records": target.records.iter().copied().collect::<Vec<_>>(),
    })
}

#[derive(Debug, Clone, Default)]
struct ParsedPecosDemMetadata {
    observables: Vec<DemOutput>,
    tracked_paulis: Vec<DemOutput>,
}

pub(crate) fn parse_pecos_dem_metadata_line(
    line: &str,
) -> Result<DemOutput, PecosDemMetadataError> {
    let line = line.trim();
    let (prefix, payload, forced_kind) =
        if let Some(payload) = line.strip_prefix("pecos_tracked_pauli") {
            (
                "pecos_tracked_pauli",
                payload.trim(),
                Some(DemOutputKind::TrackedPauli),
            )
        } else if let Some(payload) = line.strip_prefix("pecos_observable") {
            (
                "pecos_observable",
                payload.trim(),
                Some(DemOutputKind::Observable),
            )
        } else {
            return Err(PecosDemMetadataError::new(
                "missing PECOS DEM metadata prefix",
            ));
        };
    if payload.is_empty() {
        return Err(PecosDemMetadataError::new(format!(
            "{prefix} is missing its JSON payload"
        )));
    }

    let value: serde_json::Value = serde_json::from_str(payload).map_err(|err| {
        PecosDemMetadataError::new(format!("invalid {prefix} JSON payload: {err}"))
    })?;
    let mut output = parse_pecos_metadata_dem_output(0, &value)?;
    if let Some(kind) = forced_kind {
        output.kind = Some(kind);
    }
    if output.is_tracked_pauli() && !output.records.is_empty() {
        return Err(PecosDemMetadataError::new(
            "tracked Pauli metadata cannot have measurement records",
        ));
    }
    Ok(output)
}

fn parse_pecos_metadata_json(json: &str) -> Result<ParsedPecosDemMetadata, PecosDemMetadataError> {
    let root: serde_json::Value = serde_json::from_str(json).map_err(|err| {
        PecosDemMetadataError::new(format!("invalid PECOS DEM metadata JSON: {err}"))
    })?;

    let format = root
        .get("format")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| PecosDemMetadataError::new("missing metadata format"))?;
    if format != "pecos.dem.metadata" {
        return Err(PecosDemMetadataError::new(format!(
            "unsupported metadata format: {format}"
        )));
    }

    let version = root
        .get("version")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| PecosDemMetadataError::new("missing metadata version"))?;
    if version != 1 {
        return Err(PecosDemMetadataError::new(format!(
            "unsupported PECOS DEM metadata version: {version}"
        )));
    }

    for old_name in ["tracked_ops", "tracked_operators", "pauli_operators"] {
        if root.get(old_name).is_some() {
            return Err(PecosDemMetadataError::new(format!(
                "unsupported legacy metadata field: {old_name}; use tracked_paulis"
            )));
        }
    }

    let parse_array =
        |name: &str, kind: DemOutputKind| -> Result<Vec<DemOutput>, PecosDemMetadataError> {
            let Some(values) = root.get(name) else {
                return Ok(Vec::new());
            };
            let values = values.as_array().ok_or_else(|| {
                PecosDemMetadataError::new(format!("{name} metadata is not an array"))
            })?;
            values
                .iter()
                .enumerate()
                .map(|(idx, value)| {
                    let mut output = parse_pecos_metadata_dem_output(idx, value)?;
                    output.kind = Some(kind);
                    if kind == DemOutputKind::TrackedPauli && !output.records.is_empty() {
                        return Err(PecosDemMetadataError::new(format!(
                            "tracked Pauli metadata {idx} cannot have measurement records"
                        )));
                    }
                    Ok(output)
                })
                .collect()
        };

    let parsed = ParsedPecosDemMetadata {
        observables: parse_array("observables", DemOutputKind::Observable)?,
        tracked_paulis: parse_array("tracked_paulis", DemOutputKind::TrackedPauli)?,
    };

    if root.get("observables").is_none() && root.get("tracked_paulis").is_none() {
        return Err(PecosDemMetadataError::new(
            "missing observables or tracked_paulis metadata arrays",
        ));
    }

    Ok(parsed)
}

fn parse_pecos_metadata_dem_output(
    idx: usize,
    target: &serde_json::Value,
) -> Result<DemOutput, PecosDemMetadataError> {
    let object = target
        .as_object()
        .ok_or_else(|| PecosDemMetadataError::new(format!("DEM output {idx} is not an object")))?;

    let id = object
        .get("id")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| PecosDemMetadataError::new(format!("DEM output {idx} is missing id")))?;
    let id = u32::try_from(id).map_err(|_| {
        PecosDemMetadataError::new(format!("DEM output {idx} id does not fit in u32"))
    })?;

    let mut dem_output = DemOutput::new(id);
    let mut explicit_kind = None;

    if let Some(kind_value) = object.get("kind") {
        let kind = kind_value.as_str().ok_or_else(|| {
            PecosDemMetadataError::new(format!("DEM output {idx} kind is not a string"))
        })?;
        if kind != "dem_output" {
            let parsed_kind = DemOutputKind::from_metadata_str(kind).ok_or_else(|| {
                PecosDemMetadataError::new(format!("DEM output {idx} has unknown kind: {kind}"))
            })?;
            explicit_kind = Some(parsed_kind);
            dem_output = dem_output.with_kind(parsed_kind);
        }
    }

    if let Some(label_value) = object.get("label")
        && !label_value.is_null()
    {
        let label = label_value.as_str().ok_or_else(|| {
            PecosDemMetadataError::new(format!("DEM output {idx} label is not a string or null"))
        })?;
        dem_output = dem_output.with_label(label);
    }

    if let Some(pauli_value) = object.get("pauli")
        && !pauli_value.is_null()
    {
        let pauli = pauli_value.as_str().ok_or_else(|| {
            PecosDemMetadataError::new(format!("DEM output {idx} pauli is not a string or null"))
        })?;
        let pauli = pauli.parse::<PauliString>().map_err(|err| {
            PecosDemMetadataError::new(format!("DEM output {idx} has invalid PauliString: {err}"))
        })?;
        dem_output = dem_output.with_pauli(pauli);
    }

    let records = if let Some(records_value) = object.get("records") {
        if records_value.is_null() {
            Vec::new()
        } else {
            let records = records_value.as_array().ok_or_else(|| {
                PecosDemMetadataError::new(format!(
                    "DEM output {idx} records is not an array or null"
                ))
            })?;
            records
                .iter()
                .enumerate()
                .map(|(record_idx, record)| {
                    let record = record.as_i64().ok_or_else(|| {
                        PecosDemMetadataError::new(format!(
                            "DEM output {idx} record {record_idx} is not an integer"
                        ))
                    })?;
                    i32::try_from(record).map_err(|_| {
                        PecosDemMetadataError::new(format!(
                            "DEM output {idx} record {record_idx} does not fit in i32"
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?
        }
    } else {
        Vec::new()
    };

    if explicit_kind == Some(DemOutputKind::TrackedPauli) && !records.is_empty() {
        return Err(PecosDemMetadataError::new(format!(
            "tracked Pauli DEM output {idx} cannot have measurement records"
        )));
    }

    if !records.is_empty() || explicit_kind == Some(DemOutputKind::Observable) {
        dem_output = dem_output.with_records(records);
    }

    Ok(dem_output)
}

// ============================================================================
// Per-gate-type noise configuration
// ============================================================================

use pecos_core::QubitId;
use std::collections::HashMap;

/// Ordered indices for the 3 non-identity 1Q Paulis: `[X, Y, Z]`.
pub const PAULI_1Q_ORDER: [&str; 3] = ["X", "Y", "Z"];

/// Ordered indices for the 15 non-identity 2Q Pauli pairs. Row/col order:
/// `I=0, X=1, Y=2, Z=3`; pair `p1 ⊗ p2` index is `4*p1 + p2 - 1` (skip II).
/// Concretely: `["IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX",
/// "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"]`.
pub const PAULI_2Q_ORDER: [&str; 15] = [
    "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ",
];

/// Per-gate-type, optionally per-qubit noise specification. Replaces the
/// uniform scalar `NoiseConfig` with per-`GateType` per-Pauli rates, with
/// an optional per-qubit override layer for devices where `T_1`/`T_2`
/// varies qubit-to-qubit.
///
/// # Layered lookup
///
/// Rate resolution uses the most specific configured entry:
///
/// ```text
/// 1. rates_1q_per_qubit[(gate, qubit)]         // most specific
/// 2. rates_1q[gate]                            // per-gate-type default
/// 3. base.p1 / 3.0                             // base noise model
/// ```
///
/// (And analogously for 2Q with `(gate, (q_control, q_target))`.)
///
/// This lets users specify "H on qubit 0 has these rates (high `T_1`), H on
/// qubit 1 has these (low `T_1`), every other H uses the per-gate default".
///
/// # Integration with `pecos-lindblad`
///
/// The intended workflow is:
///   1. Synthesize a `PauliLindbladModel` for each gate type *and* per
///      qubit if needed via `pecos_lindblad::synthesize_superop(...)`.
///   2. Convert to `[f64; 3]` / `[f64; 15]` arrays.
///   3. Register with [`Self::with_1q_rates_for_qubit`] /
///      [`Self::with_2q_rates_for_qubits`] for heterogeneous devices, or
///      [`Self::with_1q_rates`] / [`Self::with_2q_rates`] for homogeneous
///      models.
#[derive(Debug, Clone, Default)]
pub struct PerGateTypeNoise {
    pub rates_1q: HashMap<GateType, [f64; 3]>,
    pub rates_2q: HashMap<GateType, [f64; 15]>,
    pub rates_1q_per_qubit: HashMap<(GateType, QubitId), [f64; 3]>,
    pub rates_2q_per_qubits: HashMap<(GateType, QubitId, QubitId), [f64; 15]>,
    /// Per-qubit readout (MZ) X-flip probabilities. Qubits not in this map use
    /// [`Self::p_meas`].
    pub measurement_rates: HashMap<QubitId, f64>,
    /// Per-qubit preparation (PZ) X-error probabilities. Qubits not in this map
    /// use [`Self::p_init`].
    pub init_rates: HashMap<QubitId, f64>,
    pub p_meas: f64,
    pub p_init: f64,
    pub base: NoiseConfig,
}

impl PerGateTypeNoise {
    /// Construct with empty gate maps; unspecified gates use `base`.
    #[must_use]
    pub fn from_base_noise(base: NoiseConfig) -> Self {
        Self {
            rates_1q: HashMap::new(),
            rates_2q: HashMap::new(),
            rates_1q_per_qubit: HashMap::new(),
            rates_2q_per_qubits: HashMap::new(),
            measurement_rates: HashMap::new(),
            init_rates: HashMap::new(),
            p_meas: base.p_meas,
            p_init: base.p_prep,
            base,
        }
    }

    /// Attach measurement X-flip probability for a specific qubit.
    /// Overrides [`Self::p_meas`] when set. Use for devices with
    /// heterogeneous readout fidelity.
    #[must_use]
    pub fn with_measurement_rate(mut self, q: QubitId, p: f64) -> Self {
        self.measurement_rates.insert(q, p);
        self
    }

    /// Attach preparation X-error probability for a specific qubit.
    /// Overrides [`Self::p_init`] when set.
    #[must_use]
    pub fn with_init_rate(mut self, q: QubitId, p: f64) -> Self {
        self.init_rates.insert(q, p);
        self
    }

    /// Lookup measurement X-flip rate for a qubit. Unspecified qubits use
    /// [`Self::p_meas`], which is seeded from the base `NoiseConfig::p_meas`.
    #[must_use]
    pub fn measurement_rate_on(&self, q: QubitId) -> f64 {
        *self.measurement_rates.get(&q).unwrap_or(&self.p_meas)
    }

    /// Lookup preparation X-error rate for a qubit. Unspecified qubits use
    /// [`Self::p_init`].
    #[must_use]
    pub fn init_rate_on(&self, q: QubitId) -> f64 {
        *self.init_rates.get(&q).unwrap_or(&self.p_init)
    }

    /// Attach rates for a 1Q gate type applied to any qubit.
    #[must_use]
    pub fn with_1q_rates(mut self, g: GateType, rates: [f64; 3]) -> Self {
        self.rates_1q.insert(g, rates);
        self
    }

    /// Attach rates for a 2Q gate type applied to any qubit pair.
    #[must_use]
    pub fn with_2q_rates(mut self, g: GateType, rates: [f64; 15]) -> Self {
        self.rates_2q.insert(g, rates);
        self
    }

    /// Attach rates for a 1Q gate on a specific qubit. Takes precedence
    /// over [`Self::with_1q_rates`] for that `(gate, qubit)` combination.
    #[must_use]
    pub fn with_1q_rates_for_qubit(mut self, g: GateType, q: QubitId, rates: [f64; 3]) -> Self {
        self.rates_1q_per_qubit.insert((g, q), rates);
        self
    }

    /// Return explicitly attached 1Q Pauli rates for a gate type.
    #[must_use]
    pub fn explicit_1q_rates(&self, gate: GateType) -> Option<[f64; 3]> {
        self.rates_1q.get(&gate).copied()
    }

    /// Return explicitly attached 1Q Pauli rates for a gate on a specific qubit.
    ///
    /// Per-qubit rates take precedence over gate-type rates. Unlike
    /// [`Self::rate_1q_on`], this does not fall back to the base noise model.
    #[must_use]
    pub fn explicit_1q_rates_on(&self, gate: GateType, qubit: QubitId) -> Option<[f64; 3]> {
        self.rates_1q_per_qubit
            .get(&(gate, qubit))
            .copied()
            .or_else(|| self.explicit_1q_rates(gate))
    }

    /// Attach rates for a 2Q gate on a specific ordered qubit pair.
    /// Takes precedence over [`Self::with_2q_rates`] for that
    /// `(gate, q_control, q_target)` combination.
    #[must_use]
    pub fn with_2q_rates_for_qubits(
        mut self,
        g: GateType,
        q_control: QubitId,
        q_target: QubitId,
        rates: [f64; 15],
    ) -> Self {
        self.rates_2q_per_qubits
            .insert((g, q_control, q_target), rates);
        self
    }

    /// Lookup 1Q Pauli rate for a gate. Returns `base.p1 / 3.0` if the
    /// gate type is not in the map. `pauli_idx` is 0=X, 1=Y, 2=Z.
    ///
    /// `Idle` is a no-op by default. It receives noise only from explicitly
    /// attached idle rates or from the base idle-noise model.
    #[must_use]
    pub fn rate_1q(&self, gate: GateType, pauli_idx: usize) -> f64 {
        if let Some(rates) = self.rates_1q.get(&gate) {
            return rates[pauli_idx];
        }
        if gate == GateType::Idle {
            if self.base.uses_dedicated_idle_noise() {
                let probs = self.base.idle_pauli_probs(1.0);
                return match pauli_idx {
                    0 => probs.px,
                    1 => probs.py,
                    2 => probs.pz,
                    _ => 0.0,
                };
            }
            return 0.0;
        }
        self.base.p1 / 3.0
    }

    /// Lookup 1Q Pauli rate for a gate on a specific qubit. Tries the
    /// per-qubit map first, then the per-gate-type map, then `base.p1 / 3.0`.
    /// `pauli_idx` is 0=X, 1=Y, 2=Z.
    #[must_use]
    pub fn rate_1q_on(&self, gate: GateType, qubit: QubitId, pauli_idx: usize) -> f64 {
        if let Some(rates) = self.rates_1q_per_qubit.get(&(gate, qubit)) {
            return rates[pauli_idx];
        }
        self.rate_1q(gate, pauli_idx)
    }

    /// Lookup 2Q Pauli pair rate for a gate. Returns `base.p2 / 15.0`
    /// if the gate type is not in the map. `pair_idx` follows [`PAULI_2Q_ORDER`].
    #[must_use]
    pub fn rate_2q(&self, gate: GateType, pair_idx: usize) -> f64 {
        self.rates_2q
            .get(&gate)
            .map_or(self.base.p2 / 15.0, |r| r[pair_idx])
    }

    /// Lookup 2Q Pauli pair rate for a gate on a specific ordered
    /// qubit pair. Tries `(gate, q_control, q_target)` in the per-qubits
    /// map first, then the per-gate-type map, then `base.p2 / 15.0`.
    #[must_use]
    pub fn rate_2q_on(
        &self,
        gate: GateType,
        q_control: QubitId,
        q_target: QubitId,
        pair_idx: usize,
    ) -> f64 {
        if let Some(rates) = self.rates_2q_per_qubits.get(&(gate, q_control, q_target)) {
            return rates[pair_idx];
        }
        self.rate_2q(gate, pair_idx)
    }
}

// ============================================================================
// Measurement Noise Model (MNM)
// ============================================================================

/// A measurement fault mechanism: a set of measurements that flip together.
///
/// Unlike [`FaultMechanism`], this operates directly on raw measurement
/// indices. This is useful for sampling measurement outcomes without needing
/// detector definitions.
#[derive(Clone, Default)]
pub struct MeasurementMechanism {
    /// Measurement indices that flip together, sorted canonically.
    pub measurements: SmallVec<[u32; 4]>,
}

impl MeasurementMechanism {
    /// Creates a new empty measurement mechanism.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a mechanism from unsorted measurement indices.
    #[must_use]
    pub fn from_unsorted(measurements: impl IntoIterator<Item = u32>) -> Self {
        let mut measurements: SmallVec<[u32; 4]> = measurements.into_iter().collect();
        measurements.sort_unstable();
        Self { measurements }
    }

    /// Creates a mechanism from pre-sorted measurement indices.
    #[must_use]
    pub fn from_sorted(measurements: SmallVec<[u32; 4]>) -> Self {
        debug_assert!(
            measurements.windows(2).all(|w| w[0] <= w[1]),
            "measurements must be sorted"
        );
        Self { measurements }
    }

    /// Returns true if this mechanism has no effect.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.measurements.is_empty()
    }

    /// Returns the number of measurements in this mechanism.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.measurements.len()
    }
}

impl PartialEq for MeasurementMechanism {
    fn eq(&self, other: &Self) -> bool {
        self.measurements == other.measurements
    }
}

impl Eq for MeasurementMechanism {}

impl Hash for MeasurementMechanism {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.measurements.hash(state);
    }
}

impl PartialOrd for MeasurementMechanism {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MeasurementMechanism {
    fn cmp(&self, other: &Self) -> Ordering {
        self.measurements.cmp(&other.measurements)
    }
}

impl fmt::Debug for MeasurementMechanism {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MeasurementMechanism({:?})",
            self.measurements.as_slice()
        )
    }
}

/// A measurement noise model for fast approximate raw-measurement sampling.
#[derive(Debug, Clone, Default)]
pub struct MeasurementNoiseModel {
    /// Fault mechanisms mapped to their probabilities.
    pub mechanisms: BTreeMap<MeasurementMechanism, f64>,
    /// Total number of measurements in the circuit.
    pub num_measurements: usize,
    /// Optional mapping from influence-map index to original circuit order.
    pub im_to_tc_order: Option<Vec<usize>>,
}

impl MeasurementNoiseModel {
    /// Creates a new empty measurement noise model.
    #[must_use]
    pub fn new(num_measurements: usize) -> Self {
        Self {
            mechanisms: BTreeMap::new(),
            num_measurements,
            im_to_tc_order: None,
        }
    }

    /// Sets the measurement order mapping from influence-map order to circuit order.
    #[must_use]
    pub fn with_measurement_order(mut self, im_to_tc: Vec<usize>) -> Self {
        self.im_to_tc_order = Some(im_to_tc);
        self
    }

    /// Sets the measurement order mapping in place.
    pub fn set_measurement_order(&mut self, im_to_tc: Vec<usize>) {
        self.im_to_tc_order = Some(im_to_tc);
    }

    /// Returns the number of distinct mechanisms.
    #[inline]
    #[must_use]
    pub fn num_mechanisms(&self) -> usize {
        self.mechanisms.len()
    }

    /// Adds a mechanism with probability, combining with any existing identical mechanism.
    pub fn add_mechanism(&mut self, mechanism: MeasurementMechanism, probability: f64) {
        if mechanism.is_empty() || probability <= 0.0 {
            return;
        }

        self.mechanisms
            .entry(mechanism)
            .and_modify(|p| *p = combine_probabilities(*p, probability))
            .or_insert(probability);
    }

    /// Samples measurement outcomes into a pre-sized buffer.
    pub fn sample_into<R: rand::Rng>(&self, outcomes: &mut [bool], rng: &mut R) {
        outcomes.fill(false);

        for (mechanism, &prob) in &self.mechanisms {
            if rng.random_bool(prob.clamp(0.0, 1.0)) {
                for &meas_idx in &mechanism.measurements {
                    if (meas_idx as usize) < outcomes.len() {
                        outcomes[meas_idx as usize] ^= true;
                    }
                }
            }
        }
    }

    /// Samples and returns measurement outcomes.
    #[must_use]
    pub fn sample<R: rand::Rng>(&self, rng: &mut R) -> Vec<bool> {
        let mut outcomes = vec![false; self.num_measurements];
        self.sample_into(&mut outcomes, rng);
        outcomes
    }

    /// Iterates over all mechanisms and their probabilities.
    pub fn iter(&self) -> impl Iterator<Item = (&MeasurementMechanism, &f64)> {
        self.mechanisms.iter()
    }

    /// Computes detector events from raw measurement outcomes and detector records.
    #[must_use]
    pub fn compute_detection_events(
        &self,
        outcomes: &[bool],
        detector_records: &[Vec<i32>],
    ) -> Vec<bool> {
        let tc_outcomes: Vec<bool> = if let Some(ref im_to_tc) = self.im_to_tc_order {
            let mut reordered = vec![false; outcomes.len()];
            for (im_idx, &tc_idx) in im_to_tc.iter().enumerate() {
                if im_idx < outcomes.len() && tc_idx < reordered.len() {
                    reordered[tc_idx] = outcomes[im_idx];
                }
            }
            reordered
        } else {
            outcomes.to_vec()
        };

        Self::to_detection_events_internal(&tc_outcomes, detector_records)
    }

    fn to_detection_events_internal(outcomes: &[bool], detector_records: &[Vec<i32>]) -> Vec<bool> {
        let num_measurements = outcomes.len();
        detector_records
            .iter()
            .map(|records| {
                records.iter().fold(false, |fired, &offset| {
                    if let Some(abs_idx) = record_offset_to_absolute_index(num_measurements, offset)
                        && abs_idx < num_measurements
                        && outcomes[abs_idx]
                    {
                        return !fired;
                    }
                    fired
                })
            })
            .collect()
    }

    /// Static detector-event conversion without reordering.
    #[must_use]
    pub fn to_detection_events(outcomes: &[bool], detector_records: &[Vec<i32>]) -> Vec<bool> {
        Self::to_detection_events_internal(outcomes, detector_records)
    }

    /// Samples and converts to detection events in one step.
    pub fn sample_with_detectors<R: rand::Rng>(
        &self,
        detector_records: &[Vec<i32>],
        rng: &mut R,
    ) -> (Vec<bool>, Vec<bool>) {
        let outcomes = self.sample(rng);
        let detection_events = self.compute_detection_events(&outcomes, detector_records);
        (outcomes, detection_events)
    }

    /// Computes observable flips from measurement outcomes.
    #[must_use]
    pub fn compute_observable_flips(
        &self,
        outcomes: &[bool],
        observable_records: &[Vec<i32>],
    ) -> Vec<bool> {
        self.compute_detection_events(outcomes, observable_records)
    }

    /// Samples detector events and observable flips in one step.
    pub fn sample_for_decoding<R: rand::Rng>(
        &self,
        detector_records: &[Vec<i32>],
        observable_records: &[Vec<i32>],
        rng: &mut R,
    ) -> (Vec<bool>, Vec<bool>) {
        let outcomes = self.sample(rng);
        let detection_events = self.compute_detection_events(&outcomes, detector_records);
        let observable_flips = self.compute_detection_events(&outcomes, observable_records);
        (detection_events, observable_flips)
    }

    /// Batch samples detector events and observable flips.
    pub fn sample_batch_for_decoding<R: rand::Rng>(
        &self,
        num_shots: usize,
        detector_records: &[Vec<i32>],
        observable_records: &[Vec<i32>],
        rng: &mut R,
    ) -> (Vec<Vec<bool>>, Vec<Vec<bool>>) {
        let mut all_detection_events = Vec::with_capacity(num_shots);
        let mut all_observable_flips = Vec::with_capacity(num_shots);

        for _ in 0..num_shots {
            let (det_events, obs_flips) =
                self.sample_for_decoding(detector_records, observable_records, rng);
            all_detection_events.push(det_events);
            all_observable_flips.push(obs_flips);
        }

        (all_detection_events, all_observable_flips)
    }
}

// ============================================================================
// Detector Error Model
// ============================================================================

/// A complete Detector Error Model (DEM).
///
/// This represents the noise model of a quantum circuit. It maps mechanisms
/// (detector/DEM-output effects) to their probabilities.
///
/// # Aggregation Modes
///
/// The DEM supports two modes controlled by `aggregate`:
///
/// - **Aggregated mode** (`aggregate = true`): Mechanisms with the same effect
///   are combined into a single entry using the independent error formula.
///   This is more compact but loses information about noise sources.
///
/// - **Non-aggregated mode** (`aggregate = false`, default): Each noise source
///   is kept as a separate entry. This preserves correlation information and
///   allows advanced decoders to understand the noise structure.
///
/// # Decomposed Entries
///
/// In addition to direct mechanisms, the DEM can store "decomposed" entries
/// that represent Y faults as X^Z. These are stored separately because:
///
/// 1. They help MWPM decoders understand correlation structure
/// 2. They preserve information about fault sources
///
/// When a Y fault has X-effect {`D_x`} and Z-effect {`D_z`} where both are non-empty
/// and different, it can be represented as `{D_x} ^ {D_z}` instead of XOR-ing
/// into a single mechanism.
#[derive(Debug, Clone)]
pub struct DetectorErrorModel {
    /// Detector definitions.
    pub detectors: Vec<DetectorDef>,
    /// Measurement-record observables rendered as standard `L<n>` outputs.
    pub observables: Vec<DemOutput>,
    /// PECOS tracked Paulis.
    ///
    /// These have their own ID space and are emitted only as PECOS metadata.
    pub tracked_paulis: Vec<DemOutput>,
    /// Error contributions with source tracking.
    /// Each contribution tracks whether it came from a direct (X, Z) or decomposable (Y) source.
    contributions: Vec<FaultContribution>,
    /// Count of graphlike decomposable sources per 2-detector mechanism.
    /// Key is (d0, d1) with d0 < d1. A source is "graphlike decomposable" if both
    /// component effects are non-empty and graphlike (≤2 detectors).
    /// Used to determine output format: ≥2 → 3 forms, 1 → 2 forms, 0 → 1 form.
    graphlike_decomposable_counts: BTreeMap<(u32, u32), u32>,
}

/// Structured DEM mechanism tuple: `(probability, detector_ids, observable_ids)`.
pub type MechanismTuple = (f64, Vec<u32>, Vec<u32>);

/// Detector-coordinate tuple: `(detector_id, coordinates)`.
pub type DetectorCoordinateTuple = (u32, Vec<f64>);

impl DetectorErrorModel {
    /// Creates a new empty DEM.
    #[must_use]
    pub fn new() -> Self {
        Self {
            detectors: Vec::new(),
            observables: Vec::new(),
            tracked_paulis: Vec::new(),
            contributions: Vec::new(),
            graphlike_decomposable_counts: BTreeMap::new(),
        }
    }

    /// Creates a DEM with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(num_detectors: usize, num_observables: usize) -> Self {
        Self {
            detectors: Vec::with_capacity(num_detectors),
            observables: Vec::with_capacity(num_observables),
            tracked_paulis: Vec::new(),
            contributions: Vec::new(),
            graphlike_decomposable_counts: BTreeMap::new(),
        }
    }

    /// Returns the number of detectors.
    #[inline]
    #[must_use]
    pub fn num_detectors(&self) -> usize {
        self.detectors.len()
    }

    /// Returns the number of observables.
    #[inline]
    #[must_use]
    pub fn num_observables(&self) -> usize {
        self.observables
            .iter()
            .map(|op| op.id as usize + 1)
            .max()
            .unwrap_or(0)
    }

    /// Returns the number of standard DEM `L<n>` observable outputs.
    ///
    /// This is a DEM-output alias for [`Self::num_observables`]. It does
    /// not include PECOS tracked Paulis.
    #[inline]
    #[must_use]
    pub fn num_dem_outputs(&self) -> usize {
        self.num_observables()
    }

    /// Returns the number of tracked Paulis.
    #[inline]
    #[must_use]
    pub fn num_tracked_paulis(&self) -> usize {
        self.tracked_paulis
            .iter()
            .map(|op| op.id as usize + 1)
            .max()
            .unwrap_or(0)
    }

    /// Returns standard DEM output definitions (`L<n>` observables).
    ///
    /// This DEM-output accessor does not include PECOS tracked Paulis;
    /// use [`Self::tracked_paulis`] for those.
    #[inline]
    #[must_use]
    pub fn dem_outputs(&self) -> &[DemOutput] {
        &self.observables
    }

    /// Returns mutable standard DEM output definitions (`L<n>` observables).
    ///
    /// This DEM-output accessor does not include PECOS tracked Paulis.
    #[inline]
    #[must_use]
    pub fn dem_outputs_mut(&mut self) -> &mut [DemOutput] {
        &mut self.observables
    }

    /// Iterates over observables.
    pub fn observables(&self) -> impl Iterator<Item = &DemOutput> {
        self.observables.iter()
    }

    /// Returns all tracked Pauli definitions.
    #[inline]
    #[must_use]
    pub fn tracked_paulis(&self) -> &[DemOutput] {
        &self.tracked_paulis
    }

    /// Iterates over tracked Paulis.
    pub fn iter_tracked_paulis(&self) -> impl Iterator<Item = &DemOutput> {
        self.tracked_paulis.iter()
    }

    /// Returns the number of tracked contributions.
    #[inline]
    #[must_use]
    pub fn num_contributions(&self) -> usize {
        self.contributions.len()
    }

    /// Exports PECOS-only metadata that is not representable in standard DEM syntax.
    ///
    /// The standard DEM string remains decoder-compatible and uses ordinary
    /// `logical_observable L<n>` declarations. This JSON form preserves the
    /// richer PECOS DEM-output information, including tracked Paulis.
    ///
    /// # Panics
    ///
    /// Panics only if serializing a JSON value constructed in this method fails.
    #[must_use]
    pub fn to_pecos_metadata_json(&self) -> String {
        let observables: Vec<serde_json::Value> = self
            .observables
            .iter()
            .map(pecos_metadata_dem_output_value)
            .collect();
        let tracked_paulis: Vec<serde_json::Value> = self
            .tracked_paulis
            .iter()
            .map(pecos_metadata_dem_output_value)
            .collect();

        serde_json::to_string_pretty(&serde_json::json!({
            "format": "pecos.dem.metadata",
            "version": 1,
            "observables": observables,
            "tracked_paulis": tracked_paulis,
        }))
        .expect("serializing PECOS DEM metadata should not fail")
    }

    /// Applies PECOS DEM metadata JSON to this model.
    ///
    /// This is the inverse of [`Self::to_pecos_metadata_json`] for DEM output
    /// definitions. It updates existing outputs by `id` and adds any that
    /// are present in the metadata but missing from the DEM.
    ///
    /// # Errors
    ///
    /// Returns an error if the JSON is malformed, uses an unsupported metadata
    /// version, has an unknown op kind, or contains invalid Pauli/record
    /// fields.
    pub fn apply_pecos_metadata_json(&mut self, json: &str) -> Result<(), PecosDemMetadataError> {
        let metadata = parse_pecos_metadata_json(json)?;
        for observable in metadata.observables {
            self.apply_observable_metadata(observable);
        }
        for tracked_pauli in metadata.tracked_paulis {
            self.apply_tracked_pauli_metadata(tracked_pauli);
        }
        Ok(())
    }

    /// Returns a copy of this model with PECOS DEM metadata JSON applied.
    ///
    /// # Errors
    ///
    /// See [`Self::apply_pecos_metadata_json`].
    pub fn with_pecos_metadata_json(mut self, json: &str) -> Result<Self, PecosDemMetadataError> {
        self.apply_pecos_metadata_json(json)?;
        Ok(self)
    }

    /// Converts the DEM to PECOS DEM text.
    ///
    /// This format is a strict superset of standard DEM text. It uses `D<n>`
    /// detector targets and `L<n>` measurement-defined observable targets as
    /// usual, and adds PECOS-only `TP<n>` tracked-Pauli targets for tracked
    /// operator flips. Metadata follows as `pecos_observable {json}` and
    /// `pecos_tracked_pauli {json}` statements.
    ///
    /// # Panics
    ///
    /// Panics only if serializing JSON values constructed in this method fails.
    #[must_use]
    pub fn to_pecos_string(&self) -> String {
        let mut lines = Vec::new();

        for det in &self.detectors {
            if let Some([x, y, z]) = det.coords {
                lines.push(format!("detector({x}, {y}, {z}) D{}", det.id));
            } else {
                lines.push(format!("detector D{}", det.id));
            }
        }

        for obs in &self.observables {
            lines.push(format!("logical_observable L{}", obs.id));
        }

        let mut by_effect: BTreeMap<FaultMechanism, f64> = BTreeMap::new();
        for contrib in &self.contributions {
            by_effect
                .entry(contrib.effect.clone())
                .and_modify(|p| *p = combine_independent_probs(*p, contrib.probability))
                .or_insert(contrib.probability);
        }

        for (effect, total_prob) in by_effect {
            if effect.is_empty() || total_prob <= 0.0 {
                continue;
            }

            let targets = format_pecos_mechanism_targets(&effect);
            if !targets.is_empty() {
                lines.push(format!(
                    "error({}) {}",
                    format_probability(total_prob),
                    targets
                ));
            }
        }

        let metadata_lines = self.pecos_metadata_lines();

        if metadata_lines.is_empty() {
            return lines.join("\n");
        }
        lines.extend(metadata_lines);
        lines.join("\n")
    }

    fn pecos_metadata_lines(&self) -> Vec<String> {
        let observable_lines = self.observables.iter().map(|observable| {
            let value = pecos_metadata_dem_output_value(observable);
            let payload = serde_json::to_string(&value)
                .expect("serializing PECOS observable metadata should not fail");
            format!("pecos_observable {payload}")
        });
        let tracked_pauli_lines = self.tracked_paulis.iter().map(|tracked_pauli| {
            let value = pecos_metadata_dem_output_value(tracked_pauli);
            let payload = serde_json::to_string(&value)
                .expect("serializing PECOS tracked-Pauli metadata should not fail");
            format!("pecos_tracked_pauli {payload}")
        });
        observable_lines.chain(tracked_pauli_lines).collect()
    }

    /// Applies PECOS metadata embedded in extended DEM text.
    ///
    /// Standard DEM lines are ignored by this method. PECOS extension lines
    /// are parsed and merged into the observable/tracked-Pauli definitions.
    ///
    /// # Errors
    ///
    /// Returns an error if a PECOS metadata line is malformed.
    pub fn apply_pecos_dem_metadata(
        &mut self,
        dem_text: &str,
    ) -> Result<(), PecosDemMetadataError> {
        for line in dem_text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if line.starts_with("pecos_observable") || line.starts_with("pecos_tracked_pauli") {
                self.apply_dem_output_metadata(parse_pecos_dem_metadata_line(line)?);
            } else if line.starts_with("pecos_") {
                return Err(PecosDemMetadataError::new(format!(
                    "unsupported PECOS DEM extension line: {line}"
                )));
            }
        }
        Ok(())
    }

    /// Returns a copy of this model with PECOS metadata from extended DEM text
    /// applied.
    ///
    /// # Errors
    ///
    /// See [`Self::apply_pecos_dem_metadata`].
    pub fn with_pecos_dem_metadata(
        mut self,
        dem_text: &str,
    ) -> Result<Self, PecosDemMetadataError> {
        self.apply_pecos_dem_metadata(dem_text)?;
        Ok(self)
    }

    fn apply_dem_output_metadata(&mut self, target: DemOutput) {
        if target.is_tracked_pauli() {
            self.apply_tracked_pauli_metadata(target);
        } else {
            self.apply_observable_metadata(target);
        }
    }

    fn apply_observable_metadata(&mut self, mut target: DemOutput) {
        target.kind.get_or_insert(DemOutputKind::Observable);
        if let Some(existing) = self
            .observables
            .iter_mut()
            .find(|existing| existing.id == target.id)
        {
            *existing = target;
        } else {
            self.add_observable(target);
        }
    }

    fn apply_tracked_pauli_metadata(&mut self, mut target: DemOutput) {
        target.kind = Some(DemOutputKind::TrackedPauli);
        if let Some(existing) = self
            .tracked_paulis
            .iter_mut()
            .find(|existing| existing.id == target.id)
        {
            *existing = target;
        } else {
            self.add_tracked_pauli(target);
        }
    }

    /// Returns debug info about contributions for a specific mechanism.
    ///
    /// Format: One line per contribution showing source type and probability.
    #[must_use]
    pub fn contributions_for_mechanism(&self, detectors: &[u32]) -> String {
        let target_dets: SmallVec<[u32; 4]> = detectors.iter().copied().collect();
        let mut lines = Vec::new();

        for contrib in &self.contributions {
            if contrib.effect.detectors == target_dets && contrib.effect.dem_outputs.is_empty() {
                let source_type = match &contrib.source_type {
                    FaultSourceType::Direct => "Direct".to_string(),
                    FaultSourceType::DirectOneSidedComponent => {
                        "DirectOneSidedComponent".to_string()
                    }
                    FaultSourceType::YDecomposed {
                        x_detectors,
                        x_dem_outputs,
                        z_detectors,
                        z_dem_outputs,
                    } => {
                        let x_dets: Vec<_> = x_detectors.iter().map(|d| format!("D{d}")).collect();
                        let z_dets: Vec<_> = z_detectors.iter().map(|d| format!("D{d}")).collect();
                        let x_outputs: Vec<_> =
                            x_dem_outputs.iter().map(|l| format!("L{l}")).collect();
                        let z_outputs: Vec<_> =
                            z_dem_outputs.iter().map(|l| format!("L{l}")).collect();
                        format!(
                            "YDecomposed(X=[{}{}], Z=[{}{}])",
                            x_dets.join(" "),
                            if x_outputs.is_empty() {
                                String::new()
                            } else {
                                format!(" {}", x_outputs.join(" "))
                            },
                            z_dets.join(" "),
                            if z_outputs.is_empty() {
                                String::new()
                            } else {
                                format!(" {}", z_outputs.join(" "))
                            }
                        )
                    }
                };
                lines.push(format!(
                    "  {}: prob={:.6}",
                    source_type, contrib.probability
                ));
            }
        }

        if lines.is_empty() {
            format!("No contributions found for {detectors:?}")
        } else {
            format!(
                "Contributions for {:?} ({} total):\n{}",
                detectors,
                lines.len(),
                lines.join("\n")
            )
        }
    }

    /// Returns all contributions matching a full detector/DEM-output effect.
    #[must_use]
    pub fn contributions_for_effect(
        &self,
        detectors: &[u32],
        dem_outputs: &[u32],
    ) -> Vec<FaultContribution> {
        let target =
            FaultMechanism::from_unsorted(detectors.iter().copied(), dem_outputs.iter().copied());
        self.contributions
            .iter()
            .filter(|contrib| contrib.effect == target)
            .cloned()
            .collect()
    }

    /// Returns debug info about all unique contribution effects.
    #[must_use]
    pub fn all_contribution_effects(&self) -> String {
        use std::collections::BTreeMap;

        let mut by_effect: BTreeMap<String, (usize, f64)> = BTreeMap::new();

        for contrib in &self.contributions {
            let det_str: Vec<_> = contrib
                .effect
                .detectors
                .iter()
                .map(|d| format!("D{d}"))
                .collect();
            let log_str: Vec<_> = contrib
                .effect
                .dem_outputs
                .iter()
                .map(|l| format!("L{l}"))
                .collect();
            let key = format!(
                "{}{}",
                det_str.join(" "),
                if log_str.is_empty() {
                    String::new()
                } else {
                    format!(" {}", log_str.join(" "))
                }
            );

            by_effect
                .entry(key)
                .and_modify(|(count, prob)| {
                    *count += 1;
                    *prob += contrib.probability;
                })
                .or_insert((1, contrib.probability));
        }

        let lines: Vec<_> = by_effect
            .into_iter()
            .map(|(effect, (count, prob))| {
                format!("  {effect}: {count} contrib(s), total_prob={prob:.6}")
            })
            .collect();

        format!(
            "Total contributions: {}\nUnique effects: {}\n{}",
            self.contributions.len(),
            lines.len(),
            lines.join("\n")
        )
    }

    /// Returns structured summaries for all unique contribution effects.
    #[must_use]
    pub fn contribution_effect_summaries(&self) -> Vec<ContributionEffectSummary> {
        let mut by_effect: BTreeMap<FaultMechanism, ContributionEffectSummary> = BTreeMap::new();

        for contrib in &self.contributions {
            let summary = by_effect.entry(contrib.effect.clone()).or_insert_with(|| {
                ContributionEffectSummary {
                    effect: contrib.effect.clone(),
                    num_contributions: 0,
                    total_probability: 0.0,
                    direct_count: 0,
                    direct_probability: 0.0,
                    y_decomposed_count: 0,
                    y_decomposed_probability: 0.0,
                    graphlike_decomposable_count: 0,
                }
            });

            summary.num_contributions += 1;
            summary.total_probability += contrib.probability;
            match contrib.source_type {
                FaultSourceType::Direct | FaultSourceType::DirectOneSidedComponent => {
                    summary.direct_count += 1;
                    summary.direct_probability += contrib.probability;
                }
                FaultSourceType::YDecomposed { .. } => {
                    summary.y_decomposed_count += 1;
                    summary.y_decomposed_probability += contrib.probability;
                }
            }
        }

        for summary in by_effect.values_mut() {
            if summary.effect.dem_outputs.is_empty() && summary.effect.detectors.len() == 2 {
                summary.graphlike_decomposable_count = self.graphlike_decomposable_count(
                    summary.effect.detectors[0],
                    summary.effect.detectors[1],
                );
            }
        }

        by_effect.into_values().collect()
    }

    /// Returns structured summaries of contribution render buckets before regrouping.
    ///
    /// This mirrors the per-contribution render pass used by
    /// `to_string_decomposed()`, but keeps the original effect attached so callers
    /// can see which source families collapse onto the same rendered targets.
    #[must_use]
    pub fn contribution_render_summaries(&self) -> Vec<ContributionRenderSummary> {
        self.contribution_render_summaries_with_two_detector_direct_policy(
            TwoDetectorDirectRenderPolicy::KeepDirect,
        )
    }

    /// Returns structured summaries of contribution render buckets before
    /// regrouping, using an explicit policy for direct 2-detector rendering.
    #[must_use]
    pub fn contribution_render_summaries_with_two_detector_direct_policy(
        &self,
        two_detector_direct_policy: TwoDetectorDirectRenderPolicy,
    ) -> Vec<ContributionRenderSummary> {
        #[derive(Default)]
        struct Accumulator {
            num_contributions: usize,
            total_probability: f64,
            combined_probability: f64,
            source_type_counts: BTreeMap<String, usize>,
            source_type_probabilities: BTreeMap<String, f64>,
            direct_source_family_counts: BTreeMap<String, usize>,
            direct_source_family_probabilities: BTreeMap<String, f64>,
        }

        fn source_type_label(source_type: &FaultSourceType) -> &'static str {
            match source_type {
                FaultSourceType::Direct => "Direct",
                FaultSourceType::DirectOneSidedComponent => "DirectOneSidedComponent",
                FaultSourceType::YDecomposed { .. } => "YDecomposed",
            }
        }

        fn direct_source_family_label(family: DirectSourceFamily) -> &'static str {
            match family {
                DirectSourceFamily::SingleLocation => "SingleLocation",
                DirectSourceFamily::SingleLocationY => "SingleLocationY",
                DirectSourceFamily::TwoLocationPlainY => "TwoLocationPlainY",
                DirectSourceFamily::TwoLocationComponent => "TwoLocationComponent",
                DirectSourceFamily::TwoLocationOneSidedComponent => "TwoLocationOneSidedComponent",
                DirectSourceFamily::Other => "Other",
            }
        }

        let graphlike_set = self.collect_graphlike_mechanisms();
        let graphlike_index = GraphlikeDecompositionIndex::new(&graphlike_set);
        let mut rendered_targets_cache: BTreeMap<(FaultMechanism, FaultSourceType), String> =
            BTreeMap::new();
        let mut by_render: BTreeMap<(FaultMechanism, String), Accumulator> = BTreeMap::new();

        for contrib in &self.contributions {
            if contrib.effect.is_empty() || contrib.probability <= 0.0 {
                continue;
            }

            let rendered_targets = Self::contribution_targets(
                contrib,
                &graphlike_index,
                None,
                two_detector_direct_policy,
                &mut rendered_targets_cache,
            );
            let acc = by_render
                .entry((contrib.effect.clone(), rendered_targets))
                .or_default();
            acc.num_contributions += 1;
            acc.total_probability += contrib.probability;
            acc.combined_probability =
                combine_independent_probs(acc.combined_probability, contrib.probability);

            let source_label = source_type_label(&contrib.source_type).to_string();
            *acc.source_type_counts
                .entry(source_label.clone())
                .or_insert(0) += 1;
            *acc.source_type_probabilities
                .entry(source_label)
                .or_insert(0.0) += contrib.probability;

            if let Some(family) = contrib.direct_source_family {
                let family_label = direct_source_family_label(family).to_string();
                *acc.direct_source_family_counts
                    .entry(family_label.clone())
                    .or_insert(0) += 1;
                *acc.direct_source_family_probabilities
                    .entry(family_label)
                    .or_insert(0.0) += contrib.probability;
            }
        }

        by_render
            .into_iter()
            .map(
                |((effect, rendered_targets), acc)| ContributionRenderSummary {
                    effect,
                    rendered_targets,
                    num_contributions: acc.num_contributions,
                    total_probability: acc.total_probability,
                    combined_probability: acc.combined_probability,
                    source_type_counts: acc.source_type_counts,
                    source_type_probabilities: acc.source_type_probabilities,
                    direct_source_family_counts: acc.direct_source_family_counts,
                    direct_source_family_probabilities: acc.direct_source_family_probabilities,
                },
            )
            .collect()
    }

    /// Returns per-contribution render records before final regrouping.
    ///
    /// This mirrors the same contribution render pass used by
    /// `to_string_decomposed()`, but keeps one output row per tracked
    /// contribution instead of aggregating by `(effect, rendered_targets)`.
    #[must_use]
    pub fn contribution_render_records(&self) -> Vec<ContributionRenderRecord> {
        self.contribution_render_records_with_two_detector_direct_policy(
            TwoDetectorDirectRenderPolicy::KeepDirect,
        )
    }

    /// Returns per-contribution render records before final regrouping, using
    /// an explicit policy for direct 2-detector rendering.
    #[must_use]
    pub fn contribution_render_records_with_two_detector_direct_policy(
        &self,
        two_detector_direct_policy: TwoDetectorDirectRenderPolicy,
    ) -> Vec<ContributionRenderRecord> {
        let graphlike_set = self.collect_graphlike_mechanisms();
        let graphlike_index = GraphlikeDecompositionIndex::new(&graphlike_set);
        let mut rendered_targets_cache: BTreeMap<(FaultMechanism, FaultSourceType), String> =
            BTreeMap::new();
        let mut records = Vec::new();

        for contrib in &self.contributions {
            if contrib.effect.is_empty() || contrib.probability <= 0.0 {
                continue;
            }

            let (rendered_targets, render_strategy, recorded_component_targets) =
                Self::contribution_render_details(
                    contrib,
                    &graphlike_index,
                    None,
                    two_detector_direct_policy,
                    &mut rendered_targets_cache,
                );
            records.push(ContributionRenderRecord {
                rendered_targets,
                render_strategy,
                recorded_component_targets,
                contribution: contrib.clone(),
            });
        }

        records
    }

    /// Adds a direct error contribution (X or Z channel).
    ///
    /// Direct contributions are output as direct forms (e.g., "D0 D1") rather than
    /// decomposed forms. Use this for X and Z error channels.
    ///
    /// Requires source tracking to be enabled.
    pub fn add_direct_contribution(&mut self, effect: FaultMechanism, probability: f64) {
        if effect.is_empty() || probability <= 0.0 {
            return;
        }
        self.contributions
            .push(FaultContribution::direct(effect, probability));
    }

    /// Adds a direct error contribution with source metadata.
    pub(crate) fn add_direct_contribution_with_source(
        &mut self,
        effect: FaultMechanism,
        probability: f64,
        source: SourceMetadata<'_, usize>,
    ) {
        if effect.is_empty() || probability <= 0.0 {
            return;
        }
        let location_indices = convert_location_indices(source.location_indices);
        self.contributions
            .push(FaultContribution::direct_with_source(
                effect,
                probability,
                SourceMetadata::new(
                    &location_indices,
                    source.paulis,
                    source.gate_types,
                    source.before_flags,
                ),
            ));
    }

    /// Adds a direct error contribution with source metadata and per-location
    /// component effects.
    pub(crate) fn add_direct_contribution_with_source_components(
        &mut self,
        effect: FaultMechanism,
        probability: f64,
        source: SourceMetadata<'_, usize>,
        components: DirectSourceComponents<'_>,
    ) {
        if effect.is_empty() || probability <= 0.0 {
            return;
        }
        let location_indices = convert_location_indices(source.location_indices);
        self.contributions
            .push(FaultContribution::direct_with_source_components(
                effect,
                probability,
                SourceMetadata::new(
                    &location_indices,
                    source.paulis,
                    source.gate_types,
                    source.before_flags,
                ),
                components,
            ));
    }

    /// Adds a Y-decomposed error contribution.
    ///
    /// Y-decomposed contributions are output as decomposed forms (e.g., "D0 ^ D1")
    /// when both X and Z components are graphlike. The combined effect is stored
    /// along with the X and Z components for proper decomposition output.
    ///
    /// Requires source tracking to be enabled.
    pub fn add_y_decomposed_contribution(
        &mut self,
        x_effect: &FaultMechanism,
        z_effect: &FaultMechanism,
        probability: f64,
    ) {
        if probability <= 0.0 {
            return;
        }

        // Y-containing channels are always classified as YDecomposed for source
        // tracking purposes so they can contribute to decomposed output forms
        // regardless of component structure.
        //
        // The combined effect is X XOR Z. When one is empty, the combined effect
        // is just the non-empty one.
        let combined = x_effect.xor(z_effect);
        if combined.is_empty() {
            // If combined is empty (X and Z are equal), no contribution
            return;
        }

        // If one branch is empty, the Y-containing source has the same net effect as
        // the non-empty branch and should be tracked as a direct source.
        if x_effect.is_empty() || z_effect.is_empty() {
            self.add_direct_contribution(combined, probability);
            return;
        }

        // Otherwise record as YDecomposed. The distinction between Direct and
        // YDecomposed affects output form selection.
        self.contributions.push(FaultContribution::y_decomposed(
            combined,
            x_effect,
            z_effect,
            probability,
        ));
    }

    /// Adds a Y-decomposed error contribution with source metadata.
    pub(crate) fn add_y_decomposed_contribution_with_source(
        &mut self,
        x_effect: &FaultMechanism,
        z_effect: &FaultMechanism,
        probability: f64,
        source: SourceMetadata<'_, usize>,
    ) {
        if probability <= 0.0 {
            return;
        }

        let combined = x_effect.xor(z_effect);
        if combined.is_empty() {
            return;
        }

        if x_effect.is_empty() || z_effect.is_empty() {
            self.add_direct_contribution_with_source(combined, probability, source);
            return;
        }

        let location_indices = convert_location_indices(source.location_indices);
        self.contributions
            .push(FaultContribution::y_decomposed_with_source(
                combined,
                x_effect,
                z_effect,
                probability,
                SourceMetadata::new(
                    &location_indices,
                    source.paulis,
                    source.gate_types,
                    source.before_flags,
                ),
            ));
    }

    /// Marks a 2-detector mechanism as having a graphlike decomposable source.
    ///
    /// A source is "graphlike decomposable" if both component effects (P1I and IP2)
    /// are non-empty and graphlike (≤2 detectors each). This is used to determine
    /// the output format for representation diversity:
    /// - ≥2 sources: 3 forms (direct + 2 decomposed)
    /// - 1 source: 2 forms (decomposed only)
    /// - 0 sources: 1 form (direct only)
    ///
    /// This should be called from the builder when processing 2-qubit gate channels
    /// where both component effects are graphlike.
    /// Returns grouped mechanisms as (probability, `detector_ids`, `observable_ids`) tuples.
    ///
    /// Combines contributions with the same effect using the XOR probability formula.
    /// Also returns detector coordinate map. This is the structured equivalent of
    /// `to_string()` — same data, no string intermediary.
    #[must_use]
    pub fn to_mechanisms(&self) -> (Vec<MechanismTuple>, Vec<DetectorCoordinateTuple>) {
        // Group contributions by effect
        let mut by_effect: BTreeMap<FaultMechanism, f64> = BTreeMap::new();
        for contrib in &self.contributions {
            by_effect
                .entry(contrib.effect.standard_effect())
                .and_modify(|p| *p = combine_independent_probs(*p, contrib.probability))
                .or_insert(contrib.probability);
        }

        let mechanisms: Vec<(f64, Vec<u32>, Vec<u32>)> = by_effect
            .into_iter()
            .filter(|(effect, prob)| !effect.is_standard_empty() && *prob > 0.0)
            .map(|(effect, prob)| (prob, effect.detectors.to_vec(), effect.dem_outputs.to_vec()))
            .collect();

        let coords: Vec<(u32, Vec<f64>)> = self
            .detectors
            .iter()
            .filter_map(|det| det.coords.map(|[x, y, z]| (det.id, vec![x, y, z])))
            .collect();

        (mechanisms, coords)
    }

    pub fn mark_graphlike_decomposable(&mut self, d0: u32, d1: u32) {
        let key = if d0 < d1 { (d0, d1) } else { (d1, d0) };
        *self.graphlike_decomposable_counts.entry(key).or_insert(0) += 1;
    }

    /// Merge contributions and graphlike counts from another DEM.
    /// Used for parallelized DEM construction.
    pub fn merge_contributions_from(&mut self, other: Self) {
        self.contributions.extend(other.contributions);
        for (key, count) in other.graphlike_decomposable_counts {
            *self.graphlike_decomposable_counts.entry(key).or_insert(0) += count;
        }
    }

    /// Returns the number of graphlike decomposable sources for a 2-detector mechanism.
    #[must_use]
    pub fn graphlike_decomposable_count(&self, d0: u32, d1: u32) -> u32 {
        let key = if d0 < d1 { (d0, d1) } else { (d1, d0) };
        self.graphlike_decomposable_counts
            .get(&key)
            .copied()
            .unwrap_or(0)
    }

    /// Adds a detector definition.
    pub fn add_detector(&mut self, detector: DetectorDef) {
        self.detectors.push(detector);
    }

    /// Adds a non-detector DEM output definition.
    ///
    /// Observables are stored in the standard `L<n>` namespace. Tracked
    /// Paulis are stored in PECOS metadata with a separate ID space.
    pub fn add_dem_output(&mut self, output: DemOutput) {
        if output.is_tracked_pauli() {
            self.add_tracked_pauli(output);
        } else {
            self.add_observable(output);
        }
    }

    /// Adds a standard DEM observable (`L<n>`) definition.
    pub fn add_observable(&mut self, mut observable: DemOutput) {
        observable.kind = Some(DemOutputKind::Observable);
        if let Some(existing) = self
            .observables
            .iter_mut()
            .find(|existing| existing.id == observable.id)
        {
            Self::merge_observable_definition(existing, observable);
            return;
        }
        self.observables.push(observable);
    }

    fn merge_observable_definition(existing: &mut DemOutput, incoming: DemOutput) {
        existing.kind = Some(DemOutputKind::Observable);
        merge_record_parity(&mut existing.records, incoming.records);

        if let Some(incoming_pauli) = incoming.pauli {
            if let Some(existing_pauli) = &existing.pauli {
                debug_assert_eq!(
                    existing_pauli, &incoming_pauli,
                    "conflicting Pauli metadata for observable L{}",
                    existing.id
                );
            } else {
                existing.pauli = Some(incoming_pauli);
            }
        }

        if let Some(incoming_label) = incoming.label {
            if let Some(existing_label) = &existing.label {
                debug_assert_eq!(
                    existing_label, &incoming_label,
                    "conflicting labels for observable L{}",
                    existing.id
                );
            } else {
                existing.label = Some(incoming_label);
            }
        }
    }

    /// Adds a PECOS tracked Pauli definition.
    pub fn add_tracked_pauli(&mut self, mut tracked_pauli: DemOutput) {
        tracked_pauli.kind = Some(DemOutputKind::TrackedPauli);
        self.tracked_paulis.push(tracked_pauli);
    }

    /// Converts the DEM to a string in standard DEM format.
    ///
    /// Each fault mechanism is output with its total probability, with no
    /// splitting into decomposed forms.
    ///
    /// Requires source tracking to be enabled and contributions to be populated.
    /// Use `build_with_source_tracking()` to create a DEM with contributions.
    #[must_use]
    #[allow(clippy::inherent_to_string)] // Intentional: we have two string formats
    pub fn to_string(&self) -> String {
        let mut lines = Vec::new();

        // Add detector coordinate annotations
        for det in &self.detectors {
            if let Some([x, y, z]) = det.coords {
                lines.push(format!("detector({x}, {y}, {z}) D{}", det.id));
            } else {
                lines.push(format!("detector D{}", det.id));
            }
        }

        // Add standard observable annotations
        for obs in &self.observables {
            lines.push(format!("logical_observable L{}", obs.id));
        }

        // Group contributions by effect, combining probabilities using XOR formula
        // (errors toggle detector bits, so two errors on same detector cancel)
        let mut by_effect: BTreeMap<FaultMechanism, f64> = BTreeMap::new();
        for contrib in &self.contributions {
            by_effect
                .entry(contrib.effect.standard_effect())
                .and_modify(|p| *p = combine_independent_probs(*p, contrib.probability))
                .or_insert(contrib.probability);
        }

        // Output each mechanism with its total probability
        for (effect, total_prob) in by_effect {
            if effect.is_standard_empty() || total_prob <= 0.0 {
                continue;
            }

            let targets = format_mechanism_targets(&effect);
            if !targets.is_empty() {
                lines.push(format!(
                    "error({}) {}",
                    format_probability(total_prob),
                    targets
                ));
            }
        }

        lines.join("\n")
    }

    fn collect_singleton_index(&self) -> SingletonDecompositionIndex {
        SingletonDecompositionIndex::from_contributions(&self.contributions)
    }

    fn maximally_decompose_graphlike_effect(
        effect: &FaultMechanism,
        singleton_set: &SingletonDecompositionIndex,
    ) -> Vec<FaultMechanism> {
        find_singleton_decomposition(effect, singleton_set)
            .filter(|parts| !parts.is_empty())
            .unwrap_or_else(|| vec![effect.clone()])
    }

    fn maybe_maximally_decompose_parts(
        parts: Vec<FaultMechanism>,
        singleton_set: Option<&SingletonDecompositionIndex>,
    ) -> Vec<FaultMechanism> {
        let Some(singleton_set) = singleton_set else {
            return parts;
        };

        let mut out = Vec::new();
        for part in parts {
            if part.is_graphlike() {
                out.extend(Self::maximally_decompose_graphlike_effect(
                    &part,
                    singleton_set,
                ));
            } else {
                out.push(part);
            }
        }
        out
    }

    fn recorded_component_targets(
        contrib: &FaultContribution,
        singleton_set: Option<&SingletonDecompositionIndex>,
    ) -> Option<String> {
        let (first, second) = contrib.direct_component_effects()?;
        let targets = Self::maybe_maximally_decompose_parts(
            [first, second]
                .into_iter()
                .filter(|part| !part.is_empty())
                .collect(),
            singleton_set,
        )
        .iter()
        .map(format_mechanism_targets)
        .filter(|targets| !targets.is_empty())
        .collect::<Vec<_>>()
        .join(" ^ ");
        if targets.is_empty() {
            None
        } else {
            Some(targets)
        }
    }

    fn two_detector_direct_targets(
        effect: &FaultMechanism,
        singleton_set: Option<&SingletonDecompositionIndex>,
    ) -> String {
        Self::maybe_maximally_decompose_parts(vec![effect.clone()], singleton_set)
            .iter()
            .map(format_mechanism_targets)
            .collect::<Vec<_>>()
            .join(" ^ ")
    }

    fn contribution_render_details(
        contrib: &FaultContribution,
        graphlike_index: &GraphlikeDecompositionIndex,
        singleton_set: Option<&SingletonDecompositionIndex>,
        two_detector_direct_policy: TwoDetectorDirectRenderPolicy,
        cache: &mut BTreeMap<(FaultMechanism, FaultSourceType), String>,
    ) -> (String, ContributionRenderStrategy, Option<String>) {
        let recorded_component_targets = Self::recorded_component_targets(contrib, singleton_set);
        let key = (contrib.effect.clone(), contrib.source_type.clone());
        if let Some(cached) = cache.get(&key) {
            let strategy = if contrib.decomposition_components().is_some() {
                ContributionRenderStrategy::SourceComponents
            } else if contrib.effect.num_detectors() == 2 && contrib.effect.dem_outputs.is_empty() {
                let direct_targets =
                    Self::two_detector_direct_targets(&contrib.effect, singleton_set);
                if matches!(
                    two_detector_direct_policy,
                    TwoDetectorDirectRenderPolicy::PreferRecordedComponents
                ) && recorded_component_targets.as_deref() == Some(cached.as_str())
                    && cached != &direct_targets
                {
                    ContributionRenderStrategy::RecordedComponents
                } else {
                    ContributionRenderStrategy::TwoDetectorDirect
                }
            } else if contrib.effect.is_hyperedge() {
                ContributionRenderStrategy::HyperedgeGraphlike
            } else {
                ContributionRenderStrategy::EffectDirect
            };
            return (cached.clone(), strategy, recorded_component_targets);
        }

        let effect = contrib.effect.standard_effect();
        let (targets, strategy) = if let Some((x_effect, z_effect)) =
            contrib.decomposition_components()
        {
            let x_graphlike = x_effect.is_empty() || x_effect.is_graphlike();
            let z_graphlike = z_effect.is_empty() || z_effect.is_graphlike();

            if !x_effect.is_empty() && !z_effect.is_empty() && x_graphlike && z_graphlike {
                let x_parts =
                    Self::maybe_maximally_decompose_parts(vec![x_effect.clone()], singleton_set);
                let z_parts =
                    Self::maybe_maximally_decompose_parts(vec![z_effect.clone()], singleton_set);
                let targets = x_parts
                    .iter()
                    .chain(z_parts.iter())
                    .map(format_mechanism_targets)
                    .filter(|targets| !targets.is_empty())
                    .collect::<Vec<_>>()
                    .join(" ^ ");
                let targets = if targets.is_empty() {
                    String::new()
                } else {
                    targets
                };
                (targets, ContributionRenderStrategy::SourceComponents)
            } else if effect.num_detectors() == 2 && effect.dem_outputs.is_empty() {
                let direct_targets = Self::two_detector_direct_targets(&effect, singleton_set);
                if matches!(
                    two_detector_direct_policy,
                    TwoDetectorDirectRenderPolicy::PreferRecordedComponents
                ) {
                    if let Some(component_targets) = recorded_component_targets.as_ref() {
                        if component_targets == &direct_targets {
                            (
                                direct_targets,
                                ContributionRenderStrategy::TwoDetectorDirect,
                            )
                        } else {
                            (
                                component_targets.clone(),
                                ContributionRenderStrategy::RecordedComponents,
                            )
                        }
                    } else {
                        (
                            direct_targets,
                            ContributionRenderStrategy::TwoDetectorDirect,
                        )
                    }
                } else {
                    (
                        direct_targets,
                        ContributionRenderStrategy::TwoDetectorDirect,
                    )
                }
            } else if effect.is_hyperedge() {
                if let Some(decomp) = graphlike_index.find_hyperedge_decomposition(&effect) {
                    (
                        Self::maybe_maximally_decompose_parts(decomp, singleton_set)
                            .iter()
                            .map(format_mechanism_targets)
                            .collect::<Vec<_>>()
                            .join(" ^ "),
                        ContributionRenderStrategy::HyperedgeGraphlike,
                    )
                } else {
                    (
                        format_mechanism_targets(&effect),
                        ContributionRenderStrategy::EffectDirect,
                    )
                }
            } else {
                (
                    Self::maybe_maximally_decompose_parts(vec![effect.clone()], singleton_set)
                        .iter()
                        .map(format_mechanism_targets)
                        .collect::<Vec<_>>()
                        .join(" ^ "),
                    ContributionRenderStrategy::EffectDirect,
                )
            }
        } else if effect.num_detectors() == 2 && effect.dem_outputs.is_empty() {
            let direct_targets = Self::two_detector_direct_targets(&effect, singleton_set);
            if matches!(
                two_detector_direct_policy,
                TwoDetectorDirectRenderPolicy::PreferRecordedComponents
            ) {
                if let Some(component_targets) = recorded_component_targets.as_ref() {
                    if component_targets == &direct_targets {
                        (
                            direct_targets,
                            ContributionRenderStrategy::TwoDetectorDirect,
                        )
                    } else {
                        (
                            component_targets.clone(),
                            ContributionRenderStrategy::RecordedComponents,
                        )
                    }
                } else {
                    (
                        direct_targets,
                        ContributionRenderStrategy::TwoDetectorDirect,
                    )
                }
            } else {
                (
                    direct_targets,
                    ContributionRenderStrategy::TwoDetectorDirect,
                )
            }
        } else if effect.is_hyperedge() {
            if let Some(decomp) = graphlike_index.find_hyperedge_decomposition(&effect) {
                (
                    Self::maybe_maximally_decompose_parts(decomp, singleton_set)
                        .iter()
                        .map(format_mechanism_targets)
                        .collect::<Vec<_>>()
                        .join(" ^ "),
                    ContributionRenderStrategy::HyperedgeGraphlike,
                )
            } else {
                (
                    format_mechanism_targets(&effect),
                    ContributionRenderStrategy::EffectDirect,
                )
            }
        } else {
            (
                Self::maybe_maximally_decompose_parts(vec![effect.clone()], singleton_set)
                    .iter()
                    .map(format_mechanism_targets)
                    .collect::<Vec<_>>()
                    .join(" ^ "),
                ContributionRenderStrategy::EffectDirect,
            )
        };

        cache.insert(key, targets.clone());
        (targets, strategy, recorded_component_targets)
    }

    fn contribution_targets(
        contrib: &FaultContribution,
        graphlike_index: &GraphlikeDecompositionIndex,
        singleton_set: Option<&SingletonDecompositionIndex>,
        two_detector_direct_policy: TwoDetectorDirectRenderPolicy,
        cache: &mut BTreeMap<(FaultMechanism, FaultSourceType), String>,
    ) -> String {
        Self::contribution_render_details(
            contrib,
            graphlike_index,
            singleton_set,
            two_detector_direct_policy,
            cache,
        )
        .0
    }

    /// Converts the DEM to decomposed text using source tracking.
    ///
    /// Fault mechanisms are split into direct and decomposed forms based on
    /// their source types (X/Z vs Y errors).
    ///
    /// For each unique detector effect:
    /// - Sum probabilities from direct sources (X, Z) -> output as direct form
    /// - Y-decomposed sources -> output as decomposed form (X ^ Z)
    ///
    /// Converts the DEM to a string with decomposed representations.
    ///
    /// Requires source tracking to be enabled and contributions to be populated.
    ///
    /// For 2-detector mechanisms Di Dj:
    /// - Output the direct graphlike form `Di Dj`.
    /// - Avoid introducing synthetic `Di L0 ^ Dj L0` cancellation variants,
    ///   because the edge is already graphlike and extra L0 terms can change
    ///   decoder behavior without adding new information.
    ///
    /// Hyperedges (3+ detectors) are decomposed into graphlike forms when
    /// possible. Mechanisms with up to 2 detectors are already graphlike even
    /// when they carry multiple DEM outputs.
    #[must_use]
    fn to_string_decomposed_inner(
        &self,
        maximal_decomposition: bool,
        two_detector_direct_policy: TwoDetectorDirectRenderPolicy,
    ) -> String {
        let mut lines = Vec::new();

        // Add detector coordinate annotations
        for det in &self.detectors {
            if let Some([x, y, z]) = det.coords {
                lines.push(format!("detector({x}, {y}, {z}) D{}", det.id));
            } else {
                lines.push(format!("detector D{}", det.id));
            }
        }

        // Add standard observable annotations
        for obs in &self.observables {
            lines.push(format!("logical_observable L{}", obs.id));
        }

        let graphlike_set = self.collect_graphlike_mechanisms();
        let graphlike_index = GraphlikeDecompositionIndex::new(&graphlike_set);
        let singleton_set = maximal_decomposition.then(|| self.collect_singleton_index());
        let mut by_targets: BTreeMap<String, f64> = BTreeMap::new();
        let mut rendered_targets_cache: BTreeMap<(FaultMechanism, FaultSourceType), String> =
            BTreeMap::new();

        let mut add_targets = |targets: String, probability: f64| {
            if targets.is_empty() || probability <= 0.0 {
                return;
            }
            by_targets
                .entry(targets)
                .and_modify(|p| *p = combine_independent_probs(*p, probability))
                .or_insert(probability);
        };

        // Process each tracked contribution individually, then regroup identical
        // decomposed outputs. Rewriting each error class before merging keeps
        // source-aware decompositions stable.
        for contrib in &self.contributions {
            if contrib.effect.is_empty() || contrib.probability <= 0.0 {
                continue;
            }
            let targets = Self::contribution_targets(
                contrib,
                &graphlike_index,
                singleton_set.as_ref(),
                two_detector_direct_policy,
                &mut rendered_targets_cache,
            );
            add_targets(targets, contrib.probability);
        }

        for (targets, total_prob) in by_targets {
            if !targets.is_empty() && total_prob > 0.0 {
                lines.push(format!(
                    "error({}) {}",
                    format_probability(total_prob),
                    targets
                ));
            }
        }

        lines.join("\n")
    }

    #[must_use]
    pub fn to_string_decomposed(&self) -> String {
        self.to_string_decomposed_inner(false, TwoDetectorDirectRenderPolicy::KeepDirect)
    }

    /// Converts the DEM to decomposed format with an explicit direct-2det
    /// rendering policy.
    #[must_use]
    pub fn to_string_decomposed_with_two_detector_direct_policy(
        &self,
        two_detector_direct_policy: TwoDetectorDirectRenderPolicy,
    ) -> String {
        self.to_string_decomposed_inner(false, two_detector_direct_policy)
    }

    /// Converts the DEM to a maximally decomposed graphlike form when possible.
    ///
    /// This further rewrites graphlike 2-detector effects into XORs of
    /// standalone singleton detector effects whenever those singleton effects
    /// already exist in the DEM.
    ///
    /// This is mainly useful for representation inspection or compatibility
    /// experiments. It is not generally the preferred MWPM input because
    /// replacing pair edges with singleton-heavy structure can degrade the
    /// resulting matching graph.
    #[must_use]
    pub fn to_string_decomposed_maximally(&self) -> String {
        self.to_string_decomposed_inner(true, TwoDetectorDirectRenderPolicy::KeepDirect)
    }

    /// Converts the DEM to a maximally decomposed graphlike form with an
    /// explicit direct-2det rendering policy.
    #[must_use]
    pub fn to_string_decomposed_maximally_with_two_detector_direct_policy(
        &self,
        two_detector_direct_policy: TwoDetectorDirectRenderPolicy,
    ) -> String {
        self.to_string_decomposed_inner(true, two_detector_direct_policy)
    }

    /// Collects all graphlike mechanisms from contributions.
    ///
    /// Returns a set of mechanisms with ≤2 detectors,
    /// which can be used as components for hyperedge decomposition.
    fn collect_graphlike_mechanisms(&self) -> BTreeSet<FaultMechanism> {
        let mut graphlike = BTreeSet::new();
        for contrib in &self.contributions {
            let standard = contrib.effect.standard_effect();
            if !standard.is_standard_empty() && standard.is_graphlike() {
                graphlike.insert(standard);
            }
        }
        graphlike
    }
}

impl Default for DetectorErrorModel {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Probability Combination
// ============================================================================

/// Combines two independent error probabilities.
///
/// For independent errors with probabilities p1 and p2, the probability
/// that exactly one error occurs is: p1*(1-p2) + p2*(1-p1).
///
/// This is the correct formula for combining probabilities when the same
/// fault mechanism can be triggered by multiple independent error sources.
#[inline]
#[must_use]
pub fn combine_probabilities(p1: f64, p2: f64) -> f64 {
    p1 * (1.0 - p2) + p2 * (1.0 - p1)
}

/// Formats an fault mechanism's targets as a string (e.g., "D0 D1 L0").
fn format_mechanism_targets(mechanism: &FaultMechanism) -> String {
    let mut targets = Vec::new();
    for &det in &mechanism.detectors {
        targets.push(format!("D{det}"));
    }
    for &dem_output in &mechanism.dem_outputs {
        targets.push(format!("L{dem_output}"));
    }
    targets.join(" ")
}

/// Formats a PECOS DEM mechanism's targets, including tracked Pauli `TP<n>` outputs.
fn format_pecos_mechanism_targets(mechanism: &FaultMechanism) -> String {
    let mut targets = Vec::new();
    for &det in &mechanism.detectors {
        targets.push(format!("D{det}"));
    }
    for &dem_output in &mechanism.dem_outputs {
        targets.push(format!("L{dem_output}"));
    }
    for &tracked_pauli in &mechanism.tracked_paulis {
        targets.push(format!("TP{tracked_pauli}"));
    }
    targets.join(" ")
}

/// Combines two independent error probabilities.
///
/// For two independent errors with probabilities p1 and p2, the combined
/// probability of having an odd number of errors (i.e., the XOR of the effects) is:
/// `p_combined` = p1*(1-p2) + p2*(1-p1)
fn combine_independent_probs(p1: f64, p2: f64) -> f64 {
    // For DEM probability aggregation, we use XOR combination because
    // errors toggle detector bits - if two errors both flip the same detector,
    // they cancel out (XOR behavior). We want P(odd number of errors).
    // XOR formula: P(A XOR B) = P(A)*(1-P(B)) + P(B)*(1-P(A)) = p1 + p2 - 2*p1*p2
    p1 * (1.0 - p2) + p2 * (1.0 - p1)
}

/// Formats a probability value similar to Python's %g format.
/// Uses scientific notation for very small/large values, otherwise decimal.
fn format_probability(p: f64) -> String {
    if p == 0.0 {
        return "0".to_string();
    }

    let abs_p = p.abs();

    // Use scientific notation for very small or very large values
    if (1e-4..1e6).contains(&abs_p) {
        // Regular decimal notation
        let formatted = format!("{p:.6}");
        trim_trailing_zeros(&formatted)
    } else {
        // Format with up to 6 significant figures in scientific notation
        let formatted = format!("{p:.6e}");
        // Trim trailing zeros after decimal point
        trim_trailing_zeros(&formatted)
    }
}

/// Trims trailing zeros from a number string.
fn trim_trailing_zeros(s: &str) -> String {
    if let Some(e_pos) = s.find('e') {
        // Scientific notation: trim zeros before 'e'
        let (mantissa, exponent) = s.split_at(e_pos);
        let trimmed = mantissa.trim_end_matches('0').trim_end_matches('.');
        format!("{trimmed}{exponent}")
    } else if s.contains('.') {
        // Decimal notation: trim trailing zeros
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_mechanism_xor() {
        let m1 = FaultMechanism::from_unsorted([0, 1, 2], [0]);
        let m2 = FaultMechanism::from_unsorted([1, 2, 3], [0, 1]);

        let result = m1.xor(&m2);

        // Detectors: {0, 1, 2} XOR {1, 2, 3} = {0, 3}
        assert_eq!(result.detectors.as_slice(), &[0, 3]);
        // DEM outputs: {0} XOR {0, 1} = {1}
        assert_eq!(result.dem_outputs.as_slice(), &[1]);
    }

    #[test]
    fn test_error_mechanism_equality() {
        let m1 = FaultMechanism::from_unsorted([2, 0, 1], [1, 0]);
        let m2 = FaultMechanism::from_unsorted([0, 1, 2], [0, 1]);

        assert_eq!(m1, m2);
        assert_eq!(m1.detectors.as_slice(), &[0, 1, 2]);
        assert_eq!(m1.dem_outputs.as_slice(), &[0, 1]);
    }

    #[test]
    fn test_error_mechanism_equality_and_hash_include_tracked_paulis() {
        let standard = FaultMechanism::from_unsorted([0], []);
        let with_tracked = FaultMechanism::from_unsorted_with_tracked_paulis([0], [], [0]);

        assert_ne!(standard, with_tracked);
        assert_eq!(standard.standard_effect(), with_tracked.standard_effect());

        let mut set = std::collections::HashSet::new();
        set.insert(standard);
        set.insert(with_tracked);
        assert_eq!(
            set.len(),
            2,
            "internal mechanism identity must keep tracked Paulis distinct"
        );
    }

    #[test]
    fn test_pecos_target_format_canonicalizes_tracked_paulis() {
        let mechanism = FaultMechanism::from_unsorted_with_tracked_paulis([], [], [2, 0]);
        assert_eq!(
            DecomposedFault::single(mechanism).to_pecos_targets(),
            "TP0 TP2"
        );
    }

    #[test]
    fn test_combine_probabilities() {
        // Same probability twice
        let p = combine_probabilities(0.01, 0.01);
        // Expected: 0.01 * 0.99 + 0.01 * 0.99 = 0.0198
        assert!((p - 0.0198).abs() < 1e-10);

        // One zero probability
        assert!((combine_probabilities(0.0, 0.5) - 0.5).abs() < 1e-10);
        assert!((combine_probabilities(0.5, 0.0) - 0.5).abs() < 1e-10);

        // Both zero
        assert!((combine_probabilities(0.0, 0.0)).abs() < 1e-10);
    }

    #[test]
    fn test_pecos_metadata_json_preserves_tracked_paulis() {
        use pecos_core::pauli::{X, Z};

        let mut dem = DetectorErrorModel::new();
        dem.add_dem_output(
            DemOutput::new(0)
                .with_kind(DemOutputKind::TrackedPauli)
                .with_pauli(X(0) & Z(2))
                .with_label("track_check"),
        );
        dem.add_dem_output(DemOutput::new(1).with_records([-1, -3]));

        let metadata: serde_json::Value =
            serde_json::from_str(&dem.to_pecos_metadata_json()).unwrap();
        let observables = metadata["observables"].as_array().unwrap();
        let tracked_paulis = metadata["tracked_paulis"].as_array().unwrap();

        assert_eq!(metadata["format"], "pecos.dem.metadata");
        assert_eq!(metadata["version"], 1);
        assert_eq!(tracked_paulis[0]["id"], 0);
        assert_eq!(tracked_paulis[0]["kind"], "tracked_pauli");
        assert_eq!(tracked_paulis[0]["label"], "track_check");
        assert_eq!(tracked_paulis[0]["pauli"], "+X0 Z2");
        assert_eq!(observables[0]["id"], 1);
        assert_eq!(observables[0]["kind"], "observable");
        assert_eq!(observables[0]["records"], serde_json::json!([-1, -3]));
    }

    #[test]
    fn test_dem_counts_keep_detectors_observables_and_tracked_paulis_distinct() {
        use pecos_core::pauli::X;

        let mut dem = DetectorErrorModel::new();
        dem.add_detector(DetectorDef::new(0).with_records([-1]));
        dem.add_dem_output(DemOutput::new(0).with_records([-1, -3]));
        dem.add_dem_output(
            DemOutput::new(0)
                .with_kind(DemOutputKind::TrackedPauli)
                .with_pauli(X(0)),
        );

        assert_eq!(dem.num_detectors(), 1);
        assert_eq!(dem.num_dem_outputs(), 1);
        assert_eq!(dem.num_observables(), 1);
        assert_eq!(dem.num_tracked_paulis(), 1);
        assert_eq!(dem.observables().map(|op| op.id).collect::<Vec<_>>(), [0]);
        assert_eq!(
            dem.tracked_paulis()
                .iter()
                .map(|op| op.id)
                .collect::<Vec<_>>(),
            [0]
        );
    }

    #[test]
    fn test_duplicate_observable_definitions_merge_records_by_parity() {
        use pecos_core::pauli::X;

        let mut dem = DetectorErrorModel::new();
        dem.add_observable(
            DemOutput::new(0)
                .with_records([-1, -2])
                .with_pauli(X(0))
                .with_label("logical_z"),
        );
        dem.add_observable(
            DemOutput::new(0)
                .with_records([-2, -3])
                .with_pauli(X(0))
                .with_label("logical_z"),
        );

        assert_eq!(dem.num_observables(), 1);
        let observable = &dem.dem_outputs()[0];
        assert_eq!(observable.records.as_slice(), &[-1, -3]);
        assert_eq!(observable.pauli.as_ref().unwrap().to_sparse_str(), "+X0");
        assert_eq!(observable.label.as_deref(), Some("logical_z"));
    }

    #[test]
    fn test_observable_records_are_stored_by_xor_parity() {
        let mut dem = DetectorErrorModel::new();
        dem.add_observable(DemOutput::new(0).with_records([-1, -2, -1, -3]));

        assert_eq!(dem.dem_outputs()[0].records.as_slice(), &[-2, -3]);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "conflicting labels for observable L0")]
    fn test_duplicate_observable_definitions_reject_conflicting_labels() {
        let mut dem = DetectorErrorModel::new();
        dem.add_observable(DemOutput::new(0).with_label("first"));
        dem.add_observable(DemOutput::new(0).with_label("second"));
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "conflicting Pauli metadata for observable L0")]
    fn test_duplicate_observable_definitions_reject_conflicting_paulis() {
        use pecos_core::pauli::{X, Z};

        let mut dem = DetectorErrorModel::new();
        dem.add_observable(DemOutput::new(0).with_pauli(X(0)));
        dem.add_observable(DemOutput::new(0).with_pauli(Z(0)));
    }

    #[test]
    fn test_dem_output_kind_predicates_are_mutually_exclusive() {
        use pecos_core::pauli::X;

        let observable = DemOutput::new(0)
            .with_kind(DemOutputKind::Observable)
            .with_pauli(X(0));
        assert!(observable.is_observable());
        assert!(!observable.is_tracked_pauli());

        let tracked = DemOutput::new(0)
            .with_kind(DemOutputKind::TrackedPauli)
            .with_records([-1]);
        assert!(!tracked.is_observable());
        assert!(tracked.is_tracked_pauli());

        let inferred_observable = DemOutput::new(1).with_records([-1]);
        assert!(inferred_observable.is_observable());
        assert!(!inferred_observable.is_tracked_pauli());

        let inferred_tracked = DemOutput::new(1).with_pauli(X(1));
        assert!(!inferred_tracked.is_observable());
        assert!(inferred_tracked.is_tracked_pauli());
    }

    #[test]
    fn test_generic_dem_output_metadata_uses_consistent_kind_name() {
        let mut dem = DetectorErrorModel::new();
        dem.add_dem_output(DemOutput::new(0));

        let metadata: serde_json::Value =
            serde_json::from_str(&dem.to_pecos_metadata_json()).unwrap();
        let ops = metadata["observables"].as_array().unwrap();

        assert_eq!(ops[0]["kind"], "observable");

        let recovered = DetectorErrorModel::new()
            .with_pecos_metadata_json(&dem.to_pecos_metadata_json())
            .unwrap();
        assert_eq!(recovered.num_dem_outputs(), 1);
        assert_eq!(recovered.num_tracked_paulis(), 0);
        assert_eq!(recovered.dem_outputs()[0].id, 0);
        assert_eq!(
            recovered.dem_outputs()[0].kind,
            Some(DemOutputKind::Observable)
        );
    }

    #[test]
    fn test_pecos_metadata_json_round_trips_tracked_pauli_metadata() {
        use pecos_core::pauli::{X, Z};

        let mut dem = DetectorErrorModel::new();
        dem.add_dem_output(DemOutput::new(0));
        dem.add_dem_output(DemOutput::new(1));

        let mut source = DetectorErrorModel::new();
        source.add_dem_output(
            DemOutput::new(0)
                .with_kind(DemOutputKind::TrackedPauli)
                .with_pauli(X(0) & Z(2))
                .with_label("track_check"),
        );
        source.add_dem_output(DemOutput::new(1).with_records([-1, -3]));

        dem.apply_pecos_metadata_json(&source.to_pecos_metadata_json())
            .unwrap();

        assert_eq!(
            dem.tracked_paulis()[0].kind,
            Some(DemOutputKind::TrackedPauli)
        );
        assert_eq!(
            dem.tracked_paulis()[0].label.as_deref(),
            Some("track_check")
        );
        assert_eq!(
            dem.tracked_paulis()[0]
                .pauli
                .as_ref()
                .unwrap()
                .to_sparse_str(),
            "+X0 Z2"
        );
        assert_eq!(dem.dem_outputs()[1].kind, Some(DemOutputKind::Observable));
        assert_eq!(dem.dem_outputs()[1].records.as_slice(), &[-1, -3]);
    }

    #[test]
    fn test_pecos_metadata_json_parser_requires_output_arrays() {
        let old_metadata_json = r#"{
            "format": "pecos.dem.metadata",
            "version": 1,
            "old_outputs": [
                {
                    "id": 4,
                    "kind": "old_kind",
                    "label": "old_name",
                    "pauli": null,
                    "records": []
                }
            ]
        }"#;

        let err = DetectorErrorModel::new()
            .with_pecos_metadata_json(old_metadata_json)
            .unwrap_err();
        assert!(
            err.message()
                .contains("missing observables or tracked_paulis metadata arrays")
        );
    }

    #[test]
    fn test_pecos_metadata_json_parser_rejects_legacy_tracked_fields() {
        let json = r#"{
            "format": "pecos.dem.metadata",
            "version": 1,
            "observables": [],
            "tracked_paulis": [],
            "tracked_ops": [
                {
                    "id": 0,
                    "kind": "tracked_op",
                    "label": "old_name",
                    "pauli": "+X0",
                    "records": []
                }
            ]
        }"#;

        let err = DetectorErrorModel::new()
            .with_pecos_metadata_json(json)
            .unwrap_err();
        assert!(
            err.message()
                .contains("unsupported legacy metadata field: tracked_ops; use tracked_paulis")
        );
    }

    #[test]
    fn test_pecos_metadata_json_parser_rejects_old_generic_kind_names() {
        let json = r#"{
            "format": "pecos.dem.metadata",
            "version": 1,
            "tracked_paulis": [
                {
                    "id": 4,
                    "kind": "old_kind",
                    "label": "old_name",
                    "pauli": null,
                    "records": []
                }
            ]
        }"#;

        let err = DetectorErrorModel::new()
            .with_pecos_metadata_json(json)
            .unwrap_err();
        assert!(
            err.message()
                .contains("DEM output 0 has unknown kind: old_kind")
        );

        let alias_json = r#"{
            "format": "pecos.dem.metadata",
            "version": 1,
            "tracked_paulis": [
                {
                    "id": 4,
                    "kind": "pauli_operator",
                    "label": "old_alias",
                    "pauli": "+X0",
                    "records": []
                }
            ]
        }"#;
        let err = DetectorErrorModel::new()
            .with_pecos_metadata_json(alias_json)
            .unwrap_err();
        assert!(
            err.message()
                .contains("DEM output 0 has unknown kind: pauli_operator")
        );
    }

    #[test]
    fn test_pecos_metadata_json_rejects_records_on_tracked_pauli() {
        let json = r#"{
            "format": "pecos.dem.metadata",
            "version": 1,
            "tracked_paulis": [
                {
                    "id": 0,
                    "kind": "tracked_pauli",
                    "pauli": "X0",
                    "records": [-1]
                }
            ]
        }"#;

        let err = DetectorErrorModel::new()
            .with_pecos_metadata_json(json)
            .unwrap_err();
        assert!(
            err.message()
                .contains("tracked Pauli DEM output 0 cannot have measurement records")
        );
    }

    #[test]
    fn test_pecos_dem_text_is_stim_superset_with_dem_output_metadata() {
        use pecos_core::pauli::{X, Z};

        let mut dem = DetectorErrorModel::new();
        dem.add_detector(DetectorDef::new(0));
        dem.add_dem_output(
            DemOutput::new(0)
                .with_kind(DemOutputKind::TrackedPauli)
                .with_pauli(X(0) & Z(2))
                .with_label("track_check"),
        );
        dem.add_direct_contribution(
            FaultMechanism::from_unsorted_with_tracked_paulis([0], [], [0]),
            0.01,
        );

        let stim_text = dem.to_string();
        assert!(!stim_text.contains("logical_observable L0"));
        assert!(stim_text.contains("error(0.01) D0"));
        assert!(!stim_text.contains("TP0"));
        assert!(!stim_text.contains("pecos_"));

        let pecos_text = dem.to_pecos_string();
        assert!(pecos_text.contains("error(0.01) D0 TP0"));
        assert!(pecos_text.contains("pecos_tracked_pauli"));
        assert!(pecos_text.contains(r#""kind":"tracked_pauli""#));
        assert!(pecos_text.contains(r#""pauli":"+X0 Z2""#));

        let recovered = DetectorErrorModel::new()
            .with_pecos_dem_metadata(&pecos_text)
            .unwrap();
        assert_eq!(recovered.num_dem_outputs(), 0);
        assert_eq!(recovered.num_tracked_paulis(), 1);
        assert_eq!(
            recovered.tracked_paulis()[0].kind,
            Some(DemOutputKind::TrackedPauli)
        );
        assert_eq!(
            recovered.tracked_paulis()[0]
                .pauli
                .as_ref()
                .unwrap()
                .to_sparse_str(),
            "+X0 Z2"
        );
        assert_eq!(
            recovered.tracked_paulis()[0].label.as_deref(),
            Some("track_check")
        );
    }

    #[test]
    fn test_pecos_dem_text_round_trips_observables_and_tracked_paulis() {
        use pecos_core::pauli::Z;

        let mut dem = DetectorErrorModel::new();
        dem.add_detector(DetectorDef::new(0));
        dem.add_dem_output(DemOutput::new(0).with_records([-1]));
        dem.add_dem_output(DemOutput::new(1).with_records([-2]));
        dem.add_dem_output(
            DemOutput::new(0)
                .with_kind(DemOutputKind::TrackedPauli)
                .with_pauli(Z(3))
                .with_label("tracked_z3"),
        );
        dem.add_direct_contribution(
            FaultMechanism::from_unsorted_with_tracked_paulis([0], [0], [0]),
            0.01,
        );
        dem.add_direct_contribution(FaultMechanism::from_unsorted([], [1]), 0.02);

        let stim_text = dem.to_string();
        assert!(stim_text.contains("logical_observable L0"));
        assert!(stim_text.contains("logical_observable L1"));
        assert!(!stim_text.contains("logical_observable L2"));
        assert!(!stim_text.contains("TP0"));
        assert!(!stim_text.contains("pecos_tracked_pauli"));

        let pecos_text = dem.to_pecos_string();
        assert!(pecos_text.contains("error(0.01) D0 L0 TP0"));
        assert!(pecos_text.contains("pecos_observable"));
        assert!(pecos_text.contains("pecos_tracked_pauli"));

        let recovered = DetectorErrorModel::new()
            .with_pecos_dem_metadata(&pecos_text)
            .unwrap();
        assert_eq!(recovered.num_observables(), 2);
        assert_eq!(recovered.num_dem_outputs(), 2);
        assert_eq!(recovered.num_tracked_paulis(), 1);
        assert_eq!(
            recovered
                .dem_outputs()
                .iter()
                .map(|op| op.id)
                .collect::<Vec<_>>(),
            [0, 1]
        );
        assert_eq!(
            recovered
                .tracked_paulis()
                .iter()
                .map(|op| op.id)
                .collect::<Vec<_>>(),
            [0]
        );
        assert_eq!(
            recovered.tracked_paulis()[0]
                .pauli
                .as_ref()
                .unwrap()
                .to_sparse_str(),
            "+Z3"
        );
        assert_eq!(
            recovered.tracked_paulis()[0].label.as_deref(),
            Some("tracked_z3")
        );
    }

    #[test]
    fn test_pecos_dem_text_parses_error_targets_and_metadata() {
        use crate::fault_tolerance::dem_builder::ParsedDem;
        use pecos_core::pauli::{X, Z};

        let mut dem = DetectorErrorModel::new();
        dem.add_detector(DetectorDef::new(0).with_coords([1.0, 2.0, 3.0]));
        dem.add_observable(DemOutput::new(0).with_records([-1]).with_label("L0"));
        dem.add_tracked_pauli(
            DemOutput::new(0)
                .with_pauli(X(0) & Z(2))
                .with_label("tracked_x0_z2"),
        );
        dem.add_direct_contribution(
            FaultMechanism::from_unsorted_with_tracked_paulis([0], [0], [0]),
            0.25,
        );

        let pecos_text = dem.to_pecos_string();
        let parsed: ParsedDem = pecos_text.parse().unwrap();

        assert_eq!(parsed.num_detectors, 1);
        assert_eq!(parsed.num_dem_outputs(), 1);
        assert_eq!(parsed.num_tracked_paulis(), 1);
        assert_eq!(parsed.mechanisms.len(), 1);
        assert_eq!(parsed.mechanisms[0].format_targets(), "D0 L0 TP0");
        assert_eq!(parsed.mechanisms[0].components[0].detectors, vec![0]);
        assert_eq!(parsed.mechanisms[0].components[0].observables, vec![0]);
        assert_eq!(parsed.mechanisms[0].components[0].tracked_paulis, vec![0]);
        assert_eq!(
            parsed.dem_outputs[0].as_ref().unwrap().label.as_deref(),
            Some("L0")
        );
        assert_eq!(
            parsed.tracked_paulis[0]
                .as_ref()
                .unwrap()
                .pauli
                .as_ref()
                .unwrap()
                .to_sparse_str(),
            "+X0 Z2"
        );
    }

    #[test]
    fn test_tracked_only_contribution_is_pecos_only_and_decoder_invisible() {
        use pecos_core::pauli::X;

        let mut dem = DetectorErrorModel::new();
        dem.add_tracked_pauli(DemOutput::new(0).with_pauli(X(0)).with_label("tracked_x0"));
        dem.add_direct_contribution(
            FaultMechanism::from_unsorted_with_tracked_paulis([], [], [0]),
            0.25,
        );

        let standard_text = dem.to_string();
        assert!(!standard_text.contains("error("));
        assert!(!standard_text.contains("TP0"));
        assert!(!standard_text.contains("pecos_tracked_pauli"));

        let pecos_text = dem.to_pecos_string();
        assert!(pecos_text.contains("error(0.25) TP0"));
        assert!(pecos_text.contains("pecos_tracked_pauli"));

        let (mechanisms, coords) = dem.to_mechanisms();
        assert!(mechanisms.is_empty());
        assert!(coords.is_empty());
    }

    #[test]
    fn test_standard_projection_merges_effects_that_differ_only_by_tracked_paulis() {
        use pecos_core::pauli::{X, Z};

        let mut dem = DetectorErrorModel::new();
        dem.add_detector(DetectorDef::new(0));
        dem.add_tracked_pauli(DemOutput::new(0).with_pauli(X(0)).with_label("tracked_x0"));
        dem.add_tracked_pauli(DemOutput::new(1).with_pauli(Z(0)).with_label("tracked_z0"));
        dem.add_direct_contribution(
            FaultMechanism::from_unsorted_with_tracked_paulis([0], [], [0]),
            0.1,
        );
        dem.add_direct_contribution(
            FaultMechanism::from_unsorted_with_tracked_paulis([0], [], [1]),
            0.2,
        );

        let standard_text = dem.to_string();
        let error_lines = standard_text
            .lines()
            .filter(|line| line.starts_with("error("))
            .collect::<Vec<_>>();
        assert_eq!(error_lines, ["error(0.26) D0"]);
        assert!(!standard_text.contains("TP0"));
        assert!(!standard_text.contains("TP1"));

        let (mechanisms, _coords) = dem.to_mechanisms();
        assert_eq!(mechanisms.len(), 1);
        assert!((mechanisms[0].0 - 0.26).abs() < 1e-12);
        assert_eq!(mechanisms[0].1, vec![0]);
        assert!(mechanisms[0].2.is_empty());
    }

    #[test]
    fn test_pecos_dem_preserves_effects_that_differ_by_tracked_paulis() {
        use pecos_core::pauli::{X, Z};

        let mut dem = DetectorErrorModel::new();
        dem.add_detector(DetectorDef::new(0));
        dem.add_tracked_pauli(DemOutput::new(0).with_pauli(X(0)).with_label("tracked_x0"));
        dem.add_tracked_pauli(DemOutput::new(1).with_pauli(Z(0)).with_label("tracked_z0"));
        dem.add_direct_contribution(
            FaultMechanism::from_unsorted_with_tracked_paulis([0], [], [0]),
            0.1,
        );
        dem.add_direct_contribution(
            FaultMechanism::from_unsorted_with_tracked_paulis([0], [], [1]),
            0.2,
        );

        let pecos_text = dem.to_pecos_string();
        let error_lines = pecos_text
            .lines()
            .filter(|line| line.starts_with("error("))
            .collect::<Vec<_>>();
        assert_eq!(error_lines, ["error(0.1) D0 TP0", "error(0.2) D0 TP1"]);
        assert!(pecos_text.contains(r#""label":"tracked_x0""#));
        assert!(pecos_text.contains(r#""label":"tracked_z0""#));
    }

    #[test]
    fn test_standard_dem_serialization_never_shifts_observable_ids_for_tracked_paulis() {
        use pecos_core::pauli::{X, Z};

        let mut dem = DetectorErrorModel::new();
        dem.add_detector(DetectorDef::new(0));
        dem.add_observable(DemOutput::new(0).with_records([-1]).with_label("L0"));
        dem.add_observable(DemOutput::new(2).with_records([-2]).with_label("L2"));
        dem.add_tracked_pauli(
            DemOutput::new(0)
                .with_kind(DemOutputKind::TrackedPauli)
                .with_pauli(X(0))
                .with_label("tracked_x0"),
        );
        dem.add_tracked_pauli(
            DemOutput::new(1)
                .with_kind(DemOutputKind::TrackedPauli)
                .with_pauli(Z(3))
                .with_label("tracked_z3"),
        );
        dem.add_direct_contribution(
            FaultMechanism::from_unsorted_with_tracked_paulis([0], [0, 2], [1]),
            0.01,
        );

        assert_eq!(dem.num_observables(), 3);
        assert_eq!(dem.num_dem_outputs(), 3);
        assert_eq!(dem.num_tracked_paulis(), 2);

        let standard_text = dem.to_string();
        assert!(standard_text.contains("logical_observable L0"));
        assert!(!standard_text.contains("logical_observable L1"));
        assert!(standard_text.contains("logical_observable L2"));
        assert!(!standard_text.contains("logical_observable L3"));
        assert!(standard_text.contains("error(0.01) D0 L0 L2"));
        assert!(!standard_text.contains("TP1"));
        assert!(!standard_text.contains("pecos_observable"));
        assert!(!standard_text.contains("pecos_tracked_pauli"));

        let pecos_text = dem.to_pecos_string();
        assert!(pecos_text.contains("error(0.01) D0 L0 L2 TP1"));
        assert!(pecos_text.contains(r#""kind":"observable""#));
        assert!(pecos_text.contains(r#""kind":"tracked_pauli""#));
        assert!(pecos_text.contains(r#""id":0"#));
        assert!(pecos_text.contains(r#""id":2"#));
        assert!(pecos_text.contains(r#""pauli":"+X0""#));
        assert!(pecos_text.contains(r#""pauli":"+Z3""#));

        let recovered = DetectorErrorModel::new()
            .with_pecos_dem_metadata(&pecos_text)
            .unwrap();
        assert_eq!(recovered.num_dem_outputs(), 3);
        assert_eq!(recovered.num_tracked_paulis(), 2);
        assert_eq!(
            recovered
                .dem_outputs()
                .iter()
                .map(|op| op.id)
                .collect::<Vec<_>>(),
            [0, 2]
        );
        assert_eq!(
            recovered
                .tracked_paulis()
                .iter()
                .map(|op| op.id)
                .collect::<Vec<_>>(),
            [0, 1]
        );
    }

    #[test]
    fn test_pecos_dem_text_metadata_round_trip_keeps_observable_and_tracked_id_spaces() {
        use pecos_core::pauli::{X, Y, Z};

        let mut dem = DetectorErrorModel::new();
        dem.add_detector(DetectorDef::new(0));
        dem.add_detector(DetectorDef::new(1));
        dem.add_observable(DemOutput::new(0).with_records([-1]).with_label("L0"));
        dem.add_observable(
            DemOutput::new(3)
                .with_records([-2, -1])
                .with_label("logical_aux"),
        );
        dem.add_tracked_pauli(
            DemOutput::new(0)
                .with_pauli(X(0) & Z(2))
                .with_label("tracked_x0_z2"),
        );
        dem.add_tracked_pauli(DemOutput::new(2).with_pauli(Y(5)).with_label("tracked_y5"));
        dem.add_direct_contribution(
            FaultMechanism::from_unsorted_with_tracked_paulis([0, 1], [3], [2]),
            0.125,
        );

        let standard_text = dem.to_string();
        assert!(standard_text.contains("logical_observable L0"));
        assert!(!standard_text.contains("logical_observable L1"));
        assert!(!standard_text.contains("logical_observable L2"));
        assert!(standard_text.contains("logical_observable L3"));
        assert!(standard_text.contains("error(0.125) D0 D1 L3"));
        assert!(!standard_text.contains("TP2"));
        assert!(!standard_text.contains("pecos_tracked_pauli"));

        let pecos_text = format!(
            "# ordinary comments and standard DEM lines are allowed\n{}\n",
            dem.to_pecos_string()
        );
        let recovered = DetectorErrorModel::new()
            .with_pecos_dem_metadata(&pecos_text)
            .unwrap();
        assert_eq!(recovered.num_observables(), 4);
        assert_eq!(recovered.num_dem_outputs(), 4);
        assert_eq!(recovered.num_tracked_paulis(), 3);
        assert_eq!(
            recovered
                .dem_outputs()
                .iter()
                .map(|op| (op.id, op.label.as_deref()))
                .collect::<Vec<_>>(),
            [(0, Some("L0")), (3, Some("logical_aux"))]
        );
        assert_eq!(
            recovered
                .tracked_paulis()
                .iter()
                .map(|op| (op.id, op.label.as_deref()))
                .collect::<Vec<_>>(),
            [(0, Some("tracked_x0_z2")), (2, Some("tracked_y5"))]
        );
        assert_eq!(
            recovered.tracked_paulis()[0]
                .pauli
                .as_ref()
                .unwrap()
                .to_sparse_str(),
            "+X0 Z2"
        );
        assert_eq!(
            recovered.tracked_paulis()[1]
                .pauli
                .as_ref()
                .unwrap()
                .to_sparse_str(),
            "+Y5"
        );

        let reserialized = recovered.to_pecos_string();
        assert!(reserialized.contains("logical_observable L0"));
        assert!(!reserialized.contains("logical_observable L1"));
        assert!(!reserialized.contains("logical_observable L2"));
        assert!(reserialized.contains("logical_observable L3"));
        assert!(reserialized.contains(r#""kind":"observable""#));
        assert!(reserialized.contains(r#""kind":"tracked_pauli""#));
        assert!(reserialized.contains(r#""pauli":"+X0 Z2""#));
        assert!(reserialized.contains(r#""pauli":"+Y5""#));
        assert!(
            !reserialized.contains("TP2"),
            "metadata-only recovery should not invent mechanism effects"
        );
    }

    #[test]
    fn test_pecos_dem_text_and_metadata_json_preserve_same_output_metadata() {
        use crate::fault_tolerance::dem_builder::ParsedDem;
        use pecos_core::pauli::{X, Y, Z};

        let mut dem = DetectorErrorModel::new();
        dem.add_detector(DetectorDef::new(0).with_records([-1]));
        dem.add_observable(DemOutput::new(0).with_records([-1]).with_label("L0"));
        dem.add_observable(DemOutput::new(3).with_records([-2]).with_label("L3"));
        dem.add_tracked_pauli(
            DemOutput::new(0)
                .with_pauli(X(0) & Z(2))
                .with_label("tracked_x0_z2"),
        );
        dem.add_tracked_pauli(DemOutput::new(3).with_pauli(Y(5)).with_label("tracked_y5"));
        dem.add_direct_contribution(
            FaultMechanism::from_unsorted_with_tracked_paulis([0], [3], [3]),
            0.125,
        );

        let json_recovered = DetectorErrorModel::new()
            .with_pecos_metadata_json(&dem.to_pecos_metadata_json())
            .unwrap();
        let text_recovered = DetectorErrorModel::new()
            .with_pecos_dem_metadata(&dem.to_pecos_string())
            .unwrap();

        let source_json: serde_json::Value =
            serde_json::from_str(&dem.to_pecos_metadata_json()).unwrap();
        let from_json: serde_json::Value =
            serde_json::from_str(&json_recovered.to_pecos_metadata_json()).unwrap();
        let from_text: serde_json::Value =
            serde_json::from_str(&text_recovered.to_pecos_metadata_json()).unwrap();

        assert_eq!(from_json, source_json);
        assert_eq!(from_text, source_json);

        let parsed: ParsedDem = dem.to_pecos_string().parse().unwrap();
        assert_eq!(parsed.num_dem_outputs(), 4);
        assert_eq!(parsed.num_tracked_paulis(), 4);
        assert_eq!(parsed.mechanisms[0].format_targets(), "D0 L3 TP3");
        assert_eq!(parsed.mechanisms[0].components[0].observables, vec![3]);
        assert_eq!(parsed.mechanisms[0].components[0].tracked_paulis, vec![3]);
        assert_eq!(
            parsed.dem_outputs[0].as_ref().unwrap().label.as_deref(),
            Some("L0")
        );
        assert_eq!(
            parsed.dem_outputs[3].as_ref().unwrap().label.as_deref(),
            Some("L3")
        );
        assert_eq!(
            parsed.tracked_paulis[0]
                .as_ref()
                .unwrap()
                .pauli
                .as_ref()
                .unwrap()
                .to_sparse_str(),
            "+X0 Z2"
        );
        assert_eq!(
            parsed.tracked_paulis[3].as_ref().unwrap().label.as_deref(),
            Some("tracked_y5")
        );
    }

    #[test]
    fn test_pecos_dem_metadata_parser_rejects_malformed_extension_line() {
        let err = DetectorErrorModel::new()
            .with_pecos_dem_metadata("error(0.01) D0\npecos_tracked_pauli not-json")
            .unwrap_err();
        assert!(
            err.message()
                .contains("invalid pecos_tracked_pauli JSON payload")
        );
    }

    #[test]
    fn test_pecos_dem_metadata_parser_rejects_unknown_pecos_extension_line() {
        let err = DetectorErrorModel::new()
            .with_pecos_dem_metadata(r#"pecos_old_extension {"id":1}"#)
            .unwrap_err();

        assert!(
            err.message()
                .contains("unsupported PECOS DEM extension line")
        );
    }

    #[test]
    fn test_pecos_dem_metadata_parser_rejects_legacy_tracked_extension_line() {
        let err = DetectorErrorModel::new()
            .with_pecos_dem_metadata(r#"pecos_tracked_op {"id":0,"pauli":"+X0"}"#)
            .unwrap_err();

        assert!(
            err.message()
                .contains("unsupported PECOS DEM extension line: pecos_tracked_op")
        );
    }

    #[test]
    fn test_decomposed_error_single() {
        let mechanism = FaultMechanism::from_unsorted_with_tracked_paulis([0, 1], [0], [2]);
        let decomposed = DecomposedFault::single(mechanism.clone());

        assert_eq!(decomposed.components.len(), 1);
        assert!(decomposed.is_graphlike());
        assert_eq!(decomposed.full_effect(), mechanism);
        assert_eq!(decomposed.to_stim_targets(), "D0 D1 L0");
        assert_eq!(decomposed.to_pecos_targets(), "D0 D1 L0 TP2");
    }

    #[test]
    fn test_decomposed_error_multi() {
        let m1 = FaultMechanism::from_unsorted([0, 1], []);
        let m2 = FaultMechanism::from_unsorted([2, 3], [0]);
        let decomposed = DecomposedFault::decomposed([m1.clone(), m2.clone()]);

        assert_eq!(decomposed.components.len(), 2);
        assert!(decomposed.is_graphlike());

        // Full effect should be XOR of both
        let expected = m1.xor(&m2);
        assert_eq!(decomposed.full_effect(), expected);

        // DEM format should use ^ separator for decomposed entries
        assert_eq!(decomposed.to_stim_targets(), "D0 D1 ^ D2 D3 L0");
    }

    #[test]
    fn test_dem_to_string_decomposed_keeps_two_detector_graphlike_edges_direct() {
        let mut dem = DetectorErrorModel::new();

        dem.add_detector(DetectorDef::new(0).with_coords([0.0, 0.0, 0.0]));
        dem.add_detector(DetectorDef::new(1).with_coords([1.0, 0.0, 0.0]));
        dem.add_dem_output(DemOutput::new(0));

        dem.add_direct_contribution(FaultMechanism::from_unsorted([0, 1], []), 0.01);
        dem.add_direct_contribution(FaultMechanism::from_unsorted([0], [0]), 0.02);
        dem.add_direct_contribution(FaultMechanism::from_unsorted([1], [0]), 0.03);

        let stim_str = dem.to_string_decomposed();

        assert!(stim_str.contains("logical_observable L0"));
        assert!(stim_str.contains("error(0.01) D0 D1"));
        assert!(!stim_str.contains("D0 L0 ^ D1 L0"));
    }

    #[test]
    fn test_dem_to_string_decomposed_maximally_prefers_singletons_when_available() {
        let mut dem = DetectorErrorModel::new();

        dem.add_detector(DetectorDef::new(0).with_coords([0.0, 0.0, 0.0]));
        dem.add_detector(DetectorDef::new(1).with_coords([1.0, 0.0, 0.0]));

        dem.add_direct_contribution(FaultMechanism::from_unsorted([0, 1], []), 0.01);
        dem.add_direct_contribution(FaultMechanism::from_unsorted([0], std::iter::empty()), 0.02);
        dem.add_direct_contribution(FaultMechanism::from_unsorted([1], std::iter::empty()), 0.03);

        let decomposed = dem.to_string_decomposed();
        let maximal = dem.to_string_decomposed_maximally();

        assert!(decomposed.contains("error(0.01) D0 D1"));
        assert!(!decomposed.contains("error(0.01) D0 ^ D1"));

        assert!(maximal.contains("error(0.01) D0 ^ D1"));
        assert!(!maximal.contains("error(0.01) D0 D1"));
    }

    #[test]
    fn test_contribution_effect_summaries_include_graphlike_decomposable_count() {
        let mut dem = DetectorErrorModel::new();

        dem.add_direct_contribution(
            FaultMechanism::from_unsorted([0, 1], std::iter::empty()),
            0.01,
        );
        dem.add_direct_contribution(FaultMechanism::from_unsorted([0], std::iter::empty()), 0.02);
        dem.mark_graphlike_decomposable(0, 1);
        dem.mark_graphlike_decomposable(1, 0);

        let summaries = dem.contribution_effect_summaries();

        let pair_summary = summaries
            .iter()
            .find(|summary| {
                summary.effect.detectors.as_slice() == [0, 1]
                    && summary.effect.dem_outputs.is_empty()
            })
            .expect("pair summary missing");
        assert_eq!(pair_summary.graphlike_decomposable_count, 2);

        let singleton_summary = summaries
            .iter()
            .find(|summary| {
                summary.effect.detectors.as_slice() == [0] && summary.effect.dem_outputs.is_empty()
            })
            .expect("singleton summary missing");
        assert_eq!(singleton_summary.graphlike_decomposable_count, 0);
    }

    #[test]
    fn test_dem_to_string() {
        let mut dem = DetectorErrorModel::new();

        dem.add_detector(DetectorDef::new(0).with_coords([0.0, 0.0, 0.0]));
        dem.add_detector(DetectorDef::new(1).with_coords([1.0, 0.0, 0.0]));
        dem.add_dem_output(DemOutput::new(0));

        // Add contributions directly using the source tracking API
        dem.add_direct_contribution(FaultMechanism::from_unsorted([0, 1], []), 0.01);
        dem.add_direct_contribution(FaultMechanism::from_unsorted([1], [0]), 0.005);

        let stim_str = dem.to_string();

        assert!(stim_str.contains("detector(0, 0, 0) D0"));
        assert!(stim_str.contains("detector(1, 0, 0) D1"));
        assert!(stim_str.contains("logical_observable L0"));
        assert!(stim_str.contains("error(0.01) D0 D1"));
        assert!(stim_str.contains("error(0.005) D1 L0"));
    }

    #[test]
    fn test_dem_to_string_decomposed_keeps_two_detector_one_dem_output_direct() {
        let mut dem = DetectorErrorModel::new();

        dem.add_detector(DetectorDef::new(0).with_coords([0.0, 0.0, 0.0]));
        dem.add_detector(DetectorDef::new(1).with_coords([1.0, 0.0, 0.0]));
        dem.add_dem_output(DemOutput::new(0));

        dem.add_direct_contribution(FaultMechanism::from_unsorted([0, 1], [0]), 0.01);
        dem.add_direct_contribution(FaultMechanism::from_unsorted([0], std::iter::empty()), 0.02);
        dem.add_direct_contribution(FaultMechanism::from_unsorted([1], [0]), 0.03);

        let stim_str = dem.to_string_decomposed();

        assert!(stim_str.contains("error(0.01) D0 D1 L0"));
        assert!(!stim_str.contains("error(0.01) D0 ^ D1 L0"));
    }

    #[test]
    fn test_dem_to_string_decomposed_uses_y_components_when_graphlike() {
        let mut dem = DetectorErrorModel::new();

        dem.add_detector(DetectorDef::new(0).with_coords([0.0, 0.0, 0.0]));
        dem.add_detector(DetectorDef::new(1).with_coords([1.0, 0.0, 0.0]));
        dem.add_dem_output(DemOutput::new(0));

        let x = FaultMechanism::from_unsorted([0], std::iter::empty());
        let z = FaultMechanism::from_unsorted([1], [0]);
        dem.add_y_decomposed_contribution(&x, &z, 0.01);

        let stim_str = dem.to_string_decomposed();

        assert!(stim_str.contains("error(0.01) D0 ^ D1 L0"));
        assert!(!stim_str.contains("error(0.01) D0 D1 L0"));
    }

    #[test]
    fn test_error_mechanism_with_two_detectors_and_multiple_dem_outputs_is_graphlike() {
        let effect = FaultMechanism::from_unsorted([0, 1], [0, 1]);

        assert!(effect.is_graphlike());
        assert!(!effect.is_hyperedge());
    }

    #[test]
    fn test_find_hyperedge_decomposition_returns_graphlike_subset_components() {
        let hyperedge = FaultMechanism::from_unsorted([0, 1, 2], [0]);
        let graphlike_set = BTreeSet::from([
            FaultMechanism::from_unsorted([0], std::iter::empty()),
            FaultMechanism::from_unsorted([1], std::iter::empty()),
            FaultMechanism::from_unsorted([2], [0]),
            FaultMechanism::from_unsorted([0, 1], std::iter::empty()),
        ]);

        let decomposition = find_hyperedge_decomposition(&hyperedge, &graphlike_set)
            .expect("expected a valid decomposition");
        let hyperedge_dets: BTreeSet<u32> = hyperedge.detectors.iter().copied().collect();

        let recomposed = decomposition
            .iter()
            .fold(FaultMechanism::new(), |acc, part| acc.xor(part));
        assert_eq!(recomposed, hyperedge);
        assert!(
            decomposition
                .iter()
                .all(super::FaultMechanism::is_graphlike)
        );
        assert!(
            decomposition
                .iter()
                .flat_map(|part| part.detectors.iter())
                .all(|det| hyperedge_dets.contains(det))
        );
        assert_eq!(decomposition.len(), 2);
    }

    #[test]
    fn test_find_hyperedge_decomposition_can_use_four_parts() {
        let hyperedge = FaultMechanism::from_unsorted([0, 1, 2, 3], [0]);
        let graphlike_set = BTreeSet::from([
            FaultMechanism::from_unsorted([0], std::iter::empty()),
            FaultMechanism::from_unsorted([1], std::iter::empty()),
            FaultMechanism::from_unsorted([2], std::iter::empty()),
            FaultMechanism::from_unsorted([3], [0]),
        ]);

        let decomposition = find_hyperedge_decomposition(&hyperedge, &graphlike_set)
            .expect("expected a valid decomposition");

        let recomposed = decomposition
            .iter()
            .fold(FaultMechanism::new(), |acc, part| acc.xor(part));
        assert_eq!(recomposed, hyperedge);
        assert!(
            decomposition
                .iter()
                .all(super::FaultMechanism::is_graphlike)
        );
        assert_eq!(decomposition.len(), 4);
    }

    #[test]
    fn test_contributions_for_effect_matches_observable_coupled_effects() {
        let mut dem = DetectorErrorModel::new();
        let effect = FaultMechanism::from_unsorted([0, 1], [0]);

        dem.add_direct_contribution(effect.clone(), 0.01);

        let matches = dem.contributions_for_effect(&[1, 0], &[0]);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].effect, effect);
        assert!(matches[0].is_direct());
    }

    #[test]
    fn test_contribution_effect_summaries_split_direct_and_y_contributions() {
        let mut dem = DetectorErrorModel::new();
        let effect = FaultMechanism::from_unsorted([0, 1], [0]);

        dem.add_direct_contribution(effect.clone(), 0.01);
        let x = FaultMechanism::from_unsorted([0], std::iter::empty());
        let z = FaultMechanism::from_unsorted([1], [0]);
        dem.add_y_decomposed_contribution(&x, &z, 0.02);

        let summary = dem
            .contribution_effect_summaries()
            .into_iter()
            .find(|row| row.effect == effect)
            .expect("expected a summary for the shared effect");

        assert_eq!(summary.num_contributions, 2);
        assert!((summary.total_probability - 0.03).abs() < 1e-12);
        assert_eq!(summary.direct_count, 1);
        assert!((summary.direct_probability - 0.01).abs() < 1e-12);
        assert_eq!(summary.y_decomposed_count, 1);
        assert!((summary.y_decomposed_probability - 0.02).abs() < 1e-12);
    }

    #[test]
    fn test_add_y_decomposed_contribution_routes_one_empty_branch_to_direct() {
        let mut dem = DetectorErrorModel::new();
        let x = FaultMechanism::new();
        let z = FaultMechanism::from_unsorted([1, 44], std::iter::empty());

        dem.add_y_decomposed_contribution(&x, &z, 0.02);

        let summary = dem
            .contribution_effect_summaries()
            .into_iter()
            .find(|row| row.effect == z)
            .expect("expected summary for graphlike direct effect");

        assert_eq!(summary.direct_count, 1);
        assert_eq!(summary.y_decomposed_count, 0);
        assert!((summary.direct_probability - 0.02).abs() < 1e-12);
    }

    #[test]
    fn test_direct_with_source_components_xor_back_to_effect() {
        let effect = FaultMechanism::from_unsorted([0, 1], std::iter::empty());
        let first = FaultMechanism::from_unsorted([0], std::iter::empty());
        let second = FaultMechanism::from_unsorted([1], std::iter::empty());

        let contribution = FaultContribution::direct_with_source_components(
            effect.clone(),
            0.01,
            SourceMetadata::new(
                &[3, 4],
                &[Pauli::Z, Pauli::I],
                &[GateType::CX, GateType::CX],
                &[false, false],
            ),
            DirectSourceComponents::new(&first, &second),
        );

        assert!(contribution.is_direct());
        let (left, right) = contribution
            .direct_component_effects()
            .expect("expected direct component effects");
        assert_eq!(left.xor(&right), effect);
        assert!(matches!(contribution.source_type, FaultSourceType::Direct));
    }

    #[test]
    fn test_direct_with_source_components_marks_one_sided_component_sources() {
        let effect = FaultMechanism::from_unsorted([7, 11], std::iter::empty());
        let first = effect.clone();
        let second = FaultMechanism::new();

        let contribution = FaultContribution::direct_with_source_components(
            effect.clone(),
            0.01,
            SourceMetadata::new(
                &[3, 4],
                &[Pauli::Z, Pauli::I],
                &[GateType::CX, GateType::CX],
                &[false, false],
            ),
            DirectSourceComponents::new(&first, &second),
        );

        assert!(contribution.is_direct());
        assert!(matches!(
            contribution.source_type,
            FaultSourceType::DirectOneSidedComponent
        ));
        assert_eq!(
            contribution.direct_source_family,
            Some(DirectSourceFamily::TwoLocationOneSidedComponent)
        );
        let (left, right) = contribution
            .direct_component_effects()
            .expect("expected direct component effects");
        assert_eq!(left, effect);
        assert!(right.is_empty());
        assert_eq!(
            contribution.source_gate_types.as_slice(),
            &[GateType::CX, GateType::CX]
        );
        assert_eq!(contribution.source_before_flags.as_slice(), &[false, false]);
    }

    #[test]
    fn test_add_y_decomposed_contribution_with_source_routes_metadata_to_direct() {
        let mut dem = DetectorErrorModel::new();
        let x = FaultMechanism::new();
        let z = FaultMechanism::from_unsorted([1, 44], std::iter::empty());

        dem.add_y_decomposed_contribution_with_source(
            &x,
            &z,
            0.02,
            SourceMetadata::new(&[7], &[Pauli::Y], &[GateType::H], &[false]),
        );

        let contributions = dem.contributions_for_effect(&[1, 44], &[]);
        assert_eq!(contributions.len(), 1);
        let contribution = &contributions[0];
        assert!(matches!(contribution.source_type, FaultSourceType::Direct));
        assert_eq!(contribution.location_indices.as_slice(), &[7]);
        assert_eq!(contribution.paulis.as_slice(), &[Pauli::Y]);
        assert_eq!(contribution.source_gate_types.as_slice(), &[GateType::H]);
        assert_eq!(contribution.source_before_flags.as_slice(), &[false]);
        assert_eq!(
            contribution.direct_source_family,
            Some(DirectSourceFamily::SingleLocationY)
        );
    }
}
