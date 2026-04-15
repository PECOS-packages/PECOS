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
//! detectors, and logical observables in DEM format.
//!
//! # Output Formats
//!
//! The DEM supports two output formats:
//!
//! - [`DetectorErrorModel::to_string()`] - Non-decomposed format matching Stim's
//!   `decompose_errors=False` output. Each mechanism is output once with its
//!   combined probability.
//!
//! - [`DetectorErrorModel::to_string_decomposed()`] - Decomposed format matching
//!   Stim's `decompose_errors=True` output. Hyperedge errors (3+ detectors) are
//!   decomposed into graphlike components, and 2-detector mechanisms may have
//!   multiple representations for decoder compatibility.
//!
//! Decomposed errors use the `^` separator to indicate XOR composition:
//!
//! ```text
//! error(0.01) D0 D1 ^ D2 D3
//! ```
//!
//! This indicates an error decomposed into two parts whose XOR equals the
//! original mechanism.

use pecos_core::gate_type::GateType;
use rand::RngExt;
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::hash::{Hash, Hasher};

use crate::fault_tolerance::propagator::Pauli;

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
        /// Logical effect of the X component (sorted logical IDs).
        x_logicals: SmallVec<[u32; 2]>,
        /// Detector effect of the Z component (sorted detector IDs).
        z_detectors: SmallVec<[u32; 4]>,
        /// Logical effect of the Z component (sorted logical IDs).
        z_logicals: SmallVec<[u32; 2]>,
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
    /// The detector/logical effect of this error.
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
                x_logicals: x_effect.logicals.clone(),
                z_detectors: z_effect.detectors.clone(),
                z_logicals: z_effect.logicals.clone(),
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
                x_logicals: x_effect.logicals.clone(),
                z_detectors: z_effect.detectors.clone(),
                z_logicals: z_effect.logicals.clone(),
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
                x_logicals,
                z_detectors,
                z_logicals,
            } => {
                let x = FaultMechanism::from_sorted(x_detectors.clone(), x_logicals.clone());
                let z = FaultMechanism::from_sorted(z_detectors.clone(), z_logicals.clone());
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
    /// The detector/logical effect being summarized.
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
    /// This is only non-zero for 2-detector, 0-logical effects. It reflects the
    /// dormant representation-diversity bookkeeping recorded by the DEM builder.
    pub graphlike_decomposable_count: u32,
}

/// Structured summary of how tracked contributions render before final regrouping.
#[derive(Debug, Clone)]
pub struct ContributionRenderSummary {
    /// Original full detector/logical effect before rendering.
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
    /// Kept a 2-detector, 0-logical effect graphlike as-is.
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

/// An fault mechanism: a set of detectors and logical observables that flip together.
///
/// When an error occurs, it flips a specific set of detectors and may flip
/// logical observables. Mechanisms with the same effect are aggregated together.
///
/// The detectors and logicals are stored in sorted order for canonical representation.
#[derive(Clone, Default)]
pub struct FaultMechanism {
    /// Detector indices that flip together (sorted).
    pub detectors: SmallVec<[u32; 4]>,
    /// Logical observable indices that flip together (sorted).
    pub logicals: SmallVec<[u32; 2]>,
}

impl FaultMechanism {
    /// Creates a new empty fault mechanism.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a mechanism from unsorted detector and logical indices.
    #[must_use]
    pub fn from_unsorted(
        detectors: impl IntoIterator<Item = u32>,
        logicals: impl IntoIterator<Item = u32>,
    ) -> Self {
        let mut dets: SmallVec<[u32; 4]> = detectors.into_iter().collect();
        let mut logs: SmallVec<[u32; 2]> = logicals.into_iter().collect();
        dets.sort_unstable();
        logs.sort_unstable();
        Self {
            detectors: dets,
            logicals: logs,
        }
    }

    /// Creates a mechanism from pre-sorted detector and logical indices.
    #[must_use]
    pub fn from_sorted(detectors: SmallVec<[u32; 4]>, logicals: SmallVec<[u32; 2]>) -> Self {
        debug_assert!(
            detectors.windows(2).all(|w| w[0] <= w[1]),
            "detectors must be sorted"
        );
        debug_assert!(
            logicals.windows(2).all(|w| w[0] <= w[1]),
            "logicals must be sorted"
        );
        Self {
            detectors,
            logicals,
        }
    }

    /// Returns true if this mechanism has no effect (empty).
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.detectors.is_empty() && self.logicals.is_empty()
    }

    /// Returns the number of detectors in this mechanism.
    #[inline]
    #[must_use]
    pub fn num_detectors(&self) -> usize {
        self.detectors.len()
    }

    /// Returns the number of logicals in this mechanism.
    #[inline]
    #[must_use]
    pub fn num_logicals(&self) -> usize {
        self.logicals.len()
    }

    /// XOR this mechanism with another, returning the combined effect.
    ///
    /// Used when combining correlated errors (e.g., two-qubit gate errors).
    #[must_use]
    pub fn xor(&self, other: &Self) -> Self {
        Self {
            detectors: symmetric_difference_4(&self.detectors, &other.detectors),
            logicals: symmetric_difference_2(&self.logicals, &other.logicals),
        }
    }

    /// Returns true if this mechanism is graphlike.
    ///
    /// A graphlike mechanism has at most 2 detectors.
    /// Logical observables do not affect graph-likeness; MWPM decoders attach
    /// them as frame-change masks on graph edges.
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
        self.detectors == other.detectors && self.logicals == other.logicals
    }
}

impl Eq for FaultMechanism {}

impl Hash for FaultMechanism {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.detectors.hash(state);
        self.logicals.hash(state);
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
            .then_with(|| self.logicals.cmp(&other.logicals))
    }
}

impl fmt::Debug for FaultMechanism {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FaultMechanism(dets={:?}, logs={:?})",
            self.detectors.as_slice(),
            self.logicals.as_slice()
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
            .map(|comp| {
                let mut targets = Vec::new();
                for &det in &comp.detectors {
                    targets.push(format!("D{det}"));
                }
                for &log in &comp.logicals {
                    targets.push(format!("L{log}"));
                }
                targets.join(" ")
            })
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
/// detector sets are subsets of the hyperedge. This is closer to Stim's
/// decomposition strategy than the older fixed-width 2-part/3-part search,
/// and it allows decompositions into 4+ graphlike pieces when needed.
///
/// Decompositions are filtered to only include components whose detectors are
/// subsets of the original hyperedge's detectors, matching Stim's behavior of
/// not introducing extra detector symptoms.
///
/// # Selection
///
/// The search returns the first valid decomposition found using a deterministic
/// ordering that prefers detector pairs before singlets, similar to Stim's
/// decompose pass over known graphlike symptoms.
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
    /// detector is `det`, sorted by `(logicals.len, logicals, detectors)` so the
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
                a.logicals
                    .len()
                    .cmp(&b.logicals.len())
                    .then_with(|| a.logicals.cmp(&b.logicals))
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
    // Check logicals
    for l in &a.logicals {
        if b.logicals.contains(l) {
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

/// Converts a measurement record offset (Stim-style) to an absolute measurement index.
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
// Logical Observable
// ============================================================================

/// A logical observable definition.
///
/// Logical observables track the parity of certain measurement outcomes
/// to detect logical errors.
#[derive(Debug, Clone)]
pub struct LogicalObservable {
    /// Unique observable ID.
    pub id: u32,
    /// Measurement record offsets (negative indices from end of record).
    pub records: SmallVec<[i32; 4]>,
}

impl LogicalObservable {
    /// Creates a new logical observable.
    #[must_use]
    pub fn new(id: u32) -> Self {
        Self {
            id,
            records: SmallVec::new(),
        }
    }

    /// Sets the measurement records.
    #[must_use]
    pub fn with_records(mut self, records: impl IntoIterator<Item = i32>) -> Self {
        self.records = records.into_iter().collect();
        self
    }
}

// ============================================================================
// Noise Configuration
// ============================================================================

/// Noise model configuration for DEM generation.
#[derive(Debug, Clone, Copy)]
pub struct NoiseConfig {
    /// Single-qubit depolarizing error rate.
    pub p1: f64,
    /// Two-qubit depolarizing error rate.
    pub p2: f64,
    /// Measurement error rate.
    pub p_meas: f64,
    /// Initialization (prep) error rate.
    pub p_init: f64,
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            p1: 0.01,
            p2: 0.01,
            p_meas: 0.01,
            p_init: 0.01,
        }
    }
}

impl NoiseConfig {
    /// Creates a new noise configuration.
    #[must_use]
    pub fn new(p1: f64, p2: f64, p_meas: f64, p_init: f64) -> Self {
        Self {
            p1,
            p2,
            p_meas,
            p_init,
        }
    }

    /// Creates a uniform noise configuration.
    #[must_use]
    pub fn uniform(p: f64) -> Self {
        Self {
            p1: p,
            p2: p,
            p_meas: p,
            p_init: p,
        }
    }
}

// ============================================================================
// Detector Error Model
// ============================================================================

/// A complete Detector Error Model (DEM).
///
/// This represents the noise model of a quantum circuit. It maps mechanisms
/// (detector/logical effects) to their probabilities.
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
    /// Logical observable definitions.
    pub observables: Vec<LogicalObservable>,
    /// Error contributions with source tracking.
    /// Each contribution tracks whether it came from a direct (X, Z) or decomposable (Y) source.
    contributions: Vec<FaultContribution>,
    /// Count of graphlike decomposable sources per 2-detector mechanism.
    /// Key is (d0, d1) with d0 < d1. A source is "graphlike decomposable" if both
    /// component effects are non-empty and graphlike (≤2 detectors).
    /// Used to determine output format: ≥2 → 3 forms, 1 → 2 forms, 0 → 1 form.
    graphlike_decomposable_counts: BTreeMap<(u32, u32), u32>,
}

impl DetectorErrorModel {
    /// Creates a new empty DEM.
    #[must_use]
    pub fn new() -> Self {
        Self {
            detectors: Vec::new(),
            observables: Vec::new(),
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
        self.observables.len()
    }

    /// Returns the number of tracked contributions.
    #[inline]
    #[must_use]
    pub fn num_contributions(&self) -> usize {
        self.contributions.len()
    }

    /// Returns debug info about contributions for a specific mechanism.
    ///
    /// Format: One line per contribution showing source type and probability.
    #[must_use]
    pub fn contributions_for_mechanism(&self, detectors: &[u32]) -> String {
        let target_dets: SmallVec<[u32; 4]> = detectors.iter().copied().collect();
        let mut lines = Vec::new();

        for contrib in &self.contributions {
            if contrib.effect.detectors == target_dets && contrib.effect.logicals.is_empty() {
                let source_type = match &contrib.source_type {
                    FaultSourceType::Direct => "Direct".to_string(),
                    FaultSourceType::DirectOneSidedComponent => {
                        "DirectOneSidedComponent".to_string()
                    }
                    FaultSourceType::YDecomposed {
                        x_detectors,
                        x_logicals,
                        z_detectors,
                        z_logicals,
                    } => {
                        let x_dets: Vec<_> = x_detectors.iter().map(|d| format!("D{d}")).collect();
                        let z_dets: Vec<_> = z_detectors.iter().map(|d| format!("D{d}")).collect();
                        let x_logs: Vec<_> = x_logicals.iter().map(|l| format!("L{l}")).collect();
                        let z_logs: Vec<_> = z_logicals.iter().map(|l| format!("L{l}")).collect();
                        format!(
                            "YDecomposed(X=[{}{}], Z=[{}{}])",
                            x_dets.join(" "),
                            if x_logs.is_empty() {
                                String::new()
                            } else {
                                format!(" {}", x_logs.join(" "))
                            },
                            z_dets.join(" "),
                            if z_logs.is_empty() {
                                String::new()
                            } else {
                                format!(" {}", z_logs.join(" "))
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

    /// Returns all contributions matching a full detector/logical effect.
    #[must_use]
    pub fn contributions_for_effect(
        &self,
        detectors: &[u32],
        logicals: &[u32],
    ) -> Vec<FaultContribution> {
        let target =
            FaultMechanism::from_unsorted(detectors.iter().copied(), logicals.iter().copied());
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
                .logicals
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
            if summary.effect.logicals.is_empty() && summary.effect.detectors.len() == 2 {
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
        // tracking purposes. This matches Stim's behavior where Y channels
        // contribute to decomposed output forms regardless of component structure.
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
    pub fn mark_graphlike_decomposable(&mut self, d0: u32, d1: u32) {
        let key = if d0 < d1 { (d0, d1) } else { (d1, d0) };
        *self.graphlike_decomposable_counts.entry(key).or_insert(0) += 1;
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

    /// Adds a logical observable definition.
    pub fn add_observable(&mut self, observable: LogicalObservable) {
        self.observables.push(observable);
    }

    /// Converts the DEM to a string in standard DEM format.
    ///
    /// Each fault mechanism is output with its total probability, with no
    /// splitting into decomposed forms. This matches Stim's
    /// `detector_error_model(decompose_errors=False)` output.
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

        // Add logical observable annotations
        for obs in &self.observables {
            lines.push(format!("logical_observable L{}", obs.id));
        }

        // Group contributions by effect, combining probabilities using XOR formula
        // (errors toggle detector bits, so two errors on same detector cancel)
        let mut by_effect: BTreeMap<FaultMechanism, f64> = BTreeMap::new();
        for contrib in &self.contributions {
            by_effect
                .entry(contrib.effect.clone())
                .and_modify(|p| *p = combine_independent_probs(*p, contrib.probability))
                .or_insert(contrib.probability);
        }

        // Output each mechanism with its total probability
        for (effect, total_prob) in by_effect {
            if effect.is_empty() || total_prob <= 0.0 {
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
            } else if contrib.effect.num_detectors() == 2 && contrib.effect.logicals.is_empty() {
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

        let effect = &contrib.effect;
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
            } else if effect.num_detectors() == 2 && effect.logicals.is_empty() {
                let direct_targets = Self::two_detector_direct_targets(effect, singleton_set);
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
                if let Some(decomp) = graphlike_index.find_hyperedge_decomposition(effect) {
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
                        format_mechanism_targets(effect),
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
        } else if effect.num_detectors() == 2 && effect.logicals.is_empty() {
            let direct_targets = Self::two_detector_direct_targets(effect, singleton_set);
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
            if let Some(decomp) = graphlike_index.find_hyperedge_decomposition(effect) {
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
                    format_mechanism_targets(effect),
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

    /// Converts the DEM to Stim format using source tracking (decomposed format).
    ///
    /// This matches Stim's `detector_error_model(decompose_errors=True)` output.
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
    /// when they carry multiple logical observables.
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

        // Add logical observable annotations
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
        // decomposed outputs. This is closer to Stim's decomposition pass, which
        // rewrites each error class before merging identical rewritten targets.
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
            if contrib.effect.is_graphlike() {
                graphlike.insert(contrib.effect.clone());
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
// Measurement Noise Model (MNM)
// ============================================================================

/// A measurement fault mechanism: a set of measurements that flip together.
///
/// Unlike [`FaultMechanism`] which operates on detectors, this operates directly
/// on raw measurement indices. This is useful for sampling measurement outcomes
/// without needing detector definitions.
///
/// Measurements are stored in sorted order for canonical representation.
#[derive(Clone, Default)]
pub struct MeasurementMechanism {
    /// Measurement indices that flip together (sorted).
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
        let mut meas: SmallVec<[u32; 4]> = measurements.into_iter().collect();
        meas.sort_unstable();
        Self { measurements: meas }
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

    /// Returns true if this mechanism has no effect (empty).
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

/// A Measurement Noise Model (MNM) for fast approximate sampling.
///
/// Unlike a DEM which maps fault mechanisms to detector effects, the MNM maps
/// directly to measurement effects. This allows sampling raw measurement outcomes
/// without needing detector definitions.
///
/// # Sampling Modes
///
/// - **Per-fault-location** (accurate): Sample each (location, Pauli) independently
/// - **Per-mechanism** (fast, approximate): Sample each unique measurement effect once
///
/// The MNM enables the fast per-mechanism mode while still producing raw measurement
/// outcomes that can be converted to detection events using any detector definition.
///
/// # Example
///
/// Build an MNM from a fault influence map and sample measurement outcomes.
/// In practice you will use [`MemBuilder`] to populate mechanisms; here we
/// use an empty MNM to keep the doctest self-contained.
///
/// ```
/// use pecos_qec::fault_tolerance::dem_builder::MeasurementNoiseModel;
/// use rand::SeedableRng;
/// use rand::rngs::StdRng;
///
/// let num_measurements = 4;
/// let mnm = MeasurementNoiseModel::new(num_measurements);
///
/// let mut outcomes = vec![false; num_measurements];
/// let mut rng = StdRng::seed_from_u64(0);
/// mnm.sample_into(&mut outcomes, &mut rng);
/// ```
#[derive(Debug, Clone, Default)]
pub struct MeasurementNoiseModel {
    /// Fault mechanisms mapped to their probabilities.
    /// Uses `BTreeMap` for deterministic iteration order.
    pub mechanisms: BTreeMap<MeasurementMechanism, f64>,
    /// Total number of measurements in the circuit.
    pub num_measurements: usize,
    /// Optional mapping from influence map index to `TickCircuit` index.
    /// If set, outcomes are reordered before detection event conversion.
    /// `im_to_tc`[`im_idx`] = `tc_idx`
    pub im_to_tc_order: Option<Vec<usize>>,
}

impl MeasurementNoiseModel {
    /// Creates a new empty MNM.
    #[must_use]
    pub fn new(num_measurements: usize) -> Self {
        Self {
            mechanisms: BTreeMap::new(),
            num_measurements,
            im_to_tc_order: None,
        }
    }

    /// Sets the measurement order mapping from influence map to `TickCircuit` order.
    ///
    /// This is needed when detector definitions use `TickCircuit` measurement indices
    /// but the influence map uses a different ordering.
    ///
    /// # Arguments
    ///
    /// * `im_to_tc` - Mapping where `im_to_tc[im_idx] = tc_idx`
    #[must_use]
    pub fn with_measurement_order(mut self, im_to_tc: Vec<usize>) -> Self {
        self.im_to_tc_order = Some(im_to_tc);
        self
    }

    /// Sets the measurement order mapping (mutable version).
    pub fn set_measurement_order(&mut self, im_to_tc: Vec<usize>) {
        self.im_to_tc_order = Some(im_to_tc);
    }

    /// Returns the number of distinct mechanisms.
    #[inline]
    #[must_use]
    pub fn num_mechanisms(&self) -> usize {
        self.mechanisms.len()
    }

    /// Adds an fault mechanism with the given probability.
    ///
    /// If the mechanism already exists, probabilities are combined
    /// using the independent error formula: p1*(1-p2) + p2*(1-p1).
    pub fn add_mechanism(&mut self, mechanism: MeasurementMechanism, probability: f64) {
        if mechanism.is_empty() || probability <= 0.0 {
            return;
        }

        self.mechanisms
            .entry(mechanism)
            .and_modify(|p| *p = combine_probabilities(*p, probability))
            .or_insert(probability);
    }

    /// Samples measurement outcomes into the provided buffer.
    ///
    /// Each mechanism is sampled once according to its probability.
    /// When a mechanism fires, its measurements are XOR'd into the outcomes.
    ///
    /// # Arguments
    ///
    /// * `outcomes` - Buffer to store measurement outcomes (must be pre-sized)
    /// * `rng` - Random number generator
    pub fn sample_into<R: rand::Rng>(&self, outcomes: &mut [bool], rng: &mut R) {
        // Clear outcomes
        outcomes.fill(false);

        for (mechanism, &prob) in &self.mechanisms {
            if rng.random::<f64>() < prob {
                for &meas_idx in &mechanism.measurements {
                    if (meas_idx as usize) < outcomes.len() {
                        outcomes[meas_idx as usize] ^= true;
                    }
                }
            }
        }
    }

    /// Samples and returns measurement outcomes as a vector.
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

    /// Converts measurement outcomes to detection events.
    ///
    /// Given raw measurement outcomes and detector definitions (as measurement indices),
    /// computes which detectors fire by XOR'ing the specified measurements for each detector.
    ///
    /// If `im_to_tc_order` is set, outcomes are first reordered from influence map
    /// order to `TickCircuit` order before applying detector records.
    ///
    /// # Arguments
    ///
    /// * `outcomes` - Raw measurement outcomes in influence map order (from `sample()`)
    /// * `detector_records` - For each detector, the list of measurement indices to XOR.
    ///   Indices can be negative (offset from end) or positive (absolute).
    ///   These indices refer to `TickCircuit` measurement order.
    ///
    /// # Returns
    ///
    /// Vector of detection events (true = detector fired)
    #[must_use]
    pub fn compute_detection_events(
        &self,
        outcomes: &[bool],
        detector_records: &[Vec<i32>],
    ) -> Vec<bool> {
        // Reorder outcomes from IM order to TC order if mapping is set
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

    /// Internal static helper for detection event conversion.
    fn to_detection_events_internal(outcomes: &[bool], detector_records: &[Vec<i32>]) -> Vec<bool> {
        let num_measurements = outcomes.len();
        let mut detection_events = Vec::with_capacity(detector_records.len());

        for records in detector_records {
            let mut fired = false;
            for &offset in records {
                if let Some(abs_idx) = record_offset_to_absolute_index(num_measurements, offset)
                    && abs_idx < num_measurements
                    && outcomes[abs_idx]
                {
                    fired = !fired; // XOR
                }
            }
            detection_events.push(fired);
        }

        detection_events
    }

    /// Static version without reordering (for backwards compatibility).
    #[must_use]
    pub fn to_detection_events(outcomes: &[bool], detector_records: &[Vec<i32>]) -> Vec<bool> {
        Self::to_detection_events_internal(outcomes, detector_records)
    }

    /// Samples and converts to detection events in one step.
    ///
    /// # Arguments
    ///
    /// * `detector_records` - For each detector, the measurement indices to XOR
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// Tuple of (`measurement_outcomes_in_im_order`, `detection_events`)
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
    ///
    /// This works identically to `compute_detection_events` - `XORing` measurement
    /// outcomes at the specified record positions. The difference is semantic:
    /// - Detection events indicate which detectors fired (syndrome)
    /// - Observable flips indicate which logical observables were flipped
    ///
    /// # Arguments
    ///
    /// * `outcomes` - Raw measurement outcomes in influence map order (from `sample()`)
    /// * `observable_records` - For each observable, the list of measurement indices to XOR.
    ///   Indices can be negative (offset from end) or positive (absolute).
    ///   These indices refer to `TickCircuit` measurement order.
    ///
    /// # Returns
    ///
    /// Vector of observable flips (true = observable was flipped by errors)
    #[must_use]
    pub fn compute_observable_flips(
        &self,
        outcomes: &[bool],
        observable_records: &[Vec<i32>],
    ) -> Vec<bool> {
        // Same logic as detection events - just different semantic meaning
        self.compute_detection_events(outcomes, observable_records)
    }

    /// Samples with full threshold estimation output.
    ///
    /// Returns detection events AND observable flips in one step, matching
    /// Stim's DEM sampler output format.
    ///
    /// # Arguments
    ///
    /// * `detector_records` - For each detector, the measurement indices to XOR
    /// * `observable_records` - For each observable, the measurement indices to XOR
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// Tuple of (`detection_events`, `observable_flips`)
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

    /// Batch sampling for threshold estimation.
    ///
    /// Efficiently samples multiple shots and returns detection events and observable
    /// flips for each shot.
    ///
    /// # Arguments
    ///
    /// * `num_shots` - Number of shots to sample
    /// * `detector_records` - For each detector, the measurement indices to XOR
    /// * `observable_records` - For each observable, the measurement indices to XOR
    /// * `rng` - Random number generator
    ///
    /// # Returns
    ///
    /// Tuple of (`detection_events_per_shot`, `observable_flips_per_shot`)
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
    for &log in &mechanism.logicals {
        targets.push(format!("L{log}"));
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
        // Logicals: {0} XOR {0, 1} = {1}
        assert_eq!(result.logicals.as_slice(), &[1]);
    }

    #[test]
    fn test_error_mechanism_equality() {
        let m1 = FaultMechanism::from_unsorted([2, 0, 1], [1, 0]);
        let m2 = FaultMechanism::from_unsorted([0, 1, 2], [0, 1]);

        assert_eq!(m1, m2);
        assert_eq!(m1.detectors.as_slice(), &[0, 1, 2]);
        assert_eq!(m1.logicals.as_slice(), &[0, 1]);
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
    fn test_decomposed_error_single() {
        let mechanism = FaultMechanism::from_unsorted([0, 1], [0]);
        let decomposed = DecomposedFault::single(mechanism.clone());

        assert_eq!(decomposed.components.len(), 1);
        assert!(decomposed.is_graphlike());
        assert_eq!(decomposed.full_effect(), mechanism);
        assert_eq!(decomposed.to_stim_targets(), "D0 D1 L0");
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
        dem.add_observable(LogicalObservable::new(0));

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
                summary.effect.detectors.as_slice() == [0, 1] && summary.effect.logicals.is_empty()
            })
            .expect("pair summary missing");
        assert_eq!(pair_summary.graphlike_decomposable_count, 2);

        let singleton_summary = summaries
            .iter()
            .find(|summary| {
                summary.effect.detectors.as_slice() == [0] && summary.effect.logicals.is_empty()
            })
            .expect("singleton summary missing");
        assert_eq!(singleton_summary.graphlike_decomposable_count, 0);
    }

    #[test]
    fn test_dem_to_string() {
        let mut dem = DetectorErrorModel::new();

        dem.add_detector(DetectorDef::new(0).with_coords([0.0, 0.0, 0.0]));
        dem.add_detector(DetectorDef::new(1).with_coords([1.0, 0.0, 0.0]));
        dem.add_observable(LogicalObservable::new(0));

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
    fn test_dem_to_string_decomposed_keeps_two_detector_one_logical_direct() {
        let mut dem = DetectorErrorModel::new();

        dem.add_detector(DetectorDef::new(0).with_coords([0.0, 0.0, 0.0]));
        dem.add_detector(DetectorDef::new(1).with_coords([1.0, 0.0, 0.0]));
        dem.add_observable(LogicalObservable::new(0));

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
        dem.add_observable(LogicalObservable::new(0));

        let x = FaultMechanism::from_unsorted([0], std::iter::empty());
        let z = FaultMechanism::from_unsorted([1], [0]);
        dem.add_y_decomposed_contribution(&x, &z, 0.01);

        let stim_str = dem.to_string_decomposed();

        assert!(stim_str.contains("error(0.01) D0 ^ D1 L0"));
        assert!(!stim_str.contains("error(0.01) D0 D1 L0"));
    }

    #[test]
    fn test_error_mechanism_with_two_detectors_and_multiple_logicals_is_graphlike() {
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
