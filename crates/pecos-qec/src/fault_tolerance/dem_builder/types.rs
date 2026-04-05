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
//! This module provides data structures for representing error mechanisms,
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

use rand::RngExt;
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};

// ============================================================================
// Error Source Tracking
// ============================================================================

/// Classification of error sources for decomposition decisions.
///
/// This tracks how an error contribution was generated, which determines
/// how it should be output in the decomposed DEM format:
/// - Direct errors (X, Z channels) -> output as direct form
/// - Y-decomposed errors -> output as decomposed form (X ^ Z)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorSourceType {
    /// Direct X or Z error channel - outputs as direct form only.
    /// These represent single Pauli errors that cannot be further decomposed.
    Direct,

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

/// An error contribution with source tracking.
///
/// This represents a single error source's contribution to the DEM,
/// tracking both its effect and how it was generated. Multiple contributions
/// with the same effect are grouped at output time, with their source types
/// determining how they are output (direct vs decomposed forms).
#[derive(Debug, Clone)]
pub struct ErrorContribution {
    /// The detector/logical effect of this error.
    pub effect: ErrorMechanism,

    /// Probability of this error.
    pub probability: f64,

    /// Source classification for decomposition decisions.
    pub source_type: ErrorSourceType,
}

impl ErrorContribution {
    /// Creates a new direct error contribution (X or Z channel).
    #[must_use]
    pub fn direct(effect: ErrorMechanism, probability: f64) -> Self {
        Self {
            effect,
            probability,
            source_type: ErrorSourceType::Direct,
        }
    }

    /// Creates a new Y-decomposed error contribution.
    ///
    /// The combined effect is stored along with the X and Z component effects,
    /// allowing the decomposed form (X ^ Z) to be output.
    #[must_use]
    pub fn y_decomposed(
        combined_effect: ErrorMechanism,
        x_effect: &ErrorMechanism,
        z_effect: &ErrorMechanism,
        probability: f64,
    ) -> Self {
        Self {
            effect: combined_effect,
            probability,
            source_type: ErrorSourceType::YDecomposed {
                x_detectors: x_effect.detectors.clone(),
                x_logicals: x_effect.logicals.clone(),
                z_detectors: z_effect.detectors.clone(),
                z_logicals: z_effect.logicals.clone(),
            },
        }
    }

    /// Returns true if this is a direct (non-decomposable) source.
    #[must_use]
    pub fn is_direct(&self) -> bool {
        matches!(self.source_type, ErrorSourceType::Direct)
    }

    /// Returns the X and Z components if this is a Y-decomposed source.
    #[must_use]
    pub fn decomposition_components(&self) -> Option<(ErrorMechanism, ErrorMechanism)> {
        match &self.source_type {
            ErrorSourceType::YDecomposed {
                x_detectors,
                x_logicals,
                z_detectors,
                z_logicals,
            } => {
                let x = ErrorMechanism::from_sorted(x_detectors.clone(), x_logicals.clone());
                let z = ErrorMechanism::from_sorted(z_detectors.clone(), z_logicals.clone());
                Some((x, z))
            }
            ErrorSourceType::Direct => None,
        }
    }
}

// ============================================================================
// Error Mechanism
// ============================================================================

/// An error mechanism: a set of detectors and logical observables that flip together.
///
/// When an error occurs, it flips a specific set of detectors and may flip
/// logical observables. Mechanisms with the same effect are aggregated together.
///
/// The detectors and logicals are stored in sorted order for canonical representation.
#[derive(Clone, Default)]
pub struct ErrorMechanism {
    /// Detector indices that flip together (sorted).
    pub detectors: SmallVec<[u32; 4]>,
    /// Logical observable indices that flip together (sorted).
    pub logicals: SmallVec<[u32; 2]>,
}

impl ErrorMechanism {
    /// Creates a new empty error mechanism.
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
    /// A graphlike mechanism has at most 2 detectors and at most 1 logical observable.
    /// MWPM decoders can only handle graphlike errors directly.
    #[inline]
    #[must_use]
    pub fn is_graphlike(&self) -> bool {
        self.detectors.len() <= 2 && self.logicals.len() <= 1
    }

    /// Returns true if this mechanism is a hyperedge (not graphlike).
    ///
    /// Hyperedge mechanisms have 3+ detectors or 2+ logicals and need to be
    /// decomposed into graphlike components for MWPM decoders.
    #[inline]
    #[must_use]
    pub fn is_hyperedge(&self) -> bool {
        self.detectors.len() > 2 || self.logicals.len() > 1
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

impl PartialEq for ErrorMechanism {
    fn eq(&self, other: &Self) -> bool {
        self.detectors == other.detectors && self.logicals == other.logicals
    }
}

impl Eq for ErrorMechanism {}

impl Hash for ErrorMechanism {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.detectors.hash(state);
        self.logicals.hash(state);
    }
}

impl PartialOrd for ErrorMechanism {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ErrorMechanism {
    fn cmp(&self, other: &Self) -> Ordering {
        self.detectors
            .cmp(&other.detectors)
            .then_with(|| self.logicals.cmp(&other.logicals))
    }
}

impl fmt::Debug for ErrorMechanism {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ErrorMechanism(dets={:?}, logs={:?})",
            self.detectors.as_slice(),
            self.logicals.as_slice()
        )
    }
}

// ============================================================================
// Decomposed Error
// ============================================================================

/// A decomposed error mechanism with optional decomposition into graphlike parts.
///
/// When an error affects 3+ detectors (a hyperedge), it can be decomposed into
/// a combination of graphlike errors (affecting 1-2 detectors each) connected
/// by `^` separators indicating XOR composition.
#[derive(Clone, Debug)]
pub struct DecomposedError {
    /// The component error mechanisms (separated by `^` in DEM format).
    /// For graphlike errors, this has a single element.
    /// For decomposed hyperedges, this has multiple elements.
    pub components: SmallVec<[ErrorMechanism; 2]>,
}

impl DecomposedError {
    /// Creates a new decomposed error from a single mechanism.
    #[must_use]
    pub fn single(mechanism: ErrorMechanism) -> Self {
        let mut components = SmallVec::new();
        components.push(mechanism);
        Self { components }
    }

    /// Creates a decomposed error from multiple components.
    #[must_use]
    pub fn decomposed(components: impl IntoIterator<Item = ErrorMechanism>) -> Self {
        Self {
            components: components.into_iter().collect(),
        }
    }

    /// Returns the full effect of this error (XOR of all components).
    #[must_use]
    pub fn full_effect(&self) -> ErrorMechanism {
        let mut result = ErrorMechanism::new();
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
/// A hyperedge is an error mechanism with 3+ detectors or 2+ logicals.
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
/// A vector of decompositions, where each decomposition is a vector of
/// graphlike mechanisms whose XOR equals the original hyperedge.
/// Returns an empty vector if no valid decomposition exists.
///
/// # Algorithm
///
/// Uses a type-aware search that distinguishes between:
/// - 2-part decompositions (hyperedge = A XOR B)
/// - 3-part decompositions (hyperedge = A XOR B XOR C)
///
/// This matches Stim's behavior of outputting:
/// - 1 form if only one decomposition size exists
/// - 2 forms if both 2-part and 3-part decompositions exist
///
/// Additionally, decompositions are filtered to only include those where
/// all component detectors are subsets of the original hyperedge's detectors.
/// This matches Stim's behavior of not introducing extra detectors.
///
/// # Steps
/// 1. Find decompositions, preferring those with smaller components
/// 2. Return one 2-part and one 3-part if both exist
pub fn find_hyperedge_decompositions(
    hyperedge: &ErrorMechanism,
    graphlike_set: &HashSet<ErrorMechanism>,
) -> Vec<Vec<ErrorMechanism>> {
    // If already graphlike, no decomposition needed
    if hyperedge.is_graphlike() {
        return vec![vec![hyperedge.clone()]];
    }

    // Collect the set of detectors in the hyperedge
    let hyperedge_dets: HashSet<u32> = hyperedge.detectors.iter().copied().collect();

    // Helper to check if all detectors in a decomposition are in the hyperedge
    let decomp_dets_valid = |decomp: &[ErrorMechanism]| -> bool {
        decomp
            .iter()
            .flat_map(|m| m.detectors.iter())
            .all(|d| hyperedge_dets.contains(d))
    };

    // Helper to compute the maximum component size (prefer smaller)
    let max_component_size = |decomp: &[ErrorMechanism]| -> usize {
        decomp.iter().map(|m| m.detectors.len()).max().unwrap_or(0)
    };

    // Find best 2-part and best 3-part decomposition (prefer smaller components)
    let mut two_part_decomp: Option<Vec<ErrorMechanism>> = None;
    let mut two_part_max_size: usize = usize::MAX;
    let mut three_part_decomp: Option<Vec<ErrorMechanism>> = None;
    let mut three_part_max_size: usize = usize::MAX;

    // Try 2-part decompositions
    for g1 in graphlike_set {
        // g1 must share at least one element with the hyperedge
        if !shares_element(g1, hyperedge) {
            continue;
        }

        let remainder = hyperedge.xor(g1);

        // If remainder is graphlike and in the set, we found a 2-part decomposition
        if remainder.is_graphlike() && graphlike_set.contains(&remainder) {
            // Verify: g1 XOR remainder should equal hyperedge
            let check = g1.xor(&remainder);
            if check != *hyperedge {
                continue;
            }

            // Canonicalize ordering
            let decomp = if g1 < &remainder {
                vec![g1.clone(), remainder]
            } else {
                vec![remainder, g1.clone()]
            };

            // Check that all detectors in components are in the hyperedge
            if decomp_dets_valid(&decomp) {
                let size = max_component_size(&decomp);
                if size < two_part_max_size {
                    two_part_max_size = size;
                    two_part_decomp = Some(decomp);
                }
            }
        }
    }

    // Try 3-part decompositions
    for g1 in graphlike_set {
        if !shares_element(g1, hyperedge) {
            continue;
        }

        let after_g1 = hyperedge.xor(g1);
        if after_g1.is_graphlike() {
            continue; // Would be a 2-part decomposition
        }

        for g2 in graphlike_set {
            if g2 <= g1 {
                continue; // Avoid duplicates
            }
            if !shares_element(g2, &after_g1) {
                continue;
            }

            let after_g2 = after_g1.xor(g2);

            // If remainder is graphlike and in the set, we found a 3-part decomposition
            if after_g2.is_graphlike() && graphlike_set.contains(&after_g2) {
                // Verify: g1 XOR g2 XOR after_g2 should equal hyperedge
                let check = g1.xor(g2).xor(&after_g2);
                if check != *hyperedge {
                    continue;
                }

                let mut parts = vec![g1.clone(), g2.clone(), after_g2];
                parts.sort();

                // Check that all detectors in components are in the hyperedge
                if decomp_dets_valid(&parts) {
                    let size = max_component_size(&parts);
                    if size < three_part_max_size {
                        three_part_max_size = size;
                        three_part_decomp = Some(parts);
                    }
                }
            }
        }
    }

    // Combine results: output both types if available
    let mut result = Vec::new();
    if let Some(decomp) = two_part_decomp {
        result.push(decomp);
    }
    if let Some(decomp) = three_part_decomp {
        result.push(decomp);
    }
    result
}

/// Checks if two mechanisms share at least one detector or logical.
fn shares_element(a: &ErrorMechanism, b: &ErrorMechanism) -> bool {
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
    contributions: Vec<ErrorContribution>,
    /// Count of graphlike decomposable sources per 2-detector mechanism.
    /// Key is (d0, d1) with d0 < d1. A source is "graphlike decomposable" if both
    /// component effects are non-empty and graphlike (≤2 detectors).
    /// Used to determine output format: ≥2 → 3 forms, 1 → 2 forms, 0 → 1 form.
    graphlike_decomposable_counts: HashMap<(u32, u32), u32>,
}

impl DetectorErrorModel {
    /// Creates a new empty DEM.
    #[must_use]
    pub fn new() -> Self {
        Self {
            detectors: Vec::new(),
            observables: Vec::new(),
            contributions: Vec::new(),
            graphlike_decomposable_counts: HashMap::new(),
        }
    }

    /// Creates a DEM with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(num_detectors: usize, num_observables: usize) -> Self {
        Self {
            detectors: Vec::with_capacity(num_detectors),
            observables: Vec::with_capacity(num_observables),
            contributions: Vec::new(),
            graphlike_decomposable_counts: HashMap::new(),
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
                    ErrorSourceType::Direct => "Direct".to_string(),
                    ErrorSourceType::YDecomposed {
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

    /// Adds a direct error contribution (X or Z channel).
    ///
    /// Direct contributions are output as direct forms (e.g., "D0 D1") rather than
    /// decomposed forms. Use this for X and Z error channels.
    ///
    /// Requires source tracking to be enabled.
    pub fn add_direct_contribution(&mut self, effect: ErrorMechanism, probability: f64) {
        if effect.is_empty() || probability <= 0.0 {
            return;
        }
        self.contributions
            .push(ErrorContribution::direct(effect, probability));
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
        x_effect: &ErrorMechanism,
        z_effect: &ErrorMechanism,
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

        // Always record as YDecomposed since the source is a Y-containing channel.
        // The distinction between Direct and YDecomposed affects output form selection.
        self.contributions.push(ErrorContribution::y_decomposed(
            combined,
            x_effect,
            z_effect,
            probability,
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
    /// Each error mechanism is output with its total probability, with no
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
        let mut by_effect: BTreeMap<ErrorMechanism, f64> = BTreeMap::new();
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

    /// Converts the DEM to Stim format using source tracking (decomposed format).
    ///
    /// This matches Stim's `detector_error_model(decompose_errors=True)` output.
    /// Error mechanisms are split into direct and decomposed forms based on
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
    /// - If both Di L0 and Dj L0 exist as mechanisms, outputs both direct form
    ///   and L0 cancellation form (Di L0 ^ Dj L0), with probability split based
    ///   on relative mechanism probabilities.
    /// - Otherwise, outputs decomposed forms (Di ^ Dj, Dj ^ Di) with probability split.
    ///
    /// This provides representation diversity for decoders, similar to Stim's
    /// `decompose_errors=True` behavior.
    #[must_use]
    pub fn to_string_decomposed(&self) -> String {
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

        // Find standalone detectors from contributions
        let mut standalone_detectors: std::collections::HashSet<u32> =
            std::collections::HashSet::new();
        for contrib in &self.contributions {
            if contrib.effect.num_detectors() == 1 && contrib.effect.logicals.is_empty() {
                standalone_detectors.insert(contrib.effect.detectors[0]);
            }
        }

        // Find single-detector + L0 mechanisms (Di L0) and their probabilities
        // These can be used for L0 cancellation decomposition
        let mut det_l0_probs: HashMap<u32, f64> = HashMap::new();
        for contrib in &self.contributions {
            if contrib.effect.num_detectors() == 1
                && contrib.effect.logicals.len() == 1
                && contrib.effect.logicals[0] == 0
            {
                let det_id = contrib.effect.detectors[0];
                det_l0_probs
                    .entry(det_id)
                    .and_modify(|p| *p = combine_independent_probs(*p, contrib.probability))
                    .or_insert(contrib.probability);
            }
        }

        // Group contributions by effect, combining probabilities using independent error formula
        // p_combined = p1 + p2 - p1*p2 = 1 - (1-p1)*(1-p2)
        let mut by_effect: BTreeMap<ErrorMechanism, f64> = BTreeMap::new();
        for contrib in &self.contributions {
            by_effect
                .entry(contrib.effect.clone())
                .and_modify(|p| *p = combine_independent_probs(*p, contrib.probability))
                .or_insert(contrib.probability);
        }

        // Process each unique effect
        for (effect, total_prob) in &by_effect {
            if effect.is_empty() || *total_prob <= 0.0 {
                continue;
            }

            // Check if this is a 2-detector mechanism with no logicals
            let is_2det_no_logical = effect.num_detectors() == 2 && effect.logicals.is_empty();

            if is_2det_no_logical {
                let d0 = effect.detectors[0];
                let d1 = effect.detectors[1];

                // Check if L0 cancellation decomposition is possible
                // (both Di L0 and Dj L0 exist as mechanisms)
                let has_d0_l0 = det_l0_probs.contains_key(&d0);
                let has_d1_l0 = det_l0_probs.contains_key(&d1);

                if has_d0_l0 && has_d1_l0 {
                    // L0 cancellation is possible: Di Dj can be represented as Di L0 ^ Dj L0
                    // Split probability between direct form and L0 cancellation form
                    //
                    // Heuristic: Use approximately 25% for L0 cancellation form, which
                    // matches the average ratio observed in Stim's decomposed output.
                    // The exact split varies in Stim (10-50%), but 25% is a reasonable
                    // approximation for decoder compatibility.
                    let l0_fraction = 0.25;
                    let direct_fraction = 1.0 - l0_fraction;

                    let direct_prob = total_prob * direct_fraction;
                    let l0_prob = total_prob * l0_fraction;

                    // Direct form
                    if direct_prob > 0.0 {
                        lines.push(format!(
                            "error({}) D{} D{}",
                            format_probability(direct_prob),
                            d0,
                            d1
                        ));
                    }

                    // L0 cancellation form
                    if l0_prob > 0.0 {
                        lines.push(format!(
                            "error({}) D{} L0 ^ D{} L0",
                            format_probability(l0_prob),
                            d0,
                            d1
                        ));
                    }
                } else if standalone_detectors.contains(&d0) && standalone_detectors.contains(&d1) {
                    // Both detectors have standalone mechanisms - use compact decomposition
                    // (matching Stim's approach of minimal entries)
                    let graphlike_count = self.graphlike_decomposable_count(d0, d1);

                    if graphlike_count >= 2 {
                        // Direct form only - both detectors flip together
                        lines.push(format!(
                            "error({}) D{} D{}",
                            format_probability(*total_prob),
                            d0,
                            d1
                        ));
                    } else {
                        // Decomposed form - one ordering only
                        lines.push(format!(
                            "error({}) D{} ^ D{}",
                            format_probability(*total_prob),
                            d0,
                            d1
                        ));
                    }
                } else {
                    // Neither L0 cancellation nor standalone decomposition possible
                    // Output as direct form
                    lines.push(format!(
                        "error({}) D{} D{}",
                        format_probability(*total_prob),
                        d0,
                        d1
                    ));
                }
            } else if effect.is_hyperedge() {
                // Hyperedge (3+ detectors or 2+ logicals): try to decompose
                let graphlike_set = self.collect_graphlike_mechanisms();
                let decompositions = find_hyperedge_decompositions(effect, &graphlike_set);

                if decompositions.is_empty() {
                    // No valid decomposition found - output as direct form
                    let targets = format_mechanism_targets(effect);
                    if !targets.is_empty() {
                        lines.push(format!(
                            "error({}) {}",
                            format_probability(*total_prob),
                            targets
                        ));
                    }
                } else {
                    // Split probability across decompositions
                    #[allow(clippy::cast_precision_loss)]
                    let split_prob = *total_prob / decompositions.len() as f64;
                    for decomp in decompositions {
                        let targets = decomp
                            .iter()
                            .map(format_mechanism_targets)
                            .collect::<Vec<_>>()
                            .join(" ^ ");
                        lines.push(format!(
                            "error({}) {}",
                            format_probability(split_prob),
                            targets
                        ));
                    }
                }
            } else if effect.num_detectors() == 2 && effect.num_logicals() == 1 {
                // 2-detector + 1-logical mechanism: try to decompose as D_i ^ D_j L_k
                // This matches Stim's behavior of decomposing these into components
                let graphlike_set = self.collect_graphlike_mechanisms();

                let d0 = effect.detectors[0];
                let d1 = effect.detectors[1];
                let l0 = effect.logicals[0];

                // Try decomposition: D0 ^ (D1 L0) or D1 ^ (D0 L0)
                let comp_d0 = ErrorMechanism::from_unsorted([d0], std::iter::empty());
                let comp_d1_l0 = ErrorMechanism::from_unsorted([d1], [l0]);
                let comp_d1 = ErrorMechanism::from_unsorted([d1], std::iter::empty());
                let comp_d0_l0 = ErrorMechanism::from_unsorted([d0], [l0]);

                let can_decompose_1 =
                    graphlike_set.contains(&comp_d0) && graphlike_set.contains(&comp_d1_l0);
                let can_decompose_2 =
                    graphlike_set.contains(&comp_d1) && graphlike_set.contains(&comp_d0_l0);

                if can_decompose_1 || can_decompose_2 {
                    // Output decomposed form
                    if can_decompose_1 {
                        lines.push(format!(
                            "error({}) D{} ^ D{} L{}",
                            format_probability(*total_prob),
                            d0,
                            d1,
                            l0
                        ));
                    } else {
                        lines.push(format!(
                            "error({}) D{} ^ D{} L{}",
                            format_probability(*total_prob),
                            d1,
                            d0,
                            l0
                        ));
                    }
                } else {
                    // Can't decompose - output as direct form
                    let targets = format_mechanism_targets(effect);
                    if !targets.is_empty() {
                        lines.push(format!(
                            "error({}) {}",
                            format_probability(*total_prob),
                            targets
                        ));
                    }
                }
            } else {
                // Other graphlike mechanism - output as direct form
                let targets = format_mechanism_targets(effect);
                if !targets.is_empty() {
                    lines.push(format!(
                        "error({}) {}",
                        format_probability(*total_prob),
                        targets
                    ));
                }
            }
        }

        lines.join("\n")
    }

    /// Collects all graphlike mechanisms from contributions.
    ///
    /// Returns a set of mechanisms with ≤2 detectors and ≤1 logical,
    /// which can be used as components for hyperedge decomposition.
    fn collect_graphlike_mechanisms(&self) -> HashSet<ErrorMechanism> {
        let mut graphlike = HashSet::new();
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

/// A measurement error mechanism: a set of measurements that flip together.
///
/// Unlike [`ErrorMechanism`] which operates on detectors, this operates directly
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
/// Unlike a DEM which maps error mechanisms to detector effects, the MNM maps
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
/// ```ignore
/// let mnm = MeasurementNoiseModel::from_influence_map(&influence_map, &noise);
///
/// // Sample measurement outcomes
/// let mut outcomes = vec![false; num_measurements];
/// mnm.sample_into(&mut outcomes, &mut rng);
/// ```
#[derive(Debug, Clone, Default)]
pub struct MeasurementNoiseModel {
    /// Error mechanisms mapped to their probabilities.
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

    /// Adds an error mechanism with the given probability.
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
                // Convert negative offset to absolute index
                #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)] // measurement count fits in i32
                #[allow(clippy::cast_sign_loss)]
                // negative offset + total count, or non-negative offset
                let abs_idx = if offset < 0 {
                    (num_measurements as i32 + offset) as usize
                } else {
                    offset as usize
                };

                if abs_idx < num_measurements && outcomes[abs_idx] {
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
/// error mechanism can be triggered by multiple independent error sources.
#[inline]
#[must_use]
pub fn combine_probabilities(p1: f64, p2: f64) -> f64 {
    p1 * (1.0 - p2) + p2 * (1.0 - p1)
}

/// Formats an error mechanism's targets as a string (e.g., "D0 D1 L0").
fn format_mechanism_targets(mechanism: &ErrorMechanism) -> String {
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
        let m1 = ErrorMechanism::from_unsorted([0, 1, 2], [0]);
        let m2 = ErrorMechanism::from_unsorted([1, 2, 3], [0, 1]);

        let result = m1.xor(&m2);

        // Detectors: {0, 1, 2} XOR {1, 2, 3} = {0, 3}
        assert_eq!(result.detectors.as_slice(), &[0, 3]);
        // Logicals: {0} XOR {0, 1} = {1}
        assert_eq!(result.logicals.as_slice(), &[1]);
    }

    #[test]
    fn test_error_mechanism_equality() {
        let m1 = ErrorMechanism::from_unsorted([2, 0, 1], [1, 0]);
        let m2 = ErrorMechanism::from_unsorted([0, 1, 2], [0, 1]);

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
        let mechanism = ErrorMechanism::from_unsorted([0, 1], [0]);
        let decomposed = DecomposedError::single(mechanism.clone());

        assert_eq!(decomposed.components.len(), 1);
        assert!(decomposed.is_graphlike());
        assert_eq!(decomposed.full_effect(), mechanism);
        assert_eq!(decomposed.to_stim_targets(), "D0 D1 L0");
    }

    #[test]
    fn test_decomposed_error_multi() {
        let m1 = ErrorMechanism::from_unsorted([0, 1], []);
        let m2 = ErrorMechanism::from_unsorted([2, 3], [0]);
        let decomposed = DecomposedError::decomposed([m1.clone(), m2.clone()]);

        assert_eq!(decomposed.components.len(), 2);
        assert!(decomposed.is_graphlike());

        // Full effect should be XOR of both
        let expected = m1.xor(&m2);
        assert_eq!(decomposed.full_effect(), expected);

        // DEM format should use ^ separator for decomposed entries
        assert_eq!(decomposed.to_stim_targets(), "D0 D1 ^ D2 D3 L0");
    }

    #[test]
    fn test_dem_to_string() {
        let mut dem = DetectorErrorModel::new();

        dem.add_detector(DetectorDef::new(0).with_coords([0.0, 0.0, 0.0]));
        dem.add_detector(DetectorDef::new(1).with_coords([1.0, 0.0, 0.0]));
        dem.add_observable(LogicalObservable::new(0));

        // Add contributions directly using the source tracking API
        dem.add_direct_contribution(ErrorMechanism::from_unsorted([0, 1], []), 0.01);
        dem.add_direct_contribution(ErrorMechanism::from_unsorted([1], [0]), 0.005);

        let stim_str = dem.to_string();

        assert!(stim_str.contains("detector(0, 0, 0) D0"));
        assert!(stim_str.contains("detector(1, 0, 0) D1"));
        assert!(stim_str.contains("logical_observable L0"));
        assert!(stim_str.contains("error(0.01) D0 D1"));
        assert!(stim_str.contains("error(0.005) D1 L0"));
    }
}
