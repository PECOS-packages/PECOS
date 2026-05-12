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

//! DEM (Detector Error Model) expression equivalence validation.
//!
//! This module provides utilities for comparing DEM expressions to determine
//! if they are semantically equivalent, even when their representations differ.
//!
//! # Key Concepts
//!
//! - Two DEMs are equivalent if they produce the same probability distribution
//!   over (`detector_events`, `dem_output_flips`) patterns.
//! - Decomposed DEMs (using ^) create independent error channels that are `XORed`.
//! - Different decomposition strategies can produce equivalent sampling results.
//! - For non-decomposed DEMs, mechanisms must match exactly.
//!
//! # Comparison Methods
//!
//! - **Exact comparison**: Compares aggregated mechanisms and probabilities directly.
//!   Appropriate for non-decomposed DEMs.
//!
//! - **Statistical comparison**: Samples from both DEMs and compares syndrome/observable
//!   distributions. More robust but slower.
//!
//! # Example
//!
//! ```
//! use pecos_qec::fault_tolerance::dem_builder::{
//!     ParsedDem, compare_dems_exact, compare_dems_statistical,
//! };
//!
//! let dem_str = "error(0.01) D0 D1\nerror(0.02) D1 D2 L0\n";
//!
//! let dem1 = ParsedDem::parse(dem_str).unwrap();
//! let dem2 = ParsedDem::parse(dem_str).unwrap();
//!
//! // Exact comparison
//! let result = compare_dems_exact(&dem1, &dem2, 1e-6);
//! assert!(result.equivalent);
//!
//! // Statistical comparison (more robust for decomposed DEMs)
//! let result = compare_dems_statistical(&dem1, &dem2, 10_000, 42, 0.02);
//! assert!(result.equivalent);
//! ```

use pecos_random::{PecosRng, Rng, RngExt};
use std::collections::{BTreeMap, BTreeSet};

use std::fmt;
use std::str::FromStr;

use super::types::{DemOutput, combine_probabilities, parse_pecos_dem_metadata_line};

// ============================================================================
// Parsed DEM Types
// ============================================================================

/// A single fault mechanism parsed from DEM format.
///
/// Can represent both simple mechanisms (D0 D1) and decomposed ones (D0 ^ D1).
#[derive(Debug, Clone)]
pub struct ParsedMechanism {
    /// Probability of this mechanism.
    pub probability: f64,
    /// Components of this mechanism.
    /// For simple mechanisms, this has one element.
    /// For decomposed mechanisms (with ^), this has multiple elements.
    pub components: Vec<MechanismComponent>,
}

impl ParsedMechanism {
    /// Creates a new simple mechanism (no decomposition).
    #[must_use]
    pub fn simple(probability: f64, detectors: Vec<u32>, observables: Vec<u32>) -> Self {
        Self {
            probability,
            components: vec![MechanismComponent {
                detectors,
                observables,
                tracked_paulis: Vec::new(),
            }],
        }
    }

    /// Returns true if this mechanism is decomposed (has multiple components).
    #[must_use]
    pub fn is_decomposed(&self) -> bool {
        self.components.len() > 1
    }

    /// Returns the combined effect of this mechanism (XOR of all components).
    #[must_use]
    pub fn combined_effect(&self) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let mut all_dets: BTreeSet<u32> = BTreeSet::new();
        let mut all_obs: BTreeSet<u32> = BTreeSet::new();
        let mut all_tracked_paulis: BTreeSet<u32> = BTreeSet::new();

        for comp in &self.components {
            for &d in &comp.detectors {
                if all_dets.contains(&d) {
                    all_dets.remove(&d);
                } else {
                    all_dets.insert(d);
                }
            }
            for &o in &comp.observables {
                if all_obs.contains(&o) {
                    all_obs.remove(&o);
                } else {
                    all_obs.insert(o);
                }
            }
            for &op in &comp.tracked_paulis {
                if all_tracked_paulis.contains(&op) {
                    all_tracked_paulis.remove(&op);
                } else {
                    all_tracked_paulis.insert(op);
                }
            }
        }

        // BTreeSet is already sorted, so just collect
        let dets: Vec<u32> = all_dets.into_iter().collect();
        let obs: Vec<u32> = all_obs.into_iter().collect();
        let tracked_paulis: Vec<u32> = all_tracked_paulis.into_iter().collect();
        (dets, obs, tracked_paulis)
    }

    /// Creates an effect key for this mechanism (for aggregation).
    #[must_use]
    pub fn effect_key(&self) -> EffectKey {
        let (dets, obs, tracked_paulis) = self.combined_effect();
        EffectKey {
            detectors: dets,
            observables: obs,
            tracked_paulis,
        }
    }

    /// Format the target string for this mechanism (e.g., "D0 D3 L0" or "D0 ^ D1 D3").
    #[must_use]
    pub fn format_targets(&self) -> String {
        let parts: Vec<String> = self
            .components
            .iter()
            .map(|comp| {
                let mut tokens = Vec::new();
                for &d in &comp.detectors {
                    tokens.push(format!("D{d}"));
                }
                for &o in &comp.observables {
                    tokens.push(format!("L{o}"));
                }
                for &op in &comp.tracked_paulis {
                    tokens.push(format!("TP{op}"));
                }
                tokens.join(" ")
            })
            .collect();
        parts.join(" ^ ")
    }
}

/// A single component of a mechanism (the part between ^ separators).
#[derive(Debug, Clone)]
pub struct MechanismComponent {
    /// Detector IDs in this component.
    pub detectors: Vec<u32>,
    /// Observable IDs in this component.
    pub observables: Vec<u32>,
    /// PECOS tracked-Pauli IDs in this component.
    pub tracked_paulis: Vec<u32>,
}

/// Key for aggregating mechanisms by their effect.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EffectKey {
    /// Sorted detector IDs.
    pub detectors: Vec<u32>,
    /// Sorted observable IDs.
    pub observables: Vec<u32>,
    /// Sorted tracked-Pauli IDs.
    pub tracked_paulis: Vec<u32>,
}

impl EffectKey {
    /// Creates a new effect key.
    #[must_use]
    pub fn new(mut detectors: Vec<u32>, mut observables: Vec<u32>) -> Self {
        detectors.sort_unstable();
        observables.sort_unstable();
        Self {
            detectors,
            observables,
            tracked_paulis: Vec::new(),
        }
    }
}

impl fmt::Display for EffectKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts: Vec<String> = Vec::new();
        for &d in &self.detectors {
            parts.push(format!("D{d}"));
        }
        for &o in &self.observables {
            parts.push(format!("L{o}"));
        }
        for &op in &self.tracked_paulis {
            parts.push(format!("TP{op}"));
        }
        if parts.is_empty() {
            write!(f, "(empty)")
        } else {
            write!(f, "{}", parts.join(" "))
        }
    }
}

// ============================================================================
// Parsed DEM
// ============================================================================

/// A parsed Detector Error Model.
#[derive(Debug, Clone)]
pub struct ParsedDem {
    /// All mechanisms in the DEM.
    pub mechanisms: Vec<ParsedMechanism>,
    /// Number of detectors (max ID + 1).
    pub num_detectors: u32,
    /// Total number of outputs in the DEM `L<n>` namespace.
    pub num_dem_outputs: u32,
    /// PECOS metadata for `L<n>` observables, indexed by `L<n>`.
    pub dem_outputs: Vec<Option<DemOutput>>,
    /// PECOS metadata for tracked Paulis in their own ID space.
    pub tracked_paulis: Vec<Option<DemOutput>>,
}

impl ParsedDem {
    /// Creates an empty `ParsedDem`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            mechanisms: Vec::new(),
            num_detectors: 0,
            num_dem_outputs: 0,
            dem_outputs: Vec::new(),
            tracked_paulis: Vec::new(),
        }
    }

    /// Parses a DEM from a string.
    ///
    /// Supports both standard and PECOS DEM formats.
    ///
    /// # Errors
    ///
    /// Returns `DemParseError` if the string cannot be parsed.
    pub fn parse(dem_str: &str) -> Result<Self, DemParseError> {
        dem_str.parse()
    }

    /// Total number of outputs in the DEM `L<n>` namespace.
    #[must_use]
    pub fn num_dem_outputs(&self) -> u32 {
        self.num_dem_outputs
    }

    /// Number of observables.
    #[must_use]
    pub fn num_observables(&self) -> u32 {
        self.num_dem_outputs
    }

    /// Number of tracked Paulis.
    #[must_use]
    pub fn num_tracked_paulis(&self) -> u32 {
        u32::try_from(self.tracked_paulis.len()).unwrap_or(u32::MAX)
    }

    fn record_metadata(ops: &mut Vec<Option<DemOutput>>, op: DemOutput) {
        let idx = op.id as usize;
        if ops.len() <= idx {
            ops.resize(idx + 1, None);
        }
        ops[idx] = Some(op);
    }

    /// Parses a single error line.
    fn parse_error_line(line: &str) -> Result<ParsedMechanism, DemParseError> {
        // Extract probability: error(0.01) ...
        let prob_end = line
            .find(')')
            .ok_or_else(|| DemParseError::InvalidFormat("Missing closing parenthesis".into()))?;

        let prob_str = &line[6..prob_end]; // Skip "error("
        let probability: f64 = prob_str
            .parse()
            .map_err(|_| DemParseError::InvalidProbability(prob_str.to_string()))?;

        // Get targets after probability
        let rest = &line[prob_end + 1..].trim();

        // Check for decomposition (XOR chains)
        if rest.contains('^') {
            let parts: Vec<&str> = rest.split('^').collect();
            let mut components = Vec::new();

            for part in parts {
                let part = part.trim();
                let comp = Self::parse_component(part)?;
                components.push(comp);
            }

            Ok(ParsedMechanism {
                probability,
                components,
            })
        } else {
            // Simple mechanism
            let comp = Self::parse_component(rest)?;
            Ok(ParsedMechanism {
                probability,
                components: vec![comp],
            })
        }
    }

    /// Parses a component (part between ^ separators).
    fn parse_component(s: &str) -> Result<MechanismComponent, DemParseError> {
        let mut detectors = Vec::new();
        let mut observables = Vec::new();
        let mut tracked_paulis = Vec::new();

        for token in s.split_whitespace() {
            if let Some(id_str) = token.strip_prefix('D') {
                let id: u32 = id_str
                    .parse()
                    .map_err(|_| DemParseError::InvalidDetectorId(token.to_string()))?;
                detectors.push(id);
            } else if let Some(id_str) = token.strip_prefix('L') {
                let id: u32 = id_str
                    .parse()
                    .map_err(|_| DemParseError::InvalidObservableId(token.to_string()))?;
                observables.push(id);
            } else if let Some(id_str) = token.strip_prefix("TP") {
                let id: u32 = id_str
                    .parse()
                    .map_err(|_| DemParseError::InvalidTrackedPauliId(token.to_string()))?;
                tracked_paulis.push(id);
            } else {
                return Err(DemParseError::InvalidTarget(token.to_string()));
            }
        }

        detectors.sort_unstable();
        observables.sort_unstable();
        tracked_paulis.sort_unstable();

        Ok(MechanismComponent {
            detectors,
            observables,
            tracked_paulis,
        })
    }

    /// Extracts detector ID from a detector declaration line.
    fn extract_detector_id(line: &str) -> Option<u32> {
        // Look for D followed by digits
        let d_pos = line.find('D')?;
        let rest = &line[d_pos + 1..];
        let end = rest
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(rest.len());
        rest[..end].parse().ok()
    }

    /// Extracts observable ID from an observable declaration line.
    fn extract_observable_id(line: &str) -> Option<u32> {
        let l_pos = line.find('L')?;
        let rest = &line[l_pos + 1..];
        let end = rest
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(rest.len());
        rest[..end].parse().ok()
    }

    /// Aggregates mechanisms by their effect, combining probabilities.
    ///
    /// Returns a map from effect key to aggregated probability.
    #[must_use]
    pub fn aggregate(&self) -> BTreeMap<EffectKey, f64> {
        let mut aggregated: BTreeMap<EffectKey, f64> = BTreeMap::new();

        for mech in &self.mechanisms {
            if mech.is_decomposed() {
                // For decomposed mechanisms, each component fires independently
                for comp in &mech.components {
                    let mut key = EffectKey::new(comp.detectors.clone(), comp.observables.clone());
                    key.tracked_paulis.clone_from(&comp.tracked_paulis);
                    key.tracked_paulis.sort_unstable();
                    aggregated
                        .entry(key)
                        .and_modify(|p| *p = combine_probabilities(*p, mech.probability))
                        .or_insert(mech.probability);
                }
            } else {
                // Simple mechanism
                let key = mech.effect_key();
                aggregated
                    .entry(key)
                    .and_modify(|p| *p = combine_probabilities(*p, mech.probability))
                    .or_insert(mech.probability);
            }
        }

        aggregated
    }

    /// Samples from this DEM.
    ///
    /// Returns (`detector_events`, `dem_output_flips`).
    ///
    /// # Semantics
    ///
    /// In DEM syntax, `error(p) A ^ B` means that when the error fires
    /// (with probability p), ALL components (A and B) flip together. The `^`
    /// separator is used for error tracking/decomposition but doesn't create
    /// independent firing - all components fire together as a single error.
    pub fn sample<R: Rng>(&self, rng: &mut R) -> (Vec<bool>, Vec<bool>) {
        let mut det_events = vec![false; self.num_detectors as usize];
        let mut obs_flips = vec![false; self.num_dem_outputs as usize];

        for mech in &self.mechanisms {
            // Single random check for the entire mechanism
            // All components fire together when the error occurs
            if rng.random::<f64>() < mech.probability {
                for comp in &mech.components {
                    for &d in &comp.detectors {
                        if (d as usize) < det_events.len() {
                            det_events[d as usize] ^= true;
                        }
                    }
                    for &o in &comp.observables {
                        if (o as usize) < obs_flips.len() {
                            obs_flips[o as usize] ^= true;
                        }
                    }
                }
            }
        }

        (det_events, obs_flips)
    }

    /// Samples multiple shots from this DEM.
    ///
    /// Returns (`detector_events_per_shot`, `dem_output_flips_per_shot`).
    pub fn sample_batch<R: Rng>(
        &self,
        num_shots: usize,
        rng: &mut R,
    ) -> (Vec<Vec<bool>>, Vec<Vec<bool>>) {
        let mut det_batches = Vec::with_capacity(num_shots);
        let mut obs_batches = Vec::with_capacity(num_shots);

        for _ in 0..num_shots {
            let (dets, obs) = self.sample(rng);
            det_batches.push(dets);
            obs_batches.push(obs);
        }

        (det_batches, obs_batches)
    }

    /// Convert to an optimized `DemSampler` for fast batch sampling.
    ///
    /// The `DemSampler` uses:
    /// - Geometric skip sampling for low error rates
    /// - Bit-packed arrays for efficient XOR operations
    /// - Parallel chunked processing for large DEMs
    ///
    /// This is significantly faster than `sample_batch` for large shot counts.
    ///
    /// # Note on decomposed errors
    ///
    /// In DEM syntax, `error(p) D0 ^ D1` means that when the error fires
    /// (with probability p), BOTH D0 and D1 flip together. The `^` separator is
    /// used for error tracking/decomposition but doesn't affect sampling - all
    /// components fire together.
    #[must_use]
    pub fn to_dem_sampler(&self) -> super::sampler::DemSampler {
        // Convert mechanisms to the format expected by SamplingEngine::from_mechanisms
        // Use combined_effect() to get the union of all detectors/observables
        // since all components fire together when the error occurs
        let mechanisms = self.mechanisms.iter().map(|mech| {
            let (dets, obs, _tracked_paulis) = mech.combined_effect();
            (mech.probability, dets, obs)
        });

        let engine = super::dem_sampler::SamplingEngine::from_mechanisms(
            mechanisms,
            self.num_detectors as usize,
            self.num_dem_outputs as usize,
        );
        let dem_outputs = self
            .dem_outputs
            .iter()
            .enumerate()
            .map(|(id, output)| {
                output.clone().or_else(|| {
                    #[allow(clippy::cast_possible_truncation)]
                    // parsed DEM output count fits in u32
                    {
                        Some(
                            DemOutput::new(id as u32)
                                .with_kind(crate::fault_tolerance::DemOutputKind::Observable),
                        )
                    }
                })
            })
            .collect();
        let tracked_paulis = self
            .tracked_paulis
            .iter()
            .enumerate()
            .map(|(id, output)| {
                output.clone().or_else(|| {
                    #[allow(clippy::cast_possible_truncation)]
                    // parsed tracked-Pauli count fits in u32
                    {
                        Some(
                            DemOutput::new(id as u32)
                                .with_kind(crate::fault_tolerance::DemOutputKind::TrackedPauli),
                        )
                    }
                })
            })
            .collect();

        super::sampler::DemSampler::from_engine(engine)
            .with_dem_output_metadata(dem_outputs, tracked_paulis)
    }

    /// Convert to a decomposed (graphlike) DEM string.
    ///
    /// Mechanisms with <= 2 detectors pass through unchanged. Already-decomposed
    /// mechanisms (with `^` notation) pass through unchanged.
    ///
    /// Hyperedges (3+ detectors, not already decomposed) cannot be decomposed
    /// without Pauli provenance and will cause an error. Use
    /// `coherent_dem_decomposed()` or `noise_characterization()` for proper
    /// X/Z-aware decomposition.
    ///
    /// # Errors
    ///
    /// Returns an error if any mechanism has 3+ detectors without decomposition.
    pub fn to_string_decomposed(&self) -> Result<String, String> {
        let mut lines = Vec::new();

        // Accumulate by target string to merge identical decomposed entries
        let mut by_targets: BTreeMap<String, f64> = BTreeMap::new();

        for mech in &self.mechanisms {
            if mech.probability <= 0.0 {
                continue;
            }

            if mech.is_decomposed() {
                // Already decomposed — pass through
                let targets = mech.format_targets();
                by_targets
                    .entry(targets)
                    .and_modify(|p| *p = combine_probabilities(*p, mech.probability))
                    .or_insert(mech.probability);
                continue;
            }

            // Single component
            let comp = &mech.components[0];

            if comp.detectors.len() <= 2 {
                // Graphlike — pass through
                let targets = mech.format_targets();
                by_targets
                    .entry(targets)
                    .and_modify(|p| *p = combine_probabilities(*p, mech.probability))
                    .or_insert(mech.probability);
            } else {
                // Hyperedge (3+ detectors) cannot be decomposed without Pauli
                // provenance (X/Z component info). Use `coherent_dem_decomposed()`
                // or `noise_characterization()` which track X/Z components from
                // the backward mechanism extraction.
                return Err(format!(
                    "Cannot decompose hyperedge with {} detectors ({:?}) without \
                     Pauli provenance. Use coherent_dem_decomposed() or \
                     noise_characterization() for proper X/Z decomposition.",
                    comp.detectors.len(),
                    comp.detectors,
                ));
            }
        }

        for (targets, prob) in &by_targets {
            if *prob > 0.0 {
                lines.push(format!("error({prob}) {targets}"));
            }
        }

        Ok(lines.join("\n"))
    }
}

impl Default for ParsedDem {
    fn default() -> Self {
        Self::new()
    }
}

impl FromStr for ParsedDem {
    type Err = DemParseError;

    fn from_str(dem_str: &str) -> Result<Self, Self::Err> {
        let mut mechanisms = Vec::new();
        let mut max_det: i32 = -1;
        let mut max_obs: i32 = -1;
        let mut max_tracked_pauli: i32 = -1;
        let mut dem_outputs: Vec<Option<DemOutput>> = Vec::new();
        let mut tracked_paulis: Vec<Option<DemOutput>> = Vec::new();

        for line in dem_str.lines() {
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse error lines
            if line.starts_with("error(") {
                let mech = Self::parse_error_line(line)?;

                // Update max IDs
                for comp in &mech.components {
                    for &d in &comp.detectors {
                        #[allow(clippy::cast_possible_wrap)] // detector ID fits in i32
                        {
                            max_det = max_det.max(d as i32);
                        }
                    }
                    for &o in &comp.observables {
                        #[allow(clippy::cast_possible_wrap)] // observable ID fits in i32
                        {
                            max_obs = max_obs.max(o as i32);
                        }
                    }
                    for &op in &comp.tracked_paulis {
                        #[allow(clippy::cast_possible_wrap)] // tracked-Pauli ID fits in i32
                        {
                            max_tracked_pauli = max_tracked_pauli.max(op as i32);
                        }
                    }
                }

                mechanisms.push(mech);
            }
            // Parse detector declarations
            else if line.starts_with("detector") {
                if let Some(id) = Self::extract_detector_id(line) {
                    #[allow(clippy::cast_possible_wrap)] // detector ID fits in i32
                    {
                        max_det = max_det.max(id as i32);
                    }
                }
            }
            // Parse observable declarations
            else if line.starts_with("logical_observable")
                && let Some(id) = Self::extract_observable_id(line)
            {
                #[allow(clippy::cast_possible_wrap)] // observable ID fits in i32
                {
                    max_obs = max_obs.max(id as i32);
                }
                Self::record_metadata(
                    &mut dem_outputs,
                    DemOutput::new(id).with_kind(crate::fault_tolerance::DemOutputKind::Observable),
                );
            }
            // Parse PECOS DEM-superset metadata declarations.
            else if line.starts_with("pecos_observable")
                || line.starts_with("pecos_tracked_pauli")
            {
                let op = parse_pecos_dem_metadata_line(line)
                    .map_err(|err| DemParseError::InvalidPecosMetadata(err.to_string()))?;
                if op.is_tracked_pauli() {
                    #[allow(clippy::cast_possible_wrap)] // tracked-Pauli ID fits in i32
                    {
                        max_tracked_pauli = max_tracked_pauli.max(op.id as i32);
                    }
                    Self::record_metadata(&mut tracked_paulis, op);
                } else {
                    #[allow(clippy::cast_possible_wrap)] // observable ID fits in i32
                    {
                        max_obs = max_obs.max(op.id as i32);
                    }
                    Self::record_metadata(&mut dem_outputs, op);
                }
            }
            // PECOS extensions are explicit; ordinary DEM lines remain valid,
            // but unknown PECOS extension statements should not be silently
            // accepted as historical aliases.
            else if line.starts_with("pecos_") {
                return Err(DemParseError::InvalidPecosMetadata(format!(
                    "unsupported PECOS DEM extension line: {line}"
                )));
            }
        }

        if max_obs >= 0 {
            #[allow(clippy::cast_sign_loss)] // guarded by >= 0 check
            {
                dem_outputs.resize(max_obs as usize + 1, None);
            }
        }
        if max_tracked_pauli >= 0 {
            #[allow(clippy::cast_sign_loss)] // guarded by >= 0 check
            {
                tracked_paulis.resize(max_tracked_pauli as usize + 1, None);
            }
        }

        Ok(Self {
            mechanisms,
            num_detectors: if max_det >= 0 {
                #[allow(clippy::cast_sign_loss)] // guarded by >= 0 check
                {
                    max_det as u32 + 1
                }
            } else {
                0
            },
            num_dem_outputs: if max_obs >= 0 {
                #[allow(clippy::cast_sign_loss)] // guarded by >= 0 check
                {
                    max_obs as u32 + 1
                }
            } else {
                0
            },
            dem_outputs,
            tracked_paulis,
        })
    }
}

// ============================================================================
// Parse Errors
// ============================================================================

/// Errors that can occur when parsing a DEM.
#[derive(Debug, Clone)]
pub enum DemParseError {
    /// Invalid DEM format.
    InvalidFormat(String),
    /// Invalid probability value.
    InvalidProbability(String),
    /// Invalid detector ID.
    InvalidDetectorId(String),
    /// Invalid observable ID.
    InvalidObservableId(String),
    /// Invalid tracked-Pauli ID.
    InvalidTrackedPauliId(String),
    /// Invalid target token in an error line.
    InvalidTarget(String),
    /// Invalid PECOS DEM-superset metadata.
    InvalidPecosMetadata(String),
}

impl std::fmt::Display for DemParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidFormat(msg) => write!(f, "Invalid DEM format: {msg}"),
            Self::InvalidProbability(s) => write!(f, "Invalid probability: {s}"),
            Self::InvalidDetectorId(s) => write!(f, "Invalid detector ID: {s}"),
            Self::InvalidObservableId(s) => write!(f, "Invalid observable ID: {s}"),
            Self::InvalidTrackedPauliId(s) => write!(f, "Invalid tracked Pauli ID: {s}"),
            Self::InvalidTarget(s) => write!(f, "Invalid DEM error target: {s}"),
            Self::InvalidPecosMetadata(s) => write!(f, "Invalid PECOS DEM metadata: {s}"),
        }
    }
}

impl std::error::Error for DemParseError {}

// ============================================================================
// Equivalence Result
// ============================================================================

/// Result of DEM equivalence comparison.
#[derive(Debug, Clone)]
pub struct EquivalenceResult {
    /// Whether the DEMs are equivalent within tolerance.
    pub equivalent: bool,
    /// Maximum absolute difference in rates/probabilities.
    pub max_rate_difference: f64,
    /// Maximum relative difference in rates/probabilities.
    pub max_relative_difference: f64,
    /// Correlation of detector rates (statistical comparison).
    pub correlation: f64,
    /// Per-detector rate differences (statistical comparison).
    pub detector_rate_differences: Vec<f64>,
    /// Per-observable rate differences (statistical comparison).
    pub observable_rate_differences: Vec<f64>,
    /// Additional comparison details.
    pub details: ComparisonDetails,
}

/// Additional details from DEM comparison.
#[derive(Debug, Clone, Default)]
pub struct ComparisonDetails {
    /// Number of mechanisms in first DEM.
    pub dem1_mechanism_count: usize,
    /// Number of mechanisms in second DEM.
    pub dem2_mechanism_count: usize,
    /// Mechanisms only in first DEM.
    pub only_in_dem1: Vec<String>,
    /// Mechanisms only in second DEM.
    pub only_in_dem2: Vec<String>,
    /// Probability mismatches for common mechanisms.
    pub prob_mismatches: Vec<ProbabilityMismatch>,
}

/// A probability mismatch between two DEMs.
#[derive(Debug, Clone)]
pub struct ProbabilityMismatch {
    /// Target description (e.g., "D0 D1").
    pub target: String,
    /// Probability in first DEM.
    pub dem1_prob: f64,
    /// Probability in second DEM.
    pub dem2_prob: f64,
    /// Absolute difference.
    pub difference: f64,
}

// ============================================================================
// Comparison Functions
// ============================================================================

/// Compares two DEMs for exact mechanism match.
///
/// This comparison aggregates mechanisms by effect and compares probabilities.
/// Appropriate for non-decomposed DEMs or when exact match is required.
///
/// # Arguments
///
/// * `dem1` - First DEM.
/// * `dem2` - Second DEM.
/// * `prob_tolerance` - Relative tolerance for probability comparison.
///
/// # Returns
///
/// `EquivalenceResult` with comparison statistics.
pub fn compare_dems_exact(
    dem1: &ParsedDem,
    dem2: &ParsedDem,
    prob_tolerance: f64,
) -> EquivalenceResult {
    let agg1 = dem1.aggregate();
    let agg2 = dem2.aggregate();

    let keys1: BTreeSet<_> = agg1.keys().cloned().collect();
    let keys2: BTreeSet<_> = agg2.keys().cloned().collect();

    let only_in_1: Vec<_> = keys1.difference(&keys2).cloned().collect();
    let only_in_2: Vec<_> = keys2.difference(&keys1).cloned().collect();
    let common: Vec<_> = keys1.intersection(&keys2).cloned().collect();

    // Compute probability differences for common mechanisms
    let mut prob_diffs = Vec::new();
    let mut rel_diffs = Vec::new();
    let mut mismatches = Vec::new();

    for key in &common {
        let p1 = agg1.get(key).copied().unwrap_or(0.0);
        let p2 = agg2.get(key).copied().unwrap_or(0.0);
        let diff = (p1 - p2).abs();
        let max_p = p1.max(p2).max(1e-10);
        let rel_diff = diff / max_p;

        prob_diffs.push(diff);
        rel_diffs.push(rel_diff);

        if rel_diff > prob_tolerance {
            mismatches.push(ProbabilityMismatch {
                target: key.to_string(),
                dem1_prob: p1,
                dem2_prob: p2,
                difference: diff,
            });
        }
    }

    let max_prob_diff = prob_diffs.iter().copied().fold(0.0_f64, f64::max);
    let max_rel_diff = rel_diffs.iter().copied().fold(0.0_f64, f64::max);

    // Equivalence requires same mechanism sets and all probabilities match
    let equivalent = only_in_1.is_empty() && only_in_2.is_empty() && max_rel_diff <= prob_tolerance;

    EquivalenceResult {
        equivalent,
        max_rate_difference: max_prob_diff,
        max_relative_difference: max_rel_diff,
        correlation: if equivalent { 1.0 } else { 0.0 },
        detector_rate_differences: vec![],
        observable_rate_differences: vec![],
        details: ComparisonDetails {
            dem1_mechanism_count: agg1.len(),
            dem2_mechanism_count: agg2.len(),
            only_in_dem1: only_in_1.iter().map(EffectKey::to_string).collect(),
            only_in_dem2: only_in_2.iter().map(EffectKey::to_string).collect(),
            prob_mismatches: mismatches,
        },
    }
}

/// Compares two DEMs statistically by sampling.
///
/// This is the most robust comparison method as it accounts for all
/// decomposition strategies and probability combinations. It compares
/// the joint distribution of syndrome patterns, not just marginal rates,
/// which correctly distinguishes between correlated and independent errors.
///
/// # Arguments
///
/// * `dem1` - First DEM.
/// * `dem2` - Second DEM.
/// * `num_shots` - Number of shots for sampling.
/// * `seed` - Random seed.
/// * `tolerance` - Maximum relative difference to consider equivalent.
///
/// # Returns
///
/// `EquivalenceResult` with comparison statistics.
#[allow(clippy::cast_precision_loss)] // statistical calculations use count as f64
pub fn compare_dems_statistical(
    dem1: &ParsedDem,
    dem2: &ParsedDem,
    num_shots: usize,
    seed: u64,
    tolerance: f64,
) -> EquivalenceResult {
    let mut rng1 = PecosRng::seed_from_u64(seed);
    let mut rng2 = PecosRng::seed_from_u64(seed + 1); // Different seed for independence

    // Sample from both DEMs
    let (det1, obs1) = dem1.sample_batch(num_shots, &mut rng1);
    let (det2, obs2) = dem2.sample_batch(num_shots, &mut rng2);

    // Compute detector firing rates (marginals)
    let num_det = dem1.num_detectors.max(dem2.num_detectors) as usize;
    let num_obs = dem1.num_dem_outputs.max(dem2.num_dem_outputs) as usize;

    let mut det_rates1 = vec![0.0; num_det];
    let mut det_rates2 = vec![0.0; num_det];
    let mut obs_rates1 = vec![0.0; num_obs];
    let mut obs_rates2 = vec![0.0; num_obs];

    for shot in &det1 {
        for (i, &fired) in shot.iter().enumerate() {
            if fired && i < num_det {
                det_rates1[i] += 1.0;
            }
        }
    }
    for shot in &det2 {
        for (i, &fired) in shot.iter().enumerate() {
            if fired && i < num_det {
                det_rates2[i] += 1.0;
            }
        }
    }
    for shot in &obs1 {
        for (i, &flipped) in shot.iter().enumerate() {
            if flipped && i < num_obs {
                obs_rates1[i] += 1.0;
            }
        }
    }
    for shot in &obs2 {
        for (i, &flipped) in shot.iter().enumerate() {
            if flipped && i < num_obs {
                obs_rates2[i] += 1.0;
            }
        }
    }

    // Normalize to rates
    let n = num_shots as f64;
    for r in &mut det_rates1 {
        *r /= n;
    }
    for r in &mut det_rates2 {
        *r /= n;
    }
    for r in &mut obs_rates1 {
        *r /= n;
    }
    for r in &mut obs_rates2 {
        *r /= n;
    }

    // Compute marginal rate differences
    let det_diffs: Vec<f64> = det_rates1
        .iter()
        .zip(&det_rates2)
        .map(|(a, b)| (a - b).abs())
        .collect();
    let obs_diffs: Vec<f64> = obs_rates1
        .iter()
        .zip(&obs_rates2)
        .map(|(a, b)| (a - b).abs())
        .collect();

    // Compute syndrome pattern distributions (joint distribution)
    // This captures correlations between detectors that marginals miss
    let mut pattern_counts1: BTreeMap<Vec<bool>, usize> = BTreeMap::new();
    let mut pattern_counts2: BTreeMap<Vec<bool>, usize> = BTreeMap::new();

    for shot in &det1 {
        // Pad to num_det length
        let mut pattern = shot.clone();
        pattern.resize(num_det, false);
        *pattern_counts1.entry(pattern).or_insert(0) += 1;
    }
    for shot in &det2 {
        let mut pattern = shot.clone();
        pattern.resize(num_det, false);
        *pattern_counts2.entry(pattern).or_insert(0) += 1;
    }

    // Collect all patterns seen in either DEM
    let all_patterns: BTreeSet<_> = pattern_counts1
        .keys()
        .chain(pattern_counts2.keys())
        .cloned()
        .collect();

    // Compare pattern distributions
    let mut max_pattern_diff = 0.0_f64;
    let mut max_pattern_rel_diff = 0.0_f64;

    for pattern in &all_patterns {
        let count1 = *pattern_counts1.get(pattern).unwrap_or(&0) as f64;
        let count2 = *pattern_counts2.get(pattern).unwrap_or(&0) as f64;
        let rate1 = count1 / n;
        let rate2 = count2 / n;

        let diff = (rate1 - rate2).abs();
        max_pattern_diff = max_pattern_diff.max(diff);

        let max_rate = rate1.max(rate2);
        if max_rate > 1e-6 {
            let rel_diff = diff / max_rate;
            max_pattern_rel_diff = max_pattern_rel_diff.max(rel_diff);
        }
    }

    // Use pattern distribution for equivalence check
    // Account for statistical noise: standard error is ~sqrt(p*(1-p)/n)
    // For tolerance comparison, use absolute difference with statistical margin
    let statistical_margin = 3.0 / (num_shots as f64).sqrt(); // 3-sigma

    let max_abs_diff = det_diffs
        .iter()
        .copied()
        .fold(0.0_f64, f64::max)
        .max(obs_diffs.iter().copied().fold(0.0_f64, f64::max));

    // Compute correlation of detector rates (for reporting)
    let correlation = if num_det > 1 {
        compute_correlation(&det_rates1, &det_rates2)
    } else if !det_rates1.is_empty() {
        // For single detector, use pattern match quality
        if max_pattern_diff < statistical_margin {
            1.0
        } else {
            0.0
        }
    } else {
        1.0
    };

    // Equivalence requires:
    // 1. Pattern distribution differences within tolerance + statistical margin
    // 2. Marginal rate differences within tolerance
    let pattern_equivalent = max_pattern_diff < tolerance + statistical_margin;
    let marginal_equivalent = max_abs_diff < tolerance + statistical_margin;
    let equivalent = pattern_equivalent && marginal_equivalent;

    EquivalenceResult {
        equivalent,
        max_rate_difference: max_pattern_diff.max(max_abs_diff),
        max_relative_difference: max_pattern_rel_diff,
        correlation,
        detector_rate_differences: det_diffs,
        observable_rate_differences: obs_diffs,
        details: ComparisonDetails {
            dem1_mechanism_count: dem1.mechanisms.len(),
            dem2_mechanism_count: dem2.mechanisms.len(),
            only_in_dem1: vec![],
            only_in_dem2: vec![],
            prob_mismatches: vec![],
        },
    }
}

/// Computes Pearson correlation coefficient.
#[allow(clippy::cast_precision_loss)] // length as f64 for mean calculation
fn compute_correlation(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }

    let n = a.len() as f64;
    let mean_a: f64 = a.iter().sum::<f64>() / n;
    let mean_b: f64 = b.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;

    for (ai, bi) in a.iter().zip(b.iter()) {
        let da = ai - mean_a;
        let db = bi - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let std_a = var_a.sqrt();
    let std_b = var_b.sqrt();

    if std_a < 1e-10 || std_b < 1e-10 {
        // Near-zero variance - check if values are equal
        if a.iter()
            .zip(b.iter())
            .all(|(ai, bi)| (ai - bi).abs() < 0.01)
        {
            1.0
        } else {
            0.0
        }
    } else {
        cov / (std_a * std_b)
    }
}

/// Convenience function to verify DEM equivalence.
///
/// Returns true if DEMs are equivalent within tolerance.
///
/// # Errors
/// Returns `DemParseError` if either DEM string cannot be parsed.
pub fn verify_dem_equivalence(
    dem_str1: &str,
    dem_str2: &str,
    method: &ComparisonMethod,
) -> Result<bool, DemParseError> {
    let dem1 = ParsedDem::from_str(dem_str1)?;
    let dem2 = ParsedDem::from_str(dem_str2)?;

    let result = match *method {
        ComparisonMethod::Exact { prob_tolerance } => {
            compare_dems_exact(&dem1, &dem2, prob_tolerance)
        }
        ComparisonMethod::Statistical {
            num_shots,
            seed,
            tolerance,
        } => compare_dems_statistical(&dem1, &dem2, num_shots, seed, tolerance),
    };

    Ok(result.equivalent)
}

/// Method for DEM comparison.
#[derive(Debug, Clone)]
pub enum ComparisonMethod {
    /// Exact mechanism comparison.
    Exact {
        /// Relative tolerance for probability comparison.
        prob_tolerance: f64,
    },
    /// Statistical comparison via sampling.
    Statistical {
        /// Number of shots for sampling.
        num_shots: usize,
        /// Random seed.
        seed: u64,
        /// Tolerance for rate differences.
        tolerance: f64,
    },
}

impl Default for ComparisonMethod {
    fn default() -> Self {
        Self::Exact {
            prob_tolerance: 1e-6,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_mechanism() {
        let dem_str = "error(0.01) D0 D1";
        let dem = ParsedDem::from_str(dem_str).unwrap();

        assert_eq!(dem.mechanisms.len(), 1);
        assert!(!dem.mechanisms[0].is_decomposed());
        assert!((dem.mechanisms[0].probability - 0.01).abs() < f64::EPSILON);
        assert_eq!(dem.mechanisms[0].components[0].detectors, vec![0, 1]);
    }

    #[test]
    fn test_parse_decomposed_mechanism() {
        let dem_str = "error(0.01) D0 ^ D1 D2";
        let dem = ParsedDem::from_str(dem_str).unwrap();

        assert_eq!(dem.mechanisms.len(), 1);
        assert!(dem.mechanisms[0].is_decomposed());
        assert_eq!(dem.mechanisms[0].components.len(), 2);
        assert_eq!(dem.mechanisms[0].components[0].detectors, vec![0]);
        assert_eq!(dem.mechanisms[0].components[1].detectors, vec![1, 2]);
    }

    #[test]
    fn test_parse_with_observable() {
        let dem_str = "error(0.02) D0 L0";
        let dem = ParsedDem::from_str(dem_str).unwrap();

        assert_eq!(dem.mechanisms.len(), 1);
        assert_eq!(dem.mechanisms[0].components[0].detectors, vec![0]);
        assert_eq!(dem.mechanisms[0].components[0].observables, vec![0]);
    }

    #[test]
    fn test_parse_accepts_pecos_dem_superset_metadata() {
        let dem_str = r#"
            error(0.02) D0 TP0
            pecos_tracked_pauli {"id":0,"kind":"tracked_pauli","label":"track","pauli":"+X0 Z2","records":[]}
        "#;
        let dem = ParsedDem::from_str(dem_str).unwrap();

        assert_eq!(dem.mechanisms.len(), 1);
        assert_eq!(dem.num_detectors, 1);
        assert_eq!(dem.num_dem_outputs(), 0);
        assert_eq!(dem.num_observables(), 0);
        assert_eq!(dem.num_tracked_paulis(), 1);
        assert_eq!(dem.mechanisms[0].components[0].tracked_paulis, vec![0]);
        assert_eq!(dem.mechanisms[0].format_targets(), "D0 TP0");
        assert_eq!(dem.mechanisms[0].effect_key().to_string(), "D0 TP0");
        let op = dem.tracked_paulis[0].as_ref().unwrap();
        assert_eq!(op.label.as_deref(), Some("track"));
        assert_eq!(
            op.kind,
            Some(crate::fault_tolerance::DemOutputKind::TrackedPauli)
        );
        assert_eq!(op.pauli.as_ref().unwrap().to_sparse_str(), "+X0 Z2");
    }

    #[test]
    fn test_parse_rejects_malformed_pecos_dem_superset_metadata() {
        let err = ParsedDem::from_str("pecos_tracked_pauli not-json").unwrap_err();
        assert!(matches!(err, DemParseError::InvalidPecosMetadata(_)));
    }

    #[test]
    fn test_parse_accepts_tracked_pauli_targets_without_metadata() {
        let dem = ParsedDem::from_str("error(0.125) D1 L0 TP2").unwrap();

        assert_eq!(dem.num_detectors, 2);
        assert_eq!(dem.num_dem_outputs(), 1);
        assert_eq!(dem.num_tracked_paulis(), 3);
        assert_eq!(dem.mechanisms[0].components[0].detectors, vec![1]);
        assert_eq!(dem.mechanisms[0].components[0].observables, vec![0]);
        assert_eq!(dem.mechanisms[0].components[0].tracked_paulis, vec![2]);
        assert_eq!(dem.mechanisms[0].effect_key().to_string(), "D1 L0 TP2");
    }

    #[test]
    fn test_decomposed_tracked_pauli_targets_xor_by_parity() {
        let cancels = ParsedDem::from_str("error(0.5) TP0 ^ TP0").unwrap();
        let (dets, obs, tracked_paulis) = cancels.mechanisms[0].combined_effect();
        assert!(dets.is_empty());
        assert!(obs.is_empty());
        assert!(tracked_paulis.is_empty());
        assert_eq!(cancels.mechanisms[0].effect_key().to_string(), "(empty)");

        let leaves_detector = ParsedDem::from_str("error(0.5) D0 TP0 ^ TP0").unwrap();
        let (dets, obs, tracked_paulis) = leaves_detector.mechanisms[0].combined_effect();
        assert_eq!(dets, vec![0]);
        assert!(obs.is_empty());
        assert!(tracked_paulis.is_empty());
        assert_eq!(leaves_detector.mechanisms[0].effect_key().to_string(), "D0");
    }

    #[test]
    fn test_duplicate_tracked_pauli_targets_cancel_by_parity() {
        let dem = ParsedDem::from_str("error(0.1) TP0 TP0").unwrap();
        assert_eq!(dem.mechanisms[0].components[0].tracked_paulis, vec![0, 0]);

        let (dets, obs, tracked_paulis) = dem.mechanisms[0].combined_effect();
        assert!(dets.is_empty());
        assert!(obs.is_empty());
        assert!(tracked_paulis.is_empty());
        assert_eq!(dem.mechanisms[0].effect_key().to_string(), "(empty)");
    }

    #[test]
    fn test_parse_rejects_unknown_error_targets() {
        let err = ParsedDem::from_str("error(0.125) D1 T0").unwrap_err();
        assert!(matches!(err, DemParseError::InvalidTarget(_)));
        assert!(err.to_string().contains("Invalid DEM error target: T0"));
    }

    #[test]
    fn test_parse_rejects_malformed_tracked_pauli_targets() {
        for target in ["TP", "TPx", "TP-1"] {
            let err = ParsedDem::from_str(&format!("error(0.125) {target}")).unwrap_err();
            assert!(
                matches!(err, DemParseError::InvalidTrackedPauliId(_)),
                "{target} should be rejected as a malformed tracked-Pauli target"
            );
        }
    }

    #[test]
    fn test_parsed_dem_sampler_projects_tracked_pauli_effects_but_keeps_metadata() {
        let dem = ParsedDem::from_str("error(1.0) TP0").unwrap();
        let sampler = dem.to_dem_sampler();
        let mut rng = PecosRng::seed_from_u64(123);

        assert_eq!(dem.num_tracked_paulis(), 1);
        assert_eq!(sampler.num_detectors(), 0);
        assert_eq!(sampler.num_dem_outputs(), 0);
        assert_eq!(sampler.num_tracked_paulis(), 1);
        let (detectors, dem_outputs) = sampler.sample(&mut rng);
        assert!(detectors.is_empty());
        assert!(dem_outputs.is_empty());
        let err = sampler.sample_tracked_pauli_flips(&mut rng).unwrap_err();
        assert_eq!(err.backend(), "DemSampler");
        assert_eq!(err.num_tracked_paulis(), 1);
    }

    #[test]
    fn test_parsed_dem_samplers_project_different_tracked_paulis_the_same() {
        let dem1 = ParsedDem::from_str("error(1.0) D0 TP0").unwrap();
        let dem2 = ParsedDem::from_str("error(1.0) D0 TP1").unwrap();
        let sampler1 = dem1.to_dem_sampler();
        let sampler2 = dem2.to_dem_sampler();
        let mut rng1 = PecosRng::seed_from_u64(11);
        let mut rng2 = PecosRng::seed_from_u64(11);

        assert_eq!(sampler1.num_detectors(), 1);
        assert_eq!(sampler2.num_detectors(), 1);
        assert_eq!(sampler1.num_dem_outputs(), 0);
        assert_eq!(sampler2.num_dem_outputs(), 0);
        assert_eq!(sampler1.num_tracked_paulis(), 1);
        assert_eq!(sampler2.num_tracked_paulis(), 2);
        assert_eq!(sampler1.sample(&mut rng1), sampler2.sample(&mut rng2));
    }

    #[test]
    fn test_parse_rejects_unknown_pecos_dem_extension() {
        let err = ParsedDem::from_str("pecos_old_extension {}").unwrap_err();
        assert!(matches!(err, DemParseError::InvalidPecosMetadata(_)));
        assert!(err.to_string().contains("unsupported PECOS DEM extension"));
    }

    #[test]
    fn test_aggregate() {
        let dem_str = r"
error(0.1) D0
error(0.2) D0
";
        let dem = ParsedDem::from_str(dem_str).unwrap();
        let agg = dem.aggregate();

        // Combined: 0.1*(1-0.2) + 0.2*(1-0.1) = 0.08 + 0.18 = 0.26
        let key = EffectKey::new(vec![0], vec![]);
        assert!((agg[&key] - 0.26).abs() < 1e-10);
    }

    #[test]
    fn test_compare_identical_dems() {
        let dem_str = r"
error(0.01) D0 D1
error(0.02) D1 D2
";
        let dem = ParsedDem::from_str(dem_str).unwrap();
        let result = compare_dems_exact(&dem, &dem, 1e-6);

        assert!(result.equivalent);
        assert!(result.max_rate_difference < 1e-10);
    }

    #[test]
    fn test_compare_different_probabilities() {
        let dem1 = ParsedDem::from_str("error(0.01) D0").unwrap();
        let dem2 = ParsedDem::from_str("error(0.02) D0").unwrap();

        let result = compare_dems_exact(&dem1, &dem2, 0.01);
        assert!(!result.equivalent);
        assert!((result.max_rate_difference - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_compare_different_mechanisms() {
        let dem1 = ParsedDem::from_str("error(0.01) D0 D1").unwrap();
        let dem2 = ParsedDem::from_str("error(0.01) D0 D2").unwrap();

        let result = compare_dems_exact(&dem1, &dem2, 1e-6);
        assert!(!result.equivalent);
        assert!(!result.details.only_in_dem1.is_empty());
        assert!(!result.details.only_in_dem2.is_empty());
    }

    #[test]
    fn test_compare_exact_distinguishes_tracked_pauli_targets() {
        let dem1 = ParsedDem::from_str("error(0.01) D0 TP0").unwrap();
        let dem2 = ParsedDem::from_str("error(0.01) D0 TP1").unwrap();

        let result = compare_dems_exact(&dem1, &dem2, 1e-6);
        assert!(!result.equivalent);
        assert_eq!(
            result
                .details
                .only_in_dem1
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
            ["D0 TP0"]
        );
        assert_eq!(
            result
                .details
                .only_in_dem2
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
            ["D0 TP1"]
        );

        let decomposed1 = ParsedDem::from_str("error(0.01) D0 TP0 ^ D1").unwrap();
        let decomposed2 = ParsedDem::from_str("error(0.01) D0 TP1 ^ D1").unwrap();
        let result = compare_dems_exact(&decomposed1, &decomposed2, 1e-6);
        assert!(
            !result.equivalent,
            "exact PECOS DEM comparison must include tracked targets on decomposed components"
        );
    }

    #[test]
    fn test_statistical_comparison() {
        let dem_str = "error(0.5) D0";
        let dem = ParsedDem::from_str(dem_str).unwrap();

        let result = compare_dems_statistical(&dem, &dem, 10_000, 42, 0.05);
        // Same DEM should be equivalent
        assert!(result.equivalent);
    }

    #[test]
    fn test_decomposed_equivalent_to_simple() {
        // In DEM syntax, these should be equivalent for sampling:
        // - error(0.1) D0 D1: D0 and D1 flip together with p=0.1
        // - error(0.1) D0 ^ D1: D0 and D1 flip together with p=0.1 (^ is for decomposition tracking)
        let dem1 = ParsedDem::from_str("error(0.1) D0 D1").unwrap();
        let dem2 = ParsedDem::from_str("error(0.1) D0 ^ D1").unwrap();

        let result = compare_dems_statistical(&dem1, &dem2, 50_000, 42, 0.05);
        // These SHOULD be equivalent (both flip D0 and D1 together)
        assert!(result.equivalent);
    }

    #[test]
    fn test_truly_independent_not_equivalent() {
        // D0 and D1 flip together (correlated)
        let dem1 = ParsedDem::from_str("error(0.1) D0 D1").unwrap();
        // D0 and D1 flip independently (two separate errors)
        let dem2 = ParsedDem::from_str("error(0.1) D0\nerror(0.1) D1").unwrap();

        let result = compare_dems_statistical(&dem1, &dem2, 50_000, 42, 0.05);
        // These should NOT be equivalent
        // dem1: P(D0 fires) = P(D1 fires) = P(both fire) = 0.1
        // dem2: P(D0 fires) = P(D1 fires) = 0.1, P(both fire) = 0.01
        assert!(!result.equivalent);
    }

    #[test]
    fn test_xor_cancellation() {
        // error(p) D0 ^ D0 should result in no net effect (XOR cancellation)
        let dem = ParsedDem::from_str("error(0.5) D0 ^ D0").unwrap();

        // The combined effect should be empty
        let (dets, obs, tracked_paulis) = dem.mechanisms[0].combined_effect();
        assert!(dets.is_empty());
        assert!(obs.is_empty());
        assert!(tracked_paulis.is_empty());

        // Sample and verify D0 never fires
        let mut rng = PecosRng::seed_from_u64(42);
        let (det_events, _) = dem.sample_batch(10_000, &mut rng);
        let d0_fires: usize = det_events.iter().filter(|e| !e.is_empty() && e[0]).count();
        assert_eq!(d0_fires, 0, "D0 should never fire due to XOR cancellation");
    }
}
