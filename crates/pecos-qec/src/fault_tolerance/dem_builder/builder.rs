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

//! DEM (Detector Error Model) builder implementation.
//!
//! This module provides the main builder for constructing DEMs from fault
//! influence maps and detector/observable metadata.

use super::types::{
    DetectorDef, DetectorErrorModel, ErrorMechanism, LogicalObservable, NoiseConfig,
};
use crate::fault_tolerance::propagator::{DagFaultInfluenceMap, Pauli};
use pecos_core::gate_type::GateType;
use smallvec::SmallVec;
use std::collections::BTreeMap;

// ============================================================================
// JSON Parsing Types
// ============================================================================

/// Parsed detector from JSON metadata.
#[derive(Debug, Clone)]
struct ParsedDetector {
    id: u32,
    coords: Option<[f64; 3]>,
    records: Vec<i32>,
}

/// Parsed observable from JSON metadata.
#[derive(Debug, Clone)]
struct ParsedObservable {
    id: u32,
    records: Vec<i32>,
}

// ============================================================================
// DEM Builder
// ============================================================================

/// Builder for Detector Error Models (DEMs).
///
/// Constructs a DEM from a fault influence map and detector/observable metadata.
/// Uses the per-qubit fault model for accurate depolarizing noise analysis.
///
/// # Example
///
/// ```
/// use pecos_qec::fault_tolerance::DagFaultAnalyzer;
/// use pecos_qec::fault_tolerance::dem_builder::DemBuilder;
/// use pecos_quantum::DagCircuit;
///
/// let mut dag = DagCircuit::new();
/// dag.pz(&[2]);
/// dag.cx(&[(0, 2)]);
/// dag.cx(&[(1, 2)]);
/// dag.mz(&[2]);
///
/// let analyzer = DagFaultAnalyzer::new(&dag);
/// let influence_map = analyzer.build_influence_map();
/// let detectors_json = r#"[{"id": 0, "records": [-1]}]"#;
/// let observables_json = "[]";
///
/// let dem = DemBuilder::new(&influence_map)
///     .with_noise(0.01, 0.01, 0.01, 0.01)
///     .with_detectors_json(detectors_json).unwrap()
///     .with_observables_json(observables_json).unwrap()
///     .build();
///
/// // Non-decomposed output (matches Stim's decompose_errors=False)
/// println!("{}", dem.to_string());
///
/// // Decomposed output (matches Stim's decompose_errors=True)
/// println!("{}", dem.to_string_decomposed());
/// ```
pub struct DemBuilder<'a> {
    /// Reference to the fault influence map.
    influence_map: &'a DagFaultInfluenceMap,
    /// Noise configuration.
    noise: NoiseConfig,
    /// Parsed detector definitions.
    detectors: Vec<ParsedDetector>,
    /// Parsed observable definitions.
    observables: Vec<ParsedObservable>,
    /// Total number of measurements in the circuit.
    num_measurements: usize,
    /// Optional measurement order: maps `TickCircuit` measurement index -> qubit.
    /// This allows proper mapping between record offsets and influence map indices.
    measurement_order: Option<Vec<usize>>,
}

impl<'a> DemBuilder<'a> {
    /// Creates a new DEM builder from a fault influence map.
    #[must_use]
    pub fn new(influence_map: &'a DagFaultInfluenceMap) -> Self {
        Self {
            influence_map,
            noise: NoiseConfig::default(),
            detectors: Vec::new(),
            observables: Vec::new(),
            num_measurements: influence_map.measurements.len(),
            measurement_order: None,
        }
    }

    /// Sets the noise configuration.
    #[must_use]
    pub fn with_noise(mut self, p1: f64, p2: f64, p_meas: f64, p_init: f64) -> Self {
        self.noise = NoiseConfig::new(p1, p2, p_meas, p_init);
        self
    }

    /// Sets the number of measurements (used for record offset calculation).
    #[must_use]
    pub fn with_num_measurements(mut self, num: usize) -> Self {
        self.num_measurements = num;
        self
    }

    /// Sets the measurement order from the original circuit.
    ///
    /// The measurement order is a list of qubits in the order they were measured
    /// in the original circuit (e.g., `TickCircuit`). This allows proper mapping
    /// between record offsets (which use `TickCircuit` order) and influence map
    /// indices (which may use a different order based on DAG topology).
    ///
    /// # Arguments
    ///
    /// * `order` - List of qubit indices in measurement execution order.
    ///   `order[i]` is the qubit measured at `TickCircuit` measurement index `i`.
    #[must_use]
    pub fn with_measurement_order(mut self, order: Vec<usize>) -> Self {
        self.measurement_order = Some(order);
        self
    }

    /// Parses and sets detector definitions from JSON.
    ///
    /// Expected format:
    /// ```json
    /// [
    ///   {"id": 0, "coords": [0.0, 0.0, 0.0], "records": [-1, -5]},
    ///   {"id": 1, "coords": [1.0, 0.0, 0.0], "records": [-2]}
    /// ]
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the JSON is malformed.
    pub fn with_detectors_json(mut self, json: &str) -> Result<Self, DemBuilderError> {
        self.detectors = parse_detectors_json(json)?;
        Ok(self)
    }

    /// Parses and sets observable definitions from JSON.
    ///
    /// Expected format:
    /// ```json
    /// [
    ///   {"id": 0, "records": [-1, -3, -5]}
    /// ]
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if the JSON is malformed.
    pub fn with_observables_json(mut self, json: &str) -> Result<Self, DemBuilderError> {
        self.observables = parse_observables_json(json)?;
        Ok(self)
    }

    /// Builds the Detector Error Model with source tracking.
    ///
    /// This performs fault propagation analysis and tracks error sources (X/Z vs Y)
    /// through the pipeline, enabling accurate direct/decomposed form splitting.
    ///
    /// Use `dem.to_string()` or `dem.to_string_decomposed()` for output.
    #[must_use]
    pub fn build(&self) -> DetectorErrorModel {
        let mut dem =
            DetectorErrorModel::with_capacity(self.detectors.len(), self.observables.len());

        // Add detector definitions
        for det in &self.detectors {
            let mut def = DetectorDef::new(det.id);
            if let Some(coords) = det.coords {
                def = def.with_coords(coords);
            }
            def = def.with_records(det.records.iter().copied());
            dem.add_detector(def);
        }

        // Add observable definitions
        for obs in &self.observables {
            let def = LogicalObservable::new(obs.id).with_records(obs.records.iter().copied());
            dem.add_observable(def);
        }

        // Build measurement -> detector/observable mappings
        let (meas_to_detectors, meas_to_observables) = self.build_measurement_mappings();

        // Process all fault locations with source tracking
        self.process_fault_locations_source_tracked(
            &mut dem,
            &meas_to_detectors,
            &meas_to_observables,
        );

        dem
    }

    /// Processes fault locations with source tracking.
    ///
    /// This version uses `add_direct_contribution` and `add_y_decomposed_contribution`
    /// to track error sources through the pipeline.
    fn process_fault_locations_source_tracked(
        &self,
        dem: &mut DetectorErrorModel,
        meas_to_detectors: &BTreeMap<usize, Vec<u32>>,
        meas_to_observables: &BTreeMap<usize, Vec<u32>>,
    ) {
        let locations = &self.influence_map.locations;

        // Group CX locations by node for two-qubit gate processing
        let mut cx_groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();

        for (loc_idx, loc) in locations.iter().enumerate() {
            match loc.gate_type {
                GateType::PZ | GateType::QAlloc => {
                    if self.noise.p_init > 0.0 && !loc.before {
                        self.process_prep_fault_source_tracked(
                            loc_idx,
                            dem,
                            meas_to_detectors,
                            meas_to_observables,
                        );
                    }
                }
                GateType::MZ | GateType::MeasureFree => {
                    if self.noise.p_meas > 0.0 && loc.before {
                        self.process_meas_fault_source_tracked(
                            loc_idx,
                            dem,
                            meas_to_detectors,
                            meas_to_observables,
                        );
                    }
                }
                GateType::CX | GateType::CZ => {
                    if !loc.before {
                        cx_groups.entry(loc.node).or_default().push(loc_idx);
                    }
                }
                GateType::H
                | GateType::SZ
                | GateType::SZdg
                | GateType::SX
                | GateType::SXdg
                | GateType::SY
                | GateType::SYdg
                | GateType::X
                | GateType::Y
                | GateType::Z => {
                    if self.noise.p1 > 0.0 && !loc.before {
                        self.process_single_qubit_fault_source_tracked(
                            loc_idx,
                            dem,
                            meas_to_detectors,
                            meas_to_observables,
                        );
                    }
                }
                _ => {}
            }
        }

        // Process two-qubit gates
        if self.noise.p2 > 0.0 {
            for (_, loc_indices) in cx_groups {
                if loc_indices.len() == 2 {
                    self.process_two_qubit_fault_source_tracked(
                        loc_indices[0],
                        loc_indices[1],
                        dem,
                        meas_to_detectors,
                        meas_to_observables,
                    );
                }
            }
        }
    }

    /// Processes a prep fault with source tracking.
    fn process_prep_fault_source_tracked(
        &self,
        loc_idx: usize,
        dem: &mut DetectorErrorModel,
        meas_to_detectors: &BTreeMap<usize, Vec<u32>>,
        meas_to_observables: &BTreeMap<usize, Vec<u32>>,
    ) {
        // For Z-basis prep, X error matters - this is a direct source
        let mechanism =
            self.compute_mechanism(loc_idx, Pauli::X, meas_to_detectors, meas_to_observables);
        if !mechanism.is_empty() {
            dem.add_direct_contribution(mechanism, self.noise.p_init);
        }
    }

    /// Processes a measurement fault with source tracking.
    fn process_meas_fault_source_tracked(
        &self,
        loc_idx: usize,
        dem: &mut DetectorErrorModel,
        meas_to_detectors: &BTreeMap<usize, Vec<u32>>,
        meas_to_observables: &BTreeMap<usize, Vec<u32>>,
    ) {
        // Measurement error is a bit flip (X error) - this is a direct source
        let mechanism =
            self.compute_mechanism(loc_idx, Pauli::X, meas_to_detectors, meas_to_observables);
        if !mechanism.is_empty() {
            dem.add_direct_contribution(mechanism, self.noise.p_meas);
        }
    }

    /// Processes a single-qubit gate fault with source tracking.
    fn process_single_qubit_fault_source_tracked(
        &self,
        loc_idx: usize,
        dem: &mut DetectorErrorModel,
        meas_to_detectors: &BTreeMap<usize, Vec<u32>>,
        meas_to_observables: &BTreeMap<usize, Vec<u32>>,
    ) {
        let prob = per_channel_probability(self.noise.p1, 3);

        let x_effect =
            self.compute_mechanism(loc_idx, Pauli::X, meas_to_detectors, meas_to_observables);
        let z_effect =
            self.compute_mechanism(loc_idx, Pauli::Z, meas_to_detectors, meas_to_observables);

        // X error: direct source
        if !x_effect.is_empty() {
            dem.add_direct_contribution(x_effect.clone(), prob);
        }

        // Z error: direct source
        if !z_effect.is_empty() {
            dem.add_direct_contribution(z_effect.clone(), prob);
        }

        // Y error: Y = XZ, so effect is XOR of X and Z effects
        // Handle all cases:
        // 1. Both non-empty and different: decomposable Y = X ^ Z
        // 2. X non-empty, Z empty: Y has same effect as X (direct)
        // 3. X empty, Z non-empty: Y has same effect as Z (direct)
        // 4. Both non-empty and equal: Y effect is empty (X XOR X = nothing)
        let y_effect = x_effect.xor(&z_effect);
        if !y_effect.is_empty() {
            if !x_effect.is_empty() && !z_effect.is_empty() {
                // Both non-empty, so Y is decomposable as X ^ Z
                dem.add_y_decomposed_contribution(&x_effect, &z_effect, prob);
            } else {
                // One is empty, so Y has same effect as the non-empty one (direct source)
                dem.add_direct_contribution(y_effect, prob);
            }
        }
    }

    /// Processes a two-qubit gate fault with source tracking and intra-channel decomposition.
    fn process_two_qubit_fault_source_tracked(
        &self,
        loc1: usize,
        loc2: usize,
        dem: &mut DetectorErrorModel,
        meas_to_detectors: &BTreeMap<usize, Vec<u32>>,
        meas_to_observables: &BTreeMap<usize, Vec<u32>>,
    ) {
        let prob = per_channel_probability(self.noise.p2, 15);

        // Compute base effects for X and Z on each qubit
        let x1 = self.compute_mechanism(loc1, Pauli::X, meas_to_detectors, meas_to_observables);
        let z1 = self.compute_mechanism(loc1, Pauli::Z, meas_to_detectors, meas_to_observables);
        let x2 = self.compute_mechanism(loc2, Pauli::X, meas_to_detectors, meas_to_observables);
        let z2 = self.compute_mechanism(loc2, Pauli::Z, meas_to_detectors, meas_to_observables);

        // Build effect table for all 16 Pauli combinations
        let get_single_effect = |p: u8, x: &ErrorMechanism, z: &ErrorMechanism| -> ErrorMechanism {
            match p {
                0 => ErrorMechanism::new(), // I
                1 => x.clone(),             // X
                2 => x.xor(z),              // Y = X XOR Z
                3 => z.clone(),             // Z
                _ => unreachable!("Pauli index must be 0-3"),
            }
        };

        let mut effects: [[ErrorMechanism; 4]; 4] = Default::default();
        for p1 in 0..4u8 {
            for p2 in 0..4u8 {
                let e1 = get_single_effect(p1, &x1, &z1);
                let e2 = get_single_effect(p2, &x2, &z2);
                effects[p1 as usize][p2 as usize] = e1.xor(&e2);
            }
        }

        // Process all 15 non-trivial Pauli combinations
        for p1 in 0u8..4 {
            for p2 in 0u8..4 {
                if p1 == 0 && p2 == 0 {
                    continue; // Skip II
                }

                let effect = &effects[p1 as usize][p2 as usize];
                if effect.is_empty() {
                    continue;
                }

                // Get component effects (P1I and IP2)
                let e1 = &effects[p1 as usize][0]; // P1 on qubit 1, I on qubit 2
                let e2 = &effects[0][p2 as usize]; // I on qubit 1, P2 on qubit 2

                // Check if this is a "graphlike decomposable" source:
                // - Combined effect has exactly 2 detectors and no logicals
                // - Both component effects are non-empty
                // - Both component effects are graphlike (≤2 detectors)
                if effect.num_detectors() == 2
                    && effect.logicals.is_empty()
                    && !e1.is_empty()
                    && !e2.is_empty()
                    && e1.num_detectors() <= 2
                    && e2.num_detectors() <= 2
                {
                    dem.mark_graphlike_decomposable(effect.detectors[0], effect.detectors[1]);
                }

                // Check for intra-channel decomposition (Y-containing cases)
                if let Some((a1, a2, b1, b2)) = get_y_decomposition(p1, p2) {
                    // Y-containing channels can be decomposable if both their X and Z
                    // components have non-empty, distinct effects. Otherwise they
                    // produce the effect directly without decomposition.
                    let e_a = &effects[a1 as usize][a2 as usize];
                    let e_b = &effects[b1 as usize][b2 as usize];

                    // Only truly decomposable if both components are non-empty and different.
                    // add_y_decomposed_contribution handles routing to Direct when appropriate.
                    dem.add_y_decomposed_contribution(e_a, e_b, prob);
                } else {
                    // Non-Y channel (XI, IX, ZI, IZ, XX, XZ, ZX, ZZ)
                    // These are always direct sources.
                    dem.add_direct_contribution(effect.clone(), prob);
                }
            }
        }
    }

    /// Builds mappings from measurement indices to detector/observable IDs.
    ///
    /// When `measurement_order` is provided, this properly maps between
    /// `TickCircuit` measurement indices (used in record offsets) and influence
    /// map measurement indices (used in `detector_idx`).
    ///
    /// For multi-round circuits where the same qubit is measured multiple times,
    /// we match measurements by their relative order within each qubit's measurement
    /// sequence.
    fn build_measurement_mappings(&self) -> (BTreeMap<usize, Vec<u32>>, BTreeMap<usize, Vec<u32>>) {
        let mut meas_to_detectors: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
        let mut meas_to_observables: BTreeMap<usize, Vec<u32>> = BTreeMap::new();

        // Build a mapping from (qubit, occurrence_index) to influence_map_index
        // This handles multi-round circuits where the same qubit is measured multiple times
        let tc_to_influence: BTreeMap<usize, usize> =
            if let Some(ref order) = self.measurement_order {
                // Count occurrences of each qubit in TickCircuit order
                let mut tc_qubit_counts: BTreeMap<usize, usize> = BTreeMap::new();
                let mut tc_qubit_occurrence: Vec<(usize, usize)> = Vec::with_capacity(order.len());

                for &qubit in order {
                    let count = tc_qubit_counts.entry(qubit).or_insert(0);
                    tc_qubit_occurrence.push((qubit, *count));
                    *count += 1;
                }

                // Count occurrences of each qubit in influence map order
                let mut im_qubit_counts: BTreeMap<usize, usize> = BTreeMap::new();
                let mut im_qubit_occurrence: Vec<(usize, usize)> =
                    Vec::with_capacity(self.influence_map.measurements.len());

                for &(_, qubit, _) in &self.influence_map.measurements {
                    let count = im_qubit_counts.entry(qubit).or_insert(0);
                    im_qubit_occurrence.push((qubit, *count));
                    *count += 1;
                }

                // Build (qubit, occurrence) -> influence_map_index mapping
                let qubit_occ_to_im: BTreeMap<(usize, usize), usize> = im_qubit_occurrence
                    .iter()
                    .enumerate()
                    .map(|(idx, &(qubit, occ))| ((qubit, occ), idx))
                    .collect();

                // Build TickCircuit index -> influence map index mapping
                tc_qubit_occurrence
                    .iter()
                    .enumerate()
                    .filter_map(|(tc_idx, &(qubit, occ))| {
                        qubit_occ_to_im
                            .get(&(qubit, occ))
                            .map(|&im_idx| (tc_idx, im_idx))
                    })
                    .collect()
            } else {
                // No measurement order provided, assume indices match
                (0..self.num_measurements).map(|i| (i, i)).collect()
            };

        for det in &self.detectors {
            for &rec in &det.records {
                // Convert negative record offset to absolute measurement index in TickCircuit order
                #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)] // measurement count fits in i32
                #[allow(clippy::cast_sign_loss)]
                // negative offset + total count yields valid index
                let tc_meas_idx = (self.num_measurements as i32 + rec) as usize;

                // Map to influence map index
                if let Some(&influence_idx) = tc_to_influence.get(&tc_meas_idx) {
                    meas_to_detectors
                        .entry(influence_idx)
                        .or_default()
                        .push(det.id);
                }
            }
        }

        for obs in &self.observables {
            for &rec in &obs.records {
                #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)] // measurement count fits in i32
                #[allow(clippy::cast_sign_loss)]
                // negative offset + total count yields valid index
                let tc_meas_idx = (self.num_measurements as i32 + rec) as usize;

                if let Some(&influence_idx) = tc_to_influence.get(&tc_meas_idx) {
                    meas_to_observables
                        .entry(influence_idx)
                        .or_default()
                        .push(obs.id);
                }
            }
        }

        (meas_to_detectors, meas_to_observables)
    }

    /// Computes the error mechanism for a fault at the given location and Pauli type.
    fn compute_mechanism(
        &self,
        loc_idx: usize,
        pauli: Pauli,
        meas_to_detectors: &BTreeMap<usize, Vec<u32>>,
        meas_to_observables: &BTreeMap<usize, Vec<u32>>,
    ) -> ErrorMechanism {
        // Get the Rust detector indices that this fault flips
        let rust_dets = self
            .influence_map
            .get_detector_indices(loc_idx, pauli.as_u8());

        // Convert to pre-defined detector IDs using XOR
        let mut triggered_dets: SmallVec<[u32; 4]> = SmallVec::new();
        let mut triggered_obs: SmallVec<[u32; 2]> = SmallVec::new();

        for &rust_det in rust_dets {
            let meas_idx = rust_det as usize;

            // Map to pre-defined detectors
            if let Some(det_ids) = meas_to_detectors.get(&meas_idx) {
                for &det_id in det_ids {
                    xor_toggle_4(&mut triggered_dets, det_id);
                }
            }

            // Map to observables
            if let Some(obs_ids) = meas_to_observables.get(&meas_idx) {
                for &obs_id in obs_ids {
                    xor_toggle_2(&mut triggered_obs, obs_id);
                }
            }
        }

        // Sort for canonical form
        triggered_dets.sort_unstable();
        triggered_obs.sort_unstable();

        ErrorMechanism::from_sorted(triggered_dets, triggered_obs)
    }
}

/// Toggles an element in a vec (add if not present, remove if present).
fn xor_toggle_4(vec: &mut SmallVec<[u32; 4]>, value: u32) {
    if let Some(pos) = vec.iter().position(|&v| v == value) {
        vec.remove(pos);
    } else {
        vec.push(value);
    }
}

/// Toggles an element in a vec (add if not present, remove if present).
fn xor_toggle_2(vec: &mut SmallVec<[u32; 2]>, value: u32) {
    if let Some(pos) = vec.iter().position(|&v| v == value) {
        vec.remove(pos);
    } else {
        vec.push(value);
    }
}

/// Computes the per-error probability for independent error channels.
///
/// For a depolarizing channel with total error probability `p` split among `n`
/// independent Pauli channels, this computes the probability for each channel
/// such that the combined probability of any error occurring equals `p`.
///
/// Formula: `p_each = 1 - (1-p)^(1/n)`
///
/// This is derived from: `P(at least one error) = 1 - P(no errors) = 1 - (1-p_each)^n = p`
///
/// For small `p`, this is approximately `p/n`, but the exact formula accounts
/// for the independence of error channels.
///
/// # Arguments
///
/// * `total_prob` - Total depolarizing probability (e.g., 0.02 for 2% error rate)
/// * `num_channels` - Number of independent error channels (3 for DEPOLARIZE1, 15 for DEPOLARIZE2)
///
/// # Returns
///
/// Per-channel error probability
#[inline]
fn per_channel_probability(total_prob: f64, num_channels: u32) -> f64 {
    if total_prob <= 0.0 {
        return 0.0;
    }
    if total_prob >= 1.0 {
        return 1.0;
    }
    // p_each = 1 - (1-p)^(1/n)
    1.0 - (1.0 - total_prob).powf(1.0 / f64::from(num_channels))
}

// ============================================================================
// Intra-Channel Decomposition
// ============================================================================

/// Returns the intra-channel decomposition for Y-containing Pauli cases.
///
/// For any two-qubit Pauli case (p1, p2) that contains Y, returns the
/// decomposition (a1, a2, b1, b2) such that:
///   effect(p1, p2) = effect(a1, a2) XOR effect(b1, b2)
///
/// This is based on the Pauli algebra identity Y = iXZ (phase ignored for effects):
/// - YI = XI * ZI  (tensor product: Y⊗I = (X⊗I)(Z⊗I))
/// - IY = IX * IZ
/// - XY = XX * IZ  (X⊗Y = X⊗(XZ) = (X⊗X)(I⊗Z))
/// - YX = XX * ZI
/// - YY = XX * ZZ
/// - YZ = XZ * ZI
/// - ZY = ZX * IZ
///
/// Pauli indices: I=0, X=1, Y=2, Z=3
///
/// Returns `None` if the case doesn't contain Y (no decomposition needed).
#[inline]
fn get_y_decomposition(p1: u8, p2: u8) -> Option<(u8, u8, u8, u8)> {
    // Only Y-containing cases can be decomposed
    match (p1, p2) {
        (2, 0) => Some((1, 0, 3, 0)), // YI -> XI ^ ZI
        (0, 2) => Some((0, 1, 0, 3)), // IY -> IX ^ IZ
        (1, 2) => Some((1, 1, 0, 3)), // XY -> XX ^ IZ
        (2, 1) => Some((1, 1, 3, 0)), // YX -> XX ^ ZI
        (2, 2) => Some((1, 1, 3, 3)), // YY -> XX ^ ZZ
        (2, 3) => Some((1, 3, 3, 0)), // YZ -> XZ ^ ZI
        (3, 2) => Some((3, 1, 0, 3)), // ZY -> ZX ^ IZ
        _ => None,                    // No Y involved
    }
}

// ============================================================================
// JSON Parsing
// ============================================================================

/// Parses detector definitions from JSON.
fn parse_detectors_json(json: &str) -> Result<Vec<ParsedDetector>, DemBuilderError> {
    // Simple JSON parsing without serde dependency
    // Expected format: [{"id": 0, "coords": [0.0, 0.0, 0.0], "records": [-1, -5]}, ...]

    let json = json.trim();
    if json.is_empty() || json == "[]" {
        return Ok(Vec::new());
    }

    let mut detectors = Vec::new();

    // Find all objects in the array
    let mut depth = 0;
    let mut obj_start = None;

    for (i, c) in json.char_indices() {
        match c {
            '[' if depth == 0 => depth = 1,
            '{' if depth == 1 => {
                depth = 2;
                obj_start = Some(i);
            }
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 1 {
                    if let Some(start) = obj_start {
                        // i is the byte index of '}', we want to include it
                        let obj_str = &json[start..i + c.len_utf8()];
                        let det = parse_single_detector(obj_str)?;
                        detectors.push(det);
                    }
                    obj_start = None;
                }
            }
            _ => {}
        }
    }

    Ok(detectors)
}

/// Parses a single detector object.
fn parse_single_detector(json: &str) -> Result<ParsedDetector, DemBuilderError> {
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    // detector IDs are small non-negative integers
    let id = extract_number(json, "\"id\"")
        .ok_or_else(|| DemBuilderError::ParseError("missing detector id".into()))?
        as u32;

    let coords = extract_coords(json);
    let records = extract_records(json);

    Ok(ParsedDetector {
        id,
        coords,
        records,
    })
}

/// Parses observable definitions from JSON.
fn parse_observables_json(json: &str) -> Result<Vec<ParsedObservable>, DemBuilderError> {
    let json = json.trim();
    if json.is_empty() || json == "[]" {
        return Ok(Vec::new());
    }

    let mut observables = Vec::new();

    let mut depth = 0;
    let mut obj_start = None;

    for (i, c) in json.char_indices() {
        match c {
            '[' if depth == 0 => depth = 1,
            '{' if depth == 1 => {
                depth = 2;
                obj_start = Some(i);
            }
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 1 {
                    if let Some(start) = obj_start {
                        // i is the byte index of '}', we want to include it
                        let obj_str = &json[start..i + c.len_utf8()];
                        let obs = parse_single_observable(obj_str)?;
                        observables.push(obs);
                    }
                    obj_start = None;
                }
            }
            _ => {}
        }
    }

    Ok(observables)
}

/// Parses a single observable object.
fn parse_single_observable(json: &str) -> Result<ParsedObservable, DemBuilderError> {
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    // observable IDs are small non-negative integers
    let id = extract_number(json, "\"id\"")
        .ok_or_else(|| DemBuilderError::ParseError("missing observable id".into()))?
        as u32;

    let records = extract_records(json);

    Ok(ParsedObservable { id, records })
}

/// Extracts a number after a key.
fn extract_number(json: &str, key: &str) -> Option<i64> {
    let pos = json.find(key)?;
    let rest = &json[pos + key.len()..];
    let rest = rest.trim_start_matches(|c: char| c == ':' || c.is_whitespace());

    let end = rest.find(|c: char| !c.is_ascii_digit() && c != '-' && c != '.')?;
    let num_str = &rest[..end];
    num_str.parse().ok()
}

/// Extracts coordinates array [x, y, t].
fn extract_coords(json: &str) -> Option<[f64; 3]> {
    let pos = json.find("\"coords\"")?;
    let rest = &json[pos..];
    let bracket_start = rest.find('[')?;
    let bracket_end = rest.find(']')?;
    let array_str = &rest[bracket_start + 1..bracket_end];

    let nums: Vec<f64> = array_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    if nums.len() == 3 {
        Some([nums[0], nums[1], nums[2]])
    } else {
        None
    }
}

/// Extracts records array.
fn extract_records(json: &str) -> Vec<i32> {
    if let Some(pos) = json.find("\"records\"") {
        let rest = &json[pos..];
        if let Some(bracket_start) = rest.find('[')
            && let Some(bracket_end) = rest.find(']')
        {
            let array_str = &rest[bracket_start + 1..bracket_end];
            return array_str
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
        }
    }
    Vec::new()
}

// ============================================================================
// Error Type
// ============================================================================

/// Errors that can occur during DEM building.
#[derive(Debug, Clone)]
pub enum DemBuilderError {
    /// JSON parsing error.
    ParseError(String),
}

impl std::fmt::Display for DemBuilderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParseError(msg) => write!(f, "DEM builder parse error: {msg}"),
        }
    }
}

impl std::error::Error for DemBuilderError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_detectors_json() {
        let json = r#"[
            {"id": 0, "coords": [0.0, 0.0, 0.0], "records": [-1, -5]},
            {"id": 1, "coords": [1.0, 0.0, 0.0], "records": [-2]}
        ]"#;

        let detectors = parse_detectors_json(json).unwrap();

        assert_eq!(detectors.len(), 2);
        assert_eq!(detectors[0].id, 0);
        assert_eq!(detectors[0].coords, Some([0.0, 0.0, 0.0]));
        assert_eq!(detectors[0].records, vec![-1, -5]);
        assert_eq!(detectors[1].id, 1);
        assert_eq!(detectors[1].records, vec![-2]);
    }

    #[test]
    fn test_parse_observables_json() {
        let json = r#"[{"id": 0, "records": [-1, -3, -5]}]"#;

        let observables = parse_observables_json(json).unwrap();

        assert_eq!(observables.len(), 1);
        assert_eq!(observables[0].id, 0);
        assert_eq!(observables[0].records, vec![-1, -3, -5]);
    }

    #[test]
    fn test_parse_empty_json() {
        assert!(parse_detectors_json("").unwrap().is_empty());
        assert!(parse_detectors_json("[]").unwrap().is_empty());
        assert!(parse_observables_json("").unwrap().is_empty());
    }

    #[test]
    fn test_xor_toggle() {
        let mut vec: SmallVec<[u32; 4]> = SmallVec::new();

        xor_toggle_4(&mut vec, 1);
        assert_eq!(vec.as_slice(), &[1]);

        xor_toggle_4(&mut vec, 2);
        assert_eq!(vec.as_slice(), &[1, 2]);

        xor_toggle_4(&mut vec, 1); // Toggle off
        assert_eq!(vec.as_slice(), &[2]);

        xor_toggle_4(&mut vec, 2); // Toggle off
        assert!(vec.is_empty());
    }

    #[test]
    fn test_per_channel_probability() {
        // Test DEPOLARIZE1: p=0.01, n=3
        let p1 = per_channel_probability(0.01, 3);
        // Should be 1 - (1-0.01)^(1/3) = 0.003344...
        assert!((p1 - 0.003_344_506).abs() < 1e-6);

        // Verify: combining 3 channels gives back ~p
        let combined = 1.0 - (1.0 - p1).powi(3);
        assert!((combined - 0.01).abs() < 1e-10);

        // Test DEPOLARIZE2: p=0.02, n=15
        let p2 = per_channel_probability(0.02, 15);
        // Should be 1 - (1-0.02)^(1/15) = 0.001346...
        assert!((p2 - 0.001_345_941).abs() < 1e-6);

        // Verify: combining 15 channels gives back ~p
        let combined2 = 1.0 - (1.0 - p2).powi(15);
        assert!((combined2 - 0.02).abs() < 1e-10);

        // Edge cases
        assert!((per_channel_probability(0.0, 3) - 0.0).abs() < f64::EPSILON);
        assert!((per_channel_probability(1.0, 3) - 1.0).abs() < f64::EPSILON);
        assert!((per_channel_probability(-0.1, 3) - 0.0).abs() < f64::EPSILON);

        // For small p, should be close to p/n
        let small_p = per_channel_probability(0.001, 15);
        let simple = 0.001 / 15.0;
        // Difference should be < 0.1% for small p
        assert!((small_p - simple).abs() / simple < 0.001);
    }
}
