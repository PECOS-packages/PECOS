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
//! influence maps and detector/DEM-output metadata.

use super::types::{
    DemOutput, DetectorDef, DetectorErrorModel, DirectSourceComponents, FaultMechanism,
    NoiseConfig, PerGateTypeNoise, SourceMetadata, record_offset_to_absolute_index,
};
use crate::fault_tolerance::propagator::dag::DagSpacetimeLocation;
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
/// # Simple API (recommended)
///
/// For most use cases, use the one-liner:
///
/// ```
/// use pecos_qec::fault_tolerance::dem_builder::DemBuilder;
/// use pecos_quantum::DagCircuit;
///
/// // Build DEM from circuit + noise (reads detectors from circuit metadata)
/// let dag = DagCircuit::new();
/// let dem = DemBuilder::from_circuit(&dag, 0.001, 0.01, 0.001, 0.001);
/// assert_eq!(dem.num_detectors(), 0);
/// ```
///
/// Also works with `TickCircuit`:
///
/// ```
/// use pecos_qec::fault_tolerance::dem_builder::DemBuilder;
/// use pecos_quantum::TickCircuit;
///
/// let tc = TickCircuit::new();
/// let dem = DemBuilder::from_tick_circuit(&tc, 0.001, 0.01, 0.001, 0.001);
/// assert_eq!(dem.num_detectors(), 0);
/// ```
///
/// # Advanced API
///
/// For custom influence maps, non-standard noise, or manual detector
/// definitions, use the step-by-step builder:
///
/// ```no_run
/// # use pecos_qec::fault_tolerance::dem_builder::DemBuilder;
/// # use pecos_qec::fault_tolerance::propagator::DagFaultInfluenceMap;
/// # let influence_map = DagFaultInfluenceMap::with_capacity(0);
/// let dem = DemBuilder::new(&influence_map)
///     .with_noise(0.01, 0.01, 0.01, 0.01)
///     .with_detectors_json("[]").unwrap()
///     .build();
/// ```
pub struct DemBuilder<'a> {
    /// Reference to the fault influence map.
    influence_map: &'a DagFaultInfluenceMap,
    /// Uniform-depolarizing noise configuration. When `per_gate` is also
    /// set, its per-qubit / per-Pauli overrides take precedence; this
    /// `NoiseConfig` still seeds measurement/prep scalars.
    noise: NoiseConfig,
    /// Optional per-gate-type per-Pauli noise spec. Mirrors the
    /// `DemSamplerBuilder` path so DEM text export reflects the same
    /// asymmetric noise structure that the sampler does.
    per_gate: Option<PerGateTypeNoise>,
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
    /// Build a `DetectorErrorModel` directly from a circuit and noise.
    ///
    /// One-liner for the common case. Reads detector/DEM output definitions
    /// from circuit metadata (`"detectors"`, `"observables"` attributes).
    ///
    /// ```
    /// use pecos_qec::fault_tolerance::dem_builder::DemBuilder;
    /// use pecos_quantum::DagCircuit;
    ///
    /// let dag = DagCircuit::new();
    /// let dem = DemBuilder::from_circuit(&dag, 0.001, 0.01, 0.001, 0.001);
    /// assert_eq!(dem.num_detectors(), 0);
    /// ```
    /// Build a `DetectorErrorModel` directly from a `DagCircuit` and noise.
    ///
    /// One-liner for the common case. Reads detector/DEM output definitions
    /// from circuit metadata.
    #[must_use]
    pub fn from_circuit(
        circuit: &pecos_quantum::DagCircuit,
        p1: f64,
        p2: f64,
        p_meas: f64,
        p_prep: f64,
    ) -> DetectorErrorModel {
        build_dem_from_circuit(circuit, p1, p2, p_meas, p_prep)
    }

    /// Build a `DetectorErrorModel` from a `TickCircuit` and noise.
    ///
    /// Converts to `DagCircuit` internally.
    #[must_use]
    pub fn from_tick_circuit(
        circuit: &pecos_quantum::TickCircuit,
        p1: f64,
        p2: f64,
        p_meas: f64,
        p_prep: f64,
    ) -> DetectorErrorModel {
        let dag = pecos_quantum::DagCircuit::from(circuit);
        build_dem_from_circuit(&dag, p1, p2, p_meas, p_prep)
    }

    /// Creates a new DEM builder from a fault influence map.
    #[must_use]
    pub fn new(influence_map: &'a DagFaultInfluenceMap) -> Self {
        Self {
            influence_map,
            noise: NoiseConfig::default(),
            per_gate: None,
            detectors: Vec::new(),
            observables: Vec::new(),
            num_measurements: influence_map.measurements.len(),
            measurement_order: None,
        }
    }

    /// Sets the noise configuration from individual parameters.
    #[must_use]
    pub fn with_noise(mut self, p1: f64, p2: f64, p_meas: f64, p_prep: f64) -> Self {
        self.noise = NoiseConfig::new(p1, p2, p_meas, p_prep);
        self
    }

    /// Sets the full noise configuration (supports custom weights, T1/T2, idle).
    #[must_use]
    pub fn with_noise_config(mut self, noise: NoiseConfig) -> Self {
        self.noise = noise;
        self
    }

    /// Attach per-gate-type per-Pauli noise. When present, overrides
    /// [`Self::with_noise`] scalars for gate types in the spec's maps.
    /// Mirrors
    /// [`crate::fault_tolerance::dem_builder::DemSamplerBuilder::with_per_gate_noise`]
    /// so the DEM text output reflects the same noise structure.
    #[must_use]
    pub fn with_per_gate_noise(mut self, cfg: PerGateTypeNoise) -> Self {
        self.noise.p_meas = cfg.p_meas;
        self.noise.p_prep = cfg.p_init;
        self.per_gate = Some(cfg);
        self
    }

    /// Resolve preparation X-error rate at a specific location.
    fn init_rate_for_loc(&self, loc: &DagSpacetimeLocation) -> f64 {
        if let Some(pg) = &self.per_gate
            && let Some(q) = loc.qubits.first()
        {
            return pg.init_rate_on(*q);
        }
        self.noise.p_prep
    }

    /// Resolve measurement X-flip rate at a specific location.
    fn measurement_rate_for_loc(&self, loc: &DagSpacetimeLocation) -> f64 {
        if let Some(pg) = &self.per_gate
            && let Some(q) = loc.qubits.first()
        {
            return pg.measurement_rate_on(*q);
        }
        self.noise.p_meas
    }

    /// Resolve `[rate_X, rate_Y, rate_Z]` for a 1Q gate location.
    fn rates_1q_for_loc(&self, loc: &DagSpacetimeLocation) -> [f64; 3] {
        if let Some(pg) = &self.per_gate {
            if let Some(q) = loc.qubits.first() {
                return [
                    pg.rate_1q_on(loc.gate_type, *q, 0),
                    pg.rate_1q_on(loc.gate_type, *q, 1),
                    pg.rate_1q_on(loc.gate_type, *q, 2),
                ];
            }
            return [
                pg.rate_1q(loc.gate_type, 0),
                pg.rate_1q(loc.gate_type, 1),
                pg.rate_1q(loc.gate_type, 2),
            ];
        }
        if let Some(weights) = &self.noise.p1_weights {
            use pecos_core::pauli::{X, Y, Z};
            return [
                self.noise.p1 * weights.weight_for(&X(0)),
                self.noise.p1 * weights.weight_for(&Y(0)),
                self.noise.p1 * weights.weight_for(&Z(0)),
            ];
        }
        let per = per_channel_probability(self.noise.p1, 3);
        [per, per, per]
    }

    /// Resolve `[rate_X, rate_Y, rate_Z]` for an explicit idle location.
    fn idle_rates_for_loc(&self, loc: &DagSpacetimeLocation) -> [f64; 3] {
        if let Some(pg) = &self.per_gate {
            let explicit_rates = loc
                .qubits
                .first()
                .and_then(|q| pg.explicit_1q_rates_on(GateType::Idle, *q))
                .or_else(|| pg.explicit_1q_rates(GateType::Idle));
            if let Some(rates) = explicit_rates {
                return rates;
            }
            if pg.base.uses_dedicated_idle_noise() {
                #[allow(clippy::cast_precision_loss)]
                let duration = loc.idle_duration.max(1) as f64;
                let probs = pg.base.idle_pauli_probs(duration);
                return [probs.px, probs.py, probs.pz];
            }
            return [0.0; 3];
        }

        if self.noise.uses_dedicated_idle_noise() {
            #[allow(clippy::cast_precision_loss)]
            let duration = loc.idle_duration.max(1) as f64;
            let probs = self.noise.idle_pauli_probs(duration);
            return [probs.px, probs.py, probs.pz];
        }
        [0.0; 3]
    }

    /// Resolve the 15-entry 2Q per-Pauli-pair rate array for a gate
    /// spanning two fault locations.
    fn rates_2q_for_locs(
        &self,
        loc1: &DagSpacetimeLocation,
        loc2: &DagSpacetimeLocation,
    ) -> [f64; 15] {
        if let Some(pg) = &self.per_gate {
            let gate = loc1.gate_type;
            let mut qubits = loc1
                .qubits
                .iter()
                .copied()
                .chain(loc2.qubits.iter().copied());
            if let (Some(qc), Some(qt)) = (qubits.next(), qubits.next()) {
                return std::array::from_fn(|i| pg.rate_2q_on(gate, qc, qt, i));
            }
            return std::array::from_fn(|i| pg.rate_2q(gate, i));
        }
        if let Some(weights) = &self.noise.p2_weights {
            return std::array::from_fn(|idx| {
                let flat = idx + 1;
                let p1 = flat / 4;
                let p2 = flat % 4;
                self.noise.p2 * weights.weight_for(&pauli_pair_for_weight(p1, p2))
            });
        }
        [per_channel_probability(self.noise.p2, 15); 15]
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
    /// Set the measurement order for legacy circuits without `MeasId` on gates.
    ///
    /// **Not needed for circuits built with `TickCircuit.mz()`** — the `MeasId`
    /// values on gates ensure correct ordering automatically.
    ///
    /// Only use this for circuits where MZ gates lack `meas_ids` (e.g.,
    /// circuits imported from external formats without measurement IDs).
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
    /// Each object accepts either `"id"` or `"detector_id"` as the identifier key.
    ///
    /// Expected format:
    /// ```json
    /// [
    ///   {"id": 0, "coords": [0.0, 0.0, 0.0], "records": [-1, -5]},
    ///   {"detector_id": 1, "coords": [1.0, 0.0, 0.0], "records": [-2]}
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
    /// Tracked Paulis are carried by the influence map; this helper is only
    /// for observable metadata.
    ///
    /// Each object accepts either `"id"` or `"observable_id"` as the identifier key.
    ///
    /// # Errors
    ///
    /// Returns an error if the JSON is malformed.
    pub fn with_observables_json(mut self, json: &str) -> Result<Self, DemBuilderError> {
        self.observables = parse_observables_json(json)?;
        Ok(self)
    }

    /// Sets observable definitions from measurement-record offsets.
    #[must_use]
    pub fn with_observable_records(mut self, records: Vec<Vec<i32>>) -> Self {
        self.observables = records
            .into_iter()
            .enumerate()
            .map(|(id, records)| ParsedObservable {
                #[allow(clippy::cast_possible_truncation)] // observable count fits in u32
                id: id as u32,
                records,
            })
            .collect();
        self
    }

    /// Builds the Detector Error Model with source tracking.
    ///
    /// This performs fault propagation analysis and tracks error sources (X/Z vs Y)
    /// through the pipeline, enabling accurate direct/decomposed form splitting.
    ///
    /// Use `dem.to_string()` or `dem.to_string_decomposed()` for output.
    #[must_use]
    pub fn build(&self) -> DetectorErrorModel {
        let num_influence_dem_outputs = self
            .num_influence_dem_outputs()
            .max(self.influence_map.dem_output_metadata.len());
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

        // Add non-detector outputs carried directly by the influence map.
        // Metadata-bearing outputs use separate compact ID spaces for standard
        // observables and PECOS tracked Paulis.
        if self.influence_map.dem_output_metadata.is_empty() {
            for dem_output_idx in 0..num_influence_dem_outputs {
                #[allow(clippy::cast_possible_truncation)] // DEM output count fits in u32
                dem.add_observable(DemOutput::new(dem_output_idx as u32));
            }
        } else {
            for (internal_idx, metadata) in
                self.influence_map.dem_output_metadata.iter().enumerate()
            {
                #[allow(clippy::cast_possible_truncation)] // DEM output count fits in u32
                let internal_id = internal_idx as u32;
                if let Some(dem_output_id) = self
                    .influence_map
                    .tracked_pauli_id_for_internal_dem_output(internal_id)
                {
                    dem.add_tracked_pauli(DemOutput::from_metadata(dem_output_id, metadata));
                } else if let Some(dem_output_id) = self
                    .influence_map
                    .observable_id_for_internal_dem_output(internal_id)
                {
                    dem.add_observable(DemOutput::from_metadata(dem_output_id, metadata));
                }
            }
        }

        // Add observable definitions in the standard `L<n>` namespace.
        // Observable IDs are not shifted by tracked Paulis.
        for obs in &self.observables {
            let def = DemOutput::new(obs.id).with_records(obs.records.iter().copied());
            dem.add_observable(def);
        }

        // Build measurement -> detector/DEM-output mappings
        let (meas_to_detectors, meas_to_observables) = self.build_measurement_mappings();

        // Process all fault locations with source tracking
        self.process_fault_locations_source_tracked(
            &mut dem,
            &meas_to_detectors,
            &meas_to_observables,
        );

        dem
    }

    fn num_influence_dem_outputs(&self) -> usize {
        self.influence_map
            .influences
            .max_dem_output_index()
            .map_or(0, |idx| idx + 1)
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
                GateType::PZ | GateType::QAlloc
                    if !loc.before && self.init_rate_for_loc(loc) > 0.0 =>
                {
                    self.process_prep_fault_source_tracked(
                        loc_idx,
                        dem,
                        meas_to_detectors,
                        meas_to_observables,
                    );
                }
                GateType::MZ | GateType::MeasureFree
                    if loc.before && self.measurement_rate_for_loc(loc) > 0.0 =>
                {
                    self.process_meas_fault_source_tracked(
                        loc_idx,
                        dem,
                        meas_to_detectors,
                        meas_to_observables,
                    );
                }
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
                | GateType::RZZ
                    if !loc.before =>
                {
                    cx_groups.entry(loc.node).or_default().push(loc_idx);
                }
                GateType::H
                | GateType::F
                | GateType::Fdg
                | GateType::SZ
                | GateType::SZdg
                | GateType::SX
                | GateType::SXdg
                | GateType::SY
                | GateType::SYdg
                | GateType::X
                | GateType::Y
                | GateType::Z
                | GateType::T
                | GateType::Tdg
                | GateType::RX
                | GateType::RY
                | GateType::RZ
                | GateType::U
                | GateType::R1XY
                    if !loc.before =>
                {
                    let rates = self.rates_1q_for_loc(loc);
                    if rates.iter().any(|r| *r > 0.0) {
                        self.process_single_qubit_fault_source_tracked(
                            loc_idx,
                            rates,
                            dem,
                            meas_to_detectors,
                            meas_to_observables,
                        );
                    }
                }
                GateType::Idle if !loc.before => {
                    let rates = self.idle_rates_for_loc(loc);
                    if rates.iter().any(|r| *r > 0.0) {
                        self.process_single_qubit_fault_source_tracked(
                            loc_idx,
                            rates,
                            dem,
                            meas_to_detectors,
                            meas_to_observables,
                        );
                    }
                }
                _ => {}
            }
        }

        // Process two-qubit gates.
        for (_, loc_indices) in cx_groups {
            for pair in loc_indices.chunks(2) {
                if pair.len() != 2 {
                    continue;
                }
                let loc1 = &locations[pair[0]];
                let loc2 = &locations[pair[1]];
                let rates = self.rates_2q_for_locs(loc1, loc2);
                if rates.iter().any(|r| *r > 0.0) {
                    self.process_two_qubit_fault_source_tracked(
                        pair[0],
                        pair[1],
                        rates,
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
        let loc = &self.influence_map.locations[loc_idx];
        let p = self.init_rate_for_loc(loc);
        // For Z-basis prep, X error matters - this is a direct source
        let mechanism =
            self.compute_mechanism(loc_idx, Pauli::X, meas_to_detectors, meas_to_observables);
        if !mechanism.is_empty() {
            dem.add_direct_contribution_with_source(
                mechanism,
                p,
                SourceMetadata::new(&[loc_idx], &[Pauli::X], &[loc.gate_type], &[loc.before]),
            );
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
        let loc = &self.influence_map.locations[loc_idx];
        let p = self.measurement_rate_for_loc(loc);
        // Measurement error is a bit flip (X error) - this is a direct source
        let mechanism =
            self.compute_mechanism(loc_idx, Pauli::X, meas_to_detectors, meas_to_observables);
        if !mechanism.is_empty() {
            dem.add_direct_contribution_with_source(
                mechanism,
                p,
                SourceMetadata::new(&[loc_idx], &[Pauli::X], &[loc.gate_type], &[loc.before]),
            );
        }
    }

    /// Processes a single-qubit gate fault with source tracking.
    /// `rates` is `[rate_X, rate_Y, rate_Z]` -- zero entries are skipped.
    fn process_single_qubit_fault_source_tracked(
        &self,
        loc_idx: usize,
        rates: [f64; 3],
        dem: &mut DetectorErrorModel,
        meas_to_detectors: &BTreeMap<usize, Vec<u32>>,
        meas_to_observables: &BTreeMap<usize, Vec<u32>>,
    ) {
        let [rate_x, rate_y, rate_z] = rates;

        let x_effect =
            self.compute_mechanism(loc_idx, Pauli::X, meas_to_detectors, meas_to_observables);
        let z_effect =
            self.compute_mechanism(loc_idx, Pauli::Z, meas_to_detectors, meas_to_observables);

        // X error: direct source
        if rate_x > 0.0 && !x_effect.is_empty() {
            dem.add_direct_contribution_with_source(
                x_effect.clone(),
                rate_x,
                SourceMetadata::new(
                    &[loc_idx],
                    &[Pauli::X],
                    &[self.influence_map.locations[loc_idx].gate_type],
                    &[self.influence_map.locations[loc_idx].before],
                ),
            );
        }

        // Z error: direct source
        if rate_z > 0.0 && !z_effect.is_empty() {
            dem.add_direct_contribution_with_source(
                z_effect.clone(),
                rate_z,
                SourceMetadata::new(
                    &[loc_idx],
                    &[Pauli::Z],
                    &[self.influence_map.locations[loc_idx].gate_type],
                    &[self.influence_map.locations[loc_idx].before],
                ),
            );
        }

        // Y error: Y = XZ, so effect is XOR of X and Z effects
        let y_effect = x_effect.xor(&z_effect);
        if rate_y > 0.0 && !y_effect.is_empty() {
            if !x_effect.is_empty() && !z_effect.is_empty() {
                dem.add_y_decomposed_contribution_with_source(
                    &x_effect,
                    &z_effect,
                    rate_y,
                    SourceMetadata::new(
                        &[loc_idx],
                        &[Pauli::Y],
                        &[self.influence_map.locations[loc_idx].gate_type],
                        &[self.influence_map.locations[loc_idx].before],
                    ),
                );
            } else {
                // One is empty, so Y has same effect as the non-empty one (direct source)
                dem.add_direct_contribution_with_source(
                    y_effect,
                    rate_y,
                    SourceMetadata::new(
                        &[loc_idx],
                        &[Pauli::Y],
                        &[self.influence_map.locations[loc_idx].gate_type],
                        &[self.influence_map.locations[loc_idx].before],
                    ),
                );
            }
        }
    }

    /// Processes a two-qubit gate fault with source tracking and intra-channel decomposition.
    /// `rates` is the 15-entry array in `PAULI_2Q_ORDER` order -- zero entries
    /// are skipped.
    fn process_two_qubit_fault_source_tracked(
        &self,
        loc1: usize,
        loc2: usize,
        rates: [f64; 15],
        dem: &mut DetectorErrorModel,
        meas_to_detectors: &BTreeMap<usize, Vec<u32>>,
        meas_to_observables: &BTreeMap<usize, Vec<u32>>,
    ) {
        let loc1_meta = &self.influence_map.locations[loc1];
        let loc2_meta = &self.influence_map.locations[loc2];

        // Compute base effects for X and Z on each qubit
        let x1 = self.compute_mechanism(loc1, Pauli::X, meas_to_detectors, meas_to_observables);
        let z1 = self.compute_mechanism(loc1, Pauli::Z, meas_to_detectors, meas_to_observables);
        let x2 = self.compute_mechanism(loc2, Pauli::X, meas_to_detectors, meas_to_observables);
        let z2 = self.compute_mechanism(loc2, Pauli::Z, meas_to_detectors, meas_to_observables);

        // Build effect table for all 16 Pauli combinations
        let get_single_effect = |p: u8, x: &FaultMechanism, z: &FaultMechanism| -> FaultMechanism {
            match p {
                0 => FaultMechanism::new(), // I
                1 => x.clone(),             // X
                2 => x.xor(z),              // Y = X XOR Z
                3 => z.clone(),             // Z
                _ => unreachable!("Pauli index must be 0-3"),
            }
        };

        let mut effects: [[FaultMechanism; 4]; 4] = Default::default();
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

                // Per-pair rate: index = 4*p1 + p2 - 1 (skipping II at idx 0).
                let flat = 4 * (p1 as usize) + (p2 as usize);
                let prob = rates[flat - 1];
                if prob == 0.0 {
                    continue;
                }

                // Get component effects (P1I and IP2)
                let e1 = &effects[p1 as usize][0]; // P1 on qubit 1, I on qubit 2
                let e2 = &effects[0][p2 as usize]; // I on qubit 1, P2 on qubit 2

                // Check if this is a "graphlike decomposable" source:
                // - Combined effect has exactly 2 detectors and no dem_outputs
                // - Both component effects are non-empty
                // - Both component effects are graphlike (≤2 detectors)
                let graphlike_decomposable = effect.num_detectors() == 2
                    && effect.dem_outputs.is_empty()
                    && !e1.is_empty()
                    && !e2.is_empty()
                    && e1.num_detectors() <= 2
                    && e2.num_detectors() <= 2;
                if graphlike_decomposable {
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
                    dem.add_y_decomposed_contribution_with_source(
                        e_a,
                        e_b,
                        prob,
                        SourceMetadata::new(
                            &[loc1, loc2],
                            &[Pauli::from_u8(p1), Pauli::from_u8(p2)],
                            &[loc1_meta.gate_type, loc2_meta.gate_type],
                            &[loc1_meta.before, loc2_meta.before],
                        ),
                    );
                } else {
                    // Non-Y channel (XI, IX, ZI, IZ, XX, XZ, ZX, ZZ)
                    // These are always direct sources.
                    dem.add_direct_contribution_with_source_components(
                        effect.clone(),
                        prob,
                        SourceMetadata::new(
                            &[loc1, loc2],
                            &[Pauli::from_u8(p1), Pauli::from_u8(p2)],
                            &[loc1_meta.gate_type, loc2_meta.gate_type],
                            &[loc1_meta.before, loc2_meta.before],
                        ),
                        DirectSourceComponents::new(e1, e2),
                    );
                }
            }
        }
    }

    /// Builds mappings from measurement indices to detector/DEM-output IDs.
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
        let influence_observable_ids = self.influence_map.observable_ids();

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
                if let Some(tc_meas_idx) =
                    record_offset_to_absolute_index(self.num_measurements, rec)
                    && let Some(&influence_idx) = tc_to_influence.get(&tc_meas_idx)
                {
                    meas_to_detectors
                        .entry(influence_idx)
                        .or_default()
                        .push(det.id);
                }
            }
        }

        for obs in &self.observables {
            if influence_observable_ids.contains(&obs.id) {
                continue;
            }
            for &rec in &obs.records {
                if let Some(tc_meas_idx) =
                    record_offset_to_absolute_index(self.num_measurements, rec)
                    && let Some(&influence_idx) = tc_to_influence.get(&tc_meas_idx)
                {
                    meas_to_observables
                        .entry(influence_idx)
                        .or_default()
                        .push(obs.id);
                }
            }
        }

        (meas_to_detectors, meas_to_observables)
    }

    /// Computes the fault mechanism for a fault at the given location and Pauli type.
    fn compute_mechanism(
        &self,
        loc_idx: usize,
        pauli: Pauli,
        meas_to_detectors: &BTreeMap<usize, Vec<u32>>,
        meas_to_observables: &BTreeMap<usize, Vec<u32>>,
    ) -> FaultMechanism {
        // Get the measurement indices that this fault flips
        let rust_dets = self
            .influence_map
            .get_detector_indices(loc_idx, pauli.as_u8());

        // Convert to pre-defined detector IDs using XOR
        let mut triggered_dets: SmallVec<[u32; 4]> = SmallVec::new();
        let mut triggered_obs: SmallVec<[u32; 2]> = SmallVec::new();
        let mut triggered_tracked_paulis: SmallVec<[u32; 2]> = SmallVec::new();

        for dem_output_idx in self
            .influence_map
            .get_observable_indices(loc_idx, pauli.as_u8())
        {
            xor_toggle_2(&mut triggered_obs, dem_output_idx);
        }
        for tracked_pauli_idx in self
            .influence_map
            .get_tracked_pauli_indices(loc_idx, pauli.as_u8())
        {
            xor_toggle_2(&mut triggered_tracked_paulis, tracked_pauli_idx);
        }

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
        triggered_tracked_paulis.sort_unstable();

        FaultMechanism::from_sorted_with_tracked_paulis(
            triggered_dets,
            triggered_obs,
            triggered_tracked_paulis,
        )
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

fn pauli_pair_for_weight(p1: usize, p2: usize) -> pecos_core::PauliString {
    let mut paulis = Vec::new();
    let pauli_from_index = |idx| match idx {
        0 => pecos_core::Pauli::I,
        1 => pecos_core::Pauli::X,
        2 => pecos_core::Pauli::Y,
        3 => pecos_core::Pauli::Z,
        _ => unreachable!("Pauli index must be 0-3"),
    };
    let pa1 = pauli_from_index(p1);
    let pa2 = pauli_from_index(p2);
    if pa1 != pecos_core::Pauli::I {
        paulis.push((pa1, pecos_core::QubitId::from(0usize)));
    }
    if pa2 != pecos_core::Pauli::I {
        paulis.push((pa2, pecos_core::QubitId::from(1usize)));
    }
    pecos_core::PauliString::with_phase_and_paulis(pecos_core::QuarterPhase::PlusOne, paulis)
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
    let id = extract_u32(
        json,
        &["\"id\"", "\"detector_id\""],
        "missing detector id",
        "detector id out of range",
    )?;

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
    let id = extract_u32(
        json,
        &["\"id\"", "\"observable_id\""],
        "missing observable id",
        "observable id out of range",
    )?;

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

fn extract_u32(
    json: &str,
    keys: &[&str],
    missing_message: &str,
    range_message: &str,
) -> Result<u32, DemBuilderError> {
    let value = keys
        .iter()
        .find_map(|key| extract_number(json, key))
        .ok_or_else(|| DemBuilderError::ParseError(missing_message.into()))?;
    u32::try_from(value).map_err(|_| DemBuilderError::ParseError(range_message.into()))
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
// Convenience: build DEM from circuit (free function to handle lifetimes)
// ============================================================================

/// Build a `DetectorErrorModel` from a `DagCircuit` and noise parameters.
///
/// Reads detector/DEM output definitions from circuit metadata attributes.
fn build_dem_from_circuit(
    circuit: &pecos_quantum::DagCircuit,
    p1: f64,
    p2: f64,
    p_meas: f64,
    p_prep: f64,
) -> DetectorErrorModel {
    use crate::fault_tolerance::influence_builder::InfluenceBuilder;
    use crate::fault_tolerance::propagator::DagFaultAnalyzer;
    use pecos_num::graph::Attribute;

    let mut influence_map = DagFaultAnalyzer::new(circuit).build_influence_map();
    let annotated_observable_records = observable_records_from_annotations(circuit, &influence_map);
    let annotation_map = InfluenceBuilder::new(circuit)
        .with_circuit_annotations(circuit)
        .build();
    influence_map.merge_dem_outputs_from(&annotation_map);

    // Extract metadata before building (to avoid borrow issues)
    let det_json = circuit.get_attr("detectors").and_then(|a| {
        if let Attribute::String(s) = a {
            Some(s.clone())
        } else {
            None
        }
    });
    let obs_json = circuit.get_attr("observables").and_then(|a| {
        if let Attribute::String(s) = a {
            Some(s.clone())
        } else {
            None
        }
    });
    let num_meas = circuit.get_attr("num_measurements").and_then(|a| {
        if let Attribute::String(s) = a {
            s.parse::<usize>().ok()
        } else {
            None
        }
    });

    let builder = DemBuilder::new(&influence_map).with_noise(p1, p2, p_meas, p_prep);

    let builder = if let Some(ref dj) = det_json {
        builder
            .with_detectors_json(dj)
            .unwrap_or_else(|_| DemBuilder::new(&influence_map).with_noise(p1, p2, p_meas, p_prep))
    } else {
        builder
    };

    let builder = if let Some(ref oj) = obs_json {
        builder
            .with_observables_json(oj)
            .unwrap_or_else(|_| DemBuilder::new(&influence_map).with_noise(p1, p2, p_meas, p_prep))
    } else if !annotated_observable_records.is_empty() {
        builder.with_observable_records(annotated_observable_records)
    } else {
        builder
    };

    let builder = if let Some(n) = num_meas {
        builder.with_num_measurements(n)
    } else {
        builder
    };

    builder.build()
}

fn observable_records_from_annotations(
    circuit: &pecos_quantum::DagCircuit,
    influence_map: &DagFaultInfluenceMap,
) -> Vec<Vec<i32>> {
    use pecos_quantum::AnnotationKind;

    let num_measurements = influence_map.measurements.len();
    if num_measurements == 0 {
        return Vec::new();
    }

    let mut node_to_meas_idx: BTreeMap<usize, usize> = BTreeMap::new();
    for (meas_idx, &(node, _qubit, _basis)) in influence_map.measurements.iter().enumerate() {
        node_to_meas_idx.entry(node).or_insert(meas_idx);
    }

    circuit
        .observables()
        .map(|ann| {
            if let AnnotationKind::Observable { measurement_nodes } = &ann.kind {
                measurement_nodes
                    .iter()
                    .filter_map(|node| node_to_meas_idx.get(node).copied())
                    .map(|meas_idx| {
                        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                        {
                            meas_idx as i32 - num_measurements as i32
                        }
                    })
                    .collect()
            } else {
                Vec::new()
            }
        })
        .collect()
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
    fn test_from_circuit_tracks_tracked_pauli() {
        use pecos_core::pauli::X;
        use pecos_quantum::DagCircuit;

        let mut circuit = DagCircuit::new();
        circuit.pz(&[0]);
        circuit.h(&[0]);
        circuit.tracked_pauli_labeled("x_check", X(0));

        let dem = DemBuilder::from_circuit(&circuit, 0.03, 0.0, 0.0, 0.0);

        assert_eq!(dem.num_dem_outputs(), 0);
        assert_eq!(dem.num_tracked_paulis(), 1);
        assert_eq!(dem.num_observables(), 0);
        assert_eq!(
            dem.tracked_paulis()[0].kind,
            Some(crate::fault_tolerance::DemOutputKind::TrackedPauli)
        );
        assert_eq!(dem.tracked_paulis()[0].label.as_deref(), Some("x_check"));
        assert_eq!(
            dem.tracked_paulis()[0]
                .pauli
                .as_ref()
                .unwrap()
                .to_sparse_str(),
            "+X0"
        );
        assert!(!dem.to_string().contains("logical_observable"));
        assert!(!dem.to_string().contains("TP0"));
        let pecos_text = dem.to_pecos_string();
        assert!(pecos_text.contains("TP0"));
        assert!(pecos_text.contains("pecos_tracked_pauli"));
    }

    #[test]
    fn test_tracked_pauli_and_observable_use_distinct_tracked_paulis() {
        use pecos_core::pauli::Z;
        use pecos_quantum::{Attribute, DagCircuit};

        let mut circuit = DagCircuit::new();
        circuit.pz(&[0]);
        circuit.tracked_pauli_labeled("z_check", Z(0));
        circuit.mz(&[0]);
        circuit.set_attr("num_measurements", Attribute::String("1".to_string()));
        circuit.set_attr(
            "observables",
            Attribute::String(r#"[{"id":0,"records":[-1]}]"#.to_string()),
        );

        let dem = DemBuilder::from_circuit(&circuit, 0.0, 0.0, 0.02, 0.03);

        assert_eq!(dem.num_dem_outputs(), 1);
        assert_eq!(dem.num_tracked_paulis(), 1);
        assert_eq!(dem.num_observables(), 1);
        assert_eq!(
            dem.dem_outputs()[0].kind,
            Some(crate::fault_tolerance::DemOutputKind::Observable)
        );
        assert_eq!(dem.tracked_paulis()[0].label.as_deref(), Some("z_check"));
        let dem_str = dem.to_string();
        assert!(dem_str.contains("logical_observable L0"));
        assert!(!dem_str.contains("logical_observable L1"));
        assert!(!dem_str.contains("TP0"));
        let pecos_text = dem.to_pecos_string();
        assert!(pecos_text.contains("TP0"));
        assert!(pecos_text.contains("pecos_tracked_pauli"));
        let summaries = dem.contribution_effect_summaries();
        assert!(
            summaries
                .iter()
                .any(|summary| summary.effect.dem_outputs.as_slice() == [0]),
            "observable should remain L0"
        );
        assert!(
            summaries
                .iter()
                .any(|summary| summary.effect.tracked_paulis.as_slice() == [0]),
            "tracked Pauli should remain TP0"
        );
    }

    #[test]
    fn test_tick_dag_tick_dem_keeps_detector_observable_and_tracked_pauli_distinct() {
        use pecos_core::pauli::X;
        use pecos_quantum::{DagCircuit, TickCircuit};

        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[0, 1]);
        circuit.tick().h(&[0]);
        circuit.tracked_pauli_labeled("tracked_x0", X(0));
        circuit.tick().mz(&[0, 1]);
        circuit.set_meta(
            "num_measurements",
            pecos_quantum::Attribute::String(circuit.num_measurements().to_string()),
        );
        circuit
            .add_detector_metadata(&[-2], None, Some("D0"), Some(0))
            .unwrap();
        circuit
            .add_observable_metadata(&[-1], Some(0), Some("L0"))
            .unwrap();
        let round_tripped = TickCircuit::from(&DagCircuit::from(&circuit));
        let dem = DemBuilder::from_tick_circuit(&round_tripped, 0.03, 0.0, 0.02, 0.0);

        assert_eq!(dem.num_detectors(), 1);
        assert_eq!(dem.num_observables(), 1);
        assert_eq!(dem.num_dem_outputs(), 1);
        assert_eq!(dem.dem_outputs()[0].id, 0);
        assert_eq!(dem.num_tracked_paulis(), 1);
        assert_eq!(dem.tracked_paulis()[0].id, 0);
        assert_eq!(dem.tracked_paulis()[0].label.as_deref(), Some("tracked_x0"));
        assert_eq!(
            dem.tracked_paulis()[0]
                .pauli
                .as_ref()
                .unwrap()
                .to_sparse_str(),
            "+X0"
        );

        let standard_text = dem.to_string();
        assert!(standard_text.contains("logical_observable L0"));
        assert!(!standard_text.contains("logical_observable L1"));
        assert!(!standard_text.contains("pecos_tracked_pauli"));

        let pecos_text = dem.to_pecos_string();
        assert!(pecos_text.contains("pecos_observable"));
        assert!(pecos_text.contains("pecos_tracked_pauli"));

        let summaries = dem.contribution_effect_summaries();
        assert!(
            summaries
                .iter()
                .any(|summary| summary.effect.detectors.as_slice() == [0]),
            "detector effects should survive Tick -> DAG -> Tick"
        );
        assert!(
            summaries
                .iter()
                .any(|summary| summary.effect.dem_outputs.as_slice() == [0]),
            "observable effects should remain in L0"
        );
    }

    #[test]
    fn test_circuit_observable_annotation_is_not_double_counted() {
        use pecos_quantum::DagCircuit;

        let mut circuit = DagCircuit::new();
        circuit.pz(&[0]);
        let meas = circuit.mz(&[0]);
        circuit.observable_labeled("obs0", &[meas[0]]);

        let dem = DemBuilder::from_circuit(&circuit, 0.0, 0.0, 1.0, 0.0);

        assert_eq!(dem.num_dem_outputs(), 1);
        assert_eq!(dem.num_observables(), 1);
        assert_eq!(dem.dem_outputs().len(), 1);
        assert_eq!(dem.dem_outputs()[0].id, 0);
        assert_eq!(dem.dem_outputs()[0].records.as_slice(), &[-1]);
        assert_eq!(dem.dem_outputs()[0].label.as_deref(), Some("obs0"));

        let logical_observable_lines = dem
            .to_string()
            .lines()
            .filter(|line| *line == "logical_observable L0")
            .count();
        assert_eq!(logical_observable_lines, 1);

        let summaries = dem.contribution_effect_summaries();
        assert!(
            summaries
                .iter()
                .any(|summary| summary.effect.dem_outputs.as_slice() == [0]),
            "measurement fault should flip observable L0 once, not cancel"
        );
    }

    #[test]
    fn test_from_tick_circuit_tracks_face_gate_fault_sources() {
        use pecos_core::QubitId;
        use pecos_quantum::{Attribute, TickCircuit};

        for gate_type in [GateType::F, GateType::Fdg] {
            let mut circuit = TickCircuit::new();
            circuit.tick().pz(&[QubitId(0)]);
            match gate_type {
                GateType::F => {
                    circuit.tick().f(&[QubitId(0)]);
                }
                GateType::Fdg => {
                    circuit.tick().fdg(&[QubitId(0)]);
                }
                _ => unreachable!(),
            }
            circuit.tick().mz(&[QubitId(0)]);
            circuit.set_meta("num_measurements", Attribute::String("1".to_string()));
            circuit.set_meta(
                "detectors",
                Attribute::String(r#"[{"id":0,"records":[-1]}]"#.to_string()),
            );
            circuit.set_meta("observables", Attribute::String("[]".to_string()));

            let dem = DemBuilder::from_tick_circuit(&circuit, 0.03, 0.0, 0.0, 0.0);
            let contributions = dem.contributions_for_effect(&[0], &[]);

            assert!(
                contributions
                    .iter()
                    .any(|contribution| contribution.source_gate_types.contains(&gate_type)),
                "DEM should include a tracked {gate_type:?} fault source"
            );
        }
    }

    #[test]
    fn test_fault_catalog_and_dem_cover_standard_clifford_gate_sources() {
        use crate::fault_tolerance::fault_sampler::{
            FaultCatalog, StochasticNoiseParams, build_fault_catalog,
        };
        use pecos_core::QubitId;
        use pecos_quantum::{Attribute, TickCircuit};
        use std::collections::BTreeMap;

        fn set_meta(circuit: &mut TickCircuit, num_measurements: usize, detectors: &str) {
            circuit.set_meta(
                "num_measurements",
                Attribute::String(num_measurements.to_string()),
            );
            circuit.set_meta("detectors", Attribute::String(detectors.to_string()));
            circuit.set_meta("observables", Attribute::String("[]".to_string()));
        }

        fn add_1q_gate(circuit: &mut TickCircuit, gate_type: GateType) {
            match gate_type {
                GateType::X => {
                    circuit.tick().x(&[QubitId(0)]);
                }
                GateType::Y => {
                    circuit.tick().y(&[QubitId(0)]);
                }
                GateType::Z => {
                    circuit.tick().z(&[QubitId(0)]);
                }
                GateType::H => {
                    circuit.tick().h(&[QubitId(0)]);
                }
                GateType::F => {
                    circuit.tick().f(&[QubitId(0)]);
                }
                GateType::Fdg => {
                    circuit.tick().fdg(&[QubitId(0)]);
                }
                GateType::SX => {
                    circuit.tick().sx(&[QubitId(0)]);
                }
                GateType::SXdg => {
                    circuit.tick().sxdg(&[QubitId(0)]);
                }
                GateType::SY => {
                    circuit.tick().sy(&[QubitId(0)]);
                }
                GateType::SYdg => {
                    circuit.tick().sydg(&[QubitId(0)]);
                }
                GateType::SZ => {
                    circuit.tick().sz(&[QubitId(0)]);
                }
                GateType::SZdg => {
                    circuit.tick().szdg(&[QubitId(0)]);
                }
                _ => panic!("not a 1q standard Clifford gate: {gate_type:?}"),
            }
        }

        fn add_2q_gate(circuit: &mut TickCircuit, gate_type: GateType) {
            let pair = &[(QubitId(0), QubitId(1))];
            match gate_type {
                GateType::CX => {
                    circuit.tick().cx(pair);
                }
                GateType::CY => {
                    circuit.tick().cy(pair);
                }
                GateType::CZ => {
                    circuit.tick().cz(pair);
                }
                GateType::SXX => {
                    circuit.tick().sxx(pair);
                }
                GateType::SXXdg => {
                    circuit.tick().sxxdg(pair);
                }
                GateType::SYY => {
                    circuit.tick().syy(pair);
                }
                GateType::SYYdg => {
                    circuit.tick().syydg(pair);
                }
                GateType::SZZ => {
                    circuit.tick().szz(pair);
                }
                GateType::SZZdg => {
                    circuit.tick().szzdg(pair);
                }
                GateType::SWAP => {
                    circuit.tick().swap(pair);
                }
                _ => panic!("not a 2q standard Clifford gate: {gate_type:?}"),
            }
        }

        fn dem_has_source(dem: &DetectorErrorModel, gate_type: GateType) -> bool {
            dem.contribution_render_records()
                .iter()
                .any(|record| record.contribution.source_gate_types.contains(&gate_type))
        }

        fn catalog_dem_channel_effect_probabilities(
            catalog: &FaultCatalog,
        ) -> BTreeMap<(Vec<u32>, Vec<u32>), f64> {
            let mut by_effect = BTreeMap::new();
            for location in &catalog.locations {
                if location.num_alternatives == 0 {
                    continue;
                }
                let num_alternatives = f64::from(
                    u32::try_from(location.num_alternatives)
                        .expect("fault alternative count fits in u32"),
                );
                let per_channel_probability =
                    1.0 - location.no_fault_probability.powf(1.0 / num_alternatives);
                for fault in &location.faults {
                    if fault.affected_detectors.is_empty() && fault.affected_observables.is_empty()
                    {
                        continue;
                    }
                    let detectors: Vec<u32> = fault
                        .affected_detectors
                        .iter()
                        .map(|&det| u32::try_from(det).unwrap())
                        .collect();
                    let observables: Vec<u32> = fault
                        .affected_observables
                        .iter()
                        .map(|&obs| u32::try_from(obs).unwrap())
                        .collect();
                    *by_effect.entry((detectors, observables)).or_insert(0.0) +=
                        per_channel_probability;
                }
            }
            by_effect
        }

        fn dem_effect_probabilities(
            dem: &DetectorErrorModel,
        ) -> BTreeMap<(Vec<u32>, Vec<u32>), f64> {
            dem.contribution_effect_summaries()
                .into_iter()
                .filter(|summary| {
                    !summary.effect.detectors.is_empty() || !summary.effect.dem_outputs.is_empty()
                })
                .map(|summary| {
                    (
                        (
                            summary.effect.detectors.into_iter().collect(),
                            summary.effect.dem_outputs.into_iter().collect(),
                        ),
                        summary.total_probability,
                    )
                })
                .collect()
        }

        fn assert_catalog_dem_probabilities_match(
            catalog: &FaultCatalog,
            dem: &DetectorErrorModel,
            gate_type: GateType,
        ) {
            let catalog_probs = catalog_dem_channel_effect_probabilities(catalog);
            let dem_probs = dem_effect_probabilities(dem);
            assert_eq!(
                catalog_probs.keys().collect::<Vec<_>>(),
                dem_probs.keys().collect::<Vec<_>>(),
                "{gate_type:?} should produce the same non-empty effects in the fault catalog and DEM"
            );
            for (effect, catalog_probability) in catalog_probs {
                let dem_probability = dem_probs[&effect];
                assert!(
                    (catalog_probability - dem_probability).abs() < 1e-12,
                    "{gate_type:?} effect {effect:?}: catalog probability {catalog_probability} != DEM probability {dem_probability}"
                );
            }
        }

        for gate_type in [
            GateType::X,
            GateType::Y,
            GateType::Z,
            GateType::H,
            GateType::F,
            GateType::Fdg,
            GateType::SX,
            GateType::SXdg,
            GateType::SY,
            GateType::SYdg,
            GateType::SZ,
            GateType::SZdg,
        ] {
            let mut circuit = TickCircuit::new();
            circuit.tick().pz(&[QubitId(0)]);
            add_1q_gate(&mut circuit, gate_type);
            circuit.tick().mz(&[QubitId(0)]);
            set_meta(&mut circuit, 1, r#"[{"id":0,"records":[-1]}]"#);

            let catalog = build_fault_catalog(
                &circuit,
                &StochasticNoiseParams {
                    p1: 0.03,
                    p2: 0.0,
                    p_meas: 0.0,
                    p_prep: 0.0,
                },
            )
            .unwrap();
            let locations: Vec<_> = catalog
                .locations
                .iter()
                .filter(|location| location.gate_type == gate_type)
                .collect();
            assert_eq!(locations.len(), 1, "{gate_type:?}");
            assert_eq!(locations[0].faults.len(), 3, "{gate_type:?}");

            let dem = DemBuilder::from_tick_circuit(&circuit, 0.03, 0.0, 0.0, 0.0);
            assert!(
                dem_has_source(&dem, gate_type),
                "DEM should track a source contribution for {gate_type:?}"
            );
            assert_catalog_dem_probabilities_match(&catalog, &dem, gate_type);
        }

        for gate_type in [
            GateType::CX,
            GateType::CY,
            GateType::CZ,
            GateType::SXX,
            GateType::SXXdg,
            GateType::SYY,
            GateType::SYYdg,
            GateType::SZZ,
            GateType::SZZdg,
            GateType::SWAP,
        ] {
            let mut circuit = TickCircuit::new();
            circuit.tick().pz(&[QubitId(0), QubitId(1)]);
            add_2q_gate(&mut circuit, gate_type);
            circuit.tick().mz(&[QubitId(0), QubitId(1)]);
            set_meta(
                &mut circuit,
                2,
                r#"[{"id":0,"records":[-2]},{"id":1,"records":[-1]}]"#,
            );

            let catalog = build_fault_catalog(
                &circuit,
                &StochasticNoiseParams {
                    p1: 0.0,
                    p2: 0.15,
                    p_meas: 0.0,
                    p_prep: 0.0,
                },
            )
            .unwrap();
            let locations: Vec<_> = catalog
                .locations
                .iter()
                .filter(|location| location.gate_type == gate_type)
                .collect();
            assert_eq!(locations.len(), 1, "{gate_type:?}");
            assert_eq!(locations[0].faults.len(), 15, "{gate_type:?}");

            let dem = DemBuilder::from_tick_circuit(&circuit, 0.0, 0.15, 0.0, 0.0);
            assert!(
                dem_has_source(&dem, gate_type),
                "DEM should track a source contribution for {gate_type:?}"
            );
            assert_catalog_dem_probabilities_match(&catalog, &dem, gate_type);
        }
    }

    #[test]
    fn test_parse_detectors_json() {
        let json = r#"[
            {"id": 0, "coords": [0.0, 0.0, 0.0], "records": [-1, -5]},
            {"detector_id": 1, "coords": [1.0, 0.0, 0.0], "records": [-2]}
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
        let json = r#"[{"observable_id": 0, "records": [-1, -3, -5]}]"#;

        let observables = parse_observables_json(json).unwrap();

        assert_eq!(observables.len(), 1);
        assert_eq!(observables[0].id, 0);
        assert_eq!(observables[0].records, vec![-1, -3, -5]);
    }

    #[test]
    fn test_dem_builder_accepts_observables_json_alias() {
        let influence_map = DagFaultInfluenceMap::with_capacity(0);
        let dem = DemBuilder::new(&influence_map)
            .with_observables_json(r#"[{"id": 0, "records": [-1, -3]}]"#)
            .unwrap()
            .build();

        assert_eq!(dem.num_dem_outputs(), 1);
        assert_eq!(dem.num_observables(), 1);
        assert_eq!(dem.num_tracked_paulis(), 0);
        assert_eq!(dem.dem_outputs()[0].records.as_slice(), &[-1, -3]);
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
