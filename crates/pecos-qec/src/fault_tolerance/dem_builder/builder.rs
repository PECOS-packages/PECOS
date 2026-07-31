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
    DemOutput, DetectorDef, DetectorErrorModel, DirectSourceComponents, DirectSourceFamily,
    FaultMechanism, MeasurementCrosstalkDemMode, NoiseConfig, PerGateTypeNoise,
    ReplacementBranchApproximation, SourceMetadata, record_offset_to_absolute_index,
};
use crate::fault_tolerance::propagator::dag::DagSpacetimeLocation;
use crate::fault_tolerance::propagator::{DagFaultInfluenceMap, Direction, Pauli, apply_gate};
use pecos_core::BitSet;
use pecos_core::gate_type::GateType;
use pecos_simulators::{
    PauliProp, SymbolicMeasurementResult, SymbolicSparseStab,
    symbolic_sparse_stab::MeasurementHistory,
};
use smallvec::SmallVec;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

// ============================================================================
// JSON Parsing Types
// ============================================================================

/// Parsed detector from JSON metadata.
#[derive(Debug, Clone)]
struct ParsedDetector {
    id: u32,
    coords: Option<[f64; 3]>,
    records: Vec<i32>,
    meas_ids: Vec<usize>,
}

/// Parsed observable from JSON metadata.
#[derive(Debug, Clone)]
struct ParsedObservable {
    id: u32,
    records: Vec<i32>,
    meas_ids: Vec<usize>,
    /// Human-readable name from the metadata JSON's `label` field.
    ///
    /// The metadata format has always carried this and callers already write
    /// it, but it used to be parsed and dropped, leaving circuit annotations as
    /// the only way to get a label onto an observable.
    label: Option<String>,
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
    /// Optional circuit context for future exact replacement-branch replay.
    exact_branch_context: Option<ExactBranchReplayContext<'a>>,
    /// Ideal symbolic measurement history shared by exact branch replays.
    exact_ideal_history_cache: RefCell<Option<Rc<MeasurementHistory>>>,
    /// Per-gate cache for exact replacement-branch replay effects.
    exact_branch_cache: RefCell<BTreeMap<usize, ExactBranchReplayAnalysis>>,
}

#[derive(Debug, Clone, Copy)]
struct ExactBranchReplayContext<'a> {
    circuit: &'a pecos_quantum::DagCircuit,
}

#[derive(Debug, Clone)]
struct ExactBranchReplayAnalysis {
    base_effect: FaultMechanism,
    branch_effects: [[FaultMechanism; 4]; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExactBranchReplayRequest {
    gate_node: usize,
    gate_type: GateType,
    loc_indices: [usize; 2],
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MeasurementParityExpression {
    dependencies: BitSet,
    flip: bool,
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
    ///
    /// # Panics
    ///
    /// Panics if the circuit's detector/observable metadata is malformed (use
    /// [`Self::try_from_circuit`] to handle that as an error instead).
    #[must_use]
    pub fn from_circuit(
        circuit: &pecos_quantum::DagCircuit,
        p1: f64,
        p2: f64,
        p_meas: f64,
        p_prep: f64,
    ) -> DetectorErrorModel {
        Self::try_from_circuit(circuit, p1, p2, p_meas, p_prep)
            .unwrap_or_else(|err| panic!("invalid DEM metadata on circuit: {err}"))
    }

    /// Try to build a `DetectorErrorModel` directly from a `DagCircuit` and noise.
    ///
    /// Reads detector/DEM output definitions from circuit metadata and returns
    /// parser errors instead of dropping malformed metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if detector or observable metadata is malformed.
    ///
    /// # Panics
    ///
    /// Malformed *metadata* is reported as an error, but a circuit whose
    /// metadata and annotations **contradict each other** is not a parse
    /// failure and is not reported that way: two definitions of one observable
    /// disagreeing about the label or the Pauli panic inside
    /// [`DetectorErrorModel::add_observable`] rather than returning here. Both
    /// sources must agree, or only one should supply the field.
    pub fn try_from_circuit(
        circuit: &pecos_quantum::DagCircuit,
        p1: f64,
        p2: f64,
        p_meas: f64,
        p_prep: f64,
    ) -> Result<DetectorErrorModel, DemBuilderError> {
        build_dem_from_circuit(circuit, NoiseConfig::new(p1, p2, p_meas, p_prep))
    }

    /// Try to build a `DetectorErrorModel` directly from a `DagCircuit` and
    /// full noise configuration.
    ///
    /// Reads detector/DEM output definitions from circuit metadata and returns
    /// parser errors instead of dropping malformed metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if detector or observable metadata is malformed.
    ///
    /// # Panics
    ///
    /// Malformed *metadata* is reported as an error, but a circuit whose
    /// metadata and annotations **contradict each other** is not a parse
    /// failure and is not reported that way: two definitions of one observable
    /// disagreeing about the label or the Pauli panic inside
    /// [`DetectorErrorModel::add_observable`] rather than returning here. Both
    /// sources must agree, or only one should supply the field.
    pub fn try_from_circuit_with_noise_config(
        circuit: &pecos_quantum::DagCircuit,
        noise: NoiseConfig,
    ) -> Result<DetectorErrorModel, DemBuilderError> {
        build_dem_from_circuit(circuit, noise)
    }

    /// Build a `DetectorErrorModel` from a `TickCircuit` and noise.
    ///
    /// Converts to `DagCircuit` internally.
    ///
    /// # Panics
    ///
    /// Panics if the circuit's detector/observable metadata is malformed (use
    /// [`Self::try_from_tick_circuit`] to handle that as an error instead).
    #[must_use]
    pub fn from_tick_circuit(
        circuit: &pecos_quantum::TickCircuit,
        p1: f64,
        p2: f64,
        p_meas: f64,
        p_prep: f64,
    ) -> DetectorErrorModel {
        Self::try_from_tick_circuit(circuit, p1, p2, p_meas, p_prep)
            .unwrap_or_else(|err| panic!("invalid DEM metadata on circuit: {err}"))
    }

    /// Try to build a `DetectorErrorModel` from a `TickCircuit` and noise.
    ///
    /// Converts to `DagCircuit` internally and returns parser errors instead
    /// of dropping malformed metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if detector or observable metadata is malformed.
    ///
    /// # Panics
    ///
    /// Malformed *metadata* is reported as an error, but a circuit whose
    /// metadata and annotations **contradict each other** is not a parse
    /// failure and is not reported that way: two definitions of one observable
    /// disagreeing about the label or the Pauli panic inside
    /// [`DetectorErrorModel::add_observable`] rather than returning here. Both
    /// sources must agree, or only one should supply the field.
    pub fn try_from_tick_circuit(
        circuit: &pecos_quantum::TickCircuit,
        p1: f64,
        p2: f64,
        p_meas: f64,
        p_prep: f64,
    ) -> Result<DetectorErrorModel, DemBuilderError> {
        let dag = pecos_quantum::DagCircuit::try_from(circuit)
            .map_err(|err| DemBuilderError::ConfigurationError(err.to_string()))?;
        build_dem_from_circuit(&dag, NoiseConfig::new(p1, p2, p_meas, p_prep))
    }

    /// Try to build a `DetectorErrorModel` from a `TickCircuit` and full noise
    /// configuration.
    ///
    /// Converts to `DagCircuit` internally and returns parser errors instead
    /// of dropping malformed metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if detector or observable metadata is malformed.
    ///
    /// # Panics
    ///
    /// Malformed *metadata* is reported as an error, but a circuit whose
    /// metadata and annotations **contradict each other** is not a parse
    /// failure and is not reported that way: two definitions of one observable
    /// disagreeing about the label or the Pauli panic inside
    /// [`DetectorErrorModel::add_observable`] rather than returning here. Both
    /// sources must agree, or only one should supply the field.
    pub fn try_from_tick_circuit_with_noise_config(
        circuit: &pecos_quantum::TickCircuit,
        noise: NoiseConfig,
    ) -> Result<DetectorErrorModel, DemBuilderError> {
        let dag = pecos_quantum::DagCircuit::try_from(circuit)
            .map_err(|err| DemBuilderError::ConfigurationError(err.to_string()))?;
        build_dem_from_circuit(&dag, noise)
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
            exact_branch_context: None,
            exact_ideal_history_cache: RefCell::new(None),
            exact_branch_cache: RefCell::new(BTreeMap::new()),
        }
    }

    fn clear_exact_branch_cache(&mut self) {
        self.exact_branch_cache.get_mut().clear();
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

    #[must_use]
    pub fn with_exact_branch_replay_context(
        mut self,
        circuit: &'a pecos_quantum::DagCircuit,
    ) -> Self {
        self.exact_branch_context = Some(ExactBranchReplayContext { circuit });
        self.exact_ideal_history_cache.get_mut().take();
        self.clear_exact_branch_cache();
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
        let p1_total = self.noise.p1_rate_for_gate(loc.gate_type);
        if let Some(weights) = &self.noise.p1_weights {
            use pecos_core::pauli::{X, Y, Z};
            return [
                p1_total * weights.weight_for(&X(0)),
                p1_total * weights.weight_for(&Y(0)),
                p1_total * weights.weight_for(&Z(0)),
            ];
        }
        let per = per_channel_probability(p1_total, 3);
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
                let duration = loc.idle_duration.max(0.0);
                let probs = pg.base.idle_pauli_probs(duration);
                return [probs.px, probs.py, probs.pz];
            }
            return [0.0; 3];
        }

        if self.noise.uses_dedicated_idle_noise() {
            let duration = loc.idle_duration.max(0.0);
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
                let pauli = pauli_pair_for_weight(p1, p2);
                let p2_total = self.noise.p2_rate_for_gate(loc1.gate_type);
                let weight = if self.noise.p2_replacement_approximation
                    == ReplacementBranchApproximation::BranchImpact
                    || self.noise.p2_replacement_approximation
                        == ReplacementBranchApproximation::ExactBranchReplay
                {
                    weights.post_gate_two_qubit_weight_for(&pauli)
                } else {
                    weights.two_qubit_weight_for(
                        loc1.gate_type,
                        &pauli,
                        self.noise.p2_replacement_approximation,
                    )
                };
                p2_total * weight
            });
        }
        [per_channel_probability(self.noise.p2_rate_for_gate(loc1.gate_type), 15); 15]
    }

    /// Sets the number of measurements (used for record offset calculation).
    #[must_use]
    pub fn with_num_measurements(mut self, num: usize) -> Self {
        self.num_measurements = num;
        self.clear_exact_branch_cache();
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
        self.clear_exact_branch_cache();
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
        self.clear_exact_branch_cache();
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
        self.clear_exact_branch_cache();
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
                meas_ids: Vec::new(),
                label: None,
            })
            .collect();
        self.clear_exact_branch_cache();
        self
    }

    /// Resolves a JSON `meas_id` to a circuit measurement-record index.
    ///
    /// When the circuit carries stable `MeasId`s (the traced
    /// `from_guppy`/`from_circuit` path), `meas_id` is interpreted as that
    /// **stable stamped id** and looked up in `influence_map.meas_ids` -- so a
    /// non-sequential traced id (e.g. the QIS result slot) resolves correctly
    /// regardless of compilation reordering. When no stable ids are present
    /// (the decoupled/raw builder with an empty influence map), `meas_id` is a
    /// positional measurement index (the legacy escape hatch). Returns the
    /// `0..num_measurements` record index, or `None` if the id is absent.
    fn resolve_meas_id_to_tc_index(&self, meas_id: usize) -> Option<usize> {
        if self.influence_map.meas_ids.is_empty() {
            return (meas_id < self.num_measurements).then_some(meas_id);
        }
        self.influence_map
            .meas_ids
            .iter()
            .position(|mid| mid.0 == meas_id)
    }

    fn meas_id_to_record_offset(&self, meas_id: usize) -> Option<i32> {
        let index = self.resolve_meas_id_to_tc_index(meas_id)?;
        let measurement = i64::try_from(index).ok()?;
        let total = i64::try_from(self.num_measurements).ok()?;
        i32::try_from(measurement - total).ok()
    }

    /// Fail loud if any detector/observable references a measurement that does
    /// not exist, instead of silently dropping it and weakening the DEM.
    ///
    /// `records` and `meas_ids` are alternative ways to name the *same*
    /// measurements (the parser allows neither both-empty). Each used
    /// reference must resolve in range. When an entry carries **both**, they
    /// must be redundant -- `meas_ids` must resolve to exactly the `records`
    /// set -- otherwise the DEM the builder produces (which consumes
    /// `records`) would silently differ from what `meas_ids` asked for. The
    /// surface `logical_circuit` path emits both redundantly; a non-redundant
    /// pair is a caller error and fails loud here.
    ///
    /// # Errors
    /// Returns [`DemBuilderError::ParseError`] if a used record offset is out
    /// of range, a used `meas_id` is absent, or a both-present entry's
    /// `records` and `meas_ids` disagree.
    fn validate_metadata_refs(&self) -> Result<(), DemBuilderError> {
        let check = |kind: &str, id: u32, records: &[i32], meas_ids: &[usize]| {
            for &rec in records {
                if record_offset_to_absolute_index(self.num_measurements, rec).is_none() {
                    return Err(DemBuilderError::ParseError(format!(
                        "{kind} {id} references record offset {rec}, which \
                         is out of range for a circuit with {} \
                         measurement(s)",
                        self.num_measurements
                    )));
                }
            }
            let mut resolved_offsets = Vec::with_capacity(meas_ids.len());
            for &mid in meas_ids {
                let offset = self.meas_id_to_record_offset(mid).ok_or_else(|| {
                    DemBuilderError::ParseError(format!(
                        "{kind} {id} references meas_id {mid}, which is not \
                         present in the circuit's {} measurement(s)",
                        self.num_measurements
                    ))
                })?;
                resolved_offsets.push(offset);
            }
            if !records.is_empty() && !meas_ids.is_empty() {
                let mut a = records.to_vec();
                let mut b = resolved_offsets;
                a.sort_unstable();
                b.sort_unstable();
                if a != b {
                    return Err(DemBuilderError::ParseError(format!(
                        "{kind} {id} has both 'records' and 'meas_ids' but \
                         they reference different measurements (records map \
                         to offsets {a:?}, meas_ids resolve to {b:?}); they \
                         are alternatives, not additive -- the builder would \
                         consume only 'records' and silently drop the rest"
                    )));
                }
            }
            Ok(())
        };
        for d in &self.detectors {
            check("Detector", d.id, &d.records, &d.meas_ids)?;
        }
        for o in &self.observables {
            check("Observable", o.id, &o.records, &o.meas_ids)?;
        }
        Ok(())
    }

    fn effective_record_offsets(&self, records: &[i32], meas_ids: &[usize]) -> Vec<i32> {
        if !records.is_empty() {
            return records.to_vec();
        }
        meas_ids
            .iter()
            .filter_map(|&meas_id| self.meas_id_to_record_offset(meas_id))
            .collect()
    }

    fn measurement_indices_from_refs(
        &self,
        records: &[i32],
        meas_ids: &[usize],
    ) -> Result<Vec<usize>, DemBuilderError> {
        if !records.is_empty() {
            return records
                .iter()
                .map(|&rec| {
                    record_offset_to_absolute_index(self.num_measurements, rec).ok_or_else(|| {
                        DemBuilderError::ParseError(format!(
                            "record offset {rec} is out of range for a circuit with {} measurement(s)",
                            self.num_measurements
                        ))
                    })
                })
                .collect();
        }

        meas_ids
            .iter()
            .map(|&meas_id| {
                self.resolve_meas_id_to_tc_index(meas_id).ok_or_else(|| {
                    DemBuilderError::ParseError(format!(
                        "meas_id {meas_id} is not present in the circuit's {} measurement(s)",
                        self.num_measurements
                    ))
                })
            })
            .collect()
    }

    /// Validates metadata refs, then builds the Detector Error Model.
    ///
    /// This is the fail-loud entry point. Every path that ingests
    /// detector/observable metadata derived from a circuit (the
    /// `from_circuit` family, [`DemSampler::from_circuit`], and the public
    /// Python `DemBuilder.build`) must go through here so an out-of-range
    /// record offset or `meas_id` is rejected rather than silently dropped.
    ///
    /// [`Self::build`] is the infallible counterpart, kept for the raw,
    /// decoupled construction case (e.g. an empty influence map where record
    /// offsets are opaque DEM coordinates) and so existing callers do not
    /// change behavior.
    ///
    /// Rejects a `num_measurements` that disagrees with a non-empty influence
    /// map.
    ///
    /// When the builder is fed a real circuit (the influence map has
    /// measurements), record offsets and `meas_id`s are defined against that
    /// circuit's actual measurement record. A caller-supplied
    /// `with_num_measurements` that differs would let out-of-range refs pass
    /// [`Self::validate_metadata_refs`] and silently misbind, so it is an
    /// error. An empty influence map keeps the escape hatch: the count is then
    /// purely declarative and record offsets are opaque pass-through DEM
    /// coordinates.
    fn validate_measurement_count(&self) -> Result<(), DemBuilderError> {
        let actual = self.influence_map.measurements.len();
        if actual != 0 && self.num_measurements != actual {
            return Err(DemBuilderError::ParseError(format!(
                "num_measurements={} disagrees with the {actual} measurement(s) \
                 the circuit performs; the declared count must match so \
                 detector/observable record offsets resolve correctly",
                self.num_measurements
            )));
        }
        // Internal-consistency guard: stable MeasIds must be unique. A
        // duplicate would make stamped-id resolution bind to the wrong
        // measurement; it indicates a trace/replay bug, not bad caller input.
        let mut seen = std::collections::HashSet::with_capacity(self.influence_map.meas_ids.len());
        for mid in &self.influence_map.meas_ids {
            if !seen.insert(mid.0) {
                return Err(DemBuilderError::ParseError(format!(
                    "duplicate stable MeasId {} in the traced circuit; each \
                     measurement must have a unique stamped id",
                    mid.0
                )));
            }
        }
        Ok(())
    }

    /// # Errors
    ///
    /// Returns [`DemBuilderError::ParseError`] if `num_measurements` disagrees
    /// with a non-empty influence map, a used record offset is out of range,
    /// a used `meas_id` is not present in the circuit (resolved against the
    /// stable stamped ids when available, else positionally), or a
    /// both-present entry's `records` and `meas_ids` are not redundant.
    pub fn try_build(&self) -> Result<DetectorErrorModel, DemBuilderError> {
        self.validate_measurement_count()?;
        self.validate_metadata_refs()?;
        self.validate_replacement_branch_approximation()?;
        self.validate_measurement_crosstalk_dem_mode()?;
        Ok(self.build())
    }

    /// Builds the Detector Error Model with source tracking.
    ///
    /// This performs fault propagation analysis and tracks error sources (X/Z vs Y)
    /// through the pipeline, enabling accurate direct/decomposed form splitting.
    ///
    /// Use `dem.to_string()` or `dem.to_string_decomposed()` for output.
    ///
    /// This does **not** validate metadata refs; callers ingesting
    /// circuit-derived metadata must use [`Self::try_build`] instead.
    /// # Panics
    ///
    /// Panics if the configured replacement-branch approximation is invalid;
    /// validity is established by construction-time validation.
    #[must_use]
    pub fn build(&self) -> DetectorErrorModel {
        self.validate_replacement_branch_approximation()
            .expect("invalid DEM replacement branch approximation");
        self.validate_measurement_crosstalk_dem_mode()
            .expect("invalid DEM measurement crosstalk configuration");
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
            let records = self.effective_record_offsets(&det.records, &det.meas_ids);
            def = def.with_records(records.iter().copied());
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
            let records = self.effective_record_offsets(&obs.records, &obs.meas_ids);
            let mut def = DemOutput::new(obs.id).with_records(records.iter().copied());
            if let Some(label) = &obs.label {
                def = def.with_label(label.clone());
            }
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

    fn validate_replacement_branch_approximation(&self) -> Result<(), DemBuilderError> {
        let has_replacement_branches = self
            .noise
            .p2_weights
            .as_ref()
            .is_some_and(super::types::PauliWeights::has_replacement_entries);
        if self.noise.p2_replacement_approximation
            == ReplacementBranchApproximation::ExactBranchReplay
            && has_replacement_branches
        {
            if let Some(context) = self.exact_branch_context {
                let requests =
                    context.replacement_branch_requests(&self.influence_map.locations)?;
                let weights = self
                    .noise
                    .p2_weights
                    .as_ref()
                    .expect("replacement entries exist");
                for request in &requests {
                    for (replacement_pauli, _weight) in weights.replacement_entries() {
                        let label = two_qubit_label_for_replay(replacement_pauli)?;
                        let _replacement_effect =
                            self.exact_replacement_branch_effect(context, *request, &label)?;
                    }
                }
                return Ok(());
            }
            return Err(DemBuilderError::ConfigurationError(
                "exact_branch_replay for starred p2 replacement branches requires a circuit-aware exact branch provider; use branch_impact or pauli_twirl_omitted_gate for the current Pauli-projected approximations"
                    .to_string(),
            ));
        }
        Ok(())
    }

    fn validate_measurement_crosstalk_dem_mode(&self) -> Result<(), DemBuilderError> {
        if self.noise.measurement_crosstalk_dem_mode == MeasurementCrosstalkDemMode::Omitted {
            return Ok(());
        }

        let has_local_payloads = self
            .influence_map
            .locations
            .iter()
            .any(|loc| !loc.before && loc.gate_type == GateType::MeasCrosstalkLocalPayload);
        let has_global_payloads = self
            .influence_map
            .locations
            .iter()
            .any(|loc| !loc.before && loc.gate_type == GateType::MeasCrosstalkGlobalPayload);

        if !self.noise.p_meas_crosstalk_model.is_valid() {
            return Err(DemBuilderError::ConfigurationError(
                "measurement crosstalk transition probabilities must be finite, non-negative, and have each hidden-outcome row sum <= 1"
                    .to_string(),
            ));
        }

        if self.noise.p_meas_crosstalk_global > 0.0 && !has_global_payloads {
            return Err(DemBuilderError::ConfigurationError(
                "exact deterministic measurement crosstalk DEM replay requested a positive global rate, but the influence map contains no MeasCrosstalkGlobalPayload locations"
                    .to_string(),
            ));
        }

        if self.noise.p_meas_crosstalk_local > 0.0 && !has_local_payloads {
            return Err(DemBuilderError::ConfigurationError(
                "exact deterministic measurement crosstalk DEM replay requested a positive local rate, but the influence map contains no MeasCrosstalkLocalPayload locations"
                    .to_string(),
            ));
        }

        if self.noise.p_meas_crosstalk_local <= 0.0 && self.noise.p_meas_crosstalk_global <= 0.0 {
            return Ok(());
        }

        if self.noise.p_meas_crosstalk_model.has_leakage()
            && !matches!(
                self.noise.measurement_crosstalk_dem_mode,
                MeasurementCrosstalkDemMode::ExactDeterministicLeakageAsDepolarizing
                    | MeasurementCrosstalkDemMode::AveragedHiddenLeakageAsDepolarizing
            )
        {
            return Err(DemBuilderError::ConfigurationError(
                "exact deterministic measurement crosstalk DEM replay does not yet support leakage transitions"
                    .to_string(),
            ));
        }

        let needs_circuit_context =
            self.noise.measurement_crosstalk_dem_mode != MeasurementCrosstalkDemMode::Omitted;
        let Some(context) = self.exact_branch_context else {
            if needs_circuit_context {
                return Err(DemBuilderError::ConfigurationError(
                    "measurement crosstalk DEM replay requires a circuit-aware builder context"
                        .to_string(),
                ));
            }
            return Ok(());
        };

        let requires_deterministic_hidden_measurements = matches!(
            self.noise.measurement_crosstalk_dem_mode,
            MeasurementCrosstalkDemMode::ExactDeterministic
                | MeasurementCrosstalkDemMode::ExactDeterministicLeakageAsDepolarizing
        );
        if !requires_deterministic_hidden_measurements {
            return Ok(());
        }

        for (loc_idx, loc) in self.influence_map.locations.iter().enumerate() {
            if loc.before
                || !matches!(
                    loc.gate_type,
                    GateType::MeasCrosstalkLocalPayload | GateType::MeasCrosstalkGlobalPayload
                )
            {
                continue;
            }
            let result = Self::hidden_mz_result_before_crosstalk_payload(context, loc)?;
            if !result.is_deterministic || !result.outcome.is_empty() {
                return Err(DemBuilderError::ConfigurationError(format!(
                    "exact deterministic measurement crosstalk DEM replay requires a state-independent hidden MZ result at location {loc_idx} (node {}, qubit {:?}); got deterministic={}, dependencies={:?}",
                    loc.node,
                    loc.qubits.first(),
                    result.is_deterministic,
                    result.outcome
                )));
            }
        }

        Ok(())
    }

    fn hidden_mz_result_before_crosstalk_payload(
        context: ExactBranchReplayContext<'_>,
        loc: &DagSpacetimeLocation,
    ) -> Result<SymbolicMeasurementResult, DemBuilderError> {
        let qubit = loc.qubits.first().copied().ok_or_else(|| {
            DemBuilderError::ConfigurationError(format!(
                "measurement crosstalk payload at node {} has no victim qubit",
                loc.node
            ))
        })?;
        let qubit_index = qubit.index();
        let topo_order = context.circuit.topological_order();
        let max_qubit = topo_order
            .iter()
            .filter_map(|&node| context.circuit.gate(node))
            .flat_map(|gate| gate.qubits.iter())
            .map(pecos_core::QubitId::index)
            .max()
            .unwrap_or(qubit_index);
        let mut sim = SymbolicSparseStab::new(max_qubit.max(qubit_index) + 1);

        for node in topo_order {
            if node == loc.node {
                return sim.mz(&[qubit_index]).into_iter().next().ok_or_else(|| {
                    DemBuilderError::ConfigurationError(format!(
                        "failed to synthesize hidden MZ for measurement crosstalk payload at node {}",
                        loc.node
                    ))
                });
            }

            if let Some(gate) = context.circuit.gate(node) {
                let qubits: Vec<usize> =
                    gate.qubits.iter().map(pecos_core::QubitId::index).collect();
                Self::apply_symbolic_gate_for_crosstalk_hidden_mz(
                    &mut sim,
                    node,
                    gate.gate_type,
                    &qubits,
                )?;
            }
        }

        Err(DemBuilderError::ConfigurationError(format!(
            "measurement crosstalk payload node {} was not found in the replay circuit",
            loc.node
        )))
    }

    fn exact_measurement_crosstalk_pauli_effect(
        &self,
        context: ExactBranchReplayContext<'_>,
        loc: &DagSpacetimeLocation,
        pauli: Pauli,
    ) -> Result<FaultMechanism, DemBuilderError> {
        let mut triggered_dets: SmallVec<[u32; 4]> = SmallVec::new();
        let mut triggered_obs: SmallVec<[u32; 2]> = SmallVec::new();

        for detector in &self.detectors {
            let indices =
                self.measurement_indices_from_refs(&detector.records, &detector.meas_ids)?;
            if self.measurement_parity_anticommutes_after_crosstalk_payload(
                context, loc, pauli, &indices,
            )? {
                xor_toggle_4(&mut triggered_dets, detector.id);
            }
        }

        for observable in &self.observables {
            let indices =
                self.measurement_indices_from_refs(&observable.records, &observable.meas_ids)?;
            if self.measurement_parity_anticommutes_after_crosstalk_payload(
                context, loc, pauli, &indices,
            )? {
                xor_toggle_2(&mut triggered_obs, observable.id);
            }
        }

        triggered_dets.sort_unstable();
        triggered_obs.sort_unstable();
        Ok(FaultMechanism::from_sorted_with_tracked_paulis(
            triggered_dets,
            triggered_obs,
            SmallVec::new(),
        ))
    }

    fn measurement_parity_anticommutes_after_crosstalk_payload(
        &self,
        context: ExactBranchReplayContext<'_>,
        loc: &DagSpacetimeLocation,
        pauli: Pauli,
        measurement_indices: &[usize],
    ) -> Result<bool, DemBuilderError> {
        let victim = loc.qubits.first().copied().ok_or_else(|| {
            DemBuilderError::ConfigurationError(format!(
                "measurement crosstalk payload at node {} has no victim qubit",
                loc.node
            ))
        })?;
        let victim_index = victim.index();
        let mut prop = PauliProp::new();
        for &measurement_idx in measurement_indices {
            let &(_, qubit, basis) = self
                .influence_map
                .measurements
                .get(measurement_idx)
                .ok_or_else(|| {
                    DemBuilderError::ConfigurationError(format!(
                        "measurement crosstalk exact replay has no measurement {measurement_idx}"
                    ))
                })?;
            match basis {
                0 => prop.track_z(&[qubit]),
                1 => prop.track_x(&[qubit]),
                other => {
                    return Err(DemBuilderError::ConfigurationError(format!(
                        "measurement crosstalk exact replay does not support measurement basis {other}"
                    )));
                }
            }
        }

        let topo_order = context.circuit.topological_order();
        let payload_pos = topo_order
            .iter()
            .position(|&node| node == loc.node)
            .ok_or_else(|| {
                DemBuilderError::ConfigurationError(format!(
                    "measurement crosstalk payload node {} was not found in the replay circuit",
                    loc.node
                ))
            })?;
        for &node in topo_order[payload_pos + 1..].iter().rev() {
            if let Some(gate) = context.circuit.gate(node) {
                apply_gate(&mut prop, gate, Direction::Backward);
            }
        }

        Ok(Self::pauli_anticommutes_with_prop_on_qubit(
            &prop,
            victim_index,
            pauli,
        ))
    }

    fn pauli_anticommutes_with_prop_on_qubit(prop: &PauliProp, qubit: usize, pauli: Pauli) -> bool {
        let has_x = prop.contains_x(qubit);
        let has_z = prop.contains_z(qubit);
        match pauli {
            Pauli::I => false,
            Pauli::X => has_z,
            Pauli::Z => has_x,
            Pauli::Y => has_x ^ has_z,
        }
    }

    fn apply_symbolic_gate_for_crosstalk_hidden_mz(
        sim: &mut SymbolicSparseStab,
        node: usize,
        gate_type: GateType,
        qubits: &[usize],
    ) -> Result<(), DemBuilderError> {
        use crate::fault_tolerance::symbolic_replay::{
            ArityError, Dispatch, apply_unitary_clifford,
        };

        let arity_error = |err: ArityError| match err {
            ArityError::TooFew { required, actual } => {
                DemBuilderError::ConfigurationError(format!(
                    "measurement crosstalk replay expected gate {gate_type:?} at node {node} to have at least {required} qubit(s), got {actual}"
                ))
            }
            ArityError::OddPairing { actual } => DemBuilderError::ConfigurationError(format!(
                "measurement crosstalk replay expected gate {gate_type:?} at node {node} to have an even number of qubits, got {actual}"
            )),
        };

        if apply_unitary_clifford(sim, gate_type, qubits).map_err(arity_error)? == Dispatch::Applied
        {
            return Ok(());
        }

        let require = |n: usize| -> Result<(), DemBuilderError> {
            if qubits.len() < n {
                return Err(arity_error(ArityError::TooFew {
                    required: n,
                    actual: qubits.len(),
                }));
            }
            Ok(())
        };

        match gate_type {
            GateType::MZ | GateType::MeasureFree => {
                require(1)?;
                sim.mz(qubits);
            }
            GateType::PZ | GateType::QAlloc => {
                require(1)?;
                for &qubit in qubits {
                    sim.pz(qubit);
                }
            }
            GateType::I
            | GateType::Idle
            | GateType::QFree
            | GateType::MeasCrosstalkGlobalPayload
            | GateType::MeasCrosstalkLocalPayload
            | GateType::TrackedPauliMeta => {}
            _ => {
                return Err(DemBuilderError::ConfigurationError(format!(
                    "measurement crosstalk exact deterministic replay does not support gate {gate_type:?} before payload node {node}"
                )));
            }
        }

        Ok(())
    }

    #[cfg(test)]
    fn exact_omitted_branch_base_effect(
        &self,
        context: ExactBranchReplayContext<'_>,
        request: ExactBranchReplayRequest,
    ) -> Result<FaultMechanism, DemBuilderError> {
        let branch = circuit_with_omitted_two_qubit_gate(context.circuit, request.gate_node)?;
        self.exact_omitted_branch_base_effect_for_branch(context, request, &branch)
    }

    fn exact_omitted_branch_base_effect_for_branch(
        &self,
        context: ExactBranchReplayContext<'_>,
        request: ExactBranchReplayRequest,
        branch: &pecos_quantum::DagCircuit,
    ) -> Result<FaultMechanism, DemBuilderError> {
        use crate::fault_tolerance::influence_builder::InfluenceBuilder;

        let ideal_history = self.exact_ideal_measurement_history(context);
        let branch_info = InfluenceBuilder::new(branch).run_symbolic_simulation();
        let mut triggered_dets: SmallVec<[u32; 4]> = SmallVec::new();
        let mut triggered_obs: SmallVec<[u32; 2]> = SmallVec::new();

        for detector in &self.detectors {
            let indices =
                self.measurement_indices_from_refs(&detector.records, &detector.meas_ids)?;
            if omitted_branch_flips_measurement_parity_from_histories(
                request,
                ideal_history.as_ref(),
                &branch_info.history,
                &indices,
            )? {
                xor_toggle_4(&mut triggered_dets, detector.id);
            }
        }

        for observable in &self.observables {
            let indices =
                self.measurement_indices_from_refs(&observable.records, &observable.meas_ids)?;
            if omitted_branch_flips_measurement_parity_from_histories(
                request,
                ideal_history.as_ref(),
                &branch_info.history,
                &indices,
            )? {
                xor_toggle_2(&mut triggered_obs, observable.id);
            }
        }

        triggered_dets.sort_unstable();
        triggered_obs.sort_unstable();
        Ok(FaultMechanism::from_sorted_with_tracked_paulis(
            triggered_dets,
            triggered_obs,
            SmallVec::new(),
        ))
    }

    fn exact_ideal_measurement_history(
        &self,
        context: ExactBranchReplayContext<'_>,
    ) -> Rc<MeasurementHistory> {
        use crate::fault_tolerance::influence_builder::InfluenceBuilder;

        if let Some(cached) = self.exact_ideal_history_cache.borrow().as_ref().cloned() {
            return cached;
        }

        let history = Rc::new(
            InfluenceBuilder::new(context.circuit)
                .run_symbolic_simulation()
                .history,
        );
        *self.exact_ideal_history_cache.borrow_mut() = Some(history.clone());
        history
    }

    fn exact_branch_analysis(
        &self,
        context: ExactBranchReplayContext<'_>,
        request: ExactBranchReplayRequest,
    ) -> Result<ExactBranchReplayAnalysis, DemBuilderError> {
        use crate::fault_tolerance::propagator::DagFaultAnalyzer;

        if let Some(cached) = self
            .exact_branch_cache
            .borrow()
            .get(&request.gate_node)
            .cloned()
        {
            return Ok(cached);
        }

        let branch = circuit_with_omitted_two_qubit_gate(context.circuit, request.gate_node)?;
        let base_effect =
            self.exact_omitted_branch_base_effect_for_branch(context, request, &branch)?;
        let branch_map = DagFaultAnalyzer::new(&branch).build_influence_map();
        let branch_locs = identity_location_pair_for_request(
            request,
            &self.influence_map.locations,
            &branch_map.locations,
        )?;
        let (meas_to_detectors, meas_to_observables) = self.build_measurement_mappings();
        let branch_effects = Self::two_qubit_effect_table_for_map(
            &branch_map,
            branch_locs[0],
            branch_locs[1],
            &meas_to_detectors,
            &meas_to_observables,
        );
        let analysis = ExactBranchReplayAnalysis {
            base_effect,
            branch_effects,
        };
        self.exact_branch_cache
            .borrow_mut()
            .insert(request.gate_node, analysis.clone());
        Ok(analysis)
    }

    fn exact_replacement_branch_effect(
        &self,
        context: ExactBranchReplayContext<'_>,
        request: ExactBranchReplayRequest,
        replacement_pauli_label: &str,
    ) -> Result<FaultMechanism, DemBuilderError> {
        let (p1, p2) = two_qubit_label_to_pauli_indices(replacement_pauli_label).ok_or_else(|| {
            DemBuilderError::ConfigurationError(format!(
                "exact_branch_replay replacement Pauli label {replacement_pauli_label:?} is not a two-qubit Pauli label"
            ))
        })?;
        let analysis = self.exact_branch_analysis(context, request)?;
        if p1 == 0 && p2 == 0 {
            return Ok(analysis.base_effect);
        }

        Ok(analysis
            .base_effect
            .xor(&analysis.branch_effects[p1 as usize][p2 as usize]))
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
                GateType::MeasCrosstalkLocalPayload
                    if !loc.before
                        && matches!(
                            self.noise.measurement_crosstalk_dem_mode,
                            MeasurementCrosstalkDemMode::ExactDeterministic
                                | MeasurementCrosstalkDemMode::ExactDeterministicLeakageAsDepolarizing
                                | MeasurementCrosstalkDemMode::AveragedHiddenLeakageAsDepolarizing
                        )
                        && self.noise.p_meas_crosstalk_local > 0.0 =>
                {
                    self.process_measurement_crosstalk_source_tracked(
                        loc_idx,
                        self.noise.p_meas_crosstalk_local,
                        dem,
                        meas_to_detectors,
                        meas_to_observables,
                    );
                }
                GateType::MeasCrosstalkGlobalPayload
                    if !loc.before
                        && matches!(
                            self.noise.measurement_crosstalk_dem_mode,
                            MeasurementCrosstalkDemMode::ExactDeterministic
                                | MeasurementCrosstalkDemMode::ExactDeterministicLeakageAsDepolarizing
                                | MeasurementCrosstalkDemMode::AveragedHiddenLeakageAsDepolarizing
                        )
                        && self.noise.p_meas_crosstalk_global > 0.0 =>
                {
                    self.process_measurement_crosstalk_source_tracked(
                        loc_idx,
                        self.noise.p_meas_crosstalk_global,
                        dem,
                        meas_to_detectors,
                        meas_to_observables,
                    );
                }
                gate_type if is_two_qubit_noise_gate(gate_type) && !loc.before => {}
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
        for [loc1_idx, loc2_idx] in two_qubit_after_location_pairs(locations) {
            let loc1 = &locations[loc1_idx];
            let loc2 = &locations[loc2_idx];
            let rates = self.rates_2q_for_locs(loc1, loc2);
            if rates.iter().any(|r| *r > 0.0) {
                self.process_two_qubit_fault_source_tracked(
                    loc1_idx,
                    loc2_idx,
                    rates,
                    dem,
                    meas_to_detectors,
                    meas_to_observables,
                );
            }
            if self.noise.p2_replacement_approximation
                == ReplacementBranchApproximation::BranchImpact
            {
                self.process_two_qubit_replacement_branch_impacts_source_tracked(
                    loc1_idx,
                    loc2_idx,
                    dem,
                    meas_to_detectors,
                    meas_to_observables,
                );
            }
            if self.noise.p2_replacement_approximation
                == ReplacementBranchApproximation::ExactBranchReplay
            {
                self.process_two_qubit_exact_replacement_branches_source_tracked(
                    loc1_idx, loc2_idx, dem,
                );
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

    /// Processes starred two-qubit replacement branches as explicit branch impacts.
    fn process_two_qubit_replacement_branch_impacts_source_tracked(
        &self,
        loc1: usize,
        loc2: usize,
        dem: &mut DetectorErrorModel,
        meas_to_detectors: &BTreeMap<usize, Vec<u32>>,
        meas_to_observables: &BTreeMap<usize, Vec<u32>>,
    ) {
        let Some(weights) = &self.noise.p2_weights else {
            return;
        };
        let loc1_meta = &self.influence_map.locations[loc1];
        let branch_impacts = weights.replacement_branch_impacts(loc1_meta.gate_type);
        if branch_impacts.is_empty() {
            return;
        }

        let effects =
            self.two_qubit_effect_table(loc1, loc2, meas_to_detectors, meas_to_observables);
        let loc2_meta = &self.influence_map.locations[loc2];

        for impact in branch_impacts {
            let Some((p1, p2)) = two_qubit_label_to_pauli_indices(&impact.pauli_label) else {
                continue;
            };
            let prob = self.noise.p2 * impact.relative_probability;
            Self::add_two_qubit_pauli_contribution(
                loc1,
                loc2,
                p1,
                p2,
                prob,
                &effects,
                loc1_meta,
                loc2_meta,
                dem,
                Some(DirectSourceFamily::TwoLocationReplacementBranchImpact),
            );
        }
    }

    /// Processes starred two-qubit replacement branches by replaying the exact
    /// omitted-gate branch against detector/observable metadata.
    fn process_two_qubit_exact_replacement_branches_source_tracked(
        &self,
        loc1: usize,
        loc2: usize,
        dem: &mut DetectorErrorModel,
    ) {
        let Some(weights) = &self.noise.p2_weights else {
            return;
        };
        if !weights.has_replacement_entries() {
            return;
        }

        let context = self
            .exact_branch_context
            .expect("exact_branch_replay was validated with circuit context");
        let loc1_meta = &self.influence_map.locations[loc1];
        let loc2_meta = &self.influence_map.locations[loc2];
        let request = ExactBranchReplayRequest {
            gate_node: loc1_meta.node,
            gate_type: loc1_meta.gate_type,
            loc_indices: [loc1, loc2],
        };

        for (replacement_pauli, relative_probability) in weights.replacement_entries() {
            if *relative_probability <= 0.0 {
                continue;
            }
            let label = two_qubit_label_for_replay(replacement_pauli)
                .expect("exact_branch_replay replacement Pauli was validated");
            let (p1, p2) = two_qubit_label_to_pauli_indices(&label)
                .expect("exact_branch_replay label must be a two-qubit Pauli");
            let analysis = self
                .exact_branch_analysis(context, request)
                .expect("exact_branch_replay effect was validated");
            let base_effect = analysis.base_effect.clone();
            let branch_pauli_effect = analysis.branch_effects[p1 as usize][p2 as usize].clone();
            let effect = base_effect.xor(&branch_pauli_effect);
            if effect.is_empty() {
                continue;
            }

            let source_locations = [loc1, loc2];
            let source_paulis = [Pauli::from_u8(p1), Pauli::from_u8(p2)];
            let source_gate_types = [loc1_meta.gate_type, loc2_meta.gate_type];
            let source_before_flags = [loc1_meta.before, loc2_meta.before];
            dem.add_direct_contribution_with_source_components(
                effect,
                self.noise.p2 * *relative_probability,
                SourceMetadata::new(
                    &source_locations,
                    &source_paulis,
                    &source_gate_types,
                    &source_before_flags,
                )
                .with_direct_source_family(DirectSourceFamily::TwoLocationExactReplacementBranch)
                .with_replacement_branch(),
                &DirectSourceComponents::new(&base_effect, &branch_pauli_effect),
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

    /// Processes local measurement-crosstalk payloads when hidden outcomes are
    /// deterministic and state-independent.
    fn process_measurement_crosstalk_source_tracked(
        &self,
        loc_idx: usize,
        payload_rate: f64,
        dem: &mut DetectorErrorModel,
        meas_to_detectors: &BTreeMap<usize, Vec<u32>>,
        meas_to_observables: &BTreeMap<usize, Vec<u32>>,
    ) {
        let (bit_flip_probability, leak_probability) =
            match self.noise.measurement_crosstalk_dem_mode {
                MeasurementCrosstalkDemMode::AveragedHiddenLeakageAsDepolarizing => (
                    0.5 * (self.noise.p_meas_crosstalk_model.p_0_to_1
                        + self.noise.p_meas_crosstalk_model.p_1_to_0),
                    0.5 * (self.noise.p_meas_crosstalk_model.p_0_to_leak
                        + self.noise.p_meas_crosstalk_model.p_1_to_leak),
                ),
                MeasurementCrosstalkDemMode::ExactDeterministic
                | MeasurementCrosstalkDemMode::ExactDeterministicLeakageAsDepolarizing => {
                    let context = self.exact_branch_context.expect(
                        "measurement crosstalk exact deterministic mode was validated with context",
                    );
                    let loc = &self.influence_map.locations[loc_idx];
                    let hidden = Self::hidden_mz_result_before_crosstalk_payload(context, loc)
                        .expect(
                            "measurement crosstalk exact deterministic hidden result was validated",
                        );
                    if hidden.flip {
                        (
                            self.noise.p_meas_crosstalk_model.p_1_to_0,
                            self.noise.p_meas_crosstalk_model.p_1_to_leak,
                        )
                    } else {
                        (
                            self.noise.p_meas_crosstalk_model.p_0_to_1,
                            self.noise.p_meas_crosstalk_model.p_0_to_leak,
                        )
                    }
                }
                MeasurementCrosstalkDemMode::Omitted => {
                    return;
                }
            };
        if matches!(
            self.noise.measurement_crosstalk_dem_mode,
            MeasurementCrosstalkDemMode::ExactDeterministicLeakageAsDepolarizing
                | MeasurementCrosstalkDemMode::AveragedHiddenLeakageAsDepolarizing
        ) {
            let leak_pauli_probability = leak_probability / 4.0;
            self.process_measurement_crosstalk_pauli_rates_source_tracked(
                loc_idx,
                [
                    payload_rate * (bit_flip_probability + leak_pauli_probability),
                    payload_rate * leak_pauli_probability,
                    payload_rate * leak_pauli_probability,
                ],
                dem,
                meas_to_detectors,
                meas_to_observables,
            );
        } else {
            self.process_measurement_crosstalk_pauli_rates_source_tracked(
                loc_idx,
                [payload_rate * bit_flip_probability, 0.0, 0.0],
                dem,
                meas_to_detectors,
                meas_to_observables,
            );
        }
    }

    /// Processes local measurement-crosstalk payloads as single-location Pauli
    /// source channels while preserving crosstalk source metadata.
    fn process_measurement_crosstalk_pauli_rates_source_tracked(
        &self,
        loc_idx: usize,
        rates: [f64; 3],
        dem: &mut DetectorErrorModel,
        meas_to_detectors: &BTreeMap<usize, Vec<u32>>,
        meas_to_observables: &BTreeMap<usize, Vec<u32>>,
    ) {
        let loc = &self.influence_map.locations[loc_idx];
        let [rate_x, rate_y, rate_z] = rates;
        let effect = |pauli| -> FaultMechanism {
            if loc.gate_type == GateType::MeasCrosstalkGlobalPayload {
                let context = self.exact_branch_context.expect(
                    "measurement crosstalk exact deterministic mode was validated with context",
                );
                self.exact_measurement_crosstalk_pauli_effect(context, loc, pauli)
                    .expect("global measurement crosstalk exact replay was validated")
            } else {
                self.compute_mechanism(loc_idx, pauli, meas_to_detectors, meas_to_observables)
            }
        };
        let x_effect = effect(Pauli::X);
        let z_effect = effect(Pauli::Z);

        if rate_x > 0.0 && !x_effect.is_empty() {
            dem.add_direct_contribution_with_source(
                x_effect.clone(),
                rate_x,
                SourceMetadata::new(&[loc_idx], &[Pauli::X], &[loc.gate_type], &[loc.before])
                    .with_direct_source_family(DirectSourceFamily::MeasurementCrosstalk),
            );
        }
        if rate_z > 0.0 && !z_effect.is_empty() {
            dem.add_direct_contribution_with_source(
                z_effect.clone(),
                rate_z,
                SourceMetadata::new(&[loc_idx], &[Pauli::Z], &[loc.gate_type], &[loc.before])
                    .with_direct_source_family(DirectSourceFamily::MeasurementCrosstalk),
            );
        }

        let y_effect = x_effect.xor(&z_effect);
        if rate_y > 0.0 && !y_effect.is_empty() {
            if !x_effect.is_empty() && !z_effect.is_empty() {
                dem.add_y_decomposed_contribution_with_source(
                    &x_effect,
                    &z_effect,
                    rate_y,
                    SourceMetadata::new(&[loc_idx], &[Pauli::Y], &[loc.gate_type], &[loc.before])
                        .with_direct_source_family(DirectSourceFamily::MeasurementCrosstalk),
                );
            } else {
                dem.add_direct_contribution_with_source(
                    y_effect,
                    rate_y,
                    SourceMetadata::new(&[loc_idx], &[Pauli::Y], &[loc.gate_type], &[loc.before])
                        .with_direct_source_family(DirectSourceFamily::MeasurementCrosstalk),
                );
            }
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

        let effects =
            self.two_qubit_effect_table(loc1, loc2, meas_to_detectors, meas_to_observables);

        // Process all 15 non-trivial Pauli combinations
        for p1 in 0u8..4 {
            for p2 in 0u8..4 {
                if p1 == 0 && p2 == 0 {
                    continue; // Skip II
                }

                // Per-pair rate: index = 4*p1 + p2 - 1 (skipping II at idx 0).
                let flat = 4 * (p1 as usize) + (p2 as usize);
                let prob = rates[flat - 1];
                if prob == 0.0 {
                    continue;
                }
                Self::add_two_qubit_pauli_contribution(
                    loc1, loc2, p1, p2, prob, &effects, loc1_meta, loc2_meta, dem, None,
                );
            }
        }
    }

    fn two_qubit_effect_table(
        &self,
        loc1: usize,
        loc2: usize,
        meas_to_detectors: &BTreeMap<usize, Vec<u32>>,
        meas_to_observables: &BTreeMap<usize, Vec<u32>>,
    ) -> [[FaultMechanism; 4]; 4] {
        let x1 = self.compute_mechanism(loc1, Pauli::X, meas_to_detectors, meas_to_observables);
        let z1 = self.compute_mechanism(loc1, Pauli::Z, meas_to_detectors, meas_to_observables);
        let x2 = self.compute_mechanism(loc2, Pauli::X, meas_to_detectors, meas_to_observables);
        let z2 = self.compute_mechanism(loc2, Pauli::Z, meas_to_detectors, meas_to_observables);

        let get_single_effect = |p: u8, x: &FaultMechanism, z: &FaultMechanism| -> FaultMechanism {
            match p {
                0 => FaultMechanism::new(),
                1 => x.clone(),
                2 => x.xor(z),
                3 => z.clone(),
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
        effects
    }

    fn two_qubit_effect_table_for_map(
        influence_map: &DagFaultInfluenceMap,
        loc1: usize,
        loc2: usize,
        meas_to_detectors: &BTreeMap<usize, Vec<u32>>,
        meas_to_observables: &BTreeMap<usize, Vec<u32>>,
    ) -> [[FaultMechanism; 4]; 4] {
        let x1 = Self::compute_mechanism_for_map(
            influence_map,
            loc1,
            Pauli::X,
            meas_to_detectors,
            meas_to_observables,
        );
        let z1 = Self::compute_mechanism_for_map(
            influence_map,
            loc1,
            Pauli::Z,
            meas_to_detectors,
            meas_to_observables,
        );
        let x2 = Self::compute_mechanism_for_map(
            influence_map,
            loc2,
            Pauli::X,
            meas_to_detectors,
            meas_to_observables,
        );
        let z2 = Self::compute_mechanism_for_map(
            influence_map,
            loc2,
            Pauli::Z,
            meas_to_detectors,
            meas_to_observables,
        );

        let get_single_effect = |p: u8, x: &FaultMechanism, z: &FaultMechanism| -> FaultMechanism {
            match p {
                0 => FaultMechanism::new(),
                1 => x.clone(),
                2 => x.xor(z),
                3 => z.clone(),
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
        effects
    }

    #[allow(clippy::too_many_arguments)]
    fn add_two_qubit_pauli_contribution(
        loc1: usize,
        loc2: usize,
        p1: u8,
        p2: u8,
        prob: f64,
        effects: &[[FaultMechanism; 4]; 4],
        loc1_meta: &DagSpacetimeLocation,
        loc2_meta: &DagSpacetimeLocation,
        dem: &mut DetectorErrorModel,
        direct_source_family: Option<DirectSourceFamily>,
    ) {
        let effect = &effects[p1 as usize][p2 as usize];
        if effect.is_empty() {
            return;
        }

        let e1 = &effects[p1 as usize][0];
        let e2 = &effects[0][p2 as usize];

        let graphlike_decomposable = effect.num_detectors() == 2
            && effect.dem_outputs.is_empty()
            && !e1.is_empty()
            && !e2.is_empty()
            && e1.num_detectors() <= 2
            && e2.num_detectors() <= 2;
        if graphlike_decomposable {
            dem.mark_graphlike_decomposable(effect.detectors[0], effect.detectors[1]);
        }

        let source_locations = [loc1, loc2];
        let source_paulis = [Pauli::from_u8(p1), Pauli::from_u8(p2)];
        let source_gate_types = [loc1_meta.gate_type, loc2_meta.gate_type];
        let source_before_flags = [loc1_meta.before, loc2_meta.before];

        let source_frame_components = if direct_source_family.is_none() {
            Self::two_qubit_clifford_source_frame_components(loc1_meta.gate_type, p1, p2, effects)
        } else {
            None
        };
        if let Some(parts) = source_frame_components.as_ref() {
            dem.add_direct_contribution_with_source_components(
                effect.clone(),
                prob,
                SourceMetadata::new(
                    &source_locations,
                    &source_paulis,
                    &source_gate_types,
                    &source_before_flags,
                ),
                &DirectSourceComponents::from_slice(parts.as_slice()),
            );
            return;
        }

        if let Some((a1, a2, b1, b2)) = get_y_decomposition(p1, p2) {
            let e_a = &effects[a1 as usize][a2 as usize];
            let e_b = &effects[b1 as usize][b2 as usize];
            let mut source = SourceMetadata::new(
                &source_locations,
                &source_paulis,
                &source_gate_types,
                &source_before_flags,
            );
            if direct_source_family.is_some() {
                source = source.with_replacement_branch();
            }
            dem.add_y_decomposed_contribution_with_source(e_a, e_b, prob, source);
        } else {
            let mut source = SourceMetadata::new(
                &source_locations,
                &source_paulis,
                &source_gate_types,
                &source_before_flags,
            );
            if let Some(family) = direct_source_family {
                source = source
                    .with_direct_source_family(family)
                    .with_replacement_branch();
            }

            dem.add_direct_contribution_with_source_components(
                effect.clone(),
                prob,
                source,
                &DirectSourceComponents::new(e1, e2),
            );
        }
    }

    /// Builds exact source-frame components for ordinary post-gate Pauli noise
    /// on supported two-qubit Clifford gates.
    ///
    /// A post-gate Pauli can be pulled back through the Clifford into a pre-gate
    /// Pauli. Decomposing that pre-gate Pauli into X/Z generators often exposes
    /// the graphlike source pieces that were hidden by the native gate frame.
    /// Each generator is then pushed forward again and looked up in the existing
    /// post-gate effect table, so the XOR of returned components is exactly the
    /// original post-gate effect.
    fn two_qubit_clifford_source_frame_components(
        gate_type: GateType,
        post_p1: u8,
        post_p2: u8,
        effects: &[[FaultMechanism; 4]; 4],
    ) -> Option<SmallVec<[FaultMechanism; 4]>> {
        let images = two_qubit_pre_generator_post_images(gate_type)?;
        let (pre_p1, pre_p2) = invert_two_qubit_clifford_post_pauli(images, (post_p1, post_p2))?;

        let mut components = SmallVec::new();
        for image in two_qubit_pre_pauli_generator_images(images, pre_p1, pre_p2) {
            toggle_source_component(
                &mut components,
                effects[image.0 as usize][image.1 as usize].clone(),
            );
        }

        if components.is_empty() {
            return None;
        }

        #[cfg(debug_assertions)]
        {
            let combined = components
                .iter()
                .fold(FaultMechanism::new(), |acc, part| acc.xor(part));
            debug_assert_eq!(combined, effects[post_p1 as usize][post_p2 as usize]);
        }

        Some(components)
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
            if det.records.is_empty() {
                for &meas_id in &det.meas_ids {
                    if let Some(tc_idx) = self.resolve_meas_id_to_tc_index(meas_id)
                        && let Some(&influence_idx) = tc_to_influence.get(&tc_idx)
                    {
                        meas_to_detectors
                            .entry(influence_idx)
                            .or_default()
                            .push(det.id);
                    }
                }
            } else {
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
        }

        for obs in &self.observables {
            if influence_observable_ids.contains(&obs.id) {
                continue;
            }
            if obs.records.is_empty() {
                for &meas_id in &obs.meas_ids {
                    if let Some(tc_idx) = self.resolve_meas_id_to_tc_index(meas_id)
                        && let Some(&influence_idx) = tc_to_influence.get(&tc_idx)
                    {
                        meas_to_observables
                            .entry(influence_idx)
                            .or_default()
                            .push(obs.id);
                    }
                }
            } else {
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
        Self::compute_mechanism_for_map(
            self.influence_map,
            loc_idx,
            pauli,
            meas_to_detectors,
            meas_to_observables,
        )
    }

    fn compute_mechanism_for_map(
        influence_map: &DagFaultInfluenceMap,
        loc_idx: usize,
        pauli: Pauli,
        meas_to_detectors: &BTreeMap<usize, Vec<u32>>,
        meas_to_observables: &BTreeMap<usize, Vec<u32>>,
    ) -> FaultMechanism {
        // Get the measurement indices that this fault flips
        let rust_dets = influence_map.get_detector_indices(loc_idx, pauli.as_u8());

        // Convert to pre-defined detector IDs using XOR
        let mut triggered_dets: SmallVec<[u32; 4]> = SmallVec::new();
        let mut triggered_obs: SmallVec<[u32; 2]> = SmallVec::new();
        let mut triggered_tracked_paulis: SmallVec<[u32; 2]> = SmallVec::new();

        for dem_output_idx in influence_map.get_observable_indices(loc_idx, pauli.as_u8()) {
            xor_toggle_2(&mut triggered_obs, dem_output_idx);
        }
        for tracked_pauli_idx in influence_map.get_tracked_pauli_indices(loc_idx, pauli.as_u8()) {
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

fn two_qubit_label_to_pauli_indices(label: &str) -> Option<(u8, u8)> {
    let mut chars = label.chars();
    let p1 = pauli_label_to_index(chars.next()?)?;
    let p2 = pauli_label_to_index(chars.next()?)?;
    chars.next().is_none().then_some((p1, p2))
}

fn two_qubit_label_for_replay(pauli: &pecos_core::PauliString) -> Result<String, DemBuilderError> {
    let mut label = String::with_capacity(2);
    for qubit in [0, 1] {
        label.push(pauli_index_to_label(match pauli.get(qubit) {
            pecos_core::Pauli::I => 0,
            pecos_core::Pauli::X => 1,
            pecos_core::Pauli::Y => 2,
            pecos_core::Pauli::Z => 3,
        }));
    }
    if pauli.qubits().into_iter().any(|qubit| qubit > 1) {
        return Err(DemBuilderError::ConfigurationError(format!(
            "exact_branch_replay replacement Pauli {} is not supported by the two-qubit replay path",
            pauli.to_sparse_str()
        )));
    }
    Ok(label)
}

fn pauli_index_to_label(index: u8) -> char {
    match index {
        0 => 'I',
        1 => 'X',
        2 => 'Y',
        3 => 'Z',
        _ => unreachable!("Pauli index must be 0-3"),
    }
}

fn pauli_label_to_index(label: char) -> Option<u8> {
    match label {
        'I' => Some(0),
        'X' => Some(1),
        'Y' => Some(2),
        'Z' => Some(3),
        _ => None,
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

type TwoQubitPauli = (u8, u8);
type TwoQubitGeneratorImages = [TwoQubitPauli; 4];

/// Returns post-gate images of the pre-gate generators
/// `[X1, Z1, X2, Z2]`, ignoring phase.
#[inline]
fn two_qubit_pre_generator_post_images(gate_type: GateType) -> Option<TwoQubitGeneratorImages> {
    match gate_type {
        GateType::CX => Some([
            (1, 1), // X1 -> XX
            (3, 0), // Z1 -> ZI
            (0, 1), // X2 -> IX
            (3, 3), // Z2 -> ZZ
        ]),
        GateType::CZ => Some([
            (1, 3), // X1 -> XZ
            (3, 0), // Z1 -> ZI
            (3, 1), // X2 -> ZX
            (0, 3), // Z2 -> IZ
        ]),
        GateType::SZZ | GateType::SZZdg => Some([
            (2, 3), // X1 -> YZ
            (3, 0), // Z1 -> ZI
            (3, 2), // X2 -> ZY
            (0, 3), // Z2 -> IZ
        ]),
        _ => None,
    }
}

#[inline]
fn invert_two_qubit_clifford_post_pauli(
    images: TwoQubitGeneratorImages,
    post: TwoQubitPauli,
) -> Option<TwoQubitPauli> {
    for pre_p1 in 0..4 {
        for pre_p2 in 0..4 {
            if forward_two_qubit_pauli(images, pre_p1, pre_p2) == post {
                return Some((pre_p1, pre_p2));
            }
        }
    }
    None
}

fn two_qubit_pre_pauli_generator_images(
    images: TwoQubitGeneratorImages,
    pre_p1: u8,
    pre_p2: u8,
) -> SmallVec<[TwoQubitPauli; 4]> {
    let mut out = SmallVec::new();
    if pauli_has_x(pre_p1) {
        out.push(images[0]);
    }
    if pauli_has_z(pre_p1) {
        out.push(images[1]);
    }
    if pauli_has_x(pre_p2) {
        out.push(images[2]);
    }
    if pauli_has_z(pre_p2) {
        out.push(images[3]);
    }
    out
}

#[inline]
fn forward_two_qubit_pauli(
    images: TwoQubitGeneratorImages,
    pre_p1: u8,
    pre_p2: u8,
) -> TwoQubitPauli {
    two_qubit_pre_pauli_generator_images(images, pre_p1, pre_p2)
        .into_iter()
        .fold((0, 0), xor_two_qubit_pauli)
}

#[inline]
fn xor_two_qubit_pauli(a: TwoQubitPauli, b: TwoQubitPauli) -> TwoQubitPauli {
    (xor_pauli(a.0, b.0), xor_pauli(a.1, b.1))
}

#[inline]
fn xor_pauli(a: u8, b: u8) -> u8 {
    pauli_from_bits(
        pauli_has_x(a) ^ pauli_has_x(b),
        pauli_has_z(a) ^ pauli_has_z(b),
    )
}

#[inline]
fn pauli_has_x(pauli: u8) -> bool {
    matches!(pauli, 1 | 2)
}

#[inline]
fn pauli_has_z(pauli: u8) -> bool {
    matches!(pauli, 2 | 3)
}

#[inline]
fn pauli_from_bits(has_x: bool, has_z: bool) -> u8 {
    match (has_x, has_z) {
        (false, false) => 0,
        (true, false) => 1,
        (true, true) => 2,
        (false, true) => 3,
    }
}

fn toggle_source_component(
    components: &mut SmallVec<[FaultMechanism; 4]>,
    component: FaultMechanism,
) {
    if component.is_empty() {
        return;
    }
    if let Some(index) = components
        .iter()
        .position(|existing| existing == &component)
    {
        components.remove(index);
    } else {
        components.push(component);
    }
}

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
    let json = json.trim();
    if json.is_empty() || json == "[]" {
        return Ok(Vec::new());
    }

    let parsed: serde_json::Value = serde_json::from_str(json).map_err(|err| {
        DemBuilderError::ParseError(format!("detectors JSON is malformed: {err}"))
    })?;
    let array = parsed
        .as_array()
        .ok_or_else(|| DemBuilderError::ParseError("detectors_json must be a JSON list".into()))?;
    array.iter().map(parse_single_detector).collect()
}

/// Parses a single detector object.
fn parse_single_detector(value: &serde_json::Value) -> Result<ParsedDetector, DemBuilderError> {
    let object = value
        .as_object()
        .ok_or_else(|| DemBuilderError::ParseError("detector entry must be an object".into()))?;
    reject_tracked_pauli(object, "detector")?;
    let id = extract_u32(
        object,
        &["id", "detector_id"],
        'D',
        "missing detector id",
        "detector id out of range",
    )?;

    let coords = extract_coords(object)?;
    let (records, meas_ids) = extract_measurement_refs(object, "detector")?;

    Ok(ParsedDetector {
        id,
        coords,
        records,
        meas_ids,
    })
}

/// Parses observable definitions from JSON.
fn parse_observables_json(json: &str) -> Result<Vec<ParsedObservable>, DemBuilderError> {
    let json = json.trim();
    if json.is_empty() || json == "[]" {
        return Ok(Vec::new());
    }

    let parsed: serde_json::Value = serde_json::from_str(json).map_err(|err| {
        DemBuilderError::ParseError(format!("observables JSON is malformed: {err}"))
    })?;
    let array = parsed.as_array().ok_or_else(|| {
        DemBuilderError::ParseError("observables_json must be a JSON list".into())
    })?;
    array.iter().map(parse_single_observable).collect()
}

/// Parses a single observable object.
fn parse_single_observable(value: &serde_json::Value) -> Result<ParsedObservable, DemBuilderError> {
    let object = value
        .as_object()
        .ok_or_else(|| DemBuilderError::ParseError("observable entry must be an object".into()))?;
    reject_tracked_pauli(object, "observable")?;
    let id = extract_u32(
        object,
        &["id", "observable_id"],
        'L',
        "missing observable id",
        "observable id out of range",
    )?;

    let (records, meas_ids) = extract_measurement_refs(object, "observable")?;
    // A present-but-malformed label is rejected rather than silently treated as
    // absent; the richer DEM metadata parser already holds that line.
    let label = match object.get("label") {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::String(label)) => Some(label.clone()),
        Some(other) => {
            return Err(DemBuilderError::ParseError(format!(
                "observable label must be a string or null, got {other}"
            )));
        }
    };

    Ok(ParsedObservable {
        id,
        records,
        meas_ids,
        label,
    })
}

/// Parse detector JSON into per-detector measurement-reference vectors for the
/// sampler builders, enforcing the **same** validation and resolution as
/// `DemBuilder`.
///
/// Schema/type validation (rejects malformed JSON, a non-list top level, a
/// non-object entry, non-integer values, `tracked_pauli` entries, and entries
/// referencing neither `records` nor `meas_ids`) comes from the shared serde
/// parser. On top of that, this resolves every reference against the
/// `influence_map` exactly as `DemBuilder::validate_metadata_refs` /
/// `resolve_meas_id_to_tc_index` do, and rejects fail-loud:
///   - a `records` offset that is out of range,
///   - a `meas_ids` value that does not resolve (a stamped `MeasId` absent from
///     the circuit, or -- when the circuit carries no stable ids -- an
///     out-of-range positional index), and
///   - co-present `records` + `meas_ids` that reference different measurements.
///
/// `meas_ids` are stamped stable ids when `influence_map.meas_ids` is populated
/// (the traced `from_guppy`/`from_circuit` path), and positional indices only
/// when it is empty -- matching `DemBuilder`. The returned vector uses the
/// sampler's storage convention: negative `records` offsets are kept as-is
/// (preferred when present, like `DemBuilder`), while a `meas_ids`-only entry is
/// emitted as the resolved absolute indices (positive ints).
///
/// An empty influence map (no measurements) keeps the escape hatch: refs are
/// opaque pass-through coordinates and resolution is skipped.
pub(crate) fn parse_detector_record_vectors(
    json: &str,
    influence_map: &DagFaultInfluenceMap,
) -> Result<Vec<Vec<i32>>, DemBuilderError> {
    reject_duplicate_stamped_meas_ids(influence_map)?;
    parse_detectors_json(json)?
        .iter()
        .map(|d| {
            resolve_sampler_record_vector("Detector", d.id, &d.records, &d.meas_ids, influence_map)
        })
        .collect()
}

/// Observable counterpart of [`parse_detector_record_vectors`].
pub(crate) fn parse_observable_record_vectors(
    json: &str,
    influence_map: &DagFaultInfluenceMap,
) -> Result<Vec<Vec<i32>>, DemBuilderError> {
    reject_duplicate_stamped_meas_ids(influence_map)?;
    parse_observables_json(json)?
        .iter()
        .map(|o| {
            resolve_sampler_record_vector(
                "Observable",
                o.id,
                &o.records,
                &o.meas_ids,
                influence_map,
            )
        })
        .collect()
}

/// Reject a circuit whose stable `MeasId`s are not unique, before resolving any
/// `meas_ids`. A duplicate would make stamped-id resolution bind to the first
/// occurrence (an ambiguous, silently-wrong bind); it indicates a trace/replay
/// bug, not bad caller input. Mirrors the guard in
/// `DemBuilder::validate_measurement_count` so the sampler JSON path rejects
/// exactly what `DemBuilder` does.
fn reject_duplicate_stamped_meas_ids(
    influence_map: &DagFaultInfluenceMap,
) -> Result<(), DemBuilderError> {
    let mut seen = std::collections::HashSet::with_capacity(influence_map.meas_ids.len());
    for mid in &influence_map.meas_ids {
        if !seen.insert(mid.0) {
            return Err(DemBuilderError::ParseError(format!(
                "duplicate stable MeasId {} in the traced circuit; each \
                 measurement must have a unique stamped id",
                mid.0
            )));
        }
    }
    Ok(())
}

/// Resolve a stamped/positional `meas_id` against the influence map, mirroring
/// `DemBuilder::resolve_meas_id_to_tc_index`: a stamped stable id when the
/// circuit carries them, a positional index only when it does not.
fn resolve_sampler_meas_id(influence_map: &DagFaultInfluenceMap, meas_id: usize) -> Option<usize> {
    if influence_map.meas_ids.is_empty() {
        (meas_id < influence_map.measurements.len()).then_some(meas_id)
    } else {
        influence_map
            .meas_ids
            .iter()
            .position(|mid| mid.0 == meas_id)
    }
}

/// Resolve a parsed `records`/`meas_ids` pair to the sampler's single-`Vec<i32>`
/// convention, with `DemBuilder`-equivalent validation. See
/// [`parse_detector_record_vectors`] for the contract.
fn resolve_sampler_record_vector(
    kind: &str,
    id: u32,
    records: &[i32],
    meas_ids: &[usize],
    influence_map: &DagFaultInfluenceMap,
) -> Result<Vec<i32>, DemBuilderError> {
    let num_measurements = influence_map.measurements.len();

    // Escape hatch: an empty influence map makes refs opaque pass-through
    // coordinates with no circuit to resolve against. Prefer records; emit
    // meas_ids verbatim as positional indices (there are no stable ids).
    if num_measurements == 0 {
        if !records.is_empty() {
            return Ok(records.to_vec());
        }
        return meas_ids
            .iter()
            .map(|&m| {
                i32::try_from(m).map_err(|_| {
                    DemBuilderError::ParseError(format!(
                        "{kind} {id} meas_id {m} is out of range for an i32 record vector"
                    ))
                })
            })
            .collect();
    }

    // Resolve each form to absolute measurement indices, fail-loud.
    let records_abs = records
        .iter()
        .map(|&offset| {
            record_offset_to_absolute_index(num_measurements, offset).ok_or_else(|| {
                DemBuilderError::ParseError(format!(
                    "{kind} {id} references record offset {offset}, which is out of \
                     range for a circuit with {num_measurements} measurement(s)"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let meas_ids_abs = meas_ids
        .iter()
        .map(|&meas_id| {
            resolve_sampler_meas_id(influence_map, meas_id).ok_or_else(|| {
                DemBuilderError::ParseError(format!(
                    "{kind} {id} references meas_id {meas_id}, which is not present in \
                     the circuit's {num_measurements} measurement(s)"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Co-present records and meas_ids must reference the same measurements
    // (mirrors `validate_metadata_refs`); they are alternatives, not additive.
    if !records.is_empty() && !meas_ids.is_empty() {
        let mut a = records_abs.clone();
        let mut b = meas_ids_abs.clone();
        a.sort_unstable();
        b.sort_unstable();
        if a != b {
            return Err(DemBuilderError::ParseError(format!(
                "{kind} {id} has both 'records' and 'meas_ids' but they reference \
                 different measurements (records -> {a:?}, meas_ids -> {b:?}); they \
                 are alternatives, not additive"
            )));
        }
    }

    // Prefer records (kept as Stim offsets, like `DemBuilder`); otherwise emit
    // the resolved absolute indices, which the sampler reads as positive
    // (absolute-index) record values.
    if !records.is_empty() {
        return Ok(records.to_vec());
    }
    meas_ids_abs
        .iter()
        .map(|&idx| {
            i32::try_from(idx).map_err(|_| {
                DemBuilderError::ParseError(format!(
                    "{kind} {id} resolved measurement index {idx} exceeds i32 range"
                ))
            })
        })
        .collect()
}

/// Rejects a JSON entry that declares `kind: "tracked_pauli"`.
///
/// Tracked Paulis reference qubits via `pauli`, not measurements, and are
/// only produced from circuit annotations -- never from `detectors_json` /
/// `observables_json`. The JSON parser reads only `id`/`records`, so a
/// tracked-Pauli entry here would be silently parsed as the wrong thing.
fn reject_tracked_pauli(
    object: &serde_json::Map<String, serde_json::Value>,
    kind: &str,
) -> Result<(), DemBuilderError> {
    if object.get("kind").and_then(serde_json::Value::as_str) == Some("tracked_pauli") {
        return Err(DemBuilderError::ParseError(format!(
            "{kind} entry uses kind=\"tracked_pauli\", which is not supported \
             in detectors_json/observables_json (tracked Paulis come only \
             from circuit annotations)"
        )));
    }
    Ok(())
}

/// Reads an entry id as either an unsigned integer or the DEM-label string
/// form (`prefix` is `'D'` for detectors, `'L'` for observables, e.g.
/// `"D0"`/`"L0"`); both normalize to the same integer. A string id with the
/// wrong prefix or a non-numeric body is a hard error -- silently
/// reinterpreting it would risk a mislabeled DEM.
fn extract_u32(
    object: &serde_json::Map<String, serde_json::Value>,
    keys: &[&str],
    prefix: char,
    missing_message: &str,
    range_message: &str,
) -> Result<u32, DemBuilderError> {
    let value = keys
        .iter()
        .find_map(|key| object.get(*key))
        .ok_or_else(|| DemBuilderError::ParseError(missing_message.into()))?;
    if let Some(raw) = value.as_u64() {
        return u32::try_from(raw).map_err(|_| DemBuilderError::ParseError(range_message.into()));
    }
    if let Some(s) = value.as_str() {
        let body = s.strip_prefix(prefix);
        if let Some(digits) = body
            && !digits.is_empty()
            && digits.bytes().all(|b| b.is_ascii_digit())
        {
            return digits
                .parse::<u32>()
                .map_err(|_| DemBuilderError::ParseError(range_message.into()));
        }
        return Err(DemBuilderError::ParseError(format!(
            "id {s:?} is not a valid identifier; expected an integer or the \
             {prefix:?}-prefixed form like {prefix}0"
        )));
    }
    Err(DemBuilderError::ParseError(format!(
        "{missing_message}: expected an integer or {prefix:?}-prefixed string id"
    )))
}

/// Extracts coordinates array [x, y, t].
fn extract_coords(
    object: &serde_json::Map<String, serde_json::Value>,
) -> Result<Option<[f64; 3]>, DemBuilderError> {
    let Some(coords) = object.get("coords") else {
        return Ok(None);
    };
    let array = coords
        .as_array()
        .ok_or_else(|| DemBuilderError::ParseError("detector coords must be an array".into()))?;
    if array.len() != 3 {
        return Err(DemBuilderError::ParseError(
            "detector coords must contain exactly three numbers".into(),
        ));
    }
    let mut values = [0.0; 3];
    for (idx, coord) in array.iter().enumerate() {
        values[idx] = coord
            .as_f64()
            .ok_or_else(|| DemBuilderError::ParseError("detector coords must be numeric".into()))?;
    }
    Ok(Some(values))
}

/// Extracts `records`/`meas_ids` arrays.
fn extract_measurement_refs(
    object: &serde_json::Map<String, serde_json::Value>,
    kind: &str,
) -> Result<(Vec<i32>, Vec<usize>), DemBuilderError> {
    let records = if let Some(records) = object.get("records") {
        let array = records.as_array().ok_or_else(|| {
            DemBuilderError::ParseError(format!("{kind} records must be an array"))
        })?;
        array
            .iter()
            .map(|record| {
                let raw = record.as_i64().ok_or_else(|| {
                    DemBuilderError::ParseError(format!("{kind} record offsets must be integers"))
                })?;
                i32::try_from(raw).map_err(|_| {
                    DemBuilderError::ParseError(format!("{kind} record offset out of range"))
                })
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        Vec::new()
    };

    let meas_ids = if let Some(meas_ids) = object.get("meas_ids") {
        let array = meas_ids.as_array().ok_or_else(|| {
            DemBuilderError::ParseError(format!("{kind} meas_ids must be an array"))
        })?;
        array
            .iter()
            .map(|meas_id| {
                let raw = meas_id.as_i64().ok_or_else(|| {
                    DemBuilderError::ParseError(format!("{kind} meas_ids must be integers"))
                })?;
                usize::try_from(raw).map_err(|_| {
                    DemBuilderError::ParseError(format!("{kind} meas_id out of range"))
                })
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        Vec::new()
    };

    if records.is_empty() && meas_ids.is_empty() {
        return Err(DemBuilderError::ParseError(format!(
            "{kind} entry has neither 'records' nor 'meas_ids'; it would \
             contribute nothing and silently weaken the DEM"
        )));
    }

    // `records` and `meas_ids` are alternative ways to reference the *same*
    // measurements, not additive. Co-presence is allowed but must be
    // redundant; that equality is enforced fail-loud in
    // `validate_metadata_refs` (which has the circuit context needed to
    // resolve `meas_ids`), not here at the pure-parse stage. The surface
    // `logical_circuit` path legitimately emits both (records = legacy Stim
    // offsets, meas_ids = the same measurements as stable ids).
    Ok((records, meas_ids))
}

fn is_two_qubit_noise_gate(gate_type: GateType) -> bool {
    matches!(
        gate_type,
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
    )
}

fn two_qubit_after_location_pairs(locations: &[DagSpacetimeLocation]) -> Vec<[usize; 2]> {
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (loc_idx, loc) in locations.iter().enumerate() {
        if is_two_qubit_noise_gate(loc.gate_type) && !loc.before {
            groups.entry(loc.node).or_default().push(loc_idx);
        }
    }

    groups
        .into_values()
        .flat_map(|loc_indices| {
            loc_indices
                .chunks_exact(2)
                .map(|pair| [pair[0], pair[1]])
                .collect::<Vec<_>>()
        })
        .collect()
}

impl ExactBranchReplayContext<'_> {
    fn replacement_branch_requests(
        self,
        locations: &[DagSpacetimeLocation],
    ) -> Result<Vec<ExactBranchReplayRequest>, DemBuilderError> {
        let mut requests = Vec::new();
        for [loc1_idx, loc2_idx] in two_qubit_after_location_pairs(locations) {
            let loc1 = &locations[loc1_idx];
            let loc2 = &locations[loc2_idx];
            if loc1.node != loc2.node {
                return Err(DemBuilderError::ConfigurationError(format!(
                    "exact_branch_replay expected paired two-qubit locations to share a node, got {} and {}",
                    loc1.node, loc2.node
                )));
            }
            if loc1.gate_type != loc2.gate_type {
                return Err(DemBuilderError::ConfigurationError(format!(
                    "exact_branch_replay expected paired two-qubit locations at node {} to share a gate type, got {:?} and {:?}",
                    loc1.node, loc1.gate_type, loc2.gate_type
                )));
            }
            let replacement = self.circuit.gate(loc1.node).ok_or_else(|| {
                DemBuilderError::ConfigurationError(format!(
                    "exact_branch_replay expected an original gate at node {}",
                    loc1.node
                ))
            })?;
            if !loc1
                .qubits
                .iter()
                .chain(loc2.qubits.iter())
                .all(|qubit| replacement.qubits.contains(qubit))
            {
                return Err(DemBuilderError::ConfigurationError(format!(
                    "exact_branch_replay location qubits at node {} are not all present in the omitted branch gate",
                    loc1.node
                )));
            }
            requests.push(ExactBranchReplayRequest {
                gate_node: loc1.node,
                gate_type: loc1.gate_type,
                loc_indices: [loc1_idx, loc2_idx],
            });
        }
        Ok(requests)
    }

    #[cfg(test)]
    fn omitted_branch_location_pair(
        self,
        request: ExactBranchReplayRequest,
        original_locations: &[DagSpacetimeLocation],
    ) -> Result<[usize; 2], DemBuilderError> {
        use crate::fault_tolerance::propagator::DagFaultAnalyzer;

        let branch = circuit_with_omitted_two_qubit_gate(self.circuit, request.gate_node)?;
        let branch_map = DagFaultAnalyzer::new(&branch).build_influence_map();
        identity_location_pair_for_request(request, original_locations, &branch_map.locations)
    }
}

fn omitted_branch_flips_measurement_parity_from_histories(
    request: ExactBranchReplayRequest,
    ideal_history: &MeasurementHistory,
    branch_history: &MeasurementHistory,
    measurement_indices: &[usize],
) -> Result<bool, DemBuilderError> {
    measurement_parity_differs_from_histories(
        ideal_history,
        branch_history,
        measurement_indices,
        &format!(
            "exact_branch_replay omitted gate at node {}",
            request.gate_node
        ),
    )
}

fn measurement_parity_differs_from_histories(
    ideal_history: &MeasurementHistory,
    branch_history: &MeasurementHistory,
    measurement_indices: &[usize],
    context: &str,
) -> Result<bool, DemBuilderError> {
    let ideal = measurement_parity_expression(ideal_history, measurement_indices, "ideal")?;
    let branch = measurement_parity_expression(branch_history, measurement_indices, "branch")?;
    if ideal.dependencies != branch.dependencies {
        return Err(DemBuilderError::ConfigurationError(format!(
            "{context} changes measurement dependencies for parity {measurement_indices:?}; this branch is not representable as a single deterministic DEM event"
        )));
    }
    Ok(ideal.flip ^ branch.flip)
}

fn measurement_parity_expression(
    history: &MeasurementHistory,
    measurement_indices: &[usize],
    history_label: &str,
) -> Result<MeasurementParityExpression, DemBuilderError> {
    let mut dependencies = BitSet::new();
    let mut flip = false;

    for &measurement_idx in measurement_indices {
        let result = history.get(measurement_idx).ok_or_else(|| {
            DemBuilderError::ConfigurationError(format!(
                "exact_branch_replay {history_label} history has no measurement {measurement_idx}"
            ))
        })?;
        dependencies.symmetric_difference_update(&result.outcome);
        flip ^= result.flip;
    }

    Ok(MeasurementParityExpression { dependencies, flip })
}

fn identity_location_pair_for_request(
    request: ExactBranchReplayRequest,
    original_locations: &[DagSpacetimeLocation],
    branch_locations: &[DagSpacetimeLocation],
) -> Result<[usize; 2], DemBuilderError> {
    let [orig_loc1, orig_loc2] = request.loc_indices;
    let expected_qubits = [
        *original_locations
            .get(orig_loc1)
            .and_then(|loc| loc.qubits.first())
            .ok_or_else(|| {
                DemBuilderError::ConfigurationError(format!(
                    "exact_branch_replay original location {orig_loc1} has no qubit"
                ))
            })?,
        *original_locations
            .get(orig_loc2)
            .and_then(|loc| loc.qubits.first())
            .ok_or_else(|| {
                DemBuilderError::ConfigurationError(format!(
                    "exact_branch_replay original location {orig_loc2} has no qubit"
                ))
            })?,
    ];

    let mut pair = [usize::MAX; 2];
    for (branch_idx, branch_loc) in branch_locations.iter().enumerate() {
        if branch_loc.node != request.gate_node
            || branch_loc.before
            || branch_loc.gate_type != GateType::I
        {
            continue;
        }
        if branch_loc.qubits.first() == Some(&expected_qubits[0]) {
            pair[0] = branch_idx;
        } else if branch_loc.qubits.first() == Some(&expected_qubits[1]) {
            pair[1] = branch_idx;
        }
    }

    if pair.contains(&usize::MAX) {
        return Err(DemBuilderError::ConfigurationError(format!(
            "exact_branch_replay could not find identity branch locations for omitted gate node {} on qubits {:?}",
            request.gate_node, expected_qubits
        )));
    }
    Ok(pair)
}

// ============================================================================
// Convenience: build DEM from circuit (free function to handle lifetimes)
// ============================================================================

/// Build a `DetectorErrorModel` from a `DagCircuit` and noise parameters.
///
/// Reads detector/DEM output definitions from circuit metadata attributes.
fn build_dem_from_circuit(
    circuit: &pecos_quantum::DagCircuit,
    noise: NoiseConfig,
) -> Result<DetectorErrorModel, DemBuilderError> {
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

    let builder = DemBuilder::new(&influence_map)
        .with_noise_config(noise)
        .with_exact_branch_replay_context(circuit);

    let builder = if let Some(ref dj) = det_json {
        builder.with_detectors_json(dj)?
    } else {
        builder
    };

    let builder = if let Some(ref oj) = obs_json {
        builder.with_observables_json(oj)?
    } else if !annotated_observable_records.is_empty() {
        builder.with_observable_records(annotated_observable_records)
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

    builder.try_build()
}

/// Return a branch circuit where one ideal two-qubit gate has been omitted.
///
/// Replacement-branch exact replay needs to evaluate "the hardware branch did
/// not apply this entangler" without disturbing the surrounding DAG wiring.
/// Replacing the selected node by batched identities preserves the node id,
/// qubit wires, and topological context while making the operation itself a
/// no-op on every qubit carried by the original gate.
fn circuit_with_omitted_two_qubit_gate(
    circuit: &pecos_quantum::DagCircuit,
    node: usize,
) -> Result<pecos_quantum::DagCircuit, DemBuilderError> {
    let original = circuit.gate(node).ok_or_else(|| {
        DemBuilderError::ConfigurationError(format!(
            "cannot omit gate at node {node}: no such gate node exists"
        ))
    })?;
    if !original.gate_type.is_two_qubit() {
        return Err(DemBuilderError::ConfigurationError(format!(
            "cannot omit gate at node {node}: {:?} is not a two-qubit gate",
            original.gate_type
        )));
    }

    let mut branch = circuit.clone();
    let replacement = pecos_core::Gate::simple(GateType::I, original.qubits.clone());
    *branch.gate_mut(node).expect("gate existed before clone") = replacement;
    Ok(branch)
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
// Tag-referenced detector resolution
// ============================================================================

/// Resolve `result_tags` on detector/observable JSON into record offsets.
///
/// `tag_to_ords` is the **sound** Guppy `result(tag, ...)` -> measurement
/// ordinal binding recovered structurally from the compiled HUGR
/// (reorder-immune; see `pecos_hugr_qis::result_tags`). Each referenced tag's
/// ordinals are mapped through `source_meas_ids` to stable runtime measurement
/// identities. `result_tags` is an alternative to `records`/`meas_ids` (not
/// additive): any co-present form must resolve to the same measurements.
///
/// Fail-loud (returns `Err`), never silently misbinds:
/// - **Loop guard**: if `static_meas_count != source_meas_ids.len()` the program
///   has un-unrolled runtime loops (the HUGR has one static measure op per
///   loop body), so per-occurrence tag binding is not statically available.
/// - An unknown tag, malformed `result_tags`, or invalid JSON is an error.
///
/// # Errors
/// Returns [`DemBuilderError::ParseError`] on the loop guard, an unknown tag,
/// malformed `result_tags`, or invalid JSON.
pub fn resolve_result_tags(
    detectors_json: &str,
    observables_json: &str,
    tag_to_ords: &std::collections::BTreeMap<String, Vec<Option<usize>>>,
    static_meas_count: usize,
    source_meas_ids: &[usize],
    runtime_meas_ids: &[usize],
) -> Result<(String, String), DemBuilderError> {
    if static_meas_count != source_meas_ids.len() {
        return Err(DemBuilderError::ParseError(format!(
            "result_tags (tag-referenced detectors) is not supported for Guppy \
             programs with runtime loops: the HUGR has {static_meas_count} \
             static measurement op(s) but the traced program emits \
             {} measurement(s). Per-occurrence tag binding is \
             not statically available; use positional records.",
            source_meas_ids.len()
        )));
    }
    let source_set = source_meas_ids
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let runtime_set = runtime_meas_ids
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    if source_set.len() != source_meas_ids.len()
        || runtime_set.len() != runtime_meas_ids.len()
        || source_set != runtime_set
    {
        return Err(DemBuilderError::ParseError(
            "source and runtime measurement identities must be unique and describe the same set"
                .to_string(),
        ));
    }

    let rewrite = |json: &str, kind: &str| -> Result<String, DemBuilderError> {
        if json.trim().is_empty() {
            return Ok(json.to_string());
        }
        let mut value: serde_json::Value = serde_json::from_str(json).map_err(|e| {
            DemBuilderError::ParseError(format!("invalid detector/observable JSON: {e}"))
        })?;
        let Some(entries) = value.as_array_mut() else {
            return Ok(json.to_string());
        };
        for entry in entries.iter_mut() {
            let Some(obj) = entry.as_object_mut() else {
                continue;
            };
            let Some(tags) = obj.remove("result_tags") else {
                continue;
            };

            // Resolve `result_tags` strictly into a list of record offsets.
            let tag_list = tags.as_array().ok_or_else(|| {
                DemBuilderError::ParseError(
                    "result_tags must be a JSON array of strings".to_string(),
                )
            })?;
            let mut tag_meas_ids: Vec<usize> = Vec::new();
            for tag in tag_list {
                let tag = tag.as_str().ok_or_else(|| {
                    DemBuilderError::ParseError("result_tags entries must be strings".to_string())
                })?;
                let ords = tag_to_ords.get(tag).ok_or_else(|| {
                    DemBuilderError::ParseError(format!(
                        "{kind} references result_tag {tag:?}, which the Guppy \
                         program never records via result(...)"
                    ))
                })?;
                for (occurrence, ord) in ords.iter().enumerate() {
                    let ord = ord.ok_or_else(|| {
                        DemBuilderError::ParseError(format!(
                            "{kind} references result_tag {tag:?}, but occurrence {occurrence} is not a direct scalar measurement result"
                        ))
                    })?;
                    let meas_id = source_meas_ids.get(ord).ok_or_else(|| {
                        DemBuilderError::ParseError(format!(
                            "compiler measurement ordinal {ord} is outside the traced source measurement stream"
                        ))
                    })?;
                    tag_meas_ids.push(*meas_id);
                }
            }

            // `result_tags` is an *alternative* to `records` (and `meas_ids`),
            // following the same redundancy discipline as records-vs-meas_ids:
            // co-presence is allowed only when the two forms reference the
            // *same* measurements (sorted-set equality). Additive merging
            // would either silently weaken the DEM (when callers expected
            // alternatives) or corrupt parity by double-referencing (when
            // they were actually redundant).
            let mut existing_meas_ids: Option<Vec<usize>> = None;
            if let Some(meas_ids_value) = obj.get("meas_ids") {
                let meas_ids_array = meas_ids_value.as_array().ok_or_else(|| {
                    DemBuilderError::ParseError(format!(
                        "{kind} meas_ids must be a JSON array of non-negative integers"
                    ))
                })?;
                let mut parsed = Vec::with_capacity(meas_ids_array.len());
                for meas_id in meas_ids_array {
                    let meas_id = meas_id
                        .as_u64()
                        .and_then(|value| usize::try_from(value).ok())
                        .ok_or_else(|| {
                            DemBuilderError::ParseError(format!(
                                "{kind} meas_ids entries must be non-negative integers"
                            ))
                        })?;
                    parsed.push(meas_id);
                }
                existing_meas_ids = Some(parsed);
            }
            if let Some(records_value) = obj.get("records") {
                let records_array = records_value.as_array().ok_or_else(|| {
                    DemBuilderError::ParseError(format!(
                        "{kind} records must be a JSON array of integers"
                    ))
                })?;
                let mut from_records: Vec<usize> = Vec::with_capacity(records_array.len());
                let runtime_len = i64::try_from(runtime_meas_ids.len()).map_err(|_| {
                    DemBuilderError::ParseError("traced measurement count too large".to_string())
                })?;
                for rec in records_array {
                    let r = rec.as_i64().ok_or_else(|| {
                        DemBuilderError::ParseError(format!(
                            "{kind} records entries must be integers"
                        ))
                    })?;
                    let index = runtime_len + r;
                    let index = usize::try_from(index)
                        .ok()
                        .filter(|&value| value < runtime_meas_ids.len())
                        .ok_or_else(|| {
                            DemBuilderError::ParseError(format!(
                                "{kind} record offset {r} is out of range for {} measurements",
                                runtime_meas_ids.len()
                            ))
                        })?;
                    from_records.push(runtime_meas_ids[index]);
                }
                if let Some(existing) = &existing_meas_ids {
                    let mut a = existing.clone();
                    let mut b = from_records.clone();
                    a.sort_unstable();
                    b.sort_unstable();
                    if a != b {
                        return Err(DemBuilderError::ParseError(format!(
                            "{kind} records and meas_ids reference different measurements"
                        )));
                    }
                }
                existing_meas_ids = Some(from_records);
            }

            if let Some(existing) = existing_meas_ids {
                let mut a = existing;
                let mut b = tag_meas_ids;
                a.sort_unstable();
                b.sort_unstable();
                if a != b {
                    return Err(DemBuilderError::ParseError(format!(
                        "{kind} entry has 'records'/'meas_ids' alongside 'result_tags' but \
                             they reference different measurements (existing meas_ids {a:?}, \
                             result_tags resolve to meas_ids {b:?}); they are alternatives, \
                             not additive -- provide one, or make them redundant"
                    )));
                }
            } else {
                obj.insert(
                    "meas_ids".to_string(),
                    serde_json::Value::Array(
                        tag_meas_ids
                            .into_iter()
                            .map(serde_json::Value::from)
                            .collect(),
                    ),
                );
            }
        }
        serde_json::to_string(&value)
            .map_err(|e| DemBuilderError::ParseError(format!("failed to re-serialize JSON: {e}")))
    };

    Ok((
        rewrite(detectors_json, "Detector")?,
        rewrite(observables_json, "Observable")?,
    ))
}

// ============================================================================
// Error Type
// ============================================================================

/// Errors that can occur during DEM building.
#[derive(Debug, Clone)]
pub enum DemBuilderError {
    /// JSON parsing error.
    ParseError(String),
    /// Invalid DEM builder configuration.
    ConfigurationError(String),
}

impl std::fmt::Display for DemBuilderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParseError(msg) => write!(f, "DEM builder parse error: {msg}"),
            Self::ConfigurationError(msg) => write!(f, "DEM builder configuration error: {msg}"),
        }
    }
}

impl std::error::Error for DemBuilderError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_szz_source_frame_components_pull_post_error_to_pre_generators() {
        fn dets(indices: &[u32]) -> FaultMechanism {
            FaultMechanism::from_unsorted(indices.iter().copied(), std::iter::empty())
        }

        let a = dets(&[0, 1]);
        let b = dets(&[2]);
        let c = dets(&[3, 4]);

        let mut effects: [[FaultMechanism; 4]; 4] = Default::default();
        effects[2][3] = a.clone(); // SZZ maps pre X1 to post YZ.
        effects[3][0] = b.clone(); // SZZ maps pre Z1 to post ZI.
        effects[0][3] = c.clone(); // SZZ maps pre Z2 to post IZ.
        effects[1][0] = a.xor(&b).xor(&c);

        let parts =
            DemBuilder::two_qubit_clifford_source_frame_components(GateType::SZZ, 1, 0, &effects)
                .expect("post XI should pull back through SZZ to pre YZ");

        assert_eq!(parts.len(), 3);
        assert!(parts.contains(&a));
        assert!(parts.contains(&b));
        assert!(parts.contains(&c));
        assert_eq!(
            parts
                .iter()
                .fold(FaultMechanism::new(), |acc, part| acc.xor(part)),
            effects[1][0]
        );

        effects[3][0] = a.clone();
        effects[1][0] = c.clone();

        let parts =
            DemBuilder::two_qubit_clifford_source_frame_components(GateType::SZZ, 1, 0, &effects)
                .expect("duplicate source components should cancel by XOR");

        assert_eq!(parts.as_slice(), &[c]);
    }

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
        let round_tripped =
            TickCircuit::from(&DagCircuit::try_from(&circuit).expect("valid circuit"));
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
        assert!(detectors[0].meas_ids.is_empty());
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
        assert!(observables[0].meas_ids.is_empty());
    }

    #[test]
    fn test_parse_json_accepts_meas_ids() {
        let detectors = parse_detectors_json(r#"[{"id": 0, "meas_ids": [0, 2]}]"#).unwrap();
        assert_eq!(detectors[0].records, Vec::<i32>::new());
        assert_eq!(detectors[0].meas_ids, vec![0, 2]);

        let observables =
            parse_observables_json(r#"[{"observable_id": 1, "meas_ids": [3]}]"#).unwrap();
        assert_eq!(observables[0].records, Vec::<i32>::new());
        assert_eq!(observables[0].meas_ids, vec![3]);
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
    fn test_dem_builder_resolves_meas_ids_when_records_are_absent() {
        let influence_map = DagFaultInfluenceMap::with_capacity(0);
        let dem = DemBuilder::new(&influence_map)
            .with_detectors_json(r#"[{"id": 0, "meas_ids": [0, 2]}]"#)
            .unwrap()
            .with_observables_json(r#"[{"id": 0, "meas_ids": [1]}]"#)
            .unwrap()
            .with_num_measurements(3)
            .build();

        assert_eq!(dem.detectors[0].records.as_slice(), &[-3, -1]);
        assert_eq!(dem.dem_outputs()[0].records.as_slice(), &[-2]);
    }

    #[test]
    fn test_try_build_rejects_out_of_range_record_and_meas_id() {
        let influence_map = DagFaultInfluenceMap::with_capacity(0);

        let bad_record = DemBuilder::new(&influence_map)
            .with_detectors_json(r#"[{"id": 0, "records": [-2]}]"#)
            .unwrap()
            .with_num_measurements(1)
            .try_build();
        assert!(
            bad_record.is_err(),
            "out-of-range record must fail try_build"
        );

        let bad_meas_id = DemBuilder::new(&influence_map)
            .with_detectors_json(r#"[{"id": 0, "meas_ids": [999]}]"#)
            .unwrap()
            .with_num_measurements(1)
            .try_build();
        assert!(
            bad_meas_id.is_err(),
            "out-of-range meas_id must fail try_build"
        );

        // The infallible `build` stays lax for the decoupled/raw case so
        // existing pass-through callers are unaffected.
        let _ = DemBuilder::new(&influence_map)
            .with_observables_json(r#"[{"id": 0, "records": [-1, -3]}]"#)
            .unwrap()
            .build();

        // Empty influence map keeps the escape hatch: a declared count with
        // no real measurements is allowed (opaque pass-through coordinates).
        assert!(
            DemBuilder::new(&influence_map)
                .with_detectors_json(r#"[{"id": 0, "meas_ids": [0, 2]}]"#)
                .unwrap()
                .with_num_measurements(3)
                .try_build()
                .is_ok(),
            "empty influence map must keep the declarative-count escape hatch"
        );
    }

    #[test]
    fn test_try_build_rejects_exact_branch_replay_without_provider() {
        use crate::fault_tolerance::dem_builder::PauliWeights;
        use pecos_core::pauli::Z;

        let influence_map = DagFaultInfluenceMap::with_capacity(0);
        let noise = NoiseConfig::new(0.0, 0.01, 0.0, 0.0)
            .set_p2_weights(PauliWeights::with_replacement([], [(Z(0) & Z(1), 1.0)]))
            .set_p2_replacement_approximation(ReplacementBranchApproximation::ExactBranchReplay);

        let err = DemBuilder::new(&influence_map)
            .with_noise_config(noise)
            .try_build()
            .expect_err("exact branch replay must fail loud without an exact provider");

        assert!(matches!(err, DemBuilderError::ConfigurationError(_)));
        assert!(
            err.to_string()
                .contains("circuit-aware exact branch provider"),
            "unexpected error: {err}",
        );
    }

    #[test]
    fn test_from_circuit_exact_branch_replay_emits_omitted_gate_effect() {
        use crate::fault_tolerance::dem_builder::PauliWeights;
        use pecos_core::PauliString;
        use pecos_quantum::{Attribute, DagCircuit};

        let mut circuit = DagCircuit::new();
        circuit.pz(&[0, 1]);
        circuit.x(&[0]);
        circuit.cx(&[(0, 1)]);
        circuit.mz(&[1]);
        circuit.set_attr("num_measurements", Attribute::String("1".to_string()));
        circuit.set_attr(
            "detectors",
            Attribute::String(r#"[{"id":0,"records":[-1]}]"#.to_string()),
        );

        let noise = NoiseConfig::new(0.0, 0.01, 0.0, 0.0)
            .set_p2_weights(PauliWeights::with_replacement(
                [],
                [(PauliString::identity(), 1.0)],
            ))
            .set_p2_replacement_approximation(ReplacementBranchApproximation::ExactBranchReplay);

        let dem = DemBuilder::try_from_circuit_with_noise_config(&circuit, noise)
            .expect("exact branch replay should emit representable branch effects");

        let contributions = dem.contributions_for_effect(&[0], &[]);
        assert_eq!(
            contributions.len(),
            1,
            "all effects:\n{}",
            dem.all_contribution_effects()
        );
        assert!((contributions[0].probability - 0.01).abs() < 1.0e-12);
        assert!(contributions[0].replacement_branch);
        assert_eq!(
            contributions[0].direct_source_family,
            Some(DirectSourceFamily::TwoLocationExactReplacementBranch)
        );
        assert!(
            contributions[0]
                .paulis
                .iter()
                .all(|pauli| *pauli == Pauli::I),
            "omission-only replacement branch should be recorded as *II"
        );
        let (base_effect, branch_pauli_effect) = contributions[0]
            .direct_component_effects()
            .expect("exact branch replay should preserve base/branch components");
        assert_eq!(
            base_effect.xor(&branch_pauli_effect),
            contributions[0].effect
        );
        assert_eq!(base_effect.detectors.as_slice(), &[0]);
        assert!(branch_pauli_effect.is_empty());
    }

    fn single_qubit_local_crosstalk_circuit(pre_payload_h: bool) -> pecos_quantum::DagCircuit {
        use pecos_core::{Gate, QubitId};
        use pecos_quantum::{Attribute, DagCircuit};

        let mut circuit = DagCircuit::new();
        circuit.add_gate_auto_wire(Gate::pz(&[QubitId(0)]));
        if pre_payload_h {
            circuit.add_gate_auto_wire(Gate::h(&[QubitId(0)]));
        }
        circuit.add_gate_auto_wire(Gate::meas_crosstalk_local_payload(&[QubitId(0)]));
        circuit.add_gate_auto_wire(Gate::mz(&[QubitId(0)]));
        circuit.set_attr("num_measurements", Attribute::String("1".to_string()));
        circuit.set_attr(
            "detectors",
            Attribute::String(r#"[{"id":0,"records":[-1]}]"#.to_string()),
        );
        circuit
    }

    fn single_qubit_no_crosstalk_payload_circuit() -> pecos_quantum::DagCircuit {
        use pecos_core::{Gate, QubitId};
        use pecos_quantum::{Attribute, DagCircuit};

        let mut circuit = DagCircuit::new();
        circuit.add_gate_auto_wire(Gate::pz(&[QubitId(0)]));
        circuit.add_gate_auto_wire(Gate::mz(&[QubitId(0)]));
        circuit.set_attr("num_measurements", Attribute::String("1".to_string()));
        circuit.set_attr(
            "detectors",
            Attribute::String(r#"[{"id":0,"records":[-1]}]"#.to_string()),
        );
        circuit
    }

    fn two_qubit_global_crosstalk_circuit() -> pecos_quantum::DagCircuit {
        use pecos_core::{Gate, QubitId};
        use pecos_quantum::{Attribute, DagCircuit};

        let mut circuit = DagCircuit::new();
        circuit.add_gate_auto_wire(Gate::pz(&[QubitId(0)]));
        circuit.add_gate_auto_wire(Gate::pz(&[QubitId(1)]));
        circuit.add_gate_auto_wire(Gate::meas_crosstalk_global_payload(&[QubitId(0)]));
        circuit.add_gate_auto_wire(Gate::mz(&[QubitId(1)]));
        circuit.set_attr("num_measurements", Attribute::String("1".to_string()));
        circuit.set_attr(
            "detectors",
            Attribute::String(r#"[{"id":0,"records":[-1]}]"#.to_string()),
        );
        circuit
    }

    fn two_qubit_global_crosstalk_random_victim_circuit() -> pecos_quantum::DagCircuit {
        use pecos_core::{Gate, QubitId};
        use pecos_quantum::{Attribute, DagCircuit};

        let mut circuit = DagCircuit::new();
        circuit.add_gate_auto_wire(Gate::pz(&[QubitId(0)]));
        circuit.add_gate_auto_wire(Gate::pz(&[QubitId(1)]));
        circuit.add_gate_auto_wire(Gate::h(&[QubitId(1)]));
        circuit.add_gate_auto_wire(Gate::meas_crosstalk_global_payload(&[QubitId(0)]));
        circuit.add_gate_auto_wire(Gate::mz(&[QubitId(1)]));
        circuit.set_attr("num_measurements", Attribute::String("1".to_string()));
        circuit.set_attr(
            "detectors",
            Attribute::String(r#"[{"id":0,"records":[-1]}]"#.to_string()),
        );
        circuit
    }

    #[test]
    fn test_exact_deterministic_local_measurement_crosstalk_emits_dem_source() {
        use crate::fault_tolerance::dem_builder::MeasurementCrosstalkTransitionModel;

        let circuit = single_qubit_local_crosstalk_circuit(false);
        let noise = NoiseConfig::new(0.0, 0.0, 0.0, 0.0)
            .set_measurement_crosstalk_local_rate(0.25)
            .set_measurement_crosstalk_transition_model(
                MeasurementCrosstalkTransitionModel::bit_flip(0.4, 0.0),
            )
            .set_measurement_crosstalk_dem_mode(MeasurementCrosstalkDemMode::ExactDeterministic);

        let dem = DemBuilder::try_from_circuit_with_noise_config(&circuit, noise)
            .expect("deterministic local measurement crosstalk should be representable");

        let contributions = dem.contributions_for_effect(&[0], &[]);
        assert_eq!(
            contributions.len(),
            1,
            "all effects:\n{}",
            dem.all_contribution_effects()
        );
        assert!((contributions[0].probability - 0.1).abs() < 1.0e-12);
        assert_eq!(
            contributions[0].direct_source_family,
            Some(DirectSourceFamily::MeasurementCrosstalk)
        );
        assert_eq!(
            contributions[0].source_gate_types.as_slice(),
            &[GateType::MeasCrosstalkLocalPayload]
        );
        assert_eq!(contributions[0].paulis.as_slice(), &[Pauli::X]);
    }

    #[test]
    fn test_exact_deterministic_global_measurement_crosstalk_emits_victim_dem_source() {
        use crate::fault_tolerance::dem_builder::MeasurementCrosstalkTransitionModel;

        let circuit = two_qubit_global_crosstalk_circuit();
        let noise = NoiseConfig::new(0.0, 0.0, 0.0, 0.0)
            .set_measurement_crosstalk_global_rate(0.25)
            .set_measurement_crosstalk_transition_model(
                MeasurementCrosstalkTransitionModel::bit_flip(0.4, 0.0),
            )
            .set_measurement_crosstalk_dem_mode(MeasurementCrosstalkDemMode::ExactDeterministic);

        let dem = DemBuilder::try_from_circuit_with_noise_config(&circuit, noise)
            .expect("deterministic global measurement crosstalk should be representable");

        let contributions = dem.contributions_for_effect(&[0], &[]);
        assert_eq!(
            contributions.len(),
            1,
            "all effects:\n{}",
            dem.all_contribution_effects()
        );
        assert!((contributions[0].probability - 0.1).abs() < 1.0e-12);
        assert_eq!(
            contributions[0].direct_source_family,
            Some(DirectSourceFamily::MeasurementCrosstalk)
        );
        assert_eq!(
            contributions[0].source_gate_types.as_slice(),
            &[GateType::MeasCrosstalkGlobalPayload]
        );
        assert_eq!(contributions[0].paulis.as_slice(), &[Pauli::X]);
    }

    #[test]
    fn test_exact_deterministic_local_measurement_crosstalk_requires_payloads() {
        use crate::fault_tolerance::dem_builder::MeasurementCrosstalkTransitionModel;

        let circuit = single_qubit_no_crosstalk_payload_circuit();
        let noise = NoiseConfig::new(0.0, 0.0, 0.0, 0.0)
            .set_measurement_crosstalk_local_rate(0.25)
            .set_measurement_crosstalk_transition_model(
                MeasurementCrosstalkTransitionModel::bit_flip(0.4, 0.0),
            )
            .set_measurement_crosstalk_dem_mode(MeasurementCrosstalkDemMode::ExactDeterministic);

        let err = DemBuilder::try_from_circuit_with_noise_config(&circuit, noise)
            .expect_err("positive local crosstalk rate without payloads must fail loudly");

        assert!(matches!(err, DemBuilderError::ConfigurationError(_)));
        assert!(
            err.to_string()
                .contains("no MeasCrosstalkLocalPayload locations"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_exact_deterministic_global_measurement_crosstalk_requires_payloads() {
        use crate::fault_tolerance::dem_builder::MeasurementCrosstalkTransitionModel;

        let circuit = single_qubit_no_crosstalk_payload_circuit();
        let noise = NoiseConfig::new(0.0, 0.0, 0.0, 0.0)
            .set_measurement_crosstalk_global_rate(0.25)
            .set_measurement_crosstalk_transition_model(
                MeasurementCrosstalkTransitionModel::bit_flip(0.4, 0.0),
            )
            .set_measurement_crosstalk_dem_mode(MeasurementCrosstalkDemMode::ExactDeterministic);

        let err = DemBuilder::try_from_circuit_with_noise_config(&circuit, noise)
            .expect_err("positive global crosstalk rate without payloads must fail loudly");

        assert!(matches!(err, DemBuilderError::ConfigurationError(_)));
        assert!(
            err.to_string()
                .contains("no MeasCrosstalkGlobalPayload locations"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_exact_deterministic_local_measurement_crosstalk_identity_transition_is_empty() {
        use crate::fault_tolerance::dem_builder::MeasurementCrosstalkTransitionModel;

        let circuit = single_qubit_local_crosstalk_circuit(false);
        let noise = NoiseConfig::new(0.0, 0.0, 0.0, 0.0)
            .set_measurement_crosstalk_local_rate(0.25)
            .set_measurement_crosstalk_transition_model(
                MeasurementCrosstalkTransitionModel::bit_flip(0.0, 0.0),
            )
            .set_measurement_crosstalk_dem_mode(MeasurementCrosstalkDemMode::ExactDeterministic);

        let dem = DemBuilder::try_from_circuit_with_noise_config(&circuit, noise)
            .expect("identity local measurement crosstalk should be representable");

        assert_eq!(dem.num_contributions(), 0);
    }

    #[test]
    fn test_exact_deterministic_local_measurement_crosstalk_rejects_hidden_randomness() {
        use crate::fault_tolerance::dem_builder::MeasurementCrosstalkTransitionModel;

        let circuit = single_qubit_local_crosstalk_circuit(true);
        let noise = NoiseConfig::new(0.0, 0.0, 0.0, 0.0)
            .set_measurement_crosstalk_local_rate(0.25)
            .set_measurement_crosstalk_transition_model(
                MeasurementCrosstalkTransitionModel::bit_flip(0.4, 0.0),
            )
            .set_measurement_crosstalk_dem_mode(MeasurementCrosstalkDemMode::ExactDeterministic);

        let err = DemBuilder::try_from_circuit_with_noise_config(&circuit, noise)
            .expect_err("nondeterministic hidden measurement must fail loudly");

        assert!(matches!(err, DemBuilderError::ConfigurationError(_)));
        assert!(
            err.to_string()
                .contains("state-independent hidden MZ result"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_averaged_global_measurement_crosstalk_accepts_hidden_randomness() {
        use crate::fault_tolerance::dem_builder::MeasurementCrosstalkTransitionModel;

        let circuit = two_qubit_global_crosstalk_random_victim_circuit();
        let noise = NoiseConfig::new(0.0, 0.0, 0.0, 0.0)
            .set_measurement_crosstalk_global_rate(0.25)
            .set_measurement_crosstalk_transition_model(MeasurementCrosstalkTransitionModel {
                p_0_to_1: 0.4,
                p_0_to_leak: 0.2,
                p_1_to_0: 0.0,
                p_1_to_leak: 0.0,
            })
            .set_measurement_crosstalk_dem_mode(
                MeasurementCrosstalkDemMode::AveragedHiddenLeakageAsDepolarizing,
            );

        let dem = DemBuilder::try_from_circuit_with_noise_config(&circuit, noise)
            .expect("averaged global leak2depolar crosstalk should handle random hidden MZ");

        let contributions = dem.contributions_for_effect(&[0], &[]);
        assert_eq!(
            contributions.len(),
            2,
            "all effects:\n{}",
            dem.all_contribution_effects()
        );
        assert!(contributions.iter().any(|contribution| {
            contribution.paulis.as_slice() == [Pauli::X]
                && (contribution.probability - 0.05625).abs() < 1.0e-12
        }));
        assert!(contributions.iter().any(|contribution| {
            contribution.paulis.as_slice() == [Pauli::Y]
                && (contribution.probability - 0.00625).abs() < 1.0e-12
        }));
        assert!(contributions.iter().all(|contribution| {
            contribution.direct_source_family == Some(DirectSourceFamily::MeasurementCrosstalk)
                && contribution.source_gate_types.as_slice()
                    == [GateType::MeasCrosstalkGlobalPayload]
        }));
    }

    #[test]
    fn test_exact_deterministic_local_measurement_crosstalk_rejects_leakage() {
        use crate::fault_tolerance::dem_builder::MeasurementCrosstalkTransitionModel;

        let circuit = single_qubit_local_crosstalk_circuit(false);
        let noise = NoiseConfig::new(0.0, 0.0, 0.0, 0.0)
            .set_measurement_crosstalk_local_rate(0.25)
            .set_measurement_crosstalk_transition_model(MeasurementCrosstalkTransitionModel {
                p_0_to_1: 0.4,
                p_0_to_leak: 0.2,
                p_1_to_0: 0.0,
                p_1_to_leak: 0.0,
            })
            .set_measurement_crosstalk_dem_mode(MeasurementCrosstalkDemMode::ExactDeterministic);

        let err = DemBuilder::try_from_circuit_with_noise_config(&circuit, noise)
            .expect_err("plain exact deterministic crosstalk should reject leakage");

        assert!(matches!(err, DemBuilderError::ConfigurationError(_)));
        assert!(
            err.to_string().contains("leakage transitions"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_exact_deterministic_local_measurement_crosstalk_leakage_as_depolarizing() {
        use crate::fault_tolerance::dem_builder::MeasurementCrosstalkTransitionModel;

        let circuit = single_qubit_local_crosstalk_circuit(false);
        let noise = NoiseConfig::new(0.0, 0.0, 0.0, 0.0)
            .set_measurement_crosstalk_local_rate(0.25)
            .set_measurement_crosstalk_transition_model(MeasurementCrosstalkTransitionModel {
                p_0_to_1: 0.4,
                p_0_to_leak: 0.2,
                p_1_to_0: 0.0,
                p_1_to_leak: 0.0,
            })
            .set_measurement_crosstalk_dem_mode(
                MeasurementCrosstalkDemMode::ExactDeterministicLeakageAsDepolarizing,
            );

        let dem = DemBuilder::try_from_circuit_with_noise_config(&circuit, noise)
            .expect("deterministic leak2depolar local crosstalk should be representable");

        let contributions = dem.contributions_for_effect(&[0], &[]);
        assert_eq!(contributions.len(), 2);
        let total_probability: f64 = contributions
            .iter()
            .map(|contribution| contribution.probability)
            .sum();
        assert!((total_probability - 0.125).abs() < 1.0e-12);
        assert!(contributions.iter().any(|contribution| {
            contribution.paulis.as_slice() == [Pauli::X]
                && (contribution.probability - 0.1125).abs() < 1.0e-12
        }));
        assert!(contributions.iter().any(|contribution| {
            contribution.paulis.as_slice() == [Pauli::Y]
                && (contribution.probability - 0.0125).abs() < 1.0e-12
        }));
        assert!(contributions.iter().all(|contribution| {
            contribution.source_gate_types.as_slice() == [GateType::MeasCrosstalkLocalPayload]
        }));
    }

    #[test]
    fn test_from_circuit_exact_branch_replay_skips_empty_replacement_effect() {
        use crate::fault_tolerance::dem_builder::PauliWeights;
        use pecos_core::pauli::X;
        use pecos_quantum::{Attribute, DagCircuit};

        let mut circuit = DagCircuit::new();
        circuit.pz(&[0, 1]);
        circuit.x(&[0]);
        circuit.cx(&[(0, 1)]);
        circuit.mz(&[1]);
        circuit.set_attr("num_measurements", Attribute::String("1".to_string()));
        circuit.set_attr(
            "detectors",
            Attribute::String(r#"[{"id":0,"records":[-1]}]"#.to_string()),
        );

        let noise = NoiseConfig::new(0.0, 0.01, 0.0, 0.0)
            .set_p2_weights(PauliWeights::with_replacement([], [(X(1), 1.0)]))
            .set_p2_replacement_approximation(ReplacementBranchApproximation::ExactBranchReplay);

        let dem = DemBuilder::try_from_circuit_with_noise_config(&circuit, noise)
            .expect("exact branch replay should allow empty representable effects");

        assert_eq!(dem.num_contributions(), 0);
    }

    #[test]
    fn test_from_circuit_exact_branch_replay_rejects_dependency_changing_branch() {
        use crate::fault_tolerance::dem_builder::PauliWeights;
        use pecos_core::PauliString;
        use pecos_quantum::{Attribute, DagCircuit};

        let mut circuit = DagCircuit::new();
        circuit.pz(&[0, 1]);
        circuit.h(&[0]);
        circuit.cx(&[(0, 1)]);
        circuit.mz(&[1]);
        circuit.set_attr("num_measurements", Attribute::String("1".to_string()));
        circuit.set_attr(
            "detectors",
            Attribute::String(r#"[{"id":0,"records":[-1]}]"#.to_string()),
        );

        let noise = NoiseConfig::new(0.0, 0.01, 0.0, 0.0)
            .set_p2_weights(PauliWeights::with_replacement(
                [],
                [(PauliString::identity(), 1.0)],
            ))
            .set_p2_replacement_approximation(ReplacementBranchApproximation::ExactBranchReplay);

        let err = DemBuilder::try_from_circuit_with_noise_config(&circuit, noise)
            .expect_err("dependency-changing replacement branches must stay fail-loud");

        assert!(
            err.to_string().contains("not representable"),
            "unexpected error: {err}",
        );
    }

    #[test]
    fn test_circuit_with_omitted_two_qubit_gate_preserves_wiring() {
        use pecos_core::{Gate, QubitId};
        use pecos_quantum::DagCircuit;

        let mut circuit = DagCircuit::new();
        let prep0 = circuit.add_gate_auto_wire(Gate::pz(&[QubitId(0)]));
        let prep1 = circuit.add_gate_auto_wire(Gate::pz(&[QubitId(1)]));
        let entangler = circuit.add_gate_auto_wire(Gate::szz(&[(QubitId(0), QubitId(1))]));
        let meas0 = circuit.add_gate_auto_wire(Gate::mz(&[QubitId(0)]));
        let meas1 = circuit.add_gate_auto_wire(Gate::mz(&[QubitId(1)]));

        let branch = circuit_with_omitted_two_qubit_gate(&circuit, entangler)
            .expect("two-qubit entangler can be omitted");

        assert_eq!(circuit.gate(entangler).unwrap().gate_type, GateType::SZZ);
        let replacement = branch.gate(entangler).unwrap();
        assert_eq!(replacement.gate_type, GateType::I);
        assert_eq!(replacement.qubits.as_slice(), &[QubitId(0), QubitId(1)]);

        assert_eq!(
            branch.predecessor_on_qubit(entangler, QubitId(0)),
            Some(prep0)
        );
        assert_eq!(
            branch.predecessor_on_qubit(entangler, QubitId(1)),
            Some(prep1)
        );
        assert_eq!(
            branch.successor_on_qubit(entangler, QubitId(0)),
            Some(meas0)
        );
        assert_eq!(
            branch.successor_on_qubit(entangler, QubitId(1)),
            Some(meas1)
        );
        assert_eq!(branch.topological_order(), circuit.topological_order());
    }

    #[test]
    fn test_exact_branch_replay_context_collects_two_qubit_requests() {
        use crate::fault_tolerance::propagator::DagFaultAnalyzer;
        use pecos_core::{Gate, QubitId};
        use pecos_quantum::DagCircuit;

        let mut circuit = DagCircuit::new();
        circuit.add_gate_auto_wire(Gate::pz(&[QubitId(0)]));
        circuit.add_gate_auto_wire(Gate::pz(&[QubitId(1)]));
        let entangler = circuit.add_gate_auto_wire(Gate::szz(&[(QubitId(0), QubitId(1))]));
        circuit.add_gate_auto_wire(Gate::mz(&[QubitId(0)]));
        circuit.add_gate_auto_wire(Gate::mz(&[QubitId(1)]));

        let influence_map = DagFaultAnalyzer::new(&circuit).build_influence_map();
        let requests = ExactBranchReplayContext { circuit: &circuit }
            .replacement_branch_requests(&influence_map.locations)
            .expect("two-qubit branch requests should be recoverable");

        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].gate_node, entangler);
        assert_eq!(requests[0].gate_type, GateType::SZZ);
        let branch_pair = (ExactBranchReplayContext { circuit: &circuit })
            .omitted_branch_location_pair(requests[0], &influence_map.locations)
            .expect("omitted branch identity locations should be recoverable");
        assert_ne!(branch_pair[0], branch_pair[1]);

        let loc_qubits: Vec<_> = requests[0]
            .loc_indices
            .iter()
            .flat_map(|&idx| influence_map.locations[idx].qubits.iter().copied())
            .collect();
        assert_eq!(loc_qubits, vec![QubitId(0), QubitId(1)]);
    }

    #[test]
    fn test_exact_branch_replay_base_effect_detects_deterministic_omitted_gate_flip() {
        use crate::fault_tolerance::propagator::DagFaultAnalyzer;
        use pecos_quantum::DagCircuit;

        let mut circuit = DagCircuit::new();
        circuit.pz(&[0, 1]);
        circuit.x(&[0]);
        let entangler = circuit.add_gate_auto_wire(pecos_core::Gate::cx(&[(0, 1)]));
        circuit.mz(&[1]);

        let influence_map = DagFaultAnalyzer::new(&circuit).build_influence_map();
        let context = ExactBranchReplayContext { circuit: &circuit };
        let request = context
            .replacement_branch_requests(&influence_map.locations)
            .unwrap()
            .into_iter()
            .find(|request| request.gate_node == entangler)
            .expect("CX request should be present");

        let builder = DemBuilder::new(&influence_map)
            .with_detectors_json(r#"[{"id":0,"records":[-1]}]"#)
            .unwrap()
            .with_num_measurements(1);
        let effect = builder
            .exact_omitted_branch_base_effect(context, request)
            .expect("omitting this CX only flips a deterministic measurement parity");

        assert_eq!(effect.detectors.as_slice(), &[0]);
        assert!(effect.dem_outputs.is_empty());
    }

    #[test]
    fn test_exact_branch_replay_replacement_pauli_combines_with_omitted_gate_effect() {
        use crate::fault_tolerance::propagator::DagFaultAnalyzer;
        use pecos_quantum::DagCircuit;

        let mut circuit = DagCircuit::new();
        circuit.pz(&[0, 1]);
        circuit.x(&[0]);
        let entangler = circuit.add_gate_auto_wire(pecos_core::Gate::cx(&[(0, 1)]));
        circuit.mz(&[1]);

        let influence_map = DagFaultAnalyzer::new(&circuit).build_influence_map();
        let context = ExactBranchReplayContext { circuit: &circuit };
        let request = context
            .replacement_branch_requests(&influence_map.locations)
            .unwrap()
            .into_iter()
            .find(|request| request.gate_node == entangler)
            .expect("CX request should be present");

        let builder = DemBuilder::new(&influence_map)
            .with_detectors_json(r#"[{"id":0,"records":[-1]}]"#)
            .unwrap()
            .with_num_measurements(1);

        let omitted_only = builder
            .exact_replacement_branch_effect(context, request, "II")
            .expect("omission-only branch should be deterministic here");
        assert_eq!(omitted_only.detectors.as_slice(), &[0]);
        assert!(builder.exact_ideal_history_cache.borrow().is_some());
        assert_eq!(builder.exact_branch_cache.borrow().len(), 1);

        let omitted_then_target_x = builder
            .exact_replacement_branch_effect(context, request, "IX")
            .expect("replacement X on the target should be deterministic here");
        assert!(
            omitted_then_target_x.is_empty(),
            "target X after omitted CX restores the ideal target measurement"
        );
        assert_eq!(builder.exact_branch_cache.borrow().len(), 1);

        let builder = builder.with_detectors_json("[]").unwrap();
        assert!(builder.exact_branch_cache.borrow().is_empty());
    }

    #[test]
    fn test_circuit_with_omitted_two_qubit_gate_rejects_bad_nodes() {
        use pecos_core::{Gate, QubitId};
        use pecos_quantum::DagCircuit;

        let mut circuit = DagCircuit::new();
        let prep = circuit.add_gate_auto_wire(Gate::pz(&[QubitId(0)]));

        assert!(matches!(
            circuit_with_omitted_two_qubit_gate(&circuit, prep),
            Err(DemBuilderError::ConfigurationError(_))
        ));
        assert!(matches!(
            circuit_with_omitted_two_qubit_gate(&circuit, prep + 1),
            Err(DemBuilderError::ConfigurationError(_))
        ));
    }

    #[test]
    fn test_parse_accepts_dem_label_id_form() {
        let det = parse_detectors_json(r#"[{"id": "D0", "records": [-1]}]"#).unwrap();
        assert_eq!(det[0].id, 0);
        let obs = parse_observables_json(r#"[{"id": "L7", "records": [-1]}]"#).unwrap();
        assert_eq!(obs[0].id, 7);
        // Wrong prefix / non-numeric body is a hard error, not a guess.
        assert!(parse_detectors_json(r#"[{"id": "L0", "records": [-1]}]"#).is_err());
        assert!(parse_detectors_json(r#"[{"id": "X0", "records": [-1]}]"#).is_err());
        assert!(parse_observables_json(r#"[{"id": "Lx", "records": [-1]}]"#).is_err());
    }

    #[test]
    fn test_parse_rejects_tracked_pauli_and_refless_entries() {
        assert!(
            parse_observables_json(r#"[{"kind": "tracked_pauli", "pauli": "X0"}]"#).is_err(),
            "tracked_pauli must be rejected in observables_json",
        );
        assert!(
            parse_detectors_json(r#"[{"id": 0, "kind": "tracked_pauli"}]"#).is_err(),
            "tracked_pauli must be rejected in detectors_json too",
        );
        assert!(
            parse_detectors_json(r#"[{"id": 0}]"#).is_err(),
            "an entry with neither records nor meas_ids must be rejected",
        );
        // Both-present is allowed at parse time (surface logical_circuit
        // legitimately emits redundant records+meas_ids); the
        // redundancy/fail-loud decision is made later in try_build.
        assert!(
            parse_detectors_json(r#"[{"id": 0, "records": [-1], "meas_ids": [0]}]"#).is_ok(),
            "both records and meas_ids must parse; redundancy is checked in try_build",
        );
    }

    #[test]
    fn test_try_build_mixed_records_meas_ids_must_be_redundant() {
        // Empty influence map => positional meas_id resolution (deterministic):
        // num_measurements=3, meas_id k resolves to record offset k-3.
        let influence_map = DagFaultInfluenceMap::with_capacity(0);

        // Redundant: records [-3] and meas_ids [0] both name measurement 0.
        let redundant = DemBuilder::new(&influence_map)
            .with_detectors_json(r#"[{"id": 0, "records": [-3], "meas_ids": [0]}]"#)
            .unwrap()
            .with_num_measurements(3)
            .try_build();
        assert!(
            redundant.is_ok(),
            "redundant records+meas_ids must be accepted: {redundant:?}",
        );

        // Non-redundant: records [-3] (measurement 0) vs meas_ids [1]
        // (measurement 1) -> fail loud, not silently records-only.
        let conflicting = DemBuilder::new(&influence_map)
            .with_detectors_json(r#"[{"id": 0, "records": [-3], "meas_ids": [1]}]"#)
            .unwrap()
            .with_num_measurements(3)
            .try_build();
        assert!(
            conflicting.is_err(),
            "non-redundant records+meas_ids must fail loud, not collapse to records",
        );
    }

    #[test]
    fn test_validate_measurement_count_rejects_duplicate_stamped_meas_id() {
        let mut influence_map = DagFaultInfluenceMap::with_capacity(0);
        influence_map.meas_ids = vec![pecos_core::MeasId(5), pecos_core::MeasId(5)];
        let result = DemBuilder::new(&influence_map)
            .with_detectors_json(r#"[{"id": 0, "meas_ids": [5]}]"#)
            .unwrap()
            .try_build();
        assert!(
            result.is_err(),
            "a duplicate stable MeasId must fail loud, not bind to the first",
        );
    }

    #[test]
    fn test_parse_empty_json() {
        assert!(parse_detectors_json("").unwrap().is_empty());
        assert!(parse_detectors_json("[]").unwrap().is_empty());
        assert!(parse_observables_json("").unwrap().is_empty());
    }

    #[test]
    fn test_parse_detector_json_rejects_malformed_shapes() {
        for json in [
            "{}",
            r#"[{"id":0,"records":["-1"]}]"#,
            r#"[{"id":0,"records":[-1.2]}]"#,
            r#"[{"id":0,"meas_ids":["0"]}]"#,
            r#"[{"id":0,"meas_ids":[-1]}]"#,
            r#"[{"id":0,"meas_ids":[1.2]}]"#,
            r#"[{"id":true,"records":[-1]}]"#,
        ] {
            assert!(
                parse_detectors_json(json).is_err(),
                "detectors JSON should fail loud: {json}"
            );
        }
    }

    #[test]
    fn test_parse_observable_json_rejects_malformed_shapes() {
        for json in [
            "{}",
            r#"[{"id":0,"records":["-1"]}]"#,
            r#"[{"id":0,"records":[-1.2]}]"#,
            r#"[{"id":0,"meas_ids":["0"]}]"#,
            r#"[{"id":0,"meas_ids":[-1]}]"#,
            r#"[{"id":0,"meas_ids":[1.2]}]"#,
            r#"[{"observable_id":false,"records":[-1]}]"#,
        ] {
            assert!(
                parse_observables_json(json).is_err(),
                "observables JSON should fail loud: {json}"
            );
        }
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

    /// Issue #325 regression: `from_circuit` once produced different DEMs for
    /// native `F`/`Fdg`/`SY`/`SYdg` versus their unitarily identical
    /// decompositions (86 mechanisms differed on the d=3 SZZ lowered
    /// circuit). The gate table and the dispatch are separate layers; this
    /// pins the composed public path.
    ///
    /// Construction: an identity pair -- the gate under test followed by the
    /// *decomposed* inverse -- so every circuit is noiseless-deterministic
    /// (all measurements read 0, keeping the single-record detectors valid),
    /// while p2 noise injected by the leading CX is conjugated through the
    /// pair. A wrong native conjugation leaves a residual Clifford `R` that
    /// re-maps error components across detector-visibility classes.
    ///
    /// Discriminating power (exact, not "any permutation"): a single
    /// Z-basis readout only distinguishes which Pauli `R` sends to `Z`
    /// (post-CX `X` and `Y` share a detector signature), so two variants are
    /// asserted. Variant 1 (Z-basis) partitions {X,Y} | {Z}; variant 2
    /// (X-basis preparation and readout via `H`) partitions {X} | {Y,Z}.
    /// Jointly they separate all three axes, and with distinct per-Pauli
    /// weights (X 0.5 / Y 0.3 / Z 0.2) any unsigned residual permutation
    /// changes some (probability, signature) pairing. Sign-only differences
    /// remain invisible -- `SY` versus `SYdg` differ only by signs, which no
    /// phase-free DEM test can distinguish; their cases pin each gate
    /// against its own decomposition, not against each other.
    ///
    /// Negative controls are part of the test: one confusable pair per
    /// blindness class proves each variant contributes real teeth.
    #[test]
    fn test_from_tick_circuit_propagates_f_and_sy_like_their_decompositions() {
        use crate::fault_tolerance::dem_builder::PauliWeights;
        use pecos_core::QubitId;
        use pecos_core::pauli::{X, Y, Z};
        use pecos_quantum::{Attribute, TickCircuit};

        fn apply_step(circuit: &mut TickCircuit, step: &str) {
            let mut tick = circuit.tick();
            let target = &[QubitId(1)];
            match step {
                "f" => {
                    tick.f(target);
                }
                "fdg" => {
                    tick.fdg(target);
                }
                "sy" => {
                    tick.sy(target);
                }
                "sydg" => {
                    tick.sydg(target);
                }
                "sx" => {
                    tick.sx(target);
                }
                "sxdg" => {
                    tick.sxdg(target);
                }
                "sz" => {
                    tick.sz(target);
                }
                "szdg" => {
                    tick.szdg(target);
                }
                "h" => {
                    tick.h(target);
                }
                _ => unreachable!(),
            }
        }

        /// Build the identity-pair circuit: `gate_steps` then
        /// `inverse_steps` net to the identity, so the noiseless state stays
        /// |00> and both single-record Z detectors are deterministic.
        /// `x_basis` runs the sandwiched qubit in the X basis (H before the
        /// noise source and H before readout), flipping which error axis is
        /// invisible to the detectors.
        fn build(gate_steps: &[&str], inverse_steps: &[&str], x_basis: bool) -> TickCircuit {
            let mut circuit = TickCircuit::new();
            circuit.tick().pz(&[QubitId(0), QubitId(1)]);
            if x_basis {
                apply_step(&mut circuit, "h");
            }
            circuit.tick().cx(&[(QubitId(0), QubitId(1))]);
            for step in gate_steps {
                apply_step(&mut circuit, step);
            }
            for step in inverse_steps {
                apply_step(&mut circuit, step);
            }
            if x_basis {
                apply_step(&mut circuit, "h");
            }
            circuit.tick().mz(&[QubitId(0), QubitId(1)]);
            circuit.set_meta("num_measurements", Attribute::String("2".to_string()));
            circuit.set_meta(
                "detectors",
                Attribute::String(
                    r#"[{"id":0,"records":[-2]},{"id":1,"records":[-1]}]"#.to_string(),
                ),
            );
            circuit.set_meta("observables", Attribute::String("[]".to_string()));
            circuit
        }

        fn dem(gate_steps: &[&str], inverse_steps: &[&str], x_basis: bool) -> String {
            let noise = NoiseConfig::new(0.0, 0.01, 0.0, 0.0).set_p2_weights(PauliWeights::from([
                (X(1), 0.5),
                (Y(1), 0.3),
                (Z(1), 0.2),
            ]));
            DemBuilder::try_from_tick_circuit_with_noise_config(
                &build(gate_steps, inverse_steps, x_basis),
                noise,
            )
            .expect("valid DEM metadata")
            .to_string()
        }

        // (native gate, decomposition, decomposed inverse), circuit order
        // leftmost-first; decompositions are the state-vector-verified maps.
        let cases: [(&[&str], &[&str], &[&str]); 4] = [
            (&["f"], &["sx", "sz"], &["szdg", "sxdg"]),
            (&["fdg"], &["szdg", "sxdg"], &["sx", "sz"]),
            (&["sy"], &["sx", "sz", "sxdg"], &["sx", "szdg", "sxdg"]),
            (&["sydg"], &["sx", "szdg", "sxdg"], &["sx", "sz", "sxdg"]),
        ];
        for x_basis in [false, true] {
            for (native, decomposed, inverse) in cases {
                assert_eq!(
                    dem(native, inverse, x_basis),
                    dem(decomposed, inverse, x_basis),
                    "from_circuit DEM must be identical for {native:?} and its decomposition \
                     {decomposed:?} (x_basis readout: {x_basis})"
                );
            }
        }

        // Negative controls: each readout variant must catch its class of
        // wrong conjugation. In the identity-pair construction the variant
        // blindness is a property of the RESIDUAL `R = decomposed_inverse o
        // wrong_gate`: the Z-basis variant is blind exactly when `R` fixes
        // `Z` (an unsigned X<->Y swap), the X-basis variant exactly when `R`
        // fixes `X` (an unsigned Y<->Z swap). Substituting SY for F leaves
        // `R = Fdg o SY` = X<->Y swap, the Z-blind case; substituting Fdg
        // for F leaves `R = F`, which fixes nothing and both variants catch.
        assert_ne!(
            dem(&["f"], &["szdg", "sxdg"], false),
            dem(&["fdg"], &["szdg", "sxdg"], false),
            "Z-basis variant must distinguish F from Fdg"
        );
        assert_eq!(
            dem(&["f"], &["szdg", "sxdg"], false),
            dem(&["sy"], &["szdg", "sxdg"], false),
            "documented blindness: the Z-basis variant alone cannot separate F from SY \
             (their residual is an X<->Y swap, which fixes Z); if this ever fails, \
             the readout physics changed"
        );
        assert_ne!(
            dem(&["f"], &["szdg", "sxdg"], true),
            dem(&["sy"], &["szdg", "sxdg"], true),
            "X-basis variant must distinguish F from SY (the Z-blind pair)"
        );
    }

    /// The observables metadata format carries a `label`, and callers already
    /// write one, but the parser dropped it -- so a label could only ever reach
    /// a DEM through a circuit annotation. That coupling meant the metadata
    /// format was not self-sufficient.
    #[test]
    fn test_observable_label_comes_from_metadata_without_annotations() {
        use pecos_num::graph::Attribute;
        use pecos_quantum::DagCircuit;

        let mut circuit = DagCircuit::new();
        circuit.pz(&[0]);
        let _meas = circuit.mz(&[0]);
        circuit.set_attr("num_measurements", Attribute::String("1".to_string()));
        circuit.set_attr(
            "observables",
            Attribute::String(r#"[{"id":0,"records":[-1],"label":"from_metadata"}]"#.to_string()),
        );
        circuit.set_attr("detectors", Attribute::String("[]".to_string()));

        let dem = DemBuilder::from_circuit(&circuit, 0.0, 0.0, 0.5, 0.0);
        assert_eq!(dem.num_observables(), 1);
        assert_eq!(
            dem.observables().next().unwrap().label.as_deref(),
            Some("from_metadata"),
            "the label declared in observables metadata must reach the DEM"
        );
    }

    /// A malformed label is rejected rather than silently treated as absent.
    #[test]
    fn test_observable_metadata_rejects_non_string_label() {
        let err = super::parse_observables_json(r#"[{"id":0,"records":[-1],"label":42}]"#)
            .expect_err("a non-string label must be rejected");
        assert!(
            err.to_string()
                .contains("observable label must be a string"),
            "error should name the problem, got: {err}"
        );
    }

    /// An explicit null label is equivalent to omitting it.
    #[test]
    fn test_observable_metadata_accepts_null_label() {
        let parsed = super::parse_observables_json(r#"[{"id":0,"records":[-1],"label":null}]"#)
            .expect("null label is allowed");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].label, None);
    }
}
