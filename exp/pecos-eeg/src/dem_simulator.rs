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

//! DEM-based simulator: samples from a Detector Error Model and synthesizes
//! physical measurement bitstrings.
//!
//! This module provides the pure Rust implementation of DEM-based simulation.
//! Given a circuit (as `Vec<Gate>`) and noise parameters, it:
//! 1. Builds a DEM via the EEG coherent backward mechanism extraction
//! 2. Samples detection events from the DEM
//! 3. Synthesizes physical measurement bitstrings matching the circuit's output
//!
//! # Performance
//!
//! The sampling step uses `ParsedDem::sample()` which is O(mechanisms) per shot.
//! For bulk sampling, use `to_dem_sampler()` for columnar bit-packed SIMD sampling.

use crate::dem_generator::{DemContext, DemGenerator};
use crate::expand::{ExpandedCircuit, GateIndex};
use crate::noise::UniformNoise;
use pecos_core::Gate;
use pecos_core::gate_type::GateType;
use pecos_core::pauli::pauli_bitmask::BitmaskStorage;
use pecos_qec::fault_tolerance::dem_builder::ParsedDem;
use pecos_qec::fault_tolerance::fault_sampler::{
    RawMeasurementPlan, StochasticNoiseParams, symbolic_measurement_history,
};
use pecos_quantum::TickCircuit;
use pecos_random::PecosRng;

/// Metadata needed for measurement synthesis.
#[derive(Clone, Debug)]
pub struct CircuitMeasurementMeta {
    /// Total number of physical measurements in the circuit.
    pub num_measurements: usize,
    /// Detector definitions: each is a list of measurement record offsets.
    /// Offset is relative: absolute_index = num_measurements + offset.
    pub detector_records: Vec<Vec<i32>>,
    /// Observable definitions: same format as detectors.
    pub observable_records: Vec<Vec<i32>>,
}

/// Result of a DEM simulation run.
pub struct DemSimulationResult {
    /// Per-shot measurement bitstrings (same format as gate-by-gate simulators).
    pub measurements: Vec<Vec<u8>>,
}

/// Run DEM-based simulation: build DEM, sample, produce measurement bitstrings.
///
/// Two modes:
/// 1. **Stochastic path** (idle_rz == 0): builds a TickCircuit from gates + metadata,
///    uses `DemSampler::from_tick_circuit` with `OutputMode::RawMeasurements` for
///    proper non-deterministic handling and maximum performance.
/// 2. **Coherent path** (idle_rz > 0): uses EEG DemGenerator for DEM, then
///    ParsedDem sampler + measurement synthesis (EEG handles coherent noise).
///
/// # Arguments
/// * `gates` - Circuit gates (from CommandQueue conversion)
/// * `noise` - Noise parameters
/// * `meta` - Circuit measurement metadata (num_measurements, detector/observable records)
/// * `generator` - Which DEM generator to use (for coherent path)
/// * `shots` - Number of shots to sample
/// * `seed` - Random seed
pub fn run_dem_simulation(
    gates: &[Gate],
    noise: &UniformNoise,
    meta: &CircuitMeasurementMeta,
    generator: &dyn DemGenerator,
    shots: usize,
    seed: u64,
) -> DemSimulationResult {
    // Coherent noise: use EEG path (Heisenberg walks handle idle_rz)
    if noise.idle_rz.abs() > 1e-15 {
        return run_eeg_path(gates, noise, meta, generator, shots, seed);
    }

    // Stochastic: use proper DemSampler with raw measurement output
    try_stochastic_path(gates, noise, meta, shots, seed)
        .expect("DEM simulation failed: could not build TickCircuit or DemSampler from circuit")
}

/// Stochastic raw measurement path via RawMeasurementPlan.
///
/// Builds a TickCircuit, runs symbolic simulation for MeasurementHistory,
/// then uses fault_sampler::RawMeasurementPlan for:
/// - Correct cross-reset measurement correlations (via SymbolicSparseStab PZ)
/// - Geometric/O(fired) fault sampling
/// - Raw measurement output matching gate-by-gate simulators
///
/// Returns None if idle_rz > 0 (needs EEG path for coherent noise).
fn try_stochastic_path(
    gates: &[Gate],
    noise: &UniformNoise,
    meta: &CircuitMeasurementMeta,
    shots: usize,
    seed: u64,
) -> Option<DemSimulationResult> {
    // Only use stochastic path when no coherent noise
    if noise.idle_rz.abs() > 1e-15 {
        return None;
    }

    // Build TickCircuit using typed API (proper measurement record tracking)
    let mut tc = build_tick_circuit(gates, meta);

    // Compact ticks to reduce DAG complexity (critical for performance)
    tc.compact_ticks();

    let history = symbolic_measurement_history(&tc).ok()?;

    let noise_params = StochasticNoiseParams {
        p1: noise.p1,
        p2: noise.p2,
        p_meas: noise.p_meas,
        p_prep: noise.p_prep,
    };
    let mechanisms =
        pecos_qec::fault_tolerance::fault_sampler::build_fault_table(&tc, &noise_params).ok()?;
    let plan = RawMeasurementPlan::new(&history, mechanisms);

    // Sample raw measurements (columnar, then extract rows)
    let result = plan.sample(shots, seed);
    let mut measurements = Vec::with_capacity(shots);
    for shot in 0..shots {
        let n = result.num_measurements();
        let mut meas = Vec::with_capacity(n);
        for m in 0..n {
            meas.push(u8::from(result.get(shot, m).0));
        }
        measurements.push(meas);
    }

    Some(DemSimulationResult { measurements })
}

/// Build a TickCircuit from flat gates + metadata using the typed API.
///
/// Uses `.mz()` for measurement gates (properly tracks measurement records),
/// `.pz()` for prep gates, and `try_add_gate()` for all other gates.
/// After building all gates, creates detector/observable annotations using
/// the stored measurement references. This ensures the DagCircuit conversion
/// and DagFaultAnalyzer see proper structured annotations.
fn build_tick_circuit(gates: &[Gate], meta: &CircuitMeasurementMeta) -> TickCircuit {
    use pecos_quantum::{Attribute, TickMeasRef};

    let mut tc = TickCircuit::default();
    let mut all_meas_refs: Vec<TickMeasRef> = Vec::new();

    for gate in gates {
        match gate.gate_type {
            GateType::MZ => {
                // Use the typed .mz() API to properly track measurement records
                let qubits: Vec<pecos_core::QubitId> = gate.qubits.iter().copied().collect();
                let refs = tc.tick().mz(&qubits);
                all_meas_refs.extend(refs);
            }
            GateType::PZ | GateType::QAlloc => {
                let qubits: Vec<pecos_core::QubitId> = gate.qubits.iter().copied().collect();
                tc.tick().pz(&qubits);
            }
            _ => {
                let mut tick = tc.tick();
                let _ = tick.try_add_gate(gate.clone());
            }
        }
    }

    // Create detector annotations from record definitions
    for records in &meta.detector_records {
        let det_refs: Vec<TickMeasRef> = records
            .iter()
            .filter_map(|&rec| {
                let abs_idx = (meta.num_measurements as i32 + rec) as usize;
                all_meas_refs.get(abs_idx).copied()
            })
            .collect();
        if !det_refs.is_empty() {
            tc.detector(&det_refs);
        }
    }

    // Create observable annotations from record definitions
    for records in &meta.observable_records {
        let obs_refs: Vec<TickMeasRef> = records
            .iter()
            .filter_map(|&rec| {
                let abs_idx = (meta.num_measurements as i32 + rec) as usize;
                all_meas_refs.get(abs_idx).copied()
            })
            .collect();
        if !obs_refs.is_empty() {
            tc.observable(&obs_refs);
        }
    }

    // Set metadata (for DemBuilder JSON fallback path)
    tc.set_meta(
        "num_measurements",
        Attribute::String(meta.num_measurements.to_string()),
    );
    if let Ok(det_json) = serde_json::to_string(
        &meta
            .detector_records
            .iter()
            .enumerate()
            .map(|(id, recs)| serde_json::json!({"id": id, "records": recs}))
            .collect::<Vec<_>>(),
    ) {
        tc.set_meta("detectors", Attribute::String(det_json));
    }
    if let Ok(obs_json) = serde_json::to_string(
        &meta
            .observable_records
            .iter()
            .enumerate()
            .map(|(id, recs)| serde_json::json!({"id": id, "records": recs}))
            .collect::<Vec<_>>(),
    ) {
        tc.set_meta("observables", Attribute::String(obs_json));
    }

    tc
}

/// EEG path: DEM generation + ParsedDem sampling + measurement synthesis.
///
/// Used when coherent noise (idle_rz) is present and the stochastic path
/// cannot capture the noise accurately.
fn run_eeg_path(
    gates: &[Gate],
    noise: &UniformNoise,
    meta: &CircuitMeasurementMeta,
    generator: &dyn DemGenerator,
    shots: usize,
    seed: u64,
) -> DemSimulationResult {
    // Expand circuit for EEG analysis
    let expanded = crate::expand::expand_circuit(gates);
    let gate_index = GateIndex::build(&expanded.gates, expanded.num_qubits);

    // Build detectors and observables from metadata
    let detectors = build_detectors_from_meta(meta, &expanded);
    let observables = build_observables_from_meta(meta, &expanded);

    // Generate DEM via trait
    let ctx = DemContext {
        gates: &expanded.gates,
        expanded: &expanded,
        gate_index: &gate_index,
        detectors: &detectors,
        observables: &observables,
    };
    let output = generator.generate(&ctx, noise);
    let dem_str = crate::dem_mapping::format_dem(&output.entries);

    // Parse DEM and build sampler
    let parsed_dem: ParsedDem = dem_str.parse().unwrap_or_else(|_| ParsedDem::new());
    let sampler = parsed_dem.to_dem_sampler();

    // Build measurement synthesis info
    let synthesis_info = MeasurementSynthesisInfo::build(meta, &expanded);

    // Sample and synthesize
    let mut rng = PecosRng::seed_from_u64(seed);
    let mut measurements = Vec::with_capacity(shots);

    for _ in 0..shots {
        let (det_events, obs_flips) = sampler.sample(&mut rng);
        let meas = synthesis_info.synthesize(&det_events, &obs_flips, &mut rng);
        measurements.push(meas);
    }

    DemSimulationResult { measurements }
}

/// Build EEG Detector structs from circuit metadata.
fn build_detectors_from_meta(
    meta: &CircuitMeasurementMeta,
    expanded: &ExpandedCircuit,
) -> Vec<crate::dem_mapping::Detector> {
    meta.detector_records
        .iter()
        .enumerate()
        .map(|(id, records)| {
            let mut bm = crate::Bm::default();
            for &rec in records {
                let meas_idx = (meta.num_measurements as i32 + rec) as usize;
                if meas_idx < expanded.measurement_qubit.len() {
                    let q = expanded.measurement_qubit[meas_idx];
                    bm.z_bits.set_bit(q);
                }
            }
            crate::dem_mapping::Detector { id, stabilizer: bm }
        })
        .collect()
}

/// Build EEG Observable structs from circuit metadata.
fn build_observables_from_meta(
    meta: &CircuitMeasurementMeta,
    expanded: &ExpandedCircuit,
) -> Vec<crate::dem_mapping::Observable> {
    meta.observable_records
        .iter()
        .enumerate()
        .map(|(id, records)| {
            let mut bm = crate::Bm::default();
            for &rec in records {
                let meas_idx = (meta.num_measurements as i32 + rec) as usize;
                if meas_idx < expanded.measurement_qubit.len() {
                    let q = expanded.measurement_qubit[meas_idx];
                    bm.z_bits.set_bit(q);
                }
            }
            crate::dem_mapping::Observable { id, pauli: bm }
        })
        .collect()
}

/// Precomputed info for synthesizing measurements from detection events.
struct MeasurementSynthesisInfo {
    num_meas: usize,
    /// For each measurement: Some((det_idx, other_meas_idx)) if determined by a detector.
    /// other_meas_idx == usize::MAX means single-record detector.
    meas_info: Vec<Option<(usize, usize)>>,
    /// Which measurements are non-deterministic (need random coin).
    is_non_det: Vec<bool>,
    /// Observable measurement assignments: (meas_idx, obs_idx).
    obs_meas_info: Vec<(usize, usize)>,
}

impl MeasurementSynthesisInfo {
    /// Build synthesis info from circuit metadata.
    fn build(meta: &CircuitMeasurementMeta, _expanded: &ExpandedCircuit) -> Self {
        let num_meas = meta.num_measurements;
        let mut meas_info: Vec<Option<(usize, usize)>> = vec![None; num_meas];

        // Build detector -> measurement mapping
        for (det_idx, records) in meta.detector_records.iter().enumerate() {
            let abs_records: Vec<usize> = records
                .iter()
                .map(|&r| (num_meas as i32 + r) as usize)
                .filter(|&idx| idx < num_meas)
                .collect();

            if abs_records.len() == 2 {
                let (earlier, later) = if abs_records[0] < abs_records[1] {
                    (abs_records[0], abs_records[1])
                } else {
                    (abs_records[1], abs_records[0])
                };
                if meas_info[later].is_none() {
                    meas_info[later] = Some((det_idx, earlier));
                }
            } else if abs_records.len() == 1 {
                let idx = abs_records[0];
                if meas_info[idx].is_none() {
                    meas_info[idx] = Some((det_idx, usize::MAX));
                }
            }
        }

        // Identify non-deterministic measurements
        let mut is_non_det = vec![false; num_meas];
        for idx in 0..num_meas {
            if meas_info[idx].is_none() {
                is_non_det[idx] = true;
            }
        }
        // Also: measurements referenced as "other" by a detector but not assigned themselves
        for idx in 0..num_meas {
            if let Some((_, other_idx)) = meas_info[idx]
                && other_idx != usize::MAX
                && other_idx < num_meas
                && meas_info[other_idx].is_none()
            {
                is_non_det[other_idx] = true;
            }
        }

        // Observable measurement assignments
        let mut obs_meas_info = Vec::new();
        for (obs_idx, records) in meta.observable_records.iter().enumerate() {
            for &rec in records {
                let idx = (num_meas as i32 + rec) as usize;
                if idx < num_meas {
                    obs_meas_info.push((idx, obs_idx));
                }
            }
        }

        Self {
            num_meas,
            meas_info,
            is_non_det,
            obs_meas_info,
        }
    }

    /// Synthesize a measurement bitstring from detection events + observable flips.
    fn synthesize(&self, det_events: &[bool], obs_flips: &[bool], rng: &mut PecosRng) -> Vec<u8> {
        let mut meas = vec![0u8; self.num_meas];

        // Random coins for non-deterministic measurements
        for (idx, bit) in meas.iter_mut().enumerate().take(self.num_meas) {
            if self.is_non_det[idx] {
                *bit = u8::from(rng.random_bool(0.5));
            }
        }

        // Assign measurements in index order (time order)
        for idx in 0..self.num_meas {
            if let Some((det_idx, other_idx)) = self.meas_info[idx] {
                if det_idx < det_events.len() && det_events[det_idx] {
                    if other_idx == usize::MAX {
                        meas[idx] ^= 1;
                    } else if other_idx < self.num_meas {
                        meas[idx] = u8::from(det_events[det_idx]) ^ meas[other_idx];
                    }
                } else if other_idx != usize::MAX && other_idx < self.num_meas {
                    meas[idx] = meas[other_idx];
                }
            }
        }

        // Apply observable flips
        for &(meas_idx, obs_idx) in &self.obs_meas_info {
            if obs_idx < obs_flips.len() && obs_flips[obs_idx] {
                meas[meas_idx] ^= 1;
            }
        }

        meas
    }
}
