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
    /// Detector definitions in id order, as absolute emission positions.
    pub detector_measurements: Vec<Vec<usize>>,
    /// Observable definitions in id order, as absolute emission positions.
    pub observable_measurements: Vec<Vec<usize>>,
}

/// Result of a DEM simulation run.
pub struct DemSimulationResult {
    /// Per-shot measurement bitstrings (same format as gate-by-gate simulators).
    pub measurements: Vec<Vec<u8>>,
}
/// Why a DEM simulation could not run.
///
/// Every variant carries the underlying diagnostic. This replaces an
/// `Option`/`.ok()?` chain that erased the cause and an `expect` that turned
/// any failure into a generic panic blaming the wrong layer.
#[derive(Debug, Clone)]
pub enum DemSimulationError {
    /// The circuit cannot be expanded or its annotations resolved.
    Eeg(crate::expand::EegBuildError),
    /// The circuit cannot produce a record-aligned measurement history.
    History(pecos_qec::fault_tolerance::fault_sampler::MeasurementHistoryError),
    /// Expanded records cannot be bound to the resolved emission positions.
    MeasurementCountMismatch { metadata: usize, expanded: usize },
    /// The fault table could not be built.
    FaultTable(String),
}

impl std::fmt::Display for DemSimulationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Eeg(err) => write!(f, "DEM simulation: {err}"),
            Self::History(err) => write!(f, "DEM simulation: {err}"),
            Self::MeasurementCountMismatch { metadata, expanded } => write!(
                f,
                "DEM simulation: metadata declares {metadata} measurements, but EEG expansion produced {expanded} measurement records"
            ),
            Self::FaultTable(msg) => write!(f, "DEM simulation: fault table: {msg}"),
        }
    }
}

impl std::error::Error for DemSimulationError {}

impl From<crate::expand::EegBuildError> for DemSimulationError {
    fn from(err: crate::expand::EegBuildError) -> Self {
        Self::Eeg(err)
    }
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
/// * `meta` - Circuit measurement metadata (num_measurements, detector/observable positions)
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
) -> Result<DemSimulationResult, DemSimulationError> {
    // Coherent noise: use EEG path (Heisenberg walks handle idle_rz)
    if noise.idle_rz.abs() > 1e-15 {
        return run_eeg_path(gates, noise, meta, generator, shots, seed);
    }

    // Stochastic: use proper DemSampler with raw measurement output
    stochastic_path(gates, noise, meta, shots, seed)
}

/// Stochastic raw measurement path via RawMeasurementPlan.
///
/// Builds a TickCircuit, runs symbolic simulation for MeasurementHistory,
/// then uses fault_sampler::RawMeasurementPlan for:
/// - Correct cross-reset measurement correlations (via SymbolicSparseStab PZ)
/// - Geometric/O(fired) fault sampling
/// - Raw measurement output matching gate-by-gate simulators
///
fn stochastic_path(
    gates: &[Gate],
    noise: &UniformNoise,
    meta: &CircuitMeasurementMeta,
    shots: usize,
    seed: u64,
) -> Result<DemSimulationResult, DemSimulationError> {
    // Build TickCircuit using typed API (proper measurement record tracking)
    let mut tc = build_tick_circuit(gates, meta)?;

    // Compact ticks to reduce DAG complexity (critical for performance)
    tc.compact_ticks();

    let history = symbolic_measurement_history(&tc).map_err(DemSimulationError::History)?;

    let noise_params = StochasticNoiseParams {
        p1: noise.p1,
        p2: noise.p2,
        p_meas: noise.p_meas,
        p_prep: noise.p_prep,
    };
    let mechanisms =
        pecos_qec::fault_tolerance::fault_sampler::build_fault_table(&tc, &noise_params)
            .map_err(|err| DemSimulationError::FaultTable(err.to_string()))?;
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

    Ok(DemSimulationResult { measurements })
}

/// Resolve an absolute emission position to its measurement reference.
fn resolve_measurement_ref(
    position: usize,
    all_meas_refs: &[pecos_quantum::TickMeasRef],
) -> Result<pecos_quantum::TickMeasRef, DemSimulationError> {
    all_meas_refs.get(position).copied().ok_or_else(|| {
        DemSimulationError::FaultTable(format!(
            "measurement position {position} does not resolve against {} measurements",
            all_meas_refs.len()
        ))
    })
}

/// Build a TickCircuit from flat gates and resolved measurement definitions.
/// Annotations and metadata name the same measurements, including empty definitions.
///
/// # Errors
///
/// Returns [`DemSimulationError`] when a position does not resolve against
/// the circuit's measurements.
fn build_tick_circuit(
    gates: &[Gate],
    meta: &CircuitMeasurementMeta,
) -> Result<TickCircuit, DemSimulationError> {
    use pecos_quantum::{Attribute, TickMeasRef};

    let mut tc = TickCircuit::default();
    let mut all_meas_refs: Vec<TickMeasRef> = Vec::new();

    for gate in gates {
        match gate.gate_type {
            GateType::MZ => all_meas_refs.extend(tc.tick().mz(&gate.qubits)),
            GateType::MX => all_meas_refs.extend(tc.tick().mx(&gate.qubits)),
            GateType::MPZ => all_meas_refs.extend(tc.tick().mpz(&gate.qubits)),
            GateType::MeasureFree => all_meas_refs.extend(tc.tick().mz_free(&gate.qubits)),
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

    // Create annotations from resolved emission positions.
    for records in &meta.detector_measurements {
        let mut det_refs: Vec<TickMeasRef> = Vec::with_capacity(records.len());
        for &rec in records {
            det_refs.push(resolve_measurement_ref(rec, &all_meas_refs)?);
        }
        tc.detector(&det_refs)
            .expect("refs were just resolved from this circuit");
    }

    // Create observable annotations from resolved emission positions
    for records in &meta.observable_measurements {
        let mut obs_refs: Vec<TickMeasRef> = Vec::with_capacity(records.len());
        for &rec in records {
            obs_refs.push(resolve_measurement_ref(rec, &all_meas_refs)?);
        }
        tc.observable(&obs_refs)
            .expect("refs were just resolved from this circuit");
    }

    // The reader merges this metadata with the annotations above and rejects
    // disagreement. Both use the same positions, so they agree by construction.
    tc.set_meta(
        "num_measurements",
        Attribute::String(meta.num_measurements.to_string()),
    );
    for (attribute, definitions) in [
        ("detectors", &meta.detector_measurements),
        ("observables", &meta.observable_measurements),
    ] {
        let definitions = definitions
            .iter()
            .enumerate()
            .map(|(id, positions)| {
                let records: Vec<_> = positions
                    .iter()
                    .map(|&position| position as i128 - meta.num_measurements as i128)
                    .collect();
                serde_json::json!({"id": id, "records": records})
            })
            .collect::<Vec<_>>();
        tc.set_meta(
            attribute,
            Attribute::String(serde_json::Value::Array(definitions).to_string()),
        );
    }

    Ok(tc)
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
) -> Result<DemSimulationResult, DemSimulationError> {
    // Expand circuit for EEG analysis
    let expanded = crate::expand::expand_circuit(gates)?;
    if expanded.measurement_qubit.len() != meta.num_measurements {
        return Err(DemSimulationError::MeasurementCountMismatch {
            metadata: meta.num_measurements,
            expanded: expanded.measurement_qubit.len(),
        });
    }
    let gate_index = GateIndex::build(
        &expanded.gates,
        expanded.num_qubits,
        noise,
        &expanded.expansion_gates,
    );

    // Build detectors and observables from metadata
    let detectors = build_detectors_from_meta(meta, &expanded)?;
    let observables = build_observables_from_meta(meta, &expanded)?;

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

    Ok(DemSimulationResult { measurements })
}

/// Build EEG Detector structs from circuit metadata.
fn build_detectors_from_meta(
    meta: &CircuitMeasurementMeta,
    expanded: &ExpandedCircuit,
) -> Result<Vec<crate::dem_mapping::Detector>, DemSimulationError> {
    let mut detectors = Vec::with_capacity(meta.detector_measurements.len());
    for (id, records) in meta.detector_measurements.iter().enumerate() {
        let mut bm = crate::Bm::default();
        for &rec in records {
            let q = expanded.aux_qubit_for_record(rec)?;
            bm.z_bits.xor_bit(q);
        }
        detectors.push(crate::dem_mapping::Detector { id, stabilizer: bm });
    }
    Ok(detectors)
}

/// Build EEG Observable structs from circuit metadata.
fn build_observables_from_meta(
    meta: &CircuitMeasurementMeta,
    expanded: &ExpandedCircuit,
) -> Result<Vec<crate::dem_mapping::Observable>, DemSimulationError> {
    let mut observables = Vec::with_capacity(meta.observable_measurements.len());
    for (id, records) in meta.observable_measurements.iter().enumerate() {
        let mut bm = crate::Bm::default();
        for &rec in records {
            let q = expanded.aux_qubit_for_record(rec)?;
            bm.z_bits.xor_bit(q);
        }
        observables.push(crate::dem_mapping::Observable { id, pauli: bm });
    }
    Ok(observables)
}

/// Precomputed info for synthesizing measurements from detection events.
/// Only one- and two-reference detectors assign measurements; measurements covered
/// only by larger detectors remain random coins.
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
        for (det_idx, records) in meta.detector_measurements.iter().enumerate() {
            if records.len() == 2 {
                let (earlier, later) = if records[0] < records[1] {
                    (records[0], records[1])
                } else {
                    (records[1], records[0])
                };
                if meas_info[later].is_none() {
                    meas_info[later] = Some((det_idx, earlier));
                }
            } else if records.len() == 1 {
                let idx = records[0];
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
        for (obs_idx, records) in meta.observable_measurements.iter().enumerate() {
            for &rec in records {
                obs_meas_info.push((rec, obs_idx));
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

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_qec::fault_tolerance::circuit_definitions::definitions_from_tick_circuit;

    #[test]
    fn coherent_sampling_rejects_expanded_measurement_count_mismatch() {
        let gates = [Gate::mz(&[0]), Gate::mz(&[1])];
        let meta = CircuitMeasurementMeta {
            num_measurements: 3,
            detector_measurements: vec![vec![1]],
            observable_measurements: vec![],
        };
        // The gates and the metadata reach this function separately (sim_neo
        // builds them from two walks of the circuit), so a caller can pass a
        // metadata count that disagrees with the gates. Position 1 is in range
        // of the two expanded records and would silently bind to MZ(1).
        let result = run_dem_simulation(
            &gates,
            &UniformNoise::coherent_only(0.125),
            &meta,
            &crate::dem_generator::CoherentApprox,
            1,
            1046,
        );
        let Err(error) = result else {
            panic!("mismatched measurement coordinates must be rejected");
        };
        assert!(matches!(
            error,
            DemSimulationError::MeasurementCountMismatch {
                metadata: 3,
                expanded: 2,
            }
        ));
        assert_eq!(
            error.to_string(),
            "DEM simulation: metadata declares 3 measurements, but EEG expansion produced 2 measurement records"
        );
    }

    #[test]
    fn synthesis_uses_absolute_positions() {
        let expanded = crate::expand::expand_circuit(&[Gate::mz(&[0, 1, 2])]).unwrap();
        let meta = CircuitMeasurementMeta {
            num_measurements: 3,
            detector_measurements: vec![vec![0], vec![0, 1], vec![2]],
            observable_measurements: vec![vec![2]],
        };
        let info = MeasurementSynthesisInfo::build(&meta, &expanded);
        let mut rng = PecosRng::seed_from_u64(7);
        // m0 = D0; m1 = m0 XOR D1; m2 = D2 XOR L0.
        assert_eq!(
            info.synthesize(&[true, false, false], &[true], &mut rng),
            vec![1, 1, 1]
        );
        assert_eq!(
            info.synthesize(&[false, true, true], &[true], &mut rng),
            vec![0, 1, 0]
        );
    }

    #[test]
    fn repeated_references_cancel_in_bitmasks() {
        let expanded = crate::expand::expand_circuit(&[Gate::mz(&[0, 1])]).unwrap();
        let meta = CircuitMeasurementMeta {
            num_measurements: 2,
            detector_measurements: vec![vec![0, 0], vec![0, 1, 0]],
            observable_measurements: vec![vec![1, 1], vec![1, 0, 1]],
        };
        let detectors = build_detectors_from_meta(&meta, &expanded).unwrap();
        let observables = build_observables_from_meta(&meta, &expanded).unwrap();
        assert_eq!(detectors[0].stabilizer, crate::Bm::default());
        assert_eq!(observables[0].pauli, crate::Bm::default());
        let mut first = crate::Bm::default();
        first
            .z_bits
            .set_bit(expanded.aux_qubit_for_record(0).unwrap());
        let mut second = crate::Bm::default();
        second
            .z_bits
            .set_bit(expanded.aux_qubit_for_record(1).unwrap());
        assert_eq!(detectors[1].stabilizer, second);
        assert_eq!(observables[1].pauli, first);
    }

    #[test]
    fn rebuilt_definitions_preserve_positions() {
        let gates = [
            Gate::mz(&[1, 0]),
            Gate::mpz(&[0]),
            Gate::mx(&[1]),
            Gate::mz_free(&[0]),
        ];
        let positions = vec![vec![0, 4, 0], vec![1, 2, 3]];
        let meta = CircuitMeasurementMeta {
            num_measurements: 5,
            detector_measurements: positions.clone(),
            observable_measurements: positions.clone(),
        };
        let circuit = build_tick_circuit(&gates, &meta).unwrap();
        let definitions = definitions_from_tick_circuit(&circuit).unwrap();
        assert_eq!(definitions.num_measurements, 5);
        assert_eq!(
            definitions
                .detectors
                .iter()
                .map(|d| d.measurements.clone())
                .collect::<Vec<_>>(),
            positions
        );
        assert_eq!(
            definitions
                .observables
                .iter()
                .map(|o| o.measurements.clone())
                .collect::<Vec<_>>(),
            positions
        );
        assert_eq!(
            circuit.get_meta("detectors"),
            Some(&pecos_quantum::Attribute::String(
                r#"[{"id":0,"records":[-5,-1,-5]},{"id":1,"records":[-4,-3,-2]}]"#.to_string()
            ))
        );
    }
}
