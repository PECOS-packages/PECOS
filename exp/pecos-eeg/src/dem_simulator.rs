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
//! 2. Samples detection events from the DEM with `ParsedDem::to_dem_sampler()`
//! 3. Synthesizes physical measurement bitstrings matching the circuit's output:
//!    the noiseless row from the symbolic measurement history, XOR a flip
//!    vector that reproduces the sampled detector and observable events

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
use pecos_quantum::{F2Matrix, TickCircuit};
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
    /// The symbolic history does not match the declared measurement count.
    HistoryMeasurementCountMismatch { metadata: usize, history: usize },
    /// A gate could not be added to the rebuilt circuit.
    TickGate(pecos_quantum::TickGateError),
    /// The generated DEM could not be parsed.
    DemParse(pecos_qec::fault_tolerance::dem_builder::DemParseError),
    /// A DEM detector or observable id exceeds the corresponding definitions.
    EventCountMismatch {
        kind: &'static str,
        definitions: usize,
        events: usize,
    },
    /// A sampled event pattern violates a dependent row of the parity matrix.
    InconsistentEvents { shot: usize, row: usize },
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
            Self::HistoryMeasurementCountMismatch { metadata, history } => write!(
                f,
                "DEM simulation: metadata declares {metadata} measurements, but symbolic history produced {history} measurement records"
            ),
            Self::TickGate(err) => write!(f, "DEM simulation: {err}"),
            Self::DemParse(err) => write!(f, "DEM simulation: {err}"),
            Self::EventCountMismatch {
                kind,
                definitions,
                events,
            } => write!(
                f,
                "DEM simulation: DEM has {events} {kind} slots, but only {definitions} definitions"
            ),
            Self::InconsistentEvents { shot, row } => write!(
                f,
                "DEM simulation: inconsistent events at shot {shot}, dependent row {row}"
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
///    samples the symbolic measurement history with `RawMeasurementPlan`, and
///    overlays the stochastic fault table using geometric fault sampling.
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

    // Stochastic: sample the symbolic history and physical fault mechanisms
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
/// the circuit's measurements or a gate payload is invalid.
fn build_tick_circuit(
    gates: &[Gate],
    meta: &CircuitMeasurementMeta,
) -> Result<TickCircuit, DemSimulationError> {
    use pecos_quantum::{Attribute, TickMeasRef};

    let mut tc = TickCircuit::default();
    let mut all_meas_refs: Vec<TickMeasRef> = Vec::new();

    for (tick_idx, gate) in gates.iter().enumerate() {
        // Typed measurement/preparation helpers also validate, but panic on
        // invalid payloads. Reject them here along with all other gate types.
        gate.validate().map_err(|message| {
            DemSimulationError::TickGate(pecos_quantum::TickGateError::InvalidGate {
                message,
                tick_idx: Some(tick_idx),
            })
        })?;
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
                tick.try_add_gate(gate.clone())
                    .map_err(DemSimulationError::TickGate)?;
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
///
/// Each row is the noiseless row `r` XOR a flip vector `f` solving
/// `[D; L] f = [d; o]` over GF(2), with free measurements set to 0. Every
/// detector and observable parity is therefore exact, but a measurement that
/// no detector or observable covers keeps its noiseless value and receives no
/// noise.
fn run_eeg_path(
    gates: &[Gate],
    noise: &UniformNoise,
    meta: &CircuitMeasurementMeta,
    generator: &dyn DemGenerator,
    shots: usize,
    seed: u64,
) -> Result<DemSimulationResult, DemSimulationError> {
    // Keep emission order: compact_ticks can move measurements into earlier
    // batches. Validate the symbolic history before reaching EEG conjugation,
    // which can panic on unsupported gates.
    let tc = build_tick_circuit(gates, meta)?;
    let history = symbolic_measurement_history(&tc).map_err(DemSimulationError::History)?;
    let plan = RawMeasurementPlan::new(&history, Vec::new());

    // Expand circuit for EEG analysis
    let expanded = crate::expand::expand_circuit(gates)?;
    if expanded.measurement_qubit.len() != meta.num_measurements {
        return Err(DemSimulationError::MeasurementCountMismatch {
            metadata: meta.num_measurements,
            expanded: expanded.measurement_qubit.len(),
        });
    }
    if plan.num_measurements != meta.num_measurements {
        return Err(DemSimulationError::HistoryMeasurementCountMismatch {
            metadata: meta.num_measurements,
            history: plan.num_measurements,
        });
    }
    let noiseless = plan.sample(shots, seed);
    let synthesis = MeasurementFlipSolver::new(meta);
    let mut events = vec![0; synthesis.num_events().div_ceil(64)];
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

    // Validate the DEM's namespaces even when shots == 0.
    let parsed_dem = parse_dem(&dem_str, meta)?;
    let sampler = parsed_dem.to_dem_sampler();
    let mut rng = PecosRng::seed_from_u64(dem_sampling_seed(seed));
    let mut measurements = Vec::with_capacity(shots);

    for shot in 0..shots {
        let (mut det_events, mut obs_flips) = sampler.sample(&mut rng);
        // ParsedDem counts only up to the highest id present in its text.
        // Definitions untouched by any mechanism still contribute zero events.
        det_events.resize(meta.detector_measurements.len(), false);
        obs_flips.resize(meta.observable_measurements.len(), false);
        let mut meas = (0..meta.num_measurements)
            .map(|m| u8::from(noiseless.get(shot, m).0))
            .collect::<Vec<_>>();
        synthesis.apply(&det_events, &obs_flips, shot, &mut meas, &mut events)?;
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

/// Keep DEM events independent of RawMeasurementPlan's base stream (seed)
/// and fault stream (seed + 1), including at the u64 wraparound boundary.
fn dem_sampling_seed(seed: u64) -> u64 {
    seed.wrapping_add(2)
}

/// Parse the generated DEM text and reject detector or observable ids beyond
/// the circuit's definition counts.
fn parse_dem(text: &str, meta: &CircuitMeasurementMeta) -> Result<ParsedDem, DemSimulationError> {
    let parsed: ParsedDem = text.parse().map_err(DemSimulationError::DemParse)?;
    for (kind, events, definitions) in [
        (
            "detector",
            parsed.num_detectors as usize,
            meta.detector_measurements.len(),
        ),
        (
            "observable",
            parsed.num_observables() as usize,
            meta.observable_measurements.len(),
        ),
    ] {
        if events > definitions {
            return Err(DemSimulationError::EventCountMismatch {
                kind,
                definitions,
                events,
            });
        }
    }
    Ok(parsed)
}

/// A particular solution of A f = [d; o], where A contains detector rows
/// followed by observable rows. Free measurement columns are always zero.
struct MeasurementFlipSolver {
    num_detectors: usize,
    num_observables: usize,
    /// Pivot columns belonging to A, in reduced row order.
    pivots: Vec<usize>,
    /// Packed rows of the transform T extracted from reduced [A | I].
    transform: Vec<Vec<u64>>,
}

impl MeasurementFlipSolver {
    /// Definition positions have already been validated by build_tick_circuit.
    fn new(meta: &CircuitMeasurementMeta) -> Self {
        let num_rows = meta.detector_measurements.len() + meta.observable_measurements.len();
        let mut augmented = F2Matrix::zeros(num_rows, meta.num_measurements + num_rows);
        for (row, positions) in meta
            .detector_measurements
            .iter()
            .chain(&meta.observable_measurements)
            .enumerate()
        {
            for &column in positions {
                // Definitions are parities: duplicate positions cancel.
                augmented.set(row, column, augmented.get(row, column) ^ 1);
            }
            augmented.set(row, meta.num_measurements + row, 1);
        }
        let (reduced, pivots) = augmented.row_reduce();
        let pivots = pivots
            .into_iter()
            .take_while(|&column| column < meta.num_measurements)
            .collect();
        // Reduction may also pivot in the identity block. Such rows have a
        // zero A block and encode consistency constraints on the event vector.
        let transform = (0..num_rows)
            .map(|row| {
                let mut words = vec![0; num_rows.div_ceil(64)];
                for column in 0..num_rows {
                    words[column / 64] |=
                        u64::from(reduced.get(row, meta.num_measurements + column))
                            << (column % 64);
                }
                words
            })
            .collect();
        Self {
            num_detectors: meta.detector_measurements.len(),
            num_observables: meta.observable_measurements.len(),
            pivots,
            transform,
        }
    }

    fn num_events(&self) -> usize {
        self.num_detectors + self.num_observables
    }

    /// XOR the particular flip solution into a noiseless row. The packed
    /// scratch buffer is reused across shots; the per-row product uses popcount.
    fn apply(
        &self,
        detectors: &[bool],
        observables: &[bool],
        shot: usize,
        measurements: &mut [u8],
        events: &mut [u64],
    ) -> Result<(), DemSimulationError> {
        assert_eq!(detectors.len(), self.num_detectors);
        assert_eq!(observables.len(), self.num_observables);
        assert_eq!(events.len(), self.num_events().div_ceil(64));
        events.fill(0);
        for (column, &value) in detectors.iter().chain(observables).enumerate() {
            events[column / 64] |= u64::from(value) << (column % 64);
        }
        for (row, transform) in self.transform.iter().enumerate() {
            let value = transform
                .iter()
                .zip(events.iter())
                .fold(0, |parity, (&a, &b)| parity ^ (a & b).count_ones())
                & 1;
            if let Some(&pivot) = self.pivots.get(row) {
                // row_reduce chooses the earliest measurement pivot.
                measurements[pivot] ^= u8::from(value != 0);
            } else if value != 0 {
                return Err(DemSimulationError::InconsistentEvents { shot, row });
            }
        }
        Ok(())
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
    fn synthesis_solves_overlapping_and_dependent_parities() {
        let gates = [
            Gate::x(&[0]),
            Gate::h(&[1]),
            Gate::cx(&[(1, 2)]),
            Gate::mz(&[0, 1, 2, 3, 4]),
        ];
        let meta = CircuitMeasurementMeta {
            num_measurements: 5,
            detector_measurements: vec![vec![0, 1, 2], vec![0, 2], vec![1]],
            observable_measurements: vec![vec![1, 2, 3, 4, 4]],
        };
        let circuit = build_tick_circuit(&gates, &meta).unwrap();
        let history = symbolic_measurement_history(&circuit).unwrap();
        let noiseless = RawMeasurementPlan::new(&history, Vec::new()).sample(32, 7);
        let solver = MeasurementFlipSolver::new(&meta);
        assert_eq!(solver.pivots, vec![0, 1, 2]);
        for shot in 0..32 {
            let base = (0..5)
                .map(|m| u8::from(noiseless.get(shot, m).0))
                .collect::<Vec<_>>();
            // Bell pair: r = [1, b, b, 0, 0]. D0=1 and L0=0 noiselessly.
            assert_eq!(base, vec![1, base[1], base[1], 0, 0]);
            for (d0, d1, logical) in (0..8).map(|v| (v & 1 != 0, v & 2 != 0, v & 4 != 0)) {
                let mut row = base.clone();
                solver
                    .apply(&[d0, d1, d0 ^ d1], &[logical], shot, &mut row, &mut [0])
                    .unwrap();
                // Hand solution with free columns f3=f4=0.
                assert_eq!(
                    row,
                    vec![
                        1 ^ u8::from(d0 ^ logical),
                        base[1] ^ u8::from(d0 ^ d1),
                        base[2] ^ u8::from(d0 ^ d1 ^ logical),
                        0,
                        0,
                    ]
                );
                for (positions, event) in meta
                    .detector_measurements
                    .iter()
                    .chain(&meta.observable_measurements)
                    .zip([d0, d1, d0 ^ d1, logical])
                {
                    let parity = |bits: &[u8]| positions.iter().fold(0, |p, &m| p ^ bits[m]);
                    assert_eq!(parity(&row), parity(&base) ^ u8::from(event));
                }
            }
        }
        let error = solver
            .apply(&[false, false, true], &[false], 17, &mut [0; 5], &mut [0])
            .unwrap_err();
        assert!(matches!(
            error,
            DemSimulationError::InconsistentEvents { shot: 17, row: 3 }
        ));
        assert!(error.to_string().contains("shot 17, dependent row 3"));
    }

    #[test]
    fn synthesis_packed_transform_crosses_word_boundaries() {
        let meta = CircuitMeasurementMeta {
            num_measurements: 70,
            detector_measurements: (0..70).map(|m| vec![m]).collect(),
            observable_measurements: vec![vec![0, 63, 64, 69]],
        };
        let solver = MeasurementFlipSolver::new(&meta);
        for column in 0..70 {
            let mut events = vec![false; 70];
            events[column] = true;
            let mut row = vec![0; 70];
            solver
                .apply(
                    &events,
                    &[matches!(column, 0 | 63 | 64 | 69)],
                    column,
                    &mut row,
                    &mut [0; 2],
                )
                .unwrap();
            assert_eq!(row.iter().map(|&v| v != 0).collect::<Vec<_>>(), events);
        }
    }

    #[test]
    fn coherent_empty_dem_preserves_noiseless_rows_with_padding() {
        struct EmptyDem;
        impl DemGenerator for EmptyDem {
            fn generate(
                &self,
                ctx: &DemContext<'_>,
                noise: &dyn crate::noise::NoiseSpec,
            ) -> crate::dem_generator::DemOutput {
                let output = crate::dem_generator::CoherentApprox.generate(ctx, noise);
                assert!(output.entries.is_empty());
                output
            }
            fn name(&self) -> &'static str {
                "empty"
            }
        }
        let gates = [
            Gate::pz(&[0, 1]),
            Gate::h(&[1]),
            Gate::mz(&[0, 1]),
            Gate::mz(&[1]),
            Gate::x(&[0]),
            Gate::mz(&[0]),
        ];
        let meta = CircuitMeasurementMeta {
            num_measurements: 4,
            detector_measurements: vec![vec![0], vec![1, 2]],
            observable_measurements: vec![vec![3]],
        };
        let history =
            symbolic_measurement_history(&build_tick_circuit(&gates, &meta).unwrap()).unwrap();
        let noiseless = RawMeasurementPlan::new(&history, Vec::new()).sample(128, 7);
        let rows = run_eeg_path(
            &gates,
            &UniformNoise::coherent_only(1e-4),
            &meta,
            &EmptyDem,
            128,
            7,
        )
        .unwrap()
        .measurements;
        for (shot, row) in rows.iter().enumerate() {
            assert_eq!(
                *row,
                (0..4)
                    .map(|m| u8::from(noiseless.get(shot, m).0))
                    .collect::<Vec<_>>()
            );
            assert_eq!(*row, vec![0, row[1], row[1], 1]);
        }
        assert!(rows.iter().any(|row| row[1] == 0));
        assert!(rows.iter().any(|row| row[1] == 1));
    }

    #[test]
    fn coherent_dem_seed_is_independent_of_plan_streams() {
        for seed in [0, 7, u64::MAX - 1, u64::MAX] {
            let dem_seed = dem_sampling_seed(seed);
            assert_ne!(dem_seed, seed);
            assert_ne!(dem_seed, seed.wrapping_add(1));
            let mut dem = PecosRng::seed_from_u64(dem_seed);
            let mut base = PecosRng::seed_from_u64(seed);
            assert_ne!(dem.next_u64(), base.next_u64());
        }
    }

    #[test]
    fn coherent_noiseless_history_keeps_emission_order() {
        let gates = [
            Gate::pz(&[0, 1]),
            Gate::h(&[0]),
            Gate::x(&[0]),
            Gate::mz(&[0]),
            Gate::mz(&[1]),
        ];
        let meta = CircuitMeasurementMeta {
            num_measurements: 2,
            detector_measurements: vec![],
            observable_measurements: vec![],
        };
        // Compaction could move the independent MZ(1) before MZ(0), swapping
        // the deterministic and random columns. No definitions constrain them.
        let rows = run_eeg_path(
            &gates,
            &UniformNoise::coherent_only(1e-4),
            &meta,
            &crate::dem_generator::CoherentApprox,
            128,
            7,
        )
        .unwrap()
        .measurements;
        assert!(rows.iter().all(|row| row[1] == 0));
        assert!(rows.iter().any(|row| row[0] == 0));
        assert!(rows.iter().any(|row| row[0] == 1));
    }

    #[test]
    fn synthesis_handles_empty_and_cancelled_definitions() {
        let mut meta = CircuitMeasurementMeta {
            num_measurements: 2,
            detector_measurements: vec![],
            observable_measurements: vec![],
        };
        let mut row = [1, 0];
        MeasurementFlipSolver::new(&meta)
            .apply(&[], &[], 0, &mut row, &mut [])
            .unwrap();
        assert_eq!(row, [1, 0]);
        meta.detector_measurements = vec![vec![], vec![0, 0]];
        meta.observable_measurements = vec![vec![1, 1]];
        let solver = MeasurementFlipSolver::new(&meta);
        solver
            .apply(&[false, false], &[false], 0, &mut row, &mut [0])
            .unwrap();
        assert_eq!(row, [1, 0]);
        assert!(matches!(
            solver.apply(&[false, true], &[false], 9, &mut row, &mut [0]),
            Err(DemSimulationError::InconsistentEvents { shot: 9, .. })
        ));
    }

    #[test]
    fn coherent_dem_rejects_parse_errors_and_excess_ids() {
        let meta = CircuitMeasurementMeta {
            num_measurements: 1,
            detector_measurements: vec![vec![0]],
            observable_measurements: vec![vec![0]],
        };
        assert!(matches!(
            parse_dem("invalid", &meta),
            Err(DemSimulationError::DemParse(_))
        ));
        for (text, kind) in [
            ("error(0.1) D1", "detector"),
            ("error(0.1) L1", "observable"),
        ] {
            assert!(
                matches!(parse_dem(text, &meta), Err(DemSimulationError::EventCountMismatch { kind: actual, definitions: 1, events: 2 }) if actual == kind)
            );
        }
    }

    #[test]
    fn coherent_unsupported_gate_returns_history_error() {
        let meta = CircuitMeasurementMeta {
            num_measurements: 1,
            detector_measurements: vec![vec![0]],
            observable_measurements: vec![],
        };
        let result = run_eeg_path(
            &[Gate::t(&[0]), Gate::mz(&[0])],
            &UniformNoise::coherent_only(0.1),
            &meta,
            &crate::dem_generator::CoherentApprox,
            1,
            7,
        );
        assert!(matches!(result, Err(DemSimulationError::History(_))));
    }

    #[test]
    fn rebuilding_rejects_invalid_gate_payloads() {
        let meta = CircuitMeasurementMeta {
            num_measurements: 0,
            detector_measurements: vec![],
            observable_measurements: vec![],
        };
        for gate in [Gate::h(&[0, 0]), Gate::mz(&[0, 0]), Gate::pz(&[0, 0])] {
            assert!(matches!(
                build_tick_circuit(&[gate], &meta),
                Err(DemSimulationError::TickGate(_))
            ));
        }
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
