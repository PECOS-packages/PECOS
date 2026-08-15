// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific language governing permissions and
// limitations under the License.

//! Generic CSS syndrome extraction scheduled by Tanner-graph edge coloration.
//!
//! The coloration construction follows arXiv:2308.08648. Konig's theorem gives an exact
//! Delta-edge-coloring of each bipartite Tanner graph, so every color is a depth-one matching.
//! This guarantees a valid syndrome-extraction schedule; it does not imply that the resulting
//! circuit preserves the distance of the underlying code.

use pecos_num::graph::{BipartiteEdgeColoringError, bipartite_edge_coloring};
use pecos_quantum::{Attribute, TickCircuit, TickMeasRef};
use thiserror::Error;

use crate::memory_circuit::{
    CssMemoryCircuitFinish, discover_css_logical_operators, finish_css_memory_circuit,
};
use crate::{MemoryBasis, ParityCheckMatrix, StabilizerCodeSpec};

/// Errors reported while constructing a coloration-scheduled CSS memory circuit.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ColorationMemoryError {
    /// The matrices do not describe any data qubits.
    #[error("coloration memory experiment requires at least one data qubit")]
    ZeroQubits,
    /// The two parity-check matrices have different widths.
    #[error("CSS parity-check matrices have different widths: Hx has {hx} qubits, Hz has {hz}")]
    MismatchedQubitCount {
        /// Width of Hx.
        hx: usize,
        /// Width of Hz.
        hz: usize,
    },
    /// Existing CSS validation rejected the matrices.
    #[error("invalid CSS check structure: {0}")]
    InvalidCssStructure(String),
    /// At least one syndrome cycle is required.
    #[error("coloration memory experiment requires at least one syndrome cycle")]
    ZeroRounds,
    /// The data and ancilla register sizes overflowed `usize`.
    #[error("coloration memory circuit size overflows usize")]
    SizeOverflow,
    /// Exact coloring of a Tanner graph failed.
    #[error(transparent)]
    EdgeColoring(#[from] BipartiteEdgeColoringError),
    /// A supposedly matching color layer violated a tick-circuit invariant.
    #[error("invalid coloration CNOT layer: {0}")]
    InvalidSchedule(String),
    /// A measurement reference could not be annotated.
    #[error("invalid coloration memory annotation: {0}")]
    InvalidAnnotation(String),
}

#[derive(Clone, Copy)]
enum CnotDirection {
    DataToAncilla,
    AncillaToData,
}

/// Build a memory experiment for any validated CSS check pair using exact edge coloration.
///
/// Each syndrome cycle resets one ancilla per check, applies all Z-check colors as CNOTs from
/// data to Z ancillas, applies all X-check colors as CNOTs from X ancillas to data, and measures
/// the ancillas. The entangling depth per cycle is `Delta(Hz) + Delta(Hx)`. Detectors compare
/// consecutive syndrome cycles and close the prepared-basis checks against final data
/// measurements. Logical observables are discovered from the CSS check spaces.
///
/// The schedule is deterministic for a fixed pair of matrices. Its matching layers guarantee
/// circuit validity, but no distance-preservation claim is made.
///
/// # Errors
///
/// Returns an error for zero rounds, incompatible or nonorthogonal CSS matrices, size overflow,
/// edge-coloring failure, invalid CNOT layers, or invalid measurement annotations.
pub fn coloration_memory_circuit(
    hx: &ParityCheckMatrix,
    hz: &ParityCheckMatrix,
    rounds: usize,
    basis: MemoryBasis,
) -> Result<TickCircuit, ColorationMemoryError> {
    if rounds == 0 {
        return Err(ColorationMemoryError::ZeroRounds);
    }
    if hx.num_qubits() != hz.num_qubits() {
        return Err(ColorationMemoryError::MismatchedQubitCount {
            hx: hx.num_qubits(),
            hz: hz.num_qubits(),
        });
    }
    let num_data = hx.num_qubits();
    if num_data == 0 {
        return Err(ColorationMemoryError::ZeroQubits);
    }
    StabilizerCodeSpec::builder(num_data)
        .checks_from_css(hx, hz)
        .map_err(|error| ColorationMemoryError::InvalidCssStructure(error.to_string()))?;

    let num_x_checks = hx.num_checks();
    let num_z_checks = hz.num_checks();
    let x_ancilla_start = 0;
    let data_start = num_x_checks;
    let z_ancilla_start = data_start
        .checked_add(num_data)
        .ok_or(ColorationMemoryError::SizeOverflow)?;
    z_ancilla_start
        .checked_add(num_z_checks)
        .ok_or(ColorationMemoryError::SizeOverflow)?;

    let x_edges = tanner_edges(hx);
    let z_edges = tanner_edges(hz);
    let x_coloring = bipartite_edge_coloring(num_data, num_x_checks, &x_edges)?;
    let z_coloring = bipartite_edge_coloring(num_data, num_z_checks, &z_edges)?;
    let entangling_depth = x_coloring
        .num_colors()
        .checked_add(z_coloring.num_colors())
        .ok_or(ColorationMemoryError::SizeOverflow)?;
    let cycle_depth = entangling_depth
        .checked_add(3)
        .ok_or(ColorationMemoryError::SizeOverflow)?;

    let data_qubits = (data_start..z_ancilla_start).collect::<Vec<_>>();
    let x_ancillas = (x_ancilla_start..data_start).collect::<Vec<_>>();
    let z_ancillas = (z_ancilla_start..z_ancilla_start + num_z_checks).collect::<Vec<_>>();
    let (logical_x, logical_z) = discover_css_logical_operators(hx, hz);

    let mut circuit = TickCircuit::new();
    let mut x_measurements: Vec<Vec<TickMeasRef>> = Vec::with_capacity(rounds);
    let mut z_measurements: Vec<Vec<TickMeasRef>> = Vec::with_capacity(rounds);
    for round in 0..rounds {
        let mut init = circuit.tick();
        if !x_ancillas.is_empty() {
            init.try_add_gate(pecos_core::Gate::px(&x_ancillas))
                .map_err(|error| ColorationMemoryError::InvalidSchedule(error.to_string()))?;
        }
        if !z_ancillas.is_empty() {
            init.try_add_gate(pecos_core::Gate::pz(&z_ancillas))
                .map_err(|error| ColorationMemoryError::InvalidSchedule(error.to_string()))?;
        }
        if round == 0 {
            let preparation = match basis {
                MemoryBasis::X => pecos_core::Gate::px(&data_qubits),
                MemoryBasis::Z => pecos_core::Gate::pz(&data_qubits),
            };
            init.try_add_gate(preparation)
                .map_err(|error| ColorationMemoryError::InvalidSchedule(error.to_string()))?;
        }

        append_colored_cnot_layers(
            &mut circuit,
            &z_edges,
            z_coloring.colors(),
            z_coloring.num_colors(),
            data_start,
            z_ancilla_start,
            CnotDirection::DataToAncilla,
        )?;
        append_colored_cnot_layers(
            &mut circuit,
            &x_edges,
            x_coloring.colors(),
            x_coloring.num_colors(),
            data_start,
            x_ancilla_start,
            CnotDirection::AncillaToData,
        )?;

        let z_refs = if z_ancillas.is_empty() {
            circuit.tick();
            Vec::new()
        } else {
            circuit.tick().mz(&z_ancillas)
        };
        let x_refs = if x_ancillas.is_empty() {
            circuit.tick();
            Vec::new()
        } else {
            circuit.tick().mx(&x_ancillas)
        };
        z_measurements.push(z_refs);
        x_measurements.push(x_refs);
    }

    finish_css_memory_circuit(
        &mut circuit,
        CssMemoryCircuitFinish {
            data_qubits: &data_qubits,
            hx,
            hz,
            logical_x: &logical_x,
            logical_z: &logical_z,
            x_measurements: &x_measurements,
            z_measurements: &z_measurements,
            rounds,
            basis,
            circuit_type: "coloration_css_memory",
        },
    )
    .map_err(ColorationMemoryError::InvalidAnnotation)?;
    circuit.set_meta(
        "num_ancilla_qubits",
        Attribute::String((num_x_checks + num_z_checks).to_string()),
    );
    circuit.set_meta(
        "num_x_ancillas",
        Attribute::String(num_x_checks.to_string()),
    );
    circuit.set_meta(
        "num_z_ancillas",
        Attribute::String(num_z_checks.to_string()),
    );
    circuit.set_meta(
        "x_coloration_depth",
        Attribute::String(x_coloring.num_colors().to_string()),
    );
    circuit.set_meta(
        "z_coloration_depth",
        Attribute::String(z_coloring.num_colors().to_string()),
    );
    circuit.set_meta(
        "entangling_depth_per_cycle",
        Attribute::String(entangling_depth.to_string()),
    );
    circuit.set_meta(
        "syndrome_cycle_depth",
        Attribute::String(cycle_depth.to_string()),
    );
    Ok(circuit)
}

fn tanner_edges(matrix: &ParityCheckMatrix) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();
    for (check, row) in matrix.rows().iter().enumerate() {
        for (data, &bit) in row.iter().enumerate() {
            if bit == 1 {
                edges.push((data, check));
            }
        }
    }
    edges
}

fn append_colored_cnot_layers(
    circuit: &mut TickCircuit,
    edges: &[(usize, usize)],
    colors: &[usize],
    num_colors: usize,
    data_start: usize,
    ancilla_start: usize,
    direction: CnotDirection,
) -> Result<(), ColorationMemoryError> {
    if edges.len() != colors.len() {
        return Err(ColorationMemoryError::InvalidSchedule(format!(
            "{} Tanner edges have {} colors",
            edges.len(),
            colors.len()
        )));
    }
    for color in 0..num_colors {
        let pairs = edges
            .iter()
            .zip(colors)
            .filter(|&(_, &edge_color)| edge_color == color)
            .map(|(&(data, check), _)| match direction {
                CnotDirection::DataToAncilla => (data_start + data, ancilla_start + check),
                CnotDirection::AncillaToData => (ancilla_start + check, data_start + data),
            })
            .collect::<Vec<_>>();
        if pairs.is_empty() {
            return Err(ColorationMemoryError::InvalidSchedule(format!(
                "color {color} has no Tanner edges"
            )));
        }
        circuit
            .tick()
            .try_add_gate(pecos_core::Gate::cx(&pairs))
            .map_err(|error| ColorationMemoryError::InvalidSchedule(error.to_string()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SurfaceCode;
    use crate::bivariate_bicycle::{BbMonomial, BivariateBicycleCode, bb_memory_circuit};
    use crate::fault_tolerance::dem_builder::DemBuilder;
    use crate::fault_tolerance::{
        connected_cluster_fault_distance, graphlike_fault_distance, per_observable_fault_distances,
    };
    use crate::geometry::StabilizerCheck;
    use pecos_quantum::{AnnotationKind, GateType};
    use pecos_simulators::{CircuitExecutor, DenseStab};
    use std::time::Instant;

    const A: [BbMonomial; 3] = [(3, 0), (0, 1), (0, 2)];
    const B: [BbMonomial; 3] = [(0, 3), (1, 0), (2, 0)];

    fn steane_checks() -> (ParityCheckMatrix, ParityCheckMatrix) {
        let h = ParityCheckMatrix::from_dense(vec![
            vec![1, 0, 1, 0, 1, 0, 1],
            vec![0, 1, 1, 0, 0, 1, 1],
            vec![0, 0, 0, 1, 1, 1, 1],
        ])
        .unwrap();
        (h.clone(), h)
    }

    fn checks_from_surface(checks: &[StabilizerCheck], num_data: usize) -> ParityCheckMatrix {
        let rows = checks
            .iter()
            .map(|check| {
                let mut row = vec![0_u8; num_data];
                for qubit in check.qubits() {
                    row[qubit] = 1;
                }
                row
            })
            .collect();
        ParityCheckMatrix::from_dense(rows).unwrap()
    }

    fn surface_checks() -> (ParityCheckMatrix, ParityCheckMatrix) {
        let code = SurfaceCode::rotated(3).unwrap();
        (
            checks_from_surface(code.x_stabilizers(), code.num_data_qubits()),
            checks_from_surface(code.z_stabilizers(), code.num_data_qubits()),
        )
    }

    fn sample_annotation_parities(circuit: &TickCircuit, num_qubits: usize) -> Vec<bool> {
        let mut simulator = DenseStab::new(num_qubits);
        let measurements = CircuitExecutor::new(circuit).run(&mut simulator);
        circuit
            .annotations()
            .iter()
            .filter_map(|annotation| match &annotation.kind {
                AnnotationKind::Detector {
                    measurement_ids, ..
                }
                | AnnotationKind::Observable { measurement_ids } => {
                    Some(measurement_ids.iter().fold(false, |parity, id| {
                        parity ^ measurements[id.index()].outcome
                    }))
                }
                AnnotationKind::TrackedPauli => None,
            })
            .collect()
    }

    fn assert_fault_free(
        hx: &ParityCheckMatrix,
        hz: &ParityCheckMatrix,
        basis: MemoryBasis,
        expected_detectors: usize,
    ) {
        let circuit = coloration_memory_circuit(hx, hz, 2, basis).unwrap();
        let dem = DemBuilder::try_from_tick_circuit(&circuit, 0.0, 0.0, 0.0, 0.0)
            .expect("fault-free circuit has a DEM");
        assert_eq!(dem.num_detectors(), expected_detectors);
        assert!(dem.to_mechanisms().0.is_empty());

        let total_qubits = hx.num_qubits() + hx.num_checks() + hz.num_checks();
        for _ in 0..4 {
            let samples = sample_annotation_parities(&circuit, total_qubits);
            assert_eq!(
                samples,
                vec![false; expected_detectors + hx.num_qubits() - hx.rank() - hz.rank()]
            );
        }
    }

    #[test]
    fn surface_and_steane_fault_free_memory_is_clean() {
        let (surface_hx, surface_hz) = surface_checks();
        let surface_detectors = surface_hx.num_checks() + 3 * surface_hz.num_checks();
        assert_fault_free(&surface_hx, &surface_hz, MemoryBasis::Z, surface_detectors);

        let (steane_hx, steane_hz) = steane_checks();
        for basis in [MemoryBasis::X, MemoryBasis::Z] {
            let basis_checks = match basis {
                MemoryBasis::X => steane_hx.num_checks(),
                MemoryBasis::Z => steane_hz.num_checks(),
            };
            let expected_detectors =
                steane_hx.num_checks() + steane_hz.num_checks() + 2 * basis_checks;
            assert_fault_free(&steane_hx, &steane_hz, basis, expected_detectors);
        }
    }

    #[test]
    fn bb_72_coloration_is_clean_and_depth_is_measured() {
        let code = BivariateBicycleCode::new(6, 6, &A, &B).unwrap();
        let coloration =
            coloration_memory_circuit(code.hx(), code.hz(), 2, MemoryBasis::Z).unwrap();
        let specialized = bb_memory_circuit(6, 6, &A, &B, 2, MemoryBasis::Z).unwrap();

        assert_eq!(
            coloration.get_meta("z_coloration_depth"),
            Some(&Attribute::String("6".to_string()))
        );
        assert_eq!(
            coloration.get_meta("x_coloration_depth"),
            Some(&Attribute::String("6".to_string()))
        );
        assert_eq!(
            coloration.get_meta("entangling_depth_per_cycle"),
            Some(&Attribute::String("12".to_string()))
        );
        assert_eq!(
            coloration.get_meta("syndrome_cycle_depth"),
            Some(&Attribute::String("15".to_string()))
        );
        assert_eq!(coloration.num_ticks(), 31);
        assert_eq!(specialized.num_ticks(), 18);
        let coloration_entangling_layers = coloration
            .ticks()
            .iter()
            .take(15)
            .filter(|tick| {
                tick.iter_gate_instances()
                    .any(|gate| gate.gate_type() == GateType::CX)
            })
            .count();
        let specialized_entangling_layers = specialized
            .ticks()
            .iter()
            .skip(1)
            .take(8)
            .filter(|tick| {
                tick.iter_gate_instances()
                    .any(|gate| gate.gate_type() == GateType::CX)
            })
            .count();
        assert_eq!(coloration_entangling_layers, 12);
        assert_eq!(specialized_entangling_layers, 7);
        assert_eq!((coloration.num_ticks() - 1) / 2, 15);
        assert_eq!((specialized.num_ticks() - 2) / 2, 8);
        assert_eq!(
            specialized.get_meta("syndrome_extraction_depth"),
            Some(&Attribute::String("17".to_string()))
        );

        let dem = DemBuilder::try_from_tick_circuit(&coloration, 0.0, 0.0, 0.0, 0.0).unwrap();
        assert_eq!(dem.num_detectors(), 144);
        assert!(dem.to_mechanisms().0.is_empty());
    }

    #[test]
    fn builder_is_deterministic() {
        let (hx, hz) = steane_checks();
        let first = coloration_memory_circuit(&hx, &hz, 2, MemoryBasis::Z).unwrap();
        let second = coloration_memory_circuit(&hx, &hz, 2, MemoryBasis::Z).unwrap();
        assert_eq!(format!("{first:?}"), format!("{second:?}"));
    }

    #[test]
    fn corrupted_coloring_is_rejected_as_an_overlapping_cnot_layer() {
        let edges = vec![(0, 0), (0, 1)];
        let corrupted_colors = vec![0, 0];
        let mut circuit = TickCircuit::new();

        let error = append_colored_cnot_layers(
            &mut circuit,
            &edges,
            &corrupted_colors,
            1,
            0,
            2,
            CnotDirection::DataToAncilla,
        )
        .unwrap_err();
        assert!(
            matches!(error, ColorationMemoryError::InvalidSchedule(_)),
            "the TickCircuit qubit-conflict invariant must reject two same-layer CNOTs on data 0"
        );
    }

    #[test]
    fn rejects_nonorthogonal_or_mismatched_css_checks() {
        let hx = ParityCheckMatrix::from_dense(vec![vec![1, 0]]).unwrap();
        let hz = ParityCheckMatrix::from_dense(vec![vec![1, 0]]).unwrap();
        assert!(matches!(
            coloration_memory_circuit(&hx, &hz, 1, MemoryBasis::Z),
            Err(ColorationMemoryError::InvalidCssStructure(_))
        ));
        let short = ParityCheckMatrix::from_dense(vec![vec![0]]).unwrap();
        assert!(matches!(
            coloration_memory_circuit(&hx, &short, 1, MemoryBasis::Z),
            Err(ColorationMemoryError::MismatchedQubitCount { .. })
        ));
    }

    #[test]
    fn steane_two_cycle_circuit_fault_distance_is_measured() {
        let (hx, hz) = steane_checks();
        let started = Instant::now();
        let circuit = coloration_memory_circuit(&hx, &hz, 2, MemoryBasis::Z).unwrap();
        let dem = DemBuilder::try_from_tick_circuit(&circuit, 0.001, 0.001, 0.001, 0.001)
            .expect("uniform circuit noise has a DEM");
        let build_elapsed = started.elapsed();
        let (mechanisms, _) = dem.to_mechanisms();

        let search_started = Instant::now();
        let (method, overall) = match graphlike_fault_distance(&dem) {
            Ok(result) => ("graphlike", result),
            Err(error) => {
                println!("graphlike unavailable: {error}");
                (
                    "connected_cluster",
                    connected_cluster_fault_distance(&dem, 3),
                )
            }
        };
        let search_elapsed = search_started.elapsed();
        let per_started = Instant::now();
        let per_observable = per_observable_fault_distances(&dem, 3);
        let per_elapsed = per_started.elapsed();
        println!(
            "Steane coloration DEM: {} mechanisms, build {build_elapsed:?}; {method} {overall:?}, search {search_elapsed:?}; per-observable {per_observable:?}, search {per_elapsed:?}",
            mechanisms.len()
        );

        let overall = overall.expect("a logical circuit fault exists through weight three");
        println!("Steane coloration witness:");
        for &index in &overall.mechanism_indices {
            println!("mechanism[{index}] = {:?}", mechanisms[index]);
        }
        // This assertion records a circuit measurement, not a preservation guarantee. A result
        // below the Steane code distance three is evidence of hook errors in naive coloration.
        assert_eq!(overall.distance, 2);
        assert_eq!(per_observable.len(), 1);
        assert_eq!(
            per_observable[0].as_ref().map(|result| result.distance),
            Some(2)
        );
    }
}
