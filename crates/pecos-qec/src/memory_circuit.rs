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

//! Shared construction machinery for CSS memory experiments.

use pecos_quantum::{AnnotationKind, Attribute, F2Matrix, TickCircuit, TickMeasRef};

use crate::ParityCheckMatrix;

/// Memory-experiment preparation and final-measurement basis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryBasis {
    /// Prepare and measure the encoded state in the X basis.
    X,
    /// Prepare and measure the encoded state in the Z basis.
    Z,
}

#[derive(Clone, Copy)]
pub(crate) struct CssMemoryCircuitFinish<'a> {
    pub data_qubits: &'a [usize],
    pub hx: &'a ParityCheckMatrix,
    pub hz: &'a ParityCheckMatrix,
    pub logical_x: &'a ParityCheckMatrix,
    pub logical_z: &'a ParityCheckMatrix,
    pub x_measurements: &'a [Vec<TickMeasRef>],
    pub z_measurements: &'a [Vec<TickMeasRef>],
    pub rounds: usize,
    pub basis: MemoryBasis,
    pub circuit_type: &'a str,
}

pub(crate) fn discover_css_logical_operators(
    hx: &ParityCheckMatrix,
    hz: &ParityCheckMatrix,
) -> (ParityCheckMatrix, ParityCheckMatrix) {
    let num_qubits = hx.num_qubits();
    let logical_x = quotient_basis(hz.matrix(), hx.matrix(), num_qubits);
    let logical_z = quotient_basis(hx.matrix(), hz.matrix(), num_qubits);
    (
        parity_matrix_from_rows(logical_x, num_qubits),
        parity_matrix_from_rows(logical_z, num_qubits),
    )
}

pub(crate) fn finish_css_memory_circuit(
    circuit: &mut TickCircuit,
    finish: CssMemoryCircuitFinish<'_>,
) -> Result<(), String> {
    let CssMemoryCircuitFinish {
        data_qubits,
        hx,
        hz,
        logical_x,
        logical_z,
        x_measurements,
        z_measurements,
        rounds,
        basis,
        circuit_type,
    } = finish;
    add_cycle_detectors(circuit, x_measurements, z_measurements, basis)?;

    let final_data = match basis {
        MemoryBasis::X => circuit.tick().mx(data_qubits),
        MemoryBasis::Z => circuit.tick().mz(data_qubits),
    };
    let (closing_checks, final_logicals, last_syndrome, label) = match basis {
        MemoryBasis::X => (hx, logical_x, &x_measurements[rounds - 1], "X"),
        MemoryBasis::Z => (hz, logical_z, &z_measurements[rounds - 1], "Z"),
    };
    for (check, &syndrome) in last_syndrome.iter().enumerate() {
        let mut refs = vec![syndrome];
        for (data, &bit) in closing_checks
            .row(check)
            .expect("the check index comes from its measurement list")
            .iter()
            .enumerate()
        {
            if bit == 1 {
                refs.push(final_data[data]);
            }
        }
        annotate_detector(circuit, &format!("{label}{check}_final"), &refs)?;
    }
    for logical in 0..final_logicals.num_checks() {
        let refs = final_logicals
            .row(logical)
            .expect("the logical index is in range")
            .iter()
            .enumerate()
            .filter_map(|(data, &bit)| (bit == 1).then_some(final_data[data]))
            .collect::<Vec<_>>();
        circuit
            .observable_labeled(&format!("L{logical}"), &refs)
            .map_err(|error| error.to_string())?;
    }

    let (detectors_json, observables_json, num_detectors, num_observables) =
        annotation_metadata_json(circuit);
    circuit.set_meta(
        "num_measurements",
        Attribute::String(circuit.num_measurements().to_string()),
    );
    circuit.set_meta("detectors", Attribute::String(detectors_json));
    circuit.set_meta("observables", Attribute::String(observables_json));
    circuit.set_meta(
        "num_detectors",
        Attribute::String(num_detectors.to_string()),
    );
    circuit.set_meta(
        "num_observables",
        Attribute::String(num_observables.to_string()),
    );
    circuit.set_meta(
        "num_data_qubits",
        Attribute::String(data_qubits.len().to_string()),
    );
    circuit.set_meta(
        "num_logical_qubits",
        Attribute::String(final_logicals.num_checks().to_string()),
    );
    circuit.set_meta("syndrome_cycles", Attribute::String(rounds.to_string()));
    circuit.set_meta(
        "syndrome_extraction_depth",
        Attribute::String((circuit.num_ticks() - 1).to_string()),
    );
    circuit.set_meta("circuit_type", Attribute::String(circuit_type.to_string()));
    Ok(())
}

fn add_cycle_detectors(
    circuit: &mut TickCircuit,
    x_measurements: &[Vec<TickMeasRef>],
    z_measurements: &[Vec<TickMeasRef>],
    basis: MemoryBasis,
) -> Result<(), String> {
    for round in 0..x_measurements.len() {
        if round == 0 {
            let (label, measurements) = match basis {
                MemoryBasis::X => ("X", &x_measurements[0]),
                MemoryBasis::Z => ("Z", &z_measurements[0]),
            };
            for (check, &measurement) in measurements.iter().enumerate() {
                annotate_detector(circuit, &format!("{label}{check}_r0"), &[measurement])?;
            }
            continue;
        }
        let num_checks = x_measurements[round].len().max(z_measurements[round].len());
        for check in 0..num_checks {
            if let (Some(&previous), Some(&current)) = (
                x_measurements[round - 1].get(check),
                x_measurements[round].get(check),
            ) {
                annotate_detector(circuit, &format!("X{check}_r{round}"), &[previous, current])?;
            }
            if let (Some(&previous), Some(&current)) = (
                z_measurements[round - 1].get(check),
                z_measurements[round].get(check),
            ) {
                annotate_detector(circuit, &format!("Z{check}_r{round}"), &[previous, current])?;
            }
        }
    }
    Ok(())
}

fn annotate_detector(
    circuit: &mut TickCircuit,
    label: &str,
    measurements: &[TickMeasRef],
) -> Result<(), String> {
    circuit
        .detector_labeled(label, measurements)
        .map(|_| ())
        .map_err(|error| error.to_string())
}

fn annotation_metadata_json(circuit: &TickCircuit) -> (String, String, usize, usize) {
    let mut detectors = Vec::new();
    let mut observables = Vec::new();
    for annotation in circuit.annotations() {
        match &annotation.kind {
            AnnotationKind::Detector {
                measurement_ids,
                coords: _,
            } => {
                let id = detectors.len();
                detectors.push(serde_json::json!({
                    "id": id,
                    "meas_ids": measurement_ids.iter().map(|id| id.index()).collect::<Vec<_>>(),
                    "label": annotation.label,
                }));
            }
            AnnotationKind::Observable { measurement_ids } => {
                let id = observables.len();
                observables.push(serde_json::json!({
                    "id": id,
                    "meas_ids": measurement_ids.iter().map(|id| id.index()).collect::<Vec<_>>(),
                    "label": annotation.label,
                }));
            }
            AnnotationKind::TrackedPauli => {}
        }
    }
    let num_detectors = detectors.len();
    let num_observables = observables.len();
    (
        serde_json::to_string(&detectors).expect("annotation metadata is JSON-serializable"),
        serde_json::to_string(&observables).expect("annotation metadata is JSON-serializable"),
        num_detectors,
        num_observables,
    )
}

fn quotient_basis(kernel_matrix: &F2Matrix, stabilizers: &F2Matrix, width: usize) -> Vec<Vec<u8>> {
    let (stabilizer_rref, pivots) = stabilizers.row_reduce();
    let reduced = kernel_matrix.kernel().into_iter().filter_map(|mut vector| {
        for (row, &pivot) in pivots.iter().enumerate() {
            if vector[pivot] == 1 {
                for (column, bit) in vector.iter_mut().enumerate() {
                    *bit ^= stabilizer_rref.get(row, column);
                }
            }
        }
        vector.iter().any(|&bit| bit != 0).then_some(vector)
    });
    let candidates: Vec<_> = reduced.collect();
    if candidates.is_empty() {
        return Vec::new();
    }
    let (rref, _) = F2Matrix::from_rows(candidates).row_reduce();
    let rows = rref
        .rows()
        .into_iter()
        .filter(|row| row.iter().any(|&bit| bit != 0))
        .collect::<Vec<_>>();
    debug_assert!(rows.iter().all(|row| row.len() == width));
    rows
}

fn parity_matrix_from_rows(rows: Vec<Vec<u8>>, width: usize) -> ParityCheckMatrix {
    if rows.is_empty() {
        ParityCheckMatrix::zeros(0, width)
    } else {
        ParityCheckMatrix::from_dense(rows)
            .expect("a rectangular logical-operator matrix was generated")
    }
}
