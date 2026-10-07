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

//! An annotated observable on an MX readout is sensitive to the faults that
//! flip an X measurement, not to those that flip a Z measurement.
//!
//! Observable annotations used to be propagated as Z on every referenced
//! qubit. For an X-basis memory that made faults which are physically
//! harmless (an X error just before MX) look like detector-silent logical
//! errors. The memory tests compare against direct stabilizer simulation of
//! every single-location Pauli fault, which shares no code with the DEM
//! builder's propagation.

use pecos_qec::fault_tolerance::dem_builder::{DemBuilder, DemSampler, NoiseConfig};
use pecos_qec::{
    BbMemoryBasis, MemoryBasis, ParityCheckMatrix, bb_memory_circuit, coloration_memory_circuit,
};
use pecos_quantum::{AnnotationKind, Attribute, GateType, TickCircuit};
use pecos_random::PecosRng;
use pecos_simulators::{CircuitExecutor, SparseStab};

#[test]
fn mx_observable_flips_with_z_faults_not_x_faults() {
    // |+>, noisy X gate, MX. Detector and observable are both the readout.
    // Only Z and Y faults on the X gate flip it, and each flips the detector
    // together with the observable.
    let mut circuit = TickCircuit::new();
    circuit.tick().px(&[0]);
    circuit.tick().x(&[0]);
    let readout = circuit.tick().mx(&[0]);
    circuit.observable(&[readout[0]]).unwrap();
    // The tick-circuit DEM path reads detectors from metadata, as the memory
    // builders write it; the observable comes from the annotation under test.
    circuit.set_meta(
        "detectors",
        Attribute::String(format!(
            r#"[{{"id":0,"meas_ids":[{}]}}]"#,
            readout[0].meas_id.index()
        )),
    );
    let mut sim = SparseStab::new(1);
    let outcomes = CircuitExecutor::new(&circuit).run(&mut sim).unwrap();
    assert!(
        outcomes.iter().all(|m| m.is_deterministic && !m.outcome),
        "the noiseless readout must be a deterministic 0"
    );

    let dem = DemBuilder::try_from_tick_circuit(&circuit, 0.03, 0.0, 0.0, 0.0).unwrap();
    assert_eq!((dem.num_detectors(), dem.num_observables()), (1, 1));
    let (mechanisms, _) = dem.to_mechanisms();
    assert!(!mechanisms.is_empty(), "the noisy X gate must contribute");
    for (probability, detectors, observables) in &mechanisms {
        assert_eq!(
            (detectors.as_slice(), observables.as_slice()),
            ([0].as_slice(), [0].as_slice()),
            "mechanism with probability {probability} has the wrong signature"
        );
    }
}

/// Detector and observable parities of one noiseless run.
fn annotation_parities(circuit: &TickCircuit, num_qubits: usize) -> (Vec<bool>, Vec<bool>) {
    let mut sim = SparseStab::new(num_qubits);
    let measurements = CircuitExecutor::new(circuit).run(&mut sim).unwrap();
    let parity = |ids: &[pecos_core::MeasId]| {
        ids.iter()
            .fold(false, |value, id| value ^ measurements[id.index()].outcome)
    };
    let mut detectors = Vec::new();
    let mut observables = Vec::new();
    for annotation in circuit.annotations() {
        match &annotation.kind {
            AnnotationKind::Detector {
                measurement_ids, ..
            } => detectors.push(parity(measurement_ids)),
            AnnotationKind::Observable { measurement_ids } => {
                observables.push(parity(measurement_ids));
            }
            AnnotationKind::TrackedPauli => {}
        }
    }
    (detectors, observables)
}

/// Detector and observable flips caused by an X (`z_part == false`) or Z
/// Pauli on `qubit`, inserted before tick `tick`.
fn single_fault_effect(
    circuit: &TickCircuit,
    num_qubits: usize,
    reference_observables: &[bool],
    tick: usize,
    qubit: usize,
    z_part: bool,
) -> Vec<bool> {
    let mut faulty = circuit.clone();
    {
        let mut inserted = faulty.insert_tick(tick);
        if z_part {
            inserted.z(&[qubit]);
        } else {
            inserted.x(&[qubit]);
        }
    }
    let (detectors, observables) = annotation_parities(&faulty, num_qubits);
    detectors
        .into_iter()
        .chain(
            observables
                .iter()
                .zip(reference_observables)
                .map(|(value, reference)| value ^ reference),
        )
        .collect()
}

fn xor(left: &[bool], right: &[bool]) -> Vec<bool> {
    left.iter().zip(right).map(|(a, b)| a ^ b).collect()
}

/// Assert, by direct simulation, that no single-qubit Pauli at any tick
/// boundary and no two-qubit Pauli after any CX flips an observable without
/// firing a detector; then assert the DEM has no such mechanism either.
fn assert_no_detector_silent_logical_faults(circuit: &TickCircuit, label: &str) {
    let num_qubits = circuit.all_qubits().len();
    let (reference_detectors, reference_observables) = annotation_parities(circuit, num_qubits);
    assert!(
        reference_detectors.iter().all(|&fired| !fired),
        "{label}: noiseless detectors must be quiet"
    );
    let num_detectors = reference_detectors.len();
    let num_ticks = circuit.num_ticks();

    // effects[tick][qubit] = [X effect, Z effect]; Y is their XOR because a
    // Pauli frame acts linearly on detector and observable parities.
    let effects: Vec<Vec<[Vec<bool>; 2]>> = (0..=num_ticks)
        .map(|tick| {
            (0..num_qubits)
                .map(|qubit| {
                    if tick == 0 {
                        [Vec::new(), Vec::new()]
                    } else {
                        [false, true].map(|z_part| {
                            single_fault_effect(
                                circuit,
                                num_qubits,
                                &reference_observables,
                                tick,
                                qubit,
                                z_part,
                            )
                        })
                    }
                })
                .collect()
        })
        .collect();
    let pauli_effect = |tick: usize, qubit: usize, pauli: usize| {
        let mut effect = vec![false; num_detectors + reference_observables.len()];
        if pauli & 1 != 0 {
            effect = xor(&effect, &effects[tick][qubit][0]);
        }
        if pauli & 2 != 0 {
            effect = xor(&effect, &effects[tick][qubit][1]);
        }
        effect
    };
    let silent_logical = |effect: &[bool]| {
        !effect[..num_detectors].contains(&true) && effect[num_detectors..].contains(&true)
    };

    let mut flips_observable = 0;
    for tick in 1..=num_ticks {
        for qubit in 0..num_qubits {
            for pauli in 1..4 {
                let effect = pauli_effect(tick, qubit, pauli);
                assert!(
                    !silent_logical(&effect),
                    "{label}: Pauli {pauli} on q{qubit} before tick {tick} is detector-silent"
                );
                flips_observable += usize::from(effect[num_detectors..].contains(&true));
            }
        }
    }
    assert!(
        flips_observable > 0,
        "{label}: the oracle must see observable-flipping faults"
    );
    for (tick_index, tick) in circuit.ticks().iter().enumerate() {
        for gate in tick.iter_gate_batches() {
            if gate.gate_type != GateType::CX {
                continue;
            }
            for pair in gate.qubits.chunks(2) {
                let (control, target) = (pair[0].index(), pair[1].index());
                for control_pauli in 0..4 {
                    for target_pauli in 0..4 {
                        let effect = xor(
                            &pauli_effect(tick_index + 1, control, control_pauli),
                            &pauli_effect(tick_index + 1, target, target_pauli),
                        );
                        assert!(
                            !silent_logical(&effect),
                            "{label}: two-qubit fault after CX {control}->{target} at tick {tick_index} is detector-silent"
                        );
                    }
                }
            }
        }
    }

    let dem = DemBuilder::try_from_tick_circuit(circuit, 0.001, 0.001, 0.001, 0.001).unwrap();
    let (mechanisms, _) = dem.to_mechanisms();
    for (probability, detectors, observables) in &mechanisms {
        assert!(
            !detectors.is_empty() || observables.is_empty(),
            "{label}: DEM mechanism with probability {probability} flips {observables:?} without a detector"
        );
    }
}

#[test]
fn bb72_memory_dems_have_no_detector_silent_logical_faults() {
    let a = [(3, 0), (0, 1), (0, 2)];
    let b = [(0, 3), (1, 0), (2, 0)];
    for (basis, label) in [(BbMemoryBasis::X, "BB72 X"), (BbMemoryBasis::Z, "BB72 Z")] {
        let circuit = bb_memory_circuit(6, 6, &a, &b, 1, basis).unwrap();
        assert_no_detector_silent_logical_faults(&circuit, label);

        // The sampler shares the annotation ingest. At distance 6, a shot
        // with an undetected logical flip needs several faults, which this
        // rate and shot count do not reach.
        let sampler =
            DemSampler::from_tick_circuit(&circuit, &NoiseConfig::new(0.01, 0.01, 0.01, 0.01))
                .unwrap();
        let mut rng = PecosRng::seed_from_u64(0x0b5e);
        for shot in 0..2_000 {
            let (detectors, observables) = sampler.sample(&mut rng);
            assert!(
                detectors.contains(&true) || !observables.contains(&true),
                "{label}: sampled shot {shot} flips an observable without a detector"
            );
        }
    }
}

#[test]
fn steane_memory_dems_have_no_detector_silent_logical_faults() {
    let h = ParityCheckMatrix::from_dense(vec![
        vec![1, 0, 1, 0, 1, 0, 1],
        vec![0, 1, 1, 0, 0, 1, 1],
        vec![0, 0, 0, 1, 1, 1, 1],
    ])
    .unwrap();
    for (basis, label) in [(MemoryBasis::X, "Steane X"), (MemoryBasis::Z, "Steane Z")] {
        let circuit = coloration_memory_circuit(&h, &h, 2, basis).unwrap();
        assert_no_detector_silent_logical_faults(&circuit, label);
    }
}
