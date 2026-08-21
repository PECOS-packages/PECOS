// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0

use pecos_qec::fault_tolerance::dem_builder::DemBuilder;
use pecos_qec::{SurfaceCode, connected_cluster_fault_distance, exhaustive_fault_distance};
use pecos_quantum::{Attribute, DagCircuit, TickCircuit, TickMeasRef};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let distance: usize = args[1].parse().unwrap();
    let rounds: usize = args[2].parse().unwrap();
    let max_weight: usize = args[3].parse().unwrap();
    let run_exhaustive = args.get(4).is_some_and(|arg| arg == "blind");

    let tick = build_surface_memory(distance, rounds);
    let dag = DagCircuit::try_from(&tick).unwrap();
    let dem = DemBuilder::from_circuit(&dag, 0.001, 0.001, 0.001, 0.001);
    let mechanism_count = dem.to_mechanisms().0.len();

    let started = Instant::now();
    let connected = connected_cluster_fault_distance(&dem, max_weight);
    let connected_elapsed = started.elapsed();

    println!(
        "d={distance} rounds={rounds} mechanisms={mechanism_count} max_weight={max_weight} connected={connected:?} connected_ns={}",
        connected_elapsed.as_nanos()
    );

    if run_exhaustive {
        let started = Instant::now();
        let exhaustive = exhaustive_fault_distance(&dem, max_weight);
        let exhaustive_elapsed = started.elapsed();
        println!(
            "exhaustive={exhaustive:?} exhaustive_ns={}",
            exhaustive_elapsed.as_nanos()
        );
        assert_eq!(connected, exhaustive);
    }
}

fn build_surface_memory(distance: usize, rounds: usize) -> TickCircuit {
    let code = SurfaceCode::rotated(distance).unwrap();
    let num_data = code.num_data_qubits();
    let x_ancilla = |index: usize| num_data + index;
    let z_ancilla = |index: usize| num_data + code.num_x_stabilizers() + index;
    let data_qubits: Vec<usize> = (0..num_data).collect();
    let x_ancillas: Vec<usize> = (0..code.num_x_stabilizers()).map(x_ancilla).collect();
    let z_ancillas: Vec<usize> = (0..code.num_z_stabilizers()).map(z_ancilla).collect();

    let mut circuit = TickCircuit::new();
    circuit.tick().pz(&data_qubits);
    let mut x_rounds = Vec::with_capacity(rounds);
    let mut z_rounds = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        circuit.tick().pz(&x_ancillas);
        circuit.tick().pz(&z_ancillas);
        circuit.tick().h(&x_ancillas);
        for check in code.x_stabilizers() {
            for data in check.qubits() {
                circuit.tick().cx(&[(x_ancilla(check.index), data)]);
            }
        }
        for check in code.z_stabilizers() {
            for data in check.qubits() {
                circuit.tick().cx(&[(data, z_ancilla(check.index))]);
            }
        }
        circuit.tick().h(&x_ancillas);
        x_rounds.push(circuit.tick().mz(&x_ancillas));
        z_rounds.push(circuit.tick().mz(&z_ancillas));
    }

    let final_data = circuit.tick().mz(&data_qubits);
    let mut detectors = Vec::new();
    for &measurement in &z_rounds[0] {
        detectors.push(measurement_ids(&[measurement]));
    }
    for round in 1..rounds {
        for (&current, &previous) in x_rounds[round].iter().zip(&x_rounds[round - 1]) {
            detectors.push(measurement_ids(&[current, previous]));
        }
        for (&current, &previous) in z_rounds[round].iter().zip(&z_rounds[round - 1]) {
            detectors.push(measurement_ids(&[current, previous]));
        }
    }
    for check in code.z_stabilizers() {
        let mut measurements = vec![z_rounds[rounds - 1][check.index]];
        measurements.extend(check.qubits().into_iter().map(|qubit| final_data[qubit]));
        detectors.push(measurement_ids(&measurements));
    }

    let observable = code
        .logical_z()
        .data_qubits
        .iter()
        .map(|&qubit| final_data[qubit])
        .collect::<Vec<_>>();
    circuit.set_meta(
        "num_measurements",
        Attribute::String(circuit.num_measurements().to_string()),
    );
    circuit.set_meta("detectors", Attribute::String(annotations_json(&detectors)));
    circuit.set_meta(
        "observables",
        Attribute::String(annotations_json(&[measurement_ids(&observable)])),
    );
    circuit
}

fn measurement_ids(measurements: &[TickMeasRef]) -> Vec<usize> {
    measurements
        .iter()
        .map(|measurement| measurement.meas_id.index())
        .collect()
}

fn annotations_json(annotations: &[Vec<usize>]) -> String {
    let entries = annotations
        .iter()
        .map(|ids| {
            let ids = ids
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",");
            format!(r#"{{"meas_ids":[{ids}]}}"#)
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("[{entries}]")
}
