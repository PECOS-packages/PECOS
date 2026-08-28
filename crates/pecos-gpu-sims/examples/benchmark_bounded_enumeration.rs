// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0

use pecos_gpu_sims::GpuBoundedEnumerationBackend;
use pecos_qec::{
    ParityCheckMatrix, bounded_enumeration_code_distance,
    bounded_enumeration_code_distance_with_backend,
};
use pecos_quantum::F2Matrix;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use std::time::Instant;

fn independent_dense_rows(rng: &mut SmallRng, row_count: usize, width: usize) -> Vec<Vec<u8>> {
    let mut rows = Vec::with_capacity(row_count);
    while rows.len() < row_count {
        let candidate: Vec<_> = (0..width).map(|_| u8::from(rng.random_bool(0.5))).collect();
        let mut extended = rows.clone();
        extended.push(candidate.clone());
        if F2Matrix::from_rows(extended).row_reduce().1.len() > rows.len() {
            rows.push(candidate);
        }
    }
    rows
}

fn dense_css_pair(
    num_qubits: usize,
    num_logicals: usize,
    seed: u64,
) -> (ParityCheckMatrix, ParityCheckMatrix) {
    let x_checks = num_logicals;
    let z_checks = num_qubits - num_logicals - x_checks;
    let mut rng = SmallRng::seed_from_u64(seed);
    let hz_rows = independent_dense_rows(&mut rng, z_checks, num_qubits);
    let hz = F2Matrix::from_rows(hz_rows.clone());
    let hz_kernel = hz.kernel();
    assert_eq!(hz_kernel.len(), x_checks + num_logicals);

    let hx = F2Matrix::from_rows(hz_kernel[..x_checks].to_vec());
    let mut span = hz_rows;
    let mut span_rank = z_checks;
    let mut logical_z = Vec::with_capacity(num_logicals);
    for candidate in hx.kernel() {
        let mut extended = span.clone();
        extended.push(candidate.clone());
        let rank = F2Matrix::from_rows(extended).row_reduce().1.len();
        if rank > span_rank {
            span.push(candidate.clone());
            logical_z.push(candidate);
            span_rank = rank;
        }
    }
    assert_eq!(logical_z.len(), num_logicals);
    (
        ParityCheckMatrix::from_dense(hz.rows()).unwrap(),
        ParityCheckMatrix::from_dense(logical_z).unwrap(),
    )
}

fn main() {
    let arguments: Vec<_> = std::env::args().collect();
    assert_eq!(
        arguments.len(),
        4,
        "usage: <num-qubits> <num-logicals> <cpu|gpu>"
    );
    let num_qubits: usize = arguments[1].parse().unwrap();
    let num_logicals: usize = arguments[2].parse().unwrap();
    let mode = &arguments[3];
    let seed = 0xDE05_EC55_0026_0004;
    let (h, l) = dense_css_pair(num_qubits, num_logicals, seed);

    let started = Instant::now();
    let result = match mode.as_str() {
        "cpu" => bounded_enumeration_code_distance(&h, &l, num_qubits).unwrap(),
        "gpu" => {
            let mut backend = GpuBoundedEnumerationBackend::try_new().unwrap();
            println!("adapter={:?}", backend.adapter_info());
            bounded_enumeration_code_distance_with_backend(&h, &l, num_qubits, &mut backend)
                .unwrap()
                .unwrap()
        }
        _ => panic!("mode must be cpu or gpu"),
    };
    println!(
        "dense [[{num_qubits},{num_logicals}]] {mode}: upper_bound={}, lower_bound={}, certified={}, elapsed={:?}",
        result.upper_bound(),
        result.lower_bound(),
        result.is_certified(),
        started.elapsed(),
    );
}
