// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use
// this file except in compliance with the License. You may obtain a copy of the
// License at https://www.apache.org/licenses/LICENSE-2.0

//! Measurement-only wall-clock benchmark for issue #704.
//!
//! H on every qubit, then RZ(pi/4), H, CX(q, q+1) for q = 0..T.
//! Measure qubits 0, 1, 2, with RZ(pi/4), H on the next measured qubit
//! between measurements. Preparation and re-divergence are outside the timer.

use pecos_core::{Angle64, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, StabVec};
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let runs: usize = std::env::args()
        .nth(1)
        .map_or(10, |value| value.parse().expect("runs must be an integer"));
    let theta = Angle64::from_radians(std::f64::consts::FRAC_PI_4);
    println!("qubits,t_gates,run,measurement,terms,elapsed_ns,outcome,deterministic");
    for (num_qubits, t_gates) in [(8, 6), (10, 8), (12, 10)] {
        for run in 0..=runs {
            let mut sim = StabVec::builder(num_qubits)
                .seed(42)
                .pruning_threshold(0.0)
                .build();
            for q in 0..num_qubits {
                sim.h(&[QubitId(q)]);
            }
            for q in 0..t_gates {
                sim.rz(theta, &[QubitId(q)]);
                sim.h(&[QubitId(q)]);
                sim.cx(&[(QubitId(q), QubitId(q + 1))]);
            }
            for measurement in 0..3 {
                if measurement > 0 {
                    sim.rz(theta, &[QubitId(measurement)]);
                    sim.h(&[QubitId(measurement)]);
                }
                assert!(!sim.has_shared_projection_structure());
                let terms = sim.num_terms();
                let start = Instant::now();
                let result = black_box(sim.mz(&[QubitId(measurement)]));
                let elapsed = start.elapsed();
                assert!(!sim.is_dense());
                // Run zero warms up each shape and is not reported.
                if run > 0 {
                    println!(
                        "{num_qubits},{t_gates},{run},{},{terms},{},{},{}",
                        measurement + 1,
                        elapsed.as_nanos(),
                        result[0].outcome,
                        result[0].is_deterministic,
                    );
                }
            }
        }
    }
}
