use pecos_core::{Angle64, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, CliffordRz};
use std::time::Instant;

fn main() {
    let nq: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let nrz: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);

    let theta = Angle64::from_radians(0.3);
    let mut sim = CliffordRz::new_with_seed(nq, 42);

    for q in 0..nq {
        sim.h(&[QubitId(q)]);
    }
    for q in 0..nq - 1 {
        sim.cx(&[(QubitId(q), QubitId(q + 1))]);
    }
    for r in 0..nrz {
        let q = r % nq;
        sim.rz(theta, &[QubitId(q)]);
        sim.h(&[QubitId(q)]);
    }
    let t = sim.num_terms();
    eprintln!("n={nq}, T={t}");

    // Time precompute_shared_constraints
    let iters = 1000;
    let t0 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(sim.num_terms());
    }
    let sc_time = t0.elapsed() / iters;
    eprintln!("precompute_shared_constraints: {sc_time:.1?}");

    // Time a single measurement
    let _sim2 = sim.clone();
    let t0 = Instant::now();
    let iters_meas = 100;
    for _seed in 0..iters_meas {
        let mut s = sim.clone();
        let _ = s.mz(&[QubitId(0)]);
    }
    let meas_time = t0.elapsed() / iters_meas;
    eprintln!("single mz: {meas_time:.1?}");
}
