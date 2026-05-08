use pecos_core::{Angle64, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, StabVec};
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
    let mut sim = StabVec::new_with_seed(nq, 42);

    // Create entangled state with many terms
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
    eprintln!("Terms: {}", sim.num_terms());

    // Measure qubits one at a time, timing each
    let mut total = std::time::Duration::ZERO;
    for q in 0..nq {
        let t0 = Instant::now();
        let _ = sim.mz(&[QubitId(q)]);
        let elapsed = t0.elapsed();
        total += elapsed;
        eprintln!("  mz(q={q}): {:>8.1?}, terms={}", elapsed, sim.num_terms());
    }
    eprintln!("Total measurement: {total:.1?}");
}
