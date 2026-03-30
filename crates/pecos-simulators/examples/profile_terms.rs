use pecos_core::{Angle64, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, CliffordRz};
use std::time::Instant;

/// Benchmark that actually creates many terms by interleaving H and RZ.
fn main() {
    let nq: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let nrz: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    let theta = Angle64::from_radians(0.3);
    let mut sim = CliffordRz::new_with_seed(nq, 42);

    // Create entangled state
    for q in 0..nq {
        sim.h(&[QubitId(q)]);
    }
    for q in 0..nq - 1 {
        sim.cx(&[(QubitId(q), QubitId(q + 1))]);
    }

    // Interleave H and RZ to force term doubling
    let t0 = Instant::now();
    for r in 0..nrz {
        let q = r % nq;
        sim.rz(theta, &[QubitId(q)]);
        sim.h(&[QubitId(q)]);
    }
    let t_circuit = t0.elapsed();
    eprintln!("After circuit: {} terms ({t_circuit:.1?})", sim.num_terms());

    // Measure all qubits
    let t0 = Instant::now();
    for q in 0..nq {
        let _ = sim.mz(&[QubitId(q)]);
    }
    let t_meas = t0.elapsed();
    eprintln!(
        "After measurement: {} terms ({t_meas:.1?})",
        sim.num_terms()
    );
    eprintln!("Total: {:.1?}", t_circuit + t_meas);
}
