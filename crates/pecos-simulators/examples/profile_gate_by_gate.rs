//! Reproducible gate-by-gate sampling workloads; pin the executable with taskset.
use pecos_core::QubitId;
use pecos_random::{PecosRng, RngExt};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, StabVec};
use std::time::Instant;

fn main() {
    let workload = std::env::args().nth(1).unwrap_or_else(|| "random".into());
    let rotations: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let shots: u32 = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let num_qubits: usize = std::env::args()
        .nth(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let start = Instant::now();
    for seed in 0..shots {
        let mut gates = PecosRng::seed_from_u64(701);
        let mut sim = StabVec::builder(num_qubits)
            .seed(u64::from(seed))
            .pruning_threshold(0.0)
            .build();
        for q in 0..num_qubits {
            sim.h(&[QubitId(q)]);
        }
        for step in 0..120 {
            let q = gates.random_range(0..num_qubits);
            let t = (q + gates.random_range(1..num_qubits)) % num_qubits;
            match gates.random_range(0..3) {
                0 => {
                    sim.h(&[QubitId(q)]);
                }
                1 => {
                    sim.sz(&[QubitId(q)]);
                }
                _ => {
                    sim.cx(&[(QubitId(q), QubitId(t))]);
                }
            }
            if step < rotations {
                sim.rz(pecos_core::Angle64::QUARTER_TURN / 2u64, &[QubitId(q)])
                    .h(&[QubitId(q)]);
            }
            if workload == "sxx" {
                sim.sxx(&[(QubitId(q), QubitId(t))]);
            }
        }
        let qubits: Vec<_> = (0..num_qubits).map(QubitId).collect();
        std::hint::black_box(sim.mz(&qubits));
    }
    println!(
        "{workload} n={num_qubits} t={rotations} shots={shots} ms/shot={:.6}",
        start.elapsed().as_secs_f64() * 1000.0 / f64::from(shots)
    );
}
