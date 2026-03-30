use pecos_core::{Angle64, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, CliffordRz};

fn main() {
    let nq: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let iters: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);
    let nrz: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let theta = Angle64::from_radians(0.3);
    for _ in 0..iters {
        let mut sim = CliffordRz::new_with_seed(nq, 42);
        for q in 0..nq {
            sim.h(&[QubitId(q)]);
        }
        if nq > 1 {
            sim.cx(&[(QubitId(0), QubitId(1))]);
        }
        for r in 0..nrz {
            sim.rz(theta, &[QubitId(r % nq)]);
        }
        let _ = sim.mz(&[QubitId(0)]);
    }
}
