use pecos_core::{Angle64, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, CliffordRz};
use std::time::Instant;

/// QEC-like circuit: repeated rounds of gates + measurement on ancilla qubits.
fn main() {
    let data_q: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let ancilla_q: usize = data_q;
    let rounds: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let nrz: usize = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let nq = data_q + ancilla_q;

    let theta = Angle64::from_radians(0.3);
    let mut sim = CliffordRz::new_with_seed(nq, 42);

    // Initialize
    for q in 0..data_q {
        sim.h(&[QubitId(q)]);
    }

    let mut t_gates = std::time::Duration::ZERO;
    let mut t_rz = std::time::Duration::ZERO;
    let mut t_meas = std::time::Duration::ZERO;

    // Pre-build batched gate arguments
    let cx_pairs: Vec<(QubitId, QubitId)> = (0..data_q.min(ancilla_q))
        .map(|i| (QubitId(i), QubitId(data_q + i)))
        .collect();
    let ancilla_qubits: Vec<QubitId> = (0..ancilla_q).map(|i| QubitId(data_q + i)).collect();

    for round in 0..rounds {
        let t0 = Instant::now();
        sim.cx(&cx_pairs);
        t_gates += t0.elapsed();

        let t0 = Instant::now();
        for r in 0..nrz {
            sim.rz(theta, &[QubitId(r % data_q)]);
        }
        t_rz += t0.elapsed();

        let t0 = Instant::now();
        for i in 0..ancilla_q {
            let _ = sim.mz(&[QubitId(data_q + i)]);
        }
        t_meas += t0.elapsed();

        let t0 = Instant::now();
        sim.h(&ancilla_qubits);
        t_gates += t0.elapsed();
        eprintln!("  round {round}: {} terms", sim.num_terms());
    }

    let total = t_gates + t_rz + t_meas;
    eprintln!(
        "nq={nq}, rounds={rounds}, nrz={nrz}, terms={}",
        sim.num_terms()
    );
    eprintln!(
        "  gates: {:>8.1?} ({:.0}%)",
        t_gates,
        100.0 * t_gates.as_secs_f64() / total.as_secs_f64()
    );
    eprintln!(
        "  rz:    {:>8.1?} ({:.0}%)",
        t_rz,
        100.0 * t_rz.as_secs_f64() / total.as_secs_f64()
    );
    eprintln!(
        "  meas:  {:>8.1?} ({:.0}%)",
        t_meas,
        100.0 * t_meas.as_secs_f64() / total.as_secs_f64()
    );
    eprintln!("  total: {total:>8.1?}");
}
