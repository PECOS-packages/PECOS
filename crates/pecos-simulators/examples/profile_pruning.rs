use pecos_core::QubitId;
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, StabVec};
use std::time::Instant;

fn run_test(
    nq: usize,
    nrz: usize,
    threshold: f64,
) -> Result<(), pecos_simulators::InvalidPruningThreshold> {
    let theta = pecos_core::radians!(0.3);
    let mut sim = StabVec::builder(nq)
        .seed(42)
        .pruning_threshold(threshold)?
        .build();

    for q in 0..nq {
        sim.h(&[QubitId(q)]);
    }
    for q in 0..nq - 1 {
        sim.cx(&[(QubitId(q), QubitId(q + 1))]);
    }

    let t0 = Instant::now();
    for r in 0..nrz {
        let q = r % nq;
        sim.rz(theta, &[QubitId(q)]);
        sim.h(&[QubitId(q)]);
    }
    let t_circuit = t0.elapsed();

    let terms_before = sim.num_terms();
    let t0 = Instant::now();
    let all_q: Vec<QubitId> = (0..nq).map(QubitId).collect();
    for &q in &all_q {
        let _ = sim.mz(&[q]);
    }
    let t_meas = t0.elapsed();
    let terms_after = sim.num_terms();

    eprintln!(
        "  threshold={threshold:.0e}: T={terms_before}->{terms_after}, circuit={t_circuit:.1?}, meas={t_meas:.1?}, total={:.1?}",
        t_circuit + t_meas
    );
    Ok(())
}

fn main() -> Result<(), pecos_simulators::InvalidPruningThreshold> {
    let nq: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let nrz: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(14);

    eprintln!("n={nq}, nrz={nrz}:");
    run_test(nq, nrz, 0.0)?; // exact
    run_test(nq, nrz, 1e-12)?;
    run_test(nq, nrz, 1e-8)?; // default
    run_test(nq, nrz, 1e-6)?;
    run_test(nq, nrz, 1e-4)?;
    run_test(nq, nrz, 1e-3)?;
    run_test(nq, nrz, 1e-2)?;
    Ok(())
}
