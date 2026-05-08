use pecos_core::{Angle64, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, StabVec, StateVec};

/// Verify MC sampling gives similar statistics to exact state vector.
fn main() {
    let nq = 8;
    let nrz = 6;
    let theta = Angle64::from_radians(0.3);
    let num_shots = 10000;

    // Build reference state vector
    let mut sv = StateVec::new(nq);
    for q in 0..nq {
        sv.h(&[QubitId(q)]);
    }
    for q in 0..nq - 1 {
        sv.cx(&[(QubitId(q), QubitId(q + 1))]);
    }
    for r in 0..nrz {
        let q = r % nq;
        sv.rz(theta, &[QubitId(q)]);
        sv.h(&[QubitId(q)]);
    }

    // Compute exact probabilities from state vector
    let state = sv.state();
    let dim = 1 << nq;
    let mut exact_probs = vec![0.0f64; dim];
    let norm: f64 = state.iter().map(num_complex::Complex::norm_sqr).sum();
    for x in 0..dim {
        exact_probs[x] = state[x].norm_sqr() / norm;
    }

    // Sample from StabVec (which uses MC for T > 2048, exact otherwise)
    // Force the MC path by temporarily using it
    let mut mc_counts = vec![0u32; dim];
    #[allow(clippy::cast_sign_loss)] // num_shots is a positive literal
    for seed in 0..num_shots as u64 {
        let mut crz = StabVec::new_with_seed(nq, seed);
        for q in 0..nq {
            crz.h(&[QubitId(q)]);
        }
        for q in 0..nq - 1 {
            crz.cx(&[(QubitId(q), QubitId(q + 1))]);
        }
        for r in 0..nrz {
            let q = r % nq;
            crz.rz(theta, &[QubitId(q)]);
            crz.h(&[QubitId(q)]);
        }
        let qubits: Vec<QubitId> = (0..nq).map(QubitId).collect();
        let results = crz.mz(&qubits);
        let mut idx = 0;
        for (i, r) in results.iter().enumerate() {
            if r.outcome {
                idx |= 1 << i;
            }
        }
        mc_counts[idx] += 1;
    }

    // Compare
    let tolerance = 5.0 / f64::from(num_shots).sqrt();
    let mut max_diff = 0.0f64;
    let mut total_diff = 0.0f64;
    for x in 0..dim {
        let mc_prob = f64::from(mc_counts[x]) / f64::from(num_shots);
        let diff = (mc_prob - exact_probs[x]).abs();
        max_diff = max_diff.max(diff);
        total_diff += diff;
        if diff > tolerance && exact_probs[x] > 0.01 {
            eprintln!(
                "  |{x:0>nq$b}>: exact={:.4}, mc={:.4}, diff={:.4}",
                exact_probs[x],
                mc_prob,
                diff,
                nq = nq
            );
        }
    }
    eprintln!("n={nq}, nrz={nrz}, T={} (exact path, not MC)", {
        let mut c = StabVec::new(nq);
        for q in 0..nq {
            c.h(&[QubitId(q)]);
        }
        for q in 0..nq - 1 {
            c.cx(&[(QubitId(q), QubitId(q + 1))]);
        }
        for r in 0..nrz {
            let q = r % nq;
            c.rz(theta, &[QubitId(q)]);
            c.h(&[QubitId(q)]);
        }
        c.num_terms()
    });
    eprintln!(
        "shots={num_shots}, max_diff={max_diff:.4}, total_diff={total_diff:.4}, tolerance={tolerance:.4}"
    );
}
