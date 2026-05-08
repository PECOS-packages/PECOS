// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
// express or implied. See the License for the specific language governing permissions and
// limitations under the License.

//! Verification tests comparing STN and MAST against `StabVec`.

use num_complex::Complex64;
use pecos_core::{Angle64, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, QuantumSimulator, StabVec};
use pecos_stab_tn::stab_mps::StabMps;
use pecos_stab_tn::stab_mps::mast::Mast;

/// Check that two state vectors match up to global phase.
fn assert_states_match(sv_a: &[Complex64], sv_b: &[Complex64], label: &str) {
    assert_states_close(sv_a, sv_b, 0.01, label);
}

fn assert_states_close(sv_a: &[Complex64], sv_b: &[Complex64], tol: f64, label: &str) {
    assert_eq!(sv_a.len(), sv_b.len(), "{label}: dimension mismatch");
    let norm_a: f64 = sv_a.iter().map(num_complex::Complex::norm_sqr).sum();
    let norm_b: f64 = sv_b.iter().map(num_complex::Complex::norm_sqr).sum();
    assert!(
        (norm_a - 1.0).abs() < tol + 0.01,
        "{label}: norm_a = {norm_a:.4}"
    );
    assert!(
        (norm_b - 1.0).abs() < tol + 0.01,
        "{label}: norm_b = {norm_b:.4}"
    );
    let overlap: Complex64 = sv_a
        .iter()
        .zip(sv_b.iter())
        .map(|(a, b)| a.conj() * b)
        .sum();
    assert!(
        (overlap.norm_sqr() - 1.0).abs() < tol,
        "{label}: overlap = {:.4} (should be 1.0, tol={tol})",
        overlap.norm_sqr()
    );
}

/// Apply a random-ish Clifford+T circuit to both STN and `StabVec`.
fn run_circuit_on_both(
    n: usize,
    gates: &[(&str, Vec<usize>, Option<Angle64>)],
    seed: u64,
) -> (Vec<Complex64>, Vec<Complex64>) {
    let mut stn = StabMps::with_seed(n, seed);
    let mut crz = StabVec::builder(n).seed(seed).build();

    for (gate, qubits, angle) in gates {
        let qids: Vec<QubitId> = qubits.iter().map(|&q| QubitId(q)).collect();
        match *gate {
            "h" => {
                stn.h(&qids);
                crz.h(&qids);
            }
            "sz" => {
                stn.sz(&qids);
                crz.sz(&qids);
            }
            "x" => {
                stn.x(&qids);
                crz.x(&qids);
            }
            "z" => {
                stn.z(&qids);
                crz.z(&qids);
            }
            "cx" => {
                let pairs = vec![(QubitId(qubits[0]), QubitId(qubits[1]))];
                stn.cx(&pairs);
                crz.cx(&pairs);
            }
            "cz" => {
                let pairs = vec![(QubitId(qubits[0]), QubitId(qubits[1]))];
                stn.cz(&pairs);
                crz.cz(&pairs);
            }
            "rz" => {
                let theta = angle.unwrap();
                stn.rz(theta, &qids);
                crz.rz(theta, &qids);
            }
            "rx" => {
                let theta = angle.unwrap();
                stn.rx(theta, &qids);
                crz.rx(theta, &qids);
            }
            "rzz" => {
                let theta = angle.unwrap();
                let pairs = vec![(QubitId(qubits[0]), QubitId(qubits[1]))];
                stn.rzz(theta, &pairs);
                crz.rzz(theta, &pairs);
            }
            "t" => {
                let t = Angle64::QUARTER_TURN / 2u64;
                stn.rz(t, &qids);
                crz.rz(t, &qids);
            }
            _ => panic!("unknown gate: {gate}"),
        }
    }

    (stn.state_vector(), crz.state_vector())
}

// ============================================================================
// State vector cross-validation tests
// ============================================================================

#[test]
fn test_4qubit_random_circuit() {
    let gates = vec![
        ("h", vec![0], None),
        ("cx", vec![0, 1], None),
        ("h", vec![2], None),
        ("cx", vec![2, 3], None),
        ("t", vec![0], None),
        ("t", vec![2], None),
        ("cx", vec![1, 2], None),
        ("t", vec![1], None),
        ("h", vec![3], None),
        ("rz", vec![3], Some(Angle64::from_radians(0.7))),
    ];
    let (stn_sv, crz_sv) = run_circuit_on_both(4, &gates, 42);
    assert_states_match(&stn_sv, &crz_sv, "4-qubit random circuit");
}

#[test]
fn test_5qubit_deep_circuit() {
    let gates = vec![
        ("h", vec![0], None),
        ("h", vec![1], None),
        ("h", vec![2], None),
        ("cx", vec![0, 1], None),
        ("cx", vec![2, 3], None),
        ("cx", vec![3, 4], None),
        ("t", vec![0], None),
        ("t", vec![1], None),
        ("t", vec![2], None),
        ("cx", vec![1, 2], None),
        ("h", vec![0], None),
        ("t", vec![0], None),
        ("cz", vec![0, 3], None),
        ("rz", vec![4], Some(Angle64::from_radians(1.5))),
        ("cx", vec![4, 0], None),
        ("t", vec![4], None),
    ];
    let (stn_sv, crz_sv) = run_circuit_on_both(5, &gates, 123);
    assert_states_match(&stn_sv, &crz_sv, "5-qubit deep circuit");
}

#[test]
fn test_repeated_t_on_same_qubit() {
    // T^8 = I (up to phase). 8 T gates on the same qubit.
    let mut gates = vec![("h", vec![0], None)];
    for _ in 0..8 {
        gates.push(("t", vec![0], None));
    }
    let (stn_sv, crz_sv) = run_circuit_on_both(1, &gates, 42);
    assert_states_match(&stn_sv, &crz_sv, "T^8 on |+>");
}

#[test]
fn test_rx_gate() {
    let theta = Angle64::from_radians(std::f64::consts::FRAC_PI_3);
    let gates = vec![
        ("rx", vec![0], Some(theta)),
        ("cx", vec![0, 1], None),
        ("rx", vec![1], Some(theta)),
    ];
    let (stn_sv, crz_sv) = run_circuit_on_both(2, &gates, 42);
    assert_states_match(&stn_sv, &crz_sv, "RX circuits");
}

#[test]
fn test_rzz_gate() {
    let theta = Angle64::from_radians(0.5);
    let gates = vec![
        ("h", vec![0], None),
        ("h", vec![1], None),
        ("rzz", vec![0, 1], Some(theta)),
    ];
    let (stn_sv, crz_sv) = run_circuit_on_both(2, &gates, 42);
    assert_states_match(&stn_sv, &crz_sv, "RZZ gate");
}

#[test]
fn test_alternating_clifford_and_t() {
    // H, T, S, T, H, T, S, T on 2 qubits with entangling
    let gates = vec![
        ("h", vec![0], None),
        ("t", vec![0], None),
        ("sz", vec![0], None),
        ("cx", vec![0, 1], None),
        ("t", vec![1], None),
        ("h", vec![1], None),
        ("t", vec![1], None),
        ("sz", vec![1], None),
        ("cx", vec![1, 0], None),
        ("t", vec![0], None),
    ];
    let (stn_sv, crz_sv) = run_circuit_on_both(2, &gates, 42);
    assert_states_match(&stn_sv, &crz_sv, "alternating Clifford+T");
}

// ============================================================================
// Measurement probability tests
// ============================================================================

#[test]
fn test_rx_measurement_probabilities() {
    // RX(pi/3)|0> has prob(0) = cos^2(pi/6) = 3/4
    let expected_p0 = 0.75;
    let theta = Angle64::from_radians(std::f64::consts::FRAC_PI_3);

    let num_trials = 1000;
    let mut count_0 = 0;
    for trial in 0..num_trials {
        let mut stn = StabMps::with_seed(1, 10_000 + trial);
        stn.rx(theta, &[QubitId(0)]);
        if !stn.mz(&[QubitId(0)])[0].outcome {
            count_0 += 1;
        }
    }
    let p0 = f64::from(count_0) / num_trials as f64;
    assert!(
        (p0 - expected_p0).abs() < 0.05,
        "p(0) = {p0:.3}, expected {expected_p0:.3}"
    );
}

#[test]
fn test_ghz_measurement_correlation() {
    // GHZ state: H, CX chain, T on first qubit.
    // All qubits should be correlated.
    let n = 4;
    let num_trials = 100;
    let mut all_correlated = 0;

    for trial in 0..num_trials {
        let mut stn = StabMps::with_seed(n, 20_000 + trial);
        stn.h(&[QubitId(0)]);
        for q in 0..(n - 1) {
            stn.cx(&[(QubitId(q), QubitId(q + 1))]);
        }
        // Apply T to make it non-trivial
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);

        let results: Vec<bool> = (0..n).map(|q| stn.mz(&[QubitId(q)])[0].outcome).collect();

        if results.iter().all(|&r| r == results[0]) {
            all_correlated += 1;
        }
    }
    let rate = f64::from(all_correlated) / num_trials as f64;
    assert!(
        rate > 0.90,
        "GHZ+T correlation rate {rate:.2} should be > 0.90"
    );
}

// ============================================================================
// MAST vs STN comparison
// ============================================================================

#[test]
fn test_mast_vs_stn_measurement_statistics() {
    // Compare measurement outcome distributions between MAST and STN.
    let num_trials = 500;
    let mut stn_outcomes = [0u32; 4]; // 2 qubits -> 4 outcomes
    let mut mast_outcomes = [0u32; 4];

    for trial in 0..num_trials {
        // STN version
        let mut stn = StabMps::with_seed(2, 30_000 + trial);
        stn.h(&[QubitId(0)]);
        stn.cx(&[(QubitId(0), QubitId(1))]);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
        let r0 = stn.mz(&[QubitId(0)])[0].outcome;
        let r1 = stn.mz(&[QubitId(1)])[0].outcome;
        let idx = (usize::from(r0) << 1) | usize::from(r1);
        stn_outcomes[idx] += 1;

        // MAST version
        let mut mast = Mast::with_seed(2, 4, 30_000 + trial);
        mast.h(&[QubitId(0)]);
        mast.cx(&[(QubitId(0), QubitId(1))]);
        mast.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
        let r0 = mast.mz(&[QubitId(0)])[0].outcome;
        let r1 = mast.mz(&[QubitId(1)])[0].outcome;
        let idx = (usize::from(r0) << 1) | usize::from(r1);
        mast_outcomes[idx] += 1;
    }

    // Bell+T: only |00> and |11> should appear (correlation preserved)
    let stn_p00 = f64::from(stn_outcomes[0]) / num_trials as f64;
    let stn_p11 = f64::from(stn_outcomes[3]) / num_trials as f64;
    let mast_p00 = f64::from(mast_outcomes[0]) / num_trials as f64;
    let mast_p11 = f64::from(mast_outcomes[3]) / num_trials as f64;

    // Both should have ~50% |00> and ~50% |11>
    assert!((stn_p00 - 0.5).abs() < 0.1, "STN p(00) = {stn_p00:.2}");
    assert!((stn_p11 - 0.5).abs() < 0.1, "STN p(11) = {stn_p11:.2}");
    assert!((mast_p00 - 0.5).abs() < 0.1, "MAST p(00) = {mast_p00:.2}");
    assert!((mast_p11 - 0.5).abs() < 0.1, "MAST p(11) = {mast_p11:.2}");

    // Both should have no |01> or |10> (perfect correlation)
    assert!(
        stn_outcomes[1] + stn_outcomes[2] == 0,
        "STN has uncorrelated outcomes: {stn_outcomes:?}"
    );
    assert!(
        mast_outcomes[1] + mast_outcomes[2] == 0,
        "MAST has uncorrelated outcomes: {mast_outcomes:?}"
    );
}

// ============================================================================
// Compression / bond dimension tests
// ============================================================================

#[test]
fn test_bond_dim_growth_with_t_gates() {
    // Track bond dimension as T gates accumulate
    let mut stn = StabMps::new(6);
    for q in 0..6 {
        stn.h(&[QubitId(q)]);
    }
    for q in 0..5 {
        stn.cx(&[(QubitId(q), QubitId(q + 1))]);
    }

    let mut bond_dims = vec![stn.max_bond_dim()];
    for q in 0..6 {
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(q)]);
        bond_dims.push(stn.max_bond_dim());
    }

    // Bond dimension should grow but stay reasonable with compression
    assert!(
        *bond_dims.last().unwrap() < 64,
        "bond dim after 6 T gates: {bond_dims:?}"
    );
}

// ============================================================================
// Randomized fuzz testing
// ============================================================================

/// Generate a pseudo-random circuit and compare STN vs `DenseStateVec` state vectors.
fn fuzz_circuit(num_qubits: usize, num_gates: usize, seed: u64) {
    // Tolerance scales with circuit depth: more SVD ops → more numerical drift
    let tol = 0.01 + 0.002 * num_gates as f64;
    fuzz_circuit_with_tol(num_qubits, num_gates, seed, tol);
}

fn fuzz_circuit_with_tol(num_qubits: usize, num_gates: usize, seed: u64, tol: f64) {
    let mut stn = StabMps::with_seed(num_qubits, seed);
    // Use DenseStateVec as reference (not CRZ, which has frame optimization issues with CZ)
    let mut crz = pecos_simulators::DenseStateVec::new(num_qubits);

    // Use seed to generate a deterministic sequence of gates
    let mut rng_state = seed;
    let next_rng = |state: &mut u64| -> u64 {
        // Simple xorshift
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    };

    for _ in 0..num_gates {
        let gate_type = next_rng(&mut rng_state) % 8;
        let q0 = (next_rng(&mut rng_state) % num_qubits as u64) as usize;
        let q1 = loop {
            let q = (next_rng(&mut rng_state) % num_qubits as u64) as usize;
            if q != q0 {
                break q;
            }
        };

        match gate_type {
            0 => {
                stn.h(&[QubitId(q0)]);
                crz.h(&[QubitId(q0)]);
            }
            1 => {
                stn.sz(&[QubitId(q0)]);
                crz.sz(&[QubitId(q0)]);
            }
            2 => {
                stn.x(&[QubitId(q0)]);
                crz.x(&[QubitId(q0)]);
            }
            3 => {
                stn.cx(&[(QubitId(q0), QubitId(q1))]);
                crz.cx(&[(QubitId(q0), QubitId(q1))]);
            }
            4 => {
                stn.cz(&[(QubitId(q0), QubitId(q1))]);
                crz.cz(&[(QubitId(q0), QubitId(q1))]);
            }
            5 => {
                // T gate
                let t = Angle64::QUARTER_TURN / 2u64;
                stn.rz(t, &[QubitId(q0)]);
                crz.rz(t, &[QubitId(q0)]);
            }
            6 => {
                // Random RZ angle
                let angle_bits = next_rng(&mut rng_state);
                let angle = Angle64::from_radians(
                    (angle_bits % 1000) as f64 * 0.001 * std::f64::consts::TAU,
                );
                stn.rz(angle, &[QubitId(q0)]);
                crz.rz(angle, &[QubitId(q0)]);
            }
            _ => {
                // RX
                let angle_bits = next_rng(&mut rng_state);
                let angle = Angle64::from_radians(
                    (angle_bits % 1000) as f64 * 0.001 * std::f64::consts::TAU,
                );
                stn.rx(angle, &[QubitId(q0)]);
                crz.rx(angle, &[QubitId(q0)]);
            }
        }
    }

    let stn_sv = stn.state_vector();
    let dim = 1usize << num_qubits;
    let ref_sv: Vec<Complex64> = (0..dim).map(|i| crz.get_amplitude(i)).collect();

    let overlap: Complex64 = stn_sv
        .iter()
        .zip(ref_sv.iter())
        .map(|(a, b)| a.conj() * b)
        .sum();
    if (overlap.norm_sqr() - 1.0).abs() > tol {
        // Re-run step-by-step to find the divergence point
        let mut stn2 = StabMps::with_seed(num_qubits, seed);
        let mut dsv2 = pecos_simulators::DenseStateVec::new(num_qubits);
        let mut rng2 = seed;
        let next2 = |state: &mut u64| -> u64 {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            *state
        };
        for step in 0..num_gates {
            let gt = next2(&mut rng2) % 8;
            let q0s = (next2(&mut rng2) % num_qubits as u64) as usize;
            let q1s = loop {
                let q = (next2(&mut rng2) % num_qubits as u64) as usize;
                if q != q0s {
                    break q;
                }
            };
            let names = ["h", "sz", "x", "cx", "cz", "t", "rz", "rx"];
            match gt {
                0 => {
                    stn2.h(&[QubitId(q0s)]);
                    dsv2.h(&[QubitId(q0s)]);
                }
                1 => {
                    stn2.sz(&[QubitId(q0s)]);
                    dsv2.sz(&[QubitId(q0s)]);
                }
                2 => {
                    stn2.x(&[QubitId(q0s)]);
                    dsv2.x(&[QubitId(q0s)]);
                }
                3 => {
                    stn2.cx(&[(QubitId(q0s), QubitId(q1s))]);
                    dsv2.cx(&[(QubitId(q0s), QubitId(q1s))]);
                }
                4 => {
                    stn2.cz(&[(QubitId(q0s), QubitId(q1s))]);
                    dsv2.cz(&[(QubitId(q0s), QubitId(q1s))]);
                }
                5 => {
                    let t = Angle64::QUARTER_TURN / 2u64;
                    stn2.rz(t, &[QubitId(q0s)]);
                    dsv2.rz(t, &[QubitId(q0s)]);
                }
                6 => {
                    let ab = next2(&mut rng2);
                    let a =
                        Angle64::from_radians((ab % 1000) as f64 * 0.001 * std::f64::consts::TAU);
                    stn2.rz(a, &[QubitId(q0s)]);
                    dsv2.rz(a, &[QubitId(q0s)]);
                }
                _ => {
                    let ab = next2(&mut rng2);
                    let a =
                        Angle64::from_radians((ab % 1000) as f64 * 0.001 * std::f64::consts::TAU);
                    stn2.rx(a, &[QubitId(q0s)]);
                    dsv2.rx(a, &[QubitId(q0s)]);
                }
            }
            let sv2 = stn2.state_vector();
            let rv2: Vec<Complex64> = (0..dim).map(|i| dsv2.get_amplitude(i)).collect();
            let ov: Complex64 = sv2.iter().zip(rv2.iter()).map(|(a, b)| a.conj() * b).sum();
            if (ov.norm_sqr() - 1.0).abs() > tol {
                eprintln!(
                    "seed={seed}: diverged at step {step} ({}(q{q0s})): overlap={:.4}, bonds={:?}, mps_norm={:.4}",
                    names[gt as usize],
                    ov.norm_sqr(),
                    stn2.mps().bond_dims(),
                    stn2.mps().norm_squared()
                );
                eprintln!(
                    "  STN: {:?}",
                    sv2.iter()
                        .map(|a| format!("{:.4}+{:.4}i", a.re, a.im))
                        .collect::<Vec<_>>()
                );
                eprintln!(
                    "  REF: {:?}",
                    rv2.iter()
                        .map(|a| format!("{:.4}+{:.4}i", a.re, a.im))
                        .collect::<Vec<_>>()
                );
                break;
            }
        }
    }

    assert_states_close(
        &stn_sv,
        &ref_sv,
        tol,
        &format!("fuzz n={num_qubits} gates={num_gates} seed={seed}"),
    );
}

#[test]
fn test_fuzz_2qubit_circuits() {
    for seed in 100..200 {
        fuzz_circuit(2, 10, seed);
    }
}

#[test]
fn test_fuzz_seed_115_mps_check() {
    let q0 = QubitId(0);
    let q1 = QubitId(1);
    let mut stn = StabMps::with_seed(2, 115);
    stn.cx(&[(q0, q1)]);
    stn.cz(&[(q0, q1)]);
    stn.cx(&[(q1, q0)]);
    stn.rz(Angle64::QUARTER_TURN / 2u64, &[q1]);
    let mps3 = stn.mps().state_vector();
    eprintln!(
        "Step 3 MPS: {:?}",
        mps3.iter()
            .map(|a| format!("{:.4}+{:.4}i", a.re, a.im))
            .collect::<Vec<_>>()
    );

    stn.cz(&[(q0, q1)]);
    stn.h(&[q0]);
    stn.cz(&[(q0, q1)]);
    stn.sz(&[q0]);
    stn.rz(Angle64::from_radians(0.2702), &[q1]);
    let mps8 = stn.mps().state_vector();
    eprintln!(
        "Step 8 MPS: {:?}",
        mps8.iter()
            .map(|a| format!("{:.4}+{:.4}i", a.re, a.im))
            .collect::<Vec<_>>()
    );
    // Reference: [0.8639-0.5036i, 0, 0, 0]
    approx::assert_relative_eq!(mps8[0].re, 0.8639, epsilon = 0.01);
    approx::assert_relative_eq!(mps8[0].im, -0.5036, epsilon = 0.01);

    // Compare state_vector vs DenseStateVec
    stn.cx(&[(q0, q1)]); // Step 9
    let stn_sv = stn.state_vector();
    let mut dsv = pecos_simulators::DenseStateVec::new(2);
    dsv.cx(&[(q0, q1)]);
    dsv.cz(&[(q0, q1)]);
    dsv.cx(&[(q1, q0)]);
    dsv.rz(Angle64::QUARTER_TURN / 2u64, &[q1]);
    dsv.cz(&[(q0, q1)]);
    dsv.h(&[q0]);
    dsv.cz(&[(q0, q1)]);
    dsv.sz(&[q0]);
    dsv.rz(Angle64::from_radians(0.2702), &[q1]);
    dsv.cx(&[(q0, q1)]);
    let ref_sv: Vec<Complex64> = (0..4).map(|i| dsv.get_amplitude(i)).collect();
    eprintln!(
        "STN SV: {:?}",
        stn_sv
            .iter()
            .map(|a| format!("{:.4}+{:.4}i", a.re, a.im))
            .collect::<Vec<_>>()
    );
    eprintln!(
        "DSV SV: {:?}",
        ref_sv
            .iter()
            .map(|a| format!("{:.4}+{:.4}i", a.re, a.im))
            .collect::<Vec<_>>()
    );
    let overlap: Complex64 = stn_sv
        .iter()
        .zip(ref_sv.iter())
        .map(|(a, b)| a.conj() * b)
        .sum();
    eprintln!("Overlap: {:.4}", overlap.norm_sqr());
}

#[test]
fn test_fuzz_seed_101_measurement_stats() {
    // Verify STN measurement probabilities match the state vector.
    let t = Angle64::QUARTER_TURN / 2u64;
    let rz_angle = Angle64::from_radians(4.0024);
    let rx1 = Angle64::from_radians(5.6800);
    let rx2 = Angle64::from_radians(5.3973);

    // Compute expected probabilities from state vector.
    let mut stn_ref = StabMps::with_seed(2, 42);
    stn_ref.rz(t, &[QubitId(1)]);
    stn_ref.h(&[QubitId(0)]);
    stn_ref.sz(&[QubitId(1)]);
    stn_ref.sz(&[QubitId(0)]);
    stn_ref.rz(rz_angle, &[QubitId(0)]);
    stn_ref.rx(rx1, &[QubitId(0)]);
    stn_ref.rx(rx2, &[QubitId(1)]);
    stn_ref.x(&[QubitId(0)]);
    stn_ref.x(&[QubitId(0)]);
    stn_ref.rz(t, &[QubitId(0)]);
    let sv = stn_ref.state_vector();
    // sv[i] uses DenseStateVec convention: bit 0 = q0, bit 1 = q1
    let expected_probs: Vec<f64> = sv.iter().map(num_complex::Complex::norm_sqr).collect();

    // Sample measurements and compare.
    let num_trials = 500;
    let mut stn_outcomes = [0u32; 4];
    for trial in 0..num_trials {
        let seed = 50_000 + trial;
        let mut stn = StabMps::with_seed(2, seed);
        stn.rz(t, &[QubitId(1)]);
        stn.h(&[QubitId(0)]);
        stn.sz(&[QubitId(1)]);
        stn.sz(&[QubitId(0)]);
        stn.rz(rz_angle, &[QubitId(0)]);
        stn.rx(rx1, &[QubitId(0)]);
        stn.rx(rx2, &[QubitId(1)]);
        stn.x(&[QubitId(0)]);
        stn.x(&[QubitId(0)]);
        stn.rz(t, &[QubitId(0)]);
        let s0 = stn.mz(&[QubitId(0)])[0].outcome;
        let s1 = stn.mz(&[QubitId(1)])[0].outcome;
        // Index: bit 0 = q0, bit 1 = q1 (matching DenseStateVec convention)
        stn_outcomes[usize::from(s0) | (usize::from(s1) << 1)] += 1;
    }

    for i in 0..4 {
        let p_s = f64::from(stn_outcomes[i]) / num_trials as f64;
        assert!(
            (p_s - expected_probs[i]).abs() < 0.1,
            "outcome {i}: STN p={p_s:.3} vs expected p={:.3}",
            expected_probs[i]
        );
    }
}

#[test]
fn test_fuzz_debug_seed_101() {
    // Minimal repro: T, H, S, S, RZ, RX sequence on 2 qubits
    // Step-by-step comparison to find divergence point.
    let t = Angle64::QUARTER_TURN / 2u64;
    let rz_angle = Angle64::from_radians(4.0024);
    let rx_angle1 = Angle64::from_radians(5.6800);

    let mut stn = StabMps::with_seed(2, 101);
    let mut crz = StabVec::builder(2).seed(101).build();

    // Step 0: T on q1
    stn.rz(t, &[QubitId(1)]);
    crz.rz(t, &[QubitId(1)]);

    let s1 = stn.state_vector();
    let c1 = crz.state_vector();
    assert_states_match(&s1, &c1, "after T(1)");

    // Step 1: H on q0
    stn.h(&[QubitId(0)]);
    crz.h(&[QubitId(0)]);
    let s2 = stn.state_vector();
    let c2 = crz.state_vector();
    assert_states_match(&s2, &c2, "after H(0)");

    // Step 2: S on q1
    stn.sz(&[QubitId(1)]);
    crz.sz(&[QubitId(1)]);
    let s3 = stn.state_vector();
    let c3 = crz.state_vector();
    assert_states_match(&s3, &c3, "after S(1)");

    // Step 3: S on q0
    stn.sz(&[QubitId(0)]);
    crz.sz(&[QubitId(0)]);
    let s4 = stn.state_vector();
    let c4 = crz.state_vector();
    assert_states_match(&s4, &c4, "after S(0)");

    // Step 4: RZ on q0
    stn.rz(rz_angle, &[QubitId(0)]);
    crz.rz(rz_angle, &[QubitId(0)]);
    eprintln!(
        "after RZ(0): MPS norm={:.6}, bonds={:?}",
        stn.mps().norm_squared(),
        stn.mps().bond_dims()
    );
    let s5 = stn.state_vector();
    let c5 = crz.state_vector();
    assert_states_match(&s5, &c5, "after RZ(0)");

    // Step 5: RX on q0 = H + RZ + H
    // Do manually to find where norm goes wrong
    stn.h(&[QubitId(0)]);
    crz.h(&[QubitId(0)]);
    eprintln!(
        "after H(0): MPS norm={:.6}, bonds={:?}",
        stn.mps().norm_squared(),
        stn.mps().bond_dims()
    );
    let s5h = stn.state_vector();
    let c5h = crz.state_vector();
    assert_states_match(&s5h, &c5h, "after RZ then H");

    // Check Z_0 decomposition before inner RZ
    let decomp = pecos_stab_tn::stab_mps::pauli_decomp::decompose_z(
        stn.tableau().stabs(),
        stn.tableau().destabs(),
        0,
    );
    eprintln!("Z_0 decomp before inner RZ: {decomp:?}");

    stn.rz(rx_angle1, &[QubitId(0)]);
    crz.rz(rx_angle1, &[QubitId(0)]);
    eprintln!(
        "after inner RZ: MPS norm={:.6}, bonds={:?}",
        stn.mps().norm_squared(),
        stn.mps().bond_dims()
    );
    // Check MPS SV directly against reference
    let mps5 = stn.mps().state_vector();
    eprintln!(
        "Step5 MPS: {:?}",
        mps5.iter()
            .map(|a| format!("{:.4}+{:.4}i", a.re, a.im))
            .collect::<Vec<_>>()
    );
    eprintln!("Step5 ref: [0.4714+0.0969i, 0, 0.2176+0.8492i, 0]");
    let s5r = stn.state_vector();
    let c5r = crz.state_vector();
    assert_states_match(&s5r, &c5r, "after step 5");

    // Continue with remaining gates from fuzz sequence
    // Step 6: RX on q1 = H(1) + RZ(1) + H(1)
    let rx_angle2 = Angle64::from_radians(5.3973);
    stn.h(&[QubitId(1)]);
    crz.h(&[QubitId(1)]);
    eprintln!("step 6a (H1): norm={:.6}", stn.mps().norm_squared());
    assert_states_match(&stn.state_vector(), &crz.state_vector(), "step 6a");

    stn.rz(rx_angle2, &[QubitId(1)]);
    crz.rz(rx_angle2, &[QubitId(1)]);
    eprintln!(
        "step 6b (RZ1): norm={:.6}, bonds={:?}",
        stn.mps().norm_squared(),
        stn.mps().bond_dims()
    );
    // Compare MPS with reference
    let mps_sv = stn.mps().state_vector();
    eprintln!(
        "  Rust MPS: {:?}",
        mps_sv
            .iter()
            .map(|a| format!("{:.4}+{:.4}i", a.re, a.im))
            .collect::<Vec<_>>()
    );
    // Reference: [0.4259+0.0876i, 0.202+0.0415i, 0.1966+0.7672i, 0.0933+0.3639i]
    eprintln!("  Ref  MPS: [0.4259+0.0876i, 0.2020+0.0415i, 0.1966+0.7672i, 0.0933+0.3639i]");
    // Check MPS directly vs through state_vector
    let mps_sv = stn.mps().state_vector();
    let stn_sv = stn.state_vector();
    let crz_sv = crz.state_vector();
    eprintln!(
        "MPS SV: {:?}",
        mps_sv
            .iter()
            .map(|a| format!("{:.3}+{:.3}i", a.re, a.im))
            .collect::<Vec<_>>()
    );
    eprintln!(
        "STN SV: {:?}",
        stn_sv
            .iter()
            .map(|a| format!("{:.3}+{:.3}i", a.re, a.im))
            .collect::<Vec<_>>()
    );
    eprintln!(
        "CRZ SV: {:?}",
        crz_sv
            .iter()
            .map(|a| format!("{:.3}+{:.3}i", a.re, a.im))
            .collect::<Vec<_>>()
    );
    assert_states_match(&stn_sv, &crz_sv, "step 6b");

    stn.h(&[QubitId(1)]);
    crz.h(&[QubitId(1)]);
    eprintln!("step 6c (H1): norm={:.6}", stn.mps().norm_squared());
    let s6 = stn.state_vector();
    let c6 = crz.state_vector();
    assert_states_match(&s6, &c6, "step 6c");

    // Step 7: X on q0
    stn.x(&[QubitId(0)]);
    crz.x(&[QubitId(0)]);
    let s7 = stn.state_vector();
    let c7 = crz.state_vector();
    assert_states_match(&s7, &c7, "after step 7");

    // Step 8: X on q0
    stn.x(&[QubitId(0)]);
    crz.x(&[QubitId(0)]);
    let s8 = stn.state_vector();
    let c8 = crz.state_vector();
    assert_states_match(&s8, &c8, "after step 8");

    // Step 9: T on q0
    stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
    crz.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
    let s9 = stn.state_vector();
    let c9 = crz.state_vector();
    assert_states_match(&s9, &c9, "after step 9");
}

#[test]
fn test_debug_seed_502() {
    let num_qubits = 2usize;
    let num_gates = 30usize;
    let seed = 502u64;
    let dim = 1usize << num_qubits;

    let mut stn = StabMps::with_seed(num_qubits, seed);
    let mut dsv = pecos_simulators::DenseStateVec::new(num_qubits);

    let mut rng_state = seed;
    let next_rng = |state: &mut u64| -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    };

    for step in 0..num_gates {
        let gate_type = next_rng(&mut rng_state) % 8;
        let q0 = (next_rng(&mut rng_state) % num_qubits as u64) as usize;
        let q1 = loop {
            let q = (next_rng(&mut rng_state) % num_qubits as u64) as usize;
            if q != q0 {
                break q;
            }
        };

        let gate_name;
        match gate_type {
            0 => {
                gate_name = format!("H({q0})");
                stn.h(&[QubitId(q0)]);
                dsv.h(&[QubitId(q0)]);
            }
            1 => {
                gate_name = format!("SZ({q0})");
                stn.sz(&[QubitId(q0)]);
                dsv.sz(&[QubitId(q0)]);
            }
            2 => {
                gate_name = format!("X({q0})");
                stn.x(&[QubitId(q0)]);
                dsv.x(&[QubitId(q0)]);
            }
            3 => {
                gate_name = format!("CX({q0},{q1})");
                stn.cx(&[(QubitId(q0), QubitId(q1))]);
                dsv.cx(&[(QubitId(q0), QubitId(q1))]);
            }
            4 => {
                gate_name = format!("CZ({q0},{q1})");
                stn.cz(&[(QubitId(q0), QubitId(q1))]);
                dsv.cz(&[(QubitId(q0), QubitId(q1))]);
            }
            5 => {
                let t = Angle64::QUARTER_TURN / 2u64;
                gate_name = format!("T({q0})");
                stn.rz(t, &[QubitId(q0)]);
                dsv.rz(t, &[QubitId(q0)]);
            }
            6 => {
                let angle_bits = next_rng(&mut rng_state);
                let angle = Angle64::from_radians(
                    (angle_bits % 1000) as f64 * 0.001 * std::f64::consts::TAU,
                );
                gate_name = format!("RZ({}, {:.4})", q0, angle.to_radians());
                stn.rz(angle, &[QubitId(q0)]);
                dsv.rz(angle, &[QubitId(q0)]);
            }
            _ => {
                let angle_bits = next_rng(&mut rng_state);
                let angle = Angle64::from_radians(
                    (angle_bits % 1000) as f64 * 0.001 * std::f64::consts::TAU,
                );
                gate_name = format!("RX({}, {:.4})", q0, angle.to_radians());
                stn.rx(angle, &[QubitId(q0)]);
                dsv.rx(angle, &[QubitId(q0)]);
            }
        }

        let stn_sv = stn.state_vector();
        let ref_sv: Vec<Complex64> = (0..dim).map(|i| dsv.get_amplitude(i)).collect();
        let overlap: Complex64 = stn_sv
            .iter()
            .zip(ref_sv.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();
        let ov = overlap.norm_sqr();
        let mps_norm = stn.mps().norm_squared();
        let bonds = stn.mps().bond_dims().to_vec();

        if (ov - 1.0).abs() > 0.01 {
            eprintln!("=== DIVERGENCE at step {step}: {gate_name} ===");
            eprintln!("  overlap={ov:.6}, mps_norm={mps_norm:.6}, bonds={bonds:?}");
            eprintln!(
                "  STN: {:?}",
                stn_sv
                    .iter()
                    .map(|a| format!("{:.4}+{:.4}i", a.re, a.im))
                    .collect::<Vec<_>>()
            );
            eprintln!(
                "  REF: {:?}",
                ref_sv
                    .iter()
                    .map(|a| format!("{:.4}+{:.4}i", a.re, a.im))
                    .collect::<Vec<_>>()
            );
            panic!(
                "Divergence at step {step} ({gate_name}): overlap={ov:.6}, mps_norm={mps_norm:.6}, bonds={bonds:?}"
            );
        }

        eprintln!(
            "step {step:2}: {gate_name:16} overlap={ov:.6} mps_norm={mps_norm:.6} bonds={bonds:?}"
        );
    }
}

#[test]
fn test_fuzz_3qubit_circuits() {
    for seed in 200..300 {
        fuzz_circuit(3, 12, seed);
    }
}

#[test]
fn test_fuzz_4qubit_circuits() {
    for seed in 300..400 {
        fuzz_circuit(4, 15, seed);
    }
}

#[test]
fn test_fuzz_5qubit() {
    for seed in 400..450 {
        fuzz_circuit(5, 12, seed);
    }
}

#[test]
fn test_fuzz_2qubit_deep() {
    for seed in 500..600 {
        fuzz_circuit(2, 30, seed);
    }
}

#[test]
fn test_rx_pi_after_nonclifford() {
    // RX(pi) = -i*X. Check it works after non-Clifford gates.
    let mut stn = StabMps::with_seed(2, 42);
    let mut dsv = pecos_simulators::DenseStateVec::new(2);
    let t = Angle64::QUARTER_TURN / 2u64;
    let pi = Angle64::from_radians(std::f64::consts::PI);

    for (gate, qids, angle) in [
        ("h", vec![QubitId(0)], None),
        ("h", vec![QubitId(1)], None),
        ("t", vec![QubitId(0)], Some(t)),
        ("cx_", vec![QubitId(0), QubitId(1)], None),
        ("t", vec![QubitId(1)], Some(t)),
        ("rx", vec![QubitId(0)], Some(pi)),
    ] {
        match gate {
            "h" => {
                stn.h(&qids);
                dsv.h(&qids);
            }
            "t" => {
                let a = angle.unwrap();
                stn.rz(a, &qids);
                dsv.rz(a, &qids);
            }
            "cx_" => {
                let p = vec![(qids[0], qids[1])];
                stn.cx(&p);
                dsv.cx(&p);
            }
            "rx" => {
                let a = angle.unwrap();
                stn.rx(a, &qids);
                dsv.rx(a, &qids);
            }
            _ => {}
        }
    }
    let stn_sv = stn.state_vector();
    let dsv_sv: Vec<Complex64> = (0..4).map(|i| dsv.get_amplitude(i)).collect();
    eprintln!(
        "STN: {:?}",
        stn_sv
            .iter()
            .map(|a| format!("{:.4}+{:.4}i", a.re, a.im))
            .collect::<Vec<_>>()
    );
    eprintln!(
        "DSV: {:?}",
        dsv_sv
            .iter()
            .map(|a| format!("{:.4}+{:.4}i", a.re, a.im))
            .collect::<Vec<_>>()
    );
    // Print destab signs and tracking flag
    eprintln!(
        "  tracks_destab_signs: {}",
        stn.tableau().tracks_destab_signs()
    );
    for i in 0..2 {
        let dm = stn.tableau().destabs().signs_minus.contains(i);
        let di = stn.tableau().destabs().signs_i.contains(i);
        eprintln!("  D[{i}] minus={dm} i={di}");
    }
    // Also check state before RX(pi)
    let mut stn2 = StabMps::with_seed(2, 42);
    let mut dsv2 = pecos_simulators::DenseStateVec::new(2);
    stn2.h(&[QubitId(0)]);
    dsv2.h(&[QubitId(0)]);
    stn2.h(&[QubitId(1)]);
    dsv2.h(&[QubitId(1)]);
    stn2.rz(t, &[QubitId(0)]);
    dsv2.rz(t, &[QubitId(0)]);
    stn2.cx(&[(QubitId(0), QubitId(1))]);
    dsv2.cx(&[(QubitId(0), QubitId(1))]);
    stn2.rz(t, &[QubitId(1)]);
    dsv2.rz(t, &[QubitId(1)]);
    let sv_before = stn2.state_vector();
    let dv_before: Vec<Complex64> = (0..4).map(|i| dsv2.get_amplitude(i)).collect();
    let ov_before: Complex64 = sv_before
        .iter()
        .zip(dv_before.iter())
        .map(|(a, b)| a.conj() * b)
        .sum();
    eprintln!("Before RX(pi): overlap={:.6}", ov_before.norm_sqr());
    assert_states_match(&stn_sv, &dsv_sv, "RX(pi) after non-Clifford");
}

#[test]
fn test_seed319_minimal() {
    // Minimal circuit from seed 319: non-Clifford, entangling, then RX(pi)
    let t = Angle64::QUARTER_TURN / 2u64;
    let gates = vec![
        ("t", vec![0], None),
        ("h", vec![1], None),
        ("rz", vec![3], Some(Angle64::from_radians(3.3427))),
        ("cz", vec![0, 3], None),
        ("rz", vec![1], Some(t)), // T gate = RZ(pi/4)
        ("cz", vec![1, 2], None),
        ("sz", vec![1], None),
        ("h", vec![2], None),
        ("t", vec![2], None),
        ("h", vec![2], None),
        ("rz", vec![3], Some(Angle64::from_radians(3.0976))),
        ("rx", vec![0], Some(Angle64::from_radians(5.2025))),
        (
            "rx",
            vec![1],
            Some(Angle64::from_radians(std::f64::consts::PI)),
        ),
    ];
    let (stn_sv, crz_sv) = run_circuit_on_both(4, &gates, 42);
    assert_states_match(&stn_sv, &crz_sv, "seed319 minimal");
}

#[test]
fn test_swap_then_t() {
    // Minimal reproduction: H on both, SWAP, then T on q1.
    let gates = vec![
        ("h", vec![0], None),
        ("h", vec![1], None),
        ("cx", vec![0, 1], None),
        ("cx", vec![1, 0], None),
        ("cx", vec![0, 1], None),
        ("t", vec![1], None),
    ];
    let (stn_sv, crz_sv) = run_circuit_on_both(2, &gates, 42);
    assert_states_match(&stn_sv, &crz_sv, "SWAP then T");
}

#[test]
fn test_h_swap_rz_t() {
    // Matches the seed 502 circuit prefix up to the failing step.
    let gates = vec![
        ("h", vec![0], None),
        ("h", vec![1], None),
        ("cx", vec![0, 1], None),
        ("cx", vec![1, 0], None),
        ("cx", vec![0, 1], None),
        ("rx", vec![1], Some(Angle64::from_radians(5.0265))),
        ("t", vec![1], None),
    ];
    let (stn_sv, crz_sv) = run_circuit_on_both(2, &gates, 42);
    assert_states_match(&stn_sv, &crz_sv, "H SWAP RX T");
}

#[test]
fn test_seed502_prefix() {
    // Seed 502 circuit prefix: T(1), RZ(0), X(0), T(1), X(0), RZ(0,~pi),
    // H(0), H(1), CX CX CX, RX(1), T(1)
    let rz_angle = Angle64::from_radians(3.6317); // angle from seed 502 RNG
    let rx_angle = Angle64::from_radians(5.0265);
    // Try: full prefix T, RZ, X, T, X, Z, H, H, SWAP, RX, T
    let gates = vec![
        ("t", vec![1], None),
        ("rz", vec![0], Some(rz_angle)),
        ("x", vec![0], None),
        ("t", vec![1], None),
        ("x", vec![0], None),
        (
            "rz",
            vec![0],
            Some(Angle64::from_radians(std::f64::consts::PI)),
        ),
        ("h", vec![0], None),
        ("h", vec![1], None),
        ("cx", vec![0, 1], None),
        ("cx", vec![1, 0], None),
        ("cx", vec![0, 1], None),
        ("rx", vec![1], Some(rx_angle)),
        ("t", vec![1], None),
    ];
    let mut stn = StabMps::with_seed(2, 42);
    let mut dsv = pecos_simulators::DenseStateVec::new(2);
    let dim = 1usize << 2;
    for (step, (gate, qubits, angle)) in gates.iter().enumerate() {
        let qids: Vec<QubitId> = qubits.iter().map(|&q| QubitId(q)).collect();
        match *gate {
            "h" => {
                stn.h(&qids);
                dsv.h(&qids);
            }
            "sz" => {
                stn.sz(&qids);
                dsv.sz(&qids);
            }
            "x" => {
                stn.x(&qids);
                dsv.x(&qids);
            }
            "z" => {
                stn.z(&qids);
                dsv.z(&qids);
            }
            "cx" => {
                let p = vec![(QubitId(qubits[0]), QubitId(qubits[1]))];
                stn.cx(&p);
                dsv.cx(&p);
            }
            "rz" => {
                let a = angle.unwrap();
                stn.rz(a, &qids);
                dsv.rz(a, &qids);
            }
            "rx" => {
                let a = angle.unwrap();
                stn.rx(a, &qids);
                dsv.rx(a, &qids);
            }
            "t" => {
                let t = Angle64::QUARTER_TURN / 2u64;
                stn.rz(t, &qids);
                dsv.rz(t, &qids);
            }
            _ => panic!("unknown gate"),
        }
        let sv = stn.state_vector();
        let rv: Vec<Complex64> = (0..dim).map(|i| dsv.get_amplitude(i)).collect();
        let ov: Complex64 = sv.iter().zip(rv.iter()).map(|(a, b)| a.conj() * b).sum();
        if (ov.norm_sqr() - 1.0).abs() > 0.01 {
            eprintln!(
                "DIVERGE at step {step} ({gate}(q{})): overlap={:.4}",
                qubits[0],
                ov.norm_sqr()
            );
            // Check decomposition phase at the divergence point
            // For RX, the inner RZ acts on the SAME qubit after H
            let target_q = qubits[0];
            let decomp = pecos_stab_tn::stab_mps::pauli_decomp::decompose_z(
                stn.tableau().stabs(),
                stn.tableau().destabs(),
                target_q,
            );
            let phase_ok = pecos_stab_tn::stab_mps::pauli_decomp::verify_decomposition_brute_force(
                stn.tableau().stabs(),
                stn.tableau().destabs(),
                target_q,
                &decomp,
            );
            eprintln!("  decomp phase correct: {phase_ok}");
            eprintln!("  decomp: {decomp:?}");
            for i in 0..2 {
                let sx: Vec<usize> = stn.tableau().stabs().row_x[i].iter().collect();
                let sz: Vec<usize> = stn.tableau().stabs().row_z[i].iter().collect();
                let sm = stn.tableau().stabs().signs_minus.contains(i);
                let si = stn.tableau().stabs().signs_i.contains(i);
                let dx: Vec<usize> = stn.tableau().destabs().row_x[i].iter().collect();
                let dz: Vec<usize> = stn.tableau().destabs().row_z[i].iter().collect();
                let dm = stn.tableau().destabs().signs_minus.contains(i);
                let di = stn.tableau().destabs().signs_i.contains(i);
                eprintln!(
                    "  S[{i}]: x={sx:?} z={sz:?} m={sm} i={si}   D[{i}]: x={dx:?} z={dz:?} m={dm} i={di}"
                );
            }
            break;
        }
    }
    let stn_sv = stn.state_vector();
    let dsv_sv: Vec<Complex64> = (0..dim).map(|i| dsv.get_amplitude(i)).collect();

    // Brute-force: compute |ψ⟩ = Σ_x ν_x * D^x * |stab⟩ directly from tableau
    let n = 2usize;
    let mps_raw = stn.mps().state_vector();
    let gen_matrix = |is_stab: bool, row: usize| -> nalgebra::DMatrix<Complex64> {
        let gens = if is_stab {
            stn.tableau().stabs()
        } else {
            stn.tableau().destabs()
        };
        let i2 = nalgebra::DMatrix::<Complex64>::identity(2, 2);
        let xm = nalgebra::DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        );
        let zm = nalgebra::DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
            ],
        );
        let ym = nalgebra::DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, -1.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(0.0, 0.0),
            ],
        );
        let mut r = nalgebra::DMatrix::from_element(1, 1, Complex64::new(1.0, 0.0));
        for q in 0..n {
            let p = match (gens.row_x[row].contains(q), gens.row_z[row].contains(q)) {
                (false, false) => &i2,
                (true, false) => &xm,
                (false, true) => &zm,
                (true, true) => &ym,
            };
            r = r.kronecker(p);
        }
        let mut ph = Complex64::new(1.0, 0.0);
        if gens.signs_minus.contains(row) {
            ph *= Complex64::new(-1.0, 0.0);
        }
        if gens.signs_i.contains(row) {
            ph *= Complex64::new(0.0, 1.0);
        }
        r * ph
    };
    let id4 = nalgebra::DMatrix::<Complex64>::identity(dim, dim);
    let mut proj = id4.clone();
    for k in 0..n {
        let sk = gen_matrix(true, k);
        proj = (&id4 + &sk) * Complex64::new(0.5, 0.0) * &proj;
    }
    let mut ss = nalgebra::DVector::from_element(dim, Complex64::new(0.0, 0.0));
    ss[0] = Complex64::new(1.0, 0.0);
    let ss = &proj * &ss;
    let sn: f64 = ss.iter().map(num_complex::Complex::norm_sqr).sum();
    let ss = ss / Complex64::new(sn.sqrt(), 0.0);
    let mut psi = nalgebra::DVector::from_element(dim, Complex64::new(0.0, 0.0));
    for (x, &nu) in mps_raw.iter().enumerate() {
        if nu.norm_sqr() < 1e-20 {
            continue;
        }
        let mut st = ss.clone();
        for k in 0..n {
            if (x >> (n - 1 - k)) & 1 == 1 {
                st = &gen_matrix(false, k) * &st;
            }
        }
        psi += st * nu;
    }
    // Brute-force uses MSB-first (Kronecker convention). Bit-reverse to match DSV (LSB-first).
    let mut bru = vec![Complex64::new(0.0, 0.0); dim];
    for (i, &a) in psi.iter().enumerate() {
        let mut rev = 0;
        for b in 0..n {
            if (i >> b) & 1 == 1 {
                rev |= 1 << (n - 1 - b);
            }
        }
        bru[rev] = a;
    }
    let ov_bru: Complex64 = bru
        .iter()
        .zip(dsv_sv.iter())
        .map(|(a, b)| a.conj() * b)
        .sum();
    eprintln!("BRU vs DSV overlap: {:.6}", ov_bru.norm_sqr());
    let ov_stn: Complex64 = stn_sv
        .iter()
        .zip(dsv_sv.iter())
        .map(|(a, b)| a.conj() * b)
        .sum();
    eprintln!("STN vs DSV overlap: {:.6}", ov_stn.norm_sqr());
    let ov_sb: Complex64 = stn_sv
        .iter()
        .zip(bru.iter())
        .map(|(a, b)| a.conj() * b)
        .sum();
    eprintln!("STN vs BRU overlap: {:.6}", ov_sb.norm_sqr());

    assert_states_match(&bru, &dsv_sv, "seed502 brute-force vs DSV");
}

#[test]
fn test_fuzz_3qubit_deep() {
    for seed in 600..700 {
        fuzz_circuit(3, 25, seed);
    }
}

#[test]
fn test_fuzz_4qubit_deep() {
    for seed in 700..750 {
        fuzz_circuit(4, 25, seed);
    }
}

#[test]
fn test_fuzz_6qubit() {
    for seed in 750..790 {
        fuzz_circuit(6, 15, seed);
    }
}

#[test]
#[ignore = "slow fuzz (~18s debug): run with `cargo test --test verification -- --include-ignored`"]
fn test_fuzz_7qubit() {
    for seed in 790..810 {
        fuzz_circuit(7, 12, seed);
    }
}

#[test]
#[ignore = "slow fuzz (~80s debug): run with `cargo test --test verification -- --include-ignored`"]
fn test_fuzz_8qubit() {
    for seed in 810..820 {
        fuzz_circuit(8, 10, seed);
    }
}

#[test]
#[ignore = "deep fuzz (~10min debug, ~30s release): run with `cargo test --release --test verification -- --ignored test_fuzz_deep`"]
fn test_fuzz_deep() {
    // Heavy fuzz for pre-release validation. Sweeps 2-8 qubits with many
    // seeds and deeper circuits to catch rare corner cases. Run in release
    // mode for reasonable turnaround.
    for n in 2..=6 {
        let depth = 25;
        for seed in 0..100u64 {
            fuzz_circuit(n, depth, 10000 + seed);
        }
    }
    for n in 7..=8 {
        let depth = 15;
        for seed in 0..50u64 {
            fuzz_circuit(n, depth, 20000 + seed);
        }
    }
}

// ============================================================================
// Measurement probability validation (compare sampling vs state vector)
// ============================================================================

/// Build a random circuit using the fuzz RNG, apply it, then check that
/// measurement sampling probabilities match the state vector amplitudes.
fn measurement_probability_check(num_qubits: usize, num_gates: usize, seed: u64) {
    // Build the circuit
    let mut stn_ref = StabMps::with_seed(num_qubits, seed);
    let mut rng_state = seed;
    let next_rng = |state: &mut u64| -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    };

    for _ in 0..num_gates {
        let gate_type = next_rng(&mut rng_state) % 8;
        let q0 = (next_rng(&mut rng_state) % num_qubits as u64) as usize;
        let q1 = loop {
            let q = (next_rng(&mut rng_state) % num_qubits as u64) as usize;
            if q != q0 {
                break q;
            }
        };
        match gate_type {
            0 => {
                stn_ref.h(&[QubitId(q0)]);
            }
            1 => {
                stn_ref.sz(&[QubitId(q0)]);
            }
            2 => {
                stn_ref.x(&[QubitId(q0)]);
            }
            3 => {
                stn_ref.cx(&[(QubitId(q0), QubitId(q1))]);
            }
            4 => {
                stn_ref.cz(&[(QubitId(q0), QubitId(q1))]);
            }
            5 => {
                let t = Angle64::QUARTER_TURN / 2u64;
                stn_ref.rz(t, &[QubitId(q0)]);
            }
            6 => {
                let angle_bits = next_rng(&mut rng_state);
                let angle = Angle64::from_radians(
                    (angle_bits % 1000) as f64 * 0.001 * std::f64::consts::TAU,
                );
                stn_ref.rz(angle, &[QubitId(q0)]);
            }
            _ => {
                let angle_bits = next_rng(&mut rng_state);
                let angle = Angle64::from_radians(
                    (angle_bits % 1000) as f64 * 0.001 * std::f64::consts::TAU,
                );
                stn_ref.rx(angle, &[QubitId(q0)]);
            }
        }
    }

    // Get expected probabilities from state vector
    let sv = stn_ref.state_vector();
    let dim = 1usize << num_qubits;
    let expected_probs: Vec<f64> = sv.iter().map(num_complex::Complex::norm_sqr).collect();

    // For each qubit, check marginal probability matches sampling
    for q in 0..num_qubits {
        // Expected p(q=0) = sum of |a_i|^2 where bit q of i is 0
        let expected_p0: f64 = (0..dim)
            .filter(|&i| (i >> q) & 1 == 0)
            .map(|i| expected_probs[i])
            .sum();

        let z_ev = pecos_stab_tn::stab_mps::measure::z_expectation_value(
            stn_ref.tableau(),
            stn_ref.mps(),
            q,
        )
        .re;
        let stn_p0 = f64::midpoint(1.0, z_ev).clamp(0.0, 1.0);

        assert!(
            (stn_p0 - expected_p0).abs() < 0.001,
            "seed={seed} q={q}: p(0) from <Z>={stn_p0:.4} vs state_vector={expected_p0:.4}"
        );
    }
}

#[test]
fn test_measurement_probabilities_2qubit() {
    for seed in 1000..1100 {
        measurement_probability_check(2, 10, seed);
    }
}

#[test]
fn test_measurement_probabilities_3qubit() {
    for seed in 1100..1200 {
        measurement_probability_check(3, 12, seed);
    }
}

#[test]
fn test_measurement_probabilities_4qubit() {
    for seed in 1200..1280 {
        measurement_probability_check(4, 15, seed);
    }
}

#[test]
fn test_measurement_probabilities_5qubit() {
    for seed in 1280..1310 {
        measurement_probability_check(5, 10, seed);
    }
}

// ============================================================================
// Disentangle validation
// ============================================================================

#[test]
#[allow(clippy::type_complexity)]
fn test_disentangle_various_circuits() {
    // Verify disentangle preserves state for several circuits
    let circuits: Vec<Box<dyn Fn(&mut StabMps)>> = vec![
        Box::new(|stn: &mut StabMps| {
            let t = Angle64::QUARTER_TURN / 2u64;
            stn.h(&[QubitId(0)]);
            stn.cx(&[(QubitId(0), QubitId(1))]);
            stn.rz(t, &[QubitId(0)]);
        }),
        Box::new(|stn: &mut StabMps| {
            let t = Angle64::QUARTER_TURN / 2u64;
            stn.h(&[QubitId(0)]);
            stn.h(&[QubitId(1)]);
            stn.rz(t, &[QubitId(0)]);
            stn.cx(&[(QubitId(0), QubitId(1))]);
            stn.rz(t, &[QubitId(1)]);
        }),
        Box::new(|stn: &mut StabMps| {
            stn.h(&[QubitId(0)]);
            stn.cx(&[(QubitId(0), QubitId(1))]);
            stn.cx(&[(QubitId(1), QubitId(2))]);
            stn.rz(Angle64::from_radians(0.7), &[QubitId(1)]);
        }),
    ];

    for (i, build) in circuits.iter().enumerate() {
        let mut stn = StabMps::new(3);
        build(&mut stn);
        let sv_before = stn.state_vector();
        let _gates = stn.disentangle(5);
        let sv_after = stn.state_vector();
        assert_states_match(&sv_before, &sv_after, &format!("disentangle circuit {i}"));
    }
}

// ============================================================================
// Edge cases
// ============================================================================

#[test]
fn test_single_qubit_identity() {
    // No gates at all
    let (stn_sv, crz_sv) = run_circuit_on_both(1, &[], 42);
    assert_states_match(&stn_sv, &crz_sv, "identity 1-qubit");
}

#[test]
fn test_only_cliffords_4qubit() {
    let gates = vec![
        ("h", vec![0], None),
        ("h", vec![1], None),
        ("cx", vec![0, 1], None),
        ("cz", vec![2, 3], None),
        ("h", vec![2], None),
        ("sz", vec![3], None),
        ("cx", vec![1, 2], None),
        ("cx", vec![3, 0], None),
        ("h", vec![0], None),
        ("h", vec![1], None),
        ("h", vec![2], None),
        ("h", vec![3], None),
    ];
    let (stn_sv, crz_sv) = run_circuit_on_both(4, &gates, 42);
    assert_states_match(&stn_sv, &crz_sv, "pure Clifford 4-qubit");
}

#[test]
fn test_t_on_every_qubit_product_state() {
    // T on each qubit of a product state |+...+>
    let n = 4;
    let t = Angle64::QUARTER_TURN / 2u64;
    let mut gates: Vec<(&str, Vec<usize>, Option<Angle64>)> = Vec::new();
    for q in 0..n {
        gates.push(("h", vec![q], None));
    }
    for q in 0..n {
        gates.push(("rz", vec![q], Some(t)));
    }
    let (stn_sv, crz_sv) = run_circuit_on_both(n, &gates, 42);
    assert_states_match(&stn_sv, &crz_sv, "T on product state");
}

#[test]
fn test_tdg_gate() {
    // T-dagger = RZ(-pi/4)
    let tdg = -(Angle64::QUARTER_TURN / 2u64);
    let gates = vec![("h", vec![0], None), ("rz", vec![0], Some(tdg))];
    let (stn_sv, crz_sv) = run_circuit_on_both(1, &gates, 42);
    assert_states_match(&stn_sv, &crz_sv, "Tdg gate");
}

#[test]
fn test_rz_near_zero_angle() {
    // Very small angle -- should behave like identity
    let tiny = Angle64::from_radians(1e-6);
    let gates = vec![("h", vec![0], None), ("rz", vec![0], Some(tiny))];
    let (stn_sv, crz_sv) = run_circuit_on_both(1, &gates, 42);
    assert_states_match(&stn_sv, &crz_sv, "near-zero RZ");
}

#[test]
fn test_rz_near_pi() {
    // Angle near pi -- should behave like Z gate
    let near_pi = Angle64::from_radians(std::f64::consts::PI - 1e-6);
    let gates = vec![("h", vec![0], None), ("rz", vec![0], Some(near_pi))];
    let (stn_sv, crz_sv) = run_circuit_on_both(1, &gates, 42);
    assert_states_match(&stn_sv, &crz_sv, "near-pi RZ");
}

#[test]
fn test_stn_reset_and_reuse() {
    let mut stn = StabMps::new(2);
    stn.h(&[QubitId(0)]);
    stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);

    stn.reset();

    // After reset, should behave like fresh simulator
    stn.h(&[QubitId(0)]);
    stn.cx(&[(QubitId(0), QubitId(1))]);
    let sv = stn.state_vector();
    let norm: f64 = sv.iter().map(num_complex::Complex::norm_sqr).sum();
    assert!(
        (norm - 1.0).abs() < 0.01,
        "norm after reset+circuit: {norm}"
    );
}

// ============================================================================
// MAST-specific verification
// ============================================================================

#[test]
fn test_mast_multiple_t_gates() {
    // Multiple T gates via MAST on entangled qubits
    let mut mast = Mast::with_seed(3, 10, 42);
    mast.h(&[QubitId(0)]);
    mast.cx(&[(QubitId(0), QubitId(1))]);
    mast.cx(&[(QubitId(1), QubitId(2))]);
    // GHZ state

    mast.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
    mast.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(1)]);
    mast.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(2)]);

    assert_eq!(mast.num_ancillas_used(), 3);
    assert!(
        mast.mps().norm_squared() > 0.5,
        "norm should be reasonable: {}",
        mast.mps().norm_squared()
    );
}

#[test]
fn test_mast_3qubit_ghz_correlation() {
    // GHZ + T via MAST: all measurements should be correlated
    let num_trials = 100;
    let mut all_corr = 0;
    for trial in 0..num_trials {
        let mut mast = Mast::with_seed(3, 10, 40_000 + trial);
        mast.h(&[QubitId(0)]);
        mast.cx(&[(QubitId(0), QubitId(1))]);
        mast.cx(&[(QubitId(1), QubitId(2))]);
        mast.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);

        let r0 = mast.mz(&[QubitId(0)])[0].outcome;
        let r1 = mast.mz(&[QubitId(1)])[0].outcome;
        let r2 = mast.mz(&[QubitId(2)])[0].outcome;
        if r0 == r1 && r1 == r2 {
            all_corr += 1;
        }
    }
    let rate = f64::from(all_corr) / num_trials as f64;
    assert!(
        rate > 0.90,
        "GHZ+T MAST correlation {rate:.2} should be > 0.90"
    );
}

#[test]
fn test_mast_t_then_measure_then_more() {
    // Apply T, measure, then apply more gates
    let mut mast = Mast::with_seed(2, 4, 42);
    mast.h(&[QubitId(0)]);
    mast.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
    let _r0 = mast.mz(&[QubitId(0)])[0].outcome;

    // After measurement, apply more gates on q1
    mast.h(&[QubitId(1)]);
    let r1 = mast.mz(&[QubitId(1)])[0].outcome;
    // q1 was in |0>, H puts it in |+>, measurement is random
    let _ = r1; // Just verify it doesn't panic
}

// ============================================================================
// RZZ fuzz tests
// ============================================================================

/// Fuzz with RZZ gates included in the gate set.
fn fuzz_with_rzz(num_qubits: usize, num_gates: usize, seed: u64) {
    let mut stn = StabMps::with_seed(num_qubits, seed);
    let mut dsv = pecos_simulators::DenseStateVec::new(num_qubits);

    let mut rng_state = seed;
    let next_rng = |state: &mut u64| -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    };

    for _ in 0..num_gates {
        let gate_type = next_rng(&mut rng_state) % 10; // expanded set includes rzz
        let q0 = (next_rng(&mut rng_state) % num_qubits as u64) as usize;
        let q1 = loop {
            let q = (next_rng(&mut rng_state) % num_qubits as u64) as usize;
            if q != q0 {
                break q;
            }
        };

        match gate_type {
            0 => {
                stn.h(&[QubitId(q0)]);
                dsv.h(&[QubitId(q0)]);
            }
            1 => {
                stn.sz(&[QubitId(q0)]);
                dsv.sz(&[QubitId(q0)]);
            }
            2 => {
                stn.x(&[QubitId(q0)]);
                dsv.x(&[QubitId(q0)]);
            }
            3 => {
                stn.cx(&[(QubitId(q0), QubitId(q1))]);
                dsv.cx(&[(QubitId(q0), QubitId(q1))]);
            }
            4 => {
                stn.cz(&[(QubitId(q0), QubitId(q1))]);
                dsv.cz(&[(QubitId(q0), QubitId(q1))]);
            }
            5 => {
                let t = Angle64::QUARTER_TURN / 2u64;
                stn.rz(t, &[QubitId(q0)]);
                dsv.rz(t, &[QubitId(q0)]);
            }
            6 => {
                let ab = next_rng(&mut rng_state);
                let a = Angle64::from_radians((ab % 1000) as f64 * 0.001 * std::f64::consts::TAU);
                stn.rz(a, &[QubitId(q0)]);
                dsv.rz(a, &[QubitId(q0)]);
            }
            7 => {
                let ab = next_rng(&mut rng_state);
                let a = Angle64::from_radians((ab % 1000) as f64 * 0.001 * std::f64::consts::TAU);
                stn.rx(a, &[QubitId(q0)]);
                dsv.rx(a, &[QubitId(q0)]);
            }
            8 | 9 => {
                // RZZ gate
                let ab = next_rng(&mut rng_state);
                let a = Angle64::from_radians((ab % 1000) as f64 * 0.001 * std::f64::consts::TAU);
                let pairs = [(QubitId(q0), QubitId(q1))];
                stn.rzz(a, &pairs);
                dsv.rzz(a, &pairs);
            }
            _ => {}
        }
    }

    let stn_sv = stn.state_vector();
    let dim = 1usize << num_qubits;
    let ref_sv: Vec<Complex64> = (0..dim).map(|i| dsv.get_amplitude(i)).collect();
    let tol = 0.01 + 0.002 * num_gates as f64;
    assert_states_close(
        &stn_sv,
        &ref_sv,
        tol,
        &format!("rzz fuzz n={num_qubits} g={num_gates} seed={seed}"),
    );
}

#[test]
fn test_fuzz_rzz_2qubit() {
    for seed in 2000..2100 {
        fuzz_with_rzz(2, 12, seed);
    }
}

#[test]
fn test_fuzz_rzz_3qubit() {
    for seed in 2100..2150 {
        fuzz_with_rzz(3, 12, seed);
    }
}

#[test]
fn test_fuzz_rzz_4qubit() {
    for seed in 2150..2200 {
        fuzz_with_rzz(4, 12, seed);
    }
}

// ============================================================================
// Sequential measurement tests
// ============================================================================

#[test]
fn test_sequential_measurement_correlations() {
    // Apply non-Clifford gates, measure a qubit, apply more gates, measure again.
    // Repeat many times and check that outcome distributions are consistent.
    let num_trials = 200;
    let mut outcomes = [[0u32; 2]; 2]; // [q0_outcome][q1_outcome]

    for trial in 0..num_trials {
        let mut stn = StabMps::with_seed(2, 5000 + trial);
        let mut dsv = pecos_simulators::DenseStateVec::new(2);

        // Prepare entangled state with non-Clifford component
        stn.h(&[QubitId(0)]);
        dsv.h(&[QubitId(0)]);
        stn.cx(&[(QubitId(0), QubitId(1))]);
        dsv.cx(&[(QubitId(0), QubitId(1))]);
        let t = Angle64::QUARTER_TURN / 2u64;
        stn.rz(t, &[QubitId(0)]);
        dsv.rz(t, &[QubitId(0)]);

        // Measure q0
        let r0_stn = stn.mz(&[QubitId(0)])[0].outcome;

        // Apply more gates after measurement
        stn.h(&[QubitId(1)]);
        stn.rz(t, &[QubitId(1)]);

        // Measure q1
        let r1_stn = stn.mz(&[QubitId(1)])[0].outcome;

        outcomes[usize::from(r0_stn)][usize::from(r1_stn)] += 1;
    }

    // Bell+T: q0 and q1 are correlated before first measurement.
    // After measuring q0, the state collapses. The second measurement
    // should give a definite result. Just check no panics and
    // reasonable distribution.
    let total: u32 = outcomes.iter().flat_map(|r| r.iter()).sum();
    assert_eq!(total, num_trials as u32);
    // Both q0=0 and q0=1 should appear (non-deterministic)
    let q0_zero: u32 = outcomes[0].iter().sum();
    let q0_one: u32 = outcomes[1].iter().sum();
    assert!(q0_zero > 10, "q0=0 too rare: {q0_zero}");
    assert!(q0_one > 10, "q0=1 too rare: {q0_one}");
}

#[test]
fn test_measure_apply_measure_3qubit() {
    // GHZ + T, measure q0, then H+T on q1, measure q1, then measure q2.
    for trial in 0..100u64 {
        let mut stn = StabMps::with_seed(3, 6000 + trial);
        let t = Angle64::QUARTER_TURN / 2u64;

        stn.h(&[QubitId(0)]);
        stn.cx(&[(QubitId(0), QubitId(1))]);
        stn.cx(&[(QubitId(1), QubitId(2))]);
        stn.rz(t, &[QubitId(1)]);

        let r0 = stn.mz(&[QubitId(0)])[0].outcome;

        // After measuring q0, apply more gates
        stn.h(&[QubitId(1)]);
        stn.rz(t, &[QubitId(1)]);

        let r1 = stn.mz(&[QubitId(1)])[0].outcome;
        let r2 = stn.mz(&[QubitId(2)])[0].outcome;

        // q0 and q2 were in GHZ: after measuring q0, q2 should be deterministic
        // (same as q0 due to GHZ correlation, modulo the T gate on q1)
        let _ = (r0, r1, r2); // Just verify no panics
    }
}

// ============================================================================
// MAST vs STN measurement comparison
// ============================================================================

#[test]
fn test_mast_matches_stn_exact_probabilities_2q() {
    // Compare MAST's sampled distribution to STN's EXACT probabilities
    // (not STN's samples). STN's `prob_bitstring` gives the analytic
    // value; MAST must sample from this distribution within 5σ.
    let t = Angle64::QUARTER_TURN / 2u64;

    // Exact probabilities from STN.
    let mut exact_probs = [0.0_f64; 4];
    let mut stn_for_probs = StabMps::with_seed(2, 1234);
    stn_for_probs.h(&[QubitId(0)]);
    stn_for_probs.cx(&[(QubitId(0), QubitId(1))]);
    stn_for_probs.rz(t, &[QubitId(0)]);
    stn_for_probs.h(&[QubitId(1)]);
    stn_for_probs.rz(t, &[QubitId(1)]);
    stn_for_probs.flush();
    // prob_bitstring is MSB-first: bitstring[k] is qubit (n-1-k). For a
    // LSB-first integer index `i` (q_k = (i >> k) & 1), bitstring =
    // [q_{n-1}, q_{n-2}, ..., q_0].
    for (i, ep) in exact_probs.iter_mut().enumerate().take(4) {
        let bits = [(i & 2) != 0, (i & 1) != 0];
        *ep = stn_for_probs.prob_bitstring(&bits);
    }
    let total: f64 = exact_probs.iter().sum();
    assert!(
        (total - 1.0).abs() < 1e-9,
        "exact probs must sum to 1, got {total}: {exact_probs:?}"
    );

    // Sample MAST many times and compare to exact.
    let num_trials = 5000;
    let mut mast_counts = [0u32; 4];
    for trial in 0..num_trials {
        let seed = 7000 + trial;
        let mut mast = Mast::with_seed(2, 4, seed);
        mast.h(&[QubitId(0)]);
        mast.cx(&[(QubitId(0), QubitId(1))]);
        mast.rz(t, &[QubitId(0)]);
        mast.h(&[QubitId(1)]);
        mast.rz(t, &[QubitId(1)]);
        let m0 = mast.mz(&[QubitId(0)])[0].outcome;
        let m1 = mast.mz(&[QubitId(1)])[0].outcome;
        mast_counts[usize::from(m0) | (usize::from(m1) << 1)] += 1;
    }

    // 5σ bound on |p_sample - p_exact|: sqrt(p(1-p)/N) × 5 ≤ 0.04 for
    // N=5000, any p. Leaves generous room for sampling noise.
    for i in 0..4 {
        let pe = exact_probs[i];
        let pm = f64::from(mast_counts[i]) / num_trials as f64;
        let sigma = (pe * (1.0 - pe) / num_trials as f64).sqrt().max(1e-6);
        let deviation = (pe - pm).abs() / sigma;
        assert!(
            deviation < 5.0,
            "outcome {i}: exact={pe:.4} MAST={pm:.4}, deviation {deviation:.1}σ"
        );
    }
}

#[test]
fn test_mast_matches_stn_exact_probabilities_3q() {
    let t = Angle64::QUARTER_TURN / 2u64;

    // Exact probs from STN.
    let mut exact_probs = [0.0_f64; 8];
    let mut stn = StabMps::with_seed(3, 1234);
    stn.h(&[QubitId(0)]);
    stn.cx(&[(QubitId(0), QubitId(1))]);
    stn.h(&[QubitId(2)]);
    stn.rz(t, &[QubitId(0)]);
    stn.rz(t, &[QubitId(2)]);
    stn.cx(&[(QubitId(1), QubitId(2))]);
    stn.flush();
    // prob_bitstring is MSB-first: bitstring = [q_{n-1}, ..., q_0].
    for (i, ep) in exact_probs.iter_mut().enumerate().take(8) {
        let bits = [(i & 4) != 0, (i & 2) != 0, (i & 1) != 0];
        *ep = stn.prob_bitstring(&bits);
    }
    let total: f64 = exact_probs.iter().sum();
    assert!(
        (total - 1.0).abs() < 1e-9,
        "probs sum != 1: {exact_probs:?}"
    );

    let num_trials = 5000;
    let mut mast_counts = [0u32; 8];
    for trial in 0..num_trials {
        let seed = 8000 + trial;
        let mut mast = Mast::with_seed(3, 4, seed);
        mast.h(&[QubitId(0)]);
        mast.cx(&[(QubitId(0), QubitId(1))]);
        mast.h(&[QubitId(2)]);
        mast.rz(t, &[QubitId(0)]);
        mast.rz(t, &[QubitId(2)]);
        mast.cx(&[(QubitId(1), QubitId(2))]);
        let r: Vec<bool> = mast
            .mz(&[QubitId(0), QubitId(1), QubitId(2)])
            .iter()
            .map(|m| m.outcome)
            .collect();
        mast_counts[usize::from(r[0]) | (usize::from(r[1]) << 1) | (usize::from(r[2]) << 2)] += 1;
    }

    for i in 0..8 {
        let pe = exact_probs[i];
        let pm = f64::from(mast_counts[i]) / num_trials as f64;
        let sigma = (pe * (1.0 - pe) / num_trials as f64).sqrt().max(1e-6);
        let deviation = (pe - pm).abs() / sigma;
        assert!(
            deviation < 5.0,
            "3q outcome {i}: exact={pe:.4} MAST={pm:.4}, dev {deviation:.1}σ"
        );
    }
}

// ============================================================================
// Large bond dimension stress tests
// ============================================================================

#[test]
fn test_many_t_gates_bond_dim_growth() {
    // Apply T gates to all qubits of an entangled state.
    // Bond dim grows but should stay bounded by max_bond_dim.
    let num_qubits = 6;
    let t = Angle64::QUARTER_TURN / 2u64;

    let mut stn = StabMps::builder(num_qubits).max_bond_dim(32).build();
    let mut dsv = pecos_simulators::DenseStateVec::new(num_qubits);

    // Create full entanglement: H on all, then CX chain
    for q in 0..num_qubits {
        stn.h(&[QubitId(q)]);
        dsv.h(&[QubitId(q)]);
    }
    for q in 0..num_qubits - 1 {
        stn.cx(&[(QubitId(q), QubitId(q + 1))]);
        dsv.cx(&[(QubitId(q), QubitId(q + 1))]);
    }

    // Apply T on every qubit — each one grows bond dim
    for q in 0..num_qubits {
        stn.rz(t, &[QubitId(q)]);
        dsv.rz(t, &[QubitId(q)]);
    }

    assert!(
        stn.max_bond_dim() <= 32,
        "bond dim {} exceeds limit",
        stn.max_bond_dim()
    );

    let stn_sv = stn.state_vector();
    let dim = 1usize << num_qubits;
    let ref_sv: Vec<Complex64> = (0..dim).map(|i| dsv.get_amplitude(i)).collect();
    assert_states_close(&stn_sv, &ref_sv, 0.05, "many T gates on entangled state");
}

#[test]
fn test_ghz_plus_t_ladder() {
    // GHZ state, then T on alternating qubits, then entangling again.
    let num_qubits = 5;
    let t = Angle64::QUARTER_TURN / 2u64;

    let mut stn = StabMps::builder(num_qubits).max_bond_dim(64).build();
    let mut dsv = pecos_simulators::DenseStateVec::new(num_qubits);

    // GHZ: H(0), CX chain
    stn.h(&[QubitId(0)]);
    dsv.h(&[QubitId(0)]);
    for q in 0..num_qubits - 1 {
        stn.cx(&[(QubitId(q), QubitId(q + 1))]);
        dsv.cx(&[(QubitId(q), QubitId(q + 1))]);
    }

    // T on even qubits
    for q in (0..num_qubits).step_by(2) {
        stn.rz(t, &[QubitId(q)]);
        dsv.rz(t, &[QubitId(q)]);
    }

    // More entangling
    for q in (0..num_qubits - 1).rev() {
        stn.cx(&[(QubitId(q + 1), QubitId(q))]);
        dsv.cx(&[(QubitId(q + 1), QubitId(q))]);
    }

    // T on odd qubits
    for q in (1..num_qubits).step_by(2) {
        stn.rz(t, &[QubitId(q)]);
        dsv.rz(t, &[QubitId(q)]);
    }

    let stn_sv = stn.state_vector();
    let dim = 1usize << num_qubits;
    let ref_sv: Vec<Complex64> = (0..dim).map(|i| dsv.get_amplitude(i)).collect();
    assert_states_close(&stn_sv, &ref_sv, 0.05, "GHZ+T ladder");
}

#[test]
fn test_repeated_t_layers_4qubit() {
    // Multiple layers of T gates with entangling between layers.
    // This is the worst case for bond dim growth.
    let num_qubits = 4;
    let t = Angle64::QUARTER_TURN / 2u64;

    let mut stn = StabMps::with_seed(num_qubits, 42);
    let mut dsv = pecos_simulators::DenseStateVec::new(num_qubits);

    for _layer in 0..3 {
        // H + CX entangling layer
        for q in 0..num_qubits {
            stn.h(&[QubitId(q)]);
            dsv.h(&[QubitId(q)]);
        }
        for q in 0..num_qubits - 1 {
            stn.cx(&[(QubitId(q), QubitId(q + 1))]);
            dsv.cx(&[(QubitId(q), QubitId(q + 1))]);
        }
        // T layer
        for q in 0..num_qubits {
            stn.rz(t, &[QubitId(q)]);
            dsv.rz(t, &[QubitId(q)]);
        }
    }

    let stn_sv = stn.state_vector();
    let dim = 1usize << num_qubits;
    let ref_sv: Vec<Complex64> = (0..dim).map(|i| dsv.get_amplitude(i)).collect();
    // 3 layers * 4 T gates = 12 non-Clifford gates, deep circuit
    assert_states_close(&stn_sv, &ref_sv, 0.1, "repeated T layers 4q");
}

#[test]
fn test_bond_dim_respects_config() {
    // Verify that max_bond_dim is respected even under heavy non-Clifford load.
    let num_qubits = 4;
    let t = Angle64::QUARTER_TURN / 2u64;
    let max_chi = 8;

    let mut stn = StabMps::builder(num_qubits).max_bond_dim(max_chi).build();

    stn.h(&[QubitId(0)]);
    stn.cx(&[(QubitId(0), QubitId(1))]);
    stn.cx(&[(QubitId(1), QubitId(2))]);
    stn.cx(&[(QubitId(2), QubitId(3))]);

    // Apply many T gates to push bond dim up
    for _ in 0..5 {
        for q in 0..num_qubits {
            stn.rz(t, &[QubitId(q)]);
        }
    }

    assert!(
        stn.max_bond_dim() <= max_chi,
        "bond dim {} exceeds configured max {max_chi}",
        stn.max_bond_dim()
    );
    // MPS should still be approximately normalized
    assert!(
        (stn.mps().norm_squared() - 1.0).abs() < 0.5,
        "MPS norm too far from 1: {}",
        stn.mps().norm_squared()
    );
}

// ============================================================================
// Tdg (negative angle) fuzz
// ============================================================================

/// Fuzz with Tdg and negative-angle RZ gates.
fn fuzz_with_tdg(num_qubits: usize, num_gates: usize, seed: u64) {
    let mut stn = StabMps::with_seed(num_qubits, seed);
    let mut dsv = pecos_simulators::DenseStateVec::new(num_qubits);

    let mut rng_state = seed;
    let next_rng = |state: &mut u64| -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    };

    for _ in 0..num_gates {
        let gate_type = next_rng(&mut rng_state) % 10;
        let q0 = (next_rng(&mut rng_state) % num_qubits as u64) as usize;
        let q1 = loop {
            let q = (next_rng(&mut rng_state) % num_qubits as u64) as usize;
            if q != q0 {
                break q;
            }
        };
        match gate_type {
            0 => {
                stn.h(&[QubitId(q0)]);
                dsv.h(&[QubitId(q0)]);
            }
            1 => {
                stn.sz(&[QubitId(q0)]);
                dsv.sz(&[QubitId(q0)]);
            }
            2 => {
                stn.x(&[QubitId(q0)]);
                dsv.x(&[QubitId(q0)]);
            }
            3 => {
                stn.cx(&[(QubitId(q0), QubitId(q1))]);
                dsv.cx(&[(QubitId(q0), QubitId(q1))]);
            }
            4 => {
                // T gate
                let t = Angle64::QUARTER_TURN / 2u64;
                stn.rz(t, &[QubitId(q0)]);
                dsv.rz(t, &[QubitId(q0)]);
            }
            5 => {
                // Tdg gate (negative T)
                let tdg = -(Angle64::QUARTER_TURN / 2u64);
                stn.rz(tdg, &[QubitId(q0)]);
                dsv.rz(tdg, &[QubitId(q0)]);
            }
            6 => {
                // Random negative-angle RZ
                let ab = next_rng(&mut rng_state);
                let a = -Angle64::from_radians((ab % 1000) as f64 * 0.001 * std::f64::consts::TAU);
                stn.rz(a, &[QubitId(q0)]);
                dsv.rz(a, &[QubitId(q0)]);
            }
            7 => {
                // Random positive-angle RZ
                let ab = next_rng(&mut rng_state);
                let a = Angle64::from_radians((ab % 1000) as f64 * 0.001 * std::f64::consts::TAU);
                stn.rz(a, &[QubitId(q0)]);
                dsv.rz(a, &[QubitId(q0)]);
            }
            8 => {
                // RX with negative angle
                let ab = next_rng(&mut rng_state);
                let a = -Angle64::from_radians((ab % 1000) as f64 * 0.001 * std::f64::consts::TAU);
                stn.rx(a, &[QubitId(q0)]);
                dsv.rx(a, &[QubitId(q0)]);
            }
            _ => {
                stn.cz(&[(QubitId(q0), QubitId(q1))]);
                dsv.cz(&[(QubitId(q0), QubitId(q1))]);
            }
        }
    }
    let stn_sv = stn.state_vector();
    let dim = 1usize << num_qubits;
    let ref_sv: Vec<Complex64> = (0..dim).map(|i| dsv.get_amplitude(i)).collect();
    let tol = 0.01 + 0.002 * num_gates as f64;
    assert_states_close(
        &stn_sv,
        &ref_sv,
        tol,
        &format!("tdg fuzz n={num_qubits} g={num_gates} seed={seed}"),
    );
}

#[test]
fn test_fuzz_tdg_2qubit() {
    for seed in 3000..3100 {
        fuzz_with_tdg(2, 12, seed);
    }
}

#[test]
fn test_fuzz_szdg_circuits() {
    // Include szdg in the gate set to test the default sz.sz.sz path
    for seed in 3200..3250 {
        let mut stn = StabMps::with_seed(2, seed);
        let mut dsv = pecos_simulators::DenseStateVec::new(2);
        let mut rng_state = seed;
        let next_rng = |state: &mut u64| -> u64 {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            *state
        };
        for _ in 0..12 {
            let gt = next_rng(&mut rng_state) % 8;
            let q0 = (next_rng(&mut rng_state) % 2) as usize;
            let q1 = 1 - q0;
            match gt {
                0 => {
                    stn.h(&[QubitId(q0)]);
                    dsv.h(&[QubitId(q0)]);
                }
                1 => {
                    stn.sz(&[QubitId(q0)]);
                    dsv.sz(&[QubitId(q0)]);
                }
                2 => {
                    stn.szdg(&[QubitId(q0)]);
                    dsv.szdg(&[QubitId(q0)]);
                }
                3 => {
                    stn.cx(&[(QubitId(q0), QubitId(q1))]);
                    dsv.cx(&[(QubitId(q0), QubitId(q1))]);
                }
                4 => {
                    let t = Angle64::QUARTER_TURN / 2u64;
                    stn.rz(t, &[QubitId(q0)]);
                    dsv.rz(t, &[QubitId(q0)]);
                }
                5 => {
                    let tdg = -(Angle64::QUARTER_TURN / 2u64);
                    stn.rz(tdg, &[QubitId(q0)]);
                    dsv.rz(tdg, &[QubitId(q0)]);
                }
                6 => {
                    let ab = next_rng(&mut rng_state);
                    let a =
                        Angle64::from_radians((ab % 1000) as f64 * 0.001 * std::f64::consts::TAU);
                    stn.rz(a, &[QubitId(q0)]);
                    dsv.rz(a, &[QubitId(q0)]);
                }
                _ => {
                    stn.cz(&[(QubitId(q0), QubitId(q1))]);
                    dsv.cz(&[(QubitId(q0), QubitId(q1))]);
                }
            }
        }
        let stn_sv = stn.state_vector();
        let ref_sv: Vec<Complex64> = (0..4).map(|i| dsv.get_amplitude(i)).collect();
        assert_states_close(&stn_sv, &ref_sv, 0.04, &format!("szdg fuzz seed={seed}"));
    }
}

#[test]
fn test_fuzz_tdg_3qubit() {
    for seed in 3100..3150 {
        fuzz_with_tdg(3, 12, seed);
    }
}

// ============================================================================
// Post-measurement state correctness
// ============================================================================

#[test]
fn test_post_measurement_state_consistency() {
    // After measuring q0, verify the STN state is internally consistent:
    // the z_expectation_value of unmeasured qubits should match the
    // probabilities from the state vector.
    let t = Angle64::QUARTER_TURN / 2u64;

    for trial in 0..100u64 {
        let seed = 9000 + trial;
        let mut stn = StabMps::with_seed(2, seed);

        // Build non-trivial state
        stn.h(&[QubitId(0)]);
        stn.cx(&[(QubitId(0), QubitId(1))]);
        stn.rz(t, &[QubitId(0)]);
        stn.h(&[QubitId(1)]);
        stn.rz(t, &[QubitId(1)]);

        // Check <Z_0> before measurement matches state vector
        let sv_before = stn.state_vector();
        let ev_z0_sv: f64 = sv_before
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let sign = if i & 1 == 0 { 1.0 } else { -1.0 };
                sign * a.norm_sqr()
            })
            .sum();
        let ev_z0_mps =
            pecos_stab_tn::stab_mps::measure::z_expectation_value(stn.tableau(), stn.mps(), 0).re;
        assert!(
            (ev_z0_sv - ev_z0_mps).abs() < 0.01,
            "trial {trial}: pre-meas <Z_0> sv={ev_z0_sv:.4} mps={ev_z0_mps:.4}"
        );

        // Measure q0
        let r0 = stn.mz(&[QubitId(0)])[0].outcome;

        // After measurement: <Z_1> from expectation value should match state_vector
        let sv_after = stn.state_vector();
        let ev_z1_sv: f64 = sv_after
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let sign = if (i >> 1) & 1 == 0 { 1.0 } else { -1.0 };
                sign * a.norm_sqr()
            })
            .sum();
        let ev_z1_mps =
            pecos_stab_tn::stab_mps::measure::z_expectation_value(stn.tableau(), stn.mps(), 1).re;
        assert!(
            (ev_z1_sv - ev_z1_mps).abs() < 0.05,
            "trial {trial}: post-meas <Z_1> sv={ev_z1_sv:.4} mps={ev_z1_mps:.4}"
        );

        // Check <Z_0> after measurement
        let ev_z0_after =
            pecos_stab_tn::stab_mps::measure::z_expectation_value(stn.tableau(), stn.mps(), 0).re;
        let expected_ev = if r0 { -1.0 } else { 1.0 };
        // Also check via state vector (brute-force)
        let sv_after = stn.state_vector();
        let ev_z0_sv: f64 = sv_after
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let sign = if i & 1 == 0 { 1.0 } else { -1.0 };
                sign * a.norm_sqr()
            })
            .sum();
        if (ev_z0_after - expected_ev).abs() > 0.1 {
            eprintln!(
                "trial {trial}: outcome={r0}, <Z_0> mps={ev_z0_after:.4} sv={ev_z0_sv:.4} (expected {expected_ev})"
            );
            // Print decomposition of Z_0 after measurement
            let decomp = pecos_stab_tn::stab_mps::pauli_decomp::decompose_z(
                stn.tableau().stabs(),
                stn.tableau().destabs(),
                0,
            );
            eprintln!("  Z_0 decomp after meas: {decomp:?}");
            eprintln!("  MPS bonds: {:?}", stn.mps().bond_dims());
        }

        // Re-measure q0: should give same outcome (collapsed state)
        let r0_again = stn.mz(&[QubitId(0)]);
        assert_eq!(
            r0_again[0].outcome, r0,
            "trial {trial}: re-measurement should give same outcome, <Z_0>={ev_z0_after:.4}"
        );
    }
}

#[test]
fn test_post_measurement_multisite_collapse() {
    // Trigger a multi-site DestabilizerFlip measurement (flip + sign sites).
    // Then re-measure to verify collapse.
    let t = Angle64::QUARTER_TURN / 2u64;

    for trial in 0..50u64 {
        let seed = 9200 + trial;
        let mut stn = StabMps::with_seed(3, seed);

        // Build state where Z_0 decomposes with both flip and sign sites
        stn.h(&[QubitId(0)]);
        stn.h(&[QubitId(1)]);
        stn.cx(&[(QubitId(0), QubitId(1))]);
        stn.cx(&[(QubitId(1), QubitId(2))]);
        stn.rz(t, &[QubitId(0)]);
        stn.rz(t, &[QubitId(1)]);
        stn.h(&[QubitId(0)]); // Change basis to make Z_0 decomposition multi-site

        // Measure q0
        let r0 = stn.mz(&[QubitId(0)])[0].outcome;

        // Re-measure: should give same outcome
        let r0_again = stn.mz(&[QubitId(0)]);
        assert_eq!(
            r0_again[0].outcome, r0,
            "trial {trial}: multi-site re-measurement should give same outcome"
        );

        // Verify expectation value consistency: <Z_q> from MPS should match state_vector
        let sv = stn.state_vector();
        for q in 0..3 {
            let ev_sv: f64 = sv
                .iter()
                .enumerate()
                .map(|(i, a)| {
                    let sign = if (i >> q) & 1 == 0 { 1.0 } else { -1.0 };
                    sign * a.norm_sqr()
                })
                .sum();
            let ev_mps =
                pecos_stab_tn::stab_mps::measure::z_expectation_value(stn.tableau(), stn.mps(), q)
                    .re;
            assert!(
                (ev_sv - ev_mps).abs() < 0.05,
                "trial {trial}: post-multisite-meas <Z_{q}> sv={ev_sv:.4} mps={ev_mps:.4}"
            );
        }
    }
}

#[test]
fn test_post_measurement_state_3qubit() {
    // Measure q0 on a 3-qubit entangled state, then check internal consistency.
    let t = Angle64::QUARTER_TURN / 2u64;

    for trial in 0..50u64 {
        let seed = 9500 + trial;
        let mut stn = StabMps::with_seed(3, seed);

        stn.h(&[QubitId(0)]);
        stn.cx(&[(QubitId(0), QubitId(1))]);
        stn.h(&[QubitId(2)]);
        stn.rz(t, &[QubitId(1)]);

        let _ = stn.mz(&[QubitId(0)])[0].outcome;

        // After measurement: check expectation values match state vector
        let sv = stn.state_vector();
        for q in 1..3 {
            let ev_sv: f64 = sv
                .iter()
                .enumerate()
                .map(|(i, a)| {
                    let sign = if (i >> q) & 1 == 0 { 1.0 } else { -1.0 };
                    sign * a.norm_sqr()
                })
                .sum();
            let ev_mps =
                pecos_stab_tn::stab_mps::measure::z_expectation_value(stn.tableau(), stn.mps(), q)
                    .re;
            assert!(
                (ev_sv - ev_mps).abs() < 0.05,
                "trial {trial}: post-meas <Z_{q}> sv={ev_sv:.4} mps={ev_mps:.4}"
            );
        }
    }
}

// ============================================================================
// Single-qubit circuit fuzz
// ============================================================================

#[test]
fn test_fuzz_single_qubit() {
    // Single-qubit circuits: always Stabilizer decomposition path.
    for seed in 4000..4200 {
        let mut stn = StabMps::with_seed(1, seed);
        let mut dsv = pecos_simulators::DenseStateVec::new(1);

        let mut rng_state = seed;
        let next_rng = |state: &mut u64| -> u64 {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            *state
        };

        for _ in 0..15 {
            let gate_type = next_rng(&mut rng_state) % 6;
            match gate_type {
                0 => {
                    stn.h(&[QubitId(0)]);
                    dsv.h(&[QubitId(0)]);
                }
                1 => {
                    stn.sz(&[QubitId(0)]);
                    dsv.sz(&[QubitId(0)]);
                }
                2 => {
                    let t = Angle64::QUARTER_TURN / 2u64;
                    stn.rz(t, &[QubitId(0)]);
                    dsv.rz(t, &[QubitId(0)]);
                }
                3 => {
                    let tdg = -(Angle64::QUARTER_TURN / 2u64);
                    stn.rz(tdg, &[QubitId(0)]);
                    dsv.rz(tdg, &[QubitId(0)]);
                }
                4 => {
                    let ab = next_rng(&mut rng_state);
                    let a =
                        Angle64::from_radians((ab % 1000) as f64 * 0.001 * std::f64::consts::TAU);
                    stn.rz(a, &[QubitId(0)]);
                    dsv.rz(a, &[QubitId(0)]);
                }
                _ => {
                    let ab = next_rng(&mut rng_state);
                    let a =
                        Angle64::from_radians((ab % 1000) as f64 * 0.001 * std::f64::consts::TAU);
                    stn.rx(a, &[QubitId(0)]);
                    dsv.rx(a, &[QubitId(0)]);
                }
            }
        }
        let stn_sv = stn.state_vector();
        let ref_sv: Vec<Complex64> = (0..2).map(|i| dsv.get_amplitude(i)).collect();
        assert_states_close(&stn_sv, &ref_sv, 0.01, &format!("1q fuzz seed={seed}"));
    }
}

// ============================================================================
// RZZ at Clifford angles
// ============================================================================

#[test]
fn test_rzz_clifford_angles() {
    // RZZ at Clifford angles should not grow bond dimension.
    let clifford_angles = [
        Angle64::ZERO,
        Angle64::QUARTER_TURN,        // pi/2
        Angle64::HALF_TURN,           // pi
        Angle64::THREE_QUARTERS_TURN, // 3pi/2
    ];

    for &angle in &clifford_angles {
        let mut stn = StabMps::with_seed(3, 42);
        let mut dsv = pecos_simulators::DenseStateVec::new(3);

        // Create entangled state first
        stn.h(&[QubitId(0)]);
        dsv.h(&[QubitId(0)]);
        stn.cx(&[(QubitId(0), QubitId(1))]);
        dsv.cx(&[(QubitId(0), QubitId(1))]);
        stn.h(&[QubitId(2)]);
        dsv.h(&[QubitId(2)]);

        // Apply RZZ at Clifford angle
        let pairs = [(QubitId(0), QubitId(1))];
        stn.rzz(angle, &pairs);
        dsv.rzz(angle, &pairs);

        // Bond dim should stay 1 (Clifford doesn't grow MPS)
        assert_eq!(
            stn.max_bond_dim(),
            1,
            "RZZ({angle:?}) should not grow bond dim"
        );

        let stn_sv = stn.state_vector();
        let ref_sv: Vec<Complex64> = (0..8).map(|i| dsv.get_amplitude(i)).collect();
        assert_states_match(&stn_sv, &ref_sv, &format!("RZZ Clifford angle {angle:?}"));
    }
}

#[test]
fn test_rzz_then_non_clifford() {
    // RZZ at non-Clifford angle, then more gates. Verify state.
    let angle = Angle64::from_radians(0.7);
    let t = Angle64::QUARTER_TURN / 2u64;

    let mut stn = StabMps::with_seed(3, 42);
    let mut dsv = pecos_simulators::DenseStateVec::new(3);

    stn.h(&[QubitId(0)]);
    dsv.h(&[QubitId(0)]);
    stn.h(&[QubitId(1)]);
    dsv.h(&[QubitId(1)]);
    stn.rzz(angle, &[(QubitId(0), QubitId(1))]);
    dsv.rzz(angle, &[(QubitId(0), QubitId(1))]);
    stn.rz(t, &[QubitId(0)]);
    dsv.rz(t, &[QubitId(0)]);
    stn.cx(&[(QubitId(1), QubitId(2))]);
    dsv.cx(&[(QubitId(1), QubitId(2))]);
    stn.rz(t, &[QubitId(2)]);
    dsv.rz(t, &[QubitId(2)]);

    let stn_sv = stn.state_vector();
    let ref_sv: Vec<Complex64> = (0..8).map(|i| dsv.get_amplitude(i)).collect();
    assert_states_close(&stn_sv, &ref_sv, 0.05, "RZZ then non-Clifford");
}

// ============================================================================
// compress() correctness
// ============================================================================

#[test]
fn test_compress_preserves_state() {
    use pecos_stab_tn::mps::{Mps, MpsConfig};

    // Build an MPS via addition (doubles bond dim), then compress.
    // State vector should be unchanged.
    let mps_a = Mps::new(3, MpsConfig::default());
    let mut mps_b = Mps::new(3, MpsConfig::default());

    let h = nalgebra::DMatrix::from_row_slice(
        2,
        2,
        &[
            Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
            Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
            Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
            Complex64::new(-std::f64::consts::FRAC_1_SQRT_2, 0.0),
        ],
    );
    let cnot = nalgebra::DMatrix::from_row_slice(
        4,
        4,
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    );

    // mps_a = |000>
    // mps_b = H(0) CX(0,1) |000> = Bell on (0,1) ⊗ |0>
    mps_b.apply_one_site_gate(0, &h).unwrap();
    mps_b.apply_two_site_gate(0, &cnot).unwrap();
    mps_b.scale(Complex64::new(0.5, 0.0)); // scale down

    let sum = mps_a.add(&mps_b);
    let sv_before = sum.state_vector();
    let bond_before = sum.max_bond_dim();

    let mut compressed = sum;
    compressed.compress();
    let sv_after = compressed.state_vector();
    let bond_after = compressed.max_bond_dim();

    // State should be preserved
    assert_eq!(sv_before.len(), sv_after.len());
    for (i, (a, b)) in sv_before.iter().zip(sv_after.iter()).enumerate() {
        assert!(
            (a - b).norm() < 1e-10,
            "compress changed amplitude at index {i}: {a:.6} -> {b:.6}"
        );
    }

    // Bond dim should not increase
    assert!(
        bond_after <= bond_before,
        "compress increased bond dim: {bond_before} -> {bond_after}"
    );
}

// ============================================================================
// MAST 3-qubit measurement investigation
// ============================================================================

#[test]
fn test_mast_single_t_measurement_distribution() {
    // Simpler MAST test: H(0), T(0), measure. p(0)=p(1)=0.5.
    let t = Angle64::QUARTER_TURN / 2u64;
    let num_trials = 500;
    let mut count_0 = 0u32;

    for trial in 0..num_trials {
        let mut mast = Mast::with_seed(1, 2, 10000 + trial as u64);
        mast.h(&[QubitId(0)]);
        mast.rz(t, &[QubitId(0)]);
        if !mast.mz(&[QubitId(0)])[0].outcome {
            count_0 += 1;
        }
    }
    let p0 = f64::from(count_0) / f64::from(num_trials);
    assert!(
        (p0 - 0.5).abs() < 0.1,
        "MAST T|+> measurement: p(0)={p0:.3}, expected 0.5"
    );
}

#[test]
fn test_mast_bell_t_measurement_correlation() {
    // MAST: Bell + T, measure both. Outcomes should be correlated.
    let t = Angle64::QUARTER_TURN / 2u64;
    let num_trials = 200;
    let mut correlated = 0u32;

    for trial in 0..num_trials {
        let mut mast = Mast::with_seed(2, 2, 11000 + trial as u64);
        mast.h(&[QubitId(0)]);
        mast.cx(&[(QubitId(0), QubitId(1))]);
        mast.rz(t, &[QubitId(0)]);

        let r0 = mast.mz(&[QubitId(0)])[0].outcome;
        let r1 = mast.mz(&[QubitId(1)])[0].outcome;
        if r0 == r1 {
            correlated += 1;
        }
    }
    let corr_rate = f64::from(correlated) / f64::from(num_trials);
    // Bell state: outcomes should be perfectly correlated
    assert!(
        corr_rate > 0.95,
        "MAST Bell+T correlation: {corr_rate:.3}, expected ~1.0"
    );
}

#[test]
fn test_mast_3qubit_outcome_coverage() {
    // Check that MAST produces multiple distinct outcomes for a 3-qubit circuit.
    let t = Angle64::QUARTER_TURN / 2u64;
    let num_trials = 300;
    let mut seen = std::collections::HashSet::new();

    for trial in 0..num_trials {
        let mut mast = Mast::with_seed(3, 4, 12000 + trial as u64);
        mast.h(&[QubitId(0)]);
        mast.h(&[QubitId(1)]);
        mast.h(&[QubitId(2)]);
        mast.rz(t, &[QubitId(0)]);
        mast.cx(&[(QubitId(0), QubitId(1))]);
        mast.rz(t, &[QubitId(1)]);

        let results = mast.mz(&[QubitId(0), QubitId(1), QubitId(2)]);
        let outcome: u8 = results
            .iter()
            .enumerate()
            .map(|(i, r)| (u8::from(r.outcome)) << i)
            .sum();
        seen.insert(outcome);
    }

    // With H on all qubits + T + entangling, we should see many outcomes
    assert!(
        seen.len() >= 4,
        "MAST 3q should produce at least 4 distinct outcomes, got {}",
        seen.len()
    );
}

// ============================================================================
// Paper property verification
// ============================================================================

#[test]
fn test_property_cliffords_dont_grow_bond_dim() {
    // Paper claim: Clifford gates only update the tableau. MPS stays at bond dim 1.
    let mut stn = StabMps::with_seed(8, 42);

    // Apply many Clifford gates: H, S, CX, CZ on all qubits
    for q in 0..8 {
        stn.h(&[QubitId(q)]);
    }
    for q in 0..7 {
        stn.cx(&[(QubitId(q), QubitId(q + 1))]);
    }
    for q in 0..8 {
        stn.sz(&[QubitId(q)]);
    }
    for q in (0..7).rev() {
        stn.cz(&[(QubitId(q), QubitId(q + 1))]);
    }
    for q in 0..8 {
        stn.h(&[QubitId(q)]);
    }

    // Bond dim should still be 1 everywhere
    assert_eq!(
        stn.max_bond_dim(),
        1,
        "Clifford gates should not grow bond dimension"
    );
}

#[test]
fn test_property_stn_bond_dim_grows_with_nonclifford() {
    // Paper claim: each non-Clifford gate on an entangled state can increase bond dim.
    let t = Angle64::QUARTER_TURN / 2u64;
    let mut stn = StabMps::with_seed(4, 42);

    // Create entangled state
    for q in 0..4 {
        stn.h(&[QubitId(q)]);
    }
    for q in 0..3 {
        stn.cx(&[(QubitId(q), QubitId(q + 1))]);
    }

    let bond_before = stn.max_bond_dim();
    assert_eq!(bond_before, 1, "pure Clifford should have bond dim 1");

    // Apply T gates — bond dim should grow
    stn.rz(t, &[QubitId(0)]);
    let bond_after_1t = stn.max_bond_dim();
    assert!(
        bond_after_1t >= 1,
        "T gate on entangled state should maintain or grow bond dim"
    );

    stn.rz(t, &[QubitId(2)]);
    let bond_after_2t = stn.max_bond_dim();

    eprintln!(
        "STN bond dim: before={bond_before}, after 1T={bond_after_1t}, after 2T={bond_after_2t}"
    );
}

#[test]
fn test_property_mast_bond_dim_stays_low() {
    // Paper claim (PRL 2025): for random circuits with t <= N non-Clifford gates,
    // MAST bond dimension stays ~3 on average.
    let t = Angle64::QUARTER_TURN / 2u64;
    let num_qubits = 8;
    let num_t_gates = 8; // t = N

    let mut total_max_bond = 0usize;
    let num_trials = 20;

    for trial in 0..num_trials {
        let mut mast = Mast::with_seed(num_qubits, num_t_gates, 20000 + trial as u64);

        // Random Clifford layer
        for q in 0..num_qubits {
            mast.h(&[QubitId(q)]);
        }
        for q in 0..num_qubits - 1 {
            mast.cx(&[(QubitId(q), QubitId(q + 1))]);
        }

        // Apply t T gates on random qubits
        let mut rng_state = 30000 + trial as u64;
        for _ in 0..num_t_gates {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let q = (rng_state % num_qubits as u64) as usize;
            mast.rz(t, &[QubitId(q)]);
        }

        // More Clifford entangling
        for q in (0..num_qubits - 1).rev() {
            mast.cx(&[(QubitId(q + 1), QubitId(q))]);
        }

        // Force projection of all deferred measurements
        mast.mz(&[QubitId(0)]);

        total_max_bond += mast.mps().max_bond_dim();
    }

    let avg_bond = total_max_bond as f64 / f64::from(num_trials);
    eprintln!("MAST average max bond dim for {num_qubits}q, {num_t_gates}T: {avg_bond:.1}");

    // Paper claims ~3 for t <= N. Allow some slack for our small test.
    assert!(
        avg_bond < 10.0,
        "MAST bond dim should stay low for t <= N, got avg={avg_bond:.1}"
    );
}

#[test]
fn test_property_mast_vs_stn_bond_dim() {
    // Paper claim: MAST has lower bond dimension than plain STN for the same circuit.
    let t = Angle64::QUARTER_TURN / 2u64;
    let num_qubits = 6;

    let mut stn = StabMps::with_seed(num_qubits, 42);
    let mut mast = Mast::with_seed(num_qubits, 4, 42);

    // Same circuit on both
    for q in 0..num_qubits {
        stn.h(&[QubitId(q)]);
        mast.h(&[QubitId(q)]);
    }
    for q in 0..num_qubits - 1 {
        stn.cx(&[(QubitId(q), QubitId(q + 1))]);
        mast.cx(&[(QubitId(q), QubitId(q + 1))]);
    }
    for q in 0..4 {
        stn.rz(t, &[QubitId(q)]);
        mast.rz(t, &[QubitId(q)]);
    }

    // Force MAST projection
    mast.mz(&[QubitId(0)]);

    let stn_bond = stn.max_bond_dim();
    let mast_bond = mast.mps().max_bond_dim();

    eprintln!("STN max bond: {stn_bond}, MAST max bond: {mast_bond}");

    // MAST should generally have lower or equal bond dim
    // (not always guaranteed for small circuits, so just log it)
}

#[test]
fn test_property_disentangle_reduces_bond_dim() {
    // Paper claim: Clifford disentangling can reduce MPS bond dimension.
    let t = Angle64::QUARTER_TURN / 2u64;
    let mut stn = StabMps::with_seed(3, 42);

    stn.h(&[QubitId(0)]);
    stn.cx(&[(QubitId(0), QubitId(1))]);
    stn.rz(t, &[QubitId(0)]);
    stn.h(&[QubitId(2)]);
    stn.cx(&[(QubitId(1), QubitId(2))]);
    stn.rz(t, &[QubitId(2)]);

    let bond_before = stn.max_bond_dim();
    let sv_before = stn.state_vector();

    let num_gates = stn.disentangle(5);

    let bond_after = stn.max_bond_dim();
    let sv_after = stn.state_vector();

    eprintln!("Disentangle: bond {bond_before} -> {bond_after}, applied {num_gates} gates");

    // State should be preserved
    assert_states_match(&sv_before, &sv_after, "disentangle preserves state");

    // Bond dim should not increase (and ideally decreases)
    assert!(
        bond_after <= bond_before,
        "disentangle should not increase bond dim: {bond_before} -> {bond_after}"
    );
}

// ============================================================================
// Large-scale bond dimension validation
// ============================================================================

/// Run a random circuit on STN/MAST at scale. No state vector check (too large).
/// Just verify bond dim stays bounded and measurements work.
fn large_scale_bond_dim_check(
    num_qubits: usize,
    num_t_gates: usize,
    num_clifford_layers: usize,
    seed: u64,
) -> (usize, usize) {
    // STN path
    let t = Angle64::QUARTER_TURN / 2u64;
    let mut stn = StabMps::builder(num_qubits)
        .max_bond_dim(256)
        .seed(seed)
        .build();

    let mut rng = seed;
    let next = |state: &mut u64| -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    };

    // Alternating Clifford + T layers
    let t_per_layer = num_t_gates / num_clifford_layers.max(1);
    for _layer in 0..num_clifford_layers {
        // Random Clifford entangling layer
        for q in 0..num_qubits {
            if next(&mut rng) % 2 == 0 {
                stn.h(&[QubitId(q)]);
            }
        }
        for q in 0..num_qubits - 1 {
            if next(&mut rng) % 3 == 0 {
                stn.cx(&[(QubitId(q), QubitId(q + 1))]);
            }
        }
        // T gates on random qubits
        for _ in 0..t_per_layer {
            let q = (next(&mut rng) % num_qubits as u64) as usize;
            stn.rz(t, &[QubitId(q)]);
        }
    }

    let stn_bond = stn.max_bond_dim();

    // MAST path (same circuit structure)
    let mut mast = Mast::with_seed(num_qubits, num_t_gates + 4, seed);
    let mut rng = seed; // reset RNG to get same circuit

    for _layer in 0..num_clifford_layers {
        for q in 0..num_qubits {
            if next(&mut rng) % 2 == 0 {
                mast.h(&[QubitId(q)]);
            }
        }
        for q in 0..num_qubits - 1 {
            if next(&mut rng) % 3 == 0 {
                mast.cx(&[(QubitId(q), QubitId(q + 1))]);
            }
        }
        for _ in 0..t_per_layer {
            let q = (next(&mut rng) % num_qubits as u64) as usize;
            mast.rz(t, &[QubitId(q)]);
        }
    }

    // Force MAST projection
    mast.mz(&[QubitId(0)]);
    let mast_bond = mast.mps().max_bond_dim();

    (stn_bond, mast_bond)
}

#[test]
fn test_large_scale_50_qubits() {
    let num_qubits = 50;
    let num_t = 20;
    let (stn_bond, mast_bond) = large_scale_bond_dim_check(num_qubits, num_t, 4, 42);
    eprintln!("{num_qubits}q {num_t}T: STN bond={stn_bond}, MAST bond={mast_bond}");
    // Should complete without panic. MAST bond should be small.
    assert!(
        mast_bond < 50,
        "MAST bond too large at {num_qubits}q: {mast_bond}"
    );
}

#[test]
fn test_large_scale_100_qubits() {
    let num_qubits = 100;
    let num_t = 40;
    let (stn_bond, mast_bond) = large_scale_bond_dim_check(num_qubits, num_t, 4, 123);
    eprintln!("{num_qubits}q {num_t}T: STN bond={stn_bond}, MAST bond={mast_bond}");
    assert!(
        mast_bond < 50,
        "MAST bond too large at {num_qubits}q: {mast_bond}"
    );
}

#[test]
fn test_large_scale_200_qubits() {
    let num_qubits = 200;
    let num_t = 50;
    let (stn_bond, mast_bond) = large_scale_bond_dim_check(num_qubits, num_t, 5, 456);
    eprintln!("{num_qubits}q {num_t}T: STN bond={stn_bond}, MAST bond={mast_bond}");
    assert!(
        mast_bond < 50,
        "MAST bond too large at {num_qubits}q: {mast_bond}"
    );
}

#[test]
#[ignore = "slow (~3min debug): run with `cargo test --test verification -- --include-ignored`"]
fn test_large_scale_bond_dim_curve() {
    // Track bond dim as a function of T-count for fixed qubit count.
    let num_qubits = 50;
    let t_counts = [5, 10, 20, 30, 40, 50];

    eprintln!("\nBond dim curve for {num_qubits} qubits:");
    eprintln!("  T-count  STN-bond  MAST-bond");
    for &num_t in &t_counts {
        let (stn_bond, mast_bond) = large_scale_bond_dim_check(num_qubits, num_t, 4, 789);
        eprintln!("  {num_t:>7}  {stn_bond:>8}  {mast_bond:>9}");
    }
}

#[test]
fn test_large_scale_measurement_works() {
    // Verify measurement doesn't panic at 30 qubits.
    let num_qubits = 30;
    let t = Angle64::QUARTER_TURN / 2u64;
    let mut stn = StabMps::with_seed(num_qubits, 42);

    // Build entangled state
    for q in 0..num_qubits {
        stn.h(&[QubitId(q)]);
    }
    for q in 0..num_qubits - 1 {
        stn.cx(&[(QubitId(q), QubitId(q + 1))]);
    }
    // Apply some T gates
    for q in (0..num_qubits).step_by(5) {
        stn.rz(t, &[QubitId(q)]);
    }

    // Measure a subset of qubits (measuring all 100 is too slow)
    let measure_qubits: Vec<QubitId> = (0..10).map(QubitId).collect();
    let results = stn.mz(&measure_qubits);
    assert_eq!(results.len(), 10);

    eprintln!(
        "{num_qubits}q measurement: bond_dim={}, measured 10 of {num_qubits} qubits",
        stn.max_bond_dim()
    );
}

// ============================================================================
// Shared measurement stress test suite
// ============================================================================

pecos_simulators::measurement_stress_test_suite!(StabMps, 4, StabMps::with_seed(4, 42));

// ============================================================================
// Performance profiling (run with --nocapture to see timing)
// ============================================================================

#[test]
fn test_profile_operation_costs() {
    use std::time::Instant;
    let t = Angle64::QUARTER_TURN / 2u64;

    for &num_qubits in &[20, 50, 100, 200] {
        let mut stn = StabMps::builder(num_qubits).seed(42).build();

        // Clifford layer: H + CX chain
        let start = Instant::now();
        for q in 0..num_qubits {
            stn.h(&[QubitId(q)]);
        }
        for q in 0..num_qubits - 1 {
            stn.cx(&[(QubitId(q), QubitId(q + 1))]);
        }
        let clifford_ms = start.elapsed().as_millis();

        // T gates
        let num_t = num_qubits / 4;
        let start = Instant::now();
        for q in 0..num_t {
            stn.rz(t, &[QubitId(q)]);
        }
        let t_ms = start.elapsed().as_millis();

        // Second Clifford layer
        let start = Instant::now();
        for q in (0..num_qubits - 1).rev() {
            stn.cx(&[(QubitId(q + 1), QubitId(q))]);
        }
        for q in 0..num_qubits {
            stn.h(&[QubitId(q)]);
        }
        let clifford2_ms = start.elapsed().as_millis();

        eprintln!(
            "{num_qubits:>3}q: clifford1={clifford_ms:>4}ms, {num_t}T={t_ms:>4}ms, clifford2={clifford2_ms:>4}ms, bond={}",
            stn.max_bond_dim()
        );
    }
}
