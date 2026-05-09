// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Scaffolded integration: synthesize a Pauli-Lindblad model from a
//! physical Lindbladian, collapse to scalar p1/p2 (lossy), and feed the
//! scalars to the existing uniform-depolarizing `DemStabSim`.
//!
//! **This is a scaffold, not the full bridge.** The paper's real payoff is
//! per-location per-Pauli rates, but `pecos-qec::fault_tolerance::dem_builder`
//! currently accepts only uniform-depolarizing `NoiseConfig { p1, p2,
//! p_meas, p_prep }` (4 scalar probabilities). A proper integration requires
//! generalizing `NoiseConfig` to accept a `PauliLindbladModel` per gate
//! type; that change is out of scope for the `pecos-lindblad` crate and is
//! tracked in `design/lindblad_sim_skeleton.md`.
//!
//! What this test proves *today*:
//! - Lindbladian + duration -> PauliLindbladModel works end-to-end.
//! - Summary helpers (`total_rate`, `rate_at_weight`) produce sensible numbers.
//! - Output scalars are in a range that `DemStabSim` will accept.
//! - DemStabSim runs with those scalars and returns shot batches.

use num_complex::Complex64;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use pecos_lindblad::matrix::{self, Matrix};
use pecos_lindblad::{
    DEFAULT_N_STEPS, Gate, Lindbladian, Pauli1, synthesize_identity_1q, synthesize_numerical,
};
use pecos_qec::dem_stab::DemStabSim;
use pecos_qec::fault_tolerance::dem_builder::{DetectorDef, NoiseConfig};
use pecos_quantum::DagCircuit;

fn ad_plus_pd_1q(beta_down: f64, beta_phi: f64) -> Lindbladian {
    let d = 2;
    let hamiltonian = matrix::zeros(d);
    let collapse: Vec<(Matrix, f64)> = vec![
        (matrix::sigma_minus(), beta_down),
        (matrix::pauli_1q(Pauli1::Z), beta_phi / 2.0),
    ];
    Lindbladian::new(d, hamiltonian, collapse)
}

fn ad_plus_pd_2q(beta_down: f64, beta_phi: f64) -> Lindbladian {
    let d = 4;
    let i2 = matrix::identity(2);
    let sm = matrix::sigma_minus();
    let z = matrix::pauli_1q(Pauli1::Z);
    let sm_l = matrix::kron(&sm, &i2, 2, 2);
    let sm_r = matrix::kron(&i2, &sm, 2, 2);
    let z_l = matrix::kron(&z, &i2, 2, 2);
    let z_r = matrix::kron(&i2, &z, 2, 2);
    let collapse: Vec<(Matrix, f64)> = vec![
        (sm_l, beta_down),
        (sm_r, beta_down),
        (z_l, beta_phi / 2.0),
        (z_r, beta_phi / 2.0),
    ];
    let zero_ham: Matrix = vec![Complex64::new(0.0, 0.0); d * d];
    Lindbladian::new(d, zero_ham, collapse)
}

#[test]
fn lindblad_derived_noise_config_feeds_dem_stab_sim() {
    // Step 1: physical noise parameters from a hypothetical device.
    let beta_down = 1e-4; // per time unit, e.g. inverse of T1
    let beta_phi = 2e-4; // dephasing
    let tau_1q = 40.0; // 1Q gate duration
    let omega_cx = 1.0;
    let theta = std::f64::consts::FRAC_PI_2; // full CNOT

    // Step 2: synthesize Pauli-Lindblad models for each gate family.
    let pl_1q = synthesize_identity_1q(&Gate::identity(
        1,
        ad_plus_pd_1q(beta_down, beta_phi),
        tau_1q,
    ));
    let pl_cx = synthesize_numerical(
        &Gate::cx_theta(omega_cx, theta, ad_plus_pd_2q(beta_down, beta_phi)),
        DEFAULT_N_STEPS,
    );

    // Sanity-check the summaries.
    let total_1q = pl_1q.total_rate();
    let total_2q = pl_cx.total_rate();
    assert!(
        total_1q > 0.0 && total_1q < 0.1,
        "1Q total rate out of range: {}",
        total_1q
    );
    assert!(
        total_2q > 0.0 && total_2q < 0.1,
        "2Q total rate out of range: {}",
        total_2q
    );
    // For the 1Q identity with AD+PD, only weight-1 rates should be non-zero.
    assert!((pl_1q.rate_at_weight(1) - total_1q).abs() < 1e-12);
    // For CX_theta, both weight-1 and weight-2 rates exist.
    assert!(pl_cx.rate_at_weight(1) > 0.0);
    assert!(pl_cx.rate_at_weight(2) > 0.0);

    // Step 3: lossy collapse to scalar p1, p2 for DemStabSim.
    // Caveat: DemStabSim treats p1 as uniform depolarizing (X/Y/Z equal).
    // For asymmetric AD+PD the numbers here are order-of-magnitude correct
    // but lose per-Pauli structure. This is the gap that motivates the
    // proper integration (generalize NoiseConfig to carry a PL model).
    let p1 = pl_1q.total_rate();
    let p2 = pl_cx.total_rate();
    let noise = NoiseConfig::new(p1, p2, 0.0, 0.0);

    // Step 4: build a tiny repetition-code-style circuit and sample.
    let mut dag = DagCircuit::new();
    dag.pz(&[2]);
    dag.cx(&[(0, 2)]);
    dag.cx(&[(1, 2)]);
    dag.mz(&[2]);

    let sim = DemStabSim::builder()
        .circuit(dag)
        .noise(noise)
        .detectors(vec![DetectorDef::new(0).with_records([-1])])
        .build()
        .expect("DemStabSim build");

    // Step 5: shots flow through.
    let mut rng = SmallRng::seed_from_u64(42);
    let batch = sim.sample_batch(500, &mut rng);
    assert_eq!(batch.detector_flips.len(), 500);
    assert_eq!(batch.detector_flips[0].len(), 1);
}

#[test]
fn lindblad_scalar_collapse_is_order_of_magnitude_sane() {
    // For 1Q identity under AD only (no PD), paper closed form:
    //   lambda_x = lambda_y = beta_down * tau / 4, lambda_z = 0
    //   total = beta_down * tau / 2.
    // The lossy scalar collapse should report the same total.
    let beta_down = 3e-4;
    let tau = 100.0;
    let pl = synthesize_identity_1q(&Gate::identity(1, ad_plus_pd_1q(beta_down, 0.0), tau));
    let expected = beta_down * tau / 2.0;
    assert!((pl.total_rate() - expected).abs() < 1e-12);
}
