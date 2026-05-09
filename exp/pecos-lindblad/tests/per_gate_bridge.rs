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

//! End-to-end bridge: pecos-lindblad synthesis -> PauliLindbladModel
//! adapter arrays -> pecos-qec PerGateTypeNoise -> DemStabSim.
//!
//! This is the **honest integration** the `Phase 5 scaffold` flagged as
//! missing. Closes the loop: device params -> per-gate per-Pauli rates
//! -> DEM mechanisms -> sampling -> shot batches.

use rand::SeedableRng;
use rand::rngs::SmallRng;

use pecos_lindblad::noise_models::{ad_pd_1q, ad_pd_2q};
use pecos_lindblad::{DEFAULT_N_STEPS, Gate, synthesize_identity_1q, synthesize_numerical};
use pecos_qec::dem_stab::DemStabSim;
use pecos_qec::fault_tolerance::dem_builder::{DetectorDef, NoiseConfig, PerGateTypeNoise};
use pecos_quantum::{DagCircuit, GateType};

#[test]
fn lindblad_rates_flow_through_per_gate_noise_spec() {
    // 1Q identity rates (for idle locations) and 2Q CX rates.
    let t1 = 100.0;
    let t2 = 80.0;
    let tau_1q = 0.5;
    let tau_cx = std::f64::consts::FRAC_PI_2;

    let pl_1q = synthesize_identity_1q(&Gate::identity(1, ad_pd_1q(t1, t2), tau_1q));
    let pl_cx = synthesize_numerical(
        &Gate::cx_theta(1.0, tau_cx, ad_pd_2q(t1, t1, t2, t2)),
        DEFAULT_N_STEPS,
    );

    // Bundle into a PerGateTypeNoise. The base noise model gives uncovered gate
    // types (e.g. PZ prep, MZ measure) a small uniform rate.
    let cfg = PerGateTypeNoise::from_base_noise(NoiseConfig::new(0.0, 0.0, 1e-3, 1e-3))
        .with_1q_rates(GateType::H, pl_1q.to_noise_array_1q())
        .with_2q_rates(GateType::CX, pl_cx.to_noise_array_2q());

    // Small syndrome-extraction circuit.
    let mut dag = DagCircuit::new();
    dag.pz(&[2]);
    dag.cx(&[(0, 2)]);
    dag.cx(&[(1, 2)]);
    dag.mz(&[2]);

    let sim = DemStabSim::builder()
        .circuit(dag)
        .per_gate_noise(cfg) // overrides .noise() when both set
        .detectors(vec![DetectorDef::new(0).with_records([-1])])
        .build()
        .expect("DemStabSim build");

    // Sample shots; sanity-check detector flips length matches detector count.
    let mut rng = SmallRng::seed_from_u64(42);
    let batch = sim.sample_batch(500, &mut rng);
    assert_eq!(batch.detector_flips.len(), 500);
    assert_eq!(batch.detector_flips[0].len(), 1);
}

#[test]
fn to_noise_array_1q_round_trip() {
    let pl = synthesize_identity_1q(&Gate::identity(1, ad_pd_1q(100.0, 80.0), 1.0));
    let arr = pl.to_noise_array_1q();
    // Paper: lambda_x = lambda_y, lambda_z different.
    assert!((arr[0] - arr[1]).abs() < 1e-14);
    assert!(arr[2] > 0.0);
    // Sum of rates matches total_rate.
    assert!((arr.iter().sum::<f64>() - pl.total_rate()).abs() < 1e-14);
}

#[test]
fn to_noise_array_2q_preserves_paper_structure() {
    // CX + AD+PD produces specific non-zero rates per paper eq 929-956.
    // Check the array matches the rate lookups.
    let pl = synthesize_numerical(
        &Gate::cx_theta(
            1.0,
            std::f64::consts::FRAC_PI_4,
            ad_pd_2q(100.0, 80.0, 100.0, 80.0),
        ),
        DEFAULT_N_STEPS,
    );
    let arr = pl.to_noise_array_2q();
    // Sum across the array matches total rate.
    assert!((arr.iter().sum::<f64>() - pl.total_rate()).abs() < 1e-14);
    // Several entries must be non-zero (IX, XI, IZ, ZI, ZZ and more).
    let non_zero = arr.iter().filter(|r| r.abs() > 1e-12).count();
    assert!(
        non_zero >= 6,
        "expected at least 6 non-zero rates in CX+AD+PD array, got {}",
        non_zero
    );
}

#[test]
#[should_panic(expected = "1-qubit")]
fn to_noise_array_1q_panics_on_2q_model() {
    let pl = synthesize_numerical(
        &Gate::cx_theta(
            1.0,
            std::f64::consts::FRAC_PI_4,
            ad_pd_2q(100.0, 100.0, 80.0, 80.0),
        ),
        DEFAULT_N_STEPS,
    );
    let _ = pl.to_noise_array_1q();
}
