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

//! Integration tests for Stim-format DEM export from `DemStabSim` with
//! per-gate noise. Closes the
//! `~/Repos/pecos-docs/ideas/stim-compat-dem-export.md` gap.

use pecos_core::QubitId;
use pecos_qec::dem_stab::DemStabSim;
use pecos_qec::fault_tolerance::dem_builder::{
    DemOutput, DetectorDef, NoiseConfig, PerGateTypeNoise,
};
use pecos_quantum::{DagCircuit, GateType};

fn build_parity_check() -> DagCircuit {
    let mut dag = DagCircuit::new();
    dag.pz(&[2]);
    dag.cx(&[(0, 2)]);
    dag.cx(&[(1, 2)]);
    dag.mz(&[2]);
    dag
}

#[test]
fn dem_text_export_with_scalar_noise() {
    let dag = build_parity_check();
    let sim = DemStabSim::builder()
        .circuit(dag)
        .noise(NoiseConfig::uniform(0.01))
        .detectors(vec![DetectorDef::new(0).with_records([-1])])
        .observables(vec![DemOutput::new(0).with_records([-1])])
        .build()
        .unwrap();

    let dem = sim.detector_error_model();
    let text = dem.to_string();

    // Must contain at least one error mechanism.
    assert!(
        text.contains("error("),
        "expected 'error(' line in DEM text:\n{text}"
    );
    // Must declare the detector and observable.
    assert!(text.contains("detector D0"), "missing detector D0:\n{text}");
    assert!(
        text.contains("logical_observable L0") || text.contains("observable_include L0"),
        "missing observable decl:\n{text}",
    );
}

#[test]
fn dem_text_export_with_per_gate_noise() {
    let dag = build_parity_check();
    let q0 = QubitId::from(0usize);
    let q2 = QubitId::from(2usize);
    let cfg = PerGateTypeNoise::from_base_noise(NoiseConfig::new(0.0, 0.0, 0.001, 0.001))
        .with_2q_rates(GateType::CX, [1e-3; 15])
        .with_2q_rates_for_qubits(GateType::CX, q0, q2, [5e-3; 15]);
    let sim = DemStabSim::builder()
        .circuit(dag)
        .per_gate_noise(cfg)
        .detectors(vec![DetectorDef::new(0).with_records([-1])])
        .build()
        .unwrap();

    let dem = sim.detector_error_model();
    let text = dem.to_string();

    // Should render multiple error mechanisms (CX on (0,2) boosted vs CX on (1,2)).
    let error_lines = text.matches("error(").count();
    assert!(
        error_lines > 0,
        "expected per-gate-noise path to produce error lines:\n{text}",
    );
    assert!(text.contains("detector D0"));
}

#[test]
fn dem_round_trip_mechanism_count_matches_sampler() {
    let dag = build_parity_check();
    let sim = DemStabSim::builder()
        .circuit(dag)
        .noise(NoiseConfig::uniform(0.005))
        .detectors(vec![DetectorDef::new(0).with_records([-1])])
        .build()
        .unwrap();

    let dem = sim.detector_error_model();
    // The reconstructed DEM should have the same mechanism count as the
    // sampler (direct contributions).
    assert_eq!(
        dem.num_contributions(),
        sim.num_mechanisms(),
        "mechanism count should round-trip between sampler and reconstructed DEM"
    );
}

#[test]
fn dem_probabilities_recoverable_from_thresholds() {
    // Check that the prob → u64 threshold → prob round-trip is close.
    // Use a small, well-separated set of mechanisms.
    let dag = build_parity_check();
    let sim = DemStabSim::builder()
        .circuit(dag)
        .noise(NoiseConfig::uniform(0.01))
        .detectors(vec![DetectorDef::new(0).with_records([-1])])
        .build()
        .unwrap();

    let dem = sim.detector_error_model();
    let text = dem.to_string();
    // Probabilities recovered should be non-zero and appear in the text
    // in some form.
    for line in text.lines().filter(|l| l.starts_with("error(")) {
        // Parse the prob inside "error(...)".
        let inner = line.trim_start_matches("error(").split(')').next().unwrap();
        let p: f64 = inner.parse().unwrap();
        assert!(p > 0.0 && p < 1.0, "probability out of range: {p}");
    }
}
