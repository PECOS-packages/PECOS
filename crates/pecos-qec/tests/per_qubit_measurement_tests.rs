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

//! Integration tests for per-qubit measurement and preparation rates on
//! [`PerGateTypeNoise`]. Mirrors the per-qubit gate tests but for MZ/PZ
//! locations.

use pecos_core::QubitId;
use pecos_qec::fault_tolerance::dem_builder::{DemSamplerBuilder, NoiseConfig, PerGateTypeNoise};
use pecos_qec::fault_tolerance::propagator::DagFaultAnalyzer;
use pecos_quantum::DagCircuit;

#[test]
fn per_qubit_measurement_override_takes_precedence() {
    // Unit test the lookup layering.
    let q0 = QubitId::from(0usize);
    let q1 = QubitId::from(1usize);
    let mut cfg = PerGateTypeNoise::from_base_noise(NoiseConfig::uniform(0.01));
    // Explicitly override q0, leave q1 on the base noise model.
    cfg = cfg.with_measurement_rate(q0, 0.1);
    assert!((cfg.measurement_rate_on(q0) - 0.1).abs() < 1e-14);
    assert!((cfg.measurement_rate_on(q1) - 0.01).abs() < 1e-14);
}

#[test]
fn per_qubit_init_override_takes_precedence() {
    let q0 = QubitId::from(0usize);
    let q1 = QubitId::from(1usize);
    let mut cfg = PerGateTypeNoise::from_base_noise(NoiseConfig::uniform(0.02));
    cfg = cfg.with_init_rate(q0, 0.2);
    assert!((cfg.init_rate_on(q0) - 0.2).abs() < 1e-14);
    assert!((cfg.init_rate_on(q1) - 0.02).abs() < 1e-14);
}

fn build_three_ancilla_circuit() -> DagCircuit {
    // Three prep + measure operations on three different qubits. Each
    // qubit is only touched once so per-qubit rates affect exactly one
    // mechanism each.
    let mut dag = DagCircuit::new();
    dag.pz(&[0]);
    dag.pz(&[1]);
    dag.pz(&[2]);
    dag.mz(&[0]);
    dag.mz(&[1]);
    dag.mz(&[2]);
    dag
}

#[test]
fn per_qubit_measurement_rate_raises_only_targeted_qubit() {
    // With per-qubit override on qubit 0, and scalar 0 for everyone else,
    // the total mechanism probability should reflect only the q0
    // contribution.
    let dag = build_three_ancilla_circuit();
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence = analyzer.build_influence_map();

    let q0 = QubitId::from(0usize);
    let cfg_only_q0 = PerGateTypeNoise::from_base_noise(NoiseConfig::new(0.0, 0.0, 0.0, 0.0))
        .with_measurement_rate(q0, 0.05);
    let sim_only_q0 = DemSamplerBuilder::new(&influence)
        .with_per_gate_noise(cfg_only_q0)
        .with_detectors_json(r#"[{"id": 0, "records": [-3]}, {"id": 1, "records": [-2]}, {"id": 2, "records": [-1]}]"#)
        .unwrap()
        .build().unwrap();

    // Baseline: all three qubits at the same rate.
    let cfg_uniform = PerGateTypeNoise::from_base_noise(NoiseConfig::new(0.0, 0.0, 0.05, 0.0));
    let sim_uniform = DemSamplerBuilder::new(&influence)
        .with_per_gate_noise(cfg_uniform)
        .with_detectors_json(r#"[{"id": 0, "records": [-3]}, {"id": 1, "records": [-2]}, {"id": 2, "records": [-1]}]"#)
        .unwrap()
        .build().unwrap();

    // Only q0 has meas noise in `sim_only_q0` => one mechanism.
    // Uniform has meas noise on all three => three mechanisms.
    // `average_error_probability` is per-mechanism, so it's the same
    // in both; what differs is the count.
    assert_eq!(
        sim_only_q0.num_mechanisms(),
        1,
        "expected exactly one mechanism for per-qubit q0-only override",
    );
    assert_eq!(
        sim_uniform.num_mechanisms(),
        3,
        "expected three mechanisms when all qubits share the rate",
    );
    // Per-mechanism probability should be the same 0.05 in both cases.
    let delta =
        (sim_only_q0.average_error_probability() - sim_uniform.average_error_probability()).abs();
    assert!(
        delta < 1e-12,
        "per-mech probabilities should match: {delta}"
    );
}

#[test]
fn per_qubit_init_rate_raises_only_targeted_qubit() {
    let dag = build_three_ancilla_circuit();
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence = analyzer.build_influence_map();

    let q1 = QubitId::from(1usize);
    let cfg_only_q1 = PerGateTypeNoise::from_base_noise(NoiseConfig::new(0.0, 0.0, 0.0, 0.0))
        .with_init_rate(q1, 0.05);
    let sim_only_q1 = DemSamplerBuilder::new(&influence)
        .with_per_gate_noise(cfg_only_q1)
        .with_detectors_json(r#"[{"id": 0, "records": [-3]}, {"id": 1, "records": [-2]}, {"id": 2, "records": [-1]}]"#)
        .unwrap()
        .build().unwrap();

    let cfg_uniform = PerGateTypeNoise::from_base_noise(NoiseConfig::new(0.0, 0.0, 0.0, 0.05));
    let sim_uniform = DemSamplerBuilder::new(&influence)
        .with_per_gate_noise(cfg_uniform)
        .with_detectors_json(r#"[{"id": 0, "records": [-3]}, {"id": 1, "records": [-2]}, {"id": 2, "records": [-1]}]"#)
        .unwrap()
        .build().unwrap();

    assert_eq!(
        sim_only_q1.num_mechanisms(),
        1,
        "expected exactly one mechanism for per-qubit q1-only init override",
    );
    assert_eq!(
        sim_uniform.num_mechanisms(),
        3,
        "expected three mechanisms when all qubits share the rate",
    );
}

#[test]
fn per_qubit_measurement_path_uses_base_rate_without_overrides() {
    // A PerGateTypeNoise without any per-qubit measurement rates should
    // use scalar p_meas exactly.
    let dag = build_three_ancilla_circuit();
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence = analyzer.build_influence_map();

    let cfg = PerGateTypeNoise::from_base_noise(NoiseConfig::uniform(0.05));
    let sim_per_gate = DemSamplerBuilder::new(&influence)
        .with_per_gate_noise(cfg)
        .with_detectors_json(r#"[{"id": 0, "records": [-3]}, {"id": 1, "records": [-2]}, {"id": 2, "records": [-1]}]"#)
        .unwrap()
        .build().unwrap();

    let sim_scalar = DemSamplerBuilder::new(&influence)
        .with_noise(0.05, 0.05, 0.05, 0.05)
        .with_detectors_json(r#"[{"id": 0, "records": [-3]}, {"id": 1, "records": [-2]}, {"id": 2, "records": [-1]}]"#)
        .unwrap()
        .build().unwrap();

    assert_eq!(sim_per_gate.num_mechanisms(), sim_scalar.num_mechanisms());
    let delta =
        (sim_per_gate.average_error_probability() - sim_scalar.average_error_probability()).abs();
    assert!(delta < 1e-12, "delta {delta} should be near zero");
}
