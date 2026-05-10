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

//! Integration tests for per-qubit noise variation on top of
//! [`PerGateTypeNoise`]. Verifies:
//!
//! 1. per-qubit rates override per-gate-type defaults for matching qubits.
//! 2. qubits not in the per-qubit map use the per-gate-type default.
//! 3. lookup lookup methods (`rate_1q_on`, `rate_2q_on`) layer correctly.

use pecos_core::QubitId;
use pecos_qec::fault_tolerance::dem_builder::{DemSamplerBuilder, NoiseConfig, PerGateTypeNoise};
use pecos_qec::fault_tolerance::propagator::DagFaultAnalyzer;
use pecos_quantum::{DagCircuit, GateType};

fn assert_rate_eq(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-14,
        "expected rate {expected:.16e}, got {actual:.16e}"
    );
}

#[test]
fn per_qubit_override_takes_precedence_over_per_gate_type() {
    // Direct unit test of the lookup layering, independent of DemSampler.
    let q0 = QubitId::from(0usize);
    let q1 = QubitId::from(1usize);
    let cfg = PerGateTypeNoise::from_base_noise(NoiseConfig::uniform(0.1))
        .with_1q_rates(GateType::H, [0.01, 0.02, 0.03])
        .with_1q_rates_for_qubit(GateType::H, q0, [0.001, 0.002, 0.003]);

    // qubit 0 has per-qubit override
    assert_rate_eq(cfg.rate_1q_on(GateType::H, q0, 0), 0.001);
    assert_rate_eq(cfg.rate_1q_on(GateType::H, q0, 1), 0.002);
    assert_rate_eq(cfg.rate_1q_on(GateType::H, q0, 2), 0.003);

    // qubit 1 uses the per-gate-type default.
    assert_rate_eq(cfg.rate_1q_on(GateType::H, q1, 0), 0.01);
    assert_rate_eq(cfg.rate_1q_on(GateType::H, q1, 1), 0.02);

    // Unregistered gate on qubit 0 uses the per-gate-type default (not set),
    // then to uniform base.p1/3.
    let uniform_share = 0.1_f64 / 3.0;
    assert!((cfg.rate_1q_on(GateType::X, q0, 0) - uniform_share).abs() < 1e-14);
}

#[test]
fn per_qubit_2q_override_takes_precedence() {
    let q0 = QubitId::from(0usize);
    let q1 = QubitId::from(1usize);
    let q2 = QubitId::from(2usize);
    let q3 = QubitId::from(3usize);

    let mut per_pair = [0.0; 15];
    per_pair[0] = 1e-3; // IX
    let mut gate_default = [0.0; 15];
    gate_default[0] = 5e-4; // IX

    let cfg = PerGateTypeNoise::from_base_noise(NoiseConfig::uniform(0.0))
        .with_2q_rates(GateType::CX, gate_default)
        .with_2q_rates_for_qubits(GateType::CX, q0, q1, per_pair);

    // (q0, q1) uses the specific rates
    assert_rate_eq(cfg.rate_2q_on(GateType::CX, q0, q1, 0), 1e-3);
    // (q2, q3) uses the per-gate-type default.
    assert_rate_eq(cfg.rate_2q_on(GateType::CX, q2, q3, 0), 5e-4);
    // Different ordered pair (q1, q0): NOT the same as (q0, q1). Falls back.
    assert_rate_eq(cfg.rate_2q_on(GateType::CX, q1, q0, 0), 5e-4);
}

fn build_circuit_with_two_cxs() -> DagCircuit {
    // Two CX gates on different qubit pairs. The per-qubit override
    // should apply only to the first CX.
    let mut dag = DagCircuit::new();
    dag.pz(&[4]);
    dag.cx(&[(0, 4)]);
    dag.cx(&[(1, 4)]);
    dag.mz(&[4]);
    dag
}

#[test]
fn per_qubit_cx_rate_affects_mechanism_probabilities() {
    // Two CX locations touching different qubit pairs:
    //   CX on (0, 4): per-qubit rate (10x baseline)
    //   CX on (1, 4): per-gate-type default rate
    // Total aggregated error probability should reflect the override.
    let dag = build_circuit_with_two_cxs();
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence = analyzer.build_influence_map();

    let q0 = QubitId::from(0usize);
    let q4 = QubitId::from(4usize);

    let baseline_rates = [1e-4; 15];
    let boosted_rates = [1e-3; 15];

    // Control case: just the baseline.
    let baseline_cfg = PerGateTypeNoise::from_base_noise(NoiseConfig::uniform(0.0))
        .with_2q_rates(GateType::CX, baseline_rates);
    let baseline = DemSamplerBuilder::new(&influence)
        .with_per_gate_noise(baseline_cfg)
        .with_detectors_json(r#"[{"id": 0, "records": [-1]}]"#)
        .unwrap()
        .build()
        .unwrap();

    // Override case: boost the (0, 4) pair specifically.
    let override_cfg = PerGateTypeNoise::from_base_noise(NoiseConfig::uniform(0.0))
        .with_2q_rates(GateType::CX, baseline_rates)
        .with_2q_rates_for_qubits(GateType::CX, q0, q4, boosted_rates);
    let overridden = DemSamplerBuilder::new(&influence)
        .with_per_gate_noise(override_cfg)
        .with_detectors_json(r#"[{"id": 0, "records": [-1]}]"#)
        .unwrap()
        .build()
        .unwrap();

    assert_eq!(baseline.num_mechanisms(), overridden.num_mechanisms());
    // Override should substantially raise average error probability
    // (one of two CX contributions now 10x the baseline).
    let avg_base = baseline.average_error_probability();
    let avg_over = overridden.average_error_probability();
    assert!(
        avg_over > 2.0 * avg_base,
        "expected per-qubit override to raise avg error >>2x, got base={avg_base} over={avg_over}",
    );
}

#[test]
fn per_qubit_path_uses_per_gate_type_rates_without_overrides() {
    // A config with only per-gate-type rates (no per-qubit overrides)
    // should still produce mechanisms from the per-gate-type rates.
    let dag = build_circuit_with_two_cxs();
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence = analyzer.build_influence_map();

    let cfg = PerGateTypeNoise::from_base_noise(NoiseConfig::uniform(0.0))
        .with_2q_rates(GateType::CX, [1e-3; 15]);
    let sim = DemSamplerBuilder::new(&influence)
        .with_per_gate_noise(cfg)
        .with_detectors_json(r#"[{"id": 0, "records": [-1]}]"#)
        .unwrap()
        .build()
        .unwrap();

    assert!(sim.num_mechanisms() > 0);
    assert!(sim.average_error_probability() > 0.0);
}
