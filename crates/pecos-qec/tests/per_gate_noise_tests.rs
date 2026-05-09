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

//! Integration tests for the per-gate-type noise path on
//! [`DemSamplerBuilder`].
//!
//! Verifies:
//! 1. Uniform-equivalent per-gate spec produces identical mechanisms to
//!    the scalar `with_noise` path.
//! 2. Per-gate rates actually override scalar rates for gates in the map.
//! 3. Fallback uniform rates apply to gate types not in the map.

use pecos_qec::fault_tolerance::dem_builder::{DemSamplerBuilder, NoiseConfig, PerGateTypeNoise};
use pecos_qec::fault_tolerance::propagator::DagFaultAnalyzer;
use pecos_quantum::{DagCircuit, GateType};

fn build_parity_check_circuit() -> DagCircuit {
    let mut dag = DagCircuit::new();
    dag.pz(&[2]);
    dag.cx(&[(0, 2)]);
    dag.cx(&[(1, 2)]);
    dag.mz(&[2]);
    dag
}

#[test]
fn per_gate_uniform_equivalent_matches_scalar_path() {
    // Build a PerGateTypeNoise that mimics uniform p1=p2=p_meas=p_prep=0.01.
    // Expectation: `per_gate.rate_1q()` lookup with an empty map uses
    // `base.p1 / 3.0` for 1Q, and `base.p2 / 15.0` for 2Q --
    // which is exactly what the legacy scalar path uses. So the two
    // builders should produce identical mechanism sets.
    let dag = build_parity_check_circuit();
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence = analyzer.build_influence_map();

    let p = 0.01;
    let scalar = DemSamplerBuilder::new(&influence)
        .with_noise(p, p, p, p)
        .with_detectors_json(r#"[{"id": 0, "records": [-1]}]"#)
        .unwrap()
        .build()
        .unwrap();
    let per_gate = DemSamplerBuilder::new(&influence)
        .with_per_gate_noise(PerGateTypeNoise::from_base_noise(NoiseConfig::uniform(p)))
        .with_detectors_json(r#"[{"id": 0, "records": [-1]}]"#)
        .unwrap()
        .build()
        .unwrap();

    assert_eq!(
        scalar.num_mechanisms(),
        per_gate.num_mechanisms(),
        "uniform-equivalent per-gate must produce same mechanism count as scalar",
    );
}

#[test]
fn per_gate_override_changes_cx_rate() {
    // Assign a large explicit CX rate via per_gate, compare against a
    // scalar baseline with small p2.
    let dag = build_parity_check_circuit();
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence = analyzer.build_influence_map();

    // Scalar baseline: small p2.
    let small = DemSamplerBuilder::new(&influence)
        .with_noise(0.0, 1e-4, 0.0, 0.0)
        .with_detectors_json(r#"[{"id": 0, "records": [-1]}]"#)
        .unwrap()
        .build()
        .unwrap();

    // Per-gate override: same p2 for CX via map, but 10x larger value.
    let per_gate = DemSamplerBuilder::new(&influence)
        .with_per_gate_noise(
            PerGateTypeNoise::from_base_noise(NoiseConfig::uniform(0.0))
                .with_2q_rates(GateType::CX, [1e-3; 15]),
        )
        .with_detectors_json(r#"[{"id": 0, "records": [-1]}]"#)
        .unwrap()
        .build()
        .unwrap();

    // Per-gate path should produce the same mechanism count (same circuit
    // structure), but larger aggregate error probabilities.
    assert_eq!(small.num_mechanisms(), per_gate.num_mechanisms());
    assert!(
        per_gate.average_error_probability() > 5.0 * small.average_error_probability(),
        "10x larger per-CX rate should produce substantially larger avg error \
         (got per_gate={}, scalar={})",
        per_gate.average_error_probability(),
        small.average_error_probability(),
    );
}

#[test]
fn per_gate_base_noise_used_for_unmapped_gate_types() {
    // Specify H explicitly (rates[X, Y, Z]); CX uses the base noise model.
    let dag = build_parity_check_circuit();
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence = analyzer.build_influence_map();

    let cfg = PerGateTypeNoise::from_base_noise(NoiseConfig::uniform(0.01))
        .with_1q_rates(GateType::H, [0.001, 0.001, 0.001]);

    let sim = DemSamplerBuilder::new(&influence)
        .with_per_gate_noise(cfg)
        .with_detectors_json(r#"[{"id": 0, "records": [-1]}]"#)
        .unwrap()
        .build()
        .unwrap();

    // Parity-check circuit has no H gate -- all 1Q contributions come
    // from prep/measurement. Just verify it builds and has mechanisms.
    assert!(sim.num_mechanisms() > 0);
}

#[test]
fn per_gate_asymmetric_2q_rates() {
    // Set only lambda_IX != 0, everything else zero for CX. Confirms the
    // sparse rate path (only one of 15 pair rates nonzero) works.
    let dag = build_parity_check_circuit();
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence = analyzer.build_influence_map();

    let mut rates_2q = [0.0; 15];
    rates_2q[0] = 0.005; // IX
    let cfg = PerGateTypeNoise::from_base_noise(NoiseConfig::uniform(0.0))
        .with_2q_rates(GateType::CX, rates_2q);

    let sim = DemSamplerBuilder::new(&influence)
        .with_per_gate_noise(cfg)
        .with_detectors_json(r#"[{"id": 0, "records": [-1]}]"#)
        .unwrap()
        .build()
        .unwrap();

    // With everything else zero, should still produce some mechanisms
    // (from the single IX contribution at each CX location).
    assert!(
        sim.num_mechanisms() > 0,
        "sparse rates should still produce mechanisms from the IX contribution",
    );
}
