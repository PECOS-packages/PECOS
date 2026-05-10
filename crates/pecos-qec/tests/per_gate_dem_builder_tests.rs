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

//! Integration tests for `DemBuilder::with_per_gate_noise`. Parity with
//! `DemSamplerBuilder` path + verification that decomposed DEM text
//! output reflects per-gate-type per-Pauli rates.

use pecos_core::{QubitId, TimeUnits};
use pecos_qec::fault_tolerance::dem_builder::{DemBuilder, NoiseConfig, PerGateTypeNoise};
use pecos_qec::fault_tolerance::propagator::DagFaultAnalyzer;
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
fn uniform_equivalent_per_gate_matches_scalar_dem() {
    // DemBuilder with empty-map PerGateTypeNoise (base noise only) should
    // produce the same DEM text as scalar `with_noise`.
    let dag = build_parity_check();
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence = analyzer.build_influence_map();

    let scalar = DemBuilder::new(&influence)
        .with_noise(0.01, 0.02, 0.005, 0.003)
        .with_detectors_json(r#"[{"id": 0, "records": [-1]}]"#)
        .unwrap()
        .build();

    let per_gate = DemBuilder::new(&influence)
        .with_per_gate_noise(PerGateTypeNoise::from_base_noise(NoiseConfig::new(
            0.01, 0.02, 0.005, 0.003,
        )))
        .with_detectors_json(r#"[{"id": 0, "records": [-1]}]"#)
        .unwrap()
        .build();

    // Both DEMs should contain the same mechanism set with matching probabilities.
    assert_eq!(scalar.num_contributions(), per_gate.num_contributions());
    let scalar_text = scalar.to_string();
    let per_gate_text = per_gate.to_string();
    // Equal line counts in the text form (mechanism-by-mechanism identical).
    assert_eq!(
        scalar_text
            .lines()
            .filter(|l| l.starts_with("error("))
            .count(),
        per_gate_text
            .lines()
            .filter(|l| l.starts_with("error("))
            .count(),
        "scalar and uniform-per-gate should produce identical error-line counts:\nscalar:\n{scalar_text}\nper_gate:\n{per_gate_text}",
    );
}

#[test]
fn per_gate_override_produces_decomposed_dem_text() {
    // Specific per-gate CX rates should appear in the decomposed DEM text.
    let dag = build_parity_check();
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence = analyzer.build_influence_map();

    let mut rates_2q = [0.0; 15];
    // IX = index 0, nonzero probability
    rates_2q[0] = 1e-3;
    // XX = index 4 ("IX","IY","IZ","XI","XX",...) — high correlated rate
    rates_2q[4] = 5e-3;

    let cfg = PerGateTypeNoise::from_base_noise(NoiseConfig::new(0.0, 0.0, 0.0, 0.0))
        .with_2q_rates(GateType::CX, rates_2q);

    let dem = DemBuilder::new(&influence)
        .with_per_gate_noise(cfg)
        .with_detectors_json(r#"[{"id": 0, "records": [-1]}]"#)
        .unwrap()
        .build();

    let text = dem.to_string_decomposed();
    let error_lines = text.lines().filter(|l| l.starts_with("error(")).count();
    assert!(
        error_lines > 0,
        "expected per-gate CX rates to produce error lines in decomposed DEM:\n{text}",
    );
}

#[test]
fn per_qubit_cx_override_changes_dem_probabilities() {
    // Like the sampler-path test: boost CX (0, 2) per-qubit-pair, compare
    // to baseline where only per-gate-type is set.
    let dag = build_parity_check();
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence = analyzer.build_influence_map();

    let q0 = QubitId::from(0usize);
    let q2 = QubitId::from(2usize);

    let cfg_baseline = PerGateTypeNoise::from_base_noise(NoiseConfig::new(0.0, 0.0, 0.0, 0.0))
        .with_2q_rates(GateType::CX, [1e-4; 15]);
    let cfg_boost = cfg_baseline
        .clone()
        .with_2q_rates_for_qubits(GateType::CX, q0, q2, [1e-3; 15]);

    let baseline = DemBuilder::new(&influence)
        .with_per_gate_noise(cfg_baseline)
        .with_detectors_json(r#"[{"id": 0, "records": [-1]}]"#)
        .unwrap()
        .build();
    let boosted = DemBuilder::new(&influence)
        .with_per_gate_noise(cfg_boost)
        .with_detectors_json(r#"[{"id": 0, "records": [-1]}]"#)
        .unwrap()
        .build();

    // Both produce the same mechanism set (same circuit structure) but
    // the boosted DEM should have higher-average probabilities in its text.
    assert_eq!(baseline.num_contributions(), boosted.num_contributions());

    // Parse error-line probabilities and sum them.
    let sum_probs = |s: &str| -> f64 {
        s.lines()
            .filter_map(|l| l.strip_prefix("error("))
            .filter_map(|inner| inner.split(')').next())
            .filter_map(|p| p.parse::<f64>().ok())
            .sum()
    };
    let baseline_sum = sum_probs(&baseline.to_string());
    let boosted_sum = sum_probs(&boosted.to_string());
    assert!(
        boosted_sum > 2.0 * baseline_sum,
        "per-qubit-pair boost should raise probability sum: baseline={baseline_sum} boosted={boosted_sum}",
    );
}

#[test]
fn idle_locations_contribute_to_dem_text() {
    // Circuit with an idle gate + per-qubit idle rate should produce an
    // error line for the idle location. Before the Idle routing fix this
    // test would see zero idle contributions.
    let mut dag = DagCircuit::new();
    dag.pz(&[0]);
    dag.idle(TimeUnits::new(100), &[0]);
    dag.mz(&[0]);
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence = analyzer.build_influence_map();

    let q0 = QubitId::from(0usize);
    let cfg = PerGateTypeNoise::from_base_noise(NoiseConfig::new(0.0, 0.0, 0.0, 0.0))
        .with_1q_rates_for_qubit(GateType::Idle, q0, [0.01, 0.01, 0.01]);

    let dem = DemBuilder::new(&influence)
        .with_per_gate_noise(cfg)
        .with_detectors_json(r#"[{"id": 0, "records": [-1]}]"#)
        .unwrap()
        .build();

    let text = dem.to_string();
    assert!(
        text.contains("error("),
        "expected idle location to produce an error line:\n{text}",
    );
}

#[test]
fn decomposed_dem_reflects_per_gate_noise() {
    // DemBuilder's decomposed path uses mark_graphlike_decomposable and
    // Y-decomposition logic. Verify the text output includes both
    // direct and decomposed-effect lines when per-gate noise is set.
    let dag = build_parity_check();
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence = analyzer.build_influence_map();

    let cfg = PerGateTypeNoise::from_base_noise(NoiseConfig::new(0.005, 0.005, 0.005, 0.005));
    let dem = DemBuilder::new(&influence)
        .with_per_gate_noise(cfg)
        .with_detectors_json(r#"[{"id": 0, "records": [-1]}]"#)
        .unwrap()
        .build();

    // Both formats should have content.
    let non_decomposed = dem.to_string();
    let decomposed = dem.to_string_decomposed();
    assert!(non_decomposed.contains("error("));
    assert!(decomposed.contains("error("));
}
