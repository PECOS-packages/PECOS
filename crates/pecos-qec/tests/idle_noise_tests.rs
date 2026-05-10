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

//! Integration tests for idle-gate noise. `GateType::Idle` is a no-op unless
//! noise is explicitly attached to idle locations via dedicated idle noise or
//! per-gate idle rates.

use pecos_core::{QubitId, TimeUnits};
use pecos_qec::fault_tolerance::dem_builder::{
    DemBuilder, DemSamplerBuilder, NoiseConfig, PerGateTypeNoise,
};
use pecos_qec::fault_tolerance::propagator::DagFaultAnalyzer;
use pecos_quantum::{DagCircuit, GateType};

fn build_idle_then_measure(num_idles: usize) -> DagCircuit {
    // Prep N qubits, idle each once, measure each. Very simple fixture
    // to isolate idle-gate contributions.
    let mut dag = DagCircuit::new();
    for q in 0..num_idles {
        dag.pz(&[q]);
    }
    for q in 0..num_idles {
        dag.idle(TimeUnits::new(100), &[q]);
    }
    for q in 0..num_idles {
        dag.mz(&[q]);
    }
    dag
}

#[test]
fn idle_locations_contribute_mechanisms_when_rates_set() {
    let dag = build_idle_then_measure(2);
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence = analyzer.build_influence_map();

    // No noise elsewhere; idle rates set only on qubit 0.
    let q0 = QubitId::from(0usize);
    let cfg = PerGateTypeNoise::from_base_noise(NoiseConfig::new(0.0, 0.0, 0.0, 0.0))
        .with_1q_rates_for_qubit(GateType::Idle, q0, [0.001, 0.001, 0.001]);
    let sim = DemSamplerBuilder::new(&influence)
        .with_per_gate_noise(cfg)
        .with_detectors_json(r#"[{"id": 0, "records": [-2]}, {"id": 1, "records": [-1]}]"#)
        .unwrap()
        .build()
        .unwrap();

    // Exactly one location contributes noise (idle on q0). That location
    // produces X, Y, Z mechanisms, of which X+Y generally both flip the
    // Z-basis measurement, but aggregation collapses them. Expect at
    // least one mechanism -> we used to get zero silently.
    assert!(
        sim.num_mechanisms() > 0,
        "idle on q0 should produce at least one mechanism",
    );
}

#[test]
fn idle_rates_absent_means_no_idle_contribution() {
    // Config provides no Idle rates and uses zero base noise. DEM should have
    // zero mechanisms: prep/measure are 0 and idle is a no-op by default.
    let dag = build_idle_then_measure(3);
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence = analyzer.build_influence_map();

    let cfg = PerGateTypeNoise::from_base_noise(NoiseConfig::new(0.0, 0.0, 0.0, 0.0));
    let sim = DemSamplerBuilder::new(&influence)
        .with_per_gate_noise(cfg)
        .with_detectors_json(r#"[{"id": 0, "records": [-3]}, {"id": 1, "records": [-2]}, {"id": 2, "records": [-1]}]"#)
        .unwrap()
        .build().unwrap();
    assert_eq!(sim.num_mechanisms(), 0);
}

#[test]
fn per_gate_base_p1_does_not_attach_to_idle() {
    let dag = build_idle_then_measure(2);
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence = analyzer.build_influence_map();

    let cfg = PerGateTypeNoise::from_base_noise(NoiseConfig::new(0.01, 0.0, 0.0, 0.0));
    let sim = DemSamplerBuilder::new(&influence)
        .with_per_gate_noise(cfg)
        .with_detectors_json(r#"[{"id": 0, "records": [-2]}, {"id": 1, "records": [-1]}]"#)
        .unwrap()
        .build()
        .unwrap();

    assert_eq!(sim.num_mechanisms(), 0);
}

#[test]
fn per_gate_base_idle_noise_attaches_to_idle() {
    let dag = build_idle_then_measure(2);
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence = analyzer.build_influence_map();

    let cfg = PerGateTypeNoise::from_base_noise(NoiseConfig::with_idle(0.01, 0.0, 0.0, 0.0, 0.002));
    let sim = DemSamplerBuilder::new(&influence)
        .with_per_gate_noise(cfg)
        .with_detectors_json(r#"[{"id": 0, "records": [-2]}, {"id": 1, "records": [-1]}]"#)
        .unwrap()
        .build()
        .unwrap();

    assert!(
        sim.num_mechanisms() > 0,
        "base p_idle in per-gate config should attach to idle locations",
    );
}

#[test]
fn idle_noise_respects_per_qubit_override() {
    // q0 gets boosted idle rate; q1 gets zero. Expect exactly one
    // mechanism from q0's idle.
    let dag = build_idle_then_measure(2);
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence = analyzer.build_influence_map();

    let q0 = QubitId::from(0usize);
    let cfg = PerGateTypeNoise::from_base_noise(NoiseConfig::new(0.0, 0.0, 0.0, 0.0))
        .with_1q_rates(GateType::Idle, [0.0, 0.0, 0.0])
        .with_1q_rates_for_qubit(GateType::Idle, q0, [0.01, 0.01, 0.01]);
    let sim = DemSamplerBuilder::new(&influence)
        .with_per_gate_noise(cfg)
        .with_detectors_json(r#"[{"id": 0, "records": [-2]}, {"id": 1, "records": [-1]}]"#)
        .unwrap()
        .build()
        .unwrap();

    assert!(sim.num_mechanisms() > 0);
    // q1's idle at zero rates should not contribute -- only q0's.
    assert!(sim.max_error_probability() >= 0.01 * 0.5);
}

#[test]
fn idle_with_scalar_p1_is_noop() {
    // Ordinary p1 gate noise should not attach to Idle. Idle is a no-op unless
    // idle noise is explicitly configured.
    let dag = build_idle_then_measure(2);
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence = analyzer.build_influence_map();

    let sim = DemSamplerBuilder::new(&influence)
        .with_noise(0.01, 0.0, 0.0, 0.0)
        .with_detectors_json(r#"[{"id": 0, "records": [-2]}, {"id": 1, "records": [-1]}]"#)
        .unwrap()
        .build()
        .unwrap();

    assert_eq!(sim.num_mechanisms(), 0);
}

#[test]
fn explicit_uniform_idle_noise_is_noisy() {
    let dag = build_idle_then_measure(2);
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence = analyzer.build_influence_map();

    let sim = DemSamplerBuilder::new(&influence)
        .with_noise_config(NoiseConfig::with_idle(0.01, 0.0, 0.0, 0.0, 0.002))
        .with_detectors_json(r#"[{"id": 0, "records": [-2]}, {"id": 1, "records": [-1]}]"#)
        .unwrap()
        .build()
        .unwrap();

    assert!(
        sim.num_mechanisms() > 0,
        "explicit p_idle should produce idle-location mechanisms",
    );
}

#[test]
fn dem_builder_scalar_p1_does_not_attach_to_idle() {
    let dag = build_idle_then_measure(1);
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence = analyzer.build_influence_map();

    let dem = DemBuilder::new(&influence)
        .with_noise(0.01, 0.0, 0.0, 0.0)
        .with_detectors_json(r#"[{"id": 0, "records": [-1]}]"#)
        .unwrap()
        .build();

    assert_eq!(dem.num_contributions(), 0);
}

#[test]
fn dem_builder_explicit_idle_noise_is_noisy() {
    let dag = build_idle_then_measure(1);
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence = analyzer.build_influence_map();

    let dem = DemBuilder::new(&influence)
        .with_noise_config(NoiseConfig::with_idle(0.01, 0.0, 0.0, 0.0, 0.002))
        .with_detectors_json(r#"[{"id": 0, "records": [-1]}]"#)
        .unwrap()
        .build();

    assert!(
        dem.num_contributions() > 0,
        "explicit p_idle should produce idle-location DEM contributions",
    );
}
