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

use pecos_core::pauli::{X, Y, Z};
use pecos_core::{QubitId, TimeUnits};
use pecos_qec::fault_tolerance::dem_builder::{
    DemBuilder, DemSamplerBuilder, DetectorErrorModel, FaultMechanism, IdleNoiseFamily, MemBuilder,
    NoiseConfig, PauliProbs, PerGateTypeNoise, SamplingEngine, combine_probabilities,
};
use pecos_qec::fault_tolerance::propagator::{
    DagFaultAnalyzer, DagFaultInfluenceMap, DagSpacetimeLocation, DetectorId, MeasurementId, Pauli,
};
use pecos_quantum::{DagCircuit, GateType};
use std::collections::BTreeMap;

fn idle_family(
    rate: f64,
    weights: impl IntoIterator<Item = (&'static str, f64)>,
) -> IdleNoiseFamily {
    IdleNoiseFamily::new(
        rate,
        weights
            .into_iter()
            .map(|(axis, weight)| (axis.to_string(), weight))
            .collect(),
    )
}

fn z_idle_family(rate: f64) -> IdleNoiseFamily {
    idle_family(rate, [("Z", 1.0)])
}

fn axis_rate_family(px: f64, py: f64, pz: f64) -> IdleNoiseFamily {
    idle_family(1.0, [("X", px), ("Y", py), ("Z", pz)])
}

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

fn build_nanosecond_idle_x_basis_measure() -> DagCircuit {
    let mut dag = DagCircuit::new();
    dag.pz(&[0]);
    dag.h(&[0]);
    dag.idle(TimeUnits::new(20), &[0]);
    dag.h(&[0]);
    dag.mz(&[0]);
    dag
}

fn build_unit_idle_with_pauli_tracking() -> DagCircuit {
    let mut dag = DagCircuit::new();
    dag.pz(&[0]);
    dag.idle(TimeUnits::new(1), &[0]);
    dag.tracked_pauli_labeled("tracked_x", X(0));
    dag.tracked_pauli_labeled("tracked_y", Y(0));
    dag.tracked_pauli_labeled("tracked_z", Z(0));
    dag.mz(&[0]);
    dag
}

fn build_unit_idle_tracking_x() -> DagCircuit {
    let mut dag = DagCircuit::new();
    dag.pz(&[0]);
    dag.idle(TimeUnits::new(1), &[0]);
    dag.tracked_pauli_labeled("tracked_x", X(0));
    dag.mz(&[0]);
    dag
}

fn build_unit_idle_tracking_y() -> DagCircuit {
    let mut dag = DagCircuit::new();
    dag.pz(&[0]);
    dag.idle(TimeUnits::new(1), &[0]);
    dag.tracked_pauli_labeled("tracked_y", Y(0));
    dag.mz(&[0]);
    dag
}

fn build_tracked_idle_dem(noise: NoiseConfig) -> Result<DetectorErrorModel, String> {
    let dag = build_unit_idle_with_pauli_tracking();
    DemBuilder::try_from_circuit_with_noise_config(&dag, noise).map_err(|error| error.to_string())
}

fn synthetic_idle_influence(
    x_signature: &[u32],
    y_signature: &[u32],
    z_signature: &[u32],
) -> DagFaultInfluenceMap {
    let mut influence = DagFaultInfluenceMap::with_capacity(1);
    influence.locations.push(DagSpacetimeLocation {
        node: 0,
        qubits: vec![QubitId::from(0usize)],
        before: false,
        gate_type: GateType::Idle,
        idle_duration: 1.0,
    });
    influence
        .influences
        .detectors_x
        .extend(x_signature.iter().copied());
    influence
        .influences
        .detectors_y
        .extend(y_signature.iter().copied());
    influence
        .influences
        .detectors_z
        .extend(z_signature.iter().copied());
    influence.influences.finish_location();
    influence.measurements = vec![(0, 0, 0), (1, 0, 0)];
    influence
}

fn build_synthetic_idle_dem(
    influence: &DagFaultInfluenceMap,
    noise: NoiseConfig,
) -> Result<DetectorErrorModel, String> {
    DemBuilder::new(influence)
        .with_noise_config(noise)
        .with_detectors_json(r#"[{"id": 0, "records": [-2]}, {"id": 1, "records": [-1]}]"#)
        .map_err(|error| error.to_string())?
        .try_build()
        .map_err(|error| error.to_string())
}

fn compose_xyz_mechanisms(mechanisms: PauliProbs) -> [f64; 4] {
    let PauliProbs { px, py, pz } = mechanisms;
    [
        (1.0 - px) * (1.0 - py) * (1.0 - pz) + px * py * pz,
        px * (1.0 - py) * (1.0 - pz) + (1.0 - px) * py * pz,
        (1.0 - px) * py * (1.0 - pz) + px * (1.0 - py) * pz,
        (1.0 - px) * (1.0 - py) * pz + px * py * (1.0 - pz),
    ]
}

fn compose_pauli_channels(left: [f64; 4], right: [f64; 4]) -> [f64; 4] {
    let [li, lx, ly, lz] = left;
    let [ri, rx, ry, rz] = right;
    [
        li * ri + lx * rx + ly * ry + lz * rz,
        li * rx + lx * ri + ly * rz + lz * ry,
        li * ry + ly * ri + lx * rz + lz * rx,
        li * rz + lz * ri + lx * ry + ly * rx,
    ]
}

fn idle_signature_contributions(dem: &DetectorErrorModel) -> Vec<(FaultMechanism, f64)> {
    let mut mechanisms = Vec::new();
    for record in dem.contribution_render_records() {
        let contribution = record.contribution;
        assert_eq!(contribution.source_gate_types.as_slice(), [GateType::Idle]);
        assert!(contribution.paulis.is_empty());
        mechanisms.push((contribution.effect, contribution.probability));
    }
    mechanisms
}

fn raw_idle_signature(
    influence: &DagFaultInfluenceMap,
    loc_idx: usize,
    pauli: Pauli,
) -> FaultMechanism {
    FaultMechanism::from_unsorted_with_tracked_paulis(
        influence
            .get_detector_indices(loc_idx, pauli.as_u8())
            .iter()
            .copied(),
        influence
            .get_observable_indices(loc_idx, pauli.as_u8())
            .iter()
            .copied(),
        influence
            .get_tracked_pauli_indices(loc_idx, pauli.as_u8())
            .iter()
            .copied(),
    )
}

fn idle_location(influence: &DagFaultInfluenceMap) -> usize {
    influence
        .locations
        .iter()
        .position(|location| location.gate_type == GateType::Idle && !location.before)
        .expect("after-idle fault location")
}

fn independent_signature_distribution(
    mechanisms: &[(FaultMechanism, f64)],
) -> BTreeMap<FaultMechanism, f64> {
    let mut distribution = BTreeMap::from([(FaultMechanism::new(), 1.0)]);
    for (mechanism, probability) in mechanisms {
        let mut next = BTreeMap::new();
        for (effect, mass) in distribution {
            *next.entry(effect.clone()).or_insert(0.0) += mass * (1.0 - probability);
            *next.entry(effect.xor(mechanism)).or_insert(0.0) += mass * probability;
        }
        distribution = next;
    }
    distribution
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
fn nanosecond_timeunit_idle_duration_is_preserved_in_fault_locations() {
    let dag = build_nanosecond_idle_x_basis_measure();
    let influence = DagFaultAnalyzer::new(&dag).build_influence_map();

    let idle = influence
        .locations
        .iter()
        .find(|loc| loc.gate_type == GateType::Idle)
        .expect("idle location");

    assert!((idle.idle_duration - 20.0).abs() < f64::EPSILON);
}

#[test]
fn linear_memory_z_noise_uses_idle_duration_in_dem() {
    let dag = build_nanosecond_idle_x_basis_measure();
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence = analyzer.build_influence_map();

    let dem = DemBuilder::new(&influence)
        .with_noise_config(
            NoiseConfig::new(0.0, 0.0, 0.0, 0.0).set_idle_linear(z_idle_family(1.0e-3)),
        )
        .with_detectors_json(r#"[{"id": 0, "records": [-1]}]"#)
        .unwrap()
        .build();

    assert!(
        dem.num_contributions() > 0,
        "linear Z-memory noise on an idle should produce DEM contributions",
    );
}

#[test]
fn idle_memory_pauli_probabilities_match_linear_and_quadratic_model() {
    let linear = NoiseConfig::new(0.0, 0.0, 0.0, 0.0)
        .set_idle_linear(z_idle_family(1.0e-3))
        .idle_pauli_probs(20.0);
    assert_eq!(linear.px.to_bits(), 0.0_f64.to_bits());
    assert_eq!(linear.py.to_bits(), 0.0_f64.to_bits());
    assert!((linear.pz - 0.02).abs() < 1e-15);

    let quadratic = NoiseConfig::new(0.0, 0.0, 0.0, 0.0)
        .set_idle_quadratic(z_idle_family(0.1))
        .idle_pauli_probs(2.0);
    assert_eq!(quadratic.px.to_bits(), 0.0_f64.to_bits());
    assert_eq!(quadratic.py.to_bits(), 0.0_f64.to_bits());
    assert!((quadratic.pz - 0.4).abs() < 1e-15);

    let pauli = NoiseConfig::new(0.0, 0.0, 0.0, 0.0)
        .set_idle_linear(axis_rate_family(1.0e-3, 2.0e-3, 3.0e-3))
        .set_idle_quadratic(axis_rate_family(1.0e-4, 2.0e-4, 3.0e-4))
        .idle_memory_pauli_probs(10.0);
    let expected = compose_pauli_channels(
        [0.94, 0.01, 0.02, 0.03],
        compose_xyz_mechanisms(PauliProbs {
            px: 0.01,
            py: 0.02,
            pz: 0.03,
        }),
    );
    assert!((pauli.px - expected[1]).abs() < 1e-15);
    assert!((pauli.py - expected[2]).abs() < 1e-15);
    assert!((pauli.pz - expected[3]).abs() < 1e-15);
}

#[test]
fn idle_memory_pauli_probabilities_support_quadratic_sine_model() {
    let z_sine = NoiseConfig::new(0.0, 0.0, 0.0, 0.0)
        .set_idle_quadratic_sine(z_idle_family(0.2))
        .idle_memory_pauli_probs(3.0);
    assert_eq!(z_sine.px.to_bits(), 0.0_f64.to_bits());
    assert_eq!(z_sine.py.to_bits(), 0.0_f64.to_bits());
    assert!((z_sine.pz - 0.6_f64.sin().powi(2)).abs() < 1e-15);

    let pauli_sine = NoiseConfig::new(0.0, 0.0, 0.0, 0.0)
        .set_idle_quadratic_sine(axis_rate_family(0.1, 0.2, 0.3))
        .idle_memory_pauli_probs(2.0);
    let expected = compose_xyz_mechanisms(PauliProbs {
        px: 0.2_f64.sin().powi(2),
        py: 0.4_f64.sin().powi(2),
        pz: 0.6_f64.sin().powi(2),
    });
    assert!((pauli_sine.px - expected[1]).abs() < 1e-15);
    assert!((pauli_sine.py - expected[2]).abs() < 1e-15);
    assert!((pauli_sine.pz - expected[3]).abs() < 1e-15);
}

#[test]
fn unset_idle_weight_map_is_symmetric() {
    let influence = synthetic_idle_influence(&[0], &[0, 1], &[1]);
    let implicit = NoiseConfig::new(0.0, 0.0, 0.0, 0.0)
        .set_idle_linear(IdleNoiseFamily::new(0.005, BTreeMap::new()));
    let explicit = NoiseConfig::new(0.0, 0.0, 0.0, 0.0)
        .set_idle_linear(idle_family(0.005, [("X", 1.0), ("Y", 1.0), ("Z", 1.0)]));

    assert_eq!(
        build_synthetic_idle_dem(&influence, implicit)
            .expect("implicit symmetric family")
            .to_string(),
        build_synthetic_idle_dem(&influence, explicit)
            .expect("explicit symmetric family")
            .to_string(),
    );
}

#[test]
fn single_axis_idle_map_matches_pinned_z_linear_dem() {
    let influence = synthetic_idle_influence(&[0], &[0, 1], &[1]);
    let noise =
        NoiseConfig::new(0.0, 0.0, 0.0, 0.0).set_idle_linear(idle_family(0.005, [("Z", 1.0)]));

    assert_eq!(
        build_synthetic_idle_dem(&influence, noise)
            .expect("single-axis Z family")
            .to_string(),
        "detector D0\ndetector D1\nerror(0.005) D1",
    );
}

#[test]
fn zero_idle_family_rate_is_inactive_regardless_of_weights() {
    let influence = synthetic_idle_influence(&[0], &[0, 1], &[1]);
    let omitted = NoiseConfig::new(0.0, 0.0, 0.0, 0.0);
    let configured = NoiseConfig::new(0.0, 0.0, 0.0, 0.0).set_idle_linear(IdleNoiseFamily::new(
        0.0,
        BTreeMap::from([("invalid".to_string(), f64::NAN), ("X".to_string(), -1.0)]),
    ));

    assert!(!configured.uses_dedicated_idle_noise());
    let probabilities = configured
        .try_idle_memory_pauli_probs(1.0)
        .expect("zero-rate family bypasses its map");
    assert_eq!(probabilities.px.to_bits(), 0.0_f64.to_bits());
    assert_eq!(probabilities.py.to_bits(), 0.0_f64.to_bits());
    assert_eq!(probabilities.pz.to_bits(), 0.0_f64.to_bits());
    assert_eq!(
        build_synthetic_idle_dem(&influence, configured)
            .expect("zero-rate family is inactive")
            .to_string(),
        build_synthetic_idle_dem(&influence, omitted)
            .expect("omitted family")
            .to_string(),
    );
}

#[test]
fn idle_weight_map_insertion_order_does_not_affect_dem_text() {
    let influence = synthetic_idle_influence(&[0], &[0, 1], &[1]);
    let xyz = idle_family(0.005, [("X", 0.4), ("Y", 0.6), ("Z", 1.0)]);
    let zyx = idle_family(0.005, [("Z", 1.0), ("Y", 0.6), ("X", 0.4)]);

    assert!(std::any::type_name_of_val(&xyz.weights).contains("BTreeMap"));

    assert_eq!(
        build_synthetic_idle_dem(
            &influence,
            NoiseConfig::new(0.0, 0.0, 0.0, 0.0).set_idle_linear(xyz),
        )
        .expect("XYZ insertion order")
        .to_string(),
        build_synthetic_idle_dem(
            &influence,
            NoiseConfig::new(0.0, 0.0, 0.0, 0.0).set_idle_linear(zyx),
        )
        .expect("ZYX insertion order")
        .to_string(),
    );
}

#[test]
fn invalid_idle_family_rates_and_weights_keep_existing_errors() {
    let cases = [
        (
            IdleNoiseFamily::new(-0.01, BTreeMap::from([("Z".to_string(), 1.0)])),
            "invalid linear idle rate/model [X=0, Y=0, Z=-0.01]",
        ),
        (
            IdleNoiseFamily::new(f64::INFINITY, BTreeMap::from([("Z".to_string(), 1.0)])),
            "invalid linear idle rate/model [X=0, Y=0, Z=inf]",
        ),
        (
            IdleNoiseFamily::new(0.01, BTreeMap::from([("X".to_string(), -1.0)])),
            "invalid linear idle rate/model [X=-0.01, Y=0, Z=0]",
        ),
        (
            IdleNoiseFamily::new(0.01, BTreeMap::from([("X".to_string(), f64::NAN)])),
            "invalid linear idle rate/model [X=NaN, Y=0, Z=0]",
        ),
        (
            IdleNoiseFamily::new(
                -f64::MIN_POSITIVE,
                BTreeMap::from([("X".to_string(), f64::MIN_POSITIVE)]),
            ),
            "invalid linear idle rate/model [X=-0, Y=0, Z=0]",
        ),
        (
            IdleNoiseFamily::new(f64::INFINITY, BTreeMap::from([("X".to_string(), 0.0)])),
            "invalid linear idle rate/model [X=0, Y=0, Z=0]",
        ),
        (
            IdleNoiseFamily::new(
                f64::MIN_POSITIVE,
                BTreeMap::from([("X".to_string(), -f64::MIN_POSITIVE)]),
            ),
            "invalid linear idle rate/model [X=-0, Y=0, Z=0]",
        ),
    ];

    for (family, expected) in cases {
        let error =
            build_tracked_idle_dem(NoiseConfig::new(0.0, 0.0, 0.0, 0.0).set_idle_linear(family))
                .expect_err("invalid family inputs must be rejected");
        assert!(error.contains(expected), "error={error:?}");
        assert!(error.contains("rates must be finite and non-negative"));
    }
}

#[test]
fn invalid_idle_weight_map_key_is_rejected() {
    let family = IdleNoiseFamily::new(0.01, BTreeMap::from([("not-a-pauli".to_string(), 1.0)]));
    let error =
        build_tracked_idle_dem(NoiseConfig::new(0.0, 0.0, 0.0, 0.0).set_idle_linear(family))
            .expect_err("invalid family key must be rejected");

    assert!(error.contains("invalid linear idle rate/model key \"not-a-pauli\""));
    assert!(error.contains("weights must use only X, Y, and Z"));
}

#[test]
fn equal_idle_signatures_sum_exclusive_probabilities_before_dem_merging() {
    let influence = synthetic_idle_influence(&[0], &[], &[0]);
    let noise =
        NoiseConfig::new(0.0, 0.0, 0.0, 0.0).set_idle_linear(axis_rate_family(0.25, 0.0, 0.75));
    let dem = build_synthetic_idle_dem(&influence, noise)
        .expect("equal signatures need no independent conversion");
    let contributions = idle_signature_contributions(&dem);

    assert_eq!(contributions.len(), 1);
    assert_eq!(contributions[0].1.to_bits(), 1.0_f64.to_bits());
    assert_ne!(
        contributions[0].1.to_bits(),
        combine_probabilities(0.25, 0.75).to_bits(),
        "exclusive aliases must sum, not use the independent XOR rule",
    );
    assert!(dem.idle_noise_residuals().is_empty());

    let measurement_model = MemBuilder::new(&influence)
        .with_noise_config(
            NoiseConfig::new(0.0, 0.0, 0.0, 0.0).set_idle_linear(axis_rate_family(0.25, 0.0, 0.75)),
        )
        .build();
    assert_eq!(measurement_model.mechanisms.len(), 1);
    let measurement_probability = *measurement_model
        .mechanisms
        .values()
        .next()
        .expect("equal raw-measurement signatures must emit one mechanism");
    assert_eq!(measurement_probability.to_bits(), 1.0_f64.to_bits());
    assert_ne!(
        measurement_probability.to_bits(),
        combine_probabilities(0.25, 0.75).to_bits()
    );
}

#[test]
fn empty_idle_signature_is_dropped_before_conversion() {
    let influence = synthetic_idle_influence(&[], &[0], &[0]);
    let noise =
        NoiseConfig::new(0.0, 0.0, 0.0, 0.0).set_idle_linear(axis_rate_family(0.25, 0.0, 0.75));
    let dem = build_synthetic_idle_dem(&influence, noise)
        .expect("an undetectable X branch cannot obstruct the surviving Z branch");
    let contributions = idle_signature_contributions(&dem);

    assert_eq!(contributions.len(), 1);
    assert_eq!(contributions[0].1.to_bits(), 0.75_f64.to_bits());
    assert!(dem.idle_noise_residuals().is_empty());

    let measurement_model = MemBuilder::new(&influence)
        .with_noise_config(
            NoiseConfig::new(0.0, 0.0, 0.0, 0.0).set_idle_linear(axis_rate_family(0.25, 0.0, 0.75)),
        )
        .build();
    assert_eq!(measurement_model.mechanisms.len(), 1);
    assert_eq!(
        measurement_model
            .mechanisms
            .values()
            .next()
            .expect("surviving Z measurement signature")
            .to_bits(),
        0.75_f64.to_bits()
    );
    assert!(measurement_model.idle_noise_residuals.is_empty());
}

#[test]
fn biased_xz_idle_channel_builds_with_quantified_boundary_residual() {
    let influence = synthetic_idle_influence(&[0], &[0, 1], &[1]);
    let loc_idx = idle_location(&influence);
    let noise =
        NoiseConfig::new(0.0, 0.0, 0.0, 0.0).set_idle_linear(axis_rate_family(0.0075, 0.0, 0.0225));
    let legacy_sampling_engine = SamplingEngine::from_influence_map(&influence, &[1.0], &noise);
    let mut sampler_influence = influence.clone();
    sampler_influence
        .detectors
        .push(DetectorId::single(MeasurementId {
            tick: 0,
            qubit: 0,
            basis: 0,
        }));
    let sampler = DemSamplerBuilder::new(&sampler_influence)
        .with_noise_config(noise.clone())
        .with_detectors_json(r#"[{"id": 0, "records": [-2]}, {"id": 1, "records": [-1]}]"#)
        .expect("valid detector metadata")
        .build()
        .expect("valid detector sampler");
    let sampler_dem = sampler.to_detector_error_model();
    let dem = build_synthetic_idle_dem(&influence, noise)
        .expect("ordinary biased X/Z idle noise must produce a usable DEM");
    let mechanisms = idle_signature_contributions(&dem);
    let distribution = independent_signature_distribution(&mechanisms);
    let x_effect = raw_idle_signature(&influence, loc_idx, Pauli::X);
    let y_effect = raw_idle_signature(&influence, loc_idx, Pauli::Y);
    let z_effect = raw_idle_signature(&influence, loc_idx, Pauli::Z);

    assert!(
        (distribution[&x_effect] - 0.0075).abs() < 1e-12,
        "distribution={distribution:?}, mechanisms={mechanisms:?}"
    );
    assert!((distribution[&z_effect] - 0.0225).abs() < 1e-12);
    let [residual] = dem.idle_noise_residuals() else {
        panic!("the infeasible two-signature channel must report one residual")
    };
    assert_eq!(residual.effect, y_effect);
    assert_eq!(residual.channel_weight.to_bits(), 0.03_f64.to_bits());
    assert!((distribution[&y_effect] - residual.magnitude).abs() < 1e-12);
    for sampler_residuals in [
        legacy_sampling_engine.idle_noise_residuals(),
        sampler_dem.idle_noise_residuals(),
    ] {
        let [sampler_residual] = sampler_residuals else {
            panic!("each sampler path must retain the idle-channel residual")
        };
        assert_eq!(
            sampler_residual.channel_weight.to_bits(),
            0.03_f64.to_bits()
        );
        assert_eq!(
            sampler_residual.relative_magnitude().to_bits(),
            residual.relative_magnitude().to_bits()
        );
    }
    let qx = mechanisms
        .iter()
        .find_map(|(effect, probability)| (effect == &x_effect).then_some(*probability))
        .expect("X signature mechanism");
    let qz = mechanisms
        .iter()
        .find_map(|(effect, probability)| (effect == &z_effect).then_some(*probability))
        .expect("Z signature mechanism");
    assert!((residual.magnitude - qx * qz).abs() < 1e-15);
}

#[test]
fn three_distinct_idle_signatures_match_engines_pauli_channel() {
    // pecos-engines samples one event with probability 0.01, then selects
    // X/Y/Z categorically with the configured relative weights. Include both
    // the documented 0.25/0.25/0.50 model and an asymmetric model so every
    // eigenvalue denominator is independently exercised.
    let influence = synthetic_idle_influence(&[0], &[0, 1], &[1]);
    let loc_idx = idle_location(&influence);
    let x_effect = raw_idle_signature(&influence, loc_idx, Pauli::X);
    let y_effect = raw_idle_signature(&influence, loc_idx, Pauli::Y);
    let z_effect = raw_idle_signature(&influence, loc_idx, Pauli::Z);

    for [px, py, pz] in [[0.0025, 0.0025, 0.005], [0.002, 0.003, 0.005]] {
        let noise =
            NoiseConfig::new(0.0, 0.0, 0.0, 0.0).set_idle_linear(axis_rate_family(px, py, pz));
        let dem = build_synthetic_idle_dem(&influence, noise)
            .expect("three-signature engines channel is exactly representable");
        let distribution = independent_signature_distribution(&idle_signature_contributions(&dem));

        assert!(
            (distribution[&FaultMechanism::new()] - (1.0 - px - py - pz)).abs() < 1e-12,
            "distribution={distribution:?}"
        );
        assert!((distribution[&x_effect] - px).abs() < 1e-12);
        assert!((distribution[&y_effect] - py).abs() < 1e-12);
        assert!((distribution[&z_effect] - pz).abs() < 1e-12);
        assert!(dem.idle_noise_residuals().is_empty());
    }
}

#[test]
fn idle_y_signature_is_xor_of_x_and_z_at_every_tested_location() {
    let synthetic = synthetic_idle_influence(&[0], &[0, 1], &[1]);
    let synthetic_loc = idle_location(&synthetic);
    assert_eq!(
        raw_idle_signature(&synthetic, synthetic_loc, Pauli::Y),
        raw_idle_signature(&synthetic, synthetic_loc, Pauli::X).xor(&raw_idle_signature(
            &synthetic,
            synthetic_loc,
            Pauli::Z
        )),
        "synthetic non-empty signatures",
    );

    for dag in [
        build_unit_idle_with_pauli_tracking(),
        build_unit_idle_tracking_x(),
        build_unit_idle_tracking_y(),
        build_idle_then_measure(3),
    ] {
        let influence = DagFaultAnalyzer::new(&dag).build_influence_map();
        for (loc_idx, location) in influence.locations.iter().enumerate() {
            if location.gate_type != GateType::Idle || location.before {
                continue;
            }
            let x_effect = raw_idle_signature(&influence, loc_idx, Pauli::X);
            let y_effect = raw_idle_signature(&influence, loc_idx, Pauli::Y);
            let z_effect = raw_idle_signature(&influence, loc_idx, Pauli::Z);
            assert_eq!(y_effect, x_effect.xor(&z_effect), "idle location {loc_idx}");
        }
    }
}

#[test]
fn linear_and_sine_idle_families_emit_separate_contributions() {
    let linear_probability = 0.01;
    let sine_rate: f64 = 0.2;
    let sine_probability = sine_rate.sin().powi(2);
    let noise = NoiseConfig::new(0.0, 0.0, 0.0, 0.0)
        .set_idle_linear(z_idle_family(linear_probability))
        .set_idle_quadratic_sine(z_idle_family(sine_rate));
    let dem = build_tracked_idle_dem(noise).expect("valid composed-family DEM");

    let mut z_probabilities = dem
        .contribution_render_records()
        .into_iter()
        .filter_map(|record| {
            (record.contribution.source_gate_types.as_slice() == [GateType::Idle])
                .then_some(record.contribution.probability)
        })
        .collect::<Vec<_>>();
    z_probabilities.sort_by(f64::total_cmp);

    assert_eq!(z_probabilities.len(), 2);
    assert!((z_probabilities[0] - linear_probability).abs() < 1e-15);
    assert!((z_probabilities[1] - sine_probability).abs() < 1e-15);
    assert!(
        (z_probabilities.iter().sum::<f64>() - (linear_probability + sine_probability)).abs()
            < 1e-15
    );
}

#[test]
fn nonpositive_signature_channel_character_returns_specific_error() {
    let influence = synthetic_idle_influence(&[0], &[0, 1], &[1]);
    let error = build_synthetic_idle_dem(
        &influence,
        NoiseConfig::new(0.0, 0.0, 0.0, 0.0).set_idle_linear(axis_rate_family(0.25, 0.0, 0.25)),
    )
    .expect_err("zero signature-channel characters are broken input");

    assert!(error.contains("DEM builder configuration error"));
    assert!(error.contains("location"));
    assert!(error.contains("characters"));
    assert!(error.contains("must all be positive"));
}

#[test]
fn oversized_coefficient_quadratic_mechanism_returns_specific_error() {
    let error = build_tracked_idle_dem(
        NoiseConfig::new(0.0, 0.0, 0.0, 0.0).set_idle_quadratic(z_idle_family(1.1)),
    )
    .expect_err("probabilities above one must not be clamped");

    assert!(error.contains("coefficient-quadratic idle mechanism probabilities"));
    assert!(error.contains("Z=1.1"));
    assert!(error.contains("must be finite and lie in [0, 1]"));
}

#[test]
fn negative_idle_rate_is_rejected_instead_of_clamped() {
    let error = build_tracked_idle_dem(
        NoiseConfig::new(0.0, 0.0, 0.0, 0.0).set_idle_linear(z_idle_family(-0.01)),
    )
    .expect_err("negative rates must not be clamped");

    assert!(error.contains("invalid linear idle rate/model [X=0, Y=0, Z=-0.01]"));
    assert!(error.contains("rates must be finite and non-negative"));
}

#[test]
fn negative_idle_duration_is_rejected_instead_of_clamped() {
    let dag = build_unit_idle_tracking_x();
    let mut influence = DagFaultAnalyzer::new(&dag).build_influence_map();
    let loc_idx = idle_location(&influence);
    influence.locations[loc_idx].idle_duration = -1.0;
    let error = DemBuilder::new(&influence)
        .with_noise_config(
            NoiseConfig::new(0.0, 0.0, 0.0, 0.0).set_idle_linear(z_idle_family(0.01)),
        )
        .try_build()
        .expect_err("negative idle durations must not be clamped");

    assert!(error.to_string().contains("invalid idle duration -1"));
    assert!(
        error
            .to_string()
            .contains("duration must be finite and non-negative")
    );
}

#[test]
fn identical_idle_configuration_produces_byte_identical_dem_text() {
    let influence = synthetic_idle_influence(&[0], &[0, 1], &[1]);
    let noise =
        NoiseConfig::new(0.0, 0.0, 0.0, 0.0).set_idle_linear(axis_rate_family(0.002, 0.003, 0.005));
    let build = || {
        build_synthetic_idle_dem(&influence, noise.clone())
            .expect("valid deterministic DEM")
            .to_string()
    };

    let expected = build();
    assert_eq!(
        expected,
        "detector D0\ndetector D1\nerror(0.002001) D0\nerror(0.003011) D0 D1\nerror(0.005019) D1",
        "this full dimension-two DEM text is pinned to commit 79e8aa833",
    );
    let dem = build_synthetic_idle_dem(&influence, noise.clone()).expect("valid pinned DEM");
    let probabilities = idle_signature_contributions(&dem)
        .into_iter()
        .map(|(_, probability)| probability.to_bits())
        .collect::<Vec<_>>();
    assert_eq!(
        probabilities,
        [
            0.002_000_955_040_586_616_f64.to_bits(),
            0.003_011_095_091_214_222_f64.to_bits(),
            0.005_019_131_070_643_723_f64.to_bits(),
        ],
        "dimension-two idle mechanism bits are pinned to commit 79e8aa833",
    );
    for _ in 0..16 {
        assert_eq!(build().as_bytes(), expected.as_bytes());
    }
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
