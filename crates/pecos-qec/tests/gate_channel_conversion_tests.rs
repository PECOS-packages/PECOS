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

//! Regression tests for categorical gate-Pauli channels converted to independent
//! DEM mechanisms after propagation to concrete flip signatures.

use pecos_core::QubitId;
use pecos_qec::fault_tolerance::dem_builder::{
    DemBuilder, DetectorErrorModel, DirectSourceFamily, FaultMechanism, MeasurementNoiseModel,
    MemBuilder, NoiseChannelKind, NoiseConfig, PerGateTypeNoise, combine_probabilities,
};
use pecos_qec::fault_tolerance::propagator::{DagFaultInfluenceMap, DagSpacetimeLocation};
use pecos_quantum::GateType;

const FOUR_DETECTORS: &str = r#"[
    {"id": 0, "records": [-4]},
    {"id": 1, "records": [-3]},
    {"id": 2, "records": [-2]},
    {"id": 3, "records": [-1]}
]"#;

fn synthetic_one_qubit_influence(
    gate_type: GateType,
    before: bool,
    x: &[u32],
    y: &[u32],
    z: &[u32],
) -> DagFaultInfluenceMap {
    let mut influence = DagFaultInfluenceMap::with_capacity(1);
    influence.locations.push(DagSpacetimeLocation {
        node: 0,
        qubits: vec![QubitId::from(0usize)],
        before,
        gate_type,
        idle_duration: 0.0,
    });
    influence.influences.detectors_x.extend(x.iter().copied());
    influence.influences.detectors_y.extend(y.iter().copied());
    influence.influences.detectors_z.extend(z.iter().copied());
    influence.influences.finish_location();
    influence.measurements = (0..4).map(|index| (index, index, 0)).collect();
    influence
}

fn synthetic_two_qubit_influence() -> DagFaultInfluenceMap {
    let mut influence = DagFaultInfluenceMap::with_capacity(2);
    for (qubit, x, y, z) in [
        (0, &[0][..], &[0, 1][..], &[1][..]),
        (1, &[2][..], &[2, 3][..], &[3][..]),
    ] {
        influence.locations.push(DagSpacetimeLocation {
            node: 0,
            qubits: vec![QubitId::from(qubit)],
            before: false,
            gate_type: GateType::CX,
            idle_duration: 0.0,
        });
        influence.influences.detectors_x.extend(x.iter().copied());
        influence.influences.detectors_y.extend(y.iter().copied());
        influence.influences.detectors_z.extend(z.iter().copied());
        influence.influences.finish_location();
    }
    influence.measurements = (0..4).map(|index| (index, index, 0)).collect();
    influence
}

fn independent_distribution(model: &MeasurementNoiseModel, dimension: usize) -> Vec<f64> {
    let mechanisms = model.mechanisms.iter().map(|(mechanism, &probability)| {
        let mask = mechanism
            .measurements
            .iter()
            .fold(0usize, |mask, &index| mask ^ (1usize << index));
        (mask, probability)
    });
    compose_independent_mechanisms(mechanisms, dimension)
}

fn fault_mechanism_mask(mechanism: &FaultMechanism) -> usize {
    mechanism
        .detectors
        .iter()
        .fold(0usize, |mask, &index| mask ^ (1usize << index))
}

fn compose_independent_mechanisms(
    mechanisms: impl IntoIterator<Item = (usize, f64)>,
    dimension: usize,
) -> Vec<f64> {
    let size = 1usize << dimension;
    let mut distribution = vec![0.0; size];
    distribution[0] = 1.0;
    for (mask, probability) in mechanisms {
        let previous = distribution.clone();
        for effect in 0..size {
            distribution[effect] =
                previous[effect] * (1.0 - probability) + previous[effect ^ mask] * probability;
        }
    }
    distribution
}

fn build_synthetic_dem(influence: &DagFaultInfluenceMap, noise: NoiseConfig) -> DetectorErrorModel {
    DemBuilder::new(influence)
        .with_noise_config(noise)
        .with_detectors_json(FOUR_DETECTORS)
        .expect("valid detector metadata")
        .try_build()
        .expect("valid categorical channel")
}

fn build_synthetic_per_gate_dem(
    influence: &DagFaultInfluenceMap,
    gate_type: GateType,
    rates: [f64; 3],
) -> DetectorErrorModel {
    let noise = PerGateTypeNoise::from_base_noise(NoiseConfig::new(0.0, 0.0, 0.0, 0.0))
        .with_1q_rates(gate_type, rates);
    DemBuilder::new(influence)
        .with_per_gate_noise(noise)
        .with_detectors_json(FOUR_DETECTORS)
        .expect("valid detector metadata")
        .try_build()
        .expect("valid categorical channel")
}

fn gate_signature_mechanisms(dem: &DetectorErrorModel) -> Vec<(FaultMechanism, f64)> {
    dem.contribution_render_records()
        .into_iter()
        .map(|record| {
            let contribution = record.contribution;
            assert!(contribution.paulis.is_empty());
            assert_eq!(
                contribution.direct_source_family,
                Some(DirectSourceFamily::ExclusiveSignature)
            );
            (contribution.effect, contribution.probability)
        })
        .collect()
}

#[test]
fn single_qubit_gate_mechanisms_compose_to_the_three_pauli_channel() {
    let p1 = 0.002;
    let influence = synthetic_one_qubit_influence(GateType::H, false, &[0], &[0, 1], &[1]);
    let model = MemBuilder::new(&influence)
        .with_noise_config(NoiseConfig::new(p1, 0.0, 0.0, 0.0))
        .build();
    let distribution = independent_distribution(&model, 2);
    let target = p1 / 3.0;

    assert_eq!(model.mechanisms.len(), 3);
    for &probability in model.mechanisms.values() {
        assert!((probability - 0.000_667_111_704_693_190_7).abs() < 1e-15);
    }
    assert!((distribution[0] - (1.0 - p1)).abs() < 1e-12);
    for effect_probability in &distribution[1..] {
        assert!((effect_probability - target).abs() < 1e-12);
    }

    let dem = build_synthetic_dem(&influence, NoiseConfig::new(p1, 0.0, 0.0, 0.0));
    let mechanisms = gate_signature_mechanisms(&dem);
    assert_eq!(mechanisms.len(), 3);
    assert!(dem.idle_noise_residuals().is_empty());
    let dem_distribution = compose_independent_mechanisms(
        mechanisms
            .iter()
            .map(|(effect, probability)| (fault_mechanism_mask(effect), *probability)),
        2,
    );
    for (actual, expected) in dem_distribution
        .iter()
        .zip([1.0 - p1, target, target, target])
    {
        assert!(
            (actual - expected).abs() < 1e-12,
            "distribution={dem_distribution:?}, mechanisms={mechanisms:?}"
        );
    }
}

#[test]
fn two_qubit_gate_mechanisms_compose_to_the_fifteen_pauli_channel() {
    let p2 = 0.02;
    let influence = synthetic_two_qubit_influence();
    let model = MemBuilder::new(&influence)
        .with_noise_config(NoiseConfig::new(0.0, p2, 0.0, 0.0))
        .build();
    let distribution = independent_distribution(&model, 4);
    let target = p2 / 15.0;

    assert_eq!(model.mechanisms.len(), 15);
    for &probability in model.mechanisms.values() {
        assert!((probability - 0.001_345_946_290_707_722_4).abs() < 1e-15);
    }
    assert!((distribution[0] - (1.0 - p2)).abs() < 1e-12);
    for effect_probability in &distribution[1..] {
        assert!((effect_probability - target).abs() < 1e-12);
    }

    let dem = build_synthetic_dem(&influence, NoiseConfig::new(0.0, p2, 0.0, 0.0));
    let mechanisms = gate_signature_mechanisms(&dem);
    assert_eq!(mechanisms.len(), 15);
    assert!(dem.idle_noise_residuals().is_empty());
    let dem_distribution = compose_independent_mechanisms(
        mechanisms
            .iter()
            .map(|(effect, probability)| (fault_mechanism_mask(effect), *probability)),
        4,
    );
    assert!(
        (dem_distribution[0] - (1.0 - p2)).abs() < 1e-12,
        "distribution={dem_distribution:?}, mechanisms={mechanisms:?}"
    );
    for effect_probability in &dem_distribution[1..] {
        assert!((effect_probability - target).abs() < 1e-12);
    }
}

#[test]
fn equal_gate_signatures_sum_before_independent_merging() {
    let influence = synthetic_one_qubit_influence(GateType::H, false, &[0], &[], &[0]);
    let dem = build_synthetic_per_gate_dem(&influence, GateType::H, [0.2, 0.0, 0.3]);
    let mechanisms = gate_signature_mechanisms(&dem);

    assert_eq!(mechanisms.len(), 1);
    assert_eq!(mechanisms[0].1.to_bits(), 0.5_f64.to_bits());
    assert_ne!(
        mechanisms[0].1.to_bits(),
        combine_probabilities(0.2, 0.3).to_bits()
    );
    assert!(dem.idle_noise_residuals().is_empty());
}

#[test]
fn vanishing_gate_signatures_drop_without_changing_survivors() {
    let influence = synthetic_one_qubit_influence(GateType::H, false, &[], &[1], &[1]);
    let dem = build_synthetic_per_gate_dem(&influence, GateType::H, [0.2, 0.3, 0.4]);
    let mechanisms = gate_signature_mechanisms(&dem);

    assert_eq!(mechanisms.len(), 1);
    assert!((mechanisms[0].1 - 0.7).abs() < 1e-15);
    assert!(dem.idle_noise_residuals().is_empty());
}

#[test]
fn prep_and_measurement_channels_remain_single_exact_mechanisms() {
    for (gate_type, before, noise, expected) in [
        (
            GateType::PZ,
            false,
            NoiseConfig::new(0.0, 0.0, 0.0, 0.25),
            0.25_f64,
        ),
        (
            GateType::MZ,
            true,
            NoiseConfig::new(0.0, 0.0, 0.375, 0.0),
            0.375_f64,
        ),
    ] {
        let influence = synthetic_one_qubit_influence(gate_type, before, &[0], &[], &[]);
        let model = MemBuilder::new(&influence)
            .with_noise_config(noise.clone())
            .build();
        assert_eq!(model.mechanisms.len(), 1);
        assert_eq!(
            model
                .mechanisms
                .values()
                .next()
                .expect("single prep or measurement mechanism")
                .to_bits(),
            expected.to_bits()
        );
        assert!(model.idle_noise_residuals.is_empty());

        let dem = build_synthetic_dem(&influence, noise);
        let records = dem.contribution_render_records();
        assert_eq!(records.len(), 1);
        assert_eq!(
            records[0].contribution.probability.to_bits(),
            expected.to_bits()
        );
        assert!(dem.idle_noise_residuals().is_empty());
    }
}

#[test]
fn infeasible_gate_channel_reports_kind_and_queryable_magnitude() {
    let influence = synthetic_one_qubit_influence(GateType::H, false, &[0], &[0, 1], &[1]);
    let dem = build_synthetic_per_gate_dem(&influence, GateType::H, [0.0075, 0.0, 0.0225]);

    let [residual] = dem.idle_noise_residuals() else {
        panic!("the infeasible gate channel must report one residual")
    };
    assert_eq!(residual.channel_kind, NoiseChannelKind::SingleQubitGate);
    assert_eq!(fault_mechanism_mask(&residual.effect), 0b11);
    assert_eq!(residual.channel_weight.to_bits(), 0.03_f64.to_bits());
    assert!(residual.magnitude > 0.0);
    assert_eq!(
        residual.relative_magnitude().to_bits(),
        (residual.magnitude / 0.03).to_bits()
    );
    let mechanisms = gate_signature_mechanisms(&dem);
    let distribution = compose_independent_mechanisms(
        mechanisms
            .iter()
            .map(|(effect, probability)| (fault_mechanism_mask(effect), *probability)),
        2,
    );
    assert!((distribution[0b11] - residual.magnitude).abs() < 1e-12);
}

#[test]
fn broken_gate_probabilities_and_nonpositive_characters_are_hard_errors() {
    let influence = synthetic_one_qubit_influence(GateType::H, false, &[0], &[0, 1], &[1]);
    let build = |rates| {
        DemBuilder::new(&influence)
            .with_per_gate_noise(
                PerGateTypeNoise::from_base_noise(NoiseConfig::new(0.0, 0.0, 0.0, 0.0))
                    .with_1q_rates(GateType::H, rates),
            )
            .with_detectors_json(FOUR_DETECTORS)
            .expect("valid detector metadata")
            .try_build()
    };

    let probability_error = build([0.6, 0.6, 0.0])
        .expect_err("a categorical total above one must be rejected")
        .to_string();
    assert!(probability_error.contains("one-qubit H gate"));
    assert!(probability_error.contains("total 1.2"));
    assert!(probability_error.contains("must be finite and lie in [0, 1]"));

    for rates in [[-0.1, 0.1, 0.0], [f64::NAN, 0.0, 0.0]] {
        let probability_error = build(rates)
            .expect_err("non-finite and out-of-range probabilities must be rejected")
            .to_string();
        assert!(probability_error.contains("one-qubit H gate"));
        assert!(probability_error.contains("must be finite and lie in [0, 1]"));
    }

    let character_error = build([0.25, 0.0, 0.25])
        .expect_err("a zero signature character must be rejected")
        .to_string();
    assert!(character_error.contains("one-qubit H gate"));
    assert!(character_error.contains("characters"));
    assert!(character_error.contains("must all be positive"));
}

#[test]
fn identical_gate_configuration_produces_byte_identical_dem_text() {
    let influence = synthetic_two_qubit_influence();
    let build =
        || build_synthetic_dem(&influence, NoiseConfig::new(0.0, 0.02, 0.0, 0.0)).to_string();
    let expected = build();
    for _ in 0..16 {
        assert_eq!(build().as_bytes(), expected.as_bytes());
    }
}
